# Copyright (c) 2026 Eric Hiss. All rights reserved.
# Licensed under the terms in LICENSE.txt (CC BY-NC 4.0 / Commercial dual license).
# https://github.com/EricRollei/Eric_Qwen_Edit_Experiments
"""
Eric Qwen-Edit Unify (Composite Harmonizer) Node

Takes the output of Eric_Composer_Studio's ImageComposer (a layered collage
where elements have been segmented and placed but lighting / shadows / seams
do not yet match) and runs a single Qwen-Image-Edit-2511 pass to harmonise
the whole image into a single cohesive photograph.

Three modes:

- "harmonize_only": (DEFAULT, RECOMMENDED) Send ONLY the composite to Qwen-Edit
                    with the harmonisation prompt. No segments, no relayed
                    layers, no masks, no post-composite. Cleanest path to
                    "unify lighting and remove seams" — Qwen sees the whole
                    composite as one image and relights it coherently.

- "whole":          Full Qwen-Edit pass on the composite. Same as harmonize_only
                    but ALSO sends optional segments / relayed Composer layers
                    as additional VL/ref images. Use only if you find a subject's
                    identity is being lost in harmonize_only — these extra refs
                    can fight harmonisation by anchoring subjects to their
                    original (mismatched) lighting.

- "seam_inpaint":   Run Qwen-Edit on the full composite then post-composite the
                    result back over the original through a feathered mask, so
                    untouched regions remain pixel-identical to the composite.
                    Default mask source is seam_mask from ImageComposer.
                    ADVANCED — requires correctly-wired masks. If masks don't
                    align with subjects you'll get dissolved subjects / halos.
                    Use harmonize_only first; only fall back to this if you
                    need surgical control over which areas get regenerated.

Qwen-Edit has no classic denoise/strength knob \u2014 structure is anchored by
the VAE ref-latent, not an init-latent. The seam_inpaint mode therefore
implements partial-strength harmonisation by *spatial masking* rather than
by reducing the diffusion strength.

Author: Eric Hiss (GitHub: EricRollei)
"""

from __future__ import annotations

import time
from typing import List, Optional, Tuple

import torch

import comfy.utils

from .eric_qwen_edit_utils import (
    tensor_to_pil,
    pil_to_tensor,
    prepare_image_for_pipeline,
)


# ---------------------------------------------------------------------------
# Default harmonisation prompt
# ---------------------------------------------------------------------------

_DEFAULT_HARMONIZE_PROMPT = (
    "Relight every subject in this image so they all share the same lighting "
    "direction, color temperature, exposure and white balance as the background "
    "scene. Add natural cast shadows and contact shadows where each subject meets "
    "the ground, water, or another surface. Where a subject stands in water, show "
    "realistic water interaction: ripples, displacement, partial submersion, "
    "reflections. Match atmospheric perspective, depth of field and film grain "
    "across all elements. "
    "CRITICAL: every subject currently present in the image — including any human "
    "figure, statue, sculpture, animal, or object — must remain present, intact, "
    "in exactly the same position, pose and scale. Do not remove, erase, replace, "
    "hide, restyle, or move any subject. Do not change anyone's clothing, identity, "
    "species, or material. Only adjust lighting, color, shadows and edge integration "
    "so the result reads as one continuously captured photograph."
)


# ---------------------------------------------------------------------------
# Mask helpers (numpy / PIL)
# ---------------------------------------------------------------------------

def _mask_tensor_to_np(mask: torch.Tensor):
    import numpy as np
    if mask is None:
        return None
    if mask.dim() == 3:
        m = mask[0]
    else:
        m = mask
    return m.detach().cpu().numpy().astype(np.float32)


def _resize_mask_np(m, target_w: int, target_h: int):
    import numpy as np
    import cv2
    if m is None:
        return None
    h, w = m.shape[:2]
    if (w, h) == (target_w, target_h):
        return m.astype(np.float32)
    return cv2.resize(m.astype(np.float32), (target_w, target_h), interpolation=cv2.INTER_LINEAR)


def _edge_band_from_fg(fg_mask_np, border_px: int):
    """Fallback seam: dilate(fg) - erode(fg) when no seam_mask is provided."""
    import numpy as np
    import cv2
    k = max(1, int(border_px))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * k + 1, 2 * k + 1))
    b = (fg_mask_np > 0.05).astype(np.uint8)
    dil = cv2.dilate(b, kernel, iterations=1)
    ero = cv2.erode(b, kernel, iterations=1)
    return np.clip(dil.astype(np.float32) - ero.astype(np.float32), 0.0, 1.0)


def _dilate_mask(m, px: int):
    import cv2
    import numpy as np
    if px <= 0 or m is None:
        return m
    k = int(px)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * k + 1, 2 * k + 1))
    out = cv2.dilate((m > 0.01).astype(np.uint8), kernel, iterations=1).astype(np.float32)
    # Preserve soft values where we already had them
    return np.maximum(m.astype(np.float32), out)


def _feather_mask(m, px: int):
    import cv2
    if px <= 0 or m is None:
        return m
    sigma = max(0.5, float(px))
    ksz = int(2 * round(sigma * 2) + 1)
    return cv2.GaussianBlur(m.astype("float32"), (ksz, ksz), sigma)


def _extend_down(m, px: int):
    """Vertical morphological dilation downward only (for ground-contact strip)."""
    import numpy as np
    import cv2
    if px <= 0 or m is None:
        return m
    k = int(px)
    # Anchor at the TOP of a vertical kernel so dilation only grows downward.
    kernel = np.ones((2 * k + 1, 1), dtype=np.uint8)
    b = (m > 0.05).astype(np.uint8)
    out = cv2.dilate(b, kernel, anchor=(0, 0), iterations=1).astype(np.float32)
    return np.maximum(m.astype(np.float32), out)


def _composite_with_mask_pil(original_pil, generated_pil, mask_np):
    """Blend generated over original via mask (already feathered, [0,1])."""
    import numpy as np
    from PIL import Image

    if generated_pil.size != original_pil.size:
        generated_pil = generated_pil.resize(original_pil.size, Image.LANCZOS)
    target_w, target_h = original_pil.size
    if mask_np.shape[:2] != (target_h, target_w):
        import cv2
        mask_np = cv2.resize(mask_np.astype("float32"), (target_w, target_h),
                             interpolation=cv2.INTER_LINEAR)

    orig = np.asarray(original_pil.convert("RGB"), dtype=np.float32)
    gen = np.asarray(generated_pil.convert("RGB"), dtype=np.float32)
    a = np.clip(mask_np, 0.0, 1.0)[..., None]
    out = orig * (1.0 - a) + gen * a
    return Image.fromarray(np.clip(out, 0, 255).astype("uint8"), "RGB")


# ---------------------------------------------------------------------------
# Node
# ---------------------------------------------------------------------------

class EricQwenEditUnify:
    """Unify a composited collage into a single cohesive photograph."""

    CATEGORY = "Eric Qwen-Edit"
    FUNCTION = "unify"
    RETURN_TYPES = ("IMAGE", "MASK")
    RETURN_NAMES = ("image", "used_mask")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "pipeline": ("QWEN_EDIT_PIPELINE",),
                "composite_image": ("IMAGE", {
                    "tooltip": "Composite from ImageComposer (or any other source). "
                               "This is sent to Qwen-Edit as the main reference image."
                }),
                "mode": (["harmonize_only", "whole", "seam_inpaint"], {
                    "default": "harmonize_only",
                    "tooltip": "harmonize_only (RECOMMENDED): send ONLY the composite to Qwen-Edit. "
                               "No segments, no masks, no post-compositing. Cleanest path — the model "
                               "relights the whole image coherently. Try this first.\n"
                               "whole: same as harmonize_only but ALSO forwards segments / Composer "
                               "layers as extra VL/ref images. Can fight harmonisation; use only if a "
                               "subject's identity is being lost.\n"
                               "seam_inpaint: ADVANCED. Generates with Qwen, then post-composites "
                               "via a feathered mask so untouched regions stay pixel-identical. "
                               "Requires well-aligned foreground/seam masks; otherwise produces "
                               "dissolved subjects or halos."
                }),
                "user_prompt": ("STRING", {
                    "multiline": True,
                    "default": "",
                    "tooltip": "Optional extra instruction. If empty, the baked harmonisation prompt is used as-is."
                }),
                "prompt_mode": (["append", "replace"], {
                    "default": "append",
                    "tooltip": "append: user_prompt is appended to the baked harmonise preamble. "
                               "replace: user_prompt fully replaces the baked preamble (advanced)."
                }),
            },
            "optional": {
                # Relayed-from-Composer bundle (one wire, replaces segment_X)
                "composer_layers": ("COMPOSER_LAYERS", {
                    "tooltip": "Bundle from ImageComposer's composer_layers output. Contains "
                               "occlusion-baked, tightly-cropped, neutral-filled per-layer crops "
                               "with visibility metadata. When connected, REPLACES the manual "
                               "segment_1..5 inputs."
                }),
                "layer_visibility_threshold": ("FLOAT", {
                    "default": 0.08, "min": 0.0, "max": 1.0, "step": 0.01,
                    "tooltip": "Drop relayed layers whose visible footprint is below this fraction "
                               "of their own area (default 8%). Avoids feeding heavily-occluded slivers."
                }),
                "layer_max_count": ("INT", {
                    "default": 4, "min": 0, "max": 5, "step": 1,
                    "tooltip": "Cap on relayed layers (Qwen-Edit handles up to ~4-5 images including the "
                               "composite). Lowest-visibility layers are dropped first. 0 = ignore the bundle."
                }),
                # Mask sources
                "seam_mask": ("MASK", {
                    "tooltip": "Seam mask from ImageComposer (silhouette borders of each layer). "
                               "Used as the base for seam_inpaint mode. If absent, a fallback edge "
                               "band is derived from foreground_mask."
                }),
                "foreground_mask": ("MASK", {
                    "tooltip": "Combined foreground mask from ImageComposer. Used for ground-contact "
                               "extension (strip below subjects) and as fallback if seam_mask is absent."
                }),
                "manual_mask": ("MASK", {
                    "tooltip": "Optional manual mask. If provided, OVERRIDES seam_mask + auto extensions "
                               "(use this to constrain edits to a hand-painted region)."
                }),
                # Mask shaping
                "seam_dilate_px": ("INT", {
                    "default": 16, "min": 0, "max": 256, "step": 1,
                    "tooltip": "Outward growth of the seam band in pixels. Larger = wider halo of "
                               "surrounding pixels gets harmonised. With subject_protect ON this "
                               "band lives entirely OUTSIDE each subject — subject pixels are preserved."
                }),
                "subject_protect": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "ON (recommended): seam band lives entirely outside each subject's "
                               "silhouette — subject edges and interior stay pixel-identical to the "
                               "composite. OFF: seam straddles the silhouette (more aggressive "
                               "harmonisation but can blur / halo subject edges). "
                               "Requires foreground_mask to take effect."
                }),
                "subject_edge_blend_px": ("INT", {
                    "default": 1, "min": 0, "max": 16, "step": 1,
                    "tooltip": "With subject_protect ON, allow this many pixels of soft blending "
                               "INSIDE the subject silhouette so the join doesn't look stamped. "
                               "0 = perfectly hard edge. 1–2 px is invisible but smoother."
                }),
                "ground_contact_px": ("INT", {
                    "default": 48, "min": 0, "max": 512, "step": 1,
                    "tooltip": "Vertical strip extended DOWN from foreground silhouettes so the model can "
                               "paint cast/contact shadows and ground/water interaction. 0 disables."
                }),
                "seam_feather_px": ("INT", {
                    "default": 16, "min": 0, "max": 128, "step": 1,
                    "tooltip": "Final Gaussian feather of the seam mask for smooth blend back into the composite."
                }),
                # Optional segments (multi-image style)
                "segment_1": ("IMAGE", {"tooltip": "Optional original segment (Picture 2). VL-only by default."}),
                "segment_2": ("IMAGE", {"tooltip": "Optional original segment (Picture 3)."}),
                "segment_3": ("IMAGE", {"tooltip": "Optional original segment (Picture 4)."}),
                "segment_4": ("IMAGE", {"tooltip": "Optional original segment (Picture 5)."}),
                "segment_5": ("IMAGE", {"tooltip": "Optional original segment (Picture 6)."}),
                "vl_segments": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Send segments through the VL/semantic path so the model keeps each "
                               "subject's identity straight."
                }),
                "ref_segments": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Send segments through the VAE/ref pixel path. Default OFF \u2014 segments at "
                               "DIFFERENT lighting can fight harmonisation. Enable only if a subject's "
                               "identity is being lost."
                }),
                # Qwen-Edit common params
                "negative_prompt": ("STRING", {
                    "multiline": True, "default": "",
                    "tooltip": "What to avoid (e.g. 'plastic look, oversharpened, mismatched lighting')."
                }),
                "steps": ("INT", {
                    "default": 8, "min": 1, "max": 100, "step": 1,
                    "tooltip": "Inference steps (8 for lightning LoRA, ~30-50 for base model)."
                }),
                "true_cfg_scale": ("FLOAT", {
                    "default": 3.0, "min": 1.0, "max": 20.0, "step": 0.5,
                    "tooltip": "True CFG scale. Lower (2.5–3.0) is better for harmonisation — "
                               "high CFG over-applies the prompt and can cause Qwen to delete "
                               "out-of-place subjects (statues, small animals)."
                }),
                "seed": ("INT", {
                    "default": 0, "min": 0, "max": 0xffffffffffffffff,
                    "tooltip": "Random seed."
                }),
                "max_mp": ("FLOAT", {
                    "default": 8.0, "min": 0.5, "max": 17.0, "step": 0.5,
                    "tooltip": "Max output megapixels (Qwen-Edit can run up to ~17 MP at high VRAM)."
                }),
            },
        }

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return kwargs.get("seed", 0)

    # ------------------------------------------------------------------
    # Mask construction
    # ------------------------------------------------------------------
    def _build_mask(
        self,
        target_w: int, target_h: int,
        seam_mask: Optional[torch.Tensor],
        foreground_mask: Optional[torch.Tensor],
        manual_mask: Optional[torch.Tensor],
        seam_dilate_px: int,
        ground_contact_px: int,
        seam_feather_px: int,
        subject_protect: bool = True,
        subject_edge_blend_px: int = 1,
    ):
        import numpy as np
        import cv2

        # Manual override always wins
        if manual_mask is not None:
            base = _mask_tensor_to_np(manual_mask)
            base = _resize_mask_np(base, target_w, target_h)
            return _feather_mask(np.clip(base, 0.0, 1.0), seam_feather_px)

        # Build seam component
        seam_np = _mask_tensor_to_np(seam_mask)
        if seam_np is not None:
            seam_np = _resize_mask_np(seam_np, target_w, target_h)
        fg_np = _mask_tensor_to_np(foreground_mask)
        if fg_np is not None:
            fg_np = _resize_mask_np(fg_np, target_w, target_h)

        if seam_np is None and fg_np is not None:
            # Fallback: derive a thin edge band from the foreground mask
            seam_np = _edge_band_from_fg(fg_np, border_px=3)
        if seam_np is None and fg_np is None:
            # No info at all \u2014 fall back to an all-ones mask (== whole-image)
            return np.ones((target_h, target_w), dtype=np.float32)

        m = _dilate_mask(seam_np, seam_dilate_px)

        # Ground-contact strip from foreground silhouette
        if ground_contact_px > 0 and fg_np is not None:
            strip = _extend_down(fg_np, ground_contact_px)
            # Subtract the original interior so we only ADD the band beneath subjects;
            # the subject body itself stays unmasked (preserves original pixels).
            interior = (fg_np > 0.05).astype("float32")
            band = np.clip(strip - interior, 0.0, 1.0)
            m = np.maximum(m, band)

        # Subject-edge protection: zero out the seam mask wherever the
        # foreground silhouette is solid, so the model never repaints the
        # subject's actual pixels (prevents the dissolved/halo look).
        if subject_protect and fg_np is not None:
            if subject_edge_blend_px > 0:
                k = int(subject_edge_blend_px)
                kernel = cv2.getStructuringElement(
                    cv2.MORPH_ELLIPSE, (2 * k + 1, 2 * k + 1)
                )
                interior_eroded = cv2.erode(
                    (fg_np > 0.5).astype("uint8"), kernel, iterations=1
                ).astype("float32")
                protect = cv2.GaussianBlur(
                    interior_eroded, (2 * k + 1, 2 * k + 1), max(0.5, float(k))
                )
            else:
                protect = (fg_np > 0.5).astype("float32")
            m = np.clip(m * (1.0 - protect), 0.0, 1.0)

        m = _feather_mask(np.clip(m, 0.0, 1.0), seam_feather_px)

        # Re-apply subject protect AFTER feathering as a hard cap, so the
        # feather can't bleed back into the subject's interior.
        if subject_protect and fg_np is not None:
            interior_cap = (fg_np > 0.5).astype("float32")
            if subject_edge_blend_px > 0:
                k = int(subject_edge_blend_px)
                kernel = cv2.getStructuringElement(
                    cv2.MORPH_ELLIPSE, (2 * k + 1, 2 * k + 1)
                )
                interior_cap = cv2.erode(
                    interior_cap.astype("uint8"), kernel, iterations=1
                ).astype("float32")
            m = np.clip(m * (1.0 - interior_cap), 0.0, 1.0)

        return np.clip(m, 0.0, 1.0)

    # ------------------------------------------------------------------
    # Main entry
    # ------------------------------------------------------------------
    def unify(
        self,
        pipeline: dict,
        composite_image: torch.Tensor,
        mode: str,
        user_prompt: str,
        prompt_mode: str = "append",
        composer_layers: Optional[dict] = None,
        layer_visibility_threshold: float = 0.08,
        layer_max_count: int = 4,
        seam_mask: Optional[torch.Tensor] = None,
        foreground_mask: Optional[torch.Tensor] = None,
        manual_mask: Optional[torch.Tensor] = None,
        seam_dilate_px: int = 16,
        ground_contact_px: int = 48,
        seam_feather_px: int = 16,
        subject_protect: bool = True,
        subject_edge_blend_px: int = 1,
        segment_1: Optional[torch.Tensor] = None,
        segment_2: Optional[torch.Tensor] = None,
        segment_3: Optional[torch.Tensor] = None,
        segment_4: Optional[torch.Tensor] = None,
        segment_5: Optional[torch.Tensor] = None,
        vl_segments: bool = True,
        ref_segments: bool = False,
        negative_prompt: str = "",
        steps: int = 8,
        true_cfg_scale: float = 3.0,
        seed: int = 0,
        max_mp: float = 8.0,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        import numpy as np

        pipe = pipeline["pipeline"]
        offload_vae = pipeline.get("offload_vae", False)

        # ---- Composite to PIL (kept at native size) ----
        if composite_image.dim() == 4:
            comp_pil = tensor_to_pil(composite_image[0])
        else:
            comp_pil = tensor_to_pil(composite_image)
        comp_pil = prepare_image_for_pipeline(comp_pil)
        comp_w, comp_h = comp_pil.size

        # ---- Build the prompt ----
        user_prompt = (user_prompt or "").strip()
        if prompt_mode == "replace" and user_prompt:
            final_prompt = user_prompt
        elif user_prompt:
            final_prompt = f"{_DEFAULT_HARMONIZE_PROMPT} {user_prompt}"
        else:
            final_prompt = _DEFAULT_HARMONIZE_PROMPT

        # ---- Build the input image list (composite is image_1 / main) ----
        seg_tensors = [segment_1, segment_2, segment_3, segment_4, segment_5]
        pil_images = [comp_pil]
        image_vl_flags = [True]
        image_ref_flags = [True]  # composite must be the seed

        # In harmonize_only mode, NEVER add segments or relayed layers — the
        # composite alone is the only input. Extra refs anchor subjects to
        # their original (mismatched) lighting and fight harmonisation.
        relay_used = False
        if mode == "harmonize_only":
            print("[EricQwenEdit][Unify] harmonize_only: sending composite only (no segments / relay)")
        elif composer_layers and isinstance(composer_layers, dict) \
                and composer_layers.get("entries") and layer_max_count > 0:
            entries = list(composer_layers["entries"])
            # Filter by visibility, sort highest-first, cap.
            entries = [e for e in entries
                       if float(e.get("visibility", 0.0)) >= float(layer_visibility_threshold)]
            entries.sort(key=lambda e: float(e.get("visibility", 0.0)), reverse=True)
            entries = entries[: int(layer_max_count)]
            for e in entries:
                arr = e.get("rgb")
                if arr is None:
                    continue
                # arr is (h,w,3) float32 [0,1]; convert to PIL RGB.
                import numpy as np
                from PIL import Image
                pil = Image.fromarray(
                    np.clip(arr * 255.0, 0, 255).astype("uint8"), "RGB"
                )
                pil_images.append(prepare_image_for_pipeline(pil))
                image_vl_flags.append(bool(vl_segments))
                image_ref_flags.append(bool(ref_segments))
            relay_used = True
            print(f"[EricQwenEdit][Unify] using {len(entries)} relayed layer(s) from Composer")
            for e in entries:
                print(f"[EricQwenEdit][Unify]   - {e.get('label','?')}: "
                      f"visibility={float(e.get('visibility',0))*100:.1f}%  "
                      f"bbox={e.get('bbox')}")
        elif mode != "harmonize_only":
            for seg in seg_tensors:
                if seg is None:
                    continue
                if seg.dim() == 4:
                    seg_pil = tensor_to_pil(seg[0])
                else:
                    seg_pil = tensor_to_pil(seg)
                pil_images.append(prepare_image_for_pipeline(seg_pil))
                image_vl_flags.append(bool(vl_segments))
                image_ref_flags.append(bool(ref_segments))

        num_images = len(pil_images)

        # ---- Build the post-composite mask (only used in seam_inpaint mode) ----
        if mode == "seam_inpaint":
            used_mask_np = self._build_mask(
                comp_w, comp_h,
                seam_mask, foreground_mask, manual_mask,
                seam_dilate_px, ground_contact_px, seam_feather_px,
                subject_protect=subject_protect,
                subject_edge_blend_px=subject_edge_blend_px,
            )
        else:
            used_mask_np = np.ones((comp_h, comp_w), dtype=np.float32)

        coverage = float(np.mean(used_mask_np)) * 100.0
        print(f"[EricQwenEdit][Unify] mode={mode}  composite={comp_w}x{comp_h}  "
              f"images={num_images}  mask_coverage={coverage:.1f}%")
        print(f"[EricQwenEdit][Unify] steps={steps}  cfg={true_cfg_scale}  seed={seed}  "
              f"max_mp={max_mp}")
        print(f"[EricQwenEdit][Unify] prompt: {final_prompt[:160]}...")

        # ---- Run Qwen-Edit (composite as main, full denoise, ref-anchored) ----
        device = next(pipe.transformer.parameters()).device
        generator = torch.Generator(device=device).manual_seed(seed)

        vae_device_original = None
        if offload_vae:
            vae_device_original = next(pipe.vae.parameters()).device
            if str(vae_device_original) == "cpu":
                print("[EricQwenEdit][Unify] Moving VAE to GPU for encode/decode...")
                pipe.vae = pipe.vae.to(device)

        pbar = comfy.utils.ProgressBar(steps)
        def _progress_callback(_pipe, _step, _t, cb_kwargs):
            pbar.update(1)
            return cb_kwargs

        start = time.time()
        try:
            with torch.inference_mode():
                output = pipe(
                    prompt=final_prompt,
                    image=pil_images if num_images > 1 else pil_images[0],
                    max_pixels=int(max_mp * 1024 * 1024),
                    negative_prompt=negative_prompt if negative_prompt else " ",
                    num_inference_steps=steps,
                    true_cfg_scale=true_cfg_scale,
                    generator=generator,
                    num_images_per_prompt=1,
                    # Per-image conditioning controls (multi_image extension)
                    vae_target_size=None,
                    main_image_index=0,
                    image_vl_flags=image_vl_flags,
                    image_ref_flags=image_ref_flags,
                    callback_on_step_end=_progress_callback,
                    callback_on_step_end_tensor_inputs=["latents"],
                )
        finally:
            if offload_vae and vae_device_original is not None and str(vae_device_original) == "cpu":
                print("[EricQwenEdit][Unify] Moving VAE back to CPU...")
                pipe.vae = pipe.vae.to("cpu")
                torch.cuda.empty_cache()

        elapsed = time.time() - start
        generated_pil = output.images[0]

        # ---- Post-composite (seam_inpaint mode), or pass-through (whole) ----
        if mode == "seam_inpaint":
            result_pil = _composite_with_mask_pil(comp_pil, generated_pil, used_mask_np)
        else:
            # Whole-image: still resize generated to composite size for stability
            from PIL import Image
            if generated_pil.size != comp_pil.size:
                result_pil = generated_pil.resize(comp_pil.size, Image.LANCZOS)
            else:
                result_pil = generated_pil

        result_tensor = pil_to_tensor(result_pil).unsqueeze(0)

        # Mask out (1,H,W)
        used_mask_t = torch.from_numpy(used_mask_np[None].astype("float32")).contiguous()

        print(f"[EricQwenEdit][Unify] done in {elapsed:.1f}s  ({elapsed/max(steps,1):.2f} s/step)  "
              f"out={result_pil.size[0]}x{result_pil.size[1]}")

        return (result_tensor, used_mask_t)
