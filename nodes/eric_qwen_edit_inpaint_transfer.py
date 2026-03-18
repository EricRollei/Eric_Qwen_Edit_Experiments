# Copyright (c) 2026 Eric Hiss. All rights reserved.
# Licensed under the terms in LICENSE.txt (CC BY-NC 4.0 / Commercial dual license).
# https://github.com/EricRollei/Eric_Qwen_Edit_Experiments
"""
Eric Qwen-Edit Inpaint Transfer Node

Transfers content from a reference/transfer image into the masked region
of the original.  The transfer image is resized to fill the mask's
bounding-box, pre-composited into the original, and then the edit model
*harmonises* the rough paste into the scene.  After generation the result
is composited with the original so pixels outside the mask stay untouched.

Model Credits:
- Qwen-Image-Edit developed by Qwen Team (Alibaba)
- https://github.com/QwenLM/Qwen-Image

Author: Eric Hiss (GitHub: EricRollei)
"""

import torch
import time
from typing import Tuple

import comfy.utils

from .eric_qwen_edit_utils import (
    tensor_to_pil,
    pil_to_tensor,
    prepare_image_for_pipeline,
)

# Reuse helpers from the inpaint node
from .eric_qwen_edit_inpaint import _composite_with_mask


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _bbox(mask_np, threshold=127):
    """Return (y_min, y_max, x_min, x_max) of pixels > threshold, or None."""
    import numpy as np
    region = mask_np > threshold
    if not region.any():
        return None
    rows = np.any(region, axis=1)
    cols = np.any(region, axis=0)
    y_min, y_max = np.where(rows)[0][[0, -1]]
    x_min, x_max = np.where(cols)[0][[0, -1]]
    return y_min, y_max, x_min, x_max


def _prefill_with_scaled_transfer(
    original, transfer, target_mask,
    transfer_mask=None, blend_edges=True,
):
    """
    Scale the transfer image so its masked region fits inside the target
    mask's bounding-box, then composite it into *original*.

    When *transfer_mask* is provided
    --------------------------------
    1.  Bounding-box both masks.
    2.  Compute ``scale = min(target_w/src_w, target_h/src_h)`` so the
        source masked region fits inside the target region (no skew).
    3.  Scale the **entire** transfer image + transfer_mask by that factor.
    4.  Extract only the masked pixels from the scaled transfer.
    5.  Center those pixels in the target bounding-box and composite.

    When *transfer_mask* is ``None``
    --------------------------------
    The whole transfer image is treated as the source content and scaled
    to fit inside the target mask's bounding-box the same way.

    Returns:
        (prefilled_image, transfer_for_vl)
        prefilled_image:  PIL RGB — original with transfer pasted in.
        transfer_for_vl:  PIL RGB — the (optionally cropped) transfer
                          snippet for use as a VL reference image.
    """
    from PIL import Image, ImageFilter
    import numpy as np

    target_mask_np = np.array(target_mask)
    target_box = _bbox(target_mask_np)
    if target_box is None:
        return original.copy(), transfer.copy()

    ty_min, ty_max, tx_min, tx_max = target_box
    target_w = tx_max - tx_min + 1
    target_h = ty_max - ty_min + 1

    if transfer_mask is not None:
        # ---- Scale based on masked region sizes ----
        if transfer_mask.size != transfer.size:
            transfer_mask = transfer_mask.resize(transfer.size, Image.NEAREST)

        src_mask_np = np.array(transfer_mask)
        src_box = _bbox(src_mask_np)
        if src_box is None:
            # Empty transfer mask — fall back to full image
            transfer_mask = None
            # Fall through to the no-mask path below

    if transfer_mask is not None:
        sy_min, sy_max, sx_min, sx_max = src_box
        src_w = sx_max - sx_min + 1
        src_h = sy_max - sy_min + 1

        # Scale so source masked region fits inside target masked region
        scale = min(target_w / src_w, target_h / src_h)
        new_full_w = max(int(round(transfer.size[0] * scale)), 1)
        new_full_h = max(int(round(transfer.size[1] * scale)), 1)

        scaled_transfer = transfer.resize((new_full_w, new_full_h), Image.LANCZOS)
        scaled_tmask = transfer_mask.resize((new_full_w, new_full_h), Image.NEAREST)

        # Recompute the source bbox in scaled coordinates
        scaled_sx = int(round(sx_min * scale))
        scaled_sy = int(round(sy_min * scale))
        scaled_sw = int(round(src_w * scale))
        scaled_sh = int(round(src_h * scale))

        print(f"[EricQwenEdit] Transfer region: {src_w}x{src_h} -> "
              f"scaled {scaled_sw}x{scaled_sh} to fit target {target_w}x{target_h}")

        # Build a canvas the size of the original, place the scaled
        # transfer so its masked region is centered in the target bbox.
        # Offset: where the scaled source bbox top-left goes relative
        # to the target bbox top-left
        offset_x = tx_min + (target_w - scaled_sw) // 2 - scaled_sx
        offset_y = ty_min + (target_h - scaled_sh) // 2 - scaled_sy

        canvas = Image.new("RGB", original.size, (0, 0, 0))
        canvas.paste(scaled_transfer, (offset_x, offset_y))

        mask_canvas = Image.new("L", original.size, 0)
        mask_canvas.paste(scaled_tmask, (offset_x, offset_y))

        # Only keep pixels inside BOTH the scaled transfer mask AND
        # the target mask (intersection)
        canvas_np = np.array(canvas, dtype=np.float32)
        tmask_canvas_np = np.array(mask_canvas, dtype=np.float32) / 255.0
        target_mask_f = np.array(target_mask, dtype=np.float32) / 255.0
        if target_mask.size != original.size:
            target_mask_resized = target_mask.resize(original.size, Image.NEAREST)
            target_mask_f = np.array(target_mask_resized, dtype=np.float32) / 255.0
        alpha = np.minimum(tmask_canvas_np, target_mask_f)

        # Build VL reference: the scaled transfer cropped to the source
        # masked region's bounding box
        vl_crop_x = max(scaled_sx + offset_x, 0)  # noqa: never used directly
        # Actually crop from the scaled transfer directly
        vl_ref = scaled_transfer

    else:
        # ---- No transfer mask: scale full transfer to fit target bbox ----
        t_w, t_h = transfer.size
        scale = min(target_w / t_w, target_h / t_h)
        new_w = max(int(round(t_w * scale)), 1)
        new_h = max(int(round(t_h * scale)), 1)
        resized = transfer.resize((new_w, new_h), Image.LANCZOS)

        print(f"[EricQwenEdit] Transfer (no mask): {t_w}x{t_h} -> "
              f"scaled {new_w}x{new_h} to fit target {target_w}x{target_h}")

        # Center in target bbox
        paste_x = tx_min + (target_w - new_w) // 2
        paste_y = ty_min + (target_h - new_h) // 2

        canvas = Image.new("RGB", original.size, (0, 0, 0))
        canvas.paste(resized, (paste_x, paste_y))
        canvas_np = np.array(canvas, dtype=np.float32)

        # Alpha = target mask only (the whole resized transfer is content)
        target_mask_f = np.array(target_mask, dtype=np.float32) / 255.0
        if target_mask.size != original.size:
            target_mask_resized = target_mask.resize(original.size, Image.NEAREST)
            target_mask_f = np.array(target_mask_resized, dtype=np.float32) / 255.0
        alpha = target_mask_f
        vl_ref = resized

    # Gentle edge blur for pre-fill
    if blend_edges:
        alpha_uint8 = np.clip(alpha * 255, 0, 255).astype(np.uint8)
        alpha_pil = Image.fromarray(alpha_uint8, "L")
        alpha_pil = alpha_pil.filter(ImageFilter.GaussianBlur(radius=3))
        alpha = np.array(alpha_pil, dtype=np.float32) / 255.0

    alpha = alpha[:, :, np.newaxis]
    orig_np = np.array(original.convert("RGB"), dtype=np.float32)
    merged = orig_np * (1.0 - alpha) + canvas_np * alpha
    prefilled = Image.fromarray(np.clip(merged, 0, 255).astype(np.uint8), "RGB")

    return prefilled, vl_ref


class EricQwenEditInpaintTransfer:
    """
    Inpaint Transfer — blend content from a reference/transfer image
    into the masked area of the original.

    Pre-composites the transfer into the mask region so the model sees
    the content at the right location, then harmonises/blends it.
    Post-composite preserves original pixels outside the mask.

    Use cases:
    - Paste a logo/graphic from one image into a specific spot
    - Transfer a texture or pattern into a region
    - Blend a face, object, or scene element from a reference
    - Replace a background section with content from another photo

    Mask format:
    - White (255) = area to fill with transfer content
    - Black (0) = keep original pixels
    """

    CATEGORY = "Eric Qwen-Edit"
    FUNCTION = "inpaint_transfer"
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "pipeline": ("QWEN_EDIT_PIPELINE",),
                "image": ("IMAGE", {
                    "tooltip": "Original image to edit",
                }),
                "mask": ("MASK", {
                    "tooltip": "Mask: white = area to fill with transfer content, black = keep original",
                }),
                "transfer_image": ("IMAGE", {
                    "tooltip": (
                        "Reference/transfer image whose content will be blended "
                        "into the masked area of the original."
                    ),
                }),
            },
            "optional": {
                "transfer_mask": ("MASK", {
                    "tooltip": (
                        "Optional mask on the transfer image: white = region "
                        "to extract. When provided, only the selected part of "
                        "the transfer image is used. When omitted, the full "
                        "transfer image is used."
                    ),
                }),
                "prompt": ("STRING", {
                    "multiline": True,
                    "default": "",
                    "tooltip": (
                        "Optional guidance text. If empty, a default prompt is used "
                        "that tells the model to harmonise the inserted content. "
                        "Add detail to steer the blend."
                    ),
                }),
                "negative_prompt": ("STRING", {
                    "multiline": True,
                    "default": "",
                    "tooltip": "What to avoid",
                }),
                "transfer_vl_ref": ("BOOLEAN", {
                    "default": True,
                    "tooltip": (
                        "Also send the full (un-cropped) transfer image as a "
                        "second VL reference. Gives the model extra semantic "
                        "context about the transfer content. Disable if the "
                        "pre-composite alone is sufficient."
                    ),
                }),
                "blend_strength": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.05,
                    "tooltip": (
                        "Blend strength for the post-compositing. 1.0 = fully "
                        "replace masked area with generated content. Lower values "
                        "mix in some of the original (useful for subtle blends)."
                    ),
                }),
                "feather": ("INT", {
                    "default": 8,
                    "min": 0,
                    "max": 50,
                    "step": 1,
                    "tooltip": (
                        "Gaussian blur radius for mask-edge feathering. "
                        "Higher = softer blend at mask edges. 0 = hard cut."
                    ),
                }),
                "steps": ("INT", {
                    "default": 8,
                    "min": 1,
                    "max": 100,
                    "step": 1,
                    "tooltip": "Inference steps (8 for lightning LoRA, 50 for base model)",
                }),
                "true_cfg_scale": ("FLOAT", {
                    "default": 4.0,
                    "min": 1.0,
                    "max": 20.0,
                    "step": 0.5,
                    "tooltip": "True CFG scale",
                }),
                "seed": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 0xffffffffffffffff,
                    "tooltip": "Random seed",
                }),
                "max_mp": ("FLOAT", {
                    "default": 8.0,
                    "min": 0.5,
                    "max": 16.0,
                    "step": 0.5,
                    "tooltip": "Max output megapixels.",
                }),
            },
        }

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return kwargs.get("seed", 0)

    def inpaint_transfer(
        self,
        pipeline: dict,
        image: torch.Tensor,
        mask: torch.Tensor,
        transfer_image: torch.Tensor,
        transfer_mask: torch.Tensor = None,
        prompt: str = "",
        negative_prompt: str = "",
        transfer_vl_ref: bool = True,
        blend_strength: float = 1.0,
        feather: int = 8,
        steps: int = 8,
        true_cfg_scale: float = 4.0,
        seed: int = 0,
        max_mp: float = 8.0,
    ) -> Tuple[torch.Tensor]:
        """
        Inpaint-transfer: pre-composite transfer into mask area, let the
        model harmonise, then post-composite to keep original outside mask.
        """
        pipe = pipeline["pipeline"]
        offload_vae = pipeline.get("offload_vae", False)

        # --- Convert original image ---
        if image.dim() == 4:
            pil_image = tensor_to_pil(image[0])
        else:
            pil_image = tensor_to_pil(image)
        pil_image = prepare_image_for_pipeline(pil_image)

        # --- Convert transfer image ---
        if transfer_image.dim() == 4:
            pil_transfer = tensor_to_pil(transfer_image[0])
        else:
            pil_transfer = tensor_to_pil(transfer_image)
        pil_transfer = prepare_image_for_pipeline(pil_transfer)

        # --- Convert mask ---
        if mask.dim() == 3:
            mask_np = mask[0].cpu().numpy()
        else:
            mask_np = mask.cpu().numpy()

        from PIL import Image
        import numpy as np

        mask_pil = Image.fromarray((mask_np * 255).astype(np.uint8), mode="L")

        # --- Convert transfer mask if provided ---
        tmask_pil = None
        if transfer_mask is not None:
            if transfer_mask.dim() == 3:
                tmask_np = transfer_mask[0].cpu().numpy()
            else:
                tmask_np = transfer_mask.cpu().numpy()
            tmask_pil = Image.fromarray((tmask_np * 255).astype(np.uint8), mode="L")

        # --- Pre-composite: scale transfer and paste into the mask area ---
        # Bounding-box both masks, scale the whole transfer proportionally
        # so the source masked region fits inside the target masked region,
        # then composite into the original.
        prefilled_image, vl_ref_image = _prefill_with_scaled_transfer(
            pil_image, pil_transfer, mask_pil,
            transfer_mask=tmask_pil, blend_edges=True,
        )

        input_w, input_h = pil_image.size
        input_mp = (input_w * input_h) / (1024 * 1024)
        transfer_w, transfer_h = pil_transfer.size

        # --- Build prompt ---
        if prompt and prompt.strip():
            enhanced_prompt = (
                f"Naturally blend and harmonise the edited area with "
                f"the surrounding image. {prompt}"
            )
        else:
            enhanced_prompt = (
                f"Harmonise and naturally blend the inserted content so "
                f"it looks like a seamless part of the original image. "
                f"Match lighting, color tone, and perspective."
            )

        print(f"[EricQwenEdit] Inpaint Transfer (vl_ref={transfer_vl_ref})")
        print(f"[EricQwenEdit] Original: {input_w}x{input_h} ({input_mp:.1f}MP)")
        print(f"[EricQwenEdit] Transfer: {transfer_w}x{transfer_h}")
        print(f"[EricQwenEdit] Steps: {steps}, CFG: {true_cfg_scale}, Feather: {feather}")
        print(f"[EricQwenEdit] Prompt: {enhanced_prompt[:120]}...")

        device = next(pipe.transformer.parameters()).device
        generator = torch.Generator(device=device).manual_seed(seed)

        # VAE offload handling
        vae_device_original = None
        if offload_vae:
            vae_device_original = next(pipe.vae.parameters()).device
            if str(vae_device_original) == "cpu":
                print("[EricQwenEdit] Moving VAE to GPU for encode/decode...")
                pipe.vae = pipe.vae.to(device)

        start_time = time.time()

        # ComfyUI progress bar
        pbar = comfy.utils.ProgressBar(steps)

        def _progress_callback(pipeline, step_index, timestep, cb_kwargs):
            pbar.update(1)
            return cb_kwargs

        try:
            with torch.inference_mode():
                # Build image list:
                #   [0] prefilled_image — VL + VAE (transfer already in place)
                #   [1] vl_ref_image   — VL only (extra semantic context, optional)
                if transfer_vl_ref:
                    output = pipe(
                        prompt=enhanced_prompt,
                        image=[prefilled_image, vl_ref_image],
                        max_pixels=int(max_mp * 1024 * 1024),
                        negative_prompt=negative_prompt if negative_prompt else " ",
                        num_inference_steps=steps,
                        true_cfg_scale=true_cfg_scale,
                        generator=generator,
                        num_images_per_prompt=1,
                        callback_on_step_end=_progress_callback,
                        callback_on_step_end_tensor_inputs=["latents"],
                        image_vl_flags=[True, True],
                        image_ref_flags=[True, False],  # only prefilled goes to VAE
                        main_image_index=0,
                    )
                else:
                    # Single image: just the pre-filled composite
                    output = pipe(
                        prompt=enhanced_prompt,
                        image=[prefilled_image],
                        max_pixels=int(max_mp * 1024 * 1024),
                        negative_prompt=negative_prompt if negative_prompt else " ",
                        num_inference_steps=steps,
                        true_cfg_scale=true_cfg_scale,
                        generator=generator,
                        num_images_per_prompt=1,
                        callback_on_step_end=_progress_callback,
                        callback_on_step_end_tensor_inputs=["latents"],
                    )
        finally:
            if offload_vae and vae_device_original is not None and str(vae_device_original) == "cpu":
                print("[EricQwenEdit] Moving VAE back to CPU...")
                pipe.vae = pipe.vae.to("cpu")
                torch.cuda.empty_cache()

        elapsed = time.time() - start_time

        generated = output.images[0]

        # --- Post-composite ---
        # Apply blend_strength: scale the mask so <1.0 partially preserves original
        if blend_strength < 1.0:
            mask_arr = np.array(mask_pil, dtype=np.float32)
            mask_arr = mask_arr * blend_strength
            mask_for_comp = Image.fromarray(mask_arr.astype(np.uint8), mode="L")
        else:
            mask_for_comp = mask_pil

        result = _composite_with_mask(pil_image, generated, mask_for_comp, feather_radius=feather)

        result_tensor = pil_to_tensor(result).unsqueeze(0)

        print(f"[EricQwenEdit] Inpaint Transfer complete: {result.size[0]}x{result.size[1]}")
        print(f"[EricQwenEdit] Actual time: {elapsed / 60:.1f} minutes ({elapsed / steps:.1f} sec/step)")

        return (result_tensor,)
