# Copyright (c) 2026 Eric Hiss. All rights reserved.
# Licensed under the terms in LICENSE.txt (CC BY-NC 4.0 / Commercial dual license).
# https://github.com/EricRollei/Eric_Qwen_Edit_Experiments
"""
Eric Qwen-Image Harmonize node.

Img2img harmonisation pass for composited / collaged images using the
text-to-image Qwen-Image base (NOT Qwen-Image-Edit) plus a structural
ControlNet.

Conceptual flow:
    composite ──▶ VAE.encode ──▶ latent
    latent + sigma * noise ──▶ noisy latent      (partial denoise)
    noisy latent ──▶ Qwen-Image transformer ──▶ final latent
                    └─ ControlNet guides edges/structure ─┘
    final latent ──▶ VAE.decode ──▶ output

Why this works for harmonisation
--------------------------------
Qwen-Image-Edit always starts from random noise with the source as a *soft*
reference, so it cannot do "low-strength" img2img — output is always
re-imagined and softened by 2× VAE round-trips. Qwen-Image (base text-to-image)
DOES expose latent input, so we can:
  1. encode the composite → exactly its latent
  2. add only a fraction of noise (e.g. 0.35 strength)
  3. run only the lower portion of the sigma schedule
  4. decode

At denoise ≤ 0.4 the structure is essentially locked; only colour, lighting
and contact-shadow details change. ControlNet on top adds a hard structural
constraint (edges or depth) so the model cannot move, delete or distort any
subject.

Author: Eric Hiss (GitHub: EricRollei)
"""

from __future__ import annotations

import math
import time
from typing import Optional, Tuple

import numpy as np
import torch
from PIL import Image

import comfy.utils

from .eric_qwen_edit_utils import pil_to_tensor, tensor_to_pil
from .eric_qwen_image_multistage import (
    _add_noise_flowmatch,
    _check_cancelled,
    _compute_actual_start_sigma,
    _compute_mu,
    _packed_seq_len,
    build_sigma_schedule,
)
from .eric_qwen_image_ultragen import DEFAULT_NEGATIVE_PROMPT
from .eric_qwen_upscale_vae import decode_latents_with_upscale_vae


# ---------------------------------------------------------------------------
# Default preservation-strong harmonise prompt (used if no prompt is wired)
# ---------------------------------------------------------------------------

_DEFAULT_HARMONIZE_PROMPT = (
    "A single cohesive photograph. All subjects are lit by the same light "
    "source with consistent direction, color temperature, exposure and "
    "white balance. Natural cast shadows and contact shadows where each "
    "subject meets the ground, water or another surface. Realistic water "
    "interaction including ripples, reflections and partial submersion "
    "where applicable. Matched atmospheric perspective, depth of field "
    "and film grain across the entire frame. Continuous photographic "
    "rendering, no compositing artefacts, no hard cut-out edges."
)


# ---------------------------------------------------------------------------
# Control-image preprocessors (auto modes)
# ---------------------------------------------------------------------------

def _np_uint8_from_image_tensor(img: torch.Tensor) -> np.ndarray:
    if img.dim() == 4:
        img = img[0]
    a = img.detach().cpu().numpy()
    return np.clip(a * 255.0, 0, 255).astype(np.uint8)


def _auto_canny(img_np: np.ndarray, low: int = 100, high: int = 200) -> np.ndarray:
    """Standard cv2 Canny edges. Returns a 3-channel uint8 image."""
    import cv2
    gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, int(low), int(high))
    edges3 = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)
    return edges3


def _auto_soft_edge(img_np: np.ndarray) -> np.ndarray:
    """Lightweight HED-style soft edge map (no extra model).

    Combines a small-radius Laplacian + Sobel magnitude, normalised. Produces
    softer, more gradient-aware edges than Canny — closer to what HED/PiDiNet
    would output.
    """
    import cv2
    gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    # Mild blur to suppress film grain
    gray = cv2.GaussianBlur(gray, (3, 3), 0.6)
    # Sobel magnitude
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    mag = np.sqrt(gx * gx + gy * gy)
    # Laplacian magnitude (catches finer details)
    lap = np.abs(cv2.Laplacian(gray, cv2.CV_32F, ksize=3))
    edges = 0.7 * mag + 0.3 * lap
    if edges.max() > 0:
        edges = edges / edges.max()
    edges = np.clip(edges ** 0.6, 0.0, 1.0)  # soft gamma to lift mid-tones
    edges_u8 = (edges * 255.0).astype(np.uint8)
    return cv2.cvtColor(edges_u8, cv2.COLOR_GRAY2RGB)


# ---------------------------------------------------------------------------
# Resize helpers (size aligned to VAE)
# ---------------------------------------------------------------------------

def _align_to(value: int, multiple: int) -> int:
    return max(multiple, (int(value) // multiple) * multiple)


def _resize_for_pipeline(
    img: Image.Image, max_pixels: int, vae_scale_factor: int
) -> Image.Image:
    """Resize PIL image to fit within max_pixels while preserving aspect.

    Output dimensions are aligned to (vae_scale_factor * 2) so packed
    latents work cleanly.
    """
    align = vae_scale_factor * 2
    w, h = img.size
    px = w * h
    if px > max_pixels:
        scale = math.sqrt(max_pixels / px)
        w = max(align, int(w * scale))
        h = max(align, int(h * scale))
    w = _align_to(w, align)
    h = _align_to(h, align)
    if (w, h) != img.size:
        img = img.resize((w, h), Image.LANCZOS)
    return img


# ---------------------------------------------------------------------------
# Encode a PIL image to packed Qwen-Image latents
# ---------------------------------------------------------------------------

def _encode_image_to_packed_latents(
    pil: Image.Image, pipe, device, dtype
) -> Tuple[torch.Tensor, int, int]:
    """PIL → packed latents matching the pipeline's normalisation convention.

    Mirrors the encode block in eric_qwen_upscale_vae.upscale_between_stages,
    minus the AutoencoderKLWan path. Uses the pipeline's own AutoencoderKLQwenImage.
    """
    from .eric_qwen_image_multistage import _pack_latents

    vae = pipe.vae
    z_dim = vae.config.z_dim
    mean = torch.tensor(vae.config.latents_mean).view(1, z_dim, 1, 1, 1)
    std = torch.tensor(vae.config.latents_std).view(1, z_dim, 1, 1, 1)

    arr = np.asarray(pil.convert("RGB"), dtype=np.float32) / 255.0  # H,W,3 in [0,1]
    arr = arr * 2.0 - 1.0                                            # to [-1,1]
    px = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)         # (1,3,H,W)
    # VAE expects (B, C, frames, H, W) for video AE
    px = px.unsqueeze(2).to(device=device, dtype=dtype)              # (1,3,1,H,W)

    vae_was_on = next(vae.parameters()).device
    if str(vae_was_on) != str(device):
        vae.to(device)
    try:
        with torch.no_grad():
            posterior = vae.encode(px).latent_dist
            raw = posterior.mode()                                   # (1,z,1,H',W')
        mean_d = mean.to(device=device, dtype=dtype)
        std_d = std.to(device=device, dtype=dtype)
        norm = (raw - mean_d) / std_d
        packed = _pack_latents(norm)
    finally:
        # Caller manages VAE placement; don't move it back.
        pass
    return packed, pil.size[0], pil.size[1]


# ---------------------------------------------------------------------------
# Node
# ---------------------------------------------------------------------------

class EricQwenImageHarmonize:
    """
    Img2img harmonisation pass for composited images using
    Qwen-Image (text-to-image base) + InstantX ControlNet-Union.
    """

    CATEGORY = "Eric Qwen-Image"
    FUNCTION = "harmonize"
    RETURN_TYPES = ("IMAGE", "IMAGE")
    RETURN_NAMES = ("image", "control_image_used")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "pipeline": ("QWEN_IMAGE_PIPELINE", {
                    "tooltip": "From the Qwen-Image loader (Qwen-Image base, NOT Qwen-Image-Edit)."
                }),
                "controlnet": ("QWEN_IMAGE_CONTROLNET", {
                    "tooltip": "From the Qwen-Image ControlNet Loader. Use the InstantX Union model."
                }),
                "composite_image": ("IMAGE", {
                    "tooltip": "The composited image to harmonise (e.g. ImageComposer output)."
                }),
                "prompt": ("STRING", {
                    "multiline": True,
                    "default": "",
                    "tooltip": "Description of the unified scene. If empty, a generic harmonisation "
                               "preamble is used. RECOMMENDED: wire from the Qwen ControlNet Prompt "
                               "Rewriter with the composite as input \u2014 it will list every subject "
                               "in place so Qwen cannot delete them."
                }),
            },
            "optional": {
                "cn_mode": (["auto_canny", "auto_soft_edge", "external"], {
                    "default": "auto_canny",
                    "tooltip": "How to obtain the control image:\n"
                               "auto_canny: cv2 Canny edges of the composite (default, robust).\n"
                               "auto_soft_edge: cv2 Sobel+Laplacian soft edges (smoother textures).\n"
                               "external: use the wired control_image input (depth, pose, custom canny, etc.)."
                }),
                "control_image": ("IMAGE", {
                    "tooltip": "External control image. Used only when cn_mode = external."
                }),
                "canny_low": ("INT", {
                    "default": 100, "min": 0, "max": 255, "step": 1,
                    "tooltip": "Low threshold for cv2.Canny (auto_canny only)."
                }),
                "canny_high": ("INT", {
                    "default": 200, "min": 0, "max": 255, "step": 1,
                    "tooltip": "High threshold for cv2.Canny (auto_canny only)."
                }),
                "denoise": ("FLOAT", {
                    "default": 0.42, "min": 0.05, "max": 1.0, "step": 0.05,
                    "tooltip": "How much of the schedule to run.\n"
                               "0.20\u20130.30: very light \u2014 just colour/grain matching.\n"
                               "0.30\u20130.45: typical harmonisation \u2014 lighting + soft contact shadows.\n"
                               "0.45\u20130.60: stronger relight, may shift small details.\n"
                               "0.70+: nearly full re-render \u2014 use only if structure is locked by CN."
                }),
                "sigma_schedule": (["linear", "balanced", "karras"], {
                    "default": "linear",
                    "tooltip": "Step distribution within the active denoise range. balanced is safest."
                }),
                "cn_strength": ("FLOAT", {
                    "default": 1.0, "min": 0.0, "max": 5.0, "step": 0.1,
                    "tooltip": "ControlNet conditioning scale. 1.0 standard. >1.5 over-constrains."
                }),
                "cn_start": ("FLOAT", {
                    "default": 0.0, "min": 0.0, "max": 1.0, "step": 0.05,
                    "tooltip": "Fraction of steps at which CN guidance begins."
                }),
                "cn_end": ("FLOAT", {
                    "default": 0.7, "min": 0.0, "max": 1.0, "step": 0.05,
                    "tooltip": "Fraction of steps at which CN guidance ends. Stopping at 0.8 lets the "
                               "last 20%% of steps refine textures without CN over-constraining detail."
                }),
                "negative_prompt": ("STRING", {
                    "multiline": True,
                    "default": DEFAULT_NEGATIVE_PROMPT,
                    "tooltip": "What to avoid (default = official Qwen-Image negative)."
                }),
                "steps": ("INT", {
                    "default": 40, "min": 4, "max": 100, "step": 1,
                    "tooltip": "Total inference steps. 20–28 is a good range; effective denoise "
                               "steps = round(steps * denoise)."
                }),
                "true_cfg_scale": ("FLOAT", {
                    "default": 4.5, "min": 1.0, "max": 12.0, "step": 0.5,
                    "tooltip": "True CFG. 2.5\u20133.5 is best for harmonisation."
                }),
                "max_sequence_length": ("INT", {
                    "default": 1024, "min": 128, "max": 1024, "step": 64,
                    "tooltip": "Max prompt token length for the text encoder. 1024 is the Qwen-Image "
                               "max and has negligible runtime cost — lets long override prompts "
                               "(per-subject preservation lists) survive without truncation."
                }),
                "max_mp": ("FLOAT", {
                    "default": 6.0, "min": 0.5, "max": 16.0, "step": 0.5,
                    "tooltip": "Max megapixels at which to run the harmonisation. The composite is "
                               "resized to fit; output is upsampled back to composite size with "
                               "Lanczos. Qwen-Image natively handles up to ~16 MP. Lower this "
                               "(e.g. 4–8 MP) if you hit VRAM limits."
                }),
                "seed": ("INT", {
                    "default": 0, "min": 0, "max": 0xffffffffffffffff,
                    "tooltip": "Seed (0 = random)."
                }),
                "upscale_vae": ("UPSCALE_VAE", {
                    "tooltip": "Optional. Wire the 2× Wan upscale VAE to replace the final decode "
                               "with a sharper 2× reconstruction. No extra diffusion is done — "
                               "the diffusion result is decoded through the upscale VAE instead "
                               "of the pipeline's standard VAE, then resized down to the "
                               "composite size with Lanczos. Adds ~1–3s and gives crisper "
                               "textures and edges. If unwired, the standard pipeline VAE "
                               "decode is used."
                }),
                "upscale_vae_keep_2x": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "When upscale_vae is wired and this is ON, output is kept at "
                               "2× the working resolution (NOT downsampled to the original "
                               "composite size). Useful when you want the harmoniser to "
                               "also act as a 2× upscaler. Ignored if upscale_vae is not wired."
                }),
                "refine_pass": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Run a second, lighter diffusion pass on the result of pass 1.\n"
                               "The pass-1 latent stays on the GPU \u2014 NO VAE re-encode round-trip \u2014 "
                               "so detail is not lost between passes.\n"
                               "Pass 1 = structural harmonisation (lighting, shadows, water blends).\n"
                               "Pass 2 = detail refinement (faces, fabric, marble texture).\n"
                               "Two gentle passes consistently beat one heavier pass."
                }),
                "refine_denoise": ("FLOAT", {
                    "default": 0.30, "min": 0.05, "max": 0.6, "step": 0.05,
                    "tooltip": "Pass-2 denoise. Keep low (0.18\u20130.28). Pass 1 already locked "
                               "structure; pass 2 should only sharpen detail."
                }),
                "refine_steps": ("INT", {
                    "default": 40, "min": 4, "max": 80, "step": 1,
                    "tooltip": "Pass-2 total steps. Effective = round(refine_steps * refine_denoise)."
                }),
                "refine_sigma_schedule": (["linear", "balanced", "karras"], {
                    "default": "karras",
                    "tooltip": "Pass-2 sigma schedule. balanced is best for detail formation."
                }),
                "refine_cfg": ("FLOAT", {
                    "default": 4.5, "min": 1.0, "max": 12.0, "step": 0.5,
                    "tooltip": "Pass-2 CFG. Slightly higher than pass 1 \u2014 commits harder to detail "
                               "words like 'fine skin pores', 'individual eyelashes', etc."
                }),
                "refine_cn_strength": ("FLOAT", {
                    "default": 1.0, "min": 0.0, "max": 5.0, "step": 0.1,
                    "tooltip": "Pass-2 ControlNet strength. Lower than pass 1 \u2014 structure is "
                               "already locked, so heavy CN here just over-constrains texture."
                }),
                "refine_cn_start": ("FLOAT", {
                    "default": 0.0, "min": 0.0, "max": 1.0, "step": 0.05,
                    "tooltip": "Pass-2 CN start fraction."
                }),
                "refine_cn_end": ("FLOAT", {
                    "default": 0.5, "min": 0.0, "max": 1.0, "step": 0.05,
                    "tooltip": "Pass-2 CN end fraction. Stop CN early so the last 50%% of pass-2 "
                               "steps are free to refine textures without CN over-constraining."
                }),
                "refine_recanny": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "When ON and cn_mode is auto_canny/auto_soft_edge, re-derive the "
                               "control image from the pass-1 result (now clean photographic edges, "
                               "not collage edges). Costs one extra decode-to-PIL but does NOT "
                               "round-trip the latent. When OFF (or cn_mode = external), pass 2 "
                               "reuses the pass-1 control image."
                }),
            },
        }

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return kwargs.get("seed", 0)

    # ------------------------------------------------------------------
    # Build / fetch the control image at the working resolution
    # ------------------------------------------------------------------
    def _build_control_image(
        self,
        cn_mode: str,
        composite_pil: Image.Image,
        external: Optional[torch.Tensor],
        canny_low: int,
        canny_high: int,
    ) -> Image.Image:
        if cn_mode == "external":
            if external is None:
                raise ValueError(
                    "[EricQwenImageHarmonize] cn_mode=external but no control_image was wired."
                )
            ctrl_arr = _np_uint8_from_image_tensor(external)
            ctrl_pil = Image.fromarray(ctrl_arr, "RGB")
            if ctrl_pil.size != composite_pil.size:
                ctrl_pil = ctrl_pil.resize(composite_pil.size, Image.LANCZOS)
            return ctrl_pil

        # Auto modes: derive from composite
        comp_arr = np.asarray(composite_pil.convert("RGB"))
        if cn_mode == "auto_soft_edge":
            ctrl_arr = _auto_soft_edge(comp_arr)
        else:  # auto_canny (default)
            ctrl_arr = _auto_canny(comp_arr, canny_low, canny_high)
        return Image.fromarray(ctrl_arr, "RGB")

    # ------------------------------------------------------------------
    def harmonize(
        self,
        pipeline: dict,
        controlnet: dict,
        composite_image: torch.Tensor,
        prompt: str,
        cn_mode: str = "auto_canny",
        control_image: Optional[torch.Tensor] = None,
        canny_low: int = 100,
        canny_high: int = 200,
        denoise: float = 0.42,
        sigma_schedule: str = "linear",
        cn_strength: float = 1.0,
        cn_start: float = 0.0,
        cn_end: float = 0.7,
        negative_prompt: str = DEFAULT_NEGATIVE_PROMPT,
        steps: int = 40,
        true_cfg_scale: float = 4.5,
        max_sequence_length: int = 1024,
        max_mp: float = 6.0,
        seed: int = 0,
        upscale_vae=None,
        upscale_vae_keep_2x: bool = True,
        refine_pass: bool = True,
        refine_denoise: float = 0.30,
        refine_steps: int = 40,
        refine_sigma_schedule: str = "karras",
        refine_cfg: float = 4.5,
        refine_cn_strength: float = 1.0,
        refine_cn_start: float = 0.0,
        refine_cn_end: float = 0.5,
        refine_recanny: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        from diffusers import QwenImageControlNetPipeline

        pipe = pipeline["pipeline"]
        cn_model = controlnet["model"]

        # Composite to PIL
        if composite_image.dim() == 4:
            comp_pil_orig = tensor_to_pil(composite_image[0])
        else:
            comp_pil_orig = tensor_to_pil(composite_image)
        orig_w, orig_h = comp_pil_orig.size

        # Working resolution (capped by max_mp)
        vae_scale_factor = pipe.vae_scale_factor if hasattr(pipe, "vae_scale_factor") else 8
        max_pixels = int(max_mp * 1024 * 1024)
        comp_pil = _resize_for_pipeline(comp_pil_orig, max_pixels, vae_scale_factor)
        work_w, work_h = comp_pil.size

        # Control image at working resolution
        ctrl_pil = self._build_control_image(
            cn_mode, comp_pil, control_image, canny_low, canny_high
        )

        # Final prompt
        final_prompt = (prompt or "").strip() or _DEFAULT_HARMONIZE_PROMPT

        # Device / dtype
        device = next(pipe.transformer.parameters()).device
        t_dtype = next(pipe.transformer.parameters()).dtype

        # ControlNet placement
        cn_dev = next(cn_model.parameters()).device
        cn_dt = next(cn_model.parameters()).dtype
        if str(cn_dev) != str(device) or cn_dt != t_dtype:
            print(f"[EricQwenImageHarmonize] Moving ControlNet to {device} ({t_dtype})")
            cn_model.to(device=device, dtype=t_dtype)

        # Build CN pipeline (zero-copy share of components)
        cn_pipe = QwenImageControlNetPipeline(
            scheduler=pipe.scheduler,
            vae=pipe.vae,
            text_encoder=pipe.text_encoder,
            tokenizer=pipe.tokenizer,
            transformer=pipe.transformer,
            controlnet=cn_model,
        )

        # Encode composite → packed latents (single VAE encode for the whole node)
        print(f"[EricQwenImageHarmonize] Encoding composite at {work_w}x{work_h} "
              f"(orig={orig_w}x{orig_h}, max_mp={max_mp})")
        packed, _, _ = _encode_image_to_packed_latents(comp_pil, pipe, device, t_dtype)
        clean_latents = packed.to(device=device, dtype=t_dtype)

        seq_len = _packed_seq_len(work_h, work_w, vae_scale_factor)
        mu = _compute_mu(seq_len, pipe.scheduler)

        seed_int = int(seed)

        # ── Internal helper: one full diffusion pass on supplied clean latents
        def _run_pass(
            tag: str,
            init_clean_latents: torch.Tensor,
            ctrl_pil_in: Image.Image,
            denoise_v: float,
            steps_v: int,
            sched_v: str,
            cfg_v: float,
            cn_s: float,
            cn_a: float,
            cn_b: float,
            seed_v: int,
        ):
            raw_sigmas = build_sigma_schedule(steps_v, denoise_v, sched_v)
            actual_sigma = _compute_actual_start_sigma(pipe.scheduler, raw_sigmas, mu)
            keep_steps = len(raw_sigmas)

            gen = torch.Generator(device=device).manual_seed(seed_v) if seed_v > 0 else None
            if gen is not None:
                noise = torch.randn(
                    init_clean_latents.shape, generator=gen,
                    device=device, dtype=t_dtype,
                )
            else:
                noise = torch.randn(
                    init_clean_latents.shape, device=device, dtype=t_dtype
                )
            noisy = _add_noise_flowmatch(init_clean_latents, noise, actual_sigma)

            print(f"[EricQwenImageHarmonize] [{tag}] denoise={denoise_v:.2f} -> "
                  f"{keep_steps}/{steps_v} effective steps  "
                  f"sigma_start={actual_sigma:.4f}  cfg={cfg_v}  "
                  f"cn=[{cn_a},{cn_b}]@{cn_s}  schedule={sched_v}")

            pbar = comfy.utils.ProgressBar(keep_steps)

            def _cb(_pipe, _step, _t, cb_kwargs):
                pbar.update(1)
                _check_cancelled()
                return cb_kwargs

            t0 = time.time()
            with torch.inference_mode():
                result = cn_pipe(
                    prompt=final_prompt,
                    negative_prompt=negative_prompt if negative_prompt else " ",
                    control_image=ctrl_pil_in,
                    controlnet_conditioning_scale=float(cn_s),
                    control_guidance_start=float(cn_a),
                    control_guidance_end=float(cn_b),
                    height=work_h,
                    width=work_w,
                    num_inference_steps=steps_v,
                    sigmas=raw_sigmas,
                    true_cfg_scale=float(cfg_v),
                    generator=gen,
                    latents=noisy,
                    max_sequence_length=int(max_sequence_length),
                    num_images_per_prompt=1,
                    callback_on_step_end=_cb,
                    output_type="latent",  # always keep latents on GPU
                )
            dt = time.time() - t0
            print(f"[EricQwenImageHarmonize] [{tag}] pass done in {dt:.1f}s "
                  f"({dt/max(1,keep_steps):.2f} s/step)")
            return result.images  # packed latents

        use_upscale_vae = upscale_vae is not None
        decode_label = ("upscale-VAE 2×" if use_upscale_vae else "pipeline VAE")
        print(f"[EricQwenImageHarmonize] cn_mode={cn_mode}  decode={decode_label}  "
              f"refine_pass={refine_pass}  prompt[:120]={final_prompt[:120]}")

        overall_t0 = time.time()

        # ── Pass 1: structural harmonisation ─────────────────────────────
        pass1_latents = _run_pass(
            "pass-1",
            clean_latents,
            ctrl_pil,
            denoise, steps, sigma_schedule,
            true_cfg_scale,
            cn_strength, cn_start, cn_end,
            seed_int,
        )

        final_latents = pass1_latents

        # ── Pass 2 (optional): detail refinement ────────────────────────
        if refine_pass:
            ctrl_pil_p2 = ctrl_pil
            do_recanny = (
                refine_recanny
                and cn_mode in ("auto_canny", "auto_soft_edge")
            )
            if do_recanny:
                # One auxiliary decode (pass-1 latent → PIL) ONLY to build a
                # fresh canny map. The latent itself is never re-encoded —
                # pass 2 starts from `pass1_latents` directly. Zero round-trip.
                print(f"[EricQwenImageHarmonize] refine: re-deriving "
                      f"{cn_mode} from pass-1 result (no latent round-trip)")
                t0 = time.time()
                with torch.inference_mode():
                    pil_for_canny = self._latents_to_pil(
                        cn_pipe, pass1_latents, work_h, work_w, vae_scale_factor
                    )
                ctrl_pil_p2 = self._build_control_image(
                    cn_mode, pil_for_canny, None, canny_low, canny_high
                )
                print(f"[EricQwenImageHarmonize] refine: aux decode + canny "
                      f"in {time.time()-t0:.1f}s")
            else:
                print(f"[EricQwenImageHarmonize] refine: reusing pass-1 control image")

            refine_seed = (seed_int + 1) if seed_int > 0 else 0
            final_latents = _run_pass(
                "pass-2",
                pass1_latents,  # use pass-1 result as the new "clean" base
                ctrl_pil_p2,
                refine_denoise, refine_steps, refine_sigma_schedule,
                refine_cfg,
                refine_cn_strength, refine_cn_start, refine_cn_end,
                refine_seed,
            )
            ctrl_pil = ctrl_pil_p2  # report the pass-2 control image

        # ── Final decode ────────────────────────────────────────────────
        if use_upscale_vae:
            try:
                pipe.transformer = pipe.transformer.to("cpu")
                torch.cuda.empty_cache()
            except Exception:
                pass
            print(f"[EricQwenImageHarmonize] Decoding through upscale VAE "
                  f"({work_w}x{work_h} -> {work_w*2}x{work_h*2}) ...")
            t0 = time.time()
            img_bhwc = decode_latents_with_upscale_vae(
                final_latents, upscale_vae, pipe.vae,
                work_h, work_w, vae_scale_factor,
            )
            print(f"[EricQwenImageHarmonize] Upscale VAE decode: {time.time()-t0:.1f}s")
            arr = (img_bhwc[0].numpy() * 255.0).clip(0, 255).astype(np.uint8)
            out_pil = Image.fromarray(arr, "RGB")
            target_size = (work_w * 2, work_h * 2) if upscale_vae_keep_2x else (orig_w, orig_h)
            if out_pil.size != target_size:
                out_pil = out_pil.resize(target_size, Image.LANCZOS)
        else:
            out_pil = self._latents_to_pil(
                cn_pipe, final_latents, work_h, work_w, vae_scale_factor
            )
            if out_pil.size != (orig_w, orig_h):
                out_pil = out_pil.resize((orig_w, orig_h), Image.LANCZOS)

        elapsed = time.time() - overall_t0
        print(f"[EricQwenImageHarmonize] TOTAL {elapsed:.1f}s  "
              f"output={out_pil.size[0]}x{out_pil.size[1]}")

        out_tensor = pil_to_tensor(out_pil).unsqueeze(0)

        if ctrl_pil.size != (orig_w, orig_h):
            ctrl_view = ctrl_pil.resize((orig_w, orig_h), Image.LANCZOS)
        else:
            ctrl_view = ctrl_pil
        ctrl_tensor = pil_to_tensor(ctrl_view).unsqueeze(0)

        return (out_tensor, ctrl_tensor)

    # ------------------------------------------------------------------
    @staticmethod
    def _latents_to_pil(cn_pipe, packed_latents, height, width, vae_scale_factor):
        """Decode packed Qwen latents to a PIL image using the pipeline VAE.

        Replicates QwenImagePipeline.decode_latents math directly so we can
        do this without round-tripping through Diffusers' unhappy paths.
        """
        from .eric_qwen_image_multistage import _unpack_latents

        vae = cn_pipe.vae
        z_dim = vae.config.z_dim
        device = next(vae.parameters()).device
        # If VAE got offloaded, bring it back
        v_dt = next(vae.parameters()).dtype

        spatial = _unpack_latents(packed_latents, height, width, vae_scale_factor)
        spatial = spatial.to(device=device, dtype=v_dt)

        mean_t = torch.tensor(vae.config.latents_mean).view(1, z_dim, 1, 1, 1).to(device=device, dtype=v_dt)
        std_t = torch.tensor(vae.config.latents_std).view(1, z_dim, 1, 1, 1).to(device=device, dtype=v_dt)
        # QwenImagePipeline decode normalisation: latents = latents / std_inv + mean
        # where std_inv = 1/std → equivalent to: latents * std + mean
        spatial = spatial * std_t + mean_t

        with torch.no_grad():
            decoded = vae.decode(spatial, return_dict=False)[0]
        # decoded: (1, 3, 1, H, W) in [-1, 1]
        decoded = decoded.squeeze(2)  # (1, 3, H, W)
        decoded = (decoded.clamp(-1, 1) + 1.0) / 2.0
        arr = decoded[0].permute(1, 2, 0).float().cpu().numpy()
        arr = (arr * 255.0).clip(0, 255).astype(np.uint8)
        return Image.fromarray(arr, "RGB")


NODE_CLASS_MAPPINGS = {
    "EricQwenImageHarmonize": EricQwenImageHarmonize,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "EricQwenImageHarmonize": "Eric Qwen-Image Composite Harmonize",
}
