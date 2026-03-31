# Copyright (c) 2026 Eric Hiss. All rights reserved.
# Licensed under the terms in LICENSE.txt (CC BY-NC 4.0 / Commercial dual license).
# https://github.com/EricRollei/Eric_Qwen_Edit_Experiments
"""
Eric Qwen UltraGen Inpaint CN — ControlNet-Guided Multi-Stage Inpainting

Multi-stage inpainting and outpainting node powered by the InstantX
Qwen-Image-ControlNet-Inpainting model.  Uses `QwenImageControlNetInpaintPipeline`
which understands a real mask channel (17ch conditioning: 16ch VAE-encoded
masked image + 1ch downsampled mask), unlike our Qwen-Edit inpaint node
which simulates inpainting by blanking the masked region.

Supports:
  • Object replacement — mask an object, describe the replacement
  • Background replacement — mask the background, describe the new scene
  • Text modification — mask text regions, describe new text
  • Outpainting — expand the canvas in any direction

Architecture:
  Stage 1: QwenImageControlNetInpaintPipeline at low resolution (CN draft)
  Stage 2: Same inpaint pipeline at higher resolution with CN (refine)
  Stage 3: (Optional) Standard QwenImagePipeline polish (no CN)
  Final:   Pixel-space composite — original pixels preserved outside mask

ControlNet Model:
  - InstantX/Qwen-Image-ControlNet-Inpainting
  - https://huggingface.co/InstantX/Qwen-Image-ControlNet-Inpainting

Base Model Credits:
  - Qwen-Image developed by Qwen Team (Alibaba)
  - https://github.com/QwenLM/Qwen-Image

Author: Eric Hiss (GitHub: EricRollei)
"""

import math
import torch
import numpy as np
from PIL import Image, ImageFilter
from typing import Tuple

from .eric_qwen_edit_utils import pil_to_tensor
from .eric_qwen_image_generate import (
    ASPECT_RATIOS,
    DIMENSION_ALIGNMENT,
    _align,
    compute_dimensions_from_ratio,
)
from .eric_qwen_image_multistage import (
    _unpack_latents,
    _pack_latents,
    _upscale_latents,
    _add_noise_flowmatch,
    _check_cancelled,
    _packed_seq_len,
    _compute_mu,
    _compute_actual_start_sigma,
    build_sigma_schedule,
)
from .eric_qwen_upscale_vae import (
    decode_latents_with_upscale_vae,
    upscale_between_stages,
)
from .eric_qwen_image_ultragen import (
    _apply_lora_stage_weights,
    QWEN_OFFICIAL_RESOLUTIONS,
    DEFAULT_NEGATIVE_PROMPT,
)


# ── PIL / tensor conversion helpers ─────────────────────────────────────

def _tensor_to_pil(tensor: torch.Tensor) -> Image.Image:
    """Convert a ComfyUI IMAGE tensor (B, H, W, C) float [0,1] to PIL."""
    if tensor.dim() == 4:
        tensor = tensor[0]
    arr = (tensor.cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
    return Image.fromarray(arr)


def _mask_tensor_to_pil(mask: torch.Tensor) -> Image.Image:
    """Convert a ComfyUI MASK tensor to PIL L-mode.

    ComfyUI MASK tensors are (B, H, W) or (H, W), float [0,1].
    White (1.0 / 255) = area to inpaint.
    """
    if mask.dim() == 3:
        mask = mask[0]
    arr = (mask.cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
    return Image.fromarray(arr, mode="L")


# ── Mask dilation with gradient ──────────────────────────────────────────

def _grow_mask_with_gradient(
    mask_pil: Image.Image, grow_px: int,
) -> Image.Image:
    """Dilate a mask by *grow_px* pixels with a smooth gradient falloff.

    Returns an L-mode PIL image where:
      • Original masked area  → 255 (fully white)
      • Growth border zone    → gradient 255→0 over *grow_px* pixels
      • Original keep area    → 0   (fully black)

    The pipeline will binarise this (threshold 0.5) so the model sees a
    wider inpaint region, but the gradient version is used for the final
    feathered composite so the transition is seamless.
    """
    if grow_px <= 0:
        return mask_pil.copy()

    arr_orig = np.array(mask_pil, dtype=np.float32)

    # 1) Binary dilation — iterative MaxFilter
    dilated = mask_pil.copy()
    px_left = grow_px
    while px_left > 0:
        kernel = min(px_left * 2 + 1, 31)
        if kernel < 3:
            kernel = 3
        dilated = dilated.filter(ImageFilter.MaxFilter(kernel))
        px_left -= (kernel - 1) // 2

    # 2) Gaussian-blur the dilated mask to create a gradient at edges
    gradient = dilated.filter(
        ImageFilter.GaussianBlur(radius=max(grow_px * 0.75, 1))
    )

    # 3) Ensure original masked area stays fully white
    arr_gradient = np.array(gradient, dtype=np.float32)
    result = np.maximum(arr_orig, arr_gradient)

    return Image.fromarray(result.clip(0, 255).astype(np.uint8), mode="L")


# ── Outpainting canvas helpers ──────────────────────────────────────────

OUTPAINT_MODES = [
    "disabled",
    "left",
    "right",
    "top",
    "bottom",
    "left+right",
    "top+bottom",
    "all_sides",
]


def _build_outpaint_canvas(
    image: Image.Image,
    mask: Image.Image,
    direction: str,
    expand_pixels: int,
) -> Tuple[Image.Image, Image.Image]:
    """Expand the canvas for outpainting and update the mask.

    Returns (padded_image, padded_mask) where the new region is black
    in the image and white (255) in the mask.
    """
    if direction == "disabled" or expand_pixels <= 0:
        return image, mask

    orig_w, orig_h = image.size

    # Calculate new dimensions based on direction
    pad_l = pad_r = pad_t = pad_b = 0
    if direction == "left":
        pad_l = expand_pixels
    elif direction == "right":
        pad_r = expand_pixels
    elif direction == "top":
        pad_t = expand_pixels
    elif direction == "bottom":
        pad_b = expand_pixels
    elif direction == "left+right":
        pad_l = expand_pixels
        pad_r = expand_pixels
    elif direction == "top+bottom":
        pad_t = expand_pixels
        pad_b = expand_pixels
    elif direction == "all_sides":
        pad_l = pad_r = pad_t = pad_b = expand_pixels

    new_w = orig_w + pad_l + pad_r
    new_h = orig_h + pad_t + pad_b

    # Align to 16 (pipeline requirement: divisible by vae_scale_factor * 2)
    new_w = ((new_w + 15) // 16) * 16
    new_h = ((new_h + 15) // 16) * 16

    # Build padded image (black background)
    padded_img = Image.new("RGB", (new_w, new_h), (0, 0, 0))
    padded_img.paste(image, (pad_l, pad_t))

    # Build padded mask: white in expanded region, preserve original mask
    padded_mask = Image.new("L", (new_w, new_h), 255)  # all white
    # Paste original mask into original position
    if mask.size != (orig_w, orig_h):
        mask = mask.resize((orig_w, orig_h), Image.NEAREST)
    padded_mask.paste(mask, (pad_l, pad_t))

    print(f"[UltraGenInpaintCN] Outpaint: {orig_w}×{orig_h} → {new_w}×{new_h} "
          f"(pad L={pad_l} R={pad_r} T={pad_t} B={pad_b})")

    return padded_img, padded_mask


# ── Post-compositing ────────────────────────────────────────────────────

def _composite_with_mask(
    original: Image.Image,
    generated: Image.Image,
    mask: Image.Image,
    feather_radius: int = 8,
) -> Image.Image:
    """Composite *generated* into *original* using *mask* with feathering.

    Pixels where mask is white (255) → use generated.
    Pixels where mask is black (0)   → use original.

    All inputs are PIL Images.  *mask* is mode 'L'.

    The composite is done at the GENERATED image's resolution so that
    multi-stage upscaling is preserved. Original and mask are upscaled
    to match when needed.
    """
    target_size = generated.size  # (W, H) — use the generated resolution

    # Upscale original and mask to match generated size (preserves upscaling)
    if original.size != target_size:
        original = original.resize(target_size, Image.LANCZOS)
    if mask.size != target_size:
        mask = mask.resize(target_size, Image.NEAREST)

    # Feather the mask edges
    if feather_radius > 0:
        blurred_mask = mask.filter(ImageFilter.GaussianBlur(radius=feather_radius))
    else:
        blurred_mask = mask

    orig_np = np.array(original.convert("RGB"), dtype=np.float32)
    gen_np = np.array(generated.convert("RGB"), dtype=np.float32)
    alpha = np.array(blurred_mask, dtype=np.float32) / 255.0
    alpha = alpha[:, :, np.newaxis]

    result_np = orig_np * (1.0 - alpha) + gen_np * alpha
    result_np = np.clip(result_np, 0, 255).astype(np.uint8)
    return Image.fromarray(result_np, "RGB")


# ── Aspect-ratio helpers ────────────────────────────────────────────────

def _closest_aspect_ratio(
    img_w: int, img_h: int
) -> tuple:
    """Find the ASPECT_RATIOS entry closest to the image's actual ratio."""
    img_ratio = img_w / max(img_h, 1)
    best_label = "1:1   Square"
    best_pair = (1, 1)
    best_diff = float("inf")
    for label, (wr, hr) in ASPECT_RATIOS.items():
        diff = abs(wr / hr - img_ratio)
        if diff < best_diff:
            best_diff = diff
            best_label = label
            best_pair = (wr, hr)
    return best_label, best_pair


# ═══════════════════════════════════════════════════════════════════════
#  UltraGen Inpaint CN Node
# ═══════════════════════════════════════════════════════════════════════

class EricQwenImageUltraGenInpaintCN:
    """
    ControlNet-guided multi-stage inpainting and outpainting.

    Uses InstantX/Qwen-Image-ControlNet-Inpainting which provides a
    dedicated mask channel (17ch conditioning) for high-quality
    mask-aware inpainting on the Qwen-Image base model.

    The pipeline generates the ENTIRE image from noise while the CN
    conditions on the masked source image.  After generation, a
    pixel-space composite ensures original pixels outside the mask
    are perfectly preserved.

    Stages:
      1 — Inpaint draft at low resolution (CN 100%)
      2 — Refine at higher resolution (CN 100%)
      3 — (Optional) Final polish without CN
      Post — Feathered composite with original image

    Outpainting:
      Enable the outpaint direction and expand amount to extend the
      canvas beyond the original image.  A mask is automatically
      created for the expanded region.

    Prompt tips (from InstantX):
      • Use DESCRIPTIVE prompts, not instructive.
      • Describe the ENTIRE image (inpainted area AND background).
      • Good: "A green taxi driving on a road with buildings behind"
      • Bad:  "Replace the car with a taxi"
    """

    CATEGORY = "Eric Qwen-Image"
    FUNCTION = "generate"
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)

    @classmethod
    def INPUT_TYPES(cls):
        ratio_names = list(ASPECT_RATIOS.keys())
        return {
            "required": {
                "pipeline": ("QWEN_IMAGE_PIPELINE", {
                    "tooltip": "From the Qwen-Image loader or component loader"
                }),
                "controlnet": ("QWEN_IMAGE_CONTROLNET", {
                    "tooltip": (
                        "From the Qwen-Image ControlNet Loader.\n"
                        "Load InstantX/Qwen-Image-ControlNet-Inpainting."
                    )
                }),
                "image": ("IMAGE", {
                    "tooltip": (
                        "Source image to inpaint or outpaint.\n"
                        "Can be any resolution — will be fitted to\n"
                        "generation dimensions for Stage 1."
                    )
                }),
                "mask": ("MASK", {
                    "tooltip": (
                        "Inpaint mask. White (1.0) = areas to regenerate,\n"
                        "Black (0.0) = areas to preserve.\n"
                        "Must match image dimensions."
                    )
                }),
                "prompt": ("STRING", {
                    "multiline": True,
                    "tooltip": (
                        "Describe the ENTIRE desired image, not just the\n"
                        "inpainted area. Use descriptive language.\n"
                        "Good: 'A green taxi on a road with tall buildings'\n"
                        "Bad:  'Replace the car with a taxi'"
                    )
                }),
            },
            "optional": {
                "negative_prompt": ("STRING", {
                    "multiline": True,
                    "default": DEFAULT_NEGATIVE_PROMPT,
                    "tooltip": "What to avoid in the generation."
                }),

                # ── Outpainting ──────────────────────────────────────────
                "outpaint_direction": (OUTPAINT_MODES, {
                    "default": "disabled",
                    "tooltip": (
                        "Expand the canvas for outpainting.\n"
                        "A mask is auto-generated for the expanded region."
                    )
                }),
                "outpaint_pixels": ("INT", {
                    "default": 256,
                    "min": 0,
                    "max": 2048,
                    "step": 16,
                    "tooltip": (
                        "Pixels to expand in the chosen direction.\n"
                        "Will be rounded up to nearest multiple of 16."
                    )
                }),

                # ── Compositing ──────────────────────────────────────────
                "feather": ("INT", {
                    "default": 8,
                    "min": 0,
                    "max": 50,
                    "step": 1,
                    "tooltip": (
                        "Mask-edge feathering radius for final composite.\n"
                        "Higher = softer blend at mask edges. 0 = hard edge."
                    )
                }),
                "mask_grow_px": ("INT", {
                    "default": 24,
                    "min": 0,
                    "max": 128,
                    "step": 4,
                    "tooltip": (
                        "Dilate the mask by this many pixels for Stage 2.\n"
                        "Creates a transition zone around the edit where\n"
                        "the model can regenerate for seamless blending.\n"
                        "The growth zone uses a gradient falloff in the\n"
                        "final composite. 0 = use original mask only."
                    )
                }),
                "auto_stages": ("BOOLEAN", {
                    "default": True,
                    "tooltip": (
                        "Smart stage selection based on input image size.\n"
                        "When enabled and the input image is large enough\n"
                        "(>= 50%% of S2 target MP), Stage 1 is skipped\n"
                        "and Stage 2 generates from noise at higher\n"
                        "resolution directly.  Saves time without\n"
                        "sacrificing quality for large inputs."
                    )
                }),

                # ── Aspect ratio / sizing ────────────────────────────────
                "aspect_ratio": (ratio_names, {
                    "default": "1:1   Square",
                    "tooltip": (
                        "Aspect ratio for generation dimensions.\n"
                        "Set to 'match_image' to auto-detect from input."
                    )
                }),
                "match_image_aspect": ("BOOLEAN", {
                    "default": True,
                    "tooltip": (
                        "Override aspect_ratio to match the input image.\n"
                        "Recommended for inpainting to avoid distortion."
                    )
                }),
                "seed": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 0xffffffffffffffff,
                    "tooltip": "Random seed (0 = random)"
                }),
                "seed_mode": (
                    ["same_all_stages", "offset_per_stage", "random_per_stage"],
                    {
                        "default": "offset_per_stage",
                        "tooltip": (
                            "How seeds are chosen for each stage:\n"
                            "• same_all_stages — one seed for all\n"
                            "• offset_per_stage — S2=seed+1, S3=seed+2\n"
                            "• random_per_stage — independent random per stage"
                        )
                    }
                ),

                # ── Pipeline parameters ──────────────────────────────────
                "max_sequence_length": ("INT", {
                    "default": 1024,
                    "min": 128,
                    "max": 1024,
                    "step": 64,
                    "tooltip": "Maximum prompt token length."
                }),

                # ── ControlNet parameters ────────────────────────────────
                "cn_auto_scale": ("BOOLEAN", {
                    "default": True,
                    "tooltip": (
                        "Auto-calibrate ControlNet strength to match\n"
                        "the transformer's hidden-state magnitude.\n"
                        "Compensates for finetuned transformers."
                    )
                }),
                "cn_target_strength": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.1,
                    "max": 6.0,
                    "step": 0.1,
                    "tooltip": (
                        "ControlNet influence (auto-scale mode).\n"
                        "1.0 = standard. Higher = stronger guidance."
                    )
                }),
                "controlnet_conditioning_scale": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.0,
                    "max": 50.0,
                    "step": 0.5,
                    "tooltip": (
                        "Manual ControlNet scale (when auto-scale is OFF).\n"
                        "For custom transformers, 10–30 may be needed."
                    )
                }),
                "control_guidance_start": ("FLOAT", {
                    "default": 0.0,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.05,
                    "tooltip": "When CN guidance begins (fraction of steps)."
                }),
                "control_guidance_end": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.05,
                    "tooltip": "When CN guidance ends (fraction of steps)."
                }),

                # ── Stage 2 CN parameters ────────────────────────────────
                "s2_cn_scale": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.0,
                    "max": 50.0,
                    "step": 0.5,
                    "tooltip": (
                        "ControlNet strength on Stage 2.\n"
                        "When auto_scale is ON, this is relative to S1.\n"
                        "1.0 = same as S1 (recommended for inpainting).\n"
                        "0.0 = disable CN for S2."
                    )
                }),
                "s2_cn_start": ("FLOAT", {
                    "default": 0.0,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.05,
                    "tooltip": "When CN guidance begins in Stage 2."
                }),
                "s2_cn_end": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.05,
                    "tooltip": "When CN guidance ends in Stage 2."
                }),

                # ── Stage 1 ─────────────────────────────────────────────
                "s1_mp": ("FLOAT", {
                    "default": 0.5,
                    "min": 0.3,
                    "max": 2.0,
                    "step": 0.1,
                    "tooltip": "Stage 1 resolution in megapixels."
                }),
                "s1_steps": ("INT", {
                    "default": 30,
                    "min": 1,
                    "max": 200,
                    "step": 1,
                    "tooltip": (
                        "Stage 1 inference steps.\n"
                        "InstantX recommends 30 for inpainting."
                    )
                }),
                "s1_cfg": ("FLOAT", {
                    "default": 4.0,
                    "min": 1.0,
                    "max": 20.0,
                    "step": 0.5,
                    "tooltip": (
                        "Stage 1 true CFG scale.\n"
                        "InstantX recommends 4.0 for inpainting."
                    )
                }),

                # ── Stage 2 ─────────────────────────────────────────────
                "upscale_to_stage2": ("FLOAT", {
                    "default": 4.0,
                    "min": 0.0,
                    "max": 10.0,
                    "step": 0.5,
                    "tooltip": (
                        "Upscale factor (area) from S1 to S2.\n"
                        "0 = skip S2 & S3 (output S1 only)."
                    )
                }),
                "s2_steps": ("INT", {
                    "default": 26,
                    "min": 1,
                    "max": 200,
                    "step": 1,
                    "tooltip": "Stage 2 inference steps."
                }),
                "s2_cfg": ("FLOAT", {
                    "default": 4.0,
                    "min": 1.0,
                    "max": 20.0,
                    "step": 0.5,
                    "tooltip": "Stage 2 true CFG scale."
                }),
                "s2_denoise": ("FLOAT", {
                    "default": 0.85,
                    "min": 0.1,
                    "max": 1.0,
                    "step": 0.05,
                    "tooltip": "Stage 2 denoise strength."
                }),
                "s2_sigma_schedule": (["linear", "balanced", "karras"], {
                    "default": "linear",
                    "tooltip": "Sigma schedule for Stage 2."
                }),

                # ── S2 Harmonization pass ─────────────────────────────────
                "s2_harmonize_denoise": ("FLOAT", {
                    "default": 0.35,
                    "min": 0.0,
                    "max": 0.8,
                    "step": 0.05,
                    "tooltip": (
                        "S2 Phase B: whole-image harmonization pass.\n"
                        "Runs after the CN inpaint pass (Phase A) at S2\n"
                        "resolution with NO mask and NO ControlNet, so\n"
                        "the model can harmonize boundaries between\n"
                        "inpainted and original content.\n"
                        "0.0 = disabled (skip Phase B)."
                    )
                }),
                "s2_harmonize_steps": ("INT", {
                    "default": 10,
                    "min": 1,
                    "max": 60,
                    "step": 1,
                    "tooltip": (
                        "Number of inference steps for S2 harmonization.\n"
                        "Lower = faster, higher = smoother blending."
                    )
                }),

                # ── Stage 3 ─────────────────────────────────────────────
                "upscale_to_stage3": ("FLOAT", {
                    "default": 0.0,
                    "min": 0.0,
                    "max": 8.0,
                    "step": 0.5,
                    "tooltip": (
                        "Upscale factor (area) from S2 to S3.\n"
                        "0 = disabled (2-stage output)."
                    )
                }),
                "s3_steps": ("INT", {
                    "default": 18,
                    "min": 1,
                    "max": 200,
                    "step": 1,
                    "tooltip": "Stage 3 inference steps."
                }),
                "s3_cfg": ("FLOAT", {
                    "default": 2.0,
                    "min": 1.0,
                    "max": 20.0,
                    "step": 0.5,
                    "tooltip": "Stage 3 true CFG scale."
                }),
                "s3_denoise": ("FLOAT", {
                    "default": 0.45,
                    "min": 0.1,
                    "max": 1.0,
                    "step": 0.05,
                    "tooltip": "Stage 3 denoise strength."
                }),
                "s3_sigma_schedule": (["linear", "balanced", "karras"], {
                    "default": "karras",
                    "tooltip": "Sigma schedule for Stage 3."
                }),

                # ── Upscale VAE (optional) ───────────────────────────────
                "upscale_vae": ("UPSCALE_VAE", {
                    "tooltip": (
                        "Optional: Wan2.1 2× upscale VAE.\n"
                        "Load with Eric Qwen Upscale VAE Loader."
                    )
                }),
                "upscale_vae_mode": (
                    ["disabled", "inter_stage", "final_decode", "both"], {
                    "default": "disabled",
                    "tooltip": (
                        "How the upscale VAE is used:\n"
                        "• disabled — upscale VAE ignored\n"
                        "• inter_stage — decode S2→2×→re-encode for S3\n"
                        "• final_decode — 2× upscale on final output\n"
                        "• both — inter-stage + final decode"
                    )
                }),

                # ── Detail prompt (optional) ──────────────────────────────
                "detail_prompt": ("STRING", {
                    "multiline": True,
                    "default": "",
                    "tooltip": (
                        "Optional detailed prompt for late stages (S2B\n"
                        "harmonization and S3 polish).  These stages\n"
                        "resample the ENTIRE image without a mask, so a\n"
                        "richer, more descriptive prompt produces better\n"
                        "results.\n\n"
                        "Wire the ControlNet Prompt Rewriter output here\n"
                        "for best quality.  If empty, the main prompt is\n"
                        "used for all stages."
                    )
                }),
            }
        }

    # ─────────────────────────────────────────────────────────────────
    #  Main entry
    # ─────────────────────────────────────────────────────────────────

    def generate(
        self,
        pipeline: dict,
        controlnet: dict,
        image: torch.Tensor,
        mask: torch.Tensor,
        prompt: str = "",
        negative_prompt: str = DEFAULT_NEGATIVE_PROMPT,
        # Outpainting
        outpaint_direction: str = "disabled",
        outpaint_pixels: int = 256,
        # Compositing
        feather: int = 8,
        mask_grow_px: int = 24,
        auto_stages: bool = True,
        # Aspect / sizing
        aspect_ratio: str = "1:1   Square",
        match_image_aspect: bool = True,
        seed: int = 0,
        seed_mode: str = "offset_per_stage",
        # Pipeline params
        max_sequence_length: int = 1024,
        # CN params (Stage 1)
        cn_auto_scale: bool = True,
        cn_target_strength: float = 1.0,
        controlnet_conditioning_scale: float = 1.0,
        control_guidance_start: float = 0.0,
        control_guidance_end: float = 1.0,
        # CN params (Stage 2)
        s2_cn_scale: float = 1.0,
        s2_cn_start: float = 0.0,
        s2_cn_end: float = 1.0,
        # Stage 1
        s1_mp: float = 0.5,
        s1_steps: int = 30,
        s1_cfg: float = 4.0,
        # Stage 2
        upscale_to_stage2: float = 4.0,
        s2_steps: int = 26,
        s2_cfg: float = 4.0,
        s2_denoise: float = 0.85,
        s2_sigma_schedule: str = "linear",
        # S2 Harmonization
        s2_harmonize_denoise: float = 0.35,
        s2_harmonize_steps: int = 10,
        # Stage 3
        upscale_to_stage3: float = 0.0,
        s3_steps: int = 18,
        s3_cfg: float = 2.0,
        s3_denoise: float = 0.45,
        s3_sigma_schedule: str = "karras",
        # Upscale VAE
        upscale_vae=None,
        upscale_vae_mode: str = "disabled",
        # Detail prompt
        detail_prompt: str = "",
    ) -> Tuple[torch.Tensor]:
        pipe = pipeline["pipeline"]
        cn_model = controlnet["model"]
        offload_vae = pipeline.get("offload_vae", False)

        # ── Resolve late-stage prompt ───────────────────────────────────
        late_prompt = detail_prompt.strip() if detail_prompt else ""
        if late_prompt:
            print(f"[UltraGenInpaintCN] Detail prompt for S2B/S3: "
                  f"{len(late_prompt)} chars")
        else:
            late_prompt = prompt  # fall back to the main prompt

        # ── Convert inputs to PIL ───────────────────────────────────────
        source_pil = _tensor_to_pil(image)
        mask_pil = _mask_tensor_to_pil(mask)
        print(f"[UltraGenInpaintCN] Source: {source_pil.size[0]}×{source_pil.size[1]}, "
              f"Mask: {mask_pil.size[0]}×{mask_pil.size[1]}")

        # Ensure mask matches source image size
        if mask_pil.size != source_pil.size:
            mask_pil = mask_pil.resize(source_pil.size, Image.NEAREST)

        # ── Keep full-res originals for final composite ─────────────────
        original_for_composite = source_pil.copy()
        mask_for_composite = mask_pil.copy()

        # ── Outpainting: expand canvas ──────────────────────────────────
        if outpaint_direction != "disabled" and outpaint_pixels > 0:
            source_pil, mask_pil = _build_outpaint_canvas(
                source_pil, mask_pil, outpaint_direction, outpaint_pixels,
            )
            # For outpainting, the composite uses the padded versions
            original_for_composite = source_pil.copy()
            mask_for_composite = mask_pil.copy()

        # ── Resolve aspect ratio ────────────────────────────────────────
        if match_image_aspect:
            img_w, img_h = source_pil.size
            matched_label, (w_ratio, h_ratio) = _closest_aspect_ratio(img_w, img_h)
            print(f"[UltraGenInpaintCN] Matched aspect: {matched_label} "
                  f"({w_ratio}:{h_ratio}) from {img_w}×{img_h}")
        else:
            w_ratio, h_ratio = ASPECT_RATIOS.get(aspect_ratio, (1, 1))

        # ── Determine active stages ─────────────────────────────────────
        do_stage2 = upscale_to_stage2 > 0
        do_stage3 = do_stage2 and upscale_to_stage3 > 0

        # ── Compute per-stage dimensions ────────────────────────────────
        s1_w, s1_h = compute_dimensions_from_ratio(w_ratio, h_ratio, s1_mp)
        s1_mp_actual = s1_w * s1_h / 1e6

        if do_stage2:
            s2_mp = s1_mp_actual * upscale_to_stage2
            s2_w, s2_h = compute_dimensions_from_ratio(w_ratio, h_ratio, s2_mp)
            s2_mp_actual = s2_w * s2_h / 1e6
        else:
            s2_w = s2_h = s2_mp_actual = 0

        if do_stage3:
            s3_mp = s2_mp_actual * upscale_to_stage3
            s3_w, s3_h = compute_dimensions_from_ratio(w_ratio, h_ratio, s3_mp)
            s3_mp_actual = s3_w * s3_h / 1e6
        else:
            s3_w = s3_h = s3_mp_actual = 0

        # ── Smart stage selection ──────────────────────────────────────
        input_mp = source_pil.size[0] * source_pil.size[1] / 1e6
        skip_s1 = False
        auto_threshold = 0.0
        s2a_steps = s2_steps   # effective S2A step count (may be overridden)
        s2a_cfg = s2_cfg       # effective S2A CFG (may be overridden)
        if auto_stages and do_stage2:
            s2_target_mp = s1_mp_actual * upscale_to_stage2
            auto_threshold = s2_target_mp * 0.5
            if input_mp >= auto_threshold:
                skip_s1 = True
                # S2A becomes the initial generation — use S1-grade params
                s2a_steps = max(s1_steps, s2_steps)
                s2a_cfg = s1_cfg
                print(f"[UltraGenInpaintCN] AUTO-STAGES: Input {input_mp:.2f} MP "
                      f">= threshold {auto_threshold:.2f} MP — "
                      f"skipping S1, S2A from noise "
                      f"(steps={s2a_steps}, CFG={s2a_cfg:.1f})")

                # If input is >= S2 target, raise S2 resolution to match
                # the input so we don't downscale a large image.
                if input_mp >= s2_target_mp:
                    old_s2_w, old_s2_h = s2_w, s2_h
                    s2_w, s2_h = compute_dimensions_from_ratio(
                        w_ratio, h_ratio, input_mp
                    )
                    s2_mp_actual = s2_w * s2_h / 1e6
                    print(f"[UltraGenInpaintCN] AUTO-STAGES: Input >= S2 "
                          f"target ({s2_target_mp:.2f} MP) — raised S2 "
                          f"from {old_s2_w}×{old_s2_h} to "
                          f"{s2_w}×{s2_h} ({s2_mp_actual:.2f} MP)")

                    # Recompute S3 dimensions if active (S3 scales from S2)
                    if do_stage3:
                        s3_mp = s2_mp_actual * upscale_to_stage3
                        s3_w, s3_h = compute_dimensions_from_ratio(
                            w_ratio, h_ratio, s3_mp
                        )
                        s3_mp_actual = s3_w * s3_h / 1e6
                        print(f"[UltraGenInpaintCN] AUTO-STAGES: Recomputed "
                              f"S3 to {s3_w}×{s3_h} ({s3_mp_actual:.2f} MP)")

        # ── Prepare source image and mask at each stage resolution ──────
        # Stage 1: resize source and mask
        source_s1 = source_pil.resize((s1_w, s1_h), Image.LANCZOS)
        mask_s1 = mask_pil.resize((s1_w, s1_h), Image.NEAREST)
        print(f"[UltraGenInpaintCN] S1 image+mask: {s1_w}×{s1_h}")

        # Stage 2: resize source and mask (if CN active)
        # If mask_grow_px > 0, create a dilated mask for S2 so the model
        # regenerates a wider transition zone for seamless blending.
        source_s2 = None
        mask_s2 = None
        use_cn_s2 = do_stage2 and s2_cn_scale > 0.0
        do_s2_harmonize = do_stage2 and s2_harmonize_denoise > 0.0
        if use_cn_s2 or do_s2_harmonize:
            source_s2 = source_pil.resize((s2_w, s2_h), Image.LANCZOS)
            if mask_grow_px > 0 and use_cn_s2:
                mask_s2_grown_full = _grow_mask_with_gradient(mask_pil, mask_grow_px)
                mask_s2 = mask_s2_grown_full.resize((s2_w, s2_h), Image.LANCZOS)
                # Update composite mask to use grown + gradient version
                mask_for_composite = mask_s2_grown_full.copy()
                print(f"[UltraGenInpaintCN] S2 image+mask: {s2_w}×{s2_h} "
                      f"(mask dilated +{mask_grow_px}px with gradient)")
            elif use_cn_s2:
                mask_s2 = mask_pil.resize((s2_w, s2_h), Image.NEAREST)
                print(f"[UltraGenInpaintCN] S2 image+mask: {s2_w}×{s2_h}")
            else:
                print(f"[UltraGenInpaintCN] S2 image (harmonize only): {s2_w}×{s2_h}")

        # ── Resolve upscale VAE mode ────────────────────────────────────
        use_inter_stage = False
        use_final_decode = False
        if upscale_vae is not None and upscale_vae_mode != "disabled":
            if upscale_vae_mode in ("inter_stage", "both"):
                if do_stage3:
                    use_inter_stage = True
                else:
                    print("[UltraGenInpaintCN] WARNING: inter_stage requires "
                          "3 stages. Falling back to final_decode.")
                    use_final_decode = True
            if upscale_vae_mode in ("final_decode", "both"):
                use_final_decode = True

        stage_count = 3 if do_stage3 else (2 if do_stage2 else 1)
        if skip_s1:
            stage_count -= 1
        s2_harm_steps = s2_harmonize_steps if do_s2_harmonize else 0
        total_steps = ((0 if skip_s1 else s1_steps)
                       + (s2a_steps if do_stage2 else 0)
                       + s2_harm_steps
                       + (s3_steps if do_stage3 else 0))

        # ── Diagnostic info ─────────────────────────────────────────────
        cn_params = sum(p.numel() for p in cn_model.parameters()) / 1e6
        cn_device = next(cn_model.parameters()).device
        cn_dtype = next(cn_model.parameters()).dtype
        print(f"[UltraGenInpaintCN] ControlNet: {cn_params:.0f}M params, "
              f"device={cn_device}, dtype={cn_dtype}")

        # ── Common setup ────────────────────────────────────────────────
        device = pipe._execution_device if hasattr(pipe, "_execution_device") else "cuda"

        # Ensure transformer is on GPU
        transformer_device = next(pipe.transformer.parameters()).device
        if str(transformer_device) != str(device):
            print(f"[UltraGenInpaintCN] Moving transformer to {device}")
            pipe.transformer = pipe.transformer.to(device)
            torch.cuda.empty_cache()

        # Ensure ControlNet is on same device/dtype as transformer
        t_dtype = next(pipe.transformer.parameters()).dtype
        cn_dev = next(cn_model.parameters()).device
        cn_dt = next(cn_model.parameters()).dtype
        if str(cn_dev) != str(device) or cn_dt != t_dtype:
            print(f"[UltraGenInpaintCN] Moving ControlNet to {device} ({t_dtype})")
            cn_model.to(device=device, dtype=t_dtype)

        # ── Build ControlNet Inpaint Pipeline for Stage 1 & 2 ───────────
        from diffusers import QwenImageControlNetInpaintPipeline
        cn_pipe = QwenImageControlNetInpaintPipeline(
            scheduler=pipe.scheduler,
            vae=pipe.vae,
            text_encoder=pipe.text_encoder,
            tokenizer=pipe.tokenizer,
            transformer=pipe.transformer,
            controlnet=cn_model,
        )
        assert hasattr(cn_pipe, 'controlnet') and cn_pipe.controlnet is cn_model, \
            "[UltraGenInpaintCN] FATAL: controlnet not registered on pipeline!"
        print(f"[UltraGenInpaintCN] Inpaint pipeline built (zero-copy, "
              f"controlnet={type(cn_pipe.controlnet).__name__})")

        # ── Per-stage seed logic ────────────────────────────────────────
        def _make_generator(s: int):
            return torch.Generator(device=device).manual_seed(s) if s > 0 else None

        if seed_mode == "offset_per_stage":
            gen_s1 = _make_generator(seed)
            gen_s2 = _make_generator(seed + 1 if seed > 0 else 0)
            gen_s3 = _make_generator(seed + 2 if seed > 0 else 0)
            seed_info = (f"offset: S1={seed}, S2={seed+1 if seed>0 else 'random'}, "
                         f"S3={seed+2 if seed>0 else 'random'}")
        elif seed_mode == "random_per_stage":
            gen_s1 = _make_generator(seed)
            gen_s2 = None
            gen_s3 = None
            seed_info = f"random: S1={seed if seed>0 else 'random'}, S2=random, S3=random"
        else:
            gen_s1 = _make_generator(seed)
            gen_s2 = gen_s1
            gen_s3 = gen_s1
            seed_info = f"same: {seed if seed>0 else 'random'}"
        generator = gen_s1

        # ── Print plan ──────────────────────────────────────────────────
        print(f"\n{'='*60}")
        print(f"[UltraGenInpaintCN] Inpaint multi-stage: {stage_count} stage(s), "
              f"{total_steps} total steps")
        print(f"[UltraGenInpaintCN] max_seq_len={max_sequence_length}, "
              f"seed_mode={seed_mode} ({seed_info})")
        if skip_s1:
            print(f"  Stage 1: SKIPPED (input {input_mp:.2f} MP >= "
                  f"{auto_threshold:.2f} MP threshold)")
        else:
            print(f"  Stage 1: {s1_w}×{s1_h} ({s1_mp_actual:.2f} MP), "
                  f"{s1_steps} steps, CFG={s1_cfg:.1f} [Inpaint CN]")
        if do_stage2:
            cn_label = "[Inpaint CN]" if use_cn_s2 else "[no CN]"
            grow_label = f" mask_grow=+{mask_grow_px}px" if mask_grow_px > 0 else ""
            if skip_s1:
                print(f"  Stage 2A: {s2_w}×{s2_h} ({s2_mp_actual:.2f} MP), "
                      f"{s2a_steps} steps, CFG={s2a_cfg:.1f} "
                      f"{cn_label} [FROM NOISE]{grow_label}")
            else:
                print(f"  Stage 2A: {s2_w}×{s2_h} ({s2_mp_actual:.2f} MP), "
                      f"{s2_steps} steps, CFG={s2_cfg:.1f}, "
                      f"upscale={upscale_to_stage2:.1f}×, "
                      f"denoise={s2_denoise:.2f}, σ={s2_sigma_schedule} "
                      f"{cn_label}{grow_label}")
            if do_s2_harmonize:
                print(f"  Stage 2B: {s2_w}×{s2_h} harmonize, "
                      f"{s2_harmonize_steps} steps, "
                      f"denoise={s2_harmonize_denoise:.2f} [no CN, no mask]")
        if do_stage3:
            print(f"  Stage 3: {s3_w}×{s3_h} ({s3_mp_actual:.2f} MP), "
                  f"{s3_steps} steps, CFG={s3_cfg:.1f}, "
                  f"upscale={upscale_to_stage3:.1f}×, "
                  f"denoise={s3_denoise:.2f}, σ={s3_sigma_schedule} [no CN]")
        print(f"  Post: feathered composite (radius={feather})"
              + (f" mask_grow=+{mask_grow_px}px" if mask_grow_px > 0 else ""))
        print(f"{'='*60}")

        neg = negative_prompt.strip() if negative_prompt else None
        if neg == "":
            neg = None

        import comfy.utils
        pbar = comfy.utils.ProgressBar(total_steps)

        def make_cb():
            def on_step_end(_pipe, step_idx, _timestep, cb_kwargs):
                pbar.update(1)
                _check_cancelled()
                return cb_kwargs
            return on_step_end

        vae_scale_factor = pipe.vae_scale_factor if hasattr(pipe, "vae_scale_factor") else 8

        extra_kwargs = {}
        extra_kwargs["max_sequence_length"] = max_sequence_length

        # ── Compute mu for each active refine stage ──────────────────────
        if do_stage2:
            s2_seq = _packed_seq_len(s2_h, s2_w, vae_scale_factor)
            s2_mu = _compute_mu(s2_seq, pipe.scheduler)
        if do_stage3:
            s3_seq = _packed_seq_len(s3_h, s3_w, vae_scale_factor)
            s3_mu = _compute_mu(s3_seq, pipe.scheduler)

        # ── Move VAE to GPU if offloaded ────────────────────────────────
        if offload_vae and hasattr(pipe, "vae"):
            vae_device = next(pipe.transformer.parameters()).device
            pipe.vae = pipe.vae.to(vae_device)

        # ── CN auto-scaling hooks ───────────────────────────────────────
        _cn_hooks = []
        _auto = {"calibrated": False, "factor": 1.0,
                 "cn_mag": 0.0, "hs_mag": 0.0}
        _cn_ratio = cn_target_strength / 20.0

        if cn_auto_scale:
            def _block0_calibrate(_mod, _inp, output):
                if _auto["hs_mag"] == 0.0:
                    hs = (output[1] if isinstance(output, tuple)
                          and len(output) >= 2 else output)
                    _auto["hs_mag"] = hs.abs().mean().item()
                    if _auto["cn_mag"] > 0.0:
                        _auto["factor"] = (_cn_ratio
                                           * _auto["hs_mag"]
                                           / _auto["cn_mag"])
                        _auto["calibrated"] = True

            _cn_hooks.append(
                pipe.transformer.transformer_blocks[0]
                .register_forward_hook(_block0_calibrate)
            )

            def _tf_autoscale(_mod, args, kwargs):
                cbs = kwargs.get("controlnet_block_samples")
                if cbs is None:
                    return
                if _auto["cn_mag"] == 0.0:
                    _auto["cn_mag"] = cbs[0].abs().mean().item()
                    if _auto["hs_mag"] > 0.0:
                        _auto["factor"] = (_cn_ratio
                                           * _auto["hs_mag"]
                                           / _auto["cn_mag"])
                        _auto["calibrated"] = True
                if _auto["calibrated"] and _auto["factor"] != 1.0:
                    new_kwargs = dict(kwargs)
                    new_kwargs["controlnet_block_samples"] = tuple(
                        s * _auto["factor"] for s in cbs
                    )
                    return args, new_kwargs

            _cn_hooks.append(
                pipe.transformer.register_forward_pre_hook(
                    _tf_autoscale, with_kwargs=True
                )
            )

        try:
            # ════════════════════════════════════════════════════════════
            #  STAGE 1 — Inpaint CN draft (skipped when auto_stages)
            # ════════════════════════════════════════════════════════════
            s1_latents = None
            if skip_s1:
                print(f"[UltraGenInpaintCN] -- Stage 1 SKIPPED (auto_stages: "
                      f"input {input_mp:.2f} MP >= {auto_threshold:.2f} MP) --")
            else:
                _check_cancelled()
                print(f"[UltraGenInpaintCN] -- Stage 1/{stage_count} [Inpaint CN] --")

                _apply_lora_stage_weights(pipe, pipeline, stage=1)

                if cn_auto_scale:
                    s1_cn_scale = 1.0
                    print(f"[UltraGenInpaintCN]   CN mode: auto-scale "
                          f"(strength={cn_target_strength:.1f}, "
                          f"ratio={_cn_ratio:.4f})")
                else:
                    s1_cn_scale = controlnet_conditioning_scale
                    print(f"[UltraGenInpaintCN]   CN mode: manual "
                          f"(scale={s1_cn_scale:.2f})")

                s1_output_type = "latent" if (do_stage2 or use_final_decode) else "pil"
                s1_result = cn_pipe(
                    prompt=prompt,
                    negative_prompt=neg,
                    control_image=source_s1,
                    control_mask=mask_s1,
                    controlnet_conditioning_scale=s1_cn_scale,
                    control_guidance_start=control_guidance_start,
                    control_guidance_end=control_guidance_end,
                    height=s1_h,
                    width=s1_w,
                    num_inference_steps=s1_steps,
                    true_cfg_scale=s1_cfg,
                    generator=generator,
                    callback_on_step_end=make_cb(),
                    output_type=s1_output_type,
                    **extra_kwargs,
                )

                # Report auto-scale calibration
                if _auto["calibrated"]:
                    raw_ratio = _auto["cn_mag"] / max(_auto["hs_mag"], 1e-8)
                    print(f"[UltraGenInpaintCN]   CN auto-scale: "
                          f"hs={_auto['hs_mag']:.0f}, cn={_auto['cn_mag']:.0f}, "
                          f"raw_ratio={raw_ratio:.4f}, boost={_auto['factor']:.1f}× "
                          f"(strength={cn_target_strength:.1f})")
                elif cn_auto_scale:
                    print("[UltraGenInpaintCN]   CN auto-scale: WARNING — "
                          "could not calibrate")

                if not do_stage2:
                    if use_final_decode:
                        pipe.transformer = pipe.transformer.to("cpu")
                        torch.cuda.empty_cache()
                        print(f"[UltraGenInpaintCN] Upscale VAE decode (2×) ...")
                        tensor = decode_latents_with_upscale_vae(
                            s1_result.images, upscale_vae, pipe.vae,
                            s1_h, s1_w, vae_scale_factor,
                        )
                        # Composite with original
                        result_pil = _tensor_to_pil(tensor.squeeze(0))
                        result_pil = _composite_with_mask(
                            original_for_composite, result_pil,
                            mask_for_composite, feather,
                        )
                        tensor = pil_to_tensor(result_pil).unsqueeze(0)
                        print(f"[UltraGenInpaintCN] Output: {result_pil.size[0]}×{result_pil.size[1]} (composited)")
                        return (tensor,)

                    pil_out = s1_result.images[0]
                    result = _composite_with_mask(
                        original_for_composite, pil_out,
                        mask_for_composite, feather,
                    )
                    print(f"[UltraGenInpaintCN] Output: {result.size[0]}×{result.size[1]} (composited)")
                    tensor = pil_to_tensor(result).unsqueeze(0)
                    return (tensor,)

                s1_latents = s1_result.images
                print(f"[UltraGenInpaintCN]   S1 latents: {s1_latents.shape}")

            # ════════════════════════════════════════════════════════════
            #  STAGE 2A — Content Refinement (or initial gen if S1 skipped)
            # ════════════════════════════════════════════════════════════
            _check_cancelled()
            phase_label = "2A" if do_s2_harmonize else "2"
            from_noise_label = " FROM NOISE" if skip_s1 else ""
            print(f"[UltraGenInpaintCN] -- Stage {phase_label}/{stage_count} "
                  f"{'[Inpaint CN]' if use_cn_s2 else '[no CN]'}"
                  f"{from_noise_label} --")

            _apply_lora_stage_weights(pipe, pipeline, stage=2)

            # Phase A always outputs latent (Phase B or S3 or final decode
            # will consume it).  Only output PIL if this is truly the last
            # step (no harmonize, no S3, no final decode).
            s2a_output_type = ("pil"
                               if (not do_s2_harmonize
                                   and not do_stage3
                                   and not use_final_decode)
                               else "latent")

            if skip_s1:
                # ── S2A from noise (S1 was skipped) ─────────────────────
                print(f"[UltraGenInpaintCN]   S2A from noise at "
                      f"{s2_w}×{s2_h} ({s2_mp_actual:.2f} MP), "
                      f"{s2a_steps} steps, CFG={s2a_cfg:.1f}")
                if use_cn_s2:
                    s2_actual_scale = (s2_cn_scale if not cn_auto_scale
                                       else s2_cn_scale)
                    print(f"[UltraGenInpaintCN]   S2A CN: "
                          f"scale={s2_actual_scale}, "
                          f"start={s2_cn_start}, end={s2_cn_end}"
                          f"{' (auto-scaled)' if cn_auto_scale else ''}")
                    s2a_result = cn_pipe(
                        prompt=prompt,
                        negative_prompt=neg,
                        control_image=source_s2,
                        control_mask=mask_s2,
                        controlnet_conditioning_scale=s2_actual_scale,
                        control_guidance_start=s2_cn_start,
                        control_guidance_end=s2_cn_end,
                        height=s2_h,
                        width=s2_w,
                        num_inference_steps=s2a_steps,
                        true_cfg_scale=s2a_cfg,
                        generator=gen_s2,
                        callback_on_step_end=make_cb(),
                        output_type=s2a_output_type,
                        **extra_kwargs,
                    )
                else:
                    print("[UltraGenInpaintCN]   WARNING: S1 skipped but "
                          "s2_cn_scale=0 — generating without CN or mask")
                    s2a_result = pipe(
                        prompt=prompt,
                        negative_prompt=neg,
                        height=s2_h,
                        width=s2_w,
                        num_inference_steps=s2a_steps,
                        true_cfg_scale=s2a_cfg,
                        generator=gen_s2,
                        callback_on_step_end=make_cb(),
                        output_type=s2a_output_type,
                        **extra_kwargs,
                    )

                # Report auto-scale calibration (first CN run when S1 skipped)
                if _auto["calibrated"]:
                    raw_ratio = _auto["cn_mag"] / max(_auto["hs_mag"], 1e-8)
                    print(f"[UltraGenInpaintCN]   CN auto-scale: "
                          f"hs={_auto['hs_mag']:.0f}, "
                          f"cn={_auto['cn_mag']:.0f}, "
                          f"raw_ratio={raw_ratio:.4f}, "
                          f"boost={_auto['factor']:.1f}× "
                          f"(strength={cn_target_strength:.1f})")
                elif cn_auto_scale:
                    print("[UltraGenInpaintCN]   CN auto-scale: WARNING — "
                          "could not calibrate")

            else:
                # ── S2A from S1 latents (normal flow) ───────────────────
                s2_latents = _upscale_latents(
                    s1_latents, s1_h, s1_w, s2_h, s2_w, vae_scale_factor
                )
                print(f"[UltraGenInpaintCN]   S2 latents after upscale: "
                      f"{s2_latents.shape}")

                raw_sigmas_s2 = build_sigma_schedule(
                    s2_steps, s2_denoise, s2_sigma_schedule
                )
                actual_sigma_s2 = _compute_actual_start_sigma(
                    pipe.scheduler, raw_sigmas_s2, s2_mu
                )
                print(f"[UltraGenInpaintCN]   S2 mu={s2_mu:.4f}, "
                      f"schedule={s2_sigma_schedule}, "
                      f"raw_start={raw_sigmas_s2[0]:.4f}, "
                      f"actual_start={actual_sigma_s2:.4f}")

                s2_latents = self._apply_denoise_noise(
                    s2_latents, s2_denoise, actual_sigma_s2, gen_s2, device
                )

                if use_cn_s2:
                    s2_actual_scale = (s2_cn_scale if not cn_auto_scale
                                       else s2_cn_scale)
                    print(f"[UltraGenInpaintCN]   S2A CN: "
                          f"scale={s2_actual_scale}, "
                          f"start={s2_cn_start}, end={s2_cn_end}"
                          f"{' (auto-scaled)' if cn_auto_scale else ''}")
                    s2a_result = cn_pipe(
                        prompt=prompt,
                        negative_prompt=neg,
                        height=s2_h,
                        width=s2_w,
                        num_inference_steps=s2_steps,
                        sigmas=raw_sigmas_s2,
                        true_cfg_scale=s2_cfg,
                        generator=gen_s2,
                        latents=s2_latents,
                        control_image=source_s2,
                        control_mask=mask_s2,
                        controlnet_conditioning_scale=s2_actual_scale,
                        control_guidance_start=s2_cn_start,
                        control_guidance_end=s2_cn_end,
                        callback_on_step_end=make_cb(),
                        output_type=s2a_output_type,
                        **extra_kwargs,
                    )
                else:
                    s2a_result = pipe(
                        prompt=prompt,
                        negative_prompt=neg,
                        height=s2_h,
                        width=s2_w,
                        num_inference_steps=s2_steps,
                        sigmas=raw_sigmas_s2,
                        true_cfg_scale=s2_cfg,
                        generator=gen_s2,
                        latents=s2_latents,
                        callback_on_step_end=make_cb(),
                        output_type=s2a_output_type,
                        **extra_kwargs,
                    )

            # ════════════════════════════════════════════════════════════
            #  STAGE 2B — Whole-image Harmonization (no CN, no mask)
            # ════════════════════════════════════════════════════════════
            if do_s2_harmonize:
                _check_cancelled()
                print(f"[UltraGenInpaintCN] -- Stage 2B/{stage_count} "
                      f"[harmonize, no CN] --")

                s2b_latents = s2a_result.images  # latent from Phase A
                print(f"[UltraGenInpaintCN]   S2B latents from 2A: "
                      f"{s2b_latents.shape}")

                raw_sigmas_s2b = build_sigma_schedule(
                    s2_harmonize_steps, s2_harmonize_denoise, "karras"
                )
                actual_sigma_s2b = _compute_actual_start_sigma(
                    pipe.scheduler, raw_sigmas_s2b, s2_mu
                )
                print(f"[UltraGenInpaintCN]   S2B mu={s2_mu:.4f}, "
                      f"denoise={s2_harmonize_denoise:.2f}, "
                      f"raw_start={raw_sigmas_s2b[0]:.4f}, "
                      f"actual_start={actual_sigma_s2b:.4f}")

                # Use a deterministic generator offset for reproducibility
                gen_s2b = _make_generator(seed + 10 if seed > 0 else 0)

                s2b_latents = self._apply_denoise_noise(
                    s2b_latents, s2_harmonize_denoise,
                    actual_sigma_s2b, gen_s2b, device,
                )

                s2b_output_type = ("latent"
                                   if (do_stage3 or use_final_decode)
                                   else "pil")

                s2_result = pipe(
                    prompt=late_prompt,
                    negative_prompt=neg,
                    height=s2_h,
                    width=s2_w,
                    num_inference_steps=s2_harmonize_steps,
                    sigmas=raw_sigmas_s2b,
                    true_cfg_scale=s2_cfg,
                    generator=gen_s2b,
                    latents=s2b_latents,
                    callback_on_step_end=make_cb(),
                    output_type=s2b_output_type,
                    **extra_kwargs,
                )
                print(f"[UltraGenInpaintCN]   S2B harmonization complete")
            else:
                s2_result = s2a_result

            if not do_stage3:
                if use_final_decode:
                    pipe.transformer = pipe.transformer.to("cpu")
                    torch.cuda.empty_cache()
                    print(f"[UltraGenInpaintCN] Upscale VAE decode (2×) ...")
                    tensor = decode_latents_with_upscale_vae(
                        s2_result.images, upscale_vae, pipe.vae,
                        s2_h, s2_w, vae_scale_factor,
                    )
                    result_pil = _tensor_to_pil(tensor.squeeze(0))
                    result_pil = _composite_with_mask(
                        original_for_composite, result_pil,
                        mask_for_composite, feather,
                    )
                    tensor = pil_to_tensor(result_pil).unsqueeze(0)
                    print(f"[UltraGenInpaintCN] Output: {result_pil.size[0]}×{result_pil.size[1]} (composited)")
                    return (tensor,)

                pil_out = s2_result.images[0]
                result = _composite_with_mask(
                    original_for_composite, pil_out,
                    mask_for_composite, feather,
                )
                print(f"[UltraGenInpaintCN] Output: {result.size[0]}×{result.size[1]} (composited)")
                tensor = pil_to_tensor(result).unsqueeze(0)
                return (tensor,)

            s2_final_latents = s2_result.images
            print(f"[UltraGenInpaintCN]   S2 final latents: {s2_final_latents.shape}")

            # ════════════════════════════════════════════════════════════
            #  STAGE 3 — Final Polish (standard pipeline, no CN)
            # ════════════════════════════════════════════════════════════
            _check_cancelled()
            print(f"[UltraGenInpaintCN] -- Stage 3/{stage_count} [no CN] --")

            _apply_lora_stage_weights(pipe, pipeline, stage=3)

            if use_inter_stage:
                print("[UltraGenInpaintCN]   Inter-stage VAE upscale (S2→S3) ...")
                s3_latents, s3_h, s3_w = upscale_between_stages(
                    s2_final_latents, upscale_vae, pipe.vae,
                    s2_h, s2_w, vae_scale_factor,
                )
                s3_seq = _packed_seq_len(s3_h, s3_w, vae_scale_factor)
                s3_mu = _compute_mu(s3_seq, pipe.scheduler)
            else:
                s3_latents = _upscale_latents(
                    s2_final_latents, s2_h, s2_w, s3_h, s3_w, vae_scale_factor
                )
            print(f"[UltraGenInpaintCN]   S3 latents after upscale: {s3_latents.shape}")

            raw_sigmas_s3 = build_sigma_schedule(
                s3_steps, s3_denoise, s3_sigma_schedule
            )
            actual_sigma_s3 = _compute_actual_start_sigma(
                pipe.scheduler, raw_sigmas_s3, s3_mu
            )
            print(f"[UltraGenInpaintCN]   S3 mu={s3_mu:.4f}, "
                  f"schedule={s3_sigma_schedule}, "
                  f"raw_start={raw_sigmas_s3[0]:.4f}, "
                  f"actual_start={actual_sigma_s3:.4f}")

            s3_latents = self._apply_denoise_noise(
                s3_latents, s3_denoise, actual_sigma_s3, gen_s3, device
            )

            s3_output_type = "latent" if use_final_decode else "pil"
            s3_result = pipe(
                prompt=late_prompt,
                negative_prompt=neg,
                height=s3_h,
                width=s3_w,
                num_inference_steps=s3_steps,
                sigmas=raw_sigmas_s3,
                true_cfg_scale=s3_cfg,
                generator=gen_s3,
                latents=s3_latents,
                callback_on_step_end=make_cb(),
                output_type=s3_output_type,
                **extra_kwargs,
            )

            if use_final_decode:
                pipe.transformer = pipe.transformer.to("cpu")
                torch.cuda.empty_cache()
                print(f"[UltraGenInpaintCN] Upscale VAE decode (2×) ...")
                tensor = decode_latents_with_upscale_vae(
                    s3_result.images, upscale_vae, pipe.vae,
                    s3_h, s3_w, vae_scale_factor,
                )
                result_pil = _tensor_to_pil(tensor.squeeze(0))
                result_pil = _composite_with_mask(
                    original_for_composite, result_pil,
                    mask_for_composite, feather,
                )
                tensor = pil_to_tensor(result_pil).unsqueeze(0)
                print(f"[UltraGenInpaintCN] Output: {result_pil.size[0]}×{result_pil.size[1]} (composited)")
                return (tensor,)

            pil_out = s3_result.images[0]
            result = _composite_with_mask(
                original_for_composite, pil_out,
                mask_for_composite, feather,
            )
            print(f"[UltraGenInpaintCN] Output: {result.size[0]}×{result.size[1]} (composited)")
            tensor = pil_to_tensor(result).unsqueeze(0)
            return (tensor,)

        finally:
            # Remove auto-scaling hooks
            for _h in _cn_hooks:
                _h.remove()

            # Offload VAE back to CPU
            if offload_vae and hasattr(pipe, "vae"):
                pipe.vae = pipe.vae.to("cpu")
                torch.cuda.empty_cache()

    # ─────────────────────────────────────────────────────────────────
    #  Helpers
    # ─────────────────────────────────────────────────────────────────

    @staticmethod
    def _build_sigmas(num_steps: int, denoise: float) -> list:
        """Build a linear sigma schedule (legacy helper)."""
        full_sigmas = np.linspace(1.0, 1.0 / num_steps, num_steps)
        if denoise >= 1.0:
            return full_sigmas.tolist()
        keep = max(1, int(round(num_steps * denoise)))
        return full_sigmas[-keep:].tolist()

    @staticmethod
    def _apply_denoise_noise(latents: torch.Tensor, denoise: float,
                             actual_sigma: float,
                             generator, device: str) -> torch.Tensor:
        """Add flow-matching noise to latents for partial denoise."""
        if denoise >= 1.0:
            noise = torch.randn_like(latents)
            return noise

        noise = torch.randn(latents.shape, generator=generator,
                            device=latents.device, dtype=latents.dtype)
        return _add_noise_flowmatch(latents, noise, actual_sigma)
