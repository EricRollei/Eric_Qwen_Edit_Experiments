# Copyright (c) 2026 Eric Hiss. All rights reserved.
# Licensed under the terms in LICENSE.txt (CC BY-NC 4.0 / Commercial dual license).
# https://github.com/EricRollei/Eric_Qwen_Edit_Experiments
"""
Eric Qwen UltraGen CN — ControlNet-Guided Multi-Stage Generation

Duplicate of EricQwenImageUltraGen with ControlNet support on Stage 1.
Uses QwenImageControlNetPipeline for the initial draft generation to
guide composition/structure from a control image (canny, depth, pose,
soft edge), then refines with the standard QwenImagePipeline in later
stages.

ControlNet Model:
- InstantX/Qwen-Image-ControlNet-Union (canny, soft edge, depth, pose)
- https://huggingface.co/InstantX/Qwen-Image-ControlNet-Union

Base Model Credits:
- Qwen-Image developed by Qwen Team (Alibaba)
- https://github.com/QwenLM/Qwen-Image

Author: Eric Hiss (GitHub: EricRollei)
"""

import math
import torch
import numpy as np
from PIL import Image
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


def _tensor_to_pil(tensor: torch.Tensor) -> Image.Image:
    """Convert a ComfyUI IMAGE tensor (B, H, W, C) float [0,1] to PIL."""
    if tensor.dim() == 4:
        tensor = tensor[0]
    arr = (tensor.cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
    return Image.fromarray(arr)


# ── Control-image fit modes ─────────────────────────────────────────────

CN_FIT_MODES = [
    "match_control",
    "crop_center",
    "crop_top_left",
    "crop_top_right",
    "crop_bottom_left",
    "crop_bottom_right",
    "pad_black_left",
    "pad_black_right",
    "pad_black_top",
    "stretch",
]


def _fit_control_image(
    pil_img: Image.Image, target_w: int, target_h: int, mode: str
) -> Image.Image:
    """Resize a control image to *exactly* (target_w, target_h) using
    the chosen fit strategy.  All crop/pad variants preserve aspect
    ratio of the source image; only 'stretch' distorts it.
    """
    src_w, src_h = pil_img.size

    if mode == "stretch" or mode == "match_control":
        # match_control already matched the aspect ratio, so simple resize
        return pil_img.resize((target_w, target_h), Image.LANCZOS)

    if mode.startswith("crop_"):
        # Scale so the image *covers* the target (scale by larger ratio)
        scale = max(target_w / src_w, target_h / src_h)
        new_w = int(round(src_w * scale))
        new_h = int(round(src_h * scale))
        scaled = pil_img.resize((new_w, new_h), Image.LANCZOS)

        # Compute crop box
        if mode == "crop_center":
            left = (new_w - target_w) // 2
            top = (new_h - target_h) // 2
        elif mode == "crop_top_left":
            left, top = 0, 0
        elif mode == "crop_top_right":
            left, top = new_w - target_w, 0
        elif mode == "crop_bottom_left":
            left, top = 0, new_h - target_h
        elif mode == "crop_bottom_right":
            left, top = new_w - target_w, new_h - target_h
        else:
            left = (new_w - target_w) // 2
            top = (new_h - target_h) // 2

        return scaled.crop((left, top, left + target_w, top + target_h))

    if mode.startswith("pad_black_"):
        # Scale so the image *fits inside* the target (scale by smaller ratio)
        scale = min(target_w / src_w, target_h / src_h)
        new_w = int(round(src_w * scale))
        new_h = int(round(src_h * scale))
        scaled = pil_img.resize((new_w, new_h), Image.LANCZOS)

        canvas = Image.new("RGB", (target_w, target_h), (0, 0, 0))
        pad_side = mode.replace("pad_black_", "")

        if pad_side == "left":
            x = target_w - new_w
            y = (target_h - new_h) // 2
        elif pad_side == "right":
            x = 0
            y = (target_h - new_h) // 2
        elif pad_side == "top":
            x = (target_w - new_w) // 2
            y = target_h - new_h
        else:
            x = (target_w - new_w) // 2
            y = (target_h - new_h) // 2

        canvas.paste(scaled, (x, y))
        return canvas

    # Fallback — simple resize
    return pil_img.resize((target_w, target_h), Image.LANCZOS)


def _closest_aspect_ratio(
    img_w: int, img_h: int
) -> tuple:
    """Find the ASPECT_RATIOS entry closest to the image's actual ratio.
    Returns (label, (w_ratio, h_ratio)).
    """
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
#  UltraGen CN Node
# ═══════════════════════════════════════════════════════════════════════

class EricQwenImageUltraGenCN:
    """
    ControlNet-guided multi-stage text-to-image generation.

    Identical to UltraGen but uses a ControlNet on Stage 1 to guide
    the initial composition from a control image (depth map, canny
    edges, soft edges, or pose skeleton).

    Stages 2 and 3 refine using the standard pipeline (no ControlNet)
    to avoid the ControlNet constraining fine detail.

    Supported control modes (InstantX Union model):
    • canny — edge detection map
    • soft_edge — soft edge detection (AnylineDetector)
    • depth — depth map (e.g. Depth-Anything-V2)
    • pose — skeleton/pose map (DWPose)
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
                    "tooltip": "From the Qwen-Image ControlNet Loader"
                }),
                "control_image": ("IMAGE", {
                    "tooltip": (
                        "The control/condition image (canny edges, depth map,\n"
                        "pose skeleton, or soft edges). Will be fitted to\n"
                        "generation dimensions using cn_fit_mode."
                    )
                }),
                "cn_fit_mode": (CN_FIT_MODES, {
                    "default": "match_control",
                    "tooltip": (
                        "How the control image is fitted to generation dimensions:\n"
                        "\u2022 match_control \u2014 override aspect ratio to match control image\n"
                        "\u2022 crop_center \u2014 scale to cover, center crop\n"
                        "\u2022 crop_top_left/right \u2014 crop from corner\n"
                        "\u2022 crop_bottom_left/right \u2014 crop from corner\n"
                        "\u2022 pad_black_left/right/top \u2014 fit inside, pad with black\n"
                        "\u2022 stretch \u2014 distort to fill (legacy)"
                    )
                }),
                "prompt": ("STRING", {
                    "multiline": True,
                    "tooltip": (
                        "Describe the image you want to generate.\n"
                        "For best results, use detailed descriptions (~200 words).\n"
                        "Connect a Prompt Rewriter node to auto-enhance short prompts."
                    )
                }),
            },
            "optional": {
                "negative_prompt": ("STRING", {
                    "multiline": True,
                    "default": DEFAULT_NEGATIVE_PROMPT,
                    "tooltip": (
                        "What to avoid. Default is the official Qwen-Image-2512\n"
                        "negative prompt (Chinese)."
                    )
                }),
                "aspect_ratio": (ratio_names, {
                    "default": "1:1   Square",
                    "tooltip": "Aspect ratio applied at every stage"
                }),
                "seed": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 0xffffffffffffffff,
                    "tooltip": "Random seed (0 = random)"
                }),
                "seed_mode": (["same_all_stages", "offset_per_stage", "random_per_stage"], {
                    "default": "offset_per_stage",
                    "tooltip": (
                        "How seeds are chosen for each stage:\n"
                        "• same_all_stages — one generator for all stages\n"
                        "• offset_per_stage — S2 uses seed+1, S3 uses seed+2\n"
                        "• random_per_stage — independent random seed per stage"
                    )
                }),

                # ── Pipeline parameters ──────────────────────────────────
                "max_sequence_length": ("INT", {
                    "default": 1024,
                    "min": 128,
                    "max": 1024,
                    "step": 64,
                    "tooltip": "Maximum prompt token length for the text encoder."
                }),

                # ── ControlNet parameters (Stage 1) ─────────────────────
                "cn_auto_scale": ("BOOLEAN", {
                    "default": True,
                    "tooltip": (
                        "Auto-calibrate ControlNet strength to match\n"
                        "the transformer's internal activation scale.\n"
                        "Compensates for custom/finetuned transformers\n"
                        "that have different hidden-state magnitudes\n"
                        "than the base model the ControlNet was trained on.\n"
                        "When enabled, cn_target_strength controls the\n"
                        "final CN influence and conditioning_scale is ignored."
                    )
                }),
                "cn_target_strength": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.1,
                    "max": 6.0,
                    "step": 0.1,
                    "tooltip": (
                        "ControlNet influence strength (auto-scale mode).\n"
                        "1.0 = standard (recommended starting point).\n"
                        "0.5 = subtle structural hints.\n"
                        "1.5\u20132.0 = strong structural guidance.\n"
                        "3.0+ = very strong (may over-constrain).\n"
                        "Internally maps to hidden-state ratio (1.0 = 5%)."
                    )
                }),
                "controlnet_conditioning_scale": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.0,
                    "max": 50.0,
                    "step": 0.5,
                    "tooltip": (
                        "Manual ControlNet influence strength on Stage 1.\n"
                        "Only used when cn_auto_scale is OFF.\n"
                        "With custom/finetuned transformers, values of\n"
                        "10–30 may be needed (base model uses 1.0).\n"
                        "0.0 = ControlNet effectively disabled."
                    )
                }),
                "control_guidance_start": ("FLOAT", {
                    "default": 0.0,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.05,
                    "tooltip": (
                        "When ControlNet guidance begins (fraction of S1 steps).\n"
                        "0.0 = from the start (default)."
                    )
                }),
                "control_guidance_end": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.05,
                    "tooltip": (
                        "When ControlNet guidance ends (fraction of S1 steps).\n"
                        "1.0 = through the end (default).\n"
                        "0.5 = ControlNet guides composition in early steps,\n"
                        "then frees up for creative detail in later steps."
                    )
                }),

                # ── ControlNet parameters (Stage 2) ─────────────────────
                "s2_cn_scale": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.0,
                    "max": 50.0,
                    "step": 0.5,
                    "tooltip": (
                        "ControlNet influence strength on Stage 2.\n"
                        "Only used when cn_auto_scale is OFF.\n"
                        "When auto_scale is ON, S2 uses the same\n"
                        "auto-calibrated factor as S1 (scaled by this\n"
                        "value relative to S1 — e.g. 0.5 = half S1 strength).\n"
                        "0.0 = ControlNet disabled for S2."
                    )
                }),
                "s2_cn_start": ("FLOAT", {
                    "default": 0.0,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.05,
                    "tooltip": (
                        "When ControlNet guidance begins in Stage 2\n"
                        "(fraction of S2 steps). 0.0 = from the start."
                    )
                }),
                "s2_cn_end": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.05,
                    "tooltip": (
                        "When ControlNet guidance ends in Stage 2\n"
                        "(fraction of S2 steps).\n"
                        "1.0 = through the end (default).\n"
                        "0.5 = guide structure in early S2 steps only."
                    )
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
                    "default": 15,
                    "min": 1,
                    "max": 200,
                    "step": 1,
                    "tooltip": "Stage 1 inference steps (ControlNet-guided txt2img)"
                }),
                "s1_cfg": ("FLOAT", {
                    "default": 10.0,
                    "min": 1.0,
                    "max": 20.0,
                    "step": 0.5,
                    "tooltip": "Stage 1 true CFG scale."
                }),

                # ── Stage 2 ─────────────────────────────────────────────
                "upscale_to_stage2": ("FLOAT", {
                    "default": 4.0,
                    "min": 0.0,
                    "max": 10.0,
                    "step": 0.5,
                    "tooltip": (
                        "Upscale factor (area) from Stage 1 to Stage 2.\n"
                        "Set to 0 to skip Stage 2 & 3 (output Stage 1 only)."
                    )
                }),
                "s2_steps": ("INT", {
                    "default": 26,
                    "min": 1,
                    "max": 200,
                    "step": 1,
                    "tooltip": "Stage 2 inference steps (refinement, optional ControlNet)."
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
                    "tooltip": "Sigma schedule curve for Stage 2 refinement."
                }),

                # ── Stage 3 ─────────────────────────────────────────────
                "upscale_to_stage3": ("FLOAT", {
                    "default": 2.0,
                    "min": 0.0,
                    "max": 8.0,
                    "step": 0.5,
                    "tooltip": (
                        "Upscale factor (area) from Stage 2 to Stage 3.\n"
                        "Default 0 = disabled (2-stage output)."
                    )
                }),
                "s3_steps": ("INT", {
                    "default": 18,
                    "min": 1,
                    "max": 200,
                    "step": 1,
                    "tooltip": "Stage 3 inference steps"
                }),
                "s3_cfg": ("FLOAT", {
                    "default": 2.0,
                    "min": 1.0,
                    "max": 20.0,
                    "step": 0.5,
                    "tooltip": "Stage 3 true CFG scale"
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
                    "tooltip": "Sigma schedule curve for Stage 3 final polish."
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
                    "default": "both",
                    "tooltip": (
                        "How the upscale VAE is used (requires upscale_vae).\n"
                        "• disabled — upscale VAE ignored\n"
                        "• inter_stage — decode S2→2×→re-encode for S3\n"
                        "• final_decode — 2× upscale on final output\n"
                        "• both — inter-stage + final decode"
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
        control_image: torch.Tensor,
        cn_fit_mode: str = "match_control",
        prompt: str = "",
        negative_prompt: str = DEFAULT_NEGATIVE_PROMPT,
        aspect_ratio: str = "1:1   Square",
        seed: int = 0,
        seed_mode: str = "same_all_stages",
        # Pipeline params
        max_sequence_length: int = 512,
        # ControlNet params (Stage 1)
        cn_auto_scale: bool = True,
        cn_target_strength: float = 1.0,
        controlnet_conditioning_scale: float = 1.0,
        control_guidance_start: float = 0.0,
        control_guidance_end: float = 1.0,
        # ControlNet params (Stage 2)
        s2_cn_scale: float = 1.0,
        s2_cn_start: float = 0.0,
        s2_cn_end: float = 1.0,
        # Stage 1
        s1_mp: float = 0.5,
        s1_steps: int = 15,
        s1_cfg: float = 10.0,
        # Stage 2
        upscale_to_stage2: float = 7.0,
        s2_steps: int = 30,
        s2_cfg: float = 4.0,
        s2_denoise: float = 0.80,
        s2_sigma_schedule: str = "linear",
        # Stage 3
        upscale_to_stage3: float = 2.0,
        s3_steps: int = 18,
        s3_cfg: float = 2.0,
        s3_denoise: float = 0.40,
        s3_sigma_schedule: str = "linear",
        # Upscale VAE (optional)
        upscale_vae=None,
        upscale_vae_mode: str = "disabled",
    ) -> Tuple[torch.Tensor]:
        pipe = pipeline["pipeline"]
        cn_model = controlnet["model"]
        offload_vae = pipeline.get("offload_vae", False)

        # ── Convert ComfyUI tensor to PIL for ControlNet ────────────────
        control_pil = _tensor_to_pil(control_image)
        print(f"[UltraGenCN] Control image: {control_pil.size[0]}×{control_pil.size[1]} "
              f"mode={control_pil.mode}")

        # ── Diagnostic: verify ControlNet model ─────────────────────────
        cn_params = sum(p.numel() for p in cn_model.parameters()) / 1e6
        cn_device = next(cn_model.parameters()).device
        cn_dtype = next(cn_model.parameters()).dtype
        print(f"[UltraGenCN] ControlNet model: {cn_params:.0f}M params, "
              f"device={cn_device}, dtype={cn_dtype}")

        # ── Build ControlNet pipeline for Stage 1 ───────────────────────
        # Construct directly instead of using from_pipe() which calls
        # .to(dtype) and can trigger OOM on already-loaded models.
        from diffusers import QwenImageControlNetPipeline
        cn_pipe = QwenImageControlNetPipeline(
            scheduler=pipe.scheduler,
            vae=pipe.vae,
            text_encoder=pipe.text_encoder,
            tokenizer=pipe.tokenizer,
            transformer=pipe.transformer,
            controlnet=cn_model,
        )
        # Verify controlnet is properly registered on pipeline
        assert hasattr(cn_pipe, 'controlnet') and cn_pipe.controlnet is cn_model, \
            "[UltraGenCN] FATAL: controlnet not properly registered on pipeline!"
        print(f"[UltraGenCN] ControlNet pipeline built (zero-copy from base, "
              f"controlnet={type(cn_pipe.controlnet).__name__})")

        # ── Resolve aspect ratio ────────────────────────────────────────
        if cn_fit_mode == "match_control":
            ctrl_w, ctrl_h = control_pil.size
            matched_label, (w_ratio, h_ratio) = _closest_aspect_ratio(
                ctrl_w, ctrl_h
            )
            print(f"[UltraGenCN] match_control: control {ctrl_w}\u00d7{ctrl_h} "
                  f"\u2192 {matched_label} ({w_ratio}:{h_ratio})")
        else:
            w_ratio, h_ratio = ASPECT_RATIOS.get(aspect_ratio, (1, 1))

        # ── Determine which stages are active ───────────────────────────
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

        # ── Fit control image to S1 dimensions ────────────────────────
        control_pil_s1 = _fit_control_image(
            control_pil, s1_w, s1_h, cn_fit_mode
        )
        print(f"[UltraGenCN] Control image fitted to {s1_w}\u00d7{s1_h} "
              f"(S1, mode={cn_fit_mode})")

        # ── Fit control image to S2 dimensions (if S2 CN enabled) ──────
        control_pil_s2 = None
        use_cn_s2 = do_stage2 and s2_cn_scale > 0.0
        if use_cn_s2:
            control_pil_s2 = _fit_control_image(
                control_pil, s2_w, s2_h, cn_fit_mode
            )
            print(f"[UltraGenCN] Control image fitted to {s2_w}\u00d7{s2_h} "
                  f"(S2, mode={cn_fit_mode})")

        # ── Resolve upscale VAE mode ────────────────────────────────────
        use_inter_stage = False
        use_final_decode = False
        if upscale_vae is not None and upscale_vae_mode != "disabled":
            if upscale_vae_mode in ("inter_stage", "both"):
                if do_stage3:
                    use_inter_stage = True
                else:
                    print("[UltraGenCN] WARNING: inter_stage upscale requires "
                          "3 stages. Falling back to final_decode.")
                    use_final_decode = True
            if upscale_vae_mode in ("final_decode", "both"):
                use_final_decode = True

        stage_count = 3 if do_stage3 else (2 if do_stage2 else 1)
        total_steps = s1_steps + (s2_steps if do_stage2 else 0) + (s3_steps if do_stage3 else 0)

        # ── Common setup ────────────────────────────────────────────────
        device = pipe._execution_device if hasattr(pipe, "_execution_device") else "cuda"

        # ── Ensure transformer is on GPU ────────────────────────────────
        transformer_device = next(pipe.transformer.parameters()).device
        if str(transformer_device) != str(device):
            print(f"[UltraGenCN] Moving transformer back to {device}")
            pipe.transformer = pipe.transformer.to(device)
            torch.cuda.empty_cache()

        # ── Ensure ControlNet is on same device/dtype as transformer ────
        cn_device = next(cn_model.parameters()).device
        cn_dtype = next(cn_model.parameters()).dtype
        t_dtype = next(pipe.transformer.parameters()).dtype
        if str(cn_device) != str(device) or cn_dtype != t_dtype:
            print(f"[UltraGenCN] Moving ControlNet to {device} ({t_dtype})")
            cn_model.to(device=device, dtype=t_dtype)

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
        else:  # same_all_stages
            gen_s1 = _make_generator(seed)
            gen_s2 = gen_s1
            gen_s3 = gen_s1
            seed_info = f"same: {seed if seed>0 else 'random'}"
        generator = gen_s1

        # ── Print plan ──────────────────────────────────────────────────
        print(f"\n{'='*60}")
        print(f"[UltraGenCN] ControlNet multi-stage: {stage_count} stage(s), "
              f"{total_steps} total steps")
        print(f"[UltraGenCN] ControlNet scale={controlnet_conditioning_scale:.2f}, "
              f"guidance=[{control_guidance_start:.2f}, {control_guidance_end:.2f}]")
        print(f"[UltraGenCN] max_seq_len={max_sequence_length}, "
              f"seed_mode={seed_mode} ({seed_info})")
        print(f"  Stage 1: {s1_w}×{s1_h} ({s1_mp_actual:.2f} MP), "
              f"{s1_steps} steps, CFG={s1_cfg:.1f} [ControlNet]")
        if do_stage2:
            print(f"  Stage 2: {s2_w}×{s2_h} ({s2_mp_actual:.2f} MP), "
                  f"{s2_steps} steps, CFG={s2_cfg:.1f}, "
                  f"upscale={upscale_to_stage2:.1f}× area, "
                  f"denoise={s2_denoise:.2f}, σ-schedule={s2_sigma_schedule}")
        if do_stage3:
            print(f"  Stage 3: {s3_w}×{s3_h} ({s3_mp_actual:.2f} MP), "
                  f"{s3_steps} steps, CFG={s3_cfg:.1f}, "
                  f"upscale={upscale_to_stage3:.1f}× area, "
                  f"denoise={s3_denoise:.2f}, σ-schedule={s3_sigma_schedule}")
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

        # ── Spectrum acceleration ────────────────────────────────────────
        # NOTE: Spectrum is DISABLED for Stage 1 when ControlNet is active.
        # The Spectrum forward skips the transformer block loop on cached
        # steps, which means ControlNet residuals would be ignored.
        # Spectrum can still accelerate Stages 2 & 3 (no CN by default).
        spectrum_config = getattr(pipe, "_spectrum_config", None)

        if spectrum_config is not None:
            print("[UltraGenCN] Spectrum auto-disabled for Stage 1 "
                  "(incompatible with ControlNet — cached steps would "
                  "skip controlnet residuals)")

        # ── CN auto-scaling hooks ───────────────────────────────────
        # The ControlNet was trained with the base Qwen transformer.
        # Custom/finetuned transformers can have very different
        # hidden-state magnitudes, making CN residuals negligible
        # (measured at ~0.01× without scaling). Auto-scaling measures
        # the ratio on the first transformer call and compensates.
        _cn_hooks = []
        _auto = {"calibrated": False, "factor": 1.0,
                 "cn_mag": 0.0, "hs_mag": 0.0}
        # Convert GUI value (1.0 = standard) to internal ratio (0.05)
        _cn_ratio = cn_target_strength / 20.0

        if cn_auto_scale:
            def _block0_calibrate(_mod, _inp, output):
                """Capture hidden-state magnitude from block[0] output."""
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
                """Scale CN block samples to match hidden-state magnitude."""
                cbs = kwargs.get("controlnet_block_samples")
                if cbs is None:
                    return
                # Capture CN magnitude on first call
                if _auto["cn_mag"] == 0.0:
                    _auto["cn_mag"] = cbs[0].abs().mean().item()
                    if _auto["hs_mag"] > 0.0:
                        _auto["factor"] = (_cn_ratio
                                           * _auto["hs_mag"]
                                           / _auto["cn_mag"])
                        _auto["calibrated"] = True
                # Apply scaling when calibrated
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
            #  STAGE 1 — ControlNet-guided draft (txt2img)
            # ════════════════════════════════════════════════════════════
            _check_cancelled()
            print(f"[UltraGenCN] -- Stage 1/{stage_count} [ControlNet] --")

            _apply_lora_stage_weights(pipe, pipeline, stage=1)

            # ── CN scale determination ──────────────────────────────
            if cn_auto_scale:
                s1_cn_scale = 1.0  # auto-scaling hooks handle strength
                print(f"[UltraGenCN]   CN mode: auto-scale "
                      f"(strength={cn_target_strength:.1f}, "
                      f"ratio={_cn_ratio:.4f})")
            else:
                s1_cn_scale = controlnet_conditioning_scale
                print(f"[UltraGenCN]   CN mode: manual "
                      f"(scale={s1_cn_scale:.2f})")

            print(f"[UltraGenCN]   S1 CN: guidance=[{control_guidance_start:.2f}, "
                  f"{control_guidance_end:.2f}], img={control_pil_s1.size}")

            s1_output_type = "latent" if (do_stage2 or use_final_decode) else "pil"
            s1_result = cn_pipe(
                prompt=prompt,
                negative_prompt=neg,
                control_image=control_pil_s1,
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
                print(f"[UltraGenCN]   CN auto-scale: "
                      f"hs={_auto['hs_mag']:.0f}, "
                      f"cn={_auto['cn_mag']:.0f}, "
                      f"raw_ratio={raw_ratio:.4f}, "
                      f"boost={_auto['factor']:.1f}\u00d7 "
                      f"(strength={cn_target_strength:.1f})")
            elif cn_auto_scale:
                print("[UltraGenCN]   CN auto-scale: WARNING \u2014 "
                      "could not calibrate (no block samples received?)"
                      f" [strength={cn_target_strength:.1f}]")


            if not do_stage2:
                if use_final_decode:
                    pipe.transformer = pipe.transformer.to("cpu")
                    torch.cuda.empty_cache()
                    print(f"[UltraGenCN] Upscale VAE decode (2×) ...")
                    tensor = decode_latents_with_upscale_vae(
                        s1_result.images, upscale_vae, pipe.vae,
                        s1_h, s1_w, vae_scale_factor,
                    )
                    print(f"[UltraGenCN] Output: {tensor.shape[2]}×{tensor.shape[1]} (2× upscaled)")
                    return (tensor,)
                pil_image = s1_result.images[0]
                print(f"[UltraGenCN] Output: {pil_image.size[0]}×{pil_image.size[1]}")
                tensor = pil_to_tensor(pil_image).unsqueeze(0)
                return (tensor,)

            s1_latents = s1_result.images  # packed latents
            print(f"[UltraGenCN]   S1 latents shape: {s1_latents.shape}")

            # ════════════════════════════════════════════════════════════
            #  STAGE 2 — Main Refinement (standard pipeline, no CN)
            # ════════════════════════════════════════════════════════════
            _check_cancelled()
            print(f"[UltraGenCN] -- Stage 2/{stage_count} --")

            _apply_lora_stage_weights(pipe, pipeline, stage=2)

            s2_latents = _upscale_latents(
                s1_latents, s1_h, s1_w, s2_h, s2_w, vae_scale_factor
            )
            print(f"[UltraGenCN]   S2 latents after upscale: {s2_latents.shape}")

            raw_sigmas_s2 = build_sigma_schedule(
                s2_steps, s2_denoise, s2_sigma_schedule
            )
            actual_sigma_s2 = _compute_actual_start_sigma(
                pipe.scheduler, raw_sigmas_s2, s2_mu
            )
            print(f"[UltraGenCN]   S2 mu={s2_mu:.4f}, schedule={s2_sigma_schedule}, "
                  f"raw_start={raw_sigmas_s2[0]:.4f}, "
                  f"actual_start={actual_sigma_s2:.4f}")

            s2_latents = self._apply_denoise_noise(
                s2_latents, s2_denoise, actual_sigma_s2, gen_s2, device
            )

            # Use ControlNet pipeline for S2 if scale > 0, else base
            s2_output_type = "latent" if (do_stage3 or use_final_decode) else "pil"
            if use_cn_s2:
                # When auto-scale is ON, s2_cn_scale acts as relative
                # multiplier (e.g. 0.5 = half of S1 auto-scaled strength)
                s2_actual_scale = s2_cn_scale if not cn_auto_scale else s2_cn_scale
                print(f"[UltraGenCN]   S2 ControlNet: scale={s2_actual_scale}, "
                      f"start={s2_cn_start}, end={s2_cn_end}"
                      f"{' (auto-scaled)' if cn_auto_scale else ''}")
                s2_result = cn_pipe(
                    prompt=prompt,
                    negative_prompt=neg,
                    height=s2_h,
                    width=s2_w,
                    num_inference_steps=s2_steps,
                    sigmas=raw_sigmas_s2,
                    true_cfg_scale=s2_cfg,
                    generator=gen_s2,
                    latents=s2_latents,
                    control_image=control_pil_s2,
                    controlnet_conditioning_scale=s2_actual_scale,
                    control_guidance_start=s2_cn_start,
                    control_guidance_end=s2_cn_end,
                    callback_on_step_end=make_cb(),
                    output_type=s2_output_type,
                    **extra_kwargs,
                )
            else:
                s2_result = pipe(
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
                    output_type=s2_output_type,
                    **extra_kwargs,
                )

            if not do_stage3:
                if use_final_decode:
                    pipe.transformer = pipe.transformer.to("cpu")
                    torch.cuda.empty_cache()
                    print(f"[UltraGenCN] Upscale VAE decode (2×) ...")
                    tensor = decode_latents_with_upscale_vae(
                        s2_result.images, upscale_vae, pipe.vae,
                        s2_h, s2_w, vae_scale_factor,
                    )
                    print(f"[UltraGenCN] Output: {tensor.shape[2]}×{tensor.shape[1]} (2× upscaled)")
                    return (tensor,)
                pil_image = s2_result.images[0]
                print(f"[UltraGenCN] Output: {pil_image.size[0]}×{pil_image.size[1]}")
                tensor = pil_to_tensor(pil_image).unsqueeze(0)
                return (tensor,)

            s2_final_latents = s2_result.images
            print(f"[UltraGenCN]   S2 final latents shape: {s2_final_latents.shape}")

            # ════════════════════════════════════════════════════════════
            #  STAGE 3 — Final Polish (standard pipeline, no CN)
            # ════════════════════════════════════════════════════════════
            _check_cancelled()
            print(f"[UltraGenCN] -- Stage 3/{stage_count} --")

            _apply_lora_stage_weights(pipe, pipeline, stage=3)

            if use_inter_stage:
                print("[UltraGenCN]   Inter-stage VAE upscale (S2→S3) ...")
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
            print(f"[UltraGenCN]   S3 latents after upscale: {s3_latents.shape}")

            raw_sigmas_s3 = build_sigma_schedule(
                s3_steps, s3_denoise, s3_sigma_schedule
            )
            actual_sigma_s3 = _compute_actual_start_sigma(
                pipe.scheduler, raw_sigmas_s3, s3_mu
            )
            print(f"[UltraGenCN]   S3 mu={s3_mu:.4f}, schedule={s3_sigma_schedule}, "
                  f"raw_start={raw_sigmas_s3[0]:.4f}, "
                  f"actual_start={actual_sigma_s3:.4f}")

            s3_latents = self._apply_denoise_noise(
                s3_latents, s3_denoise, actual_sigma_s3, gen_s3, device
            )

            s3_output_type = "latent" if use_final_decode else "pil"
            s3_result = pipe(
                prompt=prompt,
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
                print(f"[UltraGenCN] Upscale VAE decode (2×) ...")
                tensor = decode_latents_with_upscale_vae(
                    s3_result.images, upscale_vae, pipe.vae,
                    s3_h, s3_w, vae_scale_factor,
                )
                print(f"[UltraGenCN] Output: {tensor.shape[2]}×{tensor.shape[1]} (2× upscaled)")
                return (tensor,)

            pil_image = s3_result.images[0]
            print(f"[UltraGenCN] Output: {pil_image.size[0]}×{pil_image.size[1]}")
            tensor = pil_to_tensor(pil_image).unsqueeze(0)
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
    #  Helpers (identical to UltraGen)
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
