# Copyright (c) 2026 Eric Hiss. All rights reserved.
# Licensed under the terms in LICENSE.txt (CC BY-NC 4.0 / Commercial dual license).
# https://github.com/EricRollei/Eric_Qwen_Edit_Experiments
"""
Eric Qwen UltraGen — Quality-Focused Multi-Stage Generation (v2)

Enhanced text-to-image generation node incorporating Qwen-Image-2512 best
practices discovered through analysis of the official repository, HuggingFace
model card, and pipeline source code.

Key improvements over the original multi-stage node:
  - max_sequence_length exposed (512-1024) for detailed prompt capture
  - Official Chinese negative prompt as default
  - Spectrum acceleration support on stage 1
  - Defaults tuned from user testing (0.5 MP s1, 4x upscale, offset seeds, karras S3)
  - Full per-stage control retained for experimentation

Architecture:
  Stage 1: Generate at low resolution with high CFG for composition
  Stage 2: Upscale + refine at target resolution with most sampling steps
  Stage 3: (Optional) Final upscale + light refine for ultra-high res

Model Credits:
  - Qwen-Image developed by Qwen Team (Alibaba)
  - https://github.com/QwenLM/Qwen-Image

Author: Eric Hiss (GitHub: EricRollei)
"""

import math
import torch
import numpy as np
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


# ═══════════════════════════════════════════════════════════════════════
#  Official Qwen-Image-2512 recommended resolutions (~1.76 MP)
#  Source: https://github.com/QwenLM/Qwen-Image Quick Start
# ═══════════════════════════════════════════════════════════════════════

QWEN_OFFICIAL_RESOLUTIONS = {
    # (w_ratio, h_ratio): (width, height)   — all ~1.5-1.76 MP
    (1, 1):   (1328, 1328),   # 1.76 MP
    (16, 9):  (1664, 928),    # 1.54 MP
    (9, 16):  (928, 1664),    # 1.54 MP
    (4, 3):   (1472, 1104),   # 1.62 MP
    (3, 4):   (1104, 1472),   # 1.62 MP
    (3, 2):   (1584, 1056),   # 1.67 MP
    (2, 3):   (1056, 1584),   # 1.67 MP
}

# ═══════════════════════════════════════════════════════════════════════
#  Official negative prompt (Chinese)
#  Source: Qwen-Image-2512 official examples
#  Translation: Low resolution, low quality, limb deformity, finger
#  deformity, over-saturation, wax figure feel, face without detail,
#  overly smooth, AI look, chaotic composition, blurred text, distortion.
# ═══════════════════════════════════════════════════════════════════════

DEFAULT_NEGATIVE_PROMPT = (
    "低分辨率，低画质，肢体畸形，手指畸形，画面过饱和，蜡像感，"
    "人脸无细节，过度光滑，画面具有AI感。构图混乱。文字模糊，扭曲。"
)


# ═══════════════════════════════════════════════════════════════════════
#  UltraGen Node
# ═══════════════════════════════════════════════════════════════════════

class EricQwenImageUltraGen:
    """
    Quality-focused multi-stage text-to-image generation (v2).

    Incorporates all Qwen-Image-2512 best practices:
    • Official Chinese negative prompt as default
    • max_sequence_length up to 1024 for detailed prompts
    • Spectrum acceleration on Stage 1
    • Tuned defaults: 0.5 MP s1 → 4× upscale → 26-step s2 refinement

    Full per-stage control retained:
    - Set upscale_to_stage2 = 0 → output Stage 1 only
    - Set upscale_to_stage3 = 0 → stop after Stage 2
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
                        "negative prompt (Chinese). Translation: low resolution,\n"
                        "low quality, limb/finger deformity, over-saturation,\n"
                        "wax figure feel, no face detail, overly smooth, AI look,\n"
                        "chaotic composition, blurred text, distortion."
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
                        "• random_per_stage — independent random seed per stage\n"
                        "Different seeds per stage can improve diversity of detail\n"
                        "added during refinement while preserving S1 composition."
                    )
                }),

                # ── Pipeline parameters (NEW in v2) ─────────────────────
                "max_sequence_length": ("INT", {
                    "default": 1024,
                    "min": 128,
                    "max": 1024,
                    "step": 64,
                    "tooltip": (
                        "Maximum prompt token length for the text encoder.\n"
                        "Default 1024 for full prompt capacity.\n"
                        "Reduce to 512 if not using detailed prompts."
                    )
                }),

                # ── Stage 1 ─────────────────────────────────────────────
                "s1_mp": ("FLOAT", {
                    "default": 0.5,
                    "min": 0.3,
                    "max": 2.0,
                    "step": 0.1,
                    "tooltip": (
                        "Stage 1 resolution in megapixels.\n"
                        "Default 0.6 MP — low-res draft for composition.\n"
                        "The model's native sweet spot is ~1.76 MP."
                    )
                }),
                "s1_steps": ("INT", {
                    "default": 15,
                    "min": 1,
                    "max": 200,
                    "step": 1,
                    "tooltip": "Stage 1 inference steps (txt2img from noise)"
                }),
                "s1_cfg": ("FLOAT", {
                    "default": 10.0,
                    "min": 1.0,
                    "max": 20.0,
                    "step": 0.5,
                    "tooltip": (
                        "Stage 1 true CFG scale.\n"
                        "High CFG at low resolution helps lock in composition.\n"
                        "Pipeline uses norm-preserving CFG so high values are safer."
                    )
                }),

                # ── Stage 2 ─────────────────────────────────────────────
                "upscale_to_stage2": ("FLOAT", {
                    "default": 4.0,
                    "min": 0.0,
                    "max": 10.0,
                    "step": 0.5,
                    "tooltip": (
                        "Upscale factor (area) from Stage 1 to Stage 2.\n"
                        "Default 4.0 — from 0.5 MP to ~2.0 MP.\n"
                        "Set to 0 to skip Stage 2 & 3 (output Stage 1 only)."
                    )
                }),
                "s2_steps": ("INT", {
                    "default": 26,
                    "min": 1,
                    "max": 200,
                    "step": 1,
                    "tooltip": (
                        "Stage 2 inference steps.\n"
                        "This is the main refinement stage — most sampling\n"
                        "steps should go here for maximum quality."
                    )
                }),
                "s2_cfg": ("FLOAT", {
                    "default": 4.0,
                    "min": 1.0,
                    "max": 20.0,
                    "step": 0.5,
                    "tooltip": (
                        "Stage 2 true CFG scale.\n"
                        "4.0 matches official Qwen recommendation."
                    )
                }),
                "s2_denoise": ("FLOAT", {
                    "default": 0.85,
                    "min": 0.1,
                    "max": 1.0,
                    "step": 0.05,
                    "tooltip": (
                        "Stage 2 denoise strength.\n"
                        "1.0 = full re-denoise (ignores Stage 1 structure).\n"
                        "0.5-0.7 = refine while preserving composition.\n"
                        "Lower values preserve more of Stage 1."
                    )
                }),
                "s2_sigma_schedule": (["linear", "balanced", "karras"], {
                    "default": "linear",
                    "tooltip": (
                        "Sigma schedule curve for Stage 2 refinement.\n"
                        "• linear — uniform spacing (default, original)\n"
                        "• balanced — moderate detail focus (Karras ρ=3):\n"
                        "  27% composition / 38% detail / 35% texture\n"
                        "• karras — heavy fine-detail focus (Karras ρ=7):\n"
                        "  23% composition / 35% detail / 42% texture\n"
                        "Balanced is recommended for S2."
                    )
                }),

                # ── Stage 3 ─────────────────────────────────────────────
                "upscale_to_stage3": ("FLOAT", {
                    "default": 2.0,
                    "min": 0.0,
                    "max": 8.0,
                    "step": 0.5,
                    "tooltip": (
                        "Upscale factor (area) from Stage 2 to Stage 3.\n"
                        "Default 0 = disabled (2-stage output).\n"
                        "Set to 1.5-2.0 to push to 5-7 MP final."
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
                    "tooltip": "Stage 3 true CFG scale (lower to avoid over-sharpening)"
                }),
                "s3_denoise": ("FLOAT", {
                    "default": 0.45,
                    "min": 0.1,
                    "max": 1.0,
                    "step": 0.05,
                    "tooltip": (
                        "Stage 3 denoise strength.\n"
                        "0.3-0.5 recommended — light refinement preserves\n"
                        "Stage 2 detail while adding final polish."
                    )
                }),
                "s3_sigma_schedule": (["linear", "balanced", "karras"], {
                    "default": "karras",
                    "tooltip": (
                        "Sigma schedule curve for Stage 3 final polish.\n"
                        "• linear — uniform spacing (default)\n"
                        "• balanced — moderate detail focus (Karras ρ=3)\n"
                        "• karras — heavy fine-texture focus (Karras ρ=7)\n"
                        "Karras is recommended for S3 (fine micro-texture)."
                    )
                }),

                # ── Upscale VAE (optional) ───────────────────────────────
                "upscale_vae": ("UPSCALE_VAE", {
                    "tooltip": (
                        "Optional: Wan2.1 2× upscale VAE.\n"
                        "Load with Eric Qwen Upscale VAE Loader.\n"
                        "Model: spacepxl/Wan2.1-VAE-upscale2x\n"
                        "Compatible with Qwen latent space.\n"
                        "Behaviour controlled by upscale_vae_mode."
                    )
                }),
                "upscale_vae_mode": (
                    ["disabled", "inter_stage", "final_decode", "both"], {
                    "default": "both",
                    "tooltip": (
                        "How the upscale VAE is used (requires upscale_vae).\n"
                        "• disabled — upscale VAE ignored even if connected\n"
                        "• inter_stage — decode S2 latents at 2×, re-encode\n"
                        "  to latents, feed 2× canvas to S3. Replaces the\n"
                        "  bislerp upscale between S2→S3. (Requires 3 stages)\n"
                        "• final_decode — replace the final stage's normal\n"
                        "  VAE decode with the 2× upscale decode for a free\n"
                        "  2× resolution output image.\n"
                        "• both — inter-stage S2→S3 AND 2× final decode\n"
                        "  (stacks: S3 canvas is 2× from inter-stage, then\n"
                        "   output is another 2× from final decode = 4× total)"
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
        prompt: str,
        negative_prompt: str = DEFAULT_NEGATIVE_PROMPT,
        aspect_ratio: str = "1:1   Square",
        seed: int = 0,
        seed_mode: str = "same_all_stages",
        # Pipeline params (v2)
        max_sequence_length: int = 512,
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
        offload_vae = pipeline.get("offload_vae", False)

        # ── Resolve aspect ratio ────────────────────────────────────────
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

        # ── Resolve upscale VAE mode ────────────────────────────────────
        #   "disabled" or no VAE connected → no upscale behaviour
        #   "inter_stage"  → decode→2×→re-encode between S2 and S3
        #   "final_decode" → replace final stage VAE decode with 2× decode
        #   "both"         → inter_stage + final_decode
        use_inter_stage = False
        use_final_decode = False
        if upscale_vae is not None and upscale_vae_mode != "disabled":
            if upscale_vae_mode in ("inter_stage", "both"):
                if do_stage3:
                    use_inter_stage = True
                else:
                    print("[UltraGen] WARNING: inter_stage upscale requires "
                          "3 stages (upscale_to_stage3 > 0). Falling back "
                          "to final_decode.")
                    use_final_decode = True
            if upscale_vae_mode in ("final_decode", "both"):
                use_final_decode = True

        stage_count = 3 if do_stage3 else (2 if do_stage2 else 1)
        total_steps = s1_steps + (s2_steps if do_stage2 else 0) + (s3_steps if do_stage3 else 0)

        # ── Common setup ────────────────────────────────────────────────
        device = pipe._execution_device if hasattr(pipe, "_execution_device") else "cuda"

        # ── Ensure transformer is on GPU (may have been offloaded on previous run) ──
        transformer_device = next(pipe.transformer.parameters()).device
        if str(transformer_device) != str(device):
            print(f"[UltraGen] Moving transformer back to {device} (was on {transformer_device})")
            pipe.transformer = pipe.transformer.to(device)
            torch.cuda.empty_cache()

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
            gen_s2 = None  # random
            gen_s3 = None  # random
            seed_info = f"random: S1={seed if seed>0 else 'random'}, S2=random, S3=random"
        else:  # same_all_stages
            gen_s1 = _make_generator(seed)
            gen_s2 = gen_s1
            gen_s3 = gen_s1
            seed_info = f"same: {seed if seed>0 else 'random'}"
        generator = gen_s1  # used for S1 and pipe calls

        # ── Print plan ──────────────────────────────────────────────────
        print(f"\n{'='*60}")
        print(f"[UltraGen] Multi-stage generation: {stage_count} stage(s), "
              f"{total_steps} total steps")
        print(f"[UltraGen] max_seq_len={max_sequence_length}, "
              f"seed_mode={seed_mode} ({seed_info})")
        print(f"  Stage 1: {s1_w}×{s1_h} ({s1_mp_actual:.2f} MP), "
              f"{s1_steps} steps, CFG={s1_cfg:.1f}")
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
        if negative_prompt:
            print(f"  Neg prompt: {negative_prompt[:80]}{'...' if len(negative_prompt) > 80 else ''}")
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

        # ── Build common pipeline kwargs for seq length ─────────────────
        # NOTE: guidance_scale (guidance embedding) was investigated and found
        # to be non-functional for Qwen-Image-2512. The model config has
        # guidance_embeds=False, and the QwenTimestepProjEmbeddings class
        # doesn't accept a guidance parameter. The diffusers devs left
        # TODO comments to remove it. Real guidance control is via
        # true_cfg_scale (s1_cfg/s2_cfg/s3_cfg). If adapting this node for
        # guidance-distilled models (e.g. Flux), re-add guidance_scale here.
        # See DEV_NOTES.md for details.
        extra_kwargs = {}

        # max_sequence_length for the text encoder
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

        # ── Spectrum acceleration (stage 1 only) ────────────────────────
        _spectrum_unpatch = None
        spectrum_config = getattr(pipe, "_spectrum_config", None)
        do_cfg_s1 = s1_cfg > 1 and neg is not None

        if spectrum_config is not None and s1_steps >= spectrum_config.get("min_steps", 15):
            try:
                from ..pipelines.spectrum_forward import patch_transformer_spectrum
                calls_per_step = 2 if do_cfg_s1 else 1
                _spectrum_unpatch = patch_transformer_spectrum(
                    pipe.transformer, s1_steps, spectrum_config, calls_per_step
                )
                print("[UltraGen] Spectrum acceleration enabled for Stage 1")
            except Exception as e:
                print(f"[UltraGen] Spectrum patch failed: {e}")
                _spectrum_unpatch = None
        elif spectrum_config is not None:
            print(f"[UltraGen] Spectrum auto-disabled for S1 "
                  f"(steps={s1_steps} < min_steps={spectrum_config.get('min_steps', 15)})")

        try:
            # ════════════════════════════════════════════════════════════
            #  STAGE 1 — Draft (txt2img from noise)
            # ════════════════════════════════════════════════════════════
            _check_cancelled()
            print(f"[UltraGen] -- Stage 1/{stage_count} --")

            s1_output_type = "latent" if (do_stage2 or use_final_decode) else "pil"
            s1_result = pipe(
                prompt=prompt,
                negative_prompt=neg,
                height=s1_h,
                width=s1_w,
                num_inference_steps=s1_steps,
                true_cfg_scale=s1_cfg,
                generator=generator,
                callback_on_step_end=make_cb(),
                output_type=s1_output_type,
                **extra_kwargs,
            )

            # Unpatch Spectrum after stage 1
            if _spectrum_unpatch is not None:
                try:
                    stats = _spectrum_unpatch()
                    if stats:
                        total = stats.get("actual_forwards", 0) + stats.get("cached_steps", 0)
                        actual = stats.get("actual_forwards", 0)
                        if total > 0:
                            print(f"[UltraGen] Spectrum S1 — {actual}/{total} forwards "
                                  f"({total - actual} cached, "
                                  f"{(total - actual) / total * 100:.0f}% saved)")
                except Exception as e:
                    print(f"[UltraGen] Spectrum unpatch error: {e}")
                _spectrum_unpatch = None

            if not do_stage2:
                if use_final_decode:
                    # Offload transformer to free VRAM for upscale decode
                    pipe.transformer = pipe.transformer.to("cpu")
                    torch.cuda.empty_cache()
                    print(f"[UltraGen] Upscale VAE decode (2×) ...")
                    tensor = decode_latents_with_upscale_vae(
                        s1_result.images, upscale_vae, pipe.vae,
                        s1_h, s1_w, vae_scale_factor,
                    )
                    print(f"[UltraGen] Output: {tensor.shape[2]}×{tensor.shape[1]} (2× upscaled)")
                    return (tensor,)
                pil_image = s1_result.images[0]
                print(f"[UltraGen] Output: {pil_image.size[0]}×{pil_image.size[1]}")
                tensor = pil_to_tensor(pil_image).unsqueeze(0)
                return (tensor,)

            s1_latents = s1_result.images  # packed latents
            print(f"[UltraGen]   S1 latents shape: {s1_latents.shape}")

            # ════════════════════════════════════════════════════════════
            #  STAGE 2 — Main Refinement
            # ════════════════════════════════════════════════════════════
            _check_cancelled()
            print(f"[UltraGen] -- Stage 2/{stage_count} --")

            s2_latents = _upscale_latents(
                s1_latents, s1_h, s1_w, s2_h, s2_w, vae_scale_factor
            )
            print(f"[UltraGen]   S2 latents after upscale: {s2_latents.shape}")

            raw_sigmas_s2 = build_sigma_schedule(
                s2_steps, s2_denoise, s2_sigma_schedule
            )
            actual_sigma_s2 = _compute_actual_start_sigma(
                pipe.scheduler, raw_sigmas_s2, s2_mu
            )
            print(f"[UltraGen]   S2 mu={s2_mu:.4f}, schedule={s2_sigma_schedule}, "
                  f"raw_start={raw_sigmas_s2[0]:.4f}, "
                  f"actual_start={actual_sigma_s2:.4f}")

            s2_latents = self._apply_denoise_noise(
                s2_latents, s2_denoise, actual_sigma_s2, gen_s2, device
            )

            s2_output_type = "latent" if (do_stage3 or use_final_decode) else "pil"
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
                    # Offload transformer to free VRAM for upscale decode
                    pipe.transformer = pipe.transformer.to("cpu")
                    torch.cuda.empty_cache()
                    print(f"[UltraGen] Upscale VAE decode (2×) ...")
                    tensor = decode_latents_with_upscale_vae(
                        s2_result.images, upscale_vae, pipe.vae,
                        s2_h, s2_w, vae_scale_factor,
                    )
                    print(f"[UltraGen] Output: {tensor.shape[2]}×{tensor.shape[1]} (2× upscaled)")
                    return (tensor,)
                pil_image = s2_result.images[0]
                print(f"[UltraGen] Output: {pil_image.size[0]}×{pil_image.size[1]}")
                tensor = pil_to_tensor(pil_image).unsqueeze(0)
                return (tensor,)

            s2_final_latents = s2_result.images
            print(f"[UltraGen]   S2 final latents shape: {s2_final_latents.shape}")

            # ════════════════════════════════════════════════════════════
            #  STAGE 3 — Final Polish
            # ════════════════════════════════════════════════════════════
            _check_cancelled()
            print(f"[UltraGen] -- Stage 3/{stage_count} --")

            if use_inter_stage:
                # Decode S2 latents at 2× with upscale VAE, re-encode
                # to latents as the S3 input (replaces bislerp upscale)
                print("[UltraGen]   Inter-stage VAE upscale (S2→S3) ...")
                s3_latents, s3_h, s3_w = upscale_between_stages(
                    s2_final_latents, upscale_vae, pipe.vae,
                    s2_h, s2_w, vae_scale_factor,
                )
                # Recompute mu for the new dimensions
                s3_seq = _packed_seq_len(s3_h, s3_w, vae_scale_factor)
                s3_mu = _compute_mu(s3_seq, pipe.scheduler)
            else:
                s3_latents = _upscale_latents(
                    s2_final_latents, s2_h, s2_w, s3_h, s3_w, vae_scale_factor
                )
            print(f"[UltraGen]   S3 latents after upscale: {s3_latents.shape}")

            raw_sigmas_s3 = build_sigma_schedule(
                s3_steps, s3_denoise, s3_sigma_schedule
            )
            actual_sigma_s3 = _compute_actual_start_sigma(
                pipe.scheduler, raw_sigmas_s3, s3_mu
            )
            print(f"[UltraGen]   S3 mu={s3_mu:.4f}, schedule={s3_sigma_schedule}, "
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
                # Offload transformer to free VRAM for upscale decode
                pipe.transformer = pipe.transformer.to("cpu")
                torch.cuda.empty_cache()
                print(f"[UltraGen] Upscale VAE decode (2×) ...")
                tensor = decode_latents_with_upscale_vae(
                    s3_result.images, upscale_vae, pipe.vae,
                    s3_h, s3_w, vae_scale_factor,
                )
                print(f"[UltraGen] Output: {tensor.shape[2]}×{tensor.shape[1]} (2× upscaled)")
                return (tensor,)

            pil_image = s3_result.images[0]
            print(f"[UltraGen] Output: {pil_image.size[0]}×{pil_image.size[1]}")
            tensor = pil_to_tensor(pil_image).unsqueeze(0)
            return (tensor,)

        finally:
            # Ensure Spectrum is always unpatched
            if _spectrum_unpatch is not None:
                try:
                    _spectrum_unpatch()
                except Exception:
                    pass

            # Offload VAE back to CPU
            if offload_vae and hasattr(pipe, "vae"):
                pipe.vae = pipe.vae.to("cpu")
                torch.cuda.empty_cache()

    # ─────────────────────────────────────────────────────────────────
    #  Helpers (same proven logic as Multi-Stage node)
    # ─────────────────────────────────────────────────────────────────

    @staticmethod
    def _build_sigmas(num_steps: int, denoise: float) -> list:
        """Build a linear sigma schedule (legacy helper).

        Prefer ``build_sigma_schedule()`` from multistage helpers which
        supports linear, cosine, and karras schedule curves.
        """
        full_sigmas = np.linspace(1.0, 1.0 / num_steps, num_steps)
        if denoise >= 1.0:
            return full_sigmas.tolist()
        keep = max(1, int(round(num_steps * denoise)))
        return full_sigmas[-keep:].tolist()

    @staticmethod
    def _apply_denoise_noise(latents: torch.Tensor, denoise: float,
                             actual_sigma: float,
                             generator, device: str) -> torch.Tensor:
        """Add flow-matching noise to latents for partial denoise.

        At denoise=1.0 the latents are replaced with pure noise.
        At denoise<1.0 we blend the upscaled latents with noise at
        ``actual_sigma`` — the mu-shifted + terminal-stretched starting
        sigma that the scheduler will expect on its first step.
        """
        if denoise >= 1.0:
            noise = torch.randn_like(latents)
            return noise

        noise = torch.randn(latents.shape, generator=generator,
                            device=latents.device, dtype=latents.dtype)
        return _add_noise_flowmatch(latents, noise, actual_sigma)
