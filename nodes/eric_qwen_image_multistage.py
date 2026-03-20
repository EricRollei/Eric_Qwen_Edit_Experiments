# Copyright (c) 2026 Eric Hiss. All rights reserved.
# Licensed under the terms in LICENSE.txt (CC BY-NC 4.0 / Commercial dual license).
# https://github.com/EricRollei/Eric_Qwen_Edit_Experiments
"""
Eric Qwen-Image Multi-Stage Generate Node
Progressive multi-stage text-to-image generation with full per-stage control.

Up to 3 stages with independent steps, CFG, and upscale ratios.
Set upscale_to_stage2 = 0 to output Stage 1 only (single-stage).
Set upscale_to_stage3 = 0 to stop after Stage 2 (two-stage).

Latents are upscaled between stages via bislerp (slerp-based
interpolation that preserves vector norms and directions in latent
space) and re-noised according to per-stage denoise strength before
re-sampling.

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


# ═══════════════════════════════════════════════════════════════════════
#  Latent helpers
# ═══════════════════════════════════════════════════════════════════════

def _unpack_latents(latents: torch.Tensor, height: int, width: int,
                    vae_scale_factor: int) -> torch.Tensor:
    """Unpack flow-packed latents back to spatial (B, C, 1, H_lat, W_lat)."""
    batch_size, num_patches, channels = latents.shape
    h_lat = 2 * (int(height) // (vae_scale_factor * 2))
    w_lat = 2 * (int(width) // (vae_scale_factor * 2))
    latents = latents.view(batch_size, h_lat // 2, w_lat // 2,
                           channels // 4, 2, 2)
    latents = latents.permute(0, 3, 1, 4, 2, 5)
    latents = latents.reshape(batch_size, channels // 4, 1, h_lat, w_lat)
    return latents


def _pack_latents(latents: torch.Tensor) -> torch.Tensor:
    """Pack spatial latents (B, C, 1, H, W) back to flow format (B, seq, C*4)."""
    batch_size, c, _one, h, w = latents.shape
    latents = latents.squeeze(2)                        # (B, C, H, W)
    latents = latents.view(batch_size, c, h // 2, 2, w // 2, 2)
    latents = latents.permute(0, 2, 4, 1, 3, 5)        # (B, H/2, W/2, C, 2, 2)
    latents = latents.reshape(batch_size, (h // 2) * (w // 2), c * 4)
    return latents


def _upscale_latents(latents_packed: torch.Tensor,
                     src_h: int, src_w: int,
                     dst_h: int, dst_w: int,
                     vae_scale_factor: int) -> torch.Tensor:
    """Unpack -> bislerp upscale -> repack latents.

    Uses ComfyUI's bislerp (slerp-interpolation) instead of bicubic.
    Bislerp preserves vector norms and angular relationships between
    latent channels, producing sharper and more coherent upscaled
    latents compared to bicubic's independent-channel averaging.
    """
    import comfy.utils as comfy_utils
    spatial = _unpack_latents(latents_packed, src_h, src_w, vae_scale_factor)
    b, c, _one, h_lat, w_lat = spatial.shape
    dst_h_lat = 2 * (int(dst_h) // (vae_scale_factor * 2))
    dst_w_lat = 2 * (int(dst_w) // (vae_scale_factor * 2))
    spatial_4d = spatial.squeeze(2)  # (B, C, H, W)
    upscaled = comfy_utils.bislerp(spatial_4d, dst_w_lat, dst_h_lat)
    upscaled = upscaled.unsqueeze(2)  # (B, C, 1, H', W')
    return _pack_latents(upscaled)


def _add_noise_flowmatch(latents: torch.Tensor, noise: torch.Tensor,
                         sigma: float) -> torch.Tensor:
    """Apply flow-matching noising: x_noisy = (1 - sigma) * x + sigma * noise."""
    return (1.0 - sigma) * latents + sigma * noise


def _check_cancelled():
    """Raise InterruptProcessingException if user hit Cancel in ComfyUI."""
    import comfy.model_management
    comfy.model_management.throw_exception_if_processing_interrupted()


# ═══════════════════════════════════════════════════════════════════════
#  Scheduler-aware sigma helpers
# ═══════════════════════════════════════════════════════════════════════

def _packed_seq_len(height: int, width: int, vae_scale_factor: int) -> int:
    """Compute the packed latent sequence length for given pixel dimensions."""
    h_lat = 2 * (int(height) // (vae_scale_factor * 2))
    w_lat = 2 * (int(width) // (vae_scale_factor * 2))
    return (h_lat // 2) * (w_lat // 2)


def _compute_mu(seq_len: int, scheduler) -> float:
    """Compute the time-shift mu for a given sequence length.

    Mirrors the ``calculate_shift`` call inside QwenImagePipeline.__call__.
    """
    base_seq = scheduler.config.get("base_image_seq_len", 256)
    max_seq = scheduler.config.get("max_image_seq_len", 4096)
    base_shift = scheduler.config.get("base_shift", 0.5)
    max_shift = scheduler.config.get("max_shift", 1.15)
    m = (max_shift - base_shift) / (max_seq - base_seq)
    b = base_shift - m * base_seq
    return m * seq_len + b


def _compute_actual_start_sigma(
    scheduler, raw_sigmas: list, mu: float
) -> float:
    """Replicate the scheduler's sigma transforms to find the actual start sigma.

    Steps applied by ``FlowMatchEulerDiscreteScheduler.set_timesteps``:
    1. Dynamic time shift (exponential or linear) based on *mu*.
    2. Terminal stretch — rescale so the last sigma equals ``shift_terminal``.

    We replicate these on the raw sigmas so we can noise the latent at
    exactly the level the scheduler will expect on its first step.
    """
    sigmas = np.array(raw_sigmas, dtype=np.float64)

    # 1. Dynamic time shift
    if scheduler.config.get("use_dynamic_shifting", False):
        shift_type = scheduler.config.get("time_shift_type", "exponential")
        if shift_type == "exponential":
            exp_mu = math.exp(mu)
            sigmas = exp_mu / (exp_mu + (1.0 / sigmas - 1.0))
        else:  # "linear"
            sigmas = mu / (mu + (1.0 / sigmas - 1.0))
    else:
        static_shift = scheduler.config.get("shift", 1.0)
        sigmas = static_shift * sigmas / (1.0 + (static_shift - 1.0) * sigmas)

    # 2. Terminal stretch
    shift_terminal = scheduler.config.get("shift_terminal", None)
    if shift_terminal:
        one_minus_z = 1.0 - sigmas
        scale_factor = one_minus_z[-1] / (1.0 - shift_terminal)
        sigmas = 1.0 - (one_minus_z / scale_factor)

    return float(sigmas[0])


def build_sigma_schedule(num_steps: int, denoise: float,
                         schedule: str = "linear",
                         power: float = 1.0) -> list:
    """Build a sigma schedule for flow-matching partial denoise.

    The schedule defines the **spacing** of sigma values within the active
    denoise range.  All schedules start from the same sigma (determined by
    ``denoise``) and end at the same ``sigma_min`` — they only differ in
    how steps are distributed within that range.

    Schedules
    ---------
    linear
        Uniform spacing.  Safe default.  Equal compute at every noise level.
    balanced
        Karras-style with ρ=3.  Moderately concentrates steps at mid-to-low
        sigma — reduces time spent on composition (high sigma) while giving
        balanced coverage to detail and fine texture.  **Recommended for S2.**
    karras
        Karras-style with ρ=7 (EDM-optimal).  Heavily concentrates steps at
        low sigma (fine detail/sharpening), with large jumps through high
        sigma.  **Recommended for S3.**

    Step budget distribution (S2 example: 30 steps, denoise=0.85):

        Schedule  │ HIGH σ (composition)  │ MID σ (detail)  │ LOW σ (texture)
        ──────────┼───────────────────────┼─────────────────┼────────────────
        linear    │ 46%                   │ 38%             │ 15%
        balanced  │ 27%                   │ 38%             │ 35%
        karras    │ 23%                   │ 35%             │ 42%

    Denoise
    -------
    When ``denoise < 1.0``, the schedule covers only the lower portion of
    the sigma range (from ``sigma_start`` down to ``sigma_min``), with
    ``keep = round(num_steps * denoise)`` steps.  ``sigma_start`` is the
    same value linear truncation would produce, ensuring all schedules have
    a consistent noise level regardless of curve shape.

    Args:
        num_steps: Total number of denoising steps.
        denoise:   Fraction of schedule to use (1.0 = full, <1 = partial).
        schedule:  ``"linear"``, ``"balanced"``, or ``"karras"``.
        power:     Reserved for future use.

    Returns:
        List of sigma values (descending), length = ``keep``.
    """
    sigma_max = 1.0
    sigma_min = 1.0 / num_steps

    if denoise >= 1.0:
        keep = num_steps
        sigma_start = sigma_max
    else:
        keep = max(1, int(round(num_steps * denoise)))
        # Compute starting sigma from the linear schedule so ALL schedule
        # types begin at the same noise level for a given denoise value.
        # This was the root cause of the "echo/ghosting" bug: cosine and
        # karras had wildly different starting sigmas after truncation.
        full_linear = np.linspace(sigma_max, sigma_min, num_steps)
        sigma_start = float(full_linear[num_steps - keep])

    # Distribute 'keep' steps from sigma_start to sigma_min
    t = np.linspace(0.0, 1.0, keep)

    if schedule == "balanced":
        # Karras-style with ρ=3 — moderate concentration at low sigma.
        # Reduces high-sigma (composition) steps while giving balanced
        # coverage to mid (detail) + low (fine texture) ranges.
        rho = 3.0
        sigmas = (
            sigma_start ** (1.0 / rho)
            + t * (sigma_min ** (1.0 / rho) - sigma_start ** (1.0 / rho))
        ) ** rho
    elif schedule == "karras":
        # EDM-optimal (Karras et al., NeurIPS 2022) with ρ=7.
        # Heavy concentration at low sigma — best for fine detail stages.
        rho = 7.0
        sigmas = (
            sigma_start ** (1.0 / rho)
            + t * (sigma_min ** (1.0 / rho) - sigma_start ** (1.0 / rho))
        ) ** rho
    else:  # "linear" (default)
        sigmas = np.linspace(sigma_start, sigma_min, keep)

    sigmas = np.clip(sigmas, sigma_min, sigma_start)
    return sigmas.tolist()


# ═══════════════════════════════════════════════════════════════════════
#  Multi-Stage Generation Node
# ═══════════════════════════════════════════════════════════════════════

class EricQwenImageMultiStage:
    """
    Progressive multi-stage text-to-image generation with full per-stage
    control over steps, CFG, initial MP, upscale ratios, and denoise.

    - Set upscale_to_stage2 = 0 to output Stage 1 only (single-stage).
    - Set upscale_to_stage3 = 0 to stop after Stage 2 (two-stage).
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
                    "tooltip": "Describe the image you want to generate"
                }),
            },
            "optional": {
                "negative_prompt": ("STRING", {
                    "multiline": True,
                    "default": "",
                    "tooltip": "What to avoid in the output"
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
                # ── Stage 1 ─────────────────────────────────────────────
                "s1_mp": ("FLOAT", {
                    "default": 0.5,
                    "min": 0.3,
                    "max": 2.0,
                    "step": 0.1,
                    "tooltip": "Stage 1 initial resolution in megapixels"
                }),
                "s1_steps": ("INT", {
                    "default": 15,
                    "min": 1,
                    "max": 200,
                    "step": 1,
                    "tooltip": "Stage 1 inference steps (txt2img from noise)"
                }),
                "s1_cfg": ("FLOAT", {
                    "default": 8.0,
                    "min": 1.0,
                    "max": 20.0,
                    "step": 0.5,
                    "tooltip": "Stage 1 true CFG scale"
                }),
                # ── Stage 2 ─────────────────────────────────────────────
                "upscale_to_stage2": ("FLOAT", {
                    "default": 2.0,
                    "min": 0.0,
                    "max": 8.0,
                    "step": 0.5,
                    "tooltip": "Upscale factor (area) from Stage 1 to Stage 2 (0 = skip Stage 2 & 3, output Stage 1)"
                }),
                "s2_steps": ("INT", {
                    "default": 20,
                    "min": 1,
                    "max": 200,
                    "step": 1,
                    "tooltip": "Stage 2 inference steps"
                }),
                "s2_cfg": ("FLOAT", {
                    "default": 4.0,
                    "min": 1.0,
                    "max": 20.0,
                    "step": 0.5,
                    "tooltip": "Stage 2 true CFG scale"
                }),
                "s2_denoise": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.1,
                    "max": 1.0,
                    "step": 0.05,
                    "tooltip": "Stage 2 denoise (1.0 = full re-denoise, lower = preserve prior stage detail)"
                }),
                # ── Stage 3 ─────────────────────────────────────────────
                "upscale_to_stage3": ("FLOAT", {
                    "default": 2.0,
                    "min": 0.0,
                    "max": 8.0,
                    "step": 0.5,
                    "tooltip": "Upscale factor (area) from Stage 2 to Stage 3 (0 = skip Stage 3, output Stage 2)"
                }),
                "s3_steps": ("INT", {
                    "default": 15,
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
                    "default": 1.0,
                    "min": 0.1,
                    "max": 1.0,
                    "step": 0.05,
                    "tooltip": "Stage 3 denoise (1.0 = full re-denoise, lower = preserve prior stage detail)"
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
        negative_prompt: str = "",
        aspect_ratio: str = "1:1   Square",
        seed: int = 0,
        # Stage 1
        s1_mp: float = 0.5,
        s1_steps: int = 15,
        s1_cfg: float = 8.0,
        # Stage 2
        upscale_to_stage2: float = 2.0,
        s2_steps: int = 20,
        s2_cfg: float = 4.0,
        s2_denoise: float = 1.0,
        # Stage 3
        upscale_to_stage3: float = 2.0,
        s3_steps: int = 15,
        s3_cfg: float = 2.0,
        s3_denoise: float = 1.0,
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

        stage_count = 3 if do_stage3 else (2 if do_stage2 else 1)
        total_steps = s1_steps + (s2_steps if do_stage2 else 0) + (s3_steps if do_stage3 else 0)

        # ── Print plan ──────────────────────────────────────────────────
        print(f"[EricQwenImage-MS] Multi-stage generation: {stage_count} stage(s), "
              f"{total_steps} total steps")
        print(f"  Stage 1: {s1_w}x{s1_h} ({s1_mp_actual:.2f} MP), "
              f"{s1_steps} steps, CFG={s1_cfg:.1f}")
        if do_stage2:
            print(f"  Stage 2: {s2_w}x{s2_h} ({s2_mp_actual:.2f} MP), "
                  f"{s2_steps} steps, CFG={s2_cfg:.1f}, "
                  f"upscale={upscale_to_stage2:.1f}x area, denoise={s2_denoise:.2f}")
        if do_stage3:
            print(f"  Stage 3: {s3_w}x{s3_h} ({s3_mp_actual:.2f} MP), "
                  f"{s3_steps} steps, CFG={s3_cfg:.1f}, "
                  f"upscale={upscale_to_stage3:.1f}x area, denoise={s3_denoise:.2f}")

        # ── Common setup ────────────────────────────────────────────────
        device = pipe._execution_device if hasattr(pipe, "_execution_device") else "cuda"
        generator = torch.Generator(device=device).manual_seed(seed) if seed > 0 else None

        neg = negative_prompt.strip() if negative_prompt else None
        if neg == "":
            neg = None

        import comfy.utils
        pbar = comfy.utils.ProgressBar(total_steps)

        def make_cb():
            def on_step_end(_pipe, step_idx, _timestep, cb_kwargs):
                pbar.update(1)
                _check_cancelled()  # allow mid-step cancellation
                return cb_kwargs
            return on_step_end

        vae_scale_factor = pipe.vae_scale_factor if hasattr(pipe, "vae_scale_factor") else 8

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

        try:
            # ════════════════════════════════════════════════════════════
            #  STAGE 1 — Draft (txt2img from noise)
            # ════════════════════════════════════════════════════════════
            _check_cancelled()
            print(f"[EricQwenImage-MS] -- Stage 1/{stage_count} --")

            s1_output_type = "latent" if do_stage2 else "pil"
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
            )

            if not do_stage2:
                pil_image = s1_result.images[0]
                print(f"[EricQwenImage-MS] Output: {pil_image.size[0]}x{pil_image.size[1]}")
                tensor = pil_to_tensor(pil_image).unsqueeze(0)
                return (tensor,)

            s1_latents = s1_result.images  # packed latents
            print(f"[EricQwenImage-MS]   S1 latents shape: {s1_latents.shape}")

            # ════════════════════════════════════════════════════════════
            #  STAGE 2 — Refine
            # ════════════════════════════════════════════════════════════
            _check_cancelled()
            print(f"[EricQwenImage-MS] -- Stage 2/{stage_count} --")

            s2_latents = _upscale_latents(
                s1_latents, s1_h, s1_w, s2_h, s2_w, vae_scale_factor
            )
            print(f"[EricQwenImage-MS]   S2 latents after upscale: {s2_latents.shape}")

            raw_sigmas_s2 = self._build_sigmas(s2_steps, s2_denoise)
            actual_sigma_s2 = _compute_actual_start_sigma(
                pipe.scheduler, raw_sigmas_s2, s2_mu
            )
            print(f"[EricQwenImage-MS]   S2 mu={s2_mu:.4f}, "
                  f"raw_start={raw_sigmas_s2[0]:.4f}, "
                  f"actual_start={actual_sigma_s2:.4f}")

            s2_latents = self._apply_denoise_noise(
                s2_latents, s2_denoise, actual_sigma_s2, generator, device
            )

            s2_output_type = "latent" if do_stage3 else "pil"
            s2_result = pipe(
                prompt=prompt,
                negative_prompt=neg,
                height=s2_h,
                width=s2_w,
                num_inference_steps=s2_steps,
                sigmas=raw_sigmas_s2,
                true_cfg_scale=s2_cfg,
                generator=generator,
                latents=s2_latents,
                callback_on_step_end=make_cb(),
                output_type=s2_output_type,
            )

            if not do_stage3:
                pil_image = s2_result.images[0]
                print(f"[EricQwenImage-MS] Output: {pil_image.size[0]}x{pil_image.size[1]}")
                tensor = pil_to_tensor(pil_image).unsqueeze(0)
                return (tensor,)

            s2_final_latents = s2_result.images
            print(f"[EricQwenImage-MS]   S2 final latents shape: {s2_final_latents.shape}")

            # ════════════════════════════════════════════════════════════
            #  STAGE 3 — Final
            # ════════════════════════════════════════════════════════════
            _check_cancelled()
            print(f"[EricQwenImage-MS] -- Stage 3/{stage_count} --")

            s3_latents = _upscale_latents(
                s2_final_latents, s2_h, s2_w, s3_h, s3_w, vae_scale_factor
            )
            print(f"[EricQwenImage-MS]   S3 latents after upscale: {s3_latents.shape}")

            raw_sigmas_s3 = self._build_sigmas(s3_steps, s3_denoise)
            actual_sigma_s3 = _compute_actual_start_sigma(
                pipe.scheduler, raw_sigmas_s3, s3_mu
            )
            print(f"[EricQwenImage-MS]   S3 mu={s3_mu:.4f}, "
                  f"raw_start={raw_sigmas_s3[0]:.4f}, "
                  f"actual_start={actual_sigma_s3:.4f}")

            s3_latents = self._apply_denoise_noise(
                s3_latents, s3_denoise, actual_sigma_s3, generator, device
            )

            s3_result = pipe(
                prompt=prompt,
                negative_prompt=neg,
                height=s3_h,
                width=s3_w,
                num_inference_steps=s3_steps,
                sigmas=raw_sigmas_s3,
                true_cfg_scale=s3_cfg,
                generator=generator,
                latents=s3_latents,
                callback_on_step_end=make_cb(),
                output_type="pil",
            )

            pil_image = s3_result.images[0]
            print(f"[EricQwenImage-MS] Output: {pil_image.size[0]}x{pil_image.size[1]}")
            tensor = pil_to_tensor(pil_image).unsqueeze(0)
            return (tensor,)

        finally:
            # Offload VAE back to CPU
            if offload_vae and hasattr(pipe, "vae"):
                pipe.vae = pipe.vae.to("cpu")
                torch.cuda.empty_cache()

    # ─────────────────────────────────────────────────────────────────
    #  Helpers
    # ─────────────────────────────────────────────────────────────────

    @staticmethod
    def _build_sigmas(num_steps: int, denoise: float) -> list:
        """Build a sigma schedule for flow-matching partial denoise.

        Full schedule goes from 1.0 -> ~0.  When denoise < 1.0 we start
        from a lower sigma, effectively skipping the noisiest portion.
        """
        full_sigmas = np.linspace(1.0, 1.0 / num_steps, num_steps)
        if denoise >= 1.0:
            return full_sigmas.tolist()
        # Trim from the front -- keep the last (denoise * N) steps
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
