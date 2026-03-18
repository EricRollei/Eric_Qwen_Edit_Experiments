# Copyright (c) 2026 Eric Hiss. All rights reserved.
# Licensed under the terms in LICENSE.txt (CC BY-NC 4.0 / Commercial dual license).
# https://github.com/EricRollei/Eric_Qwen_Edit_Experiments
"""
Eric Qwen-Image Generate Node
Text-to-image generation using QwenImagePipeline (Qwen-Image / Qwen-Image-2512).

Provides resolution presets, max megapixel cap, Spectrum integration,
ComfyUI progress bar, and true CFG support.

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

# ── Resolution helpers ──────────────────────────────────────────────────

DIMENSION_ALIGNMENT = 16  # Qwen requires dimensions divisible by 16

# Aspect ratios ordered wide-landscape → square → tall-portrait
ASPECT_RATIOS = {
    "16:9  Wide Landscape":   (16, 9),
    "7:5   Landscape":        (7, 5),
    "3:2   Landscape":        (3, 2),
    "4:3   Landscape":        (4, 3),
    "5:4   Landscape":        (5, 4),
    "1:1   Square":           (1, 1),
    "4:5   Portrait":         (4, 5),
    "3:4   Portrait":         (3, 4),
    "2:3   Portrait":         (2, 3),
    "5:7   Portrait":         (5, 7),
    "9:16  Tall Portrait":    (9, 16),
}


def _align(val: int) -> int:
    """Round down to nearest multiple of DIMENSION_ALIGNMENT (min 16)."""
    return max(DIMENSION_ALIGNMENT, (val // DIMENSION_ALIGNMENT) * DIMENSION_ALIGNMENT)


def compute_dimensions_from_ratio(
    w_ratio: int, h_ratio: int, target_mp: float
) -> Tuple[int, int]:
    """Compute width and height from aspect ratio and megapixel target.

    Solves for the dimensions that match the given aspect ratio while
    hitting the target megapixel count as closely as possible, then
    aligns both axes to ``DIMENSION_ALIGNMENT`` (16 px).
    """
    target_pixels = target_mp * 1_000_000
    # w/h = w_ratio/h_ratio  =>  w = h * (w_ratio/h_ratio)
    # w * h = target_pixels   =>  h^2 * (w_ratio/h_ratio) = target_pixels
    h = math.sqrt(target_pixels * h_ratio / w_ratio)
    w = h * w_ratio / h_ratio
    return _align(int(round(w))), _align(int(round(h)))


# ═══════════════════════════════════════════════════════════════════════
#  Generation Node
# ═══════════════════════════════════════════════════════════════════════

class EricQwenImageGenerate:
    """
    Generate images from text using Qwen-Image / Qwen-Image-2512.

    Pick an aspect ratio and a megapixel target — the node computes
    the exact pixel dimensions (aligned to 16 px as Qwen requires).
    A ComfyUI progress bar tracks the denoising steps.  Spectrum
    acceleration is supported when a _spectrum_config is attached to
    the pipeline.
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
                    "tooltip": "Aspect ratio of the generated image"
                }),
                "target_mp": ("FLOAT", {
                    "default": 4.0,
                    "min": 0.25,
                    "max": 16.0,
                    "step": 0.25,
                    "tooltip": "Target megapixels — the node computes exact width/height from ratio + MP (dimensions aligned to 16 px)"
                }),
                "steps": ("INT", {
                    "default": 50,
                    "min": 1,
                    "max": 200,
                    "step": 1,
                    "tooltip": "Inference steps"
                }),
                "true_cfg_scale": ("FLOAT", {
                    "default": 4.0,
                    "min": 1.0,
                    "max": 20.0,
                    "step": 0.5,
                    "tooltip": "True CFG scale — runs two transformer passes per step (>1 to enable)"
                }),
                "seed": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 0xffffffffffffffff,
                    "tooltip": "Random seed (0 = random)"
                }),
            }
        }

    def generate(
        self,
        pipeline: dict,
        prompt: str,
        negative_prompt: str = "",
        aspect_ratio: str = "1:1   Square",
        target_mp: float = 4.0,
        steps: int = 50,
        true_cfg_scale: float = 4.0,
        seed: int = 0,
    ) -> Tuple[torch.Tensor]:
        pipe = pipeline["pipeline"]
        offload_vae = pipeline.get("offload_vae", False)

        # ── Compute dimensions from ratio + megapixel target ────────────
        w_ratio, h_ratio = ASPECT_RATIOS.get(aspect_ratio, (1, 1))
        width, height = compute_dimensions_from_ratio(w_ratio, h_ratio, target_mp)

        print(f"[EricQwenImage] Generating {width}×{height} ({width * height / 1e6:.2f} MP), "
              f"ratio={aspect_ratio}, steps={steps}, cfg={true_cfg_scale}")

        # ── Generator ───────────────────────────────────────────────────
        device = pipe._execution_device if hasattr(pipe, "_execution_device") else "cuda"
        generator = torch.Generator(device=device).manual_seed(seed) if seed > 0 else None

        # ── Negative prompt ─────────────────────────────────────────────
        neg = negative_prompt.strip() if negative_prompt else None
        if neg == "":
            neg = None

        # ── ComfyUI progress bar + cancel support ───────────────────────
        import comfy.utils
        import comfy.model_management
        pbar = comfy.utils.ProgressBar(steps)

        def on_step_end(_pipe, step_idx, _timestep, cb_kwargs):
            pbar.update(1)
            comfy.model_management.throw_exception_if_processing_interrupted()
            return cb_kwargs

        # Check for cancel before starting
        comfy.model_management.throw_exception_if_processing_interrupted()

        # ── Spectrum acceleration (if configured) ───────────────────────
        _spectrum_unpatch = None
        spectrum_config = getattr(pipe, "_spectrum_config", None)
        do_cfg = true_cfg_scale > 1 and neg is not None
        if spectrum_config is not None and steps >= spectrum_config.get("min_steps", 15):
            try:
                from ..pipelines.spectrum_forward import patch_transformer_spectrum
                calls_per_step = 2 if do_cfg else 1
                _spectrum_unpatch = patch_transformer_spectrum(
                    pipe.transformer, steps, spectrum_config, calls_per_step
                )
                print("[EricQwenImage] Spectrum acceleration enabled")
            except Exception as e:
                print(f"[EricQwenImage] Spectrum patch failed, full fidelity: {e}")
                _spectrum_unpatch = None
        elif spectrum_config is not None:
            print(f"[EricQwenImage] Spectrum auto-disabled (steps={steps} < min_steps={spectrum_config.get('min_steps', 15)})")

        # ── Move VAE back to GPU if offloaded ───────────────────────────
        if offload_vae and hasattr(pipe, "vae"):
            vae_device = next(pipe.transformer.parameters()).device
            pipe.vae = pipe.vae.to(vae_device)

        # ── Generate ────────────────────────────────────────────────────
        try:
            result = pipe(
                prompt=prompt,
                negative_prompt=neg,
                height=height,
                width=width,
                num_inference_steps=steps,
                true_cfg_scale=true_cfg_scale,
                generator=generator,
                callback_on_step_end=on_step_end,
            )
        finally:
            # Unpatch Spectrum
            if _spectrum_unpatch is not None:
                try:
                    stats = _spectrum_unpatch()
                    if stats:
                        total = stats.get("actual_forwards", 0) + stats.get("cached_steps", 0)
                        actual = stats.get("actual_forwards", 0)
                        if total > 0:
                            print(f"[EricQwenImage] Spectrum — {actual}/{total} actual forwards "
                                  f"({total - actual} cached, {(total - actual) / total * 100:.0f}% saved)")
                except Exception as e:
                    print(f"[EricQwenImage] Spectrum unpatch error: {e}")

            # Offload VAE back to CPU
            if offload_vae and hasattr(pipe, "vae"):
                pipe.vae = pipe.vae.to("cpu")
                torch.cuda.empty_cache()

        # ── Convert to ComfyUI tensor ───────────────────────────────────
        pil_image = result.images[0]
        tensor = pil_to_tensor(pil_image).unsqueeze(0)

        return (tensor,)
