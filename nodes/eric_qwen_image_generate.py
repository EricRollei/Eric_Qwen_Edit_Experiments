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

DIMENSION_ALIGNMENT = 32  # Must be divisible by 32 for VAE packing

# Pre-built presets: (label, width, height)
RESOLUTION_PRESETS = {
    "1024×1024 (1:1)":    (1024, 1024),
    "1152×896 (9:7)":     (1152, 896),
    "896×1152 (7:9)":     (896, 1152),
    "1216×832 (19:13)":   (1216, 832),
    "832×1216 (13:19)":   (832, 1216),
    "1344×768 (7:4)":     (1344, 768),
    "768×1344 (4:7)":     (768, 1344),
    "1536×640 (12:5)":    (1536, 640),
    "640×1536 (5:12)":    (640, 1536),
    "custom":             (0, 0),
}


def _align(val: int) -> int:
    """Align to DIMENSION_ALIGNMENT."""
    return max(DIMENSION_ALIGNMENT, (val // DIMENSION_ALIGNMENT) * DIMENSION_ALIGNMENT)


def compute_dimensions(
    width: int, height: int, max_mp: float
) -> Tuple[int, int]:
    """Scale dimensions down if they exceed max_mp, preserving aspect ratio."""
    max_pixels = int(max_mp * 1024 * 1024)
    current_pixels = width * height
    if current_pixels > max_pixels:
        scale = math.sqrt(max_pixels / current_pixels)
        width = int(width * scale)
        height = int(height * scale)
    return _align(width), _align(height)


# ═══════════════════════════════════════════════════════════════════════
#  Generation Node
# ═══════════════════════════════════════════════════════════════════════

class EricQwenImageGenerate:
    """
    Generate images from text using Qwen-Image / Qwen-Image-2512.

    Choose a resolution preset or set custom width/height.  Resolution is
    capped by max_mp and aligned to 32 px.  A ComfyUI progress bar tracks
    the denoising steps.  Spectrum acceleration is supported when a
    _spectrum_config is attached to the pipeline.
    """

    CATEGORY = "Eric Qwen-Image"
    FUNCTION = "generate"
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)

    @classmethod
    def INPUT_TYPES(cls):
        preset_names = list(RESOLUTION_PRESETS.keys())
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
                "resolution": (preset_names, {
                    "default": "1024×1024 (1:1)",
                    "tooltip": "Resolution preset. Choose 'custom' to use width/height below."
                }),
                "width": ("INT", {
                    "default": 1024,
                    "min": 256,
                    "max": 4096,
                    "step": 32,
                    "tooltip": "Custom width (only used when resolution = 'custom')"
                }),
                "height": ("INT", {
                    "default": 1024,
                    "min": 256,
                    "max": 4096,
                    "step": 32,
                    "tooltip": "Custom height (only used when resolution = 'custom')"
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
                "max_mp": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.25,
                    "max": 16.0,
                    "step": 0.25,
                    "tooltip": "Maximum output megapixels (for VRAM safety)"
                }),
            }
        }

    def generate(
        self,
        pipeline: dict,
        prompt: str,
        negative_prompt: str = "",
        resolution: str = "1024×1024 (1:1)",
        width: int = 1024,
        height: int = 1024,
        steps: int = 50,
        true_cfg_scale: float = 4.0,
        seed: int = 0,
        max_mp: float = 1.0,
    ) -> Tuple[torch.Tensor]:
        pipe = pipeline["pipeline"]
        offload_vae = pipeline.get("offload_vae", False)

        # ── Resolve resolution ──────────────────────────────────────────
        if resolution != "custom":
            preset = RESOLUTION_PRESETS.get(resolution, (1024, 1024))
            width, height = preset
        width, height = compute_dimensions(width, height, max_mp)

        print(f"[EricQwenImage] Generating {width}×{height} ({width * height / 1e6:.2f} MP), "
              f"steps={steps}, cfg={true_cfg_scale}")

        # ── Generator ───────────────────────────────────────────────────
        device = pipe._execution_device if hasattr(pipe, "_execution_device") else "cuda"
        generator = torch.Generator(device=device).manual_seed(seed) if seed > 0 else None

        # ── Negative prompt ─────────────────────────────────────────────
        neg = negative_prompt.strip() if negative_prompt else None
        if neg == "":
            neg = None

        # ── ComfyUI progress bar ────────────────────────────────────────
        import comfy.utils
        pbar = comfy.utils.ProgressBar(steps)

        def on_step_end(_pipe, step_idx, _timestep, cb_kwargs):
            pbar.update(1)
            return cb_kwargs

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
