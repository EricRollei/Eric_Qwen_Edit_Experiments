# Copyright (c) 2026 Eric Hiss. All rights reserved.
# Licensed under the terms in LICENSE.txt (CC BY-NC 4.0 / Commercial dual license).
# https://github.com/EricRollei/Eric_Qwen_Edit_Experiments
"""
Eric Qwen-Edit Spectrum Accelerator node.

Adds Spectrum (training-free spectral diffusion feature forecasting) support
to the Qwen-Edit pipeline.  Wire this node between the Loader and any
execution node (Image / Multi-Image / Style Transfer / Inpaint).

When enabled, the transformer's 60-block loop is skipped on predicted
(cached) steps, yielding a ~3–5× wall-clock speedup at ≥20 steps with
minimal quality loss.

Spectrum Credits:
- Jiaqi Han et al., "Adaptive Spectral Feature Forecasting" (CVPR 2026)
- https://github.com/hanjq17/Spectrum (MIT License)

Author: Eric Hiss (GitHub: EricRollei)
"""

import logging

logger = logging.getLogger(__name__)


class EricQwenEditSpectrum:
    """
    Spectrum Accelerator — training-free diffusion sampling speedup.

    Applies adaptive spectral feature forecasting to skip redundant
    transformer passes during denoising.  Reduces wall-clock time by
    ~3–5× with negligible quality loss.

    Best for:
    - ≥20 inference steps (auto-disables below ``min_steps``)
    - Full-quality 50-step baselines (not needed for 8-step lightning LoRA)
    - True CFG runs (2× transformer passes per step → double the savings)
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "pipeline": ("QWEN_EDIT_PIPELINE",),
                "enable": ("BOOLEAN", {"default": True, "tooltip": "Enable/disable Spectrum acceleration."}),
            },
            "optional": {
                "warmup_steps": ("INT", {
                    "default": 3,
                    "min": 1,
                    "max": 20,
                    "step": 1,
                    "tooltip": (
                        "Number of initial denoising steps that always run the full "
                        "transformer. More warmup = better forecaster initialization "
                        "but less speedup. 2–4 recommended."
                    ),
                }),
                "window_size": ("INT", {
                    "default": 2,
                    "min": 2,
                    "max": 10,
                    "step": 1,
                    "tooltip": (
                        "Base period between actual transformer evaluations (in steps). "
                        "window_size=2 means every other step is cached after warmup. "
                        "Higher = more aggressive caching."
                    ),
                }),
                "flex_window": ("FLOAT", {
                    "default": 0.75,
                    "min": 0.0,
                    "max": 2.0,
                    "step": 0.05,
                    "tooltip": (
                        "Window growth rate — the window_size increases by this amount "
                        "after each actual eval. Later diffusion steps change features "
                        "less, so larger windows are safe. 0 = fixed window."
                    ),
                }),
                "w": ("FLOAT", {
                    "default": 0.5,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.05,
                    "tooltip": (
                        "Blend weight between Chebyshev predictor (w=1) and Newton "
                        "forward-difference predictor (w=0). 0.5 = equal blend (recommended)."
                    ),
                }),
                "lam": ("FLOAT", {
                    "default": 0.1,
                    "min": 0.001,
                    "max": 10.0,
                    "step": 0.01,
                    "tooltip": (
                        "Ridge regularization for Chebyshev regression. Higher = smoother "
                        "predictions, lower = more responsive to recent data."
                    ),
                }),
                "M": ("INT", {
                    "default": 4,
                    "min": 1,
                    "max": 8,
                    "step": 1,
                    "tooltip": (
                        "Chebyshev polynomial degree. Higher M captures more complex "
                        "feature trajectories but risks overfitting with few data points."
                    ),
                }),
                "min_steps": ("INT", {
                    "default": 15,
                    "min": 5,
                    "max": 100,
                    "step": 1,
                    "tooltip": (
                        "Spectrum auto-disables when num_inference_steps < min_steps. "
                        "Low step counts (e.g. 8-step lightning LoRA) don't benefit."
                    ),
                }),
            },
        }

    RETURN_TYPES = ("QWEN_EDIT_PIPELINE",)
    RETURN_NAMES = ("pipeline",)
    FUNCTION = "apply_spectrum"
    CATEGORY = "Eric/QwenEdit"
    DESCRIPTION = (
        "Spectrum Accelerator — training-free ~3–5× diffusion sampling speedup. "
        "Predicts transformer outputs on skipped steps using Chebyshev polynomial "
        "regression instead of running all 60 transformer blocks. "
        "Wire between Loader and any execution node."
    )

    def apply_spectrum(
        self,
        pipeline,
        enable,
        warmup_steps=3,
        window_size=2,
        flex_window=0.75,
        w=0.5,
        lam=0.1,
        M=4,
        min_steps=15,
    ):
        pipe = pipeline["pipeline"]

        if enable:
            config = {
                "warmup_steps": warmup_steps,
                "window_size": window_size,
                "flex_window": flex_window,
                "w": w,
                "lam": lam,
                "M": M,
                "K": max(M + 2, 20),  # sliding-window buffer size
                "taylor_order": 1,
                "min_steps": min_steps,
            }
            pipe._spectrum_config = config
            logger.info(
                f"Spectrum config set: warmup={warmup_steps}, window={window_size}, "
                f"flex={flex_window}, w={w}, lam={lam}, M={M}, min_steps={min_steps}"
            )
        else:
            # Disable — remove any existing config
            if hasattr(pipe, "_spectrum_config"):
                delattr(pipe, "_spectrum_config")
            logger.info("Spectrum disabled")

        return (pipeline,)
