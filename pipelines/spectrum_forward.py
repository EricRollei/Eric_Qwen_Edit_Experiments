# Copyright (c) 2026 Eric Hiss. All rights reserved.
# Licensed under the terms in LICENSE.txt (CC BY-NC 4.0 / Commercial dual license).
# https://github.com/EricRollei/Eric_Qwen_Edit_Experiments
#
# Spectrum forward pass adapted from https://github.com/hanjq17/Spectrum (MIT License).
# Original authors: Jiaqi Han et al. (CVPR 2026)
"""
Spectrum-accelerated forward pass for QwenImageTransformer2DModel.

This module provides:
- ``patch_transformer_spectrum()``: Monkey-patches the transformer's forward
  method to use Spectrum feature forecasting, returning an unpatch callable.
- ``_qwen_spectrum_forward()``: The replacement forward that skips the heavy
  60-block loop on predicted (cached) steps while always running the cheap
  embedding + ``norm_out`` + ``proj_out`` layers.

Design notes
-------------
* Dual-forecaster support for True CFG (two transformer calls per denoising
  step — conditional and unconditional — use separate forecasters but share
  the same actual-vs-cached decision per step).
* The schedule is flexible-window: after a warmup phase of ``warmup_steps``
  actual forwards, subsequent actual forwards happen with an increasing
  period controlled by ``window_size`` and ``flex_window``.
* All state is stored as ``_spectrum_*`` attributes on the **transformer
  instance**, keeping the class definition untouched.
"""

from __future__ import annotations

import math
import logging
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import torch

from diffusers.models.modeling_outputs import Transformer2DModelOutput

try:
    from diffusers.utils import USE_PEFT_BACKEND
    from diffusers.utils.peft_utils import scale_lora_layers, unscale_lora_layers
except ImportError:
    USE_PEFT_BACKEND = False
    scale_lora_layers = None
    unscale_lora_layers = None

from .spectrum_utils import SpectrumForecaster

logger = logging.getLogger(__name__)


# ======================================================================
#  Scheduling helpers
# ======================================================================

def _should_compute(transformer) -> bool:
    """
    Decide whether the current denoising step should run the full
    transformer forward (``True``) or use the cached/predicted features
    (``False``).

    This is called **once per denoising step** (not per sub-call).
    """
    step = transformer._spectrum_step_idx
    warmup = transformer._spectrum_warmup_steps

    if step < warmup:
        return True

    consecutive = transformer._spectrum_num_consecutive_cached + 1
    curr_ws = transformer._spectrum_curr_ws

    if consecutive % max(1, math.floor(curr_ws)) == 0:
        # Time for an actual forward — increase the window for next time
        transformer._spectrum_curr_ws += transformer._spectrum_flex_window
        transformer._spectrum_curr_ws = round(transformer._spectrum_curr_ws, 3)
        return True

    return False


# ======================================================================
#  The monkey-patched forward
# ======================================================================

def _qwen_spectrum_forward(
    self,
    hidden_states: torch.Tensor,
    encoder_hidden_states: torch.Tensor = None,
    encoder_hidden_states_mask: torch.Tensor = None,
    timestep: torch.LongTensor = None,
    img_shapes: Optional[List[Tuple[int, int, int]]] = None,
    txt_seq_lens: Optional[List[int]] = None,
    guidance: torch.Tensor = None,
    attention_kwargs: Optional[Dict[str, Any]] = None,
    return_dict: bool = True,
) -> Union[torch.Tensor, Transformer2DModelOutput]:
    """
    Spectrum-accelerated forward for ``QwenImageTransformer2DModel``.

    On **actual** steps the full 60-block loop runs and the forecaster is
    updated.  On **cached** steps the block loop is skipped and the
    forecaster predicts the post-block hidden states directly.

    The cheap preamble (``img_in``, ``time_text_embed``) and the cheap
    postamble (``norm_out``, ``proj_out``) always run because they depend
    on the current timestep / latent input and are negligible cost
    compared to the 60-block loop.
    """
    # ---- LoRA bookkeeping (same as original) ----
    if attention_kwargs is not None:
        attention_kwargs = attention_kwargs.copy()
        lora_scale = attention_kwargs.pop("scale", 1.0)
    else:
        lora_scale = 1.0

    if USE_PEFT_BACKEND and scale_lora_layers is not None:
        scale_lora_layers(self, lora_scale)

    # ---- Cheap preamble (always runs) ----
    hidden_states = self.img_in(hidden_states)
    timestep = timestep.to(hidden_states.dtype)
    encoder_hidden_states = self.txt_norm(encoder_hidden_states)
    encoder_hidden_states = self.txt_in(encoder_hidden_states)

    if guidance is not None:
        guidance = guidance.to(hidden_states.dtype) * 1000

    temb = (
        self.time_text_embed(timestep, hidden_states)
        if guidance is None
        else self.time_text_embed(timestep, guidance, hidden_states)
    )

    # ---- Spectrum step/call tracking ----
    sub_call = self._spectrum_sub_call

    # On the first sub-call of each denoising step, make the actual/cached decision
    if sub_call == 0:
        self._spectrum_step_is_actual = _should_compute(self)

    actual_forward: bool = self._spectrum_step_is_actual
    forecaster: SpectrumForecaster = self._spectrum_forecasters[sub_call]

    # ---- Block loop (actual) or prediction (cached) ----
    if actual_forward:
        image_rotary_emb = self.pos_embed(
            img_shapes, txt_seq_lens, device=hidden_states.device
        )
        for index_block, block in enumerate(self.transformer_blocks):
            if torch.is_grad_enabled() and self.gradient_checkpointing:
                encoder_hidden_states, hidden_states = self._gradient_checkpointing_func(
                    block,
                    hidden_states,
                    encoder_hidden_states,
                    encoder_hidden_states_mask,
                    temb,
                    image_rotary_emb,
                )
            else:
                encoder_hidden_states, hidden_states = block(
                    hidden_states=hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    encoder_hidden_states_mask=encoder_hidden_states_mask,
                    temb=temb,
                    image_rotary_emb=image_rotary_emb,
                    joint_attention_kwargs=attention_kwargs,
                )
        # Update forecaster with the real output
        forecaster.update(float(self._spectrum_step_idx), hidden_states)
        self._spectrum_actual_count += 1
    else:
        # Predict post-block hidden states from the forecaster
        hidden_states = forecaster.predict(float(self._spectrum_step_idx))
        self._spectrum_cached_count += 1

    # ---- Cheap postamble (always runs) ----
    hidden_states = self.norm_out(hidden_states, temb)
    output = self.proj_out(hidden_states)

    if USE_PEFT_BACKEND and unscale_lora_layers is not None:
        unscale_lora_layers(self, lora_scale)

    # ---- Advance counters ----
    self._spectrum_sub_call += 1
    if self._spectrum_sub_call >= self._spectrum_calls_per_step:
        self._spectrum_sub_call = 0
        if actual_forward:
            self._spectrum_num_consecutive_cached = 0
        else:
            self._spectrum_num_consecutive_cached += 1
        self._spectrum_step_idx += 1

    if not return_dict:
        return (output,)
    return Transformer2DModelOutput(sample=output)


# ======================================================================
#  Patch / unpatch
# ======================================================================

# Attributes we set on the transformer instance (for cleanup)
_SPECTRUM_ATTRS = [
    "_spectrum_step_idx",
    "_spectrum_sub_call",
    "_spectrum_calls_per_step",
    "_spectrum_step_is_actual",
    "_spectrum_num_steps",
    "_spectrum_warmup_steps",
    "_spectrum_window_size",
    "_spectrum_curr_ws",
    "_spectrum_flex_window",
    "_spectrum_num_consecutive_cached",
    "_spectrum_forecasters",
    "_spectrum_actual_count",
    "_spectrum_cached_count",
]


def patch_transformer_spectrum(
    transformer,
    num_steps: int,
    config: dict,
    calls_per_step: int = 1,
) -> Callable[[], dict]:
    """
    Monkey-patch *transformer* to use Spectrum-accelerated forward.

    Args:
        transformer: ``QwenImageTransformer2DModel`` instance.
        num_steps: Number of denoising steps.
        config: Spectrum configuration dict (from the ComfyUI node) with keys:
            warmup_steps, window_size, flex_window, w, lam, M, K, taylor_order
        calls_per_step: 1 (no CFG) or 2 (True CFG).

    Returns:
        An ``unpatch()`` callable. When called, it restores the original
        forward and clears all ``_spectrum_*`` attributes.  Returns a dict
        of stats: ``{"actual_forwards": int, "cached_steps": int}``.
    """
    device = next(transformer.parameters()).device

    # Extract config values with sensible defaults
    warmup_steps = config.get("warmup_steps", 3)
    window_size  = config.get("window_size", 2)
    flex_window  = config.get("flex_window", 0.75)
    w            = config.get("w", 0.5)
    lam          = config.get("lam", 0.1)
    M            = config.get("M", 4)
    K            = config.get("K", 20)
    taylor_order = config.get("taylor_order", 1)

    # Create forecasters (one per sub-call slot)
    forecasters = []
    for _ in range(calls_per_step):
        forecasters.append(
            SpectrumForecaster(
                M=M, K=K, lam=lam, w=w,
                taylor_order=taylor_order, device=device,
            )
        )

    # Set state on the transformer instance
    transformer._spectrum_step_idx = 0
    transformer._spectrum_sub_call = 0
    transformer._spectrum_calls_per_step = calls_per_step
    transformer._spectrum_step_is_actual = True  # will be set each step
    transformer._spectrum_num_steps = num_steps
    transformer._spectrum_warmup_steps = warmup_steps
    transformer._spectrum_window_size = window_size
    transformer._spectrum_curr_ws = float(window_size)
    transformer._spectrum_flex_window = flex_window
    transformer._spectrum_num_consecutive_cached = 0
    transformer._spectrum_forecasters = forecasters
    transformer._spectrum_actual_count = 0
    transformer._spectrum_cached_count = 0

    # Save and replace the forward method
    original_forward = transformer.__class__.forward
    transformer.__class__.forward = _qwen_spectrum_forward

    # Estimate how many actual forwards will happen (for logging)
    est_actual = _estimate_actual_steps(num_steps, warmup_steps, window_size, flex_window)
    est_cached = num_steps - est_actual
    logger.info(
        f"Spectrum: patched ({calls_per_step}x call{'s' if calls_per_step > 1 else ''}/step), "
        f"~{est_actual} actual / ~{est_cached} cached of {num_steps} steps"
    )

    def unpatch() -> dict:
        """Restore original forward, clean up state, return stats."""
        transformer.__class__.forward = original_forward
        stats = {
            "actual_forwards": getattr(transformer, "_spectrum_actual_count", 0),
            "cached_steps": getattr(transformer, "_spectrum_cached_count", 0),
        }
        for attr in _SPECTRUM_ATTRS:
            try:
                delattr(transformer, attr)
            except AttributeError:
                pass
        return stats

    return unpatch


def _estimate_actual_steps(
    num_steps: int, warmup: int, window: int, flex: float
) -> int:
    """
    Simulate the schedule to estimate total actual-forward steps.
    Used only for logging; not performance-critical.
    """
    actual = 0
    consecutive_cached = 0
    curr_ws = float(window)

    for step in range(num_steps):
        if step < warmup:
            is_actual = True
        else:
            is_actual = (consecutive_cached + 1) % max(1, math.floor(curr_ws)) == 0
            if is_actual:
                curr_ws += flex

        if is_actual:
            actual += 1
            consecutive_cached = 0
        else:
            consecutive_cached += 1

    return actual
