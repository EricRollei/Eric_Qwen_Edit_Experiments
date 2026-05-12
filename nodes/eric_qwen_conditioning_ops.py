# Copyright (c) 2026 Eric Hiss. All rights reserved.
# Licensed under the terms in LICENSE.txt (CC BY-NC 4.0 / Commercial dual license).
# https://github.com/EricRollei/Eric_Qwen_Edit_Experiments
"""
Eric Qwen Conditioning Manipulation Nodes

Nodes for combining and interpolating QWEN_CONDITIONING tensors.

    EricQwenConditioningInterpolate  - linear/spherical blend between two conditionings
    EricQwenConditioningBlend        - weighted sum of two conditionings

Interpolation methods
---------------------
lerp  - linear interpolation. Fast, simple. Slightly reduces embedding magnitude
        at the midpoint (cuts through the interior of the embedding sphere rather
        than following its surface). Fine for moderate alpha values (0.3-1.2).

slerp - spherical linear interpolation. Follows the surface of the embedding sphere,
        preserving magnitude throughout the interpolation arc. More perceptually
        uniform steps and more graceful degradation at large alpha (>1.5) or
        negative alpha. Recommended when sweeping a wide alpha range.

Scope quick reference
---------------------
    all_tokens    - interpolate everything  (safe default)
    skip_prefix_N - keep image tokens, interpolate text tokens only
                    → expression / attribute editing (same input image)
    keep_prefix_N - interpolate image tokens only, keep text tokens from cond_A
                    → style transfer (different input images)
    last_N_tokens - last scope_N tokens only  (not useful for Qwen)

Author: Eric Hiss (GitHub: EricRollei)
"""

import math
import torch
import torch.nn.functional as F


# ──────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ──────────────────────────────────────────────────────────────────────────────

def _align_conditionings(cond_a: dict, cond_b: dict) -> tuple:
    """
    Pad the shorter conditioning to match the longer one's sequence length.
    Returns ea, ma, eb, mb - all at [1, max_seq, *].
    """
    ea = cond_a["prompt_embeds"]
    ma = cond_a["prompt_embeds_mask"]
    eb = cond_b["prompt_embeds"]
    mb = cond_b["prompt_embeds_mask"]

    seq_a, seq_b = ea.shape[1], eb.shape[1]
    max_seq = max(seq_a, seq_b)

    if seq_a < max_seq:
        pad = max_seq - seq_a
        ea = F.pad(ea, (0, 0, 0, pad))
        ma = F.pad(ma, (0, pad))
    if seq_b < max_seq:
        pad = max_seq - seq_b
        eb = F.pad(eb, (0, 0, 0, pad))
        mb = F.pad(mb, (0, pad))

    return ea, ma, eb, mb


def _make_result(embeds: torch.Tensor, mask: torch.Tensor, meta: dict) -> dict:
    """Pack tensors into a QWEN_CONDITIONING dict, stored on CPU."""
    return {
        "prompt_embeds":      embeds.cpu(),
        "prompt_embeds_mask": mask.cpu(),
        "metadata": {
            "valid_tokens": int(mask.sum().item()),
            "total_tokens": embeds.shape[1],
            **meta,
        },
    }


def _resolve_prefix_N(prefix_N: int, cond_A: dict, label: str = "",
                      scope: str = "") -> int:
    """
    Resolve the actual prefix_N value for scope-based interpolation.
    If prefix_N == 0 (auto), read image_tokens from cond_A's metadata.
    Returns 0 if metadata missing (caller falls back to all_tokens).
    """
    if prefix_N != 0:
        return prefix_N

    image_tokens = cond_A.get("metadata", {}).get("image_tokens", 0)
    if image_tokens:
        tag = f"[{label}] " if label else ""
        scope_str = f"{scope} " if scope else ""
        print(f"{tag}{scope_str}auto: prefix_N={image_tokens} (from cond_A metadata)")
        return image_tokens

    return 0


def _slerp_tokens(
    a: torch.Tensor,
    b: torch.Tensor,
    t: float,
) -> torch.Tensor:
    """
    Spherical linear interpolation for token embedding tensors.
    Shape: [batch, seq_len, hidden_dim]

    For each token position, interpolates along the arc of the unit sphere
    in 3584-dimensional space, then scales by a linearly interpolated magnitude.

    This preserves embedding magnitude throughout the interpolation (unlike lerp
    which reduces magnitude at the midpoint by cutting through the sphere interior).
    Works for any t value including extrapolation (t > 1 or t < 0).

    Falls back to lerp for tokens whose embeddings are nearly parallel or
    antiparallel (sin(theta) < 1e-6) to avoid numerical instability.
    """
    # Per-token L2 magnitudes: [batch, seq_len, 1]
    a_mag = a.norm(dim=-1, keepdim=True)
    b_mag = b.norm(dim=-1, keepdim=True)

    # Unit vectors for direction interpolation (handle zero-magnitude tokens)
    a_unit = F.normalize(a, dim=-1)
    b_unit = F.normalize(b, dim=-1)

    # Angle between unit vectors: [batch, seq_len, 1]
    # Clamp to avoid NaN in acos at exactly ±1.0
    dot = (a_unit * b_unit).sum(dim=-1, keepdim=True).clamp(-1.0 + 1e-7, 1.0 - 1e-7)
    theta = torch.acos(dot)       # angle in radians
    sin_theta = torch.sin(theta)  # denominator

    # Slerp coefficients - fall back to lerp when theta ≈ 0 (parallel tokens)
    nearly_parallel = sin_theta.abs() < 1e-6
    coeff_a = torch.where(
        nearly_parallel,
        torch.full_like(theta, 1.0 - t),
        torch.sin((1.0 - t) * theta) / sin_theta,
    )
    coeff_b = torch.where(
        nearly_parallel,
        torch.full_like(theta, t),
        torch.sin(t * theta) / sin_theta,
    )

    # Interpolated unit direction, renormalized to correct floating-point drift
    direction = F.normalize(coeff_a * a_unit + coeff_b * b_unit, dim=-1)

    # Linearly interpolated magnitude (slerp doesn't constrain magnitude)
    magnitude = (1.0 - t) * a_mag + t * b_mag

    return direction * magnitude


def _interp(
    a: torch.Tensor,
    b: torch.Tensor,
    t: float,
    method: str = "lerp",
) -> torch.Tensor:
    """
    Dispatch to lerp or slerp for two token embedding tensors.
    a, b: [batch, seq_len, hidden_dim]
    t:    interpolation coefficient (can be outside [0,1] for extrapolation)
    """
    if method == "slerp":
        return _slerp_tokens(a, b, t)
    return a + t * (b - a)   # lerp


def _apply_interpolation(
    ea: torch.Tensor, ma: torch.Tensor,
    eb: torch.Tensor, mb: torch.Tensor,
    alpha: float,
    scope: str,
    scope_N: int,
    prefix_N: int,
    method: str = "lerp",
) -> tuple:
    """
    Core interpolation logic shared by Interpolate and DirectionApply.

    method: "lerp" (default) or "slerp"

    Scope map
    ---------
    all_tokens    - interpolate entire sequence
    skip_prefix_N - keep first prefix_N tokens from A (image tokens),
                    interpolate the text tokens after
    keep_prefix_N - interpolate first prefix_N tokens (image tokens),
                    keep text tokens unchanged from A
    last_N_tokens - keep prefix from A, interpolate last scope_N tokens
    """
    max_seq = ea.shape[1]

    if scope == "skip_prefix_N":
        p = min(prefix_N, max_seq - 1)
        result_e = torch.cat([
            ea[:, :p, :],
            _interp(ea[:, p:, :], eb[:, p:, :], alpha, method),
        ], dim=1)
        result_m = torch.cat([
            ma[:, :p],
            torch.max(ma[:, p:], mb[:, p:]),
        ], dim=1)

    elif scope == "keep_prefix_N":
        p = min(prefix_N, max_seq - 1)
        result_e = torch.cat([
            _interp(ea[:, :p, :], eb[:, :p, :], alpha, method),
            ea[:, p:, :],
        ], dim=1)
        result_m = torch.cat([
            torch.max(ma[:, :p], mb[:, :p]),
            ma[:, p:],
        ], dim=1)

    elif scope == "last_N_tokens":
        n = min(scope_N, max_seq)
        result_e = torch.cat([
            ea[:, :-n, :],
            _interp(ea[:, -n:, :], eb[:, -n:, :], alpha, method),
        ], dim=1)
        result_m = torch.cat([
            ma[:, :-n],
            torch.max(ma[:, -n:], mb[:, -n:]),
        ], dim=1)

    else:  # all_tokens
        result_e = _interp(ea, eb, alpha, method)
        result_m = torch.max(ma, mb)

    return result_e, result_m


# ──────────────────────────────────────────────────────────────────────────────
# Nodes
# ──────────────────────────────────────────────────────────────────────────────

class EricQwenConditioningInterpolate:
    """
    Linear or spherical interpolation between two QWEN_CONDITIONING tensors.

    formula (lerp):   result = cond_A + alpha * (cond_B - cond_A)
    formula (slerp):  arc interpolation preserving embedding magnitude

    alpha  0.0  ->  pure cond_A  (no change - use as clean baseline)
    alpha  1.0  ->  pure cond_B  (full target)
    alpha >1.0  ->  extrapolated beyond cond_B  (exaggeration)
    alpha <0.0  ->  opposite direction  (suppression)

    Method
    ------
    lerp   - linear interpolation. Fast. Reduces embedding magnitude at midpoints
             (cuts through the embedding sphere interior). Fine for alpha 0.3-1.2.

    slerp  - spherical linear interpolation. Follows the sphere surface, preserving
             magnitude throughout the arc. More perceptually uniform steps when
             sweeping alpha. Better behavior at alpha > 1.5 or < 0 where lerp
             magnitude can grow or collapse. Slightly slower.

    Scope
    -----
    all_tokens    - interpolate entire sequence (safe default).
    skip_prefix_N - keep image tokens, interpolate text only → expression editing.
    keep_prefix_N - interpolate image tokens only, keep text → style transfer.
    last_N_tokens - NOT recommended for Qwen.

    Cheat sheet
    -----------
    Same image, different prompts  (expression, attributes) -> skip_prefix_N
    Different images, same prompt  (style transfer)         -> keep_prefix_N
    Unsure / mixed                                          -> all_tokens
    Wide alpha sweep or >1.5 extrapolation                  -> slerp method
    """

    CATEGORY = "Eric Qwen-Edit/Conditioning"
    FUNCTION = "interpolate"
    RETURN_TYPES = ("QWEN_CONDITIONING",)
    RETURN_NAMES = ("conditioning",)

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "cond_A": ("QWEN_CONDITIONING", {
                    "tooltip": "Baseline conditioning (alpha=0 result)"
                }),
                "cond_B": ("QWEN_CONDITIONING", {
                    "tooltip": "Target conditioning (alpha=1 result)"
                }),
                "alpha": ("FLOAT", {
                    "default": 1.0, "min": -5.0, "max": 5.0, "step": 0.05,
                    "tooltip": (
                        "0=cond_A (no change), 1=cond_B (full target), "
                        ">1=exaggerated, <0=opposite.\n"
                        "Text prompt dominates coarse intent; alpha shapes the character."
                    )
                }),
            },
            "optional": {
                "method": (["lerp", "slerp"], {
                    "default": "lerp",
                    "tooltip": (
                        "lerp:  linear interpolation. Fast, good for alpha 0.3-1.2.\n"
                        "slerp: spherical interpolation. Preserves embedding magnitude.\n"
                        "       More uniform perceptual steps. Better for wide alpha "
                        "       sweeps or large extrapolation (>1.5)."
                    )
                }),
                "scope": (["all_tokens", "skip_prefix_N", "keep_prefix_N",
                           "last_N_tokens"], {
                    "default": "all_tokens",
                    "tooltip": (
                        "all_tokens:    interpolate everything.\n"
                        "skip_prefix_N: text tokens only → expression editing.\n"
                        "keep_prefix_N: image tokens only → style transfer.\n"
                        "last_N_tokens: not recommended for Qwen."
                    )
                }),
                "prefix_N": ("INT", {
                    "default": 0, "min": 0, "max": 512, "step": 1,
                    "tooltip": "0 = auto from Encode metadata (recommended)."
                }),
                "scope_N": ("INT", {
                    "default": 7, "min": 1, "max": 128, "step": 1,
                    "tooltip": "Only used for last_N_tokens scope."
                }),
            },
        }

    def interpolate(
        self,
        cond_A: dict,
        cond_B: dict,
        alpha: float,
        method: str = "lerp",
        scope: str = "all_tokens",
        prefix_N: int = 0,
        scope_N: int = 7,
    ) -> tuple:
        ea, ma, eb, mb = _align_conditionings(cond_A, cond_B)

        active_prefix_N = prefix_N
        if scope in ("skip_prefix_N", "keep_prefix_N"):
            active_prefix_N = _resolve_prefix_N(prefix_N, cond_A, "EricQwenInterp", scope)
            if active_prefix_N == 0:
                print(f"[EricQwenInterp] WARNING: {scope} - image_tokens missing. "
                      "Falling back to all_tokens.")
                scope = "all_tokens"

        result_e, result_m = _apply_interpolation(
            ea, ma, eb, mb, alpha, scope, scope_N, active_prefix_N, method
        )

        src_a = cond_A.get("metadata", {}).get("source_prompt", "?")[:40]
        src_b = cond_B.get("metadata", {}).get("source_prompt", "?")[:40]
        tok_a = cond_A.get("metadata", {}).get("valid_tokens", "?")
        tok_b = cond_B.get("metadata", {}).get("valid_tokens", "?")
        scope_detail = (f" prefix={active_prefix_N}" if scope in ("skip_prefix_N", "keep_prefix_N")
                        else f" N={scope_N}" if scope == "last_N_tokens" else "")
        print(f"[EricQwenInterp] α={alpha:.3f} method={method} scope={scope}{scope_detail} | "
              f"A='{src_a}'({tok_a}) <-> B='{src_b}'({tok_b})")

        return (_make_result(
            result_e, result_m,
            {
                "source_prompt": f"interp(α={alpha:.2f}, {method}, {scope}{scope_detail})",
                "alpha":   alpha,
                "method":  method,
                "scope":   scope,
                "parent_A": src_a,
                "parent_B": src_b,
            }
        ),)


class EricQwenConditioningBlend:
    """
    Weighted blend of two QWEN_CONDITIONING tensors.

    result = (weight_A * cond_A + weight_B * cond_B) / (weight_A + weight_B)

    Compound expression example:
        Blend(cond_surprised, cond_happy, 0.4, 0.6)  ->  mostly happy, hint of surprise
    """

    CATEGORY = "Eric Qwen-Edit/Conditioning"
    FUNCTION = "blend"
    RETURN_TYPES = ("QWEN_CONDITIONING",)
    RETURN_NAMES = ("conditioning",)

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "cond_A": ("QWEN_CONDITIONING",),
                "cond_B": ("QWEN_CONDITIONING",),
                "weight_A": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 2.0, "step": 0.05}),
                "weight_B": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 2.0, "step": 0.05}),
            },
            "optional": {
                "normalize": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Normalize weights to sum to 1.0 (recommended)."
                }),
            },
        }

    def blend(self, cond_A, cond_B, weight_A, weight_B, normalize=True):
        ea, ma, eb, mb = _align_conditionings(cond_A, cond_B)
        total = weight_A + weight_B
        wa, wb = (weight_A / total, weight_B / total) if (normalize and total > 1e-6) else (weight_A, weight_B)
        result_e = wa * ea + wb * eb
        result_m = torch.max(ma, mb)
        src_a = cond_A.get("metadata", {}).get("source_prompt", "?")[:40]
        src_b = cond_B.get("metadata", {}).get("source_prompt", "?")[:40]
        print(f"[EricQwenBlend] {wa:.2f}xA + {wb:.2f}xB | A='{src_a}' B='{src_b}'")
        return (_make_result(result_e, result_m,
                             {"source_prompt": f"blend({wa:.2f}xA+{wb:.2f}xB)",
                              "weight_A": wa, "weight_B": wb,
                              "parent_A": src_a, "parent_B": src_b}),)
