# Copyright (c) 2026 Eric Hiss. All rights reserved.
# Licensed under the terms in LICENSE.txt (CC BY-NC 4.0 / Commercial dual license).
# https://github.com/EricRollei/Eric_Qwen_Edit_Experiments
"""
Eric Qwen Direction Nodes

Semantic direction vectors in QWEN_CONDITIONING embedding space.

direction = V_target - V_baseline

Normalization
-------------
Averaging N directions produces a purer direction (correct angle, consistent
across many images) but at reduced magnitude - the inconsistent components
cancel, leaving the shared semantic axis at lower RMS.

Normalization restores the magnitude to a target RMS without changing the
direction (angle). It ONLY changes the length of the vector, not which way it
points. This makes `scale=1.0` semantically consistent across different direction
files - useful when building a direction library where you want comparable
intensity across multiple direction types.

    normalized_delta = delta * (target_rms / current_rms)

Typical target_rms values:
    1.0   unit direction - abstract, always consistent
    2.0   moderate - good library default
    3.5   approximate mean of individual pair RMS - restores original strength

Author: Eric Hiss (GitHub: EricRollei)
"""

import os
import torch
import torch.nn.functional as F

from .eric_qwen_conditioning_ops import (
    _align_conditionings,
    _make_result,
    _apply_interpolation,
    _resolve_prefix_N,
)


def _get_directions_dir() -> str:
    try:
        import folder_paths
        d = os.path.join(folder_paths.models_dir, "qwen_directions")
    except Exception:
        d = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models", "qwen_directions")
    os.makedirs(d, exist_ok=True)
    return d


def _rms(delta: torch.Tensor, mask: torch.Tensor) -> float:
    valid = mask[0].bool()
    if not valid.any():
        return 0.0
    return float(delta[0][valid].pow(2).mean().sqrt())


def _normalize_delta(
    delta: torch.Tensor,
    mask: torch.Tensor,
    target_rms: float,
) -> tuple:
    """
    Scale delta tensor so its RMS over valid token positions equals target_rms.
    Only the magnitude changes - the direction (angle) is preserved exactly.
    Returns (normalized_delta, actual_rms_before, scale_factor).
    """
    current_rms = _rms(delta, mask)
    if current_rms < 1e-8:
        return delta, current_rms, 1.0
    scale = target_rms / current_rms
    return delta * scale, current_rms, scale


def _pad_direction(direction: dict, target_seq: int) -> tuple:
    de = direction["delta_embeds"]
    dm = direction["delta_mask"]
    pad = target_seq - de.shape[1]
    if pad > 0:
        de = F.pad(de, (0, 0, 0, pad))
        dm = F.pad(dm, (0, pad))
    return de, dm


def _load_direction_file(path: str) -> dict:
    from safetensors.torch import load_file
    from safetensors import safe_open
    tensors = load_file(path, device="cpu")
    metadata = {}
    with safe_open(path, framework="pt", device="cpu") as f:
        raw = f.metadata()
        if raw:
            metadata = dict(raw)
    return {
        "delta_embeds": tensors["delta_embeds"].to(torch.bfloat16),
        "delta_mask":   tensors["delta_mask"],
        "metadata":     metadata,
    }


def _average_direction_list(
    dirs: list,
    label: str = "",
    normalize: bool = False,
    target_rms: float = 2.0,
) -> dict:
    max_seq = max(d["delta_embeds"].shape[1] for d in dirs)
    padded  = [_pad_direction(d, max_seq) for d in dirs]

    avg_embeds = torch.stack([p[0] for p in padded], dim=0).mean(dim=0)
    avg_mask   = torch.stack([p[1] for p in padded], dim=0).max(dim=0).values

    raw_rms = _rms(avg_embeds, avg_mask)

    if normalize and raw_rms > 1e-8:
        avg_embeds, pre_rms, scale = _normalize_delta(avg_embeds, avg_mask, target_rms)
        final_rms = target_rms
        print(f"[EricQwenDir] {label}averaged {len(dirs)} dirs | "
              f"seq={max_seq}, raw_rms={raw_rms:.5f} "
              f"→ normalized rms={final_rms:.2f} (scale={scale:.3f})")
    else:
        final_rms = raw_rms
        print(f"[EricQwenDir] {label}averaged {len(dirs)} directions | "
              f"seq={max_seq}, valid={int(avg_mask.sum())}, rms={final_rms:.5f}")

    return {
        "delta_embeds": avg_embeds.cpu(),
        "delta_mask":   avg_mask.cpu(),
        "metadata": {
            "target_prompt":   f"avg({len(dirs)} dirs{': ' + label.strip() if label else ''})",
            "baseline_prompt": "",
            "seq_len":         max_seq,
            "valid_tokens":    int(avg_mask.sum().item()),
            "rms":             final_rms,
            "raw_rms":         raw_rms,
            "n_averaged":      len(dirs),
            "normalized":      normalize,
            "target_rms":      target_rms if normalize else None,
        },
    }


def _list_subfolders() -> list:
    root = _get_directions_dir()
    subs = sorted(
        d for d in os.listdir(root)
        if os.path.isdir(os.path.join(root, d)) and not d.startswith(".")
    )
    return ["(root folder)"] + subs


class EricQwenDirectionCompute:
    """Compute: direction = cond_target - cond_baseline"""

    CATEGORY = "Eric Qwen-Edit/Conditioning/Directions"
    FUNCTION = "compute"
    RETURN_TYPES = ("QWEN_DIRECTION",)
    RETURN_NAMES = ("direction",)

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "cond_target":   ("QWEN_CONDITIONING", {"tooltip": "What you want more of"}),
                "cond_baseline": ("QWEN_CONDITIONING", {"tooltip": "What you want to move away from"}),
            },
        }

    def compute(self, cond_target: dict, cond_baseline: dict) -> tuple:
        et, mt, eb, mb = _align_conditionings(cond_target, cond_baseline)
        delta      = et - eb
        delta_mask = torch.max(mt, mb)
        src_t  = cond_target.get("metadata",  {}).get("source_prompt", "?")[:60]
        src_b  = cond_baseline.get("metadata", {}).get("source_prompt", "?")[:60]
        rms_val = _rms(delta, delta_mask)
        per_token_rms = delta[0].norm(dim=-1)
        peak_pos = int(per_token_rms.argmax().item())
        peak_val = float(per_token_rms.max().item())
        print(f"[EricQwenDir] computed '{src_t}' - '{src_b}' | "
              f"seq={delta.shape[1]}, rms={rms_val:.5f}, "
              f"peak_token={peak_pos} (rms={peak_val:.3f})")
        return ({
            "delta_embeds": delta.cpu(),
            "delta_mask":   delta_mask.cpu(),
            "metadata": {
                "target_prompt":   src_t,
                "baseline_prompt": src_b,
                "seq_len":         delta.shape[1],
                "valid_tokens":    int(delta_mask.sum().item()),
                "rms":             rms_val,
                "peak_token":      peak_pos,
            },
        },)


class EricQwenDirectionAverage:
    """Average 2-4 QWEN_DIRECTION vectors into a debiased direction."""

    CATEGORY = "Eric Qwen-Edit/Conditioning/Directions"
    FUNCTION = "average"
    RETURN_TYPES = ("QWEN_DIRECTION",)
    RETURN_NAMES = ("direction",)

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "direction_1": ("QWEN_DIRECTION",),
                "direction_2": ("QWEN_DIRECTION",),
            },
            "optional": {
                "direction_3": ("QWEN_DIRECTION",),
                "direction_4": ("QWEN_DIRECTION",),
                "normalize": ("BOOLEAN", {"default": False}),
                "target_rms": ("FLOAT", {"default": 2.0, "min": 0.1, "max": 10.0, "step": 0.1}),
            },
        }

    def average(self, direction_1, direction_2, direction_3=None, direction_4=None,
                normalize=False, target_rms=2.0):
        dirs   = [d for d in [direction_1, direction_2, direction_3, direction_4] if d is not None]
        result = _average_direction_list(dirs, normalize=normalize, target_rms=target_rms)
        return (result,)


class EricQwenDirectionAverageFromFolder:
    """Average all QWEN_DIRECTION files in a subfolder or matching a prefix."""

    CATEGORY = "Eric Qwen-Edit/Conditioning/Directions"
    FUNCTION = "average_from_folder"
    RETURN_TYPES = ("QWEN_DIRECTION",)
    RETURN_NAMES = ("direction",)

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "folder": (_list_subfolders(), {
                    "tooltip": "Subfolder of models/qwen_directions/. Refreshes on every run."
                }),
            },
            "optional": {
                "prefix":       ("STRING",  {"default": ""}),
                "normalize":    ("BOOLEAN", {"default": True}),
                "target_rms":   ("FLOAT",   {"default": 2.0, "min": 0.1, "max": 10.0, "step": 0.1}),
                "save_as":      ("STRING",  {"default": ""}),
                "save_as_bf16": ("BOOLEAN", {"default": False}),
            },
        }

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("nan")

    def average_from_folder(self, folder, prefix="", normalize=True, target_rms=2.0,
                             save_as="", save_as_bf16=False):
        root = _get_directions_dir()
        search_dir = root if folder == "(root folder)" else os.path.join(root, folder)
        if not os.path.isdir(search_dir):
            raise FileNotFoundError(f"Subfolder not found: {search_dir}")

        all_files = sorted(f for f in os.listdir(search_dir) if f.endswith(".safetensors"))
        if prefix:
            all_files = [f for f in all_files if f.startswith(prefix)]
        if not all_files:
            raise ValueError(f"No .safetensors files in '{search_dir}'" +
                             (f" with prefix '{prefix}'" if prefix else ""))

        print(f"[EricQwenDir] AverageFromFolder: {len(all_files)} files from '{folder}'"
              + (f" prefix='{prefix}'" if prefix else ""))

        dirs, rms_vals = [], []
        for fname in all_files:
            path = os.path.join(search_dir, fname)
            try:
                d = _load_direction_file(path)
                r = float(d.get("metadata", {}).get("rms", 0))
                rms_vals.append(r)
                print(f"[EricQwenDir]   + {fname} ({os.path.getsize(path)/1e6:.2f} MB, rms={r:.4f})")
                dirs.append(d)
            except Exception as e:
                print(f"[EricQwenDir]   ! skipping {fname}: {e}")

        if not dirs:
            raise ValueError("All files failed to load.")
        if rms_vals:
            print(f"[EricQwenDir]   mean individual rms={sum(rms_vals)/len(rms_vals):.4f}")

        label  = f"'{folder}'" + (f" prefix='{prefix}'" if prefix else "") + " "
        result = _average_direction_list(dirs, label=label, normalize=normalize, target_rms=target_rms)

        if save_as.strip():
            from safetensors.torch import save_file
            safe_name = "".join(c for c in save_as.strip() if c.isalnum() or c in "_-") or "avg"
            out_path  = os.path.join(root, safe_name + ".safetensors")
            dtype     = torch.bfloat16 if save_as_bf16 else torch.float32
            save_file({"delta_embeds": result["delta_embeds"].to(dtype),
                       "delta_mask":   result["delta_mask"].float()},
                      out_path,
                      metadata={str(k): str(v) for k, v in result["metadata"].items()})
            print(f"[EricQwenDir] saved '{safe_name}.safetensors' "
                  f"({os.path.getsize(out_path)/1e6:.2f} MB) -> {root}")

        return (result,)


class EricQwenDirectionApply:
    """
    Apply a QWEN_DIRECTION to a base QWEN_CONDITIONING.

    result = base + scale × direction   (or scale × normalized_direction)

    scale  0.0  passthrough baseline (no change - A/B comparison)
    scale  1.0  standard application
    scale  1.5  amplified
    scale -1.0  opposite direction

    With a LoRA that was trained for this direction, start at scale=0.5
    and work up - the LoRA increases responsiveness so lower scale suffices.

    method
    ------
    lerp:  linear interpolation. Fast. Slightly reduces embedding magnitude
           at midpoints. Fine for scale 0.3-1.2.
    slerp: spherical interpolation. Preserves embedding magnitude along the
           arc. More perceptually uniform. Better at large scale (>1.5) or
           negative scale where lerp magnitude can become unpredictable.

    normalize_direction
    -------------------
    Rescale direction to target_rms before applying scale.
    Makes scale semantically consistent across direction files regardless
    of their natural RMS. Only changes vector length, not angle.
    """

    CATEGORY = "Eric Qwen-Edit/Conditioning/Directions"
    FUNCTION = "apply_direction"
    RETURN_TYPES = ("QWEN_CONDITIONING",)
    RETURN_NAMES = ("conditioning",)

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "conditioning": ("QWEN_CONDITIONING", {"tooltip": "Base conditioning to modify"}),
                "direction":    ("QWEN_DIRECTION",    {"tooltip": "Direction to apply"}),
                "scale": ("FLOAT", {
                    "default": 1.0, "min": -5.0, "max": 5.0, "step": 0.05,
                    "tooltip": "0=passthrough, 1=standard, >1=amplified, <0=opposite.\n"
                               "With a trained LoRA, start at 0.5 - LoRA increases responsiveness."
                }),
            },
            "optional": {
                "method": (["lerp", "slerp"], {
                    "default": "lerp",
                    "tooltip": (
                        "lerp:  linear interpolation (fast, good for scale 0.3-1.2).\n"
                        "slerp: spherical interpolation (preserves embedding magnitude;\n"
                        "       better for large scale >1.5 or negative scale)."
                    )
                }),
                "scope": (["all_tokens", "skip_prefix_N", "keep_prefix_N", "last_N_tokens"], {
                    "default": "all_tokens",
                    "tooltip": (
                        "all_tokens:    apply to entire sequence.\n"
                        "skip_prefix_N: text tokens only → expression / attribute.\n"
                        "keep_prefix_N: image tokens only → style / film / visual.\n"
                        "last_N_tokens: not recommended for Qwen."
                    )
                }),
                "prefix_N": ("INT", {
                    "default": 0, "min": 0, "max": 512, "step": 1,
                    "tooltip": "0 = auto from Encode metadata."
                }),
                "scope_N": ("INT", {
                    "default": 7, "min": 1, "max": 128, "step": 1,
                    "tooltip": "Only used for last_N_tokens scope."
                }),
                "normalize_direction": ("BOOLEAN", {
                    "default": False,
                    "tooltip": (
                        "Normalize direction to target_rms before applying scale.\n"
                        "Makes scale comparable across direction files.\n"
                        "Recommended when chaining multiple directions."
                    )
                }),
                "target_rms": ("FLOAT", {
                    "default": 2.0, "min": 0.1, "max": 10.0, "step": 0.1,
                    "tooltip": "RMS target when normalize_direction=True. Use same value across all Apply nodes."
                }),
            },
        }

    def apply_direction(
        self,
        conditioning: dict,
        direction: dict,
        scale: float,
        method: str = "lerp",
        scope: str = "all_tokens",
        prefix_N: int = 0,
        scope_N: int = 7,
        normalize_direction: bool = False,
        target_rms: float = 2.0,
    ) -> tuple:
        ec = conditioning["prompt_embeds"]
        mc = conditioning["prompt_embeds_mask"]
        ed = direction["delta_embeds"]
        md = direction["delta_mask"]

        max_seq = max(ec.shape[1], ed.shape[1])
        if ec.shape[1] < max_seq:
            ec = F.pad(ec, (0, 0, 0, max_seq - ec.shape[1]))
            mc = F.pad(mc, (0, max_seq - mc.shape[1]))
        if ed.shape[1] < max_seq:
            ed = F.pad(ed, (0, 0, 0, max_seq - ed.shape[1]))
            md = F.pad(md, (0, max_seq - md.shape[1]))

        if normalize_direction:
            ed, original_rms, norm_scale = _normalize_delta(ed, md, target_rms)
            print(f"[EricQwenDirApply] normalized: rms {original_rms:.4f} "
                  f"→ {target_rms:.2f} (x{norm_scale:.3f})")

        active_prefix_N = prefix_N
        if scope in ("skip_prefix_N", "keep_prefix_N"):
            active_prefix_N = _resolve_prefix_N(prefix_N, conditioning, "EricQwenDirApply", scope)
            if active_prefix_N == 0:
                print(f"[EricQwenDirApply] WARNING: {scope} - image_tokens missing. "
                      "Falling back to all_tokens.")
                scope = "all_tokens"

        eb          = ec + ed
        mb_combined = torch.max(mc, md)

        result_e, result_m = _apply_interpolation(
            ec, mc, eb, mb_combined, scale, scope, scope_N, active_prefix_N, method
        )

        dir_name     = direction.get("metadata", {}).get("target_prompt", "?")[:40]
        src_name     = conditioning.get("metadata", {}).get("source_prompt", "?")[:40]
        scope_detail = (f" N={scope_N}"           if scope == "last_N_tokens" else
                        f" prefix={active_prefix_N}" if scope in ("skip_prefix_N", "keep_prefix_N") else "")
        norm_str     = f" [norm→{target_rms}]" if normalize_direction else ""
        print(f"[EricQwenDirApply] scale={scale:.3f} method={method} "
              f"scope={scope}{scope_detail}{norm_str} | "
              f"base='{src_name}' dir='{dir_name}'")

        return (_make_result(
            result_e, result_m,
            {
                "source_prompt": f"apply('{dir_name}', scale={scale:.2f}, {method}{norm_str})",
                "parent":    src_name,
                "direction": dir_name,
                "scale":     scale,
                "method":    method,
            }
        ),)


class EricQwenDirectionSave:
    """Save QWEN_DIRECTION to models/qwen_directions/. Passthrough."""

    CATEGORY = "Eric Qwen-Edit/Conditioning/Directions"
    FUNCTION = "save"
    RETURN_TYPES = ("QWEN_DIRECTION",)
    RETURN_NAMES = ("direction",)
    OUTPUT_NODE = True

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "direction": ("QWEN_DIRECTION",),
                "filename":  ("STRING", {
                    "default": "my_direction",
                    "tooltip": "Subfolders supported: 'silvermax/01'"
                }),
            },
            "optional": {
                "save_as_bf16": ("BOOLEAN", {"default": False}),
            },
        }

    def save(self, direction: dict, filename: str, save_as_bf16: bool = False) -> tuple:
        from safetensors.torch import save_file
        save_dir = _get_directions_dir()
        parts      = filename.replace("\\", "/").split("/")
        safe_parts = ["".join(c for c in p if c.isalnum() or c in "_-") for p in parts]
        safe_parts = [p for p in safe_parts if p] or ["direction"]
        out_dir    = os.path.join(save_dir, *safe_parts[:-1]) if len(safe_parts) > 1 else save_dir
        os.makedirs(out_dir, exist_ok=True)
        path  = os.path.join(out_dir, safe_parts[-1] + ".safetensors")
        dtype = torch.bfloat16 if save_as_bf16 else torch.float32
        save_file({"delta_embeds": direction["delta_embeds"].to(dtype),
                   "delta_mask":   direction["delta_mask"].float()},
                  path,
                  metadata={str(k): str(v) for k, v in direction.get("metadata", {}).items()})
        print(f"[EricQwenDir] saved '{os.path.relpath(path, save_dir)}' "
              f"({os.path.getsize(path)/1e6:.2f} MB, {'bf16' if save_as_bf16 else 'fp32'})")
        return (direction,)


class EricQwenDirectionLoad:
    """Load QWEN_DIRECTION. Dropdown refreshes every run."""

    CATEGORY = "Eric Qwen-Edit/Conditioning/Directions"
    FUNCTION = "load"
    RETURN_TYPES = ("QWEN_DIRECTION",)
    RETURN_NAMES = ("direction",)

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "direction_name": (EricQwenDirectionLoad._list_directions(), {}),
            },
        }

    @staticmethod
    def _list_directions() -> list:
        root  = _get_directions_dir()
        files = []
        for dirpath, _, filenames in os.walk(root):
            for f in sorted(filenames):
                if f.endswith(".safetensors"):
                    rel = os.path.relpath(os.path.join(dirpath, f), root)
                    files.append(rel.replace("\\", "/")[:-12])
        return sorted(files) if files else ["(none - save a direction first)"]

    @classmethod
    def IS_CHANGED(cls, direction_name):
        return float("nan")

    def load(self, direction_name: str) -> tuple:
        root = _get_directions_dir()
        path = os.path.join(root, direction_name.replace("/", os.sep) + ".safetensors")
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Not found: {path}")
        direction = _load_direction_file(path)
        sz_mb     = os.path.getsize(path) / 1e6
        meta      = direction["metadata"]
        norm_str  = (f" [norm→{meta.get('target_rms','?')}]"
                     if str(meta.get("normalized", "False")) == "True" else "")
        print(f"[EricQwenDir] loaded '{direction_name}' ({sz_mb:.2f} MB) | "
              f"seq={direction['delta_embeds'].shape[1]}, "
              f"rms={meta.get('rms','?')}, n_avg={meta.get('n_averaged','1')}{norm_str}, "
              f"target='{meta.get('target_prompt','?')[:50]}'")
        return (direction,)
