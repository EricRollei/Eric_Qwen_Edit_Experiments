# Copyright (c) 2026 Eric Hiss. All rights reserved.
# Licensed under the terms in LICENSE.txt (CC BY-NC 4.0 / Commercial dual license).
# https://github.com/EricRollei/Eric_Qwen_Edit_Experiments
"""
Eric Qwen-Edit LoRA Node
Apply LoRA weights to Qwen-Image-Edit pipeline.

Supports standard LoRA, LoKR (Kronecker), and LoHa (Hadamard) adapter formats.

Model Credits:
- Qwen-Image-Edit developed by Qwen Team (Alibaba)
- https://github.com/QwenLM/Qwen-Image

Author: Eric Hiss (GitHub: EricRollei)
"""

import os
import re
import folder_paths
import torch
from typing import Tuple, List


def get_lora_list() -> List[str]:
    """Get list of available LoRA files from ComfyUI's loras folder."""
    loras = []
    
    # Get ComfyUI's standard loras folder
    lora_paths = folder_paths.get_folder_paths("loras")
    
    for lora_dir in lora_paths:
        if not os.path.isdir(lora_dir):
            continue
        
        # Walk through directory and subdirectories
        for root, dirs, files in os.walk(lora_dir):
            for file in files:
                if file.endswith(('.safetensors', '.bin', '.pt', '.pth')):
                    # Get relative path from lora_dir
                    rel_path = os.path.relpath(os.path.join(root, file), lora_dir)
                    loras.append(rel_path)
    
    # Sort alphabetically
    loras.sort()
    
    # Add "none" option at the start
    return ["none"] + loras


def get_lora_full_path(lora_name: str) -> str:
    """Get the full path for a LoRA file."""
    if lora_name == "none" or not lora_name:
        return None
    
    lora_paths = folder_paths.get_folder_paths("loras")
    
    for lora_dir in lora_paths:
        full_path = os.path.join(lora_dir, lora_name)
        if os.path.exists(full_path):
            return full_path
    
    return None


# ═══════════════════════════════════════════════════════════════════════
#  Adapter format helpers
# ═══════════════════════════════════════════════════════════════════════

def _load_state_dict(path: str) -> dict:
    """Load a state dict from a safetensors, .bin, .pt, or .pth file."""
    if path.endswith(".safetensors"):
        from safetensors.torch import load_file
        return load_file(path)
    return torch.load(path, map_location="cpu", weights_only=True)


def _adapter_module_path(key: str) -> str:
    """Extract the module path from an adapter state-dict key.

    Strips adapter-specific suffixes like ``.lokr_w1``, ``.lora_A.weight``,
    ``.alpha``, etc. so we get the bare module path that should correspond
    to a ``nn.Module`` inside the transformer.
    """
    _SUFFIXES = (
        ".lokr_w1", ".lokr_w2", ".lokr_t2",
        ".lora_A.weight", ".lora_A.default.weight",
        ".lora_B.weight", ".lora_B.default.weight",
        ".lora_down.weight", ".lora_up.weight",
        ".hada_w1_a", ".hada_w1_b", ".hada_w2_a", ".hada_w2_b",
        ".alpha", ".diff", ".diff_b",
    )
    for sfx in _SUFFIXES:
        idx = key.find(sfx)
        if idx >= 0:
            return key[:idx]
    # Fallback: strip last dotted component
    return key.rsplit(".", 1)[0] if "." in key else key


def _normalize_keys(state_dict: dict, model=None) -> dict:
    """Strip component prefixes from state-dict keys.

    Many non-diffusers LoRA files bake in a component prefix such as
    ``transformer.``, ``diffusion_model.``, or ``model.diffusion_model.``.
    Diffusers expects keys relative to the transformer module itself.

    If *model* is provided (the ``nn.Module`` the adapter will target),
    the function **auto-detects** which prefix to strip by comparing the
    adapter module paths from the state dict against the model's
    ``named_modules()``.  This makes it robust regardless of training
    tool or checkpoint convention.
    """
    # Filter out text-encoder keys first
    filtered = {}
    for k, v in state_dict.items():
        if k.startswith(("text_encoder.", "text_encoder_2.")):
            continue
        filtered[k] = v

    if not filtered:
        return filtered

    # ── Smart mode: match against actual model modules ───────────────
    if model is not None:
        model_names = {name for name, _ in model.named_modules() if name}
        sd_module_paths = {_adapter_module_path(k) for k in filtered}

        # Already matches — no stripping needed
        if sd_module_paths & model_names:
            return filtered

        # Try known prefixes (most common first)
        _KNOWN = [
            "transformer.",
            "diffusion_model.",
            "model.diffusion_model.",
            "model.",
        ]
        for pfx in _KNOWN:
            stripped = {p[len(pfx):] for p in sd_module_paths
                        if p.startswith(pfx)}
            if stripped & model_names:
                cleaned = {}
                for k, v in filtered.items():
                    cleaned[k[len(pfx):] if k.startswith(pfx) else k] = v
                return cleaned

        # Auto-detect: find ANY prefix that produces matches
        for sd_path in list(sd_module_paths)[:20]:
            for m_path in list(model_names)[:50]:
                if sd_path.endswith(m_path) and len(sd_path) > len(m_path):
                    pfx = sd_path[: -len(m_path)]
                    hit = sum(1 for p in sd_module_paths
                              if p.startswith(pfx)
                              and p[len(pfx):] in model_names)
                    if hit > len(sd_module_paths) * 0.3:
                        print(f"[LoRA] Auto-detected key prefix: '{pfx}'")
                        cleaned = {}
                        for k, v in filtered.items():
                            cleaned[k[len(pfx):]
                                    if k.startswith(pfx)
                                    else k] = v
                        return cleaned

        # Nothing matched — warn and return as-is
        print("[LoRA] WARNING: Could not match state-dict keys to model "
              "modules after trying known prefixes.")
        print(f"[LoRA]   state-dict paths (sample): "
              f"{sorted(sd_module_paths)[:3]}")
        print(f"[LoRA]   model module paths (sample): "
              f"{sorted(model_names)[:5]}")
        return filtered

    # ── Simple mode (no model): strip 'transformer.' if present ──────
    prefix = "transformer."
    if any(k.startswith(prefix) for k in filtered):
        cleaned = {}
        for k, v in filtered.items():
            cleaned[k[len(prefix):] if k.startswith(prefix) else k] = v
        return cleaned
    return filtered


def _detect_adapter_type(state_dict: dict) -> str:
    """Detect adapter format from state dict keys.

    Returns one of: ``"lokr"``, ``"loha"``, ``"lora"``, ``"unknown"``.
    """
    keys = set(state_dict.keys())
    has_lokr = any("lokr_w1" in k or "lokr_w2" in k for k in keys)
    has_loha = any("hada_w1_a" in k or "hada_w2_a" in k for k in keys)
    has_lora = any("lora_A" in k or "lora_B" in k for k in keys)
    has_lora_alt = any("lora_down" in k or "lora_up" in k for k in keys)

    if has_lokr:
        return "lokr"
    if has_loha:
        return "loha"
    if has_lora or has_lora_alt:
        return "lora"
    return "unknown"


def _load_lokr_adapter(pipe, state_dict: dict, adapter_name: str,
                       log_prefix: str = "[LoRA]",
                       weight: float = 1.0) -> None:
    """Inject a LoKR (Kronecker) adapter via PEFT.

    Diffusers' ``load_lora_weights`` only understands standard LoRA format.
    For LoKR we bypass the pipeline and use PEFT's ``inject_adapter_in_model``
    + ``set_peft_model_state_dict`` directly on the transformer.

    If PEFT injection fails (e.g. due to unresolvable key mismatch), falls
    back to **direct weight merging**: computes ``kron(w1, w2) * scale`` and
    adds the deltas straight to the transformer parameters.

    Scaling convention: for full (non-decomposed) LoKR the effective delta is
    ``kron(w1, w2) * (alpha / r)``.  We set ``alpha = r`` in the config so
    that the base scaling is 1.0 (matching ComfyUI / LyCORIS convention).
    ``set_adapters()`` then multiplies by the user-supplied weight.
    """
    try:
        _load_lokr_adapter_peft(pipe, state_dict, adapter_name, log_prefix)
    except (ValueError, RuntimeError) as peft_err:
        print(f"{log_prefix} PEFT injection failed: {peft_err}")
        print(f"{log_prefix} Falling back to direct weight merge...")
        _load_lokr_adapter_direct(pipe, state_dict, adapter_name, log_prefix,
                                  weight=weight)


def _load_lokr_adapter_peft(pipe, state_dict: dict, adapter_name: str,
                            log_prefix: str = "[LoRA]") -> None:
    """Inject LoKR adapter via PEFT's ``inject_adapter_in_model``."""
    from peft import LoKrConfig, inject_adapter_in_model, set_peft_model_state_dict

    transformer = pipe.transformer

    # Use a very large r so that PEFT creates full (non-decomposed) w1/w2
    # matrices.  alpha = r gives unit scaling (1.0).
    cfg_r = 100000
    config = LoKrConfig(
        r=cfg_r,
        alpha=float(cfg_r),
        decompose_both=False,
        decompose_factor=-1,       # default factorization (near square-root)
        target_modules=["_dummy"],  # will be overridden by state_dict keys
    )

    # Inject adapter layers (structure only, random weights)
    inject_adapter_in_model(config, transformer,
                            adapter_name=adapter_name,
                            state_dict=state_dict)

    # Load actual trained weights from state dict
    incompatible = set_peft_model_state_dict(transformer, state_dict,
                                             adapter_name=adapter_name)

    # Mark PEFT as active so the pipeline's set_adapters() works
    if not getattr(transformer, "_hf_peft_config_loaded", False):
        transformer._hf_peft_config_loaded = True

    # Log incompatible keys (alpha entries are expected here)
    if incompatible:
        unexpected = getattr(incompatible, "unexpected_keys", [])
        missing = getattr(incompatible, "missing_keys", [])
        if unexpected:
            # Filter out alpha keys which are expected to be left over
            non_alpha = [k for k in unexpected if not k.endswith(".alpha")]
            if non_alpha:
                print(f"{log_prefix} LoKR unexpected keys: {non_alpha[:5]}...")
        if missing:
            print(f"{log_prefix} LoKR missing keys: {missing[:5]}...")

    print(f"{log_prefix} LoKR adapter loaded successfully via PEFT injection")


def _load_lokr_adapter_direct(pipe, state_dict: dict, adapter_name: str,
                              log_prefix: str = "[LoRA]",
                              weight: float = 1.0) -> None:
    """Apply LoKR adapter by directly merging weights into the transformer.

    Computes ``kron(w1, w2) * (alpha / r) * weight`` and adds the delta to
    each target parameter.  When *alpha* is absent from the checkpoint the
    scale defaults to ``1.0`` (weights are assumed pre-scaled).

    This is the same approach ComfyUI's native weight patcher uses.
    The adapter is registered in ``peft_config`` so ``set_adapters()``
    can still find it, but per-stage weight adjustment is limited to a
    single re-merge at changed weight.
    """
    import math, re

    transformer = pipe.transformer
    model_sd = dict(transformer.named_parameters())

    # Group state dict keys by module path
    modules: dict[str, dict] = {}  # module_path -> {"lokr_w1": ..., ...}
    for k, v in state_dict.items():
        path = _adapter_module_path(k)
        # Extract the param name (e.g., "lokr_w1", "alpha")
        param_name = k[len(path) + 1:]  # +1 for the dot
        modules.setdefault(path, {})[param_name] = v

    applied = 0
    skipped = 0
    for mod_path, params in modules.items():
        w1 = params.get("lokr_w1")
        w2 = params.get("lokr_w2")
        if w1 is None or w2 is None:
            skipped += 1
            continue

        # Compute scaling factor.
        # When alpha IS stored, use alpha / r (LyCORIS convention).
        # When alpha is NOT stored, assume weights are pre-scaled → 1.0.
        alpha = params.get("alpha")
        if alpha is not None:
            alpha_val = alpha.item()
            r_val = min(w1.shape) if w1.ndim >= 2 else 1
            scale = (alpha_val / r_val) * weight if r_val > 0 else weight
        else:
            scale = weight

        # Compute delta = kron(w1, w2) * scale
        w1f = w1.float()
        w2f = w2.float()
        delta = torch.kron(w1f, w2f) * scale

        # Match to model param.  LoKR targets ".weight" by default.
        target_key = mod_path + ".weight"
        if target_key not in model_sd:
            # Try without .weight suffix
            target_key = mod_path
        if target_key not in model_sd:
            skipped += 1
            continue

        param = model_sd[target_key]
        if delta.shape != param.shape:
            # Try reshaping
            try:
                delta = delta.reshape(param.shape)
            except RuntimeError:
                print(f"{log_prefix} Shape mismatch for {mod_path}: "
                      f"delta {delta.shape} vs param {param.shape}, skipping")
                skipped += 1
                continue

        # Store original weights for potential unloading
        backup_key = f"_lokr_backup_{adapter_name}"
        if not hasattr(transformer, backup_key):
            setattr(transformer, backup_key, {})
        backup = getattr(transformer, backup_key)
        if target_key not in backup:
            backup[target_key] = param.data.clone()

        # Apply delta
        param.data.add_(delta.to(dtype=param.dtype, device=param.device))
        applied += 1

    # Register in peft_config for set_adapters() discovery
    if not hasattr(transformer, "peft_config"):
        transformer.peft_config = {}
    # Minimal config marker so get_list_adapters() finds this adapter
    transformer.peft_config[adapter_name] = {
        "_type": "lokr_direct",
        "_applied_modules": applied,
        "_weight": weight,
    }
    if not getattr(transformer, "_hf_peft_config_loaded", False):
        transformer._hf_peft_config_loaded = True

    print(f"{log_prefix} LoKR direct merge (weight={weight}): "
          f"applied={applied}, skipped={skipped}")
    if skipped > applied:
        sample_keys = list(state_dict.keys())[:5]
        sample_modules = sorted(model_sd.keys())[:5]
        print(f"{log_prefix} WARNING: Many modules skipped. "
              f"State-dict keys (sample): {sample_keys}")
        print(f"{log_prefix}   Model params (sample): {sample_modules}")


def _load_loha_adapter(pipe, state_dict: dict, adapter_name: str,
                       log_prefix: str = "[LoRA]",
                       weight: float = 1.0) -> None:
    """Inject a LoHa (Hadamard) adapter via PEFT.

    LoHa always uses decomposed weights (w1_a/w1_b, w2_a/w2_b), so the
    rank ``r`` is directly available from the weight shapes.

    Falls back to direct weight merge if PEFT injection fails.
    """
    try:
        _load_loha_adapter_peft(pipe, state_dict, adapter_name, log_prefix)
    except (ValueError, RuntimeError) as peft_err:
        print(f"{log_prefix} PEFT injection failed for LoHa: {peft_err}")
        print(f"{log_prefix} Falling back to direct weight merge...")
        _load_loha_adapter_direct(pipe, state_dict, adapter_name, log_prefix,
                                  weight=weight)


def _load_loha_adapter_peft(pipe, state_dict: dict, adapter_name: str,
                            log_prefix: str = "[LoRA]") -> None:
    """Inject LoHa adapter via PEFT's ``inject_adapter_in_model``."""
    from peft import LoHaConfig, inject_adapter_in_model, set_peft_model_state_dict

    transformer = pipe.transformer

    # Determine rank from the first w1_a tensor shape
    r_val = None
    alpha_val = None
    for key, val in state_dict.items():
        if ".hada_w1_a" in key and val.ndim >= 2:
            r_val = val.shape[1]  # (out_dim, rank)
            break
        if ".hada_w1_b" in key and val.ndim >= 2:
            r_val = val.shape[0]  # (rank, in_dim)
            break
    if r_val is None:
        r_val = 8  # fallback default

    # Try to read alpha from state dict
    for key, val in state_dict.items():
        if key.endswith(".alpha") and val.numel() == 1:
            alpha_val = val.item()
            break
    if alpha_val is None:
        alpha_val = float(r_val)

    config = LoHaConfig(
        r=r_val,
        alpha=alpha_val,
        target_modules=["_dummy"],  # overridden by state_dict keys
    )

    inject_adapter_in_model(config, transformer,
                            adapter_name=adapter_name,
                            state_dict=state_dict)
    incompatible = set_peft_model_state_dict(transformer, state_dict,
                                             adapter_name=adapter_name)

    if not getattr(transformer, "_hf_peft_config_loaded", False):
        transformer._hf_peft_config_loaded = True

    if incompatible:
        unexpected = getattr(incompatible, "unexpected_keys", [])
        non_alpha = [k for k in unexpected if not k.endswith(".alpha")]
        if non_alpha:
            print(f"{log_prefix} LoHa unexpected keys: {non_alpha[:5]}...")

    print(f"{log_prefix} LoHa adapter loaded successfully via PEFT injection")


def _load_loha_adapter_direct(pipe, state_dict: dict, adapter_name: str,
                              log_prefix: str = "[LoRA]",
                              weight: float = 1.0) -> None:
    """Apply LoHa adapter by directly merging weights into the transformer.

    LoHa delta = ``(w1_a @ w1_b) * (w2_a @ w2_b) * (alpha / r) * weight``.
    When *alpha* is absent, scale defaults to ``1.0``.
    """
    transformer = pipe.transformer
    model_sd = dict(transformer.named_parameters())

    # Group state dict keys by module path
    modules: dict[str, dict] = {}
    for k, v in state_dict.items():
        path = _adapter_module_path(k)
        param_name = k[len(path) + 1:]
        modules.setdefault(path, {})[param_name] = v

    applied = 0
    skipped = 0
    for mod_path, params in modules.items():
        w1_a = params.get("hada_w1_a")
        w1_b = params.get("hada_w1_b")
        w2_a = params.get("hada_w2_a")
        w2_b = params.get("hada_w2_b")
        if w1_a is None or w1_b is None or w2_a is None or w2_b is None:
            skipped += 1
            continue

        # Scaling: alpha / r when alpha is present, else 1.0
        alpha = params.get("alpha")
        if alpha is not None:
            alpha_val = alpha.item()
            r_val = w1_b.shape[0] if w1_b.ndim >= 2 else 1
            scale = (alpha_val / r_val) * weight if r_val > 0 else weight
        else:
            scale = weight

        # delta = (w1_a @ w1_b) * (w2_a @ w2_b) * scale
        w1 = (w1_a.float() @ w1_b.float())
        w2 = (w2_a.float() @ w2_b.float())
        delta = w1 * w2 * scale

        target_key = mod_path + ".weight"
        if target_key not in model_sd:
            target_key = mod_path
        if target_key not in model_sd:
            skipped += 1
            continue

        param = model_sd[target_key]
        if delta.shape != param.shape:
            try:
                delta = delta.reshape(param.shape)
            except RuntimeError:
                print(f"{log_prefix} LoHa shape mismatch for {mod_path}: "
                      f"delta {delta.shape} vs param {param.shape}, skipping")
                skipped += 1
                continue

        backup_key = f"_loha_backup_{adapter_name}"
        if not hasattr(transformer, backup_key):
            setattr(transformer, backup_key, {})
        backup = getattr(transformer, backup_key)
        if target_key not in backup:
            backup[target_key] = param.data.clone()

        param.data.add_(delta.to(dtype=param.dtype, device=param.device))
        applied += 1

    if not hasattr(transformer, "peft_config"):
        transformer.peft_config = {}
    transformer.peft_config[adapter_name] = {
        "_type": "loha_direct",
        "_applied_modules": applied,
        "_weight": weight,
    }
    if not getattr(transformer, "_hf_peft_config_loaded", False):
        transformer._hf_peft_config_loaded = True

    print(f"{log_prefix} LoHa direct merge (weight={weight}): "
          f"applied={applied}, skipped={skipped}")
    if skipped > applied:
        sample_keys = list(state_dict.keys())[:5]
        sample_modules = sorted(model_sd.keys())[:5]
        print(f"{log_prefix} WARNING: Many modules skipped. "
              f"State-dict keys (sample): {sample_keys}")
        print(f"{log_prefix}   Model params (sample): {sample_modules}")


def _rename_lora_down_up(state_dict: dict) -> dict:
    """Rename ``lora_down`` / ``lora_up`` keys to ``lora_A`` / ``lora_B``.

    Some training tools (kohya_ss, diffusers-native) use ``lora_down``/
    ``lora_up`` instead of the PEFT-standard ``lora_A``/``lora_B``.
    This normalises them so the rest of the loading pipeline only needs
    to handle one convention.
    """
    if not any("lora_down" in k or "lora_up" in k for k in state_dict):
        return state_dict

    renamed = {}
    count = 0
    for k, v in state_dict.items():
        new_k = k.replace(".lora_down.weight", ".lora_A.weight") \
                 .replace(".lora_up.weight", ".lora_B.weight")
        if new_k != k:
            count += 1
        renamed[new_k] = v

    if count:
        print(f"[LoRA] Renamed {count} lora_down/lora_up keys "
              f"to lora_A/lora_B")
    return renamed


def _load_lora_adapter(pipe, state_dict: dict, adapter_name: str,
                       log_prefix: str = "[LoRA]",
                       weight: float = 1.0) -> None:
    """Load a standard LoRA adapter with multi-level fallback.

    Handles both ``lora_A``/``lora_B`` and ``lora_down``/``lora_up``
    naming conventions (the latter is normalised to the former).

    1. ``pipe.load_lora_weights(state_dict)`` — best integration with
       diffusers' adapter management.
    2. Direct PEFT ``inject_adapter_in_model`` on the transformer —
       still supports ``set_adapters()`` for weight control.
    3. Direct weight merge (B @ A * scale) — last resort, weight is
       baked in at load time.
    """
    # Normalise lora_down/lora_up → lora_A/lora_B
    state_dict = _rename_lora_down_up(state_dict)
    # ── Try 1: Pipeline LoRA loading with normalised keys ────────────
    try:
        pipe.load_lora_weights(state_dict, adapter_name=adapter_name)
        print(f"{log_prefix} LoRA loaded via pipeline with normalised keys")
        return
    except (ValueError, RuntimeError) as e:
        print(f"{log_prefix} Pipeline LoRA load with normalised keys "
              f"failed: {str(e)[:120]}")

    # ── Try 2: Direct PEFT injection on the transformer ──────────────
    try:
        _load_lora_adapter_peft(pipe, state_dict, adapter_name, log_prefix)
        return
    except (ValueError, RuntimeError) as e:
        print(f"{log_prefix} Direct PEFT LoRA injection failed: "
              f"{str(e)[:120]}")

    # ── Try 3: Direct weight merge (last resort) ─────────────────────
    print(f"{log_prefix} Falling back to direct LoRA weight merge...")
    _load_lora_adapter_direct(pipe, state_dict, adapter_name, log_prefix,
                              weight=weight)


def _load_lora_adapter_peft(pipe, state_dict: dict, adapter_name: str,
                            log_prefix: str = "[LoRA]") -> None:
    """Inject standard LoRA adapter directly via PEFT on the transformer."""
    from peft import LoraConfig, inject_adapter_in_model, set_peft_model_state_dict

    transformer = pipe.transformer

    # Normalise lora_down/lora_up → lora_A/lora_B before PEFT sees them
    state_dict = _rename_lora_down_up(state_dict)

    # Determine rank from first lora_A tensor
    r_val = None
    alpha_val = None
    for key, val in state_dict.items():
        if "lora_A" in key and val.ndim >= 2:
            r_val = val.shape[0]
            break
    if r_val is None:
        r_val = 64  # common default

    for key, val in state_dict.items():
        if key.endswith(".alpha") and val.numel() == 1:
            alpha_val = val.item()
            break
    if alpha_val is None:
        alpha_val = float(r_val)

    config = LoraConfig(
        r=r_val,
        lora_alpha=alpha_val,
        target_modules=["_dummy"],  # overridden by state_dict keys
    )

    inject_adapter_in_model(config, transformer,
                            adapter_name=adapter_name,
                            state_dict=state_dict)
    incompatible = set_peft_model_state_dict(transformer, state_dict,
                                             adapter_name=adapter_name)

    if not getattr(transformer, "_hf_peft_config_loaded", False):
        transformer._hf_peft_config_loaded = True

    if incompatible:
        unexpected = getattr(incompatible, "unexpected_keys", [])
        non_alpha = [k for k in unexpected if not k.endswith(".alpha")]
        if non_alpha:
            print(f"{log_prefix} LoRA unexpected keys: {non_alpha[:5]}...")
        missing = getattr(incompatible, "missing_keys", [])
        if missing:
            print(f"{log_prefix} LoRA missing keys: {missing[:5]}...")

    print(f"{log_prefix} LoRA loaded via direct PEFT injection")


def _load_lora_adapter_direct(pipe, state_dict: dict, adapter_name: str,
                              log_prefix: str = "[LoRA]",
                              weight: float = 1.0) -> None:
    """Apply standard LoRA by directly merging B @ A * scale into weights.

    The user *weight* is baked in at merge time.  ``set_adapters()`` will
    not be able to change it afterwards (a limitation of direct merge).
    """
    transformer = pipe.transformer
    model_sd = dict(transformer.named_parameters())

    modules: dict[str, dict] = {}
    for k, v in state_dict.items():
        path = _adapter_module_path(k)
        param_name = k[len(path) + 1:]
        modules.setdefault(path, {})[param_name] = v

    applied = 0
    skipped = 0
    for mod_path, params in modules.items():
        lora_A = params.get("lora_A.weight") or params.get(
            "lora_A.default.weight")
        lora_B = params.get("lora_B.weight") or params.get(
            "lora_B.default.weight")
        if lora_A is None or lora_B is None:
            skipped += 1
            continue

        # Scaling: delta = B @ A * (alpha / r) * weight
        alpha = params.get("alpha")
        if alpha is not None:
            alpha_val = alpha.item()
            r_val = lora_A.shape[0]
            scale = (alpha_val / r_val) * weight if r_val > 0 else weight
        else:
            scale = weight

        delta = (lora_B.float() @ lora_A.float()) * scale

        target_key = mod_path + ".weight"
        if target_key not in model_sd:
            target_key = mod_path
        if target_key not in model_sd:
            skipped += 1
            continue

        param = model_sd[target_key]
        if delta.shape != param.shape:
            try:
                delta = delta.reshape(param.shape)
            except RuntimeError:
                print(f"{log_prefix} LoRA shape mismatch for {mod_path}: "
                      f"delta {delta.shape} vs param {param.shape}, skipping")
                skipped += 1
                continue

        # Backup for unloading
        backup_key = f"_lora_backup_{adapter_name}"
        if not hasattr(transformer, backup_key):
            setattr(transformer, backup_key, {})
        backup = getattr(transformer, backup_key)
        if target_key not in backup:
            backup[target_key] = param.data.clone()

        param.data.add_(delta.to(dtype=param.dtype, device=param.device))
        applied += 1

    # Register in peft_config for adapter discovery
    if not hasattr(transformer, "peft_config"):
        transformer.peft_config = {}
    transformer.peft_config[adapter_name] = {
        "_type": "lora_direct",
        "_applied_modules": applied,
        "_weight": weight,
    }
    if not getattr(transformer, "_hf_peft_config_loaded", False):
        transformer._hf_peft_config_loaded = True

    print(f"{log_prefix} LoRA direct merge (weight={weight}): "
          f"applied={applied}, skipped={skipped}")
    if skipped > applied:
        sample_keys = list(state_dict.keys())[:5]
        sample_modules = sorted(model_sd.keys())[:5]
        print(f"{log_prefix} WARNING: Many modules skipped. "
              f"State-dict keys (sample): {sample_keys}")
        print(f"{log_prefix}   Model params (sample): {sample_modules}")


def _set_adapters_safe(pipe, adapter_name: str, weight: float,
                       log_prefix: str = "[LoRA]") -> None:
    """Call ``pipe.set_adapters()`` with graceful handling for direct-merge
    adapters that don't have PEFT tuner layers.

    For PEFT-injected adapters ``set_adapters()`` adjusts the scaling.
    For direct-merge adapters the weight is already baked into the model
    parameters at load time, so ``set_adapters()`` is a no-op and we just
    log a note.
    """
    # Check if this is a direct-merge adapter
    transformer = getattr(pipe, "transformer", None)
    if transformer is not None:
        peft_cfg = getattr(transformer, "peft_config", {})
        cfg = peft_cfg.get(adapter_name, {})
        if isinstance(cfg, dict) and cfg.get("_type", "").endswith("_direct"):
            if abs(weight - cfg.get("_weight", 1.0)) > 1e-6:
                print(f"{log_prefix} NOTE: '{adapter_name}' was loaded via "
                      f"direct weight merge (weight={cfg.get('_weight', 1.0)})."
                      f" Changing weight requires reloading the LoRA.")
            return

    try:
        pipe.set_adapters([adapter_name], adapter_weights=[weight])
    except Exception as e:
        print(f"{log_prefix} set_adapters note: {str(e)[:100]}  "
              f"(adapter may have been loaded via direct merge)")


def load_lora_with_key_fix(pipe, lora_path: str, adapter_name: str,
                          log_prefix: str = "[LoRA]",
                          weight: float = 1.0) -> None:
    """Load a LoRA / LoKR / LoHa adapter with automatic format detection.

    1. **Fast path** — tries ``pipe.load_lora_weights()`` (handles well-
       formatted standard LoRA files).
    2. **Fallback** — loads the state dict manually, normalises keys
       (strips ``transformer.`` prefix), and detects the adapter type:

       * **Standard LoRA** (``lora_A`` / ``lora_B`` keys) — re-loads
         through the pipeline with cleaned keys.
       * **LoKR** (``lokr_w1`` / ``lokr_w2`` keys) — injects via PEFT's
         ``inject_adapter_in_model`` + ``set_peft_model_state_dict``.
       * **LoHa** (``hada_w1_a`` / ``hada_w2_a`` keys) — same approach
         as LoKR but with ``LoHaConfig``.
    """
    # ── Fast path: try standard loading ──────────────────────────────
    try:
        pipe.load_lora_weights(lora_path, adapter_name=adapter_name)
        return
    except (ValueError, RuntimeError) as e:
        err = str(e)
        # Catch errors that indicate a format / key-mapping issue we can
        # potentially work around by manual loading + key normalisation.
        is_fixable = (
            ("Target modules" in err and "not found" in err)
            or "No modules were targeted" in err
            or "state_dict" in err.lower()
            or "lora_A" in err
            or "lora_B" in err
            or "lokr" in err.lower()
            or "loha" in err.lower()
            or "hada_" in err
        )
        if not is_fixable:
            raise
        print(f"{log_prefix} Standard load failed, attempting format "
              f"detection...  ({err[:120]})")

    # ── Fallback: manual load + format detection ─────────────────────
    state_dict = _load_state_dict(lora_path)
    transformer = getattr(pipe, "transformer", None)
    state_dict = _normalize_keys(state_dict, model=transformer)

    adapter_type = _detect_adapter_type(state_dict)
    print(f"{log_prefix} Detected adapter format: {adapter_type}")

    if adapter_type == "lokr":
        _load_lokr_adapter(pipe, state_dict, adapter_name, log_prefix,
                           weight=weight)
    elif adapter_type == "loha":
        _load_loha_adapter(pipe, state_dict, adapter_name, log_prefix,
                           weight=weight)
    elif adapter_type == "lora":
        # Standard LoRA with key prefix issues — multi-level fallback
        _load_lora_adapter(pipe, state_dict, adapter_name, log_prefix,
                           weight=weight)
    else:
        raise ValueError(
            f"{log_prefix} Unrecognised adapter format.  First 5 keys: "
            f"{list(state_dict.keys())[:5]}"
        )


def _diagnose_lora_compatibility(lora_path: str, transformer) -> str:
    """Analyze LoRA compatibility against a transformer without applying it.

    Returns a multi-line diagnostic report string.
    """
    lines = []
    filename = os.path.basename(lora_path)
    sep = "=" * 60
    lines.append(sep)
    lines.append(f"LoRA Diagnostic: {filename}")
    lines.append(sep)

    # ── Load raw state dict ───────────────────────────────────────────
    try:
        sd = _load_state_dict(lora_path)
    except Exception as exc:
        return f"{sep}\nLoRA Diagnostic: {filename}\n{sep}\nERROR: Could not load file — {exc}\n{sep}"

    total_keys = len(sd)
    lines.append(f"Total tensor keys: {total_keys}")

    # Text-encoder keys (we don't apply these to the transformer)
    te_keys = [k for k in sd if k.startswith(("text_encoder.", "text_encoder_2."))]
    if te_keys:
        lines.append(f"Text encoder keys (not applied): {len(te_keys)}")

    # ── Adapter type detection ────────────────────────────────────────
    adapter_type = _detect_adapter_type(sd)
    lines.append(f"Adapter type: {adapter_type.upper()}")

    # ── Rank / alpha (LoRA only) ──────────────────────────────────────
    if adapter_type == "lora":
        sd_r = _rename_lora_down_up(sd)
        ranks, alphas = [], []
        for k, v in sd_r.items():
            if "lora_A" in k and v.ndim >= 2:
                ranks.append(v.shape[0])
            if k.endswith(".alpha") and v.numel() == 1:
                alphas.append(round(v.item(), 3))
        if ranks:
            unique_ranks = sorted(set(ranks))
            lines.append(f"Rank(s): {unique_ranks}")
            avg_rank = sum(ranks) / len(ranks)
            if avg_rank > 128:
                lines.append(f"  NOTE: High rank ({avg_rank:.0f}) — may cause instability")
        if alphas:
            unique_alphas = sorted(set(alphas))
            lines.append(f"Alpha(s): {unique_alphas}")
            if ranks:
                ratio = (sum(alphas) / len(alphas)) / (sum(ranks) / len(ranks))
                lines.append(f"  alpha/rank ratio: {ratio:.3f}  (1.0 = neutral)")
        if not ranks:
            lines.append("  WARNING: No lora_A tensors found — file may be malformed")
    elif adapter_type == "lokr":
        lines.append("Format: LoKR (Kronecker decomposition)")
    elif adapter_type == "loha":
        lines.append("Format: LoHa (Hadamard product)")
        for k, v in sd.items():
            if ".hada_w1_b" in k and v.ndim >= 2:
                lines.append(f"  Rank (from w1_b): {v.shape[0]}")
                break
    else:
        lines.append("WARNING: Unrecognised format — corrupt file or unsupported training tool.")
        lines.append(f"  First 5 raw keys: {list(sd.keys())[:5]}")

    # ── Key-prefix detection (before normalisation) ───────────────────
    raw_paths = {_adapter_module_path(k) for k in sd
                 if not k.startswith(("text_encoder.", "text_encoder_2."))}
    found_prefixes = [p for p in
                      ("transformer.", "diffusion_model.", "model.diffusion_model.", "model.", "unet.")
                      if any(rp.startswith(p) for rp in raw_paths)]
    if found_prefixes:
        lines.append(f"Key prefix(es) auto-stripped: {found_prefixes}")

    # ── Normalise keys and match against model ────────────────────────
    sd_norm = _normalize_keys(sd, model=transformer)
    sd_module_paths = {_adapter_module_path(k) for k in sd_norm}
    model_module_names = {name for name, _ in transformer.named_modules() if name}

    matched = sd_module_paths & model_module_names
    unmatched = sd_module_paths - model_module_names
    n_total = len(sd_module_paths)
    n_matched = len(matched)
    match_pct = n_matched / n_total * 100 if n_total > 0 else 0.0

    lines.append("")
    lines.append(f"Adapter modules: {n_total}")
    lines.append(f"Matched modules:  {n_matched} ({match_pct:.0f}%)")

    # ── Verdict ───────────────────────────────────────────────────────
    lines.append("")
    if match_pct >= 75:
        lines.append("VERDICT: COMPATIBLE")
        lines.append("Most adapter modules match the transformer architecture.")
        if n_total > n_matched:
            lines.append(f"  {n_total - n_matched} unmatched module(s) will be skipped.")
    elif match_pct >= 30:
        lines.append("VERDICT: PARTIAL MATCH  \u26a0")
        lines.append(f"Only {match_pct:.0f}% of modules matched.  May produce partial effects or artifacts.")
        lines.append("If output looks like noise, this LoRA targets a different architecture variant.")
        if unmatched:
            lines.append(f"  Unmatched sample: {sorted(unmatched)[:5]}")
    else:
        lines.append("VERDICT: INCOMPATIBLE  \u2717")
        lines.append(f"Only {match_pct:.0f}% of modules matched — this LoRA was almost certainly")
        lines.append("trained for a DIFFERENT architecture (e.g. SDXL, SD1.5, Flux).")
        lines.append("Applying it will likely produce corrupted output (pure noise / artifacts).")
        lines.append(f"  Sample LoRA modules:  {sorted(sd_module_paths)[:5]}")
        lines.append(f"  Sample model modules: {sorted(model_module_names)[:5]}")

    lines.append(sep)
    return "\n".join(lines)


class EricQwenEditApplyLoRA:
    """
    Apply a LoRA to the Qwen-Edit pipeline.
    
    This node loads LoRA weights onto the transformer. Multiple LoRA
    nodes can be chained to apply multiple LoRAs with different weights.
    
    Workflow example:
    [Load Model] -> [Apply LoRA (style)] -> [Apply LoRA (character)] -> [Edit Image]
    
    LoRA weight:
    - 1.0 = full strength
    - 0.5 = half strength
    - 0.0 = no effect (disabled)
    - >1.0 = amplified (may cause artifacts)
    
    LoRAs are loaded from ComfyUI's standard loras folder:
    ComfyUI/models/loras/
    """
    
    CATEGORY = "Eric Qwen-Edit"
    FUNCTION = "apply_lora"
    RETURN_TYPES = ("QWEN_EDIT_PIPELINE",)
    RETURN_NAMES = ("pipeline",)
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "pipeline": ("QWEN_EDIT_PIPELINE",),
                "lora_name": (get_lora_list(), {
                    "tooltip": "Select LoRA from ComfyUI/models/loras/"
                }),
                "weight": ("FLOAT", {
                    "default": 1.0,
                    "min": -2.0,
                    "max": 2.0,
                    "step": 0.05,
                    "tooltip": "LoRA weight strength (1.0 = full, 0.5 = half)"
                }),
            },
            "optional": {
                "lora_path_override": ("STRING", {
                    "default": "",
                    "tooltip": "Optional: Override with custom path (leave empty to use dropdown)"
                }),
                "keep_loaded": ("BOOLEAN", {
                    "default": True,
                    "tooltip": (
                        "Keep the LoRA adapter loaded between runs.\n"
                        "\u2022 True (default): load once, reuse on subsequent runs\n"
                        "  (faster; useful for final render settings).\n"
                        "\u2022 False: unload ALL adapters before loading this LoRA\n"
                        "  on every run. Ensures a clean state each generation;\n"
                        "  useful when testing or swapping LoRAs frequently."
                    )
                }),
            }
        }
    
    @classmethod
    def IS_CHANGED(cls, **kwargs):
        # Refresh LoRA list on each check
        return float("nan")
    
    def apply_lora(
        self,
        pipeline: dict,
        lora_name: str,
        weight: float = 1.0,
        lora_path_override: str = "",
        keep_loaded: bool = True,
    ) -> Tuple[dict]:
        """Apply LoRA to the pipeline."""
        pipe = pipeline["pipeline"]
        
        # Determine which path to use
        if lora_path_override and lora_path_override.strip():
            lora_path = lora_path_override.strip()
            if not os.path.exists(lora_path):
                raise ValueError(f"LoRA file not found: {lora_path}")
        else:
            if lora_name == "none" or not lora_name:
                print("[EricQwenEdit] LoRA: none selected, skipping")
                return (pipeline,)
            lora_path = get_lora_full_path(lora_name)
            if lora_path is None:
                raise ValueError(f"LoRA file not found: {lora_name}")
        
        lora_filename = os.path.basename(lora_path)
        adapter_name = os.path.splitext(lora_filename)[0]  # Remove extension for adapter name
        # PEFT/PyTorch ModuleDict keys cannot contain '.', so sanitize.
        adapter_name = re.sub(r"[.\s]", "_", adapter_name)
        
        print(f"[EricQwenEdit] Applying LoRA: {lora_filename}")
        print(f"[EricQwenEdit] Path: {lora_path}")
        print(f"[EricQwenEdit] Weight: {weight}, keep_loaded: {keep_loaded}")
        
        # If keep_loaded=False, unload everything first for a clean state each run
        if not keep_loaded:
            try:
                pipe.unload_lora_weights()
                pipeline.pop("applied_loras", None)
                print("[EricQwenEdit] Unloaded previous LoRAs (keep_loaded=False)")
            except Exception:
                pass
        
        # Check if this adapter is already loaded
        loaded_adapters = set()
        try:
            adapter_list = pipe.get_list_adapters()
            for component_adapters in adapter_list.values():
                loaded_adapters.update(component_adapters)
        except Exception:
            pass
        
        try:
            if adapter_name in loaded_adapters:
                # Already loaded — just update the weight
                _set_adapters_safe(pipe, adapter_name, weight,
                                   log_prefix="[EricQwenEdit]")
                print(f"[EricQwenEdit] LoRA already loaded, updated weight: {adapter_name} -> {weight}")
            else:
                # Load fresh (with automatic key normalization fallback)
                load_lora_with_key_fix(pipe, lora_path, adapter_name,
                                      log_prefix="[EricQwenEdit]",
                                      weight=weight)
                # For PEFT-injected adapters, apply weight via set_adapters.
                # For direct-merge adapters, weight is already baked in.
                _set_adapters_safe(pipe, adapter_name, weight,
                                   log_prefix="[EricQwenEdit]")
                print(f"[EricQwenEdit] LoRA applied successfully: {adapter_name}")
            
        except Exception as e:
            print(f"[EricQwenEdit] Error loading LoRA: {e}")
            raise
        
        # Track applied adapters in the pipeline dict
        if "applied_loras" not in pipeline:
            pipeline["applied_loras"] = {}
        pipeline["applied_loras"][adapter_name] = {
            "path": lora_path,
            "weight": weight,
            "filename": lora_filename,
        }
        
        # Return the same pipeline dict (modified in place)
        return (pipeline,)


class EricQwenEditUnloadLoRA:
    """
    Unload all LoRAs from the Qwen-Edit pipeline.
    
    Use this to reset the model to its base state before applying
    different LoRAs, or to free memory.
    """
    
    CATEGORY = "Eric Qwen-Edit"
    FUNCTION = "unload_lora"
    RETURN_TYPES = ("QWEN_EDIT_PIPELINE",)
    RETURN_NAMES = ("pipeline",)
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "pipeline": ("QWEN_EDIT_PIPELINE",),
            },
        }
    
    def unload_lora(self, pipeline: dict) -> Tuple[dict]:
        """Unload all LoRAs from the pipeline."""
        pipe = pipeline["pipeline"]
        
        try:
            pipe.unload_lora_weights()
            print("[EricQwenEdit] All LoRAs unloaded")
        except Exception as e:
            print(f"[EricQwenEdit] Note: {e}")
        
        return (pipeline,)


class EricQwenEditDiagnoseLoRA:
    """
    Diagnose LoRA compatibility with the Qwen-Edit pipeline.

    Loads the LoRA file and inspects its keys WITHOUT applying it,
    then prints a compatibility report to the console and returns it
    as a STRING.  Useful for understanding why a LoRA produces noise
    or fails to load.

    The report includes:
    - Adapter type (LoRA / LoKR / LoHa / unknown)
    - Rank and alpha values (for standard LoRA)
    - How many adapter modules match the transformer architecture
    - A COMPATIBLE / PARTIAL / INCOMPATIBLE verdict
    """

    CATEGORY = "Eric Qwen-Edit"
    FUNCTION = "diagnose"
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("report",)
    OUTPUT_NODE = True

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "pipeline": ("QWEN_EDIT_PIPELINE",),
                "lora_name": (get_lora_list(), {
                    "tooltip": "LoRA to analyze (inspected but NOT applied)"
                }),
            },
            "optional": {
                "lora_path_override": ("STRING", {
                    "default": "",
                    "tooltip": "Override with full path instead of dropdown"
                }),
            }
        }

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("nan")

    def diagnose(
        self,
        pipeline: dict,
        lora_name: str,
        lora_path_override: str = "",
    ) -> Tuple:
        pipe = pipeline["pipeline"]
        transformer = pipe.transformer

        if lora_path_override and lora_path_override.strip():
            lora_path = lora_path_override.strip()
        else:
            if lora_name == "none" or not lora_name:
                return ("No LoRA selected.",)
            lora_path = get_lora_full_path(lora_name)
            if lora_path is None:
                return (f"ERROR: LoRA file not found: {lora_name}",)

        if not os.path.exists(lora_path):
            return (f"ERROR: File not found: {lora_path}",)

        report = _diagnose_lora_compatibility(lora_path, transformer)
        print(report)
        return (report,)
