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

    if has_lokr:
        return "lokr"
    if has_loha:
        return "loha"
    if has_lora:
        return "lora"
    return "unknown"


def _load_lokr_adapter(pipe, state_dict: dict, adapter_name: str,
                       log_prefix: str = "[LoRA]") -> None:
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
        _load_lokr_adapter_direct(pipe, state_dict, adapter_name, log_prefix)


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
                              log_prefix: str = "[LoRA]") -> None:
    """Apply LoKR adapter by directly merging weights into the transformer.

    Computes ``kron(w1, w2) * (alpha / r)`` (or ``kron(w1, w2)`` when
    alpha equals r) and adds the delta to each target parameter.

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

        # Compute scaling factor
        alpha = params.get("alpha")
        alpha_val = alpha.item() if alpha is not None else 1.0
        # For full LoKR, r is inferred from the matrix dimensions.
        # ComfyUI convention: the rank dim is the smaller of w1/w2.
        # If alpha == r, scale = 1.0.
        r_val = min(w1.shape) if w1.ndim >= 2 else 1
        scale = alpha_val / r_val if r_val > 0 else 1.0

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
    }
    if not getattr(transformer, "_hf_peft_config_loaded", False):
        transformer._hf_peft_config_loaded = True

    print(f"{log_prefix} LoKR direct merge: applied={applied}, "
          f"skipped={skipped}")


def _load_loha_adapter(pipe, state_dict: dict, adapter_name: str,
                       log_prefix: str = "[LoRA]") -> None:
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
        _load_loha_adapter_direct(pipe, state_dict, adapter_name, log_prefix)


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
                              log_prefix: str = "[LoRA]") -> None:
    """Apply LoHa adapter by directly merging weights into the transformer.

    LoHa delta = ``(w1_a @ w1_b) * (w2_a @ w2_b) * (alpha / r)``.
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

        # Scaling
        alpha = params.get("alpha")
        alpha_val = alpha.item() if alpha is not None else 1.0
        r_val = w1_b.shape[0] if w1_b.ndim >= 2 else 1
        scale = alpha_val / r_val if r_val > 0 else 1.0

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
    }
    if not getattr(transformer, "_hf_peft_config_loaded", False):
        transformer._hf_peft_config_loaded = True

    print(f"{log_prefix} LoHa direct merge: applied={applied}, "
          f"skipped={skipped}")


def load_lora_with_key_fix(pipe, lora_path: str, adapter_name: str,
                          log_prefix: str = "[LoRA]") -> None:
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
        # Catch errors that indicate a format we can handle:
        #  - "Target modules ... not found" → key prefix issue (standard LoRA)
        #  - "state_dict should be empty" → non-LoRA format (LoKR / LoHa)
        #  - "lora_A" / "lora_B" in error → shape or format mismatch
        is_fixable = (
            ("Target modules" in err and "not found" in err)
            or "state_dict" in err.lower()
            or "lora_A" in err
            or "lora_B" in err
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
        _load_lokr_adapter(pipe, state_dict, adapter_name, log_prefix)
    elif adapter_type == "loha":
        _load_loha_adapter(pipe, state_dict, adapter_name, log_prefix)
    elif adapter_type == "lora":
        # Standard LoRA with key prefix issues — re-load via pipeline
        pipe.load_lora_weights(state_dict, adapter_name=adapter_name)
        print(f"{log_prefix} LoRA loaded successfully with key normalization")
    else:
        raise ValueError(
            f"{log_prefix} Unrecognised adapter format.  First 5 keys: "
            f"{list(state_dict.keys())[:5]}"
        )


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
        
        print(f"[EricQwenEdit] Applying LoRA: {lora_filename}")
        print(f"[EricQwenEdit] Path: {lora_path}")
        print(f"[EricQwenEdit] Weight: {weight}")
        
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
                pipe.set_adapters([adapter_name], adapter_weights=[weight])
                print(f"[EricQwenEdit] LoRA already loaded, updated weight: {adapter_name} -> {weight}")
            else:
                # Load fresh (with automatic key normalization fallback)
                load_lora_with_key_fix(pipe, lora_path, adapter_name,
                                      log_prefix="[EricQwenEdit]")
                pipe.set_adapters([adapter_name], adapter_weights=[weight])
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
