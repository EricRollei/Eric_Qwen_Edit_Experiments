# Copyright (c) 2026 Eric Hiss. All rights reserved.
# Licensed under the terms in LICENSE.txt (CC BY-NC 4.0 / Commercial dual license).
# https://github.com/EricRollei/Eric_Qwen_Edit_Experiments
"""
Eric Qwen-Image ControlNet Loader
Loads QwenImageControlNetModel for structure-guided generation.

Supports the InstantX/Qwen-Image-ControlNet-Union model which provides
canny, soft edge, depth, and pose control in a single checkpoint.

Also supports InstantX/Qwen-Image-ControlNet-Inpainting for mask-based
inpainting and outpainting (same QwenImageControlNetModel class).

Model Credits:
- InstantX/Qwen-Image-ControlNet-Union
- https://huggingface.co/InstantX/Qwen-Image-ControlNet-Union
- InstantX/Qwen-Image-ControlNet-Inpainting
- https://huggingface.co/InstantX/Qwen-Image-ControlNet-Inpainting

Author: Eric Hiss (GitHub: EricRollei)
"""

import gc
import os
import torch
from typing import Tuple


# ── ControlNet model cache ──────────────────────────────────────────────
_CONTROLNET_CACHE = {
    "model": None,
    "model_path": None,
    "cache_key": None,
}


def get_controlnet_cache() -> dict:
    return _CONTROLNET_CACHE


def clear_controlnet_cache() -> bool:
    """Clear the ControlNet model cache and free VRAM."""
    global _CONTROLNET_CACHE
    if _CONTROLNET_CACHE["model"] is not None:
        model = _CONTROLNET_CACHE["model"]
        try:
            model.to("cpu")
        except Exception:
            pass
        del _CONTROLNET_CACHE["model"]
        _CONTROLNET_CACHE["model"] = None
        _CONTROLNET_CACHE["model_path"] = None
        _CONTROLNET_CACHE["cache_key"] = None
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
        print("[EricQwenImage] ControlNet cache cleared, VRAM freed")
        return True
    return False


def _default_controlnet_path() -> str:
    """Return a sensible default path for the ControlNet model."""
    candidates = [
        r"H:\Training\Qwen-Image-ControlNet-Union",
        r"H:\Training\InstantX-Qwen-Image-ControlNet-Union",
        r"H:\Training\Qwen-Image-ControlNet-Inpainting",
        r"H:\Training\InstantX-Qwen-Image-ControlNet-Inpainting",
    ]
    for p in candidates:
        if os.path.isdir(p):
            return p
    return ""


def _resolve_model_path(path: str) -> str:
    """Resolve a user-provided path to a valid model directory.

    Handles several common scenarios:
    - Path is already a directory with config.json → use as-is
    - Path points to a .safetensors file → look for a sibling directory
      containing config.json (e.g. ControlNet-Union/)
    - Path points to a .safetensors file whose parent has config.json → use parent
    """
    path = path.strip().rstrip("/\\")

    # Already a valid model directory
    if os.path.isdir(path) and os.path.isfile(os.path.join(path, "config.json")):
        return path

    # User pointed at a .safetensors file (existing or not)
    if path.lower().endswith(".safetensors"):
        parent = os.path.dirname(path)
        if not os.path.isdir(parent):
            return path  # parent doesn't exist, nothing we can do

        # Check if parent dir has config.json (weights + config side-by-side)
        if os.path.isfile(os.path.join(parent, "config.json")):
            return parent

        # Scan sibling directories for one containing config.json
        # with a QwenImageControlNetModel class
        for entry in os.listdir(parent):
            candidate = os.path.join(parent, entry)
            if os.path.isdir(candidate):
                cfg = os.path.join(candidate, "config.json")
                if os.path.isfile(cfg):
                    try:
                        import json
                        with open(cfg, "r", encoding="utf-8") as f:
                            data = json.load(f)
                        if data.get("_class_name") == "QwenImageControlNetModel":
                            return candidate
                    except Exception:
                        continue

    # Fallback: just a directory without config.json — let from_pretrained
    # give its own error
    return path


# ═══════════════════════════════════════════════════════════════════════
#  ControlNet Loader
# ═══════════════════════════════════════════════════════════════════════

class EricQwenImageControlNetLoader:
    """
    Load QwenImageControlNetModel for structure-guided generation.

    Supports the InstantX/Qwen-Image-ControlNet-Union which provides
    canny, soft edge, depth, and pose control modes in a single model.

    Wire this into the UltraGen CN node's controlnet input.
    """

    CATEGORY = "Eric Qwen-Image"
    FUNCTION = "load_controlnet"
    RETURN_TYPES = ("QWEN_IMAGE_CONTROLNET",)
    RETURN_NAMES = ("controlnet",)

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        cache = get_controlnet_cache()
        if cache["model"] is None:
            return float("nan")
        return cache.get("cache_key", "")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_path": ("STRING", {
                    "default": _default_controlnet_path(),
                    "tooltip": (
                        "Path to ControlNet model directory.\n"
                        "Supports InstantX/Qwen-Image-ControlNet-Union\n"
                        "(canny, soft edge, depth, pose) and\n"
                        "InstantX/Qwen-Image-ControlNet-Inpainting\n"
                        "(mask-based inpainting and outpainting).\n"
                        "Both use QwenImageControlNetModel."
                    )
                }),
            },
            "optional": {
                "precision": (["bf16", "fp16", "fp32"], {
                    "default": "bf16",
                    "tooltip": "Model precision (bf16 recommended)"
                }),
                "device": (["cuda", "cuda:0", "cuda:1", "cpu"], {
                    "default": "cuda",
                    "tooltip": "Device to load model on"
                }),
                "keep_in_vram": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Cache model between runs"
                }),
            }
        }

    def load_controlnet(
        self,
        model_path: str,
        precision: str = "bf16",
        device: str = "cuda",
        keep_in_vram: bool = True,
    ) -> Tuple:
        from diffusers import QwenImageControlNetModel

        # Resolve early so cache_key uses canonical path
        resolved_path = _resolve_model_path(model_path)
        cache_key = f"{resolved_path}_{precision}_{device}"
        cache = get_controlnet_cache()

        if cache["model"] is not None and cache.get("cache_key") == cache_key:
            print("[EricQwenImage] Using cached ControlNet model")
            return ({"model": cache["model"], "model_path": model_path},)

        if cache["model"] is not None:
            print("[EricQwenImage] Clearing old ControlNet cache")
            clear_controlnet_cache()

        dtype_map = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}
        dtype = dtype_map.get(precision, torch.bfloat16)

        if resolved_path != model_path:
            print(f"[EricQwenImage] Resolved model path: {model_path}")
            print(f"[EricQwenImage]   → directory: {resolved_path}")

        if not os.path.isdir(resolved_path):
            raise FileNotFoundError(
                f"ControlNet model directory not found: {resolved_path}\n"
                f"Expected a directory containing config.json + "
                f"diffusion_pytorch_model.safetensors.\n"
                f"If you downloaded a single .safetensors file, place it "
                f"alongside a config.json from the model repo."
            )
        if not os.path.isfile(os.path.join(resolved_path, "config.json")):
            raise FileNotFoundError(
                f"No config.json found in {resolved_path}.\n"
                f"Download config.json from the model repository "
                f"(e.g. InstantX/Qwen-Image-ControlNet-Union) and place "
                f"it in the same directory as the .safetensors file."
            )

        print(f"[EricQwenImage] Loading ControlNet from {resolved_path}")
        model = QwenImageControlNetModel.from_pretrained(
            resolved_path,
            torch_dtype=dtype,
            local_files_only=True,
        )
        model = model.to(device)

        params_m = sum(p.numel() for p in model.parameters()) / 1e6
        print(f"[EricQwenImage] ControlNet loaded — {params_m:.0f}M params")

        if keep_in_vram:
            cache["model"] = model
            cache["model_path"] = resolved_path
            cache["cache_key"] = cache_key

        return ({"model": model, "model_path": resolved_path},)


# ═══════════════════════════════════════════════════════════════════════
#  ControlNet Unload
# ═══════════════════════════════════════════════════════════════════════

class EricQwenImageControlNetUnload:
    """
    Unload ControlNet model from VRAM.

    Use this to free VRAM after ControlNet-guided generation.
    """

    CATEGORY = "Eric Qwen-Image"
    FUNCTION = "unload"
    RETURN_TYPES = ()
    OUTPUT_NODE = True

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {}}

    def unload(self) -> Tuple:
        cleared = clear_controlnet_cache()
        if not cleared:
            print("[EricQwenImage] ControlNet was not loaded")
        return ()
