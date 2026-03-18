# Copyright (c) 2026 Eric Hiss. All rights reserved.
# Licensed under the terms in LICENSE.txt (CC BY-NC 4.0 / Commercial dual license).
# https://github.com/EricRollei/Eric_Qwen_Edit_Experiments
"""
Eric Qwen-Image Pipeline Loader
Loads QwenImagePipeline (text-to-image) for Qwen-Image / Qwen-Image-2512.

This is the generation pipeline — no input images required.
Uses diffusers' built-in QwenImagePipeline.

Model Credits:
- Qwen-Image developed by Qwen Team (Alibaba)
- https://github.com/QwenLM/Qwen-Image

Author: Eric Hiss (GitHub: EricRollei)
"""

import gc
import os
import torch
from typing import Tuple


# ── Separate cache for the generation pipeline ──────────────────────────
_GEN_PIPELINE_CACHE = {
    "pipeline": None,
    "model_path": None,
    "cache_key": None,
}


def get_gen_pipeline_cache() -> dict:
    return _GEN_PIPELINE_CACHE


def clear_gen_pipeline_cache() -> bool:
    """Clear the generation pipeline cache and free VRAM."""
    global _GEN_PIPELINE_CACHE
    if _GEN_PIPELINE_CACHE["pipeline"] is not None:
        pipe = _GEN_PIPELINE_CACHE["pipeline"]
        try:
            pipe.to("cpu")
        except Exception:
            for attr in ("transformer", "vae", "text_encoder"):
                comp = getattr(pipe, attr, None)
                if comp is not None:
                    try:
                        comp.to("cpu")
                    except Exception:
                        pass
        del _GEN_PIPELINE_CACHE["pipeline"]
        _GEN_PIPELINE_CACHE["pipeline"] = None
        _GEN_PIPELINE_CACHE["model_path"] = None
        _GEN_PIPELINE_CACHE["cache_key"] = None
        del pipe
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
        print("[EricQwenImage] Generation pipeline cache cleared, VRAM freed")
        return True
    return False


def _default_gen_path() -> str:
    """Return a sensible default path for Qwen-Image-2512."""
    candidates = [
        r"H:\Training\Qwen-Image-2512",
        r"H:\Training\Qwen-Image",
    ]
    for p in candidates:
        if os.path.isdir(p):
            return p
    return ""


# ═══════════════════════════════════════════════════════════════════════
#  Loader Node
# ═══════════════════════════════════════════════════════════════════════

class EricQwenImageLoader:
    """
    Load QwenImagePipeline for text-to-image generation.

    Supports Qwen-Image and Qwen-Image-2512.  Uses a separate pipeline
    cache from the Edit loader so both can coexist.
    """

    CATEGORY = "Eric Qwen-Image"
    FUNCTION = "load_pipeline"
    RETURN_TYPES = ("QWEN_IMAGE_PIPELINE",)
    RETURN_NAMES = ("pipeline",)

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_path": ("STRING", {
                    "default": _default_gen_path(),
                    "tooltip": "Path to Qwen-Image or Qwen-Image-2512 model directory"
                }),
            },
            "optional": {
                "precision": (["bf16", "fp16", "fp32"], {
                    "default": "bf16",
                    "tooltip": "Model precision (bf16 recommended for RTX 40/50 series)"
                }),
                "device": (["cuda", "cuda:0", "cuda:1", "cpu"], {
                    "default": "cuda",
                    "tooltip": "Device to load model on"
                }),
                "keep_in_vram": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Cache model between runs (faster but uses VRAM)"
                }),
                "offload_vae": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Move VAE to CPU during transformer inference (saves ~1 GB)"
                }),
                "attention_slicing": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Trade speed for lower peak VRAM"
                }),
                "sequential_offload": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Extreme VRAM savings via sequential CPU offload"
                }),
            }
        }

    def load_pipeline(
        self,
        model_path: str,
        precision: str = "bf16",
        device: str = "cuda",
        keep_in_vram: bool = True,
        offload_vae: bool = False,
        attention_slicing: bool = False,
        sequential_offload: bool = False,
    ) -> Tuple:
        from diffusers import QwenImagePipeline

        cache_key = f"{model_path}_{precision}_{device}_{offload_vae}_{attention_slicing}_{sequential_offload}"
        cache = get_gen_pipeline_cache()

        if cache["pipeline"] is not None and cache.get("cache_key") == cache_key:
            print("[EricQwenImage] Using cached generation pipeline")
            return ({"pipeline": cache["pipeline"], "model_path": model_path},)

        if cache["pipeline"] is not None:
            print("[EricQwenImage] Clearing old generation pipeline cache")
            clear_gen_pipeline_cache()

        dtype_map = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}
        dtype = dtype_map.get(precision, torch.bfloat16)

        print(f"[EricQwenImage] Loading QwenImagePipeline from {model_path}")
        pipeline = QwenImagePipeline.from_pretrained(
            model_path,
            torch_dtype=dtype,
            local_files_only=True,
        )

        # Optimizations
        if sequential_offload:
            print("[EricQwenImage] Enabling sequential CPU offload")
            pipeline.enable_sequential_cpu_offload()
        else:
            pipeline = pipeline.to(device)
            if offload_vae:
                print("[EricQwenImage] Moving VAE to CPU")
                pipeline.vae = pipeline.vae.to("cpu")

        pipeline.vae.enable_tiling()
        print("[EricQwenImage] VAE tiling enabled")

        if attention_slicing:
            try:
                pipeline.enable_attention_slicing(slice_size="auto")
                print("[EricQwenImage] Attention slicing enabled")
            except Exception as e:
                print(f"[EricQwenImage] Attention slicing not available: {e}")

        flash_enabled = torch.backends.cuda.flash_sdp_enabled()
        print(f"[EricQwenImage] Flash SDPA enabled: {flash_enabled}")

        if keep_in_vram:
            cache["pipeline"] = pipeline
            cache["model_path"] = model_path
            cache["cache_key"] = cache_key

        params_b = sum(p.numel() for p in pipeline.transformer.parameters()) / 1e9
        print(f"[EricQwenImage] Generation pipeline loaded — transformer: {params_b:.2f}B params")

        return ({"pipeline": pipeline, "model_path": model_path, "offload_vae": offload_vae},)


# ═══════════════════════════════════════════════════════════════════════
#  Unload Node
# ═══════════════════════════════════════════════════════════════════════

class EricQwenImageUnload:
    """
    Free VRAM by unloading the Qwen-Image generation pipeline.
    """

    CATEGORY = "Eric Qwen-Image"
    FUNCTION = "unload"
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("status",)
    OUTPUT_NODE = True

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {},
            "optional": {
                "pipeline": ("QWEN_IMAGE_PIPELINE", {
                    "tooltip": "Connect pipeline to unload it and free VRAM"
                }),
                "images": ("IMAGE", {
                    "tooltip": "Connect to an image output to trigger unload after generation"
                }),
            }
        }

    def unload(self, pipeline=None, images=None) -> Tuple[str]:
        freed = clear_gen_pipeline_cache()
        status = "Generation pipeline unloaded" if freed else "No generation pipeline was loaded"
        print(f"[EricQwenImage] {status}")
        return (status,)
