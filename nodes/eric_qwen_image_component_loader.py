# Copyright (c) 2026 Eric Hiss. All rights reserved.
# Licensed under the terms in LICENSE.txt (CC BY-NC 4.0 / Commercial dual license).
# https://github.com/EricRollei/Eric_Qwen_Edit_Experiments
"""
Eric Qwen-Image Component Loader
Load QwenImagePipeline from separate components (base + custom transformer/VAE/text_encoder).

Mirrors the Edit Component Loader but targets the generation (text-to-image)
pipeline class (QwenImagePipeline).

Model Credits:
- Qwen-Image developed by Qwen Team (Alibaba)
- https://github.com/QwenLM/Qwen-Image

Author: Eric Hiss (GitHub: EricRollei)
"""

import os
import torch
from typing import Tuple

from .eric_qwen_image_loader import (
    get_gen_pipeline_cache,
    clear_gen_pipeline_cache,
    _default_gen_path,
)


class EricQwenImageComponentLoader:
    """
    Load QwenImagePipeline from separate component directories.

    Use this when you have a fine-tuned transformer that doesn't ship as a
    complete pipeline directory.  Point base_pipeline_path to a full
    Qwen-Image / Qwen-Image-2512 model for the config, scheduler, and
    tokenizer, then override individual weight folders.

    Note: QwenImagePipeline needs scheduler, vae, text_encoder, tokenizer,
    and transformer.  There is **no processor** component (unlike the Edit
    pipeline which also needs a Qwen2VLProcessor).

    base_pipeline_path is always required — it provides model_index.json,
    scheduler/, and tokenizer/ at minimum.
    """

    CATEGORY = "Eric Qwen-Image"
    FUNCTION = "load_pipeline"
    RETURN_TYPES = ("QWEN_IMAGE_PIPELINE",)
    RETURN_NAMES = ("pipeline",)

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "base_pipeline_path": ("STRING", {
                    "default": _default_gen_path(),
                    "tooltip": "Path to complete Qwen-Image model (provides config, scheduler, tokenizer, and defaults for unset components)"
                }),
            },
            "optional": {
                "transformer_path": ("STRING", {
                    "default": "",
                    "tooltip": "Path to custom/fine-tuned transformer. Folder or single .safetensors file. Leave empty to use base."
                }),
                "vae_path": ("STRING", {
                    "default": "",
                    "tooltip": "Path to custom VAE. Leave empty to use base."
                }),
                "text_encoder_path": ("STRING", {
                    "default": "",
                    "tooltip": "Path to custom text encoder. Leave empty to use base."
                }),
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
                    "tooltip": "Keep model in VRAM between runs (faster but uses memory)"
                }),
                "offload_vae": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Move VAE to CPU during transformer inference (saves ~1 GB)"
                }),
                "attention_slicing": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Enable attention slicing (reduces VRAM, slightly slower)"
                }),
                "sequential_offload": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Aggressive CPU offloading — very slow but handles huge images"
                }),
            }
        }

    # ── Transformer format detection (same logic as Edit component loader) ──

    @staticmethod
    def _detect_transformer_format(path: str) -> str:
        if not path or not path.strip():
            return "none"
        path = path.strip()
        if os.path.isfile(path):
            if path.endswith((".safetensors", ".bin", ".pt", ".pth")):
                return "single_file"
            return "unknown"
        if os.path.isdir(path):
            if os.path.isdir(os.path.join(path, "transformer")):
                return "subfolder"
            for fname in os.listdir(path):
                if fname.startswith("diffusion_pytorch_model") and fname.endswith((".safetensors", ".bin")):
                    return "direct"
                if fname == "config.json":
                    return "direct"
            if os.path.isfile(os.path.join(path, "model_index.json")):
                return "subfolder"
            return "unknown"
        return "unknown"

    # ── Main loader ──────────────────────────────────────────────────────

    def load_pipeline(
        self,
        base_pipeline_path: str,
        transformer_path: str = "",
        vae_path: str = "",
        text_encoder_path: str = "",
        precision: str = "bf16",
        device: str = "cuda",
        keep_in_vram: bool = True,
        offload_vae: bool = False,
        attention_slicing: bool = False,
        sequential_offload: bool = False,
    ) -> Tuple:
        from diffusers import QwenImagePipeline

        cache_key = (
            f"gen_comp_{base_pipeline_path}_{transformer_path}_{vae_path}_"
            f"{text_encoder_path}_{offload_vae}_{attention_slicing}_{sequential_offload}"
        )
        cache = get_gen_pipeline_cache()

        if cache["pipeline"] is not None and cache.get("cache_key") == cache_key:
            print("[EricQwenImage] Using cached generation pipeline (component loader)")
            return ({"pipeline": cache["pipeline"], "model_path": base_pipeline_path},)

        if cache["pipeline"] is not None:
            print("[EricQwenImage] Clearing cached pipeline (different config)")
            clear_gen_pipeline_cache()

        dtype_map = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}
        dtype = dtype_map.get(precision, torch.bfloat16)

        kwargs = {}

        # ── Transformer override ──
        transformer_path = transformer_path.strip() if transformer_path else ""
        if transformer_path:
            fmt = self._detect_transformer_format(transformer_path)
            print(f"[EricQwenImage] Transformer override: {transformer_path} (format: {fmt})")
            from diffusers import QwenImageTransformer2DModel

            if fmt == "subfolder":
                kwargs["transformer"] = QwenImageTransformer2DModel.from_pretrained(
                    transformer_path, subfolder="transformer", torch_dtype=dtype, local_files_only=True)
            elif fmt == "direct":
                kwargs["transformer"] = QwenImageTransformer2DModel.from_pretrained(
                    transformer_path, torch_dtype=dtype, local_files_only=True)
            elif fmt == "single_file":
                print("[EricQwenImage] Loading base transformer config, applying custom weights...")
                base_t = QwenImageTransformer2DModel.from_pretrained(
                    base_pipeline_path, subfolder="transformer", torch_dtype=dtype, local_files_only=True)
                from safetensors.torch import load_file as safetensors_load
                if transformer_path.endswith(".safetensors"):
                    state_dict = safetensors_load(transformer_path)
                else:
                    state_dict = torch.load(transformer_path, map_location="cpu")
                base_t.load_state_dict(state_dict, strict=False)
                kwargs["transformer"] = base_t
            else:
                raise ValueError(f"Cannot determine transformer format: {transformer_path}")
            print("[EricQwenImage] Custom transformer loaded")

        # ── VAE override ──
        vae_path = vae_path.strip() if vae_path else ""
        if vae_path:
            from diffusers import AutoencoderKLQwenImage
            print(f"[EricQwenImage] VAE override: {vae_path}")
            if os.path.isdir(os.path.join(vae_path, "vae")):
                kwargs["vae"] = AutoencoderKLQwenImage.from_pretrained(
                    vae_path, subfolder="vae", torch_dtype=dtype, local_files_only=True)
            else:
                kwargs["vae"] = AutoencoderKLQwenImage.from_pretrained(
                    vae_path, torch_dtype=dtype, local_files_only=True)
            print("[EricQwenImage] Custom VAE loaded")

        # ── Text encoder override ──
        text_encoder_path = text_encoder_path.strip() if text_encoder_path else ""
        if text_encoder_path:
            from transformers import Qwen2_5_VLForConditionalGeneration
            print(f"[EricQwenImage] Text encoder override: {text_encoder_path}")
            if os.path.isdir(os.path.join(text_encoder_path, "text_encoder")):
                kwargs["text_encoder"] = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                    text_encoder_path, subfolder="text_encoder", torch_dtype=dtype, local_files_only=True)
            else:
                kwargs["text_encoder"] = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                    text_encoder_path, torch_dtype=dtype, local_files_only=True)
            print("[EricQwenImage] Custom text encoder loaded")

        # ── Assemble pipeline ──
        component_info = list(kwargs.keys()) if kwargs else ["all from base"]
        print(f"[EricQwenImage] Loading generation pipeline from: {base_pipeline_path}")
        print(f"[EricQwenImage] Component overrides: {component_info}")

        pipeline = QwenImagePipeline.from_pretrained(
            base_pipeline_path,
            torch_dtype=dtype,
            local_files_only=True,
            **kwargs,
        )

        # ── Optimizations ──
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

        if keep_in_vram:
            cache["pipeline"] = pipeline
            cache["model_path"] = base_pipeline_path
            cache["cache_key"] = cache_key

        params_b = sum(p.numel() for p in pipeline.transformer.parameters()) / 1e9
        print(f"[EricQwenImage] Component pipeline loaded — transformer: {params_b:.2f}B params")

        return ({"pipeline": pipeline, "model_path": base_pipeline_path, "offload_vae": offload_vae},)
