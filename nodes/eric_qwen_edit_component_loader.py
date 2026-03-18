# Copyright (c) 2026 Eric Hiss. All rights reserved.
# Licensed under the terms in LICENSE.txt (CC BY-NC 4.0 / Commercial dual license).
# https://github.com/EricRollei/Eric_Qwen_Edit_Experiments
"""
Eric Qwen-Edit Component Loader
Load Qwen-Image-Edit pipeline from separate components (base pipeline + custom transformer).

This allows using fine-tuned transformer weights with the base pipeline configuration,
without needing a complete HuggingFace model directory for the fine-tune.

Model Credits:
- Qwen-Image-Edit developed by Qwen Team (Alibaba)
- https://github.com/QwenLM/Qwen-Image

Author: Eric Hiss (GitHub: EricRollei)
"""

import os
import torch
from typing import Tuple

from .eric_qwen_edit_utils import (
    get_pipeline_cache,
    clear_pipeline_cache,
    get_default_paths,
)


class EricQwenEditComponentLoader:
    """
    Load the Qwen-Edit pipeline from separate components.

    Use this when you have a fine-tuned transformer that doesn't come as a
    complete pipeline directory. Point base_pipeline_path to a full Qwen-Image-Edit
    model (for the config, scheduler, tokenizer, processor, and optionally
    VAE/text_encoder), then override individual components with custom paths.

    Typical use case:
    - base_pipeline_path: Qwen/Qwen-Image-Edit-2511 (full model)
    - transformer_path: /path/to/finetune/transformer (just the transformer weights)

    The loader will assemble the pipeline using the base config and swap in
    the custom transformer (and optional custom VAE / text_encoder).

    Supported transformer_path formats:
    - Directory containing diffusion_pytorch_model*.safetensors + config.json
    - Full model directory with a 'transformer' subfolder
    - Single .safetensors file (loaded as state dict into base transformer)
    """

    CATEGORY = "Eric Qwen-Edit"
    FUNCTION = "load_pipeline"
    RETURN_TYPES = ("QWEN_EDIT_PIPELINE",)
    RETURN_NAMES = ("pipeline",)

    @classmethod
    def INPUT_TYPES(cls):
        default_paths = get_default_paths()

        return {
            "required": {
                "base_pipeline_path": ("STRING", {
                    "default": default_paths.get("qwen_edit_2511", ""),
                    "tooltip": "Path to complete Qwen-Image-Edit model (provides config, scheduler, tokenizer, processor)"
                }),
            },
            "optional": {
                "transformer_path": ("STRING", {
                    "default": "",
                    "tooltip": "Path to custom/fine-tuned transformer. Can be a folder or single .safetensors file. Leave empty to use base."
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
                    "tooltip": "Move VAE to CPU during transformer inference (saves ~2GB VRAM)"
                }),
                "attention_slicing": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Enable attention slicing (reduces VRAM, slightly slower)"
                }),
                "sequential_offload": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Aggressive CPU offloading - very slow but handles huge images"
                }),
            }
        }

    def _detect_transformer_format(self, path: str) -> str:
        """Detect the format of the transformer path.
        
        Returns one of:
        - 'subfolder': path contains a 'transformer' subfolder
        - 'direct': path is a directory with model files directly
        - 'single_file': path is a single .safetensors / .bin file
        - 'unknown': cannot determine
        """
        if not path or not path.strip():
            return 'none'
        
        path = path.strip()
        
        if os.path.isfile(path):
            if path.endswith(('.safetensors', '.bin', '.pt', '.pth')):
                return 'single_file'
            return 'unknown'
        
        if os.path.isdir(path):
            # Check if it has a 'transformer' subfolder
            if os.path.isdir(os.path.join(path, 'transformer')):
                return 'subfolder'
            # Check if it contains model weight files directly
            for fname in os.listdir(path):
                if fname.startswith('diffusion_pytorch_model') and fname.endswith(('.safetensors', '.bin')):
                    return 'direct'
                if fname == 'config.json':
                    return 'direct'
            # Check for model index (full pipeline dir - user may have pointed to the wrong thing)
            if os.path.isfile(os.path.join(path, 'model_index.json')):
                return 'subfolder'
            return 'unknown'
        
        return 'unknown'

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
        """Load pipeline from base + optional component overrides."""
        from ..pipelines import QwenEditPipeline

        # Build cache key from all relevant params
        cache_key = (
            f"comp_{base_pipeline_path}_{transformer_path}_{vae_path}_"
            f"{text_encoder_path}_{offload_vae}_{attention_slicing}_{sequential_offload}"
        )

        cache = get_pipeline_cache()
        if cache["pipeline"] is not None and cache.get("cache_key") == cache_key:
            print("[EricQwenEdit] Using cached pipeline (component loader)")
            return ({"pipeline": cache["pipeline"], "model_path": base_pipeline_path},)

        if cache["pipeline"] is not None:
            print("[EricQwenEdit] Clearing cached pipeline (loading different model/settings)")
            clear_pipeline_cache()

        dtype_map = {
            "bf16": torch.bfloat16,
            "fp16": torch.float16,
            "fp32": torch.float32,
        }
        dtype = dtype_map.get(precision, torch.bfloat16)

        # Determine what component overrides we need
        kwargs = {}

        # --- Transformer override ---
        transformer_path = transformer_path.strip() if transformer_path else ""
        if transformer_path:
            fmt = self._detect_transformer_format(transformer_path)
            print(f"[EricQwenEdit] Transformer override: {transformer_path} (format: {fmt})")

            from diffusers import QwenImageTransformer2DModel

            if fmt == 'subfolder':
                # Load from the 'transformer' subfolder
                kwargs["transformer"] = QwenImageTransformer2DModel.from_pretrained(
                    transformer_path,
                    subfolder="transformer",
                    torch_dtype=dtype,
                    local_files_only=True,
                )
            elif fmt == 'direct':
                # Directory contains model files directly
                kwargs["transformer"] = QwenImageTransformer2DModel.from_pretrained(
                    transformer_path,
                    torch_dtype=dtype,
                    local_files_only=True,
                )
            elif fmt == 'single_file':
                # Load base transformer first, then swap state dict
                print("[EricQwenEdit] Loading base transformer config, then applying custom weights...")
                base_transformer = QwenImageTransformer2DModel.from_pretrained(
                    base_pipeline_path,
                    subfolder="transformer",
                    torch_dtype=dtype,
                    local_files_only=True,
                )
                from safetensors.torch import load_file as safetensors_load
                if transformer_path.endswith('.safetensors'):
                    state_dict = safetensors_load(transformer_path)
                else:
                    state_dict = torch.load(transformer_path, map_location="cpu")
                base_transformer.load_state_dict(state_dict, strict=False)
                kwargs["transformer"] = base_transformer
            else:
                raise ValueError(
                    f"Cannot determine transformer format for: {transformer_path}\n"
                    "Expected: a directory with model files, a path with 'transformer' subfolder, "
                    "or a single .safetensors file."
                )
            print(f"[EricQwenEdit] Custom transformer loaded")

        # --- VAE override ---
        vae_path = vae_path.strip() if vae_path else ""
        if vae_path:
            from diffusers import AutoencoderKLQwenImage
            print(f"[EricQwenEdit] VAE override: {vae_path}")
            if os.path.isdir(os.path.join(vae_path, "vae")):
                kwargs["vae"] = AutoencoderKLQwenImage.from_pretrained(
                    vae_path, subfolder="vae", torch_dtype=dtype, local_files_only=True,
                )
            else:
                kwargs["vae"] = AutoencoderKLQwenImage.from_pretrained(
                    vae_path, torch_dtype=dtype, local_files_only=True,
                )
            print("[EricQwenEdit] Custom VAE loaded")

        # --- Text encoder override ---
        text_encoder_path = text_encoder_path.strip() if text_encoder_path else ""
        if text_encoder_path:
            from transformers import Qwen2_5_VLForConditionalGeneration
            print(f"[EricQwenEdit] Text encoder override: {text_encoder_path}")
            if os.path.isdir(os.path.join(text_encoder_path, "text_encoder")):
                kwargs["text_encoder"] = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                    text_encoder_path, subfolder="text_encoder", torch_dtype=dtype, local_files_only=True,
                )
            else:
                kwargs["text_encoder"] = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                    text_encoder_path, torch_dtype=dtype, local_files_only=True,
                )
            print("[EricQwenEdit] Custom text encoder loaded")

        # --- Load the pipeline ---
        component_info = list(kwargs.keys()) if kwargs else ["all from base"]
        print(f"[EricQwenEdit] Loading pipeline from base: {base_pipeline_path}")
        print(f"[EricQwenEdit] Component overrides: {component_info}")

        pipeline = QwenEditPipeline.from_pretrained(
            base_pipeline_path,
            torch_dtype=dtype,
            local_files_only=True,
            **kwargs,
        )

        # Apply optimizations
        if sequential_offload:
            print("[EricQwenEdit] Enabling sequential CPU offload")
            pipeline.enable_sequential_cpu_offload()
        else:
            pipeline = pipeline.to(device)
            if offload_vae:
                print("[EricQwenEdit] Moving VAE to CPU")
                pipeline.vae = pipeline.vae.to("cpu")

        pipeline.vae.enable_tiling()
        print("[EricQwenEdit] VAE tiling enabled")

        if attention_slicing:
            try:
                pipeline.enable_attention_slicing(slice_size="auto")
                print("[EricQwenEdit] Attention slicing enabled")
            except Exception as e:
                print(f"[EricQwenEdit] Attention slicing not available: {e}")

        flash_enabled = torch.backends.cuda.flash_sdp_enabled()
        print(f"[EricQwenEdit] Flash SDPA enabled: {flash_enabled}")

        if keep_in_vram:
            cache["pipeline"] = pipeline
            cache["model_path"] = base_pipeline_path
            cache["cache_key"] = cache_key

        print(f"[EricQwenEdit] Component pipeline loaded successfully")
        print(f"[EricQwenEdit] Transformer parameters: {sum(p.numel() for p in pipeline.transformer.parameters()) / 1e9:.2f}B")

        return ({"pipeline": pipeline, "model_path": base_pipeline_path, "offload_vae": offload_vae},)
