# Copyright (c) 2026 Eric Hiss. All rights reserved.
# Licensed under the terms in LICENSE.txt (CC BY-NC 4.0 / Commercial dual license).
# https://github.com/EricRollei/Eric_Qwen_Edit_Experiments
"""
Eric Qwen-Edit Loader
Load Qwen-Image-Edit pipeline with smart resolution handling.

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


class EricQwenEditLoader:
    """
    Load the Qwen-Edit pipeline with smart resolution handling.
    
    This loader uses a custom pipeline that:
    - Preserves input resolution by default (no forced upscaling)
    - Configurable max_mp cap for VRAM safety
    - VAE tiling for high-resolution decode
    
    Supported models:
    - Qwen-Image-Edit-2509
    - Qwen-Image-Edit-2511
    
    Performance options:
    - offload_vae: Move VAE to CPU during transformer inference
    - attention_slicing: Reduce VRAM at cost of some speed
    - sequential_offload: Aggressive CPU offloading for very high res
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
                "model_path": ("STRING", {
                    "default": default_paths.get("qwen_edit_2511", ""),
                    "tooltip": "Path to Qwen-Image-Edit model (2509 or 2511)"
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
        """
        Load the Qwen-Edit pipeline.
        
        Args:
            model_path: Path to Qwen-Image-Edit model
            precision: Model precision
            device: Target device
            keep_in_vram: Whether to cache the pipeline
            offload_vae: Move VAE to CPU during inference
            attention_slicing: Enable attention slicing
            sequential_offload: Aggressive CPU offloading
            
        Returns:
            Tuple containing the pipeline wrapper
        """
        from ..pipelines import QwenEditPipeline
        
        print(f"[EricQwenEdit] Loading model from: {model_path}")
        
        # Check cache - include optimization settings
        cache = get_pipeline_cache()
        cache_key = f"{model_path}_{offload_vae}_{attention_slicing}_{sequential_offload}"
        if (cache["pipeline"] is not None and 
            cache.get("cache_key") == cache_key):
            print("[EricQwenEdit] Using cached pipeline")
            return ({"pipeline": cache["pipeline"], "model_path": model_path},)
        
        # Clear existing cache if loading different model/settings
        if cache["pipeline"] is not None:
            print("[EricQwenEdit] Clearing cached pipeline (loading different model/settings)")
            clear_pipeline_cache()
        
        # Set precision
        dtype_map = {
            "bf16": torch.bfloat16,
            "fp16": torch.float16,
            "fp32": torch.float32,
        }
        dtype = dtype_map.get(precision, torch.bfloat16)
        
        # Load pipeline
        print(f"[EricQwenEdit] Loading pipeline with precision: {precision}")
        pipeline = QwenEditPipeline.from_pretrained(
            model_path,
            torch_dtype=dtype,
            local_files_only=True,
        )
        
        # Apply optimizations before moving to device
        if sequential_offload:
            print("[EricQwenEdit] Enabling sequential CPU offload (slow but handles large images)")
            pipeline.enable_sequential_cpu_offload()
        else:
            # Move to device normally
            pipeline = pipeline.to(device)
            
            # Optional: offload VAE to CPU
            if offload_vae:
                print("[EricQwenEdit] Moving VAE to CPU (will transfer for encode/decode)")
                pipeline.vae = pipeline.vae.to("cpu")
        
        # Enable VAE tiling for high-res decode
        pipeline.vae.enable_tiling()
        print("[EricQwenEdit] VAE tiling enabled")
        
        # Enable attention slicing if requested
        if attention_slicing:
            try:
                pipeline.enable_attention_slicing(slice_size="auto")
                print("[EricQwenEdit] Attention slicing enabled")
            except Exception as e:
                print(f"[EricQwenEdit] Attention slicing not available: {e}")
        
        # Report flash attention status
        flash_enabled = torch.backends.cuda.flash_sdp_enabled()
        print(f"[EricQwenEdit] Flash SDPA enabled: {flash_enabled}")
        
        # Cache if requested
        if keep_in_vram:
            cache["pipeline"] = pipeline
            cache["model_path"] = model_path
            cache["cache_key"] = cache_key
        
        print(f"[EricQwenEdit] Pipeline loaded successfully")
        print(f"[EricQwenEdit] Transformer parameters: {sum(p.numel() for p in pipeline.transformer.parameters()) / 1e9:.2f}B")
        print(f"[EricQwenEdit] Options: offload_vae={offload_vae}, attention_slicing={attention_slicing}, sequential_offload={sequential_offload}")
        
        return ({"pipeline": pipeline, "model_path": model_path, "offload_vae": offload_vae},)


class EricQwenEditUnload:
    """
    Unload the Qwen-Edit pipeline from VRAM.
    
    Use this node to free GPU memory when done with Qwen-Edit.
    Connect the pipeline output or any downstream output to trigger unload.
    """
    
    CATEGORY = "Eric Qwen-Edit"
    FUNCTION = "unload"
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("status",)
    OUTPUT_NODE = True
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {},
            "optional": {
                "pipeline": ("QWEN_EDIT_PIPELINE", {
                    "tooltip": "Connect pipeline to unload it and free VRAM"
                }),
                "images": ("IMAGE", {
                    "tooltip": "Connect to an image output to trigger unload after generation"
                }),
            }
        }
    
    def unload(self, pipeline=None, images=None):
        """Unload the pipeline and free VRAM."""
        import gc
        
        unloaded_items = []
        
        # If a pipeline dict was passed, clean it up directly
        if pipeline is not None and isinstance(pipeline, dict):
            pipe = pipeline.get("pipeline")
            if pipe is not None:
                # Unload LoRAs from the live pipeline
                try:
                    adapter_list = pipe.get_list_adapters()
                    if any(adapter_list.values()):
                        pipe.unload_lora_weights()
                        unloaded_items.append("LoRA adapters")
                except Exception:
                    pass
                
                # Move to CPU to free VRAM
                try:
                    pipe.to("cpu")
                    unloaded_items.append("pipeline (moved to CPU)")
                except Exception:
                    try:
                        for attr_name in ["transformer", "vae", "text_encoder"]:
                            component = getattr(pipe, attr_name, None)
                            if component is not None:
                                component.to("cpu")
                        unloaded_items.append("pipeline components (moved to CPU)")
                    except Exception:
                        pass
                
                # Clear tracking data from the pipeline dict
                pipeline.pop("applied_loras", None)
        
        # Also clear the global cache
        if clear_pipeline_cache():
            unloaded_items.append("global cache")
        
        # Extra GC pass
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        if unloaded_items:
            status = f"Unloaded: {', '.join(unloaded_items)}"
        else:
            status = "No pipeline was loaded"
        
        print(f"[EricQwenEdit] {status}")
        return (status,)
