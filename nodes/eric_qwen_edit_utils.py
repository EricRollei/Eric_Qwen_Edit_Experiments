# Copyright (c) 2026 Eric Hiss. All rights reserved.
# Licensed under the terms in LICENSE.txt (CC BY-NC 4.0 / Commercial dual license).
# https://github.com/EricRollei/Eric_Qwen_Edit_Experiments
"""
Eric Qwen-Edit Utilities
Helper functions for image processing and pipeline management.

Author: Eric Hiss (GitHub: EricRollei)
"""

import torch
import numpy as np
from PIL import Image
from typing import Optional
import gc

# Pipeline cache for keeping model in VRAM
_PIPELINE_CACHE = {
    "pipeline": None,
    "model_path": None,
}


def get_pipeline_cache() -> dict:
    """Get the global pipeline cache."""
    return _PIPELINE_CACHE


def clear_pipeline_cache() -> bool:
    """Clear the pipeline cache and free VRAM aggressively."""
    global _PIPELINE_CACHE
    
    if _PIPELINE_CACHE["pipeline"] is not None:
        pipe = _PIPELINE_CACHE["pipeline"]
        
        # 1. Unload any LoRA adapters first
        try:
            adapter_list = pipe.get_list_adapters()
            if any(adapter_list.values()):
                pipe.unload_lora_weights()
                print("[EricQwenEdit] LoRA adapters unloaded")
        except Exception as e:
            print(f"[EricQwenEdit] Note during LoRA cleanup: {e}")
        
        # 2. Move all components to CPU to free GPU memory
        try:
            pipe.to("cpu")
            print("[EricQwenEdit] Pipeline moved to CPU")
        except Exception as e:
            # If sequential offload was used, .to() may not work normally
            print(f"[EricQwenEdit] Note during CPU move: {e}")
            try:
                for attr_name in ["transformer", "vae", "text_encoder"]:
                    component = getattr(pipe, attr_name, None)
                    if component is not None:
                        component.to("cpu")
            except Exception:
                pass
        
        # 3. Clear references
        del _PIPELINE_CACHE["pipeline"]
        _PIPELINE_CACHE["pipeline"] = None
        _PIPELINE_CACHE["model_path"] = None
        _PIPELINE_CACHE.pop("cache_key", None)
        del pipe
        
        # 4. Aggressive garbage collection
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
        
        print("[EricQwenEdit] Pipeline cache cleared, VRAM freed")
        return True
    return False


def get_default_paths() -> dict:
    """Get default model paths."""
    return {
        "qwen_edit_2511": r"H:\Training\Qwen-Image-Edit-2511",
        "qwen_edit_2509": r"H:\Training\Qwen-Image-Edit-2509",
    }


def tensor_to_pil(tensor: torch.Tensor) -> Image.Image:
    """Convert a ComfyUI tensor to PIL Image."""
    if tensor.dim() == 4:
        tensor = tensor[0]
    
    # ComfyUI tensors are [H, W, C] in range [0, 1]
    np_image = tensor.cpu().numpy()
    np_image = (np_image * 255).clip(0, 255).astype(np.uint8)
    
    return Image.fromarray(np_image)


def pil_to_tensor(image: Image.Image) -> torch.Tensor:
    """Convert a PIL Image to ComfyUI tensor."""
    np_image = np.array(image).astype(np.float32) / 255.0
    return torch.from_numpy(np_image)


def prepare_image_for_pipeline(image: Image.Image) -> Image.Image:
    """Prepare an image for the pipeline (ensure RGB)."""
    if image.mode != "RGB":
        image = image.convert("RGB")
    return image


def prepare_mask_for_pipeline(mask: Image.Image) -> Image.Image:
    """Prepare a mask for the inpainting pipeline (ensure L mode)."""
    if mask.mode != "L":
        mask = mask.convert("L")
    return mask
