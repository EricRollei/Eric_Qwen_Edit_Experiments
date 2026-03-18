# Copyright (c) 2026 Eric Hiss. All rights reserved.
# Licensed under the terms in LICENSE.txt (CC BY-NC 4.0 / Commercial dual license).
# https://github.com/EricRollei/Eric_Qwen_Edit_Experiments
"""
Eric Qwen-Edit LoRA Node
Apply LoRA weights to Qwen-Image-Edit pipeline.

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
                # Load fresh
                pipe.load_lora_weights(
                    lora_path,
                    adapter_name=adapter_name,
                )
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
