# Copyright (c) 2026 Eric Hiss. All rights reserved.
# Licensed under the terms in LICENSE.txt (CC BY-NC 4.0 / Commercial dual license).
# https://github.com/EricRollei/Eric_Qwen_Edit_Experiments
"""
Eric Qwen-Edit Image Node
Edit images using Qwen-Image-Edit with smart resolution handling.

Model Credits:
- Qwen-Image-Edit developed by Qwen Team (Alibaba)
- https://github.com/QwenLM/Qwen-Image

Author: Eric Hiss (GitHub: EricRollei)
"""

import math
import torch
import time
from typing import Tuple, Optional

import comfy.utils
from PIL import Image

from .eric_qwen_edit_utils import (
    tensor_to_pil,
    pil_to_tensor,
    prepare_image_for_pipeline,
)

DIMENSION_ALIGNMENT = 32  # Must be divisible by 32 for VAE packing


class EricQwenEditImage:
    """
    Edit an image using Qwen-Image-Edit.
    
    This node uses a custom pipeline that preserves input resolution
    by default, rather than forcing all outputs to 1MP.
    
    Resolution behavior:
    - Input smaller than max_mp: output matches input size
    - Input larger than max_mp: output scaled down to max_mp
    - All outputs aligned to 32 pixels
    
    Performance note:
    - Attention scales O(n²) with resolution
    - 1MP: ~1x time, 4MP: ~16x time, 8MP: ~64x time
    - Consider using lower max_mp for faster iteration
    
    Examples:
    - "Change the background to a sunset"
    - "Make the person smile"
    - "Add a hat to the person"
    - "Change the color of the car to red"
    """
    
    CATEGORY = "Eric Qwen-Edit"
    FUNCTION = "edit"
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "pipeline": ("QWEN_EDIT_PIPELINE",),
                "image": ("IMAGE", {
                    "tooltip": "Image to edit"
                }),
                "prompt": ("STRING", {
                    "multiline": True,
                    "default": "Edit this image to...",
                    "tooltip": "Describe the edit to apply"
                }),
            },
            "optional": {
                "negative_prompt": ("STRING", {
                    "multiline": True,
                    "default": "",
                    "tooltip": "What to avoid"
                }),
                "steps": ("INT", {
                    "default": 8,
                    "min": 1,
                    "max": 100,
                    "step": 1,
                    "tooltip": "Inference steps (8 for lightning LoRA, 50 for base model)"
                }),
                "true_cfg_scale": ("FLOAT", {
                    "default": 4.0,
                    "min": 1.0,
                    "max": 20.0,
                    "step": 0.5,
                    "tooltip": "True CFG scale (main quality control)"
                }),
                "seed": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 0xffffffffffffffff,
                    "tooltip": "Random seed"
                }),
                "max_mp": ("FLOAT", {
                    "default": 8.0,
                    "min": 0.5,
                    "max": 16.0,
                    "step": 0.5,
                    "tooltip": "Max output megapixels. Lower = faster. Input preserved up to this cap."
                }),
                "upscale_to_max_mp": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "When enabled, upscale small inputs to fill the max_mp budget (preserving aspect ratio). Saves needing a separate Scale Image node."
                }),
            }
        }
    
    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return kwargs.get("seed", 0)
    
    def edit(
        self,
        pipeline: dict,
        image: torch.Tensor,
        prompt: str,
        negative_prompt: str = "",
        steps: int = 50,
        true_cfg_scale: float = 4.0,
        seed: int = 0,
        max_mp: float = 8.0,
        upscale_to_max_mp: bool = False,
    ) -> Tuple[torch.Tensor]:
        """Edit an image using Qwen-Edit."""
        pipe = pipeline["pipeline"]
        offload_vae = pipeline.get("offload_vae", False)
        
        # Convert image
        if image.dim() == 4:
            pil_image = tensor_to_pil(image[0])
        else:
            pil_image = tensor_to_pil(image)
        pil_image = prepare_image_for_pipeline(pil_image)
        
        input_w, input_h = pil_image.size
        input_mp = (input_w * input_h) / (1024 * 1024)
        
        # Upscale small images to fill the max_mp budget
        if upscale_to_max_mp and input_mp < max_mp:
            max_pixels = int(max_mp * 1024 * 1024)
            scale = math.sqrt(max_pixels / (input_w * input_h))
            new_w = max(DIMENSION_ALIGNMENT, int(input_w * scale) // DIMENSION_ALIGNMENT * DIMENSION_ALIGNMENT)
            new_h = max(DIMENSION_ALIGNMENT, int(input_h * scale) // DIMENSION_ALIGNMENT * DIMENSION_ALIGNMENT)
            print(f"[EricQwenEdit] Upscaling input {input_w}x{input_h} ({input_mp:.1f}MP) "
                  f"-> {new_w}x{new_h} ({new_w * new_h / 1e6:.1f}MP) to fill max_mp={max_mp}")
            pil_image = pil_image.resize((new_w, new_h), Image.LANCZOS)
            input_w, input_h = new_w, new_h
            input_mp = (input_w * input_h) / (1024 * 1024)
        
        effective_mp = min(input_mp, max_mp)
        
        # Estimate time (very rough)
        # Base: ~2 sec/step at 1MP, scales quadratically
        estimated_time_per_step = 2 * (effective_mp ** 2)
        estimated_total = estimated_time_per_step * steps
        
        print(f"[EricQwenEdit] Image Edit")
        print(f"[EricQwenEdit] Input: {input_w}x{input_h} ({input_mp:.1f}MP)")
        print(f"[EricQwenEdit] Effective: ~{effective_mp:.1f}MP, Steps: {steps}, CFG: {true_cfg_scale}")
        print(f"[EricQwenEdit] Estimated time: ~{estimated_total/60:.1f} minutes (rough estimate)")
        print(f"[EricQwenEdit] Prompt: {prompt[:80]}...")
        
        # Get device from transformer
        device = next(pipe.transformer.parameters()).device
        generator = torch.Generator(device=device).manual_seed(seed)
        
        # If VAE was offloaded, move it to GPU for encode/decode
        vae_device_original = None
        if offload_vae:
            vae_device_original = next(pipe.vae.parameters()).device
            if str(vae_device_original) == "cpu":
                print("[EricQwenEdit] Moving VAE to GPU for encode/decode...")
                pipe.vae = pipe.vae.to(device)
        
        start_time = time.time()
        
        # ComfyUI progress bar
        pbar = comfy.utils.ProgressBar(steps)
        def _progress_callback(pipeline, step_index, timestep, cb_kwargs):
            pbar.update(1)
            return cb_kwargs
        
        # Run pipeline
        try:
            with torch.inference_mode():
                output = pipe(
                    prompt=prompt,
                    image=pil_image,
                    max_pixels=int(max_mp * 1024 * 1024),
                    negative_prompt=negative_prompt if negative_prompt else " ",
                    num_inference_steps=steps,
                    true_cfg_scale=true_cfg_scale,
                    generator=generator,
                    num_images_per_prompt=1,
                    callback_on_step_end=_progress_callback,
                    callback_on_step_end_tensor_inputs=["latents"],
                )
        finally:
            # Move VAE back to CPU if it was offloaded
            if offload_vae and vae_device_original is not None and str(vae_device_original) == "cpu":
                print("[EricQwenEdit] Moving VAE back to CPU...")
                pipe.vae = pipe.vae.to("cpu")
                torch.cuda.empty_cache()
        
        elapsed = time.time() - start_time
        
        result = output.images[0]
        result_tensor = pil_to_tensor(result).unsqueeze(0)
        
        print(f"[EricQwenEdit] Edit complete: {result.size[0]}x{result.size[1]}")
        print(f"[EricQwenEdit] Actual time: {elapsed/60:.1f} minutes ({elapsed/steps:.1f} sec/step)")
        
        return (result_tensor,)
