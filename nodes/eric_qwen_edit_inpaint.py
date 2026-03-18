# Copyright (c) 2026 Eric Hiss. All rights reserved.
# Licensed under the terms in LICENSE.txt (CC BY-NC 4.0 / Commercial dual license).
# https://github.com/EricRollei/Eric_Qwen_Edit_Experiments
"""
Eric Qwen-Edit Inpaint Node
Inpaint images using Qwen-Image-Edit with smart resolution handling.

IMPORTANT: Qwen-Edit is an *edit* model, NOT a dedicated inpainting model.
It has NO native mask input channel.  To simulate inpainting:

1. The masked region is blanked out (white/gray) or overlaid in the image
   sent to BOTH the VL (semantic) and VAE (pixel-level) encoders — the
   model sees a gap in semantic AND latent space and generates content there.
2. After generation, the output is composited with the original image using
   the mask: original pixels are kept outside the mask, generated pixels
   are used only inside the mask, with edge feathering for smooth blending.

Model Credits:
- Qwen-Image-Edit developed by Qwen Team (Alibaba)
- https://github.com/QwenLM/Qwen-Image

Author: Eric Hiss (GitHub: EricRollei)
"""

import torch
import time
from typing import Tuple, Optional

import comfy.utils

from .eric_qwen_edit_utils import (
    tensor_to_pil,
    pil_to_tensor,
    prepare_image_for_pipeline,
    prepare_mask_for_pipeline,
)


def _apply_mask_to_image(image, mask, mode="blank_white"):
    """
    Create a version of *image* where the masked region is visually
    modified so the model can see where to edit.

    Args:
        image: PIL Image (RGB).
        mask:  PIL Image (L). White (>127) = area to inpaint.
        mode:  One of:
            "blank_white"   — fill masked area with solid white
            "blank_gray"    — fill masked area with 50% gray
            "color_overlay" — semi-transparent magenta overlay

    Returns:
        PIL Image (RGB) – same size as *image*.
    """
    from PIL import Image
    import numpy as np

    # Ensure mask matches image size
    if mask.size != image.size:
        mask = mask.resize(image.size, Image.NEAREST)

    mask_np = np.array(mask)
    region = mask_np > 127  # boolean mask

    if mode == "blank_white":
        img_np = np.array(image.convert("RGB")).copy()
        img_np[region] = 255  # white
        return Image.fromarray(img_np, "RGB")

    elif mode == "blank_gray":
        img_np = np.array(image.convert("RGB")).copy()
        img_np[region] = 128  # neutral gray
        return Image.fromarray(img_np, "RGB")

    elif mode == "color_overlay":
        base = image.convert("RGBA")
        ov_np = np.zeros((*mask_np.shape, 4), dtype=np.uint8)
        ov_np[region] = (255, 0, 80, 140)  # bright magenta, ~55% opacity
        overlay = Image.fromarray(ov_np, "RGBA")
        composited = Image.alpha_composite(base, overlay)
        return composited.convert("RGB")

    else:
        raise ValueError(f"Unknown mask_mode: {mode!r}")


def _composite_with_mask(original, generated, mask, feather_radius=5):
    """
    Composite *generated* into *original* using *mask* with edge feathering.

    Pixels where mask is white (255) → use generated.
    Pixels where mask is black (0)   → use original.
    Feathered edge for smooth blending.

    All inputs/output are PIL Images.  *mask* is mode 'L'.
    """
    from PIL import Image, ImageFilter
    import numpy as np

    # Resize mask and generated to match original if needed
    if mask.size != original.size:
        mask = mask.resize(original.size, Image.NEAREST)
    if generated.size != original.size:
        generated = generated.resize(original.size, Image.LANCZOS)

    # Feather the mask edges for smooth blending
    if feather_radius > 0:
        blurred_mask = mask.filter(ImageFilter.GaussianBlur(radius=feather_radius))
    else:
        blurred_mask = mask

    # Composite: original * (1 - alpha) + generated * alpha
    orig_np = np.array(original.convert("RGB"), dtype=np.float32)
    gen_np = np.array(generated.convert("RGB"), dtype=np.float32)
    alpha = np.array(blurred_mask, dtype=np.float32) / 255.0
    alpha = alpha[:, :, np.newaxis]  # broadcast to RGB

    result_np = orig_np * (1.0 - alpha) + gen_np * alpha
    result_np = np.clip(result_np, 0, 255).astype(np.uint8)

    return Image.fromarray(result_np, "RGB")


class EricQwenEditInpaint:
    """
    Inpaint an image using Qwen-Image-Edit.
    
    Qwen-Edit has NO built-in mask channel.  This node simulates inpainting:
    
    1. The masked region is blanked out in the input image (white/gray/overlay).
    2. This blanked image is sent as a SINGLE image through both the VL
       (semantic) and VAE (pixel-latent) encoders — the model sees a gap
       in both semantic and pixel space and generates content to fill it.
    3. After generation, the result is composited with the original: pixels
       outside the mask come from the original, pixels inside come from
       the generated image, with feathered edges for smooth blending.
    
    Mask format:
    - White (255) = areas to inpaint/regenerate
    - Black (0) = areas to preserve
    
    Prompt tips:
    - Describe WHAT to generate, not WHERE.  The blank region tells the
      model where.
    - Good: "a logo of a black cat"
    - Avoid: "in the masked area" — the model has no mask concept.
    """
    
    CATEGORY = "Eric Qwen-Edit"
    FUNCTION = "inpaint"
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "pipeline": ("QWEN_EDIT_PIPELINE",),
                "image": ("IMAGE", {
                    "tooltip": "Image to inpaint"
                }),
                "mask": ("MASK", {
                    "tooltip": "Mask indicating areas to inpaint (white = inpaint, black = keep)"
                }),
                "prompt": ("STRING", {
                    "multiline": True,
                    "default": "a beautiful garden with flowers",
                    "tooltip": (
                        "Describe WHAT to generate in the blank area. "
                        "The model sees a blank hole — it will fill it with "
                        "whatever you describe. Don't mention 'mask'."
                    ),
                }),
            },
            "optional": {
                "negative_prompt": ("STRING", {
                    "multiline": True,
                    "default": "",
                    "tooltip": "What to avoid"
                }),
                "mask_mode": (["blank_white", "blank_gray", "color_overlay"], {
                    "default": "blank_white",
                    "tooltip": (
                        "How the masked region appears to the model. "
                        "blank_white: solid white hole (best for most cases). "
                        "blank_gray: neutral gray hole. "
                        "color_overlay: semi-transparent pink (model sees original underneath)."
                    ),
                }),
                "feather": ("INT", {
                    "default": 8,
                    "min": 0,
                    "max": 50,
                    "step": 1,
                    "tooltip": (
                        "Gaussian blur radius for mask-edge feathering during "
                        "post-compositing. Higher = softer blend between "
                        "generated and original pixels at mask edges. 0 = hard edge."
                    ),
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
                    "tooltip": "True CFG scale"
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
                    "tooltip": "Max output megapixels. Lower = faster."
                }),
            }
        }
    
    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return kwargs.get("seed", 0)
    
    def inpaint(
        self,
        pipeline: dict,
        image: torch.Tensor,
        mask: torch.Tensor,
        prompt: str,
        negative_prompt: str = "",
        mask_mode: str = "blank_white",
        feather: int = 8,
        steps: int = 50,
        true_cfg_scale: float = 4.0,
        seed: int = 0,
        max_mp: float = 8.0,
    ) -> Tuple[torch.Tensor]:
        """
        Inpaint: blank the masked region → generate → composite back.

        The blanked image goes through BOTH the VL and VAE encoders as a
        single image so the model sees the gap in semantic AND latent space.
        After generation the output is composited with the original using the
        mask so that unmasked areas are pixel-perfect from the original.
        """
        pipe = pipeline["pipeline"]
        offload_vae = pipeline.get("offload_vae", False)
        
        # Convert image
        if image.dim() == 4:
            pil_image = tensor_to_pil(image[0])
        else:
            pil_image = tensor_to_pil(image)
        pil_image = prepare_image_for_pipeline(pil_image)
        
        # Convert mask to PIL L-mode
        if mask.dim() == 3:
            mask_np = mask[0].cpu().numpy()
        else:
            mask_np = mask.cpu().numpy()
        
        from PIL import Image
        import numpy as np
        mask_pil = Image.fromarray((mask_np * 255).astype(np.uint8), mode='L')
        
        # Build the blanked image — sent to BOTH VL and VAE paths
        blanked_image = _apply_mask_to_image(pil_image, mask_pil, mode=mask_mode)
        
        input_w, input_h = pil_image.size
        input_mp = (input_w * input_h) / (1024 * 1024)
        effective_mp = min(input_mp, max_mp)
        
        # Build prompt that references the visible blank
        if mask_mode in ("blank_white", "blank_gray"):
            blank_desc = "white" if mask_mode == "blank_white" else "gray"
            enhanced_prompt = (
                f"Replace the {blank_desc} blank area in the image with: {prompt}"
            )
        else:  # color_overlay
            enhanced_prompt = (
                f"Modify the area highlighted in pink/red in the image: {prompt}"
            )
        
        print(f"[EricQwenEdit] Inpaint (mode={mask_mode}, feather={feather})")
        print(f"[EricQwenEdit] Input: {input_w}x{input_h} ({input_mp:.1f}MP)")
        print(f"[EricQwenEdit] Effective: ~{effective_mp:.1f}MP, Steps: {steps}, CFG: {true_cfg_scale}")
        print(f"[EricQwenEdit] Prompt: {enhanced_prompt[:100]}...")
        
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
        
        try:
            with torch.inference_mode():
                # Single blanked image goes through BOTH VL and VAE paths.
                # The model sees the gap everywhere — semantic AND pixel-latent.
                output = pipe(
                    prompt=enhanced_prompt,
                    image=blanked_image,
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
        
        generated = output.images[0]
        
        # Post-composite: keep original pixels outside mask, generated inside
        result = _composite_with_mask(pil_image, generated, mask_pil, feather_radius=feather)
        
        result_tensor = pil_to_tensor(result).unsqueeze(0)
        
        print(f"[EricQwenEdit] Inpaint complete: {result.size[0]}x{result.size[1]}")
        print(f"[EricQwenEdit] Actual time: {elapsed/60:.1f} minutes ({elapsed/steps:.1f} sec/step)")
        
        return (result_tensor,)
