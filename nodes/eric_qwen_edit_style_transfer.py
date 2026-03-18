# Copyright (c) 2026 Eric Hiss. All rights reserved.
# Licensed under the terms in LICENSE.txt (CC BY-NC 4.0 / Commercial dual license).
# https://github.com/EricRollei/Eric_Qwen_Edit_Experiments
"""
Eric Qwen-Edit Style Transfer Node
Apply the style of one image to the content of another using Qwen-Image-Edit.

Uses Qwen-Image-Edit-2511's multi-image support with style-focused prompt templates
to transfer artistic styles, color palettes, textures, and visual aesthetics.

The style_image (Picture 1) provides the aesthetic; the content_image (Picture 2)
provides the subject. By default, only the content image goes through the VAE/ref
path (ref_style=False, ref_content=True) so the model preserves the content's
structure while applying style from the VL/semantic path.

Model Credits:
- Qwen-Image-Edit developed by Qwen Team (Alibaba)
- https://github.com/QwenLM/Qwen-Image

Author: Eric Hiss (GitHub: EricRollei)
"""

import torch
import time
from typing import Tuple

import comfy.utils

from .eric_qwen_edit_utils import (
    tensor_to_pil,
    pil_to_tensor,
    prepare_image_for_pipeline,
)


# Pre-built style transfer prompt templates
STYLE_TEMPLATES = {
    "full_style": (
        "Apply the complete artistic style, color palette, textures, and visual "
        "aesthetics from Picture 1 to the content and composition of Picture 2. "
        "Maintain the subjects and layout of Picture 2 while transforming the visual "
        "style to match Picture 1."
    ),
    "color_palette": (
        "Apply the color palette and color grading from Picture 1 to Picture 2. "
        "Keep the content, composition, and details of Picture 2 unchanged, but "
        "transform the colors to match the tones and hues of Picture 1."
    ),
    "lighting": (
        "Apply the lighting style, mood, and atmosphere from Picture 1 to Picture 2. "
        "Keep the content of Picture 2 but transform the lighting to match Picture 1."
    ),
    "artistic_medium": (
        "Transform Picture 2 to look as if it were created using the same artistic "
        "medium and technique as Picture 1 (e.g., oil painting, watercolor, sketch, "
        "digital art). Preserve the subject matter of Picture 2."
    ),
    "texture": (
        "Apply the surface textures and material qualities from Picture 1 to the "
        "elements in Picture 2. Maintain the composition of Picture 2."
    ),
    "custom": "",
}


class EricQwenEditStyleTransfer:
    """
    Apply the style of a reference image to a content image.

    Uses Qwen-Image-Edit-2511's multi-image support with dual conditioning paths:
    - VL path (semantic): Both images go through text encoder by default
    - VAE/ref path (pixel): Only content image by default (controls structure)

    This means the style image influences the output semantically (colors, mood,
    technique) while the content image provides pixel-level structure.

    Toggle ref_style=True if you want the style image to also inject pixel-level
    detail (stronger style transfer but may alter content structure).

    Style modes:
    - full_style: Transfer complete visual style (colors, textures, mood)
    - color_palette: Transfer only the color grading/palette
    - lighting: Transfer lighting and atmosphere
    - artistic_medium: Transfer the art medium/technique
    - texture: Transfer surface textures and materials
    - custom: Write your own transfer prompt

    The style_image is passed as Picture 1 and the content_image as Picture 2.
    """

    CATEGORY = "Eric Qwen-Edit"
    FUNCTION = "style_transfer"
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "pipeline": ("QWEN_EDIT_PIPELINE",),
                "style_image": ("IMAGE", {
                    "tooltip": "Reference image providing the style (passed as Picture 1)"
                }),
                "content_image": ("IMAGE", {
                    "tooltip": "Image to apply the style to (passed as Picture 2)"
                }),
                "style_mode": (list(STYLE_TEMPLATES.keys()), {
                    "default": "full_style",
                    "tooltip": "Type of style transfer to apply"
                }),
            },
            "optional": {
                "custom_prompt": ("STRING", {
                    "multiline": True,
                    "default": "",
                    "tooltip": "When non-empty, overrides the style_mode template entirely. "
                               "Reference images as Picture 1 (style) and Picture 2 (content)."
                }),
                "additional_guidance": ("STRING", {
                    "multiline": True,
                    "default": "",
                    "tooltip": "Extra instructions appended to the style prompt (e.g., 'but keep faces realistic')"
                }),
                "style_strength": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.1,
                    "max": 2.0,
                    "step": 0.1,
                    "tooltip": "How strongly to apply the style (affects CFG emphasis). "
                               "1.0=balanced, >1=stronger style, <1=more original content"
                }),
                "vae_target_size": ("INT", {
                    "default": 1024,
                    "min": 0,
                    "max": 2048,
                    "step": 64,
                    "tooltip": "Fixed resolution for VAE encoding. 0 (default) = encode refs "
                               "at output resolution (matches Edit node behavior, best for high-res). "
                               "Set to e.g. 1024 to force refs to ~1MP (only useful at low output res)."
                }),
                # Per-image conditioning toggles
                "vl_style": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Include style image in VL/semantic path (text encoder sees its content)"
                }),
                "vl_content": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Include content image in VL/semantic path"
                }),
                "ref_style": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Include style image in VAE/ref path (pixel-level latent). "
                               "Default False — style is semantic-only, preventing structure bleed."
                }),
                "ref_content": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Include content image in VAE/ref path (preserves its pixel structure)"
                }),
                "negative_prompt": ("STRING", {
                    "multiline": True,
                    "default": "blurry, distorted, low quality",
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
                    "tooltip": "Max output megapixels. VAE refs scale to match output resolution."
                }),
            }
        }

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return kwargs.get("seed", 0)

    def style_transfer(
        self,
        pipeline: dict,
        style_image: torch.Tensor,
        content_image: torch.Tensor,
        style_mode: str,
        custom_prompt: str = "",
        additional_guidance: str = "",
        style_strength: float = 1.0,
        vae_target_size: int = 1024,
        vl_style: bool = True,
        vl_content: bool = True,
        ref_style: bool = False,
        ref_content: bool = True,
        negative_prompt: str = "blurry, distorted, low quality",
        steps: int = 50,
        true_cfg_scale: float = 4.0,
        seed: int = 0,
        max_mp: float = 8.0,
    ) -> Tuple[torch.Tensor]:
        """Apply style transfer using Qwen-Edit with per-image conditioning control."""
        pipe = pipeline["pipeline"]
        offload_vae = pipeline.get("offload_vae", False)

        # Convert images to PIL
        if style_image.dim() == 4:
            style_pil = tensor_to_pil(style_image[0])
        else:
            style_pil = tensor_to_pil(style_image)
        style_pil = prepare_image_for_pipeline(style_pil)

        if content_image.dim() == 4:
            content_pil = tensor_to_pil(content_image[0])
        else:
            content_pil = tensor_to_pil(content_image)
        content_pil = prepare_image_for_pipeline(content_pil)

        # Build the prompt — custom_prompt always overrides when non-empty
        if custom_prompt and custom_prompt.strip():
            prompt = custom_prompt.strip()
        else:
            prompt = STYLE_TEMPLATES.get(style_mode, STYLE_TEMPLATES["full_style"])

        if additional_guidance and additional_guidance.strip():
            prompt = f"{prompt} {additional_guidance.strip()}"

        # Adjust CFG based on style_strength
        effective_cfg = true_cfg_scale * style_strength

        # Per-image flags: [style_image, content_image]
        image_vl_flags = [vl_style, vl_content]
        image_ref_flags = [ref_style, ref_content]

        # Content image is main (index 1) if it has ref; otherwise style (index 0)
        main_image_index = 1 if ref_content else 0

        # Ensure at least one image has ref=True
        if not ref_style and not ref_content:
            print("[EricQwenEdit] Warning: no ref images. Forcing ref_content=True.")
            image_ref_flags[1] = True
            main_image_index = 1

        # vae_target_size=0 means dynamic
        effective_vae_size = vae_target_size if vae_target_size > 0 else None

        style_vl = "VL" if vl_style else "--"
        style_ref = "REF" if ref_style else "---"
        content_vl = "VL" if vl_content else "--"
        content_ref = "REF" if ref_content else "---"

        print(f"[EricQwenEdit] Style Transfer")
        print(f"[EricQwenEdit] Style image:   {style_pil.size[0]}x{style_pil.size[1]} [{style_vl}+{style_ref}]")
        print(f"[EricQwenEdit] Content image: {content_pil.size[0]}x{content_pil.size[1]} [{content_vl}+{content_ref}] [MAIN]")
        print(f"[EricQwenEdit] VAE target size: {effective_vae_size or 'dynamic'}")
        print(f"[EricQwenEdit] Mode: {style_mode}, Strength: {style_strength}, CFG: {effective_cfg:.1f}")
        print(f"[EricQwenEdit] Prompt: {prompt[:100]}...")

        device = next(pipe.transformer.parameters()).device
        generator = torch.Generator(device=device).manual_seed(seed)

        # Handle VAE offload
        vae_device_original = None
        if offload_vae:
            vae_device_original = next(pipe.vae.parameters()).device
            if str(vae_device_original) == "cpu":
                pipe.vae = pipe.vae.to(device)

        start_time = time.time()

        # ComfyUI progress bar
        pbar = comfy.utils.ProgressBar(steps)
        def _progress_callback(pipeline, step_index, timestep, cb_kwargs):
            pbar.update(1)
            return cb_kwargs

        try:
            with torch.inference_mode():
                output = pipe(
                    prompt=prompt,
                    image=[style_pil, content_pil],
                    max_pixels=int(max_mp * 1024 * 1024),
                    negative_prompt=negative_prompt if negative_prompt else " ",
                    num_inference_steps=steps,
                    true_cfg_scale=effective_cfg,
                    generator=generator,
                    num_images_per_prompt=1,
                    # Per-image conditioning controls
                    vae_target_size=effective_vae_size,
                    main_image_index=main_image_index,
                    image_vl_flags=image_vl_flags,
                    image_ref_flags=image_ref_flags,
                    callback_on_step_end=_progress_callback,
                    callback_on_step_end_tensor_inputs=["latents"],
                )
        finally:
            if offload_vae and vae_device_original is not None and str(vae_device_original) == "cpu":
                pipe.vae = pipe.vae.to("cpu")
                torch.cuda.empty_cache()

        elapsed = time.time() - start_time

        result = output.images[0]
        result_tensor = pil_to_tensor(result).unsqueeze(0)

        print(f"[EricQwenEdit] Style transfer complete: {result.size[0]}x{result.size[1]}")
        print(f"[EricQwenEdit] Actual time: {elapsed/60:.1f} minutes ({elapsed/steps:.1f} sec/step)")

        return (result_tensor,)
