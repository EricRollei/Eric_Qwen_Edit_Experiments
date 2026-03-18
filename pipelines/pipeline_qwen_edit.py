# Copyright (c) 2026 Eric Hiss. All rights reserved.
# Licensed under the terms in LICENSE.txt (CC BY-NC 4.0 / Commercial dual license).
# https://github.com/EricRollei/Eric_Qwen_Edit_Experiments
#
# Pipeline code derived from HuggingFace Diffusers (Apache 2.0).
# Original Qwen-Image pipeline: Copyright 2025 Qwen Team.

"""
Qwen-Edit Pipeline with Smart Resolution Handling

This pipeline provides intelligent resolution handling that preserves input image
dimensions (up to a configurable maximum) rather than forcing a fixed output size.

Key improvements over base QwenImageEditPlusPipeline:
- Preserves input resolution by default (aligned to 32px)
- Configurable max_pixels cap for VRAM safety
- Explicit height/width override support
- Clean, well-documented codebase
"""

import inspect
import math
from typing import Any, Callable, Dict, List, Optional, Union

import numpy as np
import torch
from transformers import Qwen2_5_VLForConditionalGeneration, Qwen2Tokenizer, Qwen2VLProcessor

from diffusers.image_processor import PipelineImageInput, VaeImageProcessor
from diffusers.loaders import QwenImageLoraLoaderMixin
from diffusers.models import AutoencoderKLQwenImage, QwenImageTransformer2DModel
from diffusers.schedulers import FlowMatchEulerDiscreteScheduler
from diffusers.utils import logging
from diffusers.utils.torch_utils import randn_tensor
from diffusers.pipelines.pipeline_utils import DiffusionPipeline

from .pipeline_output import QwenEditPipelineOutput


logger = logging.get_logger(__name__)

# Default constants
CONDITION_IMAGE_SIZE = 384 * 384  # Size for condition encoder input
DEFAULT_MAX_PIXELS = 16 * 1024 * 1024  # 16MP default cap
DIMENSION_ALIGNMENT = 32  # Must be divisible by this


def calculate_shift(
    image_seq_len: int,
    base_seq_len: int = 256,
    max_seq_len: int = 4096,
    base_shift: float = 0.5,
    max_shift: float = 1.15,
) -> float:
    """Calculate timestep shift based on image sequence length."""
    m = (max_shift - base_shift) / (max_seq_len - base_seq_len)
    b = base_shift - m * base_seq_len
    mu = image_seq_len * m + b
    return mu


def retrieve_timesteps(
    scheduler,
    num_inference_steps: Optional[int] = None,
    device: Optional[Union[str, torch.device]] = None,
    timesteps: Optional[List[int]] = None,
    sigmas: Optional[List[float]] = None,
    **kwargs,
):
    """
    Retrieve timesteps from scheduler, handling custom timesteps/sigmas.
    
    Returns:
        Tuple[torch.Tensor, int]: Timestep schedule and number of inference steps.
    """
    if timesteps is not None and sigmas is not None:
        raise ValueError("Only one of `timesteps` or `sigmas` can be passed.")
    
    if timesteps is not None:
        accepts_timesteps = "timesteps" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
        if not accepts_timesteps:
            raise ValueError(f"Scheduler {scheduler.__class__} doesn't support custom timesteps.")
        scheduler.set_timesteps(timesteps=timesteps, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    elif sigmas is not None:
        accept_sigmas = "sigmas" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
        if not accept_sigmas:
            raise ValueError(f"Scheduler {scheduler.__class__} doesn't support custom sigmas.")
        scheduler.set_timesteps(sigmas=sigmas, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    else:
        scheduler.set_timesteps(num_inference_steps, device=device, **kwargs)
        timesteps = scheduler.timesteps
    
    return timesteps, num_inference_steps


def retrieve_latents(
    encoder_output: torch.Tensor, 
    generator: Optional[torch.Generator] = None, 
    sample_mode: str = "sample"
) -> torch.Tensor:
    """Extract latents from VAE encoder output."""
    if hasattr(encoder_output, "latent_dist") and sample_mode == "sample":
        return encoder_output.latent_dist.sample(generator)
    elif hasattr(encoder_output, "latent_dist") and sample_mode == "argmax":
        return encoder_output.latent_dist.mode()
    elif hasattr(encoder_output, "latents"):
        return encoder_output.latents
    else:
        raise AttributeError("Could not access latents of provided encoder_output")


def calculate_dimensions(target_pixels: int, aspect_ratio: float) -> tuple:
    """
    Calculate width and height for a target pixel count and aspect ratio.
    
    Args:
        target_pixels: Target total pixel count
        aspect_ratio: Width / height ratio
        
    Returns:
        Tuple of (width, height) aligned to DIMENSION_ALIGNMENT
    """
    width = math.sqrt(target_pixels * aspect_ratio)
    height = width / aspect_ratio
    
    width = round(width / DIMENSION_ALIGNMENT) * DIMENSION_ALIGNMENT
    height = round(height / DIMENSION_ALIGNMENT) * DIMENSION_ALIGNMENT
    
    return int(width), int(height)


def crop_to_cover(img, target_w: int, target_h: int):
    """
    Scale-to-cover then center-crop a PIL image to exact (target_w, target_h).
    
    The image is scaled uniformly so it fully covers the target dimensions
    (the smaller scaling axis determines the scale factor), then the excess
    is center-cropped.  This avoids any aspect-ratio distortion.
    
    Args:
        img: PIL Image
        target_w: Desired output width
        target_h: Desired output height
        
    Returns:
        Cropped PIL Image at exactly (target_w, target_h)
    """
    from PIL import Image
    
    src_w, src_h = img.size
    src_aspect = src_w / src_h
    target_aspect = target_w / target_h
    
    # Scale so the image fully covers the target rectangle
    if src_aspect > target_aspect:
        # Source is wider — scale by height, crop width
        scale = target_h / src_h
    else:
        # Source is taller — scale by width, crop height
        scale = target_w / src_w
    
    scaled_w = round(src_w * scale)
    scaled_h = round(src_h * scale)
    
    # Ensure at least target size (rounding might shave a pixel)
    scaled_w = max(scaled_w, target_w)
    scaled_h = max(scaled_h, target_h)
    
    resized = img.resize((scaled_w, scaled_h), Image.LANCZOS)
    
    # Center-crop
    left = (scaled_w - target_w) // 2
    top = (scaled_h - target_h) // 2
    return resized.crop((left, top, left + target_w, top + target_h))


def compute_output_dimensions(
    input_width: int,
    input_height: int,
    target_width: Optional[int] = None,
    target_height: Optional[int] = None,
    max_pixels: int = DEFAULT_MAX_PIXELS,
) -> tuple:
    """
    Compute output dimensions with smart defaults.
    
    Resolution handling logic:
    1. If explicit width/height provided, use those (aligned to 32)
    2. Otherwise, preserve input resolution
    3. If input exceeds max_pixels, scale down proportionally
    4. Always align final dimensions to DIMENSION_ALIGNMENT
    
    Args:
        input_width: Input image width
        input_height: Input image height
        target_width: Explicit target width (optional)
        target_height: Explicit target height (optional)
        max_pixels: Maximum allowed pixels (VRAM safety cap)
        
    Returns:
        Tuple of (output_width, output_height)
    """
    # If explicit dimensions provided, use them
    if target_width is not None and target_height is not None:
        out_w = target_width // DIMENSION_ALIGNMENT * DIMENSION_ALIGNMENT
        out_h = target_height // DIMENSION_ALIGNMENT * DIMENSION_ALIGNMENT
        return out_w, out_h
    
    # Preserve input resolution
    input_pixels = input_width * input_height
    aspect_ratio = input_width / input_height
    
    if input_pixels > max_pixels:
        # Scale down to fit within budget
        scale = math.sqrt(max_pixels / input_pixels)
        out_w = int(input_width * scale)
        out_h = int(input_height * scale)
    else:
        # Preserve input size
        out_w = input_width
        out_h = input_height
    
    # Align to required divisor
    out_w = out_w // DIMENSION_ALIGNMENT * DIMENSION_ALIGNMENT
    out_h = out_h // DIMENSION_ALIGNMENT * DIMENSION_ALIGNMENT
    
    # Ensure minimum size
    out_w = max(out_w, DIMENSION_ALIGNMENT)
    out_h = max(out_h, DIMENSION_ALIGNMENT)
    
    return out_w, out_h


class QwenEditPipeline(DiffusionPipeline, QwenImageLoraLoaderMixin):
    """
    Qwen-Edit Pipeline with smart resolution handling.
    
    This pipeline supports image editing with:
    - Preserved input resolution (no forced upscaling)
    - Configurable max_pixels cap for VRAM safety
    - VAE tiling support for high-resolution decode
    - Explicit dimension override when needed
    
    Compatible with:
    - Qwen-Image-Edit-2509
    - Qwen-Image-Edit-2511
    - Other Qwen-Image-Edit variants
    
    Args:
        transformer: MMDiT transformer for denoising
        scheduler: Flow matching scheduler
        vae: VAE for encoding/decoding images
        text_encoder: Qwen2.5-VL for multimodal encoding
        tokenizer: Qwen tokenizer
        processor: Qwen2VL processor
    """
    
    model_cpu_offload_seq = "text_encoder->transformer->vae"
    _callback_tensor_inputs = ["latents", "prompt_embeds"]
    
    def __init__(
        self,
        scheduler: FlowMatchEulerDiscreteScheduler,
        vae: AutoencoderKLQwenImage,
        text_encoder: Qwen2_5_VLForConditionalGeneration,
        tokenizer: Qwen2Tokenizer,
        processor: Qwen2VLProcessor,
        transformer: QwenImageTransformer2DModel,
    ):
        super().__init__()
        
        self.register_modules(
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            processor=processor,
            transformer=transformer,
            scheduler=scheduler,
        )
        
        self.vae_scale_factor = 2 ** len(self.vae.temperal_downsample) if getattr(self, "vae", None) else 8
        self.latent_channels = self.vae.config.z_dim if getattr(self, "vae", None) else 16
        self.image_processor = VaeImageProcessor(vae_scale_factor=self.vae_scale_factor * 2)
        self.tokenizer_max_length = 1024
        
        # Prompt template for image editing
        self.prompt_template = (
            "<|im_start|>system\n"
            "Describe the key features of the input image (color, shape, size, texture, objects, background), "
            "then explain how the user's text instruction should alter or modify the image. "
            "Generate a new image that meets the user's requirements while maintaining consistency with the "
            "original input where appropriate.<|im_end|>\n"
            "<|im_start|>user\n{}<|im_end|>\n"
            "<|im_start|>assistant\n"
        )
        self.prompt_template_start_idx = 64
    
    def _extract_masked_hidden(self, hidden_states: torch.Tensor, mask: torch.Tensor) -> list:
        """Extract hidden states using attention mask."""
        bool_mask = mask.bool()
        valid_lengths = bool_mask.sum(dim=1)
        selected = hidden_states[bool_mask]
        split_result = torch.split(selected, valid_lengths.tolist(), dim=0)
        return split_result
    
    def _get_prompt_embeds(
        self,
        prompt: Union[str, List[str]],
        image: Optional[torch.Tensor] = None,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> tuple:
        """Generate prompt embeddings with optional image conditioning."""
        device = device or self._execution_device
        dtype = dtype or self.text_encoder.dtype
        
        prompt = [prompt] if isinstance(prompt, str) else prompt
        
        # Build image prompt section
        img_template = "Picture {}: <|vision_start|><|image_pad|><|vision_end|>"
        if isinstance(image, list):
            img_prompt = "".join(img_template.format(i + 1) for i in range(len(image)))
        elif image is not None:
            img_prompt = img_template.format(1)
        else:
            img_prompt = ""
        
        # Format full prompts
        txt = [self.prompt_template.format(img_prompt + p) for p in prompt]
        
        # Process through encoder
        model_inputs = self.processor(
            text=txt,
            images=image,
            padding=True,
            return_tensors="pt",
        ).to(device)
        
        outputs = self.text_encoder(
            input_ids=model_inputs.input_ids,
            attention_mask=model_inputs.attention_mask,
            pixel_values=model_inputs.pixel_values,
            image_grid_thw=model_inputs.image_grid_thw,
            output_hidden_states=True,
        )
        
        hidden_states = outputs.hidden_states[-1]
        split_hidden = self._extract_masked_hidden(hidden_states, model_inputs.attention_mask)
        split_hidden = [h[self.prompt_template_start_idx:] for h in split_hidden]
        
        # Pad to max sequence length
        attn_masks = [torch.ones(h.size(0), dtype=torch.long, device=h.device) for h in split_hidden]
        max_len = max(h.size(0) for h in split_hidden)
        
        prompt_embeds = torch.stack([
            torch.cat([h, h.new_zeros(max_len - h.size(0), h.size(1))]) 
            for h in split_hidden
        ])
        attention_mask = torch.stack([
            torch.cat([m, m.new_zeros(max_len - m.size(0))]) 
            for m in attn_masks
        ])
        
        return prompt_embeds.to(dtype=dtype, device=device), attention_mask
    
    def encode_prompt(
        self,
        prompt: Union[str, List[str]],
        image: Optional[torch.Tensor] = None,
        device: Optional[torch.device] = None,
        num_images_per_prompt: int = 1,
        prompt_embeds: Optional[torch.Tensor] = None,
        prompt_embeds_mask: Optional[torch.Tensor] = None,
        max_sequence_length: int = 1024,
    ) -> tuple:
        """Encode prompt with optional image conditioning."""
        device = device or self._execution_device
        prompt = [prompt] if isinstance(prompt, str) else prompt
        batch_size = len(prompt) if prompt_embeds is None else prompt_embeds.shape[0]
        
        if prompt_embeds is None:
            prompt_embeds, prompt_embeds_mask = self._get_prompt_embeds(prompt, image, device)
        
        _, seq_len, _ = prompt_embeds.shape
        prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
        prompt_embeds = prompt_embeds.view(batch_size * num_images_per_prompt, seq_len, -1)
        prompt_embeds_mask = prompt_embeds_mask.repeat(1, num_images_per_prompt, 1)
        prompt_embeds_mask = prompt_embeds_mask.view(batch_size * num_images_per_prompt, seq_len)
        
        return prompt_embeds, prompt_embeds_mask
    
    def check_inputs(
        self,
        prompt,
        height,
        width,
        negative_prompt=None,
        prompt_embeds=None,
        negative_prompt_embeds=None,
        prompt_embeds_mask=None,
        negative_prompt_embeds_mask=None,
        callback_on_step_end_tensor_inputs=None,
        max_sequence_length=None,
    ):
        """Validate pipeline inputs."""
        multiple_of = self.vae_scale_factor * 2
        if height % multiple_of != 0 or width % multiple_of != 0:
            logger.warning(
                f"height and width should be divisible by {multiple_of}. "
                f"Got {height}x{width}, will be adjusted."
            )
        
        if callback_on_step_end_tensor_inputs is not None:
            invalid = [k for k in callback_on_step_end_tensor_inputs if k not in self._callback_tensor_inputs]
            if invalid:
                raise ValueError(f"Invalid callback tensor inputs: {invalid}")
        
        if prompt is not None and prompt_embeds is not None:
            raise ValueError("Cannot provide both prompt and prompt_embeds")
        elif prompt is None and prompt_embeds is None:
            raise ValueError("Must provide either prompt or prompt_embeds")
        elif prompt is not None and not isinstance(prompt, (str, list)):
            raise ValueError(f"prompt must be str or list, got {type(prompt)}")
        
        if negative_prompt is not None and negative_prompt_embeds is not None:
            raise ValueError("Cannot provide both negative_prompt and negative_prompt_embeds")
        
        if prompt_embeds is not None and prompt_embeds_mask is None:
            raise ValueError("prompt_embeds requires prompt_embeds_mask")
        if negative_prompt_embeds is not None and negative_prompt_embeds_mask is None:
            raise ValueError("negative_prompt_embeds requires negative_prompt_embeds_mask")
        
        if max_sequence_length is not None and max_sequence_length > 1024:
            raise ValueError(f"max_sequence_length cannot exceed 1024, got {max_sequence_length}")
    
    @staticmethod
    def _pack_latents(latents, batch_size, num_channels, height, width):
        """Pack latents into sequence format for transformer."""
        latents = latents.view(batch_size, num_channels, height // 2, 2, width // 2, 2)
        latents = latents.permute(0, 2, 4, 1, 3, 5)
        latents = latents.reshape(batch_size, (height // 2) * (width // 2), num_channels * 4)
        return latents
    
    @staticmethod
    def _unpack_latents(latents, height, width, vae_scale_factor):
        """Unpack latents from sequence format back to spatial."""
        batch_size, num_patches, channels = latents.shape
        
        height = 2 * (int(height) // (vae_scale_factor * 2))
        width = 2 * (int(width) // (vae_scale_factor * 2))
        
        latents = latents.view(batch_size, height // 2, width // 2, channels // 4, 2, 2)
        latents = latents.permute(0, 3, 1, 4, 2, 5)
        latents = latents.reshape(batch_size, channels // 4, 1, height, width)
        
        return latents
    
    def _encode_vae_image(self, image: torch.Tensor, generator: torch.Generator) -> torch.Tensor:
        """Encode image through VAE."""
        if isinstance(generator, list):
            latents = [
                retrieve_latents(self.vae.encode(image[i:i+1]), generator=generator[i], sample_mode="argmax")
                for i in range(image.shape[0])
            ]
            latents = torch.cat(latents, dim=0)
        else:
            latents = retrieve_latents(self.vae.encode(image), generator=generator, sample_mode="argmax")
        
        # Normalize
        latents_mean = (
            torch.tensor(self.vae.config.latents_mean)
            .view(1, self.latent_channels, 1, 1, 1)
            .to(latents.device, latents.dtype)
        )
        latents_std = (
            torch.tensor(self.vae.config.latents_std)
            .view(1, self.latent_channels, 1, 1, 1)
            .to(latents.device, latents.dtype)
        )
        latents = (latents - latents_mean) / latents_std
        
        return latents
    
    def prepare_latents(
        self,
        images,
        batch_size,
        num_channels,
        height,
        width,
        dtype,
        device,
        generator,
        latents=None,
    ) -> tuple:
        """Prepare noise latents and encode input images."""
        height = 2 * (int(height) // (self.vae_scale_factor * 2))
        width = 2 * (int(width) // (self.vae_scale_factor * 2))
        
        shape = (batch_size, 1, num_channels, height, width)
        
        # Encode input images
        image_latents = None
        if images is not None:
            if not isinstance(images, list):
                images = [images]
            
            all_latents = []
            for image in images:
                image = image.to(device=device, dtype=dtype)
                if image.shape[1] != self.latent_channels:
                    img_latents = self._encode_vae_image(image, generator)
                else:
                    img_latents = image
                
                # Handle batch expansion
                if batch_size > img_latents.shape[0] and batch_size % img_latents.shape[0] == 0:
                    img_latents = torch.cat([img_latents] * (batch_size // img_latents.shape[0]), dim=0)
                elif batch_size > img_latents.shape[0]:
                    raise ValueError(f"Cannot expand batch size {img_latents.shape[0]} to {batch_size}")
                
                h, w = img_latents.shape[3:]
                img_latents = self._pack_latents(img_latents, batch_size, num_channels, h, w)
                all_latents.append(img_latents)
            
            image_latents = torch.cat(all_latents, dim=1)
        
        # Generate noise latents
        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(f"Generator list length {len(generator)} != batch size {batch_size}")
        
        if latents is None:
            latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
            latents = self._pack_latents(latents, batch_size, num_channels, height, width)
        else:
            latents = latents.to(device=device, dtype=dtype)
        
        return latents, image_latents
    
    @property
    def guidance_scale(self):
        return self._guidance_scale
    
    @property
    def attention_kwargs(self):
        return self._attention_kwargs
    
    @property
    def num_timesteps(self):
        return self._num_timesteps
    
    @property
    def current_timestep(self):
        return self._current_timestep
    
    @property
    def interrupt(self):
        return self._interrupt
    
    @torch.no_grad()
    def __call__(
        self,
        image: Optional[PipelineImageInput] = None,
        prompt: Union[str, List[str]] = None,
        negative_prompt: Union[str, List[str]] = None,
        true_cfg_scale: float = 4.0,
        height: Optional[int] = None,
        width: Optional[int] = None,
        max_pixels: int = DEFAULT_MAX_PIXELS,
        num_inference_steps: int = 50,
        sigmas: Optional[List[float]] = None,
        guidance_scale: Optional[float] = None,
        num_images_per_prompt: int = 1,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.Tensor] = None,
        prompt_embeds: Optional[torch.Tensor] = None,
        prompt_embeds_mask: Optional[torch.Tensor] = None,
        negative_prompt_embeds: Optional[torch.Tensor] = None,
        negative_prompt_embeds_mask: Optional[torch.Tensor] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        attention_kwargs: Optional[Dict[str, Any]] = None,
        callback_on_step_end: Optional[Callable[[int, int, Dict], None]] = None,
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
        max_sequence_length: int = 512,
        # Per-image conditioning controls (multi-image only)
        vae_target_size: Optional[int] = None,
        main_image_index: int = 0,
        image_vl_flags: Optional[List[bool]] = None,
        image_ref_flags: Optional[List[bool]] = None,
    ):
        """
        Generate edited images using Qwen-Edit.
        
        Resolution handling:
        - If height/width provided: use those (aligned to 32)
        - Otherwise: preserve input resolution up to max_pixels cap
        
        Args:
            image: Input image(s) to edit
            prompt: Text prompt describing the edit
            negative_prompt: What to avoid
            true_cfg_scale: CFG scale (>1 enables CFG)
            height: Output height (optional, derived from input if None)
            width: Output width (optional, derived from input if None)
            max_pixels: Max pixels cap (default 16MP)
            num_inference_steps: Denoising steps
            generator: Random generator
            output_type: "pil", "np", or "latent"
            vae_target_size: Fixed resolution for VAE encoding (e.g. 1024).
                None = encode each ref at output-proportional resolution (default).
                Only set this if you want to force all refs to a specific size.
            main_image_index: Which image (0-based) is the primary reference.
                Its VAE latent is used as the main latent output. Default 0.
            image_vl_flags: Per-image list of bools controlling VL (semantic)
                encoding. None = all True. Set False to skip VL for an image.
            image_ref_flags: Per-image list of bools controlling VAE (pixel)
                encoding. None = all True. Set False to skip VAE ref for an image.
            
        Returns:
            QwenEditPipelineOutput with generated images
        """
        # Get input dimensions
        if isinstance(image, list):
            input_image = image[-1]
        else:
            input_image = image
        input_width, input_height = input_image.size
        
        # Compute output dimensions
        output_width, output_height = compute_output_dimensions(
            input_width, input_height, width, height, max_pixels
        )
        height, width = output_height, output_width
        
        # Align
        multiple_of = self.vae_scale_factor * 2
        width = width // multiple_of * multiple_of
        height = height // multiple_of * multiple_of
        
        logger.info(f"QwenEdit: {input_width}x{input_height} -> {width}x{height}")
        
        # Validate
        self.check_inputs(prompt, height, width, negative_prompt, prompt_embeds,
                         negative_prompt_embeds, prompt_embeds_mask, 
                         negative_prompt_embeds_mask, callback_on_step_end_tensor_inputs,
                         max_sequence_length)
        
        self._guidance_scale = guidance_scale
        self._attention_kwargs = attention_kwargs or {}
        self._current_timestep = None
        self._interrupt = False
        
        # Batch size
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]
        
        device = self._execution_device
        
        # Preprocess images
        if image is not None and not (isinstance(image, torch.Tensor) and image.size(1) == self.latent_channels):
            if not isinstance(image, list):
                image = [image]
            
            num_images = len(image)
            
            # Default flags: all images go through both paths
            if image_vl_flags is None:
                image_vl_flags = [True] * num_images
            if image_ref_flags is None:
                image_ref_flags = [True] * num_images
            
            # Clamp main_image_index
            if main_image_index >= num_images:
                logger.warning(f"main_image_index={main_image_index} >= num_images={num_images}, resetting to 0")
                main_image_index = 0
            
            condition_images = []
            vae_images = []
            vae_image_sizes = []
            
            # First pass: determine the main image's VAE target dimensions
            # so we can crop-to-cover secondaries to the same spatial shape.
            main_img = image[main_image_index]
            main_w, main_h = main_img.size
            if vae_target_size is not None:
                vae_pixels = vae_target_size * vae_target_size
                main_vae_w, main_vae_h = calculate_dimensions(vae_pixels, main_w / main_h)
            else:
                main_vae_w, main_vae_h = compute_output_dimensions(main_w, main_h, max_pixels=max_pixels)
            
            for idx, img in enumerate(image):
                img_w, img_h = img.size
                
                # VL path: semantic conditioning through text encoder
                # (each image keeps its own aspect ratio at ~384px — no crop needed)
                if idx < len(image_vl_flags) and image_vl_flags[idx]:
                    cond_w, cond_h = calculate_dimensions(CONDITION_IMAGE_SIZE, img_w / img_h)
                    condition_images.append(self.image_processor.resize(img, cond_h, cond_w))
                
                # VAE/Ref path: pixel-level latent conditioning
                if idx < len(image_ref_flags) and image_ref_flags[idx]:
                    if idx == main_image_index:
                        # Main image: scale normally (preserves its aspect ratio)
                        vae_w, vae_h = main_vae_w, main_vae_h
                        vae_image_sizes.append((vae_w, vae_h))
                        vae_images.append(self.image_processor.preprocess(img, vae_h, vae_w).unsqueeze(2))
                    else:
                        # Non-main image: crop-to-cover the main image's dimensions
                        # so ALL ref latents have the same spatial shape, avoiding
                        # aspect-ratio mismatches in the transformer's position embeddings.
                        img_aspect = img_w / img_h
                        main_aspect = main_vae_w / main_vae_h
                        if abs(img_aspect - main_aspect) > 0.05:
                            logger.info(
                                f"QwenEdit: Crop-to-cover image {idx} "
                                f"({img_w}x{img_h}, {img_aspect:.2f}) -> "
                                f"main aspect ({main_vae_w}x{main_vae_h}, {main_aspect:.2f})"
                            )
                        cropped = crop_to_cover(img, main_vae_w, main_vae_h)
                        vae_image_sizes.append((main_vae_w, main_vae_h))
                        vae_images.append(self.image_processor.preprocess(cropped, main_vae_h, main_vae_w).unsqueeze(2))
        
        # CFG check
        has_neg = negative_prompt is not None or (negative_prompt_embeds is not None and negative_prompt_embeds_mask is not None)
        if true_cfg_scale > 1 and not has_neg:
            logger.warning(f"true_cfg_scale={true_cfg_scale} but no negative_prompt")
        do_cfg = true_cfg_scale > 1 and has_neg
        
        # Encode prompts
        prompt_embeds, prompt_embeds_mask = self.encode_prompt(
            image=condition_images, prompt=prompt, prompt_embeds=prompt_embeds,
            prompt_embeds_mask=prompt_embeds_mask, device=device,
            num_images_per_prompt=num_images_per_prompt, max_sequence_length=max_sequence_length)
        
        if do_cfg:
            negative_prompt_embeds, negative_prompt_embeds_mask = self.encode_prompt(
                image=condition_images, prompt=negative_prompt, 
                prompt_embeds=negative_prompt_embeds, prompt_embeds_mask=negative_prompt_embeds_mask,
                device=device, num_images_per_prompt=num_images_per_prompt,
                max_sequence_length=max_sequence_length)
        
        # Prepare latents
        num_channels = self.transformer.config.in_channels // 4
        latents, image_latents = self.prepare_latents(
            vae_images, batch_size * num_images_per_prompt, num_channels,
            height, width, prompt_embeds.dtype, device, generator, latents)
        
        img_shapes = [[
            (1, height // self.vae_scale_factor // 2, width // self.vae_scale_factor // 2),
            *[(1, vh // self.vae_scale_factor // 2, vw // self.vae_scale_factor // 2) for vw, vh in vae_image_sizes]
        ]] * batch_size
        
        # Timesteps
        sigmas = np.linspace(1.0, 1 / num_inference_steps, num_inference_steps) if sigmas is None else sigmas
        image_seq_len = latents.shape[1]
        mu = calculate_shift(image_seq_len,
            self.scheduler.config.get("base_image_seq_len", 256),
            self.scheduler.config.get("max_image_seq_len", 4096),
            self.scheduler.config.get("base_shift", 0.5),
            self.scheduler.config.get("max_shift", 1.15))
        timesteps, num_inference_steps = retrieve_timesteps(self.scheduler, num_inference_steps, device, sigmas=sigmas, mu=mu)
        num_warmup_steps = max(len(timesteps) - num_inference_steps * self.scheduler.order, 0)
        self._num_timesteps = len(timesteps)
        
        # Guidance
        if self.transformer.config.guidance_embeds and guidance_scale is None:
            raise ValueError("guidance_scale required for guidance-distilled model")
        elif self.transformer.config.guidance_embeds:
            guidance = torch.full([1], guidance_scale, device=device, dtype=torch.float32).expand(latents.shape[0])
        else:
            guidance = None
        
        txt_seq_lens = prompt_embeds_mask.sum(dim=1).tolist() if prompt_embeds_mask is not None else None
        neg_txt_seq_lens = negative_prompt_embeds_mask.sum(dim=1).tolist() if negative_prompt_embeds_mask is not None else None
        
        # Spectrum acceleration: patch transformer forward if configured
        _spectrum_unpatch = None
        spectrum_config = getattr(self, '_spectrum_config', None)
        if spectrum_config is not None and num_inference_steps >= spectrum_config.get('min_steps', 15):
            try:
                from .spectrum_forward import patch_transformer_spectrum
                calls_per_step = 2 if do_cfg else 1
                _spectrum_unpatch = patch_transformer_spectrum(
                    self.transformer, num_inference_steps, spectrum_config, calls_per_step
                )
            except Exception as e:
                logger.warning(f"QwenEdit: Spectrum patch failed, running at full fidelity: {e}")
                _spectrum_unpatch = None
        elif spectrum_config is not None:
            logger.info(
                f"QwenEdit: Spectrum auto-disabled (steps={num_inference_steps} < "
                f"min_steps={spectrum_config.get('min_steps', 15)})"
            )
        
        # Denoising loop
        self.scheduler.set_begin_index(0)
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                if self.interrupt:
                    continue
                self._current_timestep = t
                
                latent_input = latents if image_latents is None else torch.cat([latents, image_latents], dim=1)
                timestep = t.expand(latents.shape[0]).to(latents.dtype)
                
                with self.transformer.cache_context("cond"):
                    noise_pred = self.transformer(
                        hidden_states=latent_input, timestep=timestep / 1000, guidance=guidance,
                        encoder_hidden_states_mask=prompt_embeds_mask, encoder_hidden_states=prompt_embeds,
                        img_shapes=img_shapes, txt_seq_lens=txt_seq_lens,
                        attention_kwargs=self._attention_kwargs, return_dict=False)[0]
                    noise_pred = noise_pred[:, :latents.size(1)]
                
                if do_cfg:
                    with self.transformer.cache_context("uncond"):
                        neg_pred = self.transformer(
                            hidden_states=latent_input, timestep=timestep / 1000, guidance=guidance,
                            encoder_hidden_states_mask=negative_prompt_embeds_mask, 
                            encoder_hidden_states=negative_prompt_embeds,
                            img_shapes=img_shapes, txt_seq_lens=neg_txt_seq_lens,
                            attention_kwargs=self._attention_kwargs, return_dict=False)[0]
                    neg_pred = neg_pred[:, :latents.size(1)]
                    combined = neg_pred + true_cfg_scale * (noise_pred - neg_pred)
                    cond_norm = torch.norm(noise_pred, dim=-1, keepdim=True)
                    combined_norm = torch.norm(combined, dim=-1, keepdim=True)
                    noise_pred = combined * (cond_norm / combined_norm)
                
                latents_dtype = latents.dtype
                latents = self.scheduler.step(noise_pred, t, latents, return_dict=False)[0]
                if latents.dtype != latents_dtype and torch.backends.mps.is_available():
                    latents = latents.to(latents_dtype)
                
                if callback_on_step_end is not None:
                    cb_kwargs = {k: locals()[k] for k in callback_on_step_end_tensor_inputs}
                    cb_out = callback_on_step_end(self, i, t, cb_kwargs)
                    latents = cb_out.pop("latents", latents)
                    prompt_embeds = cb_out.pop("prompt_embeds", prompt_embeds)
                
                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()
        
        # Spectrum: unpatch transformer and log stats
        if _spectrum_unpatch is not None:
            try:
                stats = _spectrum_unpatch()
                if stats:
                    total = stats.get("actual_forwards", 0) + stats.get("cached_steps", 0)
                    actual = stats.get("actual_forwards", 0)
                    if total > 0:
                        logger.info(
                            f"QwenEdit: Spectrum done — {actual}/{total} actual forwards "
                            f"({total - actual} cached, {(total - actual) / total * 100:.0f}% saved)"
                        )
            except Exception as e:
                logger.warning(f"QwenEdit: Spectrum unpatch error: {e}")
        
        self._current_timestep = None
        
        # Decode
        if output_type == "latent":
            image = latents
        else:
            latents = self._unpack_latents(latents, height, width, self.vae_scale_factor)
            latents = latents.to(self.vae.dtype)
            
            latents_mean = torch.tensor(self.vae.config.latents_mean).view(1, self.vae.config.z_dim, 1, 1, 1).to(latents.device, latents.dtype)
            latents_std = 1.0 / torch.tensor(self.vae.config.latents_std).view(1, self.vae.config.z_dim, 1, 1, 1).to(latents.device, latents.dtype)
            latents = latents / latents_std + latents_mean
            
            image = self.vae.decode(latents, return_dict=False)[0][:, :, 0]
            image = self.image_processor.postprocess(image, output_type=output_type)
        
        self.maybe_free_model_hooks()
        
        if not return_dict:
            return (image,)
        return QwenEditPipelineOutput(images=image)
