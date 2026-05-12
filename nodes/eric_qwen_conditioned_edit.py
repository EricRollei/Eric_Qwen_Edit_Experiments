# Copyright (c) 2026 Eric Hiss. All rights reserved.
# Licensed under the terms in LICENSE.txt (CC BY-NC 4.0 / Commercial dual license).
# https://github.com/EricRollei/Eric_Qwen_Edit_Experiments
"""
Eric Qwen Conditioned Edit Node

Run the Qwen-Edit diffusion pipeline using a pre-computed QWEN_CONDITIONING
instead of a raw text prompt.  Enables the full conditioning manipulation
workflow: encode → interpolate / apply direction / blend → edit.

The image input here is used for the VAE pixel path - it sets the output
resolution and provides the pixel-level reference.  It should be the same
image that was encoded, or a higher-res version for editing at larger scale
than you encoded.

Author: Eric Hiss (GitHub: EricRollei)
"""

import math
import time
import torch
import comfy.utils
from PIL import Image

from .eric_qwen_edit_utils import tensor_to_pil, pil_to_tensor, prepare_image_for_pipeline

DIMENSION_ALIGNMENT = 32   # VAE packing requirement


class EricQwenConditionedEdit:
    """
    Edit an image using pre-computed QWEN_CONDITIONING.

    PixelSmile expression control example
    --------------------------------------
    With the PixelSmile LoRA loaded:

        Encode(image, "person with neutral expression")  → neutral_cond
        Encode(image, "person with happy expression")    → happy_cond
        Interpolate(neutral_cond, happy_cond, alpha=0.8) → mixed_cond
        Conditioned Edit(pipeline, image, mixed_cond)    → result

    Style direction example
    -----------------------
        Encode(photo,   "person in room")          → photo_cond
        Encode(painting,"painting of person")      → paint_cond
        Interpolate(photo_cond, paint_cond, α=1.0) → styled_cond
        Conditioned Edit(pipeline, photo, styled_cond)

    CFG notes
    ---------
    CFG is active ONLY when negative_conditioning is connected AND
    true_cfg_scale > 1.  Without a negative, the pipeline runs single-pass
    (effectively cfg=1) regardless of what true_cfg_scale is set to.
    The default of 1.5 is intentionally conservative - the direction/interpolation
    already provides steering without needing aggressive CFG.

    upscale_to_max_mp
    -----------------
    When enabled, small input images are upscaled to fill the max_mp budget
    before being passed to the pipeline.  This avoids the lossy two-step of
    upscaling outside the node and then having the pipeline resize again.
    Aspect ratio is preserved; output is aligned to 32px.
    """

    CATEGORY = "Eric Qwen-Edit/Conditioning"
    FUNCTION = "edit"
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "pipeline": ("QWEN_EDIT_PIPELINE",),
                "image": ("IMAGE", {
                    "tooltip": (
                        "Image to edit - drives the VAE pixel path and sets output resolution. "
                        "Usually the same image you encoded."
                    )
                }),
                "conditioning": ("QWEN_CONDITIONING", {
                    "tooltip": "From Encode node, optionally passed through Interpolate / Apply Direction / Blend"
                }),
            },
            "optional": {
                "negative_conditioning": ("QWEN_CONDITIONING", {
                    "tooltip": (
                        "Optional. When connected, enables true CFG with true_cfg_scale.\n"
                        "Without this, true_cfg_scale is ignored entirely.\n"
                        "For direction-based edits, try without negative first - "
                        "the direction already provides steering."
                    )
                }),
                "steps": ("INT", {
                    "default": 8, "min": 1, "max": 100, "step": 1,
                    "tooltip": "Denoising steps (8 for Lightning LoRA, 50 for base model)"
                }),
                "true_cfg_scale": ("FLOAT", {
                    "default": 1.5, "min": 1.0, "max": 20.0, "step": 0.1,
                    "tooltip": (
                        "CFG strength - ONLY active when negative_conditioning is connected.\n"
                        "1.5 is a gentle nudge. 3.0+ is strong. "
                        "For direction edits start low (1.2-1.5) since the direction already steers."
                    )
                }),
                "seed": ("INT", {
                    "default": 0, "min": 0, "max": 0xFFFFFFFFFFFFFFFF,
                }),
                "max_mp": ("FLOAT", {
                    "default": 8.0, "min": 0.5, "max": 16.0, "step": 0.5,
                    "tooltip": "Maximum output megapixels. Lower = faster."
                }),
                "upscale_to_max_mp": ("BOOLEAN", {
                    "default": False,
                    "tooltip": (
                        "When enabled, upscale small inputs to fill the max_mp budget "
                        "(preserving aspect ratio, aligned to 32px). "
                        "More efficient than upscaling before this node since it avoids "
                        "a lossy double-resize."
                    )
                }),
            },
        }

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return kwargs.get("seed", 0)

    def edit(
        self,
        pipeline: dict,
        image: torch.Tensor,
        conditioning: dict,
        negative_conditioning: dict = None,
        steps: int = 8,
        true_cfg_scale: float = 1.5,
        seed: int = 0,
        max_mp: float = 8.0,
        upscale_to_max_mp: bool = False,
    ) -> tuple:
        pipe        = pipeline["pipeline"]
        offload_vae = pipeline.get("offload_vae", False)
        device      = next(pipe.transformer.parameters()).device
        enc_dtype   = pipe.text_encoder.dtype

        # ComfyUI tensor → PIL RGB
        if image.dim() == 4:
            pil_image = tensor_to_pil(image[0])
        else:
            pil_image = tensor_to_pil(image)
        pil_image = prepare_image_for_pipeline(pil_image)

        input_w, input_h = pil_image.size
        input_mp = (input_w * input_h) / (1024 * 1024)

        # ── Optional upscale to fill max_mp budget ────────────────────────────
        if upscale_to_max_mp and input_mp < max_mp:
            max_pixels = int(max_mp * 1024 * 1024)
            scale  = math.sqrt(max_pixels / (input_w * input_h))
            new_w  = max(DIMENSION_ALIGNMENT,
                         int(input_w * scale) // DIMENSION_ALIGNMENT * DIMENSION_ALIGNMENT)
            new_h  = max(DIMENSION_ALIGNMENT,
                         int(input_h * scale) // DIMENSION_ALIGNMENT * DIMENSION_ALIGNMENT)
            print(
                f"[EricQwenCondEdit] upscale {input_w}×{input_h} ({input_mp:.1f}MP) "
                f"→ {new_w}×{new_h} ({new_w * new_h / 1e6:.1f}MP)"
            )
            pil_image = pil_image.resize((new_w, new_h), Image.LANCZOS)

        # ── Move conditioning tensors to device ───────────────────────────────
        prompt_embeds      = conditioning["prompt_embeds"].to(device=device, dtype=enc_dtype)
        prompt_embeds_mask = conditioning["prompt_embeds_mask"].to(device=device)

        neg_embeds = None
        neg_mask   = None
        if negative_conditioning is not None:
            neg_embeds = negative_conditioning["prompt_embeds"].to(device=device, dtype=enc_dtype)
            neg_mask   = negative_conditioning["prompt_embeds_mask"].to(device=device)

        do_cfg = (neg_embeds is not None) and (true_cfg_scale > 1.0)

        src  = conditioning.get("metadata", {}).get("source_prompt", "?")[:60]
        vtok = conditioning.get("metadata", {}).get("valid_tokens", "?")
        print(
            f"[EricQwenCondEdit] '{src}...' | "
            f"size={pil_image.size}, tokens={vtok}, steps={steps}, "
            f"cfg={'%.1f' % true_cfg_scale if do_cfg else 'off'}, max_mp={max_mp}"
        )

        generator = torch.Generator(device=device).manual_seed(seed)

        # ── VAE offload handling ───────────────────────────────────────────────
        vae_device_orig = None
        if offload_vae:
            vae_device_orig = next(pipe.vae.parameters()).device
            if str(vae_device_orig) == "cpu":
                print("[EricQwenCondEdit] VAE → GPU")
                pipe.vae = pipe.vae.to(device)

        pbar = comfy.utils.ProgressBar(steps)

        def _progress_cb(pl, step_idx, timestep, cb_kwargs):
            pbar.update(1)
            return cb_kwargs

        start = time.time()
        try:
            with torch.inference_mode():
                output = pipe(
                    image=pil_image,
                    prompt=None,                   # use pre-computed embeds
                    prompt_embeds=prompt_embeds,
                    prompt_embeds_mask=prompt_embeds_mask,
                    negative_prompt=None,
                    negative_prompt_embeds=neg_embeds,
                    negative_prompt_embeds_mask=neg_mask,
                    true_cfg_scale=true_cfg_scale if do_cfg else 1.0,
                    max_pixels=int(max_mp * 1024 * 1024),
                    num_inference_steps=steps,
                    generator=generator,
                    num_images_per_prompt=1,
                    callback_on_step_end=_progress_cb,
                    callback_on_step_end_tensor_inputs=["latents"],
                )
        finally:
            if offload_vae and vae_device_orig is not None and str(vae_device_orig) == "cpu":
                print("[EricQwenCondEdit] VAE → CPU")
                pipe.vae = pipe.vae.to("cpu")
                torch.cuda.empty_cache()

        elapsed = time.time() - start
        result  = output.images[0]
        print(
            f"[EricQwenCondEdit] done {result.size[0]}×{result.size[1]} in "
            f"{elapsed / 60:.1f}min ({elapsed / steps:.1f}s/step)"
        )

        return (pil_to_tensor(result).unsqueeze(0),)
