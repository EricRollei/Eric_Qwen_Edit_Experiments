# Copyright (c) 2026 Eric Hiss. All rights reserved.
# Licensed under the terms in LICENSE.txt (CC BY-NC 4.0 / Commercial dual license).
# https://github.com/EricRollei/Eric_Qwen_Edit_Experiments
"""
Eric Qwen Grounded Generate Node
Generate a new image from a pre-researched grounded prompt plus visual
reference images, typically sourced from EricGenSearcherNode.

Author: Eric Hiss (GitHub: EricRollei)
"""

import math
import time
from typing import Optional, Tuple, List

import torch
import comfy.utils
from PIL import Image

from .eric_qwen_edit_utils import (
    tensor_to_pil,
    pil_to_tensor,
    prepare_image_for_pipeline,
)

DIMENSION_ALIGNMENT = 32

_ASPECT_RATIOS = {
    "1:1":   (1,  1),
    "4:3":   (4,  3),
    "3:4":   (3,  4),
    "16:9":  (16, 9),
    "9:16":  (9,  16),
    "3:2":   (3,  2),
    "2:3":   (2,  3),
    "21:9":  (21, 9),
    "9:21":  (9,  21),
    "5:4":   (5,  4),
    "4:5":   (4,  5),
}


def _compute_output_size(
    aspect_ratio:  str,
    output_mp:     float,
    custom_width:  int,
    custom_height: int,
) -> Tuple[int, int]:
    """Return (width, height) aligned to DIMENSION_ALIGNMENT."""
    if aspect_ratio == "custom":
        w = max(256, (custom_width  // DIMENSION_ALIGNMENT) * DIMENSION_ALIGNMENT)
        h = max(256, (custom_height // DIMENSION_ALIGNMENT) * DIMENSION_ALIGNMENT)
        return w, h
    rw, rh = _ASPECT_RATIOS.get(aspect_ratio, (1, 1))
    mp_pixels = output_mp * 1024 * 1024
    scale = math.sqrt(mp_pixels / (rw * rh))
    w = max(256, int(rw * scale) // DIMENSION_ALIGNMENT * DIMENSION_ALIGNMENT)
    h = max(256, int(rh * scale) // DIMENSION_ALIGNMENT * DIMENSION_ALIGNMENT)
    return w, h


class EricQwenGroundedGenerate:
    """
    Generate a new image from a grounded prompt + visual reference images.

    Purpose-built for the Gen-Searcher -> Qwen-Image workflow.

    ## Reference Mode (ref_mode)

    This is the most important quality setting:

    - vl_only (default): All reference images go through the VL/semantic path only
      (REF=False for all). The model sees and understands what the subject looks
      like via the vision encoder, but is NOT constrained by pixel-level latents
      from web thumbnails. Gives the highest quality output since the model is
      free to generate at full quality. Best when references are web images.

    - primary_ref: ref_image_1 goes through both VL+REF paths. Use when ref_image_1
      is a high-quality photo you specifically want the output to resemble closely.
      May reduce quality if ref_image_1 is a low-res web thumbnail.

    - all_ref: All images go through VL+REF. Only use with curated high-quality
      input images, NOT with random web search results.

    ## Output Resolution

    Controlled by aspect_ratio + output_mp. The pipeline's max_pixels parameter
    is set to match. The LAST reference image is resized to the target dimensions
    so the pipeline uses it as the output canvas size.
    """

    CATEGORY = "Eric/QwenImage"
    FUNCTION = "generate"
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "pipeline": ("QWEN_EDIT_PIPELINE",),
                "grounded_prompt": ("STRING", {
                    "multiline": True,
                    "default":   "",
                    "tooltip":   "Rich prompt from EricGenSearcherNode (or any detailed prompt).",
                }),
                "ref_image_1": ("IMAGE", {
                    "tooltip": "Primary visual reference image.",
                }),
                "aspect_ratio": (
                    ["1:1", "16:9", "9:16", "4:3", "3:4", "3:2", "2:3",
                     "21:9", "9:21", "5:4", "4:5", "custom"],
                    {
                        "default": "1:1",
                        "tooltip": (
                            "Output aspect ratio.\n"
                            "  16:9 = landscape (~1920×1080 at 2MP)\n"
                            "  9:16 = portrait  (~1080×1920 at 2MP)\n"
                            "  1:1  = square    (~1448×1448 at 2MP)\n"
                            "  custom = use output_width / output_height exactly"
                        ),
                    }
                ),
                "output_mp": ("FLOAT", {
                    "default": 2.0,
                    "min":     0.25,
                    "max":     16.0,
                    "step":    0.25,
                    "tooltip": (
                        "Output megapixels:\n"
                        "  0.5 MP → quick test\n"
                        "  1.0 MP → ~1024×1024 (1:1)\n"
                        "  2.0 MP → ~1448×1448 (1:1)  ~1920×1080 (16:9)\n"
                        "  4.0 MP → ~2048×2048 (1:1)  (slow)\n"
                        "Ignored when aspect_ratio=custom."
                    ),
                }),
                "ref_mode": (
                    ["vl_only", "primary_ref", "all_ref"],
                    {
                        "default": "vl_only",
                        "tooltip": (
                            "How reference images condition the generation:\n\n"
                            "vl_only (RECOMMENDED for web images):\n"
                            "  All refs → VL/semantic path only (REF=False).\n"
                            "  Model understands subject appearance but generates\n"
                            "  at full quality without pixel-level constraints.\n"
                            "  Best when references are web search thumbnails.\n\n"
                            "primary_ref:\n"
                            "  ref_image_1 → VL+REF (pixel level).\n"
                            "  Others → VL only.\n"
                            "  Use when ref_image_1 is a high-quality photo.\n\n"
                            "all_ref:\n"
                            "  All images → VL+REF.\n"
                            "  Only use with curated high-quality inputs.\n"
                            "  NOT recommended for web search results."
                        ),
                    }
                ),
            },
            "optional": {
                "ref_image_2": ("IMAGE", {"tooltip": "Optional 2nd reference image."}),
                "ref_image_3": ("IMAGE", {"tooltip": "Optional 3rd reference image."}),
                "ref_image_4": ("IMAGE", {"tooltip": "Optional 4th reference image."}),
                "negative_prompt": ("STRING", {
                    "multiline": True,
                    "default":   "",
                }),
                "steps": ("INT", {
                    "default": 8, "min": 1, "max": 100, "step": 1,
                    "tooltip": "8 for lightning LoRA, 50 for base model.",
                }),
                "true_cfg_scale": ("FLOAT", {
                    "default": 4.0, "min": 1.0, "max": 20.0, "step": 0.5,
                }),
                "seed": ("INT", {
                    "default": 0, "min": 0, "max": 0xffffffffffffffff,
                }),
                "custom_width": ("INT", {
                    "default": 1920, "min": 256, "max": 8192, "step": 32,
                    "tooltip": "Used only when aspect_ratio=custom.",
                }),
                "custom_height": ("INT", {
                    "default": 1080, "min": 256, "max": 8192, "step": 32,
                    "tooltip": "Used only when aspect_ratio=custom.",
                }),
                "add_generation_prefix": ("BOOLEAN", {
                    "default": False,
                    "tooltip": (
                        "Prepend a directive telling the model to create a new image.\n"
                        "OFF (default): use the grounded prompt as-is. The Qwen-Image\n"
                        "  model understands image generation from detailed prompts.\n"
                        "ON: prepends 'Generate a completely new image...'\n"
                        "  May fight against the model's edit-focused fine-tuning."
                    ),
                }),
            },
        }

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return kwargs.get("seed", 0)

    def generate(
        self,
        pipeline:              dict,
        grounded_prompt:       str,
        ref_image_1:           torch.Tensor,
        aspect_ratio:          str   = "1:1",
        output_mp:             float = 2.0,
        ref_mode:              str   = "vl_only",
        ref_image_2:           Optional[torch.Tensor] = None,
        ref_image_3:           Optional[torch.Tensor] = None,
        ref_image_4:           Optional[torch.Tensor] = None,
        negative_prompt:       str   = "",
        steps:                 int   = 8,
        true_cfg_scale:        float = 4.0,
        seed:                  int   = 0,
        custom_width:          int   = 1920,
        custom_height:         int   = 1080,
        add_generation_prefix: bool  = False,
    ) -> Tuple[torch.Tensor]:

        pipe        = pipeline["pipeline"]
        offload_vae = pipeline.get("offload_vae", False)

        # ── Collect and convert reference images ──────────────────────
        raw_tensors = [ref_image_1, ref_image_2, ref_image_3, ref_image_4]
        pil_images: List[Image.Image] = []
        for t in raw_tensors:
            if t is None:
                break
            src = t[0] if t.dim() == 4 else t
            pil_images.append(prepare_image_for_pipeline(tensor_to_pil(src)))
        num_refs = len(pil_images)

        # ── Compute output dimensions ─────────────────────────────────
        target_w, target_h = _compute_output_size(
            aspect_ratio, output_mp, custom_width, custom_height
        )
        actual_mp = (target_w * target_h) / (1_024 * 1_024)

        # Resize the LAST reference image to the target output size.
        # The pipeline detects output resolution from image[-1] when no explicit
        # width/height is provided. This is more reliable than explicit params
        # and matches how the multi-image node controls output resolution.
        pil_images[-1] = pil_images[-1].resize((target_w, target_h), Image.LANCZOS)

        # ── Determine REF/VL flags based on ref_mode ──────────────────
        # vl_only:     all VL=True, all REF=False
        # primary_ref: image_1 VL+REF, rest VL only
        # all_ref:     all VL=True, all REF=True
        image_vl_flags  = [True] * num_refs
        if ref_mode == "vl_only":
            image_ref_flags = [False] * num_refs
            # Pipeline requires at least one REF image - force image_1
            image_ref_flags[0] = True
        elif ref_mode == "primary_ref":
            image_ref_flags = [False] * num_refs
            image_ref_flags[0] = True
        else:  # all_ref
            image_ref_flags = [True] * num_refs

        # ── Build final prompt ────────────────────────────────────────
        if add_generation_prefix:
            final_prompt = (
                "Generate a completely new, original image. "
                "Use the reference pictures for visual context only. "
                "Create a fresh composition following this description: "
                + grounded_prompt
            )
        else:
            final_prompt = grounded_prompt

        print(f"[EricQwenGrounded] ── Grounded Generate ──")
        print(f"[EricQwenGrounded] References: {num_refs}  ref_mode: {ref_mode}")
        print(f"[EricQwenGrounded] Output: {target_w}×{target_h} ({actual_mp:.2f}MP, {aspect_ratio})")
        print(f"[EricQwenGrounded] Steps: {steps}  CFG: {true_cfg_scale}  Seed: {seed}")
        for i, img in enumerate(pil_images):
            vl  = "VL"  if image_vl_flags[i]  else "--"
            ref = "REF" if image_ref_flags[i] else "---"
            canvas = " [canvas/output size]" if i == len(pil_images) - 1 else ""
            print(f"[EricQwenGrounded]   ref_{i+1}: {img.size[0]}×{img.size[1]} "
                  f"[{vl}+{ref}]{canvas}")
        print(f"[EricQwenGrounded] Prompt ({len(final_prompt.split())} words): "
              f"{final_prompt[:120]}...")

        # ── Device / generator ────────────────────────────────────────
        device    = next(pipe.transformer.parameters()).device
        generator = torch.Generator(device=device).manual_seed(seed)

        # ── VAE offload handling ──────────────────────────────────────
        vae_device_original = None
        if offload_vae:
            vae_device_original = next(pipe.vae.parameters()).device
            if str(vae_device_original) == "cpu":
                print("[EricQwenGrounded] Moving VAE to GPU...")
                pipe.vae = pipe.vae.to(device)

        start_time = time.time()
        pbar = comfy.utils.ProgressBar(steps)

        def _cb(pipeline, step_index, timestep, cb_kwargs):
            pbar.update(1)
            return cb_kwargs

        try:
            with torch.inference_mode():
                output = pipe(
                    prompt              = final_prompt,
                    image               = pil_images,
                    # Use max_pixels only - let image[-1] set dimensions.
                    # This is the same approach as EricQwenEditMultiImage and
                    # is more reliable than passing explicit width/height.
                    max_pixels          = int(actual_mp * 1_024 * 1_024),
                    negative_prompt     = negative_prompt if negative_prompt else " ",
                    num_inference_steps = steps,
                    true_cfg_scale      = true_cfg_scale,
                    generator           = generator,
                    num_images_per_prompt = 1,
                    main_image_index    = 0,
                    image_vl_flags      = image_vl_flags,
                    image_ref_flags     = image_ref_flags,
                    vae_target_size     = None,  # dynamic - encode at output resolution
                    callback_on_step_end = _cb,
                    callback_on_step_end_tensor_inputs = ["latents"],
                )
        finally:
            if offload_vae and vae_device_original is not None \
                    and str(vae_device_original) == "cpu":
                print("[EricQwenGrounded] Moving VAE back to CPU...")
                pipe.vae = pipe.vae.to("cpu")
                torch.cuda.empty_cache()

        elapsed = time.time() - start_time
        result  = output.images[0]
        print(f"[EricQwenGrounded] Done: {result.size[0]}×{result.size[1]} "
              f"in {elapsed/60:.1f}min ({elapsed/steps:.1f}s/step)")

        return (pil_to_tensor(result).unsqueeze(0),)
