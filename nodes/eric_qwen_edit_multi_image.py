# Copyright (c) 2026 Eric Hiss. All rights reserved.
# Licensed under the terms in LICENSE.txt (CC BY-NC 4.0 / Commercial dual license).
# https://github.com/EricRollei/Eric_Qwen_Edit_Experiments
"""
Eric Qwen-Edit Multi-Image Fusion Node
Combine multiple images using Qwen-Image-Edit-2511's native multi-image support.

Qwen-Image-Edit-2511 can accept multiple input images and compose them
based on a text prompt. Images are internally labeled Picture 1, Picture 2, etc.

The model processes images through TWO separate conditioning paths:
  1) VL path  (384x384): semantic understanding via Qwen2.5-VL text encoder
  2) VAE path (~1024px):  pixel-level reference latents concatenated with noise

Per-image control over these paths is critical for quality:
  - A "reference" image should go through BOTH paths (VL + VAE)
  - A "style-only" reference might only need VL (semantic) without VAE (pixels)
  - The "main" image's VAE latent seeds the denoising process

Composition modes help auto-generate spacial prompts, and the per-image
VL/ref toggles let you fine-tune which images the model uses for semantic
understanding vs pixel reconstruction.

Model Credits:
- Qwen-Image-Edit developed by Qwen Team (Alibaba)
- https://github.com/QwenLM/Qwen-Image

Author: Eric Hiss (GitHub: EricRollei)
"""

import torch
import time
from typing import Tuple, Optional, List

import comfy.utils

from .eric_qwen_edit_utils import (
    tensor_to_pil,
    pil_to_tensor,
    prepare_image_for_pipeline,
)


# Position descriptors for auto-layout
_POSITIONS_2 = ["on the left side", "on the right side"]
_POSITIONS_3 = ["on the left", "in the center", "on the right"]
_POSITIONS_4 = ["on the far left", "on the center-left", "on the center-right", "on the far right"]


class EricQwenEditMultiImage:
    """
    Combine multiple images using Qwen-Image-Edit-2511's multi-image composition.

    Pass 2-4 images along with a prompt describing how to compose them.
    Images are internally labeled as Picture 1, Picture 2, etc.

    ## Conditioning Paths

    Each image can independently go through:
    - **VL path** (semantic): Image is resized to ~384px and fed through the
      Qwen2.5-VL text encoder alongside the prompt. Gives the model semantic
      understanding of the image's contents.
    - **VAE/ref path** (pixel): Image is VAE-encoded at ~1024px into a latent
      that gets concatenated with the noise during denoising. Gives the model
      pixel-level detail for reconstruction/preservation.

    Use the per-image vl_X / ref_X toggles to control this. For example, to
    keep subjects separate, you might set ref=False on secondary images so
    only the main subject provides pixel-level reference.

    ## Main Image

    The main_image (default: image_1) is the primary reference. Its VAE latent
    is the one that seeds denoising. Other images with ref=True also contribute
    latents, but the main image has the strongest reconstruction influence.

    ## Composition Modes

    - raw:   Your prompt exactly as written.
    - group: Auto-prepends spatial placement ("Picture 1 on the left...").
    - scene: Picture 1 as background, others placed into it.
    - merge: Fuse features from all images into one.

    ## VAE Target Size

    By default, VAE encoding uses a fixed 1024px target (matching model
    training). Set vae_target_size=0 to use the original dynamic sizing.
    """

    CATEGORY = "Eric Qwen-Edit"
    FUNCTION = "fuse"
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "pipeline": ("QWEN_EDIT_PIPELINE",),
                "image_1": ("IMAGE", {
                    "tooltip": "First input image (Picture 1) — default main image"
                }),
                "image_2": ("IMAGE", {
                    "tooltip": "Second input image (Picture 2)"
                }),
                "prompt": ("STRING", {
                    "multiline": True,
                    "default": "All subjects are sitting together on a couch, looking at the camera.",
                    "tooltip": "In 'raw' mode: write the full prompt referencing Picture 1, Picture 2, etc. "
                               "In other modes: describe the desired scene/action (image references are auto-added)."
                }),
                "composition_mode": (["group", "scene", "merge", "raw"], {
                    "default": "group",
                    "tooltip": "group: place subjects side-by-side with positions | "
                               "scene: use Picture 1 as background, place others into it | "
                               "merge: fuse features into one subject | "
                               "raw: your prompt exactly as-is"
                }),
            },
            "optional": {
                "image_3": ("IMAGE", {
                    "tooltip": "Optional third image (Picture 3)"
                }),
                "image_4": ("IMAGE", {
                    "tooltip": "Optional fourth image (Picture 4)"
                }),
                "subject_label": ("STRING", {
                    "default": "person",
                    "tooltip": "What to call each subject in auto-generated prompts "
                               "(e.g. 'person', 'woman', 'character', 'bear', 'product')"
                }),
                "main_image": (["image_1", "image_2", "image_3", "image_4"], {
                    "default": "image_1",
                    "tooltip": "Which image is the primary reference. Its VAE latent "
                               "seeds the denoising process for strongest reconstruction."
                }),
                "vae_target_size": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 2048,
                    "step": 64,
                    "tooltip": "Fixed resolution for VAE encoding. 0 (default) = encode refs "
                               "at output resolution (matches Edit node behavior, best for high-res). "
                               "Set to e.g. 1024 to force all refs to ~1MP (only useful at low output res)."
                }),
                # Per-image VL (semantic) toggles
                "vl_1": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Include image_1 in VL/semantic path (text encoder understands its content)"
                }),
                "vl_2": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Include image_2 in VL/semantic path"
                }),
                "vl_3": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Include image_3 in VL/semantic path"
                }),
                "vl_4": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Include image_4 in VL/semantic path"
                }),
                # Per-image VAE/ref (pixel) toggles
                "ref_1": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Include image_1 in VAE/ref path (pixel-level latent reference)"
                }),
                "ref_2": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Include image_2 in VAE/ref path"
                }),
                "ref_3": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Include image_3 in VAE/ref path (default False — VL-only for secondary images)"
                }),
                "ref_4": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Include image_4 in VAE/ref path (default False — VL-only for secondary images)"
                }),
                "negative_prompt": ("STRING", {
                    "multiline": True,
                    "default": "",
                    "tooltip": "What to avoid in the output"
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
                    "tooltip": "Max output megapixels. VAE refs scale to match output resolution."
                }),
            }
        }

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return kwargs.get("seed", 0)

    def _build_prompt(
        self,
        user_prompt: str,
        composition_mode: str,
        num_images: int,
        subject_label: str,
    ) -> str:
        """Build the final prompt based on composition mode."""

        if composition_mode == "raw":
            return user_prompt

        positions = {
            2: _POSITIONS_2,
            3: _POSITIONS_3,
            4: _POSITIONS_4,
        }.get(num_images, _POSITIONS_4[:num_images])

        if composition_mode == "group":
            parts = []
            for i in range(num_images):
                parts.append(
                    f"The {subject_label} from Picture {i+1} is {positions[i]}"
                )
            preamble = ". ".join(parts) + "."
            identity_note = (
                f" Each {subject_label} is a distinct individual — "
                f"preserve their unique appearance from their respective pictures."
            )
            return f"{preamble}{identity_note} {user_prompt}"

        elif composition_mode == "scene":
            subject_parts = []
            scene_positions = _POSITIONS_3 if num_images - 1 <= 3 else _POSITIONS_4
            for i in range(1, num_images):
                pos = scene_positions[i - 1] if i - 1 < len(scene_positions) else ""
                subject_parts.append(
                    f"the {subject_label} from Picture {i+1} is placed {pos} in the scene"
                )
            subjects_str = ", and ".join(subject_parts)
            preamble = (
                f"Using the scene/background from Picture 1, {subjects_str}. "
                f"Each {subject_label} is a distinct individual — preserve their unique appearance."
            )
            return f"{preamble} {user_prompt}"

        elif composition_mode == "merge":
            refs = " and ".join(f"Picture {i+1}" for i in range(num_images))
            preamble = (
                f"Merge and combine the features from {refs} into a single image."
            )
            return f"{preamble} {user_prompt}"

        return user_prompt

    def fuse(
        self,
        pipeline: dict,
        image_1: torch.Tensor,
        image_2: torch.Tensor,
        prompt: str,
        composition_mode: str = "group",
        image_3: Optional[torch.Tensor] = None,
        image_4: Optional[torch.Tensor] = None,
        subject_label: str = "person",
        main_image: str = "image_1",
        vae_target_size: int = 1024,
        vl_1: bool = True,
        vl_2: bool = True,
        vl_3: bool = True,
        vl_4: bool = True,
        ref_1: bool = True,
        ref_2: bool = True,
        ref_3: bool = False,
        ref_4: bool = False,
        negative_prompt: str = "",
        steps: int = 50,
        true_cfg_scale: float = 4.0,
        seed: int = 0,
        max_mp: float = 8.0,
    ) -> Tuple[torch.Tensor]:
        """Fuse multiple images using Qwen-Edit with per-image conditioning control."""
        pipe = pipeline["pipeline"]
        offload_vae = pipeline.get("offload_vae", False)

        # Collect images and per-image flags
        all_tensors = [image_1, image_2, image_3, image_4]
        all_vl_flags = [vl_1, vl_2, vl_3, vl_4]
        all_ref_flags = [ref_1, ref_2, ref_3, ref_4]

        pil_images: List = []
        image_vl_flags: List[bool] = []
        image_ref_flags: List[bool] = []

        for idx, img_tensor in enumerate(all_tensors):
            if img_tensor is None:
                break
            if img_tensor.dim() == 4:
                pil_img = tensor_to_pil(img_tensor[0])
            else:
                pil_img = tensor_to_pil(img_tensor)
            pil_images.append(prepare_image_for_pipeline(pil_img))
            image_vl_flags.append(all_vl_flags[idx])
            image_ref_flags.append(all_ref_flags[idx])

        num_images = len(pil_images)

        # Resolve main_image_index
        main_map = {"image_1": 0, "image_2": 1, "image_3": 2, "image_4": 3}
        main_image_index = main_map.get(main_image, 0)
        if main_image_index >= num_images:
            print(f"[EricQwenEdit] main_image={main_image} exceeds image count, using image_1")
            main_image_index = 0

        # Main image must have ref=True (it provides the seed latent)
        if not image_ref_flags[main_image_index]:
            print(f"[EricQwenEdit] Forcing ref=True for main image ({main_image})")
            image_ref_flags[main_image_index] = True

        # Build structured prompt
        final_prompt = self._build_prompt(
            prompt, composition_mode, num_images, subject_label
        )

        # vae_target_size=0 means dynamic (None to pipeline)
        effective_vae_size = vae_target_size if vae_target_size > 0 else None

        print(f"[EricQwenEdit] Multi-Image Fusion")
        print(f"[EricQwenEdit] Input images: {num_images}, Mode: {composition_mode}")
        print(f"[EricQwenEdit] Main image: {main_image} (index {main_image_index})")
        print(f"[EricQwenEdit] VAE target size: {effective_vae_size or 'dynamic'}")
        for i, img in enumerate(pil_images):
            vl_str = "VL" if image_vl_flags[i] else "--"
            ref_str = "REF" if image_ref_flags[i] else "---"
            main_str = " [MAIN]" if i == main_image_index else ""
            print(f"[EricQwenEdit]   Picture {i+1}: {img.size[0]}x{img.size[1]} [{vl_str}+{ref_str}]{main_str}")
        print(f"[EricQwenEdit] Steps: {steps}, CFG: {true_cfg_scale}")
        print(f"[EricQwenEdit] Final prompt: {final_prompt[:150]}...")

        device = next(pipe.transformer.parameters()).device
        generator = torch.Generator(device=device).manual_seed(seed)

        # Handle VAE offload
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
                output = pipe(
                    prompt=final_prompt,
                    image=pil_images,
                    max_pixels=int(max_mp * 1024 * 1024),
                    negative_prompt=negative_prompt if negative_prompt else " ",
                    num_inference_steps=steps,
                    true_cfg_scale=true_cfg_scale,
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
                print("[EricQwenEdit] Moving VAE back to CPU...")
                pipe.vae = pipe.vae.to("cpu")
                torch.cuda.empty_cache()

        elapsed = time.time() - start_time

        result = output.images[0]
        result_tensor = pil_to_tensor(result).unsqueeze(0)

        print(f"[EricQwenEdit] Fusion complete: {result.size[0]}x{result.size[1]}")
        print(f"[EricQwenEdit] Actual time: {elapsed/60:.1f} minutes ({elapsed/steps:.1f} sec/step)")

        return (result_tensor,)
