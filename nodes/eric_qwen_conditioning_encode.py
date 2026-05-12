# Copyright (c) 2026 Eric Hiss. All rights reserved.
# Licensed under the terms in LICENSE.txt (CC BY-NC 4.0 / Commercial dual license).
# https://github.com/EricRollei/Eric_Qwen_Edit_Experiments
"""
Eric Qwen Conditioning Encode Node

Runs the Qwen2.5-VL text+vision encoder on an image + prompt and returns
the resulting hidden-state tensor pair as a QWEN_CONDITIONING dict.  The
output can be passed directly to Eric Qwen Conditioned Edit, or first routed
through conditioning manipulation nodes (Interpolate, Blend, Apply Direction).

This is Layer 1 of the conditioning system.  Encoding takes a few seconds
(much less than a full diffusion run), so it is practical to encode several
variants - neutral, target expression, reference style - and then manipulate
them before running the edit node once.

Author: Eric Hiss (GitHub: EricRollei)
"""
import math
import torch
from .eric_qwen_edit_utils import tensor_to_pil, prepare_image_for_pipeline
from ..pipelines.pipeline_qwen_edit import calculate_dimensions, CONDITION_IMAGE_SIZE

# Qwen2.5-VL visual encoder constants
_PATCH_SIZE  = 14   # spatial patch size in pixels
_MERGE_SIZE  = 2    # spatial merge factor (2×2 patches → 1 token)


def _estimate_image_tokens(vl_w: int, vl_h: int) -> int:
    """
    Estimate the number of visual tokens produced by Qwen2.5-VL for a given
    VL conditioning image size.

    Formula: (H // patch_size // merge_size) × (W // patch_size // merge_size)
    Constants: patch_size=14, merge_size=2  (Qwen2.5-VL defaults)

    This is the right value to use as prefix_N in the Interpolate / Apply
    Direction nodes when scope=skip_prefix_N.
    """
    h_tokens = math.ceil(vl_h / _PATCH_SIZE) // _MERGE_SIZE
    w_tokens = math.ceil(vl_w / _PATCH_SIZE) // _MERGE_SIZE
    return h_tokens * w_tokens


class EricQwenConditioningEncode:
    """
    Encode an image + prompt through the Qwen2.5-VL encoder.

    Returns a QWEN_CONDITIONING dict:
        {
            "prompt_embeds":      Tensor [1, padded_seq, 3584]  (stored on CPU)
            "prompt_embeds_mask": Tensor [1, padded_seq]         (1=valid, 0=pad)
            "metadata": {
                "source_prompt":   str    - first 120 chars of prompt
                "valid_tokens":    int    - non-padding tokens in sequence
                "total_tokens":    int    - full padded sequence length
                "image_size":      str    - original input image WxH
                "vl_size":         str    - VL conditioning image WxH (~384px)
                "image_tokens":    int    - estimated visual token count
                                           USE THIS as prefix_N in skip_prefix_N scope
                "text_tokens":     int    - estimated instruction + template tokens
            }
        }

    Console log example
    -------------------
        [EricQwenEncode] image 749x1062 -> VL 320x448
        [EricQwenEncode] result: 198/198 valid tokens
        [EricQwenEncode] token layout: ~176 image + ~22 text  (prefix_N=176 for skip_prefix_N)

    Typical workflow
    ----------------
    Loader → [LoRA] → Encode(neutral) ──┐
                    → Encode(target)  ──┤→ Interpolate(α) → Conditioned Edit
    """

    CATEGORY = "Eric Qwen-Edit/Conditioning"
    FUNCTION = "encode"
    RETURN_TYPES = ("QWEN_CONDITIONING",)
    RETURN_NAMES = ("conditioning",)

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "pipeline": ("QWEN_EDIT_PIPELINE", {
                    "tooltip": "Loaded Qwen-Edit pipeline (from any Loader node)"
                }),
                "image": ("IMAGE", {
                    "tooltip": (
                        "Reference image - used for VL semantic conditioning. "
                        "Resized to ~384px internally; full resolution is NOT needed here."
                    )
                }),
                "prompt": ("STRING", {
                    "multiline": True,
                    "default": "Edit this image to...",
                    "tooltip": "Edit instruction - exactly as you would type in Eric Qwen-Edit Image"
                }),
            },
            "optional": {
                "max_sequence_length": ("INT", {
                    "default": 512,
                    "min": 128,
                    "max": 1024,
                    "step": 64,
                    "tooltip": (
                        "Max prompt token budget. 512 is sufficient for most prompts. "
                        "Raise to 768-1024 only for very long detailed descriptions."
                    )
                }),
            },
        }

    def encode(
        self,
        pipeline: dict,
        image: torch.Tensor,
        prompt: str,
        max_sequence_length: int = 512,
    ) -> tuple:
        pipe = pipeline["pipeline"]
        device = next(pipe.transformer.parameters()).device

        # ComfyUI [B, H, W, C] or [H, W, C] → PIL RGB
        if image.dim() == 4:
            pil_image = tensor_to_pil(image[0])
        else:
            pil_image = tensor_to_pil(image)
        pil_image = prepare_image_for_pipeline(pil_image)
        img_w, img_h = pil_image.size

        # Resize to ~384px for the VL conditioning path
        cond_w, cond_h = calculate_dimensions(CONDITION_IMAGE_SIZE, img_w / img_h)
        condition_image = pipe.image_processor.resize(pil_image, cond_h, cond_w)

        # Estimate image token count for this VL size - used as prefix_N hint
        estimated_image_tokens = _estimate_image_tokens(cond_w, cond_h)

        print(
            f"[EricQwenEncode] image {img_w}×{img_h} → VL {cond_w}×{cond_h} | "
            f"prompt: {prompt[:70]}{'...' if len(prompt) > 70 else ''}"
        )

        with torch.inference_mode():
            prompt_embeds, prompt_embeds_mask = pipe.encode_prompt(
                prompt=prompt,
                image=[condition_image],
                device=device,
                num_images_per_prompt=1,
                max_sequence_length=max_sequence_length,
            )

        valid_tokens = int(prompt_embeds_mask.sum().item())
        total_tokens = prompt_embeds.shape[1]
        estimated_text_tokens = max(0, valid_tokens - estimated_image_tokens)

        print(
            f"[EricQwenEncode] result: {valid_tokens}/{total_tokens} valid tokens"
        )
        print(
            f"[EricQwenEncode] token layout: ~{estimated_image_tokens} image "
            f"+ ~{estimated_text_tokens} text  "
            f"(prefix_N={estimated_image_tokens} for skip_prefix_N scope)"
        )

        conditioning = {
            "prompt_embeds":      prompt_embeds.cpu(),
            "prompt_embeds_mask": prompt_embeds_mask.cpu(),
            "metadata": {
                "source_prompt":   prompt[:120],
                "valid_tokens":    valid_tokens,
                "total_tokens":    total_tokens,
                "image_size":      f"{img_w}x{img_h}",
                "vl_size":         f"{cond_w}x{cond_h}",
                "image_tokens":    estimated_image_tokens,
                "text_tokens":     estimated_text_tokens,
            },
        }
        return (conditioning,)
