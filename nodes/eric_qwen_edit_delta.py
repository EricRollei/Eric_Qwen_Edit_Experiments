# Copyright (c) 2026 Eric Hiss. All rights reserved.
# Licensed under the terms in LICENSE.txt (CC BY-NC 4.0 / Commercial dual license).
# https://github.com/EricRollei/Eric_Qwen_Edit_Experiments
"""
Eric Qwen-Edit Delta / Smart Overlay Node
Compare before/after images, generate a change mask, and composite edits
onto the original high-resolution image.

Use case:
- Have a 20MP original image
- Run Qwen-Edit at 2MP for speed
- This node detects what changed, upscales the edit, and overlays
  only the changed pixels onto the original 20MP image

This is a pure image-processing node — no AI model needed.

Author: Eric Hiss (GitHub: EricRollei)
"""

import torch
import numpy as np
from PIL import Image, ImageFilter
from typing import Tuple, Optional

from .eric_qwen_edit_utils import (
    tensor_to_pil,
    pil_to_tensor,
)


class EricQwenEditDelta:
    """
    Smart Overlay: Compare edited vs original, composite only the changes.

    This node solves the high-res editing problem:
    1. You have a 20MP original image
    2. You run Qwen-Edit at 2MP for speed
    3. This node:
       a. Compares the original (resized to edit resolution) with the edit
       b. Generates a change mask showing what was modified
       c. Upscales the edit to match the original resolution
       d. Composites only the changed areas onto the original

    The result preserves all the original resolution and detail in unchanged
    areas, while applying the AI edit only where needed.

    Outputs:
    - composite: The final blended high-res image
    - change_mask: Mask showing modified areas (white = changed)
    - upscaled_edit: The edit upscaled to original resolution (before compositing)
    """

    CATEGORY = "Eric Qwen-Edit"
    FUNCTION = "compute_delta"
    RETURN_TYPES = ("IMAGE", "MASK", "IMAGE")
    RETURN_NAMES = ("composite", "change_mask", "upscaled_edit")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "original_image": ("IMAGE", {
                    "tooltip": "The original high-resolution image (before editing)"
                }),
                "edited_image": ("IMAGE", {
                    "tooltip": "The edited image (may be lower resolution than original)"
                }),
            },
            "optional": {
                "threshold": ("FLOAT", {
                    "default": 0.05,
                    "min": 0.001,
                    "max": 0.5,
                    "step": 0.005,
                    "tooltip": "Pixel difference threshold to detect changes (0-1 range). "
                               "Lower = more sensitive, higher = only major changes."
                }),
                "blur_radius": ("INT", {
                    "default": 5,
                    "min": 0,
                    "max": 50,
                    "step": 1,
                    "tooltip": "Gaussian blur radius for mask edge softening. "
                               "Higher = smoother transitions between edited and original regions."
                }),
                "expand_mask": ("INT", {
                    "default": 3,
                    "min": 0,
                    "max": 30,
                    "step": 1,
                    "tooltip": "Dilate the mask by this many pixels to catch edge artifacts."
                }),
                "upscale_method": (["lanczos", "bilinear", "bicubic", "nearest"], {
                    "default": "lanczos",
                    "tooltip": "Method for upscaling the edited image to original resolution."
                }),
                "input_mask": ("MASK", {
                    "tooltip": "Optional: override the auto-detected mask with a manual one. "
                               "This confines the overlay to only the masked region."
                }),
            }
        }

    def _compute_change_mask(
        self,
        original_resized: np.ndarray,
        edited: np.ndarray,
        threshold: float,
        expand_pixels: int,
        blur_radius: int,
    ) -> np.ndarray:
        """
        Compute a soft change mask between two same-sized images.

        Args:
            original_resized: Original image resized to match edited (H, W, 3), float32 0-1
            edited: Edited image (H, W, 3), float32 0-1
            threshold: Minimum per-pixel difference to count as changed
            expand_pixels: Dilate mask by this many pixels
            blur_radius: Gaussian blur for soft edges

        Returns:
            Mask as float32 array (H, W) in 0-1 range
        """
        # Compute per-pixel difference (max across channels)
        diff = np.abs(original_resized.astype(np.float32) - edited.astype(np.float32))
        diff_max = np.max(diff, axis=2)  # Max difference across R,G,B

        # Apply threshold
        mask = (diff_max > threshold).astype(np.float32)

        # Convert to PIL for morphology operations
        mask_pil = Image.fromarray((mask * 255).astype(np.uint8), mode='L')

        # Expand (dilate) the mask to catch boundary artifacts
        if expand_pixels > 0:
            mask_pil = mask_pil.filter(ImageFilter.MaxFilter(size=expand_pixels * 2 + 1))

        # Blur for soft edges
        if blur_radius > 0:
            mask_pil = mask_pil.filter(ImageFilter.GaussianBlur(radius=blur_radius))

        # Convert back to float
        mask_out = np.array(mask_pil).astype(np.float32) / 255.0

        return mask_out

    def compute_delta(
        self,
        original_image: torch.Tensor,
        edited_image: torch.Tensor,
        threshold: float = 0.05,
        blur_radius: int = 5,
        expand_mask: int = 3,
        upscale_method: str = "lanczos",
        input_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compare images and composite only the changed areas."""

        # Extract single images from batch
        if original_image.dim() == 4:
            orig_np = original_image[0].cpu().numpy()
        else:
            orig_np = original_image.cpu().numpy()

        if edited_image.dim() == 4:
            edit_np = edited_image[0].cpu().numpy()
        else:
            edit_np = edited_image.cpu().numpy()

        orig_h, orig_w = orig_np.shape[:2]
        edit_h, edit_w = edit_np.shape[:2]

        print(f"[EricQwenEdit] Edit Delta")
        print(f"[EricQwenEdit] Original: {orig_w}x{orig_h} ({orig_w*orig_h/1e6:.1f}MP)")
        print(f"[EricQwenEdit] Edited: {edit_w}x{edit_h} ({edit_w*edit_h/1e6:.1f}MP)")

        # Convert to PIL for resize operations
        orig_pil = Image.fromarray((orig_np * 255).clip(0, 255).astype(np.uint8))
        edit_pil = Image.fromarray((edit_np * 255).clip(0, 255).astype(np.uint8))

        # Map upscale method names to PIL constants
        resample_map = {
            "lanczos": Image.LANCZOS,
            "bilinear": Image.BILINEAR,
            "bicubic": Image.BICUBIC,
            "nearest": Image.NEAREST,
        }
        resample = resample_map.get(upscale_method, Image.LANCZOS)

        # Step 1: Resize original to edit resolution for comparison
        orig_at_edit_res = orig_pil.resize((edit_w, edit_h), resample)
        orig_at_edit_res_np = np.array(orig_at_edit_res).astype(np.float32) / 255.0

        # Step 2: Compute change mask at edit resolution
        if input_mask is not None:
            # Use provided mask instead of auto-detection
            if input_mask.dim() == 3:
                mask_np = input_mask[0].cpu().numpy()
            else:
                mask_np = input_mask.cpu().numpy()

            # Resize input mask to edit resolution if needed
            mask_h, mask_w = mask_np.shape[:2]
            if mask_h != edit_h or mask_w != edit_w:
                mask_pil = Image.fromarray((mask_np * 255).astype(np.uint8), mode='L')
                mask_pil = mask_pil.resize((edit_w, edit_h), Image.BILINEAR)
                change_mask_edit = np.array(mask_pil).astype(np.float32) / 255.0
            else:
                change_mask_edit = mask_np.astype(np.float32)

            # Apply blur to user mask for softer edges
            if blur_radius > 0:
                mask_pil = Image.fromarray((change_mask_edit * 255).astype(np.uint8), mode='L')
                mask_pil = mask_pil.filter(ImageFilter.GaussianBlur(radius=blur_radius))
                change_mask_edit = np.array(mask_pil).astype(np.float32) / 255.0

            print(f"[EricQwenEdit] Using provided input mask")
        else:
            change_mask_edit = self._compute_change_mask(
                orig_at_edit_res_np, edit_np, threshold, expand_mask, blur_radius
            )
            changed_pixels = (change_mask_edit > 0.5).sum()
            total_pixels = change_mask_edit.shape[0] * change_mask_edit.shape[1]
            print(f"[EricQwenEdit] Auto-detected changes: {changed_pixels}/{total_pixels} pixels "
                  f"({100*changed_pixels/total_pixels:.1f}%)")

        # Step 3: Upscale both the edit and the mask to original resolution
        edit_upscaled = edit_pil.resize((orig_w, orig_h), resample)
        mask_upscaled_pil = Image.fromarray(
            (change_mask_edit * 255).clip(0, 255).astype(np.uint8), mode='L'
        ).resize((orig_w, orig_h), Image.BILINEAR)

        edit_upscaled_np = np.array(edit_upscaled).astype(np.float32) / 255.0
        mask_upscaled_np = np.array(mask_upscaled_pil).astype(np.float32) / 255.0

        # Step 4: Composite — blend edited areas onto original
        mask_3ch = mask_upscaled_np[:, :, np.newaxis]  # (H, W, 1) for broadcasting
        composite_np = orig_np * (1.0 - mask_3ch) + edit_upscaled_np * mask_3ch
        composite_np = composite_np.clip(0.0, 1.0)

        print(f"[EricQwenEdit] Composite: {orig_w}x{orig_h}")
        print(f"[EricQwenEdit] Upscale method: {upscale_method}")

        # Convert to tensors
        composite_tensor = torch.from_numpy(composite_np).unsqueeze(0)
        mask_tensor = torch.from_numpy(mask_upscaled_np).unsqueeze(0)
        upscaled_edit_tensor = torch.from_numpy(edit_upscaled_np).unsqueeze(0)

        return (composite_tensor, mask_tensor, upscaled_edit_tensor)


class EricQwenEditApplyMask:
    """
    Simple utility: apply a mask to blend two images.

    Takes a foreground image, a background image, and a mask.
    White areas of the mask show the foreground, black areas show the background.

    Useful for manually compositing edited regions with fine control.
    """

    CATEGORY = "Eric Qwen-Edit"
    FUNCTION = "apply_mask"
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "foreground": ("IMAGE", {
                    "tooltip": "Image to show in white areas of mask"
                }),
                "background": ("IMAGE", {
                    "tooltip": "Image to show in black areas of mask"
                }),
                "mask": ("MASK", {
                    "tooltip": "Blend mask: white=foreground, black=background"
                }),
            },
            "optional": {
                "blur_mask": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 50,
                    "step": 1,
                    "tooltip": "Additional blur applied to mask for softer blending"
                }),
            }
        }

    def apply_mask(
        self,
        foreground: torch.Tensor,
        background: torch.Tensor,
        mask: torch.Tensor,
        blur_mask: int = 0,
    ) -> Tuple[torch.Tensor]:
        """Blend two images using a mask."""
        # Extract from batch
        if foreground.dim() == 4:
            fg_np = foreground[0].cpu().numpy()
        else:
            fg_np = foreground.cpu().numpy()

        if background.dim() == 4:
            bg_np = background[0].cpu().numpy()
        else:
            bg_np = background.cpu().numpy()

        if mask.dim() == 3:
            mask_np = mask[0].cpu().numpy()
        else:
            mask_np = mask.cpu().numpy()

        # Ensure same size — resize foreground and mask to match background
        bg_h, bg_w = bg_np.shape[:2]
        fg_h, fg_w = fg_np.shape[:2]

        if fg_h != bg_h or fg_w != bg_w:
            fg_pil = Image.fromarray((fg_np * 255).clip(0, 255).astype(np.uint8))
            fg_pil = fg_pil.resize((bg_w, bg_h), Image.LANCZOS)
            fg_np = np.array(fg_pil).astype(np.float32) / 255.0

        mask_h, mask_w = mask_np.shape[:2]
        if mask_h != bg_h or mask_w != bg_w:
            mask_pil = Image.fromarray((mask_np * 255).clip(0, 255).astype(np.uint8), mode='L')
            mask_pil = mask_pil.resize((bg_w, bg_h), Image.BILINEAR)
            mask_np = np.array(mask_pil).astype(np.float32) / 255.0

        # Optional blur
        if blur_mask > 0:
            mask_pil = Image.fromarray((mask_np * 255).astype(np.uint8), mode='L')
            mask_pil = mask_pil.filter(ImageFilter.GaussianBlur(radius=blur_mask))
            mask_np = np.array(mask_pil).astype(np.float32) / 255.0

        # Blend
        mask_3ch = mask_np[:, :, np.newaxis]
        result = bg_np * (1.0 - mask_3ch) + fg_np * mask_3ch
        result = result.clip(0.0, 1.0)

        return (torch.from_numpy(result).unsqueeze(0),)
