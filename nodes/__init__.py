# Copyright (c) 2026 Eric Hiss. All rights reserved.
# Licensed under the terms in LICENSE.txt (CC BY-NC 4.0 / Commercial dual license).
# https://github.com/EricRollei/Eric_Qwen_Edit_Experiments
"""
Eric Qwen-Edit & Qwen-Image Node Definitions
"""

# ── Edit nodes ──────────────────────────────────────────────────────────
from .eric_qwen_edit_loader import EricQwenEditLoader, EricQwenEditUnload
from .eric_qwen_edit_node import EricQwenEditImage
from .eric_qwen_edit_inpaint import EricQwenEditInpaint
from .eric_qwen_edit_inpaint_transfer import EricQwenEditInpaintTransfer
from .eric_qwen_edit_lora import EricQwenEditApplyLoRA, EricQwenEditUnloadLoRA
from .eric_qwen_edit_component_loader import EricQwenEditComponentLoader
from .eric_qwen_edit_multi_image import EricQwenEditMultiImage
from .eric_qwen_edit_style_transfer import EricQwenEditStyleTransfer
from .eric_qwen_edit_delta import EricQwenEditDelta, EricQwenEditApplyMask
from .eric_qwen_edit_spectrum import EricQwenEditSpectrum

# ── Generation nodes (Qwen-Image / Qwen-Image-2512) ────────────────────
from .eric_qwen_image_loader import EricQwenImageLoader, EricQwenImageUnload
from .eric_qwen_image_component_loader import EricQwenImageComponentLoader
from .eric_qwen_image_generate import EricQwenImageGenerate
from .eric_qwen_image_lora import EricQwenImageApplyLoRA, EricQwenImageUnloadLoRA
from .eric_qwen_image_multistage import EricQwenImageMultiStage
from .eric_qwen_image_ultragen import EricQwenImageUltraGen
from .eric_qwen_image_spectrum import EricQwenImageSpectrum
from .eric_qwen_prompt_rewriter import EricQwenPromptRewriter
from .eric_qwen_upscale_vae import EricQwenUpscaleVAELoader

NODE_CLASS_MAPPINGS = {
    # Edit
    "Eric Qwen-Edit Loader": EricQwenEditLoader,
    "Eric Qwen-Edit Unload": EricQwenEditUnload,
    "Eric Qwen-Edit Image": EricQwenEditImage,
    "Eric Qwen-Edit Inpaint": EricQwenEditInpaint,
    "Eric Qwen-Edit Inpaint Transfer": EricQwenEditInpaintTransfer,
    "Eric Qwen-Edit Apply LoRA": EricQwenEditApplyLoRA,
    "Eric Qwen-Edit Unload LoRA": EricQwenEditUnloadLoRA,
    "Eric Qwen-Edit Component Loader": EricQwenEditComponentLoader,
    "Eric Qwen-Edit Multi-Image": EricQwenEditMultiImage,
    "Eric Qwen-Edit Style Transfer": EricQwenEditStyleTransfer,
    "Eric Qwen-Edit Delta": EricQwenEditDelta,
    "Eric Qwen-Edit Apply Mask": EricQwenEditApplyMask,
    "Eric Qwen-Edit Spectrum": EricQwenEditSpectrum,
    # Generation
    "Eric Qwen-Image Loader": EricQwenImageLoader,
    "Eric Qwen-Image Unload": EricQwenImageUnload,
    "Eric Qwen-Image Component Loader": EricQwenImageComponentLoader,
    "Eric Qwen-Image Generate": EricQwenImageGenerate,
    "Eric Qwen-Image Apply LoRA": EricQwenImageApplyLoRA,
    "Eric Qwen-Image Unload LoRA": EricQwenImageUnloadLoRA,
    "Eric Qwen-Image Multi-Stage": EricQwenImageMultiStage,
    "Eric Qwen-Image UltraGen": EricQwenImageUltraGen,
    "Eric Qwen-Image Spectrum": EricQwenImageSpectrum,
    "Eric Qwen Prompt Rewriter": EricQwenPromptRewriter,
    # Utility
    "Eric Qwen Upscale VAE Loader": EricQwenUpscaleVAELoader,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    # Edit
    "Eric Qwen-Edit Loader": "Eric Qwen-Edit Load Model",
    "Eric Qwen-Edit Unload": "Eric Qwen-Edit Unload",
    "Eric Qwen-Edit Image": "Eric Qwen-Edit Image",
    "Eric Qwen-Edit Inpaint": "Eric Qwen-Edit Inpaint",
    "Eric Qwen-Edit Inpaint Transfer": "Eric Qwen-Edit Inpaint Transfer",
    "Eric Qwen-Edit Apply LoRA": "Eric Qwen-Edit Apply LoRA",
    "Eric Qwen-Edit Unload LoRA": "Eric Qwen-Edit Unload LoRA",
    "Eric Qwen-Edit Component Loader": "Eric Qwen-Edit Component Loader",
    "Eric Qwen-Edit Multi-Image": "Eric Qwen-Edit Multi-Image Fusion",
    "Eric Qwen-Edit Style Transfer": "Eric Qwen-Edit Style Transfer",
    "Eric Qwen-Edit Delta": "Eric Qwen-Edit Delta Overlay",
    "Eric Qwen-Edit Apply Mask": "Eric Qwen-Edit Apply Mask",
    "Eric Qwen-Edit Spectrum": "Eric Qwen-Edit Spectrum Accelerator",
    # Generation
    "Eric Qwen-Image Loader": "Eric Qwen-Image Load Model",
    "Eric Qwen-Image Unload": "Eric Qwen-Image Unload",
    "Eric Qwen-Image Component Loader": "Eric Qwen-Image Component Loader",
    "Eric Qwen-Image Generate": "Eric Qwen-Image Generate",
    "Eric Qwen-Image Apply LoRA": "Eric Qwen-Image Apply LoRA",
    "Eric Qwen-Image Unload LoRA": "Eric Qwen-Image Unload LoRA",
    "Eric Qwen-Image Multi-Stage": "Eric Qwen-Image Multi-Stage Generate",
    "Eric Qwen-Image UltraGen": "Eric Qwen-Image UltraGen",
    "Eric Qwen-Image Spectrum": "Eric Qwen-Image Spectrum Accelerator",
    "Eric Qwen Prompt Rewriter": "Eric Qwen Prompt Rewriter",
    # Utility
    "Eric Qwen Upscale VAE Loader": "Eric Qwen Upscale VAE Loader (2×)",
}
