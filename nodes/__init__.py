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
from .eric_qwen_edit_lora import EricQwenEditApplyLoRA, EricQwenEditUnloadLoRA, EricQwenEditDiagnoseLoRA
from .eric_qwen_edit_component_loader import EricQwenEditComponentLoader
from .eric_qwen_edit_multi_image import EricQwenEditMultiImage
from .eric_qwen_edit_unify import EricQwenEditUnify
from .eric_qwen_edit_style_transfer import EricQwenEditStyleTransfer
from .eric_qwen_edit_delta import EricQwenEditDelta, EricQwenEditApplyMask
from .eric_qwen_edit_spectrum import EricQwenEditSpectrum

# ── Conditioning system (encode → manipulate → conditioned edit) ────────
from .eric_qwen_conditioning_encode import EricQwenConditioningEncode
from .eric_qwen_conditioned_edit import EricQwenConditionedEdit
from .eric_qwen_conditioning_ops import (
    EricQwenConditioningInterpolate,
    EricQwenConditioningBlend,
)
from .eric_qwen_direction_ops import (
    EricQwenDirectionCompute,
    EricQwenDirectionApply,
    EricQwenDirectionAverage,
    EricQwenDirectionAverageFromFolder,
    EricQwenDirectionSave,
    EricQwenDirectionLoad,
)
from .eric_qwen_direction_inspect import EricQwenDirectionInspect

# ── Generation nodes (Qwen-Image / Qwen-Image-2512) ────────────────────
from .eric_qwen_image_loader import EricQwenImageLoader, EricQwenImageUnload
from .eric_qwen_image_component_loader import EricQwenImageComponentLoader
from .eric_qwen_image_generate import EricQwenImageGenerate
from .eric_qwen_image_lora import EricQwenImageApplyLoRA, EricQwenImageUnloadLoRA, EricQwenImageDiagnoseLoRA
from .eric_qwen_image_multistage import EricQwenImageMultiStage
from .eric_qwen_image_ultragen import EricQwenImageUltraGen
from .eric_qwen_image_spectrum import EricQwenImageSpectrum
from .eric_qwen_image_controlnet_loader import EricQwenImageControlNetLoader, EricQwenImageControlNetUnload
from .eric_qwen_image_ultragen_cn import EricQwenImageUltraGenCN
from .eric_qwen_image_ultragen_inpaint_cn import EricQwenImageUltraGenInpaintCN
from .eric_qwen_image_harmonize import EricQwenImageHarmonize
from .eric_qwen_prompt_rewriter import EricQwenPromptRewriter
from .eric_qwen_inpaint_prompt_rewriter import EricQwenInpaintPromptRewriter
from .eric_qwen_controlnet_prompt_rewriter import EricQwenControlNetPromptRewriter
from .eric_qwen_upscale_vae import EricQwenUpscaleVAELoader

# ── Gen-Searcher pipeline ──────────────────────────────────────────────
from .eric_qwen_grounded_generate import EricQwenGroundedGenerate
from .eric_qwen_reference_describer import EricQwenReferenceDescriber

NODE_CLASS_MAPPINGS = {
    # ── Edit ──────────────────────────────────────────────────────────────
    "Eric Qwen-Edit Loader":               EricQwenEditLoader,
    "Eric Qwen-Edit Unload":               EricQwenEditUnload,
    "Eric Qwen-Edit Image":                EricQwenEditImage,
    "Eric Qwen-Edit Inpaint":              EricQwenEditInpaint,
    "Eric Qwen-Edit Inpaint Transfer":     EricQwenEditInpaintTransfer,
    "Eric Qwen-Edit Apply LoRA":           EricQwenEditApplyLoRA,
    "Eric Qwen-Edit Unload LoRA":          EricQwenEditUnloadLoRA,
    "Eric Qwen-Edit Diagnose LoRA":        EricQwenEditDiagnoseLoRA,
    "Eric Qwen-Edit Component Loader":     EricQwenEditComponentLoader,
    "Eric Qwen-Edit Multi-Image":          EricQwenEditMultiImage,
    "Eric Qwen-Edit Unify":                EricQwenEditUnify,
    "Eric Qwen-Edit Style Transfer":       EricQwenEditStyleTransfer,
    "Eric Qwen-Edit Delta":                EricQwenEditDelta,
    "Eric Qwen-Edit Apply Mask":           EricQwenEditApplyMask,
    "Eric Qwen-Edit Spectrum":             EricQwenEditSpectrum,
    # ── Conditioning system ───────────────────────────────────────────────
    "Eric Qwen Conditioning Encode":            EricQwenConditioningEncode,
    "Eric Qwen Conditioned Edit":               EricQwenConditionedEdit,
    "Eric Qwen Conditioning Interpolate":       EricQwenConditioningInterpolate,
    "Eric Qwen Conditioning Blend":             EricQwenConditioningBlend,
    "Eric Qwen Direction Compute":              EricQwenDirectionCompute,
    "Eric Qwen Direction Apply":                EricQwenDirectionApply,
    "Eric Qwen Direction Average":              EricQwenDirectionAverage,
    "Eric Qwen Direction Average From Folder":  EricQwenDirectionAverageFromFolder,
    "Eric Qwen Direction Save":                 EricQwenDirectionSave,
    "Eric Qwen Direction Load":                 EricQwenDirectionLoad,
    "Eric Qwen Direction Inspect":              EricQwenDirectionInspect,
    # ── Generation ────────────────────────────────────────────────────────
    "Eric Qwen-Image Loader":              EricQwenImageLoader,
    "Eric Qwen-Image Unload":              EricQwenImageUnload,
    "Eric Qwen-Image Component Loader":    EricQwenImageComponentLoader,
    "Eric Qwen-Image Generate":            EricQwenImageGenerate,
    "Eric Qwen-Image Apply LoRA":          EricQwenImageApplyLoRA,
    "Eric Qwen-Image Unload LoRA":         EricQwenImageUnloadLoRA,
    "Eric Qwen-Image Diagnose LoRA":       EricQwenImageDiagnoseLoRA,
    "Eric Qwen-Image Multi-Stage":         EricQwenImageMultiStage,
    "Eric Qwen-Image UltraGen":            EricQwenImageUltraGen,
    "Eric Qwen-Image Spectrum":            EricQwenImageSpectrum,
    "Eric Qwen-Image ControlNet Loader":   EricQwenImageControlNetLoader,
    "Eric Qwen-Image ControlNet Unload":   EricQwenImageControlNetUnload,
    "Eric Qwen-Image UltraGen CN":         EricQwenImageUltraGenCN,
    "Eric Qwen-Image UltraGen Inpaint CN": EricQwenImageUltraGenInpaintCN,
    "Eric Qwen-Image Harmonize":           EricQwenImageHarmonize,
    "Eric Qwen Prompt Rewriter":           EricQwenPromptRewriter,
    "Eric Qwen Inpaint Prompt Rewriter":   EricQwenInpaintPromptRewriter,
    "Eric Qwen ControlNet Prompt Rewriter": EricQwenControlNetPromptRewriter,
    # ── Utility ───────────────────────────────────────────────────────────
    "Eric Qwen Upscale VAE Loader":        EricQwenUpscaleVAELoader,
    # ── Gen-Searcher ──────────────────────────────────────────────────────
    "Eric Qwen Grounded Generate":         EricQwenGroundedGenerate,
    "Eric Qwen Reference Describer":       EricQwenReferenceDescriber,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    # ── Edit ──────────────────────────────────────────────────────────────
    "Eric Qwen-Edit Loader":               "Eric Qwen-Edit Load Model",
    "Eric Qwen-Edit Unload":               "Eric Qwen-Edit Unload",
    "Eric Qwen-Edit Image":                "Eric Qwen-Edit Image",
    "Eric Qwen-Edit Inpaint":              "Eric Qwen-Edit Inpaint",
    "Eric Qwen-Edit Inpaint Transfer":     "Eric Qwen-Edit Inpaint Transfer",
    "Eric Qwen-Edit Apply LoRA":           "Eric Qwen-Edit Apply LoRA",
    "Eric Qwen-Edit Unload LoRA":          "Eric Qwen-Edit Unload LoRA",
    "Eric Qwen-Edit Diagnose LoRA":        "Eric Qwen-Edit Diagnose LoRA",
    "Eric Qwen-Edit Component Loader":     "Eric Qwen-Edit Component Loader",
    "Eric Qwen-Edit Multi-Image":          "Eric Qwen-Edit Multi-Image Fusion",
    "Eric Qwen-Edit Unify":                "Eric Qwen-Edit Composite Unify",
    "Eric Qwen-Edit Style Transfer":       "Eric Qwen-Edit Style Transfer",
    "Eric Qwen-Edit Delta":                "Eric Qwen-Edit Delta Overlay",
    "Eric Qwen-Edit Apply Mask":           "Eric Qwen-Edit Apply Mask",
    "Eric Qwen-Edit Spectrum":             "Eric Qwen-Edit Spectrum Accelerator",
    # ── Conditioning system ───────────────────────────────────────────────
    "Eric Qwen Conditioning Encode":            "Eric Qwen Conditioning Encode",
    "Eric Qwen Conditioned Edit":               "Eric Qwen Conditioned Edit",
    "Eric Qwen Conditioning Interpolate":       "Eric Qwen Conditioning Interpolate",
    "Eric Qwen Conditioning Blend":             "Eric Qwen Conditioning Blend",
    "Eric Qwen Direction Compute":              "Eric Qwen Direction Compute",
    "Eric Qwen Direction Apply":                "Eric Qwen Direction Apply",
    "Eric Qwen Direction Average":              "Eric Qwen Direction Average",
    "Eric Qwen Direction Average From Folder":  "Eric Qwen Direction Average From Folder",
    "Eric Qwen Direction Save":                 "Eric Qwen Direction Save",
    "Eric Qwen Direction Load":                 "Eric Qwen Direction Load",
    "Eric Qwen Direction Inspect":              "Eric Qwen Direction Inspect",
    # ── Generation ────────────────────────────────────────────────────────
    "Eric Qwen-Image Loader":              "Eric Qwen-Image Load Model",
    "Eric Qwen-Image Unload":              "Eric Qwen-Image Unload",
    "Eric Qwen-Image Component Loader":    "Eric Qwen-Image Component Loader",
    "Eric Qwen-Image Generate":            "Eric Qwen-Image Generate",
    "Eric Qwen-Image Apply LoRA":          "Eric Qwen-Image Apply LoRA",
    "Eric Qwen-Image Unload LoRA":         "Eric Qwen-Image Unload LoRA",
    "Eric Qwen-Image Diagnose LoRA":       "Eric Qwen-Image Diagnose LoRA",
    "Eric Qwen-Image Multi-Stage":         "Eric Qwen-Image Multi-Stage Generate",
    "Eric Qwen-Image UltraGen":            "Eric Qwen-Image UltraGen",
    "Eric Qwen-Image Spectrum":            "Eric Qwen-Image Spectrum Accelerator",
    "Eric Qwen-Image ControlNet Loader":   "Eric Qwen-Image ControlNet Loader",
    "Eric Qwen-Image ControlNet Unload":   "Eric Qwen-Image ControlNet Unload",
    "Eric Qwen-Image UltraGen CN":         "Eric Qwen-Image UltraGen (ControlNet)",
    "Eric Qwen-Image UltraGen Inpaint CN": "Eric Qwen-Image UltraGen Inpaint (ControlNet)",
    "Eric Qwen-Image Harmonize":           "Eric Qwen-Image Composite Harmonize",
    "Eric Qwen Prompt Rewriter":           "Eric Qwen Prompt Rewriter",
    "Eric Qwen Inpaint Prompt Rewriter":   "Eric Qwen Inpaint Prompt Rewriter (Vision)",
    "Eric Qwen ControlNet Prompt Rewriter": "Eric Qwen ControlNet Prompt Rewriter (Vision)",
    # ── Utility ───────────────────────────────────────────────────────────
    "Eric Qwen Upscale VAE Loader":        "Eric Qwen Upscale VAE Loader (2×)",
    # ── Gen-Searcher ──────────────────────────────────────────────────────
    "Eric Qwen Grounded Generate":         "Eric Qwen Grounded Generate",
    "Eric Qwen Reference Describer":       "Eric Qwen Reference Describer (VL→Text)",
}
