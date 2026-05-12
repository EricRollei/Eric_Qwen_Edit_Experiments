# Copyright (c) 2026 Eric Hiss. All rights reserved.
# Licensed under the terms in LICENSE.txt (CC BY-NC 4.0 / Commercial dual license).
# https://github.com/EricRollei/Eric_Qwen_Edit_Experiments
"""
Eric Qwen-Image LoRA Node
Apply / unload LoRA weights on the Qwen-Image (text-to-image) pipeline.

Multiple Apply LoRA nodes can be chained to stack several LoRAs with
independent weights.

Model Credits:
- Qwen-Image developed by Qwen Team (Alibaba)
- https://github.com/QwenLM/Qwen-Image

Author: Eric Hiss (GitHub: EricRollei)
"""

import os
import re
from typing import Tuple

# Re-use the same helpers from the edit LoRA node (they are model-agnostic)
from .eric_qwen_edit_lora import (get_lora_list, get_lora_full_path,
                                  load_lora_with_key_fix,
                                  _set_adapters_safe,
                                  _diagnose_lora_compatibility)


# ═══════════════════════════════════════════════════════════════════════
#  Apply LoRA
# ═══════════════════════════════════════════════════════════════════════

class EricQwenImageApplyLoRA:
    """
    Apply a LoRA to the Qwen-Image generation pipeline.

    This node loads LoRA weights onto the transformer.  Multiple LoRA
    nodes can be chained to apply several LoRAs with different weights.

    Workflow example:
    [Load Model] → [Apply LoRA (style)] → [Apply LoRA (subject)] → [Generate]

    Per-stage LoRA weights (for UltraGen multi-stage generation):
    - weight_stage1: strength for Stage 1 (draft / composition)
    - weight_stage2: strength for Stage 2 (refinement)
    - weight_stage3: strength for Stage 3 (final polish)
    - 0.0 = LoRA disabled for that stage (default)
    - 1.0 = full strength
    - >1.0 = amplified (may cause artifacts)
    - For non-UltraGen nodes, weight_stage1 is used as the effective weight.

    Examples:
    - Detail LoRA:    S1=0, S2=0.8, S3=1.0 — no detail in draft, full in polish
    - Lightning LoRA:  S1=0.5, S2=1.0, S3=0 — skip fast-LoRA in short final stage
    - Style LoRA:      S1=1.0, S2=0.7, S3=0.3 — strong in draft, fade in later stages

    LoRAs are loaded from ComfyUI's standard loras folder:
    ComfyUI/models/loras/
    """

    CATEGORY = "Eric Qwen-Image"
    FUNCTION = "apply_lora"
    RETURN_TYPES = ("QWEN_IMAGE_PIPELINE",)
    RETURN_NAMES = ("pipeline",)

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "pipeline": ("QWEN_IMAGE_PIPELINE",),
                "lora_name": (get_lora_list(), {
                    "tooltip": "Select LoRA from ComfyUI/models/loras/"
                }),
                "weight_stage1": ("FLOAT", {
                    "default": 0.0,
                    "min": -2.0,
                    "max": 2.0,
                    "step": 0.05,
                    "tooltip": "Stage 1 (draft/composition) weight. Also used as the weight for non-UltraGen nodes. 0 = disabled."
                }),
                "weight_stage2": ("FLOAT", {
                    "default": 0.0,
                    "min": -2.0,
                    "max": 2.0,
                    "step": 0.05,
                    "tooltip": "Stage 2 (refinement) weight. 0 = disabled."
                }),
                "weight_stage3": ("FLOAT", {
                    "default": 0.0,
                    "min": -2.0,
                    "max": 2.0,
                    "step": 0.05,
                    "tooltip": "Stage 3 (final polish) weight. 0 = disabled."
                }),
            },
            "optional": {
                "lora_path_override": ("STRING", {
                    "default": "",
                    "tooltip": "Optional: Override with custom path (leave empty to use dropdown)"
                }),
                "keep_loaded": ("BOOLEAN", {
                    "default": True,
                    "tooltip": (
                        "Keep the LoRA adapter loaded between runs.\n"
                        "\u2022 True (default): load once, reuse on subsequent runs\n"
                        "  (faster; useful for final render settings).\n"
                        "\u2022 False: unload ALL adapters before loading this LoRA\n"
                        "  on every run. Ensures a clean state each generation;\n"
                        "  useful when testing or swapping LoRAs frequently."
                    )
                }),
            }
        }

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        # Refresh LoRA list on each check
        return float("nan")

    def apply_lora(
        self,
        pipeline: dict,
        lora_name: str,
        weight_stage1: float = 0.0,
        weight_stage2: float = 0.0,
        weight_stage3: float = 0.0,
        lora_path_override: str = "",
        keep_loaded: bool = True,
    ) -> Tuple[dict]:
        """Apply LoRA to the Qwen-Image pipeline."""
        pipe = pipeline["pipeline"]

        # Determine which path to use
        if lora_path_override and lora_path_override.strip():
            lora_path = lora_path_override.strip()
            if not os.path.exists(lora_path):
                raise ValueError(f"LoRA file not found: {lora_path}")
        else:
            if lora_name == "none" or not lora_name:
                print("[EricQwenImage] LoRA: none selected, skipping")
                return (pipeline,)
            lora_path = get_lora_full_path(lora_name)
            if lora_path is None:
                raise ValueError(f"LoRA file not found: {lora_name}")

        lora_filename = os.path.basename(lora_path)
        adapter_name = os.path.splitext(lora_filename)[0]
        # PEFT/PyTorch ModuleDict keys cannot contain '.', so sanitize the
        # adapter name (filenames like 'Qwen-Image-2512-Lightning-8steps-V1.0-fp32'
        # are common). Replace '.' and whitespace with '_'.
        adapter_name = re.sub(r"[.\s]", "_", adapter_name)

        # Use weight_stage1 as the immediate effective weight (for non-UltraGen
        # single-pass generation).  UltraGen overrides per stage before each pipe() call.
        effective_weight = weight_stage1

        print(f"[EricQwenImage] Applying LoRA: {lora_filename}")
        print(f"[EricQwenImage] Path: {lora_path}")
        print(f"[EricQwenImage] Stage weights: S1={weight_stage1}, S2={weight_stage2}, S3={weight_stage3}, keep_loaded={keep_loaded}")

        # If keep_loaded=False, unload everything first for a clean state each run
        if not keep_loaded:
            try:
                pipe.unload_lora_weights()
                pipeline.pop("applied_loras", None)
                print("[EricQwenImage] Unloaded previous LoRAs (keep_loaded=False)")
            except Exception:
                pass

        # Check if this adapter is already loaded
        loaded_adapters = set()
        try:
            adapter_list = pipe.get_list_adapters()
            for component_adapters in adapter_list.values():
                loaded_adapters.update(component_adapters)
        except Exception:
            pass

        try:
            if adapter_name in loaded_adapters:
                # Already loaded — just update the weight
                _set_adapters_safe(pipe, adapter_name, effective_weight,
                                   log_prefix="[EricQwenImage]")
                print(f"[EricQwenImage] LoRA already loaded, updated weight: {adapter_name} -> {effective_weight}")
            else:
                # Load fresh (with automatic key normalization fallback)
                load_lora_with_key_fix(pipe, lora_path, adapter_name,
                                      log_prefix="[EricQwenImage]",
                                      weight=effective_weight)
                _set_adapters_safe(pipe, adapter_name, effective_weight,
                                   log_prefix="[EricQwenImage]")
                print(f"[EricQwenImage] LoRA applied successfully: {adapter_name}")

        except Exception as e:
            print(f"[EricQwenImage] Error loading LoRA: {e}")
            raise

        # Track applied adapters in the pipeline dict
        if "applied_loras" not in pipeline:
            pipeline["applied_loras"] = {}
        pipeline["applied_loras"][adapter_name] = {
            "path": lora_path,
            "filename": lora_filename,
            "weight_stage1": weight_stage1,
            "weight_stage2": weight_stage2,
            "weight_stage3": weight_stage3,
        }

        return (pipeline,)


# ═══════════════════════════════════════════════════════════════════════
#  Unload LoRA
# ═══════════════════════════════════════════════════════════════════════

class EricQwenImageUnloadLoRA:
    """
    Unload all LoRAs from the Qwen-Image generation pipeline.

    Use this to reset the model to its base state before applying
    different LoRAs, or to free memory.
    """

    CATEGORY = "Eric Qwen-Image"
    FUNCTION = "unload_lora"
    RETURN_TYPES = ("QWEN_IMAGE_PIPELINE",)
    RETURN_NAMES = ("pipeline",)

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "pipeline": ("QWEN_IMAGE_PIPELINE",),
            },
        }

    def unload_lora(self, pipeline: dict) -> Tuple[dict]:
        """Unload all LoRAs from the pipeline."""
        pipe = pipeline["pipeline"]

        try:
            pipe.unload_lora_weights()
            print("[EricQwenImage] All LoRAs unloaded")
        except Exception as e:
            print(f"[EricQwenImage] Note: {e}")

        # Clear tracking
        pipeline.pop("applied_loras", None)

        return (pipeline,)


# ═══════════════════════════════════════════════════════════════
#  Diagnose LoRA
# ═══════════════════════════════════════════════════════════════

class EricQwenImageDiagnoseLoRA:
    """
    Diagnose LoRA compatibility with the Qwen-Image pipeline.

    Loads the LoRA file and inspects its keys WITHOUT applying it,
    then prints a compatibility report to the console and returns it
    as a STRING.  Useful for understanding why a LoRA produces noise
    or fails to load.

    The report includes:
    - Adapter type (LoRA / LoKR / LoHa / unknown)
    - Rank and alpha values (for standard LoRA)
    - How many adapter modules match the transformer architecture
    - A COMPATIBLE / PARTIAL / INCOMPATIBLE verdict
    """

    CATEGORY = "Eric Qwen-Image"
    FUNCTION = "diagnose"
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("report",)
    OUTPUT_NODE = True

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "pipeline": ("QWEN_IMAGE_PIPELINE",),
                "lora_name": (get_lora_list(), {
                    "tooltip": "LoRA to analyze (inspected but NOT applied)"
                }),
            },
            "optional": {
                "lora_path_override": ("STRING", {
                    "default": "",
                    "tooltip": "Override with full path instead of dropdown"
                }),
            }
        }

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("nan")

    def diagnose(
        self,
        pipeline: dict,
        lora_name: str,
        lora_path_override: str = "",
    ) -> Tuple:
        pipe = pipeline["pipeline"]
        transformer = pipe.transformer

        if lora_path_override and lora_path_override.strip():
            lora_path = lora_path_override.strip()
        else:
            if lora_name == "none" or not lora_name:
                return ("No LoRA selected.",)
            lora_path = get_lora_full_path(lora_name)
            if lora_path is None:
                return (f"ERROR: LoRA file not found: {lora_name}",)

        if not os.path.exists(lora_path):
            return (f"ERROR: File not found: {lora_path}",)

        report = _diagnose_lora_compatibility(lora_path, transformer)
        print(report)
        return (report,)
