# Copyright (c) 2026 Eric Hiss. All rights reserved.
# Licensed under the terms in LICENSE.txt (CC BY-NC 4.0 / Commercial dual license).
# https://github.com/EricRollei/Eric_Qwen_Edit_Experiments
"""
Eric Qwen ControlNet Prompt Rewriter

Vision-LLM-powered prompt rewriter for ControlNet-guided image generation.
Takes the reference image + a short user intent + the chosen CN type and
produces a full descriptive prompt suitable for QwenImageControlNetPipeline.

Unlike inpainting, the ControlNet pipeline does NOT retain source pixels —
it only extracts structural guidance (edges, depth, pose, etc.).  The prompt
must therefore be a COMPLETE scene description telling the model what to
render within those structural constraints.

The VLM sees the original reference image and understands its composition,
subjects, spatial layout, lighting etc.  It then writes a rich descriptive
prompt that applies the user's creative direction while preserving the
structural elements the chosen ControlNet type will enforce.

Also outputs a cn_type_index integer (0-3) matching the selected control
type, which downstream nodes can use via a switch to auto-select the
correct preprocessor.

Works with any OpenAI-compatible vision API:
  - Ollama  (qwen3-vl, llava, gemma3, etc.)
  - LM Studio
  - OpenAI  (gpt-4o-mini, gpt-4o)
  - DeepSeek VL
  - Any other multimodal chat completions endpoint

Author: Eric Hiss (GitHub: EricRollei)
"""

import re
import torch
import numpy as np
from PIL import Image
from typing import Tuple

# Reuse VLM infrastructure from the inpaint rewriter
from .eric_qwen_inpaint_prompt_rewriter import (
    _pil_to_base64,
    _tensor_to_pil,
    _call_vision_api,
)
from .eric_qwen_prompt_rewriter import (
    _resolve_api_key,
)


# ═══════════════════════════════════════════════════════════════════════
#  ControlNet type definitions
# ═══════════════════════════════════════════════════════════════════════

CN_TYPES = ["Canny", "SoftEdge", "Depth", "Pose"]
CN_TYPE_INDEX = {name: idx for idx, name in enumerate(CN_TYPES)}

# What each CN type preserves — used to inform the VLM
CN_TYPE_DESCRIPTIONS = {
    "Canny": (
        "Canny edge detection. Preserves hard edges, outlines, and contours "
        "of all objects. The model will follow the exact edge map — object "
        "shapes, silhouettes, and boundaries will be maintained. Interior "
        "textures, colors, materials, and fine surface details are NOT "
        "preserved and must be described in the prompt."
    ),
    "SoftEdge": (
        "Soft edge detection (AnyLine/HED). Preserves softer, more organic "
        "edges and contours with graduated boundaries. Gives the model more "
        "freedom for interior details and subtle shapes than Canny, while "
        "still maintaining the overall composition and major structural "
        "lines. Colors, textures, materials are NOT preserved."
    ),
    "Depth": (
        "Depth map estimation. Preserves the spatial depth relationships — "
        "what is near vs far, the 3D layout and relative positions of "
        "objects. Does NOT preserve edges, shapes, outlines, colors or "
        "textures — only the depth arrangement. The model has maximum "
        "freedom to change object appearance while keeping spatial layout."
    ),
    "Pose": (
        "Pose skeleton detection (DWPose/OpenPose). Preserves body pose, "
        "joint positions, and limb angles for human figures. Does NOT "
        "preserve face details, clothing, body proportions, background, "
        "or anything other than the skeletal pose. The model has complete "
        "freedom to change the person's appearance, outfit, setting, etc."
    ),
}


# ═══════════════════════════════════════════════════════════════════════
#  System prompts — ControlNet generation
# ═══════════════════════════════════════════════════════════════════════

CONTROLNET_SYSTEM_PROMPT_EN = """You write detailed image generation prompts for AI art, guided by a ControlNet structural constraint.

The user will show you a reference image and describe what they want the new image to look like. A ControlNet will extract structural guidance from the reference image, but the model will NOT see the original pixels — only the structural map.

Your job: Write a COMPLETE descriptive prompt for the new image. The prompt must describe EVERYTHING the model needs to render — subject, appearance, clothing, environment, materials, lighting, colors, atmosphere, style. The ControlNet only provides structure; your prompt provides all visual content.

CONTROLNET TYPE: {cn_type}
WHAT IT PRESERVES: {cn_description}
WHAT IT DOES NOT PRESERVE (you must describe): Everything else — colors, textures, materials, skin, clothing details, background appearance, lighting mood, artistic style.

Prompt structure (follow this order):
1. Main subject with full appearance details (face, hair, body, clothing, accessories)
2. Action/pose context (what the subject is doing — the pose is structurally guided but needs narrative context)
3. Environment and background (setting, surfaces, objects, depth)
4. Lighting, atmosphere, and mood
5. Artistic style and rendering quality

Rules:
- One flowing paragraph, 200-400 words
- Descriptive language only (describe what IS, not what to do)
- Include rich material/texture detail — the model has no other visual reference
- Describe specific colors, patterns, reflections, surface qualities
- Describe spatial relationships and depth cues
- Include atmospheric effects (haze, particles, light rays, shadows)
- NEVER reference the original image ("like the original", "same as before")
- NEVER use instructive language ("change", "replace", "make it")
- Apply the user's creative direction throughout
- Output ONLY the prompt — no commentary, no quotes, no markdown"""

CONTROLNET_SYSTEM_PROMPT_ZH = """你为AI图像生成撰写详细的提示词，配合ControlNet结构引导使用。

用户会展示一张参考图像并描述新图像的期望效果。ControlNet将从参考图像提取结构引导，但模型不会看到原始像素——只有结构图。

你的任务：为新图像撰写完整的描述性提示词。提示词必须描述模型需要渲染的一切——主体、外观、服装、环境、材质、光照、色彩、氛围、风格。ControlNet只提供结构；你的提示词提供所有视觉内容。

控制类型：{cn_type}
保留内容：{cn_description}
不保留（需要描述）：其他一切——颜色、纹理、材质、皮肤、服装细节、背景外观、光照氛围、艺术风格。

提示词结构（按此顺序）：
1. 主体及完整外观细节（面部、头发、身体、服装、配饰）
2. 动作/姿态上下文（主体在做什么）
3. 环境和背景（场景、表面、物体、纵深）
4. 光照、氛围和情绪
5. 艺术风格和渲染质量

规则：
- 一段连贯文字，200-400字
- 使用描述性语言（描述“是什么”）
- 包含丰富的材质/纹理细节
- 描述具体的颜色、图案、反射、表面质感
- 描述空间关系和深度线索
- 包含大气效果（雾霽、光线、阴影）
- 绝不引用原始图像
- 绝不使用指令性语言
- 将用户的创意方向融入全文
- 仅输出提示词"""


# ═══════════════════════════════════════════════════════════════════════
#  ComfyUI Node
# ═══════════════════════════════════════════════════════════════════════

class EricQwenControlNetPromptRewriter:
    """
    Vision-LLM-powered prompt rewriter for ControlNet-guided generation.

    Sends the reference image to a vision model along with the user's
    creative intent and the selected CN type.  The VLM writes a FULL
    descriptive prompt — because the ControlNet only provides structure,
    not appearance.  The prompt must describe everything the diffusion
    model needs to render.

    Also outputs a cn_type_index (0=Canny, 1=SoftEdge, 2=Depth, 3=Pose)
    that can drive a downstream switch node to auto-select the correct
    preprocessor.

    Works with any OpenAI-compatible vision API (Ollama, LM Studio,
    OpenAI, etc.).  API keys loaded from env vars or api_keys.ini.
    """

    CATEGORY = "Eric Qwen-Image"
    FUNCTION = "rewrite"
    RETURN_TYPES = ("STRING", "INT",)
    RETURN_NAMES = ("prompt", "cn_type_index",)

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE", {
                    "tooltip": "Reference image for the VLM to analyse."
                }),
                "creative_direction": ("STRING", {
                    "multiline": True,
                    "default": "",
                    "tooltip": (
                        "Describe how the new image should differ from\n"
                        "the reference. Can be brief — the VLM will\n"
                        "expand it into a full descriptive prompt.\n"
                        "Example: 'cyberpunk warrior in neon-lit alley'\n"
                        "Example: 'oil painting in Rembrandt style'\n"
                        "Example: 'same scene but underwater'"
                    )
                }),
                "cn_type": (CN_TYPES, {
                    "default": "Canny",
                    "tooltip": (
                        "Which ControlNet preprocessor will be used.\n"
                        "Informs the VLM about what structural info\n"
                        "is preserved vs what must be described:\n"
                        "• Canny — hard edges/outlines (idx 0)\n"
                        "• SoftEdge — soft contours (idx 1)\n"
                        "• Depth — spatial depth layout (idx 2)\n"
                        "• Pose — body skeleton only (idx 3)"
                    )
                }),
                "api_url": ("STRING", {
                    "default": "http://localhost:1234/v1",
                    "tooltip": (
                        "OpenAI-compatible vision API URL.\n"
                        "  Ollama: http://localhost:11434/v1\n"
                        "  LM Studio: http://localhost:1234/v1\n"
                        "  OpenAI: https://api.openai.com/v1"
                    )
                }),
                "model": ("STRING", {
                    "default": "qwen3-vl",
                    "tooltip": (
                        "Vision model name. Must support image input.\n"
                        "  Ollama: qwen3-vl, qwen2.5-vl:7b, llava\n"
                        "  OpenAI: gpt-4o-mini, gpt-4o"
                    )
                }),
            },
            "optional": {
                "language": (["English", "Chinese"], {
                    "default": "English",
                    "tooltip": "Language for the rewritten prompt."
                }),
                "temperature": ("FLOAT", {
                    "default": 0.7,
                    "min": 0.0,
                    "max": 2.0,
                    "step": 0.1,
                    "tooltip": "VLM temperature. Lower = more faithful."
                }),
                "max_tokens": ("INT", {
                    "default": 2048,
                    "min": 256,
                    "max": 8192,
                    "step": 128,
                    "tooltip": "Max tokens for the VLM response."
                }),
                "custom_instructions": ("STRING", {
                    "multiline": True,
                    "default": "",
                    "tooltip": (
                        "Additional instructions for the VLM.\n"
                        "Example: 'Use a cinematic photography style'\n"
                        "Example: 'Emphasize dramatic lighting'\n"
                        "Appended to the system prompt."
                    )
                }),
                "passthrough": ("BOOLEAN", {
                    "default": False,
                    "tooltip": (
                        "Skip VLM rewriting. Passes creative_direction\n"
                        "through as-is (for A/B testing)."
                    )
                }),
                "image_max_side": ("INT", {
                    "default": 1024,
                    "min": 256,
                    "max": 2048,
                    "step": 128,
                    "tooltip": (
                        "Max pixel size (longest side) for the image\n"
                        "sent to the VLM. Larger = more detail but\n"
                        "slower and more tokens."
                    )
                }),
            }
        }

    def rewrite(
        self,
        image: torch.Tensor,
        creative_direction: str,
        cn_type: str = "Canny",
        api_url: str = "http://localhost:11434/v1",
        model: str = "qwen3-vl",
        # Optional
        language: str = "English",
        temperature: float = 0.7,
        max_tokens: int = 2048,
        custom_instructions: str = "",
        passthrough: bool = False,
        image_max_side: int = 1024,
    ) -> Tuple[str, int]:
        # Always output the CN type index
        cn_index = CN_TYPE_INDEX.get(cn_type, 0)
        print(f"[CNPromptRewriter] CN type: {cn_type} (index={cn_index})")

        if passthrough:
            text = creative_direction or ""
            print("[CNPromptRewriter] Passthrough mode — text unchanged")
            return (text, cn_index)

        if not creative_direction or not creative_direction.strip():
            print("[CNPromptRewriter] WARNING: No creative direction provided.")
            return ("", cn_index)

        # ── Convert image ───────────────────────────────────────────────
        source_pil = _tensor_to_pil(image)
        print(f"[CNPromptRewriter] Reference image: "
              f"{source_pil.size[0]}×{source_pil.size[1]}")

        # ── Encode image for API ────────────────────────────────────────
        image_b64 = _pil_to_base64(source_pil, max_side=image_max_side)
        img_kb = len(image_b64) * 3 // 4 // 1024
        print(f"[CNPromptRewriter] Image encoded: ~{img_kb} KB")

        # ── Build system prompt with CN type info ───────────────────────
        cn_desc = CN_TYPE_DESCRIPTIONS.get(cn_type, "")
        if language == "English":
            sys_prompt = CONTROLNET_SYSTEM_PROMPT_EN.format(
                cn_type=cn_type,
                cn_description=cn_desc,
            )
        else:
            sys_prompt = CONTROLNET_SYSTEM_PROMPT_ZH.format(
                cn_type=cn_type,
                cn_description=cn_desc,
            )

        if custom_instructions and custom_instructions.strip():
            sys_prompt += f"\n\nAdditional instructions: {custom_instructions.strip()}"

        # ── Build user message ──────────────────────────────────────────
        user_text = (
            f"Creative direction: {creative_direction.strip()}\n\n"
            f"Look at this reference image. A {cn_type} ControlNet will "
            f"extract {self._cn_short_desc(cn_type)} from it to guide "
            f"the new image's structure.\n\n"
            f"Write a complete descriptive prompt for the new image that "
            f"applies the creative direction above. Describe everything "
            f"the model needs to render — the ControlNet only provides "
            f"structure, not appearance."
        )

        print(f"[CNPromptRewriter] User text: {user_text[:120]}"
              f"{'...' if len(user_text) > 120 else ''}")

        # ── Resolve API key ─────────────────────────────────────────────
        api_key = _resolve_api_key(api_url)
        if not api_key and any(
            svc in api_url.lower()
            for svc in ("deepseek", "openai", "anthropic")
        ):
            print("[CNPromptRewriter] WARNING: No API key found for remote service.")

        # ── Call the VLM ────────────────────────────────────────────────
        print(f"[CNPromptRewriter] Calling VLM: {api_url} (model={model})")

        try:
            result = _call_vision_api(
                api_url=api_url,
                model=model,
                system_prompt=sys_prompt,
                user_text=user_text,
                image_b64_url=image_b64,
                api_key=api_key,
                temperature=temperature,
                max_tokens=max_tokens,
            )

            # Clean up artifacts
            if result.startswith('"') and result.endswith('"'):
                result = result[1:-1]
            if result.startswith("'") and result.endswith("'"):
                result = result[1:-1]

            # Strip thinking-model tags
            result = re.sub(
                r'<think>.*?</think>', '', result, flags=re.DOTALL
            ).strip()

            # Collapse to single block
            result = result.replace("\n", " ").strip()
            result = re.sub(r' {2,}', ' ', result)

            word_count = len(result.split())
            print(f"[CNPromptRewriter] Result ({word_count} words): "
                  f"{result[:150]}{'...' if len(result) > 150 else ''}")

            if word_count < 30:
                print("[CNPromptRewriter] WARNING: Output is very short. "
                      "Try a larger/better vision model.")

            return (result, cn_index)

        except Exception as e:
            print(f"[CNPromptRewriter] ERROR ({type(e).__name__}): {e}")
            # Fallback: return the user's creative direction
            fallback = creative_direction or ""
            print(f"[CNPromptRewriter] Falling back to user description: "
                  f"{fallback[:100]}")
            return (fallback, cn_index)

    @staticmethod
    def _cn_short_desc(cn_type: str) -> str:
        """One-phrase description for the user message."""
        return {
            "Canny": "hard edges and contours",
            "SoftEdge": "soft edges and structural lines",
            "Depth": "a depth map (near/far spatial layout)",
            "Pose": "a pose skeleton (body joint positions)",
        }.get(cn_type, "structural guidance")
