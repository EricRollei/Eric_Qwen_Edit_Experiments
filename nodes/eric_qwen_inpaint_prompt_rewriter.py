# Copyright (c) 2026 Eric Hiss. All rights reserved.
# Licensed under the terms in LICENSE.txt (CC BY-NC 4.0 / Commercial dual license).
# https://github.com/EricRollei/Eric_Qwen_Edit_Experiments
"""
Eric Qwen Inpaint Prompt Rewriter

Vision-LLM-powered prompt rewriter for inpainting and outpainting workflows.
Takes the source image + mask + user intent and produces a full descriptive
prompt suitable for QwenImageControlNetInpaintPipeline.

The mask is visualised as a bright outline on the source image so the VLM
can see both the masked region content (for spatial context) and clearly
identify which area will be changed.

Works with any OpenAI-compatible vision API:
  - Ollama  (qwen3-vl, llava, gemma3, etc.)
  - LM Studio
  - OpenAI  (gpt-4o-mini, gpt-4o)
  - DeepSeek VL
  - Any other multimodal chat completions endpoint

Author: Eric Hiss (GitHub: EricRollei)
"""

import base64
import io
import json
import logging
import re
import urllib.request
import urllib.error
import numpy as np
from PIL import Image, ImageDraw, ImageFilter
from typing import Tuple

import torch

from .eric_qwen_prompt_rewriter import (
    _resolve_api_key,
)

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════
#  Mask visualisation — outline only
# ═══════════════════════════════════════════════════════════════════════

def _draw_mask_outline(
    image: Image.Image,
    mask: Image.Image,
    outline_color: tuple = (255, 50, 50),
    outline_width: int = 3,
) -> Image.Image:
    """Draw a bright outline around the mask boundary on the image.

    The interior of the mask is left visible so the VLM can see the
    existing content and understand spatial context.

    Args:
        image:  RGB source image.
        mask:   L-mode mask (white = inpaint region).
        outline_color: RGB color for the outline (default: bright red).
        outline_width: Pixel width of the outline stroke.

    Returns:
        RGB image with outline drawn on top.
    """
    if mask.size != image.size:
        mask = mask.resize(image.size, Image.NEAREST)

    # Binarise the mask
    mask_np = np.array(mask)
    binary = (mask_np > 127).astype(np.uint8)

    # Create outline by dilating - eroding (morphological gradient)
    from PIL import ImageFilter as _IF

    mask_bin = Image.fromarray(binary * 255, mode="L")

    # Dilate: max-filter expands white regions
    dilated = mask_bin.filter(_IF.MaxFilter(size=outline_width * 2 + 1))
    # Erode: min-filter shrinks white regions
    eroded = mask_bin.filter(_IF.MinFilter(size=outline_width * 2 + 1))

    # Outline = dilated - eroded
    dilated_np = np.array(dilated)
    eroded_np = np.array(eroded)
    outline_np = ((dilated_np > 127) & ~(eroded_np > 127)).astype(np.uint8) * 255
    outline_mask = Image.fromarray(outline_np, mode="L")

    # Create colored overlay for the outline
    overlay = Image.new("RGB", image.size, outline_color)
    result = image.copy().convert("RGB")
    result.paste(overlay, mask=outline_mask)

    return result


# ═══════════════════════════════════════════════════════════════════════
#  PIL ↔ base64
# ═══════════════════════════════════════════════════════════════════════

def _pil_to_base64(image: Image.Image, max_side: int = 1024) -> str:
    """Convert PIL image to base64 data URL, optionally downsizing.

    Most VLMs work best with images ≤ 1024px on the longest side.
    """
    w, h = image.size
    if max(w, h) > max_side:
        scale = max_side / max(w, h)
        image = image.resize(
            (int(w * scale), int(h * scale)), Image.LANCZOS
        )

    buf = io.BytesIO()
    image.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/png;base64,{b64}"


def _tensor_to_pil(tensor: torch.Tensor) -> Image.Image:
    """Convert ComfyUI IMAGE tensor (B, H, W, C) to PIL."""
    if tensor.dim() == 4:
        tensor = tensor[0]
    arr = (tensor.cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
    return Image.fromarray(arr)


def _mask_to_pil(mask: torch.Tensor) -> Image.Image:
    """Convert ComfyUI MASK tensor (B, H, W) to PIL L-mode."""
    if mask.dim() == 3:
        mask = mask[0]
    arr = (mask.cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
    return Image.fromarray(arr, mode="L")


# ═══════════════════════════════════════════════════════════════════════
#  Vision API call
# ═══════════════════════════════════════════════════════════════════════

def _resolve_model_name(api_url: str, requested_model: str) -> str:
    """Query the API server for loaded models and resolve the actual name.

    If the requested model is found, use it.  Otherwise pick the first
    loaded model and warn.  Returns the original name on any error.
    """
    url = api_url.rstrip("/")
    if url.endswith("/chat/completions"):
        url = url.rsplit("/chat/completions", 1)[0]
    if not url.endswith("/v1"):
        url += "/v1"
    models_url = url + "/models"

    try:
        req = urllib.request.Request(models_url, method="GET")
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            model_ids = [m["id"] for m in data.get("data", [])]

            if not model_ids:
                print(f"[InpaintRewriter] No models reported by {models_url}")
                return requested_model

            print(f"[InpaintRewriter] Server has {len(model_ids)} model(s) "
                  f"available")

            if requested_model in model_ids:
                return requested_model

            # Try partial match (user may have typed a substring)
            for mid in model_ids:
                if requested_model.lower() in mid.lower():
                    print(f"[InpaintRewriter] Partial match: "
                          f"'{requested_model}' -> '{mid}'")
                    return mid

            # Not found at all — use first available model
            fallback = model_ids[0]
            print(f"[InpaintRewriter] WARNING: Model '{requested_model}' "
                  f"not found on server.")
            print(f"[InpaintRewriter] Available: "
                  f"{', '.join(model_ids[:5])}"
                  f"{'...' if len(model_ids) > 5 else ''}")
            print(f"[InpaintRewriter] Using: '{fallback}'")
            return fallback

    except Exception as e:
        print(f"[InpaintRewriter] Could not query models endpoint: {e}")
        return requested_model


def _parse_sse_content(raw: str) -> str:
    """Extract content from SSE (Server-Sent Events) streaming response.

    Handles the case where the server ignores stream:false and sends
    chunked SSE data instead of a single JSON response.
    """
    content_parts = []
    for line in raw.split("\n"):
        line = line.strip()
        if not line.startswith("data: "):
            continue
        data_str = line[6:]  # Strip 'data: ' prefix
        if data_str == "[DONE]":
            break
        try:
            chunk = json.loads(data_str)
            delta = chunk.get("choices", [{}])[0].get("delta", {})
            piece = delta.get("content", "")
            if piece:
                content_parts.append(piece)
        except (json.JSONDecodeError, IndexError, KeyError):
            continue
    return "".join(content_parts)


def _call_vision_api(
    api_url: str,
    model: str,
    system_prompt: str,
    user_text: str,
    image_b64_url: str,
    api_key: str = "",
    temperature: float = 0.7,
    max_tokens: int = 2048,
    timeout: int = 180,
) -> str:
    """Call an OpenAI-compatible vision chat completions endpoint.

    Sends a multimodal message with both image and text.
    Falls back to SSE streaming parse if server ignores stream:false.
    """
    # Resolve actual model name (handles 404 from unloaded models)
    model = _resolve_model_name(api_url, model)

    url = api_url.rstrip("/")
    if not url.endswith("/chat/completions"):
        if url.endswith("/v1"):
            url += "/chat/completions"
        else:
            url += "/v1/chat/completions"

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": image_b64_url},
                    },
                    {
                        "type": "text",
                        "text": user_text,
                    },
                ],
            },
        ],
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stream": False,
    }

    headers = {"Content-Type": "application/json"}
    if api_key and api_key.strip():
        headers["Authorization"] = f"Bearer {api_key.strip()}"

    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers=headers, method="POST")

    print(f"[InpaintRewriter] POST {url}")
    print(f"[InpaintRewriter] Payload size: {len(data)} bytes ")

    # Handle 307/308 redirects preserving POST
    class _PostRedirectHandler(urllib.request.HTTPRedirectHandler):
        def redirect_request(self, req, fp, code, msg, headers, newurl):
            if code in (307, 308):
                new_req = urllib.request.Request(
                    newurl, data=req.data, headers=dict(req.headers),
                    method=req.get_method(),
                )
                return new_req
            return super().redirect_request(req, fp, code, msg, headers, newurl)

    opener = urllib.request.build_opener(_PostRedirectHandler)

    try:
        with opener.open(req, timeout=timeout) as resp:
            raw = resp.read().decode("utf-8")
            content_type = resp.headers.get("Content-Type", "")
            print(f"[InpaintRewriter] Response status: {resp.status}, "
                  f"Content-Type: {content_type}, "
                  f"body size: {len(raw)} bytes")

            # Try standard JSON parse first
            try:
                body = json.loads(raw)
                message = body["choices"][0]["message"]
                content = message.get("content", "").strip()

                reasoning = message.get("reasoning_content", "")
                if reasoning:
                    print(f"[InpaintRewriter] Model reasoning: "
                          f"{len(reasoning)} chars (discarded)")

                if content:
                    print(f"[InpaintRewriter] Got JSON response: "
                          f"{len(content)} chars")
                    return content
                else:
                    print(f"[InpaintRewriter] WARNING: JSON response has "
                          f"empty content field")
                    print(f"[InpaintRewriter] Raw keys: {list(message.keys())}")
                    # Some models put content in reasoning_content only
                    if reasoning:
                        return reasoning.strip()
            except (json.JSONDecodeError, KeyError, IndexError) as parse_err:
                print(f"[InpaintRewriter] JSON parse failed: {parse_err}")
                print(f"[InpaintRewriter] Raw response (first 500): "
                      f"{raw[:500]}")

            # Fallback: try parsing as SSE streaming data
            if "data: " in raw:
                print("[InpaintRewriter] Attempting SSE stream parse...")
                content = _parse_sse_content(raw)
                if content:
                    print(f"[InpaintRewriter] SSE parse recovered "
                          f"{len(content)} chars")
                    return content.strip()
                else:
                    print("[InpaintRewriter] SSE parse yielded no content")

            raise RuntimeError(
                f"Could not extract content from API response. "
                f"Raw ({len(raw)} bytes): {raw[:500]}"
            )

    except urllib.error.HTTPError as e:
        error_body = e.read().decode("utf-8", errors="replace")
        raise RuntimeError(
            f"Vision API error {e.code}: {error_body[:500]}"
        ) from e
    except urllib.error.URLError as e:
        raise RuntimeError(
            f"Cannot connect to vision API at {url}: {e.reason}\n"
            f"Make sure your VLM server is running and the model supports vision."
        ) from e


# ═══════════════════════════════════════════════════════════════════════
#  System prompts
# ═══════════════════════════════════════════════════════════════════════

INPAINT_SYSTEM_PROMPT_EN = """You write short, focused prompts for AI inpainting.

The user will show you an image with a marked area and describe what to replace it with. Write a prompt that emphasizes the NEW content that should appear in the marked area.

CRITICAL: The AI inpainting model already has the original image as conditioning. It does NOT need a description of unchanged areas — it already sees them. Your prompt must tell it WHAT TO GENERATE in the masked region.

Prompt structure (follow this order):
1. Detailed description of the new/changed content — materials, colors, textures, shape, form. This MUST be at least 70% of the prompt.
2. Brief subject/context phrase (e.g. "worn by a young woman", "on a wooden table").
3. Brief style/lighting phrase (e.g. "soft natural lighting, photorealistic").

Rules:
- One flowing paragraph, 40-80 words total
- Start IMMEDIATELY with the new content — your very first words must describe the replacement object/material
- Descriptive language only (describe what IS, not what to do or change)
- NEVER use narrative or storytelling language ("beyond the frame", "dissolves into", "transforms", "reveals")
- NEVER describe what was there before or how the change happens — only describe the final result
- Do not describe unchanged areas (background, hair, accessories, etc.) — the model already sees them
- Output ONLY the prompt text — no commentary, no quotes, no markdown"""


INPAINT_SYSTEM_PROMPT_ZH = """你为AI修图（inpainting）撰写简短、聚焦的提示词。

用户会展示一张标记了修改区域的图像，并描述要替换为什么内容。你的提示词必须以新内容为重心。

关键：AI修图模型已经拥有原始图像作为参考，不需要描述未改变的部分——它已经看到了。你的提示词必须告诉它在遮罩区域生成什么。

提示词结构（按此顺序）：
1. 详细描述新内容——材质、颜色、纹理、形状。至少占提示词的70%。
2. 简短的主体/上下文短语（如"一位年轻女性穿着"、"放在木桌上"）。
3. 简短的风格/光照短语（如"柔和自然光，写实风格"）。

规则：
- 一段连贯文字，40-80字
- 直接从新内容开始——第一个词必须描述替换物品/材质
- 使用描述性语言（描述"是什么"），不用指令性语言
- 绝不使用叙事性语言（"画面之外"、"融化成"、"变成"、"展现"）
- 绝不描述之前的内容或变化过程——只描述最终结果
- 不描述未改变区域（背景、头发、配饰等）
- 仅输出提示词，不要评论或解释"""


OUTPAINT_SYSTEM_PROMPT_EN = """You write prompts for AI outpainting (extending an image beyond its edges).

The user will show you an image and describe what lies beyond the current edges. Write a prompt that emphasizes the NEW extended content while briefly anchoring it to the existing scene.

CRITICAL: The model already has the original image. Focus your prompt on what to GENERATE in the new areas.

Prompt structure:
- FIRST: Detailed description of the new extended content — environment, objects, surfaces, depth, how it continues from the visible edges.
- THEN: Brief context anchoring it to the existing scene (subject, setting, lighting continuity).

Rules:
- One flowing paragraph, 80-150 words total
- Start with the extension content, not the existing image
- Maintain visual continuity (lighting, perspective, style)
- Descriptive language only
- Output ONLY the prompt — no commentary, no markdown"""


OUTPAINT_SYSTEM_PROMPT_ZH = """你为AI扩图（outpainting）撰写提示词。

用户会展示一张图像并描述边缘之外的内容。你的提示词应以新扩展内容为重心，同时简要关联现有画面。

关键：模型已拥有原始图像。请聚焦于在新区域生成什么。

提示词结构：
- 首先：详细描述扩展区域的新内容——环境、物体、表面、纵深、如何从可见边缘延续。
- 然后：简要关联现有场景（主体、场景、光照连续性）。

规则：
- 一段连贯文字，80-150字
- 从扩展内容开始，不要从现有图像开始
- 保持视觉连续性（光照、透视、风格）
- 使用描述性语言
- 仅输出提示词"""


# ═══════════════════════════════════════════════════════════════════════
#  ComfyUI Node
# ═══════════════════════════════════════════════════════════════════════

class EricQwenInpaintPromptRewriter:
    """
    Vision-LLM-powered prompt rewriter for inpainting and outpainting.

    Sends the source image (with mask outline) to a vision model along
    with the user's change description.  The VLM writes a SHORT,
    CHANGE-FOCUSED prompt that front-loads the new content — exactly
    what inpainting ControlNet needs.  The pipeline already has the
    source image as conditioning, so the prompt should emphasize WHAT
    TO GENERATE rather than describing unchanged areas.

    For outpainting, focuses the prompt on the extended content while
    briefly anchoring to the existing scene.

    Works with any OpenAI-compatible vision API (Ollama, LM Studio,
    OpenAI, etc.).  API keys loaded from env vars or api_keys.ini.
    """

    CATEGORY = "Eric Qwen-Image"
    FUNCTION = "rewrite"
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("prompt",)

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE", {
                    "tooltip": "Source image for the VLM to analyse."
                }),
                "change_description": ("STRING", {
                    "multiline": True,
                    "default": "",
                    "tooltip": (
                        "Describe what you want to change in the masked area.\n"
                        "Example: 'Replace the car with a green taxi'\n"
                        "Example: 'Change the sky to a dramatic sunset'"
                    )
                }),
                "api_url": ("STRING", {
                    "default": "http://localhost:1234/v1",
                    "tooltip": (
                        "OpenAI-compatible vision API URL.\n"
                        "  LM Studio: http://localhost:1234/v1\n"
                        "  Ollama: http://localhost:11434/v1\n"
                        "  OpenAI: https://api.openai.com/v1"
                    )
                }),
            },
            "optional": {
                "model": ("STRING", {
                    "default": "",
                    "tooltip": (
                        "Model name override. Leave empty to auto-detect\n"
                        "the loaded model from the API server.\n"
                        "  Ollama: qwen3-vl, qwen2.5-vl:7b\n"
                        "  OpenAI: gpt-4o-mini, gpt-4o"
                    )
                }),
                "mask": ("MASK", {
                    "tooltip": (
                        "Inpaint mask (white = area to change).\n"
                        "A red outline will be drawn on the image\n"
                        "for the VLM. Not needed for outpainting."
                    )
                }),
                "outpaint_description": ("STRING", {
                    "multiline": True,
                    "default": "",
                    "tooltip": (
                        "What to add in the outpaint expansion area.\n"
                        "Example: 'Mountain landscape extending into the distance'\n"
                        "Only used when no mask is provided or when\n"
                        "you want outpainting context in the prompt."
                    )
                }),
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
                "outline_color_r": ("INT", {
                    "default": 255, "min": 0, "max": 255,
                    "tooltip": "Mask outline red channel."
                }),
                "outline_color_g": ("INT", {
                    "default": 50, "min": 0, "max": 255,
                    "tooltip": "Mask outline green channel."
                }),
                "outline_color_b": ("INT", {
                    "default": 50, "min": 0, "max": 255,
                    "tooltip": "Mask outline blue channel."
                }),
                "outline_width": ("INT", {
                    "default": 3,
                    "min": 1,
                    "max": 10,
                    "step": 1,
                    "tooltip": "Pixel width of the mask outline."
                }),
                "custom_instructions": ("STRING", {
                    "multiline": True,
                    "default": "",
                    "tooltip": (
                        "Additional instructions for the VLM.\n"
                        "Example: 'Use a cinematic photography style'\n"
                        "Appended to the system prompt."
                    )
                }),
                "passthrough": ("BOOLEAN", {
                    "default": False,
                    "tooltip": (
                        "Skip VLM rewriting. Passes change_description\n"
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
        change_description: str,
        api_url: str = "http://localhost:1234/v1",
        model: str = "",
        # Optional
        mask: torch.Tensor = None,
        outpaint_description: str = "",
        language: str = "English",
        temperature: float = 0.7,
        max_tokens: int = 2048,
        outline_color_r: int = 255,
        outline_color_g: int = 50,
        outline_color_b: int = 50,
        outline_width: int = 3,
        custom_instructions: str = "",
        passthrough: bool = False,
        image_max_side: int = 1024,
    ) -> Tuple[str]:
        if passthrough:
            text = change_description or outpaint_description or ""
            print("[InpaintRewriter] Passthrough mode — text unchanged")
            return (text,)

        # ── Convert inputs ──────────────────────────────────────────────
        source_pil = _tensor_to_pil(image)
        print(f"[InpaintRewriter] Source image: "
              f"{source_pil.size[0]}×{source_pil.size[1]}")

        # ── Determine mode: inpaint vs outpaint ────────────────────────
        has_mask = mask is not None
        has_outpaint = bool(outpaint_description and outpaint_description.strip())
        has_change = bool(change_description and change_description.strip())

        if has_mask and has_change:
            mode = "inpaint"
        elif has_outpaint:
            mode = "outpaint"
        elif has_change:
            # Change description without mask — treat as general inpaint
            mode = "inpaint_no_mask"
        else:
            print("[InpaintRewriter] WARNING: No change or outpaint description. "
                  "Returning empty prompt.")
            return ("",)

        print(f"[InpaintRewriter] Mode: {mode}")

        # ── Build the VLM image ─────────────────────────────────────────
        if mode == "inpaint" and has_mask:
            mask_pil = _mask_to_pil(mask)
            outline_color = (outline_color_r, outline_color_g, outline_color_b)
            vlm_image = _draw_mask_outline(
                source_pil, mask_pil,
                outline_color=outline_color,
                outline_width=outline_width,
            )
            print(f"[InpaintRewriter] Drew mask outline "
                  f"(color={outline_color}, width={outline_width}px)")
        else:
            vlm_image = source_pil

        # ── Encode image for API ────────────────────────────────────────
        image_b64 = _pil_to_base64(vlm_image, max_side=image_max_side)
        img_kb = len(image_b64) * 3 // 4 // 1024
        print(f"[InpaintRewriter] Image encoded: ~{img_kb} KB")

        # ── Select system prompt ────────────────────────────────────────
        if mode in ("inpaint", "inpaint_no_mask"):
            sys_prompt = (INPAINT_SYSTEM_PROMPT_EN if language == "English"
                          else INPAINT_SYSTEM_PROMPT_ZH)
        else:
            sys_prompt = (OUTPAINT_SYSTEM_PROMPT_EN if language == "English"
                          else OUTPAINT_SYSTEM_PROMPT_ZH)

        if custom_instructions and custom_instructions.strip():
            sys_prompt += (f"\n\n## Additional Instructions\n"
                           f"{custom_instructions.strip()}")

        # ── Build user message text ─────────────────────────────────────
        if mode == "inpaint":
            user_text = (
                f"Replace the marked area with: {change_description.strip()}\n\n"
                f"The red outline shows the area to change. "
                f"Write a short, focused prompt describing the new content "
                f"in rich detail (materials, textures, colors, form). "
                f"Keep description of unchanged areas to a minimum."
            )
        elif mode == "inpaint_no_mask":
            user_text = (
                f"Apply this change: {change_description.strip()}\n\n"
                f"Write a short, focused prompt describing the changed content "
                f"in rich detail. Keep description of unchanged areas minimal."
            )
        else:  # outpaint
            parts = []
            if has_change:
                parts.append(f"Changes to existing content: "
                             f"{change_description.strip()}.")
            parts.append(
                f"Extend the scene with: {outpaint_description.strip()}\n\n"
                f"Write a prompt focused on the new extended areas. "
                f"Describe the expansion content in detail. "
                f"Briefly anchor to the existing scene for continuity."
            )
            user_text = " ".join(parts)

        print(f"[InpaintRewriter] User text: {user_text[:120]}"
              f"{'...' if len(user_text) > 120 else ''}")

        # ── Resolve API key ─────────────────────────────────────────────
        api_key = _resolve_api_key(api_url)
        if not api_key and any(
            svc in api_url.lower()
            for svc in ("deepseek", "openai", "anthropic")
        ):
            print("[InpaintRewriter] WARNING: No API key found for remote service.")

        # ── Call the VLM ────────────────────────────────────────────────
        print(f"[InpaintRewriter] Calling VLM: {api_url} (model={model})")

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
            print(f"[InpaintRewriter] Result ({word_count} words): "
                  f"{result[:150]}{'...' if len(result) > 150 else ''}")

            if word_count < 20:
                print("[InpaintRewriter] WARNING: Output is very short. "
                      "Try a larger/better vision model.")

            return (result,)

        except Exception as e:
            print(f"[InpaintRewriter] ERROR ({type(e).__name__}): {e}")
            # Fallback: return the user's description
            fallback = change_description or outpaint_description or ""
            print(f"[InpaintRewriter] Falling back to user description: "
                  f"{fallback[:100]}")
            return (fallback,)
