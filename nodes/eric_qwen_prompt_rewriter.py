# Copyright (c) 2026 Eric Hiss. All rights reserved.
# Licensed under the terms in LICENSE.txt (CC BY-NC 4.0 / Commercial dual license).
# https://github.com/EricRollei/Eric_Qwen_Edit_Experiments
"""
Eric Qwen Prompt Rewriter Node
Enhances terse text prompts into rich, detailed image descriptions using
the Qwen-Image-2512 recommended prompt enhancement methodology.

Supports any OpenAI-compatible API endpoint:
  - Ollama  (http://localhost:11434/v1)
  - LM Studio (http://localhost:1234/v1)
  - vLLM  (http://localhost:8000/v1)
  - DeepSeek API (https://api.deepseek.com/v1)
  - OpenAI API (https://api.openai.com/v1)
  - Any other OpenAI-compatible service

API keys are loaded from (in priority order):
  1. Environment variable: DEEPSEEK_API_KEY, OPENAI_API_KEY, or ERIC_QWEN_API_KEY
  2. Config file: <node_root>/api_keys.ini
  3. Empty string (works for local servers like Ollama / LM Studio)

The system prompt is distilled from Qwen's official prompt_utils_2512.py
and produces detailed ~200-word descriptions tuned for image generation.

Author: Eric Hiss (GitHub: EricRollei)
"""

import configparser
import json
import logging
import os
import urllib.request
import urllib.error
from pathlib import Path
from typing import Tuple

logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════════════
#  API-key resolution — never stored in the workflow JSON
# ═══════════════════════════════════════════════════════════════════════

# Path to the node package root (one level up from nodes/)
_PACKAGE_ROOT = Path(__file__).resolve().parent.parent
_CONFIG_PATH = _PACKAGE_ROOT / "api_keys.ini"

# Map of api_url substrings → env-var names to try (in order)
_ENV_KEY_MAP = {
    "deepseek": ["DEEPSEEK_API_KEY"],
    "openai":   ["OPENAI_API_KEY"],
    "anthropic": ["ANTHROPIC_API_KEY"],
}


def _resolve_api_key(api_url: str) -> str:
    """Resolve API key from environment variables or config file.

    Priority:
      1. Service-specific env var (e.g. DEEPSEEK_API_KEY for deepseek URLs)
      2. Generic env var: ERIC_QWEN_API_KEY
      3. Config file:  <package_root>/api_keys.ini  [api_keys] section
      4. Empty string  (fine for local servers)
    """
    url_lower = api_url.lower()

    # 1. Service-specific env var
    for url_fragment, env_names in _ENV_KEY_MAP.items():
        if url_fragment in url_lower:
            for env_name in env_names:
                val = os.environ.get(env_name, "").strip()
                if val:
                    logger.info(f"API key loaded from env ${env_name}")
                    return val

    # 2. Generic env var
    generic = os.environ.get("ERIC_QWEN_API_KEY", "").strip()
    if generic:
        logger.info("API key loaded from env $ERIC_QWEN_API_KEY")
        return generic

    # 3. Config file
    if _CONFIG_PATH.exists():
        cfg = configparser.ConfigParser()
        cfg.read(str(_CONFIG_PATH), encoding="utf-8")
        if cfg.has_section("api_keys"):
            # Try service-specific key first
            for url_fragment in _ENV_KEY_MAP:
                if url_fragment in url_lower:
                    val = cfg.get("api_keys", url_fragment, fallback="").strip()
                    if val:
                        logger.info(f"API key loaded from api_keys.ini [{url_fragment}]")
                        return val
            # Try default key
            val = cfg.get("api_keys", "default", fallback="").strip()
            if val:
                logger.info("API key loaded from api_keys.ini [default]")
                return val

    # 4. No key found — fine for local servers
    return ""

# ═══════════════════════════════════════════════════════════════════════
#  System prompt — distilled from Qwen-Image-2512 official guidelines
# ═══════════════════════════════════════════════════════════════════════

SYSTEM_PROMPT_EN = """You are a world-class image prompt rewriting expert. Your task is to take the user's original image description and EXPAND it into a rich, precise, and aesthetically compelling English prompt optimized for AI image generation.

CRITICAL: Your output MUST be LONGER and MORE DETAILED than the input. If the user gives you 20 words, you must return at least 150 words. If the user gives you 100 words, you must return at least 200 words. The whole point is to ADD rich visual detail.

## Core Requirements
1. Use fluent, natural descriptive language in a single continuous paragraph. No markdown lists, numbered items, headings, or bullet points.
2. ALWAYS enrich and expand visual details:
   - Add specific lighting details: direction, color temperature, intensity, quality (soft/hard), shadow behavior
   - Add environment/atmosphere: time of day, weather, ambient particles, haze, depth of field
   - Add material/texture details: surface qualities, reflections, translucency, wear patterns
   - Add color palette specifics: dominant hues, accent colors, saturation levels, tonal range
   - Add composition details: perspective, focal length feel, framing, depth layers
   - All added content must align stylistically and logically with the existing description
3. Never modify proper nouns (names, brands, locations, titles, etc.).
4. If the image should contain visible text, enclose it in English double quotation marks and describe its position, font style, color, and size.
5. If the image contains no text, state: "The image contains no recognizable text."
6. Clearly specify the overall artistic style (realistic photography, anime illustration, concept art, watercolor, 3D rendering, etc.) with specific sub-style details.

## For Portraits (when a human subject is central)
- Specify ethnicity, gender, approximate age, face shape, eye details, nose, mouth, expression
- Describe skin tone, texture, any visible pores or freckles, makeup if present
- Detail clothing: type, fabric texture, color, fit, wrinkles, patterns
- Hairstyle: color, length, texture, style, how light catches it
- Describe pose, gaze direction, hand placement, body language
- Depict background setting with depth, lighting direction/intensity/color temperature, mood
- MINIMUM 200 words for portraits

## For General Images (landscapes, still lifes, abstract, architecture)
- Describe subject type, form, color, material, state (static/moving), scale
- Specify spatial layering: foreground elements, midground subjects, background atmosphere
- Detail lighting: source direction, quality, contrast ratio, dominant hues, shadow softness
- Describe surface textures: smooth, rough, metallic, organic, transparent, weathered
- Include scene type, time of day, weather conditions, season, emotional tone
- Add atmospheric perspective, color grading feel, overall mood
- MINIMUM 150 words for general scenes

## Output Rules
- Output ONLY the rewritten prompt text — nothing else
- Do NOT explain, confirm, ask questions, or add any commentary
- Do NOT include any thinking, reasoning, or meta-text
- Your output MUST be a single flowing description paragraph
- MINIMUM output: 150 words. Target: 200-300 words.
- If the input is very short (under 30 words), aim for 250+ words of rich detail"""


SYSTEM_PROMPT_ZH = """你是一位世界顶级的图像Prompt改写专家。你的任务是将用户的原始图像描述**大幅扩展**为丰富、精确、富有美感的中文Prompt，专为AI图像生成优化。

关键要求：你的输出必须比输入更长、更详细。如果用户给出20个字，你必须返回至少200个字。核心目的是添加丰富的视觉细节。

## 基础要求
1. 使用流畅自然的描述性语言，以连贯段落形式输出。禁止使用列表、编号、标题或任何结构化格式。
2. 始终丰富和扩展画面细节：
   - 补充具体的光影细节：方向、色温、强度、质感、阴影表现
   - 补充环境氛围：时间、天气、空气中的微粒、雾气、景深效果
   - 补充材质纹理：表面质感、反射、透明度、磨损痕迹
   - 补充色彩细节：主色调、点缀色、饱和度、色调范围
   - 所有补充内容必须与原描述风格和逻辑一致
3. 严禁修改任何专有名词。
4. 若图像含文字，用中文双引号标注并描述位置、字体、颜色。若无文字，明确说明"图像中未出现任何可识别文字"。
5. 明确指定艺术风格（写实摄影、动漫插画、概念图等）并细化子风格。

## 人像场景
- 详细描述人种、性别、年龄、面部特征、表情、皮肤质感
- 详细描述服装面料、发型、饰品
- 描述姿态、视线方向、肢体语言
- 描述背景场景、光影色调、情绪氛围
- 最少200字

## 通用场景
- 描述主体、空间层次、光影色彩、表面质感、场景氛围
- 包含时间、天气、情绪基调、色彩风格
- 最少150字

## 输出要求
- 仅输出改写后的Prompt文本，不要输出任何其他内容
- 不要解释、不要确认、不要提问、不要额外回复
- 不要输出思考过程或元文本
- 输出必须是单个连贯的描述段落
- 最少输出150字，目标200-300字"""


# ═══════════════════════════════════════════════════════════════════════
#  API helper
# ═══════════════════════════════════════════════════════════════════════

def _call_openai_compatible(
    api_url: str,
    model: str,
    system_prompt: str,
    user_prompt: str,
    api_key: str = "",
    temperature: float = 0.7,
    max_tokens: int = 1024,
    timeout: int = 120,
) -> str:
    """Call an OpenAI-compatible chat completions endpoint.

    Works with Ollama, LM Studio, vLLM, DeepSeek, OpenAI, etc.
    Uses only stdlib (urllib) to avoid extra dependencies.

    Handles DeepSeek thinking-model responses where reasoning_content
    consumes part of the token budget.
    """
    # Normalize URL — ensure it ends with /chat/completions
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
            {"role": "user", "content": user_prompt},
        ],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }

    headers = {
        "Content-Type": "application/json",
    }
    if api_key and api_key.strip():
        headers["Authorization"] = f"Bearer {api_key.strip()}"

    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers=headers, method="POST")

    # Custom redirect handler that preserves POST method on 307/308
    # (urllib's default converts POST to GET on redirect, breaking APIs
    # behind CDNs like CloudFront that return 307 Temporary Redirect)
    class _PostRedirectHandler(urllib.request.HTTPRedirectHandler):
        def redirect_request(self, req, fp, code, msg, headers, newurl):
            if code in (307, 308):
                new_req = urllib.request.Request(
                    newurl, data=req.data, headers=dict(req.headers),
                    method=req.get_method()
                )
                return new_req
            return super().redirect_request(req, fp, code, msg, headers, newurl)

    opener = urllib.request.build_opener(_PostRedirectHandler)

    try:
        with opener.open(req, timeout=timeout) as resp:
            body = json.loads(resp.read().decode("utf-8"))
            message = body["choices"][0]["message"]
            content = message.get("content", "").strip()

            # DeepSeek thinking models may return reasoning_content
            # separately — log it but use the main content field
            reasoning = message.get("reasoning_content", "")
            if reasoning:
                logger.info(f"DeepSeek reasoning tokens used: {len(reasoning)} chars")

            return content
    except urllib.error.HTTPError as e:
        error_body = e.read().decode("utf-8", errors="replace")
        raise RuntimeError(
            f"Prompt rewrite API error {e.code}: {error_body}"
        ) from e
    except urllib.error.URLError as e:
        raise RuntimeError(
            f"Cannot connect to prompt rewrite API at {url}: {e.reason}\n"
            f"Make sure your LLM server is running (Ollama, LM Studio, etc.)"
        ) from e


# ═══════════════════════════════════════════════════════════════════════
#  ComfyUI Node
# ═══════════════════════════════════════════════════════════════════════

class EricQwenPromptRewriter:
    """
    Enhance image prompts using a local or remote LLM.

    Rewrites terse prompts into rich ~200-word descriptions using the
    Qwen-Image-2512 recommended methodology.  Connect to any
    OpenAI-compatible API (Ollama, LM Studio, DeepSeek, etc.).

    API keys are loaded securely from environment variables or
    api_keys.ini — never stored in the workflow file.

    Output connects to the prompt input of any generation node.
    """

    CATEGORY = "Eric/QwenImage"
    FUNCTION = "rewrite"
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("enhanced_prompt",)

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {
                    "multiline": True,
                    "tooltip": "Original image description to enhance"
                }),
                "api_url": ("STRING", {
                    "default": "http://localhost:11434/v1",
                    "tooltip": (
                        "OpenAI-compatible API base URL. Examples:\n"
                        "  Ollama: http://localhost:11434/v1\n"
                        "  LM Studio: http://localhost:1234/v1\n"
                        "  DeepSeek: https://api.deepseek.com/v1\n"
                        "  OpenAI: https://api.openai.com/v1"
                    )
                }),
                "model": ("STRING", {
                    "default": "qwen3:8b",
                    "tooltip": (
                        "Model name on the API server. Examples:\n"
                        "  Ollama: qwen3:8b, llama3.1:8b\n"
                        "  DeepSeek: deepseek-chat\n"
                        "  OpenAI: gpt-4o-mini"
                    )
                }),
            },
            "optional": {
                "language": (["English", "Chinese"], {
                    "default": "English",
                    "tooltip": "Language for the rewritten prompt"
                }),
                "temperature": ("FLOAT", {
                    "default": 0.7,
                    "min": 0.0,
                    "max": 2.0,
                    "step": 0.1,
                    "tooltip": "LLM temperature — lower = more faithful, higher = more creative"
                }),
                "max_tokens": ("INT", {
                    "default": 2048,
                    "min": 256,
                    "max": 8192,
                    "step": 128,
                    "tooltip": (
                        "Maximum tokens for the LLM response. Set higher for thinking "
                        "models (DeepSeek-R1) where reasoning tokens consume part of "
                        "this budget. 2048 is a safe default."
                    )
                }),
                "custom_instructions": ("STRING", {
                    "multiline": True,
                    "default": "",
                    "tooltip": (
                        "Optional additional instructions appended to the system prompt.\n"
                        "Example: 'Focus on cinematic lighting and dramatic shadows.'\n"
                        "Leave empty to use the default Qwen-style rewriting."
                    )
                }),
                "passthrough": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "If True, skip rewriting and pass the prompt through unchanged (for A/B testing)"
                }),
            }
        }

    def rewrite(
        self,
        prompt: str,
        api_url: str = "http://localhost:11434/v1",
        model: str = "qwen3:8b",
        language: str = "English",
        temperature: float = 0.7,
        max_tokens: int = 2048,
        custom_instructions: str = "",
        passthrough: bool = False,
    ) -> Tuple[str]:
        if passthrough:
            print("[EricQwenPrompt] Passthrough mode — prompt unchanged")
            return (prompt,)

        # Resolve API key from env / config file (never from workflow JSON)
        api_key = _resolve_api_key(api_url)
        if not api_key and any(svc in api_url.lower() for svc in ("deepseek", "openai", "anthropic")):
            print(
                "[EricQwenPrompt] WARNING: No API key found for remote service.\n"
                "  Set an environment variable (e.g. DEEPSEEK_API_KEY) or create\n"
                f"  {_CONFIG_PATH}\n"
                "  See api_keys.ini.example for format."
            )

        # Select system prompt by language
        sys_prompt = SYSTEM_PROMPT_EN if language == "English" else SYSTEM_PROMPT_ZH

        # Append custom instructions if provided
        if custom_instructions and custom_instructions.strip():
            sys_prompt += f"\n\n## Additional Instructions\n{custom_instructions.strip()}"

        print(f"[EricQwenPrompt] Rewriting prompt via {api_url} (model={model})")
        print(f"[EricQwenPrompt] Original ({len(prompt.split())} words): "
              f"{prompt[:120]}{'...' if len(prompt) > 120 else ''}")

        try:
            enhanced = _call_openai_compatible(
                api_url=api_url,
                model=model,
                system_prompt=sys_prompt,
                user_prompt=prompt,
                api_key=api_key,
                temperature=temperature,
                max_tokens=max_tokens,
            )

            # Clean up — remove any leading/trailing quotes the LLM might add
            if enhanced.startswith('"') and enhanced.endswith('"'):
                enhanced = enhanced[1:-1]
            if enhanced.startswith("'") and enhanced.endswith("'"):
                enhanced = enhanced[1:-1]

            # Strip thinking-model artifacts (e.g. <think>...</think> tags
            # that some models like Qwen3 or DeepSeek-R1 may include)
            import re
            enhanced = re.sub(r'<think>.*?</think>', '', enhanced, flags=re.DOTALL).strip()

            # Remove newlines for cleaner prompt (single block)
            enhanced = enhanced.replace("\n", " ").strip()
            # Collapse multiple spaces
            enhanced = re.sub(r' {2,}', ' ', enhanced)

            orig_words = len(prompt.split())
            new_words = len(enhanced.split())
            print(f"[EricQwenPrompt] Enhanced ({new_words} words): "
                  f"{enhanced[:120]}{'...' if len(enhanced) > 120 else ''}")
            print(f"[EricQwenPrompt] Word count: {orig_words} -> {new_words} "
                  f"({'↑' if new_words > orig_words else '↓'}{abs(new_words - orig_words)})")

            if new_words < orig_words:
                print("[EricQwenPrompt] WARNING: Output is shorter than input. "
                      "Try a different model or increase max_tokens.")

            return (enhanced,)

        except RuntimeError as e:
            print(f"[EricQwenPrompt] ERROR: {e}")
            print("[EricQwenPrompt] Falling back to original prompt")
            return (prompt,)
