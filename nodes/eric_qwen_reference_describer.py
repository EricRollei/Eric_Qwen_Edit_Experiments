# Copyright (c) 2026 Eric Hiss. All rights reserved.
# Licensed under the terms in LICENSE.txt (CC BY-NC 4.0 / Commercial dual license).
# https://github.com/EricRollei/Eric_Qwen_Edit_Experiments
"""
Eric Qwen Reference Describer Node

Bridges the Gen-Searcher pipeline to the pure Qwen-Image generation pipeline.

Problem:
  Qwen-Image-Edit is an edit model - when given reference images it tries to
  EDIT them rather than generate something new. Web thumbnail references
  contaminate the output with their pixel noise.

Solution:
  Use the VL model (Qwen3-VL-8B, already running in SGLang) to READ the
  reference images and extract their visual characteristics as text.
  Merge those text descriptions with the grounded prompt from Gen-Searcher.
  Feed the combined text to EricQwenImageGenerate (pure t2i - no image input,
  no pixel contamination, full generation quality).

Workflow:
  EricGenSearcherNode
    → grounded_prompt ─────────────────────────────────────────────┐
    → ref_image_1..4  → EricQwenReferenceDescriber → combined_prompt → EricQwenImageGenerate

The VL model is queried with each reference image and asked to describe
the visual characteristics relevant to image generation:
  - Subject appearance (face, hair, build, expression)
  - Clothing and accessories (colors, textures, style, fit)
  - Setting and environment (architecture, lighting, atmosphere)
  - Pose, composition, camera angle

These descriptions are synthesized into a final combined prompt that
gives the generation model all the visual knowledge from web research
without any pixel-level contamination from thumbnails.

Author: Eric Hiss (GitHub: EricRollei)
"""

import base64
import io
import json
import urllib.error
import urllib.request
from typing import List, Optional, Tuple

import torch
from PIL import Image

from .eric_qwen_edit_utils import tensor_to_pil


# ---------------------------------------------------------------------------
#  Per-image VL query prompts
#  Different focus modes ask the VL model different questions about each image
# ---------------------------------------------------------------------------

_FOCUS_PROMPTS = {
    "subject_appearance": (
        "Examine this reference image carefully. Describe ONLY the visual characteristics "
        "of the main subject that would be needed to depict them accurately in a new generated image. "
        "Cover: face shape, skin tone, hair color/texture/style, eye color, build/physique, "
        "distinctive facial features. Be precise and concise - no more than 3 sentences. "
        "Do not describe the background, setting, or clothing."
    ),
    "clothing_and_style": (
        "Examine this reference image carefully. Describe ONLY the clothing, accessories, "
        "and styling worn by the main subject. Cover: garment types, colors, textures, "
        "fit/silhouette, visible logos or details, footwear, jewelry. Be precise and concise "
        "- no more than 3 sentences. Do not describe the person's face or the background."
    ),
    "setting_and_environment": (
        "Examine this reference image carefully. Describe ONLY the setting, environment, "
        "and lighting. Cover: location type, architectural elements, time of day, light "
        "direction and quality, color palette, atmosphere/mood. Be precise and concise "
        "- no more than 3 sentences. Do not describe any people in the image."
    ),
    "full_scene": (
        "Examine this reference image carefully. Provide a concise description of the "
        "complete visual scene covering: (1) subject appearance and clothing, "
        "(2) setting and environment, (3) lighting and color palette, "
        "(4) composition and camera perspective. "
        "Be precise and factual - no more than 4 sentences total. "
        "This description will be used to guide image generation."
    ),
    "auto": (
        "Examine this reference image carefully. Extract the key visual characteristics "
        "that would be most useful for recreating or depicting this scene/subject in a new "
        "AI-generated image. Focus on the most visually distinctive and important details: "
        "appearance, clothing, setting, lighting. Be precise and concise - no more than "
        "3-4 sentences. Prioritize specifics over generalities."
    ),
}

# System prompt for the VL description calls
_VL_SYSTEM = (
    "You are a precise visual analyst assisting with AI image generation. "
    "When asked to describe images, be factual, specific, and concise. "
    "Use clear visual language. Never use vague terms like 'beautiful' or 'stunning'. "
    "Describe exactly what you see in concrete visual terms."
)

# Template for combining descriptions into the final prompt
_COMBINE_TEMPLATE = """{grounded_prompt}

Visual reference details:
{descriptions}"""

# Synthesis prompt - asks the VL model to weave everything together
_SYNTHESIS_PROMPT = (
    "You are writing the final image generation prompt for a high-quality AI image generator.\n\n"
    "Here is the base prompt:\n{grounded_prompt}\n\n"
    "Here are visual details extracted from reference images:\n{descriptions}\n\n"
    "Write a single flowing paragraph that combines the base prompt with the most "
    "important and relevant visual details from the references. "
    "The paragraph should read as a natural, detailed image generation prompt. "
    "Prioritize details that refine or add specificity to the base prompt. "
    "Do NOT include contradictory information. Do NOT use bullet points or headers. "
    "Output ONLY the final combined prompt paragraph, nothing else."
)


# ---------------------------------------------------------------------------
#  VL API helpers
# ---------------------------------------------------------------------------

def _pil_to_base64(img: Image.Image, max_dim: int = 512) -> str:
    """Convert PIL image to base64 JPEG for API transmission.

    Resizes to max_dim on the longest side to keep the API call fast.
    The VL model needs to understand the content, not render it at full res.
    """
    w, h = img.size
    if max(w, h) > max_dim:
        if w >= h:
            new_w, new_h = max_dim, max(1, int(h * max_dim / w))
        else:
            new_h, new_w = max_dim, max(1, int(w * max_dim / h))
        img = img.resize((new_w, new_h), Image.LANCZOS)
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format="JPEG", quality=85)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def _vl_describe_image(
    img_b64:  str,
    focus:    str,
    api_url:  str,
    model:    str,
    timeout:  int = 60,
) -> str:
    """Call the VL model to describe a single reference image.

    Uses the OpenAI-compatible vision format supported by SGLang.
    """
    user_content = [
        {
            "type":      "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"},
        },
        {
            "type": "text",
            "text": _FOCUS_PROMPTS.get(focus, _FOCUS_PROMPTS["auto"]),
        },
    ]

    url = api_url.rstrip("/")
    if not url.endswith("/chat/completions"):
        url = url.rstrip("/v1").rstrip("/") + "/v1/chat/completions"

    payload = {
        "model":      model,
        "max_tokens": 256,
        "stream":     False,
        "messages": [
            {"role": "system", "content": _VL_SYSTEM},
            {"role": "user",   "content": user_content},
        ],
    }

    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            data = json.loads(r.read())
        return data["choices"][0]["message"].get("content", "").strip()
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", errors="replace")[:200]
        print(f"[EricRefDescriber] VL API error {e.code}: {body}")
        return ""
    except Exception as e:
        print(f"[EricRefDescriber] VL API failed: {e}")
        return ""


def _vl_synthesize(
    grounded_prompt: str,
    descriptions:    str,
    api_url:         str,
    model:           str,
    timeout:         int = 90,
) -> str:
    """Ask the VL model to synthesize a final combined prompt."""
    text = _SYNTHESIS_PROMPT.format(
        grounded_prompt=grounded_prompt,
        descriptions=descriptions,
    )
    url = api_url.rstrip("/")
    if not url.endswith("/chat/completions"):
        url = url.rstrip("/v1").rstrip("/") + "/v1/chat/completions"

    payload = {
        "model":      model,
        "max_tokens": 512,
        "stream":     False,
        "messages": [
            {"role": "user", "content": text},
        ],
    }
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            data = json.loads(r.read())
        return data["choices"][0]["message"].get("content", "").strip()
    except Exception as e:
        print(f"[EricRefDescriber] Synthesis call failed: {e}")
        return ""


def _detect_model(api_url: str) -> str:
    """Query /v1/models and return the first model id."""
    url = api_url.rstrip("/")
    if url.endswith("/chat/completions"):
        url = url.rsplit("/chat/completions", 1)[0]
    if not url.endswith("/v1"):
        url = url.rstrip("/") + "/v1"
    try:
        req = urllib.request.Request(url + "/models", method="GET")
        with urllib.request.urlopen(req, timeout=8) as r:
            models = json.loads(r.read()).get("data", [])
            if models:
                return models[0]["id"]
    except Exception:
        pass
    return "Gen-Searcher-8B"


# ---------------------------------------------------------------------------
#  ComfyUI Node
# ---------------------------------------------------------------------------

class EricQwenReferenceDescriber:
    """
    Convert reference images + grounded prompt into a pure text prompt
    for EricQwenImageGenerate (the high-quality t2i pipeline).

    Uses the running SGLang VL server (Gen-Searcher-8B / Qwen3-VL-8B)
    to describe each reference image's visual characteristics, then
    synthesizes everything into a single detailed generation prompt.

    This avoids the pixel-contamination problem of feeding web thumbnails
    into the edit pipeline - the visual knowledge from references is
    extracted as clean text and drives the pure generation model instead.

    Typical workflow:
      EricGenSearcherNode
        → grounded_prompt ──────────────────────────────────────────────┐
        → ref_image_1..4  → EricQwenReferenceDescriber → combined_prompt → EricQwenImageGenerate
    """

    CATEGORY     = "Eric/QwenImage"
    FUNCTION     = "describe_and_combine"
    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("combined_prompt", "descriptions_log")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "grounded_prompt": ("STRING", {
                    "multiline": True,
                    "default":   "",
                    "tooltip":   "Grounded prompt from EricGenSearcherNode.",
                }),
                "ref_image_1": ("IMAGE", {
                    "tooltip": "Primary reference image.",
                }),
                "vl_api_url": ("STRING", {
                    "default": "http://localhost:30000",
                    "tooltip": (
                        "URL of the SGLang VL server (Gen-Searcher-8B).\n"
                        "This is the same server used by EricGenSearcherNode.\n"
                        "Default: http://localhost:30000"
                    ),
                }),
            },
            "optional": {
                "ref_image_2": ("IMAGE", {"tooltip": "Optional 2nd reference."}),
                "ref_image_3": ("IMAGE", {"tooltip": "Optional 3rd reference."}),
                "ref_image_4": ("IMAGE", {"tooltip": "Optional 4th reference."}),
                "model_name":  ("STRING", {
                    "default": "",
                    "tooltip": "VL model override. Leave blank to auto-detect.",
                }),
                "description_focus": (
                    ["auto", "full_scene", "subject_appearance",
                     "clothing_and_style", "setting_and_environment"],
                    {
                        "default": "auto",
                        "tooltip": (
                            "What to focus on when describing reference images:\n\n"
                            "auto: let the VL model decide what's most important (recommended)\n"
                            "full_scene: subject + clothing + setting + lighting\n"
                            "subject_appearance: face, hair, build only\n"
                            "clothing_and_style: garments and accessories only\n"
                            "setting_and_environment: location and lighting only"
                        ),
                    }
                ),
                "synthesis_mode": (
                    ["synthesize", "append"],
                    {
                        "default": "synthesize",
                        "tooltip": (
                            "synthesize (recommended): ask the VL model to weave the\n"
                            "  descriptions into the grounded prompt as a single paragraph.\n"
                            "  Produces the most natural and coherent result.\n\n"
                            "append: simply append the descriptions to the grounded prompt.\n"
                            "  Faster (one fewer API call) but less coherent."
                        ),
                    }
                ),
                "max_image_dim": ("INT", {
                    "default": 512,
                    "min":     256,
                    "max":     1024,
                    "step":    128,
                    "tooltip": (
                        "Max dimension when sending reference images to the VL model.\n"
                        "512px is sufficient for the model to extract visual details.\n"
                        "Higher = slower API calls, diminishing returns."
                    ),
                }),
            },
        }

    @classmethod
    def IS_CHANGED(cls, grounded_prompt, **kwargs):
        return float("nan")  # always re-run

    def describe_and_combine(
        self,
        grounded_prompt:    str,
        ref_image_1:        torch.Tensor,
        vl_api_url:         str = "http://localhost:30000",
        ref_image_2:        Optional[torch.Tensor] = None,
        ref_image_3:        Optional[torch.Tensor] = None,
        ref_image_4:        Optional[torch.Tensor] = None,
        model_name:         str = "",
        description_focus:  str = "auto",
        synthesis_mode:     str = "synthesize",
        max_image_dim:      int = 512,
    ) -> Tuple[str, str]:

        # Resolve model
        model = model_name.strip() if model_name.strip() else _detect_model(vl_api_url)
        print(f"[EricRefDescriber] VL model: {model}  url: {vl_api_url}")
        print(f"[EricRefDescriber] Focus: {description_focus}  Synthesis: {synthesis_mode}")

        # Collect reference images
        raw_tensors = [ref_image_1, ref_image_2, ref_image_3, ref_image_4]
        pil_images: List[Image.Image] = []
        for t in raw_tensors:
            if t is None:
                break
            src = t[0] if t.dim() == 4 else t
            pil_images.append(tensor_to_pil(src))

        print(f"[EricRefDescriber] Processing {len(pil_images)} reference image(s)...")
        print(f"[EricRefDescriber] Grounded prompt ({len(grounded_prompt.split())} words): "
              f"{grounded_prompt[:100]}...")

        # Describe each reference image
        desc_lines: List[str] = []
        for i, pil in enumerate(pil_images):
            print(f"[EricRefDescriber] Describing ref {i+1} ({pil.size[0]}×{pil.size[1]})...")
            b64 = _pil_to_base64(pil, max_image_dim)
            desc = _vl_describe_image(b64, description_focus, vl_api_url, model)
            if desc:
                desc_lines.append(f"Reference {i+1}: {desc}")
                print(f"[EricRefDescriber]   → {desc[:100]}...")
            else:
                print(f"[EricRefDescriber]   → (no description returned)")

        descriptions_text = "\n".join(desc_lines)

        if not desc_lines:
            # VL model returned nothing - fall back to grounded prompt as-is
            print("[EricRefDescriber] WARNING: No descriptions obtained. Using grounded prompt only.")
            return (grounded_prompt, "(VL descriptions unavailable)")

        # Combine descriptions with grounded prompt
        if synthesis_mode == "synthesize":
            print("[EricRefDescriber] Synthesizing combined prompt...")
            combined = _vl_synthesize(
                grounded_prompt, descriptions_text, vl_api_url, model
            )
            if not combined:
                print("[EricRefDescriber] Synthesis failed - falling back to append mode")
                combined = _COMBINE_TEMPLATE.format(
                    grounded_prompt=grounded_prompt,
                    descriptions=descriptions_text,
                )
        else:
            # append mode - simple concatenation
            combined = _COMBINE_TEMPLATE.format(
                grounded_prompt=grounded_prompt,
                descriptions=descriptions_text,
            )

        print(f"[EricRefDescriber] Combined prompt ({len(combined.split())} words): "
              f"{combined[:120]}...")

        return (combined, descriptions_text)
