# Copyright (c) 2026 Eric Hiss. All rights reserved.
# Licensed under the terms in LICENSE.txt (CC BY-NC 4.0 / Commercial dual license).
# https://github.com/EricRollei/Eric_Qwen_Edit_Experiments
"""
Eric Gen-Searcher Node
Agentic web research for grounded image generation using Gen-Searcher-8B.

Author: Eric Hiss (GitHub: EricRollei)
"""

import configparser
import glob
import io
import json
import logging
import os
import re
import shutil
import subprocess
import tempfile
import urllib.error
import urllib.request
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from PIL import Image

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
#  Constants
# ---------------------------------------------------------------------------

_PACKAGE_ROOT = Path(__file__).resolve().parent.parent
_CONFIG_PATH  = _PACKAGE_ROOT / "api_keys.ini"
_REF_MAX_DIM  = 768
_CANDIDATE_MULTIPLIER = 3   # collect 3× slots so we have fallbacks

# gallery-dl - already installed in the ComfyUI Python environment
_GALLERY_DL_EXE  = r"A:\Comfy25\ComfyUI_windows_portable\python_embeded\Scripts\gallery-dl.exe"
_GALLERY_DL_CONF = r"A:\Comfy25\ComfyUI_windows_portable\ComfyUI\custom_nodes\download_tools\configs\gallery-dl.conf"

# ---------------------------------------------------------------------------
#  Domain classification
# ---------------------------------------------------------------------------

# CDN/redirect endpoints that return HTML instead of image data on direct fetch.
_BLOCKED_CDN_DOMAINS = {
    "lookaside.instagram.com",
    "lookaside.fbsbx.com",
    "lookaside.facebook.com",
    "scontent.fbcdn.net",
    "scontent.instagram.com",
    "external.fwfw1-1.fna.fbcdn.net",
    "external.fwfw1-2.fna.fbcdn.net",
    "t.co",
}

# Page domains gallery-dl can download from.
_GALLERY_DL_PAGE_DOMAINS = {
    "instagram.com", "www.instagram.com",
    "twitter.com",   "www.twitter.com",
    "x.com",         "www.x.com",
    "pinterest.com", "www.pinterest.com", "pin.it",
    "500px.com",     "www.500px.com",
    "flickr.com",    "www.flickr.com",
    "deviantart.com", "www.deviantart.com",
    "bsky.app",      "bluesky.com",
    "reddit.com",    "www.reddit.com", "old.reddit.com",
    "tiktok.com",    "www.tiktok.com", "vm.tiktok.com",
    "facebook.com",  "www.facebook.com", "fb.com",
    "tumblr.com",    "www.tumblr.com",
}

# Platforms requiring a sacrificial account - disabled by default.
# Instagram and Facebook flag automated access aggressively.
_INSTAGRAM_DOMAINS = {"instagram.com", "www.instagram.com"}
_FACEBOOK_DOMAINS  = {
    "facebook.com", "www.facebook.com", "fb.com",
    "lookaside.fbsbx.com", "lookaside.facebook.com",
    "scontent.fbcdn.net",
}


def _domain_of(url: str) -> str:
    try:
        from urllib.parse import urlparse
        return urlparse(url).netloc.lower()
    except Exception:
        return ""


def _matches_domain_set(url: str, domain_set: set) -> bool:
    host = _domain_of(url)
    return any(host == d or host.endswith("." + d) for d in domain_set)


def _is_blocked_cdn(url: str) -> bool:
    return _matches_domain_set(url, _BLOCKED_CDN_DOMAINS)


def _is_gallery_dl_page(url: str) -> bool:
    return _matches_domain_set(url, _GALLERY_DL_PAGE_DOMAINS)


# ---------------------------------------------------------------------------
#  System prompt
# ---------------------------------------------------------------------------

_GEN_SEARCHER_SYSTEM = """You are Gen-Searcher, an expert AI assistant for knowledge-grounded image generation. Your task is to research the user's image generation prompt, gather visual references, and then write a final detailed image generation prompt.

You have access to three tools:
- search(query): Search the web for text information
- image_search(query): Search for reference images of a subject
- browse(url): Read the full content of a web page

## Research Strategy
- Aim for 3-6 searches - enough to be thorough, not exhaustive
- Prioritize visual details: colors, textures, shapes, proportions, distinctive features
- Stop when you have enough visual information to write a complete description

## CRITICAL - Final Output Format
When you have enough information, output ONLY a single paragraph of descriptive text.
- NO bullet points, NO lists, NO analysis, NO reasoning
- NO phrases like "I will...", "From my analysis...", "The user wants..."
- Just one flowing paragraph describing exactly what to generate
- Must be 150-300 words of vivid visual description
- Start directly with the subject, style, lighting, etc.

Example of correct output:
"A cinematic portrait of a young woman with long dark braided hair, wearing an oversized magenta Adidas hoodie with white logo detailing on the hood panels, standing on a red clay tennis court. The hoodie features drawstring closures and a relaxed athletic fit. She wears white tennis shorts and clean white sneakers. Soft afternoon light falls from the left, casting gentle shadows. The background shows blurred court lines and a net. Shot at eye level with a medium telephoto perspective, shallow depth of field. Photorealistic, high detail."

Do not explain your research. Do not list what you found. Just write the prompt."""


# ---------------------------------------------------------------------------
#  Analysis output detection
# ---------------------------------------------------------------------------

_ANALYSIS_MARKERS = [
    "from my analysis", "from the search results", "i have gathered",
    "i will describe", "i will focus", "i will make", "i will create",
    "i will combine", "the user asks", "the user wants",
    "i have enough information", "based on the search",
    "based on the images", "based on the results", "let me ", "to summarize",
]


def _is_analysis_output(text: str) -> bool:
    lower = text.lower()
    if text.count("\n-") + text.count("\n*") + text.count("\n*") >= 3:
        return True
    return any(m in lower for m in _ANALYSIS_MARKERS)


def _cleanup_prompt(raw_output: str, api_url: str, model: str, messages: list) -> str:
    print("[EricGenSearcher] Analysis output detected - requesting clean prompt...")
    cleanup_messages = messages + [{
        "role": "user",
        "content": (
            "Your previous response contained analysis and bullet points. "
            "Now write ONLY the final image generation prompt - a single flowing paragraph "
            "of 150-300 words describing exactly what to generate. "
            "No bullets, no analysis, no 'I will...', no explanations. "
            "Start directly with the subject description."
        ),
    }]
    try:
        response = _chat_completion(api_url, cleanup_messages, None, model, max_tokens=1024)
        cleaned  = response["choices"][0]["message"].get("content", "").strip()
        cleaned  = re.sub(r"<think>.*?</think>", "", cleaned, flags=re.DOTALL).strip()
        if cleaned and not _is_analysis_output(cleaned):
            print(f"[EricGenSearcher] Cleanup succeeded ({len(cleaned.split())} words)")
            return cleaned
        print("[EricGenSearcher] Cleanup still looks like analysis - using raw output")
    except Exception as e:
        print(f"[EricGenSearcher] Cleanup call failed: {e}")
    return raw_output


# ---------------------------------------------------------------------------
#  API key resolution
# ---------------------------------------------------------------------------

def _resolve_api_key(service: str) -> str:
    env_map = {
        "serper":   ["SERPER_API_KEY"],
        "jina":     ["JINA_API_KEY"],
        "deepseek": ["DEEPSEEK_API_KEY"],
        "openai":   ["OPENAI_API_KEY"],
    }
    for env_name in env_map.get(service, [f"{service.upper()}_API_KEY"]):
        val = os.environ.get(env_name, "").strip()
        if val:
            return val
    if _CONFIG_PATH.exists():
        cfg = configparser.ConfigParser()
        cfg.read(str(_CONFIG_PATH), encoding="utf-8")
        if cfg.has_section("api_keys"):
            val = cfg.get("api_keys", service, fallback="").strip()
            if val:
                return val
    return ""


# ---------------------------------------------------------------------------
#  Tool implementations
# ---------------------------------------------------------------------------

def _serper_search(query: str, api_key: str, num_results: int = 5) -> str:
    if not api_key:
        return f"[search unavailable: no Serper API key]\nQuery: {query}"
    url  = "https://google.serper.dev/search"
    body = json.dumps({"q": query, "num": num_results}).encode()
    req  = urllib.request.Request(
        url, data=body,
        headers={"X-API-KEY": api_key, "Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=15) as r:
            data = json.loads(r.read())
    except Exception as e:
        return f"[search error: {e}]"
    parts = [
        f"**{i.get('title','')}**\n{i.get('snippet','')}\nURL: {i.get('link','')}"
        for i in data.get("organic", [])[:num_results]
    ]
    return "\n\n".join(parts) if parts else "[no search results]"


def _serper_image_search(query: str, api_key: str, num_results: int = 10) -> Tuple[str, List[Dict]]:
    """Image search via Serper.dev. Returns (summary, [{img, page}, ...])."""
    if not api_key:
        return "[image_search unavailable: no Serper API key]", []
    url  = "https://google.serper.dev/images"
    body = json.dumps({"q": query, "num": num_results}).encode()
    req  = urllib.request.Request(
        url, data=body,
        headers={"X-API-KEY": api_key, "Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=15) as r:
            data = json.loads(r.read())
    except Exception as e:
        return f"[image_search error: {e}]", []
    images  = data.get("images", [])[:num_results]
    entries = [
        {"img": img.get("imageUrl", ""), "page": img.get("link", "")}
        for img in images if img.get("imageUrl")
    ]
    lines   = [f"- {img.get('title','')}: {img.get('imageUrl','')}" for img in images]
    return f"Found {len(entries)} images for '{query}':\n" + "\n".join(lines), entries


def _jina_browse(url: str, api_key: str = "") -> str:
    jina_url = f"https://r.jina.ai/{url}"
    headers  = {"Accept": "text/plain"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    req = urllib.request.Request(jina_url, headers=headers, method="GET")
    try:
        with urllib.request.urlopen(req, timeout=20) as r:
            content = r.read().decode("utf-8", errors="replace")
        return content[:4000] + "\n[... truncated ...]" if len(content) > 4000 else content
    except Exception as e:
        return f"[browse error for {url}: {e}]"


# ---------------------------------------------------------------------------
#  Image download - direct + gallery-dl fallback
# ---------------------------------------------------------------------------

def _download_direct(url: str, timeout: int = 10) -> Optional[Image.Image]:
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"}, method="GET")
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return Image.open(io.BytesIO(r.read())).convert("RGB")
    except Exception as e:
        print(f"[EricGenSearcher] Direct download failed {url[:80]}: {e}")
        return None


def _download_with_gallery_dl(page_url: str, timeout: int = 45) -> Optional[Image.Image]:
    """Download using gallery-dl subprocess. Returns first downloaded image."""
    if not os.path.exists(_GALLERY_DL_EXE):
        print(f"[EricGenSearcher] gallery-dl not found at {_GALLERY_DL_EXE}")
        return None

    tmp_dir = tempfile.mkdtemp(prefix="gensearch_")
    try:
        result = subprocess.run(
            [_GALLERY_DL_EXE, "--config", _GALLERY_DL_CONF,
             "--dest", tmp_dir, "--range", "1", "--no-skip", page_url],
            capture_output=True, text=True, timeout=timeout,
        )
        found = []
        for ext in ("*.jpg", "*.jpeg", "*.png", "*.webp", "*.gif", "*.avif"):
            found.extend(glob.glob(os.path.join(tmp_dir, "**", ext), recursive=True))
        if found:
            found.sort(key=os.path.getmtime)
            img = Image.open(found[0]).convert("RGB")
            print(f"[EricGenSearcher] gallery-dl ✓ {os.path.basename(found[0])} "
                  f"({img.size[0]}×{img.size[1]})")
            return img
        stderr = result.stderr.strip()[:120] if result.stderr else ""
        print(f"[EricGenSearcher] gallery-dl: no images from {page_url[:60]}"
              f"{' - ' + stderr if stderr else ''}")
    except subprocess.TimeoutExpired:
        print(f"[EricGenSearcher] gallery-dl timeout ({timeout}s) for {page_url[:60]}")
    except Exception as e:
        print(f"[EricGenSearcher] gallery-dl exception: {e}")
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)
    return None


def _download_smart(
    entry:         Dict,
    use_instagram: bool = False,
    use_facebook:  bool = False,
) -> Optional[Image.Image]:
    """Download strategy with per-platform enable flags.

    1. Check if the platform is disabled - skip immediately if so.
    2. Direct HTTP for normal CDN URLs.
    3. gallery-dl for social media page URLs.
    4. Last-resort direct attempt on blocked CDN URLs.
    """
    img_url  = entry.get("img",  "")
    page_url = entry.get("page", "")

    # --- Platform gate ---
    # Check page URL first (more reliable domain), fall back to img URL
    check_url = page_url or img_url
    if check_url:
        if not use_instagram and _matches_domain_set(check_url, _INSTAGRAM_DOMAINS):
            print(f"[EricGenSearcher] Skipping Instagram (disabled) - enable with use_instagram=True")
            return None
        if not use_facebook and _matches_domain_set(check_url, _FACEBOOK_DOMAINS):
            print(f"[EricGenSearcher] Skipping Facebook (disabled) - enable with use_facebook=True")
            return None

    # Step 1: direct download if not a known-blocked CDN
    if img_url and not _is_blocked_cdn(img_url):
        img = _download_direct(img_url)
        if img:
            return img

    # Step 2: gallery-dl via the source page URL
    if page_url and _is_gallery_dl_page(page_url):
        img = _download_with_gallery_dl(page_url)
        if img:
            return img

    # Step 3: last-resort direct attempt on blocked CDN URL
    if img_url and _is_blocked_cdn(img_url):
        img = _download_direct(img_url)
        if img:
            return img

    return None


# ---------------------------------------------------------------------------
#  Low-level API call
# ---------------------------------------------------------------------------

def _chat_completion(
    api_url: str, messages: list, tools: Optional[list], model: str,
    api_key: str = "", timeout: int = 120, max_tokens: int = 2048,
) -> dict:
    url = api_url.rstrip("/")
    if not url.endswith("/chat/completions"):
        url = url.rstrip("/v1").rstrip("/") + "/v1/chat/completions"
    payload: dict = {"model": model, "messages": messages,
                     "max_tokens": max_tokens, "stream": False}
    if tools:
        payload["tools"] = tools
        payload["tool_choice"] = "auto"
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    req = urllib.request.Request(
        url, data=json.dumps(payload).encode(), headers=headers, method="POST"
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read())
    except urllib.error.HTTPError as e:
        raise RuntimeError(f"API error {e.code}: {e.read().decode()[:400]}") from e
    except urllib.error.URLError as e:
        raise RuntimeError(
            f"Cannot connect to Gen-Searcher at {api_url}.\n"
            f"Is SGLang running? (bash /mnt/h/scripts/gen_searcher_setup/04_serve.sh)\n"
            f"Error: {e.reason}"
        ) from e


def _detect_model(api_url: str) -> str:
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
#  Inline tool call parser
# ---------------------------------------------------------------------------

def _parse_inline_tool_calls(content: str) -> list:
    tool_calls = []
    for i, match in enumerate(re.findall(r"<tool_call>\s*(.*?)\s*</tool_call>", content, re.DOTALL)):
        try:
            data = json.loads(match.strip())
            name = data.get("name", "")
            if not name:
                continue
            tool_calls.append({
                "id": f"inline_{i}", "type": "function",
                "function": {"name": name, "arguments": json.dumps(data.get("arguments", {}))},
            })
            print(f"[EricGenSearcher] Inline tool call: {name}({data.get('arguments',{})})")
        except json.JSONDecodeError as e:
            print(f"[EricGenSearcher] Inline parse error: {e}")
    return tool_calls


# ---------------------------------------------------------------------------
#  Tool definitions
# ---------------------------------------------------------------------------

_TOOLS = [
    {"type": "function", "function": {
        "name": "search",
        "description": "Search the web for text information about a topic, person, place, object, or event.",
        "parameters": {"type": "object",
                       "properties": {"query": {"type": "string"}},
                       "required": ["query"]},
    }},
    {"type": "function", "function": {
        "name": "image_search",
        "description": "Search for reference images. Returns image URLs showing what the subject looks like.",
        "parameters": {"type": "object",
                       "properties": {"query": {"type": "string"}},
                       "required": ["query"]},
    }},
    {"type": "function", "function": {
        "name": "browse",
        "description": "Read the full content of a web page URL for detailed visual information.",
        "parameters": {"type": "object",
                       "properties": {"url": {"type": "string"}},
                       "required": ["url"]},
    }},
]


# ---------------------------------------------------------------------------
#  Execute a single tool call
# ---------------------------------------------------------------------------

def _execute_tool(fn_name, fn_args, serper_key, jina_key,
                  enable_browse, log_lines, image_entries, max_candidates) -> str:
    if fn_name == "search":
        q = fn_args.get("query", "")
        log_lines.append(f"[search] {q}")
        print(f"[EricGenSearcher] search: {q}")
        return _serper_search(q, serper_key)

    elif fn_name == "image_search":
        q = fn_args.get("query", "")
        log_lines.append(f"[image_search] {q}")
        print(f"[EricGenSearcher] image_search: {q}")
        summary, entries = _serper_image_search(q, serper_key)
        existing_imgs = {e["img"] for e in image_entries}
        for entry in entries:
            if len(image_entries) >= max_candidates:
                break
            if entry["img"] and entry["img"] not in existing_imgs:
                image_entries.append(entry)
                existing_imgs.add(entry["img"])
        return summary

    elif fn_name == "browse":
        url = fn_args.get("url", "")
        if enable_browse:
            log_lines.append(f"[browse] {url}")
            print(f"[EricGenSearcher] browse: {url}")
            return _jina_browse(url, jina_key)
        log_lines.append(f"[browse skipped] {url}")
        return "[browse disabled]"

    return f"[unknown tool: {fn_name}]"


# ---------------------------------------------------------------------------
#  Agent loop
# ---------------------------------------------------------------------------

def _run_agent_loop(
    prompt, api_url, model, serper_key, jina_key,
    max_hops=8, enable_browse=True, max_ref_images=4,
) -> Tuple[str, List[Dict], str]:
    max_candidates = max_ref_images * _CANDIDATE_MULTIPLIER
    messages = [
        {"role": "system", "content": _GEN_SEARCHER_SYSTEM},
        {"role": "user",   "content": prompt},
    ]
    image_entries: List[Dict] = []
    log_lines:     List[str]  = []
    hops     = 0
    grounded = prompt

    while hops < max_hops:
        response   = _chat_completion(api_url, messages, _TOOLS, model, max_tokens=2048)
        choice     = response["choices"][0]
        message    = choice["message"]
        content    = (message.get("content") or "").strip()

        tool_calls = message.get("tool_calls") or []
        if not tool_calls and content:
            tool_calls = _parse_inline_tool_calls(content)

        if not tool_calls:
            grounded = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL).strip()
            grounded = re.sub(r"<tool_call>.*", "", grounded, flags=re.DOTALL).strip()
            if not grounded:
                grounded = prompt
            if _is_analysis_output(grounded):
                grounded = _cleanup_prompt(grounded, api_url, model, messages + [message])
            break

        messages.append(message)
        using_inline = (message.get("tool_calls") is None and bool(tool_calls))

        for tc in tool_calls:
            fn_name = tc["function"]["name"]
            try:
                fn_args = json.loads(tc["function"].get("arguments", "{}"))
            except json.JSONDecodeError:
                fn_args = {}
            result = _execute_tool(fn_name, fn_args, serper_key, jina_key,
                                   enable_browse, log_lines, image_entries, max_candidates)
            if using_inline:
                messages.append({"role": "user", "content": f"<tool_response>\n{result}\n</tool_response>"})
            else:
                messages.append({"role": "tool", "tool_call_id": tc.get("id", f"c{hops}"), "content": result})

        hops += 1

    else:
        print(f"[EricGenSearcher] max_hops={max_hops} reached, requesting final prompt")
        messages.append({
            "role": "user",
            "content": (
                "You have gathered enough information. "
                "Write ONLY the final image generation prompt now - "
                "a single paragraph of 150-300 words. "
                "No bullets, no analysis, no explanations. Start with the subject."
            ),
        })
        resp = _chat_completion(api_url, messages, None, model, max_tokens=1024)
        content  = resp["choices"][0]["message"].get("content", "").strip()
        grounded = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL).strip() or prompt
        if _is_analysis_output(grounded):
            grounded = _cleanup_prompt(grounded, api_url, model, messages)

    search_log = "\n".join(log_lines) if log_lines else "(no searches performed)"
    print(f"[EricGenSearcher] Done. {hops} hop(s), {len(image_entries)} candidate entries")
    print(f"[EricGenSearcher] Prompt ({len(grounded.split())} words): {grounded[:120]}...")
    return grounded, image_entries, search_log


# ---------------------------------------------------------------------------
#  Image resize helpers
# ---------------------------------------------------------------------------

def _pil_to_tensor(img: Image.Image) -> torch.Tensor:
    import numpy as np
    return torch.from_numpy(np.array(img).astype(np.float32) / 255.0)


def _resize_preserve_aspect(img: Image.Image, max_dim: int = _REF_MAX_DIM) -> Image.Image:
    w, h = img.size
    if max(w, h) <= max_dim:
        new_w = max(32, (w // 32) * 32)
        new_h = max(32, (h // 32) * 32)
        if new_w == w and new_h == h:
            return img
        return img.resize((new_w, new_h), Image.LANCZOS)
    if w >= h:
        new_w = max_dim
        new_h = max(32, int(h * max_dim / w))
    else:
        new_h = max_dim
        new_w = max(32, int(w * max_dim / h))
    new_w = max(32, (new_w // 32) * 32)
    new_h = max(32, (new_h // 32) * 32)
    return img.resize((new_w, new_h), Image.LANCZOS)


def _pad_to_size(tensor: torch.Tensor, target_h: int, target_w: int) -> torch.Tensor:
    h, w, c = tensor.shape
    if h == target_h and w == target_w:
        return tensor
    padded = torch.zeros(target_h, target_w, c, dtype=tensor.dtype)
    top  = (target_h - h) // 2
    left = (target_w - w) // 2
    padded[top:top+h, left:left+w, :] = tensor
    return padded


def _build_reference_batch(tensors: List[torch.Tensor]) -> torch.Tensor:
    if not tensors:
        return torch.ones(1, _REF_MAX_DIM, _REF_MAX_DIM, 3) * 0.5
    max_h = max(t.shape[1] for t in tensors)
    max_w = max(t.shape[2] for t in tensors)
    padded = [_pad_to_size(t[0], max_h, max_w).unsqueeze(0) for t in tensors]
    return torch.cat(padded, dim=0)


# ---------------------------------------------------------------------------
#  ComfyUI Node
# ---------------------------------------------------------------------------

class EricGenSearcherNode:
    """
    Agentic web research for grounded image generation.

    Downloads reference images at native aspect ratio.
    Uses gallery-dl as a fallback for social media URLs.
    Instagram and Facebook are disabled by default to protect your account
    - enable them only with a dedicated dummy account.
    """

    CATEGORY     = "Eric/QwenImage"
    FUNCTION     = "research"
    RETURN_TYPES = ("STRING", "IMAGE", "STRING", "IMAGE", "IMAGE", "IMAGE", "IMAGE")
    RETURN_NAMES = ("grounded_prompt", "reference_images", "search_log",
                    "ref_image_1", "ref_image_2", "ref_image_3", "ref_image_4")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {
                    "multiline": True,
                    "default":   "a portrait of ...",
                    "tooltip":   "Image generation prompt. Gen-Searcher researches and expands it.",
                }),
                "agent_api_url": ("STRING", {
                    "default": "http://localhost:30000",
                    "tooltip": "SGLang server URL.",
                }),
            },
            "optional": {
                "model_name":        ("STRING",  {"default": ""}),
                "max_hops":          ("INT",     {"default": 8, "min": 1, "max": 20, "step": 1}),
                "ref_image_count":   ("INT",     {"default": 4, "min": 1, "max": 4,  "step": 1}),
                "enable_browse":     ("BOOLEAN", {"default": True}),
                "fallback_on_error": ("BOOLEAN", {"default": True}),
                "use_instagram":     ("BOOLEAN", {
                    "default": False,
                    "tooltip": (
                        "Allow gallery-dl to download from Instagram via Firefox cookies.\n"
                        "WARNING: Instagram flags automated access aggressively.\n"
                        "Only enable with a dedicated dummy/sacrificial account.\n"
                        "Default: OFF to protect your main account."
                    ),
                }),
                "use_facebook":      ("BOOLEAN", {
                    "default": False,
                    "tooltip": (
                        "Allow gallery-dl to download from Facebook.\n"
                        "WARNING: Facebook flags automated access aggressively.\n"
                        "Only enable with a dedicated dummy/sacrificial account.\n"
                        "Default: OFF to protect your main account."
                    ),
                }),
            },
        }

    @classmethod
    def IS_CHANGED(cls, prompt, **kwargs):
        return float("nan")

    def research(
        self,
        prompt:            str,
        agent_api_url:     str  = "http://localhost:30000",
        model_name:        str  = "",
        max_hops:          int  = 8,
        ref_image_count:   int  = 4,
        enable_browse:     bool = True,
        fallback_on_error: bool = True,
        use_instagram:     bool = False,
        use_facebook:      bool = False,
    ):
        serper_key = _resolve_api_key("serper")
        jina_key   = _resolve_api_key("jina")
        if not serper_key:
            print(f"[EricGenSearcher] WARNING: No Serper key. Add to {_CONFIG_PATH}: serper = your-key")

        model = model_name.strip() if model_name.strip() else _detect_model(agent_api_url)
        print(f"[EricGenSearcher] model={model}  url={agent_api_url}")
        print(f"[EricGenSearcher] prompt: {prompt[:100]}{'...' if len(prompt) > 100 else ''}")

        gdl_available = os.path.exists(_GALLERY_DL_EXE)
        platforms = []
        if gdl_available:
            platforms.append("Pinterest/Flickr/500px/Reddit/Twitter")
            if use_instagram:
                platforms.append("Instagram ⚠")
            if use_facebook:
                platforms.append("Facebook ⚠")
            print(f"[EricGenSearcher] gallery-dl enabled for: {', '.join(platforms)}")
        else:
            print(f"[EricGenSearcher] gallery-dl not found - direct download only")

        placeholder = torch.ones(1, _REF_MAX_DIM, _REF_MAX_DIM, 3) * 0.5

        try:
            grounded, image_entries, search_log = _run_agent_loop(
                prompt=prompt, api_url=agent_api_url, model=model,
                serper_key=serper_key, jina_key=jina_key,
                max_hops=max_hops, enable_browse=enable_browse,
                max_ref_images=ref_image_count,
            )
        except RuntimeError as e:
            if fallback_on_error:
                print(f"[EricGenSearcher] ERROR (fallback): {e}")
                return (prompt, placeholder, f"ERROR: {e}",
                        placeholder, placeholder, placeholder, placeholder)
            raise

        print(f"[EricGenSearcher] {len(image_entries)} candidates, need {ref_image_count}...")
        good_images: List[torch.Tensor] = []

        for entry in image_entries:
            if len(good_images) >= ref_image_count:
                break
            img = _download_smart(entry,
                                  use_instagram=use_instagram,
                                  use_facebook=use_facebook)
            if img is not None:
                img = _resize_preserve_aspect(img, _REF_MAX_DIM)
                ar  = img.size[0] / img.size[1]
                src = "gdl" if _is_blocked_cdn(entry.get("img", "")) else "direct"
                print(f"[EricGenSearcher]   ref {len(good_images)+1}: "
                      f"{img.size[0]}×{img.size[1]} AR={ar:.2f} [{src}]")
                good_images.append(_pil_to_tensor(img).unsqueeze(0))

        print(f"[EricGenSearcher] Downloaded {len(good_images)}/{ref_image_count}")

        fill       = good_images[-1] if good_images else placeholder
        individual = good_images[:]
        while len(individual) < 4:
            individual.append(fill.clone())

        ref_tensor = _build_reference_batch(individual[:ref_image_count])
        print(f"[EricGenSearcher] batch shape: {ref_tensor.shape}")

        return (grounded, ref_tensor, search_log,
                individual[0], individual[1], individual[2], individual[3])
