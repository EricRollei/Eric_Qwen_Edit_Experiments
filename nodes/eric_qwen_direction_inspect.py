# Copyright (c) 2026 Eric Hiss. All rights reserved.
# Licensed under the terms in LICENSE.txt (CC BY-NC 4.0 / Commercial dual license).
# https://github.com/EricRollei/Eric_Qwen_Edit_Experiments
"""
Eric Qwen Direction Inspect Node

Visualizes per-token delta magnitude as a spatial heatmap, showing where a
direction vector's signal is concentrated in the image and in the text tokens.

The Qwen2.5-VL token sequence (after system prefix stripping) is:

    [template start ~3]  [image vision tokens ~176-196]  [instruction text ~13-25]

Each image vision token corresponds to a 28×28 pixel region of the VL
conditioning image (patch_size=14, merge_size=2) arranged in a raster grid.
Token index i within the image block maps to:
    row = i // n_cols
    col = i % n_cols
    pixel_y ≈ [row*28, (row+1)*28] in the VL image
    pixel_x ≈ [col*28, (col+1)*28] in the VL image

Because self-attention distributes information globally, each token does NOT
encode ONLY its local patch - it encodes that patch in the context of the entire
image. However the spatial map still tells you which region contributed most to
the direction's delta at each position.

Visualization layout
--------------------
    ┌──────────────────────────────────────────────────────────┐
    │  [direction name]  rms=[X]  n_avg=[N]  seq=[M]          │
    │                                                          │
    │  ┌──────────────────────┐  Signal distribution:         │
    │  │  Spatial heatmap     │  Image tokens: XX%            │
    │  │  (rows × cols grid)  │  Text tokens:  XX%            │
    │  │  (image token block) │  Header:       XX%            │
    │  └──────────────────────┘                               │
    │                                                          │
    │  Text token magnitudes (bar):                           │
    │  ████████████ token 196-207                             │
    └──────────────────────────────────────────────────────────┘

Author: Eric Hiss (GitHub: EricRollei)
"""

import math
import torch
import numpy as np
from PIL import Image, ImageDraw, ImageFont


# ──────────────────────────────────────────────────────────────────────────────
# Spatial layout helpers
# ──────────────────────────────────────────────────────────────────────────────

def _parse_vl_size(vl_size_str: str) -> tuple:
    """Parse 'WxH' string from conditioning metadata → (vl_w, vl_h)."""
    try:
        parts = vl_size_str.lower().replace("×", "x").split("x")
        return int(parts[0]), int(parts[1])
    except Exception:
        return None, None


def _grid_dims_from_vl_size(vl_w: int, vl_h: int) -> tuple:
    """
    Compute (n_rows, n_cols) from VL image dimensions.
    Formula: ceil(dim/14)//2 for each axis.
    """
    n_rows = (math.ceil(vl_h / 14)) // 2
    n_cols = (math.ceil(vl_w / 14)) // 2
    return n_rows, n_cols


def _infer_grid_dims(image_tokens: int, aspect_hint: float = None) -> tuple:
    """
    Infer (n_rows, n_cols) from image_tokens count when VL size is unknown.

    Tries all factorisations and picks the one whose aspect ratio is closest
    to aspect_hint (w/h). Defaults to a near-square grid favouring portrait
    orientation (rows >= cols) when no hint is given.
    """
    if image_tokens <= 0:
        return None, None

    best = None
    best_score = float("inf")

    for cols in range(1, image_tokens + 1):
        if image_tokens % cols != 0:
            continue
        rows = image_tokens // cols
        if aspect_hint is not None:
            # aspect_hint is w/h - cols/rows approximates that
            score = abs(cols / rows - aspect_hint)
        else:
            # Prefer portrait (rows >= cols), near-square
            score = abs(rows / cols - 1.3) + (0 if rows >= cols else 100)
        if score < best_score:
            best_score = score
            best = (rows, cols)

    return best if best else (image_tokens, 1)


# ──────────────────────────────────────────────────────────────────────────────
# Colormap (no matplotlib dependency - pure numpy/PIL)
# ──────────────────────────────────────────────────────────────────────────────

# Pre-defined colormaps as (N, 3) uint8 arrays
_COLORMAPS = {}

def _build_colormaps():
    """Build colormaps as 256×3 uint8 lookup tables."""
    t = np.linspace(0, 1, 256)

    # "hot": black → red → yellow → white
    r = np.clip(t * 3, 0, 1)
    g = np.clip(t * 3 - 1, 0, 1)
    b = np.clip(t * 3 - 2, 0, 1)
    _COLORMAPS["hot"] = (np.stack([r, g, b], axis=1) * 255).astype(np.uint8)

    # "viridis"-like: dark purple → blue → green → yellow
    r = np.clip(t * 2 - 0.5, 0, 1) * 0.9 + np.clip(1 - t, 0, 0.2)
    g = np.clip(t * 1.5 - 0.1, 0, 1)
    b = np.clip(0.8 - t * 1.2, 0, 1) + np.clip(t - 0.7, 0, 0.3) * 0.5
    _COLORMAPS["viridis"] = (np.stack([r, g, b], axis=1) * 255).astype(np.uint8)

    # "coolwarm": blue → white → red
    r_cw = np.clip(t * 2 - 0.5, 0, 1)
    g_cw = np.clip(1 - abs(t * 2 - 1) * 1.5, 0, 1)
    b_cw = np.clip(1.5 - t * 2, 0, 1)
    _COLORMAPS["coolwarm"] = (np.stack([r_cw, g_cw, b_cw], axis=1) * 255).astype(np.uint8)

    # "plasma": dark purple → pink → yellow
    r_p = np.clip(0.05 + t * 0.85 + np.sin(t * 3.14) * 0.1, 0, 1)
    g_p = np.clip(t * 0.6 - 0.1 + np.sin(t * 6.28) * 0.05, 0, 1)
    b_p = np.clip(0.55 - t * 0.6 + np.cos(t * 3.14) * 0.1, 0, 1)
    _COLORMAPS["plasma"] = (np.stack([r_p, g_p, b_p], axis=1) * 255).astype(np.uint8)

    # "gray"
    gray = np.stack([t, t, t], axis=1)
    _COLORMAPS["gray"] = (gray * 255).astype(np.uint8)

_build_colormaps()


def _apply_colormap(values: np.ndarray, cmap_name: str = "hot") -> np.ndarray:
    """
    Map float values in [0, 1] to RGB colors using the named colormap.
    values: arbitrary shape, float32
    Returns: same shape + (3,) uint8
    """
    lut = _COLORMAPS.get(cmap_name, _COLORMAPS["hot"])
    indices = (np.clip(values, 0, 1) * 255).astype(np.int32)
    return lut[indices]


# ──────────────────────────────────────────────────────────────────────────────
# Drawing helpers
# ──────────────────────────────────────────────────────────────────────────────

def _try_font(size: int):
    """Load a font, falling back to default."""
    try:
        return ImageFont.truetype("arial.ttf", size)
    except Exception:
        try:
            return ImageFont.truetype("DejaVuSans.ttf", size)
        except Exception:
            return ImageFont.load_default()


def _draw_text_safe(draw: ImageDraw.ImageDraw, xy: tuple, text: str,
                    fill=(255, 255, 255), font=None):
    """Draw text with a dark shadow for legibility on any background."""
    x, y = xy
    draw.text((x + 1, y + 1), text, fill=(0, 0, 0), font=font)
    draw.text((x, y), text, fill=fill, font=font)


# ──────────────────────────────────────────────────────────────────────────────
# Main visualization function
# ──────────────────────────────────────────────────────────────────────────────

def _build_heatmap_image(
    delta: torch.Tensor,     # [1, seq_len, hidden_dim]
    delta_mask: torch.Tensor, # [1, seq_len]
    metadata: dict,
    n_rows: int,
    n_cols: int,
    image_token_offset: int,  # where image tokens start in the sequence
    image_tokens: int,
    original_image: Image.Image = None,  # PIL image for overlay
    colormap: str = "hot",
    overlay_alpha: float = 0.4,
    cell_size: int = 32,
) -> Image.Image:
    """
    Build the full heatmap visualization as a PIL Image.
    """
    font_sm = _try_font(11)
    font_md = _try_font(13)
    font_lg = _try_font(15)

    # ── Per-token L2 magnitude ────────────────────────────────────────────────
    per_token_rms = delta[0].norm(dim=-1).float().numpy()  # [seq_len]
    seq_len = len(per_token_rms)

    # Identify token regions
    img_start = image_token_offset
    img_end   = img_start + image_tokens
    txt_start = img_end

    img_magnitudes  = per_token_rms[img_start:img_end]   # [image_tokens]
    txt_magnitudes  = per_token_rms[txt_start:]           # [text_tokens]
    hdr_magnitudes  = per_token_rms[:img_start]           # [header_tokens]

    # Distribution stats
    total_signal = per_token_rms.sum() + 1e-8
    img_pct  = img_magnitudes.sum() / total_signal * 100
    txt_pct  = txt_magnitudes.sum() / total_signal * 100
    hdr_pct  = hdr_magnitudes.sum() / total_signal * 100

    # ── Spatial grid heatmap ─────────────────────────────────────────────────
    grid_vals = np.zeros(n_rows * n_cols)
    usable = min(image_tokens, n_rows * n_cols)
    grid_vals[:usable] = img_magnitudes[:usable]
    grid_vals = grid_vals.reshape(n_rows, n_cols)

    # Normalize to [0, 1]
    grid_max = grid_vals.max()
    grid_min = grid_vals.min()
    if grid_max > grid_min:
        grid_norm = (grid_vals - grid_min) / (grid_max - grid_min)
    else:
        grid_norm = grid_vals * 0

    # Render to pixel array: each cell = cell_size × cell_size
    cell_px = cell_size
    hmap_h = n_rows * cell_px
    hmap_w = n_cols * cell_px
    hmap_rgb = np.zeros((hmap_h, hmap_w, 3), dtype=np.uint8)
    for r in range(n_rows):
        for c in range(n_cols):
            color = _apply_colormap(np.array([grid_norm[r, c]]), colormap)[0]
            y0, y1 = r * cell_px, (r + 1) * cell_px
            x0, x1 = c * cell_px, (c + 1) * cell_px
            hmap_rgb[y0:y1, x0:x1] = color

    hmap_img = Image.fromarray(hmap_rgb, "RGB")

    # Overlay on original image if provided
    if original_image is not None:
        orig_resized = original_image.resize((hmap_w, hmap_h), Image.LANCZOS).convert("RGB")
        overlay_arr = np.array(orig_resized).astype(float)
        hmap_arr    = np.array(hmap_img).astype(float)
        blended = (overlay_arr * overlay_alpha + hmap_arr * (1 - overlay_alpha)).astype(np.uint8)
        hmap_img = Image.fromarray(blended, "RGB")

    # Draw grid lines at every 5 cells for readability
    hmap_draw = ImageDraw.Draw(hmap_img)
    for r in range(0, n_rows + 1, 5):
        y = r * cell_px
        hmap_draw.line([(0, y), (hmap_w, y)], fill=(100, 100, 100), width=1)
    for c in range(0, n_cols + 1, 5):
        x = c * cell_px
        hmap_draw.line([(x, 0), (x, hmap_h)], fill=(100, 100, 100), width=1)

    # Annotate peak cell
    peak_idx   = int(grid_vals.argmax())
    peak_row   = peak_idx // n_cols
    peak_col   = peak_idx % n_cols
    peak_val   = grid_vals.max()
    px0 = peak_col * cell_px
    py0 = peak_row * cell_px
    hmap_draw.rectangle([px0, py0, px0 + cell_px - 1, py0 + cell_px - 1],
                         outline=(255, 255, 0), width=2)

    # ── Text token bar chart ─────────────────────────────────────────────────
    n_txt = len(txt_magnitudes)
    bar_h = 60
    bar_w = max(hmap_w, n_txt * 20 + 40)
    bar_img = Image.new("RGB", (bar_w, bar_h), (30, 30, 30))
    bar_draw = ImageDraw.Draw(bar_img)

    if n_txt > 0:
        txt_max = txt_magnitudes.max()
        if txt_max > 0:
            for i, val in enumerate(txt_magnitudes):
                bh = int((val / txt_max) * (bar_h - 20))
                bx = 20 + i * max(1, bar_w // (n_txt + 2))
                bw = max(1, bar_w // (n_txt + 2) - 2)
                by0 = bar_h - 18 - bh
                color = _apply_colormap(
                    np.array([val / (txt_max + 1e-8)]), colormap
                )[0].tolist()
                bar_draw.rectangle([bx, by0, bx + bw, bar_h - 18], fill=tuple(color))
                bar_draw.text((bx, bar_h - 16), str(txt_start + i),
                              fill=(180, 180, 180), font=font_sm)
        bar_draw.text((2, 2), "Text tokens", fill=(200, 200, 200), font=font_sm)

    # ── Stats panel ───────────────────────────────────────────────────────────
    stats_w  = max(260, bar_w - hmap_w)
    stats_h  = hmap_h
    stats_img = Image.new("RGB", (stats_w, stats_h), (25, 25, 35))
    sdraw = ImageDraw.Draw(stats_img)

    dir_name   = metadata.get("target_prompt", "?")[:35]
    rms_val    = metadata.get("rms", "?")
    n_avg      = metadata.get("n_averaged", 1)
    normalized = str(metadata.get("normalized", "False")) == "True"
    norm_str   = f" [norm→{metadata.get('target_rms','?')}]" if normalized else ""

    lines = [
        ("Direction:", (220, 220, 100), font_md),
        (f"  {dir_name}", (200, 200, 200), font_sm),
        ("", None, None),
        (f"rms:     {rms_val}{norm_str}", (180, 230, 180), font_sm),
        (f"n_avg:   {n_avg}", (180, 230, 180), font_sm),
        (f"seq_len: {seq_len}", (180, 230, 180), font_sm),
        (f"grid:    {n_rows}×{n_cols}", (180, 230, 180), font_sm),
        ("", None, None),
        ("Signal distribution:", (220, 180, 100), font_md),
        (f"  image tokens: {img_pct:4.1f}%", (180, 210, 255), font_sm),
        (f"  text tokens:  {txt_pct:4.1f}%", (255, 200, 150), font_sm),
        (f"  header:       {hdr_pct:4.1f}%", (160, 160, 160), font_sm),
        ("", None, None),
        (f"Peak token: {img_start + peak_idx}", (255, 255, 100), font_sm),
        (f"  row {peak_row}, col {peak_col}", (200, 200, 200), font_sm),
        (f"  rms = {peak_val:.4f}", (200, 200, 200), font_sm),
        ("", None, None),
        ("Token legend:", (200, 200, 200), font_sm),
        (f"  header: 0-{img_start - 1}", (160, 160, 160), font_sm),
        (f"  image:  {img_start}-{img_end - 1}", (180, 210, 255), font_sm),
        (f"  text:   {txt_start}-{seq_len - 1}", (255, 200, 150), font_sm),
    ]

    # Top-5 tokens by magnitude
    top5_idx = np.argsort(per_token_rms)[::-1][:5]
    sdraw.text((5, stats_h - 120), "Top 5 tokens by magnitude:", fill=(200, 200, 200), font=font_sm)
    for rank, tidx in enumerate(top5_idx):
        val = per_token_rms[tidx]
        if img_start <= tidx < img_end:
            rel = tidx - img_start
            r_pos = rel // n_cols
            c_pos = rel % n_cols
            region_str = f"img row{r_pos},col{c_pos}"
            col_fill = (180, 210, 255)
        elif tidx >= txt_start:
            region_str = f"text tok {tidx - txt_start}"
            col_fill = (255, 200, 150)
        else:
            region_str = "header"
            col_fill = (160, 160, 160)
        line = f"  #{tidx}: {val:.4f} ({region_str})"
        sdraw.text((5, stats_h - 105 + rank * 14), line, fill=col_fill, font=font_sm)

    y_cursor = 8
    for text, color, font in lines:
        if text == "" or font is None:
            y_cursor += 6
            continue
        sdraw.text((5, y_cursor), text, fill=color, font=font)
        try:
            bbox = font.getbbox(text)
            y_cursor += (bbox[3] - bbox[1]) + 4
        except Exception:
            y_cursor += 16

    # Colorbar strip on stats panel right edge
    bar_strip_w = 16
    cbar_h = hmap_h - 40
    cbar_vals = np.linspace(1, 0, cbar_h)
    cbar_rgb = _apply_colormap(cbar_vals, colormap)  # [cbar_h, 3]
    cbar_arr = np.repeat(cbar_rgb[:, np.newaxis, :], bar_strip_w, axis=1)
    cbar_img = Image.fromarray(cbar_arr.astype(np.uint8), "RGB")
    stats_img.paste(cbar_img, (stats_w - bar_strip_w - 2, 20))
    sdraw.text((stats_w - bar_strip_w - 18, 10), "max",
               fill=(200, 200, 200), font=font_sm)
    sdraw.text((stats_w - bar_strip_w - 18, 20 + cbar_h),
               "min", fill=(200, 200, 200), font=font_sm)

    # ── Assemble final image ─────────────────────────────────────────────────
    # Header bar
    header_h = 36
    total_w  = hmap_w + stats_w
    header   = Image.new("RGB", (total_w, header_h), (20, 20, 40))
    hdraw    = ImageDraw.Draw(header)

    title = f"Direction Inspect  |  {n_rows}×{n_cols} spatial grid  |  {seq_len} tokens"
    _draw_text_safe(hdraw, (8, 8), title, fill=(230, 230, 100), font=font_md)
    cmap_label = f"colormap: {colormap}"
    cmap_bbox  = font_sm.getbbox(cmap_label) if hasattr(font_sm, "getbbox") else (0, 0, 80, 14)
    _draw_text_safe(hdraw, (total_w - (cmap_bbox[2] - cmap_bbox[0]) - 10, 11),
                    cmap_label, fill=(180, 180, 180), font=font_sm)

    # Main row: heatmap + stats
    main_row = Image.new("RGB", (total_w, hmap_h), (20, 20, 30))
    main_row.paste(hmap_img, (0, 0))
    main_row.paste(stats_img, (hmap_w, 0))

    # Footer: text token bar
    footer_w = total_w
    bar_padded = Image.new("RGB", (footer_w, bar_h), (20, 20, 30))
    bar_padded.paste(bar_img, (0, 0))

    total_h = header_h + hmap_h + bar_h + 4
    final = Image.new("RGB", (total_w, total_h), (15, 15, 20))
    final.paste(header,   (0, 0))
    final.paste(main_row, (0, header_h))
    final.paste(bar_padded, (0, header_h + hmap_h + 4))

    return final


# ──────────────────────────────────────────────────────────────────────────────
# ComfyUI node
# ──────────────────────────────────────────────────────────────────────────────

class EricQwenDirectionInspect:
    """
    Visualize a QWEN_DIRECTION as a spatial heatmap.

    Shows per-token delta magnitude in two forms:

    1. Spatial grid - the image vision tokens arranged in their original spatial
       layout (rows × cols). Each cell represents a 28×28 pixel region of the
       VL conditioning image. Bright = large delta (high signal). Warm colors
       indicate where the direction's energy is concentrated.

    2. Text token bar chart - the instruction text tokens shown as vertical bars,
       revealing which text positions carry the most directional signal.

    For same-image expression directions (skip_prefix_N use case):
        Expect: mostly uniform / near-zero in spatial grid,
                bright bars in the text token region.

    For cross-image style/film directions (keep_prefix_N use case):
        Expect: bright spatial grid covering the affected image regions,
                near-zero text token bars (same prompt on both sides).

    For averaged directions (DirectionAverageFromFolder output):
        Expect: more diffuse / spatially spread spatial grid compared to
                individual pair directions - confirmation that image-specific
                content cancelled and the shared character remains.

    Connecting a QWEN_CONDITIONING lets the node read the exact grid dimensions
    (rows, cols) from the VL image metadata. Without it, the grid is inferred
    from the image_tokens count.
    """

    CATEGORY = "Eric Qwen-Edit/Conditioning/Directions"
    FUNCTION = "inspect"
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("heatmap",)

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "direction": ("QWEN_DIRECTION", {
                    "tooltip": "Direction to inspect"
                }),
            },
            "optional": {
                "conditioning": ("QWEN_CONDITIONING", {
                    "tooltip": (
                        "Optional - connects spatial layout metadata.\n"
                        "Provides exact grid dimensions from the VL image size.\n"
                        "Connect the conditioning used to compute this direction."
                    )
                }),
                "original_image": ("IMAGE", {
                    "tooltip": (
                        "Optional - the original image for overlay context.\n"
                        "The heatmap is blended on top of the source image so you\n"
                        "can see which face regions correspond to high-delta tokens."
                    )
                }),
                "colormap": (["hot", "viridis", "plasma", "coolwarm", "gray"], {
                    "default": "hot",
                    "tooltip": (
                        "hot:      black→red→yellow→white  (high contrast, shows peaks clearly)\n"
                        "viridis:  dark blue→green→yellow  (perceptually uniform)\n"
                        "plasma:   purple→pink→yellow       (perceptually uniform, vivid)\n"
                        "coolwarm: blue→white→red           (good for signed values)\n"
                        "gray:     black→white              (neutral)"
                    )
                }),
                "overlay_alpha": ("FLOAT", {
                    "default": 0.35, "min": 0.0, "max": 1.0, "step": 0.05,
                    "tooltip": (
                        "Original image blend when original_image is connected.\n"
                        "0.0 = pure heatmap, 1.0 = pure original, 0.35 = good default."
                    )
                }),
                "cell_size": ("INT", {
                    "default": 32, "min": 8, "max": 80, "step": 4,
                    "tooltip": "Pixel size of each spatial grid cell. Larger = bigger output image."
                }),
                "image_token_offset": ("INT", {
                    "default": 0, "min": 0, "max": 16, "step": 1,
                    "tooltip": (
                        "Number of template header tokens before the image block.\n"
                        "0 = auto-detect (finds where significant signal starts).\n"
                        "Typically 3-6 for Qwen2.5-VL chat template."
                    )
                }),
            },
        }

    def inspect(
        self,
        direction: dict,
        conditioning: dict = None,
        original_image=None,
        colormap: str = "hot",
        overlay_alpha: float = 0.35,
        cell_size: int = 32,
        image_token_offset: int = 0,
    ) -> tuple:
        import torch

        delta      = direction["delta_embeds"]   # [1, seq_len, hidden_dim]
        delta_mask = direction["delta_mask"]      # [1, seq_len]
        metadata   = direction.get("metadata", {})
        seq_len    = delta.shape[1]

        # ── Determine grid dimensions ─────────────────────────────────────────
        n_rows, n_cols = None, None
        image_tokens   = 0
        vl_size_str    = None

        # Try conditioning metadata first (most accurate)
        if conditioning is not None:
            cond_meta = conditioning.get("metadata", {})
            image_tokens = int(cond_meta.get("image_tokens", 0))
            vl_size_str  = cond_meta.get("vl_size", "")
            if vl_size_str:
                vl_w, vl_h = _parse_vl_size(vl_size_str)
                if vl_w and vl_h:
                    n_rows, n_cols = _grid_dims_from_vl_size(vl_w, vl_h)

        # Fall back to direction metadata
        if (n_rows is None) and conditioning is None:
            image_tokens = int(metadata.get("image_tokens", 0))
            vl_size_str  = metadata.get("vl_size", "")
            if vl_size_str:
                vl_w, vl_h = _parse_vl_size(vl_size_str)
                if vl_w and vl_h:
                    n_rows, n_cols = _grid_dims_from_vl_size(vl_w, vl_h)

        # Infer from image_tokens if still unknown
        if (n_rows is None) and image_tokens > 0:
            n_rows, n_cols = _infer_grid_dims(image_tokens)

        # Last resort: treat entire valid sequence as 1D (no spatial info)
        if n_rows is None:
            valid_tokens = int(delta_mask.sum().item())
            image_tokens = valid_tokens
            # Try a squarish grid
            n_cols = max(1, int(math.sqrt(valid_tokens)))
            n_rows = math.ceil(valid_tokens / n_cols)
            print(f"[EricQwenInspect] No VL size info - using inferred grid {n_rows}×{n_cols}")

        # ── Auto-detect image token offset ────────────────────────────────────
        per_token_rms = delta[0].norm(dim=-1).float()

        if image_token_offset == 0:
            # Find where signal meaningfully begins - skip leading near-zero tokens
            threshold = float(per_token_rms.max()) * 0.02
            offset = 0
            for i in range(min(16, seq_len)):
                if float(per_token_rms[i]) < threshold:
                    offset = i + 1
                else:
                    break
            image_token_offset = offset

        print(
            f"[EricQwenInspect] seq={seq_len}, grid={n_rows}×{n_cols}, "
            f"image_tokens={image_tokens}, offset={image_token_offset}"
        )

        # ── Console report ─────────────────────────────────────────────────────
        img_start = image_token_offset
        img_end   = img_start + image_tokens
        txt_start = img_end

        per_token_np = per_token_rms.numpy()
        img_mag  = per_token_np[img_start:img_end]
        txt_mag  = per_token_np[txt_start:]
        total_signal = per_token_np.sum() + 1e-8

        img_pct = img_mag.sum() / total_signal * 100
        txt_pct = txt_mag.sum() / total_signal * 100

        print(f"[EricQwenInspect] Signal: image={img_pct:.1f}% | text={txt_pct:.1f}%")

        # Top 10 tokens
        top10 = np.argsort(per_token_np)[::-1][:10]
        print(f"[EricQwenInspect] Top 10 token magnitudes:")
        for rank, tidx in enumerate(top10):
            val = per_token_np[tidx]
            if img_start <= tidx < img_end:
                rel = tidx - img_start
                r_pos, c_pos = rel // n_cols, rel % n_cols
                loc = f"image row={r_pos} col={c_pos}"
            elif tidx >= txt_start:
                loc = f"text[{tidx - txt_start}]"
            else:
                loc = "header"
            print(f"  #{rank+1:2d}: token {tidx:3d}  rms={val:.5f}  ({loc})")

        # ── Build PIL image ────────────────────────────────────────────────────
        # Convert original_image tensor if provided
        orig_pil = None
        if original_image is not None:
            import torch
            img_tensor = original_image
            if img_tensor.dim() == 4:
                img_tensor = img_tensor[0]
            # [H, W, C] float → PIL
            arr = (img_tensor.cpu().float().numpy() * 255).clip(0, 255).astype(np.uint8)
            orig_pil = Image.fromarray(arr, "RGB")

        # Add spatial metadata to direction metadata for the visualization
        vis_metadata = dict(metadata)
        vis_metadata["image_tokens"] = image_tokens
        vis_metadata["grid"] = f"{n_rows}x{n_cols}"

        final_pil = _build_heatmap_image(
            delta=delta,
            delta_mask=delta_mask,
            metadata=vis_metadata,
            n_rows=n_rows,
            n_cols=n_cols,
            image_token_offset=image_token_offset,
            image_tokens=image_tokens,
            original_image=orig_pil,
            colormap=colormap,
            overlay_alpha=overlay_alpha,
            cell_size=cell_size,
        )

        # Convert PIL → ComfyUI tensor [1, H, W, C] float32
        out_arr = np.array(final_pil).astype(np.float32) / 255.0
        out_tensor = torch.from_numpy(out_arr).unsqueeze(0)

        return (out_tensor,)
