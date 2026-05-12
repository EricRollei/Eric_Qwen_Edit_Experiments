# Eric Qwen Conditioning System
## Semantic Direction Vectors for Precise Image Control

*Part of [Eric_Qwen_Edit_Experiments](https://github.com/EricRollei/Eric_Qwen_Edit_Experiments)*

---

## Overview

The conditioning system adds a new layer of control to Qwen-Edit workflows: instead of describing edits purely through text prompts, you can capture the *difference* between two encoded images as a reusable direction vector and apply it to any image at any intensity.

This enables:
- **Expression control** (with PixelSmile LoRA) - smooth slider from neutral to happy to exaggerated
- **Film emulation** - encode a Nik Silver Efex conversion and apply it to any photo
- **Attribute editing** - lighting, skin quality, color grade, grain, colorization
- **Compound edits** - chain multiple directions in one pass
- **Direction inspection** - visualize where signal lives spatially in any direction file

Direction files are tiny (1-3 MB) compared to LoRAs, fully portable across images, and reusable across sessions.

---

## The Concept: Embedding Space Arithmetic

### How Qwen encodes an image

When you call `Conditioning Encode`, Qwen2.5-VL processes your image and prompt together and produces a sequence of embedding vectors - the `prompt_embeds` tensor. After the system prefix is stripped, this sequence has a specific spatial structure:

```
[template start ~3 tokens]  [image vision tokens ~176-196]  [instruction text ~13-25 tokens]
```

The image vision tokens encode what the model *sees*. The text tokens encode what you *want done*.

### What a direction is

A direction vector is simply the difference between two conditionings:

```
direction = V_target - V_baseline
```

For example:
```
V_happy   = Encode(portrait, "make the person smile warmly")
V_neutral = Encode(portrait, "give the person a neutral expression")
direction = V_happy - V_neutral         ←  "smile direction"
```

This direction captures what changes between the two instructions. Applied to any portrait at inference time, it nudges generation toward the same change.

### The relationship to CFG

This is mathematically identical to Classifier-Free Guidance:

```
CFG:       output = V_uncond + scale × (V_cond − V_uncond)
Direction: output = V_base   + scale  × (V_target − V_base)
```

The only difference is CFG uses an empty string as baseline while directions use a semantically meaningful baseline. Directions are CFG with a better-chosen reference point.

### Alpha / scale extrapolation

Because the embedding space is smooth and geometrically consistent, values outside [0, 1] work predictably:

| Value | Meaning |
|---|---|
| 0.0 | No change - pure baseline (use for A/B comparison) |
| 0.5 | Half intensity |
| 1.0 | Full target |
| 1.5 | Amplified beyond target |
| -1.0 | Opposite direction (suppression) |

Both Interpolate (alpha) and DirectionApply (scale) support the range **-5.0 to +5.0** for exploration.

---

## Token Positions and the Spatial Map

### The actual token budget

`max_sequence_length=512` is the total cap including image tokens. With ~195 image tokens, your actual instruction text budget is approximately:

```
512 - 195 (image) - ~10 (chat template overhead) ≈ 307 instruction tokens
```

That's roughly 200-250 words - more than enough for any practical edit instruction.

### Spatial layout of image tokens

Each image vision token corresponds to a **28×28 pixel region** of the VL conditioning image, arranged in a raster grid. The grid dimensions come from:

```python
import math
n_rows = math.ceil(vl_h / 14) // 2    # patch_size=14, merge_size=2
n_cols = math.ceil(vl_w / 14) // 2
```

Token index `i` within the image block maps to:
```
row     = i // n_cols
col     = i % n_cols
pixel_y ≈ [row * 28, (row+1) * 28]  in the VL image
pixel_x ≈ [col * 28, (col+1) * 28]  in the VL image
```

For a typical centred portrait (VL size 320×448, grid 11×16):

| Token rows | Grid position | Approximate face region |
|---|---|---|
| 0-15 (row 0) | Top of frame | Background above head, top of hair |
| 16-31 (row 1) | Second row | Hair, top of head |
| 32-47 (row 2) | Third row | Forehead |
| 48-63 (row 3) | Fourth row | Eyes, upper cheeks |
| 64-79 (row 4) | Fifth row | Nose, mid-cheeks |
| 80-95 (row 5) | Sixth row | Mouth, lower cheeks |
| 96-111 (row 6) | Seventh row | Chin, jaw |
| 112-127 (row 7) | Eighth row | Neck |
| 128-175 (rows 8-10) | Lower rows | Chest, clothing |

### Important: tokens are not isolated patches

By the time tokens reach `prompt_embeds`, they have passed through Qwen2.5-VL's self-attention layers. Every token has attended to every other token. Token #64 (nose region) doesn't encode *only* the nose - it encodes the nose *in the context of the full image*, including the lighting, the expression of the eyes, the background. This is why direction deltas are rarely perfectly localised even for targeted attribute changes.

### Per-token magnitude

Each token's delta magnitude is the L2 norm of its 3584-dimensional vector:

```python
per_token_rms = delta[0].norm(dim=-1)   # shape: [seq_len]
```

This tells you how much each position changed between the two source conditionings. The **Direction Inspect** node visualises this as a spatial heatmap - the fastest way to understand what any direction is actually doing.

---

## Interpolation Methods: Lerp vs Slerp

### Linear interpolation (lerp) - default

```
result = A + t * (B - A)
```

Draws a straight line between two points in embedding space. Fast and simple. At the midpoint (t=0.5) the embedding magnitude is slightly reduced because the straight line cuts through the interior of the embedding "sphere" rather than following its surface. Fine for moderate alpha values (0.3-1.2).

### Spherical linear interpolation (slerp)

```
result = sin((1-t)θ)/sin(θ) × A + sin(tθ)/sin(θ) × B
where θ = angle between A and B
```

Travels along the arc of the embedding sphere's surface, preserving magnitude throughout the interpolation. Produces more perceptually uniform steps when sweeping alpha across a wide range. More graceful at large extrapolation (alpha > 1.5) or strong suppression (alpha < -1.0) where lerp magnitude can grow or collapse unpredictably.

**When to use slerp:** wide alpha sweeps, alpha > 1.5, production quality interpolation where uniformity of steps matters. For typical single-value expression edits (alpha 0.5-1.2), lerp and slerp are nearly identical.

---

## Direction Normalization

### Why averaged directions have reduced magnitude

Averaging N direction vectors:
- **Consistent components** (same semantic axis in all N pairs) reinforce - magnitude preserved
- **Inconsistent components** (image-specific, prompt-wording variation) cancel - magnitude reduced

The result points in a purer direction (correct angle) but at reduced magnitude. With 14 diverse Silvermax pairs, averaged RMS dropped to **28% of individual pair values** - high purity, strong cancellation.

### What normalization does

Normalization rescales the vector magnitude to a `target_rms` value **without changing the direction (angle)**:

```python
normalized_delta = delta * (target_rms / current_rms)
```

Only the length changes. The semantic meaning - which direction the vector points - is preserved exactly.

### Choosing target_rms

| Value | Meaning |
|---|---|
| 1.0 | Unit direction - abstract, always consistent |
| 2.0 | Moderate - good library default |
| 3.5 | Approximate mean of typical individual pair RMS - restores original strength |

Use the same `target_rms` across your entire direction library so that `scale=1.0` has consistent intensity regardless of which direction file you load.

### normalize_direction in DirectionApply

When `normalize_direction=True` on the Apply node, the direction is normalized to `target_rms` before scale is applied. This makes scale semantically consistent across different direction files - particularly useful when chaining multiple directions in compound edits where you want equal weighting.

---

## Text Prompts and Directions: Coarse and Fine Control

**Text prompts and directions serve different roles and work best together.**

### Text prompt = coarse control

The text prompt defines *what kind of edit happens*. "Convert to black and white" is strong enough that the model follows it regardless of what direction is applied.

### Direction = fine control

The direction shapes the *character* of the edit within the space the prompt defined. For a B&W conversion, the direction doesn't determine whether it becomes B&W - it shapes *how*: the tonal curve, the contrast character, the luminosity weighting across colour channels. These are things words can't describe precisely but that are cleanly captured in image token deltas.

### What testing revealed

**Both + and - scale produce B&W when the prompt says B&W.** Testing showed `scale=+1.0` and `scale=-1.0` on a Silvermax direction both produce B&W conversions because the text prompt dominates that decision. The direction modifies the *tonal character*, not whether it's B&W. The Silvermax direction at +scale pushes toward the Silvermax tonal interpretation; at -scale it pushes away from it toward a flatter result.

**Always use a descriptive text prompt.** "Portrait" alone gives the denoiser almost no operational context. "Convert to black and white with strong contrast and rich tonal range" constrains the output space; the direction then shapes the specific character within that constrained space.

**`scale=0` is your baseline.** Zero passes conditioning unchanged - useful for A/B comparison with same image, same prompt, same seed.

### The two-channel model

```
Text prompt:  ████████░░  coarse intent - what kind of edit
Direction:    ░░░░████░░  fine character - how it looks
```

---

## Token Scope: Where the Signal Lives

Different edit types have their signal in different token regions.

### Same-image edits (expression, attributes)

Both Encode nodes receive the **same image**. Image token delta ≈ zero. Signal lives in text tokens:

```
[image tokens: delta ≈ 0] [text tokens: delta = real signal]
```

→ Use `scope = skip_prefix_N`. Skips near-zero image tokens, applies only the text delta.

### Cross-image edits (style, film emulation)

Two Encode nodes receive **different images** (e.g. color photo vs B&W conversion). Signal lives in image tokens:

```
[image tokens: delta = color/style signal] [text tokens: delta ≈ 0 if prompts match]
```

→ Use `scope = keep_prefix_N`. Applies only image token delta, keeps text tokens unchanged.

### Scope cheat sheet

| Scope | Use for | What it does |
|---|---|---|
| `all_tokens` | General use, unsure | Interpolates everything |
| `skip_prefix_N` | Expression, attribute editing (same image) | Keeps image tokens from A, interpolates text tokens |
| `keep_prefix_N` | Style transfer, film emulation (different images) | Interpolates image tokens, keeps text from A |
| `last_N_tokens` | *(not recommended for Qwen)* | Template end tokens sit here, not instructions |

### prefix_N

Boundary between image and text token regions. Auto-calculated from VL image dimensions:

```python
prefix_N = (math.ceil(vl_h / 14) // 2) * (math.ceil(vl_w / 14) // 2)
```

The Encode node prints this every run. Set `prefix_N = 0` in all manipulation nodes to auto-read from conditioning metadata.

| Input aspect | VL size | prefix_N | Grid |
|---|---|---|---|
| Tall portrait (749×1062) | 320×448 | 176 | 11×16 |
| Portrait (1440×1795) | 352×416 | 195 | 15×13 |
| Landscape (1440×960) | 480×320 | 187 | 11×17 |
| Square (1440×1440) | 384×384 | 196 | 14×14 |

---

## LoRA Compatibility

**Direction creation:** LoRAs target the DiT denoiser, not the Qwen2.5-VL text encoder. `prompt_embeds` is identical regardless of which LoRA is loaded. **Directions are fully portable across LoRA configurations.**

**Direction application:** LoRAs significantly affect how the denoiser responds to directions.

| LoRA | Best for |
|---|---|
| PixelSmile | Expression directions - precise, surgically local face changes |
| Base model | Style/film directions - global properties, color, tone, lighting |
| FireRed / other edit LoRAs | Attribute directions - intermediate precision |

---

## Nodes Reference

All nodes appear under **Eric Qwen-Edit/Conditioning** and **Eric Qwen-Edit/Conditioning/Directions**.

---

### Eric Qwen Conditioning Encode

Runs Qwen2.5-VL on an image + prompt, returns `QWEN_CONDITIONING`.

| Input | Notes |
|---|---|
| pipeline | From any loader node |
| image | Resized to ~384px internally for VL encoding |
| prompt | Edit instruction - same as Eric Qwen-Edit Image |
| max_sequence_length | Default 512. Text budget ≈ 307 tokens after image tokens |

**Console output:**
```
[EricQwenEncode] image 749×1062 → VL 320×448
[EricQwenEncode] result: 198/198 valid tokens
[EricQwenEncode] token layout: ~176 image + ~22 text  (prefix_N=176 for skip_prefix_N scope)
```

---

### Eric Qwen Conditioned Edit

Runs diffusion using pre-computed `QWEN_CONDITIONING` instead of a raw text prompt.

| Input | Notes |
|---|---|
| pipeline | Any loader |
| image | VAE path - sets output resolution |
| conditioning | From Encode or any manipulation node |
| negative_conditioning | Optional - enables true CFG when connected |
| steps | 8 for Lightning LoRAs, 50 for base |
| true_cfg_scale | Only active when negative_conditioning is connected |

---

### Eric Qwen Conditioning Interpolate

Linear or spherical interpolation between two conditionings.

```
result = cond_A + alpha × (cond_B - cond_A)
```

| Input | Notes |
|---|---|
| cond_A | Baseline (alpha=0) |
| cond_B | Target (alpha=1) |
| alpha | -5.0 to 5.0 |
| method | `lerp` (default, fast) or `slerp` (preserves magnitude, better for wide sweeps) |
| scope | all_tokens / skip_prefix_N / keep_prefix_N / last_N_tokens |
| prefix_N | 0 = auto from metadata |
| scope_N | Only for last_N_tokens |

---

### Eric Qwen Conditioning Blend

Weighted sum of two conditionings. Equivalent to Interpolate with `alpha = weight_B / (weight_A + weight_B)`. More readable for compound expressions.

```
result = (weight_A × cond_A + weight_B × cond_B) / (weight_A + weight_B)
```

---

### Eric Qwen Direction Compute

```
direction = cond_target - cond_baseline
```

Console also prints `peak_token` position and its RMS - showing where the direction's energy is most concentrated spatially.

---

### Eric Qwen Direction Apply

```
result = base_conditioning + scale × direction   (optionally: scale × normalized_direction)
```

| Input | Notes |
|---|---|
| conditioning | Base conditioning |
| direction | From DirectionCompute or DirectionLoad |
| scale | -5.0 to 5.0. 0=passthrough baseline |
| scope | all_tokens / skip_prefix_N / keep_prefix_N / last_N_tokens |
| prefix_N | 0 = auto |
| normalize_direction | When True, scale direction to target_rms before applying. Makes scale semantically consistent across direction files. |
| target_rms | Target RMS for normalization. Use same value across all Apply nodes in compound edits. |

---

### Eric Qwen Direction Average

Average 2-4 directions into a debiased direction. Cancels image-specific content, leaves shared semantic axis.

| Input | Notes |
|---|---|
| direction_1, 2 | Required |
| direction_3, 4 | Optional |
| normalize | Rescale result to target_rms (recommended for averaged directions) |
| target_rms | 1.0=unit, 2.0=moderate default, 3.5≈original pair strength |

For averaging more than 4, use **Direction Average From Folder**.

---

### Eric Qwen Direction Average From Folder

Average all direction files in a subfolder or matching a prefix. Primary tool for building debiased direction libraries.

| Input | Notes |
|---|---|
| folder | Subfolder dropdown - refreshes automatically |
| prefix | Optional filename filter |
| normalize | True (default) - rescale to target_rms after averaging |
| target_rms | Default 2.0 |
| save_as | Optional filename to save result |
| save_as_bf16 | Half file size |

Console prints each loaded file with its RMS, the mean pair RMS, and the final averaged/normalized RMS. The ratio `averaged_rms / mean_pair_rms` indicates purity - lower is purer.

**Example from 14 Silvermax pairs:**
```
[EricQwenDir]   mean individual rms=3.8320
[EricQwenDir] averaged 14 directions | rms=1.0625
[EricQwenDir] normalized: rms 1.0625 → 2.00 (scale=1.883)
```
1.06 / 3.83 = 28% - high purity.

---

### Eric Qwen Direction Save

Saves to `models/qwen_directions/<filename>.safetensors`. Passthrough - direction returned unchanged.

**Subfolder support:** `silvermax/pair_01` → creates subfolder automatically.

---

### Eric Qwen Direction Load

Loads from `models/qwen_directions/`. Dropdown lists all files recursively including subfolders. Refreshes every run.

---

### Eric Qwen Direction Inspect

**NEW.** Visualizes a direction's per-token delta magnitude as a spatial heatmap and text token bar chart. No GPU required - pure tensor math + PIL rendering.

| Input | Notes |
|---|---|
| direction | Direction to inspect |
| conditioning | Optional - provides exact grid dimensions from vl_size metadata |
| original_image | Optional - overlay heatmap on source image for spatial context |
| colormap | hot / viridis / plasma / coolwarm / gray |
| overlay_alpha | 0=pure heatmap, 1=pure original, 0.35=default blend |
| cell_size | Pixel size of each grid cell (default 32) |
| image_token_offset | 0=auto-detect header token count |

**Output:** ComfyUI IMAGE tensor of the full visualization.

**What to expect by direction type:**

| Direction type | Spatial grid | Text token bars |
|---|---|---|
| Expression (same image, different prompts) | Near-zero / uniform | Bright - all signal here |
| Film emulation (different images, same prompt) | Bright across affected regions | Near-zero |
| Averaged direction (14 pairs) | Diffuse, spread - specific content cancelled | Minimal |
| Single-pair direction | May show image-specific hotspots | Varies |

The visualization includes:
- Spatial heatmap with grid lines every 5 cells and peak cell highlighted in yellow
- Stats panel: signal distribution (image vs text %), top-5 tokens by magnitude with spatial coordinates
- Colorbar strip
- Text token bar chart for the instruction region

**Console output:**
```
[EricQwenInspect] seq=208, grid=15×13, image_tokens=195, offset=3
[EricQwenInspect] Signal: image=87.3% | text=12.7%
[EricQwenInspect] Top 10 token magnitudes:
    #1: token 201  rms=0.43210  (text[6])
    #2: token 198  rms=0.38140  (text[3])
    #3: token  82  rms=0.12300  (image row=6 col=4)
    ...
```

---

## Workflow Patterns

### Pattern 1: Live Expression Interpolation (PixelSmile)

```
Loader → Apply LoRA (PixelSmile)
  ├→ Encode(image, "neutral, relaxed expression")    → cond_A ──┐
  └→ Encode(image, "smile warmly, happy expression") → cond_B ──┤
                                          Interpolate(alpha=0.8, method=lerp, scope=skip_prefix_N)
                                                                  ↓
                                          Conditioned Edit(pipeline, image)
```

### Pattern 2: Film Direction Application

```
Loader (no LoRA needed)
  ↓
Encode(B&W image, "colorize with natural realistic colors") → base_cond
  ↓
Direction Load("silvermax_final") ─→ Direction Apply(scale=1.0, scope=keep_prefix_N)
                                                           ↓
                                     Conditioned Edit(pipeline, B&W image)
```

### Pattern 3: Building a Direction Library

```
For each image pair:
  Color → Encode("photograph") → cond_target  ──┐
  Film  → Encode("photograph") → cond_baseline ──┤→ DirectionCompute → DirectionSave("silvermax/01")

After all pairs:
  DirectionAverageFromFolder(folder="silvermax", normalize=True, target_rms=2.0, save_as="silvermax_final")
```

### Pattern 4: Compound Directions

```
Encode(image, "editorial portrait") → base_cond
  → DirectionApply(lighting_dir, scale=0.5, normalize_direction=True, target_rms=2.0)
  → DirectionApply(color_warm_dir, scale=0.4, normalize_direction=True, target_rms=2.0)
  → DirectionApply(skin_dir, scale=0.3, normalize_direction=True, target_rms=2.0)
  → Conditioned Edit(pipeline, image)
```

### Pattern 5: A/B Baseline Comparison

```
Run A: DirectionApply(base_cond, direction, scale=0.0)   → baseline (no direction)
Run B: DirectionApply(base_cond, direction, scale=1.0)   → with direction
Run C: DirectionApply(base_cond, direction, scale=-1.0)  → opposite direction
```

Same seed across all three runs isolates exactly what the direction contributes.

### Pattern 6: Inspect Before Apply

```
DirectionLoad("silvermax_final")
  ├→ DirectionInspect(conditioning=my_encode_output, original_image=portrait) → heatmap
  └→ DirectionApply(base_cond, scale=1.0, scope=keep_prefix_N)
```

---

## What Works Without a Style LoRA

| Direction type | Scale range | Status |
|---|---|---|
| B&W → colorization | 0.8-1.2 | Proven - use descriptive text prompt |
| Color → film B&W | 0.8-1.2 | Proven - use scale=-1.0 on colorize direction |
| Film emulation (tone/contrast) | 0.5-1.0 | Proven - Silvermax tonal character transfers |
| Color grading (warm/cool) | 0.4-1.0 | Works well |
| Lighting mood | 0.4-0.8 | Works - global property |
| Skin quality | 0.3-0.6 | Subtle, lower scale |
| Background tone | 0.4-0.8 | Works well |

**Requires a LoRA:**
- Expression changes - needs PixelSmile or equivalent
- Precise local geometric changes - model drifts without tight LoRA control

---

## Building a Direction Library: Diversity Guidelines

For maximum cross-image transferability:

- Mix of skin tones (4-5 different ethnicities/complexions)
- Mix of hair colors (blonde, brunette, dark, red - film renders differently per channel)
- Mix of lighting (studio, outdoor, natural, artificial)
- Mix of backgrounds (clean, busy, dark, light)
- Mix of focal distances (tight, half-body, environmental)
- If you want gender-neutral transfer: include both male and female subjects

**Pair count guidelines:**

| Pairs | Quality |
|---|---|
| 1 | Image-specific, use only for that exact image |
| 3 | Major improvement, most obvious content cancels |
| 5 | Good - sufficient for most applications |
| 8-10 | Excellent - well-generalised |
| 14+ | Maximum - diminishing returns beyond this |

With 14 diverse pairs, averaged RMS typically drops to 25-30% of individual pair values.

---

## Training a Style LoRA

The direction system works today without a style LoRA but the response is loose - the model understands the direction but may drift composition or intensity at higher scale values. A LoRA trained specifically for a style axis would give the same precision as PixelSmile gives for expressions.

### The training script

`training/train_style_lora.py` implements symmetric LoRA training for style directions.

**Architecture:** based on PixelSmile's symmetric joint training approach, simplified for style (no ArcFace identity loss, no CLIP contrastive loss - those are expression-specific):

```
L_total = L_forward + L_backward + λ_sym × L_symmetry

L_forward  = flow_matching_loss(model(color, forward_prompt), bw_target)
L_backward = flow_matching_loss(model(bw, backward_prompt), color_target)
L_symmetry = magnitude(L_forward - L_backward)²   # round-trip consistency
```

**Why symmetric training helps:** standard LoRA training optimizes prompt following but doesn't teach the model to traverse the conditioning-space axis between two conditionings smoothly. The symmetry constraint forces the model to learn a clean, reversible axis. Round-trip consistency (A→B→A returns to A) eliminates the one-directional drift that standard training produces.

**Usage:**
```bash
cd training/

# Generate a config template first:
python train_style_lora.py --generate_config configs/silvermax.yaml

# Edit the config, then train:
python train_style_lora.py --config configs/silvermax.yaml

# Or inline:
python train_style_lora.py \
    --data_dir ./training_data/silvermax \
    --forward_prompt "convert to Silvermax film black and white, high contrast" \
    --backward_prompt "colorize with natural realistic colors" \
    --lora_rank 16 \
    --steps 2000
```

**Requirements:**
```bash
pip install diffusers>=0.31 peft>=0.13 accelerate transformers pillow torch pyyaml
```

**Dataset format:**
```
training_data/silvermax/
    pair_001_a.jpg    ← color source
    pair_001_b.jpg    ← Silvermax B&W conversion
    pair_002_a.jpg
    pair_002_b.jpg
    ...
```

Files named `*_a.*` and `*_b.*` are automatically paired. Alternative: `*_color.*` and `*_bw.*` or `*_silver.*`. Falls back to consecutive alphabetical pairing for any even-numbered image set.

**LoRA target modules** (from DiffSynth-Studio, Qwen-Image-Edit-2511 specific):
```
to_q, to_k, to_v, add_q_proj, add_k_proj, add_v_proj,
to_out.0, to_add_out, img_mlp.net.2, img_mod.1, txt_mlp.net.2, txt_mod.1
```

**Important:** the `compute_flow_matching_loss` function in the training script contains a note about verifying the exact `pipeline.transformer.forward()` call signature against your installed diffusers version. Check `pixelsmile/train.py` at `github.com/Ammmob/PixelSmile` for the authoritative Qwen-Edit-2511 training forward pass implementation before running.

**LoRA rank guidance:**
- Style/film emulation: rank 16-32 (global changes, higher-dimensional)
- Expression: rank 4-8 (local face changes, lower-dimensional)

**Difference from PixelSmile training:**
- No ArcFace identity loss (style doesn't require face identity preservation)
- No CLIP contrastive loss (no need to disentangle similar styles)
- Simpler dataset: your 14 image pairs are perfect training data as-is
- PixelSmile's full training uses a 60K-image FFE dataset; 14-50 diverse pairs is sufficient for a single style axis

---

## File Organisation

```
ComfyUI/models/qwen_directions/
├── silvermax_final.safetensors       ← averaged + normalized, ready-to-use
├── colorize_portrait.safetensors     ← single-pair, image-specific
├── silvermax/                        ← subfolder of source pairs
│   ├── 01.safetensors
│   ├── 02.safetensors
│   └── ...
└── expressions/
    ├── smile_happy.safetensors
    └── smile_subtle.safetensors
```

**Naming tips:**
- Use zero-padded numbers (`01`, `02` not `1`, `2`) for correct alphabetical sort order
- DirectionSave accepts `/` separators and creates subfolders automatically
- DirectionLoad and DirectionAverageFromFolder traverse subfolders recursively

---

## Technical Reference

### QWEN_CONDITIONING dict schema

```python
{
    "prompt_embeds":      Tensor[1, seq_len, 3584],  # bfloat16, CPU
    "prompt_embeds_mask": Tensor[1, seq_len],         # 1=valid, 0=padding
    "metadata": {
        "source_prompt":  str,    # first 120 chars of prompt
        "valid_tokens":   int,    # non-padding token count
        "total_tokens":   int,    # padded sequence length
        "image_size":     str,    # original image WxH
        "vl_size":        str,    # VL conditioning image WxH (~384px)
        "image_tokens":   int,    # visual token count - use as prefix_N
        "text_tokens":    int,    # instruction + template token count
    }
}
```

### QWEN_DIRECTION dict schema

```python
{
    "delta_embeds": Tensor[1, seq_len, 3584],  # V_target - V_baseline, bfloat16
    "delta_mask":   Tensor[1, seq_len],         # union of source masks
    "metadata": {
        "target_prompt":   str,
        "baseline_prompt": str,
        "seq_len":         int,
        "valid_tokens":    int,
        "rms":             float,   # overall RMS magnitude
        "raw_rms":         float,   # RMS before normalization (if normalized)
        "n_averaged":      int,     # number of pairs averaged
        "normalized":      bool,    # whether normalization was applied
        "target_rms":      float,   # normalization target (if normalized)
        "peak_token":      int,     # token position with highest magnitude
    }
}
```

### Spatial grid formula

```python
import math
n_rows = math.ceil(vl_h / 14) // 2    # patch_size=14, merge_size=2
n_cols = math.ceil(vl_w / 14) // 2
prefix_N = n_rows * n_cols

# Token i in image block → spatial position:
row = i // n_cols
col = i % n_cols
```

### Slerp formula reference

```python
def slerp(a, b, t):
    """Spherical interpolation for embedding tensors [batch, seq, hidden]."""
    a_mag = a.norm(dim=-1, keepdim=True)
    b_mag = b.norm(dim=-1, keepdim=True)
    a_unit = F.normalize(a, dim=-1)
    b_unit = F.normalize(b, dim=-1)
    dot = (a_unit * b_unit).sum(dim=-1, keepdim=True).clamp(-1+1e-7, 1-1e-7)
    theta = torch.acos(dot)
    sin_theta = torch.sin(theta)
    parallel = sin_theta.abs() < 1e-6
    coeff_a = torch.where(parallel, 1.0 - t, torch.sin((1-t)*theta) / sin_theta)
    coeff_b = torch.where(parallel, t,       torch.sin(t*theta)     / sin_theta)
    direction = F.normalize(coeff_a * a_unit + coeff_b * b_unit, dim=-1)
    magnitude = (1-t) * a_mag + t * b_mag
    return direction * magnitude
```

---

*Screenshots of example workflows to be added.*
