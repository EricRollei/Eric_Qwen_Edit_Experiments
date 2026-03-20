# Developer Notes — Guidance Embedding & Future Model Support

These notes capture research findings from building the Eric Qwen-Edit &
Qwen-Image node set.  They're intended as a reference when adapting nodes
for other models (Flux, Z-Image Turbo, future Qwen variants, etc.).

---

## Guidance Embedding vs. True CFG

There are **two separate guidance mechanisms** in flow-matching / diffusion
models.  They are often confused because diffusers uses the same parameter
name (`guidance_scale`) for both, depending on the model.

### 1. Guidance Embedding (guidance-distilled models)

- A scalar value fed into the transformer as an **input embedding**.
- The timestep embedding module accepts `(timestep, guidance, hidden_states)`
  and conditions the forward pass on the guidance value.
- Only **one** forward pass per denoising step — the guidance is "baked in"
  as a conditioning signal.
- The model must be **trained** (distilled) with this embedding to make it
  work.  You can't just add it after the fact.
- Config flag:  `transformer.config.guidance_embeds = True`
- Pipeline behavior: creates `guidance = torch.full([1], guidance_scale)`
  and passes it to `transformer.forward(guidance=guidance)`.

**Models that use this:**
- **Flux.1-dev** — Flux was guidance-distilled; its transformer has a
  `CombinedTimestepGuidanceEmbeddings` layer that accepts guidance.
- **Stable Diffusion 3.5 Medium** — also guidance-distilled.
- Future Qwen variants (if Alibaba releases a distilled version).

### 2. True Classifier-Free Guidance (CFG)

- Standard CFG: run the transformer **twice** per step (once with the
  prompt, once with no prompt / negative prompt), then blend the outputs:
    `output = uncond + cfg_scale * (cond - uncond)`
- Pipeline parameter: `true_cfg_scale`
- 2× the compute cost per step.
- Works with **any** model, no training changes needed.
- Norm-preserving variant (used by QwenImagePipeline): after blending,
  rescale the result to match the conditional output's norm.  This prevents
  over-saturation at high CFG values.

**Models that use this:**
- **Qwen-Image-2512** (true_cfg_scale, called s1_cfg/s2_cfg/s3_cfg in our nodes)
- Virtually any diffusion model supports standard CFG.

---

## Qwen-Image-2512 Specifics

### Why guidance_scale doesn't work

Investigated March 2026.  Findings:

1. `transformer.config.guidance_embeds = False` — set by the model authors.
2. `QwenTimestepProjEmbeddings.forward(self, timestep, hidden_states)` only
   takes 2 positional args.  The guidance path in the transformer's forward
   method tries to call `self.time_text_embed(timestep, guidance, hidden_states)`
   with 3 args → would crash with TypeError.
3. The diffusers source has `# TODO: this should probably be removed` on both
   the `guidance_embeds` init param and the `guidance` forward param.
4. The pipeline prints "*guidance_scale is passed as X, but ignored since
   the model is not guidance-distilled*" when `guidance_embeds=False`.

**Conclusion:**  Qwen-Image-2512 was not guidance-distilled.  The guidance
embedding path is dead code inherited from the Flux architecture template.
Real guidance control is exclusively through `true_cfg_scale`.

### Official Qwen-Image Recommendations
- 50 steps, true_cfg_scale = 4.0
- Native resolution: ~1.76 MP (1328×1328, 1664×928, etc.)
- Chinese negative prompt for quality
- Prompt enhancement via LLM rewriting (200+ word prompts)

---

## Adapting for Flux / Other Guidance-Distilled Models

If building an UltraGen-style node for Flux or similar:

1. **Re-add `guidance_scale` as a node input** (was removed from Qwen UltraGen
   since it's non-functional there).
2. The pipeline will check `transformer.config.guidance_embeds` and
   automatically create the guidance tensor.
3. For Flux specifically:
   - `guidance_scale` controls prompt adherence (typical: 3.5-7.0)
   - Flux doesn't use `true_cfg_scale` / negative prompts in its standard
     pipeline — guidance is embedded, not CFG.
   - Flux's `FluxPipeline` has different internal structure (no VAE
     scale factor of 8, different latent packing, etc.)
4. The Spectrum acceleration mechanism (`patch_transformer_spectrum`) should
   work on any transformer-based diffusion model — it's architecture-agnostic,
   operating on the transformer's forward pass.

---

## Z-Image Turbo / Distilled Models

Distilled (few-step) models typically:
- Don't benefit from high step counts (4-8 steps is optimal)
- Don't benefit from Spectrum (too few steps to cache)
- May or may not use guidance embedding (check `guidance_embeds` config)
- Often require `true_cfg_scale=1.0` (CFG baked into distillation)

When adapting, check:
```python
print(pipe.transformer.config.guidance_embeds)      # True = has guidance embedding
print(pipe.transformer.config)                        # Full config dump
print(type(pipe.transformer.time_text_embed))         # Check embedding class
import inspect
print(inspect.signature(pipe.transformer.time_text_embed.forward))  # Does it accept guidance?
```

---

## Sigma Schedule Curves for Multi-Stage Refinement

Updated March 2026.  `build_sigma_schedule()` in `eric_qwen_image_multistage.py`
provides three sigma schedule curves for flow-matching denoise.

### Background

The sigma schedule defines the **spacing** of noise levels the sampler walks
through during denoising.  Not all noise levels contribute equally:

| Sigma range | What happens              |
|------------|---------------------------|
| 0.5 – 1.0 | Global composition, large shapes |
| 0.15 – 0.5 | Object details, textures   |
| 0.0 – 0.15 | Fine detail, micro-textures, sharpness |

A **linear** schedule spreads compute equally across all levels.  For
refinement stages (S2, S3) where composition is already locked, this wastes
steps on the high-sigma range that should be rushed through.

### Critical Design: Consistent Starting Sigma

All schedules **start from the same sigma** for a given denoise value.  The
starting sigma is computed from the linear schedule's truncation point:

```python
full_linear = np.linspace(1.0, sigma_min, num_steps)
sigma_start = full_linear[num_steps - keep]   # keep = round(N × denoise)
```

The schedule curve then distributes `keep` steps from `sigma_start` → `sigma_min`.
This ensures the noise level is consistent regardless of curve shape.

> **Bug history (March 2026):** The original implementation built a full schedule
> from σ=1.0 and truncated.  For non-linear schedules, truncation produced wildly
> different starting sigmas: cosine started at 0.977 (near pure noise, causing
> echoes/ghosting) while karras started at 0.683 (too low, insufficient detail).
> Linear was unaffected.  Fixed by computing sigma_start independently, then
> distributing steps within the correct range.

### Available Schedules

**Linear** — `np.linspace(sigma_start, sigma_min, keep)`
- Uniform spacing.  Safe default.  Equal compute at every noise level.

**Balanced** — Karras-style with ρ = 3
- Moderate concentration at mid-to-low sigma.
- Reduces compute at high sigma (composition) while balancing detail + texture.
- Recommended for **Stage 2** — preserves composition, adds mid-level and
  fine detail with good coverage across both ranges.

**Karras** — EDM-optimal (Karras et al. NeurIPS 2022) with ρ = 7
- `(sigma_start^(1/ρ) + t × (sigma_min^(1/ρ) − sigma_start^(1/ρ)))^ρ`
- Heavily concentrates steps at **low sigma** (fine detail/sharpness).
- Large jumps through high sigma → rushes past composition.
- Recommended for **Stage 3** where micro-texture and sharpening dominate.

### Step Budget Distribution

Measured at S2 defaults (30 steps, denoise = 0.85, keep = 26):

| Schedule | HIGH σ (composition) | MID σ (detail) | LOW σ (texture) |
|----------|---------------------|----------------|-----------------|
| Linear   | 50%                 | 38%            | 11%             |
| Balanced | 30%                 | 38%            | 30%             |
| Karras   | 26%                 | 34%            | 38%             |

### Interaction with Denoise

When `denoise < 1.0`, the schedule covers only the portion from `sigma_start`
to `sigma_min`, with `keep = round(num_steps × denoise)` steps.  The curve
determines how those steps are distributed within that fixed range:

- **Linear + denoise=0.85**: 26 steps, uniform from σ=0.87 to ~0.03
- **Balanced + denoise=0.85**: 26 steps, moderately packed toward lower σ
- **Karras + denoise=0.85**: 26 steps, heavily packed near σ=0.03–0.10

### Practical Recommendations

| Stage | Schedule | Why |
|-------|----------|-----|
| S1 (txt2img from noise) | Linear | Full denoise, all sigma ranges matter equally |
| S2 (main refinement) | Balanced | 30/38/30 split — composition preserved, good detail + texture |
| S3 (final polish) | Karras | Heavy low-σ focus — maximum sharpening, fine micro-texture |

Linear remains the safest default for experimentation.  Switch to
balanced/karras once you have a composition you like from S1.

---

## Wan2.1 / Qwen 2× Upscale VAE Integration

### Background

The `spacepxl/Wan2.1-VAE-upscale2x` model is a decoder-only finetune of
the Wan2.1 VAE architecture.  Wan2.1 and Qwen-Image share an identical
latent space (z_dim=16, same normalization scheme) and architecturally
identical VAE encoders/decoders.  This was confirmed by the model author,
cross-model testing in the community, and code analysis.

### How it works

The upscale VAE's decoder outputs **12 channels** instead of 3.
After decode, `F.pixel_shuffle(decoded, 2)` rearranges the 12 channels
into 3 channels at 2× spatial resolution — a free 2× super-resolution
step performed entirely in VAE decode space.

### Decode path

```
packed_latents [B, seq, C*4]   (from pipe output_type="latent")
    → _unpack_latents()        → [B, 16, 1, H/8, W/8]
    → latent normalization     → latents / latents_std + latents_mean
    → upscale_vae.decode()     → [B, 12, 1, H/8, W/8]
    → squeeze(2)               → [B, 12, H/8, W/8]
    → pixel_shuffle(2)         → [B, 3, H/4, W/4]   (2× resolution)
    → normalize [-1,1]→[0,1]
    → permute to [B, H, W, C]  (ComfyUI IMAGE format)
```

### UltraGen integration

The `upscale_vae` optional input plus `upscale_vae_mode` dropdown control
how the upscale VAE is used:

| Mode | Behaviour |
|------|-----------|
| `disabled` | Upscale VAE ignored even if connected (default — safe) |
| `inter_stage` | Decode S2 latents at 2× with upscale VAE, re-encode with standard Qwen VAE, feed 2× latents to S3.  Replaces bislerp upscale between S2→S3.  Requires 3 stages. |
| `final_decode` | Replace the final stage's normal VAE decode with 2× upscale decode.  Works with any stage count (1, 2, or 3). |
| `both` | Inter-stage S2→S3 AND 2× final decode.  S3 operates on a 2× canvas from inter-stage, then the output image is another 2× from final decode → effectively **4× total** vs S2 resolution. |

When `upscale_vae_mode` is `disabled` or no VAE is connected, UltraGen
behaves exactly as before — no code paths are altered.

**Inter-stage flow (S2→S3):**
```
S2 packed latents [B, seq, C*4]
  → unpack → denormalize → upscale_vae.decode() → [B, 12, 1, H, W]
  → squeeze → pixel_shuffle(2) → [B, 3, 2H, 2W] pixels
  → pipe_vae.encode() → posterior.mode() → raw latents
  → normalize: (raw - mean) * std → packed latents at 2× resolution
  → feed to S3 as starting latents (with denoise noise added)
```

**Final decode flow:**
```
Final stage packed latents (output_type="latent")
  → same as decode_latents_with_upscale_vae()
  → output image at 2× the final stage resolution
```

### Nodes

| Node | Purpose |
|------|---------|
| **Eric Qwen Upscale VAE Loader** | Loads the Wan2.1 upscale VAE |
| UltraGen `upscale_vae` input | Connects loader → UltraGen for 2× decode |

### Model source

- HuggingFace: `spacepxl/Wan2.1-VAE-upscale2x`
- Subfolder: `diffusers/Wan2.1_VAE_upscale2x_imageonly_real_v1`
- Class: `diffusers.AutoencoderKLWan`
- Size: ~200 MB

---

*Last updated: March 19, 2026 — Eric Hiss*
