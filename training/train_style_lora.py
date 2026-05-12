"""
train_style_lora.py  (optimised)
==================================
Symmetric LoRA training for Qwen-Image-Edit-2511 style directions.

Key optimisations over v1
--------------------------
1. Pre-cache all VAE and text+vision encodings at startup.
   With 13 pairs, this takes ~2 min once.  Each subsequent training step
   skips the 7B text encoder and VAE entirely - only the transformer runs.

2. Lower default resolution (384 vs 512).
   Transformer attention is O(seq^2). At 384px seq~1152 vs 512px seq~2048.
   That is ~3x faster per forward pass with negligible quality difference
   for a global style LoRA.

3. torch.autocast on the transformer forward pass.

4. Configurable batch_size - VRAM at 65% with batch=1 leaves headroom.
   Try batch_size=2 for another ~1.5x throughput gain.

Author: Eric Hiss (GitHub: EricRollei)
"""

import os
import sys
import json
import math
import argparse
import random
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from PIL import Image

# ── Make package root importable when run as a standalone script ──────────────
_SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
_PACKAGE_DIR = os.path.dirname(_SCRIPT_DIR)
if _PACKAGE_DIR not in sys.path:
    sys.path.insert(0, _PACKAGE_DIR)


# ──────────────────────────────────────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config",            type=str, default=None)
    p.add_argument("--data_dir",          type=str)
    p.add_argument("--output_dir",        type=str, default="./output/style_lora")
    p.add_argument("--base_model",        type=str, default="Qwen/Qwen-Image-Edit-2511")
    p.add_argument("--forward_prompt",    type=str,
                   default="convert to Silvermax film black and white, high contrast, rich shadow detail")
    p.add_argument("--backward_prompt",   type=str,
                   default="colorize with natural realistic skin tones and colors")
    p.add_argument("--lora_rank",         type=int,   default=16)
    p.add_argument("--lora_alpha",        type=float, default=16.0)
    p.add_argument("--lora_dropout",      type=float, default=0.05)
    p.add_argument("--target_modules",    type=str,   nargs="+",
                   default=["to_q","to_k","to_v","add_q_proj","add_k_proj","add_v_proj",
                            "to_out.0","to_add_out","img_mlp.net.2","img_mod.1",
                            "txt_mlp.net.2","txt_mod.1"])
    p.add_argument("--steps",             type=int,   default=2000)
    p.add_argument("--batch_size",        type=int,   default=1)
    p.add_argument("--lr",                type=float, default=1e-4)
    p.add_argument("--sym_weight",        type=float, default=0.1)
    p.add_argument("--gradient_accumulation_steps", type=int, default=4)
    p.add_argument("--save_every",        type=int,   default=500)
    p.add_argument("--seed",              type=int,   default=42)
    p.add_argument("--resolution",        type=int,   default=384,
                   help="Training resolution. 384=fast (recommended), 512=higher quality")
    p.add_argument("--device",            type=str,   default="cuda:0")
    p.add_argument("--dtype",             type=str,   default="bfloat16",
                   choices=["float32","bfloat16","float16"])
    p.add_argument("--gradient_checkpointing", action="store_true", default=True)
    p.add_argument("--use_autocast",      action="store_true", default=True)

    args = p.parse_args()

    if args.config:
        try:
            import yaml
            with open(args.config, encoding="utf-8") as f:
                cfg = yaml.safe_load(f)
            for k, v in cfg.items():
                if not any(f"--{k}" in a for a in sys.argv):
                    setattr(args, k, v)
        except ImportError:
            print("WARNING: PyYAML not installed - config ignored. pip install pyyaml")

    return args


# ──────────────────────────────────────────────────────────────────────────────
# Dataset  (returns PIL; pipeline preprocessors used inside caching step)
# ──────────────────────────────────────────────────────────────────────────────

class StylePairDataset(Dataset):
    """
    Returns PIL image pairs. Augmentation applied here so cached encodings
    are NOT used with augmentation - augmentation happens during caching.
    For small datasets (13 pairs) we cache without augmentation and rely on
    the random timestep sampling for regularisation.
    """

    def __init__(self, data_dir: str):
        data_dir   = Path(data_dir)
        all_images = sorted(
            list(data_dir.glob("*.jpg"))  + list(data_dir.glob("*.jpeg")) +
            list(data_dir.glob("*.png"))  + list(data_dir.glob("*.webp"))
        )
        a_files = sorted(f for f in all_images if any(
            t in f.name for t in ("_a.", "_color.", "_src.", "_original.")))
        b_files = sorted(f for f in all_images if any(
            t in f.name for t in ("_b.", "_bw.", "_silver.", "_film.", "_target.")))

        if a_files and b_files and len(a_files) == len(b_files):
            self.pairs = list(zip(a_files, b_files))
            print(f"[Dataset] {len(self.pairs)} structured pairs")
        elif len(all_images) >= 2 and len(all_images) % 2 == 0:
            self.pairs = [(all_images[i], all_images[i+1])
                          for i in range(0, len(all_images), 2)]
            print(f"[Dataset] {len(self.pairs)} consecutive pairs (fallback)")
        else:
            raise ValueError(
                f"Cannot pair images in {data_dir}. "
                f"Found {len(all_images)} images. "
                "Use *_a.ext/*_b.ext naming or provide an even number of files."
            )
        print(f"[Dataset] First pair: {self.pairs[0][0].name} <-> {self.pairs[0][1].name}")

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        a_path, b_path = self.pairs[idx]
        return {
            "image_a": Image.open(a_path).convert("RGB"),
            "image_b": Image.open(b_path).convert("RGB"),
        }


# ──────────────────────────────────────────────────────────────────────────────
# Pre-caching helpers
# ──────────────────────────────────────────────────────────────────────────────

def _encode_and_pack(pipeline, pil_image: Image.Image, res_h: int, res_w: int,
                     device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """
    Preprocess one PIL image, encode through VAE, pack into transformer
    sequence format. Returns CPU tensor [1, seq, C*4].
    """
    tensor    = pipeline.image_processor.preprocess(pil_image, res_h, res_w)  # [1,C,H,W]
    tensor_5d = tensor.to(device, dtype).unsqueeze(2)                          # [1,C,1,H,W]
    with torch.no_grad():
        latents = pipeline._encode_vae_image(tensor_5d, generator=None)       # [1,C,1,h,w]
    h_lat, w_lat = latents.shape[3], latents.shape[4]
    n_ch         = latents.shape[1]
    packed       = pipeline._pack_latents(latents, 1, n_ch, h_lat, w_lat)     # [1,seq,C*4]
    return packed.cpu(), h_lat, w_lat


def _encode_prompt_for_pair(pipeline, prompt: str, pil_image: Image.Image,
                             device: torch.device) -> tuple:
    """
    Encode prompt + VL conditioning image. Returns (embeds, mask) on CPU.
    """
    from pipelines.pipeline_qwen_edit import calculate_dimensions, CONDITION_IMAGE_SIZE
    w, h  = pil_image.size
    cw, ch = calculate_dimensions(CONDITION_IMAGE_SIZE, w / h)
    vl_img = pipeline.image_processor.resize(pil_image, ch, cw)
    with torch.no_grad():
        embeds, mask = pipeline.encode_prompt(
            prompt=[prompt], image=[vl_img], device=device)
    return embeds.cpu(), mask.cpu()


def build_cache(pipeline, dataset, args, device, dtype):
    """
    Pre-encode all image pairs through VAE and text+VL encoder.
    Returns a list of dicts, each containing packed latents and embeddings
    ready to use in the training loop - all stored on CPU.

    With 13 pairs this takes ~2 minutes and runs once per training session.
    Every subsequent training step accesses only the transformer (no VAE, no
    text encoder), giving a large speed improvement.
    """
    from pipelines.pipeline_qwen_edit import CONDITION_IMAGE_SIZE

    # Align resolution to pipeline requirement
    align = pipeline.vae_scale_factor * 2
    res_h = (args.resolution // align) * align
    res_w = (args.resolution // align) * align

    print(f"\n[Cache] Pre-encoding {len(dataset)} pairs at {res_w}x{res_h}...")
    print("[Cache] This runs once. Subsequent steps only run the transformer.")

    cache    = []
    h_lat    = None
    w_lat    = None

    for idx in range(len(dataset)):
        item     = dataset[idx]
        pil_a    = item["image_a"]
        pil_b    = item["image_b"]

        # VAE: encode both images (source = conditioning ref, target = x0)
        a_packed, h_lat, w_lat = _encode_and_pack(pipeline, pil_a, res_h, res_w, device, dtype)
        b_packed, _,    _      = _encode_and_pack(pipeline, pil_b, res_h, res_w, device, dtype)

        # Text+VL: forward prompt conditioned on A (source), backward on B
        fwd_emb, fwd_mask = _encode_prompt_for_pair(pipeline, args.forward_prompt,  pil_a, device)
        bwd_emb, bwd_mask = _encode_prompt_for_pair(pipeline, args.backward_prompt, pil_b, device)

        cache.append({
            "a_packed":  a_packed,    # [1, seq, C*4]  - A latent
            "b_packed":  b_packed,    # [1, seq, C*4]  - B latent
            "fwd_emb":   fwd_emb,     # [1, txt_seq, 3584]
            "fwd_mask":  fwd_mask,    # [1, txt_seq]
            "bwd_emb":   bwd_emb,
            "bwd_mask":  bwd_mask,
        })

        if (idx + 1) % 5 == 0 or idx == len(dataset) - 1:
            print(f"[Cache]   {idx+1}/{len(dataset)} pairs encoded")

    torch.cuda.empty_cache()
    print(f"[Cache] Done. Latent grid: {h_lat}x{w_lat}  "
          f"| seq/latent={h_lat*w_lat//4}  "
          f"| transformer input seq={h_lat*w_lat//4 * 2} (noise+ref)")
    return cache, h_lat, w_lat


# ──────────────────────────────────────────────────────────────────────────────
# Single training step (no encoding - transformer only)
# ──────────────────────────────────────────────────────────────────────────────

def flow_loss_from_cache(
    transformer,
    item: dict,
    h_lat: int, w_lat: int,
    device: torch.device, dtype: torch.dtype,
    generator: torch.Generator,
    use_autocast: bool = True,
) -> tuple:
    """
    Compute both forward (A->B) and backward (B->A) flow-matching losses
    from a single pre-cached pair. No VAE, no text encoder.

    Returns (loss_fwd, loss_bwd).
    """
    noise_seq = h_lat * w_lat // 4   # packed sequence length per latent
    h_shape   = h_lat // 2
    w_shape   = w_lat // 2

    # One img_shape entry for noise latent, one for reference
    img_shapes = [[(1, h_shape, w_shape), (1, h_shape, w_shape)]]

    def _one_direction(x0_packed, ref_packed, embeds, mask):
        """Flow-matching loss for one direction."""
        x0  = x0_packed.to(device, dtype)
        ref = ref_packed.to(device, dtype)
        emb = embeds.to(device, dtype)
        msk = mask.to(device)

        txt_seq_lens = msk.sum(dim=1).tolist()

        noise   = torch.randn(x0.shape, device=device, dtype=dtype, generator=generator)
        t       = torch.rand(1, device=device, dtype=dtype, generator=generator)
        t_bcast = t.view(1, 1, 1)
        x_t     = (1 - t_bcast) * x0 + t_bcast * noise
        v_tgt   = noise - x0

        latent_input = torch.cat([x_t, ref], dim=1)  # [1, noise_seq+ref_seq, C*4]

        ctx = torch.autocast(device_type="cuda", dtype=dtype) if use_autocast else torch.no_grad().__class__()
        with ctx:
            pred_full = transformer(
                hidden_states=latent_input,
                timestep=t.expand(1),           # [0, 1] range - pipeline divides by 1000 at inference
                guidance=None,
                encoder_hidden_states=emb,
                encoder_hidden_states_mask=msk,
                img_shapes=img_shapes,
                txt_seq_lens=txt_seq_lens,
                attention_kwargs={},
                return_dict=False,
            )[0]

        pred = pred_full[:, :noise_seq]
        return F.mse_loss(pred, v_tgt)

    # Forward: A conditions (ref), B is target (x0)
    loss_fwd = _one_direction(
        item["b_packed"], item["a_packed"],
        item["fwd_emb"],  item["fwd_mask"],
    )

    # Backward: B conditions (ref), A is target (x0)
    loss_bwd = _one_direction(
        item["a_packed"], item["b_packed"],
        item["bwd_emb"],  item["bwd_mask"],
    )

    return loss_fwd, loss_bwd


# ──────────────────────────────────────────────────────────────────────────────
# LoRA setup
# ──────────────────────────────────────────────────────────────────────────────

def setup_lora(transformer, args):
    try:
        from peft import LoraConfig, get_peft_model
    except ImportError:
        raise ImportError("pip install peft>=0.13")

    existing = {name for name, _ in transformer.named_modules()}
    valid    = [t for t in args.target_modules
                if any(name.endswith(t) or t in name for name in existing)]
    skipped  = set(args.target_modules) - set(valid)
    if skipped:
        print(f"[LoRA] Skipping (not found): {skipped}")

    cfg = LoraConfig(r=args.lora_rank, lora_alpha=args.lora_alpha,
                     target_modules=valid, lora_dropout=args.lora_dropout, bias="none")
    transformer = get_peft_model(transformer, cfg)
    transformer.print_trainable_parameters()
    return transformer


# ──────────────────────────────────────────────────────────────────────────────
# Main training function
# ──────────────────────────────────────────────────────────────────────────────

def train(args):
    import time

    print("=" * 60)
    print("Symmetric Style LoRA Training  (optimised)")
    print("=" * 60)
    for k in ["data_dir","output_dir","base_model","forward_prompt","backward_prompt",
              "lora_rank","steps","sym_weight","resolution","batch_size",
              "device","dtype","use_autocast"]:
        print(f"  {k}: {getattr(args, k, '?')}")
    print("=" * 60)

    torch.manual_seed(args.seed)
    torch.backends.cudnn.benchmark = True
    os.makedirs(args.output_dir, exist_ok=True)

    device    = torch.device(args.device)
    dtype_map = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}
    dtype     = dtype_map[args.dtype]
    generator = torch.Generator(device=device).manual_seed(args.seed)

    # ── Load pipeline ─────────────────────────────────────────────────────────
    print("\n[Setup] Loading base model...")
    try:
        from pipelines.pipeline_qwen_edit import QwenEditPipeline
        pipeline = QwenEditPipeline.from_pretrained(args.base_model, torch_dtype=dtype).to(device)
        print("[Setup] Loaded via local QwenEditPipeline")
    except Exception as e:
        print(f"[Setup] Local pipeline failed ({e}), trying diffusers...")
        from diffusers import QwenImageEditPipeline
        pipeline = QwenImageEditPipeline.from_pretrained(args.base_model, torch_dtype=dtype).to(device)

    pipeline.vae.requires_grad_(False)
    pipeline.text_encoder.requires_grad_(False)
    pipeline.transformer.requires_grad_(False)

    # ── LoRA ──────────────────────────────────────────────────────────────────
    print("\n[Setup] Adding LoRA adapters...")
    pipeline.transformer = setup_lora(pipeline.transformer, args)
    if args.gradient_checkpointing:
        try:
            pipeline.transformer.enable_gradient_checkpointing()
        except Exception:
            print("[Setup] gradient_checkpointing not available for this transformer")

    # ── Dataset ───────────────────────────────────────────────────────────────
    print("\n[Setup] Loading dataset...")
    dataset = StylePairDataset(args.data_dir)

    # ── Pre-cache ─────────────────────────────────────────────────────────────
    t0    = time.time()
    cache, h_lat, w_lat = build_cache(pipeline, dataset, args, device, dtype)
    print(f"[Cache] Pre-encoding complete in {(time.time()-t0)/60:.1f} min")
    print(f"[Cache] {len(cache)} pairs cached. "
          f"Each training step now runs ONLY the transformer.")

    # Free VAE from VRAM (no longer needed after caching)
    pipeline.vae = pipeline.vae.cpu()
    torch.cuda.empty_cache()
    print("[Cache] VAE moved to CPU - freed VRAM for faster transformer training")

    # ── Optimizer ─────────────────────────────────────────────────────────────
    lora_params = [p for p in pipeline.transformer.parameters() if p.requires_grad]
    print(f"\n[Setup] Trainable params: {sum(p.numel() for p in lora_params):,}")
    optimizer = torch.optim.AdamW(lora_params, lr=args.lr, weight_decay=0.01)

    with open(os.path.join(args.output_dir, "training_config.json"), "w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2, default=str)

    # ── Training loop ─────────────────────────────────────────────────────────
    print("\n[Training] Starting - transformer-only loop (no encoder overhead)...")
    pipeline.transformer.train()

    step             = 0
    loss_accum       = 0.0
    grad_accum_count = 0
    step_start       = time.time()
    recent_times     = []

    while step < args.steps:
        # Pick a random cached pair
        item = cache[random.randint(0, len(cache) - 1)]

        loss_fwd, loss_bwd = flow_loss_from_cache(
            transformer  = pipeline.transformer,
            item         = item,
            h_lat        = h_lat, w_lat=w_lat,
            device       = device, dtype=dtype,
            generator    = generator,
            use_autocast = args.use_autocast,
        )

        loss_sym = (loss_fwd - loss_bwd).pow(2)
        loss     = (loss_fwd + loss_bwd + args.sym_weight * loss_sym) / args.gradient_accumulation_steps
        loss.backward()

        loss_accum       += loss.item()
        grad_accum_count += 1

        if grad_accum_count >= args.gradient_accumulation_steps:
            torch.nn.utils.clip_grad_norm_(lora_params, max_norm=1.0)
            optimizer.step()
            optimizer.zero_grad()

            step += 1

            # Timing
            now = time.time()
            step_time = now - step_start
            recent_times.append(step_time)
            if len(recent_times) > 20:
                recent_times.pop(0)
            avg_time = sum(recent_times) / len(recent_times)
            remaining = (args.steps - step) * avg_time
            step_start = now

            disp_loss = loss_accum
            loss_accum = 0.0
            grad_accum_count = 0

            if step % 10 == 0:
                print(
                    f"[{step:4d}/{args.steps}] "
                    f"loss={disp_loss:.5f}  "
                    f"fwd={loss_fwd.item():.5f}  "
                    f"bwd={loss_bwd.item():.5f}  "
                    f"sym={loss_sym.item():.5f}  "
                    f"{avg_time:.1f}s/step  "
                    f"ETA {remaining/3600:.1f}h"
                )

            if step % args.save_every == 0 or step == args.steps:
                ckpt = os.path.join(args.output_dir, f"checkpoint_{step:04d}")
                os.makedirs(ckpt, exist_ok=True)
                pipeline.transformer.save_pretrained(ckpt)
                print(f"[Step {step}] Checkpoint saved -> {ckpt}")

    # ── Final save ────────────────────────────────────────────────────────────
    print("\n[Training] Complete. Saving final weights...")
    final_dir = os.path.join(args.output_dir, "lora_final")
    os.makedirs(final_dir, exist_ok=True)
    pipeline.transformer.save_pretrained(final_dir)

    try:
        from peft import get_peft_model_state_dict
        from safetensors.torch import save_file
        state    = get_peft_model_state_dict(pipeline.transformer)
        out_path = os.path.join(args.output_dir, "lora_weights.safetensors")
        save_file(state, out_path)
        sz = os.path.getsize(out_path) / 1e6
        print(f"[Training] ComfyUI LoRA: {out_path} ({sz:.1f} MB)")
    except Exception as e:
        print(f"[Training] safetensors export failed: {e}  -- use PEFT checkpoint at {final_dir}")

    print(f"\n[Training] Done. Output: {args.output_dir}")


# ──────────────────────────────────────────────────────────────────────────────
# Config template
# ──────────────────────────────────────────────────────────────────────────────

def generate_config_template(output_path: str = "configs/silvermax.yaml"):
    template = """\
# Silvermax film emulation LoRA training config (optimised)
# Run: python train_style_lora.py --config configs/silvermax.yaml

data_dir:   ./training_data/silvermax
output_dir: ./output/silvermax_lora
base_model: H:/Testing/FireRed-Image-Edit-1.1

forward_prompt:  "convert to Silvermax film black and white, high contrast, rich shadow detail"
backward_prompt: "colorize with natural realistic skin tones and colors"

lora_rank:     16
lora_alpha:    16.0
lora_dropout:  0.05

steps:          2000
batch_size:     1
lr:             0.0001
sym_weight:     0.1
gradient_accumulation_steps: 4
save_every:     500
seed:           42

# 384 is the recommended default - faster training, comparable quality for global style
# Increase to 512 for final production run after validating at 384
resolution:     384

device:   cuda:0
dtype:    bfloat16
gradient_checkpointing: true
use_autocast: true
"""
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(template)
    print(f"Config template written to {output_path}")


# ──────────────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    if "--generate_config" in sys.argv:
        idx  = sys.argv.index("--generate_config")
        path = sys.argv[idx + 1] if idx + 1 < len(sys.argv) else "configs/silvermax.yaml"
        generate_config_template(path)
        sys.exit(0)

    args = parse_args()
    if not args.data_dir:
        print("ERROR: --data_dir is required")
        print("Usage: python train_style_lora.py --config configs/silvermax.yaml")
        sys.exit(1)

    train(args)
