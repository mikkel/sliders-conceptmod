"""Music3 condition-encoder slider (notrigger-style).

Train LoRA on MiniMaxMusic3ConditionEncoder first. The transformer / AR LM
stay frozen and are not loaded for the dummy path. Real training consumes
cached frame_hiddens only.

Same contract as conceptmod/notrigger/train_notrigger.py:
  +scale on the neutral hidden should match the static positive encoding
  -scale on the neutral hidden should match the static negative encoding
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("HF_HOME", "/ml2/music/.cache/huggingface")
os.environ.setdefault("HUGGINGFACE_HUB_CACHE", os.environ["HF_HOME"] + "/hub")
os.environ.setdefault("HF_HUB_CACHE", os.environ["HUGGINGFACE_HUB_CACHE"])
os.environ.setdefault("TRANSFORMERS_CACHE", os.environ["HUGGINGFACE_HUB_CACHE"])
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

import torch
import torch.nn.functional as F
import yaml
from tqdm import tqdm

from conceptmod.textsliders.lora import LoRANetwork

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL = Path("/ml2/music/models/MiniMax-Music3")
DEFAULT_CACHE = ROOT / "cache" / "energy"
DEFAULT_SAVE = ROOT / "models" / "energy-encoder"

NEUTRAL_PROMPT = (
    "Genre: pop. BPM: 110. Mid energy, balanced mix, clear lead vocal, "
    "drums, bass, and guitar. Neither aggressive nor quiet."
)
POSITIVE_PROMPT = (
    "Genre: pop. BPM: 140. Extremely high energy, loud, aggressive, "
    "pounding drums, distorted guitars, shouted vocals, dense mix."
)
NEGATIVE_PROMPT = (
    "Genre: pop. BPM: 70. Extremely quiet, calm, ambient, whisper-soft, "
    "sparse, almost silent, no drums, gentle pads."
)
LYRICS = "[verse]\nI can feel it in the air tonight\n[chorus]\nLouder now or fade away"


@dataclass
class PromptSet:
    target: str
    positive: str
    negative: str
    neutral: str
    lyrics: str
    guidance: float = 4.0


def flush() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def load_prompts(path: Path | None) -> PromptSet:
    if path is None or not path.exists():
        return PromptSet(
            target=NEUTRAL_PROMPT,
            positive=POSITIVE_PROMPT,
            negative=NEGATIVE_PROMPT,
            neutral=NEUTRAL_PROMPT,
            lyrics=LYRICS,
        )
    data = yaml.safe_load(path.read_text())
    row = data[0] if isinstance(data, list) else data
    return PromptSet(
        target=row.get("target") or row.get("neutral") or NEUTRAL_PROMPT,
        positive=row.get("positive") or POSITIVE_PROMPT,
        negative=row.get("negative") or NEGATIVE_PROMPT,
        neutral=row.get("neutral") or row.get("target") or NEUTRAL_PROMPT,
        lyrics=row.get("lyrics") or LYRICS,
        guidance=float(row.get("guidance") or row.get("guidance_scale") or 4.0),
    )


def load_condition_encoder(model_dir: Path, device: torch.device, dtype: torch.dtype):
    from diffusers.models import MiniMaxMusic3ConditionEncoder

    encoder = MiniMaxMusic3ConditionEncoder.from_pretrained(
        str(model_dir),
        subfolder="condition_encoder",
        local_files_only=True,
        torch_dtype=dtype,
    )
    encoder.to(device)
    encoder.eval()
    encoder.requires_grad_(False)
    return encoder


def prompt_key(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:16]


@torch.no_grad()
def make_dummy_hiddens(
    encoder,
    frames: int,
    device: torch.device,
    dtype: torch.dtype,
    seed: int,
    encoded_delta: float = 25.0,
) -> dict[str, torch.Tensor]:
    """Correlated dummy AR hiddens with a calibrated encoded pos/neg gap.

    A raw 0.5 * unit-direction offset collapses through the encoder to ~0.06 L2,
    which makes MSE numerically useless and high LRs explode pperc. Scale the
    hidden-space direction so ||enc(pos)-enc(neu)|| ≈ encoded_delta.
    """
    hidden_dim = encoder.config.num_condition_layers * encoder.config.condition_hidden_dim
    g = torch.Generator(device="cpu").manual_seed(seed)
    base = torch.randn(1, frames, hidden_dim, generator=g, dtype=torch.float32)
    direction = torch.randn(1, frames, hidden_dim, generator=g, dtype=torch.float32)
    direction = direction / direction.norm().clamp_min(1e-6)
    probe = encoder((base + direction).to(device=device, dtype=dtype)).float()
    origin = encoder(base.to(device=device, dtype=dtype)).float()
    probe_norm = torch.norm(probe - origin).clamp_min(1e-8)
    scale = float(encoded_delta) / float(probe_norm.item())
    print(f"dummy hidden scale={scale:.3f} (probe encoded L2={probe_norm.item():.4f} -> {encoded_delta})")
    return {
        "neutral": base.to(device=device, dtype=dtype),
        "positive": (base + scale * direction).to(device=device, dtype=dtype),
        "negative": (base - scale * direction).to(device=device, dtype=dtype),
        "target": base.to(device=device, dtype=dtype),
    }


def save_hiddens(cache_dir: Path, hiddens: dict[str, torch.Tensor]) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    packed = {k: v.detach().cpu() for k, v in hiddens.items()}
    torch.save(packed, cache_dir / "frame_hiddens.pt")
    meta = {k: list(v.shape) for k, v in packed.items()}
    (cache_dir / "meta.json").write_text(json.dumps(meta, indent=2))


def load_hiddens(cache_dir: Path, device: torch.device, dtype: torch.dtype) -> dict[str, torch.Tensor]:
    path = cache_dir / "frame_hiddens.pt"
    if not path.exists():
        raise FileNotFoundError(f"missing cached hiddens: {path}")
    packed = torch.load(path, map_location="cpu", weights_only=True)
    return {k: v.to(device=device, dtype=dtype) for k, v in packed.items()}


@torch.no_grad()
def cache_real_hiddens(
    model_dir: Path,
    prompts: PromptSet,
    cache_dir: Path,
    device: torch.device,
    dtype: torch.dtype,
    duration: float,
    seed: int,
) -> dict[str, torch.Tensor]:
    """Run the AR stage once per branch and store frame_hiddens."""
    from diffusers import ModularPipeline

    load_order = (
        "tokenizer",
        "scheduler",
        "vocoder",
        "condition_encoder",
        "rvq_depth_decoder",
        "transformer",
        "language_model",
    )
    pipe = ModularPipeline.from_pretrained(str(model_dir), local_files_only=True)
    local_kwargs = {
        "pretrained_model_name_or_path": str(model_dir),
        "local_files_only": True,
        "dtype": dtype,
    }
    for name in load_order:
        pipe.load_components(names=name, **local_kwargs)
    pipe.to(device)

    branches = {
        "neutral": prompts.neutral,
        "positive": prompts.positive,
        "negative": prompts.negative,
        "target": prompts.target,
    }
    hiddens: dict[str, torch.Tensor] = {}
    for name, caption in branches.items():
        print(f"AR cache {name} ({duration}s)…", flush=True)
        generator = torch.Generator(device=str(device)).manual_seed(seed)
        # Generate audio so the modular pipeline runs the semantic step; then
        # steal frame_hiddens from a dedicated encode pass below if exposed.
        # The pipeline does not return frame_hiddens, so call the AR block.
        from diffusers.modular_pipelines.minimax_music3.encoders import (
            MiniMaxMusic3SemanticGenerationStep,
            MiniMaxMusic3TextEncoderStep,
        )
        from diffusers.modular_pipelines.modular_pipeline import PipelineState

        state = PipelineState()
        state.set("prompt", caption)
        state.set("lyrics", prompts.lyrics)
        state.set("audio_duration", float(duration))
        state.set("generator", generator)
        text_step = MiniMaxMusic3TextEncoderStep()
        ar_step = MiniMaxMusic3SemanticGenerationStep()
        pipe, state = text_step(pipe, state)
        pipe, state = ar_step(pipe, state)
        frame_hiddens = state.get("frame_hiddens")
        if frame_hiddens is None:
            raise RuntimeError(f"AR step did not produce frame_hiddens for {name}")
        hiddens[name] = frame_hiddens.detach().to(device="cpu", dtype=torch.float32)
        print(f"  {name}: {tuple(hiddens[name].shape)}", flush=True)
        flush()

    min_frames = min(int(v.shape[1]) for v in hiddens.values())
    if min_frames < 8:
        raise RuntimeError(f"AR produced too few frames ({min_frames})")
    hiddens = {k: v[:, :min_frames].contiguous() for k, v in hiddens.items()}
    save_hiddens(cache_dir, hiddens)

    del pipe
    flush()
    return {k: v.to(device=device, dtype=dtype) for k, v in hiddens.items()}


def fixed_distance_loss(trainable: torch.Tensor, target: torch.Tensor, step: torch.Tensor):
    diff = target - trainable
    current = torch.norm(diff, dim=-1, keepdim=True)
    direction = diff / (current + 1e-8)
    clamped = torch.clamp(step.view(1, 1, 1), -current, current)
    dest = trainable + direction * clamped
    return ((trainable - dest) ** 2).mean()


def encode(encoder, hiddens: torch.Tensor) -> torch.Tensor:
    return encoder(hiddens)


def set_lora_scale(network: LoRANetwork, scale: float) -> None:
    network.set_lora_slider(scale)
    for lora in network.unet_loras:
        lora.multiplier = float(scale)


def train(args) -> Path:
    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if args.precision == "bf16" else torch.float32
    prompts = load_prompts(Path(args.prompts_file) if args.prompts_file else None)
    cache_dir = Path(args.cache_dir)
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = save_dir / "metrics.jsonl"

    static = load_condition_encoder(Path(args.model_dir), device, dtype)
    trainable = load_condition_encoder(Path(args.model_dir), device, dtype)

    if args.dummy:
        hiddens = make_dummy_hiddens(
            static,
            args.dummy_frames,
            device,
            dtype,
            args.seed,
            encoded_delta=args.dummy_delta,
        )
        print("using correlated dummy frame_hiddens", {k: tuple(v.shape) for k, v in hiddens.items()})
    elif args.skip_ar:
        hiddens = load_hiddens(cache_dir, device, dtype)
        print("loaded cached frame_hiddens", {k: tuple(v.shape) for k, v in hiddens.items()})
    else:
        hiddens = cache_real_hiddens(
            Path(args.model_dir),
            prompts,
            cache_dir,
            device,
            dtype,
            args.duration,
            args.seed,
        )

    if args.full_ft:
        network = None
        trainable.requires_grad_(True)
        trainable.train()
        params = [p for p in trainable.parameters() if p.requires_grad]
        n_params = sum(p.numel() for p in params)
        print(f"full encoder finetune params={n_params}")
    else:
        network = LoRANetwork(
            trainable,
            rank=args.rank,
            multiplier=1.0,
            delimiter="-",
            alpha=args.alpha,
            target_replace=["MiniMaxMusic3ConditionEncoder"],
            prefix="lora_te",
            train_method="full",
        ).to(device, dtype=torch.float32)
        params = [p for p in network.parameters() if p.requires_grad]
        n_params = sum(p.numel() for p in params)
        print(f"encoder LoRA modules={len(network.unet_loras)} params={n_params} rank={args.rank}")
        if len(network.unet_loras) == 0:
            raise RuntimeError("LoRA wrapped 0 modules; Conv1d targeting failed")

    if args.optim == "sgd":
        optimizer = torch.optim.SGD(params, lr=args.lr)
    else:
        optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=1e-6)

    with torch.no_grad():
        pos_tgt = encode(static, hiddens["positive"]).float()
        neg_tgt = encode(static, hiddens["negative"]).float()
        neu_ref = encode(static, hiddens["neutral"]).float()
    pos_step = torch.norm(pos_tgt - neu_ref).mean() / args.split
    neg_step = torch.norm(neg_tgt - neu_ref).mean() / args.split
    print(
        f"target distances  pos={torch.norm(pos_tgt - neu_ref).item():.4f} "
        f"neg={torch.norm(neg_tgt - neu_ref).item():.4f} "
        f"step_p={pos_step.item():.4f} step_n={neg_step.item():.4f}"
    )

    history = []
    pbar = tqdm(range(args.steps), desc="encoder")
    for step in pbar:
        if network is not None:
            set_lora_scale(network, 1.0)
        pred_pos = encode(trainable, hiddens["neutral"]).float()
        if args.loss == "mse":
            ploss = F.mse_loss(pred_pos, pos_tgt)
        else:
            ploss = fixed_distance_loss(pred_pos, pos_tgt, pos_step)

        if network is not None:
            set_lora_scale(network, -1.0)
            pred_neg = encode(trainable, hiddens["neutral"]).float()
            if args.loss == "mse":
                nloss = F.mse_loss(pred_neg, neg_tgt)
            else:
                nloss = fixed_distance_loss(pred_neg, neg_tgt, neg_step)
            v_pos = pred_pos - neu_ref
            v_neg = pred_neg - neu_ref
            v_pos_t = pos_tgt - neu_ref
            v_neg_t = neg_tgt - neu_ref
            sim_pos = F.cosine_similarity(v_pos.flatten(1), v_neg_t.flatten(1), dim=-1).mean()
            sim_neg = F.cosine_similarity(v_neg.flatten(1), v_pos_t.flatten(1), dim=-1).mean()
            collapse = F.cosine_similarity(v_pos.flatten(1), v_neg.flatten(1), dim=-1).mean()
            regularizer = args.lambda_sim * (
                sim_pos.clamp_min(0) + sim_neg.clamp_min(0) + collapse.clamp_min(0)
            )
            ndist = torch.norm(pred_neg - neg_tgt).mean()
            n0 = torch.norm(neg_tgt - neu_ref).mean().clamp_min(1e-6)
            nperc = (ndist / n0).item()
            nloss_v = nloss
        else:
            nloss_v = pred_pos.new_zeros(())
            regularizer = pred_pos.new_zeros(())
            collapse = pred_pos.new_zeros(())
            nperc = 0.0

        pdist = torch.norm(pred_pos - pos_tgt).mean()
        p0 = torch.norm(pos_tgt - neu_ref).mean().clamp_min(1e-6)
        pperc = (pdist / p0).item()

        loss = ploss + nloss_v + regularizer
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_value_(params, clip_value=1.0)
        optimizer.step()

        row = {
            "step": step,
            "loss": float(loss.detach().item()),
            "ploss": float(ploss.detach().item()),
            "nloss": float(nloss_v.detach().item()),
            "pperc": pperc,
            "nperc": nperc,
            "collapse": float(collapse.detach().item()),
            "lr": args.lr,
        }
        history.append(row)
        pbar.set_description(
            f"loss {row['loss']:.4f} p%{pperc*100:.1f} n%{nperc*100:.1f} col {row['collapse']:.2f}"
        )
        with metrics_path.open("a") as fh:
            fh.write(json.dumps(row) + "\n")

        if args.save_every and step > 0 and step % args.save_every == 0:
            ckpt = save_dir / f"{args.name}_step{step}.safetensors"
            if network is not None:
                network.save_weights(ckpt, dtype=torch.float32)
            else:
                from safetensors.torch import save_file

                save_file({k: v.detach().cpu() for k, v in trainable.state_dict().items()}, str(ckpt))

    last = save_dir / f"{args.name}_last.safetensors"
    if network is not None:
        network.save_weights(last, dtype=torch.float32)
    else:
        from safetensors.torch import save_file

        save_file({k: v.detach().cpu() for k, v in trainable.state_dict().items()}, str(last))
    summary = {
        "first_loss": history[0]["loss"] if history else None,
        "last_loss": history[-1]["loss"] if history else None,
        "best_loss": min(h["loss"] for h in history) if history else None,
        "first_pperc": history[0]["pperc"] if history else None,
        "last_pperc": history[-1]["pperc"] if history else None,
        "first_nperc": history[0]["nperc"] if history else None,
        "last_nperc": history[-1]["nperc"] if history else None,
        "checkpoint": str(last),
        "modules": 0 if network is None else len(network.unet_loras),
        "params": n_params,
        "dummy": bool(args.dummy),
        "full_ft": bool(args.full_ft),
        "rank": args.rank,
        "lr": args.lr,
        "optim": args.optim,
        "loss": args.loss,
        "steps": args.steps,
    }
    (save_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    if summary["last_loss"] is None or summary["first_loss"] is None:
        raise RuntimeError("no training steps ran")
    moved = summary["last_pperc"] is not None and summary["last_pperc"] < summary["first_pperc"] * 0.9
    if not moved:
        print(
            "WARNING: pperc did not drop 10%. Try --lr, --rank, --optim, --loss, or --full_ft.",
            file=sys.stderr,
        )
    return last


def parse_args(argv: list[str] | None = None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model_dir", default=str(DEFAULT_MODEL))
    p.add_argument("--prompts_file", default=str(ROOT / "conceptmod/textsliders/data/prompts-music3.yaml"))
    p.add_argument("--cache_dir", default=str(DEFAULT_CACHE))
    p.add_argument("--save_dir", default=str(DEFAULT_SAVE))
    p.add_argument("--name", default="energy-encoder")
    p.add_argument("--device", type=int, default=0)
    p.add_argument("--rank", type=int, default=64)
    p.add_argument("--alpha", type=float, default=64.0)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--steps", type=int, default=4000)
    p.add_argument("--optim", choices=["sgd", "adamw"], default="adamw")
    p.add_argument("--loss", choices=["mse", "curriculum"], default="mse")
    p.add_argument("--lambda_sim", type=float, default=0.05)
    p.add_argument("--split", type=float, default=20.0)
    p.add_argument("--precision", choices=["bf16", "fp32"], default="bf16")
    p.add_argument("--duration", type=float, default=4.0)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--dummy", action="store_true")
    p.add_argument("--dummy_frames", type=int, default=32)
    p.add_argument("--dummy_delta", type=float, default=25.0, help="target ||enc(pos)-enc(neu)|| for dummy data")
    p.add_argument("--full_ft", action="store_true", help="finetune the whole encoder (capacity check, +side only)")
    p.add_argument("--skip_ar", action="store_true")
    p.add_argument("--save_every", type=int, default=100)
    return p.parse_args(argv)


if __name__ == "__main__":
    train(parse_args())
