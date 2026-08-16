#!/usr/bin/env python3
"""Train a textual concept slider on MiniMax Music 3's flow transformer.

Does not import conceptmod PromptSettings (pydantic v1). Reuses LoRANetwork
with target MiniMaxMusic3Attention. Dummy mode never loads the 8B LM.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

_HF_HOME = "/ml2/music/.cache/huggingface"
os.environ["HF_HOME"] = _HF_HOME
os.environ["HUGGINGFACE_HUB_CACHE"] = f"{_HF_HOME}/hub"
os.environ["HF_HUB_CACHE"] = f"{_HF_HOME}/hub"
os.environ["TRANSFORMERS_CACHE"] = f"{_HF_HOME}/hub"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

import torch
import yaml
from tqdm.auto import tqdm

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from conceptmod.textsliders.lora import LoRANetwork

DEFAULT_MODEL_DIR = Path("/ml2/music/models/MiniMax-Music3")
DEFAULT_SAVE_DIR = Path("/ml2/music/sliders-conceptmod/models/energy-slider")
DEFAULT_CACHE_DIR = Path("/ml2/music/sliders-conceptmod/cache/energy")
DEFAULT_CONFIG = Path(__file__).resolve().parent / "data" / "config-music3.yaml"
DEFAULT_PROMPTS = Path(__file__).resolve().parent / "data" / "prompts-music3.yaml"
TARGET_REPLACE_ATTN = ["MiniMaxMusic3Attention"]
# Attention-only misses proj_in / preprocess_conv, where the condition is
# concatenated onto the latent. Full wraps those plus every block FF.
TARGET_REPLACE_FULL = [
    "MiniMaxMusic3Attention",
    "MiniMaxMusic3TransformerBlock",
    "MiniMaxMusic3Transformer1DModel",
]
TARGET_REPLACE = TARGET_REPLACE_ATTN
DEFAULT_LYRICS = "[verse]\nI keep walking through the night\nlooking for a little light\n[chorus]\nI will not go quietly"
# Match /ml2/music/app/generator.py _LOAD_ORDER, then drop unused modules after cache.
AR_COMPONENTS = (
    "tokenizer",
    "scheduler",
    "vocoder",
    "condition_encoder",
    "rvq_depth_decoder",
    "transformer",
    "language_model",
)
UNLOAD_AFTER_CACHE = ("language_model", "rvq_depth_decoder", "tokenizer", "vocoder", "scheduler", "transformer")
MIN_AR_FRAMES = 8
AR_RETRY_LIMIT = 5


@dataclass
class SliderPrompt:
    target: str
    positive: str
    negative: str
    neutral: str
    lyrics: str
    action: str = "enhance"
    guidance_scale: float = 4.0
    batch_size: int = 1
    unconditional: str = ""


def _load_yaml(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


@dataclass
class PromptsMeta:
    plus_label: str = ""
    minus_label: str = ""
    recommended_range: list[float] = field(default_factory=lambda: [-2.0, 2.0])


def _expand_attributes(item: dict) -> list[dict]:
    """Replicate a prompt row once per attribute, prefixing the attribute to all
    four captions (upstream sliders' disentanglement mechanism): the axis delta
    is then averaged over rows where the orthogonal trait is pinned both ways."""
    attributes = item.get("attributes")
    if not attributes:
        return [item]
    rows = []
    for attribute in attributes:
        prefix = str(attribute).strip()
        row = dict(item)
        for key in ("target", "positive", "negative", "neutral"):
            value = item.get(key)
            if value:
                row[key] = f"{prefix} {value}"
        rows.append(row)
    return rows


def load_prompts(path: Path) -> tuple[list[SliderPrompt], PromptsMeta]:
    raw = _load_yaml(path)
    meta = PromptsMeta()
    if isinstance(raw, dict):
        meta.plus_label = str(raw.get("plus_label") or "")
        meta.minus_label = str(raw.get("minus_label") or "")
        rng = raw.get("recommended_range")
        if isinstance(rng, (list, tuple)) and len(rng) == 2:
            meta.recommended_range = [float(rng[0]), float(rng[1])]
        raw = raw.get("rows")
    if not isinstance(raw, list) or not raw:
        raise ValueError(f"prompts file is empty: {path}")
    prompts: list[SliderPrompt] = []
    for item in raw:
        if not isinstance(item, dict) or "target" not in item:
            raise ValueError(f"each prompt must be a mapping with target: {item!r}")
        for row in _expand_attributes(item):
            target = str(row["target"])
            prompts.append(
                SliderPrompt(
                    target=target,
                    positive=str(row.get("positive") or target),
                    negative=str(row.get("negative") or ""),
                    neutral=str(row.get("neutral") or target),
                    lyrics=str(row.get("lyrics") or DEFAULT_LYRICS),
                    action=str(row.get("action") or "enhance"),
                    guidance_scale=float(row.get("guidance_scale", 4.0)),
                    batch_size=int(row.get("batch_size", 1)),
                    unconditional=str(row.get("unconditional") or ""),
                )
            )
    return prompts, meta


def load_config_defaults(path: Path | None) -> dict[str, Any]:
    if path is None or not path.exists():
        return {}
    raw = _load_yaml(path) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"config must be a mapping: {path}")
    return raw


def _prompt_hash(prompt: str, lyrics: str, duration: float, seed: int) -> str:
    payload = f"{prompt}\n---\n{lyrics}\n---\n{duration:.3f}\n---\n{seed}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def _cache_path(cache_dir: Path, prompt: str, lyrics: str, duration: float, seed: int) -> Path:
    return cache_dir / f"{_prompt_hash(prompt, lyrics, duration, seed)}.pt"


def _flush() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _pick_device(index: int, dummy: bool) -> torch.device:
    if not torch.cuda.is_available():
        return torch.device("cpu")
    free, _total = torch.cuda.mem_get_info(index)
    if dummy and free < 2 * 1024**3:
        print(f"cuda:{index} has {free / 1024**3:.1f} GiB free; dummy falling back to CPU")
        return torch.device("cpu")
    return torch.device(f"cuda:{index}")


def _set_lora_multiplier(network: LoRANetwork, value: float) -> None:
    for lora in network.unet_loras:
        lora.multiplier = value


def _align_conditions(conditions: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    lengths = [int(tensor.shape[1]) for tensor in conditions.values()]
    if not lengths:
        raise ValueError("no conditions to align")
    min_len = min(lengths)
    if min_len < 1:
        raise ValueError("condition latent length is zero")
    aligned = {}
    for name, tensor in conditions.items():
        cropped = tensor[:, :min_len].contiguous()
        aligned[name] = cropped
        if tensor.shape[1] != min_len:
            print(f"cropped {name} condition {tensor.shape[1]} -> {min_len}")
    return aligned


def _make_dummy_transformer() -> torch.nn.Module:
    from diffusers import MiniMaxMusic3Transformer1DModel

    return MiniMaxMusic3Transformer1DModel(
        in_channels=128,
        condition_dim=2048,
        num_layers=2,
        num_attention_heads=4,
        attention_head_dim=32,
        ff_inner_dim=128,
        rotary_dim=16,
        fourier_embedding_dim=64,
    )


def _load_transformer(model_dir: Path, device: torch.device) -> torch.nn.Module:
    from diffusers import MiniMaxMusic3Transformer1DModel

    transformer = MiniMaxMusic3Transformer1DModel.from_pretrained(
        str(model_dir),
        subfolder="transformer",
        torch_dtype=torch.bfloat16,
        local_files_only=True,
    )
    transformer.requires_grad_(False)
    transformer.eval()
    if hasattr(transformer, "enable_gradient_checkpointing"):
        transformer.enable_gradient_checkpointing()
    return transformer.to(device)


def _load_ar_pipeline(model_dir: Path, device: torch.device):
    from diffusers import ModularPipeline

    local_kwargs = {
        "pretrained_model_name_or_path": str(model_dir),
        "local_files_only": True,
        "dtype": torch.bfloat16,
    }
    pipe = ModularPipeline.from_pretrained(str(model_dir), local_files_only=True)
    for name in AR_COMPONENTS:
        print(f"loading {name} for AR cache")
        pipe.load_components(names=name, **local_kwargs)
    if device.type == "cuda":
        pipe.to(str(device))
    return pipe


def _unload_ar(pipe: Any) -> None:
    for name in UNLOAD_AFTER_CACHE:
        if hasattr(pipe, name):
            setattr(pipe, name, None)
    _flush()


def _min_ar_frames(duration: float) -> int:
    return max(MIN_AR_FRAMES, int(float(duration) * 12.5))


def _attach_lm_slider(pipe: Any, weights: Path, device: torch.device) -> LoRANetwork:
    from safetensors.torch import load_file

    network = LoRANetwork(
        pipe.language_model,
        rank=8,
        alpha=8.0,
        multiplier=1.0,
        target_replace=["Qwen3Attention"],
        train_method="full",
        delimiter="-",
        prefix="lora_te",
    )
    network.to(device)
    state = load_file(str(weights), device="cpu")
    missing, unexpected = network.load_state_dict(state, strict=False)
    print(
        f"AR LM slider {weights.name} modules={len(network.unet_loras)} "
        f"missing={len(missing)} unexpected={len(unexpected)}",
        flush=True,
    )
    if missing or not network.unet_loras:
        raise RuntimeError(f"failed to wrap LM slider: missing={len(missing)}")
    for lora in network.unet_loras:
        lora.multiplier = 0.0
    return network


def _encode_condition(
    pipe: Any,
    prompt: str,
    lyrics: str,
    duration: float,
    seed: int,
    device: torch.device,
    lm_network: LoRANetwork | None = None,
    lm_scale: float = 0.0,
) -> torch.Tensor:
    from diffusers.modular_pipelines.modular_pipeline import PipelineState

    min_frames = _min_ar_frames(duration)
    last_error = "AR produced no frames"
    for attempt in range(AR_RETRY_LIMIT):
        attempt_seed = int(seed) + attempt
        generator = torch.Generator(device=device).manual_seed(attempt_seed)
        state = PipelineState()
        state.set("prompt", prompt)
        state.set("lyrics", lyrics)
        state.set("audio_duration", float(duration))
        state.set("generator", generator)
        blocks = pipe._blocks.sub_blocks
        if lm_network is not None:
            lm_network.set_lora_slider(float(lm_scale))
            with lm_network:
                blocks["text_encoder"](pipe, state)
                blocks["semantic_generator"](pipe, state)
        else:
            blocks["text_encoder"](pipe, state)
            blocks["semantic_generator"](pipe, state)
        frame_hiddens = state.get("frame_hiddens")
        if frame_hiddens is None:
            last_error = "AR stage did not produce frame_hiddens"
            print(f"AR retry {attempt + 1}/{AR_RETRY_LIMIT}: {last_error} seed={attempt_seed}")
            continue
        n_frames = int(frame_hiddens.shape[1])
        print(f"AR frames={n_frames} min={min_frames} seed={attempt_seed} prompt={prompt[:60]!r}")
        if n_frames < min_frames:
            last_error = f"AR early EOS: {n_frames} < {min_frames} frames"
            print(f"AR retry {attempt + 1}/{AR_RETRY_LIMIT}: {last_error}")
            continue
        condition = pipe.condition_encoder(frame_hiddens.to(pipe._execution_device))
        return condition.detach().to(dtype=torch.bfloat16).cpu()
    raise RuntimeError(f"{last_error} after {AR_RETRY_LIMIT} seeds starting at {seed}")


def _cache_or_encode(
    pipe: Any | None,
    cache_dir: Path,
    prompt: str,
    lyrics: str,
    duration: float,
    seed: int,
    device: torch.device,
    skip_ar: bool,
    lm_network: LoRANetwork | None = None,
    lm_scale: float = 0.0,
) -> torch.Tensor:
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_prompt = prompt if lm_network is None else f"__lm{lm_scale:+.3f}__\n{prompt}"
    path = _cache_path(cache_dir, cache_prompt, lyrics, duration, seed)
    if path.exists():
        payload = torch.load(path, map_location="cpu", weights_only=False)
        condition = payload["condition"] if isinstance(payload, dict) else payload
        print(f"cache hit {path.name} shape={tuple(condition.shape)} lm={lm_scale:g}")
        return condition
    if skip_ar:
        raise FileNotFoundError(f"--skip_ar set but cache missing: {path}")
    if pipe is None:
        raise RuntimeError("AR pipeline is not loaded")
    print(f"encoding condition -> {path.name} lm={lm_scale:g} prompt={prompt[:60]!r}")
    condition = _encode_condition(
        pipe, prompt, lyrics, duration, int(seed), device, lm_network=lm_network, lm_scale=lm_scale
    )
    torch.save(
        {
            "condition": condition,
            "prompt": prompt,
            "lyrics": lyrics,
            "duration": float(duration),
            "seed": int(seed),
            "lm_scale": float(lm_scale),
        },
        path,
    )
    return condition


def build_conditions(
    prompts: list[SliderPrompt],
    cache_dir: Path,
    duration: float,
    seeds: list[int],
    device: torch.device,
    model_dir: Path,
    skip_ar: bool,
    dummy: bool,
    lm_weights: Path | None = None,
) -> list[tuple[SliderPrompt, dict[str, torch.Tensor]]]:
    """Return one (prompt, conditions) entry per (prompt row, AR seed)."""
    if dummy:
        latent_len = 32
        entries = []
        for prompt in prompts:
            for _seed in seeds:
                noise = {
                    name: torch.randn(1, latent_len, 2048)
                    for name in ("target", "positive", "negative", "neutral")
                }
                entries.append((prompt, noise))
        return entries

    use_lm = lm_weights is not None
    jobs: list[tuple[str, str, float, int]] = []
    for prompt in prompts:
        base = prompt.neutral or prompt.target
        for seed in seeds:
            if use_lm:
                # Same caption; LM slider writes the axis-shifted AR plan the transformer should match.
                jobs.extend(
                    [
                        (base, prompt.lyrics, 0.0, seed),
                        (base, prompt.lyrics, 1.0, seed),
                        (base, prompt.lyrics, -1.0, seed),
                    ]
                )
            else:
                jobs.extend(
                    [
                        (prompt.target, prompt.lyrics, 0.0, seed),
                        (prompt.positive, prompt.lyrics, 0.0, seed),
                        (prompt.negative, prompt.lyrics, 0.0, seed),
                        (prompt.neutral, prompt.lyrics, 0.0, seed),
                    ]
                )

    def _key(text: str, lyrics: str, scale: float, seed: int) -> str:
        hashed = text if not use_lm else f"__lm{scale:+.3f}__\n{text}"
        return _prompt_hash(hashed, lyrics, duration, seed)

    need_ar = False
    for text, lyrics, scale, seed in jobs:
        hashed = text if not use_lm else f"__lm{scale:+.3f}__\n{text}"
        if not _cache_path(cache_dir, hashed, lyrics, duration, seed).exists():
            need_ar = True

    pipe = None
    lm_network = None
    if need_ar:
        if skip_ar:
            raise FileNotFoundError(f"--skip_ar set but missing LM/AR cache in {cache_dir}")
        pipe = _load_ar_pipeline(model_dir, device)
        if use_lm:
            lm_network = _attach_lm_slider(pipe, lm_weights, device)

    encoded: dict[str, torch.Tensor] = {}
    try:
        for text, lyrics, scale, seed in jobs:
            cache_key = _key(text, lyrics, scale, seed)
            if cache_key in encoded:
                continue
            encoded[cache_key] = _cache_or_encode(
                pipe,
                cache_dir,
                text,
                lyrics,
                duration,
                seed,
                device,
                skip_ar,
                lm_network=lm_network,
                lm_scale=scale,
            )
    finally:
        if pipe is not None:
            _unload_ar(pipe)
            del pipe
            _flush()

    entries = []
    for prompt in prompts:
        base = prompt.neutral or prompt.target
        for seed in seeds:
            if use_lm:
                branch = {
                    "target": encoded[_key(base, prompt.lyrics, 0.0, seed)],
                    "positive": encoded[_key(base, prompt.lyrics, 1.0, seed)],
                    "negative": encoded[_key(base, prompt.lyrics, -1.0, seed)],
                    "neutral": encoded[_key(base, prompt.lyrics, 0.0, seed)],
                }
            else:
                branch = {
                    "target": encoded[_key(prompt.target, prompt.lyrics, 0.0, seed)],
                    "positive": encoded[_key(prompt.positive, prompt.lyrics, 0.0, seed)],
                    "negative": encoded[_key(prompt.negative, prompt.lyrics, 0.0, seed)],
                    "neutral": encoded[_key(prompt.neutral, prompt.lyrics, 0.0, seed)],
                }
            entries.append((prompt, _align_conditions(branch)))
    return entries


def _velocity_target(
    vel_neu: torch.Tensor,
    vel_pos: torch.Tensor,
    vel_neg: torch.Tensor,
    guidance: float,
    action: str,
) -> torch.Tensor:
    delta = guidance * (vel_pos - vel_neg)
    if action == "erase":
        return vel_neu - delta
    if action != "enhance":
        raise ValueError(f"action must be enhance or erase, got {action!r}")
    return vel_neu + delta


def _condition_hash(condition: torch.Tensor) -> str:
    data = condition.to(torch.float32).cpu().contiguous().numpy().tobytes()
    return hashlib.sha256(data).hexdigest()[:16]


INFERENCE_CFG = 1.7  # ClassifierFreeGuidance default in MiniMaxMusic3ChunkDenoiseInner


@torch.no_grad()
def _generate_x0(
    transformer: torch.nn.Module,
    condition: torch.Tensor,
    model_dir: Path,
    device: torch.device,
    seed: int,
    num_steps: int = 30,
    amp: bool = True,
) -> torch.Tensor:
    """Flow-integrate a clean latent for one condition, replicating inference:
    sigmas linspace(1, 1/N), CFG 1.7 with a zeros condition on the uncond branch.
    Single window at the condition's full length (training forwards use the same
    length, so the anchor stays self-consistent even past the 200-frame chunk)."""
    import numpy as np
    from diffusers import FlowMatchEulerDiscreteScheduler

    scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
        str(model_dir), subfolder="scheduler", local_files_only=True
    )
    sigmas = np.linspace(1.0, 1.0 / num_steps, num_steps)
    scheduler.set_timesteps(sigmas=sigmas, device=device)

    dtype = torch.bfloat16 if amp else torch.float32
    cond = condition.to(device=device, dtype=dtype)
    zeros = torch.zeros_like(cond)
    generator = torch.Generator(device=device.type).manual_seed(int(seed))
    latents = torch.randn(
        (1, 128, int(cond.shape[1])), device=device, dtype=dtype, generator=generator
    )
    for t in scheduler.timesteps:
        timestep = t.expand(latents.shape[0]).to(latents.dtype)
        if amp and device.type == "cuda":
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                vel_cond = transformer(latents, timestep, cond, return_dict=False)[0]
                vel_uncond = transformer(latents, timestep, zeros, return_dict=False)[0]
        else:
            vel_cond = transformer(latents, timestep, cond, return_dict=False)[0]
            vel_uncond = transformer(latents, timestep, zeros, return_dict=False)[0]
        velocity = vel_uncond + INFERENCE_CFG * (vel_cond - vel_uncond)
        latents = scheduler.step(velocity, t, latents, return_dict=False)[0]
    return latents.to(torch.float32).cpu()


def build_x0_banks(
    entries: list[tuple[SliderPrompt, dict[str, torch.Tensor]]],
    transformer: torch.nn.Module,
    cache_dir: Path,
    model_dir: Path,
    device: torch.device,
    seed: int,
    per_row: int,
    num_steps: int,
    dummy: bool,
) -> list[list[torch.Tensor]]:
    """One bank of clean latents per (row, cond-seed) entry, cached on disk.
    Generated from the *neutral* condition with LoRA multipliers already at 0."""
    banks: list[list[torch.Tensor]] = []
    cache_dir.mkdir(parents=True, exist_ok=True)
    for index, (_prompt, conds) in enumerate(entries):
        neutral = conds["neutral"]
        bank: list[torch.Tensor] = []
        for i in range(max(1, per_row)):
            x0_seed = int(seed) * 100 + i
            if dummy:
                bank.append(torch.randn(1, 128, int(neutral.shape[1])))
                continue
            path = cache_dir / f"x0_{_condition_hash(neutral)}_s{x0_seed}.pt"
            if path.exists():
                payload = torch.load(path, map_location="cpu", weights_only=False)
                bank.append(payload["latents"])
                print(f"x0 cache hit {path.name}", flush=True)
                continue
            print(f"generating x0 anchor {index + 1}/{len(entries)} seed={x0_seed}", flush=True)
            latents = _generate_x0(
                transformer, neutral, model_dir, device, x0_seed, num_steps=num_steps
            )
            torch.save({"latents": latents, "seed": x0_seed}, path)
            bank.append(latents)
        banks.append(bank)
    return banks


def train(args: argparse.Namespace) -> Path:
    config = load_config_defaults(Path(args.config_file) if args.config_file else None)
    prompts_path = Path(args.prompts_file or config.get("prompts_file") or DEFAULT_PROMPTS)
    if not prompts_path.is_absolute():
        candidate = _REPO_ROOT / prompts_path
        prompts_path = candidate if candidate.exists() else Path(__file__).resolve().parent / prompts_path
    prompts, prompts_meta = load_prompts(prompts_path)
    if args.plus_label:
        prompts_meta.plus_label = args.plus_label
    if args.minus_label:
        prompts_meta.minus_label = args.minus_label

    train_cfg = config.get("train") or {}
    save_cfg = config.get("save") or {}
    network_cfg = config.get("network") or {}
    pretrained = config.get("pretrained_model") or {}

    name = args.name or save_cfg.get("name") or "energy"
    rank = int(args.rank if args.rank is not None else network_cfg.get("rank", 4))
    alpha = float(args.alpha if args.alpha is not None else network_cfg.get("alpha", 1.0))
    steps = int(args.steps if args.steps is not None else train_cfg.get("iterations", 80))
    duration = float(args.duration if args.duration is not None else 4.0)
    lr = float(args.lr if args.lr is not None else train_cfg.get("lr", 1e-4))
    if args.guidance is not None:
        for prompt in prompts:
            prompt.guidance_scale = float(args.guidance)
    target_replace = TARGET_REPLACE_FULL if args.targets == "full" else TARGET_REPLACE_ATTN
    lr_name = str(train_cfg.get("lr_scheduler", "cosine"))
    model_dir = Path(args.model_dir or pretrained.get("name_or_path") or DEFAULT_MODEL_DIR)
    cache_dir = Path(args.cache_dir or DEFAULT_CACHE_DIR)
    save_dir = Path(args.save_dir or save_cfg.get("path") or DEFAULT_SAVE_DIR)
    if args.dummy:
        steps = min(steps, 2)
    device = _pick_device(int(args.device), dummy=bool(args.dummy))
    seed = int(args.seed)
    torch.manual_seed(seed)
    if args.cond_seeds:
        cond_seeds = [int(part) for part in str(args.cond_seeds).split(",") if part.strip()]
    else:
        cond_seeds = [seed]

    print(
        f"train music3 slider name={name} rank={rank} alpha={alpha} steps={steps} "
        f"duration={duration} device={device} dummy={bool(args.dummy)} "
        f"targets={args.targets} lm_condition={bool(args.lm_weights)} "
        f"xt_mode={args.xt_mode} bidirectional={bool(args.bidirectional)} cond_seeds={cond_seeds}"
    )

    entries = build_conditions(
        prompts=prompts,
        cache_dir=cache_dir,
        duration=duration,
        seeds=cond_seeds,
        device=device,
        model_dir=model_dir,
        skip_ar=bool(args.skip_ar),
        dummy=bool(args.dummy),
        lm_weights=Path(args.lm_weights) if args.lm_weights else None,
    )

    if args.dummy:
        transformer = _make_dummy_transformer().to(device)
        transformer.requires_grad_(False)
        transformer.train()
    else:
        transformer = _load_transformer(model_dir, device)

    network = LoRANetwork(
        transformer,
        rank=rank,
        alpha=alpha,
        multiplier=1.0,
        target_replace=target_replace,
        train_method="full",
        delimiter="-",
    )
    network.to(device)
    if len(network.unet_loras) == 0:
        raise RuntimeError(f"LoRANetwork found 0 modules for {target_replace}")
    print(f"LoRA modules: {len(network.unet_loras)}")
    _set_lora_multiplier(network, 0.0)

    for param in transformer.parameters():
        param.requires_grad_(False)
    for param in network.parameters():
        param.requires_grad_(True)

    if args.xt_mode == "noise":
        x0_banks: list[list[torch.Tensor]] = []
    else:
        x0_banks = build_x0_banks(
            entries,
            transformer,
            cache_dir=cache_dir,
            model_dir=model_dir,
            device=device,
            seed=seed,
            per_row=int(args.x0_per_row),
            num_steps=int(args.x0_steps),
            dummy=bool(args.dummy),
        )

    optimizer = torch.optim.AdamW(network.parameters(), lr=lr)
    if lr_name == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(steps, 1), eta_min=1e-6)
    else:
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda _: 1.0)
    loss_fn = torch.nn.MSELoss()
    amp_enabled = device.type == "cuda" and not args.dummy
    amp_dtype = torch.bfloat16

    save_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{name}_alpha{alpha}_rank{rank}_{args.targets}"
    if args.dummy:
        stem = f"{stem}_dummy"
    log_path = save_dir / f"{stem}_train.jsonl"
    log_handle = log_path.open("w", encoding="utf-8")
    last_loss = None
    mid_step = max(1, steps // 2)
    progress = tqdm(range(steps), desc="train")
    for step in progress:
        prompt, conds = entries[step % len(entries)]
        x0_bank = x0_banks[step % len(entries)] if x0_banks else None
        batch = max(1, int(prompt.batch_size))
        latent_len = int(conds["target"].shape[1])
        noise = torch.randn(batch, 128, latent_len, device=device, dtype=torch.float32)
        # Model convention (denoise.py): flow time t in [0,1], 0 = noise, 1 = clean;
        # x_t = (1-t)·ε + t·x0. Pure noise is only on-manifold near t=0, so anchor
        # x_t to a generated clean latent except in the explicit noise/mix modes.
        use_noise = x0_bank is None or (args.xt_mode == "mix" and step % 10 < 3)
        if use_noise:
            hi = 0.35 if x0_bank is not None else 0.95
            timestep = torch.empty(batch, device=device, dtype=torch.float32).uniform_(0.02, hi)
            latents = noise
        else:
            timestep = torch.empty(batch, device=device, dtype=torch.float32).uniform_(0.02, 0.98)
            x0 = x0_bank[step % len(x0_bank)].to(device=device, dtype=torch.float32)
            if x0.shape[0] != batch:
                x0 = x0.expand(batch, -1, -1)
            t_mix = timestep.view(batch, 1, 1)
            latents = (1.0 - t_mix) * noise + t_mix * x0

        def _cond(name: str) -> torch.Tensor:
            tensor = conds[name].to(device=device, dtype=torch.float32)
            if tensor.shape[0] != batch:
                tensor = tensor.expand(batch, -1, -1).contiguous()
            return tensor

        cond_pos = _cond("positive")
        cond_neg = _cond("negative")
        cond_neu = _cond("neutral")
        cond_tgt = _cond("target")
        if amp_enabled:
            latents = latents.to(amp_dtype)
            timestep = timestep.to(amp_dtype)
            cond_pos = cond_pos.to(amp_dtype)
            cond_neg = cond_neg.to(amp_dtype)
            cond_neu = cond_neu.to(amp_dtype)
            cond_tgt = cond_tgt.to(amp_dtype)

        _set_lora_multiplier(network, 0.0)
        with torch.no_grad():
            if amp_enabled:
                with torch.autocast(device_type="cuda", dtype=amp_dtype):
                    vel_pos = transformer(latents, timestep, cond_pos, return_dict=False)[0]
                    vel_neg = transformer(latents, timestep, cond_neg, return_dict=False)[0]
                    vel_neu = transformer(latents, timestep, cond_neu, return_dict=False)[0]
            else:
                vel_pos = transformer(latents, timestep, cond_pos, return_dict=False)[0]
                vel_neg = transformer(latents, timestep, cond_neg, return_dict=False)[0]
                vel_neu = transformer(latents, timestep, cond_neu, return_dict=False)[0]

        def _lora_forward(slider: float) -> torch.Tensor:
            network.set_lora_slider(slider)
            with network:
                if amp_enabled:
                    with torch.autocast(device_type="cuda", dtype=amp_dtype):
                        return transformer(latents, timestep, cond_tgt, return_dict=False)[0]
                return transformer(latents, timestep, cond_tgt, return_dict=False)[0]

        vel_plus = _lora_forward(1.0)
        target_plus = _velocity_target(
            vel_neu=vel_neu.float(),
            vel_pos=vel_pos.float(),
            vel_neg=vel_neg.float(),
            guidance=float(prompt.guidance_scale),
            action=prompt.action,
        )
        loss = loss_fn(vel_plus.float(), target_plus)
        vel_minus = None
        if args.bidirectional:
            # -s is not the linear inverse of +s through 36 layers; train it explicitly.
            vel_minus = _lora_forward(-1.0)
            inverse_action = "erase" if prompt.action == "enhance" else "enhance"
            target_minus = _velocity_target(
                vel_neu=vel_neu.float(),
                vel_pos=vel_pos.float(),
                vel_neg=vel_neg.float(),
                guidance=float(prompt.guidance_scale),
                action=inverse_action,
            )
            loss = 0.5 * (loss + loss_fn(vel_minus.float(), target_minus))
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_value_(network.parameters(), clip_value=1.0)
        optimizer.step()
        scheduler.step()
        last_loss = float(loss.detach().cpu())
        with torch.no_grad():
            def _cos(a: torch.Tensor, b: torch.Tensor) -> float:
                return torch.nn.functional.cosine_similarity(
                    a.flatten().unsqueeze(0), b.flatten().unsqueeze(0)
                ).item()

            true_dir = (vel_pos.float() - vel_neg.float())
            delta_plus = vel_plus.float() - vel_neu.float()
            cos_pos = _cos(delta_plus, true_dir)
            cos_neg = None
            collapse = None
            if vel_minus is not None:
                delta_minus = vel_minus.float() - vel_neu.float()
                cos_neg = _cos(delta_minus, -true_dir)
                collapse = _cos(delta_plus, delta_minus)
        postfix = {"loss": f"{last_loss:.5f}", "cos": f"{cos_pos:.3f}"}
        if collapse is not None:
            postfix["col"] = f"{collapse:.2f}"
        progress.set_postfix(**postfix)
        print(
            f"step {step + 1}/{steps} loss={last_loss:.6f} cos={cos_pos:.4f}"
            + (
                f" cos_neg={cos_neg:.4f} collapse={collapse:.4f}"
                if cos_neg is not None
                else ""
            ),
            flush=True,
        )
        record = {"step": step + 1, "loss": last_loss, "cos": cos_pos}
        if cos_neg is not None:
            record["cos_neg"] = cos_neg
            record["collapse"] = collapse
        log_handle.write(json.dumps(record) + "\n")
        log_handle.flush()
        if (step + 1) == mid_step:
            mid_path = save_dir / f"{stem}_step{step + 1}.safetensors"
            network.save_weights(str(mid_path), dtype=torch.float32)
            print(f"saved mid checkpoint {mid_path}", flush=True)

    log_handle.close()
    weights_path = save_dir / f"{stem}_last.safetensors"
    network.save_weights(str(weights_path), dtype=torch.float32)
    meta = {
        "schema": 3,
        "name": name,
        "kind": "transformer",
        "prefix": "lora_unet",
        "rank": rank,
        "alpha": alpha,
        "steps": steps,
        "duration": duration,
        "dummy": bool(args.dummy),
        "target_replace": target_replace,
        "lm_weights": args.lm_weights,
        "delimiter": "-",
        "train_method": "full",
        "targets": args.targets,
        "xt_mode": args.xt_mode,
        "bidirectional": bool(args.bidirectional),
        "cond_seeds": cond_seeds,
        "x0_per_row": int(args.x0_per_row),
        "plus_label": prompts_meta.plus_label,
        "minus_label": prompts_meta.minus_label,
        "recommended_range": prompts_meta.recommended_range,
        "loss": last_loss,
        "prompts": [asdict(prompt) for prompt in prompts],
        "weights": str(weights_path),
    }
    meta_path = save_dir / f"{stem}_last.json"
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"saved {weights_path}")
    print(f"saved {meta_path}")
    if not args.dummy and not args.no_calibrate and entries:
        from conceptmod.textsliders.calibrate_scale import (
            calibrate_network,
            merge_calibration,
            print_table,
        )

        print("calibrating unit_scale on trained weights", flush=True)
        result = calibrate_network(
            transformer,
            network,
            entries[0][1],
            guidance=float(entries[0][0].guidance_scale),
            device=device,
            seed=seed,
        )
        print_table(result)
        meta = merge_calibration(meta, result)
        meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
        print(f"updated {meta_path} unit_scale={result['unit_scale']:g}", flush=True)
    return weights_path


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", type=str, default="energy")
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--alpha", type=float, default=8.0)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--duration", type=float, default=4.0)
    parser.add_argument("--config_file", type=str, default=str(DEFAULT_CONFIG))
    parser.add_argument("--prompts_file", type=str, default=str(DEFAULT_PROMPTS))
    parser.add_argument("--model_dir", type=str, default=str(DEFAULT_MODEL_DIR))
    parser.add_argument("--cache_dir", type=str, default=str(DEFAULT_CACHE_DIR))
    parser.add_argument("--save_dir", type=str, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--skip_ar", action="store_true", help="use existing condition cache only")
    parser.add_argument(
        "--targets",
        choices=["attn", "full"],
        default="attn",
        help="attn=MiniMaxMusic3Attention only; full also LoRAs proj_in/FF/convs",
    )
    parser.add_argument(
        "--lm_weights",
        type=str,
        default=None,
        help="if set, build pos/neg conditions by running AR with this LM slider ±1 on the neutral caption",
    )
    parser.add_argument("--guidance", type=float, default=None, help="override prompt guidance_scale")
    parser.add_argument(
        "--xt_mode",
        choices=["anchor", "mix", "noise"],
        default="anchor",
        help="anchor=x_t from generated clean latents; mix=30%% pure-noise steps; noise=legacy pure randn",
    )
    parser.add_argument(
        "--bidirectional",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="also train the -1 direction against vel_neu - g*(vel_pos - vel_neg)",
    )
    parser.add_argument(
        "--cond_seeds",
        type=str,
        default=None,
        help="comma-separated AR seeds; each prompt row gets one condition set per seed (default: --seed)",
    )
    parser.add_argument("--x0_per_row", type=int, default=2, help="clean-latent anchors per condition set")
    parser.add_argument("--x0_steps", type=int, default=30, help="Euler steps when generating x0 anchors")
    parser.add_argument("--plus_label", type=str, default=None, help="override sidecar plus_label")
    parser.add_argument("--minus_label", type=str, default=None, help="override sidecar minus_label")
    parser.add_argument(
        "--no_calibrate",
        action="store_true",
        help="skip writing unit_scale into the sidecar after the last step",
    )
    parser.add_argument(
        "--dummy",
        action="store_true",
        help="random conditions + tiny transformer, 2 steps, never loads the 8B LM",
    )
    return parser.parse_args(argv)


def main() -> None:
    train(parse_args())


if __name__ == "__main__":
    main()
