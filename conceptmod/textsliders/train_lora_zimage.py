#!/usr/bin/env python3
"""Opt-in Z-Image Turbo (ZiT) image-slider trainer.

UNI analog — not Music 3 lyric-hold, not Anima / Krea / H3:

- +1 → + concept prompt velocity
- scale 0 → neutral prompt velocity
- no minus teacher (canary only)
- hold unused prompt tokens to encode(neu); do not hold concept words

Velocity-space CFG geometry from conceptmod ``backends/zimage.py``:

    v(z, t, c) − v(z, t, '')

Model card: ``Tongyi-MAI/Z-Image-Turbo``, 6B, LoRA 16, 768px, sample
8 steps, CFG 0. Dummy mode never loads Hub weights. The default Music 3
trainer is unchanged.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable

import torch
import torch.nn as nn
import yaml
from tqdm.auto import tqdm

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from conceptmod.textsliders.lora import LoRANetwork
from conceptmod.textsliders.slider_targets import (
    ZImageHoldError,
    expand_attributes_zimage,
    zimage_canary_minus,
    zimage_cfg,
    zimage_cfg_delta,
    zimage_concept_token_ids,
    zimage_uni_loss,
    zimage_uni_teachers,
    zimage_unused_token_hold,
)

DEFAULT_MODEL_ID = "Tongyi-MAI/Z-Image-Turbo"
DEFAULT_CONFIG = Path(__file__).resolve().parent / "data" / "config-zimage.yaml"
DEFAULT_PROMPTS = Path(__file__).resolve().parent / "data" / "prompts-zimage.yaml"
DEFAULT_SAVE_DIR = Path("models/zimage-slider")
TARGET_REPLACE = ["Attention"]
DUMMY_VOCAB = {
    "<pad>": 0,
    "a": 1,
    "an": 2,
    "person": 3,
    "old": 4,
    "young": 5,
    "male": 6,
    "female": 7,
    "portrait": 8,
    "of": 9,
    "elderly": 10,
    "aged": 11,
}


@dataclass
class SliderPrompt:
    target: str
    positive: str
    negative: str
    neutral: str
    action: str = "enhance"
    guidance_scale: float = 0.0
    resolution: int = 768
    batch_size: int = 1
    concept_words: str = ""


@dataclass
class PromptsMeta:
    plus_label: str = ""
    minus_label: str = ""
    recommended_range: list[float] = field(default_factory=lambda: [0.0, 2.0])
    concept_words: str = ""


def _load_yaml(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def load_prompts(path: Path) -> tuple[list[SliderPrompt], PromptsMeta]:
    raw = _load_yaml(path)
    meta = PromptsMeta()
    if isinstance(raw, dict):
        meta.plus_label = str(raw.get("plus_label") or "")
        meta.minus_label = str(raw.get("minus_label") or "")
        rng = raw.get("recommended_range")
        if isinstance(rng, (list, tuple)) and len(rng) == 2:
            meta.recommended_range = [float(rng[0]), float(rng[1])]
        meta.concept_words = str(raw.get("concept_words") or "")
        raw = raw.get("rows")
    if not isinstance(raw, list) or not raw:
        raise ValueError(f"prompts file is empty: {path}")
    prompts: list[SliderPrompt] = []
    for item in raw:
        if not isinstance(item, dict) or "target" not in item:
            raise ValueError(f"each prompt must be a mapping with target: {item!r}")
        row_concept = str(item.get("concept_words") or meta.concept_words)
        for row in expand_attributes_zimage(item):
            target = str(row["target"])
            prompts.append(
                SliderPrompt(
                    target=target,
                    positive=str(row.get("positive") or target),
                    negative=str(row.get("negative") or ""),
                    neutral=str(row.get("neutral") or target),
                    action=str(row.get("action") or "enhance"),
                    guidance_scale=float(row.get("guidance_scale", 0.0)),
                    resolution=int(row.get("resolution", 768)),
                    batch_size=int(row.get("batch_size", 1)),
                    concept_words=str(row.get("concept_words") or row_concept),
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


def dummy_tokenize(text: str) -> list[int]:
    """Whitespace tokenizer. Never talks to Hub."""
    ids: list[int] = []
    for word in str(text).lower().replace(",", " ").split():
        ids.append(DUMMY_VOCAB.get(word, 1))
    return ids or [0]


class Attention(nn.Module):
    """Class name LoRANetwork matches (``to_q`` / ``to_k`` / ``to_v`` / ``to_out.0``)."""

    def __init__(self, dim: int = 32) -> None:
        super().__init__()
        self.to_q = nn.Linear(dim, dim)
        self.to_k = nn.Linear(dim, dim)
        self.to_v = nn.Linear(dim, dim)
        self.to_out = nn.ModuleList([nn.Linear(dim, dim)])

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        q = self.to_q(hidden)
        k = self.to_k(hidden)
        v = self.to_v(hidden)
        return self.to_out[0]((q + k + v) / 3.0)


class DummyZImageTransformer(nn.Module):
    """Tiny flow stand-in. Velocity is a linear map of z plus text mean."""

    def __init__(self, dim: int = 32, channels: int = 4, spatial: int = 4) -> None:
        super().__init__()
        self.attn = Attention(dim)
        self.in_proj = nn.Linear(channels, dim)
        self.out_proj = nn.Linear(dim, channels)
        self.text_proj = nn.Linear(dim, dim)
        self.channels = channels
        self.spatial = spatial
        self.embed_dim = dim

    def forward(
        self, z: torch.Tensor, timestep: torch.Tensor, embeds: torch.Tensor
    ) -> torch.Tensor:
        b, c, h, w = z.shape
        flat = z.reshape(b, c, h * w).transpose(1, 2)
        hidden = self.in_proj(flat)
        text = self.text_proj(embeds.mean(dim=1, keepdim=True))
        t_scale = timestep.reshape(b, 1, 1).to(hidden.dtype)
        hidden = self.attn(hidden + text + t_scale)
        out = self.out_proj(hidden).transpose(1, 2).reshape(b, c, h, w)
        return out


class DummyTextEncoder(nn.Module):
    def __init__(self, dim: int = 32, vocab: int = 32) -> None:
        super().__init__()
        self.emb = nn.Embedding(vocab, dim)

    def encode(self, token_ids: list[int]) -> torch.Tensor:
        ids = torch.tensor(token_ids or [0], dtype=torch.long)
        return self.emb(ids)


def _set_lora_multiplier(network: LoRANetwork, value: float) -> None:
    for lora in network.unet_loras:
        lora.multiplier = value


def _pick_device(index: int, dummy: bool) -> torch.device:
    if dummy or not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(f"cuda:{index}")


class _LiveZImage:
    """Thin wrapper around conceptmod's Z-Image Turbo velocity convention."""

    def __init__(self, model_id: str, device: torch.device, resolution: int) -> None:
        from diffusers import ZImagePipeline

        self.pipe = ZImagePipeline.from_pretrained(model_id, torch_dtype=torch.bfloat16)
        self.pipe.to(str(device))
        self.pipe.set_progress_bar_config(disable=True)
        self.device = device
        self.resolution = resolution
        self.transformer = self.pipe.transformer
        self.transformer.requires_grad_(False)
        self.transformer.eval()
        if hasattr(self.transformer, "enable_gradient_checkpointing"):
            self.transformer.enable_gradient_checkpointing()

    def tokenize(self, text: str) -> list[int]:
        tokenizer = getattr(self.pipe, "tokenizer", None)
        if tokenizer is None:
            raise RuntimeError("Z-Image pipeline has no tokenizer")
        encoded = tokenizer(text, add_special_tokens=False)
        ids = encoded["input_ids"] if isinstance(encoded, dict) else encoded
        if ids and isinstance(ids[0], list):
            ids = ids[0]
        return [int(t) for t in ids]

    def encode_text(self, prompt: str) -> torch.Tensor:
        embeds_list = self.pipe._encode_prompt(prompt, device=self.device)
        return embeds_list[0]

    def predict_v(
        self, embeds: torch.Tensor, z: torch.Tensor, timestep: torch.Tensor
    ) -> torch.Tensor:
        dtype = torch.bfloat16
        t = timestep.expand(z.shape[0]).to(self.device).float()
        t_model = (1000.0 - t) / 1000.0
        latents = list(z.to(dtype).unsqueeze(2).unbind(dim=0))
        text = [embeds.to(dtype)]
        out = self.transformer(latents, t_model, text, return_dict=False)[0]
        return -torch.stack([o.squeeze(1) for o in out], dim=0).float()


def _dummy_bundle(device: torch.device) -> tuple[nn.Module, DummyTextEncoder, Callable[[str], list[int]]]:
    transformer = DummyZImageTransformer().to(device)
    transformer.requires_grad_(False)
    transformer.eval()
    encoder = DummyTextEncoder().to(device)
    encoder.requires_grad_(False)
    return transformer, encoder, dummy_tokenize


def train(args: argparse.Namespace) -> Path:
    config = load_config_defaults(Path(args.config_file) if args.config_file else None)
    prompts, meta = load_prompts(Path(args.prompts_file))
    if args.guidance is not None:
        for prompt in prompts:
            prompt.guidance_scale = float(args.guidance)
    dummy = bool(args.dummy)
    steps = 2 if dummy else int(args.steps)
    rank = int(args.rank)
    alpha = float(args.alpha)
    resolution = int(args.resolution)
    device = _pick_device(int(args.device), dummy)
    save_dir = Path(args.save_dir or DEFAULT_SAVE_DIR)
    save_dir.mkdir(parents=True, exist_ok=True)
    name = str(args.name)
    stem = f"{name}_alpha{alpha}_rank{rank}_uni"

    torch.manual_seed(int(args.seed))
    random.seed(int(args.seed))

    encoder: DummyTextEncoder | None = None
    live: _LiveZImage | None = None
    tokenize_fn: Callable[[str], list[int]]
    if dummy:
        transformer, encoder, tokenize_fn = _dummy_bundle(device)
        latent_shape = (transformer.channels, transformer.spatial, transformer.spatial)
        print(
            "dummy ZiT trainer: tiny Attention + hash tokenizer, no Hub, no GPU weights",
            flush=True,
        )
    else:
        live = _LiveZImage(str(args.model_id), device, resolution)
        transformer = live.transformer
        tokenize_fn = live.tokenize
        vae_scale = getattr(live.pipe, "vae_scale_factor", 16)
        h = 2 * (resolution // (vae_scale * 2))
        latent_shape = (live.transformer.config.in_channels, h, h)

    network = LoRANetwork(
        transformer,
        rank=rank,
        alpha=alpha,
        multiplier=1.0,
        target_replace=TARGET_REPLACE,
        train_method="full",
        delimiter="-",
        prefix="lora_unet",
    ).to(device)
    if not network.unet_loras:
        raise RuntimeError("LoRANetwork wrapped 0 Attention modules")
    optimizer = torch.optim.AdamW(network.parameters(), lr=float(args.lr), weight_decay=1e-6)
    print(
        f"zimage UNI modules={len(network.unet_loras)} rank={rank} alpha={alpha} "
        f"res={resolution} steps={steps} cfg={args.sample_guidance} dummy={dummy}",
        flush=True,
    )

    embed_cache: dict[str, torch.Tensor] = {}
    id_cache: dict[str, list[int]] = {}

    def _ids(text: str) -> list[int]:
        if text not in id_cache:
            id_cache[text] = tokenize_fn(text)
        return id_cache[text]

    def _encode(text: str) -> torch.Tensor:
        if text not in embed_cache:
            if dummy:
                assert encoder is not None
                embed_cache[text] = encoder.encode(_ids(text)).detach()
            else:
                assert live is not None
                embed_cache[text] = live.encode_text(text).detach()
        return embed_cache[text]

    def _predict(embeds: torch.Tensor, z: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        if dummy:
            return transformer(z, t, embeds.unsqueeze(0).expand(z.shape[0], -1, -1))
        assert live is not None
        return live.predict_v(embeds, z, t)

    log_path = save_dir / f"{stem}.jsonl"
    log_handle = log_path.open("w", encoding="utf-8")
    last_canary: dict[str, float | bool] | None = None
    progress = tqdm(range(steps), disable=dummy)
    for step in progress:
        prompt = prompts[step % len(prompts)]
        guidance = float(args.sample_guidance if args.sample_guidance is not None else prompt.guidance_scale)
        z = torch.randn((1, *latent_shape), device=device)
        t = torch.full((1,), 500.0 if not dummy else 0.5, device=device)
        emb_pos = _encode(prompt.positive).to(device)
        emb_neu = _encode(prompt.neutral).to(device)
        emb_uncond = _encode("").to(device)
        ids_pos = _ids(prompt.positive)
        ids_neu = _ids(prompt.neutral)
        concept_ids = zimage_concept_token_ids(prompt.concept_words, tokenize_fn)
        _set_lora_multiplier(network, 0.0)
        with torch.no_grad():
            vel_pos = _predict(emb_pos, z, t).float()
            vel_neu = _predict(emb_neu, z, t).float()
            vel_uncond = _predict(emb_uncond, z, t).float()
            tgt_plus, tgt_zero = zimage_uni_teachers(
                vel_pos, vel_neu, vel_uncond, guidance=guidance
            )
            vel_neg = None
            if prompt.negative:
                vel_neg = _predict(_encode(prompt.negative).to(device), z, t).float()

        token_hold = None
        try:
            token_hold = zimage_unused_token_hold(
                emb_pos, emb_neu, ids_pos, ids_neu, concept_ids
            )
        except ZImageHoldError as exc:
            if not dummy:
                raise
            print(f"dummy skip token hold: {exc}", flush=True)

        def _student(scale: float, embeds: torch.Tensor) -> torch.Tensor:
            network.set_lora_slider(scale)
            with network:
                return _predict(embeds, z, t).float()

        pred_plus = _student(1.0, emb_pos)
        pred_zero = _student(0.0, emb_neu)
        pred_unused = _student(1.0, emb_neu)
        loss = zimage_uni_loss(
            pred_plus,
            tgt_plus,
            pred_zero,
            tgt_zero,
            pred_unused=pred_unused,
            tgt_unused=tgt_zero,
            unused_weight=float(args.unused_weight),
            unused_token_hold=token_hold,
            token_hold_weight=float(args.token_hold_weight),
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(network.parameters(), max_norm=1.0)
        optimizer.step()

        record: dict[str, Any] = {
            "step": step + 1,
            "loss": float(loss.detach().cpu()),
            "cfg_delta_norm": float(zimage_cfg_delta(vel_pos, vel_uncond).norm().cpu()),
            "canary_scored": False,
        }
        if vel_neg is not None:
            with torch.no_grad():
                pred_minus = _student(-1.0, emb_pos)
                last_canary = zimage_canary_minus(pred_minus, vel_neg)
                record["canary"] = last_canary
        log_handle.write(json.dumps(record) + "\n")
        log_handle.flush()
        progress.set_postfix(loss=f"{record['loss']:.5f}")

    log_handle.close()
    weights_path = save_dir / f"{stem}_last.safetensors"
    network.save_weights(str(weights_path), dtype=torch.float32)
    sidecar = {
        "schema": 1,
        "name": name,
        "kind": "zimage",
        "recipe": "uni",
        "model_id": DEFAULT_MODEL_ID if dummy else str(args.model_id),
        "rank": rank,
        "alpha": alpha,
        "resolution": resolution,
        "sample_steps": int(args.sample_steps),
        "sample_guidance": float(args.sample_guidance),
        "steps": steps,
        "dummy": dummy,
        "plus_label": args.plus_label or meta.plus_label,
        "minus_label": args.minus_label or meta.minus_label,
        "recommended_range": meta.recommended_range,
        "concept_words": meta.concept_words,
        "teacher": {
            "plus": "+ concept prompt velocity",
            "zero": "neutral prompt velocity",
            "minus": "canary only",
            "unused_hold": "unused prompt tokens → encode(neu); concept words not held",
            "cfg": "v(z,t,c) - v(z,t,'')",
        },
        "canary": last_canary,
        "weights": str(weights_path),
        "config": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
        "prompts_meta": asdict(meta),
    }
    sidecar_path = save_dir / f"{stem}_last.json"
    sidecar_path.write_text(json.dumps(sidecar, indent=2), encoding="utf-8")
    print(f"saved {weights_path}", flush=True)
    print(f"sidecar {sidecar_path}", flush=True)
    return weights_path


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", type=str, default="age-zit")
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--alpha", type=float, default=16.0)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--resolution", type=int, default=768)
    parser.add_argument("--sample_steps", type=int, default=8)
    parser.add_argument("--sample_guidance", type=float, default=0.0)
    parser.add_argument("--guidance", type=float, default=None, help="override yaml guidance_scale")
    parser.add_argument("--unused_weight", type=float, default=1.0)
    parser.add_argument("--token_hold_weight", type=float, default=1.0)
    parser.add_argument("--model_id", type=str, default=DEFAULT_MODEL_ID)
    parser.add_argument("--config_file", type=str, default=str(DEFAULT_CONFIG))
    parser.add_argument("--prompts_file", type=str, default=str(DEFAULT_PROMPTS))
    parser.add_argument("--save_dir", type=str, default=str(DEFAULT_SAVE_DIR))
    parser.add_argument("--plus_label", type=str, default=None)
    parser.add_argument("--minus_label", type=str, default=None)
    parser.add_argument(
        "--dummy",
        action="store_true",
        help="tiny Attention + whitespace tokenizer, 2 steps, never loads Hub weights",
    )
    return parser.parse_args(argv)


def main() -> None:
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    train(parse_args())


if __name__ == "__main__":
    main()
