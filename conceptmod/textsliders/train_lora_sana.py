#!/usr/bin/env python3
"""Opt-in Sana 0.6B image-slider trainer (cheap test backend).

UNI analog — not Music 3 lyric-hold, not ZiT / Krea / Anima / H3:

- +1 → + concept prompt (CFG-composed at 4.5)
- scale 0 → neutral prompt
- no minus teacher (canary only)
- hold unused prompt tokens / pins to encode(neu); do not hold concept words

Velocity-space CFG geometry from conceptmod ``backends/sana.py``:

    direction(c) = v(z, t, c) − v(z, t, '')
    v_cfg        = v('') + 4.5 · (v(c) − v(''))     # g != 1
                 = v(c)                             # g == 1

CFG 4.5 is live. Default train method is ``xattn`` (conceptmod 0.6B).
``--lora RANK`` trains a peft-style LoRA on attn ``to_q/to_k/to_v/to_out.0``
instead. Dummy mode never loads Hub weights. The default Music 3 trainer
is unchanged.

Hub: ``Efficient-Large-Model/Sana_600M_512px_diffusers``.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

import torch
import torch.nn as nn
import yaml
from tqdm.auto import tqdm

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from conceptmod.textsliders.lora import LoRANetwork
from conceptmod.textsliders.slider_targets import (
    SANA_CFG,
    SANA_CONTROL_PROMPT,
    SANA_DEFAULT_LR,
    SANA_DEFAULT_STEPS,
    SANA_MODEL_ID,
    SANA_RESOLUTION,
    SANA_SAMPLE_STEPS,
    SANA_TRAIN_METHOD,
    SanaHoldError,
    expand_attributes_sana,
    sana_canary_minus,
    sana_cfg_delta,
    sana_concept_token_ids,
    sana_live_train_card,
    sana_live_train_command,
    sana_uni_loss,
    sana_uni_teachers,
    sana_unused_token_hold,
    zimage_require_concept_in_prompt,
)

DEFAULT_CONFIG = Path(__file__).resolve().parent / "data" / "config-sana.yaml"
DEFAULT_PROMPTS = Path(__file__).resolve().parent / "data" / "prompts-sana.yaml"
DEFAULT_SAVE_DIR = Path("models/sana-slider")
TARGET_REPLACE = ["Attention"]
_FOREIGN_BACKENDS = ("z-image", "zimage", "zit", "krea", "anima", "minimax", "h3")
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
    "bowl": 12,
    "fruit": 13,
    "on": 14,
    "table": 15,
}


@dataclass
class SliderPrompt:
    target: str
    positive: str
    negative: str
    neutral: str
    action: str = "enhance"
    guidance_scale: float = SANA_CFG
    resolution: int = SANA_RESOLUTION
    batch_size: int = 1
    concept_words: str = ""


@dataclass
class PromptsMeta:
    plus_label: str = ""
    minus_label: str = ""
    recommended_range: list[float] = field(default_factory=lambda: [0.0, 2.0])
    concept_words: str = ""
    control_prompt: str = SANA_CONTROL_PROMPT


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
        meta.control_prompt = str(raw.get("control_prompt") or SANA_CONTROL_PROMPT)
        raw = raw.get("rows")
    if not isinstance(raw, list) or not raw:
        raise ValueError(f"prompts file is empty: {path}")
    prompts: list[SliderPrompt] = []
    for item in raw:
        if not isinstance(item, dict) or "target" not in item:
            raise ValueError(f"each prompt must be a mapping with target: {item!r}")
        row_concept = str(item.get("concept_words") or meta.concept_words)
        for row in expand_attributes_sana(item):
            target = str(row["target"])
            prompts.append(
                SliderPrompt(
                    target=target,
                    positive=str(row.get("positive") or target),
                    negative=str(row.get("negative") or ""),
                    neutral=str(row.get("neutral") or target),
                    action=str(row.get("action") or "enhance"),
                    guidance_scale=float(row.get("guidance_scale", SANA_CFG)),
                    resolution=int(row.get("resolution", SANA_RESOLUTION)),
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


def assert_sana_only(model_id: str) -> None:
    lowered = str(model_id).lower()
    for name in _FOREIGN_BACKENDS:
        if name in lowered:
            raise ValueError(
                f"this trainer is Sana-only; refused foreign backend {name!r}"
            )


class Attention(nn.Module):
    """Class name LoRANetwork matches (``to_q`` / ``to_k`` / ``to_v`` / ``to_out.0``)."""

    def __init__(self, dim: int = 32) -> None:
        super().__init__()
        self.to_q = nn.Linear(dim, dim)
        self.to_k = nn.Linear(dim, dim)
        self.to_v = nn.Linear(dim, dim)
        self.to_out = nn.ModuleList([nn.Linear(dim, dim)])

    def forward(self, hidden: torch.Tensor, context: torch.Tensor | None = None) -> torch.Tensor:
        ctx = hidden if context is None else context
        q = self.to_q(hidden)
        k = self.to_k(ctx)
        v = self.to_v(ctx)
        return self.to_out[0]((q + k + v) / 3.0)


class DummySanaTransformer(nn.Module):
    """Tiny flow stand-in with ``attn1`` (self) and ``attn2`` (xattn).

    conceptmod Sana ``xattn`` trains names containing ``attn2``.
    """

    def __init__(self, dim: int = 32, channels: int = 4, spatial: int = 4) -> None:
        super().__init__()
        self.attn1 = Attention(dim)
        self.attn2 = Attention(dim)
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
        hidden = self.attn1(hidden + t_scale)
        hidden = self.attn2(hidden, text.expand_as(hidden))
        return self.out_proj(hidden).transpose(1, 2).reshape(b, c, h, w)


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


def _xattn_parameters(model: nn.Module) -> list[nn.Parameter]:
    params: list[nn.Parameter] = []
    for name, param in model.named_parameters():
        if "attn2" in name:
            param.requires_grad_(True)
            params.append(param)
        else:
            param.requires_grad_(False)
    if not params:
        raise RuntimeError("train_method xattn selected no attn2 parameters")
    return params


class _LiveSana:
    """Thin wrapper around conceptmod's Sana 0.6B velocity convention."""

    def __init__(self, model_id: str, device: torch.device, resolution: int) -> None:
        from diffusers import SanaPipeline

        self.pipe = SanaPipeline.from_pretrained(model_id, torch_dtype=torch.bfloat16)
        self.pipe.to(str(device))
        self.pipe.set_progress_bar_config(disable=True)
        self.device = device
        self.resolution = resolution
        self.transformer = self.pipe.transformer
        self.transformer.requires_grad_(False)
        self.transformer.eval()
        self.transformer.to(torch.float32)
        self.frozen = None

    def enable_xattn(self) -> list[nn.Parameter]:
        import copy

        self.frozen = copy.deepcopy(self.transformer)
        self.frozen.requires_grad_(False)
        self.frozen.eval()
        return _xattn_parameters(self.transformer)

    def tokenize(self, text: str) -> list[int]:
        tokenizer = getattr(self.pipe, "tokenizer", None)
        if tokenizer is None:
            raise RuntimeError("Sana pipeline has no tokenizer")
        encoded = tokenizer(text, add_special_tokens=False)
        ids = getattr(encoded, "input_ids", None)
        if ids is None:
            if isinstance(encoded, Mapping):
                ids = encoded["input_ids"]
            else:
                ids = encoded
        if hasattr(ids, "tolist"):
            ids = ids.tolist()
        if ids and isinstance(ids[0], (list, tuple)):
            ids = ids[0]
        return [int(t) for t in ids]

    def encode_text(self, prompt: str) -> torch.Tensor:
        embeds, _mask, _, _ = self.pipe.encode_prompt(
            prompt,
            do_classifier_free_guidance=False,
            device=self.device,
            clean_caption=False,
            complex_human_instruction=None,
        )
        return embeds[0] if embeds.ndim == 3 and embeds.shape[0] == 1 else embeds

    def _forward(self, model, z: torch.Tensor, timestep: torch.Tensor, embeds: torch.Tensor):
        cfg = model.config
        t = timestep.expand(z.shape[0]).to(self.device)
        t = t * getattr(cfg, "timestep_scale", 1.0)
        hidden = embeds if embeds.ndim == 3 else embeds.unsqueeze(0)
        mask = torch.ones(hidden.shape[:2], device=self.device, dtype=torch.long)
        v = model(
            z.to(next(model.parameters()).dtype),
            encoder_hidden_states=hidden.to(next(model.parameters()).dtype),
            encoder_attention_mask=mask,
            timestep=t,
            return_dict=False,
        )[0]
        latent_channels = cfg.in_channels
        if getattr(cfg, "out_channels", latent_channels) // 2 == latent_channels:
            v = v.chunk(2, dim=1)[0]
        return v.float()

    def predict_v(
        self, embeds: torch.Tensor, z: torch.Tensor, timestep: torch.Tensor, *, frozen: bool = False
    ) -> torch.Tensor:
        model = self.frozen if frozen and self.frozen is not None else self.transformer
        if frozen:
            with torch.no_grad():
                return self._forward(model, z, timestep, embeds)
        return self._forward(model, z, timestep, embeds)


def resolve_sana_concept_ids(
    plus_ids: Sequence[int],
    neu_ids: Sequence[int],
    concept_words: str | Iterable[str],
    tokenize_fn: Callable[[str], list[int]],
) -> set[int]:
    """Declared concept ids, or the plus−neu support split if BPE missed.

    Gemma/Sana pieces for a word in a sentence often differ from the bare
    word. ``sana_concept_token_ids`` already adds ``" " + word``. If those
    ids still miss the + prompt, use ``set(plus) - set(neu)``. Fail only
    when that fallback is empty.
    """
    concept_ids = sana_concept_token_ids(concept_words, tokenize_fn)
    plus = {int(t) for t in plus_ids}
    banned = {int(t) for t in concept_ids}
    if banned and any(tid in banned for tid in plus):
        return banned
    fallback = plus - {int(t) for t in neu_ids}
    if fallback:
        return fallback
    zimage_require_concept_in_prompt(plus_ids, concept_ids)
    return banned


def _dummy_bundle(
    device: torch.device,
) -> tuple[DummySanaTransformer, DummyTextEncoder, Callable[[str], list[int]]]:
    transformer = DummySanaTransformer().to(device)
    transformer.requires_grad_(False)
    transformer.eval()
    encoder = DummyTextEncoder().to(device)
    encoder.requires_grad_(False)
    return transformer, encoder, dummy_tokenize


def train(args: argparse.Namespace) -> Path:
    load_config_defaults(Path(args.config_file) if args.config_file else None)
    assert_sana_only(str(args.model_id))
    prompts, meta = load_prompts(Path(args.prompts_file))
    if args.guidance is not None:
        for prompt in prompts:
            prompt.guidance_scale = float(args.guidance)
    dummy = bool(args.dummy)
    steps = 2 if dummy else int(args.steps)
    lora_rank = args.lora
    use_lora = lora_rank is not None
    train_method = str(args.train_method)
    if use_lora and train_method == "lora":
        train_method = "xattn"
    resolution = int(args.resolution)
    device = _pick_device(int(args.device), dummy)
    save_dir = Path(args.save_dir or DEFAULT_SAVE_DIR)
    save_dir.mkdir(parents=True, exist_ok=True)
    name = str(args.name)
    method_tag = f"lora{lora_rank}" if use_lora else train_method
    stem = f"{name}_{method_tag}_uni"
    control_prompt = str(args.control_prompt or meta.control_prompt or SANA_CONTROL_PROMPT)

    torch.manual_seed(int(args.seed))
    random.seed(int(args.seed))

    encoder: DummyTextEncoder | None = None
    live: _LiveSana | None = None
    network: LoRANetwork | None = None
    frozen_dummy: DummySanaTransformer | None = None
    tokenize_fn: Callable[[str], list[int]]
    if dummy:
        transformer, encoder, tokenize_fn = _dummy_bundle(device)
        latent_shape = (transformer.channels, transformer.spatial, transformer.spatial)
        print(
            "dummy Sana trainer: tiny attn1/attn2 + hash tokenizer, no Hub, no GPU weights",
            flush=True,
        )
    else:
        live = _LiveSana(str(args.model_id), device, resolution)
        transformer = live.transformer
        tokenize_fn = live.tokenize
        vae_scale = getattr(live.pipe, "vae_scale_factor", 32)
        patch = getattr(getattr(transformer, "config", None), "patch_size", 1)
        scale = vae_scale * patch
        h = resolution // scale
        latent_shape = (int(getattr(transformer.config, "in_channels", 32)), h, h)

    if use_lora:
        network = LoRANetwork(
            transformer,
            rank=int(lora_rank),
            alpha=float(args.alpha if args.alpha is not None else lora_rank),
            multiplier=1.0,
            target_replace=TARGET_REPLACE,
            train_method=train_method,
            delimiter="-",
            prefix="lora_unet",
        ).to(device)
        if not network.unet_loras:
            raise RuntimeError("LoRANetwork wrapped 0 Attention modules")
        trainable = list(network.parameters())
    else:
        if dummy:
            import copy

            frozen_dummy = copy.deepcopy(transformer)
            frozen_dummy.requires_grad_(False)
            frozen_dummy.eval()
            trainable = _xattn_parameters(transformer)
        else:
            assert live is not None
            trainable = live.enable_xattn()
    optimizer = torch.optim.AdamW(trainable, lr=float(args.lr), weight_decay=1e-6)
    print(
        f"sana UNI method={train_method} lora={lora_rank} res={resolution} "
        f"steps={steps} cfg={args.sample_guidance} dummy={dummy} "
        f"control={control_prompt!r}",
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

    def _predict(
        embeds: torch.Tensor, z: torch.Tensor, t: torch.Tensor, *, frozen: bool = False
    ) -> torch.Tensor:
        if dummy:
            model = frozen_dummy if frozen and frozen_dummy is not None else transformer
            text = embeds.unsqueeze(0).expand(z.shape[0], -1, -1)
            return model(z, t, text)
        assert live is not None
        return live.predict_v(embeds, z, t, frozen=frozen)

    log_path = save_dir / f"{stem}.jsonl"
    log_handle = log_path.open("w", encoding="utf-8")
    last_canary: dict[str, float | bool] | None = None
    progress = tqdm(range(steps), disable=dummy)
    for step in progress:
        prompt = prompts[step % len(prompts)]
        guidance = float(
            args.sample_guidance if args.sample_guidance is not None else prompt.guidance_scale
        )
        z = torch.randn((1, *latent_shape), device=device)
        t = torch.full((1,), 500.0 if not dummy else 0.5, device=device)
        emb_pos = _encode(prompt.positive).to(device)
        emb_neu = _encode(prompt.neutral).to(device)
        emb_uncond = _encode("").to(device)
        ids_pos = _ids(prompt.positive)
        ids_neu = _ids(prompt.neutral)
        concept_ids = resolve_sana_concept_ids(
            ids_pos, ids_neu, prompt.concept_words, tokenize_fn
        )
        if network is not None:
            _set_lora_multiplier(network, 0.0)
        with torch.no_grad():
            vel_pos = _predict(emb_pos, z, t, frozen=True).float()
            vel_neu = _predict(emb_neu, z, t, frozen=True).float()
            vel_uncond = _predict(emb_uncond, z, t, frozen=True).float()
            tgt_plus, tgt_zero = sana_uni_teachers(
                vel_pos, vel_neu, vel_uncond, guidance=guidance
            )
            vel_neg = None
            if prompt.negative:
                vel_neg = _predict(_encode(prompt.negative).to(device), z, t, frozen=True).float()

        token_hold = None
        try:
            token_hold = sana_unused_token_hold(
                emb_pos, emb_neu, ids_pos, ids_neu, concept_ids
            )
        except SanaHoldError as exc:
            if not dummy:
                raise
            print(f"dummy skip token hold: {exc}", flush=True)

        def _student(scale: float, embeds: torch.Tensor) -> torch.Tensor:
            if network is not None:
                network.set_lora_slider(scale)
                with network:
                    return _predict(embeds, z, t, frozen=False).float()
            # xattn: scale 0 is the frozen copy; +1 is the trained transformer.
            return _predict(embeds, z, t, frozen=scale == 0.0).float()

        pred_plus = _student(1.0, emb_pos)
        pred_zero = _student(0.0, emb_neu)
        pred_unused = _student(1.0, emb_neu)
        loss = sana_uni_loss(
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
        torch.nn.utils.clip_grad_norm_(trainable, max_norm=1.0)
        optimizer.step()

        record: dict[str, Any] = {
            "step": step + 1,
            "loss": float(loss.detach().cpu()),
            "cfg_delta_norm": float(sana_cfg_delta(vel_pos, vel_uncond).norm().cpu()),
            "canary_scored": False,
            "control_prompt": control_prompt,
        }
        if vel_neg is not None:
            with torch.no_grad():
                pred_minus = _student(-1.0 if network is not None else 1.0, emb_pos)
                last_canary = sana_canary_minus(pred_minus, vel_neg)
                record["canary"] = last_canary
        log_handle.write(json.dumps(record) + "\n")
        log_handle.flush()
        progress.set_postfix(loss=f"{record['loss']:.5f}")

    log_handle.close()
    weights_path = save_dir / f"{stem}_last.safetensors"
    if network is not None:
        network.save_weights(str(weights_path), dtype=torch.float32)
    else:
        from safetensors.torch import save_file

        trained = {
            k: v.detach().contiguous().cpu()
            for k, v in transformer.state_dict().items()
            if "attn2" in k
        }
        save_file(trained, str(weights_path))
    sidecar = {
        "schema": 1,
        "name": name,
        "kind": "sana",
        "recipe": "uni",
        "model_id": SANA_MODEL_ID if dummy else str(args.model_id),
        "train_method": train_method,
        "lora_rank": lora_rank,
        "resolution": resolution,
        "sample_steps": int(args.sample_steps),
        "sample_guidance": float(args.sample_guidance),
        "steps": steps,
        "dummy": dummy,
        "plus_label": args.plus_label or meta.plus_label,
        "minus_label": args.minus_label or meta.minus_label,
        "recommended_range": meta.recommended_range,
        "concept_words": meta.concept_words,
        "control_prompt": control_prompt,
        "teacher": {
            "plus": "+ concept prompt, Sana CFG v_u + g*(v_c - v_u)",
            "zero": "neutral prompt velocity",
            "minus": "canary only",
            "unused_hold": "unused prompt tokens → encode(neu); concept words not held",
            "cfg": "v(z,t,c) - v(z,t,'')",
            "cfg_compose": "v_u + g * (v_c - v_u)",
        },
        "canary": last_canary,
        "weights": str(weights_path),
        "live_card": sana_live_train_card(),
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
    parser.add_argument("--name", type=str, default="age-sana")
    parser.add_argument(
        "--train_method",
        type=str,
        default=SANA_TRAIN_METHOD,
        choices=["xattn", "selfattn", "attn", "full", "noxattn", "lora"],
        help="conceptmod 0.6B default is xattn. lora is an alias for --lora",
    )
    parser.add_argument(
        "--lora",
        type=int,
        default=None,
        metavar="RANK",
        help="train a LoRA of this rank instead of direct xattn weights",
    )
    parser.add_argument("--alpha", type=float, default=None)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--steps", type=int, default=SANA_DEFAULT_STEPS)
    parser.add_argument("--lr", type=float, default=SANA_DEFAULT_LR)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--resolution", type=int, default=SANA_RESOLUTION)
    parser.add_argument("--sample_steps", type=int, default=SANA_SAMPLE_STEPS)
    parser.add_argument("--sample_guidance", type=float, default=SANA_CFG)
    parser.add_argument("--guidance", type=float, default=None, help="override yaml guidance_scale")
    parser.add_argument("--unused_weight", type=float, default=1.0)
    parser.add_argument("--token_hold_weight", type=float, default=1.0)
    parser.add_argument("--model_id", type=str, default=SANA_MODEL_ID)
    parser.add_argument("--config_file", type=str, default=str(DEFAULT_CONFIG))
    parser.add_argument("--prompts_file", type=str, default=str(DEFAULT_PROMPTS))
    parser.add_argument("--save_dir", type=str, default=str(DEFAULT_SAVE_DIR))
    parser.add_argument("--plus_label", type=str, default=None)
    parser.add_argument("--minus_label", type=str, default=None)
    parser.add_argument(
        "--control_prompt",
        type=str,
        default=SANA_CONTROL_PROMPT,
        help="conceptmod fruit-bowl control; not a teacher",
    )
    parser.add_argument(
        "--dummy",
        action="store_true",
        help="tiny attn1/attn2 + whitespace tokenizer, 2 steps, never loads Hub weights",
    )
    args = parser.parse_args(argv)
    if args.train_method == "lora" and args.lora is None:
        args.lora = 8
    return args


def main(argv: list[str] | None = None) -> Path:
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    return train(parse_args(argv))


if __name__ == "__main__":
    print(sana_live_train_command(), flush=True)
    main()
