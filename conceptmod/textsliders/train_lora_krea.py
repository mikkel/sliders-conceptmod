#!/usr/bin/env python3
"""Opt-in Krea image concept-slider trainer.

UNI analog (not Music 3 lyric-hold):

- student +1 → + concept prompt velocity
- student scale 0 → neutral prompt velocity
- no minus teacher (canary only)
- unused prompt tokens hold to encode(neu); concept words are not held

Velocity-space CFG geometry from conceptmod when it maps:

    v(z, t, c) − v(z, t, '')

Official card: train LoRAs on ``krea/Krea-2-Raw`` (28 steps, CFG 4.5),
rank 16, 512 px. Run on Turbo (local ComfyUI ``.safetensors``, 8 steps,
CFG 0). Default Music 3 trainers are unchanged.

``--dummy`` never loads Hub weights or a 12B transformer. CI uses that.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import torch
import torch.nn as nn
import yaml
from tqdm.auto import tqdm

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from conceptmod.textsliders.slider_targets import (
    KREA_DEFAULT_RANK,
    KREA_DEFAULT_RESOLUTION,
    KREA_HOLD_WEIGHT,
    KREA_RAW_CFG,
    KREA_RAW_MODEL,
    KREA_RAW_STEPS,
    expand_attributes_krea,
    krea_cfg_direction,
    krea_concept_words,
    krea_hold_unused_embeds,
    krea_looks_turbo,
    krea_minus_canary,
    krea_plus_neu_loss,
    krea_plus_neu_teachers,
    krea_sample_card,
    krea_unused_hold_loss,
    krea_unused_hold_mask,
    krea_word_tokens,
)

DEFAULT_CONFIG = Path(__file__).resolve().parent / "data" / "config-krea.yaml"
DEFAULT_PROMPTS = Path(__file__).resolve().parent / "data" / "prompts-krea.yaml"
DEFAULT_SAVE_DIR = Path("models/krea-slider")

# Banned in this trainer. Anima / ZiT / H3 are a separate PR.
_FOREIGN_BACKENDS = ("anima", "zit", "h3", "z-image", "zimage")


@dataclass
class KreaSliderPrompt:
    target: str
    positive: str
    neutral: str
    negative: str = ""
    attributes: list[str] = field(default_factory=list)
    action: str = "enhance"
    guidance_scale: float = KREA_RAW_CFG
    batch_size: int = 1
    unconditional: str = ""


@dataclass
class PromptsMeta:
    plus_label: str = ""
    minus_label: str = ""
    recommended_range: list[float] = field(default_factory=lambda: [-2.0, 2.0])


def _load_yaml(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def load_prompts(path: Path) -> tuple[list[KreaSliderPrompt], PromptsMeta]:
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
    prompts: list[KreaSliderPrompt] = []
    for item in raw:
        if not isinstance(item, dict) or "positive" not in item:
            raise ValueError(f"each prompt must be a mapping with positive: {item!r}")
        for row in expand_attributes_krea(item):
            target = str(row.get("target") or row.get("neutral") or row["positive"])
            prompts.append(
                KreaSliderPrompt(
                    target=target,
                    positive=str(row["positive"]),
                    neutral=str(row.get("neutral") or target),
                    negative=str(row.get("negative") or ""),
                    attributes=[str(a) for a in (item.get("attributes") or [])],
                    action=str(row.get("action") or "enhance"),
                    guidance_scale=float(row.get("guidance_scale", KREA_RAW_CFG)),
                    batch_size=int(row.get("batch_size", 1)),
                    unconditional=str(row.get("unconditional") or ""),
                )
            )
    return prompts, meta


def unused_words_for(prompt: KreaSliderPrompt) -> list[str]:
    words: list[str] = []
    for attr in prompt.attributes:
        words.extend(krea_word_tokens(attr))
    return words


def resolve_krea_card(
    model_id: str,
    sample_steps: int | None,
    sample_guidance: float | None,
) -> dict[str, float | int | str]:
    card = krea_sample_card(model_id)
    if sample_steps is not None:
        card["sample_steps"] = int(sample_steps)
    if sample_guidance is not None:
        card["sample_guidance"] = float(sample_guidance)
    return card


class DummyKreaEncode:
    """Deterministic word-table encode. No tokenizer, no Hub."""

    def __init__(self, dim: int = 8, seed: int = 0):
        self.dim = dim
        self._table: dict[str, torch.Tensor] = {}
        self._rng = torch.Generator().manual_seed(int(seed) + 17)

    def _vec(self, token: str) -> torch.Tensor:
        if token not in self._table:
            self._table[token] = torch.randn(self.dim, generator=self._rng)
        return self._table[token]

    def encode(self, prompt: str) -> tuple[torch.Tensor, list[str]]:
        tokens = krea_word_tokens(prompt) or [""]
        embeds = torch.stack([self._vec(tok) for tok in tokens], dim=0)
        return embeds, tokens


class DummyKreaDiT(nn.Module):
    """Tiny velocity head + multiplier LoRA. CPU only."""

    def __init__(self, dim: int = 8, rank: int = 2):
        super().__init__()
        self.proj = nn.Linear(dim, dim, bias=False)
        self.lora_down = nn.Linear(dim, rank, bias=False)
        self.lora_up = nn.Linear(rank, dim, bias=False)
        nn.init.zeros_(self.lora_up.weight)
        self.scale = 0.0

    def forward(self, z: torch.Tensor, text: torch.Tensor) -> torch.Tensor:
        pooled = text.mean(dim=0, keepdim=True).expand_as(z)
        base = self.proj(z + pooled)
        delta = self.lora_up(self.lora_down(z + pooled))
        return base + float(self.scale) * delta


class DummyKreaBackend:
    """Mock Krea velocity backend. Never downloads weights."""

    def __init__(self, dim: int = 8, rank: int = 2, seed: int = 0):
        self.dim = dim
        self.encode_table = DummyKreaEncode(dim=dim, seed=seed)
        self.dit = DummyKreaDiT(dim=dim, rank=rank)
        self.encoder_lora = False

    def encode_text(self, prompt: str) -> tuple[torch.Tensor, list[str]]:
        return self.encode_table.encode(prompt)

    def predict_v(
        self,
        prompt: str,
        z: torch.Tensor,
        *,
        scale: float = 0.0,
        pin_unused: bool = False,
        neu_prompt: str | None = None,
        unused_words: Sequence[str] | None = None,
    ) -> torch.Tensor:
        embeds, tokens = self.encode_text(prompt)
        if pin_unused and neu_prompt is not None:
            neu_embeds, neu_tokens = self.encode_text(neu_prompt)
            mask = krea_unused_hold_mask(tokens, neu_tokens, unused_words)
            embeds = krea_hold_unused_embeds(
                embeds, neu_embeds, tokens, neu_tokens, mask
            )
        self.dit.scale = float(scale)
        return self.dit(z, embeds)

    def trainable_parameters(self) -> list[nn.Parameter]:
        return list(self.dit.lora_down.parameters()) + list(
            self.dit.lora_up.parameters()
        )


def assert_krea_only(model_id: str) -> None:
    lowered = str(model_id).lower()
    for name in _FOREIGN_BACKENDS:
        if name in lowered:
            raise ValueError(
                f"this trainer is Krea-only; refused foreign backend {name!r}"
            )


def krea_step_loss(
    backend: DummyKreaBackend,
    prompt: KreaSliderPrompt,
    z: torch.Tensor,
    *,
    guidance: float,
    hold_weight: float = KREA_HOLD_WEIGHT,
) -> tuple[torch.Tensor, dict[str, float]]:
    """One UNI step. Minus is computed for the canary and dropped from the loss."""
    unused = unused_words_for(prompt)
    with torch.no_grad():
        v_pos = backend.predict_v(prompt.positive, z, scale=0.0)
        v_neu = backend.predict_v(prompt.neutral, z, scale=0.0)
        v_uncond = backend.predict_v(prompt.unconditional, z, scale=0.0)
        v_neg = (
            backend.predict_v(prompt.negative, z, scale=0.0)
            if prompt.negative
            else None
        )
    tgt_plus, tgt_zero = krea_plus_neu_teachers(
        v_pos, v_neu, v_uncond, guidance=guidance
    )
    pred_plus = backend.predict_v(
        prompt.positive,
        z,
        scale=1.0,
        pin_unused=True,
        neu_prompt=prompt.neutral,
        unused_words=unused,
    )
    pred_zero = backend.predict_v(prompt.neutral, z, scale=0.0)
    loss = krea_plus_neu_loss(pred_plus, tgt_plus, pred_zero, tgt_zero)

    pos_embeds, pos_tokens = backend.encode_text(prompt.positive)
    neu_embeds, neu_tokens = backend.encode_text(prompt.neutral)
    unused_mask = krea_unused_hold_mask(pos_tokens, neu_tokens, unused)
    hold = krea_unused_hold_loss(
        pos_embeds, neu_embeds, pos_tokens, neu_tokens, unused_mask
    )
    # The pin is applied on the +1 student condition. The hold term scores
    # unused pos-token embeds against encode(neu) so a free encoder cannot
    # move unused attributes. Concept words are excluded by the mask.
    if float(hold_weight) > 0.0:
        loss = loss + float(hold_weight) * hold

    stats = {
        "loss": float(loss.detach()),
        "hold": float(hold.detach()),
        "cfg_dir_norm": float(krea_cfg_direction(v_pos, v_uncond).norm().detach()),
        "concept_words": ",".join(sorted(krea_concept_words(prompt.positive, prompt.neutral))),
        "minus_teacher": 0.0,
    }
    if v_neg is not None:
        canary = krea_minus_canary(v_neg, v_uncond)
        stats["canary_minus_norm"] = float(canary.norm().detach())
    return loss, stats


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", type=str, default="krea-age")
    parser.add_argument("--rank", type=int, default=KREA_DEFAULT_RANK)
    parser.add_argument("--resolution", type=int, default=KREA_DEFAULT_RESOLUTION)
    parser.add_argument("--model_id", type=str, default=KREA_RAW_MODEL)
    parser.add_argument("--prompts_file", type=str, default=str(DEFAULT_PROMPTS))
    parser.add_argument("--config_file", type=str, default=str(DEFAULT_CONFIG))
    parser.add_argument("--save_dir", type=str, default=str(DEFAULT_SAVE_DIR))
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--sample_steps", type=int, default=None)
    parser.add_argument("--sample_guidance", type=float, default=None)
    parser.add_argument("--hold_weight", type=float, default=KREA_HOLD_WEIGHT)
    parser.add_argument(
        "--recipe",
        choices=["uni"],
        default="uni",
        help="uni: +1 → + concept, 0 → neu, no minus teacher, unused-token hold. "
        "Not Music 3 lyric-hold. Not the Music 3 default.",
    )
    parser.add_argument(
        "--dummy",
        action="store_true",
        help="tiny CPU backend, 2 steps, never loads Krea / Hub weights",
    )
    return parser.parse_args(argv)


def train(args: argparse.Namespace) -> Path:
    assert_krea_only(args.model_id)
    prompts_path = Path(args.prompts_file)
    if not prompts_path.is_absolute():
        candidate = _REPO_ROOT / prompts_path
        prompts_path = candidate if candidate.exists() else Path(args.prompts_file)
    prompts, meta = load_prompts(prompts_path)
    card = resolve_krea_card(args.model_id, args.sample_steps, args.sample_guidance)
    steps = int(args.steps)
    if args.dummy:
        steps = min(steps, 2)
        device = torch.device("cpu")
        backend = DummyKreaBackend(dim=8, rank=min(2, int(args.rank)), seed=int(args.seed))
    else:
        device = torch.device(f"cuda:{int(args.device)}" if torch.cuda.is_available() else "cpu")
        backend = _load_live_backend(args, device)

    torch.manual_seed(int(args.seed))
    params = backend.trainable_parameters()
    opt = torch.optim.AdamW(params, lr=float(args.lr))
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    log_path = save_dir / f"{args.name}_train.jsonl"
    last_stats: dict[str, float] = {}
    guidance = float(card["sample_guidance"])

    print(
        f"train krea slider name={args.name} recipe={args.recipe} "
        f"rank={args.rank} res={args.resolution} model={args.model_id} "
        f"variant={card['variant']} sample_steps={card['sample_steps']} "
        f"cfg={card['sample_guidance']} dummy={bool(args.dummy)} "
        f"minus_teacher=off unused_hold=on"
    )

    progress = tqdm(range(steps), desc="train-krea")
    for step in progress:
        prompt = prompts[step % len(prompts)]
        z = torch.randn(1, backend.dim, device=device)
        loss, stats = krea_step_loss(
            backend,
            prompt,
            z,
            guidance=guidance,
            hold_weight=float(args.hold_weight),
        )
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        last_stats = stats
        progress.set_postfix({"loss": f"{stats['loss']:.4f}"})
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps({"step": step, **stats}) + "\n")

    sidecar = {
        "kind": "krea",
        "recipe": "uni",
        "name": args.name,
        "model_id": args.model_id,
        "rank": int(args.rank),
        "resolution": int(args.resolution),
        "official": "train LoRAs on Raw, run on Turbo",
        "minus_teacher": False,
        "minus_canary": True,
        "token_hold": "unused_to_neu",
        "lyric_hold": False,
        "dummy": bool(args.dummy),
        "plus_label": meta.plus_label,
        "minus_label": meta.minus_label,
        "recommended_range": meta.recommended_range,
        "last": last_stats,
        **card,
    }
    sidecar_path = save_dir / f"{args.name}_last.json"
    sidecar_path.write_text(json.dumps(sidecar, indent=2), encoding="utf-8")
    print(f"wrote {sidecar_path}")
    return sidecar_path


def _load_live_backend(args: argparse.Namespace, device: torch.device):
    """Lazy live loader. Tests never call this. No Anima / ZiT / H3."""
    raise RuntimeError(
        "live Krea train needs a GPU and Hub or a local ComfyUI Turbo "
        f".safetensors (model_id={args.model_id!r}). CI uses --dummy. "
        "Do not download weights from this path in tests."
    )


def main() -> None:
    train(parse_args())


if __name__ == "__main__":
    main()
