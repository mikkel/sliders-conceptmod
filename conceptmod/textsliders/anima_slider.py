"""Anima image-slider geometry: UNI + unused-token hold.

Image analog of Music 3 UNI (not lyric-hold — images have no lyrics /
``<|audio_start|>``):

- student +1 fits the + concept prompt (CFG teacher)
- student scale 0 fits the neutral prompt
- no minus teacher unless the yaml still declares one as a canary
- hold unused prompt tokens / attributes (subject, composition, pins)
  to encode(neu); do **not** hold the concept words

Velocity-space CFG is conceptmod's ``v(z, t, c) − v(z, t, '')``.

CPU-pure. No Hub, no GPU, no Anima weights. Does not change the Music 3
trainer default (``--lm_target v9`` / ``--pole_mode hidden``).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

import torch
import torch.nn.functional as F
import yaml

DEFAULT_MODEL_ID = "circlestone-labs/Anima-Base-v1.0-Diffusers"
DEFAULT_RANK = 16
DEFAULT_ALPHA = 16.0
DEFAULT_RESOLUTION = 768
DEFAULT_SAMPLE_STEPS = 40
DEFAULT_CFG = 4.0
DEFAULT_HOLD_WEIGHT = 1.0
LORA_TARGETS = ("to_q", "to_k", "to_v", "to_out.0")
# CircleStone: do not train the LLM adapter.
FROZEN_MODULES = ("text_conditioner",)
_TOKEN_RE = re.compile(r"[a-z0-9]+")


@dataclass
class AnimaSliderRow:
    target: str
    positive: str
    neutral: str
    negative: str = ""
    attributes: list[str] = field(default_factory=list)
    action: str = "enhance"
    guidance_scale: float = DEFAULT_CFG
    resolution: int = DEFAULT_RESOLUTION
    batch_size: int = 1
    pins: list[str] = field(default_factory=list)

    @property
    def has_minus_canary(self) -> bool:
        return bool(str(self.negative).strip())

    @property
    def infer_prompt(self) -> str:
        """Slider inference prompt: target, else neu."""
        return self.target.strip() or self.neutral


@dataclass
class AnimaPromptsMeta:
    plus_label: str = ""
    minus_label: str = ""
    recommended_range: list[float] = field(default_factory=lambda: [-2.0, 2.0])


def word_tokens(text: str) -> list[str]:
    """Whitespace / alnum tokenizer. Images have no lyric special tokens."""
    return _TOKEN_RE.findall((text or "").lower())


def unused_vocab(
    target: str,
    neutral: str,
    attributes: Iterable[str] | None = None,
    extra_pins: Iterable[str] | None = None,
) -> set[str]:
    """Subject, composition, and pinned attributes — never concept words."""
    vocab = set(word_tokens(target)) | set(word_tokens(neutral))
    for item in list(attributes or []) + list(extra_pins or []):
        vocab.update(word_tokens(str(item)))
    return vocab


def concept_tokens(positive: str, unused: set[str]) -> list[str]:
    """Tokens in the + prompt that are not unused / pinned."""
    return [tok for tok in word_tokens(positive) if tok not in unused]


def unused_token_mask(tokens: Sequence[str], unused: set[str]) -> list[bool]:
    return [tok in unused for tok in tokens]


def align_unused_positions(
    pos_tokens: Sequence[str],
    neu_tokens: Sequence[str],
    unused: set[str],
) -> list[tuple[int, int]]:
    """Pair unused + tokens with the next matching unused neu token.

    Fail closed (empty) if the + prompt has no unused tokens — the hold
    would otherwise have nothing to pin.
    """
    pairs: list[tuple[int, int]] = []
    neu_idx = 0
    for pos_i, tok in enumerate(pos_tokens):
        if tok not in unused:
            continue
        found = None
        for j in range(neu_idx, len(neu_tokens)):
            if neu_tokens[j] == tok:
                found = j
                break
        if found is None:
            for j, neu_tok in enumerate(neu_tokens):
                if neu_tok == tok:
                    found = j
                    break
        if found is None:
            continue
        pairs.append((pos_i, found))
        neu_idx = found + 1
    return pairs


def splice_unused_embeds(
    pos_embeds: torch.Tensor,
    neu_embeds: torch.Tensor,
    pairs: Sequence[tuple[int, int]],
) -> torch.Tensor:
    """Copy encode(neu) into unused + positions. Concept words stay."""
    held = pos_embeds.clone()
    for pos_i, neu_i in pairs:
        held[..., pos_i, :] = neu_embeds[..., neu_i, :]
    return held


def anima_cfg_delta(v_cond: torch.Tensor, v_uncond: torch.Tensor) -> torch.Tensor:
    """conceptmod velocity-space CFG direction: ``v(z,t,c) − v(z,t,'')``."""
    return v_cond - v_uncond


def anima_uni_teachers(
    v_pos: torch.Tensor,
    v_neu: torch.Tensor,
    v_uncond: torch.Tensor,
    v_neg: torch.Tensor | None = None,
) -> dict[str, torch.Tensor | None]:
    """UNI teachers. Minus is a canary only — never a teacher."""
    del v_neg
    return {
        "plus": anima_cfg_delta(v_pos, v_uncond),
        "zero": anima_cfg_delta(v_neu, v_uncond),
        "minus": None,
    }


def anima_uni_loss(
    student_plus: torch.Tensor,
    teacher_plus: torch.Tensor,
    student_zero: torch.Tensor,
    teacher_zero: torch.Tensor,
    student_minus: torch.Tensor | None = None,
    teacher_minus: torch.Tensor | None = None,
) -> torch.Tensor:
    """``MSE(+) + MSE(0)``. Minus tensors are accepted as a canary and ignored."""
    del student_minus, teacher_minus
    return F.mse_loss(student_plus, teacher_plus) + F.mse_loss(
        student_zero, teacher_zero
    )


def anima_unused_hold_loss(
    pred: torch.Tensor,
    tgt: torch.Tensor,
    pairs: Sequence[tuple[int, int]] | None = None,
    pred_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Masked MSE of unused positions to encode(neu). Concept words skipped.

    ``pred`` / ``tgt`` are ``(..., T, D)``. ``pairs`` maps pred unused
    indices onto tgt unused indices. A boolean ``pred_mask`` over the
    last token axis is also accepted (same-length sequences).
    """
    if pairs is not None:
        if not pairs:
            return pred.reshape(-1)[:1].new_zeros(())
        pred_idx = [p for p, _ in pairs]
        tgt_idx = [n for _, n in pairs]
        return F.mse_loss(pred[..., pred_idx, :], tgt[..., tgt_idx, :])
    if pred_mask is None:
        raise ValueError("anima_unused_hold_loss needs pairs or pred_mask")
    mask = pred_mask.to(dtype=pred.dtype)
    while mask.ndim < pred.ndim:
        mask = mask.unsqueeze(-1)
    denom = mask.sum().clamp_min(1.0)
    return ((pred - tgt).pow(2) * mask).sum() / denom


def minus_canary_cosine(
    student_plus: torch.Tensor,
    v_neg: torch.Tensor,
    v_uncond: torch.Tensor,
) -> torch.Tensor:
    """Log-only: how aligned student +1 is with a declared minus pole."""
    minus = anima_cfg_delta(v_neg, v_uncond).flatten().unsqueeze(0)
    plus = student_plus.flatten().unsqueeze(0)
    return F.cosine_similarity(plus, minus, dim=1, eps=1e-6).mean()


def expand_attributes_anima(row: dict) -> list[dict]:
    """Prefix each attribute onto captions and keep it as an unused pin."""
    attributes = row.get("attributes") or []
    if not attributes:
        item = dict(row)
        item.setdefault("pins", [])
        return [item]
    rows = []
    for attribute in attributes:
        prefix = str(attribute).strip()
        item = dict(row)
        for key in ("target", "positive", "neutral", "negative"):
            value = row.get(key)
            if value:
                item[key] = f"{prefix} {value}"
        pins = [str(p).strip() for p in (row.get("pins") or []) if str(p).strip()]
        if prefix and prefix not in pins:
            pins.append(prefix)
        item["pins"] = pins
        item["attributes"] = [prefix]
        rows.append(item)
    return rows


def _as_row(item: dict) -> AnimaSliderRow:
    if "target" not in item and "positive" not in item:
        raise ValueError(f"anima prompt row needs target or positive: {item!r}")
    target = str(item.get("target") or item.get("neutral") or "")
    positive = str(item.get("positive") or target)
    neutral = str(item.get("neutral") or target)
    attributes = [str(a).strip() for a in (item.get("attributes") or []) if str(a).strip()]
    pins = [str(p).strip() for p in (item.get("pins") or []) if str(p).strip()]
    for attr in attributes:
        if attr not in pins:
            pins.append(attr)
    return AnimaSliderRow(
        target=target,
        positive=positive,
        neutral=neutral,
        negative=str(item.get("negative") or ""),
        attributes=attributes,
        action=str(item.get("action") or "enhance"),
        guidance_scale=float(item.get("guidance_scale", DEFAULT_CFG)),
        resolution=int(item.get("resolution", DEFAULT_RESOLUTION)),
        batch_size=int(item.get("batch_size", 1)),
        pins=pins,
    )


def load_anima_prompts(path: Path | str) -> tuple[list[AnimaSliderRow], AnimaPromptsMeta]:
    path = Path(path)
    with path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    meta = AnimaPromptsMeta()
    if isinstance(raw, dict):
        meta.plus_label = str(raw.get("plus_label") or "")
        meta.minus_label = str(raw.get("minus_label") or "")
        rng = raw.get("recommended_range")
        if isinstance(rng, (list, tuple)) and len(rng) == 2:
            meta.recommended_range = [float(rng[0]), float(rng[1])]
        raw = raw.get("rows", raw.get("prompts"))
    if not isinstance(raw, list) or not raw:
        raise ValueError(f"anima prompts file is empty: {path}")
    rows: list[AnimaSliderRow] = []
    for item in raw:
        if not isinstance(item, dict):
            raise ValueError(f"each anima prompt must be a mapping: {item!r}")
        for expanded in expand_attributes_anima(item):
            rows.append(_as_row(expanded))
    return rows, meta


def row_token_plan(row: AnimaSliderRow) -> dict[str, Any]:
    unused = unused_vocab(row.target, row.neutral, row.attributes, row.pins)
    pos_tokens = word_tokens(row.positive)
    neu_tokens = word_tokens(row.neutral)
    return {
        "unused": unused,
        "concept": concept_tokens(row.positive, unused),
        "pos_tokens": pos_tokens,
        "neu_tokens": neu_tokens,
        "pos_hold_mask": unused_token_mask(pos_tokens, unused),
        "pairs": align_unused_positions(pos_tokens, neu_tokens, unused),
    }


def live_train_card(
    *,
    name: str = "smile-anima",
    prompts_file: str = "conceptmod/textsliders/data/prompts-anima.yaml",
    model_id: str = DEFAULT_MODEL_ID,
    rank: int = DEFAULT_RANK,
    resolution: int = DEFAULT_RESOLUTION,
    sample_steps: int = DEFAULT_SAMPLE_STEPS,
    cfg: float = DEFAULT_CFG,
    device: str = "cuda:0",
) -> dict[str, Any]:
    """Documented live train card. CI never downloads these weights."""
    return {
        "name": name,
        "model_id": model_id,
        "arch": "2B Cosmos-Predict2 DiT, Qwen3+T5, Qwen-Image VAE",
        "lora": {
            "rank": rank,
            "alpha": float(rank),
            "targets": list(LORA_TARGETS),
            "train_text_conditioner": False,
            "frozen_ref": "base transformer with adapter disabled",
        },
        "resolution": resolution,
        "sample_steps": sample_steps,
        "cfg": cfg,
        "device": device,
        "prompts_file": prompts_file,
        "recipe": "uni_plus_neu + unused_token_hold",
        "music3_default_untouched": {"lm_target": "v9", "pole_mode": "hidden"},
    }


def live_train_command(
    *,
    name: str = "smile-anima",
    prompts_file: str = "conceptmod/textsliders/data/prompts-anima.yaml",
    model_id: str = DEFAULT_MODEL_ID,
    rank: int = DEFAULT_RANK,
    resolution: int = DEFAULT_RESOLUTION,
    sample_steps: int = DEFAULT_SAMPLE_STEPS,
    cfg: float = DEFAULT_CFG,
    device: str = "cuda:0",
    save_dir: str = "models/smile-anima",
) -> str:
    return (
        "HF_HUB_OFFLINE=1 python conceptmod/textsliders/train_lora_anima.py \\\n"
        f"  --name {name} \\\n"
        f"  --prompts_file {prompts_file} \\\n"
        f"  --model_id {model_id} \\\n"
        f"  --rank {rank} --resolution {resolution} "
        f"--sample_steps {sample_steps} --cfg {cfg} \\\n"
        f"  --device {device} --save_dir {save_dir}"
    )
