"""UNI analog for H3 image sliders — AR encode, not Music 3 lyric-hold.

HunyuanImage-3.0 is autoregressive MoE. UNI lives on ``encode`` /
last hidden, not a fake flow-matching velocity.

* student +1 last hidden → ``encode(pos)`` last hidden
* student scale 0 last hidden → ``encode(neu)`` last hidden
* no minus teacher (minus MSE is a logged canary only)
* hold unused prompt tokens to ``encode(neu)``
* do **not** hold concept words (tokens in + that are absent from neu)
"""

from __future__ import annotations

from typing import Iterable, Sequence

import torch
import torch.nn.functional as F


def pin_unused_attributes(
    positive: str,
    neutral: str,
    attributes: Sequence[str],
) -> list[tuple[str, str]]:
    """One (pos, neu) row per unused attribute, both captions pinned."""
    attrs = [a.strip() for a in attributes if a and a.strip()]
    if not attrs:
        return [(positive, neutral)]
    rows = []
    for attr in attrs:
        rows.append((_pin_phrase(positive, attr), _pin_phrase(neutral, attr)))
    return rows


def _pin_phrase(prompt: str, attr: str) -> str:
    prompt = prompt.strip()
    attr = attr.strip()
    if not attr:
        return prompt
    low = prompt.lower()
    if attr.lower() in low.split() or f" {attr.lower()} " in f" {low} ":
        return prompt
    if not prompt:
        return attr
    return f"{attr} {prompt}"


def concept_token_ids(tokenizer, positive: str, neutral: str) -> set[int]:
    pos = set(_encode_ids(tokenizer, positive))
    neu = set(_encode_ids(tokenizer, neutral))
    return pos - neu


def unused_token_ids(tokenizer, attributes: Sequence[str]) -> set[int]:
    ids: set[int] = set()
    for attr in attributes:
        ids.update(_encode_ids(tokenizer, attr))
    return ids


def unused_hold_mask(
    token_ids: Sequence[int],
    unused_ids: Iterable[int],
    concept_ids: Iterable[int],
) -> torch.Tensor:
    unused = set(int(x) for x in unused_ids)
    concept = set(int(x) for x in concept_ids)
    flags = [bool(int(tid) in unused and int(tid) not in concept) for tid in token_ids]
    if not flags:
        return torch.zeros(0, dtype=torch.bool)
    return torch.tensor(flags, dtype=torch.bool)


def last_hidden(embeds: torch.Tensor) -> torch.Tensor:
    if embeds.dim() == 3:
        return embeds[:, -1]
    return embeds[-1]


def h3_uni_hidden_loss(
    pred_plus: torch.Tensor,
    tgt_plus: torch.Tensor,
    pred_zero: torch.Tensor,
    tgt_zero: torch.Tensor,
) -> torch.Tensor:
    """``MSE(+ → encode(pos)) + MSE(0 → encode(neu))``. No minus term."""
    return F.mse_loss(pred_plus, tgt_plus) + F.mse_loss(pred_zero, tgt_zero)


# Back-compat alias used by earlier velocity drafts; same algebra.
h3_uni_velocity_loss = h3_uni_hidden_loss


def h3_minus_canary(pred_minus: torch.Tensor, tgt_minus: torch.Tensor) -> torch.Tensor:
    """Logged only. Never added to the train loss."""
    return F.mse_loss(pred_minus, tgt_minus)


def h3_unused_hold_loss(
    student_embeds: torch.Tensor,
    neu_embeds: torch.Tensor,
    hold_mask: torch.Tensor,
    *,
    hold_weight: float = 1.0,
) -> torch.Tensor:
    """MSE of unused-token hidden to ``encode(neu)``. Concept words free."""
    if hold_mask.numel() == 0 or not bool(hold_mask.any()):
        return student_embeds.reshape(-1)[:1].sum() * 0.0
    if student_embeds.dim() == 3:
        student_embeds = student_embeds[0]
        neu_embeds = neu_embeds[0]
    mask = hold_mask.to(device=student_embeds.device, dtype=torch.bool)
    n = min(mask.numel(), student_embeds.shape[0], neu_embeds.shape[0])
    mask = mask[:n]
    if not bool(mask.any()):
        return student_embeds.reshape(-1)[:1].sum() * 0.0
    return float(hold_weight) * F.mse_loss(student_embeds[:n][mask], neu_embeds[:n][mask])


def h3_uni_total_loss(
    pred_plus: torch.Tensor,
    tgt_plus: torch.Tensor,
    pred_zero: torch.Tensor,
    tgt_zero: torch.Tensor,
    student_embeds: torch.Tensor | None = None,
    neu_embeds: torch.Tensor | None = None,
    hold_mask: torch.Tensor | None = None,
    *,
    hold_weight: float = 1.0,
) -> torch.Tensor:
    loss = h3_uni_hidden_loss(pred_plus, tgt_plus, pred_zero, tgt_zero)
    if student_embeds is not None and neu_embeds is not None and hold_mask is not None:
        loss = loss + h3_unused_hold_loss(
            student_embeds, neu_embeds, hold_mask, hold_weight=hold_weight,
        )
    return loss


def _encode_ids(tokenizer, text: str) -> list[int]:
    if hasattr(tokenizer, "encode"):
        return [int(x) for x in tokenizer.encode(text, add_special_tokens=False)]
    batch = tokenizer(text)
    ids = batch["input_ids"]
    if hasattr(ids, "tolist"):
        ids = ids.tolist()
    if ids and isinstance(ids[0], list):
        ids = ids[0]
    return [int(x) for x in ids]
