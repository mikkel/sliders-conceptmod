"""UNI analog for MiniMax-H3 — packed-sequence velocity, not Music 3 lyric-hold.

MiniMax-H3's Omni-Transformer is a CFG-distilled flow model. The released
``MiniMaxH3Transformer3DModel`` returns **data-pointing velocity**
(``x0 = x_t + sigma * v``) on a packed multimodal sequence (text + video +
audio, MM-RoPE). This module uses that actual forward. It does **not** wrap
the stack as a conceptmod ``predict_v`` / Euler DiT.

* student +1 (LoRA on, plus-concept packed sequence) → teacher plus velocity
* student scale 0 (adapter off, neu packed sequence) → teacher neu velocity
* no minus teacher (yaml negative is a logged canary only)
* hold unused prompt tokens / pinned yaml attributes to ``encode(neu)``
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


def apply_unused_hold(
    plus_hidden: torch.Tensor,
    neu_hidden: torch.Tensor,
    plus_ids: Sequence[int],
    neu_ids: Sequence[int],
    hold_mask: torch.Tensor,
) -> torch.Tensor:
    """Copy ``encode(neu)`` onto unused (non-concept) plus-token rows.

    Alignment is by token id: each held plus position takes the first matching
    neu-row hidden. Concept words are left as ``encode(plus)``.
    """
    if hold_mask.numel() == 0 or not bool(hold_mask.any()):
        return plus_hidden
    held = plus_hidden.clone()
    neu_index = {int(tid): i for i, tid in enumerate(neu_ids)}
    mask = hold_mask.tolist()
    n = min(len(mask), held.shape[-2] if held.dim() >= 2 else 0, len(plus_ids))
    for i in range(n):
        if not mask[i]:
            continue
        j = neu_index.get(int(plus_ids[i]))
        if j is None or j >= neu_hidden.shape[-2]:
            continue
        if held.dim() == 3:
            held[:, i] = neu_hidden[:, j]
        else:
            held[i] = neu_hidden[j]
    return held


def velocity_pair(
    video: torch.Tensor,
    audio: torch.Tensor,
) -> torch.Tensor:
    """Flatten video + audio velocity into one teacher / student tensor."""
    return torch.cat((video.reshape(video.shape[0], -1), audio.reshape(audio.shape[0], -1)), dim=-1)


def minimax_h3_uni_velocity_loss(
    pred_plus: torch.Tensor,
    tgt_plus: torch.Tensor,
    pred_zero: torch.Tensor,
    tgt_zero: torch.Tensor,
) -> torch.Tensor:
    """``MSE(+ → v_plus) + MSE(0 → v_neu)``. No minus term."""
    return F.mse_loss(pred_plus, tgt_plus) + F.mse_loss(pred_zero, tgt_zero)


def minimax_h3_minus_canary(pred_minus: torch.Tensor, tgt_minus: torch.Tensor) -> torch.Tensor:
    """Logged only. Never added to the train loss."""
    return F.mse_loss(pred_minus, tgt_minus)


def minimax_h3_unused_hold_loss(
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


def minimax_h3_uni_total_loss(
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
    loss = minimax_h3_uni_velocity_loss(pred_plus, tgt_plus, pred_zero, tgt_zero)
    if student_embeds is not None and neu_embeds is not None and hold_mask is not None:
        loss = loss + minimax_h3_unused_hold_loss(
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
