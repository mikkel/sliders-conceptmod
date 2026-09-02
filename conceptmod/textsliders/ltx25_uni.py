"""UNI analog for LTX-2.5 — embed-match default, velocity kept opt-in.

**Default (validated 2026-09-02, dual RTX A6000):** post-connector
video embed-match. Student is ``encode(neu)+LoRA(scale)``. Teacher is
frozen ``encode(plus)``. Loss is MSE + rel-L2 on **valid**
post-connector **video** hidden states. LoRA hosts are video
connectors + TE last-N attn ``q/k/v/o``. **DiT stays frozen.**

DiT velocity UNI (attn1/attn2 LoRA, connector-only LoRA, TE-attn LoRA
with velocity loss) is the **failed** path: live loss sat ~3.17 and
plus/neu velocity cos ~0.9999. Decode concepts live in the text path.
Keep ``--recipe ltx25_uni_velocity`` as opt-in only — do **not** make
it the smile/chiaro default.

Working diagnostic: encode plus vs neu; post-connector video
``mean_cos`` ~0.68. Transplanting plus concept-token (or full plus)
embeds onto neu conditioning produces teeth/smile while holding
identity. That is why the teacher is frozen plus embeds, not DiT
velocity.

Sana / H3 caption-coupling lesson (do not re-learn):

* student +1 (LoRA on, **neu** caption — the infer path) → frozen
  teacher on plus
* plus is **teacher-only**. If student(+1) trains on the plus caption,
  scale 1 on neu will not hit the concept.
* no minus teacher (yaml negative is a logged canary only)
* hold every **non-concept** token (default) to the matching
  ``encode(neu)`` row **PRE-connector**. Yaml attributes (male/female)
  are a subset of that hold.
* ``hold_mode=attributes`` is the leaky subset (shared subject stays free).
* do **not** hold concept words (token ids in + that are absent from neu).
* fail closed if the + prompt has no concept-word tokens.

The velocity helpers below stay so ``--recipe ltx25_uni_velocity``
still runs. They are not the smile/chiaro card.
"""

from __future__ import annotations

from typing import Iterable, Sequence

import torch
import torch.nn.functional as F

HOLD_MODE_NON_CONCEPT = "non_concept"
HOLD_MODE_ATTRIBUTES = "attributes"
HOLD_MODES = (HOLD_MODE_NON_CONCEPT, HOLD_MODE_ATTRIBUTES)
DEFAULT_HOLD_MODE = HOLD_MODE_NON_CONCEPT

RECIPE_EMBED = "ltx25_uni_embed"
RECIPE_VELOCITY = "ltx25_uni_velocity"
RECIPE_ALIASES = {
    "embed": RECIPE_EMBED,
    "embed_match": RECIPE_EMBED,
    "embed-match": RECIPE_EMBED,
    "ltx25_uni_embed": RECIPE_EMBED,
    "ltx25_embed_match": RECIPE_EMBED,
    "velocity": RECIPE_VELOCITY,
    "dit": RECIPE_VELOCITY,
    "ltx25_uni_velocity": RECIPE_VELOCITY,
}
RECIPE_CHOICES = tuple(dict.fromkeys((RECIPE_EMBED, RECIPE_VELOCITY, *RECIPE_ALIASES)))
DEFAULT_RECIPE = RECIPE_EMBED
DEFAULT_TE_LAST_N = 4
DEFAULT_EMBED_REL_L2_WEIGHT = 1.0
DEFAULT_EMBED_REL_L2_EPS = 1e-6
# Live post-connector plus vs neu mean_cos sat ~0.68 (working gap).
# Velocity plus vs neu cos ~0.9999 is the dead teacher.
EMBED_GAP_COS_LIVE = 0.68


def resolve_ltx25_recipe(recipe: str | None) -> str:
    """Canonical recipe. Embed-match is the smile/chiaro default."""
    raw = DEFAULT_RECIPE if recipe is None else str(recipe).strip().lower()
    resolved = RECIPE_ALIASES.get(raw, raw)
    if resolved not in (RECIPE_EMBED, RECIPE_VELOCITY):
        raise ValueError(
            f"recipe must be one of {RECIPE_CHOICES}, got {recipe!r}. "
            f"{RECIPE_EMBED} is the validated smile/chiaro card; "
            f"{RECIPE_VELOCITY} is the failed DiT velocity path (opt-in only)."
        )
    return resolved


def is_embed_recipe(recipe: str | None) -> bool:
    return resolve_ltx25_recipe(recipe) == RECIPE_EMBED


class LTX25HoldError(ValueError):
    """Raised when the + prompt has no concept-word tokens (fail closed)."""


def pin_unused_attributes(
    positive: str,
    neutral: str,
    attributes: Sequence[str],
    *,
    bare_captions: bool = True,
) -> list[tuple[str, str]]:
    """One (pos, neu) row per unused attribute.

    Smile cards keep **bare captions** (Anima / Krea lesson): attributes
    pin unused gender for hold bookkeeping and are **not** prefixed onto
    target / positive / neutral. Pass ``bare_captions=False`` to prefix
    like H3 / Sana age yaml.
    """
    attrs = [a.strip() for a in attributes if a and a.strip()]
    if not attrs or bare_captions:
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


def declared_concept_token_ids(tokenizer, concept_words: str | Iterable[str]) -> set[int]:
    if isinstance(concept_words, str):
        words = [w.strip() for w in concept_words.split(",") if w.strip()]
    else:
        words = [str(w).strip() for w in concept_words if str(w).strip()]
    ids: set[int] = set()
    for word in words:
        ids.update(_encode_ids(tokenizer, word))
        ids.update(_encode_ids(tokenizer, f" {word}"))
    return ids


def resolve_concept_token_ids(
    tokenizer,
    positive: str,
    neutral: str,
    concept_words: str | Iterable[str] = "",
) -> set[int]:
    """Declared concept ids, or plus−neu if the tokenizer split missed.

    Fail closed when both the declared set and the plus−neu support are empty,
    or when declared words exist and none appear in plus *and* plus−neu is empty.
    """
    plus_ids = _encode_ids(tokenizer, positive)
    neu_ids = _encode_ids(tokenizer, neutral)
    fallback = set(plus_ids) - set(neu_ids)
    declared = declared_concept_token_ids(tokenizer, concept_words)
    plus = set(plus_ids)
    if declared and any(tid in plus for tid in declared):
        return declared
    if fallback:
        return fallback
    require_concept_tokens(plus_ids, declared or fallback)
    return declared


def require_concept_tokens(
    plus_ids: Sequence[int],
    concept_ids: Iterable[int],
) -> None:
    """Fail closed when the + prompt has no concept-word tokens."""
    banned = {int(t) for t in concept_ids}
    if not banned:
        raise LTX25HoldError("plus prompt has no concept-word tokens")
    if not any(int(t) in banned for t in plus_ids):
        raise LTX25HoldError("plus prompt has no concept-word tokens")


def resolve_hold_mode(hold_mode: str | None) -> str:
    mode = DEFAULT_HOLD_MODE if hold_mode is None else str(hold_mode).strip().lower()
    if mode not in HOLD_MODES:
        raise ValueError(
            f"hold_mode must be one of {HOLD_MODES}, got {hold_mode!r}. "
            f"{HOLD_MODE_NON_CONCEPT} holds every plus token that is not a "
            f"concept word; {HOLD_MODE_ATTRIBUTES} is the leaky attribute-only "
            f"hold that leaves shared subject identity free."
        )
    return mode


def unused_hold_mask(
    token_ids: Sequence[int],
    unused_ids: Iterable[int],
    concept_ids: Iterable[int],
    *,
    hold_mode: str | None = None,
) -> torch.Tensor:
    """Boolean hold flags for plus-token rows.

    Concept tokens (ids in plus but not in neu) are never held.

    * ``non_concept`` (default): hold every other plus token, including
      shared subject words. Matching ``encode(neu)`` rows pin identity.
    * ``attributes``: only yaml unused-attribute ids (male/female).
      Shared subject tokens stay free and leak.
    """
    mode = resolve_hold_mode(hold_mode)
    unused = set(int(x) for x in unused_ids)
    concept = set(int(x) for x in concept_ids)
    flags = []
    for tid in token_ids:
        tid_i = int(tid)
        if tid_i in concept:
            flags.append(False)
            continue
        if mode == HOLD_MODE_NON_CONCEPT:
            flags.append(True)
        else:
            flags.append(tid_i in unused)
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
    """Copy ``encode(neu)`` onto held (non-concept) plus-token rows.

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
    """Flatten video + audio transformer output into one teacher / student tensor."""
    return torch.cat((video.reshape(video.shape[0], -1), audio.reshape(audio.shape[0], -1)), dim=-1)


def ltx25_uni_velocity_loss(
    pred_plus: torch.Tensor,
    tgt_plus: torch.Tensor,
    pred_zero: torch.Tensor,
    tgt_zero: torch.Tensor,
) -> torch.Tensor:
    """``MSE(student_neu@1 → v_plus) + MSE(student_neu@0 → v_neu)``. No minus term."""
    return F.mse_loss(pred_plus, tgt_plus) + F.mse_loss(pred_zero, tgt_zero)


def ltx25_minus_canary(pred_minus: torch.Tensor, tgt_minus: torch.Tensor) -> torch.Tensor:
    """Logged only. Never added to the train loss."""
    return F.mse_loss(pred_minus, tgt_minus)


def ltx25_unused_hold_loss(
    student_embeds: torch.Tensor,
    neu_embeds: torch.Tensor,
    hold_mask: torch.Tensor,
    *,
    hold_weight: float = 1.0,
) -> torch.Tensor:
    """MSE of held-token hidden to ``encode(neu)``. Concept words free."""
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


def _as_bt_d(hidden: torch.Tensor) -> torch.Tensor:
    """Coerce video hidden to ``(B, T, D)``."""
    if hidden.ndim == 2:
        return hidden.unsqueeze(0)
    if hidden.ndim != 3:
        raise ValueError(f"post-connector hidden must be (B, T, D), got {tuple(hidden.shape)}")
    return hidden


def valid_hidden_rows(
    hidden: torch.Tensor,
    attention_mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Keep valid post-connector rows. Returns ``(B, T, D)`` + weight ``(B, T, 1)``.

    Pad / register-replaced positions with mask 0 do not enter embed-match.
    A missing mask treats every row as valid. ``T`` is unchanged so plus/neu
    connector outputs (same pad target) stay index-aligned.
    """
    hidden = _as_bt_d(hidden)
    if attention_mask is None:
        weight = hidden.new_ones(hidden.shape[0], hidden.shape[1], 1)
        return hidden, weight
    mask = attention_mask.to(device=hidden.device)
    if mask.ndim == 3:
        mask = mask.reshape(mask.shape[0], mask.shape[1])
    if mask.shape[0] != hidden.shape[0] or mask.shape[1] != hidden.shape[1]:
        raise ValueError(
            f"valid mask shape {tuple(mask.shape)} != hidden ({hidden.shape[0]}, {hidden.shape[1]})"
        )
    weight = (mask > 0).to(dtype=hidden.dtype).unsqueeze(-1)
    return hidden, weight


def align_valid_pair(
    pred: torch.Tensor,
    tgt: torch.Tensor,
    pred_mask: torch.Tensor | None = None,
    tgt_mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Align student/teacher post-connector video for MSE + rel-L2.

    Same ``T``: AND the valid masks (index-aligned after left-pad).
    Different ``T``: mean-pool each valid set to ``(B, 1, D)`` — the
    live transplant diagnostic used mean_cos, not a token zip.
    """
    pred_h, pred_w = valid_hidden_rows(pred, pred_mask)
    tgt_h, tgt_w = valid_hidden_rows(tgt, tgt_mask)
    if pred_h.shape[0] != tgt_h.shape[0]:
        raise ValueError(
            f"batch mismatch: student {tuple(pred_h.shape)} vs teacher {tuple(tgt_h.shape)}"
        )
    if pred_h.shape[1] == tgt_h.shape[1] and pred_h.shape[-1] == tgt_h.shape[-1]:
        weight = pred_w * tgt_w
        if float(weight.sum()) <= 0:
            weight = pred_h.new_ones(pred_h.shape[0], pred_h.shape[1], 1)
        return pred_h, tgt_h, weight
    # T or D differs: pool valid rows. Dummy/live pad target is normally equal.
    pred_pool = _masked_mean_tokens(pred_h, pred_w)
    tgt_pool = _masked_mean_tokens(tgt_h, tgt_w)
    ones = pred_pool.new_ones(pred_pool.shape[0], 1, 1)
    return pred_pool, tgt_pool, ones


def _masked_mean_tokens(hidden: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    denom = weight.sum(dim=1, keepdim=True).clamp(min=1.0)
    return (hidden * weight).sum(dim=1, keepdim=True) / denom


def ltx25_embed_mse(
    pred: torch.Tensor,
    tgt: torch.Tensor,
    pred_mask: torch.Tensor | None = None,
    tgt_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Masked MSE on valid post-connector video rows."""
    pred_h, tgt_h, weight = align_valid_pair(pred, tgt, pred_mask, tgt_mask)
    denom = weight.sum().clamp(min=1.0) * pred_h.shape[-1]
    return ((pred_h - tgt_h).pow(2) * weight).sum() / denom


def ltx25_embed_rel_l2(
    pred: torch.Tensor,
    tgt: torch.Tensor,
    pred_mask: torch.Tensor | None = None,
    tgt_mask: torch.Tensor | None = None,
    *,
    eps: float = DEFAULT_EMBED_REL_L2_EPS,
) -> torch.Tensor:
    """``||s−t||² / (||t||²+eps)`` on valid post-connector video rows.

    Scale-invariant sibling of Krea embed rel-L2. ``tgt`` must already
    be stopgrad. High-cos / wrong-magnitude students still score high.
    """
    pred_h, tgt_h, weight = align_valid_pair(pred, tgt, pred_mask, tgt_mask)
    diff_sq = ((pred_h - tgt_h).pow(2) * weight).sum()
    tgt_sq = ((tgt_h.pow(2) * weight).sum()).clamp_min(float(eps))
    return diff_sq / tgt_sq


def ltx25_embed_match_loss(
    pred: torch.Tensor,
    tgt: torch.Tensor,
    pred_mask: torch.Tensor | None = None,
    tgt_mask: torch.Tensor | None = None,
    *,
    rel_l2_weight: float = DEFAULT_EMBED_REL_L2_WEIGHT,
    eps: float = DEFAULT_EMBED_REL_L2_EPS,
) -> torch.Tensor:
    """Student ``encode(neu)+LoRA`` → stopgrad frozen ``encode(plus)``.

    MSE + rel-L2 on **valid post-connector video** hidden states.
    No DiT velocity term. ``tgt`` must already be stopgrad.
    """
    loss = ltx25_embed_mse(pred, tgt, pred_mask, tgt_mask)
    rel_w = float(rel_l2_weight)
    if rel_w > 0.0:
        loss = loss + rel_w * ltx25_embed_rel_l2(
            pred, tgt, pred_mask, tgt_mask, eps=eps,
        )
    return loss


def post_connector_mean_cos(
    a: torch.Tensor,
    b: torch.Tensor,
    a_mask: torch.Tensor | None = None,
    b_mask: torch.Tensor | None = None,
) -> dict[str, float]:
    """Valid-row mean-pool cosine + L2. Live plus/neu sat ~0.68."""
    a_h, b_h, _w = align_valid_pair(a, b, a_mask, b_mask)
    return cosine_l2(a_h, b_h)


def ltx25_uni_total_loss(
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
    loss = ltx25_uni_velocity_loss(pred_plus, tgt_plus, pred_zero, tgt_zero)
    if student_embeds is not None and neu_embeds is not None and hold_mask is not None:
        loss = loss + ltx25_unused_hold_loss(
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


def cosine_l2(a: torch.Tensor, b: torch.Tensor) -> dict[str, float]:
    """Packed velocity (or any tensor) cosine + L2. Flattened, float32."""
    x = a.detach().float().reshape(-1)
    y = b.detach().float().reshape(-1)
    n = min(int(x.numel()), int(y.numel()))
    if n == 0:
        return {"cos": 0.0, "l2": 0.0}
    x, y = x[:n], y[:n]
    cos = F.cosine_similarity(x.unsqueeze(0), y.unsqueeze(0), dim=-1)
    l2 = torch.linalg.vector_norm(x - y)
    return {"cos": float(cos.item()), "l2": float(l2.item())}


DEAD_GAP_COS = 0.999


def expression_gap_is_dead(metrics: dict[str, float], *, cos_floor: float = DEAD_GAP_COS) -> bool:
    """True when frozen plus vs neu velocity is collapsed (cos ≈ 1).

    First-live ``--diag``: if this is True, do **not** silently train —
    document ``transformer_full/`` as the SFT fallback. Distilled 8-sigma
    / CFG=1 geometry may hide the smile axis (**hypothesis**).
    """
    return float(metrics.get("cos", 0.0)) >= float(cos_floor)


def _as_token_rows(hidden: torch.Tensor) -> torch.Tensor:
    if hidden.dim() == 3:
        return hidden[0]
    return hidden


def hold_effectiveness_metrics(
    plus_hidden: torch.Tensor,
    neu_hidden: torch.Tensor,
    plus_ids: Sequence[int],
    neu_ids: Sequence[int],
    hold_mask: torch.Tensor,
    concept_ids: Iterable[int] | None = None,
    held_hidden: torch.Tensor | None = None,
) -> dict[str, float | int]:
    """Encoder hold check after ``apply_unused_hold`` (PRE-connector rows)."""
    plus_h = _as_token_rows(plus_hidden).detach().float()
    neu_h = _as_token_rows(neu_hidden).detach().float()
    held_h = plus_h if held_hidden is None else _as_token_rows(held_hidden).detach().float()
    neu_index = {int(tid): i for i, tid in enumerate(neu_ids)}
    concept = set(int(x) for x in (concept_ids or []))
    mask = hold_mask.detach().reshape(-1).tolist()
    n = min(len(mask), held_h.shape[0], plus_h.shape[0], len(plus_ids))
    held_l2: list[float] = []
    concept_l2: list[float] = []
    neu_mean = neu_h.mean(dim=0) if neu_h.numel() else plus_h.new_zeros(plus_h.shape[-1])
    for i in range(n):
        tid = int(plus_ids[i])
        if mask[i]:
            j = neu_index.get(tid)
            if j is None or j >= neu_h.shape[0]:
                continue
            held_l2.append(float(torch.linalg.vector_norm(held_h[i] - neu_h[j]).item()))
        is_concept = tid in concept if concept else not bool(mask[i])
        if is_concept:
            concept_l2.append(float(torch.linalg.vector_norm(plus_h[i] - neu_mean).item()))
    return {
        "n_held": len(held_l2),
        "n_free": int(sum(1 for i in range(n) if not mask[i])),
        "n_concept": len(concept_l2),
        "held_max_abs": max(held_l2) if held_l2 else 0.0,
        "held_mean_abs": float(sum(held_l2) / len(held_l2)) if held_l2 else 0.0,
        "concept_mean_abs": float(sum(concept_l2) / len(concept_l2)) if concept_l2 else 0.0,
    }


def embed_gap_energy_frac(
    plus_hidden: torch.Tensor,
    neu_hidden: torch.Tensor,
    plus_ids: Sequence[int],
    neu_ids: Sequence[int],
    hold_mask: torch.Tensor,
    concept_ids: Iterable[int] | None = None,
    held_hidden: torch.Tensor | None = None,
) -> dict[str, float]:
    """Split ``||held_plus − aligned_neu||^2`` after PRE-connector hold."""
    plus_h = _as_token_rows(plus_hidden).detach().float()
    neu_h = _as_token_rows(neu_hidden).detach().float()
    held_h = plus_h if held_hidden is None else _as_token_rows(held_hidden).detach().float()
    neu_index = {int(tid): i for i, tid in enumerate(neu_ids)}
    concept = set(int(x) for x in (concept_ids or []))
    mask = hold_mask.detach().reshape(-1).tolist()
    n = min(len(mask), held_h.shape[0], plus_h.shape[0], len(plus_ids))
    neu_mean = neu_h.mean(dim=0) if neu_h.numel() else plus_h.new_zeros(plus_h.shape[-1])
    concept_e = held_e = leak_e = 0.0
    for i in range(n):
        tid = int(plus_ids[i])
        j = neu_index.get(tid)
        ref = neu_h[j] if j is not None and j < neu_h.shape[0] else neu_mean
        energy = float(((held_h[i] - ref) ** 2).sum().item())
        if mask[i]:
            held_e += energy
        elif tid in concept or (not concept and not mask[i]):
            concept_e += energy
        else:
            leak_e += energy
    total = concept_e + held_e + leak_e
    if total <= 0.0:
        return {
            "concept": 0.0,
            "held": 0.0,
            "unheld_nonconcept": 0.0,
            "concept_frac": 0.0,
            "held_frac": 0.0,
            "unheld_nonconcept_frac": 0.0,
        }
    return {
        "concept": concept_e,
        "held": held_e,
        "unheld_nonconcept": leak_e,
        "concept_frac": concept_e / total,
        "held_frac": held_e / total,
        "unheld_nonconcept_frac": leak_e / total,
    }
