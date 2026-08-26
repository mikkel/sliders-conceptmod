"""Plus-only exam: cover the + caption, keep + off-caption low.

A separate scale from the bipolar pair-exam, leftover-sheet
``leak_frac``, and the compiled board. Two scored numbers:

- ``off_caption`` — share of the + continuation that neither the +
  caption nor the pair's shared song sings. Lower is better.
- ``cover`` — ``min(overlap_with_pos, 1 − blend_toward_mid)`` in
  ``[0, 1]``. High means the + student stays on the + caption and
  nearer to pos than to mid / to neg.

A hit is high cover AND low off-caption. ``leak_frac``, collapse,
pair-odd cos, minus overlap, and ``exam_score`` are not scored.
−1 is logged as an unconstrained canary.

CPU only. No Hub, no GPU, no Music 3 weights. Does not change the live
default (``--lm_target v9`` / ``--pole_mode hidden``).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from analysis.slider2d.exam import (
    PairField,
    SharedResidual,
    close_field,
    divergent_field,
    rollouts,
    teacher_points,
    unused_e_field,
)
from conceptmod.textsliders.slider_targets import (
    EVEN_BLEND_SCALE as LIVE_EVEN_BLEND_SCALE,
    lm_faithful_plus,
    lm_plus_loss,
    lm_slider_loss,
)

# Re-export so tests can pin the live half-step without importing the trainer.
assert LIVE_EVEN_BLEND_SCALE == 0.5

PLUS_OFF_MAX = 0.05
PLUS_COVER_MIN = 0.85

PLUS_CELLS = {
    "divergent": divergent_field,
    "close": close_field,
    "unused_e": unused_e_field,
}

# Recipes this scale compares. Bipolar rows train ±; faithful_plus trains +.
PLUS_RECIPES: list[dict] = [
    {
        "name": "faithful_plus",
        "teacher": "faithful_plus",
        "plus_only": True,
    },
    {
        "name": "leftover_gate_bipolar",
        "teacher": "faithful_sub_e_if_unused",
        "plus_only": False,
    },
    {
        "name": "faithful_even_blend",
        "teacher": "faithful_gate_odd_sub_even_blend",
        "plus_only": False,
        "even_scale": 0.5,
    },
    {
        "name": "pair_odd_midpoint",
        "teacher": "pair_odd",
        "plus_only": False,
    },
]


def plus_teacher(
    field: PairField,
    row: int,
    *,
    teacher: str,
    leak_dir: torch.Tensor | None = None,
    even_scale: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """+ teacher and a canary minus reference (never a plus-exam score)."""
    mode = str(teacher).strip().lower()
    pos, neg, neu = field.poles(row)
    if mode == "faithful_plus":
        plus = lm_faithful_plus(pos, neg, neu, leak_dir, slider_dir=field.short_u())
        return plus, neg
    return teacher_points(
        field, row, teacher=teacher, leak_dir=leak_dir, even_scale=even_scale
    )


def plus_cover(overlap_pos: float, blend_toward_mid: float) -> float:
    """Sortable cover in ``[0, 1]``. High is on-caption and not a blend."""
    return max(0.0, min(1.0, min(float(overlap_pos), 1.0 - float(blend_toward_mid))))


def blend_toward_mid(
    student: torch.Tensor,
    pos: torch.Tensor,
    mid: torch.Tensor,
    neg: torch.Tensor,
) -> float:
    """0 at pos, 1 at mid or neg. Uses the nearer of mid / neg."""
    d_pos = float((student - pos).norm())
    d_mid = float((student - mid).norm())
    d_neg = float((student - neg).norm())
    nearer = min(d_mid, d_neg)
    return d_pos / (d_pos + nearer + 1e-8)


def plus_bags(field: PairField) -> dict[str, frozenset[int]]:
    """+ caption bag, − caption bag, shared-song bag, plus-exam corpus.

    Shared song is lyrics plus tokens both pole captions sing. Plus-exam
    off-caption is words in neither the + bag nor that shared song — so
    singing the − pole's unique track words is off-caption here.
    """
    head = field.readout()
    plus_words: set[int] = set()
    minus_words: set[int] = set()
    lyrics: set[int] = set()
    for row in range(int(field.rows)):
        pos, neg, _neu = field.poles(row)
        up, _ = rollouts(field, pos, head, row=row, sign=1.0)
        down, _ = rollouts(field, neg, head, row=row, sign=-1.0)
        for seq in up:
            plus_words |= set(seq)
        for seq in down:
            minus_words |= set(seq)
        lyrics.add(head.index(f"lyric{row}"))
    plus_words |= lyrics
    minus_words |= lyrics
    shared = (plus_words & minus_words) | lyrics
    return {
        "pos": frozenset(plus_words),
        "neg": frozenset(minus_words),
        "shared": frozenset(shared),
        "plus_corpus": frozenset(plus_words | shared),
        "minus_corpus": frozenset(minus_words | shared),
    }


def _token_share(seqs: list[list[int]], bag: frozenset[int]) -> float:
    hits: list[float] = []
    for seq in seqs:
        if not seq:
            continue
        hits.append(sum(1.0 for tok in seq if tok in bag) / float(len(seq)))
    return sum(hits) / len(hits) if hits else 0.0


def _off_share(seqs: list[list[int]], corpus: frozenset[int]) -> float:
    hits: list[float] = []
    for seq in seqs:
        if not seq:
            continue
        hits.append(sum(1.0 for tok in seq if tok not in corpus) / float(len(seq)))
    return sum(hits) / len(hits) if hits else 0.0


def _continue(
    field: PairField, hidden: torch.Tensor, *, row: int, sign: float
) -> list[list[int]]:
    seqs, _ = rollouts(field, hidden, field.readout(), row=row, sign=sign)
    return seqs


def nearest_pole(
    student: torch.Tensor,
    pos: torch.Tensor,
    neu: torch.Tensor,
    neg: torch.Tensor,
) -> str:
    dist = {
        "pos": float((student - pos).norm()),
        "neu": float((student - neu).norm()),
        "neg": float((student - neg).norm()),
    }
    return min(dist, key=dist.get)


def fit_plus_exam(
    field: PairField,
    *,
    teacher: str,
    leak_dir: torch.Tensor | None = None,
    even_scale: float = 1.0,
    plus_only: bool = False,
    steps: int = 400,
    lr: float = 0.08,
    seed: int = 0,
) -> SharedResidual:
    """Fit one shared residual. ``plus_only`` drops minus MSE."""
    targets = [
        plus_teacher(
            field, row, teacher=teacher, leak_dir=leak_dir, even_scale=even_scale
        )
        for row in range(int(field.rows))
    ]
    neutrals = [field.poles(row)[2] for row in range(int(field.rows))]
    torch.manual_seed(int(seed))
    residual = SharedResidual.create(field)
    opt = torch.optim.Adam(residual.parameters(), lr=float(lr))
    for _ in range(int(steps)):
        total = None
        for (t_plus, t_minus), neu in zip(targets, neutrals):
            pred_plus = neu + residual.delta(1.0)
            pred_minus = neu + residual.delta(-1.0)
            if plus_only:
                term = lm_plus_loss(pred_plus, t_plus)
            else:
                term = lm_slider_loss(pred_plus, pred_minus, t_plus, t_minus)
            total = term if total is None else total + term
        loss = total / float(len(targets))
        opt.zero_grad()
        loss.backward()
        opt.step()
    return residual.snapshot()


def score_plus_exam(
    name: str,
    field: PairField,
    *,
    teacher: str,
    leak_dir: torch.Tensor | None = None,
    even_scale: float = 1.0,
    plus_only: bool = False,
    steps: int = 400,
    seed: int = 0,
) -> dict:
    """Fit, then score only the + continuation. −1 is a canary."""
    residual = fit_plus_exam(
        field,
        teacher=teacher,
        leak_dir=leak_dir,
        even_scale=even_scale,
        plus_only=plus_only,
        steps=steps,
        seed=seed,
    )
    bags = plus_bags(field)
    d_plus = residual.delta(1.0)
    d_minus = residual.delta(-1.0)
    overlap_rows: list[float] = []
    off_rows: list[float] = []
    blend_rows: list[float] = []
    cover_rows: list[float] = []
    canary_overlap: list[float] = []
    canary_off: list[float] = []
    canary_landed: list[str] = []
    sings_plus: list[str] = []
    sings_minus: list[str] = []
    head = field.readout()
    for row in range(int(field.rows)):
        pos, neg, neu = field.poles(row)
        mid = 0.5 * (pos + neg)
        student_plus = neu + d_plus
        student_minus = neu + d_minus
        plus_seqs = _continue(field, student_plus, row=row, sign=1.0)
        minus_seqs = _continue(field, student_minus, row=row, sign=-1.0)
        overlap = _token_share(plus_seqs, bags["pos"])
        off = _off_share(plus_seqs, bags["plus_corpus"])
        blend = blend_toward_mid(student_plus, pos, mid, neg)
        cover = plus_cover(overlap, blend)
        overlap_rows.append(overlap)
        off_rows.append(off)
        blend_rows.append(blend)
        cover_rows.append(cover)
        canary_overlap.append(_token_share(minus_seqs, bags["neg"]))
        canary_off.append(_off_share(minus_seqs, bags["minus_corpus"]))
        canary_landed.append(nearest_pole(student_minus, pos, neu, neg))
        sings_plus.append(" ".join(head.tokens[t] for t in plus_seqs[0]))
        sings_minus.append(" ".join(head.tokens[t] for t in minus_seqs[0]))
    cover = sum(cover_rows) / len(cover_rows)
    off_caption = sum(off_rows) / len(off_rows)
    overlap_pos = sum(overlap_rows) / len(overlap_rows)
    blend = sum(blend_rows) / len(blend_rows)
    hit = bool(cover >= PLUS_COVER_MIN and off_caption <= PLUS_OFF_MAX)
    landed = max(set(canary_landed), key=canary_landed.count)
    canary_off_mean = sum(canary_off) / len(canary_off)
    canary_dangerous = bool(landed == "pos" or canary_off_mean > PLUS_OFF_MAX)
    return {
        "name": name,
        "cell": field.kind,
        "teacher": teacher,
        "plus_only": bool(plus_only),
        "even_scale": float(even_scale),
        "cover": cover,
        "off_caption": off_caption,
        "overlap_pos": overlap_pos,
        "blend_toward_mid": blend,
        "hit": hit,
        "sings_plus": " | ".join(sings_plus),
        "canary": {
            "scored": False,
            "minus_overlap_neg": sum(canary_overlap) / len(canary_overlap),
            "minus_off_caption": canary_off_mean,
            "minus_landed": landed,
            "minus_sings": " | ".join(sings_minus),
            "dangerous": canary_dangerous,
        },
        "pole_cos": float(
            F.cosine_similarity(
                d_plus.flatten().unsqueeze(0),
                (field.poles(0)[0] - field.poles(0)[2]).flatten().unsqueeze(0),
            ).squeeze()
        ),
    }


def plus_exam_table(*, steps: int = 400, seed: int = 0) -> dict[str, list[dict]]:
    """Score every plus-exam recipe on the two cheap pair types (and unused_e)."""
    out: dict[str, list[dict]] = {}
    for cell_name, ctor in PLUS_CELLS.items():
        field = ctor(seed=seed)
        leak = field.declared_e()
        rows = []
        for cand in PLUS_RECIPES:
            rows.append(
                score_plus_exam(
                    cand["name"],
                    field,
                    teacher=cand["teacher"],
                    leak_dir=leak,
                    even_scale=float(cand.get("even_scale", 1.0)),
                    plus_only=bool(cand["plus_only"]),
                    steps=steps,
                    seed=seed,
                )
            )
        out[cell_name] = rows
    return out


def plus_helps(table: dict[str, list[dict]]) -> dict:
    """Yes/no from the numbers: does plus-only beat the bipolar recipes on +?"""
    by: dict[str, dict[str, dict]] = {}
    for cell, rows in table.items():
        by[cell] = {r["name"]: r for r in rows}
    plus_hits = [by[c]["faithful_plus"]["hit"] for c in ("divergent", "close")]
    bipolar_hits = {
        name: [by[c][name]["hit"] for c in ("divergent", "close")]
        for name in ("leftover_gate_bipolar", "faithful_even_blend", "pair_odd_midpoint")
    }
    plus_cover = [by[c]["faithful_plus"]["cover"] for c in ("divergent", "close")]
    plus_off = [by[c]["faithful_plus"]["off_caption"] for c in ("divergent", "close")]
    leftover_cover = [
        by[c]["leftover_gate_bipolar"]["cover"] for c in ("divergent", "close")
    ]
    leftover_off = [
        by[c]["leftover_gate_bipolar"]["off_caption"] for c in ("divergent", "close")
    ]
    fails = [
        name
        for name, hits in bipolar_hits.items()
        if any(not h for h in hits) and all(plus_hits)
    ]
    beats_leftover = all(
        (pc > lc + 1e-6) or (po + 1e-6 < lo)
        for pc, po, lc, lo in zip(plus_cover, plus_off, leftover_cover, leftover_off)
    )
    # Yes if plus-only hits both required pairs and a bipolar fail does not,
    # or if it is materially cleaner than leftover-gate on +.
    yes = bool(all(plus_hits) and (fails or beats_leftover))
    return {
        "yes": yes,
        "plus_hits_required": plus_hits,
        "bipolar_hits": bipolar_hits,
        "beats_leftover_gate": beats_leftover,
        "alleviates": fails,
    }
