"""Plus+neu exam: cover the + caption, hold scale 0 on the song.

A separate scale from the plus-only exam, the bipolar pair-exam,
leftover-sheet ``leak_frac``, and the compiled board. Three scored
numbers:

- ``cover`` — same as plus-only:
  ``min(overlap_with_pos, 1 − blend_toward_mid)`` in ``[0, 1]``.
- ``off_caption`` — share of the + continuation that neither the +
  caption nor the pair's shared song sings. Lower is better.
- ``neu_hold`` — student at scale 0 stays on the yaml lyrics / neutral
  caption: ``min(overlap_with_neu, 1 − drift_from_neu)`` in ``[0, 1]``.
  ``overlap_with_neu`` is the share of the scale-0 continuation that
  the lyrics field or the neu caption itself sings.
  ``drift_from_neu`` is
  ``‖s0 − neu‖ / (‖s0 − neu‖ + min(‖s0 − pos‖, ‖s0 − mid‖))`` —
  0 at neu, 1 at pos or at mid.

A hit is high cover AND low off-caption AND high neu_hold.
``leak_frac``, collapse, pair-odd cos, minus overlap, and ``exam_score``
are not scored. −1 is logged as an unconstrained canary.

The student here has a free origin: ``δ(σ) = σ·w_odd + |σ|·w_even + w0``.
A multiplier LoRA has no ``w0``; this extra capacity is how the CPU
fixture can rank whether a recipe pins scale 0 (UNI) or lets the
student leave the song. Plus-only (``faithful_plus``) is still the
leftover-gated + teacher with no ``MSE(0)``.

CPU only. No Hub, no GPU, no Music 3 weights. Does not change the live
default (``--lm_target v9`` / ``--pole_mode hidden``).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from analysis.slider2d.exam import (
    PairField,
    close_field,
    divergent_field,
    rollouts,
    teacher_points,
    unused_e_field,
)
from analysis.slider2d.plus_exam import (
    PLUS_COVER_MIN,
    PLUS_OFF_MAX,
    _continue,
    _off_share,
    _token_share,
    blend_toward_mid,
    nearest_pole,
    plus_bags,
    plus_cover,
    plus_teacher,
)
from conceptmod.textsliders.slider_targets import (
    lm_faithful_plus_neu,
    lm_plus_loss,
    lm_plus_neu_loss,
    lm_slider_loss,
)

PLUS_NEU_HOLD_MIN = 0.85

PLUS_NEU_CELLS = {
    "divergent": divergent_field,
    "close": close_field,
    "unused_e": unused_e_field,
}

# Recipes this scale compares. UNI trains + and 0; plus-only trains +;
# the rest train ±.
PLUS_NEU_RECIPES: list[dict] = [
    {
        "name": "faithful_plus_neu",
        "teacher": "faithful_plus_neu",
        "plus_neu": True,
        "plus_only": False,
    },
    {
        "name": "faithful_plus",
        "teacher": "faithful_plus",
        "plus_neu": False,
        "plus_only": True,
    },
    {
        "name": "leftover_gate_bipolar",
        "teacher": "faithful_sub_e_if_unused",
        "plus_neu": False,
        "plus_only": False,
    },
    {
        "name": "faithful_even_blend",
        "teacher": "faithful_gate_odd_sub_even_blend",
        "plus_neu": False,
        "plus_only": False,
        "even_scale": 0.5,
    },
    {
        "name": "pair_odd_midpoint",
        "teacher": "pair_odd",
        "plus_neu": False,
        "plus_only": False,
    },
]


@dataclass
class OriginResidual:
    """Shared residual with a free origin so scale 0 can leave the song.

    ``δ(σ) = σ·w_odd + |σ|·w_even + w0``. Scale 0 is ``w0``, not forced
    to zero. UNI's ``MSE(0)`` pins ``w0``; plus-only and bipolar do not.
    """

    w: torch.Tensor
    w_even: torch.Tensor
    w0: torch.Tensor

    @classmethod
    def create(cls, field: PairField) -> "OriginResidual":
        return cls(
            torch.zeros(field.dim, requires_grad=True),
            torch.zeros(field.dim, requires_grad=True),
            torch.zeros(field.dim, requires_grad=True),
        )

    def delta(self, scale: float) -> torch.Tensor:
        return float(scale) * self.w + abs(float(scale)) * self.w_even + self.w0

    def parameters(self) -> list[torch.Tensor]:
        return [self.w, self.w_even, self.w0]

    def snapshot(self) -> "OriginResidual":
        return OriginResidual(
            self.w.detach().clone(),
            self.w_even.detach().clone(),
            self.w0.detach().clone(),
        )


def plus_neu_teacher(
    field: PairField,
    row: int,
    *,
    teacher: str,
    leak_dir: torch.Tensor | None = None,
    even_scale: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """+ teacher and a canary minus reference (never a plus+neu score)."""
    mode = str(teacher).strip().lower()
    pos, neg, neu = field.poles(row)
    if mode == "faithful_plus_neu":
        plus = lm_faithful_plus_neu(
            pos, neg, neu, leak_dir, slider_dir=field.short_u()
        )
        return plus, neg
    if mode == "faithful_plus":
        return plus_teacher(
            field, row, teacher=teacher, leak_dir=leak_dir, even_scale=even_scale
        )
    return teacher_points(
        field, row, teacher=teacher, leak_dir=leak_dir, even_scale=even_scale
    )


def drift_from_neu(
    student: torch.Tensor,
    neu: torch.Tensor,
    pos: torch.Tensor,
    mid: torch.Tensor,
) -> float:
    """0 at neu, 1 at pos or mid. Uses the nearer of pos / mid."""
    d_neu = float((student - neu).norm())
    d_pos = float((student - pos).norm())
    d_mid = float((student - mid).norm())
    nearer = min(d_pos, d_mid)
    return d_neu / (d_neu + nearer + 1e-8)


def neu_hold(overlap_neu: float, drift: float) -> float:
    """Sortable hold in ``[0, 1]``. High is on-lyrics and still at neu."""
    return max(0.0, min(1.0, min(float(overlap_neu), 1.0 - float(drift))))


def neu_bags(field: PairField) -> frozenset[int]:
    """Lyrics field plus tokens the neu caption itself sings."""
    head = field.readout()
    words: set[int] = set()
    for row in range(int(field.rows)):
        _pos, _neg, neu = field.poles(row)
        seqs, _ = rollouts(field, neu, head, row=row, sign=0.0)
        for seq in seqs:
            words |= set(seq)
        words.add(head.index(f"lyric{row}"))
    return frozenset(words)


def fit_plus_neu_exam(
    field: PairField,
    *,
    teacher: str,
    leak_dir: torch.Tensor | None = None,
    even_scale: float = 1.0,
    plus_only: bool = False,
    plus_neu: bool = False,
    steps: int = 400,
    lr: float = 0.08,
    seed: int = 0,
) -> OriginResidual:
    """Fit one shared residual. UNI adds ``MSE(0)``; plus-only does not."""
    if plus_only and plus_neu:
        raise ValueError("plus_only and plus_neu are mutually exclusive")
    targets = [
        plus_neu_teacher(
            field, row, teacher=teacher, leak_dir=leak_dir, even_scale=even_scale
        )
        for row in range(int(field.rows))
    ]
    neutrals = [field.poles(row)[2] for row in range(int(field.rows))]
    torch.manual_seed(int(seed))
    residual = OriginResidual.create(field)
    opt = torch.optim.Adam(residual.parameters(), lr=float(lr))
    for _ in range(int(steps)):
        total = None
        for (t_plus, t_minus), neu in zip(targets, neutrals):
            pred_plus = neu + residual.delta(1.0)
            pred_minus = neu + residual.delta(-1.0)
            pred_zero = neu + residual.delta(0.0)
            if plus_neu:
                term = lm_plus_neu_loss(pred_plus, t_plus, pred_zero, neu)
            elif plus_only:
                term = lm_plus_loss(pred_plus, t_plus)
            else:
                term = lm_slider_loss(pred_plus, pred_minus, t_plus, t_minus)
            total = term if total is None else total + term
        loss = total / float(len(targets))
        opt.zero_grad()
        loss.backward()
        opt.step()
    return residual.snapshot()


def score_plus_neu_exam(
    name: str,
    field: PairField,
    *,
    teacher: str,
    leak_dir: torch.Tensor | None = None,
    even_scale: float = 1.0,
    plus_only: bool = False,
    plus_neu: bool = False,
    steps: int = 400,
    seed: int = 0,
) -> dict:
    """Fit, then score + cover / off-caption and scale-0 neu_hold."""
    residual = fit_plus_neu_exam(
        field,
        teacher=teacher,
        leak_dir=leak_dir,
        even_scale=even_scale,
        plus_only=plus_only,
        plus_neu=plus_neu,
        steps=steps,
        seed=seed,
    )
    bags = plus_bags(field)
    neu_bag = neu_bags(field)
    d_plus = residual.delta(1.0)
    d_minus = residual.delta(-1.0)
    d_zero = residual.delta(0.0)
    overlap_rows: list[float] = []
    off_rows: list[float] = []
    blend_rows: list[float] = []
    cover_rows: list[float] = []
    neu_overlap_rows: list[float] = []
    neu_drift_rows: list[float] = []
    neu_hold_rows: list[float] = []
    canary_overlap: list[float] = []
    canary_off: list[float] = []
    canary_landed: list[str] = []
    sings_plus: list[str] = []
    sings_zero: list[str] = []
    sings_minus: list[str] = []
    head = field.readout()
    for row in range(int(field.rows)):
        pos, neg, neu = field.poles(row)
        mid = 0.5 * (pos + neg)
        student_plus = neu + d_plus
        student_minus = neu + d_minus
        student_zero = neu + d_zero
        plus_seqs = _continue(field, student_plus, row=row, sign=1.0)
        zero_seqs = _continue(field, student_zero, row=row, sign=0.0)
        minus_seqs = _continue(field, student_minus, row=row, sign=-1.0)
        overlap = _token_share(plus_seqs, bags["pos"])
        off = _off_share(plus_seqs, bags["plus_corpus"])
        blend = blend_toward_mid(student_plus, pos, mid, neg)
        cover = plus_cover(overlap, blend)
        neu_ov = _token_share(zero_seqs, neu_bag)
        drift = drift_from_neu(student_zero, neu, pos, mid)
        hold = neu_hold(neu_ov, drift)
        overlap_rows.append(overlap)
        off_rows.append(off)
        blend_rows.append(blend)
        cover_rows.append(cover)
        neu_overlap_rows.append(neu_ov)
        neu_drift_rows.append(drift)
        neu_hold_rows.append(hold)
        canary_overlap.append(_token_share(minus_seqs, bags["neg"]))
        canary_off.append(_off_share(minus_seqs, bags["minus_corpus"]))
        canary_landed.append(nearest_pole(student_minus, pos, neu, neg))
        sings_plus.append(" ".join(head.tokens[t] for t in plus_seqs[0]))
        sings_zero.append(" ".join(head.tokens[t] for t in zero_seqs[0]))
        sings_minus.append(" ".join(head.tokens[t] for t in minus_seqs[0]))
    cover = sum(cover_rows) / len(cover_rows)
    off_caption = sum(off_rows) / len(off_rows)
    overlap_pos = sum(overlap_rows) / len(overlap_rows)
    blend = sum(blend_rows) / len(blend_rows)
    hold = sum(neu_hold_rows) / len(neu_hold_rows)
    overlap_neu = sum(neu_overlap_rows) / len(neu_overlap_rows)
    drift = sum(neu_drift_rows) / len(neu_drift_rows)
    hit = bool(
        cover >= PLUS_COVER_MIN
        and off_caption <= PLUS_OFF_MAX
        and hold >= PLUS_NEU_HOLD_MIN
    )
    landed = max(set(canary_landed), key=canary_landed.count)
    canary_off_mean = sum(canary_off) / len(canary_off)
    canary_dangerous = bool(landed == "pos" or canary_off_mean > PLUS_OFF_MAX)
    return {
        "name": name,
        "cell": field.kind,
        "teacher": teacher,
        "plus_only": bool(plus_only),
        "plus_neu": bool(plus_neu),
        "even_scale": float(even_scale),
        "cover": cover,
        "off_caption": off_caption,
        "neu_hold": hold,
        "overlap_pos": overlap_pos,
        "blend_toward_mid": blend,
        "overlap_neu": overlap_neu,
        "drift_from_neu": drift,
        "hit": hit,
        "sings_plus": " | ".join(sings_plus),
        "sings_zero": " | ".join(sings_zero),
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


def plus_neu_exam_table(*, steps: int = 400, seed: int = 0) -> dict[str, list[dict]]:
    """Score every plus+neu recipe on the two cheap pair types (and unused_e)."""
    out: dict[str, list[dict]] = {}
    for cell_name, ctor in PLUS_NEU_CELLS.items():
        field = ctor(seed=seed)
        leak = field.declared_e()
        rows = []
        for cand in PLUS_NEU_RECIPES:
            rows.append(
                score_plus_neu_exam(
                    cand["name"],
                    field,
                    teacher=cand["teacher"],
                    leak_dir=leak,
                    even_scale=float(cand.get("even_scale", 1.0)),
                    plus_only=bool(cand["plus_only"]),
                    plus_neu=bool(cand["plus_neu"]),
                    steps=steps,
                    seed=seed,
                )
            )
        out[cell_name] = rows
    return out


def plus_neu_sort_key(row: dict) -> tuple:
    """In-box first, then neu_hold, then cover, then off-caption (asc)."""
    return (
        0 if row["hit"] else 1,
        -float(row["neu_hold"]),
        -float(row["cover"]),
        float(row["off_caption"]),
        str(row["name"]),
    )


def plus_neu_rank(table: dict[str, list[dict]]) -> list[dict]:
    """Combined rank on the two required pairs (divergent, close)."""
    by: dict[str, dict[str, dict]] = {}
    for cell, rows in table.items():
        by[cell] = {r["name"]: r for r in rows}
    names = [c["name"] for c in PLUS_NEU_RECIPES]
    ranked: list[dict] = []
    for name in names:
        cells = [by[c][name] for c in ("divergent", "close")]
        ranked.append(
            {
                "name": name,
                "in_box": all(r["hit"] for r in cells),
                "hit_divergent": cells[0]["hit"],
                "hit_close": cells[1]["hit"],
                "neu_hold": sum(r["neu_hold"] for r in cells) / 2.0,
                "cover": sum(r["cover"] for r in cells) / 2.0,
                "off_caption": sum(r["off_caption"] for r in cells) / 2.0,
                "plus_neu": cells[0]["plus_neu"],
                "plus_only": cells[0]["plus_only"],
            }
        )
    ranked.sort(
        key=lambda r: (
            0 if r["in_box"] else 1,
            -float(r["neu_hold"]),
            -float(r["cover"]),
            float(r["off_caption"]),
            str(r["name"]),
        )
    )
    for i, row in enumerate(ranked, start=1):
        row["rank"] = i
    return ranked


def plus_neu_helps(table: dict[str, list[dict]]) -> dict:
    """Does UNI beat leftover-gated plus-only on neu_hold and keep + cover?"""
    by: dict[str, dict[str, dict]] = {}
    for cell, rows in table.items():
        by[cell] = {r["name"]: r for r in rows}
    required = ("divergent", "close")
    uni = [by[c]["faithful_plus_neu"] for c in required]
    plus = [by[c]["faithful_plus"] for c in required]
    uni_hits = [r["hit"] for r in uni]
    plus_hits = [r["hit"] for r in plus]
    beats_hold = all(
        u["neu_hold"] > p["neu_hold"] + 1e-6 for u, p in zip(uni, plus)
    )
    keeps_cover = all(
        u["cover"] + 1e-6 >= min(p["cover"], PLUS_COVER_MIN) for u, p in zip(uni, plus)
    )
    return {
        "uni_hits_required": uni_hits,
        "plus_hits_required": plus_hits,
        "beats_plus_on_neu_hold": beats_hold,
        "keeps_plus_cover": keeps_cover,
        "uni_cover": [r["cover"] for r in uni],
        "plus_cover": [r["cover"] for r in plus],
        "uni_neu_hold": [r["neu_hold"] for r in uni],
        "plus_neu_hold": [r["neu_hold"] for r in plus],
    }
