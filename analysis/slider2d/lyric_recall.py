"""+1 lyric_recall: yaml lyrics on a continuation, not last-hidden cover.

The plus+neu exam scores caption-word cover and scale-0 neu_hold off a
**last hidden**. Live UNI (``faithful_plus_neu``) matches that last
token to raw ``h+`` and still shreds lyrics. Two failures, not mixed:

1. Two-song yaml (v4): ``h+`` is another track. High ``c+`` = arrived
   at the other song. Not this page.
2. Last-token transplant (uni-v2, same-room yaml): student =
   encode(neu tokens, LoRA @ +1), teacher last = encode(pos tokens).
   Loss is last-hidden MSE only. The last real token is the
   continue-from token. The LoRA still rewrites every prefix token,
   including the yaml lyrics. Matching last hidden to ``h+`` does not
   match the KV of the neu prefix. Generation then has a pos-like last
   hidden and a neu KV. + REF (pos caption, no LoRA) still sings the
   yaml line.

The last-hidden plus+neu fixture cannot see (2): decoding from a last
hidden that already equals ``h+`` reproduces the + caption's own
rollout, lyrics included. This cell is the smallest sequence that can:

- prefix tokens = yaml ``lyrics`` (shared song)
- last token = continue
- LoRA residual on all positions (shared weights + prefix extra)
- last hidden is causal in the prefix (attention mix)

``lyric_recall`` is scored on the student's **sung line** — the
continuation read off the prefix KV — against the yaml lyric sheet,
not against + caption words (punk / loud / grit). Scale-0 lyric_recall
and + REF are logged so REF+ high + student +1 low diagnoses a
last-token transplant.

A hit is high +1 lyric_recall AND high cover (still the concept) AND
high neu_hold at 0. Rank key is lyric_recall@+1, then cover. Not
scored: ``leak_frac``, ``c+``, ``p%``, pair-odd, ``exam_score``.

CPU only. No Hub, no GPU, no Music 3 weights. Does not change the live
default (``--lm_target v9`` / ``--pole_mode hidden``).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from analysis.slider2d.exam import PairField, close_field, divergent_field, rollouts
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
)
from analysis.slider2d.plus_neu_exam import (
    PLUS_NEU_HOLD_MIN,
    drift_from_neu,
    neu_bags,
    neu_hold,
    plus_neu_teacher,
)
from conceptmod.textsliders.slider_targets import (
    lm_plus_loss,
    lm_plus_neu_loss,
    lm_plus_neu_prefix_loss,
    lm_project_last_delta_off_lyric,
    lm_slider_loss,
)

LYRIC_RECALL_MIN = 0.85
PREFIX_LEN = 4
ATTEND = 1.0

LYRIC_CELLS = {
    "divergent": divergent_field,
    "close": close_field,
}

LYRIC_RECIPES: list[dict] = [
    {
        "name": "faithful_plus_neu",
        "teacher": "faithful_plus_neu",
        "plus_neu": True,
        "plus_only": False,
        "prefix_hold": False,
        "prefix_orth": False,
    },
    {
        "name": "faithful_plus",
        "teacher": "faithful_plus",
        "plus_neu": False,
        "plus_only": True,
        "prefix_hold": False,
        "prefix_orth": False,
    },
    {
        "name": "faithful_plus_neu_prefix",
        "teacher": "faithful_plus_neu",
        "plus_neu": True,
        "plus_only": False,
        "prefix_hold": True,
        "prefix_orth": False,
    },
    {
        "name": "leftover_gate_bipolar",
        "teacher": "faithful_sub_e_if_unused",
        "plus_neu": False,
        "plus_only": False,
        "prefix_hold": False,
        "prefix_orth": False,
    },
    {
        "name": "pair_odd_midpoint",
        "teacher": "pair_odd",
        "plus_neu": False,
        "plus_only": False,
        "prefix_hold": False,
        "prefix_orth": False,
    },
]


@dataclass
class SequenceResidual:
    """LoRA on every position: shared weights plus a prefix extra.

    Live last-token MSE applies one LoRA at every token; attention also
    routes last-hidden gradients into the prefix. Shared ``δ`` is that
    one weight. Prefix extra is the activation-dependent leftover that
    prefix-hold can use to cancel the shared rewrite on lyric tokens
    while the last token still carries the concept.

    Last-token-only has no prefix extra target, so the shared path
    rewrites lyrics. Prefix-hold sets ``shared + prefix_extra = 0`` on
    the lyric tokens and leaves ``shared`` on the continue token.
    """

    w: torch.Tensor
    w_even: torch.Tensor
    w0: torch.Tensor
    p: torch.Tensor
    p_even: torch.Tensor
    p0: torch.Tensor

    @classmethod
    def create(cls, field: PairField) -> "SequenceResidual":
        z = torch.zeros
        return cls(
            z(field.dim, requires_grad=True),
            z(field.dim, requires_grad=True),
            z(field.dim, requires_grad=True),
            z(field.dim, requires_grad=True),
            z(field.dim, requires_grad=True),
            z(field.dim, requires_grad=True),
        )

    def shared(self, scale: float) -> torch.Tensor:
        s = float(scale)
        return s * self.w + abs(s) * self.w_even + self.w0

    def prefix_delta(self, scale: float) -> torch.Tensor:
        s = float(scale)
        return self.shared(s) + s * self.p + abs(s) * self.p_even + self.p0

    def last_delta(self, scale: float) -> torch.Tensor:
        return self.shared(scale)

    def parameters(self) -> list[torch.Tensor]:
        return [self.w, self.w_even, self.w0, self.p, self.p_even, self.p0]

    def snapshot(self) -> "SequenceResidual":
        return SequenceResidual(
            self.w.detach().clone(),
            self.w_even.detach().clone(),
            self.w0.detach().clone(),
            self.p.detach().clone(),
            self.p_even.detach().clone(),
            self.p0.detach().clone(),
        )


def lyric_embeds(field: PairField, row: int) -> torch.Tensor:
    """Same-room yaml lyrics: prefix tokens sit on the row's lyric axis."""
    base = float(field.lyric) * field.lyric_dir(row)
    return base.unsqueeze(0).expand(int(PREFIX_LEN), -1).clone()


def lyric_bag(field: PairField, row: int) -> frozenset[int]:
    """Yaml ``lyrics`` field for this prompt row — not + caption words."""
    return frozenset({field.readout().index(f"lyric{int(row)}")})


def encode_sequence(
    field: PairField,
    residual: SequenceResidual,
    row: int,
    scale: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Student encode(neu tokens, LoRA @ scale) → (prefix, last).

    Last hidden = continue content + attend · mean(prefix). Teacher
    encode(pos) is the same prefix (same lyrics) with last = ``h+``.
    """
    prefix_base = lyric_embeds(field, row)
    _pos, _neg, neu = field.poles(row)
    last_base = neu - float(ATTEND) * prefix_base.mean(0)
    prefix = prefix_base + residual.prefix_delta(scale)
    last = last_base + residual.last_delta(scale) + float(ATTEND) * prefix.mean(0)
    return prefix, last


def ref_plus_sequence(field: PairField, row: int) -> tuple[torch.Tensor, torch.Tensor]:
    """+ REF: encode(pos tokens), no student. Prefix is still the yaml line."""
    pos, _neg, _neu = field.poles(row)
    return lyric_embeds(field, row), pos


def sung_line(field: PairField, prefix: torch.Tensor) -> list[int]:
    """Continuation of the stored lyrics: greedy next token at each prefix pos."""
    head = field.readout()
    return [int(head.policy(h).argmax()) for h in prefix]


def lyric_recall(field: PairField, prefix: torch.Tensor, row: int) -> float:
    """Sortable [0, 1]: yaml-lyric overlap on the sung-line continuation.

    Reference is the yaml ``lyrics`` field, not + caption words. The
    sung line is the greedy continuation of the prefix KV. Overlap is
    the share of those tokens that are on the lyric sheet; it equals
    ``1 − off_lyric`` on this bag. Gate: ``≥ 0.85``.
    """
    bag = lyric_bag(field, row)
    seq = sung_line(field, prefix)
    if not seq:
        return 0.0
    return sum(1.0 for t in seq if t in bag) / float(len(seq))


def off_lyric(field: PairField, prefix: torch.Tensor, row: int) -> float:
    """Share of sung-line tokens off the yaml lyric sheet. Lower is better."""
    bag = lyric_bag(field, row)
    seq = sung_line(field, prefix)
    if not seq:
        return 1.0
    return sum(1.0 for t in seq if t not in bag) / float(len(seq))


def last_hidden_cannot_see_transplant(field: PairField) -> dict:
    """Naive lyric score on a last-hidden continuation that already equals ``h+``.

    Matching last hidden to ``h+`` reproduces the + caption rollout,
    which includes whatever lyric mass lives in ``h+``. That number is
    not +1 lyric survival after a prefix rewrite. Logged so the page
    can say the old fixture cannot show the miss.
    """
    bags = [lyric_bag(field, row) for row in range(int(field.rows))]
    shares: list[float] = []
    for row in range(int(field.rows)):
        pos, _neg, _neu = field.poles(row)
        seqs = _continue(field, pos, row=row, sign=1.0)
        shares.append(_token_share(seqs, bags[row]))
    return {
        "last_hidden_lyric_share": sum(shares) / len(shares) if shares else 0.0,
        "note": (
            "last-hidden continuation of h+ still carries lyric mass from "
            "the pole vector; it cannot see a prefix KV rewrite"
        ),
    }


def fit_lyric_exam(
    field: PairField,
    *,
    teacher: str,
    leak_dir: torch.Tensor | None = None,
    even_scale: float = 1.0,
    plus_only: bool = False,
    plus_neu: bool = False,
    prefix_hold: bool = False,
    prefix_orth: bool = False,
    steps: int = 400,
    lr: float = 0.08,
    seed: int = 0,
) -> SequenceResidual:
    """Fit a sequence LoRA. Prefix-hold adds MSE(prefix → encode(neu))."""
    if plus_only and plus_neu:
        raise ValueError("plus_only and plus_neu are mutually exclusive")
    targets = [
        plus_neu_teacher(
            field, row, teacher=teacher, leak_dir=leak_dir, even_scale=even_scale
        )
        for row in range(int(field.rows))
    ]
    torch.manual_seed(int(seed))
    residual = SequenceResidual.create(field)
    opt = torch.optim.Adam(residual.parameters(), lr=float(lr))
    for _ in range(int(steps)):
        total = None
        for row, (t_plus, t_minus) in enumerate(targets):
            _pos, _neg, neu = field.poles(row)
            prefix_neu = lyric_embeds(field, row)
            pref_p, last_p = encode_sequence(field, residual, row, 1.0)
            pref_m, last_m = encode_sequence(field, residual, row, -1.0)
            pref_0, last_0 = encode_sequence(field, residual, row, 0.0)
            tgt_last = t_plus
            if prefix_orth:
                delta = t_plus - neu
                tgt_last = neu + lm_project_last_delta_off_lyric(delta, prefix_neu)
            if prefix_hold:
                term = lm_plus_neu_prefix_loss(
                    last_p, tgt_last, last_0, neu, pref_p, prefix_neu
                )
            elif plus_neu:
                term = lm_plus_neu_loss(last_p, tgt_last, last_0, neu)
            elif plus_only:
                term = lm_plus_loss(last_p, tgt_last)
            else:
                term = lm_slider_loss(last_p, last_m, tgt_last, t_minus)
            total = term if total is None else total + term
        loss = total / float(len(targets))
        opt.zero_grad()
        loss.backward()
        opt.step()
    return residual.snapshot()


def score_lyric_exam(
    name: str,
    field: PairField,
    *,
    teacher: str,
    leak_dir: torch.Tensor | None = None,
    even_scale: float = 1.0,
    plus_only: bool = False,
    plus_neu: bool = False,
    prefix_hold: bool = False,
    prefix_orth: bool = False,
    steps: int = 400,
    seed: int = 0,
) -> dict:
    """Fit, then score +1 lyric_recall, cover, neu_hold, off-caption."""
    residual = fit_lyric_exam(
        field,
        teacher=teacher,
        leak_dir=leak_dir,
        even_scale=even_scale,
        plus_only=plus_only,
        plus_neu=plus_neu,
        prefix_hold=prefix_hold,
        prefix_orth=prefix_orth,
        steps=steps,
        seed=seed,
    )
    bags = plus_bags(field)
    neu_bag = neu_bags(field)
    overlap_rows: list[float] = []
    off_rows: list[float] = []
    blend_rows: list[float] = []
    cover_rows: list[float] = []
    neu_overlap_rows: list[float] = []
    neu_drift_rows: list[float] = []
    neu_hold_rows: list[float] = []
    lyric_plus: list[float] = []
    lyric_zero: list[float] = []
    lyric_ref: list[float] = []
    off_lyric_plus: list[float] = []
    sings_plus: list[str] = []
    sings_zero: list[str] = []
    sings_lyric: list[str] = []
    sings_ref: list[str] = []
    canary_overlap: list[float] = []
    canary_off: list[float] = []
    canary_landed: list[str] = []
    head = field.readout()
    for row in range(int(field.rows)):
        pos, neg, neu = field.poles(row)
        mid = 0.5 * (pos + neg)
        pref_p, last_p = encode_sequence(field, residual, row, 1.0)
        pref_0, last_0 = encode_sequence(field, residual, row, 0.0)
        _pref_m, last_m = encode_sequence(field, residual, row, -1.0)
        ref_pref, ref_last = ref_plus_sequence(field, row)
        plus_seqs = _continue(field, last_p, row=row, sign=1.0)
        zero_seqs = _continue(field, last_0, row=row, sign=0.0)
        minus_seqs = _continue(field, last_m, row=row, sign=-1.0)
        overlap = _token_share(plus_seqs, bags["pos"])
        off = _off_share(plus_seqs, bags["plus_corpus"])
        blend = blend_toward_mid(last_p, pos, mid, neg)
        cover = plus_cover(overlap, blend)
        neu_ov = _token_share(zero_seqs, neu_bag)
        drift = drift_from_neu(last_0, neu, pos, mid)
        hold = neu_hold(neu_ov, drift)
        overlap_rows.append(overlap)
        off_rows.append(off)
        blend_rows.append(blend)
        cover_rows.append(cover)
        neu_overlap_rows.append(neu_ov)
        neu_drift_rows.append(drift)
        neu_hold_rows.append(hold)
        lyric_plus.append(lyric_recall(field, pref_p, row))
        lyric_zero.append(lyric_recall(field, pref_0, row))
        lyric_ref.append(lyric_recall(field, ref_pref, row))
        off_lyric_plus.append(off_lyric(field, pref_p, row))
        canary_overlap.append(_token_share(minus_seqs, bags["neg"]))
        canary_off.append(_off_share(minus_seqs, bags["minus_corpus"]))
        canary_landed.append(nearest_pole(last_m, pos, neu, neg))
        sings_plus.append(" ".join(head.tokens[t] for t in plus_seqs[0]))
        sings_zero.append(" ".join(head.tokens[t] for t in zero_seqs[0]))
        sings_lyric.append(" ".join(head.tokens[t] for t in sung_line(field, pref_p)))
        sings_ref.append(" ".join(head.tokens[t] for t in sung_line(field, ref_pref)))
        _ = ref_last
    cover = sum(cover_rows) / len(cover_rows)
    off_caption = sum(off_rows) / len(off_rows)
    hold = sum(neu_hold_rows) / len(neu_hold_rows)
    recall = sum(lyric_plus) / len(lyric_plus)
    recall0 = sum(lyric_zero) / len(lyric_zero)
    recall_ref = sum(lyric_ref) / len(lyric_ref)
    off_ly = sum(off_lyric_plus) / len(off_lyric_plus)
    old_box = bool(
        cover >= PLUS_COVER_MIN
        and off_caption <= PLUS_OFF_MAX
        and hold >= PLUS_NEU_HOLD_MIN
    )
    hit = bool(
        recall >= LYRIC_RECALL_MIN
        and cover >= PLUS_COVER_MIN
        and hold >= PLUS_NEU_HOLD_MIN
    )
    transplant = bool(recall_ref >= LYRIC_RECALL_MIN and recall < LYRIC_RECALL_MIN)
    landed = max(set(canary_landed), key=canary_landed.count)
    canary_off_mean = sum(canary_off) / len(canary_off)
    return {
        "name": name,
        "cell": field.kind,
        "teacher": teacher,
        "plus_only": bool(plus_only),
        "plus_neu": bool(plus_neu),
        "prefix_hold": bool(prefix_hold),
        "prefix_orth": bool(prefix_orth),
        "lyric_recall": recall,
        "lyric_recall_zero": recall0,
        "lyric_recall_ref_plus": recall_ref,
        "off_lyric": off_ly,
        "cover": cover,
        "off_caption": off_caption,
        "neu_hold": hold,
        "overlap_pos": sum(overlap_rows) / len(overlap_rows),
        "overlap_neu": sum(neu_overlap_rows) / len(neu_overlap_rows),
        "old_box": old_box,
        "hit": hit,
        "last_token_transplant": transplant,
        "sings_plus": " | ".join(sings_plus),
        "sings_zero": " | ".join(sings_zero),
        "sings_lyric": " | ".join(sings_lyric),
        "sings_ref_plus": " | ".join(sings_ref),
        "canary": {
            "scored": False,
            "minus_overlap_neg": sum(canary_overlap) / len(canary_overlap),
            "minus_off_caption": canary_off_mean,
            "minus_landed": landed,
        },
        "pole_cos": float(
            F.cosine_similarity(
                (encode_sequence(field, residual, 0, 1.0)[1] - field.poles(0)[2])
                .flatten()
                .unsqueeze(0),
                (field.poles(0)[0] - field.poles(0)[2]).flatten().unsqueeze(0),
            ).squeeze()
        ),
    }


def lyric_exam_table(*, steps: int = 400, seed: int = 0) -> dict[str, list[dict]]:
    """Score every lyric-recall recipe on divergent (grit-like) and close."""
    out: dict[str, list[dict]] = {}
    for cell_name, ctor in LYRIC_CELLS.items():
        field = ctor(seed=seed)
        leak = field.declared_e()
        rows = []
        for cand in LYRIC_RECIPES:
            rows.append(
                score_lyric_exam(
                    cand["name"],
                    field,
                    teacher=cand["teacher"],
                    leak_dir=leak,
                    even_scale=float(cand.get("even_scale", 1.0)),
                    plus_only=bool(cand["plus_only"]),
                    plus_neu=bool(cand["plus_neu"]),
                    prefix_hold=bool(cand["prefix_hold"]),
                    prefix_orth=bool(cand["prefix_orth"]),
                    steps=steps,
                    seed=seed,
                )
            )
        out[cell_name] = rows
    return out


def lyric_rank(table: dict[str, list[dict]]) -> list[dict]:
    """Combined rank: lyric_recall@+1, then cover. Required pairs only."""
    by: dict[str, dict[str, dict]] = {}
    for cell, rows in table.items():
        by[cell] = {r["name"]: r for r in rows}
    names = [c["name"] for c in LYRIC_RECIPES]
    required = [c for c in ("divergent", "close") if c in by]
    ranked: list[dict] = []
    for name in names:
        cells = [by[c][name] for c in required]
        ranked.append(
            {
                "name": name,
                "in_box": all(r["hit"] for r in cells),
                "hit_divergent": cells[0]["hit"] if "divergent" in by else None,
                "hit_close": cells[1]["hit"] if "close" in by else None,
                "lyric_recall": sum(r["lyric_recall"] for r in cells) / len(cells),
                "cover": sum(r["cover"] for r in cells) / len(cells),
                "neu_hold": sum(r["neu_hold"] for r in cells) / len(cells),
                "off_caption": sum(r["off_caption"] for r in cells) / len(cells),
                "old_box": all(r["old_box"] for r in cells),
                "last_token_transplant": any(r["last_token_transplant"] for r in cells),
                "prefix_hold": cells[0]["prefix_hold"],
                "plus_neu": cells[0]["plus_neu"],
                "plus_only": cells[0]["plus_only"],
            }
        )
    ranked.sort(
        key=lambda r: (
            -float(r["lyric_recall"]),
            -float(r["cover"]),
            -float(r["neu_hold"]),
            str(r["name"]),
        )
    )
    for i, row in enumerate(ranked, start=1):
        row["rank"] = i
    return ranked


def lyric_verdict(table: dict[str, list[dict]]) -> dict:
    """Did we replicate the last-token transplant, and does prefix-hold fix it?"""
    by: dict[str, dict[str, dict]] = {}
    for cell, rows in table.items():
        by[cell] = {r["name"]: r for r in rows}
    required = [c for c in ("divergent", "close") if c in by]
    uni = [by[c]["faithful_plus_neu"] for c in required]
    prefix = [by[c]["faithful_plus_neu_prefix"] for c in required]
    plus = [by[c]["faithful_plus"] for c in required]
    replicated = any(r["last_token_transplant"] and r["old_box"] for r in uni)
    prefix_lifts = all(
        p["lyric_recall"] + 1e-6 >= u["lyric_recall"] for p, u in zip(prefix, uni)
    ) and any(
        p["lyric_recall"] > u["lyric_recall"] + 1e-6 for p, u in zip(prefix, uni)
    )
    prefix_keeps_cover = all(
        p["cover"] + 1e-6 >= min(u["cover"], PLUS_COVER_MIN) for p, u in zip(prefix, uni)
    )
    prefix_hits = all(r["hit"] for r in prefix)
    return {
        "replicated_last_token_transplant": replicated,
        "uni_old_box": [r["old_box"] for r in uni],
        "uni_lyric_recall": [r["lyric_recall"] for r in uni],
        "uni_cover": [r["cover"] for r in uni],
        "uni_neu_hold": [r["neu_hold"] for r in uni],
        "uni_ref_plus": [r["lyric_recall_ref_plus"] for r in uni],
        "prefix_lyric_recall": [r["lyric_recall"] for r in prefix],
        "prefix_cover": [r["cover"] for r in prefix],
        "prefix_neu_hold": [r["neu_hold"] for r in prefix],
        "prefix_lifts_lyric_recall": prefix_lifts,
        "prefix_keeps_cover": prefix_keeps_cover,
        "prefix_hits_required": prefix_hits,
        "plus_lyric_recall": [r["lyric_recall"] for r in plus],
        "last_hidden_blind": {
            cell: last_hidden_cannot_see_transplant(LYRIC_CELLS[cell]())
            for cell in required
        },
    }
