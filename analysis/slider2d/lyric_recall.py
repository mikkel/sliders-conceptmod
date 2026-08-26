"""OOD detection with existing metrics + last-token transplant fixture.

The plus+neu exam scores cover / neu_hold / off-caption off a **last
hidden**. Live UNI (``faithful_plus_neu``) matches that last token to
raw ``h+`` and still shreds lyrics on grit-like plus LoRAs. This cell
does two things, in this order:

1. Score **existing** logged metrics (plus-exam off-caption, pair-exam
   ``same_words`` / ``off_corpus`` / coherence, sheet garble,
   sheet ``lyric_mass`` / continuation vs the yaml lyric sheet) on
   last-hidden vs prefix sung-line. Show which ones would have flagged
   grit-like +1 shred vs gender/tempo keep. Do not invent a new scale
   unless an existing one cannot see the miss.
2. Keep a cheap sequence fixture so those metrics have a prefix sung
   line to read. Last-hidden-only scoring cannot see a prefix KV
   rewrite: decoding from a last hidden that already equals ``h+``
   reproduces the + caption rollout.

Prefix-hold (``faithful_plus_neu_prefix``) is secondary — already
wired, not the point. Not scored: ``leak_frac``, ``c+``, ``p%``,
pair-odd, ``exam_score``. CPU only. No Hub, no GPU, no Music 3
weights. Does not change the live default (``--lm_target v9`` /
``--pole_mode hidden``).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from analysis.slider2d.exam import (
    EXAM_COHERENCE,
    EXAM_MATCH_KEPT,
    EXAM_ROLL_OFF_MAX,
    EXAM_ROLL_OVERLAP,
    OFF_SHEET_TOKENS,
    TOKEN_SIDE,
    PairField,
    close_field,
    divergent_field,
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
)
from analysis.slider2d.sheet import GARBLE_MAX
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
    lm_plus_neu_lyric_loss,
    lm_plus_neu_prefix_loss,
    lm_project_last_delta_off_lyric,
    lm_project_onto_lyric_span,
    lm_slider_loss,
)

# Existing pair-exam continuation floor, applied to the yaml lyric sheet
# (sheet ``lyric_mass`` / continuation vs lyrics). Not a new scale.
LYRIC_RECALL_MIN = EXAM_ROLL_OVERLAP
GENDER_MOVE_MIN = EXAM_ROLL_OVERLAP
PREFIX_LEN = 4
VD_LEN = 2
ATTEND = 1.0

# Existing gates reused as-is. A metric "flags" when it would have
# called grit-like shred OOD (or gender-like keep OOD — a false alarm).
EXISTING_OOD_METRICS = (
    "plus_off_caption",
    "pair_off_corpus",
    "pair_same_words",
    "pair_coherence",
    "sheet_garble",
    "sheet_lyric_mass",
)

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
        "lyric_hold": False,
        "last_delta_orth": False,
    },
    {
        "name": "faithful_plus",
        "teacher": "faithful_plus",
        "plus_neu": False,
        "plus_only": True,
        "prefix_hold": False,
        "prefix_orth": False,
        "lyric_hold": False,
        "last_delta_orth": False,
    },
    {
        "name": "faithful_plus_neu_prefix",
        "teacher": "faithful_plus_neu",
        "plus_neu": True,
        "plus_only": False,
        "prefix_hold": True,
        "prefix_orth": False,
        "lyric_hold": False,
        "last_delta_orth": False,
    },
    {
        "name": "faithful_plus_neu_lyric",
        "teacher": "faithful_plus_neu",
        "plus_neu": True,
        "plus_only": False,
        "prefix_hold": False,
        "prefix_orth": False,
        "lyric_hold": True,
        "last_delta_orth": False,
    },
    {
        "name": "faithful_plus_neu_orth",
        "teacher": "faithful_plus_neu",
        "plus_neu": True,
        "plus_only": False,
        "prefix_hold": False,
        "prefix_orth": False,
        "lyric_hold": False,
        "last_delta_orth": True,
    },
    {
        "name": "leftover_gate_bipolar",
        "teacher": "faithful_sub_e_if_unused",
        "plus_neu": False,
        "plus_only": False,
        "prefix_hold": False,
        "prefix_orth": False,
        "lyric_hold": False,
        "last_delta_orth": False,
    },
    {
        "name": "pair_odd_midpoint",
        "teacher": "pair_odd",
        "plus_neu": False,
        "plus_only": False,
        "prefix_hold": False,
        "prefix_orth": False,
        "lyric_hold": False,
        "last_delta_orth": False,
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
    v: torch.Tensor
    v_even: torch.Tensor
    v0: torch.Tensor

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

    def lyric_delta(self, scale: float) -> torch.Tensor:
        return self.prefix_delta(scale)

    def vocal_delta(self, scale: float) -> torch.Tensor:
        s = float(scale)
        return self.shared(s) + s * self.v + abs(s) * self.v_even + self.v0

    def last_delta(self, scale: float) -> torch.Tensor:
        return self.shared(scale)

    def parameters(self) -> list[torch.Tensor]:
        return [
            self.w,
            self.w_even,
            self.w0,
            self.p,
            self.p_even,
            self.p0,
            self.v,
            self.v_even,
            self.v0,
        ]

    def snapshot(self) -> "SequenceResidual":
        return SequenceResidual(
            self.w.detach().clone(),
            self.w_even.detach().clone(),
            self.w0.detach().clone(),
            self.p.detach().clone(),
            self.p_even.detach().clone(),
            self.p0.detach().clone(),
            self.v.detach().clone(),
            self.v_even.detach().clone(),
            self.v0.detach().clone(),
        )


def lyric_embeds(field: PairField, row: int) -> torch.Tensor:
    """Same-room yaml lyrics: lyric-span tokens sit on the row's lyric axis."""
    if int(PREFIX_LEN) <= 0 or float(field.lyric) <= 1e-8:
        raise RuntimeError("empty lyrics; lyric-hold cannot locate a yaml lyrics span")
    base = float(field.lyric) * field.lyric_dir(row)
    return base.unsqueeze(0).expand(int(PREFIX_LEN), -1).clone()


def vocal_embeds(field: PairField, row: int, *, pole: str) -> torch.Tensor:
    """Vocal Details span. Neu is ungendered; pos carries woman / concept.

    Not yaml ``lyrics``. Whole-prefix hold pins this to neu; lyric-hold
    leaves it free so gender can still move.
    """
    del row
    if pole == "neu":
        base = 0.85 * field.sheet_dir()
    elif pole == "pos":
        if str(field.kind) == "close":
            base = float(field.delivery) * field.delivery_dir() + 0.15 * field.sheet_dir()
        else:
            track = field.plus_track()
            base = track + 0.15 * field.sheet_dir()
    else:
        raise ValueError(f"vocal pole must be neu or pos, got {pole!r}")
    return base.unsqueeze(0).expand(int(VD_LEN), -1).clone()


def prefix_embeds(field: PairField, row: int, *, pole: str) -> torch.Tensor:
    """Full prefix: Vocal Details then yaml lyrics."""
    return torch.cat([vocal_embeds(field, row, pole=pole), lyric_embeds(field, row)], dim=0)


def vocal_span(prefix: torch.Tensor) -> torch.Tensor:
    if prefix.shape[0] < int(VD_LEN):
        raise RuntimeError("prefix has no Vocal Details span")
    return prefix[: int(VD_LEN)]


def lyric_span(prefix: torch.Tensor) -> torch.Tensor:
    if prefix.shape[0] <= int(VD_LEN):
        raise RuntimeError("empty lyrics; lyric-hold cannot locate a yaml lyrics span")
    return prefix[int(VD_LEN) :]


def lyric_bag(field: PairField, row: int) -> frozenset[int]:
    """Yaml ``lyrics`` field for this prompt row — not + caption words."""
    return frozenset({field.readout().index(f"lyric{int(row)}")})


def encode_sequence(
    field: PairField,
    residual: SequenceResidual,
    row: int,
    scale: float,
    last_delta_orth: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Student encode(neu tokens, LoRA @ scale) → (prefix, last).

    Prefix is Vocal Details (ungendered neu) then yaml lyrics. Last
    hidden = continue content + attend · mean(prefix). Teacher
    encode(pos) last is ``h+``; + REF prefix uses pos Vocal Details
    and the same lyrics.

    ``last_delta_orth`` keeps the last-token UNI residual and lets
    lyric tokens keep only the in-span residual. Vocal Details stay
    on the unprojected vocal extra (not a prefix hold).
    """
    vd_base = vocal_embeds(field, row, pole="neu")
    ly_base = lyric_embeds(field, row)
    prefix_base = torch.cat([vd_base, ly_base], dim=0)
    _pos, _neg, neu = field.poles(row)
    last_base = neu - float(ATTEND) * prefix_base.mean(0)
    vd = vd_base + residual.vocal_delta(scale)
    ly_delta = residual.lyric_delta(scale)
    if last_delta_orth:
        if ly_base.numel() == 0 or float(ly_base.norm()) <= 1e-8:
            raise RuntimeError("lyric span is empty or near-zero; fail closed")
        ly_delta = lm_project_onto_lyric_span(
            ly_delta,
            ly_base,
            fail_closed=True,
        )
    ly = ly_base + ly_delta
    prefix = torch.cat([vd, ly], dim=0)
    last = last_base + residual.last_delta(scale) + float(ATTEND) * prefix.mean(0)
    return prefix, last


def gender_move(field: PairField, prefix: torch.Tensor, row: int) -> float:
    """Vocal Details free: +1 VD may differ from neu (woman allowed).

    Cosine of (student VD − neu VD) with (pos − neu). Whole-prefix hold
    pins VD to neu so this is 0. Last-token UNI, lyric-hold, and
    last-delta-orth leave VD free so gender can still move.
    """
    vd = vocal_span(prefix).mean(0)
    vd_neu = vocal_embeds(field, row, pole="neu").mean(0)
    pos, _neg, neu = field.poles(row)
    got = vd - vd_neu
    want = pos - neu
    if float(got.norm()) <= 1e-6 or float(want.norm()) <= 1e-6:
        return 0.0
    return float(
        F.cosine_similarity(got.flatten().unsqueeze(0), want.flatten().unsqueeze(0))
        .clamp(min=0.0)
        .squeeze()
    )


def ref_plus_sequence(field: PairField, row: int) -> tuple[torch.Tensor, torch.Tensor]:
    """+ REF: encode(pos tokens), no student. Lyrics stay the yaml line."""
    pos, _neg, _neu = field.poles(row)
    return prefix_embeds(field, row, pole="pos"), pos


def sung_line(field: PairField, prefix: torch.Tensor) -> list[int]:
    """Continuation of the stored lyrics: greedy next token at each prefix pos."""
    head = field.readout()
    return [int(head.policy(h).argmax()) for h in prefix]


def lyric_recall(field: PairField, prefix: torch.Tensor, row: int) -> float:
    """Sheet ``lyric_mass`` on the yaml lyrics span (existing lyric sheet).

    Same number as pair-exam continuation overlap against the yaml
    ``lyrics`` field, not + caption / Vocal Details words. Gate is the
    existing pair-exam continuation floor ``EXAM_ROLL_OVERLAP`` (0.85).
    """
    bag = lyric_bag(field, row)
    seq = sung_line(field, lyric_span(prefix) if prefix.shape[0] > int(VD_LEN) else prefix)
    if not seq:
        return 0.0
    return sum(1.0 for t in seq if t in bag) / float(len(seq))


def off_lyric(field: PairField, prefix: torch.Tensor, row: int) -> float:
    """Share of sung-line tokens off the yaml lyric sheet. Lower is better."""
    bag = lyric_bag(field, row)
    seq = sung_line(field, lyric_span(prefix) if prefix.shape[0] > int(VD_LEN) else prefix)
    if not seq:
        return 1.0
    return sum(1.0 for t in seq if t not in bag) / float(len(seq))


def sings_vocal_line(field: PairField, prefix: torch.Tensor, row: int) -> str:
    """Readable Vocal Details label from the *move* off neu.

    Pinned-to-neu stays ``lead``. A delivery / woman rewrite says
    ``woman``. A grit caption rewrite says ``punk``.
    """
    neu = vocal_embeds(field, row, pole="neu")
    labels = []
    for hidden, base in zip(vocal_span(prefix), neu):
        delta = hidden - base
        if float(delta.norm()) <= 1e-4:
            labels.append("lead")
            continue
        scores = {
            "woman": float(delta @ field.delivery_dir()),
            "punk": float(delta @ field.plus_track()),
        }
        labels.append(max(scores, key=scores.get))
    return " ".join(labels)


def _match_kept(seqs: list[list[int]], refs: list[list[int]]) -> float:
    """Pair-exam ``same_words`` / ``roll_match_kept`` on token draws."""
    hits: list[float] = []
    if not refs:
        return 0.0
    for index, seq in enumerate(seqs):
        ref = refs[index % len(refs)]
        n = min(len(seq), len(ref))
        if n <= 0:
            continue
        hits.append(sum(1.0 for x, y in zip(seq[:n], ref[:n]) if x == y) / float(n))
    return sum(hits) / len(hits) if hits else 0.0


def _coherence(field: PairField, seqs: list[list[int]]) -> float:
    """Pair-exam ``roll_coherence``: consecutive tokens stay on one side."""
    head = field.readout()
    scores: list[float] = []
    for seq in seqs:
        sides = [TOKEN_SIDE.get(head.tokens[t], 0.0) for t in seq]
        pairs = list(zip(sides, sides[1:]))
        if not pairs:
            scores.append(1.0)
            continue
        scores.append(sum(1.0 for x, y in pairs if x * y >= 0.0) / float(len(pairs)))
    return sum(scores) / len(scores) if scores else 1.0


def _garble_share(field: PairField, seqs: list[list[int]]) -> float:
    """Sheet garble: share of ``garble_hi`` / ``garble_lo`` tokens."""
    head = field.readout()
    off = {head.index(name) for name in OFF_SHEET_TOKENS if name in head.tokens}
    hits: list[float] = []
    for seq in seqs:
        if not seq:
            continue
        hits.append(sum(1.0 for t in seq if t in off) / float(len(seq)))
    return sum(hits) / len(hits) if hits else 0.0


def existing_metrics_on_seqs(
    field: PairField,
    seqs: list[list[int]],
    row: int,
    teacher_plus: list[list[int]],
) -> dict:
    """Existing logged metrics on one continuation surface.

    ``sheet_lyric_mass`` is continuation vs the yaml lyric sheet — the
    only existing column that can split grit shred from gender keep
    once the sung line is the prefix KV, not the last hidden.
    """
    bags = plus_bags(field)
    pair_corpus = frozenset(bags["pos"] | bags["neg"] | bags["shared"])
    lyric = lyric_bag(field, row)
    values = {
        "plus_off_caption": _off_share(seqs, bags["plus_corpus"]),
        "pair_off_corpus": _off_share(seqs, pair_corpus),
        "pair_same_words": _match_kept(seqs, teacher_plus),
        "pair_coherence": _coherence(field, seqs),
        "sheet_garble": _garble_share(field, seqs),
        "sheet_lyric_mass": _token_share(seqs, lyric),
    }
    flags = {
        "plus_off_caption": values["plus_off_caption"] > PLUS_OFF_MAX,
        "pair_off_corpus": values["pair_off_corpus"] > EXAM_ROLL_OFF_MAX,
        "pair_same_words": values["pair_same_words"] < EXAM_MATCH_KEPT,
        "pair_coherence": values["pair_coherence"] < EXAM_COHERENCE,
        "sheet_garble": values["sheet_garble"] > GARBLE_MAX,
        "sheet_lyric_mass": values["sheet_lyric_mass"] < EXAM_ROLL_OVERLAP,
    }
    return {"values": values, "flags": flags}


def _mean_existing(rows: list[dict]) -> dict:
    if not rows:
        empty = {name: 0.0 for name in EXISTING_OOD_METRICS}
        return {
            "values": empty,
            "flags": {name: False for name in EXISTING_OOD_METRICS},
        }
    values = {
        name: sum(r["values"][name] for r in rows) / len(rows)
        for name in EXISTING_OOD_METRICS
    }
    flags = {
        "plus_off_caption": values["plus_off_caption"] > PLUS_OFF_MAX,
        "pair_off_corpus": values["pair_off_corpus"] > EXAM_ROLL_OFF_MAX,
        "pair_same_words": values["pair_same_words"] < EXAM_MATCH_KEPT,
        "pair_coherence": values["pair_coherence"] < EXAM_COHERENCE,
        "sheet_garble": values["sheet_garble"] > GARBLE_MAX,
        "sheet_lyric_mass": values["sheet_lyric_mass"] < EXAM_ROLL_OVERLAP,
    }
    return {"values": values, "flags": flags}


def existing_ood_verdict(table: dict[str, list[dict]]) -> dict:
    """Which existing metrics flag grit UNI shred vs gender UNI keep."""
    by: dict[str, dict[str, dict]] = {
        cell: {r["name"]: r for r in rows} for cell, rows in table.items()
    }
    uni_div = by["divergent"]["faithful_plus_neu"]
    uni_close = by["close"]["faithful_plus_neu"]
    useful: list[str] = []
    false_alarm: list[str] = []
    blind: list[str] = []
    for surface in ("last_hidden", "prefix_sung"):
        grit_flags = uni_div["existing"][surface]["flags"]
        keep_flags = uni_close["existing"][surface]["flags"]
        for name in EXISTING_OOD_METRICS:
            key = f"{surface}.{name}"
            if grit_flags[name] and not keep_flags[name]:
                useful.append(key)
            elif keep_flags[name] and not grit_flags[name]:
                false_alarm.append(key)
            elif grit_flags[name] and keep_flags[name]:
                blind.append(f"{key} (flags both)")
            else:
                blind.append(key)
    return {
        "useful": useful,
        "false_alarm": false_alarm,
        "blind": blind,
        "only_prefix_lyric_sheet": useful == ["prefix_sung.sheet_lyric_mass"],
        "grit": uni_div["existing"],
        "gender": uni_close["existing"],
        "grit_sings_prefix": uni_div["sings_lyric"],
        "gender_sings_prefix": uni_close["sings_lyric"],
        "grit_sings_last_hidden": uni_div["sings_plus"],
        "gender_sings_last_hidden": uni_close["sings_plus"],
    }


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
    lyric_hold: bool = False,
    last_delta_orth: bool = False,
    steps: int = 400,
    lr: float = 0.08,
    seed: int = 0,
) -> SequenceResidual:
    """Fit a sequence LoRA. Prefix-hold / lyric-hold add prefix MSE.

    ``last_delta_orth`` is encode-side in-span projection on lyric
    tokens, not a lyric-token hold to neu and not a whole-prefix hold.
    Last-token UNI stays ``MSE(last + → raw h+) + MSE(last 0 → h0)``.
    """
    if plus_only and plus_neu:
        raise ValueError("plus_only and plus_neu are mutually exclusive")
    if prefix_hold and lyric_hold:
        raise ValueError("prefix_hold and lyric_hold are mutually exclusive")
    if prefix_hold and last_delta_orth:
        raise ValueError("prefix_hold and last_delta_orth are mutually exclusive")
    if lyric_hold and last_delta_orth:
        raise ValueError("lyric_hold and last_delta_orth are mutually exclusive")
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
            prefix_neu = prefix_embeds(field, row, pole="neu")
            ly_neu = lyric_span(prefix_neu)
            pref_p, last_p = encode_sequence(
                field, residual, row, 1.0, last_delta_orth=last_delta_orth
            )
            pref_m, last_m = encode_sequence(
                field, residual, row, -1.0, last_delta_orth=last_delta_orth
            )
            pref_0, last_0 = encode_sequence(
                field, residual, row, 0.0, last_delta_orth=last_delta_orth
            )
            tgt_last = t_plus
            if prefix_orth:
                delta = t_plus - neu
                tgt_last = neu + lm_project_last_delta_off_lyric(delta, ly_neu)
            if prefix_hold:
                term = lm_plus_neu_prefix_loss(
                    last_p, tgt_last, last_0, neu, pref_p, prefix_neu
                )
            elif lyric_hold:
                term = lm_plus_neu_lyric_loss(
                    last_p, tgt_last, last_0, neu, lyric_span(pref_p), ly_neu
                )
            elif last_delta_orth:
                # Last-token UNI. Lyric-span projection is already in
                # encode_sequence; do not also pin the prefix extra to
                # neu (that is lyric-token hold / prefix-hold).
                term = lm_plus_neu_loss(last_p, tgt_last, last_0, neu)
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
    lyric_hold: bool = False,
    last_delta_orth: bool = False,
    steps: int = 400,
    seed: int = 0,
) -> dict:
    """Fit, then score +1 lyric_recall, gender_move, cover, neu_hold."""
    residual = fit_lyric_exam(
        field,
        teacher=teacher,
        leak_dir=leak_dir,
        even_scale=even_scale,
        plus_only=plus_only,
        plus_neu=plus_neu,
        prefix_hold=prefix_hold,
        prefix_orth=prefix_orth,
        lyric_hold=lyric_hold,
        last_delta_orth=last_delta_orth,
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
    sings_vocal: list[str] = []
    sings_ref: list[str] = []
    gender_rows: list[float] = []
    canary_overlap: list[float] = []
    canary_off: list[float] = []
    canary_landed: list[str] = []
    last_ood_rows: list[dict] = []
    prefix_ood_rows: list[dict] = []
    head = field.readout()
    for row in range(int(field.rows)):
        pos, neg, neu = field.poles(row)
        mid = 0.5 * (pos + neg)
        pref_p, last_p = encode_sequence(
            field, residual, row, 1.0, last_delta_orth=last_delta_orth
        )
        pref_0, last_0 = encode_sequence(
            field, residual, row, 0.0, last_delta_orth=last_delta_orth
        )
        _pref_m, last_m = encode_sequence(
            field, residual, row, -1.0, last_delta_orth=last_delta_orth
        )
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
        gender_rows.append(gender_move(field, pref_p, row))
        canary_overlap.append(_token_share(minus_seqs, bags["neg"]))
        canary_off.append(_off_share(minus_seqs, bags["minus_corpus"]))
        canary_landed.append(nearest_pole(last_m, pos, neu, neg))
        lyric_seq = sung_line(field, lyric_span(pref_p))
        teacher_plus = _continue(field, pos, row=row, sign=1.0)
        last_ood_rows.append(existing_metrics_on_seqs(field, plus_seqs, row, teacher_plus))
        prefix_ood_rows.append(
            existing_metrics_on_seqs(field, [lyric_seq], row, teacher_plus)
        )
        sings_plus.append(" ".join(head.tokens[t] for t in plus_seqs[0]))
        sings_zero.append(" ".join(head.tokens[t] for t in zero_seqs[0]))
        sings_lyric.append(" ".join(head.tokens[t] for t in lyric_seq))
        sings_vocal.append(sings_vocal_line(field, pref_p, row))
        sings_ref.append(" ".join(head.tokens[t] for t in sung_line(field, lyric_span(ref_pref))))
        _ = ref_last
    cover = sum(cover_rows) / len(cover_rows)
    off_caption = sum(off_rows) / len(off_rows)
    hold = sum(neu_hold_rows) / len(neu_hold_rows)
    recall = sum(lyric_plus) / len(lyric_plus)
    recall0 = sum(lyric_zero) / len(lyric_zero)
    recall_ref = sum(lyric_ref) / len(lyric_ref)
    off_ly = sum(off_lyric_plus) / len(off_lyric_plus)
    move = sum(gender_rows) / len(gender_rows)
    old_box = bool(
        cover >= PLUS_COVER_MIN
        and off_caption <= PLUS_OFF_MAX
        and hold >= PLUS_NEU_HOLD_MIN
    )
    lyric_hit = bool(
        recall >= LYRIC_RECALL_MIN
        and cover >= PLUS_COVER_MIN
        and hold >= PLUS_NEU_HOLD_MIN
    )
    gender_hit = bool(
        move >= GENDER_MOVE_MIN
        and cover >= PLUS_COVER_MIN
        and hold >= PLUS_NEU_HOLD_MIN
    )
    hit = lyric_hit
    want_box = bool(
        recall >= LYRIC_RECALL_MIN
        and move >= GENDER_MOVE_MIN
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
        "lyric_hold": bool(lyric_hold),
        "last_delta_orth": bool(last_delta_orth),
        "lyric_recall": recall,
        "gender_move": move,
        "lyric_hit": lyric_hit,
        "gender_hit": gender_hit,
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
        "want_box": want_box,
        "last_token_transplant": transplant,
        "sings_plus": " | ".join(sings_plus),
        "sings_zero": " | ".join(sings_zero),
        "sings_lyric": " | ".join(sings_lyric),
        "sings_vocal": " | ".join(sings_vocal),
        "sings_ref_plus": " | ".join(sings_ref),
        "existing": {
            "last_hidden": _mean_existing(last_ood_rows),
            "prefix_sung": _mean_existing(prefix_ood_rows),
        },
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
                    prefix_orth=bool(cand.get("prefix_orth", False)),
                    lyric_hold=bool(cand.get("lyric_hold", False)),
                    last_delta_orth=bool(cand.get("last_delta_orth", False)),
                    steps=steps,
                    seed=seed,
                )
            )
        out[cell_name] = rows
    return out


def lyric_rank(table: dict[str, list[dict]]) -> list[dict]:
    """Want-box first, then lyric_recall@+1, cover, neu_hold, gender_move."""
    by: dict[str, dict[str, dict]] = {}
    for cell, rows in table.items():
        by[cell] = {r["name"]: r for r in rows}
    names = [c["name"] for c in LYRIC_RECIPES]
    required = [c for c in ("divergent", "close") if c in by]
    ranked: list[dict] = []
    for name in names:
        cells = [by[c][name] for c in required]
        grit = by["divergent"][name] if "divergent" in by else None
        gender = by["close"][name] if "close" in by else None
        grit_lyric = (
            grit is not None and float(grit["lyric_recall"]) >= LYRIC_RECALL_MIN
        )
        gender_free = (
            gender is not None and float(gender["gender_move"]) >= GENDER_MOVE_MIN
        )
        cover_ok = all(
            r["cover"] >= PLUS_COVER_MIN and r["neu_hold"] >= PLUS_NEU_HOLD_MIN
            for r in cells
        )
        ranked.append(
            {
                "name": name,
                "in_box": all(r["hit"] for r in cells),
                "want_box": bool(grit_lyric and gender_free and cover_ok),
                "split_want": bool(grit_lyric and gender_free),
                "hit_divergent": cells[0]["hit"] if "divergent" in by else None,
                "hit_close": cells[1]["hit"] if "close" in by else None,
                "grit_lyric_hit": grit_lyric,
                "gender_move_hit": gender_free,
                "lyric_recall": sum(r["lyric_recall"] for r in cells) / len(cells),
                "lyric_recall_grit": None if grit is None else grit["lyric_recall"],
                "lyric_recall_gender": None if gender is None else gender["lyric_recall"],
                "gender_move": None if gender is None else gender["gender_move"],
                "gender_move_grit": None if grit is None else grit["gender_move"],
                "cover": sum(r["cover"] for r in cells) / len(cells),
                "neu_hold": sum(r["neu_hold"] for r in cells) / len(cells),
                "gender_move": sum(r.get("gender_move", 0.0) for r in cells)
                / len(cells),
                "off_caption": sum(r["off_caption"] for r in cells) / len(cells),
                "lyric_recall_ref_plus": sum(r["lyric_recall_ref_plus"] for r in cells)
                / len(cells),
                "old_box": all(r["old_box"] for r in cells),
                "last_token_transplant": any(r["last_token_transplant"] for r in cells),
                "prefix_hold": cells[0]["prefix_hold"],
                "lyric_hold": cells[0].get("lyric_hold", False),
                "last_delta_orth": bool(cells[0].get("last_delta_orth", False)),
                "plus_neu": cells[0]["plus_neu"],
                "plus_only": cells[0]["plus_only"],
            }
        )
    ranked.sort(
        key=lambda r: (
            -int(bool(r["want_box"])),
            -float(r["lyric_recall"]),
            -float(r["cover"]),
            -float(r["neu_hold"]),
            -float(r["gender_move"]),
            str(r["name"]),
        )
    )
    for i, row in enumerate(ranked, start=1):
        row["rank"] = i
    return ranked


def lyric_verdict(table: dict[str, list[dict]]) -> dict:
    """Did we replicate the last-token transplant, and does prefix/lyric-hold split?"""
    by: dict[str, dict[str, dict]] = {}
    for cell, rows in table.items():
        by[cell] = {r["name"]: r for r in rows}
    required = [c for c in ("divergent", "close") if c in by]
    uni = [by[c]["faithful_plus_neu"] for c in required]
    prefix = [by[c]["faithful_plus_neu_prefix"] for c in required]
    lyric = [by[c]["faithful_plus_neu_lyric"] for c in required] if "faithful_plus_neu_lyric" in by[required[0]] else []
    plus = [by[c]["faithful_plus"] for c in required]
    orth = (
        [by[c]["faithful_plus_neu_orth"] for c in required]
        if all("faithful_plus_neu_orth" in by[c] for c in required)
        else []
    )
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
    grit = by.get("divergent", {})
    gender = by.get("close", {})
    uni_g = grit.get("faithful_plus_neu")
    uni_c = gender.get("faithful_plus_neu")
    pref_g = grit.get("faithful_plus_neu_prefix")
    pref_c = gender.get("faithful_plus_neu_prefix")
    lyr_g = grit.get("faithful_plus_neu_lyric")
    lyr_c = gender.get("faithful_plus_neu_lyric")
    uni_grit_lyric_miss = (
        "divergent" in by
        and float(by["divergent"]["faithful_plus_neu"]["lyric_recall"])
        < LYRIC_RECALL_MIN
    )
    uni_gender_hit = (
        "close" in by
        and float(by["close"]["faithful_plus_neu"]["gender_move"]) >= GENDER_MOVE_MIN
    )
    prefix_grit_lyric_hit = (
        "divergent" in by
        and float(by["divergent"]["faithful_plus_neu_prefix"]["lyric_recall"])
        >= LYRIC_RECALL_MIN
    )
    prefix_gender_miss = (
        "close" in by
        and float(by["close"]["faithful_plus_neu_prefix"]["gender_move"])
        < GENDER_MOVE_MIN
    )
    orth_beats_split = bool(orth) and all(r.get("want_box", False) for r in orth)
    return {
        "replicated_last_token_transplant": replicated,
        "uni_old_box": [r["old_box"] for r in uni],
        "uni_lyric_recall": [r["lyric_recall"] for r in uni],
        "uni_cover": [r["cover"] for r in uni],
        "uni_neu_hold": [r["neu_hold"] for r in uni],
        "uni_ref_plus": [r["lyric_recall_ref_plus"] for r in uni],
        "uni_gender_move": [r.get("gender_move") for r in uni],
        "prefix_lyric_recall": [r["lyric_recall"] for r in prefix],
        "prefix_cover": [r["cover"] for r in prefix],
        "prefix_neu_hold": [r["neu_hold"] for r in prefix],
        "prefix_gender_move": [r.get("gender_move") for r in prefix],
        "prefix_lifts_lyric_recall": prefix_lifts,
        "prefix_keeps_cover": prefix_keeps_cover,
        "prefix_hits_required": prefix_hits,
        "lyric_lyric_recall": [r["lyric_recall"] for r in lyric],
        "lyric_cover": [r["cover"] for r in lyric],
        "lyric_neu_hold": [r["neu_hold"] for r in lyric],
        "lyric_gender_move": [r.get("gender_move") for r in lyric],
        "uni_hits_gender_misses_grit": bool(
            uni_c
            and uni_g
            and uni_c.get("gender_move", 0) >= GENDER_MOVE_MIN
            and uni_g["lyric_recall"] < LYRIC_RECALL_MIN
        ),
        "prefix_hits_grit_misses_gender": bool(
            pref_c
            and pref_g
            and pref_g["lyric_recall"] >= LYRIC_RECALL_MIN
            and pref_c.get("gender_move", 1) < GENDER_MOVE_MIN
        ),
        "lyric_hits_both": bool(
            lyr_c
            and lyr_g
            and lyr_g["lyric_recall"] >= LYRIC_RECALL_MIN
            and lyr_c.get("gender_move", 0) >= GENDER_MOVE_MIN
            and lyr_g["cover"] >= PLUS_COVER_MIN
            and lyr_c["cover"] >= PLUS_COVER_MIN
            and lyr_g["neu_hold"] >= PLUS_NEU_HOLD_MIN
            and lyr_c["neu_hold"] >= PLUS_NEU_HOLD_MIN
        ),
        "orth_lyric_recall": [r["lyric_recall"] for r in orth],
        "orth_cover": [r["cover"] for r in orth],
        "orth_neu_hold": [r["neu_hold"] for r in orth],
        "orth_gender_move": [r.get("gender_move", 0.0) for r in orth],
        "uni_grit_lyric_miss": uni_grit_lyric_miss,
        "uni_gender_hit": uni_gender_hit,
        "prefix_grit_lyric_hit": prefix_grit_lyric_hit,
        "prefix_gender_miss": prefix_gender_miss,
        "orth_beats_split": orth_beats_split,
        "plus_lyric_recall": [r["lyric_recall"] for r in plus],
        "last_hidden_blind": {
            cell: last_hidden_cannot_see_transplant(LYRIC_CELLS[cell]())
            for cell in required
        },
        "existing_ood": existing_ood_verdict(table) if "divergent" in by and "close" in by else None,
    }
