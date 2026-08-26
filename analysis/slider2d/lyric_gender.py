"""Joint UNI exam: grit lyric survival vs prefix gender_move.

Lifted from PR #52's joint cell and scored with the live ``--lm_target``
losses already on main. Default trainer is untouched
(``--lm_target v9`` / ``--pole_mode hidden``). No new live flags.

The existing per-recipe boards (``lm-lyric-hold``, ``lm-roles``,
``lm-lyric-orth``) each use a slightly different residual. This cell
puts every live UNI recipe on **one** fixture and **one** seed so the
expect leaderboard is a real ranking, not a paste of incomparable
pages.

    [ caption span (Vocal Details) ][ lyric span ][ <|audio_start|> ]

The LoRA is a shared linear map ``W`` on each position's own
activation. The continue token re-reads the prefix, so last-token
UNI rewrites lyric tokens on grit-like pairs (``punk punk``) and
whole-prefix hold pins Vocal Details on gender-like pairs (woman
never arrives). ``gender_move`` is the share of the + caption's
readable Vocal Details gender margin the student put back into the
**neutral** prefix at +1.

Not scored: ``exam_score``, ``leak_frac``, ``c+``, ``p%``. Not folded
into the compiled bipolar board.

CPU only. No Hub, no GPU, no Music 3 weights.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from analysis.slider2d.exam import (
    EXAM_ROLL_OVERLAP,
    PairField,
    close_field,
    divergent_field,
)
from analysis.slider2d.lyric_recall import LYRIC_RECALL_MIN, lyric_bag, sung_line
from analysis.slider2d.plus_exam import (
    PLUS_COVER_MIN,
    PLUS_OFF_MAX,
    _continue,
    _off_share,
    _token_share,
    blend_toward_mid,
    plus_bags,
    plus_cover,
)
from analysis.slider2d.plus_neu_exam import (
    PLUS_NEU_HOLD_MIN,
    drift_from_neu,
    neu_bags,
    neu_hold,
)
from conceptmod.textsliders.slider_targets import (
    lm_faithful_plus_neu,
    lm_plus_neu_loss,
    lm_plus_neu_lyric_loss,
    lm_plus_neu_orth_loss,
    lm_plus_neu_prefix_loss,
    lm_plus_neu_roles_loss,
)

GENDER_MOVE_MIN = EXAM_ROLL_OVERLAP

# Sequence shape. Two Vocal-Details caption positions, four lyric
# positions, one <|audio_start|>.
CAPTION_LEN = 2
LYRIC_LEN = 4
ATTEND = 1.0
# Residual-stream component every position shares. ~0.84 caption-vs-lyric
# activation cosine; without it a shared map can specialize per span
# for free and the grit shred disappears.
COMMON_STREAM = 2.5
# <|audio_start|> private channel. Small: the continue token mostly
# reads the prompt, which is why holding the whole prefix costs cover.
AUDIO_CHANNEL = 0.25
# Share of h+ − h0 the + caption states in Vocal Details.
CAPTION_SHARE = 0.5
# Gender margin below this reads as "no gender word".
GENDER_DEADBAND = 1e-3


def gender_cell(**kwargs) -> PairField:
    """gender-v4 with the moved Vocal Details attribute made readable.

    ``close_field`` puts all of a close pair's motion in ``delivery``,
    a zero column, so nothing can score whether the woman arrived.
    Negative ``unused`` because ``female`` reads ``−ĝ``.
    """
    base = {"unused": -0.55}
    base.update(kwargs)
    return close_field(**base)


LYRIC_GENDER_CELLS = {
    "divergent": divergent_field,
    "close": gender_cell,
}

# Live --lm_target names on main. prefix@0.25 is the already-wired
# prefix_weight ablation, not a new flag.
EXPECT_RECIPES: list[dict] = [
    {
        "name": "faithful_plus_neu",
        "hold": "none",
        "hold_weight": 0.0,
        "is": "last-hidden UNI only (#46)",
    },
    {
        "name": "faithful_plus_neu_prefix",
        "hold": "prefix",
        "hold_weight": 1.0,
        "is": "whole encode(neu) prefix hold (#48)",
    },
    {
        "name": "faithful_plus_neu_prefix@0.25",
        "hold": "prefix",
        "hold_weight": 0.25,
        "is": "same prefix hold, lower prefix_weight (already wired)",
    },
    {
        "name": "faithful_plus_neu_lyric",
        "hold": "lyric",
        "hold_weight": 1.0,
        "is": "yaml lyrics → encode(neu); caption span free (#50)",
    },
    {
        "name": "faithful_plus_neu_roles",
        "hold": "roles",
        "hold_weight": 1.0,
        "is": "lyrics → encode(neu), Vocal Details → encode(pos) (#49)",
    },
    {
        "name": "faithful_plus_neu_orth",
        "hold": "orth",
        "hold_weight": 1.0,
        "is": "live last-delta / lyric-span UNI (#51); last token still raw h+",
    },
]

REQUIRED_RECIPES = (
    "faithful_plus_neu",
    "faithful_plus_neu_prefix",
    "faithful_plus_neu_lyric",
    "faithful_plus_neu_roles",
    "faithful_plus_neu_orth",
)


@dataclass
class SpanLoRA:
    """A shared linear map on each position's own activation.

    ``δ(σ, x) = (σ·W_odd + |σ|·W_even + W_zero) x``. One weight,
    applied everywhere — which is what a LoRA on attention is.
    """

    odd: torch.Tensor
    even: torch.Tensor
    zero: torch.Tensor

    @classmethod
    def create(cls, out_dim: int, in_dim: int) -> "SpanLoRA":
        z = torch.zeros
        return cls(
            z(out_dim, in_dim, requires_grad=True),
            z(out_dim, in_dim, requires_grad=True),
            z(out_dim, in_dim, requires_grad=True),
        )

    def matrix(self, scale: float) -> torch.Tensor:
        s = float(scale)
        return s * self.odd + abs(s) * self.even + self.zero

    def apply(self, scale: float, acts: torch.Tensor) -> torch.Tensor:
        return acts @ self.matrix(scale).T

    def parameters(self) -> list[torch.Tensor]:
        return [self.odd, self.even, self.zero]

    def snapshot(self) -> "SpanLoRA":
        return SpanLoRA(
            self.odd.detach().clone(),
            self.even.detach().clone(),
            self.zero.detach().clone(),
        )


@dataclass(frozen=True)
class Sequence:
    """One row's frozen prompt: base hiddens plus the LoRA's inputs."""

    base: torch.Tensor
    acts: torch.Tensor
    caption_mask: torch.Tensor
    lyric_mask: torch.Tensor
    ref_caption: torch.Tensor

    @property
    def prefix_len(self) -> int:
        return int(self.base.shape[0]) - 1


def build_sequence(field: PairField, row: int) -> Sequence:
    """Assemble one row: caption, lyrics, audio_start."""
    pos, _neg, neu = field.poles(row)
    dim = int(field.dim)
    caption = float(field.base_sheet) * field.sheet_dir()
    lyric = float(field.lyric) * field.lyric_dir(row)
    base = torch.stack(
        [caption] * int(CAPTION_LEN) + [lyric] * int(LYRIC_LEN) + [neu]
    )
    acts = torch.zeros(base.shape[0], dim + 2)
    acts[:, :dim] = base
    acts[:, dim] = float(COMMON_STREAM)
    acts[-1, dim + 1] = float(AUDIO_CHANNEL)
    caption_mask = torch.zeros(1, base.shape[0], dtype=torch.long)
    caption_mask[0, : int(CAPTION_LEN)] = 1
    lyric_mask = torch.zeros(1, base.shape[0], dtype=torch.long)
    lyric_mask[0, int(CAPTION_LEN) : int(CAPTION_LEN) + int(LYRIC_LEN)] = 1
    ref_caption = caption + float(CAPTION_SHARE) * (pos - neu)
    return Sequence(base, acts, caption_mask, lyric_mask, ref_caption)


def encode_sequence(
    seq: Sequence, lora: SpanLoRA, scale: float
) -> tuple[torch.Tensor, torch.Tensor]:
    """Student encode(neu tokens, LoRA @ scale) → (prefix, last)."""
    delta = lora.apply(scale, seq.acts)
    hidden = seq.base + delta
    prefix = hidden[:-1]
    last = hidden[-1] + float(ATTEND) * delta[:-1].mean(0)
    return prefix, last


def lyric_hiddens(prefix: torch.Tensor) -> torch.Tensor:
    return prefix[int(CAPTION_LEN) : int(CAPTION_LEN) + int(LYRIC_LEN)]


def caption_hiddens(prefix: torch.Tensor) -> torch.Tensor:
    return prefix[: int(CAPTION_LEN)]


def caption_hidden(prefix: torch.Tensor) -> torch.Tensor:
    return prefix[0]


def caption_teacher(seq: Sequence) -> torch.Tensor:
    """encode(pos) Vocal Details, same length as the neu caption span."""
    return seq.ref_caption.unsqueeze(0).expand(int(CAPTION_LEN), -1).clone()


def lyric_recall_span(field: PairField, prefix: torch.Tensor, row: int) -> float:
    """Sheet lyric_mass on the sung line vs the yaml lyric sheet."""
    bag = lyric_bag(field, row)
    tokens = sung_line(field, lyric_hiddens(prefix))
    if not tokens:
        return 0.0
    return sum(1.0 for t in tokens if t in bag) / float(len(tokens))


def gender_margin(field: PairField, hidden: torch.Tensor) -> float:
    """logit(female) − logit(male) under the frozen readout."""
    head = field.readout()
    logits = head.logits(hidden)
    return float(logits[head.index("female")] - logits[head.index("male")])


def gender_word(field: PairField, hidden: torch.Tensor) -> str:
    margin = gender_margin(field, hidden)
    if abs(margin) <= float(GENDER_DEADBAND):
        return "—"
    return "female" if margin > 0.0 else "male"


def gender_move(field: PairField, seq: Sequence, prefix: torch.Tensor) -> float | None:
    """Share of the + caption's Vocal Details gender the student restored.

    1.0 means the neutral prefix at +1 reads as female as the + caption
    does. ``None`` on a cell whose pair does not move a readable
    attribute (the grit cell pins its singer).
    """
    ref = gender_margin(field, seq.ref_caption) - gender_margin(field, seq.base[0])
    if abs(ref) <= float(GENDER_DEADBAND):
        return None
    got = gender_margin(field, caption_hidden(prefix)) - gender_margin(
        field, seq.base[0]
    )
    return max(0.0, min(1.0, got / ref))


def caption_move(seq: Sequence, prefix: torch.Tensor) -> float:
    """How far the caption span moved toward encode(pos) Vocal Details."""
    want = seq.ref_caption - seq.base[0]
    denom = float(want.norm())
    if denom <= 1e-8:
        return 0.0
    got = caption_hidden(prefix) - seq.base[0]
    return max(0.0, min(1.0, float(got @ want) / denom**2))


def lyric_pin(seq: Sequence, prefix: torch.Tensor) -> float:
    """How tightly +1 lyrics stayed at encode(neu). Logged leftover, not a gate.

    Lyric-hold pins this to 1. Live orth can still move in-span. That is
    the leftover the expect tie-break reads when two HITs share a
    bottleneck.
    """
    neu = lyric_hiddens(seq.base[:-1]).reshape(-1)
    got = lyric_hiddens(prefix).reshape(-1)
    if float(neu.norm()) <= 1e-8 or float(got.norm()) <= 1e-8:
        return 0.0
    return float(
        torch.nn.functional.cosine_similarity(
            got.unsqueeze(0), neu.unsqueeze(0)
        )
        .clamp(min=0.0, max=1.0)
        .squeeze()
    )


def _live_term(
    *,
    hold: str,
    hold_weight: float,
    last_p: torch.Tensor,
    tgt_last: torch.Tensor,
    last_0: torch.Tensor,
    neu: torch.Tensor,
    prefix_p: torch.Tensor,
    seq: Sequence,
) -> torch.Tensor:
    """Dispatch to the live ``--lm_target`` losses on main. No new flags."""
    lyric_p = lyric_hiddens(prefix_p)
    lyric_neu = lyric_hiddens(seq.base[:-1])
    prefix_neu = seq.base[:-1]
    if hold == "none":
        return lm_plus_neu_loss(last_p, tgt_last, last_0, neu)
    if hold == "prefix":
        return lm_plus_neu_prefix_loss(
            last_p,
            tgt_last,
            last_0,
            neu,
            prefix_p,
            prefix_neu,
            prefix_weight=float(hold_weight),
        )
    if hold == "lyric":
        return lm_plus_neu_lyric_loss(
            last_p,
            tgt_last,
            last_0,
            neu,
            lyric_p,
            lyric_neu,
            lyric_weight=float(hold_weight),
        )
    if hold == "roles":
        return lm_plus_neu_roles_loss(
            last_p,
            tgt_last,
            last_0,
            neu,
            lyric_p,
            lyric_neu,
            caption_hiddens(prefix_p),
            caption_teacher(seq),
            lyric_weight=float(hold_weight),
            concept_weight=float(hold_weight),
        )
    if hold == "orth":
        return lm_plus_neu_orth_loss(
            last_p,
            tgt_last,
            last_0,
            neu,
            lyric_p,
            lyric_neu,
            lyric_weight=float(hold_weight),
            fail_closed=True,
        )
    raise ValueError(f"unknown hold {hold!r}")


def fit_lyric_gender(
    field: PairField,
    *,
    hold: str,
    hold_weight: float,
    steps: int = 400,
    lr: float = 0.08,
    seed: int = 0,
) -> SpanLoRA:
    """Fit one shared LoRA with a live UNI loss."""
    rows = int(field.rows)
    seqs = [build_sequence(field, row) for row in range(rows)]
    torch.manual_seed(int(seed))
    lora = SpanLoRA.create(int(field.dim), int(field.dim) + 2)
    opt = torch.optim.Adam(lora.parameters(), lr=float(lr))
    for _ in range(int(steps)):
        total = None
        for row, seq in enumerate(seqs):
            pos, neg, neu = field.poles(row)
            tgt_last = lm_faithful_plus_neu(pos, neg, neu)
            prefix_p, last_p = encode_sequence(seq, lora, 1.0)
            _prefix_0, last_0 = encode_sequence(seq, lora, 0.0)
            term = _live_term(
                hold=hold,
                hold_weight=hold_weight,
                last_p=last_p,
                tgt_last=tgt_last,
                last_0=last_0,
                neu=neu,
                prefix_p=prefix_p,
                seq=seq,
            )
            total = term if total is None else total + term
        loss = total / float(rows)
        opt.zero_grad()
        loss.backward()
        opt.step()
    return lora.snapshot()


def score_lyric_gender(
    name: str,
    field: PairField,
    *,
    hold: str,
    hold_weight: float,
    steps: int = 400,
    seed: int = 0,
) -> dict:
    """Fit, then score lyric survival, prefix gender, cover and neu_hold."""
    lora = fit_lyric_gender(
        field, hold=hold, hold_weight=hold_weight, steps=steps, seed=seed
    )
    bags = plus_bags(field)
    neu_bag = neu_bags(field)
    head = field.readout()
    rows = int(field.rows)
    cover_rows: list[float] = []
    off_rows: list[float] = []
    hold_rows: list[float] = []
    overlap_rows: list[float] = []
    recall_plus: list[float] = []
    recall_zero: list[float] = []
    recall_ref: list[float] = []
    move_rows: list[float] = []
    caption_rows: list[float] = []
    pin_rows: list[float] = []
    sings_lyric: list[str] = []
    sings_lyric_ref: list[str] = []
    sings_plus: list[str] = []
    reads_plus: list[str] = []
    reads_neu: list[str] = []
    reads_ref: list[str] = []
    for row in range(rows):
        pos, neg, neu = field.poles(row)
        mid = 0.5 * (pos + neg)
        seq = build_sequence(field, row)
        prefix_p, last_p = encode_sequence(seq, lora, 1.0)
        prefix_0, last_0 = encode_sequence(seq, lora, 0.0)
        plus_seqs = _continue(field, last_p, row=row, sign=1.0)
        zero_seqs = _continue(field, last_0, row=row, sign=0.0)
        overlap = _token_share(plus_seqs, bags["pos"])
        cover_rows.append(plus_cover(overlap, blend_toward_mid(last_p, pos, mid, neg)))
        off_rows.append(_off_share(plus_seqs, bags["plus_corpus"]))
        overlap_rows.append(overlap)
        hold_rows.append(
            neu_hold(
                _token_share(zero_seqs, neu_bag),
                drift_from_neu(last_0, neu, pos, mid),
            )
        )
        recall_plus.append(lyric_recall_span(field, prefix_p, row))
        recall_zero.append(lyric_recall_span(field, prefix_0, row))
        recall_ref.append(lyric_recall_span(field, seq.base[:-1], row))
        move = gender_move(field, seq, prefix_p)
        if move is not None:
            move_rows.append(move)
        caption_rows.append(caption_move(seq, prefix_p))
        pin_rows.append(lyric_pin(seq, prefix_p))
        sings_lyric.append(
            " ".join(head.tokens[t] for t in sung_line(field, lyric_hiddens(prefix_p)))
        )
        sings_lyric_ref.append(
            " ".join(
                head.tokens[t] for t in sung_line(field, lyric_hiddens(seq.base[:-1]))
            )
        )
        sings_plus.append(" ".join(head.tokens[t] for t in plus_seqs[0]))
        reads_plus.append(gender_word(field, caption_hidden(prefix_p)))
        reads_neu.append(gender_word(field, seq.base[0]))
        reads_ref.append(gender_word(field, seq.ref_caption))

    def mean(values: list[float]) -> float:
        return sum(values) / len(values) if values else 0.0

    cover = mean(cover_rows)
    off_caption = mean(off_rows)
    hold_score = mean(hold_rows)
    recall = mean(recall_plus)
    move = mean(move_rows) if move_rows else None
    lyric_hit = bool(recall >= LYRIC_RECALL_MIN)
    gender_hit = None if move is None else bool(move >= GENDER_MOVE_MIN)
    cover_hit = bool(cover >= PLUS_COVER_MIN)
    hold_hit = bool(hold_score >= PLUS_NEU_HOLD_MIN)
    if field.kind == "close":
        hit = bool(lyric_hit and gender_hit)
    else:
        hit = bool(lyric_hit and cover_hit and hold_hit)
    return {
        "name": name,
        "cell": field.kind,
        "hold": hold,
        "hold_weight": float(hold_weight),
        "lyric_recall": recall,
        "lyric_recall_zero": mean(recall_zero),
        "lyric_recall_ref_plus": mean(recall_ref),
        "gender_move": move,
        "caption_move": mean(caption_rows),
        "lyric_pin": mean(pin_rows),
        "cover": cover,
        "off_caption": off_caption,
        "neu_hold": hold_score,
        "overlap_pos": mean(overlap_rows),
        "lyric_hit": lyric_hit,
        "gender_hit": gender_hit,
        "cover_hit": cover_hit,
        "neu_hold_hit": hold_hit,
        "hit": hit,
        "sings_lyric": " | ".join(sings_lyric),
        "sings_lyric_ref_plus": " | ".join(sings_lyric_ref),
        "sings_plus": " | ".join(sings_plus),
        "reads_vocal_neu": " | ".join(reads_neu),
        "reads_vocal_plus": " | ".join(reads_plus),
        "reads_vocal_ref_plus": " | ".join(reads_ref),
    }


def lyric_gender_table(*, steps: int = 400, seed: int = 0) -> dict[str, list[dict]]:
    """Score every live UNI recipe on the grit-like and gender-like cells."""
    out: dict[str, list[dict]] = {}
    for cell_name, ctor in LYRIC_GENDER_CELLS.items():
        field = ctor(seed=seed)
        out[cell_name] = [
            score_lyric_gender(
                cand["name"],
                field,
                hold=str(cand["hold"]),
                hold_weight=float(cand["hold_weight"]),
                steps=steps,
                seed=seed,
            )
            for cand in EXPECT_RECIPES
        ]
    return out
