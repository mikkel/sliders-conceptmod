"""Lyric survival vs prefix gender: one UNI recipe that has to do both.

``docs/lm-lyric-recall.md`` (#48) established the miss: UNI
(``faithful_plus_neu``) matches the ``<|audio_start|>`` last hidden to
raw ``h+`` and the LoRA still rewrites every prefix token, so a
grit-like plus caption shreds the yaml lyrics while sitting inside the
plus+neu want-box. The obvious clamp — ``faithful_plus_neu_prefix``,
hold the **whole** ``encode(neu)`` prefix — buys the lyrics back and
kills the other failure: on a close pair the concept *is* a prefix
caption span (``One female lead singer. A woman is singing…``) and the
listen path is the neutral caption plus the LoRA, so pinning the whole
prefix to ``encode(neu)`` pins the woman away.

This cell scores that trade on both pair types at once, on a sequence
fixture whose prefix has the two spans the live prompt has:

    [ caption span (Vocal Details) ][ lyric span ][ <|audio_start|> ]

The LoRA is a rank-free shared linear map ``W`` on each position's own
activation, not a per-position free vector. That is the whole
mechanism: the continue token reads the prefix, so the last-token
gradient lands on ``W`` through the lyric activations too, and the
lyric hiddens move by an amount that scales with ``‖h+ − h0‖``. Grit
(divergent, two tracks) shreds; gender (close, one song) does not.
Holding a span sets ``W`` to ~0 on **that span's activations**; the
caption span and the lyric span sit on different directions, so the
holds really are separable, and ``<|audio_start|>`` keeps a private
input channel of its own so a fully-pinned prefix can still reach
``h+`` at the last token (which is what live prefix-hold did: cover
0.93, gender dead).

Scored, per cell:

- divergent / grit-like: ``lyric_recall`` @ +1, ``cover``, ``neu_hold``
- close / gender-like: ``lyric_recall`` @ +1 and ``gender_move`` — the
  share of the + caption's readable Vocal Details gender margin that
  the student put back into the *neutral* prefix at +1. The caption
  span at +1 is *allowed* to differ from neu here; that is the point.

Not scored anywhere: ``exam_score``, ``leak_frac``, ``c+``, ``p%``,
pair-odd, collapse. This board is not folded into the compiled bipolar
scoreboard.

CPU only. No Hub, no GPU, no Music 3 weights. Does not change the live
default (``--lm_target v9`` / ``--pole_mode hidden``).
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
from analysis.slider2d.lyric_recall import LYRIC_RECALL_MIN, lyric_bag, sung_line
from conceptmod.textsliders.slider_targets import (
    lm_faithful_plus_neu,
    lm_plus_neu_loss,
    lm_plus_neu_span_loss,
    lm_project_last_delta_off_lyric,
)

# Same floor family as everything else on this scale: the pair-exam
# continuation gate. ``gender_move`` gets the same number so a "HIT" on
# one axis means the same thing as a "HIT" on the other.
GENDER_MOVE_MIN = EXAM_ROLL_OVERLAP

# Sequence shape. Two Vocal-Details caption positions, four lyric
# positions, one <|audio_start|>.
CAPTION_LEN = 2
LYRIC_LEN = 4
# How much of the continue token's hidden is re-read from the prefix.
ATTEND = 1.0
# The residual-stream component every position shares. Real LM hiddens
# at different positions are not orthogonal — a dominant common
# direction puts neighbouring-token cosines up around 0.8–0.9 — and
# that is exactly the channel one LoRA weight uses to rewrite every
# position at once. Without it a shared linear map could specialize
# per span for free, no last-token loss would ever touch the lyrics,
# and the fixture could not reproduce the shred that started this.
# 2.5 puts the caption-vs-lyric activation cosine at ~0.84.
COMMON_STREAM = 2.5
# Size of the <|audio_start|> token's own input channel — the direction
# no caption or lyric token has. Small: the continue token's hidden is
# mostly a read of the prompt, which is why holding the whole prefix
# costs last-token accuracy at all.
AUDIO_CHANNEL = 0.25
# Share of ``h+ − h0`` the + caption states in its Vocal Details span
# rather than at the continue token.
CAPTION_SHARE = 0.5
# Gender margin below this reads as "no gender word" — the neutral
# caption is ``One lead singer``, not a man.
GENDER_DEADBAND = 1e-3


def gender_cell(**kwargs) -> PairField:
    """gender-v4 with the moved Vocal Details attribute made readable.

    ``close_field`` puts all of a close pair's motion in ``delivery``,
    a zero column, so nothing can score whether the woman arrived. The
    yaml pair does name her (``A man is singing`` → ``A woman is
    singing``), so here that part of the pole difference sits on the
    attribute axis ``ĝ`` that ``male`` / ``female`` read. Negative
    because ``female`` reads ``−ĝ``: the + caption is the woman.
    ``shared`` is still solved from the logged gender-v4 pair cos.
    """
    base = {"unused": -0.55}
    base.update(kwargs)
    return close_field(**base)


LYRIC_GENDER_CELLS = {
    "divergent": divergent_field,
    "close": gender_cell,
}

CELL_IS = {
    "divergent": (
        "energy-v4 shape: two tracks, concept in caption words, "
        "‖h+ − h0‖ large — this is where UNI shreds lyrics"
    ),
    "close": (
        "gender-v4 shape: one song, the moved attribute is a prefix "
        "Vocal Details span — this is where prefix-hold kills the woman"
    ),
}

# Every row is UNI (raw h+ at +1, h0 at scale 0). They differ only in
# which prefix positions the hold covers, and at what weight.
LYRIC_GENDER_RECIPES: list[dict] = [
    {
        "name": "faithful_plus_neu",
        "hold": "none",
        "hold_weight": 0.0,
        "is": "baseline: last-hidden MSE only (#46)",
    },
    {
        "name": "faithful_plus_neu_prefix",
        "hold": "prefix",
        "hold_weight": 1.0,
        "is": "baseline: hold the whole encode(neu) prefix (#48)",
    },
    {
        "name": "faithful_plus_neu_prefix@0.25",
        "hold": "prefix",
        "hold_weight": 0.25,
        "is": "the same clamp, softer — a slide, not a split",
    },
    {
        "name": "faithful_plus_neu_orth",
        "hold": "orth",
        "hold_weight": 0.0,
        "is": "last-delta projected off the lyric span; no prefix term",
    },
    {
        "name": "concept_prefix_teacher",
        "hold": "concept",
        "hold_weight": 1.0,
        "is": "caption span → encode(pos) caption span, lyric span → encode(neu)",
        "live": False,
    },
    {
        "name": "faithful_plus_neu_lyric",
        "hold": "lyric",
        "hold_weight": 1.0,
        "is": "hold the lyric span only; caption span free",
    },
]

# Why ``concept_prefix_teacher`` is on the board but not in ``LM_RECIPES``:
# it needs a position-by-position correspondence between the + caption's
# span and the neutral caption's span. The fixture has one because both
# spans are ``CAPTION_LEN`` positions by construction. The live yaml does
# not — ``One female lead singer. A woman is singing…`` and ``One lead
# singer`` are different token counts, so there is no position i to MSE
# against position i. A pooled variant (mean over each caption span) is
# buildable, but it needs a second sequence teacher and a pooling choice,
# and this board says the extra teacher buys nothing the lyric hold does
# not already get.
FIXTURE_ONLY = frozenset(
    {c["name"] for c in LYRIC_GENDER_RECIPES if c.get("live") is False}
)

PICK = "faithful_plus_neu_lyric"
BASELINES = ("faithful_plus_neu", "faithful_plus_neu_prefix")


@dataclass
class SpanLoRA:
    """A shared linear map on each position's own activation.

    ``δ(σ, x) = (σ·W_odd + |σ|·W_even + W_zero) x``. One weight, applied
    everywhere — which is what a LoRA on attention is. It is *not* a
    free vector per position: whether the hold on one span is cheap
    depends on whether that span's activations are separable from the
    ones the loss still needs, and here they are (the caption span
    lives on ŝ, the lyric span on the row's l̂, the continue token has
    its own channel). That separability is the claim the board tests,
    not an assumption baked into the parameterization.
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
        """``[N, in_dim]`` activations to ``[N, out_dim]`` deltas."""
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
    """One row's frozen prompt: base hiddens plus the LoRA's inputs.

    ``base`` is ``encode(neu)``: the caption span, the lyric span, and
    the ``<|audio_start|>`` hidden, which is ``h0`` itself. ``acts`` is
    what each position feeds the LoRA: its own hidden, plus the shared
    residual-stream channel every position carries, plus — at the
    continue token only — a private channel no prefix token has.
    """

    base: torch.Tensor
    acts: torch.Tensor
    caption_mask: torch.Tensor
    lyric_mask: torch.Tensor
    ref_caption: torch.Tensor

    @property
    def prefix_len(self) -> int:
        return int(self.base.shape[0]) - 1

    def prefix_mask(self) -> torch.Tensor:
        mask = torch.ones(1, int(self.base.shape[0]), dtype=torch.long)
        mask[0, -1] = 0
        return mask


def build_sequence(field: PairField, row: int) -> Sequence:
    """Assemble one row the way ``_assemble`` does: caption, lyrics, audio_start."""
    pos, _neg, neu = field.poles(row)
    dim = int(field.dim)
    caption = float(field.base_sheet) * field.sheet_dir()
    lyric = float(field.lyric) * field.lyric_dir(row)
    base = torch.stack(
        [caption] * int(CAPTION_LEN) + [lyric] * int(LYRIC_LEN) + [neu]
    )
    # The LoRA sees each position's own hidden, the shared stream, and
    # — at the continue token — its own channel. Two extra input dims;
    # the readout never sees them.
    acts = torch.zeros(base.shape[0], dim + 2)
    acts[:, :dim] = base
    acts[:, dim] = float(COMMON_STREAM)
    acts[-1, dim + 1] = float(AUDIO_CHANNEL)
    caption_mask = torch.zeros(1, base.shape[0], dtype=torch.long)
    caption_mask[0, : int(CAPTION_LEN)] = 1
    lyric_mask = torch.zeros(1, base.shape[0], dtype=torch.long)
    lyric_mask[0, int(CAPTION_LEN) : int(CAPTION_LEN) + int(LYRIC_LEN)] = 1
    # + REF: the pos caption with the slider off. Its Vocal Details span
    # states CAPTION_SHARE of the move; the lyric span is the same line,
    # which is why + REF still sings it.
    ref_caption = caption + float(CAPTION_SHARE) * (pos - neu)
    return Sequence(base, acts, caption_mask, lyric_mask, ref_caption)


def encode_sequence(
    seq: Sequence, lora: SpanLoRA, scale: float
) -> tuple[torch.Tensor, torch.Tensor]:
    """Student ``encode(neu tokens, LoRA @ scale)`` → (prefix hiddens, last).

    The continue token's hidden takes its own delta plus ``ATTEND`` times
    the mean prefix delta: attention re-reads a rewritten prefix. That
    coupling is why last-token MSE alone puts mass on the lyric span.
    """
    delta = lora.apply(scale, seq.acts)
    hidden = seq.base + delta
    prefix = hidden[:-1]
    last = hidden[-1] + float(ATTEND) * delta[:-1].mean(0)
    return prefix, last


def lyric_hiddens(seq: Sequence, prefix: torch.Tensor) -> torch.Tensor:
    """Only the written-line positions, which is what a listener hears sung."""
    return prefix[int(CAPTION_LEN) : int(CAPTION_LEN) + int(LYRIC_LEN)]


def caption_hidden(prefix: torch.Tensor) -> torch.Tensor:
    """The Vocal Details position. One vector; every caption position is equal."""
    return prefix[0]


def lyric_recall_span(field: PairField, seq: Sequence, prefix: torch.Tensor, row: int) -> float:
    """Sheet ``lyric_mass`` on the sung line — the existing yaml lyric sheet."""
    bag = lyric_bag(field, row)
    seq_tokens = sung_line(field, lyric_hiddens(seq, prefix))
    if not seq_tokens:
        return 0.0
    return sum(1.0 for t in seq_tokens if t in bag) / float(len(seq_tokens))


def gender_margin(field: PairField, hidden: torch.Tensor) -> float:
    """``logit(female) − logit(male)`` under the frozen readout.

    The neutral Vocal Details span is ungendered, so this is 0 there;
    the + caption's span is the woman, so it is positive there. Reading
    the margin rather than a policy difference keeps the number linear
    in the attribute coordinate the caption actually moved.
    """
    head = field.readout()
    logits = head.logits(hidden)
    return float(logits[head.index("female")] - logits[head.index("male")])


def gender_word(field: PairField, hidden: torch.Tensor) -> str:
    """Which gender word the Vocal Details position reads, if either."""
    margin = gender_margin(field, hidden)
    if abs(margin) <= float(GENDER_DEADBAND):
        return "—"
    return "female" if margin > 0.0 else "male"


def gender_move(field: PairField, seq: Sequence, prefix: torch.Tensor) -> float | None:
    """Share of the + caption's Vocal Details gender the student restored.

    1.0 means the neutral prefix at +1 reads as female as the + caption
    does. ``None`` on a cell whose pair does not move a readable
    attribute at all (the divergent cell pins its singer).
    """
    ref = gender_margin(field, seq.ref_caption) - gender_margin(field, seq.base[0])
    if abs(ref) <= float(GENDER_DEADBAND):
        return None
    got = gender_margin(field, caption_hidden(prefix)) - gender_margin(field, seq.base[0])
    return max(0.0, min(1.0, got / ref))


def caption_move(seq: Sequence, prefix: torch.Tensor) -> float:
    """Cell-agnostic sibling of ``gender_move``: how far the caption span went.

    ``gender_move`` needs a readable attribute; this one is defined on
    every cell and says the same thing geometrically. Logged, not scored.
    """
    want = seq.ref_caption - seq.base[0]
    denom = float(want.norm())
    if denom <= 1e-8:
        return 0.0
    got = caption_hidden(prefix) - seq.base[0]
    return max(0.0, min(1.0, float(got @ want) / denom**2))


def hold_spec(
    seq: Sequence,
    prefix_pred: torch.Tensor,
    *,
    hold: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None:
    """(predicted, target, mask) for the held positions, in ``[1, T, H]`` shape.

    ``prefix`` holds every prefix position to ``encode(neu)``. ``lyric``
    holds the written line only. ``concept`` also teaches the caption
    span toward ``encode(pos)``'s caption span — the fixture can do that
    because both spans are the same length here; the live yaml's cannot.
    """
    if hold == "prefix":
        mask = seq.prefix_mask()
    elif hold in ("lyric", "concept"):
        mask = seq.lyric_mask if hold == "lyric" else seq.prefix_mask()
    else:
        return None
    target = seq.base.clone()
    target[-1] = 0.0
    if hold == "concept":
        target[: int(CAPTION_LEN)] = seq.ref_caption
    pad = torch.cat([prefix_pred, prefix_pred.new_zeros(1, prefix_pred.shape[1])])
    return pad.unsqueeze(0), target.unsqueeze(0), mask


def fit_lyric_gender(
    field: PairField,
    *,
    hold: str,
    hold_weight: float,
    steps: int = 400,
    lr: float = 0.08,
    seed: int = 0,
) -> SpanLoRA:
    """Fit one shared LoRA. Every variant is UNI; the hold is the variable."""
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
            if hold == "orth":
                tgt_last = neu + lm_project_last_delta_off_lyric(
                    tgt_last - neu, lyric_hiddens(seq, seq.base[:-1])
                )
            prefix_p, last_p = encode_sequence(seq, lora, 1.0)
            _prefix_0, last_0 = encode_sequence(seq, lora, 0.0)
            held = hold_spec(seq, prefix_p, hold=hold)
            if held is None:
                term = lm_plus_neu_loss(last_p, tgt_last, last_0, neu)
            else:
                pred_seq, tgt_seq, mask = held
                term = lm_plus_neu_span_loss(
                    last_p,
                    tgt_last,
                    last_0,
                    neu,
                    pred_seq,
                    tgt_seq,
                    mask,
                    span_weight=float(hold_weight),
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
        recall_plus.append(lyric_recall_span(field, seq, prefix_p, row))
        recall_zero.append(lyric_recall_span(field, seq, prefix_0, row))
        recall_ref.append(lyric_recall_span(field, seq, seq.base[:-1], row))
        move = gender_move(field, seq, prefix_p)
        if move is not None:
            move_rows.append(move)
        caption_rows.append(caption_move(seq, prefix_p))
        sings_lyric.append(
            " ".join(head.tokens[t] for t in sung_line(field, lyric_hiddens(seq, prefix_p)))
        )
        sings_lyric_ref.append(
            " ".join(
                head.tokens[t]
                for t in sung_line(field, lyric_hiddens(seq, seq.base[:-1]))
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
    if field.kind == "close":
        # Close pair: the two scored columns are lyric survival and
        # whether the woman arrived in the prefix. cover / neu_hold are
        # logged — the caption span is *supposed* to move here.
        hit = bool(lyric_hit and gender_hit)
    else:
        hit = bool(lyric_hit and cover >= PLUS_COVER_MIN and hold_score >= PLUS_NEU_HOLD_MIN)
    return {
        "name": name,
        "cell": field.kind,
        "hold": hold,
        "hold_weight": float(hold_weight),
        "live_flag": name not in FIXTURE_ONLY,
        "lyric_recall": recall,
        "lyric_recall_zero": mean(recall_zero),
        "lyric_recall_ref_plus": mean(recall_ref),
        "gender_move": move,
        "caption_move": mean(caption_rows),
        "cover": cover,
        "off_caption": off_caption,
        "neu_hold": hold_score,
        "overlap_pos": mean(overlap_rows),
        "lyric_hit": lyric_hit,
        "gender_hit": gender_hit,
        "hit": hit,
        "sings_lyric": " | ".join(sings_lyric),
        "sings_lyric_ref_plus": " | ".join(sings_lyric_ref),
        "sings_plus": " | ".join(sings_plus),
        "reads_vocal_neu": " | ".join(reads_neu),
        "reads_vocal_plus": " | ".join(reads_plus),
        "reads_vocal_ref_plus": " | ".join(reads_ref),
    }


def lyric_gender_table(*, steps: int = 400, seed: int = 0) -> dict[str, list[dict]]:
    """Score every UNI hold variant on the grit-like and gender-like cells."""
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
            for cand in LYRIC_GENDER_RECIPES
        ]
    return out


def lyric_gender_rank(table: dict[str, list[dict]]) -> list[dict]:
    """Both cells or nothing: a one-recipe win has to hit grit and gender."""
    by = {cell: {r["name"]: r for r in rows} for cell, rows in table.items()}
    required = [c for c in ("divergent", "close") if c in by]
    ranked: list[dict] = []
    for cand in LYRIC_GENDER_RECIPES:
        name = cand["name"]
        cells = [by[c][name] for c in required]
        div = by.get("divergent", {}).get(name)
        close = by.get("close", {}).get(name)
        ranked.append(
            {
                "name": name,
                "is": cand["is"],
                "live_flag": name not in FIXTURE_ONLY,
                "both": all(r["hit"] for r in cells),
                "hit_divergent": None if div is None else div["hit"],
                "hit_close": None if close is None else close["hit"],
                "grit_lyric_recall": None if div is None else div["lyric_recall"],
                "grit_cover": None if div is None else div["cover"],
                "grit_neu_hold": None if div is None else div["neu_hold"],
                "gender_lyric_recall": None if close is None else close["lyric_recall"],
                "gender_move": None if close is None else close["gender_move"],
                "gender_cover": None if close is None else close["cover"],
                "lyric_recall": sum(r["lyric_recall"] for r in cells) / len(cells),
            }
        )
    ranked.sort(
        key=lambda r: (
            0 if r["both"] else 1,
            -float(r["lyric_recall"]),
            -float(r["gender_move"] or 0.0),
            str(r["name"]),
        )
    )
    for i, row in enumerate(ranked, start=1):
        row["rank"] = i
    return ranked


def lyric_gender_verdict(table: dict[str, list[dict]]) -> dict:
    """Did one recipe take both cells, and did the two baselines split?"""
    by = {cell: {r["name"]: r for r in rows} for cell, rows in table.items()}
    grit, close = by["divergent"], by["close"]
    uni, prefix, pick = (
        "faithful_plus_neu",
        "faithful_plus_neu_prefix",
        PICK,
    )
    baselines_split = bool(
        not grit[uni]["lyric_hit"]
        and close[uni]["gender_hit"]
        and grit[prefix]["lyric_hit"]
        and not close[prefix]["gender_hit"]
    )
    winners = [row["name"] for row in lyric_gender_rank(table) if row["both"]]
    live_winners = [name for name in winners if name not in FIXTURE_ONLY]
    return {
        "pick": pick,
        "one_recipe_wins": bool(pick in winners),
        "winners": winners,
        "live_winners": live_winners,
        "fixture_only": sorted(FIXTURE_ONLY),
        "baselines_split_as_reported": baselines_split,
        "grit": {
            name: {
                "lyric_recall": grit[name]["lyric_recall"],
                "cover": grit[name]["cover"],
                "neu_hold": grit[name]["neu_hold"],
                "hit": grit[name]["hit"],
            }
            for name in grit
        },
        "gender": {
            name: {
                "lyric_recall": close[name]["lyric_recall"],
                "gender_move": close[name]["gender_move"],
                "cover": close[name]["cover"],
                "hit": close[name]["hit"],
            }
            for name in close
        },
        "gates": {
            "lyric_recall": LYRIC_RECALL_MIN,
            "gender_move": GENDER_MOVE_MIN,
            "cover": PLUS_COVER_MIN,
            "neu_hold": PLUS_NEU_HOLD_MIN,
            "off_caption_max": PLUS_OFF_MAX,
        },
    }


# Fixture knobs the verdict could be hiding behind, and the values the
# sweep re-runs the whole board at. Published, not hidden: a one-recipe
# win that only exists at one hand-picked constant is not a win.
SENSITIVITY_KNOBS: dict[str, tuple[float, ...]] = {
    "COMMON_STREAM": (1.5, 2.0, 2.5, 3.5, 5.0),
    "AUDIO_CHANNEL": (0.0, 0.25, 1.0, 2.0),
    "CAPTION_SHARE": (0.3, 0.5, 0.8),
    "ATTEND": (0.5, 1.0, 2.0),
    "CAPTION_LEN": (1, 2, 4),
    "LYRIC_LEN": (2, 4, 8),
}


def sensitivity(*, steps: int = 200, seed: int = 0) -> list[dict]:
    """Re-run the board once per knob value. One row per (knob, value).

    Each row says whether the two baselines still split the way the live
    runs did and whether the pick still takes both cells. A knob value
    where the split itself disappears is a setting where the live
    failure does not exist, so it cannot vote on the recipe.
    """
    module = __import__(__name__, fromlist=["_"])
    defaults = {name: getattr(module, name) for name in SENSITIVITY_KNOBS}
    out: list[dict] = []
    try:
        for knob, values in SENSITIVITY_KNOBS.items():
            for value in values:
                for name, default in defaults.items():
                    setattr(module, name, default)
                setattr(module, knob, value)
                table = lyric_gender_table(steps=steps, seed=seed)
                verdict = lyric_gender_verdict(table)
                grit = {r["name"]: r for r in table["divergent"]}
                close = {r["name"]: r for r in table["close"]}
                out.append(
                    {
                        "knob": knob,
                        "value": value,
                        "default": bool(value == defaults[knob]),
                        "baselines_split": verdict["baselines_split_as_reported"],
                        "pick_wins_both": verdict["one_recipe_wins"],
                        "winners": verdict["winners"],
                        "uni_grit_lyric_recall": grit["faithful_plus_neu"]["lyric_recall"],
                        "uni_gender_move": close["faithful_plus_neu"]["gender_move"],
                        "prefix_grit_lyric_recall": grit["faithful_plus_neu_prefix"][
                            "lyric_recall"
                        ],
                        "prefix_gender_move": close["faithful_plus_neu_prefix"][
                            "gender_move"
                        ],
                        "pick_grit_lyric_recall": grit[PICK]["lyric_recall"],
                        "pick_grit_cover": grit[PICK]["cover"],
                        "pick_gender_move": close[PICK]["gender_move"],
                    }
                )
    finally:
        for name, default in defaults.items():
            setattr(module, name, default)
    return out


def sensitivity_verdict(rows: list[dict]) -> dict:
    """Is the pick's win a property of the failure, or of one constant?"""
    split_rows = [r for r in rows if r["baselines_split"]]
    holds = [r for r in split_rows if r["pick_wins_both"]]
    return {
        "settings": len(rows),
        "settings_reproducing_the_live_split": len(split_rows),
        "pick_wins_both_where_split_reproduces": len(holds),
        "robust": bool(split_rows and len(holds) == len(split_rows)),
        "counterexamples": [
            {"knob": r["knob"], "value": r["value"], "winners": r["winners"]}
            for r in split_rows
            if not r["pick_wins_both"]
        ],
    }
