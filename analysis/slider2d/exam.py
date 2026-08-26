"""The pair-exam cell: what one scored token cannot see about a pole pair.

The sheet cell (#22) added a next-token readout and showed that a hidden
midpoint teacher walks off the caption sheet. It scores **one** token at
one position, and it has exactly one energy-like field: an *unused
attribute* sitting inside ``a`` on an otherwise same-song pair. Three live
Music 3 runs then landed in an order that field cannot produce:

===================  ==============================  ==========  ======
run                  recipe                          pair        listen
===================  ==============================  ==========  ======
``energy-lm-v16``    ``faithful_sub_e`` + KL         divergent   FAIL
``energy-lm-v18``    ``faithful`` + KL               divergent   PASS
``gender-lm-v16``    ``faithful`` + KL               close       FAIL
===================  ==============================  ==========  ======

Same loss on all three, and the same *teacher* on the last two. No live
log column orders them either: ``gender-lm-v16`` has the best ``c+``
(0.854), the smallest loss (0.0091) and the lowest ``p%`` of the three,
and it is the one whose lyrics came out garbled.

What separates them is the **pair**. This cell adds the two properties of
a pair the sheet field has no coordinate for, and the one scoring axis
that can read them.

``divergence`` — one song or two
    A close pair is two captions of the *same* song with one attribute
    moved (gender-v4: "A man is singing" → "A woman is singing"). A
    divergent pair is two captions of *different* songs (energy-v4:
    pop-punk BPM 168 slammed vs ambient lullaby BPM 52 near-silence).

    That is not a cosmetic difference, because it changes what ``c`` *is*.
    The sheet cell models the pole pair as ``h± = h0 ± a + c`` with one
    shared ``c``, and reads ``c`` as "what both pole captions say and the
    neutral does not — genre, BPM, mood, mix". True for a close pair. On a
    divergent pair the two poles say **contradictory** things about genre
    and BPM, so ``c = ½(h₊+h₋) − h0`` is not shared specificity: it is
    half of each song at once, a state no caption occupies. Modelling the
    two tracks as two separate features instead of two ends of one signed
    axis is the whole change, and it is what makes ``½(h₊+h₋)`` a third
    song rather than a coherent midpoint.

    It also changes what a declared ``leak_*`` pair means. energy-v4
    declares ``ê`` as ``"Pop-punk mix, BPM 168."`` / ``"Ambient lullaby
    mix, BPM 52."`` and calls it "leftover unused" — but the poles move
    genre and BPM too, by the yaml's own header ("genre, BPM, mood, mix
    and instrumentation all move with the axis"). So on energy-v4 ``ê`` is
    a restatement of most of ``a``, and subtracting ``ê_⊥`` deletes the
    slider instead of a leftover.

``visible share`` — how much of ``a`` the scored token can read
    Live ``semantic_kl`` is one next-token distribution at
    ``<|audio_start|>`` over the semantic band of ``lm_head``. Genre and
    BPM move that distribution hard. Which of two voices sings the same
    song barely moves it at all — that arrives in the vocal frames, later.
    So a close pair puts almost all of ``a`` in the readout's null space,
    where a KL loss has exactly **zero** gradient. The loss still falls to
    ~0. Nothing arrived. A small KL loss is a Goodhart metric in the same
    way a perfect pair-odd lock is.

``rollout`` — more than one token
    Both properties are invisible to a single-token score, so each hidden
    state is decoded for ``out_steps`` tokens through a frozen transition
    that (a) mixes the delivery block into the readable block, the way a
    residual stream carries content forward, and (b) feeds the emitted
    token's own content back so error compounds. The *real pole's* own
    rollout is the ground truth; a recipe is scored on how much of that
    continuation it reproduces, whether it stays inside the song's
    vocabulary, and whether consecutive tokens belong to the same song.

That last column is the one that names the ``energy-lm-v16`` failure. Its
target is ``mid ± â`` with the whole track part of ``â`` eaten by ``ê``,
so both ends sit at ``mid`` — pop-punk and ambient lullaby at once. The
one-token policy there is *bimodal*, split across the two poles' words, so
a single-token score against the union of both sheets looks survivable.
Sampled over eight steps it alternates between the two songs. "Random
words, like the midpoint got pulled."

CPU only. No Hub, no GPU, no Music 3 weights. Does not change the live
trainer default (still ``--lm_target v9`` / ``--pole_mode hidden``).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from analysis.slider2d.field import cosine
from analysis.slider2d.sheet import nucleus
from conceptmod.textsliders.slider_targets import (
    DUAL_BAND_WEIGHT,
    lm_axis_hold,
    lm_blind_projector,
    lm_dual_band_pole_loss,
    lm_even_axis_hold,
    lm_even_leftover_dir,
    lm_faithful_gate_odd_sub_even,
    lm_faithful_guard_e,
    lm_faithful_sub_e,
    lm_faithful_sub_e_if_unused,
    lm_faithful_sub_even_blend,
    lm_faithful_sub_even_blend_guard,
    lm_faithful_sub_even_blend_if_unused,
    lm_faithful_sub_even_e,
    lm_faithful_sub_even_e_guard,
    lm_faithful_sub_even_e_if_unused,
    lm_hidden_targets,
    lm_hold_dir,
    lm_next_token_logits,
    lm_pair_odd_sub_e,
    lm_readout_null_basis,
    lm_semantic_null_pole_loss,
    lm_semantic_pole_loss,
    lm_slider_loss,
    lm_unit,
    lm_unrolled_semantic_pole_loss,
)


# -- the live exam --------------------------------------------------------

# Three runs, 2026-08-25, after the #24 wire at f8d71a8. rank 8 / alpha 8 /
# lr 5e-4 / 800 steps / seed 7 / --no-early_stop / endreg 1.0 / hold 0,
# pair-odd early-stop gates deliberately left off. Last-step log columns as
# reported, and the listen verdict this cell has to reproduce.
LIVE_EXAM = {
    "energy-lm-v16": {
        "cell": "divergent",
        "teacher": "faithful_sub_e",
        "pole_mode": "semantic_kl",
        "prompts": "prompts-energy-v4.yaml",
        "c_plus": 0.690,
        "c_minus": 0.660,
        "collapse": -0.313,
        "pperc": 0.73,
        "nperc": 0.76,
        "loss": 0.0188,
        "listen": "fail",
        "heard": "random words, like the midpoint got pulled",
    },
    "energy-lm-v18": {
        "cell": "divergent",
        "teacher": "faithful",
        "pole_mode": "semantic_kl",
        "prompts": "prompts-energy-v4.yaml",
        "c_plus": 0.797,
        "c_minus": 0.719,
        "collapse": -0.354,
        "pperc": 0.60,
        "nperc": 0.70,
        "loss": 0.0167,
        "listen": "pass",
        "heard": "sounds pretty good; a genre/BPM ride, and those are the poles",
    },
    "gender-lm-v16": {
        "cell": "close",
        "teacher": "faithful",
        "pole_mode": "semantic_kl",
        "prompts": "prompts-gender-v4.yaml",
        "c_plus": 0.854,
        "c_minus": 0.668,
        "collapse": -0.458,
        "pperc": 0.523,
        "nperc": 0.777,
        "loss": 0.0091,
        "listen": "fail",
        "heard": "garbled lyrics; the KL loss is tiny and the hidden never arrived",
    },
}

# Which board row and cell each live run is the exam for. ``energy-lm-v18``
# and ``gender-lm-v16`` are the *same recipe* on different pairs — that is
# the point of the exam, and the reason the board has to be recipe × cell.
LIVE_ROW = {
    "energy-lm-v16": ("semantic_kl_sub_e", "divergent"),
    "energy-lm-v18": ("semantic_kl_poles", "divergent"),
    "gender-lm-v16": ("semantic_kl_poles", "close"),
}

# Logged ``cos(pos−neu, neg−neu)`` on the two yamls the exam ran on.
# energy-v4 is a range because its three genre rows differ.
LIVE_PAIR_COS = {
    "energy-v4": (-0.11, 0.14),
    "gender-v4": -0.08,
}


def energy_probe_cos() -> float:
    lo, hi = LIVE_PAIR_COS["energy-v4"]
    return 0.5 * (lo + hi)


# -- gates ---------------------------------------------------------------
#
# Every gate is a property of what the student *sings over a
# continuation*. ``c+``, ``c−`` and the ±1 collapse are logged and never
# scored — inherited from #22, and the exam sharpens it: of the three live
# runs the one with the best ``c+`` is one of the two that garbled.
#
# ``p%`` / ``n%`` and the pole loss are logged and never scored either.
# The live run with the smallest loss and the lowest ``p%`` also garbled.
EXAM_ROLL_OVERLAP = 0.85
EXAM_MATCH_KEPT = 0.75
EXAM_ROLL_OFF_MAX = 0.05
EXAM_COHERENCE = 0.90
EXAM_ROLL_SWING = 0.60
# Logged, not gated. The rollout commits after the first token, so an
# attribute that only tilts the first token's mass is averaged away here.
# ``leak_tok`` reads step 0 alone and reproduces the #22 sheet cell's
# number to three places, which is where leak is scored.
EXAM_LEAK_LOCK = 0.20
# Diagnosis flags, not gates.
EXAM_KL_SOLVED = 0.90
EXAM_INVISIBLE_ARRIVED = 0.50
EXAM_AXIS_EATEN = 0.30
# A verdict this close to a gate is not seed-robust, and the table says so
# rather than pretending otherwise. The three live exam rows clear every
# gate they decide by more than this, at every seed tested.
EXAM_NEAR_GATE = 0.05


AXIS_TOKENS = ("slam", "hush")
TRACK_TOKENS = ("punk", "lull")
UNUSED_TOKENS = ("male", "female")
OFF_SHEET_TOKENS = ("garble_hi", "garble_lo")

# Which of the two songs a word belongs to. Lyric and unused-attribute
# words belong to neither, so they never make a continuation incoherent.
# ``punk`` next to ``lull`` does: that is the audible signature of a state
# stuck at the average of two different tracks.
TOKEN_SIDE = {
    "slam": 1.0,
    "punk": 1.0,
    "garble_hi": 1.0,
    "hush": -1.0,
    "lull": -1.0,
    "garble_lo": -1.0,
}


@dataclass(frozen=True)
class Readout:
    """Frozen linear next-token head over a tiny vocabulary.

    Same device as ``sheet.Readout``: ``gain`` is an inverse temperature,
    calibrated once on the pole policies and then held fixed, so the sheet
    is never a function of what a student did. The one new property is that
    a whole hidden block — delivery — is a **zero column** here. Live that
    is the part of the 3584-wide state the semantic band of ``lm_head``
    does not read at this position. Hidden MSE pins it; semantic KL cannot
    see it, and gets no gradient on it at all.
    """

    tokens: tuple[str, ...]
    weight: torch.Tensor
    gain: float = 1.0

    def logits(self, hidden: torch.Tensor) -> torch.Tensor:
        return float(self.gain) * lm_next_token_logits(hidden, self.weight)

    def policy(self, hidden: torch.Tensor) -> torch.Tensor:
        return torch.softmax(self.logits(hidden), dim=-1)

    def index(self, token: str) -> int:
        return self.tokens.index(token)

    def null_dims(self) -> list[int]:
        seen = self.weight.abs().sum(dim=0)
        return [i for i in range(self.weight.shape[1]) if float(seen[i]) == 0.0]

    def content(self, token: int) -> torch.Tensor:
        """The emitted token's own content, for rollout feedback.

        Saying a word puts that word in the stream. Off-caption words are
        anti-loaded on the shared caption direction, so saying one
        de-specifies the state and the next step is further off — garble
        compounds, which is most of why a rollout sees what one token does
        not.
        """
        row = self.weight[int(token)]
        return row / row.norm().clamp_min(1e-8)


def shared_from_probe_cos(
    probe_cos: float, *, track: float, odd_norm: float
) -> float:
    """Size the shared term so the field prints the live trainer's cos.

    With ``pos−neu = track·p̂ + shared·ŝ + o`` and
    ``neg−neu = track·q̂ + shared·ŝ − o`` (``p̂ ⊥ q̂``, both ⊥ ŝ ⊥ o)::

        a = ½(pos−neg) = ½·track·(p̂−q̂) + o
        c = ½(pos+neg) − neu = ½·track·(p̂+q̂) + shared·ŝ
        cos(pos−neu, neg−neu) = (‖c‖²−‖a‖²)/(‖c‖²+‖a‖²)

    ``‖a‖² = ½track² + ‖o‖²`` and ``‖c‖² = ½track² + shared²``, so the
    track halves cancel and ``shared`` is pinned by the logged cos and
    ``‖o‖`` alone. That is how each cell here is calibrated to a real
    number: energy-v4 logs −0.11 … +0.14, gender-v4 logs −0.08.
    """
    cos = float(probe_cos)
    if not -1.0 < cos < 1.0:
        raise ValueError(f"probe cos must be in (-1, 1), got {probe_cos!r}")
    half = 0.5 * float(track) ** 2
    a_sq = half + float(odd_norm) ** 2
    # cos = (half + s² − a_sq) / (half + s² + a_sq)
    s_sq = (cos * (half + a_sq) - half + a_sq) / (1.0 - cos)
    if s_sq < 0.0:
        raise ValueError(
            f"probe cos {cos} is unreachable with track={track} odd={odd_norm}"
        )
    return s_sq**0.5


@dataclass(frozen=True)
class PairField:
    """A pole pair with a divergence, a visible share, and a continuation.

    Dim layout, all orthonormal::

        0                     û    the declared slider probe — the yaml's
                                   ``slider_positive`` / ``_negative``
                                   adjectives (loud ↔ quiet, belted ↔
                                   whispered).
        1                     p̂    the + pole's own track: pop-punk, BPM
                                   168, slammed mix.
        2                     q̂    the − pole's own track: ambient lullaby,
                                   BPM 52, sparse. A *separate feature*,
                                   not −p̂ — no caption says both.
        3                     ŝ    content both poles state and the neutral
                                   does not. Real shared specificity, which
                                   only exists when the poles are captions
                                   of the same song.
        4                     d̂    delivery / vocal detail. **Zero column in
                                   the readout**; the transition mixes it
                                   into û over the rollout.
        5                     ĝ    an attribute the captions leave unpinned.
                                   energy-v4 pins the singer with
                                   ``attributes``, so this is 0 there.
        6 .. 6+rows-1         l̂_r  the row's written lyric.

    Poles, per row with scale ``k``::

        h0 = lyric·l̂_r + base_sheet·ŝ
        h± = h0 + k·(track·(p̂ or q̂) + shared·ŝ ± (slider·û
                     + delivery·d̂ + unused·ĝ))

    so ``a = ½(h₊−h₋) = k·(½·track·(p̂−q̂) + slider·û + delivery·d̂
    + unused·ĝ)`` and ``c = ½(h₊+h₋) − h0 = k·(½·track·(p̂+q̂)
    + shared·ŝ)``. On a divergent pair (``track > 0``) that ``c`` holds
    half of each song's genre at once. On a close pair (``track = 0``) it is
    ``shared·ŝ``, exactly the sheet cell's reading.

    ``base_sheet`` is the neutral's *own* caption specificity. The v4
    neutrals are full Structured Captions — identical to ``target`` — not
    shrugs, so this is not zero, and how far ``t± = h0 ± a`` falls off the
    sheet depends on it.
    """

    kind: str = "divergent"
    rows: int = 3
    # Pole shape.
    slider: float = 0.90
    track: float = 2.00
    delivery: float = 0.70
    unused: float = 0.00
    shared: float | None = None
    probe_cos_target: float | None = None
    lyric: float = 1.00
    base_sheet: float = 1.20
    # ê, from the yaml's ``leak_*`` pair, in the same units.
    e_track: float = 2.00
    e_on_u: float = 0.30
    e_unused: float = 0.00
    row_scales: tuple[float, ...] = (1.0, 0.92, 1.08)
    gain: float = 2.5
    sheet_p: float = 0.90
    # Rollout: how much of the emitted token rides forward, how much of the
    # delivery block becomes readable per step, and how long we listen.
    carry: float = 0.60
    mix: float = 0.80
    out_steps: int = 8
    draws: int = 4
    seed: int = 0

    def __post_init__(self) -> None:
        if int(self.rows) < 1:
            raise ValueError(f"rows must be ≥ 1, got {self.rows!r}")
        if len(self.row_scales) < int(self.rows):
            raise ValueError("row_scales must cover every row")
        if float(self.slider) <= 0.0:
            raise ValueError("slider must be > 0 (it is the declared axis)")
        if int(self.out_steps) < 2:
            raise ValueError("out_steps must be ≥ 2 or this is the sheet cell")
        if self.shared is None and self.probe_cos_target is None:
            raise ValueError("give either shared or probe_cos_target")

    # -- geometry --------------------------------------------------------

    @property
    def dim(self) -> int:
        return 6 + int(self.rows)

    def _basis(self, index: int) -> torch.Tensor:
        out = torch.zeros(self.dim)
        out[index] = 1.0
        return out

    def short_u(self) -> torch.Tensor:
        return self._basis(0)

    def plus_track(self) -> torch.Tensor:
        return self._basis(1)

    def minus_track(self) -> torch.Tensor:
        return self._basis(2)

    def sheet_dir(self) -> torch.Tensor:
        return self._basis(3)

    def delivery_dir(self) -> torch.Tensor:
        return self._basis(4)

    def unused_dir(self) -> torch.Tensor:
        return self._basis(5)

    def lyric_dir(self, row: int) -> torch.Tensor:
        return self._basis(6 + int(row))

    def odd_shape(self) -> torch.Tensor:
        """The part of the pole content that flips sign: ``o``."""
        return (
            float(self.slider) * self.short_u()
            + float(self.delivery) * self.delivery_dir()
            + float(self.unused) * self.unused_dir()
        )

    def shared_size(self) -> float:
        if self.shared is not None:
            return float(self.shared)
        return shared_from_probe_cos(
            float(self.probe_cos_target),
            track=float(self.track),
            odd_norm=float(self.odd_shape().norm()),
        )

    def odd(self, row: int = 0) -> torch.Tensor:
        """``a = ½(h₊−h₋)``: half the track split, plus the flipping part."""
        scale = float(self.row_scales[int(row)])
        split = 0.5 * float(self.track) * (self.plus_track() - self.minus_track())
        return scale * (split + self.odd_shape())

    def common_vec(self, row: int = 0) -> torch.Tensor:
        """``c = ½(h₊+h₋) − h0``: half of *each* track, plus real shared."""
        scale = float(self.row_scales[int(row)])
        blend = 0.5 * float(self.track) * (self.plus_track() + self.minus_track())
        return scale * (blend + self.shared_size() * self.sheet_dir())

    def poles(self, row: int = 0) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        neu = float(self.lyric) * self.lyric_dir(row) + float(self.base_sheet) * self.sheet_dir()
        a = self.odd(row)
        c = self.common_vec(row)
        return neu + a + c, neu - a + c, neu

    def probe_cos(self, row: int = 0) -> float:
        """``cos(pos−neu, neg−neu)`` — the number the live trainer prints."""
        pos, neg, neu = self.poles(row)
        return cosine(pos - neu, neg - neu)

    def declared_e(self) -> torch.Tensor | None:
        """The yaml's ``leak_*`` pair as a direction, or ``None`` if absent.

        energy-v4's pair names the same genres and BPMs the poles move, so
        it points along the track split — which is most of ``a``.
        """
        vec = (
            0.5 * float(self.e_track) * (self.plus_track() - self.minus_track())
            + float(self.e_on_u) * self.short_u()
            + float(self.e_unused) * self.unused_dir()
        )
        return None if float(vec.norm()) <= 1e-8 else vec

    def declared_e_even(self) -> torch.Tensor | None:
        """Even leftover of the yaml ``leak_*`` pair: half of each leak caption.

        ``declared_e`` is the leak-pair *difference* (track split on
        energy-v4). This is the leak-pair *sum* — the blend of the two
        mix/BPM fragments. Unused leftover (``e_unused``) is odd and does
        not appear here. Near-zero when the pair declares no track ê.
        """
        vec = 0.5 * float(self.e_track) * (self.plus_track() + self.minus_track())
        return None if float(vec.norm()) <= 1e-8 else vec

    def declared_e_poles(self, neu: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        """Stand-in leak₊ / leak₋ embeddings whose odd matches ``declared_e``.

        Live the trainer already encodes both leak captions. This is the
        CPU fixture for that pair: leak₊ sits on the plus track, leak₋ on
        the minus track, with the same û / unused mix ``declared_e`` uses.
        """
        base = torch.zeros(self.dim) if neu is None else neu
        half = 0.5 * float(self.e_track)
        plus = (
            base
            + half * self.plus_track()
            + 0.5 * float(self.e_on_u) * self.short_u()
            + 0.5 * float(self.e_unused) * self.unused_dir()
        )
        minus = (
            base
            + half * self.minus_track()
            - 0.5 * float(self.e_on_u) * self.short_u()
            - 0.5 * float(self.e_unused) * self.unused_dir()
        )
        return plus, minus

    def has_unused(self) -> bool:
        """Is there an unpinned attribute for a leak column to measure?"""
        return abs(float(self.unused)) > 1e-8

    def visible_dirs(self) -> list[torch.Tensor]:
        """Directions some vocabulary row reads."""
        return [
            self.short_u(),
            self.plus_track(),
            self.minus_track(),
            self.sheet_dir(),
            self.unused_dir(),
        ]

    # -- the two coordinates the sheet field does not have ---------------

    def divergence(self) -> float:
        """Share of ``‖a‖`` that is one-song-versus-the-other track content.

        0 is a close pair. Near 1 is two different tracks, where a declared
        genre/BPM ``ê`` is the axis and subtracting it deletes the slider.
        """
        a = self.odd(0)
        track = (
            float(a @ self.plus_track()) ** 2 + float(a @ self.minus_track()) ** 2
        ) ** 0.5
        return track / float(a.norm().clamp_min(1e-8))

    def invisible_share(self) -> float:
        """Share of ``‖a‖`` in the readout's null space.

        Where a semantic-KL loss has exactly zero gradient. Live it is the
        part of the 3584-wide state the semantic band does not read at
        ``<|audio_start|>``; on a close pair that is the axis itself.
        """
        a = self.odd(0)
        return abs(float(a @ self.delivery_dir())) / float(a.norm().clamp_min(1e-8))

    def visible_share(self) -> float:
        """Share of ``‖a‖`` the scored next-token readout can see."""
        return (1.0 - self.invisible_share() ** 2) ** 0.5

    def readout(self) -> Readout:
        """Vocabulary and head. One structure, shared by every cell.

        ``punk`` and ``lull`` are the two songs' own words and read their
        own track feature. At ``mid`` both are lit at half strength and
        neither wins, which is what makes that policy bimodal — no
        hand-written "blend token" is needed.

        ``slam`` / ``hush`` are the axis adjectives on û. On a close pair
        the axis starts in the delivery block instead, and the transition
        brings it here a step later, which is why those pairs are nearly
        invisible at ``<|audio_start|>`` and perfectly audible once the
        vocal enters.

        ``garble_*`` is the sheet cell's device kept intact: further along
        û than any caption reaches (1.30 vs 1.00) and anti-loaded on ŝ. It
        wins where the caption specificity has been stripped.
        """
        tokens = (
            list(AXIS_TOKENS)
            + list(TRACK_TOKENS)
            + list(UNUSED_TOKENS)
            + list(OFF_SHEET_TOKENS)
        )
        tokens += [f"lyric{r}" for r in range(int(self.rows))]
        weight = torch.zeros(len(tokens), self.dim)
        weight[0, 0], weight[0, 3] = 1.00, 1.00  # slam
        weight[1, 0], weight[1, 3] = -1.00, 1.00  # hush
        weight[2, 1], weight[2, 3] = 1.00, 1.00  # punk
        weight[3, 2], weight[3, 3] = 1.00, 1.00  # lull
        weight[4, 5], weight[4, 3] = 1.00, 1.00  # male
        weight[5, 5], weight[5, 3] = -1.00, 1.00  # female
        weight[6, 0], weight[6, 3] = 1.30, -1.50  # garble_hi
        weight[7, 0], weight[7, 3] = -1.30, -1.50  # garble_lo
        for r in range(int(self.rows)):
            weight[8 + r, 6 + r] = 1.20
        return Readout(tuple(tokens), weight, gain=float(self.gain))

    def transition(self) -> torch.Tensor:
        """One decode step, frozen: delivery content becomes readable.

        ``A = I + mix·(û ⊗ d̂)``. The residual stream carries the delivery
        block forward and it reaches the tokens a few frames in — which is
        why a male-versus-female pair is nearly invisible at
        ``<|audio_start|>`` and perfectly audible by the time the vocal
        enters. It is also the only place a hidden-MSE fit can be rewarded
        for content a semantic-KL fit had no gradient on.
        """
        step = torch.eye(self.dim)
        step[0, 4] += float(self.mix)
        return step


# -- named fields --------------------------------------------------------


def divergent_field(**kwargs) -> PairField:
    """energy-v4: pop-punk BPM 168 slammed vs ambient lullaby BPM 52.

    Track content dominates ``a``; the singer is pinned by ``attributes``
    so there is no unpinned attribute to leak; and the declared ``ê``
    (``"Pop-punk mix, BPM 168."`` / ``"Ambient lullaby mix, BPM 52."``)
    restates exactly that track content. ``shared`` is sized so the field
    prints the logged energy-v4 pair cos.
    """
    base = {
        "kind": "divergent",
        "slider": 0.90,
        "track": 2.00,
        "delivery": 0.70,
        "unused": 0.00,
        "probe_cos_target": energy_probe_cos(),
        "e_track": 2.00,
        "e_on_u": 0.30,
        "e_unused": 0.00,
    }
    base.update(kwargs)
    return PairField(**base)


def close_field(**kwargs) -> PairField:
    """gender-v4: one song, "A man is singing" → "A woman is singing".

    No track motion — Global Metadata and Arrangement are identical across
    the pair, only Vocal Details moves — no declared ``ê``, and almost all
    of ``a`` is delivery, which the scored token cannot read.
    """
    base = {
        "kind": "close",
        "slider": 0.12,
        "track": 0.00,
        "delivery": 1.00,
        "unused": 0.00,
        "probe_cos_target": LIVE_PAIR_COS["gender-v4"],
        "e_track": 0.00,
        "e_on_u": 0.00,
        "e_unused": 0.00,
    }
    base.update(kwargs)
    return PairField(**base)


def unused_e_field(**kwargs) -> PairField:
    """The sheet cell's energy-like cell, re-expressed on this readout.

    One song, plus an attribute inside ``a`` that the captions leave
    unpinned and a declared ``ê`` that names it. This is the cell where
    ``faithful`` really does leak and where subtracting ``ê`` really is
    free. It stays on the board — it just is not the energy stand-in any
    more, because energy-v4 is not this.
    """
    base = {
        "kind": "unused_e",
        "slider": 1.00,
        "track": 0.00,
        "delivery": 0.35,
        "unused": 0.45,
        "probe_cos_target": energy_probe_cos(),
        "e_track": 0.00,
        "e_on_u": 0.00,
        "e_unused": 1.00,
    }
    base.update(kwargs)
    return PairField(**base)


CELLS = {
    "divergent": divergent_field,
    "close": close_field,
    "unused_e": unused_e_field,
}

CELL_IS = {
    "divergent": "two tracks (energy-v4), ê restates the pole difference",
    "close": "one song, one attribute moved (gender-v4), axis is delivery",
    "unused_e": "one song plus an unpinned attribute inside a (the #22 cell)",
}


# -- teachers ------------------------------------------------------------


TEACHERS = (
    "pair_odd",
    "pair_odd_sub_e",
    "faithful",
    "faithful_sub_e",
    "faithful_sub_e_if_unused",
    "faithful_guard_e",
    "faithful_sub_even_e",
    "faithful_sub_even_e_if_unused",
    "faithful_sub_even_e_guard",
    "faithful_sub_even_blend",
    "faithful_sub_even_blend_if_unused",
    "faithful_sub_even_blend_guard",
    "faithful_gate_odd_sub_even",
    "faithful_gate_odd_sub_even_blend",
)
# Live race modes plus fixture-only ``unrolled_kl`` (no live --pole_mode).
POLE_MODES = (
    "hidden",
    "semantic_kl",
    "semantic_kl_null",
    "hidden_kl",
    "unrolled_kl",
    "dual_band",
)


def hold_direction(field: PairField, leak_dir: torch.Tensor | None) -> torch.Tensor | None:
    """``ê_⊥ = ê − (ê·û)û`` — what the live trainer holds and subtracts."""
    if leak_dir is None:
        return None
    return lm_hold_dir(leak_dir, slider_dir=field.short_u(), mode="slider")


def _even_dir(field: PairField, leak_dir: torch.Tensor | None) -> torch.Tensor | None:
    """Leak-pair even leftover, from the fixture leak embeddings."""
    declared = field.declared_e_even()
    if declared is not None:
        return lm_hold_dir(declared, slider_dir=field.short_u(), mode="slider")
    if leak_dir is None:
        return None
    leak_plus, leak_minus = field.declared_e_poles(field.poles(0)[2])
    return lm_even_leftover_dir(
        leak_plus, leak_minus, field.poles(0)[2], slider_dir=field.short_u()
    )


def teacher_points(
    field: PairField,
    row: int,
    *,
    teacher: str = "pair_odd",
    leak_dir: torch.Tensor | None = None,
    common_beta: float = 0.0,
    even_scale: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """The two hidden states one recipe aims at, via the live functions."""
    pos, neg, neu = field.poles(row)
    mode = str(teacher).strip().lower()
    even_dir = _even_dir(field, leak_dir)
    if mode == "pair_odd":
        return lm_hidden_targets(
            pos, neg, neu, target_mode="symmetric", common_beta=float(common_beta)
        )
    if mode == "faithful":
        return pos, neg
    if mode == "faithful_sub_e_if_unused":
        return lm_faithful_sub_e_if_unused(
            pos, neg, neu, leak_dir, slider_dir=field.short_u()
        )
    if mode == "faithful_sub_even_e_if_unused":
        if leak_dir is None:
            return pos, neg
        return lm_faithful_sub_even_e_if_unused(
            pos,
            neg,
            neu,
            leak_dir,
            slider_dir=field.short_u(),
            scale=float(even_scale),
        )
    if mode == "faithful_sub_even_blend":
        return lm_faithful_sub_even_blend(
            pos, neg, neu, even_dir, scale=float(even_scale)
        )
    if mode == "faithful_sub_even_blend_if_unused":
        return lm_faithful_sub_even_blend_if_unused(
            pos,
            neg,
            neu,
            leak_dir,
            even_dir,
            slider_dir=field.short_u(),
            scale=float(even_scale),
        )
    if mode == "faithful_sub_even_blend_guard":
        return lm_faithful_sub_even_blend_guard(
            pos, neg, neu, even_dir, scale=float(even_scale)
        )
    if mode == "faithful_gate_odd_sub_even":
        return lm_faithful_gate_odd_sub_even(
            pos,
            neg,
            neu,
            leak_dir,
            slider_dir=field.short_u(),
            scale=float(even_scale),
        )
    if mode == "faithful_gate_odd_sub_even_blend":
        return lm_faithful_gate_odd_sub_even(
            pos,
            neg,
            neu,
            leak_dir,
            slider_dir=field.short_u(),
            even_dir=even_dir,
            scale=float(even_scale),
        )
    if mode in (
        "faithful_guard_e",
        "faithful_sub_even_e",
        "faithful_sub_even_e_guard",
        "faithful_gate_odd_sub_even",
        "faithful_gate_odd_sub_even_blend",
    ) and leak_dir is None:
        return pos, neg
    if leak_dir is None and mode not in (
        "faithful_sub_even_blend",
        "faithful_sub_even_blend_if_unused",
        "faithful_sub_even_blend_guard",
    ):
        raise ValueError(f"{mode} needs a declared ê")
    if mode == "pair_odd_sub_e":
        return lm_pair_odd_sub_e(pos, neg, neu, leak_dir, slider_dir=field.short_u())
    if mode == "faithful_sub_e":
        return lm_faithful_sub_e(pos, neg, neu, leak_dir, slider_dir=field.short_u())
    if mode == "faithful_guard_e":
        return lm_faithful_guard_e(pos, neg, neu, leak_dir, slider_dir=field.short_u())
    if mode == "faithful_sub_even_e":
        return lm_faithful_sub_even_e(
            pos,
            neg,
            neu,
            leak_dir,
            slider_dir=field.short_u(),
            scale=float(even_scale),
        )
    if mode == "faithful_sub_even_e_guard":
        return lm_faithful_sub_even_e_guard(
            pos,
            neg,
            neu,
            leak_dir,
            slider_dir=field.short_u(),
            scale=float(even_scale),
        )
    raise ValueError(f"teacher must be one of {TEACHERS}, got {teacher!r}")


def target_geometry(
    field: PairField,
    *,
    teacher: str,
    leak_dir: torch.Tensor | None = None,
    common_beta: float = 0.0,
    even_scale: float = 1.0,
) -> dict:
    """Where the target point sits. No optimizer in the loop.

    ``axis_eaten`` is the share of ``a`` the recipe deletes and
    ``visible_axis_eaten`` the share of the part the scored token can read.
    On a divergent pair with a genre/BPM ``ê`` both are large, and the
    target ends up nearer ``mid`` than the pole it claims to be —
    ``blend_teacher``.
    """
    row = 0
    pos, neg, _neu = field.poles(row)
    mid = 0.5 * (pos + neg)
    a = field.odd(row)
    t_plus, t_minus = teacher_points(
        field,
        row,
        teacher=teacher,
        leak_dir=leak_dir,
        common_beta=common_beta,
        even_scale=even_scale,
    )
    # What is left of the slider axis: the target pair's own odd part.
    kept = 0.5 * (t_plus - t_minus)
    a_norm = float(a.norm().clamp_min(1e-8))
    vis = field.visible_dirs()
    a_vis = sum(float(a @ d) ** 2 for d in vis) ** 0.5
    kept_vis = sum(float(kept @ d) ** 2 for d in vis) ** 0.5
    to_pole = float((t_plus - pos).norm()) / a_norm
    to_mid = float((t_plus - mid).norm()) / a_norm
    return {
        "axis_eaten": max(0.0, 1.0 - float(kept.norm()) / a_norm),
        "visible_axis_eaten": max(0.0, 1.0 - kept_vis / (a_vis + 1e-8)),
        "off_caption": to_pole,
        "to_mid": to_mid,
        "blend_teacher": bool(to_mid < to_pole),
        "is_caption": str(teacher).strip().lower() == "faithful" or float(common_beta) == 1.0,
    }


def teacher_geometry_table(cell: str) -> list[dict]:
    """Every live target point on one cell, with no optimizer at all."""
    field = CELLS[cell]()
    e = field.declared_e()
    out = []
    for name, teacher in (
        ("caption", "faithful"),
        ("pair_odd", "pair_odd"),
        ("pair_odd_sub_e", "pair_odd_sub_e"),
        ("faithful_sub_e", "faithful_sub_e"),
    ):
        if teacher.endswith("sub_e") and e is None:
            continue
        geom = target_geometry(field, teacher=teacher, leak_dir=e)
        geom.update({"name": name, "cell": cell, "teacher": teacher})
        out.append(geom)
    return out


# -- the rollout ---------------------------------------------------------


def rollout(
    field: PairField,
    hidden: torch.Tensor,
    readout: Readout,
    *,
    row: int,
    sign: float,
    draw: int = 0,
) -> tuple[list[int], list[torch.Tensor]]:
    """Decode ``out_steps`` tokens from one hidden state.

    Nucleus-truncated sampling at ``sheet_p`` — the live top-p decode —
    with a generator seeded per ``(field, row, sign, draw)`` so the teacher
    and every student see the same randomness on the same draw.

    Returns the tokens and the policy at each step. The leak column reads
    the policies, so leak is measured in probability mass and stays
    comparable with the #22 sheet cell's single-token swing.
    """
    gen = torch.Generator().manual_seed(
        int(field.seed) * 9973
        + int(row) * 101
        + int(draw) * 1009
        + (7 if sign >= 0 else 13)
    )
    step = field.transition()
    state = hidden
    said: list[int] = []
    seen: list[torch.Tensor] = []
    for k in range(int(field.out_steps)):
        if k > 0:
            state = step @ state + float(field.carry) * readout.content(said[-1])
        policy = readout.policy(state)
        seen.append(policy)
        keep = sorted(nucleus(policy, float(field.sheet_p)))
        weights = torch.tensor([float(policy[i]) for i in keep])
        pick = int(torch.multinomial(weights / weights.sum(), 1, generator=gen))
        said.append(int(keep[pick]))
    return said, seen


def rollouts(
    field: PairField,
    hidden: torch.Tensor,
    readout: Readout,
    *,
    row: int,
    sign: float,
) -> tuple[list[list[int]], list[list[torch.Tensor]]]:
    """``draws`` independent continuations from one hidden state.

    A pole state's policy is peaked and every draw says the same thing. A
    state stuck at the average of two tracks has a bimodal policy, and the
    draws disagree — which is the audible failure and the reason one
    greedy token cannot see it.
    """
    pairs = [
        rollout(field, hidden, readout, row=row, sign=sign, draw=d)
        for d in range(int(field.draws))
    ]
    return [p[0] for p in pairs], [p[1] for p in pairs]


def _side(readout: Readout, token: int) -> float:
    return TOKEN_SIDE.get(readout.tokens[int(token)], 0.0)


def rollout_report(
    field: PairField,
    plus: list[torch.Tensor],
    minus: list[torch.Tensor],
    *,
    readout: Readout,
    teacher_plus: list[list[list[int]]] | None = None,
    teacher_minus: list[list[list[int]]] | None = None,
    corpus: frozenset[int] | None = None,
) -> dict:
    """Score per-row ±1 states over their continuations.

    Works on target points as well as on fits, which is how this cell can
    say the ``faithful_sub_e`` target sings the midpoint before any
    optimizer has run.

    ``roll_overlap`` is the scored column: the share of what the student
    sings that the real pole's own continuations also sing. Bag membership,
    not position, because live decoding is sampled and a caption's own
    draws are not identical to each other either. ``roll_match`` is the
    strict position-wise version, logged.
    """
    match: list[float] = []
    overlap: list[float] = []
    off: list[float] = []
    coherent: list[float] = []
    lead: dict[float, list[float]] = {1.0: [], -1.0: []}
    side_mass: dict[float, list[float]] = {1.0: [], -1.0: []}
    unused_mass: dict[float, list[float]] = {1.0: [], -1.0: []}
    first_side: dict[float, list[float]] = {1.0: [], -1.0: []}
    first_unused: dict[float, list[float]] = {1.0: [], -1.0: []}
    said: list[str] = []
    male, female = (readout.index(t) for t in UNUSED_TOKENS)
    sides = torch.tensor(
        [TOKEN_SIDE.get(tok, 0.0) for tok in readout.tokens], dtype=torch.float32
    )
    for row, (h_plus, h_minus) in enumerate(zip(plus, minus)):
        for sign, hidden in ((1.0, h_plus), (-1.0, h_minus)):
            draws, policies = rollouts(field, hidden, readout, row=row, sign=sign)
            said.append(" ".join(readout.tokens[t] for t in draws[0]))
            refs = None
            if teacher_plus is not None and teacher_minus is not None:
                refs = (teacher_plus if sign > 0 else teacher_minus)[row]
            for seen in policies:
                stack = torch.stack(seen)
                side_mass[sign].append(float((stack @ sides).mean()))
                unused_mass[sign].append(
                    float((stack[:, male] - stack[:, female]).mean())
                )
                first_side[sign].append(float(stack[0] @ sides))
                first_unused[sign].append(float(stack[0, male] - stack[0, female]))
            for draw, seq in enumerate(draws):
                if refs is not None:
                    ref = refs[draw % len(refs)]
                    match.append(
                        sum(1.0 for x, y in zip(seq, ref) if x == y) / float(len(ref))
                    )
                    bag = frozenset().union(*(frozenset(r) for r in refs))
                    overlap.append(sum(1.0 for x in seq if x in bag) / float(len(seq)))
                if corpus is not None:
                    off.append(sum(1.0 for x in seq if x not in corpus) / float(len(seq)))
                said_sides = [_side(readout, t) for t in seq]
                pairs = list(zip(said_sides, said_sides[1:]))
                if pairs:
                    coherent.append(
                        sum(1.0 for x, y in pairs if x * y >= 0.0) / float(len(pairs))
                    )
                lead[sign].append(sum(said_sides) / float(len(said_sides)))

    def mean(vals: list[float]) -> float | None:
        return sum(vals) / len(vals) if vals else None

    def odd_part(bag: dict[float, list[float]]) -> float:
        return 0.5 * ((mean(bag[1.0]) or 0.0) - (mean(bag[-1.0]) or 0.0))

    return {
        "roll_match": mean(match),
        "roll_overlap": mean(overlap),
        "roll_off_corpus": mean(off),
        "roll_coherence": mean(coherent),
        "roll_swing": odd_part(lead),
        "roll_swing_mass": odd_part(side_mass),
        "roll_unused_swing": odd_part(unused_mass),
        "first_swing_mass": odd_part(first_side),
        "first_unused_swing": odd_part(first_unused),
        "sings": " | ".join(said),
    }


def teacher_rollouts(
    field: PairField, readout: Readout
) -> tuple[list[list[list[int]]], list[list[list[int]]], frozenset[int]]:
    """Ground truth: what the real pole captions sing, and the song's words.

    The corpus is the union over rows, both signs and every draw, plus
    every written lyric, so a token is only off-corpus if no caption of
    this song sings it anywhere. That keeps the off-corpus column from
    turning on sampling ties between a pole's own words.
    """
    plus, minus = [], []
    words: set[int] = set()
    for row in range(int(field.rows)):
        pos, neg, _neu = field.poles(row)
        up, _up_p = rollouts(field, pos, readout, row=row, sign=1.0)
        down, _down_p = rollouts(field, neg, readout, row=row, sign=-1.0)
        plus.append(up)
        minus.append(down)
        for seq in up + down:
            words |= set(seq)
        words.add(readout.index(f"lyric{row}"))
    return plus, minus, frozenset(words)


def teacher_self_match(
    plus: list[list[list[int]]], minus: list[list[list[int]]]
) -> float:
    """How much a real pole's own continuations agree with each other.

    The ceiling for ``roll_match``. It is not 1: a caption's policy is
    peaked but not a point mass, and on a divergent pair the + pole can
    reasonably start with its genre word or its intensity word. Dividing
    by it makes position-wise agreement comparable across cells — "does
    the student agree with the pole as much as the pole agrees with
    itself" — instead of penalising the multi-modal pair for being
    multi-modal.
    """
    scores: list[float] = []
    for refs in plus + minus:
        for i in range(len(refs)):
            for j in range(i + 1, len(refs)):
                scores.append(
                    sum(1.0 for x, y in zip(refs[i], refs[j]) if x == y)
                    / float(len(refs[i]))
                )
    return sum(scores) / len(scores) if scores else 1.0


# -- students ------------------------------------------------------------


@dataclass
class SharedResidual:
    """One residual serving every row: ``δ(σ) = σ·w_odd + |σ|·w_even``.

    ``train.Residual``'s LM student, the same one the sheet cell fits. A
    live LoRA is one set of weights added at the prompt-last position of
    every prompt row, so the residual is shared and only the neutral it is
    added to changes.
    """

    w: torch.Tensor
    w_even: torch.Tensor

    @classmethod
    def create(cls, field: PairField) -> "SharedResidual":
        return cls(
            torch.zeros(field.dim, requires_grad=True),
            torch.zeros(field.dim, requires_grad=True),
        )

    def delta(self, scale: float) -> torch.Tensor:
        return float(scale) * self.w + abs(float(scale)) * self.w_even

    def parameters(self) -> list[torch.Tensor]:
        return [self.w, self.w_even]

    def snapshot(self) -> "SharedResidual":
        return SharedResidual(self.w.detach().clone(), self.w_even.detach().clone())


def fit_exam(
    field: PairField,
    *,
    pole_mode: str = "hidden",
    teacher: str = "pair_odd",
    leak_dir: torch.Tensor | None = None,
    hold_weight: float = 0.0,
    common_beta: float = 0.0,
    unroll_steps: int = 1,
    blind_weight: float = DUAL_BAND_WEIGHT,
    blind_cut: float = 0.0,
    even_scale: float = 1.0,
    even_hold: bool = False,
    steps: int = 400,
    lr: float = 0.08,
    seed: int = 0,
) -> tuple[SharedResidual, float, float]:
    """Fit one shared residual with the live pole loss of ``pole_mode``.

    Returns the residual, the pole loss at ``δ = 0`` — the head room the
    loss has to work with — and the pole loss at the end.
    """
    mode = str(pole_mode).strip().lower()
    if mode not in POLE_MODES:
        raise ValueError(f"pole_mode must be one of {POLE_MODES}, got {pole_mode!r}")
    head = field.readout()
    null_basis = lm_readout_null_basis(head.weight) if mode == "semantic_kl_null" else None
    blind = (
        lm_blind_projector(head.weight, cut=float(blind_cut))
        if mode == "dual_band"
        else None
    )
    held = hold_direction(field, leak_dir)
    even_held = None
    if even_hold:
        even_held = _even_dir(field, leak_dir)
        if even_held is None:
            even_held = held
    lam = float(hold_weight) if (held is not None or even_held is not None) else 0.0
    targets = [
        teacher_points(
            field,
            row,
            teacher=teacher,
            leak_dir=leak_dir,
            common_beta=common_beta,
            even_scale=even_scale,
        )
        for row in range(int(field.rows))
    ]
    neutrals = [field.poles(row)[2] for row in range(int(field.rows))]

    torch.manual_seed(int(seed))
    residual = SharedResidual.create(field)
    opt = torch.optim.Adam(residual.parameters(), lr=float(lr))

    def step_loss() -> torch.Tensor:
        total = None
        for (t_plus, t_minus), neu in zip(targets, neutrals):
            pred_plus = neu + residual.delta(1.0)
            pred_minus = neu + residual.delta(-1.0)
            hold = None
            if even_held is not None and lam > 0.0:
                hold = lm_even_axis_hold(pred_plus, pred_minus, neu, even_held)
            elif held is not None and lam > 0.0:
                hold = lm_axis_hold(pred_plus, pred_minus, neu, held)
            hold_w = lam if hold is not None else 0.0
            if mode in ("hidden", "hidden_kl"):
                term = lm_slider_loss(
                    pred_plus,
                    pred_minus,
                    t_plus,
                    t_minus,
                    hold=hold,
                    hold_weight=hold_w,
                )
                if mode == "hidden_kl":
                    term = term + 1e-3 * lm_semantic_pole_loss(
                        head.logits(pred_plus),
                        head.logits(pred_minus),
                        head.logits(t_plus),
                        head.logits(t_minus),
                    )
            elif mode == "semantic_kl_null":
                term = lm_semantic_null_pole_loss(
                    head.logits(pred_plus),
                    head.logits(pred_minus),
                    head.logits(t_plus),
                    head.logits(t_minus),
                    pred_plus,
                    pred_minus,
                    t_plus,
                    t_minus,
                    head.weight,
                    null_basis=null_basis,
                    hold=hold,
                    hold_weight=hold_w,
                )
            elif mode == "unrolled_kl":
                scaled = head.weight * float(head.gain)
                term = lm_unrolled_semantic_pole_loss(
                    pred_plus,
                    pred_minus,
                    t_plus,
                    t_minus,
                    scaled,
                    field.transition(),
                    unroll_steps=unroll_steps,
                    hold=hold,
                    hold_weight=hold_w,
                )
            elif mode == "dual_band":
                term = lm_dual_band_pole_loss(
                    pred_plus,
                    pred_minus,
                    t_plus,
                    t_minus,
                    pred_plus_logits=head.logits(pred_plus),
                    pred_minus_logits=head.logits(pred_minus),
                    tgt_plus_logits=head.logits(t_plus),
                    tgt_minus_logits=head.logits(t_minus),
                    blind_projector=blind,
                    blind_weight=float(blind_weight),
                    hold=hold,
                    hold_weight=hold_w,
                )
            else:
                term = lm_semantic_pole_loss(
                    head.logits(pred_plus),
                    head.logits(pred_minus),
                    head.logits(t_plus),
                    head.logits(t_minus),
                    hold=hold,
                    hold_weight=hold_w,
                )
            total = term if total is None else total + term
        return total / float(len(targets))

    head_room = float(step_loss().detach())
    loss = torch.tensor(0.0)
    for _ in range(int(steps)):
        loss = step_loss()
        opt.zero_grad()
        loss.backward()
        opt.step()
    return residual.snapshot(), head_room, float(loss.detach())


# -- scoring -------------------------------------------------------------


def score_exam(
    name: str,
    field: PairField,
    *,
    pole_mode: str = "hidden",
    teacher: str = "pair_odd",
    leak_dir: torch.Tensor | None = None,
    hold_weight: float = 0.0,
    common_beta: float = 0.0,
    unroll_steps: int = 1,
    blind_weight: float = DUAL_BAND_WEIGHT,
    blind_cut: float = 0.0,
    even_scale: float = 1.0,
    even_hold: bool = False,
    steps: int = 400,
    seed: int = 0,
) -> dict:
    """Fit one recipe, then listen to it for ``out_steps`` tokens."""
    residual, head_room, final = fit_exam(
        field,
        pole_mode=pole_mode,
        teacher=teacher,
        leak_dir=leak_dir,
        hold_weight=hold_weight,
        common_beta=common_beta,
        unroll_steps=unroll_steps,
        blind_weight=blind_weight,
        blind_cut=blind_cut,
        even_scale=even_scale,
        even_hold=even_hold,
        steps=steps,
        seed=seed,
    )
    head = field.readout()
    t_plus_roll, t_minus_roll, words = teacher_rollouts(field, head)
    d_plus = residual.delta(1.0)
    d_minus = residual.delta(-1.0)
    plus = [field.poles(r)[2] + d_plus for r in range(int(field.rows))]
    minus = [field.poles(r)[2] + d_minus for r in range(int(field.rows))]
    row = rollout_report(
        field,
        plus,
        minus,
        readout=head,
        teacher_plus=t_plus_roll,
        teacher_minus=t_minus_roll,
        corpus=words,
    )
    ceiling = rollout_report(
        field,
        [field.poles(r)[0] for r in range(int(field.rows))],
        [field.poles(r)[1] for r in range(int(field.rows))],
        readout=head,
        teacher_plus=t_plus_roll,
        teacher_minus=t_minus_roll,
        corpus=words,
    )
    geom = target_geometry(
        field,
        teacher=teacher,
        leak_dir=leak_dir,
        common_beta=common_beta,
        even_scale=even_scale,
    )

    a = field.odd(0)
    pos, _neg, neu = field.poles(0)
    targets = [
        teacher_points(
            field,
            r,
            teacher=teacher,
            leak_dir=leak_dir,
            common_beta=common_beta,
            even_scale=even_scale,
        )
        for r in range(int(field.rows))
    ]
    percs: list[tuple[float, float]] = []
    for r, (t_plus, t_minus) in enumerate(targets):
        base = field.poles(r)[2]
        percs.append(
            (
                float((plus[r] - t_plus).norm())
                / float((t_plus - base).norm().clamp_min(1e-8)),
                float((minus[r] - t_minus).norm())
                / float((t_minus - base).norm().clamp_min(1e-8)),
            )
        )
    pperc = sum(p for p, _ in percs) / len(percs)
    nperc = sum(n for _, n in percs) / len(percs)

    d_hat = field.delivery_dir()
    want = abs(float((targets[0][0] - field.poles(0)[2]) @ d_hat))
    got = abs(float(d_plus @ d_hat))
    invisible_kept = got / want if want > 1e-8 else None
    self_match = teacher_self_match(t_plus_roll, t_minus_roll)
    swing_kept = row["roll_swing"] / (abs(ceiling["roll_swing"]) + 1e-8)
    # Leak in probability mass over the whole continuation, against the
    # concept mass the same continuation moved. Same shape as the sheet
    # cell's single-token ``leak_tok``, so the two cells agree on a number.
    leak_tok = (
        row["first_unused_swing"] / (abs(row["first_swing_mass"]) + 1e-8)
        if field.has_unused()
        else None
    )
    leak_roll = (
        row["roll_unused_swing"] / (abs(row["roll_swing_mass"]) + 1e-8)
        if field.has_unused()
        else None
    )

    out = dict(row)
    out.update(
        {
            "name": name,
            "cell": field.kind,
            "pole_mode": pole_mode,
            "teacher": teacher,
            "hold_weight": float(hold_weight) if leak_dir is not None else 0.0,
            "common_beta": float(common_beta),
            # Pair coordinates the sheet field has no column for.
            "divergence": field.divergence(),
            "visible_share": field.visible_share(),
            "invisible_share": field.invisible_share(),
            "probe_cos": field.probe_cos(0),
            # Live log columns. Logged, never gated.
            "pair_odd_cos": cosine(d_plus, a),
            "collapse": cosine(d_plus, d_minus),
            "pole_cos": cosine(d_plus, pos - neu),
            "pperc": pperc,
            "nperc": nperc,
            "perc_gap": abs(pperc - nperc),
            "loss": final,
            "loss_floor": head_room,
            "loss_solved": 1.0 - final / (head_room + 1e-12),
            # What the loss could not see, and whether it arrived anyway.
            "invisible_kept": invisible_kept,
            "leak_tok": leak_tok,
            "leak_roll": leak_roll,
            "roll_swing_kept": swing_kept,
            "roll_swing_mass_kept": row["roll_swing_mass"]
            / (abs(ceiling["roll_swing_mass"]) + 1e-8),
            "roll_match_kept": row["roll_match"] / (self_match + 1e-8),
            "teacher_self_match": self_match,
            "teacher_roll_swing": ceiling["roll_swing"],
        }
    )
    out.update(geom)
    out["axis"] = exam_verdicts(out)
    out["pass"] = all(v == "right" for v in out["axis"].values())
    out["near_gate"] = near_gate(out)
    out["kl_small_hidden_far"] = bool(
        out["loss_solved"] >= EXAM_KL_SOLVED
        and invisible_kept is not None
        and invisible_kept < EXAM_INVISIBLE_ARRIVED
    )
    out["reason"] = exam_reason(out)
    return out


def exam_verdicts(row: dict) -> dict[str, str]:
    """Gates. Every one is something the student sings over a continuation.

    ``pair_odd_cos``, ``collapse``, ``p%`` / ``n%`` and the pole loss are
    deliberately absent. On the live exam the run with the best ``c+``, the
    smallest loss and the lowest ``p%`` is one of the two that garbled.
    """
    out = {
        "continuation": "right"
        if _at_least(row.get("roll_overlap"), EXAM_ROLL_OVERLAP)
        else "needs_help",
        "same_words": "right"
        if _at_least(row.get("roll_match_kept"), EXAM_MATCH_KEPT)
        else "needs_help",
        "off_caption": "right"
        if _at_most(row.get("roll_off_corpus"), EXAM_ROLL_OFF_MAX)
        else "needs_help",
        "coherence": "right"
        if _at_least(row.get("roll_coherence"), EXAM_COHERENCE)
        else "needs_help",
        "swing": "right"
        if _at_least(row.get("roll_swing_kept"), EXAM_ROLL_SWING)
        else "needs_help",
    }
    return out


# gate name -> (column, bound, "floor"/"cap", how close counts as near).
# The tolerance is per gate because the columns do not share a scale: the
# off-caption cap is 0.05 out of 1, so 0.05 either side of it is the whole
# healthy range.
GATE_BOUNDS = {
    "continuation": ("roll_overlap", EXAM_ROLL_OVERLAP, "floor", 0.05),
    "same_words": ("roll_match_kept", EXAM_MATCH_KEPT, "floor", 0.05),
    "off_caption": ("roll_off_corpus", EXAM_ROLL_OFF_MAX, "cap", 0.01),
    "coherence": ("roll_coherence", EXAM_COHERENCE, "floor", 0.03),
    "swing": ("roll_swing_kept", EXAM_ROLL_SWING, "floor", 0.05),
}


def near_gate(row: dict) -> list[str]:
    """Gates this row sits close enough to that the verdict could flip.

    A published verdict that turns on one of these is not seed-robust, and
    the table says which ones rather than pretending the number is sharp.
    The three live exam rows clear every gate that decides them by more
    than this, at every seed tested.
    """
    out = []
    for gate, (key, bound, kind, tol) in GATE_BOUNDS.items():
        value = row.get(key)
        if value is None or gate not in row.get("axis", {}):
            continue
        size = abs(float(value)) if kind == "cap" else float(value)
        if abs(size - float(bound)) <= float(tol):
            out.append(gate)
    return out


def _at_least(value, floor: float) -> bool:
    return value is not None and float(value) >= float(floor)


def _at_most(value, cap: float) -> bool:
    return value is not None and float(value) <= float(cap)


def exam_reason(row: dict) -> str:
    """One phrase naming what happened, in the words of the live listen."""
    gates = row["axis"]
    bits: list[str] = []
    if gates.get("coherence") != "right":
        bits.append("alternates between the two songs")
    if gates.get("off_caption") != "right":
        bits.append("words no caption of this song sings")
    if gates.get("swing") != "right":
        bits.append("no audible swing")
    if gates.get("continuation") != "right" and not bits:
        bits.append("sings words the pole caption does not")
    if gates.get("same_words") != "right" and not bits:
        bits.append("continuation drifts off the pole's own")
    why: list[str] = []
    if row.get("blend_teacher"):
        why.append("blend teacher")
    if float(row.get("visible_axis_eaten") or 0.0) > EXAM_AXIS_EATEN:
        why.append("axis eaten by ê")
    if row.get("kl_small_hidden_far"):
        why.append("KL-small / hidden-far")
    if row.get("pass"):
        return "on-continuation" + (f" (despite {', '.join(why)})" if why else "")
    if why:
        bits.append("because " + " + ".join(why))
    return "; ".join(bits) if bits else "off the teacher's continuation"


# -- cells ---------------------------------------------------------------


def recipes(field: PairField) -> list[tuple[str, dict]]:
    """The live recipes this pair can express, named as the board names them."""
    e = field.declared_e()
    out: list[tuple[str, dict]] = [
        ("pair_odd_midpoint", {"pole_mode": "hidden", "teacher": "pair_odd"}),
        ("faithful_raw", {"pole_mode": "hidden", "teacher": "faithful"}),
        ("semantic_kl_midpoint", {"pole_mode": "semantic_kl", "teacher": "pair_odd"}),
        ("semantic_kl_poles", {"pole_mode": "semantic_kl", "teacher": "faithful"}),
        (
            "faithful_sub_e_if_unused",
            {"pole_mode": "hidden", "teacher": "faithful_sub_e_if_unused", "leak_dir": e},
        ),
        (
            "semantic_kl_null",
            {"pole_mode": "semantic_kl_null", "teacher": "faithful"},
        ),
        (
            "hidden_kl_poles",
            {"pole_mode": "hidden_kl", "teacher": "faithful"},
        ),
        (
            "unrolled_kl",
            {"pole_mode": "unrolled_kl", "teacher": "faithful"},
        ),
        (
            "faithful_guard_e",
            {"pole_mode": "hidden", "teacher": "faithful_guard_e", "leak_dir": e},
        ),
        ("dual_band_poles", {"pole_mode": "dual_band", "teacher": "faithful"}),
        (
            "dual_band_guard_e",
            {"pole_mode": "dual_band", "teacher": "faithful_guard_e", "leak_dir": e},
        ),
        ("dual_band_midpoint", {"pole_mode": "dual_band", "teacher": "pair_odd"}),
    ]
    if e is not None:
        out += [
            (
                "hold_e_perp_l8",
                {
                    "pole_mode": "hidden",
                    "teacher": "pair_odd",
                    "leak_dir": e,
                    "hold_weight": 8.0,
                },
            ),
            (
                "pair_odd_sub_e",
                {"pole_mode": "hidden", "teacher": "pair_odd_sub_e", "leak_dir": e},
            ),
            (
                "faithful_sub_e",
                {"pole_mode": "hidden", "teacher": "faithful_sub_e", "leak_dir": e},
            ),
            (
                "semantic_kl_sub_e",
                {
                    "pole_mode": "semantic_kl",
                    "teacher": "faithful_sub_e",
                    "leak_dir": e,
                },
            ),
        ]
    return out


def exam_cell(cell: str, *, steps: int = 400, seed: int = 0) -> list[dict]:
    """Score every live recipe this pair can express."""
    if cell not in CELLS:
        raise ValueError(f"cell must be one of {sorted(CELLS)}, got {cell!r}")
    field = CELLS[cell](seed=seed)
    return [
        score_exam(name, field, steps=steps, seed=seed, **kwargs)
        for name, kwargs in recipes(field)
    ]


def exam_table(*, steps: int = 400, seed: int = 0) -> dict[str, list[dict]]:
    return {cell: exam_cell(cell, steps=steps, seed=seed) for cell in CELLS}


def live_exam_rows(table: dict[str, list[dict]]) -> list[dict]:
    """Join each live run to the fixture row that predicts it."""
    out = []
    for run, (recipe, cell) in LIVE_ROW.items():
        found = next((r for r in table[cell] if r["name"] == recipe), None)
        if found is None:
            raise KeyError(f"{recipe!r} is not scored on the {cell!r} cell")
        live = LIVE_EXAM[run]
        predicted = "pass" if found["pass"] else "fail"
        out.append(
            {
                "run": run,
                "recipe": recipe,
                "cell": cell,
                "predicted": predicted,
                "listen": live["listen"],
                "agrees": predicted == live["listen"],
                "reason": found["reason"],
                "heard": live["heard"],
                "row": found,
                "live": live,
            }
        )
    return out


def pair_coordinate_table() -> list[dict]:
    """The pair coordinates, per cell, with no fitting at all."""
    out = []
    for name, ctor in CELLS.items():
        field = ctor()
        e = field.declared_e()
        held = hold_direction(field, e)
        a = field.odd(0)
        overlap = (
            abs(float(a @ lm_unit(held))) / float(a.norm().clamp_min(1e-8))
            if held is not None
            else None
        )
        out.append(
            {
                "cell": name,
                "is": CELL_IS[name],
                "divergence": field.divergence(),
                "visible_share": field.visible_share(),
                "invisible_share": field.invisible_share(),
                "probe_cos": field.probe_cos(0),
                "common_share": float(field.common_vec(0).norm() / a.norm()),
                "declared_e": e is not None,
                "e_overlap_a": overlap,
                "has_unused": field.has_unused(),
            }
        )
    return out


DIVERGENCE_GRID = (0.0, 0.4, 0.8, 1.2, 1.6, 2.0, 2.6)


def divergence_sweep(
    grid: tuple[float, ...] = DIVERGENCE_GRID,
    *,
    steps: int = 200,
    seed: int = 0,
) -> list[dict]:
    """Walk the pair from one song to two, with everything else fixed.

    ``track`` grows and the declared ``ê`` grows with it, because the yaml
    writes ``ê`` out of the same genre and BPM the poles moved. ``shared``
    is re-solved at each point so the field keeps printing the logged
    energy-v4 pair cos, which is what makes this a sweep of *divergence*
    and not of the collapse number.
    """
    out = []
    for track in grid:
        field = divergent_field(track=track, e_track=track, seed=seed)
        e = field.declared_e()
        poles = score_exam(
            "semantic_kl_poles",
            field,
            pole_mode="semantic_kl",
            teacher="faithful",
            steps=steps,
            seed=seed,
        )
        row = {
            "track": float(track),
            "divergence": field.divergence(),
            "poles_match": poles["roll_match_kept"],
            "poles_overlap": poles["roll_overlap"],
            "poles_coherence": poles["roll_coherence"],
            "poles_swing": poles["roll_swing_kept"],
            "poles_pass": poles["pass"],
            "sub_e_axis_eaten": None,
            "sub_e_overlap": None,
            "sub_e_match": None,
            "sub_e_coherence": None,
            "sub_e_swing": None,
            "sub_e_blend": None,
            "sub_e_pass": None,
        }
        if e is not None:
            sub = score_exam(
                "semantic_kl_sub_e",
                field,
                pole_mode="semantic_kl",
                teacher="faithful_sub_e",
                leak_dir=e,
                steps=steps,
                seed=seed,
            )
            row.update(
                {
                    "sub_e_axis_eaten": sub["visible_axis_eaten"],
                    "sub_e_overlap": sub["roll_overlap"],
                    "sub_e_match": sub["roll_match_kept"],
                    "sub_e_coherence": sub["roll_coherence"],
                    "sub_e_swing": sub["roll_swing_kept"],
                    "sub_e_blend": sub["blend_teacher"],
                    "sub_e_pass": sub["pass"],
                }
            )
        out.append(row)
    return out


VISIBLE_GRID = (0.02, 0.10, 0.25, 0.45, 0.65, 0.85, 0.99)


def visible_sweep(
    grid: tuple[float, ...] = VISIBLE_GRID,
    *,
    steps: int = 200,
    seed: int = 0,
) -> list[dict]:
    """Walk the visible share of ``a`` on a close pair, at fixed ``‖a‖``.

    At 0.02 almost the whole axis is delivery detail the scored token
    cannot read and semantic KL has no gradient on it. At 0.99 the axis is
    the token. Hidden MSE runs alongside because it does not care where the
    axis lives.
    """
    out = []
    base = close_field()
    norm = float(base.odd_shape().norm())
    for share in grid:
        slider = max(1e-3, share * norm)
        delivery = max(0.0, (norm**2 - slider**2) ** 0.5)
        field = close_field(slider=slider, delivery=delivery, seed=seed)
        kl = score_exam(
            "semantic_kl_poles",
            field,
            pole_mode="semantic_kl",
            teacher="faithful",
            steps=steps,
            seed=seed,
        )
        mse = score_exam(
            "faithful_raw",
            field,
            pole_mode="hidden",
            teacher="faithful",
            steps=steps,
            seed=seed,
        )
        out.append(
            {
                "visible_share": field.visible_share(),
                "kl_loss": kl["loss"],
                "kl_solved": kl["loss_solved"],
                "kl_invisible_kept": kl["invisible_kept"],
                "kl_pperc": kl["pperc"],
                "kl_c_plus": kl["pair_odd_cos"],
                "kl_match": kl["roll_match"],
                "kl_swing": kl["roll_swing_kept"],
                "kl_pass": kl["pass"],
                "mse_invisible_kept": mse["invisible_kept"],
                "mse_match": mse["roll_match"],
                "mse_swing": mse["roll_swing_kept"],
                "mse_pass": mse["pass"],
            }
        )
    return out


def first_below(
    sweep: list[dict], key: str, threshold: float, coord: str
) -> float | None:
    """Smallest sweep coordinate where a column drops below a gate.

    Reported instead of the boolean pass column, which flips on and off
    near a gate because the rollout is sampled. A monotone measured column
    crossing a stated floor is the claim; the boolean is not.
    """
    for row in sweep:
        value = row.get(key)
        if value is not None and float(value) < float(threshold):
            return float(row[coord])
    return None


def first_above(
    sweep: list[dict], key: str, threshold: float, coord: str
) -> float | None:
    """Smallest sweep coordinate where a column reaches a gate."""
    for row in sweep:
        value = row.get(key)
        if value is not None and float(value) >= float(threshold):
            return float(row[coord])
    return None


def floatable(row: dict) -> dict:
    """JSON-safe subset: drop tensors and nested history."""
    out = {}
    for key, value in row.items():
        if isinstance(value, (int, float, str, bool)) or value is None:
            out[key] = value
        elif isinstance(value, list) and all(isinstance(v, str) for v in value):
            out[key] = list(value)
        elif isinstance(value, dict) and all(isinstance(v, str) for v in value.values()):
            out[key] = value
    return out
