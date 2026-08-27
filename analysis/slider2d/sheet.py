"""The sheet cell: what hidden-only fields cannot see about lyric garble.

Every existing 2-D / high-D cell scores a residual against hidden-space
geometry — ``cos(d+, a)``, ``cos(d+, d−)``, leak along ê. Those fields have
no notion of *what the model says next*, so they cannot distinguish "the
student reached the teacher point" from "the student reached a point no
caption occupies". v15 shipped LM halves whose live logs were the best in
the campaign (gender-lm-v4: cos± 0.97, collapse −0.95) and that sing words
which are not on the sheet. Both facts are true at once, and no cell in
this repo could hold them at once.

The smallest structure that can is a **frozen linear readout** on top of the
hidden field: ``logits = hidden @ W.T``, softmax, a tiny vocabulary. That
turns each hidden state into a next-token policy, which gives three things
the hidden fields lack:

- a **sheet**: the nucleus of a *real caption's* policy — the tokens the
  positive / negative pole hidden actually puts its mass on, plus the row's
  written lyric token.
- an **off-sheet** region: tokens on neither pole's sheet. A token can be
  further along the concept direction û than any real caption ever goes and
  still be off-sheet, because real captions always carry the shared
  caption component the readout uses to stay in-distribution.
- a **null space**: dims the readout cannot see. Hidden MSE pins them;
  semantic KL does not. Live that is most of the 3584-wide hidden state.

## Why the pair-odd midpoint is off the sheet

``a = ½(h+−h−)``, ``c = ½(h++h−) − h0``. Then ``h± = h0 ± a + c`` exactly,
so the v9 target ``t± = h0 ± a`` is the real pole **minus its whole common
component**. ``c`` is not noise: it is what both pole captions say and the
neutral skeleton does not — genre, BPM, mood, mix, instruments, all the
Structured Caption specificity that makes the continuation a song and not
a shrug. Deleting it is what walks the student off the sheet.

The live trainer already prints the size of ``c``. ``cos(pos−neu, neg−neu)``
is exactly ``(|c|²−|a|²)/(|c|²+|a|²)`` for ``c ⊥ a``, so

    |c| / |a| = √((1 + cos) / (1 − cos))

and the v4 probe table (MUSIC3.md) reads: gender −0.08, rapslow −0.08,
rhyme −0.05, distortion −0.03, energy 0.03, tempo 0.06, triphop 0.10,
breath 0.24, live 0.32. Every shipped LM axis has a common component
between **0.92× and 1.39×** the norm of the entire pair-odd axis, and
``--lm_target v9`` throws all of it away (``common_beta`` is ignored by
v9 — ``train_lm_slider_music3.py`` prints a note and drops it). The
trainer's only warning fires above +0.3, i.e. it treats a *large* ``c``
as the problem (inherited collapse) and a ``cos`` near 0 as healthy. Near
0 is the case where the deleted piece is the same size as the signal.

## What the cell shows

Hidden MSE onto ``t±`` reaches ``cos(d+, a) ≈ 0.97`` and
``cos(d+, d−) ≈ −0.95`` — the live gender-lm-v4 log to two decimals — while
the policy at ±1 moves most of its mass off the sheet and, past a flip
point in ``|c|/|a|``, hands the turn to an off-sheet token. Semantic KL
onto a real caption's policy stays on the sheet and prints *worse*
pair-odd numbers, because ``d± = ±a + c`` is not odd: its collapse rises
to ``cos(pos−neu, neg−neu)`` itself. Under the v16 default a logged
collapse near −1 is the bug, not the health check.

CPU only. No Hub, no GPU, no Music 3 weights. Does not change the live
trainer default (still ``--lm_target v9`` / ``--pole_mode hidden``; see
``docs/lm-sheet-goodhart.md``).
"""

from __future__ import annotations

from dataclasses import dataclass, field as dc_field

import torch

from analysis.slider2d.field import cosine
from analysis.slider2d.highd import BEND_GENDER
from conceptmod.textsliders.slider_targets import (
    DUAL_BAND_WEIGHT,
    LEAK_HOLD_WEIGHT,
    lm_axis_hold,
    lm_blind_projector,
    lm_dual_band_pole_loss,
    lm_faithful_gate_odd_sub_even,
    lm_faithful_guard_e,
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
)


# Live v4 probe table (MUSIC3.md "Structured-caption + end-regularized LM
# halves"): cos(pos−neu, neg−neu) per axis, on the captions that trained the
# shipped halves. This is the only live measurement of the common component.
LIVE_PROBE_COS = {
    "rapslow": -0.08,
    "gender": -0.08,
    "rhyme": -0.05,
    "distortion": -0.03,
    "energy": 0.03,
    "tempo": 0.06,
    "triphop": 0.10,
    "breath": 0.24,
    "live": 0.32,
}
# The trainer only warns above this.
LIVE_COLLAPSE_WARN = 0.3
# Live gender-lm-v4 final-50-step window, the best log in the campaign.
LIVE_V4_C_PLUS = 0.97
LIVE_V4_COLLAPSE = -0.95

# Gates. Pair-odd cos and collapse are deliberately absent: they are logged,
# never scored. That is the whole point of the cell.
#
# The sheet gate is *relative* to the caption's own on-sheet mass. A nucleus
# cut at ``sheet_p`` leaves the pole itself holding about ``sheet_p`` of its
# mass on its own sheet, so an absolute 0.90 floor would fail the ground
# truth. ``on_sheet_kept`` is the share of that ceiling the student holds.
SHEET_LOCK = 0.90
GARBLE_MAX = 0.05
LEAK_LOCK = 0.20
SWING_FLOOR = 0.60
# An argmax off the sheet is garble you can hear in one token.
ARGMAX_LOCK = 1.0
# "Looks locked" in the live log's own terms.
LOCK_LOOK_C_PLUS = 0.90
LOCK_LOOK_COLLAPSE = -0.85


def common_share(probe_cos: float) -> float:
    """``|c|/|a|`` from the live-logged ``cos(pos−neu, neg−neu)``.

    ``h±−h0 = ±a + c`` with ``c ⊥ a`` gives
    ``cos = (|c|²−|a|²)/(|c|²+|a|²)``, so ``|c|/|a| = √((1+cos)/(1−cos))``.
    ``cos = −1`` is a perfectly odd pair (the midpoint *is* the pole, and
    v9 is exact); ``cos = 0`` means the deleted common component is the
    same size as the whole slider axis.
    """
    cos = float(probe_cos)
    if not -1.0 < cos < 1.0:
        raise ValueError(f"probe cos must be in (-1, 1), got {probe_cos!r}")
    return ((1.0 + cos) / (1.0 - cos)) ** 0.5


def probe_cos(share: float) -> float:
    """Inverse of :func:`common_share` — what the live trainer would print."""
    ratio = float(share) ** 2
    return (ratio - 1.0) / (ratio + 1.0)


CONCEPT_TOKENS = ("slam", "hush")
LEAK_TOKENS = ("male", "female")
OFF_SHEET_TOKENS = ("garble_hi", "garble_lo")


@dataclass(frozen=True)
class Readout:
    """Frozen linear next-token head over a tiny vocabulary.

    ``gain`` is an inverse temperature. A real caption's continuation is
    peaked (a handful of plausible next tokens); with ``gain`` near 0 every
    policy is uniform and no sheet exists to fall off. It is calibrated
    once, on the *pole* policies, and then held fixed across every recipe —
    the sheet must not be a function of what the student did.
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
        """Hidden dims no vocabulary row reads. MSE pins these; KL cannot."""
        seen = self.weight.abs().sum(dim=0)
        return [i for i in range(self.weight.shape[1]) if float(seen[i]) == 0.0]


def nucleus(policy: torch.Tensor, sheet_p: float = 0.90) -> set[int]:
    """Token ids holding the first ``sheet_p`` of a policy's mass.

    Standard nucleus (top-p) support, so the sheet is not a hand-picked
    token list: it is whatever the real caption would plausibly say next.
    """
    order = torch.argsort(policy, descending=True)
    total = 0.0
    out: set[int] = set()
    for idx in order.tolist():
        out.add(int(idx))
        total += float(policy[idx])
        if total >= float(sheet_p):
            break
    return out


@dataclass(frozen=True)
class SheetField:
    """Hidden field with a written lyric per row and a readout on top.

    Dim layout, all orthonormal:

        0                       û   concept, slam ↔ hush
        1                       ê   unused attribute (the singer's gender)
        2                       ŝ   sheet: Structured-Caption specificity
                                    that both poles have and the neutral
                                    skeleton does not
        3 .. 3+rows-1           one written-lyric axis per prompt row
        3+rows .. +null_dims    production detail the readout cannot see

    Poles, per row ``r``::

        h0_r = lyric · l̂_r
        h±_r = h0_r ± (slider·û + leak·ê + null·n̂) + common·|a|·ŝ

    so ``a`` carries the concept, the unused attribute and the invisible
    detail, ``c = common·|a|·ŝ`` is shared, and ``t± = h0 ± a`` has no ŝ
    at all. ``row_scales`` varies pole strength across rows the way live
    caption rows do (mean pairwise target cos 0.32, MUSIC3.md), so one
    shared residual has to compromise instead of hitting a single point.
    """

    rows: int = 3
    slider: float = 1.0
    leak: float = 0.45
    null: float = 0.50
    common: float = dc_field(default_factory=lambda: common_share(LIVE_PROBE_COS["gender"]))
    lyric: float = 1.00
    null_dims: int = 2
    row_scales: tuple[float, ...] = (1.0, 0.85, 1.15)
    gain: float = 2.5
    sheet_p: float = 0.90
    bend: float = BEND_GENDER
    seed: int = 0

    def __post_init__(self) -> None:
        if int(self.rows) < 1:
            raise ValueError(f"rows must be ≥ 1, got {self.rows!r}")
        if len(self.row_scales) < int(self.rows):
            raise ValueError("row_scales must cover every row")
        if float(self.slider) <= 0.0:
            raise ValueError("slider must be > 0 (it is the intended axis)")

    # -- geometry ---------------------------------------------------------

    @property
    def dim(self) -> int:
        return 3 + int(self.rows) + int(self.null_dims)

    def _basis(self, index: int) -> torch.Tensor:
        out = torch.zeros(self.dim)
        out[index] = 1.0
        return out

    def short_u(self) -> torch.Tensor:
        return self._basis(0)

    def leak_e(self) -> torch.Tensor:
        return self._basis(1)

    def sheet_dir(self) -> torch.Tensor:
        return self._basis(2)

    def lyric_dir(self, row: int) -> torch.Tensor:
        return self._basis(3 + int(row))

    def null_dirs(self) -> list[torch.Tensor]:
        start = 3 + int(self.rows)
        return [self._basis(i) for i in range(start, start + int(self.null_dims))]

    def null_mix(self) -> torch.Tensor:
        dirs = self.null_dirs()
        if not dirs:
            return torch.zeros(self.dim)
        return lm_unit(sum(dirs[1:], dirs[0]))

    def odd(self, row: int = 0) -> torch.Tensor:
        """``a`` for one row: concept + unused attribute + invisible detail."""
        scale = float(self.row_scales[int(row)])
        return scale * (
            float(self.slider) * self.short_u()
            + float(self.leak) * self.leak_e()
            + float(self.null) * self.null_mix()
        )

    def common_vec(self, row: int = 0) -> torch.Tensor:
        """``c``, sized as a multiple of ``||a||`` so ``common`` is the ratio."""
        return float(self.common) * float(self.odd(row).norm()) * self.sheet_dir()

    def poles(self, row: int = 0) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        neu = float(self.lyric) * self.lyric_dir(row)
        a = self.odd(row)
        c = self.common_vec(row)
        return neu + a + c, neu - a + c, neu

    def probe_cos(self, row: int = 0) -> float:
        """``cos(pos−neu, neg−neu)`` — the number the live trainer prints."""
        pos, neg, neu = self.poles(row)
        return cosine(pos - neu, neg - neu)

    def readout(self) -> Readout:
        """Vocabulary and head.

        Every real-word row loads ŝ: a token is only a plausible
        continuation if the state says "this is a fully specified caption
        of this song". ``garble_*`` is the opposite — it is *further* along
        û than any caption reaches (1.3 vs 1.0) and anti-loaded on ŝ. No
        real pole is ever near it, because real poles always carry ``c``.
        Strip ``c`` and the loudest token wins even though nothing on the
        sheet would say it. Lyric rows read only their own row's axis, so
        row 1's lyric is off row 0's sheet.
        """
        tokens = list(CONCEPT_TOKENS) + list(LEAK_TOKENS) + list(OFF_SHEET_TOKENS)
        tokens += [f"lyric{r}" for r in range(int(self.rows))]
        weight = torch.zeros(len(tokens), self.dim)
        weight[0, 0], weight[0, 2] = 1.0, 1.0  # slam
        weight[1, 0], weight[1, 2] = -1.0, 1.0  # hush
        weight[2, 1], weight[2, 2] = 1.0, 1.0  # male
        weight[3, 1], weight[3, 2] = -1.0, 1.0  # female
        weight[4, 0], weight[4, 2] = 1.30, -1.50  # garble_hi
        weight[5, 0], weight[5, 2] = -1.30, -1.50  # garble_lo
        for r in range(int(self.rows)):
            weight[6 + r, 3 + r] = 1.20
        return Readout(tuple(tokens), weight, gain=float(self.gain))

    def gate_matrix(self) -> torch.Tensor:
        gen = torch.Generator().manual_seed(int(self.seed) + 5)
        q, _r = torch.linalg.qr(torch.randn(self.dim, self.dim, generator=gen))
        return q


def gender_like_field(**kwargs) -> SheetField:
    """Clean pair, no declared ê: the poles *are* the concept.

    Gender's own probe cos is −0.08, so even the axis with nothing to leak
    still deletes a common component 0.92× the size of its whole axis.
    """
    base = {
        "leak": 0.0,
        "null": 0.0,
        "null_dims": 0,
        "common": common_share(LIVE_PROBE_COS["gender"]),
    }
    base.update(kwargs)
    return SheetField(**base)


def leaky_field(**kwargs) -> SheetField:
    """Energy-like: unused gender inside ``a``, plus invisible mix detail."""
    base = {"common": common_share(LIVE_PROBE_COS["energy"])}
    base.update(kwargs)
    return SheetField(**base)


def hold_direction(field: SheetField, leak_dir: torch.Tensor | None) -> torch.Tensor | None:
    """What the live trainer holds: ``ê_⊥ = ê − (ê·û)û``."""
    if leak_dir is None:
        return None
    return lm_hold_dir(leak_dir, slider_dir=field.short_u(), mode="slider")


# -- the sheet ------------------------------------------------------------


@dataclass(frozen=True)
class Sheet:
    """Ground truth for one row: what each real pole caption would say.

    Built from the *poles*, never from a target or a fit, so every recipe
    is scored against the same sheet.
    """

    row: int
    plus: frozenset[int]
    minus: frozenset[int]
    lyric: int

    def written(self) -> frozenset[int]:
        return frozenset({self.lyric})

    def on(self, sign: float) -> frozenset[int]:
        return self.plus if sign >= 0.0 else self.minus

    def anywhere(self) -> frozenset[int]:
        return self.plus | self.minus | self.written()


def row_sheet(field: SheetField, readout: Readout, row: int) -> Sheet:
    pos, neg, _neu = field.poles(row)
    return Sheet(
        row=int(row),
        plus=frozenset(nucleus(readout.policy(pos), field.sheet_p)),
        minus=frozenset(nucleus(readout.policy(neg), field.sheet_p)),
        lyric=readout.index(f"lyric{int(row)}"),
    )


def sheets(field: SheetField, readout: Readout | None = None) -> list[Sheet]:
    head = readout if readout is not None else field.readout()
    return [row_sheet(field, head, r) for r in range(int(field.rows))]


def corpus(book: list[Sheet]) -> frozenset[int]:
    """Every token some real caption of this song supports.

    Union over rows and both signs, plus every written lyric. Anything
    outside is a word the song has no sheet for — the off-sheet region.
    Taking the union rather than one pole's nucleus keeps the off-sheet set
    from depending on argsort ties between the two poles' own words.
    """
    out: frozenset[int] = frozenset()
    for sheet in book:
        out = out | sheet.anywhere()
    return out


def policy_report(
    policy: torch.Tensor,
    sheet: Sheet,
    off_sheet: frozenset[int],
    *,
    sign: float,
) -> dict[str, float]:
    """On-sheet mass, off-sheet mass, argmax and lyric mass for one policy."""
    on = sheet.on(sign)
    top = int(torch.argmax(policy))
    return {
        "on_sheet": float(sum(float(policy[i]) for i in on)),
        "garble": float(sum(float(policy[i]) for i in off_sheet)),
        "lyric_mass": float(policy[sheet.lyric]),
        "argmax_on_sheet": 1.0 if top in on else 0.0,
    }


def argmax_token(policy: torch.Tensor, readout: Readout) -> str:
    return readout.tokens[int(torch.argmax(policy))]


def hidden_sheet_report(
    field: SheetField,
    hiddens_plus: list[torch.Tensor],
    hiddens_minus: list[torch.Tensor],
    *,
    readout: Readout | None = None,
    sheet_list: list[Sheet] | None = None,
) -> dict:
    """Sheet columns for a set of per-row ±1 hidden states.

    Works on target points as well as on fits, which is how the cell shows
    the midpoint is off the sheet with no optimizer in the loop.
    """
    head = readout if readout is not None else field.readout()
    book = sheet_list if sheet_list is not None else sheets(field, head)
    off_sheet = frozenset(
        i for i in range(len(head.tokens)) if i not in corpus(book)
    )
    slam, hush = (head.index(t) for t in CONCEPT_TOKENS)
    male, female = (head.index(t) for t in LEAK_TOKENS)
    acc: dict[str, list[float]] = {}
    concept_swing: list[float] = []
    leak_swing: list[float] = []
    kl_rows: list[float] = []
    said: list[str] = []
    for row, (h_plus, h_minus) in enumerate(zip(hiddens_plus, hiddens_minus)):
        sheet = book[row]
        p_plus = head.policy(h_plus)
        p_minus = head.policy(h_minus)
        said.append(argmax_token(p_plus, head))
        said.append(argmax_token(p_minus, head))
        for sign, policy in ((1.0, p_plus), (-1.0, p_minus)):
            tag = "plus" if sign > 0 else "minus"
            for key, value in policy_report(policy, sheet, off_sheet, sign=sign).items():
                acc.setdefault(key, []).append(value)
                acc.setdefault(f"{key}_{tag}", []).append(value)
        # Odd part of the token swing: what the slider actually moved.
        concept_swing.append(
            0.5
            * float(
                (p_plus[slam] - p_plus[hush]) - (p_minus[slam] - p_minus[hush])
            )
        )
        leak_swing.append(
            0.5
            * float(
                (p_plus[male] - p_plus[female]) - (p_minus[male] - p_minus[female])
            )
        )
        pos, neg, _neu = field.poles(row)
        kl_rows.append(
            0.5
            * float(
                _kl(head.policy(pos), p_plus) + _kl(head.policy(neg), p_minus)
            )
        )
    out: dict = {key: sum(vals) / len(vals) for key, vals in acc.items()}
    out["says"] = ",".join(said)
    out["concept_swing"] = sum(concept_swing) / len(concept_swing)
    out["leak_swing"] = sum(leak_swing) / len(leak_swing)
    out["leak_tok"] = out["leak_swing"] / (abs(out["concept_swing"]) + 1e-8)
    out["kl_pole"] = sum(kl_rows) / len(kl_rows)
    return out


def _kl(p: torch.Tensor, q: torch.Tensor) -> float:
    return float((p * (p.clamp_min(1e-12).log() - q.clamp_min(1e-12).log())).sum())


def teacher_swings(field: SheetField, readout: Readout | None = None) -> dict:
    """The real poles' own sheet report — the ceiling every recipe is read against."""
    head = readout if readout is not None else field.readout()
    plus, minus = [], []
    for row in range(int(field.rows)):
        pos, neg, _neu = field.poles(row)
        plus.append(pos)
        minus.append(neg)
    return hidden_sheet_report(field, plus, minus, readout=head)


# -- students and teachers ------------------------------------------------


STUDENTS = ("odd_even", "bend")


@dataclass
class SharedResidual:
    """One residual serving every row: ``δ(σ) = σ·w + |σ|·even``.

    A live LoRA is one set of weights added at the prompt-last position of
    every prompt row, so the residual is shared and only the neutral it is
    added to changes.

    ``odd_even`` is ``train.Residual``'s LM student: ``even`` is a free
    vector. That is the right capacity here, because the whole question is
    whether the target has an even part worth learning. A symmetric
    pair-odd teacher has none, so ``w_even`` never leaves zero and the fit
    prints ``cos(d+, a) = 1``, ``cos(d+, d−) = −1``; a real-caption teacher
    has ``c``, and the student must find it.

    ``bend`` is ``highd.BendResidual``: no free even term, and the even
    reply is a fixed rotation of the odd one, sized by the ±1 asymmetry
    ``highd`` derives from the gender-lm-v4 collapse log. It reproduces the
    live 0.97 / −0.95 pair instead of the exact 1.00 / −1.00, which is what
    ``live_log_table`` uses it for. It cannot represent ``c``, so it is not
    the student for the recipe comparison — ``||w||`` and the rotation are
    the only levers it has, and an optimizer handed a caption target will
    inflate ``||w||`` into the readout's null space to synthesize an even
    reply. That is a property of the caricature, not of a LoRA.
    """

    w: torch.Tensor
    kind: str = "odd_even"
    w_even: torch.Tensor | None = None
    gate: torch.Tensor | None = None
    bend: float = 0.0

    @classmethod
    def create(cls, field: SheetField, kind: str = "odd_even") -> "SharedResidual":
        mode = str(kind).strip().lower()
        if mode not in STUDENTS:
            raise ValueError(f"student must be one of {STUDENTS}, got {kind!r}")
        w = torch.zeros(field.dim, requires_grad=True)
        if mode == "odd_even":
            return cls(w, mode, torch.zeros(field.dim, requires_grad=True))
        return cls(w, mode, None, field.gate_matrix(), float(field.bend))

    def even(self) -> torch.Tensor:
        if self.w_even is not None:
            return self.w_even
        if self.gate is None or float(self.bend) == 0.0:
            return torch.zeros_like(self.w)
        norm = self.w.norm().clamp_min(1e-8)
        along = self.w / norm
        spun = self.gate @ self.w
        across = spun - (spun @ along) * along
        across = across / across.norm().clamp_min(1e-8)
        return float(self.bend) * norm * across

    def delta(self, scale: float) -> torch.Tensor:
        return float(scale) * self.w + abs(float(scale)) * self.even()

    def parameters(self) -> list[torch.Tensor]:
        params = [self.w]
        if self.w_even is not None:
            params.append(self.w_even)
        return params

    def snapshot(self) -> "SharedResidual":
        even = None if self.w_even is None else self.w_even.detach().clone()
        return SharedResidual(self.w.detach().clone(), self.kind, even, self.gate, self.bend)


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
POLE_MODES = ("hidden", "semantic_kl", "semantic_kl_null", "dual_band")


def _sub_e(h: torch.Tensor, neu: torch.Tensor, held: torch.Tensor | None) -> torch.Tensor:
    if held is None:
        return h
    unit = lm_unit(held)
    return h - ((h - neu).flatten() @ unit) * unit


def teacher_points(
    field: SheetField,
    row: int,
    *,
    teacher: str = "pair_odd",
    leak_dir: torch.Tensor | None = None,
    common_beta: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """The two hidden states one recipe aims at, for one row.

    ``pair_odd``: ``t± = h0 ± a`` — live ``--lm_target v9``. Not a caption.
    ``pair_odd_sub_e``: ``lm_pair_odd_sub_e``, live ``--lm_target
    pair_odd_sub_e`` (#20). Still not a caption: it drops ``ê_⊥`` *and*
    the whole common component.
    ``faithful``: the raw pole hiddens. v6, and the only live target that
    is a real caption. Also ``--lm_target symmetric --common_beta 1``,
    which ``lm_hidden_targets`` makes exactly equal to ``(pos, neg)``.
    ``faithful_sub_e``: the pole with ``ê_⊥`` subtracted. Not literally a
    caption either, but one small step off one: it keeps ``c``.
    """
    pos, neg, neu = field.poles(row)
    mode = str(teacher).strip().lower()
    if mode == "pair_odd":
        return lm_hidden_targets(
            pos, neg, neu, target_mode="symmetric", common_beta=float(common_beta)
        )
    if mode == "pair_odd_sub_e":
        if leak_dir is None:
            raise ValueError("pair_odd_sub_e needs a declared ê")
        return lm_pair_odd_sub_e(pos, neg, neu, leak_dir, slider_dir=field.short_u())
    if mode == "faithful":
        return pos, neg
    if mode == "faithful_sub_e":
        if leak_dir is None:
            raise ValueError("faithful_sub_e needs a declared ê")
        held = hold_direction(field, leak_dir)
        return _sub_e(pos, neu, held), _sub_e(neg, neu, held)
    if mode == "faithful_sub_e_if_unused":
        return lm_faithful_sub_e_if_unused(
            pos, neg, neu, leak_dir, slider_dir=field.short_u()
        )
    if mode == "faithful_guard_e":
        if leak_dir is None:
            return pos, neg
        return lm_faithful_guard_e(pos, neg, neu, leak_dir, slider_dir=field.short_u())
    if mode == "faithful_sub_even_e":
        if leak_dir is None:
            raise ValueError("faithful_sub_even_e needs a declared ê")
        return lm_faithful_sub_even_e(pos, neg, neu, leak_dir, slider_dir=field.short_u())
    if mode == "faithful_sub_even_e_if_unused":
        return lm_faithful_sub_even_e_if_unused(
            pos, neg, neu, leak_dir, slider_dir=field.short_u()
        )
    if mode == "faithful_sub_even_e_guard":
        if leak_dir is None:
            return pos, neg
        return lm_faithful_sub_even_e_guard(
            pos, neg, neu, leak_dir, slider_dir=field.short_u()
        )
    # Leftover-sheet unused leftover is odd. Leak-pair even (ê_even) is
    # the energy-v4 track *sum*; it is near-zero here, so even_dir=None
    # is the honest leftover-sheet teacher, not a missing wire.
    if mode == "faithful_sub_even_blend":
        return lm_faithful_sub_even_blend(pos, neg, neu, None)
    if mode == "faithful_sub_even_blend_if_unused":
        return lm_faithful_sub_even_blend_if_unused(
            pos, neg, neu, leak_dir, None, slider_dir=field.short_u()
        )
    if mode == "faithful_sub_even_blend_guard":
        return lm_faithful_sub_even_blend_guard(pos, neg, neu, None)
    if mode == "faithful_gate_odd_sub_even":
        return lm_faithful_gate_odd_sub_even(
            pos, neg, neu, leak_dir, slider_dir=field.short_u()
        )
    if mode == "faithful_gate_odd_sub_even_blend":
        return lm_faithful_gate_odd_sub_even(
            pos, neg, neu, leak_dir, slider_dir=field.short_u(), even_dir=None
        )
    raise ValueError(f"teacher must be one of {TEACHERS}, got {teacher!r}")


def teacher_sheet_row(
    name: str,
    field: SheetField,
    *,
    teacher: str = "pair_odd",
    leak_dir: torch.Tensor | None = None,
    common_beta: float = 0.0,
) -> dict:
    """Sheet report for the *target points themselves*. No optimizer.

    This is the load-bearing measurement: whatever a student does, hidden
    MSE has no reason to land anywhere but here, and if here is off the
    sheet then a perfect fit sings garble.
    """
    head = field.readout()
    plus, minus = [], []
    for row in range(int(field.rows)):
        t_plus, t_minus = teacher_points(
            field, row, teacher=teacher, leak_dir=leak_dir, common_beta=common_beta
        )
        plus.append(t_plus)
        minus.append(t_minus)
    report = hidden_sheet_report(field, plus, minus, readout=head)
    a = field.odd(0)
    _pos, _neg, neu = field.poles(0)
    report.update(
        {
            "name": name,
            "teacher": teacher,
            "common_beta": float(common_beta),
            "common": float(field.common),
            "probe_cos": field.probe_cos(0),
            "is_caption": teacher.strip().lower() == "faithful" or float(common_beta) == 1.0,
            # How far the target sits from the caption it claims to be, in
            # units of the whole slider axis.
            "off_caption": float((plus[0] - field.poles(0)[0]).norm() / a.norm().clamp_min(1e-8)),
            "sheet_dir_kept": float((plus[0] - neu) @ field.sheet_dir())
            / float(field.common_vec(0).norm() + 1e-8),
        }
    )
    return report


def fit_sheet(
    field: SheetField,
    *,
    pole_mode: str = "hidden",
    teacher: str = "pair_odd",
    leak_dir: torch.Tensor | None = None,
    hold_weight: float = 0.0,
    common_beta: float = 0.0,
    student: str = "odd_even",
    blind_weight: float = DUAL_BAND_WEIGHT,
    steps: int = 400,
    lr: float = 0.08,
    seed: int = 0,
) -> SharedResidual:
    """Fit one shared residual with the live pole loss of ``pole_mode``.

    ``hidden`` is ``lm_slider_loss`` (+ optional ``lm_axis_hold``), the
    live default. ``semantic_kl`` is ``lm_semantic_pole_loss`` on the
    readout's logits; the brief's v16 sets hold to 0 there, and this
    function does not stop you passing one so the choice stays visible.
    """
    mode = str(pole_mode).strip().lower()
    if mode not in POLE_MODES:
        raise ValueError(f"pole_mode must be one of {POLE_MODES}, got {pole_mode!r}")
    head = field.readout()
    null_basis = lm_readout_null_basis(head.weight) if mode == "semantic_kl_null" else None
    blind = lm_blind_projector(head.weight) if mode == "dual_band" else None
    held = hold_direction(field, leak_dir)
    lam = float(hold_weight) if held is not None else 0.0
    targets = [
        teacher_points(
            field, row, teacher=teacher, leak_dir=leak_dir, common_beta=common_beta
        )
        for row in range(int(field.rows))
    ]
    neutrals = [field.poles(row)[2] for row in range(int(field.rows))]

    torch.manual_seed(int(seed))
    residual = SharedResidual.create(field, student)
    opt = torch.optim.Adam(residual.parameters(), lr=float(lr))

    def step_loss() -> torch.Tensor:
        total = None
        for (t_plus, t_minus), neu in zip(targets, neutrals):
            pred_plus = neu + residual.delta(1.0)
            pred_minus = neu + residual.delta(-1.0)
            hold = None
            if held is not None and lam > 0.0:
                hold = lm_axis_hold(pred_plus, pred_minus, neu, held)
            if mode == "hidden":
                term = lm_slider_loss(
                    pred_plus,
                    pred_minus,
                    t_plus,
                    t_minus,
                    hold=hold,
                    hold_weight=lam if hold is not None else 0.0,
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
                    hold_weight=lam if hold is not None else 0.0,
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
                    hold_weight=lam if hold is not None else 0.0,
                )
            else:
                term = lm_semantic_pole_loss(
                    head.logits(pred_plus),
                    head.logits(pred_minus),
                    head.logits(t_plus),
                    head.logits(t_minus),
                    hold=hold,
                    hold_weight=lam if hold is not None else 0.0,
                )
            total = term if total is None else total + term
        return total / float(len(targets))

    for _ in range(int(steps)):
        loss = step_loss()
        opt.zero_grad()
        loss.backward()
        opt.step()
    return residual.snapshot()


def score_sheet(
    name: str,
    field: SheetField,
    *,
    pole_mode: str = "hidden",
    teacher: str = "pair_odd",
    leak_dir: torch.Tensor | None = None,
    hold_weight: float = 0.0,
    common_beta: float = 0.0,
    student: str = "odd_even",
    steps: int = 400,
    seed: int = 0,
) -> dict:
    """Fit one recipe and report hidden geometry *and* sheet behaviour."""
    residual = fit_sheet(
        field,
        pole_mode=pole_mode,
        teacher=teacher,
        leak_dir=leak_dir,
        hold_weight=hold_weight,
        common_beta=common_beta,
        student=student,
        steps=steps,
        seed=seed,
    )
    head = field.readout()
    d_plus = residual.delta(1.0)
    d_minus = residual.delta(-1.0)
    plus = [field.poles(r)[2] + d_plus for r in range(int(field.rows))]
    minus = [field.poles(r)[2] + d_minus for r in range(int(field.rows))]
    row = hidden_sheet_report(field, plus, minus, readout=head)

    a = field.odd(0)
    pos, _neg, neu = field.poles(0)
    ceiling = teacher_swings(field, head)
    held = hold_direction(field, leak_dir)
    on_u = float(d_plus @ field.short_u())
    row.update(
        {
            "name": name,
            "pole_mode": pole_mode,
            "teacher": teacher,
            "student": student,
            "hold_weight": float(hold_weight) if held is not None else 0.0,
            "common": float(field.common),
            "common_beta": float(common_beta),
            "probe_cos": field.probe_cos(0),
            # Live log columns. Logged, never gated.
            "pair_odd_cos": cosine(d_plus, a),
            "collapse": cosine(d_plus, d_minus),
            # The honest hidden-space lock: cos to the real pole displacement.
            "pole_cos": cosine(d_plus, pos - neu),
            "sheet_dir_kept": float(d_plus @ field.sheet_dir())
            / float(field.common_vec(0).norm() + 1e-8),
            "leak_hidden": abs(float(d_plus @ field.leak_e())) / (abs(on_u) + 1e-8),
            "null_kept": _null_kept(field, d_plus),
            "swing_kept": row["concept_swing"] / (abs(ceiling["concept_swing"]) + 1e-8),
            "on_sheet_kept": row["on_sheet"] / (ceiling["on_sheet"] + 1e-8),
            "teacher_on_sheet": ceiling["on_sheet"],
            "teacher_garble": ceiling["garble"],
            "teacher_leak_tok": ceiling["leak_tok"],
        }
    )
    row["axis"] = sheet_verdicts(row)
    row["pass"] = all(v == "right" for v in row["axis"].values())
    row["looks_locked"] = (
        row["pair_odd_cos"] >= LOCK_LOOK_C_PLUS and row["collapse"] <= LOCK_LOOK_COLLAPSE
    )
    row["misleading_lock"] = bool(row["looks_locked"] and not row["pass"])
    return row


def _null_kept(field: SheetField, delta: torch.Tensor) -> float:
    """Share of the teacher's readout-invisible content the fit copied."""
    dirs = field.null_dirs()
    if not dirs:
        return 0.0
    a = field.odd(0)
    want = sum(float(a @ d) ** 2 for d in dirs) ** 0.5
    got = sum(float(delta @ d) ** 2 for d in dirs) ** 0.5
    return got / (want + 1e-8) if want > 1e-8 else 0.0


def sheet_verdicts(row: dict) -> dict[str, str]:
    """Sheet / garble / leak / audibility. Pair-odd cos and collapse are not here."""
    return {
        "sheet": "right" if row["on_sheet_kept"] >= SHEET_LOCK else "needs_help",
        "garble": "right" if row["garble"] <= GARBLE_MAX else "needs_help",
        "argmax": "right" if row["argmax_on_sheet"] >= ARGMAX_LOCK else "needs_help",
        "leak": "right" if abs(row["leak_tok"]) <= LEAK_LOCK else "needs_help",
        "slider": "right" if row["swing_kept"] >= SWING_FLOOR else "needs_help",
    }


# -- cells ----------------------------------------------------------------


def gender_cell(*, steps: int = 400, seed: int = 0) -> list[dict]:
    """Gender-like: clean pair, no declared ê, hold 0.

    Nothing to leak, so the only question left is whether the target is a
    caption. Live gender-lm-v4 is the best log in the campaign.
    """
    field = gender_like_field()
    rows = [
        score_sheet(
            "v9_hidden",
            field,
            pole_mode="hidden",
            teacher="pair_odd",
            steps=steps,
            seed=seed,
        ),
        score_sheet(
            "hidden_beta1",
            field,
            pole_mode="hidden",
            teacher="pair_odd",
            common_beta=1.0,
            steps=steps,
            seed=seed,
        ),
        score_sheet(
            "v6_faithful",
            field,
            pole_mode="hidden",
            teacher="faithful",
            steps=steps,
            seed=seed,
        ),
        score_sheet(
            "kl_on_midpoint",
            field,
            pole_mode="semantic_kl",
            teacher="pair_odd",
            steps=steps,
            seed=seed,
        ),
        score_sheet(
            "v16_semantic_kl",
            field,
            pole_mode="semantic_kl",
            teacher="faithful",
            steps=steps,
            seed=seed,
        ),
        score_sheet(
            "faithful_guard_e",
            field,
            pole_mode="hidden",
            teacher="faithful_guard_e",
            steps=steps,
            seed=seed,
        ),
        score_sheet(
            "dual_band_poles",
            field,
            pole_mode="dual_band",
            teacher="faithful",
            steps=steps,
            seed=seed,
        ),
        score_sheet(
            "dual_band_guard_e",
            field,
            pole_mode="dual_band",
            teacher="faithful_guard_e",
            steps=steps,
            seed=seed,
        ),
        score_sheet(
            "dual_band_midpoint",
            field,
            pole_mode="dual_band",
            teacher="pair_odd",
            steps=steps,
            seed=seed,
        ),
    ]
    return rows


def leaky_cell(*, steps: int = 400, seed: int = 0) -> list[dict]:
    """Energy-like: unused gender inside ``a``, ê declared, plus invisible mix."""
    field = leaky_field()
    e = field.leak_e()
    rows = [
        score_sheet(
            "v9_hidden",
            field,
            pole_mode="hidden",
            teacher="pair_odd",
            steps=steps,
            seed=seed,
        ),
        score_sheet(
            "v9_hold_e",
            field,
            pole_mode="hidden",
            teacher="pair_odd",
            leak_dir=e,
            hold_weight=LEAK_HOLD_WEIGHT,
            steps=steps,
            seed=seed,
        ),
        score_sheet(
            "v15_pair_odd_sub_e",
            field,
            pole_mode="hidden",
            teacher="pair_odd_sub_e",
            leak_dir=e,
            steps=steps,
            seed=seed,
        ),
        score_sheet(
            "v6_faithful",
            field,
            pole_mode="hidden",
            teacher="faithful",
            steps=steps,
            seed=seed,
        ),
        score_sheet(
            "faithful_sub_e",
            field,
            pole_mode="hidden",
            teacher="faithful_sub_e",
            leak_dir=e,
            steps=steps,
            seed=seed,
        ),
        score_sheet(
            "kl_on_midpoint",
            field,
            pole_mode="semantic_kl",
            teacher="pair_odd",
            steps=steps,
            seed=seed,
        ),
        score_sheet(
            "v16_semantic_kl",
            field,
            pole_mode="semantic_kl",
            teacher="faithful",
            steps=steps,
            seed=seed,
        ),
        score_sheet(
            "v16_semantic_kl_sub_e",
            field,
            pole_mode="semantic_kl",
            teacher="faithful_sub_e",
            leak_dir=e,
            steps=steps,
            seed=seed,
        ),
        score_sheet(
            "faithful_guard_e",
            field,
            pole_mode="hidden",
            teacher="faithful_guard_e",
            leak_dir=e,
            steps=steps,
            seed=seed,
        ),
        score_sheet(
            "dual_band_poles",
            field,
            pole_mode="dual_band",
            teacher="faithful",
            steps=steps,
            seed=seed,
        ),
        score_sheet(
            "dual_band_guard_e",
            field,
            pole_mode="dual_band",
            teacher="faithful_guard_e",
            leak_dir=e,
            steps=steps,
            seed=seed,
        ),
        score_sheet(
            "dual_band_midpoint",
            field,
            pole_mode="dual_band",
            teacher="pair_odd",
            steps=steps,
            seed=seed,
        ),
    ]
    return rows


def live_log_table(*, steps: int = 400, seed: int = 0) -> list[dict]:
    """Reproduce the live log columns, then read the sheet at the same fit.

    The ``odd_even`` student prints an exact 1.00 / −1.00 on a symmetric
    pair-odd teacher, because that teacher has no even part to learn. Live
    prints 0.97 / −0.95 (gender-lm-v4) because the ±1 replies of a real
    stack are not exact mirrors — ``highd``'s ``bend``. Both students walk
    the same distance off the sheet, so the 0.97 / −0.95 pair is not a
    weaker version of the lock, it is the same lock with the live rounding.
    """
    out = []
    for cell, field in (("gender", gender_like_field()), ("energy", leaky_field())):
        for student in STUDENTS:
            row = score_sheet(
                f"{cell}_{student}",
                field,
                pole_mode="hidden",
                teacher="pair_odd",
                student=student,
                steps=steps,
                seed=seed,
            )
            row["cell"] = cell
            out.append(row)
    return out


def teacher_sheet_table(field: SheetField | None = None) -> list[dict]:
    """Where every live target point sits relative to the sheet. No fitting."""
    cell = field if field is not None else leaky_field()
    e = cell.leak_e()
    return [
        teacher_sheet_row("caption", cell, teacher="faithful"),
        teacher_sheet_row("faithful_sub_e", cell, teacher="faithful_sub_e", leak_dir=e),
        teacher_sheet_row("pair_odd_beta1", cell, teacher="pair_odd", common_beta=1.0),
        teacher_sheet_row("pair_odd_beta05", cell, teacher="pair_odd", common_beta=0.5),
        teacher_sheet_row("pair_odd", cell, teacher="pair_odd"),
        teacher_sheet_row("pair_odd_sub_e", cell, teacher="pair_odd_sub_e", leak_dir=e),
    ]


COMMON_GRID = (0.0, 0.2, 0.4, 0.6, 0.8, 0.92, 1.0, 1.11, 1.28, 1.39, 1.6)


def common_sweep(
    grid: tuple[float, ...] = COMMON_GRID,
    *,
    steps: int = 200,
    seed: int = 0,
) -> list[dict]:
    """Off-sheet-ness against ``|c|/|a|``, with the live cos it implies.

    ``common = 0`` is a perfectly odd pair: the midpoint *is* the caption
    and v9 is exactly right. The live band is 0.92 … 1.39.
    """
    out = []
    for share in grid:
        field = leaky_field(common=float(share))
        e = field.leak_e()
        hidden = score_sheet(
            "hidden", field, pole_mode="hidden", teacher="pair_odd", steps=steps, seed=seed
        )
        kl = score_sheet(
            "semantic_kl",
            field,
            pole_mode="semantic_kl",
            teacher="faithful_sub_e",
            leak_dir=e,
            steps=steps,
            seed=seed,
        )
        out.append(
            {
                "common": float(share),
                "probe_cos": probe_cos(share),
                "hidden_on_sheet": hidden["on_sheet"],
                "hidden_garble": hidden["garble"],
                "hidden_argmax_on_sheet": hidden["argmax_on_sheet"],
                "hidden_pair_odd_cos": hidden["pair_odd_cos"],
                "hidden_collapse": hidden["collapse"],
                "hidden_pass": hidden["pass"],
                "kl_on_sheet": kl["on_sheet"],
                "kl_garble": kl["garble"],
                "kl_argmax_on_sheet": kl["argmax_on_sheet"],
                "kl_pair_odd_cos": kl["pair_odd_cos"],
                "kl_collapse": kl["collapse"],
                "kl_pass": kl["pass"],
            }
        )
    return out


def flip_point(sweep: list[dict], key: str = "hidden_argmax_on_sheet") -> float | None:
    """Smallest ``common`` in the sweep where hidden MSE stops being on-sheet."""
    for row in sweep:
        if float(row[key]) < 1.0:
            return float(row["common"])
    return None


def first_above(sweep: list[dict], key: str, threshold: float) -> float | None:
    """Smallest ``common`` where one sweep column crosses a gate.

    Off-sheet behaviour arrives in two stages: the argmax drifts to a token
    that is in the song's vocabulary but not this pole's sheet (singing the
    other verse), and only later does mass leave the vocabulary entirely
    (singing a word the song has no sheet for). ``flip_point`` finds the
    first, this finds the second.
    """
    for row in sweep:
        if float(row[key]) > float(threshold):
            return float(row["common"])
    return None


BETA_GRID = (0.0, 0.25, 0.5, 0.75, 1.0)


def beta_sweep(
    grid: tuple[float, ...] = BETA_GRID,
    *,
    steps: int = 200,
    seed: int = 0,
) -> list[dict]:
    """``--common_beta`` walks the hidden-MSE target from midpoint to caption.

    The flag exists live today and ``lm_hidden_targets`` makes ``β=1``
    exactly ``(pos, neg)``. ``--lm_target v9`` hard-codes β=0 and prints a
    note if you set it, so this ladder is only reachable via
    ``--lm_target symmetric``.
    """
    field = leaky_field()
    out = []
    for beta in grid:
        row = score_sheet(
            f"beta{beta:g}",
            field,
            pole_mode="hidden",
            teacher="pair_odd",
            common_beta=float(beta),
            steps=steps,
            seed=seed,
        )
        out.append(
            {
                "common_beta": float(beta),
                "on_sheet": row["on_sheet"],
                "garble": row["garble"],
                "argmax_on_sheet": row["argmax_on_sheet"],
                "pair_odd_cos": row["pair_odd_cos"],
                "collapse": row["collapse"],
                "leak_tok": row["leak_tok"],
                "null_kept": row["null_kept"],
                "kl_pole": row["kl_pole"],
                "pass": row["pass"],
            }
        )
    return out


def null_space_table(*, steps: int = 200, seed: int = 0) -> list[dict]:
    """What each pole_mode does with content the readout cannot see.

    Hidden MSE has to match the pole on every dim, including the ones no
    token reads. Semantic KL cannot see them, so it leaves them alone.
    Live the readout's row space is small next to the 3584-wide hidden
    state, so this is the bulk of what the two losses disagree about — and
    none of it can change a token either way.
    """
    field = leaky_field()
    e = field.leak_e()
    return [
        score_sheet(
            "hidden_faithful",
            field,
            pole_mode="hidden",
            teacher="faithful",
            steps=steps,
            seed=seed,
        ),
        score_sheet(
            "kl_faithful",
            field,
            pole_mode="semantic_kl",
            teacher="faithful",
            steps=steps,
            seed=seed,
        ),
        score_sheet(
            "kl_faithful_sub_e",
            field,
            pole_mode="semantic_kl",
            teacher="faithful_sub_e",
            leak_dir=e,
            steps=steps,
            seed=seed,
        ),
        score_sheet(
            "kl_null_faithful",
            field,
            pole_mode="semantic_kl_null",
            teacher="faithful",
            steps=steps,
            seed=seed,
        ),
    ]


def live_probe_table() -> list[dict]:
    """The live v4 probe cos table read as a common-component size."""
    return [
        {
            "axis": axis,
            "probe_cos": cos,
            "common_share": common_share(cos),
            "warned": cos > LIVE_COLLAPSE_WARN,
        }
        for axis, cos in sorted(LIVE_PROBE_COS.items(), key=lambda kv: kv[1])
    ]


def floatable(row: dict) -> dict:
    """JSON-safe subset: drop tensors and nested history."""
    out = {}
    for key, value in row.items():
        if isinstance(value, (int, float, str, bool)) or value is None:
            out[key] = value
        elif isinstance(value, dict) and all(isinstance(v, str) for v in value.values()):
            out[key] = value
    return out
