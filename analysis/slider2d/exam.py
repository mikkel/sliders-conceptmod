"""Live-exam fields the leftover unused-gender sheet cannot see.

The 2026-08-25 Music 3 listens failed the #23 leftover sheet:

- energy-lm-v16 (``semantic_kl`` + ``faithful_sub_e`` on energy-v4) garbled.
  energy-v4 poles are two tracks. Leftover ê *is* most of that gap
  (pop-punk / BPM 168 vs ambient lullaby / BPM 52). ``faithful_sub_e``
  is ``mid ± â`` with ``mid = ½(h++h−)`` — a third song.
- energy-lm-v18 (``semantic_kl`` + ``faithful`` on the same pair) sang.
  Genre / BPM *are* the poles. Unused-gender leak inside a same-song
  pair is a different cell.
- gender-lm-v16 (same recipe as energy-v18, on a close same-song pair)
  garbled. One-token KL at ``<|audio_start|>`` looked done (loss 0.009,
  p% / n% 0.52 / 0.78) while the 3584-wide hidden never arrived.

This module adds the three axes the leftover sheet was missing:

1. A **divergent two-track** field where ê is the bulk of ``pos−neg``.
2. A **close same-song** field whose first-token policies nearly match
   and whose continuation needs the hidden gender dim.
3. **Rollout** (teacher-forced step-1) plus **hidden-far while KL-small**
   (live p% / n%).

Pair-odd cos and ±1 collapse are logged, never scored. CPU only.
Does not change the live trainer default (``v9`` / ``hidden``).
"""

from __future__ import annotations

from dataclasses import dataclass, field as dc_field

import torch

from analysis.slider2d.field import cosine
from analysis.slider2d.highd import BEND_GENDER
from analysis.slider2d.sheet import (
    ARGMAX_LOCK,
    GARBLE_MAX,
    HIDDEN_FAR,
    KL_SMALL,
    LIVE_PROBE_COS,
    OFF_SHEET_TOKENS,
    SHEET_LOCK,
    SWING_FLOOR,
    Readout,
    _null_kept,
    common_share,
    fit_sheet,
    hidden_sheet_report,
    hold_direction,
    teacher_points,
)
from conceptmod.textsliders.slider_targets import lm_unit


# Live energy-v4 leftover ê *is* the two tracks. The declared slider
# shorts are a smaller leftover after ê_⊥ is removed.
DIVERGENT_TRACK = 1.20
DIVERGENT_ENERGY = 0.25
DIVERGENT_NULL = 0.20
# Logged pair cos on energy-v4 sat around −0.11 … +0.14.
DIVERGENT_PROBE = 0.03

# Close pair: a small first-token cue plus a larger continuation
# gender dim the audio-start head cannot see. Live gender-v16 matched
# the cue (KL 0.009) and left the rest of the 3584-wide state behind.
CLOSE_SEEN = 0.15
CLOSE_HIDDEN = 1.00
CLOSE_CONT_GENDER = 1.20

TRACK_TOKENS = ("punk", "lullaby")
BLEND_TOKENS = ("blend",)
GENDER_TOKENS = ("female", "male")
START_TOKENS = ("start",)
VERSE_TOKENS = ("verse_f", "verse_m")


@dataclass(frozen=True)
class ExamField:
    """Hidden field for the live exam cells.

    Dim layout, all orthonormal::

        0   û   declared slider (energy leftover, or gender)
        1   ê   leftover track (divergent) or unused (close: 0)
        2   ŝ   shared specified-song component
        3 .. 3+rows-1   written lyric
        then null dims the first-token readout cannot see

    ``kind="divergent"``: ê is the two tracks and most of ``||a||``.
    ``kind="close"``: ê is off, û is the singer, and the first-token
    readout barely loads û — live one-token KL at audio-start.
    """

    kind: str = "divergent"
    rows: int = 3
    slider: float = DIVERGENT_ENERGY
    leak: float = DIVERGENT_TRACK
    null: float = DIVERGENT_NULL
    common: float = dc_field(default_factory=lambda: common_share(DIVERGENT_PROBE))
    lyric: float = 1.00
    null_dims: int = 2
    row_scales: tuple[float, ...] = (1.0, 0.85, 1.15)
    gain: float = 2.5
    sheet_p: float = 0.90
    bend: float = BEND_GENDER
    seed: int = 0
    first_seen: float = 1.0
    cont_seen: float = 0.0

    def __post_init__(self) -> None:
        if self.kind not in ("divergent", "close"):
            raise ValueError(f"kind must be divergent or close, got {self.kind!r}")
        if int(self.rows) < 1:
            raise ValueError(f"rows must be ≥ 1, got {self.rows!r}")
        if len(self.row_scales) < int(self.rows):
            raise ValueError("row_scales must cover every row")

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

    def hidden_u(self) -> torch.Tensor:
        """Continuation-only gender. First-token KL cannot see this dim."""
        return self._basis(1)

    def odd(self, row: int = 0) -> torch.Tensor:
        scale = float(self.row_scales[int(row)])
        if self.kind == "close":
            # Most of the singer lives off the audio-start band.
            return scale * (
                float(self.first_seen) * self.short_u()
                + float(self.slider) * self.hidden_u()
            )
        return scale * (
            float(self.slider) * self.short_u()
            + float(self.leak) * self.leak_e()
            + float(self.null) * self.null_mix()
        )

    def common_vec(self, row: int = 0) -> torch.Tensor:
        return float(self.common) * float(self.odd(row).norm()) * self.sheet_dir()

    def poles(self, row: int = 0) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        neu = float(self.lyric) * self.lyric_dir(row)
        a = self.odd(row)
        c = self.common_vec(row)
        return neu + a + c, neu - a + c, neu

    def probe_cos(self, row: int = 0) -> float:
        pos, neg, neu = self.poles(row)
        return cosine(pos - neu, neg - neu)

    def e_share(self, row: int = 0) -> float:
        """How much of ``||a||`` is leftover ê — the two-track fraction."""
        a = self.odd(row)
        return float(abs(a @ self.leak_e()) / a.norm().clamp_min(1e-8))

    def concept_tokens(self) -> tuple[str, str]:
        if self.kind == "divergent":
            return TRACK_TOKENS
        return GENDER_TOKENS

    def leak_tokens(self) -> tuple[str, str] | None:
        # Genre/BPM *are* the poles on the divergent cell. Unused gender
        # inside a same-song pair is the leftover sheet, not this one.
        return None

    def readout(self) -> Readout:
        """First-token head — the live semantic-KL band at audio-start."""
        if self.kind == "divergent":
            return self._divergent_readout()
        return self._close_readout(step=0)

    def cont_readout(self) -> Readout | None:
        """Step-1 / teacher-forced continuation. ``None`` if unused."""
        if self.kind == "close" and float(self.cont_seen) > 0.0:
            return self._close_readout(step=1)
        return None

    def _divergent_readout(self) -> Readout:
        tokens = (
            list(TRACK_TOKENS)
            + list(BLEND_TOKENS)
            + list(OFF_SHEET_TOKENS)
            + [f"lyric{r}" for r in range(int(self.rows))]
        )
        weight = torch.zeros(len(tokens), self.dim)
        # Tracks need ê *and* ŝ. The blend is ŝ alone, louder, so it
        # wins at mid (ê ≈ 0) and loses at either real pole.
        weight[0, 1], weight[0, 2] = 1.20, 1.00  # punk
        weight[1, 1], weight[1, 2] = -1.20, 1.00  # lullaby
        weight[2, 2] = 1.60  # blend / third song
        weight[3, 0], weight[3, 2] = 1.30, -1.50  # garble_hi
        weight[4, 0], weight[4, 2] = -1.30, -1.50  # garble_lo
        for r in range(int(self.rows)):
            weight[5 + r, 3 + r] = 1.20
        return Readout(tuple(tokens), weight, gain=float(self.gain))

    def _close_readout(self, step: int) -> Readout:
        tokens = (
            list(START_TOKENS)
            + list(GENDER_TOKENS)
            + list(VERSE_TOKENS)
            + list(OFF_SHEET_TOKENS)
            + [f"lyric{r}" for r in range(int(self.rows))]
        )
        weight = torch.zeros(len(tokens), self.dim)
        if int(step) == 0:
            # First token: shared start / lyric. A weak load on dim 0
            # only — dim 1 (the bulk of the singer) is the null space
            # of this head, the way most of a 3584-wide state is.
            weight[0, 2] = 1.40  # start ← ŝ
            weight[1, 0], weight[1, 2] = 0.25, 0.70  # female (seen cue)
            weight[2, 0], weight[2, 2] = -0.25, 0.70  # male
            weight[3, 0], weight[3, 2] = 0.05, 0.15
            weight[4, 0], weight[4, 2] = -0.05, 0.15
            weight[5, 0], weight[5, 2] = 1.30, -1.50
            weight[6, 0], weight[6, 2] = -1.30, -1.50
        else:
            # Continuation reads dim 1 (hidden singer). Copying only
            # the first-token cue leaves û_hidden at 0 and garbles.
            weight[0, 2] = 0.25
            weight[1, 1], weight[1, 2] = 0.20, 0.25
            weight[2, 1], weight[2, 2] = -0.20, 0.25
            weight[3, 1], weight[3, 2] = float(self.cont_seen), 0.40  # verse_f
            weight[4, 1], weight[4, 2] = -float(self.cont_seen), 0.40  # verse_m
            weight[5, 2] = 1.00  # garble wins when hidden û ≈ 0
            weight[6, 2] = 1.00
        for r in range(int(self.rows)):
            # Written lyric is a first-token caption fact, not the verse.
            if int(step) == 0:
                weight[7 + r, 3 + r] = 1.20
        return Readout(tuple(tokens), weight, gain=float(self.gain))

    def gate_matrix(self) -> torch.Tensor:
        gen = torch.Generator().manual_seed(int(self.seed) + 5)
        q, _r = torch.linalg.qr(torch.randn(self.dim, self.dim, generator=gen))
        return q


def divergent_field(**kwargs) -> ExamField:
    """Two tracks. Leftover ê is most of the pole gap."""
    base = {
        "kind": "divergent",
        "slider": DIVERGENT_ENERGY,
        "leak": DIVERGENT_TRACK,
        "null": DIVERGENT_NULL,
        "null_dims": 2,
        "common": common_share(DIVERGENT_PROBE),
        "first_seen": 1.0,
        "cont_seen": 0.0,
    }
    base.update(kwargs)
    return ExamField(**base)


def close_pair_field(**kwargs) -> ExamField:
    """Same song, male vs female. First token is close; continuation is not."""
    base = {
        "kind": "close",
        "slider": CLOSE_HIDDEN,
        "leak": 0.0,
        "null": 0.0,
        "null_dims": 0,
        "common": common_share(LIVE_PROBE_COS["gender"]),
        "first_seen": CLOSE_SEEN,
        "cont_seen": CLOSE_CONT_GENDER,
    }
    base.update(kwargs)
    return ExamField(**base)


def _report(
    field: ExamField,
    plus: list[torch.Tensor],
    minus: list[torch.Tensor],
    *,
    readout: Readout | None = None,
    include_written: bool = True,
) -> dict:
    head = readout if readout is not None else field.readout()
    leak = field.leak_tokens()
    return hidden_sheet_report(
        field,  # type: ignore[arg-type]
        plus,
        minus,
        readout=head,
        concept_tokens=field.concept_tokens(),
        leak_tokens=leak,
        include_written=include_written,
    )


def rollout_report(
    field: ExamField,
    plus: list[torch.Tensor],
    minus: list[torch.Tensor],
) -> dict:
    """Teacher-forced step-1 sheet. Missing û garbles the verse."""
    cont = field.cont_readout()
    if cont is None:
        return {
            "rollout_on_sheet": None,
            "rollout_on_sheet_kept": None,
            "rollout_garble": None,
            "rollout_argmax_on_sheet": None,
            "rollout_says": None,
            "rollout_kl_pole": None,
        }
    ceiling = _report(field, *_pole_lists(field), readout=cont, include_written=False)
    row = _report(field, plus, minus, readout=cont, include_written=False)
    kept = row["on_sheet"] / (ceiling["on_sheet"] + 1e-8)
    return {
        "rollout_on_sheet": row["on_sheet"],
        "rollout_on_sheet_kept": kept,
        "rollout_garble": row["garble"],
        "rollout_argmax_on_sheet": row["argmax_on_sheet"],
        "rollout_says": row["says"],
        "rollout_kl_pole": row["kl_pole"],
        "rollout_swing_kept": row["concept_swing"] / (abs(ceiling["concept_swing"]) + 1e-8),
        "teacher_rollout_on_sheet": ceiling["on_sheet"],
        "teacher_rollout_garble": ceiling["garble"],
    }


def _pole_lists(field: ExamField) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    plus, minus = [], []
    for row in range(int(field.rows)):
        pos, neg, _neu = field.poles(row)
        plus.append(pos)
        minus.append(neg)
    return plus, minus


def exam_verdicts(row: dict, *, kind: str) -> dict[str, str]:
    """Exam gates. Pair-odd cos / collapse / unused-gender leak are not here.

    Divergent: the sheet of the *tracks*. Genre/BPM movement is the
    concept, not a leak fail. Blend / off-caption is the fail.
    Close: one-token on-sheet is not enough. Rollout plus hidden-far
    while KL-small is the gender-v16 flag.
    """
    out = {
        "sheet": "right" if row["on_sheet_kept"] >= SHEET_LOCK else "needs_help",
        "garble": "right" if row["garble"] <= GARBLE_MAX else "needs_help",
        "argmax": "right" if row["argmax_on_sheet"] >= ARGMAX_LOCK else "needs_help",
        "slider": "right" if row["swing_kept"] >= SWING_FLOOR else "needs_help",
    }
    if kind == "divergent":
        out["caption"] = "right" if float(row.get("off_caption", 0.0)) <= 0.35 else "needs_help"
        out["blend"] = "right" if "blend" not in str(row.get("says", "")).split(",") else "needs_help"
    else:
        rollout_kept = row.get("rollout_on_sheet_kept")
        rollout_garble = row.get("rollout_garble")
        rollout_argmax = row.get("rollout_argmax_on_sheet")
        if rollout_kept is not None:
            out["rollout"] = "right" if float(rollout_kept) >= SHEET_LOCK else "needs_help"
        if rollout_garble is not None:
            out["rollout_garble"] = "right" if float(rollout_garble) <= GARBLE_MAX else "needs_help"
        if rollout_argmax is not None:
            out["rollout_argmax"] = "right" if float(rollout_argmax) >= ARGMAX_LOCK else "needs_help"
        out["hidden"] = "needs_help" if row.get("kl_small_hidden_far") else "right"
    return out


def score_exam(
    name: str,
    field: ExamField,
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
    """Fit one recipe on an exam field and score first-token + rollout."""
    residual = fit_sheet(
        field,  # type: ignore[arg-type]
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
    row = _report(field, plus, minus, readout=head)
    row.update(rollout_report(field, plus, minus))

    a = field.odd(0)
    pos, _neg, neu = field.poles(0)
    ceiling = _report(field, *_pole_lists(field), readout=head)
    held = hold_direction(field, leak_dir)  # type: ignore[arg-type]
    t_plus, t_minus = teacher_points(
        field,  # type: ignore[arg-type]
        0,
        teacher=teacher,
        leak_dir=leak_dir,
        common_beta=common_beta,
    )
    pperc_rows: list[float] = []
    for r in range(int(field.rows)):
        tgt_plus, tgt_minus = teacher_points(
            field,  # type: ignore[arg-type]
            r,
            teacher=teacher,
            leak_dir=leak_dir,
            common_beta=common_beta,
        )
        neu_r = field.poles(r)[2]
        pperc_rows.append(
            float((neu_r + d_plus - tgt_plus).norm() / (tgt_plus - neu_r).norm().clamp_min(1e-8))
        )
        pperc_rows.append(
            float((neu_r + d_minus - tgt_minus).norm() / (tgt_minus - neu_r).norm().clamp_min(1e-8))
        )
    pperc = sum(pperc_rows) / len(pperc_rows)
    off_caption = float((t_plus - pos).norm() / a.norm().clamp_min(1e-8))
    row.update(
        {
            "name": name,
            "kind": field.kind,
            "pole_mode": pole_mode,
            "teacher": teacher,
            "student": student,
            "hold_weight": float(hold_weight) if held is not None else 0.0,
            "common": float(field.common),
            "common_beta": float(common_beta),
            "probe_cos": field.probe_cos(0),
            "e_share": field.e_share(0),
            "pair_odd_cos": cosine(d_plus, a),
            "collapse": cosine(d_plus, d_minus),
            "pole_cos": cosine(d_plus, pos - neu),
            "sheet_dir_kept": float(d_plus @ field.sheet_dir())
            / float(field.common_vec(0).norm() + 1e-8),
            "null_kept": _null_kept(field, d_plus),  # type: ignore[arg-type]
            "swing_kept": row["concept_swing"] / (abs(ceiling["concept_swing"]) + 1e-8),
            "on_sheet_kept": row["on_sheet"] / (ceiling["on_sheet"] + 1e-8),
            "teacher_on_sheet": ceiling["on_sheet"],
            "teacher_garble": ceiling["garble"],
            "off_caption": off_caption,
            "pperc": pperc,
            "kl_small_hidden_far": bool(row["kl_pole"] <= KL_SMALL and pperc >= HIDDEN_FAR),
        }
    )
    row["axis"] = exam_verdicts(row, kind=field.kind)
    row["pass"] = all(v == "right" for v in row["axis"].values())
    row["flag"] = "kl_small_hidden_far" if row["kl_small_hidden_far"] else ""
    return row


def teacher_exam_row(
    name: str,
    field: ExamField,
    *,
    teacher: str = "pair_odd",
    leak_dir: torch.Tensor | None = None,
    common_beta: float = 0.0,
) -> dict:
    """Where the target point sits. No optimizer."""
    plus, minus = [], []
    for row in range(int(field.rows)):
        t_plus, t_minus = teacher_points(
            field,  # type: ignore[arg-type]
            row,
            teacher=teacher,
            leak_dir=leak_dir,
            common_beta=common_beta,
        )
        plus.append(t_plus)
        minus.append(t_minus)
    report = _report(field, plus, minus)
    report.update(rollout_report(field, plus, minus))
    a = field.odd(0)
    pos, _neg, neu = field.poles(0)
    report.update(
        {
            "name": name,
            "kind": field.kind,
            "teacher": teacher,
            "common_beta": float(common_beta),
            "common": float(field.common),
            "probe_cos": field.probe_cos(0),
            "e_share": field.e_share(0),
            "is_caption": teacher.strip().lower() == "faithful" or float(common_beta) == 1.0,
            "off_caption": float((plus[0] - pos).norm() / a.norm().clamp_min(1e-8)),
            "sheet_dir_kept": float((plus[0] - neu) @ field.sheet_dir())
            / float(field.common_vec(0).norm() + 1e-8),
        }
    )
    report["axis"] = exam_verdicts(
        {
            **report,
            "on_sheet_kept": report["on_sheet"]
            / (_report(field, *_pole_lists(field))["on_sheet"] + 1e-8),
            "swing_kept": 1.0,
            "kl_small_hidden_far": False,
        },
        kind=field.kind,
    )
    return report


def divergent_cell(*, steps: int = 400, seed: int = 0) -> list[dict]:
    """Energy-v4 stand-in: two tracks, leftover ê is the gap."""
    field = divergent_field()
    e = field.leak_e()
    return [
        score_exam(
            "v9_hidden",
            field,
            pole_mode="hidden",
            teacher="pair_odd",
            steps=steps,
            seed=seed,
        ),
        score_exam(
            "v6_faithful",
            field,
            pole_mode="hidden",
            teacher="faithful",
            steps=steps,
            seed=seed,
        ),
        score_exam(
            "faithful_sub_e",
            field,
            pole_mode="hidden",
            teacher="faithful_sub_e",
            leak_dir=e,
            steps=steps,
            seed=seed,
        ),
        score_exam(
            "v16_semantic_kl",
            field,
            pole_mode="semantic_kl",
            teacher="faithful",
            steps=steps,
            seed=seed,
        ),
        score_exam(
            "v16_semantic_kl_sub_e",
            field,
            pole_mode="semantic_kl",
            teacher="faithful_sub_e",
            leak_dir=e,
            steps=steps,
            seed=seed,
        ),
        score_exam(
            "kl_on_midpoint",
            field,
            pole_mode="semantic_kl",
            teacher="pair_odd",
            steps=steps,
            seed=seed,
        ),
        score_exam(
            "v15_pair_odd_sub_e",
            field,
            pole_mode="hidden",
            teacher="pair_odd_sub_e",
            leak_dir=e,
            steps=steps,
            seed=seed,
        ),
    ]


def close_cell(*, steps: int = 400, seed: int = 0) -> list[dict]:
    """Gender-v4 stand-in: close same-song pair, first-token KL vs rollout."""
    field = close_pair_field()
    return [
        score_exam(
            "v9_hidden",
            field,
            pole_mode="hidden",
            teacher="pair_odd",
            steps=steps,
            seed=seed,
        ),
        score_exam(
            "hidden_beta1",
            field,
            pole_mode="hidden",
            teacher="pair_odd",
            common_beta=1.0,
            steps=steps,
            seed=seed,
        ),
        score_exam(
            "v6_faithful",
            field,
            pole_mode="hidden",
            teacher="faithful",
            steps=steps,
            seed=seed,
        ),
        score_exam(
            "kl_on_midpoint",
            field,
            pole_mode="semantic_kl",
            teacher="pair_odd",
            steps=steps,
            seed=seed,
        ),
        score_exam(
            "v16_semantic_kl",
            field,
            pole_mode="semantic_kl",
            teacher="faithful",
            steps=steps,
            seed=seed,
        ),
    ]


def divergent_teacher_table() -> list[dict]:
    field = divergent_field()
    e = field.leak_e()
    return [
        teacher_exam_row("caption", field, teacher="faithful"),
        teacher_exam_row("faithful_sub_e", field, teacher="faithful_sub_e", leak_dir=e),
        teacher_exam_row("pair_odd", field, teacher="pair_odd"),
        teacher_exam_row("pair_odd_sub_e", field, teacher="pair_odd_sub_e", leak_dir=e),
    ]


# Live exam mapping. One row per listen, not per leftover-sheet recipe.
LIVE_EXAM = (
    {
        "live": "energy-lm-v16",
        "recipe": "v16_semantic_kl_sub_e",
        "scoreboard_id": "semantic_kl_sub_e",
        "cell": "divergent",
        "pole_mode": "semantic_kl",
        "teacher": "faithful_sub_e",
        "listen": "FAIL",
        "why": "mid ± â on two tracks is a third song; leftover ê was the poles",
    },
    {
        "live": "energy-lm-v18",
        "recipe": "v16_semantic_kl",
        "scoreboard_id": "semantic_kl_poles",
        "cell": "divergent",
        "pole_mode": "semantic_kl",
        "teacher": "faithful",
        "listen": "PASS",
        "why": "raw poles; genre/BPM ride *is* the slider",
    },
    {
        "live": "gender-lm-v16",
        "recipe": "v16_semantic_kl",
        "scoreboard_id": "semantic_kl_poles",
        "cell": "close",
        "pole_mode": "semantic_kl",
        "teacher": "faithful",
        "listen": "FAIL",
        "why": "one-token KL small, hidden residual high, rollout garbles",
    },
)


def live_exam_rows(*, steps: int = 400, seed: int = 0) -> list[dict]:
    """The three listens, scored on the cell that corresponds to each."""
    divergent = {row["name"]: row for row in divergent_cell(steps=steps, seed=seed)}
    close = {row["name"]: row for row in close_cell(steps=steps, seed=seed)}
    out = []
    for spec in LIVE_EXAM:
        src = divergent if spec["cell"] == "divergent" else close
        row = dict(src[spec["recipe"]])
        predicted = "PASS" if row["pass"] else "FAIL"
        if (not row["pass"]) and row.get("kl_small_hidden_far"):
            predicted = "FAIL/flagged"
        out.append(
            {
                **spec,
                "predicted": predicted,
                "pass": row["pass"],
                "flag": row.get("flag", ""),
                "on_sheet_kept": row["on_sheet_kept"],
                "garble": row["garble"],
                "argmax_on_sheet": row["argmax_on_sheet"],
                "says": row["says"],
                "off_caption": row.get("off_caption"),
                "e_share": row.get("e_share"),
                "pperc": row.get("pperc"),
                "kl_pole": row.get("kl_pole"),
                "kl_small_hidden_far": row.get("kl_small_hidden_far"),
                "rollout_on_sheet_kept": row.get("rollout_on_sheet_kept"),
                "rollout_garble": row.get("rollout_garble"),
                "rollout_says": row.get("rollout_says"),
                "pair_odd_cos": row.get("pair_odd_cos"),
                "collapse": row.get("collapse"),
                "listen_match": predicted.split("/")[0] == spec["listen"],
            }
        )
    return out
