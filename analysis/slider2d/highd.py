"""High-D hold-ê: what the orthonormal 2-D cells cannot show.

The 2-D overlap / rich / faithful cells put the declared short û *and*
the whole intended concept on one axis, and give the student a free
per-coordinate residual. Three live facts fall out of that fixture:

1. ``λ`` is not a portable number. ``F.mse_loss`` averages over the
   hidden width, ``lm_axis_hold`` does not, so the ê component of the
   fit settles at ``a_ê / (1 + λ·D/2)``. λ=8 on a 2-D cell is a
   stiffness of 8; λ=8 on a Music 3 hidden state is a stiffness of
   ``4·D``. Once ``λ·D/2 ≫ 1`` every λ in {0.3, 1, 8} lands on the same
   residual — λ only buys stiffness, not less leak.
2. Short û is a probe, not the concept. Live ``a`` keeps 37% of itself
   on the short caption axis (energy-v14 gate log) and spends the rest
   on loudness wording the short pair misses *plus* unused mix / BPM /
   genre. Holding a pole-synonym ê eats the off-û loudness with the
   leak, so cos(d+, û) *rises* while cos(d+, intended) falls. On a
   2-D field those two cosines are the same number.
3. Trainer ``c+`` is cos(d+, full pair-odd). A working hold has to
   lower it: the ceiling is ``√(1−p²)`` with ``p = |â·ê̂_⊥|``. Gender's
   0.97 is "I copied pair-odd", not "the slider locked".

What high-D still cannot show is the ±1 polarity break. With a
symmetric pair-odd teacher and any residual that is linear in the
slider scale, the pole MSE splits into ``|w_odd − a|² + |w_even|²``
and the hold splits the same way, so ``w_even`` stays 0 and
``cos(d+, d−) = −1`` for every ê, λ and D. Live ±1 is two forward
passes with the LoRA multiplier flipped, through attention softmax and
SwiGLU MLPs; none of that is odd in the multiplier, so the two replies
are only approximately mirrors. ``bend`` is the size of the non-mirror
part and it is a property of the net, not of ê. Gender-v14 (collapse
−0.95) implies bend ≈ 0.16; energy-v14 (collapse +0.18) implies
bend ≈ 1.2, i.e. an even reply larger than the odd one.

CPU only. No Hub, no GPU, no Music 3 weights. Does not change the live
trainer default.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import torch

from analysis.slider2d.energy import CONCEPT_SCALE
from analysis.slider2d.field import cosine
from conceptmod.textsliders.slider_targets import (
    LEAK_HOLD_WEIGHT,
    leftover_bipolar,
    lm_axis_hold,
    lm_hidden_targets,
    lm_hold_dir,
    lm_odd_align,
    lm_slider_loss,
    lm_unit,
)


# Live energy-v14 log (hold ê_⊥û, λ=8, opposite-energy leak captions).
LIVE_GATE_ALIGN = 0.37
LIVE_C_PLUS = 0.31
LIVE_COLLAPSE = 0.18
# Live gender-v14 (no declared ê, so hold = 0).
LIVE_GENDER_C_PLUS = 0.97
LIVE_GENDER_COLLAPSE = -0.95
LIVE_GENDER_LOSS = 0.009
# Same-loudness rewrite of the leak captions: loss 278 by step 13.
LIVE_SPIKE_LOSS = 278.0

# ±1 response asymmetry implied by the two live collapse logs:
# bend = √((1+collapse)/(1−collapse)) for a pure rotation.
BEND_GENDER = 0.16
BEND_ENERGY = 1.20

# Live-like split of the pair-odd: 37% on the short caption axis, the
# rest loudness wording the short pair misses plus unused mix/BPM/genre.
DEFAULT_CONTENT = 0.62
DEFAULT_LEFTOVER = 0.69
DEFAULT_DIM = 8

HOLD_LAMBDAS = (0.0, 0.3, 1.0, 8.0)
DIM_GRID = (2, 4, 8, 16, 64, 1024)
BEND_GRID = (0.0, BEND_GENDER, 0.5, 1.0, BEND_ENERGY, 1.5)
CONTENT_GRID = (0.0, 0.25, 0.5, 0.75, 0.9, 1.0)

SLIDER_LOCK = 0.90
LEAK_LOCK = 0.20
BIPOLAR_MAX = -0.85
V12_LOOK = 0.90
# c+ this far under the closed form is not the hold's doing.
HOLD_EXPLAINS_TOL = 0.06


def hold_shrink(hold_weight: float, dim: int) -> float:
    """Fraction of the teacher's ê component the fit keeps.

    ``F.mse_loss`` divides by ``dim``; ``lm_axis_hold`` does not. The ê
    stationarity condition of ``2·MSE/dim + λ·(w·ê)²`` is therefore
    ``(2/dim)(s − t) + λ s = 0``, i.e. ``s = t / (1 + λ·dim/2)``.
    ``analysis.slider2d.faithful.hold_e_shrink`` is this at ``dim=2``.
    """
    lam = float(hold_weight)
    if lam < 0.0:
        raise ValueError(f"hold_weight must be ≥ 0, got {hold_weight!r}")
    if int(dim) < 1:
        raise ValueError(f"dim must be ≥ 1, got {dim!r}")
    return 1.0 / (1.0 + lam * float(dim) / 2.0)


def lambda_eff(hold_weight: float, dim: int) -> float:
    """Stiffness the hold actually applies: ``λ·D/2``."""
    return float(hold_weight) * float(dim) / 2.0


def hold_predictions(
    cover: float,
    hold_weight: float,
    dim: int,
    *,
    shrink: float | None = None,
) -> dict[str, float]:
    """Closed form for a linear student, symmetric teacher, one held axis.

    ``cover = p = |â·ê̂_⊥|`` is the share of the pair-odd that the held
    direction covers. With ``k = hold_shrink(λ, D)`` the fit is
    ``d = a − (1−k)(a·ê̂)ê̂``, so

        c+    = (1 − (1−k)p²) / √(1 − p² + k²p²)
        perc  = (1−k)·p
        c+_if_held = √(1−p²)        # k → 0, hold has done its job

    ``shrink=0`` is the λ→∞ limit, which is also what subtracting the
    same axis from the teacher does in one step.
    """
    p = float(cover)
    if not 0.0 <= p <= 1.0:
        raise ValueError(f"cover must be in [0, 1], got {cover!r}")
    k = hold_shrink(hold_weight, dim) if shrink is None else float(shrink)
    kept = 1.0 - (1.0 - k) * p * p
    norm = max(1.0 - p * p + k * k * p * p, 0.0) ** 0.5
    return {
        "shrink": k,
        "lambda_eff": lambda_eff(hold_weight, dim),
        "c_plus_predicted": kept / max(norm, 1e-8),
        "perc_predicted": (1.0 - k) * p,
        "c_plus_if_held": max(1.0 - p * p, 0.0) ** 0.5,
    }


def hold_spike(a_along_e: float, hold_weight: float) -> float:
    """Hold term while the fit still carries the teacher's ê component.

    ``λ·(a·ê̂)²``. This is the un-normalized half of the loss and the
    reason a live run can print 278 while the 2-D cell prints 0.9.
    """
    return float(hold_weight) * float(a_along_e) ** 2


@dataclass(frozen=True)
class HighDLeakField:
    """Already-odd poles in R^dim: short û, off-û concept, unused leftover.

    ``align`` is the live gate log ``|a·û|/||a||``. ``content`` is the
    loudness wording the short pair misses — intended, not leak.
    ``leftover`` is unused mix / BPM / genre / syntax, spread over the
    remaining dims. ``bend`` is the ±1 response asymmetry of the
    LoRA'd stack (0 = perfectly odd, as every existing 2-D cell
    assumes).
    """

    dim: int = DEFAULT_DIM
    align: float = LIVE_GATE_ALIGN
    content: float = DEFAULT_CONTENT
    leftover: float = DEFAULT_LEFTOVER
    scale: float = CONCEPT_SCALE
    bend: float = 0.0
    bend_parallel: float = 0.0
    seed: int = 0

    def __post_init__(self) -> None:
        if int(self.dim) < 2:
            raise ValueError(f"dim must be ≥ 2, got {self.dim!r}")
        if not 0.0 < float(self.align) <= 1.0:
            raise ValueError(f"align must be in (0, 1], got {self.align!r}")
        if float(self.content) > 0.0 and int(self.dim) < 3:
            raise ValueError("off-û content needs dim ≥ 3")
        if float(self.leftover) > 0.0 and self._n_leftover() < 1:
            raise ValueError("leftover needs a spare dim")

    def _has_content(self) -> bool:
        return float(self.content) > 0.0

    def _n_leftover(self) -> int:
        return int(self.dim) - (2 if self._has_content() else 1)

    def short_u(self) -> torch.Tensor:
        out = torch.zeros(int(self.dim))
        out[0] = 1.0
        return out

    def content_dir(self) -> torch.Tensor | None:
        if not self._has_content():
            return None
        out = torch.zeros(int(self.dim))
        out[1] = 1.0
        return out

    def leftover_dirs(self) -> list[torch.Tensor]:
        start = 2 if self._has_content() else 1
        dirs = []
        for i in range(start, int(self.dim)):
            out = torch.zeros(int(self.dim))
            out[i] = 1.0
            dirs.append(out)
        return dirs

    def leftover_mix(self) -> torch.Tensor:
        """Unit direction of the leftover the poles actually carry."""
        dirs = self.leftover_dirs()
        if not dirs or float(self.leftover) <= 0.0:
            return torch.zeros(int(self.dim))
        gen = torch.Generator().manual_seed(int(self.seed))
        weights = torch.randn(len(dirs), generator=gen).abs() + 0.3
        acc = torch.zeros(int(self.dim))
        for weight, direction in zip(weights, dirs):
            acc = acc + float(weight) * direction
        return lm_unit(acc)

    def odd(self) -> torch.Tensor:
        """Pair-odd teacher ``a`` with ``|a·û|/||a|| == align``."""
        raw = float(self.align) * self.short_u()
        content = self.content_dir()
        if content is not None:
            raw = raw + float(self.content) * content
        raw = raw + float(self.leftover) * self.leftover_mix()
        return float(self.scale) * lm_unit(raw)

    def intended(self) -> torch.Tensor:
        """The concept: short-û loudness plus the loudness wording off û."""
        raw = float(self.align) * self.short_u()
        content = self.content_dir()
        if content is not None:
            raw = raw + float(self.content) * content
        return lm_unit(raw)

    def poles(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        a = self.odd()
        neu = torch.zeros(int(self.dim))
        return a.clone(), (-a).clone(), neu

    def gate_align(self) -> float:
        pos, neg, _neu = self.poles()
        return float(lm_odd_align(pos, neg, self.short_u()))

    def gate_matrix(self) -> torch.Tensor:
        """Fixed orthogonal ``G``: ``||G w|| = ||w||``, generically ⊥ w."""
        gen = torch.Generator().manual_seed(int(self.seed) + 5)
        q, _r = torch.linalg.qr(torch.randn(int(self.dim), int(self.dim), generator=gen))
        return q


def leak_axis(
    field: HighDLeakField,
    *,
    on_u: float,
    on_content: float,
    on_leftover: float,
    leftover_match: float = 1.0,
    noise: float = 0.0,
) -> torch.Tensor:
    """Declared ê written as what its captions actually say.

    ``on_u`` is the part the short slider pair already says (loud vs
    quiet). ``on_content`` is loudness the short pair misses — density,
    "slammed", "168" heard as loud. ``on_leftover`` is genre / BPM
    wording / mix adjectives / syntax that the concept does not own.
    A pole synonym is content-heavy; leftover-only ê is the opposite.

    ``leftover_match`` is how well ê's unused part lines up with the
    unused mix the poles actually carry — one declared caption pair
    cannot name four kinds of leftover at once. ``noise`` rotates
    ``ê_⊥`` into junk, which is what a tiny messy leftover does.
    """
    u = field.short_u()
    content = field.content_dir()
    if float(on_content) != 0.0 and content is None:
        raise ValueError("this field has no off-û content dim")
    mix = field.leftover_mix()
    match = float(leftover_match)
    if not -1.0 <= match <= 1.0:
        raise ValueError(f"leftover_match must be in [-1, 1], got {leftover_match!r}")
    leftover = mix
    if float(on_leftover) != 0.0 and abs(match) < 1.0:
        others = [
            d
            for d in field.leftover_dirs()
            if abs(float(d @ mix)) < 0.999
        ]
        if not others:
            raise ValueError("leftover_match < 1 needs ≥ 2 leftover dims")
        gen = torch.Generator().manual_seed(int(field.seed) + 11)
        spare = torch.zeros(int(field.dim))
        for direction in others:
            spare = spare + float(torch.randn((), generator=gen)) * direction
        spare = spare - (spare @ mix) * mix
        leftover = lm_unit(match * mix + (1.0 - match * match) ** 0.5 * lm_unit(spare))
    axis = float(on_u) * u + float(on_leftover) * leftover
    if content is not None:
        axis = axis + float(on_content) * content
    nu = float(noise)
    if nu > 0.0:
        gen = torch.Generator().manual_seed(int(field.seed) + 23)
        wobble = torch.randn(int(field.dim), generator=gen)
        wobble = wobble - (wobble @ u) * u
        axis = lm_unit(axis) + nu * lm_unit(wobble)
    if float(axis.norm()) <= 1e-8:
        raise ValueError("declared ê is empty")
    return lm_unit(axis)


def hold_direction(field: HighDLeakField, leak_dir: torch.Tensor | None) -> torch.Tensor | None:
    """What the live trainer holds: ``ê_⊥ = ê − (ê·û)û``."""
    if leak_dir is None:
        return None
    return lm_hold_dir(leak_dir, slider_dir=field.short_u(), mode="slider")


def hold_cover(field: HighDLeakField, held: torch.Tensor | None) -> float:
    """``p = |â·ê̂_⊥|`` — the share of the teacher the hold direction covers."""
    if held is None:
        return 0.0
    a = field.odd()
    return abs(float(lm_unit(a) @ lm_unit(held)))


def hold_split(field: HighDLeakField, held: torch.Tensor | None) -> dict[str, float]:
    """What the held direction is made of: concept content vs unused leftover."""
    if held is None:
        return {"hold_on_u": 0.0, "hold_on_content": 0.0, "hold_on_leftover": 0.0}
    unit = lm_unit(held)
    content = field.content_dir()
    leftover = sum(float(unit @ d) ** 2 for d in field.leftover_dirs()) ** 0.5
    return {
        "hold_on_u": abs(float(unit @ field.short_u())),
        "hold_on_content": 0.0 if content is None else abs(float(unit @ content)),
        "hold_on_leftover": leftover,
    }


@dataclass
class BendResidual:
    """``δ(s) = s·w + bend·G w`` — a stack whose ±1 replies are not mirrors.

    ``bend = 0`` is every existing 2-D cell: exactly odd in the slider
    scale, so ``cos(d+, d−) = −1`` by construction. The even reply has
    norm ``bend·||w||`` and ``parallel`` splits it between a gain
    (``+1`` and ``−1`` pushing different distances along the same
    direction) and a rotation (⊥ ``w``, taken from a fixed ``G``).
    ``free_even`` adds the ``odd_even`` residual's free ``|s|·w_even``
    term; the pole MSE and the hold are both even in it, so it never
    leaves zero.
    """

    w: torch.Tensor
    gate: torch.Tensor
    bend: float = 0.0
    parallel: float = 0.0
    w_even: torch.Tensor | None = None

    @classmethod
    def create(cls, field: HighDLeakField, *, free_even: bool = False) -> "BendResidual":
        return cls(
            torch.zeros(int(field.dim), requires_grad=True),
            field.gate_matrix(),
            float(field.bend),
            float(field.bend_parallel),
            torch.zeros(int(field.dim), requires_grad=True) if free_even else None,
        )

    def even(self) -> torch.Tensor:
        out = torch.zeros_like(self.w)
        if float(self.bend) != 0.0:
            par = float(self.parallel)
            norm = self.w.norm().clamp_min(1e-8)
            along = self.w / norm
            spun = self.gate @ self.w
            across = spun - (spun @ along) * along
            across = across / across.norm().clamp_min(1e-8)
            direction = par * along + (1.0 - par * par) ** 0.5 * across
            out = out + float(self.bend) * norm * direction
        if self.w_even is not None:
            out = out + self.w_even
        return out

    def delta(self, scale: float) -> torch.Tensor:
        return float(scale) * self.w + abs(float(scale)) * self.even()

    def parameters(self) -> list[torch.Tensor]:
        params = [self.w]
        if self.w_even is not None:
            params.append(self.w_even)
        return params

    def snapshot(self) -> "BendResidual":
        return BendResidual(
            self.w.detach().clone(),
            self.gate,
            self.bend,
            self.parallel,
            None if self.w_even is None else self.w_even.detach().clone(),
        )


TEACHERS = ("pair_odd", "pair_odd_sub_e", "pair_odd_sub_raw_e")


def teacher_poles(
    field: HighDLeakField,
    *,
    teacher: str = "pair_odd",
    leak_dir: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """``pair_odd``: ``t± = h0 ± a`` (live default ``--lm_target v9``).

    ``pair_odd_sub_e``: drop the axis the hold would have held,
    ``ê_⊥ = ê − (ê·û)û``, out of the poles first
    (``h± − ((h±−h0)·ê_⊥)ê_⊥``), then take the same symmetric teacher.
    That is the λ→∞ hold equilibrium reached in one step, with no
    stiffness. ``pair_odd_sub_raw_e`` subtracts raw ê and is the trap:
    ê's own û component goes with it. Neither is wired live.
    """
    pos, neg, neu = field.poles()
    mode = str(teacher).strip().lower()
    if mode in ("pair_odd_sub_e", "pair_odd_sub_raw_e"):
        if leak_dir is None:
            raise ValueError(f"{mode} needs a declared ê")
        axis = leak_dir if mode == "pair_odd_sub_raw_e" else hold_direction(field, leak_dir)
        if axis is not None:
            unit = lm_unit(axis)
            pos = pos - ((pos - neu).flatten() @ unit) * unit
            neg = neg - ((neg - neu).flatten() @ unit) * unit
    elif mode != "pair_odd":
        raise ValueError(f"teacher must be one of {TEACHERS}, got {teacher!r}")
    return lm_hidden_targets(pos, neg, neu, target_mode="symmetric")


def fit_highd(
    field: HighDLeakField,
    *,
    leak_dir: torch.Tensor | None = None,
    hold_weight: float = 0.0,
    teacher: str = "pair_odd",
    steps: int = 400,
    lr: float = 0.08,
    seed: int = 0,
    clip_value: float | None = None,
    free_even: bool = False,
    history_every: int = 100,
) -> tuple[BendResidual, list[dict]]:
    """Live LM slider loss on the high-D field. Same helpers as the trainer."""
    held = hold_direction(field, leak_dir)
    lam = float(hold_weight) if held is not None else 0.0
    tgt_plus, tgt_minus = teacher_poles(field, teacher=teacher, leak_dir=leak_dir)
    _pos, _neg, neu = field.poles()
    torch.manual_seed(int(seed))
    residual = BendResidual.create(field, free_even=free_even)
    opt = torch.optim.Adam(residual.parameters(), lr=float(lr))
    history: list[dict] = []

    def step_loss() -> torch.Tensor:
        pred_plus = neu + residual.delta(1.0)
        pred_minus = neu + residual.delta(-1.0)
        hold = None
        if held is not None and lam > 0.0:
            hold = lm_axis_hold(pred_plus, pred_minus, neu, held)
        return lm_slider_loss(
            pred_plus,
            pred_minus,
            tgt_plus,
            tgt_minus,
            hold=hold,
            hold_weight=lam if hold is not None else 0.0,
        )

    for i in range(int(steps) + 1):
        loss = step_loss()
        if i % int(history_every) == 0 or i == int(steps):
            history.append({"step": i, **fit_metrics(residual.snapshot(), field, loss=float(loss.detach()))})
        if i == int(steps):
            break
        opt.zero_grad()
        loss.backward()
        if clip_value is not None:
            torch.nn.utils.clip_grad_value_(residual.parameters(), float(clip_value))
        opt.step()
    return residual.snapshot(), history


def fit_metrics(
    residual: BendResidual,
    field: HighDLeakField,
    *,
    loss: float,
) -> dict[str, float]:
    """Live-log columns: trainer c+, collapse, perc, loss."""
    a = field.odd()
    _pos, _neg, neu = field.poles()
    d_plus = residual.delta(1.0)
    d_minus = residual.delta(-1.0)
    perc = float((d_plus - a).norm() / a.norm().clamp_min(1e-8))
    perc_minus = float((d_minus + a).norm() / a.norm().clamp_min(1e-8))
    norm_plus = float(d_plus.norm())
    norm_minus = float(d_minus.norm())
    return {
        "loss": float(loss),
        "c_plus": cosine(d_plus, a),
        "c_minus": cosine(d_minus, -a),
        "collapse": cosine(d_plus, d_minus),
        "perc": perc,
        "perc_minus": perc_minus,
        "norm_plus": norm_plus,
        "norm_minus": norm_minus,
        # Live logs pperc / nperc per pole; an even reply shows up here first.
        "norm_ratio": norm_plus / max(norm_minus, 1e-8),
        "neutral_norm": float(neu.norm()),
    }


def score_highd(
    name: str,
    field: HighDLeakField,
    *,
    leak_dir: torch.Tensor | None = None,
    hold_weight: float = 0.0,
    teacher: str = "pair_odd",
    e_label: str = "",
    steps: int = 400,
    seed: int = 0,
    clip_value: float | None = None,
    free_even: bool = False,
) -> dict:
    residual, history = fit_highd(
        field,
        leak_dir=leak_dir,
        hold_weight=hold_weight,
        teacher=teacher,
        steps=steps,
        seed=seed,
        clip_value=clip_value,
        free_even=free_even,
    )
    held = hold_direction(field, leak_dir)
    used_hold = float(hold_weight) if held is not None else 0.0
    cover = hold_cover(field, held)
    a = field.odd()
    u = field.short_u()
    intended = field.intended()
    d_plus = residual.delta(1.0)
    d_minus = residual.delta(-1.0)
    leftover = sum(float(d_plus @ direction) ** 2 for direction in field.leftover_dirs()) ** 0.5
    teacher_leftover = sum(float(a @ direction) ** 2 for direction in field.leftover_dirs()) ** 0.5
    on_intended = float(d_plus @ intended)
    subtracted = teacher.strip().lower() != "pair_odd"
    row = dict(history[-1])
    row.pop("step", None)
    row.update(
        hold_predictions(
            cover,
            used_hold,
            int(field.dim),
            shrink=0.0 if subtracted else None,
        )
    )
    row.update(hold_split(field, held))
    row.update(leftover_bipolar(d_plus, d_minus))
    row.update(
        {
            "name": name,
            "e_label": e_label,
            "dim": int(field.dim),
            "bend": float(field.bend),
            "teacher": teacher,
            "hold_weight": float(hold_weight),
            "used_hold": used_hold,
            "gate_align": field.gate_align(),
            "e_dot_u": 0.0 if leak_dir is None else float(lm_unit(leak_dir) @ u),
            "hold_norm": 0.0 if held is None else float(held.norm()),
            "hold_cover": cover,
            "hold_off": held is None,
            "cos_short_u": cosine(d_plus, u),
            "cos_intended": cosine(d_plus, intended),
            "leftover_norm": leftover,
            "leftover_teacher": teacher_leftover,
            "leftover_kept": leftover / (teacher_leftover + 1e-8) if teacher_leftover > 1e-8 else 0.0,
            "leftover_leak": leftover / (abs(on_intended) + 1e-8),
            "proj_intended": on_intended,
            "history": history,
        }
    )
    row["c_plus_gap"] = row["c_plus"] - row["c_plus_predicted"]
    row["axis"] = highd_verdicts(row)
    row["pass"] = all(v == "right" for v in row["axis"].values())
    row["looks_like_v12"] = row["c_plus"] >= V12_LOOK
    row["hold_explains_c_plus"] = row["c_plus_gap"] >= -HOLD_EXPLAINS_TOL
    return row


def highd_verdicts(row: dict) -> dict[str, str]:
    """Leak / slider / ±1 on the concept, not on the short probe."""
    return {
        "slider": "right" if row["cos_intended"] >= SLIDER_LOCK else "needs_help",
        "leak": "right" if abs(row["leftover_leak"]) <= LEAK_LOCK else "needs_help",
        "collapse": "right" if row["collapse"] <= BIPOLAR_MAX else "needs_help",
    }


def gender_like_field(*, bend: float = BEND_GENDER, dim: int = DEFAULT_DIM) -> HighDLeakField:
    """Clean pair: the poles *are* the singer, so no ê and no leftover."""
    return HighDLeakField(dim=dim, align=1.0, content=0.0, leftover=0.0, bend=bend)


def energy_field(
    *,
    bend: float = 0.0,
    dim: int = DEFAULT_DIM,
    align: float = LIVE_GATE_ALIGN,
) -> HighDLeakField:
    return HighDLeakField(dim=dim, align=align, bend=bend)


def energy_2d_field() -> HighDLeakField:
    """The orthonormal 2-D energy cell in this harness.

    ``align`` is the mean live |odd·û|/||odd|| of the existing energy
    rows (0.48 / 0.68 → 0.58) and there is no off-û concept, so cos to
    û and cos to the concept are the same number — the blind spot.
    """
    return HighDLeakField(dim=2, align=0.58, content=0.0, leftover=0.81, bend=0.0)


def synonym_e(field: HighDLeakField, *, on_u: float = 0.80, noise: float = 0.0) -> torch.Tensor:
    """Energy-v4 leak captions: the poles restated. Content-heavy ê."""
    off = (1.0 - float(on_u) ** 2) ** 0.5
    return leak_axis(
        field,
        on_u=on_u,
        on_content=0.92 * off,
        on_leftover=0.39 * off,
        noise=noise,
    )


def medium_pin_e(field: HighDLeakField) -> torch.Tensor:
    """The "same loudness" rewrite: no loud/quiet word, density still ∥ poles."""
    return leak_axis(field, on_u=0.30, on_content=0.88, on_leftover=0.37)


def leftover_only_e(field: HighDLeakField, *, leftover_match: float = 0.85) -> torch.Tensor:
    """Proposed ê: genre + BPM wording, no density / loudness / delivery / gain."""
    return leak_axis(
        field,
        on_u=0.05,
        on_content=0.0,
        on_leftover=1.0,
        leftover_match=leftover_match,
    )


def cell_table(*, steps: int = 400, seed: int = 0) -> list[dict]:
    """Every live bullet as one row on one harness."""
    energy = energy_field()
    bent = energy_field(bend=BEND_ENERGY)
    flat = energy_2d_field()
    specs = [
        ("gender_like_no_e", gender_like_field(), None, 0.0, "pair_odd", "none (clean pair)"),
        (
            "energy_2d_synonym_l8",
            flat,
            leak_axis(flat, on_u=0.5, on_content=0.0, on_leftover=0.866),
            LEAK_HOLD_WEIGHT,
            "pair_odd",
            "synonym ê on the 2-D field",
        ),
        ("energy_highd_pair_odd", energy, None, 0.0, "pair_odd", "none (v9 with no leak pair)"),
        (
            "energy_highd_synonym_l8",
            energy,
            synonym_e(energy),
            LEAK_HOLD_WEIGHT,
            "pair_odd",
            "synonym ê",
        ),
        (
            "energy_highd_tiny_l8",
            energy,
            synonym_e(energy, on_u=0.98),
            LEAK_HOLD_WEIGHT,
            "pair_odd",
            "synonym ê, tiny ê_⊥",
        ),
        (
            "energy_highd_tiny_messy_l8",
            energy,
            synonym_e(energy, on_u=0.98, noise=0.6),
            LEAK_HOLD_WEIGHT,
            "pair_odd",
            "synonym ê, tiny + messy ê_⊥",
        ),
        (
            "energy_highd_medium_pin_l8",
            energy,
            medium_pin_e(energy),
            LEAK_HOLD_WEIGHT,
            "pair_odd",
            "same-loudness pin, density still ∥ poles",
        ),
        (
            "energy_highd_leftover_l0.3",
            energy,
            leftover_only_e(energy),
            0.3,
            "pair_odd",
            "leftover-only ê (genre + BPM)",
        ),
        (
            "energy_highd_leftover_l1",
            energy,
            leftover_only_e(energy),
            1.0,
            "pair_odd",
            "leftover-only ê (genre + BPM)",
        ),
        (
            "energy_highd_leftover_l8",
            energy,
            leftover_only_e(energy),
            LEAK_HOLD_WEIGHT,
            "pair_odd",
            "leftover-only ê (genre + BPM)",
        ),
        (
            "energy_highd_sub_e_synonym",
            energy,
            synonym_e(energy),
            0.0,
            "pair_odd_sub_e",
            "synonym ê_⊥ subtracted from a",
        ),
        (
            "energy_highd_sub_e_leftover",
            energy,
            leftover_only_e(energy),
            0.0,
            "pair_odd_sub_e",
            "leftover-only ê_⊥ subtracted from a",
        ),
        (
            "energy_highd_sub_raw_e_leftover",
            energy,
            leftover_only_e(energy),
            0.0,
            "pair_odd_sub_raw_e",
            "leftover-only raw ê subtracted from a",
        ),
        (
            "energy_bend_synonym_l8",
            bent,
            synonym_e(bent),
            LEAK_HOLD_WEIGHT,
            "pair_odd",
            f"synonym ê, bend {BEND_ENERGY:g}",
        ),
        (
            "energy_bend_leftover_l1",
            bent,
            leftover_only_e(bent),
            1.0,
            "pair_odd",
            f"leftover-only ê, bend {BEND_ENERGY:g}",
        ),
    ]
    return [
        score_highd(
            name,
            field,
            leak_dir=axis,
            hold_weight=lam,
            teacher=teacher,
            e_label=label,
            steps=steps,
            seed=seed,
        )
        for name, field, axis, lam, teacher, label in specs
    ]


def synonym_cover(dim: int = DEFAULT_DIM) -> float:
    """``p`` for the energy-v4 synonym ê. Set by ê's wording, not by D."""
    field = energy_field(dim=int(dim))
    return hold_cover(field, hold_direction(field, synonym_e(field)))


def lambda_dim_sweep(
    *,
    lambdas: Iterable[float] = HOLD_LAMBDAS,
    dims: Iterable[int] = DIM_GRID,
    cover: float | None = None,
) -> list[dict]:
    """Closed-form λ × D grid: what λ buys once ``λ·D/2 ≫ 1``."""
    p = synonym_cover() if cover is None else float(cover)
    rows = []
    for dim in dims:
        for lam in lambdas:
            rows.append(
                {"dim": int(dim), "hold_weight": float(lam), "cover": p}
                | hold_predictions(p, lam, int(dim))
            )
    return rows


def lambda_fit_sweep(
    *,
    lambdas: Iterable[float] = HOLD_LAMBDAS,
    dims: Sequence[int] = (4, 8, 64),
    steps: int = 400,
    seed: int = 0,
) -> list[dict]:
    """Fitted counterpart of ``lambda_dim_sweep`` — the closed form is real."""
    rows = []
    for dim in dims:
        field = energy_field(dim=int(dim))
        axis = synonym_e(field)
        for lam in lambdas:
            rows.append(
                score_highd(
                    f"d{dim}_l{lam:g}",
                    field,
                    leak_dir=axis if lam > 0.0 else None,
                    hold_weight=lam,
                    e_label="synonym ê",
                    steps=steps,
                    seed=seed,
                )
            )
    return rows


def bend_sweep(
    *,
    bends: Iterable[float] = BEND_GRID,
    hold_weight: float = LEAK_HOLD_WEIGHT,
    steps: int = 400,
    seed: int = 0,
) -> list[dict]:
    """±1 polarity vs the response asymmetry, held ê fixed."""
    rows = []
    for bend in bends:
        field = energy_field(bend=float(bend))
        axis = synonym_e(field)
        rows.append(
            score_highd(
                f"bend{bend:g}_l{hold_weight:g}",
                field,
                leak_dir=axis,
                hold_weight=hold_weight,
                e_label="synonym ê",
                steps=steps,
                seed=seed,
            )
        )
        gender = gender_like_field(bend=float(bend))
        rows.append(
            score_highd(
                f"bend{bend:g}_gender",
                gender,
                leak_dir=None,
                hold_weight=0.0,
                e_label="none (clean pair)",
                steps=steps,
                seed=seed,
            )
        )
    return rows


def content_sweep(
    *,
    fractions: Iterable[float] = CONTENT_GRID,
    hold_weight: float = LEAK_HOLD_WEIGHT,
    steps: int = 400,
    seed: int = 0,
) -> list[dict]:
    """Leftover-only ê → pole synonym at fixed λ. ê's wording is the knob.

    ``fraction`` is how much of ``ê_⊥`` is off-û loudness content
    instead of unused leftover.
    """
    rows = []
    field = energy_field()
    for fraction in fractions:
        share = float(fraction)
        axis = leak_axis(
            field,
            on_u=0.05,
            on_content=share,
            on_leftover=(1.0 - share * share) ** 0.5,
        )
        rows.append(
            score_highd(
                f"content{share:g}_l{hold_weight:g}",
                field,
                leak_dir=axis,
                hold_weight=hold_weight,
                e_label=f"ê_⊥ is {share:g} concept content",
                steps=steps,
                seed=seed,
            )
        )
    return rows


MATCH_GRID = (0.5, 0.7, 0.85, 0.95, 1.0)


def match_sweep(
    *,
    matches: Iterable[float] = MATCH_GRID,
    hold_weight: float = 1.0,
    steps: int = 400,
    seed: int = 0,
) -> list[dict]:
    """Leftover-only ê, swept on how completely it names the leak.

    One declared caption pair is one direction. Whatever unused mix the
    poles carry off that direction survives any λ.
    """
    rows = []
    field = energy_field()
    for match in matches:
        axis = leftover_only_e(field, leftover_match=float(match))
        row = score_highd(
            f"match{match:g}_l{hold_weight:g}",
            field,
            leak_dir=axis,
            hold_weight=hold_weight,
            e_label=f"leftover-only ê naming {match:g} of the leak",
            steps=steps,
            seed=seed,
        )
        row["leftover_match"] = float(match)
        rows.append(row)
    return rows


def polarity_grid(
    *,
    lambdas: Iterable[float] = (0.0, 1.0, 8.0, 64.0, 1024.0),
    dims: Iterable[int] = (4, 8, 64),
    e_specs: Sequence[tuple[str, float, float]] = (
        ("synonym", 0.80, 0.0),
        ("tiny", 0.98, 0.0),
        ("tiny_messy", 0.98, 0.6),
    ),
    steps: int = 200,
    seed: int = 0,
) -> list[dict]:
    """±1 across every geometry knob, with a *free* even parameter.

    Nothing here breaks polarity: the pole MSE splits as
    ``|w_odd − a|² + |w_even|²`` and the hold splits the same way.
    """
    rows = []
    for dim in dims:
        field = energy_field(dim=int(dim))
        for label, on_u, noise in e_specs:
            axis = synonym_e(field, on_u=float(on_u), noise=float(noise))
            for lam in lambdas:
                rows.append(
                    score_highd(
                        f"d{dim}_{label}_l{lam:g}",
                        field,
                        leak_dir=axis if lam > 0.0 else None,
                        hold_weight=lam,
                        e_label=label,
                        steps=steps,
                        seed=seed,
                        free_even=True,
                    )
                )
    return rows


def bend_for_collapse(collapse: float) -> float:
    """``bend`` implied by a logged ±1 cosine, for a pure rotation.

    With ``d± = ±w + b·||w||·q̂``, ``q̂ ⊥ w``:
    ``cos(d+, d−) = (b² − 1)/(b² + 1)`` → ``b = √((1+c)/(1−c))``.
    A gain share (``parallel > 0``) makes the same collapse need a
    larger ``bend``, so this is a lower bound on the asymmetry.
    """
    c = float(collapse)
    if not -1.0 <= c < 1.0:
        raise ValueError(f"collapse must be in [-1, 1), got {collapse!r}")
    return ((1.0 + c) / (1.0 - c)) ** 0.5


PARALLEL_GRID = (0.0, 0.2, 0.4, 0.5, 0.6, 0.65, 0.7, 0.8)
ANALOGUE_BENDS = tuple(0.4 + 0.4 * i for i in range(14))


def _bent_row(
    name: str,
    *,
    bend: float,
    parallel: float,
    hold_weight: float = LEAK_HOLD_WEIGHT,
    dim: int = DEFAULT_DIM,
    steps: int = 400,
    seed: int = 0,
    e_label: str = "synonym ê",
) -> dict:
    field = energy_field(dim=int(dim))
    field = HighDLeakField(
        dim=field.dim,
        align=field.align,
        content=field.content,
        leftover=field.leftover,
        scale=field.scale,
        bend=float(bend),
        bend_parallel=float(parallel),
        seed=field.seed,
    )
    row = score_highd(
        name,
        field,
        leak_dir=synonym_e(field),
        hold_weight=hold_weight,
        e_label=e_label,
        steps=steps,
        seed=seed,
    )
    row["bend_parallel"] = float(parallel)
    return row


def calibrate_bend(
    target_collapse: float,
    *,
    parallel: float = 0.0,
    hold_weight: float = LEAK_HOLD_WEIGHT,
    dim: int = DEFAULT_DIM,
    steps: int = 400,
    seed: int = 0,
    tol: float = 1e-3,
    max_iter: int = 32,
) -> float:
    """±1 asymmetry that reproduces a logged collapse on this cell.

    Collapse is monotone in ``bend`` for a pure rotation
    (``parallel = 0``), so bisect. Mixed gain / rotation is not
    monotone — use ``live_v14_analogue``, which searches. The fixture
    cannot derive the live value either way: it reads it off the log.
    """
    lo, hi = 0.0, 16.0

    def collapse_at(bend: float) -> float:
        return _bent_row(
            "calibrate",
            bend=bend,
            parallel=parallel,
            hold_weight=hold_weight,
            dim=dim,
            steps=steps,
            seed=seed,
        )["collapse"]

    if collapse_at(hi) < float(target_collapse):
        raise ValueError(f"collapse {target_collapse!r} is out of reach at bend ≤ {hi}")
    for _ in range(int(max_iter)):
        mid = 0.5 * (lo + hi)
        if collapse_at(mid) < float(target_collapse):
            lo = mid
        else:
            hi = mid
        if hi - lo < tol:
            break
    return 0.5 * (lo + hi)


def live_v14_analogue(
    *,
    parallels: Iterable[float] = PARALLEL_GRID,
    bends: Iterable[float] = ANALOGUE_BENDS,
    steps: int = 300,
    seed: int = 0,
) -> tuple[dict, list[dict]]:
    """Closest analogue of the live energy-v14 log this field can make.

    Searches the ±1 asymmetry — its size ``bend`` and its gain /
    rotation split — for the cell whose ``(collapse, c+)`` is nearest
    the live pair. Both live numbers are inputs: the fixture reads the
    stack's asymmetry off the log, it does not predict it.
    """
    rows = []
    for parallel in parallels:
        for bend in bends:
            row = _bent_row(
                f"analogue_par{parallel:g}_b{bend:g}",
                bend=float(bend),
                parallel=float(parallel),
                steps=steps,
                seed=seed,
                e_label=(
                    f"synonym ê, ±1 asymmetry {bend:g} "
                    f"({parallel:g} gain / {1.0 - float(parallel):g} rotation)"
                ),
            )
            row["calibrated_bend"] = float(bend)
            rows.append(row)

    def miss(row: dict) -> float:
        return (row["collapse"] - LIVE_COLLAPSE) ** 2 + (row["c_plus"] - LIVE_C_PLUS) ** 2

    best = dict(min(rows, key=miss))
    best["name"] = "energy_live_v14_analogue"
    return best, rows


def compact(row: dict, *, history: bool = True) -> dict:
    out = {}
    for key, value in row.items():
        if key in ("history", "axis"):
            continue
        if isinstance(value, (int, float, bool, str)):
            out[key] = value
    out["axis"] = dict(row.get("axis", {}))
    if history:
        out["history"] = [
            {k: (int(v) if k == "step" else float(v)) for k, v in snap.items()}
            for snap in row.get("history", [])
        ]
    return out
