"""Live energy-v14 signature: c+ vs slider-cos, live-D tiny ê_⊥, ±1 polarity.

The existing 2-D cells (``overlap.py``) show ê·û overlap and that holding
ê_⊥û locks the slider at ρ=0.5. They do **not** show the live energy-v14
signature:

1. trainer c+ (cos with the pair-odd teacher ``a``) and slider-cos (cos
   with the declared û) as first-class columns. Gender's 0.97 c+ is "I
   copied pair-odd", not slider lock — and the leaky-energy no-hold run
   prints the *same* c+/loss/collapse. A working hold must print a worse
   c+; people read that as failure.
2. live is not 2-D. û and ê are short captions in a huge hidden space.
   MSE is a mean over D dims while the hold is a squared dot with a
   *unit* ê, so the held component of a linear fit lands at
   ``s_e = t_e / (1 + λ·D/2)`` — λ=8 is a 9× shrink in 2-D and a ~4000×
   annihilation at live D. And after dropping û, ê_⊥ is one messy
   direction out of a (D−2)-dimensional leftover: holding it is violent
   *and* barely cures the heard leak.
3. the ±1 polarity break (live collapse **+0.18**, loss 278 by step 13).
   A linear residual can never show it: with symmetric targets the loss
   separates over odd/even weights, the even weight gets exactly zero
   gradient, and collapse is −1.0 to machine precision under any hold,
   any λ, any D. The break needs hidden states that are *curved* in the
   adapter — a LoRA at −1 is −ΔW in weight space, not −Δh in hidden
   space.

This module adds a D=1024 field with that geometry (a heard-loudness
plane the declared probe only partially spans, a shared genre/BPM/mix
leftover, per-row wording, caption-syntax junk) and two students:

- ``linear``: odd+even free residual, the class every existing cell
  uses. Its even weight provably stays 0 (the separability point).
- ``curved``: an adapter applied through a fixed saturating layer
  (tanh around a non-zero operating point). Its ± deltas are only
  approximately opposite, like a real net, and the λ=8 fight at live D
  actually breaks ±1 and blows the loss up ~3000× in the first steps.

CPU only. No Hub, no GPU, no Music 3 weights. Does not change the live
trainer default (``--lm_target v9``, hold λ=8 on declared ê_⊥û).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from functools import lru_cache

import torch
import torch.nn.functional as F

from conceptmod.textsliders.slider_targets import (
    LEAK_HOLD_WEIGHT,
    lm_axis_hold,
    lm_hidden_targets,
    lm_hold_dir,
    lm_slider_loss,
    lm_unit,
)


# Live hidden width is ~2048; 1024 keeps the same λ·D/2 physics and runs fast.
SIG_DIM = 1024

# Live energy rows: |odd·û|/||odd|| logged 0.48 / 0.68 (same as energy.py).
SIG_ALIGNS = (0.48, 0.48, 0.68, 0.68)
SIG_SCALE = 1.2

# The declared short caption ("Extremely high energy…") is a probe of the
# heard loudness, not all of it: dense/slammed wording is loudness too.
HEARD_SPLIT_DEG = 40.0
# Per-row wording share of the leftover (each pair words its genre/BPM/mix
# differently).
ROW_NOISE_DEG = 25.0

# Declared ê variants. `opposite` is the energy-v4 leak captions ("Dense
# slammed mix, BPM 168, pop-punk" vs "Sparse airy mix, BPM 52, ambient
# lullaby") — a pole synonym: mostly heard loudness, a little wording.
# `pinned` is the "medium energy on both sides" rewrite — the explicit
# level words are gone but slammed/dense vs sparse/airy still encodes
# loud vs quiet in Qwen, so ê stays heard-dominant and ê_⊥û stays messy.
OPPOSITE_HEARD = 0.85
PINNED_HEARD = 0.62

# Verdict bands (match energy.py / overlap.py where they overlap).
SLIDER_LOCK = 0.90
LEAK_LOCK = 0.20
COLLAPSE_LOCK = -0.85
V12_LOOK_CPLUS = 0.90
POLARITY_BROKEN = -0.50

# Live steps; also the fixture's default.
SIG_STEPS = 800


@dataclass(frozen=True)
class SigAxes:
    """Fixed orthonormal caption directions in R^dim.

    Everything past the named dims is padding the pair never moves — the
    MSE mean still runs over it, exactly like live hidden dims.
    """

    u_probe: torch.Tensor      # declared short loud/quiet û
    u_dense: torch.Tensor      # dense/slammed loudness wording the probe misses
    leftover: torch.Tensor     # shared genre / BPM / mix wording
    row_noise: tuple[torch.Tensor, ...]  # per-row wording
    junk: torch.Tensor         # caption syntax, never in the teacher


@lru_cache(maxsize=4)
def sig_axes(dim: int = SIG_DIM) -> SigAxes:
    if dim < 12:
        raise ValueError(f"signature field needs at least 12 dims, got {dim}")

    def axis(*pairs: tuple[int, float]) -> torch.Tensor:
        v = torch.zeros(dim)
        for idx, val in pairs:
            v[idx] = val
        return v

    return SigAxes(
        u_probe=axis((0, 1.0)),
        u_dense=axis((1, 1.0)),
        leftover=lm_unit(axis((2, 0.7), (3, 0.5), (4, 0.5))),
        row_noise=tuple(axis((5 + r, 1.0)) for r in range(4)),
        junk=lm_unit(axis((9, 0.8), (10, 0.6))),
    )


@dataclass(frozen=True)
class SignatureField:
    """Four leaky energy rows in live-D; declared û is the probe axis only.

    ``gender_like=True`` replaces the rows with one clean pair whose odd
    *is* the concept (no leftover, no ê) and keeps the declared û a short
    probe at 0.20 — the live gender-v1/v14 geometry. Do not invent a junk
    ê there: ``leak_axis`` refuses.
    """

    dim: int = SIG_DIM
    aligns: tuple[float, ...] = SIG_ALIGNS
    scale: float = SIG_SCALE
    heard_split_deg: float = HEARD_SPLIT_DEG
    row_noise_deg: float = ROW_NOISE_DEG
    gender_like: bool = False
    gender_align: float = 0.20

    @property
    def axes(self) -> SigAxes:
        return sig_axes(self.dim)

    @property
    def heard(self) -> torch.Tensor:
        phi = math.radians(self.heard_split_deg)
        return math.cos(phi) * self.axes.u_probe + math.sin(phi) * self.axes.u_dense

    @property
    def intended(self) -> torch.Tensor:
        """What the slider should move: the concept (gender) / heard loudness."""
        return self.axes.u_probe.clone() if self.gender_like else self.heard

    @property
    def declared_u(self) -> torch.Tensor:
        if self.gender_like:
            a = float(self.gender_align)
            return a * self.axes.u_probe + math.sqrt(1.0 - a * a) * self.axes.junk
        return self.axes.u_probe.clone()

    def odd(self, row: int) -> torch.Tensor:
        if self.gender_like:
            return self.scale * self.axes.u_probe
        a = float(self.aligns[row])
        theta = math.radians(self.row_noise_deg)
        left = math.cos(theta) * self.axes.leftover + math.sin(theta) * self.axes.row_noise[row]
        return self.scale * (a * self.heard + math.sqrt(1.0 - a * a) * left)

    def odds(self) -> list[torch.Tensor]:
        n = 1 if self.gender_like else len(self.aligns)
        return [self.odd(r) for r in range(n)]

    def leak_axis(self, kind: str) -> torch.Tensor | None:
        """Declared ê. ``None`` on the clean gender-like pair (v9 hold off)."""
        name = str(kind).strip().lower()
        if name == "none":
            return None
        if self.gender_like:
            raise ValueError("gender-like cell declares no ê — do not invent one")
        ax = self.axes
        if name == "leftover_only":
            # genre+BPM wording only — no density / loudness / delivery words.
            return ax.leftover.clone()
        mess = lm_unit(0.7 * ax.leftover + 0.3 * ax.junk)
        if name == "opposite":
            rho = OPPOSITE_HEARD
        elif name == "pinned":
            rho = PINNED_HEARD
            mess = lm_unit(0.5 * ax.leftover + 0.5 * ax.junk)
        else:
            raise ValueError(f"leak kind must be none/opposite/pinned/leftover_only, got {kind!r}")
        return lm_unit(rho * self.heard + math.sqrt(max(1.0 - rho * rho, 0.0)) * mess)

    def leak_geometry(self, kind: str) -> dict[str, float]:
        if self.gender_like or str(kind).strip().lower() == "none":
            return {"e_dot_u": 0.0, "e_perp_norm": 0.0, "e_perp_dot_ahat": 0.0}
        e = self.leak_axis(kind)
        u = lm_unit(self.declared_u)
        perp = e - (e @ u) * u
        mean_odd = lm_unit(torch.stack(self.odds()).mean(0))
        return {
            "e_dot_u": float(e @ u),
            "e_perp_norm": float(perp.norm()),
            "e_perp_dot_ahat": 0.0 if float(perp.norm()) < 1e-8 else float(lm_unit(perp) @ mean_odd),
        }


def assert_signature_geometry(field: SignatureField | None = None) -> None:
    """The live-D field must keep the live numbers, not drift into a cheat."""
    field = field or SignatureField()
    for r, odd in enumerate(field.odds()):
        align = float((odd @ lm_unit(field.declared_u)).abs() / odd.norm())
        want = field.aligns[r] * math.cos(math.radians(field.heard_split_deg))
        if abs(align - want) > 1e-4:
            raise AssertionError(f"row {r} probe align {align:.3f} != {want:.3f}")
        if align > 0.80:
            raise AssertionError("probe align must stay live-middling (< 0.80), not the 0.95 cheat")
    geo = field.leak_geometry("opposite")
    if geo["e_perp_dot_ahat"] < 0.60:
        raise AssertionError("opposite-energy ê_⊥ must overlap the pair-odd (synonym in disguise)")
    left = field.leak_geometry("leftover_only")
    if abs(left["e_dot_u"]) > 1e-6:
        raise AssertionError("leftover-only ê must not contain the probe û")
    if abs(float(field.leak_axis("leftover_only") @ field.heard)) > 1e-6:
        raise AssertionError("leftover-only ê must not contain heard loudness")
    gender = SignatureField(dim=field.dim, gender_like=True)
    try:
        gender.leak_axis("opposite")
    except ValueError:
        pass
    else:
        raise AssertionError("gender-like cell must refuse an invented ê")


class LinearStudent:
    """Odd+even free residual (same class as ``Residual('odd_even')``)."""

    kind = "linear"

    def __init__(self, dim: int = SIG_DIM, seed: int = 0):
        del seed
        self.w_odd = torch.zeros(dim, requires_grad=True)
        self.w_even = torch.zeros(dim, requires_grad=True)

    def delta(self, scale: float) -> torch.Tensor:
        return float(scale) * self.w_odd + abs(float(scale)) * self.w_even

    def parameters(self) -> list[torch.Tensor]:
        return [self.w_odd, self.w_even]


class CurvedStudent:
    """Adapter through a fixed saturating layer around a non-zero point.

    ``delta(s) = C·tanh(z0 + s·w + |s|·w_even) − C·tanh(z0)``. The −1 pole
    applies the adapter negated in *pre-activation* space (a LoRA
    multiplier), so the ± hidden deltas are only approximately opposite:
    the curvature of tanh at z0 is an even response the student controls
    only through the same weights the teacher pins. The linear student
    has the same even DOF and provably cannot excite it — curvature is
    what couples it to the loss.
    """

    kind = "curved"

    def __init__(self, dim: int = SIG_DIM, seed: int = 0, z0_scale: float = 0.6,
                 gain: float = 2.0, cond: float = 0.8):
        g = torch.Generator().manual_seed(seed + 13)
        q = torch.linalg.qr(torch.randn(dim, dim, generator=g))[0]
        self.C = gain * (q * torch.logspace(0, -cond, dim))
        self.z0 = torch.randn(dim, generator=g) * z0_scale
        self.w_odd = torch.zeros(dim, requires_grad=True)
        self.w_even = torch.zeros(dim, requires_grad=True)
        self.base = (self.C @ torch.tanh(self.z0)).detach()

    def delta(self, scale: float) -> torch.Tensor:
        s = float(scale)
        pre = self.z0 + s * self.w_odd + abs(s) * self.w_even
        return self.C @ torch.tanh(pre) - self.base

    def parameters(self) -> list[torch.Tensor]:
        return [self.w_odd, self.w_even]


def make_student(kind: str, dim: int = SIG_DIM, seed: int = 0):
    name = str(kind).strip().lower()
    if name == "linear":
        return LinearStudent(dim, seed=seed)
    if name == "curved":
        return CurvedStudent(dim, seed=seed)
    raise ValueError(f"student must be linear or curved, got {kind!r}")


def _fit_metrics(student, field: SignatureField, hold_dir: torch.Tensor | None,
                 hold_weight: float) -> dict:
    """The live train-log numbers plus the fixture's geometry columns."""
    odds = field.odds()
    neu = torch.zeros(field.dim)
    ax = field.axes
    with torch.no_grad():
        d_plus = student.delta(1.0)
        d_minus = student.delta(-1.0)
        cplus, losses, percs = [], [], []
        for a in odds:
            tgt_plus, tgt_minus = lm_hidden_targets(neu + a, neu - a, neu)
            cplus.append(float(F.cosine_similarity(d_plus[None], a[None])))
            percs.append(float((d_plus - a).norm() / a.norm().clamp_min(1e-8)))
            hold = None
            used = 0.0
            if hold_dir is not None and hold_weight > 0.0:
                hold = lm_axis_hold(neu + d_plus, neu + d_minus, neu, hold_dir)
                used = float(hold_weight)
            losses.append(float(lm_slider_loss(
                neu + d_plus, neu + d_minus, tgt_plus, tgt_minus, hold=hold, hold_weight=used,
            )))
        n = len(odds)
        u = lm_unit(field.declared_u)
        intended = lm_unit(field.intended)
        proj_intended = float(d_plus @ intended)
        if field.gender_like:
            in_span = proj_intended * intended
        else:
            plane = torch.stack([ax.u_probe, ax.u_dense])
            in_span = plane.T @ (plane @ d_plus)
        leftover_norm = float((d_plus - in_span).norm())
    return {
        "cos_teacher": sum(cplus) / n,
        "collapse": float(F.cosine_similarity(d_plus[None], d_minus[None])),
        "perc": sum(percs) / n,
        "loss": sum(losses) / n,
        "slider_cos": float(F.cosine_similarity(d_plus[None], u[None])),
        "intended_cos": float(F.cosine_similarity(d_plus[None], intended[None])),
        "proj_intended": proj_intended,
        "leak_ratio": leftover_norm / (abs(proj_intended) + 1e-8),
        "norm_plus": float(d_plus.norm()),
        "norm_minus": float(d_minus.norm()),
        "w_even_norm": float(student.w_even.detach().norm()),
    }


def train_signature(
    name: str,
    *,
    field: SignatureField | None = None,
    student: str = "linear",
    leak: str = "none",
    hold_weight: float = 0.0,
    subtract_e: bool = False,
    steps: int = SIG_STEPS,
    lr: float = 0.08,
    seed: int = 0,
    history_every: int = 25,
) -> dict:
    """One signature cell. Teacher is full pair-odd (v9); hold per live rule.

    The hold direction is ``lm_hold_dir(ê, û, mode="slider")`` — the live
    ê_⊥û — and ``lm_axis_hold`` unit-normalizes it, so a tiny messy
    leftover is penalized at full λ, exactly like the trainer.

    ``subtract_e=True`` is the proposed ``pair_odd_sub_e`` teacher change:
    subtract the declared ê_⊥û component from ``a`` instead of holding
    it. The hold is then off (nothing left to fight). ê is orthogonalized
    to û first for the same reason the hold is — a raw synonym ê would
    subtract the slider itself (shown by ``sub_e_synonym_raw``).
    """
    field = field or SignatureField()
    torch.manual_seed(seed)
    st = make_student(student, field.dim, seed=seed)
    opt = torch.optim.Adam(st.parameters(), lr=lr)
    neu = torch.zeros(field.dim)
    odds = field.odds()

    e_raw = field.leak_axis(leak) if leak != "none" else None
    e_perp = None
    if e_raw is not None:
        e_perp = lm_hold_dir(e_raw, slider_dir=field.declared_u, mode="slider")
    hold_dir = None
    used_hold = 0.0
    if e_perp is not None and hold_weight > 0.0 and not subtract_e:
        hold_dir = e_perp
        used_hold = float(hold_weight)

    teacher_odds = odds
    if subtract_e and e_raw is not None:
        unit_e = lm_unit(e_perp if e_perp is not None else e_raw)
        teacher_odds = [a - (a @ unit_e) * unit_e for a in odds]

    def loss_fn() -> torch.Tensor:
        d_plus = st.delta(1.0)
        d_minus = st.delta(-1.0)
        total = neu.new_zeros(())
        for t_odd in teacher_odds:
            tgt_plus, tgt_minus = lm_hidden_targets(neu + t_odd, neu - t_odd, neu)
            hold = None
            if hold_dir is not None:
                hold = lm_axis_hold(neu + d_plus, neu + d_minus, neu, hold_dir)
            total = total + lm_slider_loss(
                neu + d_plus, neu + d_minus, tgt_plus, tgt_minus,
                hold=hold, hold_weight=used_hold,
            )
        return total / len(teacher_odds)

    def snap(step: int, loss_val: float) -> dict:
        fit = _fit_metrics(st, field, hold_dir, used_hold)
        return {"step": step, "loss_train": float(loss_val), **{
            k: fit[k] for k in ("loss", "cos_teacher", "collapse", "perc", "slider_cos")
        }}

    history = [snap(0, float(loss_fn().detach()))]
    for i in range(steps):
        loss = loss_fn()
        opt.zero_grad()
        loss.backward()
        opt.step()
        if (i + 1) % history_every == 0 or (i + 1) == steps or i < 12:
            history.append(snap(i + 1, float(loss.detach())))

    metrics = _fit_metrics(st, field, hold_dir, used_hold)
    late = [h for h in history if h["step"] >= steps // 2]
    early = [h for h in history if 0 < h["step"] < steps // 2]
    metrics.update(field.leak_geometry(leak))
    metrics.update(
        {
            "name": name,
            "student": st.kind,
            "leak_kind": leak,
            "hold_weight": float(hold_weight) if not subtract_e else 0.0,
            "used_hold": used_hold,
            "subtract_e": bool(subtract_e),
            "steps": int(steps),
            "collapse_late_max": max(h["collapse"] for h in late),
            "cos_teacher_min": min(h["cos_teacher"] for h in history[1:]),
            "loss_max": max(h["loss_train"] for h in history[1:]),
            "loss_spike": max(h["loss_train"] for h in history[1:]) / max(history[0]["loss_train"], 1e-9),
            "polarity_broken": max(h["collapse"] for h in late) > POLARITY_BROKEN,
            "polarity_broken_early": bool(early) and max(h["collapse"] for h in early) > POLARITY_BROKEN,
            "looks_like_v12": (
                metrics["cos_teacher"] >= V12_LOOK_CPLUS and metrics["perc"] <= 0.40
            ),
            "slider_locked": metrics["intended_cos"] >= SLIDER_LOCK,
            "leak_ok": abs(metrics["leak_ratio"]) <= LEAK_LOCK,
            "bipolar_ok": metrics["collapse"] <= COLLAPSE_LOCK,
            "history": history,
        }
    )
    return metrics


def linear_shrink_factor(dim: int, hold_weight: float) -> float:
    """Closed form for the held component of a linear fit.

    Per pole the loss is ``mse = ‖d ∓ a‖²/D`` plus ``(λ/2)(d·ê)²``, so the
    optimum along a unit ê (orthogonal to everything else the teacher
    wants) is

        s_e = t_e / (1 + λ·D/2)

    In 2-D that is ``t_e/(1+λ)`` — the shrink-in-place the overlap doc
    derived. At D=1024 the same λ=8 is 4097×; λ=0.3 is already 155×.
    λ does not mean the same thing at a different D.
    """
    return 1.0 / (1.0 + float(hold_weight) * float(dim) / 2.0)


def measured_shrink_factor(dim: int, hold_weight: float, *, steps: int = 800,
                           lr: float = 0.05, seed: int = 0) -> float:
    """Fit one row whose odd is exactly ê and measure s_e/t_e."""
    torch.manual_seed(seed)
    e = torch.zeros(dim)
    e[0] = 1.0
    a = 1.2 * e
    w = torch.zeros(dim, requires_grad=True)
    opt = torch.optim.Adam([w], lr=lr)
    for _ in range(steps):
        loss = F.mse_loss(w, a) + F.mse_loss(-w, -a) + float(hold_weight) * 0.5 * (
            (w @ e) ** 2 + ((-w) @ e) ** 2
        )
        opt.zero_grad()
        loss.backward()
        opt.step()
    return float((w.detach() @ e) / (a @ e))


def signature_table(*, dim: int = SIG_DIM, steps: int = SIG_STEPS, seed: int = 0) -> list[dict]:
    """Every signature cell; both students where the contrast matters."""
    energy = SignatureField(dim=dim)
    gender = SignatureField(dim=dim, gender_like=True)
    rows = [
        train_signature("gender_like_linear", field=gender, student="linear",
                        steps=steps, seed=seed),
        train_signature("gender_like_curved", field=gender, student="curved",
                        steps=steps, seed=seed),
        train_signature("energy_no_hold_linear", field=energy, student="linear",
                        steps=steps, seed=seed),
        train_signature("energy_no_hold_curved", field=energy, student="curved",
                        steps=steps, seed=seed),
        train_signature("synonym_perp_l8_linear", field=energy, student="linear",
                        leak="opposite", hold_weight=LEAK_HOLD_WEIGHT, steps=steps, seed=seed),
        train_signature("synonym_perp_l8_curved", field=energy, student="curved",
                        leak="opposite", hold_weight=LEAK_HOLD_WEIGHT, steps=steps, seed=seed),
        train_signature("pinned_perp_l8_curved", field=energy, student="curved",
                        leak="pinned", hold_weight=LEAK_HOLD_WEIGHT, steps=steps, seed=seed),
    ]
    for lam in (0.3, 1.0, LEAK_HOLD_WEIGHT):
        rows.append(
            train_signature(f"leftover_only_l{lam:g}_curved", field=energy, student="curved",
                            leak="leftover_only", hold_weight=lam, steps=steps, seed=seed)
        )
    rows.append(
        train_signature("sub_e_leftover_linear", field=energy, student="linear",
                        leak="leftover_only", subtract_e=True, steps=steps, seed=seed)
    )
    rows.append(
        train_signature("sub_e_leftover_curved", field=energy, student="curved",
                        leak="leftover_only", subtract_e=True, steps=steps, seed=seed)
    )
    rows.append(
        train_signature("sub_e_synonym_curved", field=energy, student="curved",
                        leak="opposite", subtract_e=True, steps=steps, seed=seed)
    )
    return rows


def compact(row: dict) -> dict:
    out = {}
    for key, value in row.items():
        if key == "history":
            out[key] = [
                {k: (int(v) if k == "step" else float(v)) for k, v in h.items()}
                for h in value
            ]
        elif isinstance(value, (int, float, bool, str)):
            out[key] = value
    return out
