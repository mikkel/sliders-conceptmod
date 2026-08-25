"""Lyric-garble Goodhart: a sheet the hidden-only cells cannot see.

v15 pole term is hidden MSE onto the pair-odd midpoint
``t± = h0 ± ½(h+ − h−)``. That point is not a real caption. Hitting it
prints pair-odd cos ~0.96 and collapse ~−0.95 — the live gender-v15
"locked" look — while the next-token policy at ``t±`` is off the
caption sheet.

v16 ``--pole_mode semantic_kl`` fits the next-token policy of a real
caption hidden (gender: ``encode(pos)`` / ``encode(neg)``; leaky:
ê-cleaned real poles). Pair-odd cos / collapse are logs only and will
look worse. That is expected.

This field is the smallest extra structure that can see the garble:

- a tiny vocab + linear readout from hidden
- a **sheet**: tokens / policy mass of the real +/− captions (and a
  shared lyric token)
- an **off-sheet** region around the synthetic midpoint: argmax is a
  third token (and a leftover token on leaky poles)

A linear readout is enough. Softmax of a hidden-space blend is not a
blend of the two policies; the midpoint prefers a third token. MSE to
that midpoint maximizes it. KL to the pole logits does not — if the
student can hold the shared even. An odd-linear student (live-like
LoRA multiplier) is stuck on the midpoint ray; with no bias, argmax
is constant along a ray, so odd+KL stays off-sheet. The
``odd_even`` residual (``s·w_odd + |s|·w_even``) is the curved
student that can stay on-sheet.

CPU only. No Hub, no GPU, no Music 3 weights. Does not change the
live trainer default.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from analysis.slider2d.field import cosine
from analysis.slider2d.train import Residual
from conceptmod.textsliders.slider_targets import (
    leftover_bipolar,
    lm_e_cleaned_captions,
    lm_hidden_targets,
    lm_pair_odd_sub_e,
    lm_policy_logits,
    lm_semantic_kl,
    lm_slider_loss,
    lm_unit,
)


TOKEN_POS = 0
TOKEN_NEG = 1
TOKEN_OOD_POS = 2
TOKEN_OOD_NEG = 3
TOKEN_LYRIC = 4
TOKEN_LEAK = 5
TOKEN_NAMES = ("pos", "neg", "ood+", "ood-", "lyric", "leak")
SHEET_TOKENS = (TOKEN_POS, TOKEN_NEG, TOKEN_LYRIC)
OFFSHEET_TOKENS = (TOKEN_OOD_POS, TOKEN_OOD_NEG, TOKEN_LEAK)

# Hidden: dim 0 = pair-odd, dim 1 = shared even (real captions only),
# dim 2 = leftover unused attr (energy / leaky cells).
ODD_SCALE = 1.0
EVEN_SCALE = 0.75
LEAK_SCALE = 0.80

# Linear readout. Chosen so:
#   argmax(h+) = pos, argmax(h-) = neg, argmax(t±) = ood±
#   leftover-heavy pair-odd midpoint prefers leak
#   ê-cleaned midpoint still prefers ood (even dropped)
#   ê-cleaned real poles stay on-sheet
POS_ROW = (1.0, 4.0, 0.0)
NEG_ROW = (-1.0, 4.0, 0.0)
OOD_POS_ROW = (2.5, 0.0, 0.0)
OOD_NEG_ROW = (-2.5, 0.0, 0.0)
LYRIC_ROW = (0.0, 3.0, 0.0)
LEAK_ROW = (0.0, 0.0, 3.5)

LOOKS_LOCKED_COS = 0.90
LOOKS_LOCKED_COLLAPSE = -0.85
ON_SHEET_MASS = 0.50
OFF_SHEET_MASS = 0.40
STEPS = 300
LR = 0.08
WINDOW = 50


def readout_weight(dim: int) -> torch.Tensor:
    """``[vocab, dim]`` LM-head stand-in. Extra hidden dims stay unused."""
    rows = (
        POS_ROW,
        NEG_ROW,
        OOD_POS_ROW,
        OOD_NEG_ROW,
        LYRIC_ROW,
        LEAK_ROW,
    )
    weight = torch.zeros(len(rows), int(dim))
    width = min(int(dim), 3)
    for i, row in enumerate(rows):
        weight[i, :width] = torch.tensor(row[:width])
    return weight


@dataclass(frozen=True)
class SheetField:
    """Caption hiddens + a linear next-token sheet.

    Gender-like: dim 2, no leftover, no ê.
    Energy-like / leaky: dim 3, leftover on dim 2, ê = leftover axis,
    short û on dim 0.
    """

    dim: int = 2
    odd: float = ODD_SCALE
    even: float = EVEN_SCALE
    leftover: float = 0.0

    def short_u(self) -> torch.Tensor:
        return lm_unit(torch.tensor([1.0] + [0.0] * (self.dim - 1)))

    def leak_dir(self) -> torch.Tensor | None:
        if self.dim < 3 or abs(float(self.leftover)) < 1e-8:
            return None
        axis = torch.zeros(self.dim)
        axis[2] = 1.0
        return axis

    def weight(self) -> torch.Tensor:
        return readout_weight(self.dim)

    def poles(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        pos = torch.zeros(self.dim)
        neg = torch.zeros(self.dim)
        neu = torch.zeros(self.dim)
        pos[0] = float(self.odd)
        neg[0] = -float(self.odd)
        pos[1] = float(self.even)
        neg[1] = float(self.even)
        if self.dim >= 3:
            pos[2] = float(self.leftover)
            neg[2] = -float(self.leftover)
        return pos, neg, neu

    def pair_odd(self) -> torch.Tensor:
        pos, neg, _neu = self.poles()
        return (pos - neg) / 2.0

    def midpoints(self) -> tuple[torch.Tensor, torch.Tensor]:
        pos, neg, neu = self.poles()
        return lm_hidden_targets(pos, neg, neu, target_mode="symmetric")

    def cleaned_midpoints(self) -> tuple[torch.Tensor, torch.Tensor]:
        leak = self.leak_dir()
        pos, neg, neu = self.poles()
        if leak is None:
            return self.midpoints()
        return lm_pair_odd_sub_e(pos, neg, neu, leak, slider_dir=self.short_u())

    def cleaned_captions(self) -> tuple[torch.Tensor, torch.Tensor]:
        leak = self.leak_dir()
        pos, neg, neu = self.poles()
        if leak is None:
            return pos, neg
        return lm_e_cleaned_captions(pos, neg, neu, leak, slider_dir=self.short_u())


def gender_sheet() -> SheetField:
    """Clean pair, no ê. Live gender / v15 default path."""
    return SheetField(dim=2, leftover=0.0)


def energy_sheet() -> SheetField:
    """Leaky leftover in the odd teacher. Live energy ``pair_odd_sub_e``."""
    return SheetField(dim=3, leftover=LEAK_SCALE)


def policy_at(hidden: torch.Tensor, field: SheetField) -> dict[str, float | str | bool]:
    """Next-token policy of one hidden. The sheet metric lives here."""
    logits = lm_policy_logits(hidden, field.weight())
    probs = F.softmax(logits, dim=-1)
    index = int(torch.argmax(probs))
    sheet = sum(float(probs[i]) for i in SHEET_TOKENS if i < int(probs.numel()))
    ood = float(probs[TOKEN_OOD_POS]) + float(probs[TOKEN_OOD_NEG])
    leak = float(probs[TOKEN_LEAK]) if int(probs.numel()) > TOKEN_LEAK else 0.0
    return {
        "argmax": TOKEN_NAMES[index],
        "argmax_i": index,
        "on_sheet": index in SHEET_TOKENS,
        "on_sheet_mass": sheet,
        "ood_mass": ood,
        "leak_mass": leak,
        "lyric_mass": float(probs[TOKEN_LYRIC]),
        "pos_mass": float(probs[TOKEN_POS]),
        "neg_mass": float(probs[TOKEN_NEG]),
        "ood_rate": 0.0 if index in SHEET_TOKENS else 1.0,
    }


def field_geometry(field: SheetField) -> dict[str, float | str | bool]:
    """Properties of the field itself, before any student."""
    pos, neg, _neu = field.poles()
    t_plus, t_minus = field.midpoints()
    c_plus, c_minus = field.cleaned_captions()
    m_plus, m_minus = field.cleaned_midpoints()
    a = field.pair_odd()
    p_pos = policy_at(pos, field)
    p_neg = policy_at(neg, field)
    p_t = policy_at(t_plus, field)
    p_m = policy_at(m_plus, field)
    p_c = policy_at(c_plus, field)
    return {
        "cos_caption_to_mid": cosine(pos, t_plus),
        "cos_caption_to_odd": cosine(pos, a),
        "mid_argmax": str(p_t["argmax"]),
        "caption_argmax": str(p_pos["argmax"]),
        "mid_neq_caption": p_t["argmax"] != p_pos["argmax"],
        "mid_on_sheet": bool(p_t["on_sheet"]),
        "caption_on_sheet": bool(p_pos["on_sheet"]) and bool(policy_at(neg, field)["on_sheet"]),
        "mid_ood_mass": float(p_t["ood_mass"]),
        "caption_sheet_mass": float(p_pos["on_sheet_mass"]),
        "cleaned_mid_argmax": str(p_m["argmax"]),
        "cleaned_caption_argmax": str(p_c["argmax"]),
        "minus_caption_argmax": str(p_neg["argmax"]),
        "minus_mid_argmax": str(policy_at(t_minus, field)["argmax"]),
        "cleaned_minus_argmax": str(policy_at(c_minus, field)["argmax"]),
        "ray_argmax_stable": ray_argmax_stable(field),
    }


def ray_argmax_stable(field: SheetField, scales: tuple[float, ...] = (0.25, 1.0, 4.0)) -> bool:
    """Linear readout, no bias: argmax is constant along a ray (except 0)."""
    t_plus, _ = field.midpoints()
    names = {str(policy_at(float(s) * t_plus, field)["argmax"]) for s in scales}
    return len(names) == 1


def teachers(
    field: SheetField,
    *,
    pole_mode: str,
    teacher: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pole teachers for one loss.

    ``hidden``: the v15 targets — pair-odd / pair_odd_sub_e midpoints,
    or raw captions for the faithful control.
    ``semantic_kl``: real captions (gender) or ê-cleaned captions
    (leaky). The midpoint is never the KL teacher.
    """
    pos, neg, neu = field.poles()
    mode = str(pole_mode).strip().lower()
    kind = str(teacher).strip().lower()
    leak = field.leak_dir()
    if mode == "hidden":
        if kind == "faithful":
            return pos, neg
        if kind == "pair_odd_sub_e":
            if leak is None:
                raise ValueError("pair_odd_sub_e needs a leftover ê")
            return lm_pair_odd_sub_e(pos, neg, neu, leak, slider_dir=field.short_u())
        if kind != "pair_odd":
            raise ValueError(f"unknown hidden teacher {teacher!r}")
        return lm_hidden_targets(pos, neg, neu, target_mode="symmetric")
    if mode != "semantic_kl":
        raise ValueError(f"pole_mode must be hidden or semantic_kl, got {pole_mode!r}")
    if kind == "pair_odd_sub_e":
        if leak is None:
            raise ValueError("pair_odd_sub_e needs a leftover ê")
        return lm_e_cleaned_captions(pos, neg, neu, leak, slider_dir=field.short_u())
    if kind in ("pair_odd", "faithful"):
        return pos, neg
    raise ValueError(f"unknown kl teacher {teacher!r}")


def fit_sheet(
    field: SheetField,
    *,
    pole_mode: str,
    teacher: str = "pair_odd",
    student: str = "odd_even",
    steps: int = STEPS,
    lr: float = LR,
    seed: int = 0,
    window: int = WINDOW,
) -> tuple[Residual, list[dict], dict[str, float]]:
    pos, neg, neu = field.poles()
    tgt_plus, tgt_minus = teachers(field, pole_mode=pole_mode, teacher=teacher)
    torch.manual_seed(int(seed))
    residual = Residual.create(str(student), dim=int(field.dim))
    opt = torch.optim.Adam(residual.parameters(), lr=float(lr))
    weight = field.weight()
    history: list[dict] = []
    tail: list[dict] = []

    def step_loss() -> torch.Tensor:
        pred_plus = neu + residual.delta(1.0)
        pred_minus = neu + residual.delta(-1.0)
        if str(pole_mode).strip().lower() == "hidden":
            return lm_slider_loss(pred_plus, pred_minus, tgt_plus, tgt_minus)
        return lm_semantic_kl(pred_plus, pred_minus, tgt_plus, tgt_minus, weight)

    for i in range(int(steps) + 1):
        loss = step_loss()
        snap = fit_metrics(residual.snapshot(), field, loss=float(loss.detach()), teacher=teacher)
        if i % 100 == 0 or i == int(steps):
            history.append({"step": i, **snap})
        tail.append(snap)
        if len(tail) > int(window):
            tail.pop(0)
        if i == int(steps):
            break
        opt.zero_grad()
        loss.backward()
        opt.step()
    return residual.snapshot(), history, window_mean(tail)


def window_mean(snaps: list[dict]) -> dict[str, float]:
    if not snaps:
        raise ValueError("window_mean needs at least one snapshot")
    skip = {"argmax_plus", "argmax_minus"}
    keys = [k for k in snaps[0] if k not in skip and k != "step"]
    out = {key: sum(float(s[key]) for s in snaps) / len(snaps) for key in keys}
    out["argmax_plus"] = snaps[-1]["argmax_plus"]
    out["argmax_minus"] = snaps[-1]["argmax_minus"]
    return out


def fit_metrics(
    residual: Residual,
    field: SheetField,
    *,
    loss: float,
    teacher: str,
) -> dict:
    pos, _neg, neu = field.poles()
    a = field.pair_odd()
    d_plus = residual.delta(1.0)
    d_minus = residual.delta(-1.0)
    h_plus = neu + d_plus
    h_minus = neu + d_minus
    p_plus = policy_at(h_plus, field)
    p_minus = policy_at(h_minus, field)
    leak_dir = field.leak_dir()
    leftover = 0.0 if leak_dir is None else abs(float(d_plus @ lm_unit(leak_dir)))
    teacher_leftover = 0.0 if leak_dir is None else abs(float(a @ lm_unit(leak_dir)))
    mid_plus, _mid_minus = field.midpoints()
    if str(teacher).strip().lower() == "pair_odd_sub_e":
        score_axis = field.cleaned_midpoints()[0] - neu
    else:
        score_axis = a
    return {
        "loss": float(loss),
        "c_plus": cosine(d_plus, score_axis),
        "c_plus_pair_odd": cosine(d_plus, a),
        "c_minus": cosine(d_minus, -score_axis),
        "collapse": cosine(d_plus, d_minus),
        "cos_to_mid": cosine(h_plus, mid_plus),
        "cos_to_caption": cosine(h_plus, pos),
        "even_plus": float(d_plus[1]) if int(d_plus.numel()) > 1 else 0.0,
        "leftover": leftover,
        "leftover_teacher": teacher_leftover,
        "leftover_leak": leftover / (abs(float(d_plus[0])) + 1e-8),
        "on_sheet_mass": 0.5 * (float(p_plus["on_sheet_mass"]) + float(p_minus["on_sheet_mass"])),
        "on_sheet_plus": 1.0 if p_plus["on_sheet"] else 0.0,
        "on_sheet_minus": 1.0 if p_minus["on_sheet"] else 0.0,
        "ood_mass": 0.5 * (float(p_plus["ood_mass"]) + float(p_minus["ood_mass"])),
        "leak_mass": 0.5 * (float(p_plus["leak_mass"]) + float(p_minus["leak_mass"])),
        "lyric_mass": 0.5 * (float(p_plus["lyric_mass"]) + float(p_minus["lyric_mass"])),
        "ood_rate": 0.5 * (float(p_plus["ood_rate"]) + float(p_minus["ood_rate"])),
        "argmax_plus": str(p_plus["argmax"]),
        "argmax_minus": str(p_minus["argmax"]),
        **leftover_bipolar(d_plus, d_minus),
    }


def looks_locked(row: dict) -> bool:
    """The v15 early-stop look: high teacher cos, bipolar collapse.

    Gender hidden: teacher *is* pair-odd, so this is the 0.96 lock.
    ``pair_odd_sub_e`` hidden locks onto the ê-cleaned midpoint; raw
    pair-odd cos is then lower and is the wrong column for "did the
    fit hit its teacher".
    """
    return float(row["c_plus"]) >= LOOKS_LOCKED_COS and float(row["collapse"]) <= LOOKS_LOCKED_COLLAPSE


def stays_on_sheet(row: dict) -> bool:
    return (
        float(row["on_sheet_plus"]) >= 0.5
        and float(row["on_sheet_minus"]) >= 0.5
        and float(row["on_sheet_mass"]) >= ON_SHEET_MASS
    )


def sheet_verdicts(row: dict) -> dict[str, str]:
    on = stays_on_sheet(row)
    locked = looks_locked(row)
    return {
        "sheet": "right" if on else "needs_help",
        # Pair-odd lock is honest only when the student is on-sheet.
        # Hidden MSE that looks locked while off-sheet is the Goodhart.
        "lock_honest": "right" if (on or not locked) else "needs_help",
    }


def score_sheet(
    name: str,
    field: SheetField,
    *,
    pole_mode: str,
    teacher: str = "pair_odd",
    student: str = "odd_even",
    steps: int = STEPS,
    seed: int = 0,
    label: str = "",
) -> dict:
    residual, history, window = fit_sheet(
        field,
        pole_mode=pole_mode,
        teacher=teacher,
        student=student,
        steps=steps,
        seed=seed,
    )
    final = dict(history[-1])
    final.pop("step", None)
    row = dict(window)
    row.update({f"{key}_final": value for key, value in final.items()})
    geo = field_geometry(field)
    row.update(
        {
            "name": name,
            "label": label,
            "pole_mode": str(pole_mode),
            "teacher": str(teacher),
            "student": str(student),
            "dim": int(field.dim),
            "hold_weight": 0.0,
            "window": WINDOW,
            "history": history,
            "mid_argmax": geo["mid_argmax"],
            "caption_argmax": geo["caption_argmax"],
            "mid_neq_caption": geo["mid_neq_caption"],
            "ray_argmax_stable": geo["ray_argmax_stable"],
            "cos_caption_to_mid": geo["cos_caption_to_mid"],
        }
    )
    row["looks_locked"] = looks_locked(row)
    row["on_sheet"] = stays_on_sheet(row)
    row["goodhart"] = bool(row["looks_locked"] and not row["on_sheet"])
    row["axis"] = sheet_verdicts(row)
    row["pass"] = all(v == "right" for v in row["axis"].values())
    del residual
    return row


def cell_table(*, steps: int = STEPS, seed: int = 0) -> list[dict]:
    """Gender (no ê) and leaky energy, hidden vs semantic_kl."""
    gender = gender_sheet()
    energy = energy_sheet()
    specs = [
        (
            "gender_hidden_odd",
            gender,
            "hidden",
            "pair_odd",
            "odd",
            "v15 gender: odd LoRA, hidden MSE onto midpoint",
        ),
        (
            "gender_hidden_odd_even",
            gender,
            "hidden",
            "pair_odd",
            "odd_even",
            "v15 gender: even capacity, hidden MSE still drops even",
        ),
        (
            "gender_hidden_faithful_odd_even",
            gender,
            "hidden",
            "faithful",
            "odd_even",
            "control: hidden MSE onto real captions (not the midpoint)",
        ),
        (
            "gender_kl_odd",
            gender,
            "semantic_kl",
            "pair_odd",
            "odd",
            "v16 gender, odd LoRA: ray-stuck, still off-sheet",
        ),
        (
            "gender_kl_odd_even",
            gender,
            "semantic_kl",
            "pair_odd",
            "odd_even",
            "v16 gender: KL onto encode(pos)/encode(neg)",
        ),
        (
            "energy_hidden_pair_odd",
            energy,
            "hidden",
            "pair_odd",
            "odd_even",
            "leaky v15: hidden MSE onto raw pair-odd (keeps leftover)",
        ),
        (
            "energy_hidden_sub_e",
            energy,
            "hidden",
            "pair_odd_sub_e",
            "odd_even",
            "leaky v15: hidden MSE onto ê-cleaned midpoint",
        ),
        (
            "energy_kl_pair_odd",
            energy,
            "semantic_kl",
            "pair_odd",
            "odd_even",
            "leaky KL onto raw encode(pos)/encode(neg)",
        ),
        (
            "energy_kl_sub_e",
            energy,
            "semantic_kl",
            "pair_odd_sub_e",
            "odd_even",
            "v16 leaky: KL onto ê-cleaned real poles, hold 0",
        ),
    ]
    return [
        score_sheet(
            name,
            field,
            pole_mode=pole_mode,
            teacher=teacher,
            student=student,
            steps=steps,
            seed=seed,
            label=label,
        )
        for name, field, pole_mode, teacher, student, label in specs
    ]


def compact(row: dict) -> dict:
    skip = {"history"}
    return {k: v for k, v in row.items() if k not in skip}
