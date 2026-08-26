"""CPU-pure slider targets extracted from the live trainers.

The GPU trainers can import these later; the 2-D fixture already does.
Formulas are copied from:

- ``PromptEmbedsPair._enhance`` / ``_erase`` in ``prompt_util.py``
  (SD / XL / SD3 / Flux / Cascade noise-prediction sliders)
- ``_slider_loss`` and ``_target_delta`` in ``train_lora_music3.py``
- LM raw vs ``--symmetric`` targets in ``train_lm_slider_music3.py``
- Hub v9 LM recipe (sidecar ``target_mode`` / ``leakage_floor`` / ``anchor_*``)
  plus the leak fix. Default ``--lm_target v9`` is **full pair-odd**
  ``a = ½(h+−h−)``, ``t± = h0 ± a``, κ = 0, and a hold along a declared
  leak axis ``ê`` (YAML ``leak_positive`` / ``leak_negative``):
  penalize ``(h(±1)−h0) · ê_⊥`` with ``ê_⊥ = ê − (ê·û)û`` when a
  slider direction is declared. Opposite-energy leak captions
  (energy-v4 slammed/168/pop-punk vs airy/52/lullaby) overlap the
  slider and the pair-odd; holding raw ê punches the slider itself.
  Short ``slider_positive`` is a name / probe, not the teacher — do
  not replace ``a`` with ``(a·û)û``. If no ``ê`` is declared (clean
  pair, or ``attributes`` already pin the unused axis), hold is 0.
  ``--lm_target pair_odd_sub_e`` is the leaky-axis teacher: drop
  ``ê_⊥`` out of pair-odd first (the λ→∞ hold limit, no stiffness).
  Subtract ``ê_⊥``, not raw ê.   ``--lm_target faithful_sub_e`` is the
  same leftover odd on the real poles (midpoint stays ½(h++h−));
  it is not the default. ``--lm_target faithful_sub_e_if_unused``
  subtracts only when ``|ê̂_⊥ · â|`` is below ``UNUSED_E_OVERLAP_MAX``
  (leftover is actually unused); otherwise it keeps the raw poles.
  Gender stays ``v9`` with no ê / hold 0.
  ``--lm_target faithful_guard_e`` is the threshold-free sibling:
  subtract leftover ê only while the cleaned target stays nearer its
  own caption than the pair midpoint (``lm_blend_guard``). No
  ``leak_*`` → raw poles. ``--lm_target faithful_even_blend`` leftover-
  gates the odd part and subtracts ``EVEN_BLEND_SCALE`` of leak-pair
  even leftover (opt-in; default stays ``v9``). ``--pole_mode dual_band``
  is KL on the
  semantic band plus hidden MSE on the centered-readout blind band
  (``P_blind`` from SVD). Neither is the default.
  ``--lm_target v9_project`` is the old slider-level project+hold
  onto û; ``v9_always`` never gates.
- Encoder MSE in ``train_encoder_music3.py``

No Hub, no GPU, no model weights.
"""

from __future__ import annotations

from typing import Iterable, Sequence

import torch
import torch.nn.functional as F


# Live gender-v1 sat at 0.20; live energy rows sat at 0.48 and 0.68.
# A slider-level floor in that gap is one teacher for the whole LoRA.
# Per-row 0.50 splits energy (mixed teacher) and is not the default.
SLIDER_ALIGN_MIN = 0.50

# Pair-aware leftover gate. ``|ê̂_⊥ · â|`` is the share of the pair-odd
# axis that subtracting ê_⊥ would delete (ê_⊥ = ê − (ê·û)û).
# Measured on the pair-exam / sheet fixtures (seed-free geometry):
#   unused leftover   Field2D 0.324 / sheet 0.373 / unused_e 0.391
#   energy-v4 tracks  divergent 0.778  (ê restates pop-punk 168 vs lullaby 52)
# Subtract only below this floor. ``||ê_⊥||/||a||`` does not separate
# those clusters (unused ~0.87, divergent ~0.78) and is not the gate.
UNUSED_E_OVERLAP_MAX = 0.50

# Weight on the blind-band term of ``lm_dual_band_pole_loss``. The pair-exam
# cell is flat in this over 0.5 … 32 (six doublings), because the term's job
# is to supply a gradient where the KL has none at all, not to outweigh it.
DUAL_BAND_WEIGHT = 1.0
# Spectral cut for ``lm_blind_projector``. 0 is the exact null space of the
# readout, which is what a caricature head with an unread block has. A live
# ``lm_head`` band whose row space fills the hidden width has no exact null
# space and needs a cut > 0 — the directions it reads so weakly that a KL on
# its policy has no usable gradient there.
BLIND_SPECTRAL_CUT = 0.0

# Soft hold along ê fights the full-odd teacher. λ=1 leaves energy leak
# ~0.69 (odd·û ≈ 0.58 leftover). λ=8 is the first value that lands leak
# ≤ 0.20 on unused-ê; +/− same-dir stays ~0 (live-good band ≲ 6%).
# λ=8 is *not* safe on a raw ê that overlaps the slider — hold then
# punches û. The live path orthogonalizes ê to û first.
LEAK_HOLD_WEIGHT = 8.0
SAME_DIR_MAX = 0.06
HOLD_DIR_EPS = 1e-6
# Half the leak-pair even leftover. Scale 1.0 (drop all ê_even) fails
# exam_divergent on energy-v4; 0.25 … 0.90 still pass with leak_frac < 0.
# 0.5 is the named half-step: enough to cross zero, far from the α=1 fail.
EVEN_BLEND_SCALE = 0.5


def sd_noise_target(
    neutral: torch.Tensor,
    positive: torch.Tensor,
    negative: torch.Tensor,
    guidance: float,
    action: str = "enhance",
) -> torch.Tensor:
    """Teacher noise (or any same-shaped score) the SD slider fits.

    ``prompt_util.PromptEmbedsPair._enhance`` / ``_erase``:

        enhance:  neu + g * (pos - neg)
        erase:    neu - g * (pos - neg)
    """
    if action == "erase":
        return neutral - guidance * (positive - negative)
    if action == "enhance":
        return neutral + guidance * (positive - negative)
    raise ValueError("action must be erase or enhance")


def music3_axis_delta(
    direction: float,
    vel_pos: torch.Tensor,
    vel_neg: torch.Tensor,
    guidance: float,
    action: str = "enhance",
) -> torch.Tensor:
    """``_target_delta`` with ``--target_mode axis`` (Music 3 default).

    ``axis = sign * g * (vel_pos - vel_neg)``, then ``direction * axis``.
    ``sign`` is +1 for enhance, -1 for erase.
    """
    if action not in ("enhance", "erase"):
        raise ValueError(f"action must be enhance or erase, got {action!r}")
    sign = 1.0 if action == "enhance" else -1.0
    axis = sign * guidance * (vel_pos - vel_neg)
    return direction * axis


def music3_pole_delta(
    direction: float,
    vel_pos: torch.Tensor,
    vel_neg: torch.Tensor,
    vel_neu: torch.Tensor,
    guidance: float,
    action: str = "enhance",
) -> torch.Tensor:
    """``_target_delta`` with ``--target_mode pole``.

    Each slider sign aims at its own caption's displacement from neutral:
    ``abs(direction) * g * (nearer_pole - vel_neu)``.
    """
    if action not in ("enhance", "erase"):
        raise ValueError(f"action must be enhance or erase, got {action!r}")
    sign = 1.0 if action == "enhance" else -1.0
    toward_pos = (direction * sign) >= 0.0
    pole = vel_pos if toward_pos else vel_neg
    return abs(direction) * guidance * (pole - vel_neu)


def music3_slider_loss(
    vel: torch.Tensor,
    vel_neu: torch.Tensor,
    axis: torch.Tensor,
    kind: str,
    mag_weight: float,
    gain_weight: float = 0.0,
    gain_mode: str = "penalty",
    gain_tw: float = 1.0,
) -> torch.Tensor:
    """Verbatim extract of ``train_lora_music3._slider_loss``.

    ``axis`` is the signed target delta, ``guidance * (vel_pos - vel_neg)`` for +1.
    """
    delta = vel - vel_neu
    unit = vel_neu.flatten()
    unit = unit / unit.norm().clamp_min(1e-8)
    if gain_weight > 0.0:
        g_delta = delta.flatten() @ unit
        if gain_mode == "match":
            g_target = axis.flatten() @ unit
        elif gain_mode == "penalty":
            g_target = axis.new_zeros(())
        else:
            raise ValueError(f"unknown gain_mode {gain_mode!r}")
        gain = (g_delta - g_target) / axis.norm().clamp_min(1e-8)
        gain_term = gain_weight * float(gain_tw) * gain.pow(2)
    else:
        gain_term = 0.0
    if kind == "mse":
        return F.mse_loss(vel, vel_neu + axis) + gain_term
    if kind == "nmse":
        scale = axis.pow(2).mean().clamp_min(1e-8)
        return F.mse_loss(vel, vel_neu + axis) / scale + gain_term
    if kind == "nmse_ortho":
        d_par = delta.flatten() @ unit
        a_par = axis.flatten() @ unit
        d_perp = delta - (d_par * unit).view_as(delta)
        a_perp = axis - (a_par * unit).view_as(axis)
        scale = a_perp.pow(2).mean().clamp_min(1e-8)
        return (d_perp - a_perp).pow(2).mean() / scale + gain_term
    if kind == "cos":
        cos = F.cosine_similarity(
            delta.flatten().unsqueeze(0), axis.flatten().unsqueeze(0)
        ).squeeze()
        ratio = delta.norm() / axis.norm().clamp_min(1e-8)
        return (1.0 - cos) + mag_weight * (ratio - 1.0).pow(2) + gain_term
    raise ValueError(f"unknown loss {kind!r}")


def resolve_lm_target_mode(
    *,
    symmetric: bool = True,
    target_mode: str | None = None,
) -> str:
    """``faithful`` (v6 raw poles) or ``symmetric`` (v4/v9 polarity).

    Sidecar name is ``target_mode``. The live trainer still exposes
    ``--symmetric`` as the polarity step inside ``--lm_target v9``.
    """
    if target_mode is None:
        return "symmetric" if symmetric else "faithful"
    mode = str(target_mode).strip().lower()
    if mode not in ("faithful", "symmetric"):
        raise ValueError(f"target_mode must be faithful or symmetric, got {target_mode!r}")
    return mode


def lm_hidden_targets(
    pos: torch.Tensor,
    neg: torch.Tensor,
    neu: torch.Tensor,
    *,
    symmetric: bool = True,
    target_mode: str | None = None,
    target_scale: float = 1.0,
    common_beta: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """LM slider pole targets from ``train_lm_slider_music3.py``.

    ``target_mode symmetric`` (default, also ``--symmetric``):
        ``tgt(±1) = neu ± (pos-neg)/2 * target_scale + β * common``
    ``target_mode faithful`` (v6 raw, also ``--no-symmetric``):
        ``tgt(+1), tgt(-1) = pos, neg``
    """
    mode = resolve_lm_target_mode(symmetric=symmetric, target_mode=target_mode)
    if mode == "symmetric":
        axis = (pos - neg) / 2.0 * float(target_scale)
        common = (pos + neg) / 2.0 - neu
        return neu + axis + float(common_beta) * common, neu - axis + float(common_beta) * common
    if float(target_scale) != 1.0:
        raise ValueError("--target_scale is defined for symmetric targets only")
    return pos, neg


def lm_pair_collapse(pos: torch.Tensor, neg: torch.Tensor, neu: torch.Tensor) -> torch.Tensor:
    """``r = cos(h+ − h0, h− − h0)``. Raw even-mode collapse from the Hub v9 note."""
    return F.cosine_similarity(
        (pos - neu).flatten().unsqueeze(0),
        (neg - neu).flatten().unsqueeze(0),
    ).squeeze()


def lm_anchor_kappa(
    pos: torch.Tensor,
    neg: torch.Tensor,
    neu: torch.Tensor,
    leakage_floor: float,
    *,
    autocal: bool = True,
    kappa: float | None = None,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Largest κ ∈ [0, 1] whose perfect-fit ±1 cosine stays ≤ ``leakage_floor``.

    Hub v9 / sidecar ``--anchor_autocal --leakage_floor``:

        r = cos(h+ − h0, h− − h0)
        ρ² = (1 − r) / (1 + r)
        a perfect fit on the blend realizes cos(d+, d−) = (κ² − ρ²) / (κ² + ρ²)
        κ = √(ρ² · (1 + floor) / (1 − floor))   clamped to [0, 1]

    This sizes the *even* blend back toward raw ``h±``. It does not touch the
    odd teacher ``(pos − neg) / 2``, so unused-attribute leak in that axis
    is unchanged.
    """
    if not autocal:
        value = 1.0 if kappa is None else float(kappa)
        return pos.new_tensor(min(1.0, max(0.0, value)))
    floor = float(leakage_floor)
    if floor >= 1.0:
        return pos.new_tensor(1.0)
    if floor <= -1.0:
        return pos.new_tensor(0.0)
    r = lm_pair_collapse(pos, neg, neu).clamp(-1.0 + eps, 1.0 - eps)
    rho2 = (1.0 - r) / (1.0 + r)
    raw = (rho2 * (1.0 + floor) / (1.0 - floor)).clamp(min=0.0)
    return raw.sqrt().clamp(0.0, 1.0)


def lm_anchor_targets(
    pos: torch.Tensor,
    neg: torch.Tensor,
    neu: torch.Tensor,
    kappa: torch.Tensor | float,
    *,
    target_scale: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """``anchor± = (1 − κ)(h0 ± a) + κ h±`` with ``a = (pos − neg) / 2``."""
    axis = (pos - neg) / 2.0 * float(target_scale)
    k = kappa if torch.is_tensor(kappa) else pos.new_tensor(float(kappa))
    return (1.0 - k) * (neu + axis) + k * pos, (1.0 - k) * (neu - axis) + k * neg


def lm_perfect_fit_collapse(
    kappa: torch.Tensor | float,
    rho2: torch.Tensor | float,
) -> torch.Tensor:
    """``cos(d+, d−) = (κ² − ρ²) / (κ² + ρ²)`` for a perfect fit on the blend."""
    k2 = kappa * kappa if torch.is_tensor(kappa) else kappa ** 2
    r2 = rho2 if torch.is_tensor(rho2) else torch.tensor(float(rho2))
    k2_t = k2 if torch.is_tensor(k2) else r2.new_tensor(float(k2))
    return (k2_t - r2) / (k2_t + r2).clamp_min(1e-8)


def lm_unit(direction: torch.Tensor) -> torch.Tensor:
    flat = direction.flatten()
    return flat / flat.norm().clamp_min(1e-8)


def lm_odd_align(
    pos: torch.Tensor,
    neg: torch.Tensor,
    slider_dir: torch.Tensor,
) -> torch.Tensor:
    """|odd · û| / ||odd|| with ``odd = (pos − neg) / 2`` and û unit(slider_dir).

    Live gender-v1 logged 0.20 on the short declared captions. The energetic
    2-D leak cell is ~0.95 because û is the pole polarity (energetic↔calm).
    This is *not* a cosine of d+ after the fit — it is the teacher overlap
    that decides how much of the pair project-odd will keep.
    """
    odd = ((pos - neg) / 2.0).flatten()
    unit = lm_unit(slider_dir)
    return (odd @ unit).abs() / odd.norm().clamp_min(1e-8)


def lm_should_project_odd(
    pos: torch.Tensor,
    neg: torch.Tensor,
    slider_dir: torch.Tensor,
    project_align_min: float | None,
) -> tuple[bool, torch.Tensor]:
    """Whether one row should project the odd teacher onto û.

    ``project_align_min is None`` is always-project (``v9_always``).
    Otherwise this is the *per-row* floor. Live energy at 0.48 / 0.68
    splits on a hard 0.50 row gate — do not use this as the default.
    Prefer ``lm_project_decisions`` with ``scope="slider"``. The trainer
    must also drop the orthogonal hold when this returns False —
    holding ⊥ the rejected û still eats the pair.
    """
    align = lm_odd_align(pos, neg, slider_dir)
    if project_align_min is None:
        return True, align
    return bool(float(align) >= float(project_align_min)), align


def lm_mean_odd_align(
    pairs: Sequence[tuple[torch.Tensor, torch.Tensor]],
    slider_dir: torch.Tensor,
) -> torch.Tensor:
    """Mean ``|odd·û|/||odd||`` across every (pos, neg) row of one slider."""
    if not pairs:
        raise ValueError("lm_mean_odd_align needs at least one (pos, neg) pair")
    aligns = [lm_odd_align(pos, neg, slider_dir) for pos, neg in pairs]
    stacked = torch.stack([a.reshape(()) for a in aligns])
    return stacked.mean()


def lm_project_decisions(
    aligns: Sequence[float | torch.Tensor],
    project_align_min: float | None,
    scope: str = "slider",
) -> list[bool]:
    """Per-row project+hold decisions from a list of ``|odd·û|/||odd||``.

    ``project_align_min is None`` → always project (old v9).
    ``scope="slider"`` → one decision from the mean (live default).
    ``scope="row"`` → hard per-row floor (v12; mixes live energy).
    """
    if not aligns:
        raise ValueError("lm_project_decisions needs at least one align")
    values = [float(a) for a in aligns]
    if project_align_min is None:
        return [True] * len(values)
    floor = float(project_align_min)
    mode = str(scope).strip().lower()
    if mode == "slider":
        mean = sum(values) / len(values)
        return [mean >= floor] * len(values)
    if mode == "row":
        return [v >= floor for v in values]
    raise ValueError(f"project_align_scope must be 'slider' or 'row', got {scope!r}")


def lm_teachers_mixed(decisions: Sequence[bool]) -> bool:
    """True when some rows project and others fall back (mixed teacher)."""
    kinds = {bool(d) for d in decisions}
    return len(kinds) > 1


def lm_pair_odd_sub_e(
    pos: torch.Tensor,
    neg: torch.Tensor,
    neu: torch.Tensor,
    leak_dir: torch.Tensor,
    *,
    slider_dir: torch.Tensor,
    target_scale: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pair-odd teacher with the hold axis removed.

        a = (pos − neg) / 2
        ê_⊥ = ê − (ê·û)û          # same axis ``lm_axis_hold`` uses
        â = a − (a · ê̂_⊥) ê̂_⊥
        tgt(±1) = neu ± â · target_scale

    This is the λ→∞ hold equilibrium in one step: no ``λ·D/2``
    stiffness. Subtract ``ê_⊥``, not raw ê — raw ê takes û with it.
    If ê is already parallel to û, ``ê_⊥`` vanishes and the teacher
    stays full pair-odd (hold would have been off).
    """
    axis = (pos - neg) / 2.0 * float(target_scale)
    held = lm_hold_dir(leak_dir, slider_dir=slider_dir, mode="slider")
    if held is not None:
        unit = lm_unit(held)
        axis = axis - ((axis.flatten() @ unit) * unit).view_as(axis)
    return neu + axis, neu - axis


def lm_faithful_sub_e(
    pos: torch.Tensor,
    neg: torch.Tensor,
    neu: torch.Tensor,
    leak_dir: torch.Tensor,
    *,
    slider_dir: torch.Tensor,
    target_scale: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Real caption poles with leftover ê removed from the odd part only.

        a = (pos − neg) / 2
        ê_⊥ = ê − (ê·û)û          # same leftover geometry as pair_odd_sub_e
        â = a − (a · ê̂_⊥) ê̂_⊥
        mid = ½(pos + neg)
        tgt(±1) = mid ± â · target_scale

    Midpoint stays ½(h++h−). Teacher is a real caption minus leftover
    unused, not ``t± = h0 ± a``. ``pair_odd_sub_e`` is the
    midpoint-minus-ê teacher (v15) and is not this function — it is
    this leftover odd plus ``h0`` instead of ``mid``.
    """
    plus, minus = lm_pair_odd_sub_e(
        pos, neg, neu, leak_dir, slider_dir=slider_dir, target_scale=target_scale
    )
    common = (pos + neg) / 2.0 - neu
    return plus + common, minus + common


def lm_e_overlap_a(
    pos: torch.Tensor,
    neg: torch.Tensor,
    leak_dir: torch.Tensor,
    *,
    slider_dir: torch.Tensor | None = None,
) -> torch.Tensor:
    """``|ê̂_⊥ · â|`` — share of ``a`` that subtracting leftover ê would drop.

    ``a = (pos − neg) / 2``, ``â = a/||a||``. ``ê̂`` is the hold/subtract
    direction: ``ê_⊥ = ê − (ê·û)û`` when ``slider_dir`` is set, else raw ê.
    When ê is already parallel to û the leftover vanishes and this is 0.
    """
    axis = ((pos - neg) / 2.0).flatten()
    held = lm_hold_dir(
        leak_dir,
        slider_dir=slider_dir,
        mode="slider" if slider_dir is not None else "raw",
    )
    if held is None:
        return axis.new_tensor(0.0)
    return (lm_unit(held) @ lm_unit(axis)).abs()


def lm_e_is_unused(
    pos: torch.Tensor,
    neg: torch.Tensor,
    leak_dir: torch.Tensor,
    *,
    slider_dir: torch.Tensor | None = None,
    floor: float = UNUSED_E_OVERLAP_MAX,
) -> tuple[bool, torch.Tensor]:
    """True when leftover ê is unused, not a restatement of the pair.

    Subtract (or hold) ê only then. ``floor`` is ``UNUSED_E_OVERLAP_MAX``.
    """
    overlap = lm_e_overlap_a(pos, neg, leak_dir, slider_dir=slider_dir)
    return bool(float(overlap) < float(floor)), overlap


def lm_mean_e_overlap_a(
    pairs: Sequence[tuple[torch.Tensor, torch.Tensor]],
    leak_dir: torch.Tensor,
    *,
    slider_dir: torch.Tensor | None = None,
) -> torch.Tensor:
    """Mean ``|ê̂_⊥ · â|`` across every (pos, neg) row of one slider."""
    if not pairs:
        raise ValueError("lm_mean_e_overlap_a needs at least one (pos, neg) pair")
    overlaps = [lm_e_overlap_a(pos, neg, leak_dir, slider_dir=slider_dir) for pos, neg in pairs]
    stacked = torch.stack([o.reshape(()) for o in overlaps])
    return stacked.mean()


def lm_e_unused_decision(
    overlaps: Sequence[float | torch.Tensor],
    floor: float = UNUSED_E_OVERLAP_MAX,
) -> bool:
    """Slider-level leftover gate: one decision from the mean ``|ê̂_⊥ · â|``."""
    if not overlaps:
        raise ValueError("lm_e_unused_decision needs at least one overlap")
    mean = sum(float(o) for o in overlaps) / len(overlaps)
    return mean < float(floor)


def lm_faithful_sub_e_if_unused(
    pos: torch.Tensor,
    neg: torch.Tensor,
    neu: torch.Tensor,
    leak_dir: torch.Tensor | None = None,
    *,
    slider_dir: torch.Tensor | None = None,
    target_scale: float = 1.0,
    unused: bool | None = None,
    floor: float = UNUSED_E_OVERLAP_MAX,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Raw poles, or ê-cleaned poles only when leftover ê is unused.

    No declared ê → ``(pos, neg)``. ``|ê̂_⊥ · â|`` at or above ``floor``
    means ê restates the pair (energy-v4 genre/BPM) → keep the poles.
    Below the floor, leftover is unused (same-song unused gender) →
    ``lm_faithful_sub_e``. ``unused`` is the already-resolved slider-level
    decision when the caller gated on the mean overlap.
    """
    if leak_dir is None:
        return pos, neg
    if unused is None:
        unused, _overlap = lm_e_is_unused(
            pos, neg, leak_dir, slider_dir=slider_dir, floor=floor
        )
    if not unused:
        return pos, neg
    if slider_dir is None:
        raise ValueError(
            "faithful_sub_e_if_unused subtracts ê_⊥ = ê−(ê·û)û; "
            "needs a declared slider_dir (do not subtract raw ê)"
        )
    return lm_faithful_sub_e(
        pos, neg, neu, leak_dir, slider_dir=slider_dir, target_scale=target_scale
    )


def lm_blend_guard(
    tgt_plus: torch.Tensor,
    tgt_minus: torch.Tensor,
    pos: torch.Tensor,
    neg: torch.Tensor,
) -> dict[str, float | bool]:
    """Is a target still nearer its own caption than the pair's midpoint?

        mid = ½(pos + neg)
        to_pole = max(‖t₊ − pos‖, ‖t₋ − neg‖)
        to_mid  = min(‖t₊ − mid‖, ‖t₋ − mid‖)
        admissible ⟺ to_pole < to_mid

    No threshold is chosen here. ``mid`` is the one point on the segment
    that is not either caption, and on a divergent pair it is a state no
    caption occupies at all — half of each song. A teacher that has drifted
    closer to ``mid`` than to the pole it claims to be is a *blend* teacher,
    and the ±1 ends of a blend sing both songs at once.

    This is a test on the target **point**, so it can be run at setup from
    the four declared captions, before a single step. It says nothing about
    the loss.
    """
    mid = 0.5 * (pos + neg)
    to_pole = max(
        float((tgt_plus - pos).norm()), float((tgt_minus - neg).norm())
    )
    to_mid = min(float((tgt_plus - mid).norm()), float((tgt_minus - mid).norm()))
    return {
        "to_pole": to_pole,
        "to_mid": to_mid,
        "admissible": bool(to_pole < to_mid),
    }


def lm_faithful_guard_e(
    pos: torch.Tensor,
    neg: torch.Tensor,
    neu: torch.Tensor,
    leak_dir: torch.Tensor,
    *,
    slider_dir: torch.Tensor,
    target_scale: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """``lm_faithful_sub_e`` when the blend guard admits it, else raw poles.

    ``faithful_sub_e`` is right when ``ê`` names a leftover the captions
    left unpinned and wrong when ``ê`` restates the pole difference: on
    energy-v4 the declared pair (``"Pop-punk mix, BPM 168."`` /
    ``"Ambient lullaby mix, BPM 52."``) is the same genre and BPM the poles
    move, so subtracting ``ê_⊥`` deletes the slider and the target lands
    nearer ``½(h₊+h₋)`` than the caption. :func:`lm_blend_guard` is exactly
    that condition, so this teacher subtracts ``ê_⊥`` only while what is
    left of the axis is longer than what was taken, and otherwise keeps the
    caption it was already aiming at.

    One declaration, both pair types: a yaml may keep its ``leak_*`` pair
    without that pair being able to eat the axis. The guard is per row, so
    ``lm_teachers_mixed`` can report a yaml whose rows disagree.
    """
    plus, minus = lm_faithful_sub_e(
        pos, neg, neu, leak_dir, slider_dir=slider_dir, target_scale=target_scale
    )
    if lm_blend_guard(plus, minus, pos, neg)["admissible"]:
        return plus, minus
    if float(target_scale) != 1.0:
        raise ValueError("--target_scale is defined for symmetric targets only")
    return pos, neg


def lm_even_residual(
    pos: torch.Tensor,
    neg: torch.Tensor,
    neu: torch.Tensor,
) -> torch.Tensor:
    """``c = ½((h₊−h0) + (h₋−h0))`` — the even residual."""
    return 0.5 * ((pos - neu) + (neg - neu))


def lm_even_leftover_dir(
    leak_pos: torch.Tensor,
    leak_neg: torch.Tensor,
    neu: torch.Tensor,
    *,
    slider_dir: torch.Tensor | None = None,
) -> torch.Tensor | None:
    """Even leftover of a declared leak pair.

    ``ê`` is ``leak₊ − leak₋`` (odd). The even counterpart is
    ``ê_even = ½((leak₊−h0) + (leak₋−h0))`` — half of each leak caption
    at once. On energy-v4 that is the blend of the two mix/BPM fragments.
    Orthogonalize to ``û`` when a slider direction is declared, same as
    ``ê_⊥``. Near-zero leftover returns ``None``.
    """
    even = lm_even_residual(leak_pos, leak_neg, neu)
    if slider_dir is not None:
        return lm_hold_dir(even, slider_dir=slider_dir, mode="slider")
    if float(even.norm()) <= HOLD_DIR_EPS:
        return None
    return even


def lm_subtract_even_dir(
    pos: torch.Tensor,
    neg: torch.Tensor,
    neu: torch.Tensor,
    direction: torch.Tensor | None,
    *,
    scale: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Keep the odd teacher; drop only the even component along ``direction``.

        c  = ½((h₊−h0) + (h₋−h0))
        c' = c − scale · (c · d̂) d̂
        t± = h0 + c' ± a

    ``scale = 1`` removes that leftover even. This is not ``t± = h0 ± a``:
    even orthogonal to ``d`` (shared song / caption sheet) stays. A missing
    or near-zero direction is a no-op.
    """
    if direction is None or abs(float(scale)) <= 1e-12:
        return pos, neg
    unit = lm_unit(direction)
    even = lm_even_residual(pos, neg, neu)
    coeff = even.flatten() @ unit
    if float(coeff.abs()) <= HOLD_DIR_EPS:
        return pos, neg
    drop = float(scale) * (coeff * unit).view_as(even)
    return pos - drop, neg - drop


def lm_faithful_sub_even_e(
    pos: torch.Tensor,
    neg: torch.Tensor,
    neu: torch.Tensor,
    leak_dir: torch.Tensor,
    *,
    slider_dir: torch.Tensor,
    target_scale: float = 1.0,
    scale: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Raw-pole odd; leftover ``ê_⊥`` subtracted from even only.

        ê_⊥ = ê − (ê·û)û
        c'  = c − scale · (c · ê̂_⊥) ê̂_⊥
        t±  = h0 + c' ± a

    Leftover-gate / ``faithful_sub_e`` subtract unused ê from the *odd*
    part and leave even alone — that is why they still sit at
    ``leak_frac > 0``. This is the even sibling. Does not delete all of
    ``c`` and is not ``t± = h0 ± a``.
    """
    if float(target_scale) != 1.0:
        raise ValueError("--target_scale is defined for symmetric targets only")
    held = lm_hold_dir(leak_dir, slider_dir=slider_dir, mode="slider")
    return lm_subtract_even_dir(pos, neg, neu, held, scale=scale)


def lm_faithful_sub_even_e_if_unused(
    pos: torch.Tensor,
    neg: torch.Tensor,
    neu: torch.Tensor,
    leak_dir: torch.Tensor | None = None,
    *,
    slider_dir: torch.Tensor | None = None,
    target_scale: float = 1.0,
    unused: bool | None = None,
    floor: float = UNUSED_E_OVERLAP_MAX,
    scale: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Subtract even leftover ê only when leftover ê is unused.

    Same unused floor as ``faithful_sub_e_if_unused``. Energy-v4 restates
    the tracks (overlap 0.778) → keep the poles. Unused leftover
    (0.32–0.39) → drop even-along-ê_⊥.
    """
    if leak_dir is None:
        return pos, neg
    if unused is None:
        unused, _overlap = lm_e_is_unused(
            pos, neg, leak_dir, slider_dir=slider_dir, floor=floor
        )
    if not unused:
        return pos, neg
    if slider_dir is None:
        raise ValueError(
            "faithful_sub_even_e_if_unused subtracts ê_⊥ from even; "
            "needs a declared slider_dir"
        )
    return lm_faithful_sub_even_e(
        pos,
        neg,
        neu,
        leak_dir,
        slider_dir=slider_dir,
        target_scale=target_scale,
        scale=scale,
    )


def lm_faithful_sub_even_e_guard(
    pos: torch.Tensor,
    neg: torch.Tensor,
    neu: torch.Tensor,
    leak_dir: torch.Tensor,
    *,
    slider_dir: torch.Tensor,
    target_scale: float = 1.0,
    scale: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """``lm_faithful_sub_even_e`` only while the blend guard admits it."""
    plus, minus = lm_faithful_sub_even_e(
        pos,
        neg,
        neu,
        leak_dir,
        slider_dir=slider_dir,
        target_scale=target_scale,
        scale=scale,
    )
    if lm_blend_guard(plus, minus, pos, neg)["admissible"]:
        return plus, minus
    if float(target_scale) != 1.0:
        raise ValueError("--target_scale is defined for symmetric targets only")
    return pos, neg


def lm_faithful_sub_even_blend(
    pos: torch.Tensor,
    neg: torch.Tensor,
    neu: torch.Tensor,
    even_dir: torch.Tensor | None,
    *,
    scale: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Subtract the leak-pair even leftover; keep caption even orthogonal to it.

    ``even_dir`` is ``lm_even_leftover_dir(leak₊, leak₋, h0)`` — the blend
    of the two leak captions, not odd ê. Missing ``even_dir`` is a no-op.
    """
    return lm_subtract_even_dir(pos, neg, neu, even_dir, scale=scale)


def lm_faithful_sub_even_blend_if_unused(
    pos: torch.Tensor,
    neg: torch.Tensor,
    neu: torch.Tensor,
    leak_dir: torch.Tensor | None,
    even_dir: torch.Tensor | None,
    *,
    slider_dir: torch.Tensor | None = None,
    unused: bool | None = None,
    floor: float = UNUSED_E_OVERLAP_MAX,
    scale: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Subtract leak-pair even leftover only when odd ê is unused leftover."""
    if leak_dir is None or even_dir is None:
        return pos, neg
    if unused is None:
        unused, _overlap = lm_e_is_unused(
            pos, neg, leak_dir, slider_dir=slider_dir, floor=floor
        )
    if not unused:
        return pos, neg
    return lm_faithful_sub_even_blend(pos, neg, neu, even_dir, scale=scale)


def lm_faithful_sub_even_blend_guard(
    pos: torch.Tensor,
    neg: torch.Tensor,
    neu: torch.Tensor,
    even_dir: torch.Tensor | None,
    *,
    scale: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Subtract leak-pair even leftover only while the blend guard admits it."""
    plus, minus = lm_faithful_sub_even_blend(pos, neg, neu, even_dir, scale=scale)
    if even_dir is None or lm_blend_guard(plus, minus, pos, neg)["admissible"]:
        return plus, minus
    return pos, neg


def lm_faithful_gate_odd_sub_even(
    pos: torch.Tensor,
    neg: torch.Tensor,
    neu: torch.Tensor,
    leak_dir: torch.Tensor | None,
    *,
    slider_dir: torch.Tensor | None = None,
    even_dir: torch.Tensor | None = None,
    unused: bool | None = None,
    floor: float = UNUSED_E_OVERLAP_MAX,
    scale: float = 1.0,
    target_scale: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Leftover-gate the odd part, then subtract leftover even.

    Odd: ``faithful_sub_e_if_unused`` (ê_⊥ out of ``a`` only when unused).
    Even: drop ``even_dir`` if given, else even-along-ê_⊥. Caption even
    orthogonal to that direction stays.
    """
    plus, minus = lm_faithful_sub_e_if_unused(
        pos,
        neg,
        neu,
        leak_dir,
        slider_dir=slider_dir,
        target_scale=target_scale,
        unused=unused,
        floor=floor,
    )
    direction = even_dir
    if direction is None and leak_dir is not None and slider_dir is not None:
        direction = lm_hold_dir(leak_dir, slider_dir=slider_dir, mode="slider")
    return lm_subtract_even_dir(plus, minus, neu, direction, scale=scale)


def lm_even_axis_hold(
    pred_plus: torch.Tensor,
    pred_minus: torch.Tensor,
    neu: torch.Tensor,
    leak_dir: torch.Tensor,
) -> torch.Tensor:
    """Hold the *even* residual along ``ê`` only.

    ``lm_axis_hold`` penalizes ``(h(±1)−h0)·ê`` on each pole — odd and even
    leftover together. This penalizes ``(c · ê̂)²`` with
    ``c = ½((h₊−h0)+(h₋−h0))``, so leftover even is held and leftover odd
    is not. Teacher stays whatever the caller set (usually the captions).
    """
    unit = lm_unit(leak_dir)
    even = lm_even_residual(pred_plus, pred_minus, neu).flatten()
    return (even @ unit).pow(2)


def lm_project_odd_axis(
    pos: torch.Tensor,
    neg: torch.Tensor,
    neu: torch.Tensor,
    slider_dir: torch.Tensor,
    *,
    target_scale: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Symmetric poles whose odd teacher is only the declared slider component.

        a = (pos − neg) / 2
        â = (a · û) û          # drop everything orthogonal to slider_dir
        tgt(±1) = neu ± â · target_scale

    ``slider_dir`` is the *declared* concept axis (energetic↔calm on the 2-D
    field), not an attributes-prefixed caption. Unused-gender leak lives in
    ``a`` and is dropped here. The targets stay odd around ``neu`` (collapse
    −1). Hub ``leakage_floor`` / κ cannot do this — they only size even
    blend-back toward raw ``h±``.
    """
    axis = (pos - neg) / 2.0 * float(target_scale)
    unit = lm_unit(slider_dir)
    projected = (axis.flatten() @ unit) * unit
    return neu + projected.view_as(axis), neu - projected.view_as(axis)


def lm_ortho_hold(
    pred_plus: torch.Tensor,
    pred_minus: torch.Tensor,
    neu: torch.Tensor,
    slider_dir: torch.Tensor,
) -> torch.Tensor:
    """Mean-squared residual orthogonal to the declared slider direction."""
    unit = lm_unit(slider_dir)

    def _ortho(pred: torch.Tensor) -> torch.Tensor:
        delta = (pred - neu).flatten()
        return delta - (delta @ unit) * unit

    return 0.5 * (_ortho(pred_plus).pow(2).mean() + _ortho(pred_minus).pow(2).mean())


def lm_hold_dir(
    leak_dir: torch.Tensor,
    *,
    slider_dir: torch.Tensor | None = None,
    odd_dir: torch.Tensor | None = None,
    mode: str = "raw",
) -> torch.Tensor | None:
    """Direction the hold actually penalizes.

    ``raw``: declared ê.
    ``slider``: ê_⊥ = ê − (ê·û)û. Hold cannot punch the slider name.
    ``odd``: ê_⊥ = ê − (ê·â)â. Hold cannot punch the pair-odd teacher.
    Near-zero leftover returns ``None`` (hold is off). Missing ``slider_dir``
    / ``odd_dir`` for that mode falls back to raw ê — do not invent an axis.
    """
    kind = str(mode).strip().lower()
    axis = leak_dir.flatten()
    if kind == "raw":
        out = axis
    elif kind == "slider":
        if slider_dir is None:
            out = axis
        else:
            unit = lm_unit(slider_dir)
            out = axis - (axis @ unit) * unit
    elif kind in ("odd", "pair_odd", "teacher"):
        if odd_dir is None:
            out = axis
        else:
            unit = lm_unit(odd_dir)
            out = axis - (axis @ unit) * unit
    else:
        raise ValueError(f"hold dir mode must be raw/slider/odd, got {mode!r}")
    if float(out.norm()) <= HOLD_DIR_EPS:
        return None
    return out.reshape_as(leak_dir) if out.numel() == leak_dir.numel() else out


def lm_axis_hold(
    pred_plus: torch.Tensor,
    pred_minus: torch.Tensor,
    neu: torch.Tensor,
    leak_dir: torch.Tensor,
) -> torch.Tensor:
    """Mean-squared residual *along* a declared leak axis ``ê``.

    Penalize ``(h(±1) − h0) · ê``. Teacher stays the full pair-odd; this
    term is what should eat unused mix / BPM / genre inside ``a``. Short
    ``slider_positive`` is not this axis — do not hold ``û_⊥``.
    """
    unit = lm_unit(leak_dir)

    def _along(pred: torch.Tensor) -> torch.Tensor:
        delta = (pred - neu).flatten()
        return (delta @ unit).pow(2)

    return 0.5 * (_along(pred_plus) + _along(pred_minus))


def leftover_bipolar(d_plus: torch.Tensor, d_minus: torch.Tensor) -> dict[str, float]:
    """+/− leftover: ``leak_frac = cos(d+, d−)``, ``same_dir`` = even / (even+odd).

    Live Hub / pair-symmetric sits around leak_frac ≈ −0.94 and same-dir 2–4%.
    This is a bipolar check, not the unused-attr leak we optimize.
    """
    even = 0.5 * (d_plus + d_minus)
    odd = 0.5 * (d_plus - d_minus)
    even_n = float(even.norm())
    odd_n = float(odd.norm())
    return {
        "leak_frac": float(
            F.cosine_similarity(
                d_plus.flatten().unsqueeze(0), d_minus.flatten().unsqueeze(0)
            ).squeeze()
        ),
        "same_dir": even_n / (even_n + odd_n + 1e-8),
        "even_norm": even_n,
        "odd_norm": odd_n,
    }


def lm_slider_loss(
    pred_plus: torch.Tensor,
    pred_minus: torch.Tensor,
    tgt_plus: torch.Tensor,
    tgt_minus: torch.Tensor,
    *,
    anchor_plus: torch.Tensor | None = None,
    anchor_minus: torch.Tensor | None = None,
    anchor_weight: float = 0.0,
    hold: torch.Tensor | None = None,
    hold_weight: float = 0.0,
) -> torch.Tensor:
    """Pole MSE plus optional v9 anchor MSE and orthogonal hold.

    Endreg / planreg / collapse_weight are AR-only and are not expressed on
    the CPU field. Semantic-KL poles are ``lm_semantic_pole_loss``; they
    need a readout, which ``analysis/slider2d/sheet.py`` supplies.
    """
    pole = F.mse_loss(pred_plus, tgt_plus) + F.mse_loss(pred_minus, tgt_minus)
    weight = float(anchor_weight)
    if weight > 0.0 and anchor_plus is not None and anchor_minus is not None:
        pole = pole + weight * (
            F.mse_loss(pred_plus, anchor_plus) + F.mse_loss(pred_minus, anchor_minus)
        )
    hold_w = float(hold_weight)
    if hold_w > 0.0 and hold is not None:
        pole = pole + hold_w * hold
    return pole


def lm_next_token_logits(
    hidden: torch.Tensor,
    readout: torch.Tensor,
    *,
    bias: torch.Tensor | None = None,
) -> torch.Tensor:
    """``hidden @ readout.T`` — next-token logits from a prompt-last hidden.

    ``readout`` is the slice of ``lm_head.weight`` that decides what the AR
    model says next. Live that is the semantic-code band
    ``weight[_AUDIO_CODE_OFFSET : + _SEMANTIC_VOCAB_SIZE]`` — the rows
    ``_frame_margins`` already multiplies out for the end margin. The pole
    hiddens the LM slider fits are exactly the states this head reads, so
    this is the only place the "does the target say real words" question can
    be asked at all.
    """
    logits = hidden @ readout.transpose(-1, -2)
    if bias is not None:
        logits = logits + bias
    return logits


def lm_semantic_kl(pred_logits: torch.Tensor, tgt_logits: torch.Tensor) -> torch.Tensor:
    """``KL(teacher ‖ student)`` on next-token policies.

    Teacher first: the target caption's policy is the reference, so mass it
    puts nowhere costs the student nothing and mass it puts somewhere the
    student misses is paid in full. Reverse KL would let the student drop
    the caption's rarer real tokens.

    This term only sees the readout's row space. Hidden MSE also pins the
    head's null space, which on a Music 3 hidden state is most of the
    width — pole content that cannot change a single token.
    """
    return F.kl_div(
        F.log_softmax(pred_logits, dim=-1),
        F.log_softmax(tgt_logits, dim=-1),
        log_target=True,
        reduction="batchmean",
    )


def lm_semantic_pole_loss(
    pred_plus_logits: torch.Tensor,
    pred_minus_logits: torch.Tensor,
    tgt_plus_logits: torch.Tensor,
    tgt_minus_logits: torch.Tensor,
    *,
    hold: torch.Tensor | None = None,
    hold_weight: float = 0.0,
) -> torch.Tensor:
    """``lm_slider_loss`` with the pole MSE replaced by semantic KL.

    The targets must be logits of a hidden state some real caption
    actually produces — ``encode(positive)``, or that pole with a declared
    ê subtracted. Feeding the pair-odd midpoint ``h0 ± ½(h+−h−)`` through
    ``lm_next_token_logits`` and calling this "semantic" does not help:
    the midpoint is not a caption, and matching its policy is the same
    off-sheet target with a different metric on top. See
    ``analysis/slider2d/sheet.py``.
    """
    pole = lm_semantic_kl(pred_plus_logits, tgt_plus_logits) + lm_semantic_kl(
        pred_minus_logits, tgt_minus_logits
    )
    hold_w = float(hold_weight)
    if hold_w > 0.0 and hold is not None:
        pole = pole + hold_w * hold
    return pole


def blind_band_tolerance(
    readout: torch.Tensor, top: float, *, cut: float = BLIND_SPECTRAL_CUT
) -> float:
    """Singular values at or below this count as blind.

    ``cut`` is the stated spectral cut; the floor underneath it is the usual
    numerical rank tolerance, so ``cut = 0`` means "the exact null space" and
    not "singular values that happen to round to zero".
    """
    eps = torch.finfo(readout.dtype).eps if readout.is_floating_point() else 1e-7
    floor = float(eps) * max(readout.shape) * float(top)
    return max(float(cut) * float(top), floor)


def lm_blind_projector(
    readout: torch.Tensor,
    *,
    cut: float = BLIND_SPECTRAL_CUT,
    center: bool = True,
) -> torch.Tensor | None:
    """Projector onto the hidden directions a next-token KL cannot use.

    A softmax policy is unchanged by adding a constant to every logit, so
    what the KL can see is the row space of the *centered* readout
    ``W − mean_row(W)``. Everything orthogonal to it is invisible to
    :func:`lm_semantic_kl` — the gradient there is exactly zero, not small.

    ``cut`` widens "invisible" to "read too weakly to matter": keep the
    right-singular directions with ``s ≤ cut · s_max``. ``cut = 0`` is the
    exact null space. Returns ``None`` when that subspace is empty, which is
    the honest answer for a band whose row space fills the hidden width —
    then :func:`lm_dual_band_pole_loss` degenerates to plain semantic KL and
    the caller has to raise ``cut`` rather than believe it got a free fix.

    One SVD of the frozen band at setup; the projector is a constant.
    """
    weight = readout - readout.mean(dim=0, keepdim=True) if center else readout
    _u, sing, vh = torch.linalg.svd(weight, full_matrices=True)
    top = float(sing.max()) if sing.numel() else 0.0
    if top <= 0.0:
        return torch.eye(weight.shape[1], dtype=weight.dtype, device=weight.device)
    tol = blind_band_tolerance(weight, top, cut=cut)
    keep = [i for i in range(vh.shape[0]) if i >= sing.numel() or float(sing[i]) <= tol]
    if not keep:
        return None
    basis = vh[keep]
    return basis.transpose(-1, -2) @ basis


def lm_blind_residual(
    vec: torch.Tensor, projector: torch.Tensor | None
) -> torch.Tensor:
    """The part of ``vec`` inside the readout's blind band."""
    if projector is None:
        return torch.zeros_like(vec)
    flat = vec.flatten()
    return (projector.to(dtype=flat.dtype) @ flat).reshape_as(vec)


def lm_dual_band_pole_loss(
    pred_plus: torch.Tensor,
    pred_minus: torch.Tensor,
    tgt_plus: torch.Tensor,
    tgt_minus: torch.Tensor,
    *,
    pred_plus_logits: torch.Tensor,
    pred_minus_logits: torch.Tensor,
    tgt_plus_logits: torch.Tensor,
    tgt_minus_logits: torch.Tensor,
    blind_projector: torch.Tensor | None,
    blind_weight: float = DUAL_BAND_WEIGHT,
    hold: torch.Tensor | None = None,
    hold_weight: float = 0.0,
) -> torch.Tensor:
    """Semantic KL where the head reads, hidden MSE where it cannot.

        loss = KL(t₊ ‖ p₊) + KL(t₋ ‖ p₋)
             + blind_weight · ( ‖P_blind(p₊ − t₊)‖² + ‖P_blind(p₋ − t₋)‖² )

    Neither band is new; splitting them is. ``semantic_kl`` alone is the
    live v16 loss and on a close pair it reaches its floor without the axis
    arriving, because which of two voices sings the same song barely moves
    the one scored distribution — that content is in ``P_blind``, where the
    KL's gradient is zero. Hidden MSE alone pins ``P_blind`` but also
    insists on matching the readable band in Euclidean distance, which is
    not the quantity anyone listens to.

    Distinct from ``semantic_kl_null``: that pins ``ker(W)`` of the
    *uncentered* readout via coefficient MSE. This pins the *centered*
    semantic band's SVD blind projector via residual MSE. Do not alias.

    The targets must still be states a real caption produces; this is a
    loss, and no loss fixes a target point that is not a caption.
    """
    total = lm_semantic_kl(pred_plus_logits, tgt_plus_logits) + lm_semantic_kl(
        pred_minus_logits, tgt_minus_logits
    )
    weight = float(blind_weight)
    if weight > 0.0 and blind_projector is not None:
        blind = lm_blind_residual(pred_plus - tgt_plus, blind_projector).pow(2).mean()
        blind = blind + lm_blind_residual(
            pred_minus - tgt_minus, blind_projector
        ).pow(2).mean()
        total = total + weight * blind
    hold_w = float(hold_weight)
    if hold_w > 0.0 and hold is not None:
        total = total + hold_w * hold
    return total


def lm_readout_null_basis(
    readout: torch.Tensor, *, rtol: float = 1e-5
) -> torch.Tensor | None:
    """Orthonormal basis for ``ker(W)`` where ``W`` is the semantic readout.

    Vectors in this subspace cannot change ``W @ h``, so semantic KL has zero
    gradient on them. Hidden MSE on this component is what pins delivery /
    vocal detail on a close pair without replacing the caption target.
    """
    weight = readout.detach().float()
    if weight.ndim != 2 or weight.shape[0] == 0 or weight.shape[1] == 0:
        return None
    _, singular, vh = torch.linalg.svd(weight, full_matrices=True)
    if singular.numel() == 0:
        return None
    thresh = float(rtol) * float(singular[0].clamp_min(1e-8))
    keep = singular <= thresh
    if not bool(keep.any()):
        return None
    basis = vh[keep].transpose(0, 1)
    return basis


def lm_null_space_mse(
    pred_plus: torch.Tensor,
    pred_minus: torch.Tensor,
    tgt_plus: torch.Tensor,
    tgt_minus: torch.Tensor,
    readout: torch.Tensor,
    *,
    null_basis: torch.Tensor | None = None,
    rtol: float = 1e-5,
) -> torch.Tensor:
    """MSE on the readout null-space component of the pole residual."""
    basis = null_basis
    if basis is None:
        basis = lm_readout_null_basis(readout, rtol=rtol)
    if basis is None:
        return pred_plus.new_tensor(0.0)

    def _term(pred: torch.Tensor, tgt: torch.Tensor) -> torch.Tensor:
        delta = (pred - tgt).flatten().float()
        coeff = basis.T @ delta
        return coeff.pow(2).mean()

    return _term(pred_plus, tgt_plus) + _term(pred_minus, tgt_minus)


def lm_semantic_null_pole_loss(
    pred_plus_logits: torch.Tensor,
    pred_minus_logits: torch.Tensor,
    tgt_plus_logits: torch.Tensor,
    tgt_minus_logits: torch.Tensor,
    pred_plus: torch.Tensor,
    pred_minus: torch.Tensor,
    tgt_plus: torch.Tensor,
    tgt_minus: torch.Tensor,
    readout: torch.Tensor,
    *,
    null_weight: float = 1.0,
    null_basis: torch.Tensor | None = None,
    hold: torch.Tensor | None = None,
    hold_weight: float = 0.0,
) -> torch.Tensor:
    """Semantic KL on the readout row space plus hidden MSE on ``ker(W)``.

    Real caption poles stay the teacher. The KL term matches what the scored
    token can see; the null-space MSE pins what it cannot — delivery on a
    close pair, without ``faithful_sub_e``'s blend teacher on a divergent one.

    Canonical live ``--pole_mode semantic_kl_null``. ``semantic_kl_plus_hidden``
    and ``semantic_kl_pin`` are aliases of this same loss, not forks.
    """
    pole = lm_semantic_pole_loss(
        pred_plus_logits,
        pred_minus_logits,
        tgt_plus_logits,
        tgt_minus_logits,
        hold=hold,
        hold_weight=hold_weight,
    )
    null_w = float(null_weight)
    if null_w > 0.0:
        pole = pole + null_w * lm_null_space_mse(
            pred_plus,
            pred_minus,
            tgt_plus,
            tgt_minus,
            readout,
            null_basis=null_basis,
        )
    return pole


def lm_unrolled_semantic_pole_loss(
    pred_plus: torch.Tensor,
    pred_minus: torch.Tensor,
    tgt_plus: torch.Tensor,
    tgt_minus: torch.Tensor,
    readout: torch.Tensor,
    transition: torch.Tensor,
    *,
    unroll_steps: int = 1,
    hold: torch.Tensor | None = None,
    hold_weight: float = 0.0,
) -> torch.Tensor:
    """Semantic KL at token 0 and after each residual-mix step.

    ``transition`` is the frozen mix that carries readout-invisible
    content into the scored band (the pair-exam ``A = I + mix·(û⊗d̂)``).
    One-token KL has no gradient on that content; after the mix it
    does. Fixture-only: the live trainer has no frozen mix, so this is
    not a ``--pole_mode``.
    """
    if int(unroll_steps) < 0:
        raise ValueError(f"unroll_steps must be ≥ 0, got {unroll_steps!r}")
    p_plus, p_minus = pred_plus, pred_minus
    t_plus, t_minus = tgt_plus, tgt_minus
    total = None
    for k in range(int(unroll_steps) + 1):
        if k > 0:
            p_plus = transition @ p_plus
            p_minus = transition @ p_minus
            t_plus = transition @ t_plus
            t_minus = transition @ t_minus
        term = lm_semantic_pole_loss(
            lm_next_token_logits(p_plus, readout),
            lm_next_token_logits(p_minus, readout),
            lm_next_token_logits(t_plus, readout),
            lm_next_token_logits(t_minus, readout),
        )
        total = term if total is None else total + term
    pole = total / float(int(unroll_steps) + 1)
    hold_w = float(hold_weight)
    if hold_w > 0.0 and hold is not None:
        pole = pole + hold_w * hold
    return pole


def encoder_mse_loss(
    pred_pos: torch.Tensor,
    pred_neg: torch.Tensor,
    pos_tgt: torch.Tensor,
    neg_tgt: torch.Tensor,
) -> torch.Tensor:
    """Encoder-only slider: MSE each pole to the frozen teacher encoding."""
    return F.mse_loss(pred_pos, pos_tgt) + F.mse_loss(pred_neg, neg_tgt)


def expand_attributes_sd(row: dict, attributes: Iterable[str]) -> list[dict]:
    """``prompt_util.load_prompts_from_yaml`` prefixing (all five caption fields)."""
    attrs = [a.strip() for a in attributes if str(a).strip()]
    if not attrs:
        return [dict(row)]
    out = []
    for att in attrs:
        copy_ = dict(row)
        for key in ("target", "positive", "neutral", "negative", "unconditional"):
            if key in copy_ and copy_[key] is not None:
                copy_[key] = f"{att} {copy_[key]}"
        out.append(copy_)
    return out


def expand_attributes_music3(row: dict) -> list[dict]:
    """``train_lora_music3._expand_attributes`` (target/pos/neg/neu only)."""
    attributes = row.get("attributes")
    if not attributes:
        return [dict(row)]
    rows = []
    for attribute in attributes:
        prefix = str(attribute).strip()
        item = dict(row)
        for key in ("target", "positive", "negative", "neutral"):
            value = row.get(key)
            if value:
                item[key] = f"{prefix} {value}"
        rows.append(item)
    return rows
