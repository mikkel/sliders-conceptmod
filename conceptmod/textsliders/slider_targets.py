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
  ``leak_*`` → raw poles.   ``--lm_target faithful_even_blend`` leftover-
  gates the odd part and subtracts ``EVEN_BLEND_SCALE`` of leak-pair
  even leftover (opt-in; default stays ``v9``).   ``--lm_target
  faithful_plus`` trains the + pole only: teacher is leftover-gated
  ``h+`` (raw pos when leftover ê is unused or undeclared). No pair-odd,
  no ``h0 ± a``, no minus MSE. Inference may still expose a −1 fader;
  that fader is unconstrained.   ``--lm_target faithful_plus_neu`` is UNI:
  student +1 fits raw ``h+`` (never leftover-gated) and student scale 0
  fits ``h0``. Last-hidden MSE only — the LoRA still rewrites the lyric
  prefix.   ``--lm_target faithful_plus_neu_prefix`` is UNI plus a prefix
  hold: +1 last hidden → raw ``h+``, +1 prefix hidden → encode(neu)
  prefix (not encode(pos) prefix), scale 0 → ``h0``. That pins Vocal
  Details too. ``--lm_target faithful_plus_neu_lyric`` is UNI plus a
  lyric-token hold: +1 lyric hiddens → encode(neu) yaml ``lyrics``
  span only. Vocal Details / metadata stay free. ``--lm_target
  faithful_plus_neu_roles`` is UNI plus a role split: yaml lyric tokens
  → encode(neu), Vocal Details / caption tokens → encode(pos) (so
  woman can move on a neu listen), last → raw ``h+``, scale 0 →
  ``h0``. No minus teacher. ``--lm_target faithful_plus_neu_orth`` is
  last-token UNI plus a last-delta / lyric-span projection: +1 last →
  raw ``h+``, 0 → ``h0``, and the last-token update must not lie in
  the lyric-token hidden span (lyric tokens keep only the in-span
  residual). Vocal Details are not held. Fail closed if the lyric
  span cannot be found. Opt-in; default stays ``v9``.
  ``--pole_mode dual_band`` is KL on the
  semantic band plus hidden MSE on the centered-readout blind band
  (``P_blind`` from SVD). Neither is the default.
  ``--lm_target v9_project`` is the old slider-level project+hold
  onto û; ``v9_always`` never gates.
- Encoder MSE in ``train_encoder_music3.py``
- Z-Image Turbo UNI analog in ``train_lora_zimage.py``: +1 → +
  concept prompt, scale 0 → neu, no minus teacher, unused prompt
  tokens held to encode(neu). Velocity-space CFG geometry is
  ``v(z,t,c) − v(z,t,'')``. Opt-in; Music 3 defaults stay put.
- Krea image UNI (opt-in ``train_lora_krea.py``, not the Music 3
  default): student +1 fits ``v(z,t,pos)`` (CFG-composed when
  guidance > 0), student scale 0 fits ``v(z,t,neu)``. Concept
  direction is conceptmod's ``v(z,t,c) − v(z,t,'')``. Unused
  prompt tokens hold to encode(neu); concept words are not held.
  No minus teacher (canary only). Not lyric-hold.
  ``--lm_target embed`` (alias ``--recipe embed_uni``) is TE-only
  stacked-embed UNI: student ``E_θ(neu)`` fits stopgrad
  ``E_frozen(pos)`` on ``[B, seq, layers, dim]``. Live gap diag:
  DiT velocity neu/plus cos≈0.9999 (useless); TE embeds
  ``[1,512,12,2560]`` neu/plus cos≈0.67 (early ~0.91, mid/late
  ~0.57–0.73). Do not teach v-space on that path. Not Anima
  ``same_crop`` / ``embed_struct``.

No Hub, no GPU, no model weights.
"""

from __future__ import annotations

from dataclasses import dataclass
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


def lm_faithful_plus(
    pos: torch.Tensor,
    neg: torch.Tensor,
    neu: torch.Tensor,
    leak_dir: torch.Tensor | None = None,
    *,
    slider_dir: torch.Tensor | None = None,
    target_scale: float = 1.0,
    unused: bool | None = None,
    floor: float = UNUSED_E_OVERLAP_MAX,
) -> torch.Tensor:
    """Plus-only teacher: leftover-gated ``h+``. Never a minus target.

    Teacher is the + caption when no leftover ê is unused. When ``leak_*``
    exists and leftover ê is unused, the + teacher is the + half of
    ``lm_faithful_sub_e`` — leftover-gate on the + side only. Pair
    geometry may still read ``neg`` to decide the gate; the minus caption
    is not a teacher. No pair-odd, no ``h0 ± a``, no minus MSE.
    """
    plus, _minus = lm_faithful_sub_e_if_unused(
        pos,
        neg,
        neu,
        leak_dir,
        slider_dir=slider_dir,
        target_scale=target_scale,
        unused=unused,
        floor=floor,
    )
    return plus


def lm_faithful_plus_neu(
    pos: torch.Tensor,
    neg: torch.Tensor,
    neu: torch.Tensor,
    leak_dir: torch.Tensor | None = None,
    *,
    slider_dir: torch.Tensor | None = None,
    target_scale: float = 1.0,
) -> torch.Tensor:
    """UNI plus+neu teacher: raw ``h+``. Never leftover-gates.

    ``--lm_target faithful_plus_neu``: student +1 fits the + caption
    (``pos``), not a leftover-gated leftover and not ``h0 ± a``.
    ``leak_dir`` / ``slider_dir`` are accepted so the call site can stay
    the same as ``lm_faithful_plus``; they do not change the teacher.
    ``neg`` is not a teacher. Scale-0 supervision is ``neu`` itself and
    lives in ``lm_plus_neu_loss``, not here. Prefix-hold lives in
    ``lm_plus_neu_prefix_loss`` (``faithful_plus_neu_prefix``) and
    ``lm_plus_neu_lyric_loss`` (``faithful_plus_neu_lyric``); this
    function is still the last-token + teacher.
    """
    del neg, neu, leak_dir, slider_dir, target_scale
    return pos


def lm_faithful_plus_neu_prefix(
    pos: torch.Tensor,
    neg: torch.Tensor,
    neu: torch.Tensor,
    leak_dir: torch.Tensor | None = None,
    *,
    slider_dir: torch.Tensor | None = None,
    target_scale: float = 1.0,
) -> torch.Tensor:
    """UNI prefix-hold last-token teacher: still raw ``h+``.

    ``--lm_target faithful_plus_neu_prefix`` keeps the same last-token
    + teacher as ``faithful_plus_neu``. The prefix hold (student +1
    prefix → encode(neu) prefix) is a sequence loss, not a different
    last-hidden point. ``leak_dir`` / leftover-gate never apply.
    """
    return lm_faithful_plus_neu(
        pos,
        neg,
        neu,
        leak_dir,
        slider_dir=slider_dir,
        target_scale=target_scale,
    )


def lm_faithful_plus_neu_lyric(
    pos: torch.Tensor,
    neg: torch.Tensor,
    neu: torch.Tensor,
    leak_dir: torch.Tensor | None = None,
    *,
    slider_dir: torch.Tensor | None = None,
    target_scale: float = 1.0,
) -> torch.Tensor:
    """UNI lyric-hold last-token teacher: still raw ``h+``.

    ``--lm_target faithful_plus_neu_lyric`` keeps the same last-token
    + teacher as ``faithful_plus_neu``. The lyric hold (student +1
    yaml-lyrics tokens → encode(neu) lyrics) is a sequence loss, not
    a different last-hidden point. Vocal Details is not held.
    ``leak_dir`` / leftover-gate never apply.
    """
    return lm_faithful_plus_neu(
        pos,
        neg,
        neu,
        leak_dir,
        slider_dir=slider_dir,
        target_scale=target_scale,
    )


def lm_faithful_plus_neu_roles(
    pos: torch.Tensor,
    neg: torch.Tensor,
    neu: torch.Tensor,
    leak_dir: torch.Tensor | None = None,
    *,
    slider_dir: torch.Tensor | None = None,
    target_scale: float = 1.0,
) -> torch.Tensor:
    """UNI role-split last-token teacher: still raw ``h+``.

    ``--lm_target faithful_plus_neu_roles`` keeps the same last-token
    + teacher as ``faithful_plus_neu``. Lyrics → encode(neu) and
    Vocal Details → encode(pos) are sequence losses, not a different
    last-hidden point. ``leak_dir`` / leftover-gate never apply.
    """
    return lm_faithful_plus_neu(
        pos,
        neg,
        neu,
        leak_dir,
        slider_dir=slider_dir,
        target_scale=target_scale,
    )


def lm_faithful_plus_neu_orth(
    pos: torch.Tensor,
    neg: torch.Tensor,
    neu: torch.Tensor,
    leak_dir: torch.Tensor | None = None,
    *,
    slider_dir: torch.Tensor | None = None,
    target_scale: float = 1.0,
) -> torch.Tensor:
    """UNI last-delta-orth last-token teacher: still raw ``h+``.

    ``--lm_target faithful_plus_neu_orth`` keeps the same last-token
    + teacher as ``faithful_plus_neu``. The lyric-span projection is a
    sequence constraint (last-token update ⊥ lyric-token hiddens), not
    a different last-hidden point. ``leak_dir`` / leftover-gate never
    apply.
    """
    return lm_faithful_plus_neu(
        pos,
        neg,
        neu,
        leak_dir,
        slider_dir=slider_dir,
        target_scale=target_scale,
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


def lm_plus_loss(
    pred_plus: torch.Tensor,
    tgt_plus: torch.Tensor,
    *,
    hold: torch.Tensor | None = None,
    hold_weight: float = 0.0,
) -> torch.Tensor:
    """Plus-only hidden MSE. No minus term, no pair-odd, no ``h0 ± a``.

    ``--lm_target faithful_plus``: student +1 fits leftover-gated ``h+``.
    Inference may still expose a −1 fader; this loss does not teach it.
    """
    pole = F.mse_loss(pred_plus, tgt_plus)
    hold_w = float(hold_weight)
    if hold_w > 0.0 and hold is not None:
        pole = pole + hold_w * hold
    return pole


def lm_plus_neu_loss(
    pred_plus: torch.Tensor,
    tgt_plus: torch.Tensor,
    pred_zero: torch.Tensor,
    tgt_zero: torch.Tensor,
) -> torch.Tensor:
    """UNI hidden MSE: student +1 fits ``h+``, student 0 fits ``h0``.

    ``--lm_target faithful_plus_neu``: ``MSE(+) + MSE(0)`` only. No minus
    MSE, no pair-odd, no ``h0 ± a``, no leftover-gate. ``tgt_plus`` is
    raw pos; ``tgt_zero`` is the neutral caption.
    """
    return F.mse_loss(pred_plus, tgt_plus) + F.mse_loss(pred_zero, tgt_zero)


def lm_plus_neu_prefix_loss(
    pred_plus: torch.Tensor,
    tgt_plus: torch.Tensor,
    pred_zero: torch.Tensor,
    tgt_zero: torch.Tensor,
    pred_plus_prefix: torch.Tensor,
    tgt_neu_prefix: torch.Tensor,
    *,
    prefix_weight: float = 1.0,
) -> torch.Tensor:
    """UNI + prefix hold: last-token ``h+`` / ``h0``, prefix → encode(neu).

    ``--lm_target faithful_plus_neu_prefix``: ``MSE(last +) + MSE(last 0)
    + MSE(prefix + → encode(neu) prefix)``. Prefix teacher is the
    same-room lyric / neu tokens, not encode(pos) prefix. No minus
    teacher, no leftover-gate. ``tgt_plus`` is raw last-token ``h+``.
    """
    last = lm_plus_neu_loss(pred_plus, tgt_plus, pred_zero, tgt_zero)
    return last + float(prefix_weight) * F.mse_loss(pred_plus_prefix, tgt_neu_prefix)


def lm_plus_neu_lyric_loss(
    pred_plus: torch.Tensor,
    tgt_plus: torch.Tensor,
    pred_zero: torch.Tensor,
    tgt_zero: torch.Tensor,
    pred_plus_lyric: torch.Tensor,
    tgt_neu_lyric: torch.Tensor,
    *,
    lyric_weight: float = 1.0,
) -> torch.Tensor:
    """UNI + lyric-token hold: last-token ``h+`` / ``h0``, lyrics → encode(neu).

    ``--lm_target faithful_plus_neu_lyric``: ``MSE(last +) + MSE(last 0)
    + MSE(lyric + → encode(neu) lyric)``. Lyric tensors are the yaml
    ``lyrics`` span only — not Vocal Details / metadata. ``tgt_plus``
    is raw last-token ``h+``. No minus teacher, no leftover-gate.
    """
    last = lm_plus_neu_loss(pred_plus, tgt_plus, pred_zero, tgt_zero)
    return last + float(lyric_weight) * F.mse_loss(pred_plus_lyric, tgt_neu_lyric)


class RoleSpanError(ValueError):
    """Required lyric or concept span is missing. Fail closed."""


def lm_find_subsequence(haystack: Sequence[int], needle: Sequence[int]) -> int | None:
    """First index of ``needle`` in ``haystack``, or None."""
    ids = [int(x) for x in haystack]
    pat = [int(x) for x in needle]
    if not pat or len(pat) > len(ids):
        return None
    last = len(ids) - len(pat)
    for start in range(last + 1):
        if ids[start : start + len(pat)] == pat:
            return start
    return None


def lm_role_span_bounds(
    token_ids: Sequence[int],
    *,
    lyrics_start_id: int | None,
    lyrics_end_id: int | None,
    caption_start_id: int | None,
    caption_end_id: int | None,
    vocal_details_ids: Sequence[int] | None = None,
    arrangement_ids: Sequence[int] | None = None,
) -> dict[str, tuple[int, int] | str]:
    """Lyric and concept ``[start, end)`` the tokenizer can actually locate.

    Lyrics are the exclusive span between ``<|lyrics_start|>`` and
    ``<|lyrics_end|>``. Concept prefers Vocal Details (after the heading,
    until Arrangement or ``<|caption_end|>``) because woman lives there
    on gender-v4. If that heading is not a unique token run, concept is
    the exclusive caption span between ``<|caption_start|>`` and
    ``<|caption_end|>``. Fail closed when a required span is empty.
    """
    ids = [int(x) for x in token_ids]
    if lyrics_start_id is None or lyrics_end_id is None:
        raise RoleSpanError("lyrics special tokens are required")
    if caption_start_id is None or caption_end_id is None:
        raise RoleSpanError("caption special tokens are required")
    try:
        lyric_lo = ids.index(int(lyrics_start_id)) + 1
        lyric_hi = ids.index(int(lyrics_end_id))
    except ValueError as exc:
        raise RoleSpanError("lyrics span markers are missing") from exc
    if lyric_hi < lyric_lo:
        raise RoleSpanError("lyrics end precedes lyrics start")
    if lyric_hi <= lyric_lo:
        raise RoleSpanError("lyrics span is empty")
    try:
        cap_lo = ids.index(int(caption_start_id)) + 1
        cap_hi = ids.index(int(caption_end_id))
    except ValueError as exc:
        raise RoleSpanError("caption span markers are missing") from exc
    if cap_hi <= cap_lo:
        raise RoleSpanError("caption span is empty")

    concept_lo, concept_hi = cap_lo, cap_hi
    source = "caption"
    heading = list(vocal_details_ids or ())
    if heading:
        found = lm_find_subsequence(ids[cap_lo:cap_hi], heading)
        if found is not None:
            concept_lo = cap_lo + found + len(heading)
            stop = cap_hi
            arr = list(arrangement_ids or ())
            if arr:
                arr_at = lm_find_subsequence(ids[concept_lo:cap_hi], arr)
                if arr_at is not None:
                    stop = concept_lo + arr_at
            if stop > concept_lo:
                concept_hi = stop
                source = "vocal_details"

    if concept_hi <= concept_lo:
        raise RoleSpanError("concept span is empty")
    return {
        "lyric": (int(lyric_lo), int(lyric_hi)),
        "concept": (int(concept_lo), int(concept_hi)),
        "source": source,
    }


def lm_gather_span(hidden: torch.Tensor, start: int, end: int) -> torch.Tensor:
    """Slice ``[start, end)`` from a ``[T, H]`` or ``[B, T, H]`` hidden."""
    lo, hi = int(start), int(end)
    if hi <= lo:
        raise RoleSpanError("span is empty")
    if hidden.dim() == 2:
        return hidden[lo:hi]
    if hidden.dim() == 3:
        return hidden[:, lo:hi]
    raise RoleSpanError(f"hidden must be [T, H] or [B, T, H], got {tuple(hidden.shape)}")


def lm_span_mse(pred: torch.Tensor, tgt: torch.Tensor) -> torch.Tensor:
    """Token-wise MSE when lengths match; pooled MSE when they do not.

    Pos Vocal Details is longer than neu on gender-v4 (woman is extra
    text). Lyrics stay token-wise: same yaml line, same length.
    """
    pred_flat = pred.reshape(-1, pred.shape[-1])
    tgt_flat = tgt.reshape(-1, tgt.shape[-1])
    if pred_flat.shape[0] == 0 or tgt_flat.shape[0] == 0:
        raise RoleSpanError("span MSE got an empty role span")
    if pred_flat.shape[0] == tgt_flat.shape[0]:
        return F.mse_loss(pred_flat, tgt_flat)
    return F.mse_loss(pred_flat.mean(0), tgt_flat.mean(0))


def lm_plus_neu_roles_loss(
    pred_plus: torch.Tensor,
    tgt_plus: torch.Tensor,
    pred_zero: torch.Tensor,
    tgt_zero: torch.Tensor,
    pred_lyric: torch.Tensor,
    tgt_neu_lyric: torch.Tensor,
    pred_concept: torch.Tensor,
    tgt_pos_concept: torch.Tensor,
    *,
    lyric_weight: float = 1.0,
    concept_weight: float = 1.0,
) -> torch.Tensor:
    """UNI + role split: lyrics → encode(neu), concept → encode(pos).

    ``--lm_target faithful_plus_neu_roles``: ``MSE(last + → h+) +
    MSE(last 0 → h0) + MSE(lyric + → encode(neu) lyrics) +
    MSE(concept + → encode(pos) same role)``. No minus, no leftover-gate.
    Last token stays raw ``h+``. Concept teacher is encode(pos) Vocal
    Details (or the whole caption if that heading cannot be located).
    """
    last = lm_plus_neu_loss(pred_plus, tgt_plus, pred_zero, tgt_zero)
    lyric = lm_span_mse(pred_lyric, tgt_neu_lyric)
    concept = lm_span_mse(pred_concept, tgt_pos_concept)
    return last + float(lyric_weight) * lyric + float(concept_weight) * concept


def lm_plus_neu_orth_loss(
    pred_plus: torch.Tensor,
    tgt_plus: torch.Tensor,
    pred_zero: torch.Tensor,
    tgt_zero: torch.Tensor,
    pred_plus_lyric: torch.Tensor,
    tgt_neu_lyric: torch.Tensor,
    *,
    lyric_weight: float = 1.0,
    fail_closed: bool = True,
) -> torch.Tensor:
    """UNI last-token MSE plus last-delta off the lyric-token span.

    ``--lm_target faithful_plus_neu_orth``: ``MSE(last + → raw h+) +
    MSE(last 0 → h0)`` and the last-token update must not lie in the
    lyric span. Lyric tokens keep only the in-span residual
    (``||(I−P)(student_lyric − encode(neu) lyric)||²``). That is not
    a lyric-token *hold* to neu: in-span motion stays free. Vocal
    Details (outside the lyric span) are not held. Fail closed if
    the lyric span cannot be found. ``tgt_plus`` is raw last-token
    ``h+``.
    """
    last = lm_plus_neu_loss(pred_plus, tgt_plus, pred_zero, tgt_zero)
    off = lm_project_last_delta_off_lyric(
        pred_plus_lyric - tgt_neu_lyric,
        tgt_neu_lyric,
        fail_closed=fail_closed,
    )
    return last + float(lyric_weight) * off.pow(2).mean()


def lm_lyric_span_basis(
    lyric_span: torch.Tensor,
    *,
    fail_closed: bool = False,
) -> torch.Tensor | None:
    """Orthonormal basis for the lyric-token hidden span. Rank-aware.

    ``lyric_span`` is ``[..., H]`` (encode(neu) lyric-token hiddens).
    Repeated copies of one lyric vector count as rank 1 — do not keep
    the leftover QR columns (those are numerical junk and would eat
    the concept). Fail closed when the span cannot be found.
    """
    if lyric_span.numel() == 0:
        if fail_closed:
            raise RuntimeError("lyric span cannot be found (empty)")
        return None
    span = lyric_span.reshape(-1, lyric_span.shape[-1])
    if float(span.norm()) <= 1e-8:
        if fail_closed:
            raise RuntimeError("lyric span cannot be found (near-zero)")
        return None
    q, r = torch.linalg.qr(span.T.float(), mode="reduced")
    diag = r.abs() if r.ndim == 1 else torch.diagonal(r).abs()
    tol = 1e-6 * float(diag.max().clamp_min(1e-8))
    rank = int((diag > tol).sum().item())
    if rank <= 0:
        if fail_closed:
            raise RuntimeError("lyric span cannot be found (rank 0)")
        return None
    return q[:, :rank].to(dtype=lyric_span.dtype)


def lm_project_last_delta_off_lyric(
    last_delta: torch.Tensor,
    lyric_span: torch.Tensor,
    *,
    fail_closed: bool = False,
) -> torch.Tensor:
    """Project a last-token delta off the span of lyric-token hiddens.

    Cheap sibling of prefix-hold: the concept update at the continue
    token is orthogonal to the yaml lyric positions. ``lyric_span`` is
    ``[..., H]`` (encode(neu) lyric-token hiddens). Used by
    ``--lm_target faithful_plus_neu_orth``. Fail closed if the lyric
    span cannot be found. Rank-aware: identical lyric rows are rank 1.
    """
    basis = lm_lyric_span_basis(lyric_span, fail_closed=fail_closed)
    if basis is None:
        return last_delta
    hidden = int(lyric_span.shape[-1])
    if last_delta.ndim >= 1 and last_delta.shape[-1] == hidden:
        flat = last_delta.reshape(-1, hidden)
        return (flat - (flat @ basis) @ basis.T).reshape_as(last_delta)
    vec = last_delta.reshape(-1)
    return last_delta - (basis @ (basis.T @ vec)).reshape_as(last_delta)


def lm_project_onto_lyric_span(
    last_delta: torch.Tensor,
    lyric_span: torch.Tensor,
    *,
    fail_closed: bool = False,
) -> torch.Tensor:
    """Keep only the component of a delta that already lies in the lyric span."""
    return last_delta - lm_project_last_delta_off_lyric(
        last_delta, lyric_span, fail_closed=fail_closed
    )


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
    plus_only: bool = False,
) -> torch.Tensor:
    """Pole MSE plus optional v9 anchor MSE and orthogonal hold.

    Endreg / planreg / collapse_weight are AR-only and are not expressed on
    the CPU field. Semantic-KL poles are ``lm_semantic_pole_loss``; they
    need a readout, which ``analysis/slider2d/sheet.py`` supplies.
    ``plus_only`` drops the minus MSE (``faithful_plus``).
    """
    if plus_only:
        return lm_plus_loss(pred_plus, tgt_plus, hold=hold, hold_weight=hold_weight)
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



# ---------------------------------------------------------------------------
# Z-Image Turbo (ZiT) image-slider UNI analog
# ---------------------------------------------------------------------------
# Opt-in ``train_lora_zimage.py``. Not Music 3 lyric-hold. Not Anima / Krea / H3.
# Teacher is velocity-space UNI:
#   +1 → v(z, t, + concept)
#   scale 0 → v(z, t, neu)
#   no minus teacher (canary only)
# Unused prompt tokens hold to encode(neu); concept words are not held.
# CFG geometry from conceptmod ``backends/zimage.py``:
#   v(z, t, c) − v(z, t, '')
# Default sample guidance is 0, so CFG is off and the teacher is raw v.


class ZImageHoldError(ValueError):
    """Concept words missing from the + prompt. Fail closed."""


def zimage_cfg_delta(vel_c: torch.Tensor, vel_uncond: torch.Tensor) -> torch.Tensor:
    """Velocity-space CFG increment: ``v(z,t,c) − v(z,t,'')``."""
    return vel_c - vel_uncond


def zimage_cfg(
    vel_c: torch.Tensor, vel_uncond: torch.Tensor, guidance: float
) -> torch.Tensor:
    """conceptmod zimage ``_cfg``: ``v + g * (v − v_u)``. ``g=0`` is raw ``v``."""
    if guidance and float(guidance) > 0.0:
        return vel_c + float(guidance) * (vel_c - vel_uncond)
    return vel_c


def zimage_uni_teachers(
    vel_pos: torch.Tensor,
    vel_neu: torch.Tensor,
    vel_uncond: torch.Tensor | None = None,
    *,
    guidance: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """UNI teachers: +1 target is the + concept velocity, 0 is neu.

    The student that fits these teachers runs on the **neutral** caption
    (the infer path). Concept still comes from the + caption for teacher
    velocities only. When ``guidance > 0`` both poles go through
    :func:`zimage_cfg`. ``vel_uncond`` is required then. Minus is never
    a teacher.
    """
    if vel_uncond is None:
        return vel_pos, vel_neu
    return (
        zimage_cfg(vel_pos, vel_uncond, guidance),
        zimage_cfg(vel_neu, vel_uncond, guidance),
    )


def zimage_uni_loss(
    pred_plus: torch.Tensor,
    tgt_plus: torch.Tensor,
    pred_zero: torch.Tensor,
    tgt_zero: torch.Tensor,
    *,
    pred_unused: torch.Tensor | None = None,
    tgt_unused: torch.Tensor | None = None,
    unused_weight: float = 0.0,
    unused_token_hold: torch.Tensor | None = None,
    token_hold_weight: float = 0.0,
) -> torch.Tensor:
    """UNI velocity MSE: student +1/+0 on neu → (+ teacher, v_neu).

    No minus MSE. ``pred_unused`` is an optional extra hold (for example
    a later control-prompt velocity hold). Do **not** pass student +1 on
    neu as ``pred_unused`` with target ``v_neu`` — that cancels the infer
    path. Frozen-embed token hold is off by default (no LoRA grad).
    """
    loss = F.mse_loss(pred_plus, tgt_plus) + F.mse_loss(pred_zero, tgt_zero)
    if pred_unused is not None and tgt_unused is not None and float(unused_weight) > 0.0:
        loss = loss + float(unused_weight) * F.mse_loss(pred_unused, tgt_unused)
    if unused_token_hold is not None and float(token_hold_weight) > 0.0:
        loss = loss + float(token_hold_weight) * unused_token_hold
    return loss


def zimage_canary_minus(
    pred_minus: torch.Tensor, vel_neg: torch.Tensor
) -> dict[str, float | bool]:
    """Unscored −1 canary. Never a teacher."""
    overlap = F.cosine_similarity(
        pred_minus.flatten().unsqueeze(0), vel_neg.flatten().unsqueeze(0)
    ).squeeze()
    return {
        "scored": False,
        "minus_overlap_neg": float(overlap),
    }


def expand_attributes_zimage(row: dict) -> list[dict]:
    """Pin unused attributes onto target / positive / neutral (and canary neg).

    Same prefixing as Music 3, but the minus caption is canary-only.
    """
    attributes = row.get("attributes")
    if not attributes:
        return [dict(row)]
    rows = []
    for attribute in attributes:
        prefix = str(attribute).strip()
        if not prefix:
            continue
        item = dict(row)
        for key in ("target", "positive", "negative", "neutral"):
            value = row.get(key)
            if value:
                item[key] = f"{prefix} {value}"
        rows.append(item)
    return rows or [dict(row)]


def zimage_concept_token_ids(
    concept_words: str | Iterable[str],
    tokenize_fn,
) -> set[int]:
    """Token ids for declared concept words. Empty words are skipped."""
    if isinstance(concept_words, str):
        words = [w.strip() for w in concept_words.split(",") if w.strip()]
    else:
        words = [str(w).strip() for w in concept_words if str(w).strip()]
    ids: set[int] = set()
    for word in words:
        ids.update(int(t) for t in tokenize_fn(word))
    return ids


def zimage_unused_token_positions(
    token_ids: Sequence[int], concept_ids: Iterable[int]
) -> list[int]:
    """Positions whose tokens are not concept words."""
    banned = {int(t) for t in concept_ids}
    return [i for i, tid in enumerate(token_ids) if int(tid) not in banned]


def zimage_require_concept_in_prompt(
    token_ids: Sequence[int], concept_ids: Iterable[int]
) -> None:
    """Fail closed when the + prompt has no concept-word tokens."""
    banned = {int(t) for t in concept_ids}
    if not banned:
        raise ZImageHoldError("concept_words are required")
    if not any(int(t) in banned for t in token_ids):
        raise ZImageHoldError("concept words were not found in the + prompt")


def zimage_gather_unused(
    hidden: torch.Tensor, token_ids: Sequence[int], concept_ids: Iterable[int]
) -> torch.Tensor:
    """Unused-token slice of a ``[T, H]`` or ``[B, T, H]`` embed."""
    positions = zimage_unused_token_positions(token_ids, concept_ids)
    if not positions:
        raise ZImageHoldError("unused prompt span is empty")
    idx = torch.tensor(positions, device=hidden.device, dtype=torch.long)
    if hidden.dim() == 2:
        return hidden.index_select(0, idx)
    if hidden.dim() == 3:
        return hidden.index_select(1, idx)
    raise ZImageHoldError(f"embeds must be [T, H] or [B, T, H], got {tuple(hidden.shape)}")


def zimage_unused_token_hold(
    pred_plus_embeds: torch.Tensor,
    neu_embeds: torch.Tensor,
    plus_ids: Sequence[int],
    neu_ids: Sequence[int],
    concept_ids: Iterable[int],
    *,
    fail_closed: bool = True,
) -> torch.Tensor:
    """MSE unused +1 tokens → encode(neu). Concept words are not held.

    Lengths may differ (``an old person`` vs ``a person``). Same-length
    unused spans are token-wise; otherwise pooled. Fail closed if
    concept words are missing from the + prompt.
    """
    if fail_closed:
        zimage_require_concept_in_prompt(plus_ids, concept_ids)
    pred = zimage_gather_unused(pred_plus_embeds, plus_ids, concept_ids)
    tgt = zimage_gather_unused(neu_embeds, neu_ids, concept_ids)
    pred_flat = pred.reshape(-1, pred.shape[-1])
    tgt_flat = tgt.reshape(-1, tgt.shape[-1])
    if pred_flat.shape[0] == tgt_flat.shape[0]:
        return F.mse_loss(pred_flat, tgt_flat)
    return F.mse_loss(pred_flat.mean(0), tgt_flat.mean(0))


# ---------------------------------------------------------------------------
# Krea image UNI (opt-in). Not Music 3 lyric-hold. No minus teacher.
# ---------------------------------------------------------------------------

KREA_RAW_MODEL = "krea/Krea-2-Raw"
KREA_DEFAULT_RANK = 16
KREA_DEFAULT_RESOLUTION = 512
KREA_RAW_STEPS = 28
KREA_RAW_CFG = 4.5
KREA_TURBO_STEPS = 8
KREA_TURBO_CFG = 0.0
KREA_HOLD_WEIGHT = 1.0
# Smile / happy yaml: unused-token hold is a near-constant on frozen TE
# and dominates the logged loss (live smile-krea: hold≈7.31 of ≈7.35).
# Age yaml still wants the stock 1.0 (prefixed unused gender). Do not
# change KREA_HOLD_WEIGHT — pass --hold_weight 0.1 on the smile card.
KREA_SMILE_HOLD_WEIGHT = 0.1
KREA_CONTROL_PROMPT = "a bowl of fruit on a table"
KREA_SAMPLE_SCALES = (0.0, 0.25, 0.5, 1.0)
KREA_LORA_TARGETS = ("to_q", "to_k", "to_v", "to_out.0")
KREA_TE_LORA_TARGETS = ("q_proj", "k_proj", "v_proj", "o_proj")
KREA_DEFAULT_LORA_TARGETS = "dit"
KREA_LORA_TARGET_CHOICES = ("dit", "te", "dit+te")
# Velocity UNI is the default (``--lm_target v``). Embed UNI is
# TE-only: live DiT v-gap is microscopic (neu/plus cos≈0.9999);
# stacked TE embeds [1,512,12,2560] carry the smile Δ (cos≈0.67).
KREA_LM_TARGET_DEFAULT = "v"
KREA_LM_TARGETS = ("v", "embed")
KREA_LM_TARGET_ALIASES = {
    "velocity": "v",
    "dit": "v",
    "uni": "v",
    "embed_uni": "embed",
    "te": "embed",
    "te_embed": "embed",
    "text_encoder": "embed",
}
KREA_LM_TARGET_CHOICES = KREA_LM_TARGETS + tuple(KREA_LM_TARGET_ALIASES)
KREA_RECIPE_CHOICES = ("uni", "embed_uni")
KREA_RECIPE_DEFAULT = "uni"
# Live Qwen3-VL stack is 12 layers. Diag: layers 0–1 ~0.91,
# layers 2–11 ~0.57–0.73. Weight mid/late more.
KREA_EMBED_N_LAYERS_LIVE = 12
KREA_EMBED_EARLY_LAYERS = 2
KREA_EMBED_EARLY_WEIGHT = 0.25
KREA_EMBED_MID_LATE_WEIGHT = 1.0
KREA_EMBED_COSINE_WEIGHT = 1.0
KREA_DUMMY_EMBED_LAYERS = 4
KREA_DUMMY_EMBED_SEQ = 8


@dataclass(frozen=True)
class KreaLoraSpec:
    """Which Krea modules receive PEFT LoRA.

    Default is DiT attn (``to_q/k/v/out``). Anima lesson: expression
    can live in the text path, so ``te`` / ``dit+te`` attach Qwen3-VL
    attention ``q_proj/k_proj/v_proj/o_proj``. Age yaml stays DiT-only
    unless the caller passes ``--lora_targets``.
    """

    label: str
    train_dit: bool
    train_te: bool

    @property
    def encoder_lora(self) -> bool:
        return self.train_te

    @property
    def dit_lora_only(self) -> bool:
        return self.train_dit and not self.train_te

    @property
    def te_parking(self) -> bool:
        """Park Qwen3-VL after encode only when TE is frozen (48GB)."""
        return not self.train_te

    @property
    def frozen_modules(self) -> tuple[str, ...]:
        frozen: list[str] = []
        if not self.train_dit:
            frozen.append("transformer")
        if not self.train_te:
            frozen.append("text_encoder")
        return tuple(frozen)

    @property
    def adapted_module_names(self) -> list[str]:
        names: list[str] = []
        if self.train_dit:
            names.append("transformer")
        if self.train_te:
            names.append("text_encoder")
        return names

    @property
    def dit_lora_targets(self) -> list[str]:
        return list(KREA_LORA_TARGETS) if self.train_dit else []

    @property
    def te_lora_targets(self) -> list[str]:
        return list(KREA_TE_LORA_TARGETS) if self.train_te else []


def resolve_krea_lora_targets(lora_targets: str | None = None) -> KreaLoraSpec:
    """Parse ``dit`` / ``te`` / ``dit+te``. Aliases: ``text_encoder``, ``transformer``."""
    raw = str(lora_targets if lora_targets is not None else KREA_DEFAULT_LORA_TARGETS)
    label = raw.strip().lower().replace(" ", "")
    aliases = {
        "transformer": "dit",
        "text_encoder": "te",
        "encoder": "te",
        "dit+text_encoder": "dit+te",
        "transformer+te": "dit+te",
        "transformer+text_encoder": "dit+te",
        "dit+encoder": "dit+te",
        "te+dit": "dit+te",
        "text_encoder+dit": "dit+te",
    }
    label = aliases.get(label, label)
    if label not in KREA_LORA_TARGET_CHOICES:
        raise ValueError(
            f"krea lora_targets must be one of {KREA_LORA_TARGET_CHOICES} "
            f"(aliases: text_encoder, transformer), got {lora_targets!r}"
        )
    parts = set(label.split("+"))
    return KreaLoraSpec(
        label=label,
        train_dit="dit" in parts,
        train_te="te" in parts,
    )


def resolve_krea_lm_target(
    lm_target: str | None = None,
    recipe: str | None = None,
) -> str:
    """``v`` (velocity UNI, default) or ``embed`` (TE-only stacked embeds).

    ``--recipe embed_uni`` is an alias for ``--lm_target embed``.
    Conflicting flags fail closed. Music 3 ``--lm_target v9`` is a
    different trainer and is not accepted here.
    """
    raw_lm = None if lm_target is None else str(lm_target).strip().lower()
    raw_recipe = None if recipe is None else str(recipe).strip().lower()
    from_recipe = None
    if raw_recipe:
        if raw_recipe not in KREA_RECIPE_CHOICES:
            raise ValueError(
                f"krea recipe must be one of {KREA_RECIPE_CHOICES}, got {recipe!r}"
            )
        # ``uni`` is the default velocity recipe and must not override
        # an explicit ``--lm_target embed``.
        if raw_recipe == "embed_uni":
            from_recipe = "embed"
    from_lm = None
    if raw_lm:
        from_lm = KREA_LM_TARGET_ALIASES.get(raw_lm, raw_lm)
        if from_lm not in KREA_LM_TARGETS:
            raise ValueError(
                f"krea lm_target must be one of {KREA_LM_TARGET_CHOICES}, "
                f"got {lm_target!r}"
            )
    if from_lm and from_recipe and from_lm != from_recipe:
        # argparse default is ``--lm_target v``. ``--recipe embed_uni``
        # is the explicit opt-in and wins over that default.
        if from_recipe == "embed" and from_lm == "v":
            return "embed"
        raise ValueError(
            f"krea lm_target={lm_target!r} conflicts with recipe={recipe!r}"
        )
    return from_lm or from_recipe or KREA_LM_TARGET_DEFAULT


def krea_embed_requires_te(lora_spec: KreaLoraSpec) -> None:
    """Fail closed: embed UNI has no concept signal without TE LoRA."""
    if not lora_spec.train_te or lora_spec.train_dit:
        raise ValueError(
            "--lm_target embed is TE-only (DiT stays frozen); "
            "pass --lora_targets te. dit / dit+te still teach v-space "
            "and the live DiT neu/plus gap is cos≈0.9999."
        )


def force_krea_embed_lora_targets(
    lora_targets: str | None = None,
    *,
    lm_target: str | None = None,
    recipe: str | None = None,
) -> KreaLoraSpec:
    """Embed UNI forces ``te`` (DiT frozen). Velocity keeps the caller spec."""
    lm = resolve_krea_lm_target(lm_target, recipe)
    if lm == "embed":
        return resolve_krea_lora_targets("te")
    return resolve_krea_lora_targets(lora_targets)


def _is_lora_delta_layer(module) -> bool:
    return any(
        hasattr(module, name)
        for name in ("lora_A", "lora_B", "lora_embedding_A", "lora_embedding_B")
    )


def apply_continuous_lora_scale(module, scale: float) -> int:
    """Multiply LoRA delta by ``scale`` (continuous 0..1).

    PEFT ``PeftModel.set_adapter_scale`` often no-ops or only toggles,
    which made live smile grids at 0.25 / 0.5 / 1.0 byte-identical.
    This writes the layer ``scaling`` dict (and ``lora_scale`` /
    ``multiplier`` when present) the same way PEFT
    ``rescale_adapter_scale`` does: ``scaling[name] = base[name] * scale``.
    ``base`` is snapshotted on first call (the init ``alpha/r``).
    Works on ``get_peft_model`` wrappers and on duck-typed mocks.
    """
    scale = float(scale)
    updated = 0
    for child in module.modules():
        is_lora = _is_lora_delta_layer(child)
        scaling = getattr(child, "scaling", None)
        if is_lora and isinstance(scaling, dict) and scaling:
            base = getattr(child, "_krea_base_scaling", None)
            if not isinstance(base, dict) or any(key not in base for key in scaling):
                child._krea_base_scaling = {k: float(v) for k, v in scaling.items()}
                base = child._krea_base_scaling
            for key in scaling:
                scaling[key] = base[key] * scale
            updated += 1
        if is_lora and hasattr(child, "lora_scale") and not callable(child.lora_scale):
            child.lora_scale = scale
            updated += 1
        if is_lora and hasattr(child, "multiplier") and not callable(child.multiplier):
            child.multiplier = scale
            updated += 1
    return updated


def krea_looks_turbo(model_id: str) -> bool:
    """Local ComfyUI Turbo files are named with ``turbo``; hub Raw is not."""
    return "turbo" in str(model_id).lower()


def krea_sample_card(model_id: str) -> dict[str, float | int | str]:
    """Live sample card. Train LoRAs on Raw; run on Turbo."""
    if krea_looks_turbo(model_id):
        return {
            "variant": "turbo",
            "sample_steps": KREA_TURBO_STEPS,
            "sample_guidance": KREA_TURBO_CFG,
        }
    return {
        "variant": "raw",
        "sample_steps": KREA_RAW_STEPS,
        "sample_guidance": KREA_RAW_CFG,
    }


def krea_cfg_direction(v_cond: torch.Tensor, v_uncond: torch.Tensor) -> torch.Tensor:
    """Conceptmod Krea concept direction: ``v(z,t,c) − v(z,t,'')``."""
    return v_cond - v_uncond


def krea_cfg_compose(
    v_cond: torch.Tensor,
    v_uncond: torch.Tensor,
    guidance: float,
) -> torch.Tensor:
    """Krea convention: ``cond + g*(cond − uncond)`` when ``g > 0``.

    Turbo trains and samples at ``g = 0`` (CFG off). Raw samples at 4.5.
    """
    if guidance and float(guidance) > 0.0:
        return v_cond + float(guidance) * (v_cond - v_uncond)
    return v_cond


def krea_plus_neu_teachers(
    v_pos: torch.Tensor,
    v_neu: torch.Tensor,
    v_uncond: torch.Tensor | None = None,
    *,
    guidance: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """UNI teachers: +1 → + concept velocity, scale 0 → ``v(neu)``.

    When ``guidance > 0`` and ``v_uncond`` is given, the + teacher is
    CFG-composed ``v(pos) + g*(v(pos)−v(''))`` so Raw's 4.5 sample card
    maps. Scale 0 is the slider off: raw ``v(neu)``, never CFG of neu
    and never leftover-gated. ``v_neg`` is not a teacher.
    """
    if v_uncond is not None and float(guidance) > 0.0:
        return krea_cfg_compose(v_pos, v_uncond, guidance), v_neu
    return v_pos, v_neu


def krea_plus_neu_loss(
    pred_plus: torch.Tensor,
    tgt_plus: torch.Tensor,
    pred_zero: torch.Tensor,
    tgt_zero: torch.Tensor,
) -> torch.Tensor:
    """UNI velocity MSE: student +1 fits ``v(pos)``, student 0 fits ``v(neu)``.

    No minus MSE, no pair-odd, no ``h0 ± a``. Image analog of
    ``lm_plus_neu_loss`` — not lyric-hold. Not used by
    ``--lm_target embed``.
    """
    return F.mse_loss(pred_plus, tgt_plus) + F.mse_loss(pred_zero, tgt_zero)


def krea_embed_as_stacked(embeds: torch.Tensor) -> torch.Tensor:
    """Normalize TE hidden states to ``[B, T, L, D]``.

    Live Krea ``get_text_hidden_states`` is ``[B, T, 12, 2560]``.
    Dummy word-table encodes are ``[T, D]`` (one implicit layer) or
    stacked dummy TE ``[B, T, L, D]`` / ``[T, L, D]``.
    """
    if embeds.dim() == 4:
        return embeds
    if embeds.dim() == 3:
        # [T, L, D] if last dim is hidden; [B, T, D] has no layer stack.
        # Dummy stacked TE always writes 4D. Treat 3D as [B, T, D] → L=1.
        return embeds.unsqueeze(2)
    if embeds.dim() == 2:
        return embeds.unsqueeze(0).unsqueeze(2)
    raise ValueError(f"krea embeds must be 2D/3D/4D, got {tuple(embeds.shape)}")


def krea_embed_layer_weights(
    n_layers: int,
    *,
    early_layers: int = KREA_EMBED_EARLY_LAYERS,
    early_weight: float = KREA_EMBED_EARLY_WEIGHT,
    mid_late_weight: float = KREA_EMBED_MID_LATE_WEIGHT,
    device=None,
    dtype=None,
) -> torch.Tensor:
    """Per-layer weights. Layers ``0 .. early-1`` are down-weighted.

    Live diag: early (0–1) neu/plus cos≈0.91; mid/late (2–11) ≈0.57–0.73.
    Dummy uses 4 layers with the same split (0–1 vs 2–3).
    """
    n = max(int(n_layers), 1)
    weights = torch.full((n,), float(mid_late_weight), device=device, dtype=dtype)
    n_early = min(max(int(early_layers), 0), n)
    if n_early:
        weights[:n_early] = float(early_weight)
    return weights


def krea_embed_mse(
    pred: torch.Tensor,
    tgt: torch.Tensor,
    *,
    layer_weights: torch.Tensor | None = None,
) -> torch.Tensor:
    """Layer-weighted MSE on stacked TE embeds.

    Same ``[B, T, L, D]`` → token-and-hidden MSE per layer, then a
    weighted mean over layers. Different sequence length → pool T
    (live pads to 512 so this is dummy-only).
    """
    pred = krea_embed_as_stacked(pred)
    tgt = krea_embed_as_stacked(tgt)
    if pred.shape[-1] != tgt.shape[-1] or pred.shape[2] != tgt.shape[2]:
        raise ValueError(
            f"krea embed shapes must share layers/dim, got {tuple(pred.shape)} "
            f"vs {tuple(tgt.shape)}"
        )
    if pred.shape[1] != tgt.shape[1]:
        pred = pred.mean(dim=1, keepdim=True)
        tgt = tgt.mean(dim=1, keepdim=True)
    # (pred − tgt)² mean over B, T, D → [L]
    se = (pred - tgt).pow(2).mean(dim=(0, 1, 3))
    weights = layer_weights
    if weights is None:
        weights = krea_embed_layer_weights(
            int(se.numel()), device=se.device, dtype=se.dtype
        )
    else:
        weights = weights.to(device=se.device, dtype=se.dtype).reshape(-1)
        if int(weights.numel()) != int(se.numel()):
            raise ValueError(
                f"layer_weights length {int(weights.numel())} != {int(se.numel())} layers"
            )
    denom = weights.sum().clamp_min(1e-8)
    return (se * weights).sum() / denom


def krea_embed_cosine(
    pred: torch.Tensor,
    tgt: torch.Tensor,
    *,
    layer_weights: torch.Tensor | None = None,
) -> torch.Tensor:
    """Layer-weighted cosine of stacked embeds (diag-style per layer)."""
    pred = krea_embed_as_stacked(pred)
    tgt = krea_embed_as_stacked(tgt)
    if pred.shape[1] != tgt.shape[1]:
        pred = pred.mean(dim=1, keepdim=True)
        tgt = tgt.mean(dim=1, keepdim=True)
    n_layers = int(pred.shape[2])
    cosines = []
    for layer in range(n_layers):
        a = pred[:, :, layer].reshape(pred.shape[0], -1)
        b = tgt[:, :, layer].reshape(tgt.shape[0], -1)
        cosines.append(F.cosine_similarity(a, b, dim=-1, eps=1e-6).mean())
    stacked = torch.stack(cosines)
    weights = layer_weights
    if weights is None:
        weights = krea_embed_layer_weights(
            n_layers, device=stacked.device, dtype=stacked.dtype
        )
    else:
        weights = weights.to(device=stacked.device, dtype=stacked.dtype).reshape(-1)
    denom = weights.sum().clamp_min(1e-8)
    return (stacked * weights).sum() / denom


def krea_embed_uni_loss(
    pred: torch.Tensor,
    tgt: torch.Tensor,
    *,
    layer_weights: torch.Tensor | None = None,
    cosine_weight: float = KREA_EMBED_COSINE_WEIGHT,
) -> torch.Tensor:
    """TE-only UNI: student ``E_θ(neu)`` → stopgrad ``E_frozen(pos)``.

    Layer-weighted MSE on the full stack, plus optional ``1 − cos``.
    ``tgt`` must already be stopgrad. No DiT velocity term. No Anima
    structure lock.
    """
    mse = krea_embed_mse(pred, tgt, layer_weights=layer_weights)
    weight = float(cosine_weight)
    if weight <= 0.0:
        return mse
    cos = krea_embed_cosine(pred, tgt, layer_weights=layer_weights)
    return mse + weight * (1.0 - cos)


def krea_minus_canary(v_neg: torch.Tensor, v_uncond: torch.Tensor) -> torch.Tensor:
    """Minus CFG direction for logging only. Never a teacher."""
    return krea_cfg_direction(v_neg, v_uncond)


def krea_word_tokens(text: str) -> list[str]:
    """Whitespace tokens. Image analog of a lyric span — not Music 3 lyrics."""
    cleaned = str(text).replace(",", " ").replace(".", " ")
    return [part.lower() for part in cleaned.split() if part.strip()]


def krea_concept_words(positive: str, neutral: str) -> set[str]:
    """Words in the + prompt that are not in neu — do not hold these."""
    return set(krea_word_tokens(positive)) - set(krea_word_tokens(neutral))


def krea_unused_hold_mask(
    pos_tokens: Sequence[str],
    neu_tokens: Sequence[str],
    unused_words: Iterable[str] | None = None,
) -> list[bool]:
    """True where a pos token is held to encode(neu).

    Unused = shared skeleton (in neu) or a declared unused attribute.
    Concept words (in pos, not in neu) are never held, even if someone
    lists them as an attribute.
    """
    concept = set(pos_tokens) - set(neu_tokens)
    unused = {str(w).lower().strip() for w in (unused_words or ()) if str(w).strip()}
    neu_set = set(neu_tokens)
    mask: list[bool] = []
    for tok in pos_tokens:
        if tok in concept:
            mask.append(False)
        elif tok in unused or tok in neu_set:
            mask.append(True)
        else:
            mask.append(False)
    return mask


def krea_token_rows(embeds: torch.Tensor) -> torch.Tensor:
    """Token-major ``(T, F)`` view for hold math.

    Dummy encodes are ``(T, D)``. Live Krea ``get_text_hidden_states`` is
    ``(B, T, L, D)`` — flatten the layer stack into ``F`` so unused hold
    stays one row per tokenizer token, not ``T*L`` rows.
    """
    if embeds.dim() == 4:
        batch, tokens, layers, hidden = embeds.shape
        return embeds.reshape(batch * tokens, layers * hidden)
    if embeds.dim() == 3:
        return embeds.reshape(-1, embeds.shape[-1])
    return embeds


def krea_neu_token_lookup(
    neu_embeds: torch.Tensor,
    neu_tokens: Sequence[str],
) -> dict[str, torch.Tensor]:
    """First encode(neu) vector for each unused / shared token."""
    neu_embeds = krea_token_rows(neu_embeds)
    lookup: dict[str, torch.Tensor] = {}
    for i, tok in enumerate(neu_tokens):
        if i >= int(neu_embeds.shape[0]):
            break
        if tok not in lookup:
            lookup[tok] = neu_embeds[i]
    return lookup


def krea_hold_unused_embeds(
    pos_embeds: torch.Tensor,
    neu_embeds: torch.Tensor,
    pos_tokens: Sequence[str],
    neu_tokens: Sequence[str],
    unused_mask: Sequence[bool] | None = None,
) -> torch.Tensor:
    """Copy encode(neu) onto unused pos-token positions. Concept words stay."""
    mask = unused_mask
    if mask is None:
        mask = krea_unused_hold_mask(pos_tokens, neu_tokens)
    lookup = krea_neu_token_lookup(neu_embeds, neu_tokens)
    flat = krea_token_rows(pos_embeds).clone()
    for i, (tok, hold) in enumerate(zip(pos_tokens, mask)):
        if i >= int(flat.shape[0]):
            break
        if hold and tok in lookup:
            flat[i] = lookup[tok]
    return flat.reshape_as(pos_embeds)


def krea_unused_hold_loss(
    pred_embeds: torch.Tensor,
    neu_embeds: torch.Tensor,
    pos_tokens: Sequence[str],
    neu_tokens: Sequence[str],
    unused_mask: Sequence[bool] | None = None,
) -> torch.Tensor:
    """MSE unused student tokens → encode(neu). Concept words excluded.

    Empty unused span is 0 (a prompt with no shared tokens has no hold),
    not a fail-closed lyric span.
    """
    mask = unused_mask
    if mask is None:
        mask = krea_unused_hold_mask(pos_tokens, neu_tokens)
    held = krea_hold_unused_embeds(
        pred_embeds, neu_embeds, pos_tokens, neu_tokens, mask
    )
    pred_flat = krea_token_rows(pred_embeds)
    held_flat = krea_token_rows(held)
    keep = [i for i, flag in enumerate(mask) if flag and i < int(pred_flat.shape[0])]
    if not keep:
        return pred_flat.new_zeros(())
    idx = torch.tensor(keep, device=pred_flat.device)
    return F.mse_loss(pred_flat.index_select(0, idx), held_flat.index_select(0, idx))


def expand_attributes_krea(row: dict, *, prefix: bool = True) -> list[dict]:
    """Pin unused attributes. Age yaml prefixes; happy/smile stays bare.

    ``prefix=True`` (default, age slider): Music 3-style ``male a photo…``
    so unused gender is present both ways.

    ``prefix=False`` (happy/smile yaml ``bare_captions``): captions stay
    as written. Attributes remain unused-token pins only — the Anima
    smile lesson. Expansion does not make minus a teacher.
    """
    attributes = [
        str(a).strip() for a in (row.get("attributes") or []) if str(a).strip()
    ]
    if not attributes:
        return [dict(row)]
    if not prefix:
        item = dict(row)
        item["attributes"] = attributes
        return [item]
    rows = []
    for attribute in attributes:
        item = dict(row)
        item["attributes"] = attributes
        for key in ("target", "positive", "negative", "neutral"):
            value = row.get(key)
            if value:
                item[key] = f"{attribute} {value}"
        rows.append(item)
    return rows


# ---------------------------------------------------------------------------
# Sana 0.6B image UNI (opt-in). Not Music 3 lyric-hold.
# Cheap test backend: Efficient-Large-Model/Sana_600M_512px_diffusers.
# Train xattn (conceptmod default) or LoRA. 512px. Sample 20 steps, CFG 4.5.
# CFG compose is conceptmod ``backends/sana.py``:
#   v = v_u + g * (v_c - v_u)     # g != 1
#   v = v_c                       # g == 1
# The increment ``v(z,t,c) − v(z,t,'')`` is live at CFG 4.5.
# ---------------------------------------------------------------------------

SANA_MODEL_ID = "Efficient-Large-Model/Sana_600M_512px_diffusers"
SANA_RESOLUTION = 512
SANA_SAMPLE_STEPS = 20
SANA_CFG = 4.5
SANA_TRAIN_METHOD = "xattn"
SANA_LORA_TARGETS = ("to_q", "to_k", "to_v", "to_out.0")
SANA_CONTROL_PROMPT = "a bowl of fruit on a table"
SANA_DEFAULT_LR = 2e-5
SANA_DEFAULT_STEPS = 500
SANA_HOLD_WEIGHT = 1.0


def sana_cfg_delta(v_cond: torch.Tensor, v_uncond: torch.Tensor) -> torch.Tensor:
    """Velocity-space CFG increment from conceptmod: ``v(z,t,c) − v(z,t,'')``."""
    return v_cond - v_uncond


def sana_cfg(
    v_cond: torch.Tensor,
    v_uncond: torch.Tensor,
    guidance: float,
) -> torch.Tensor:
    """conceptmod ``SanaBackend`` sampling CFG.

    ``v_u + g * (v_c − v_u)`` when ``g != 1``. ``g == 1`` is identity
    (the backend skips the uncond pass). This is **not** the Z-Image /
    Krea compose ``v_c + g * (v_c − v_u)``.
    """
    g = float(guidance)
    if g == 1.0:
        return v_cond
    return v_uncond + g * (v_cond - v_uncond)


def sana_uni_teachers(
    v_pos: torch.Tensor,
    v_neu: torch.Tensor,
    v_uncond: torch.Tensor | None = None,
    *,
    guidance: float = SANA_CFG,
) -> tuple[torch.Tensor, torch.Tensor]:
    """UNI teachers: +1 → CFG-composed + concept, scale 0 → raw ``v(neu)``.

    The student that fits these teachers runs on the **neutral** caption
    (the infer path). Concept still comes from the + caption for teacher
    velocities only. CFG 4.5 is live, so the + teacher uses
    :func:`sana_cfg`. Scale 0 is the slider off: raw ``v(neu)``, never
    leftover-gated. Minus is not a teacher.
    """
    if v_uncond is not None and float(guidance) != 1.0:
        return sana_cfg(v_pos, v_uncond, guidance), v_neu
    return v_pos, v_neu


def sana_uni_loss(
    pred_plus: torch.Tensor,
    tgt_plus: torch.Tensor,
    pred_zero: torch.Tensor,
    tgt_zero: torch.Tensor,
    *,
    pred_unused: torch.Tensor | None = None,
    tgt_unused: torch.Tensor | None = None,
    unused_weight: float = 0.0,
    unused_token_hold: torch.Tensor | None = None,
    token_hold_weight: float = 0.0,
) -> torch.Tensor:
    """UNI velocity MSE: student +1/+0 on neu → (CFG(+), v_neu).

    Image analog of ``lm_plus_neu_loss`` — not lyric-hold. No minus MSE.
    Unused velocity hold and frozen-embed token hold are off by default.
    """
    return zimage_uni_loss(
        pred_plus,
        tgt_plus,
        pred_zero,
        tgt_zero,
        pred_unused=pred_unused,
        tgt_unused=tgt_unused,
        unused_weight=unused_weight,
        unused_token_hold=unused_token_hold,
        token_hold_weight=token_hold_weight,
    )


def sana_canary_minus(
    pred_minus: torch.Tensor, vel_neg: torch.Tensor
) -> dict[str, float | bool]:
    """Unscored −1 canary. Never a teacher."""
    return zimage_canary_minus(pred_minus, vel_neg)


SanaHoldError = ZImageHoldError


def expand_attributes_sana(row: dict) -> list[dict]:
    """Pin unused attributes onto target / pos / neu (and canary neg)."""
    return expand_attributes_zimage(row)


def sana_concept_token_ids(concept_words: str | Iterable[str], tokenize_fn) -> set[int]:
    """Bare word ids plus leading-space BPE pieces (Gemma/Sana).

    Standalone ``happy`` is not the same id as `` happy`` inside ``a happy person``.
    """
    ids = zimage_concept_token_ids(concept_words, tokenize_fn)
    if isinstance(concept_words, str):
        words = [w.strip() for w in concept_words.split(",") if w.strip()]
    else:
        words = [str(w).strip() for w in concept_words if str(w).strip()]
    for word in words:
        ids.update(int(t) for t in tokenize_fn(" " + word))
    return ids


def sana_unused_token_hold(
    pred_plus_embeds: torch.Tensor,
    neu_embeds: torch.Tensor,
    plus_ids: Sequence[int],
    neu_ids: Sequence[int],
    concept_ids: Iterable[int],
    *,
    fail_closed: bool = True,
) -> torch.Tensor:
    """MSE unused +1 tokens → encode(neu). Concept words are not held."""
    return zimage_unused_token_hold(
        pred_plus_embeds,
        neu_embeds,
        plus_ids,
        neu_ids,
        concept_ids,
        fail_closed=fail_closed,
    )


def sana_live_train_card() -> dict[str, object]:
    """Documented first GPU look (Modal / RunPod). CI never downloads this."""
    return {
        "model_id": SANA_MODEL_ID,
        "arch": "0.6B flow-matching linear DiT, Gemma-2, DC-AE",
        "train_method": SANA_TRAIN_METHOD,
        "lora": {
            "optional": True,
            "targets": list(SANA_LORA_TARGETS),
            "note": "conceptmod default for 0.6B is xattn, not LoRA",
        },
        "resolution": SANA_RESOLUTION,
        "sample_steps": SANA_SAMPLE_STEPS,
        "sample_guidance": SANA_CFG,
        "cfg_compose": "v_u + g * (v_c - v_u)",
        "cfg_delta": "v(z,t,c) - v(z,t,'')",
        "control_prompt": SANA_CONTROL_PROMPT,
        "lr": SANA_DEFAULT_LR,
        "steps": SANA_DEFAULT_STEPS,
        "device": "cuda:0",
        "recipe": "uni: student +1/0 on neu, teacher CFG(+)",
        "music3_default_untouched": {"lm_target": "v9", "pole_mode": "hidden"},
    }


def sana_live_train_command() -> str:
    return (
        "CUDA_VISIBLE_DEVICES=0 python conceptmod/textsliders/train_lora_sana.py \\\n"
        "  --name happy-sana \\\n"
        "  --prompts_file conceptmod/textsliders/data/prompts-sana.yaml \\\n"
        f"  --model_id {SANA_MODEL_ID} \\\n"
        "  --train_method xattn \\\n"
        f"  --resolution {SANA_RESOLUTION} \\\n"
        f"  --sample_steps {SANA_SAMPLE_STEPS} --sample_guidance {SANA_CFG} \\\n"
        f"  --control_prompt \"{SANA_CONTROL_PROMPT}\" \\\n"
        f"  --steps {SANA_DEFAULT_STEPS} --lr {SANA_DEFAULT_LR} --seed 7 --device 0 \\\n"
        "  --save_dir models/sana-slider"
    )
