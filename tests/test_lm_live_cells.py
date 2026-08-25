"""Live gender-like + energy-like 2-D cells, and the slider-level v9 default.

Energy is not the old leak-0 cell. û must not be the pole names.
"""

from __future__ import annotations

import pytest
import torch

from analysis.slider2d.energy import (
    ENERGY_POLES,
    LIVE_ENERGY_ALIGNS,
    SHORT_NEG,
    SHORT_POS,
    EnergyLiveField2D,
    assert_not_pole_name_cheat,
    energy_pairs,
    energy_row_aligns,
    pair_odd_as_u_hides_energy,
    score_energy_policy,
)
from analysis.slider2d.field import E_SLIDER
from analysis.slider2d.live_compare import live_policy_table
from analysis.slider2d.mismatch import (
    LIVE_GENDER_V1_ALIGN,
    leak_cell_align,
    score_leak_policy,
)
from analysis.slider2d.train import music3_pairs, pair_slider_dir
from conceptmod.textsliders.slider_targets import (
    SLIDER_ALIGN_MIN,
    lm_mean_odd_align,
    lm_odd_align,
    lm_project_decisions,
    lm_teachers_mixed,
)
from conceptmod.textsliders.train_lm_slider_music3 import (
    parse_args,
    resolve_lm_loss_weights,
    resolve_lm_recipe,
    resolve_v9_gate,
)


_LIVE = None


def live_table():
    global _LIVE
    if _LIVE is None:
        _LIVE = live_policy_table(steps=200, seed=0)
    return _LIVE


def test_energy_declared_u_is_not_pole_names():
    field = EnergyLiveField2D()
    assert_not_pole_name_cheat(field)
    assert field.declared_captions == (SHORT_POS, SHORT_NEG)
    assert field.pole_names == ENERGY_POLES
    assert "energetic" not in {n for pair in field.pole_names for n in pair}
    assert "calm" not in {n for pair in field.pole_names for n in pair}
    assert field.declared_captions != field.pole_names[0]


def test_energy_aligns_match_live_not_the_095_cheat():
    field = EnergyLiveField2D()
    aligns = energy_row_aligns(field)
    assert aligns == pytest.approx(list(LIVE_ENERGY_ALIGNS), abs=1e-6)
    assert sum(abs(a - 0.48) < 1e-6 for a in aligns) == 2
    assert sum(abs(a - 0.68) < 1e-6 for a in aligns) == 2
    assert leak_cell_align() > 0.90
    assert max(aligns) < 0.80
    assert min(aligns) > LIVE_GENDER_V1_ALIGN
    mean = sum(aligns) / len(aligns)
    assert mean == pytest.approx(0.58, abs=1e-6)
    assert mean > SLIDER_ALIGN_MIN > LIVE_GENDER_V1_ALIGN
    # Setting û = pole-odd hides the middling numbers (old leak-0 lie).
    assert pair_odd_as_u_hides_energy(field)


def test_energy_u_is_intended_axis_pair_is_leaky():
    field = EnergyLiveField2D()
    for i, (pos_name, neg_name) in enumerate(field.pole_names):
        pos, neg = field.embed(pos_name), field.embed(neg_name)
        odd = (pos - neg) / 2.0
        assert abs(float(odd @ field.unused)) > 0.20
        assert float(lm_odd_align(pos, neg, field.declared_u)) == pytest.approx(
            field.aligns[i], abs=1e-6
        )
        assert not torch.allclose(odd / odd.norm(), field.declared_u, atol=1e-4)


def test_setting_u_to_pole_names_is_not_energy():
    """If someone sets û = pole-odd, project is identity and still leaks."""
    field = EnergyLiveField2D(aligns=(0.48,))
    r = score_energy_policy(
        "u_is_pole_odd",
        project_odd=True,
        hold_weight=1.0,
        use_declared_u=False,
        use_pole_odd_u=True,
        field=field,
    )
    assert r["odd_align"] > 0.99
    assert abs(r["leak_ratio"]) > 0.20
    assert r["axis"]["leak"] == "needs_help"
    assert r["pass"] is False
    # The short declared û on the same poles is the live 0.48, not 1.0.
    assert energy_row_aligns(field)[0] == pytest.approx(0.48, abs=1e-6)


def test_old_leak_cell_is_not_the_energy_cell():
    leak_pair = music3_pairs(False)[0]
    energy_pair = energy_pairs()[0]
    assert leak_pair.positive.name == "energetic"
    assert leak_pair.negative.name == "calm"
    assert energy_pair.positive.name != "energetic"
    assert energy_pair.negative.name != "calm"
    assert torch.allclose(pair_slider_dir(leak_pair), E_SLIDER)
    # Pole polarity on energy *is* the intended axis. The cheat is û = embed odd.
    assert torch.allclose(pair_slider_dir(energy_pair), E_SLIDER)
    field = EnergyLiveField2D()
    cheat = field.odd(0)
    assert not torch.allclose(cheat / cheat.norm(), field.declared_u, atol=1e-4)


def test_gender_like_hub_passes_always_project_fails():
    table = live_table()
    hub = next(r for r in table["gender"] if r["name"] == "hub")
    always = next(r for r in table["gender"] if r["name"] == "always_project_hold")
    assert hub["pass"] is True
    assert hub["mixed"] is False
    assert always["pass"] is False
    assert always["strength"] < 0.30
    assert always["cos_concept"] == pytest.approx(LIVE_GENDER_V1_ALIGN, abs=0.05)


def test_energy_like_hub_leaks_always_project_is_clean():
    table = live_table()
    hub = next(r for r in table["energy"] if r["name"] == "hub")
    always = next(r for r in table["energy"] if r["name"] == "always_project_hold")
    assert hub["pass"] is False
    assert abs(hub["leak_ratio"]) > 0.20
    assert always["pass"] is True
    assert abs(always["leak_ratio"]) <= 0.20
    assert always["mixed"] is False
    assert always["cos_intended"] > 0.90
    assert always["strength_on_u"] >= 0.50


def test_gated_050_fails_energy_mixed_rows():
    table = live_table()
    gender = next(r for r in table["gender"] if r["name"] == "gated_row_0.50")
    energy = next(r for r in table["energy"] if r["name"] == "gated_row_0.50")
    assert gender["pass"] is True
    assert gender["mixed"] is False
    assert energy["mixed"] is True
    assert energy["pass"] is False
    assert energy["decisions"] == [False, False, True, True]
    assert lm_teachers_mixed(energy["decisions"])


def test_slider_level_passes_both_cells_same_path():
    table = live_table()
    gender = next(r for r in table["gender"] if r["name"] == "slider_align_0.50")
    energy = next(r for r in table["energy"] if r["name"] == "slider_align_0.50")
    assert gender["pass"] is True
    assert energy["pass"] is True
    assert gender["mixed"] is False
    assert energy["mixed"] is False
    assert gender["decisions"] == [False]
    assert energy["decisions"] == [True, True, True, True]
    assert abs(energy["leak_ratio"]) <= 0.20
    assert energy["cos_intended"] > 0.90
    assert energy["strength_on_u"] >= 0.50
    assert gender["cos_concept"] > 0.90
    assert gender["strength"] > 0.80
    # Better than Hub on energy: leak-low, still on the intended û.
    hub = next(r for r in table["energy"] if r["name"] == "hub")
    assert abs(energy["leak_ratio"]) < abs(hub["leak_ratio"])
    assert energy["cos_intended"] > hub["cos_intended"]


def test_old_leak0_cell_always_project_still_clean():
    r = score_leak_policy("always_project_hold", project_odd=True, hold_weight=1.0)
    assert r["pass"] is True
    assert abs(r["leak_ratio"]) < 0.05
    assert r["odd_align"] > 0.90


def test_project_decisions_slider_vs_row():
    energy = list(LIVE_ENERGY_ALIGNS)
    gender = [LIVE_GENDER_V1_ALIGN]
    assert lm_project_decisions(energy, 0.50, "row") == [False, False, True, True]
    assert lm_project_decisions(energy, 0.50, "slider") == [True, True, True, True]
    assert lm_project_decisions(gender, 0.50, "slider") == [False]
    assert lm_project_decisions(gender, 0.50, "row") == [False]
    assert lm_project_decisions(energy, None, "slider") == [True, True, True, True]
    assert lm_teachers_mixed(lm_project_decisions(energy, 0.50, "row"))
    assert not lm_teachers_mixed(lm_project_decisions(energy, 0.50, "slider"))


def test_mean_odd_align_is_slider_level():
    field = EnergyLiveField2D()
    pairs = [(field.embed(p), field.embed(n)) for p, n in field.pole_names]
    mean = float(lm_mean_odd_align(pairs, field.declared_u))
    assert mean == pytest.approx(0.58, abs=1e-6)


def test_bare_trainer_is_slider_level_v9():
    args = parse_args(["--prompts_file", "prompts.yaml"])
    assert args.lm_target == "v9"
    assert resolve_lm_recipe(lm_target=args.lm_target, symmetric=True) == "v9"
    floor, scope = resolve_v9_gate(
        recipe="v9",
        project_align_min=args.project_align_min,
        project_align_scope=args.project_align_scope,
    )
    assert floor == pytest.approx(SLIDER_ALIGN_MIN)
    assert scope == "slider"
    hold, anchor = resolve_lm_loss_weights("v9", hold_weight=None, anchor_weight=None)
    assert hold == 1.0
    assert anchor == 0.0
    always_floor, always_scope = resolve_v9_gate(
        recipe="v9_always", project_align_min=0.50, project_align_scope="row"
    )
    assert always_floor is None
    assert resolve_lm_recipe(lm_target="v9_always", symmetric=True) == "v9_always"
    row = parse_args(
        [
            "--prompts_file",
            "prompts.yaml",
            "--project_align_scope",
            "row",
            "--project_align_min",
            "0.50",
        ]
    )
    r_floor, r_scope = resolve_v9_gate(
        recipe="v9",
        project_align_min=row.project_align_min,
        project_align_scope=row.project_align_scope,
    )
    assert r_scope == "row"
    assert r_floor == pytest.approx(0.50)
    _ = always_scope


def test_soft_per_row_blend_is_not_the_default():
    """A λ=align blend that keeps gender still leaks energy. Discarded."""
    # Gender: λ=0.20 keeps most of the pair (would pass). Energy: leftover
    # leak = (1-λ) β/α ≈ 0.52 * 1.83 ≈ 0.95 (fails leak).
    energy_align = 0.48
    beta_over_alpha = ((1.0 / energy_align**2) - 1.0) ** 0.5
    leak = (1.0 - energy_align) * beta_over_alpha
    assert leak > 0.20
    args = parse_args(["--prompts_file", "prompts.yaml"])
    assert args.lm_target == "v9"
    assert args.project_align_scope in (None, "slider")
