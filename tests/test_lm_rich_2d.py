"""Richer poles vs leak on the existing CPU fixture."""

from __future__ import annotations

import pytest
import torch

from analysis.slider2d.field import E_ATTR, E_SLIDER, Field2D
from analysis.slider2d.rich import (
    ALIGN_GENDER_V1,
    DEFAULT_ODD_BPM,
    DEFAULT_RICH,
    E_BPM,
    E_GENDER,
    E_RICH,
    E_U,
    POLE_NEG,
    POLE_POS,
    RichField,
    ScaledField2D,
    beats_v9_on_rich,
    core_table,
    field_at_short_align,
    score_rich,
    short_u_vs,
    slider_richness_sweep,
    unused_richness_sweep,
)
from analysis.slider2d.train import Residual, infer_dim, music3_pairs, subtract_axis, train_lm
from analysis.slider2d.faithful import leak_teacher, score_leak_lm
from analysis.slider2d.mismatch import MismatchField2D, mismatch_pairs, score_against_odd
from conceptmod.textsliders.slider_targets import lm_hidden_targets, lm_odd_align
from conceptmod.textsliders.train_lm_slider_music3 import parse_args, resolve_lm_recipe


_CORE = None


def core():
    global _CORE
    if _CORE is None:
        _CORE = {r["name"]: r for r in core_table(steps=160, seed=0)}
    return _CORE


def test_live_default_is_still_v9_hold_e():
    args = parse_args(["--prompts_file", "prompts.yaml"])
    assert args.lm_target == "v9"
    assert args.pole_mode == "hidden"
    assert resolve_lm_recipe(lm_target="v9", symmetric=True) == "v9"


def test_scaled_field_identity_matches_field2d():
    inner = Field2D()
    scaled = ScaledField2D(1.0, 1.0)
    for name in ("energetic", "calm", "song"):
        assert torch.allclose(scaled.embed(name, 0.5), inner.embed(name, 0.5), atol=1e-6)


def test_onaxis_slider_words_do_not_raise_leak():
    """Extra energetic/calm on û: leak falls (same unused, more slider)."""
    base = ScaledField2D(1.0, 1.0)
    rich = ScaledField2D(2.0, 1.0)
    pairs = music3_pairs(False)
    a = train_lm(base, pairs, target_mode="symmetric", steps=80, seed=0)
    b = train_lm(rich, pairs, target_mode="symmetric", steps=80, seed=0)
    from analysis.slider2d.train import score_residual

    la = abs(score_residual(a)["leak_ratio"])
    lb = abs(score_residual(b)["leak_ratio"])
    assert lb < la - 0.05


def test_subtract_e_zeros_declared_axis_keeps_the_rest():
    h0 = torch.zeros(4)
    h = torch.tensor([1.0, 0.7, 0.4, 0.8])
    cleaned = subtract_axis(h, h0, E_GENDER)
    assert float(cleaned @ E_GENDER) == pytest.approx(0.0, abs=1e-6)
    assert float(cleaned @ E_U) == pytest.approx(1.0, abs=1e-6)
    assert float(cleaned @ E_RICH) == pytest.approx(0.8, abs=1e-6)
    assert float(cleaned @ E_BPM) == pytest.approx(0.4, abs=1e-6)
    plus, minus = lm_hidden_targets(cleaned, -cleaned, h0, target_mode="symmetric")
    assert torch.allclose(plus, cleaned)
    assert torch.allclose(minus, -cleaned)


def test_rich_field_short_align_below_cheat_when_richness_is_off_axis():
    field = RichField(odd_gender=0.0, even_gender=0.0, odd_bpm=0.0)
    cap = field.slider / (field.slider ** 2 + field.rich ** 2) ** 0.5
    assert field.align_short() == pytest.approx(cap, abs=1e-5)
    assert field.align_short() < 0.90
    assert field.align_rich() == pytest.approx(1.0, abs=1e-5)


def test_field_at_short_align_hits_or_caps():
    hit = field_at_short_align(0.48, rich=0.0)
    assert hit.align_short() == pytest.approx(0.48, abs=0.01)
    capped = field_at_short_align(0.95, rich=DEFAULT_RICH)
    assert capped.align_short() < 0.90
    assert capped.odd_gender == pytest.approx(0.0, abs=1e-6)


def test_residual_dim_follows_the_field():
    field = RichField()
    assert infer_dim(field) == 4
    residual = Residual.create("odd_even", dim=4)
    assert residual.w_odd.numel() == 4


def test_core_faithful_copies_unused_and_richness():
    row = core()["faithful"]
    assert row["rich_kept"] == pytest.approx(1.0, abs=0.08)
    assert abs(row["leak_ratio"]) > 0.20
    assert row["axis"]["leak"] == "needs_help"


def test_core_pair_odd_keeps_odd_unused_and_richness():
    row = core()["pair_odd"]
    assert row["rich_kept"] == pytest.approx(1.0, abs=0.08)
    # Even gender dropped; odd gender + BPM remain.
    assert abs(row["leak_gender"]) == pytest.approx(RichField().odd_gender / RichField().slider, abs=0.08)
    assert abs(row["leak_bpm"]) == pytest.approx(DEFAULT_ODD_BPM / RichField().slider, abs=0.08)
    assert abs(row["leak_ratio"]) > 0.20


def test_core_v9_hold_e_shrinks_gender_keeps_rich_and_bpm():
    v9 = core()["v9"]
    odd = core()["pair_odd"]
    assert v9["rich_kept"] == pytest.approx(1.0, abs=0.10)
    assert abs(v9["leak_gender"]) < abs(odd["leak_gender"]) - 0.10
    assert abs(v9["leak_bpm"]) == pytest.approx(abs(odd["leak_bpm"]), abs=0.08)
    assert v9["cos_plus_minus"] < -0.85


def test_core_subtract_e_zeros_gender_keeps_rich_and_bpm():
    row = core()["pair_odd_sub_e"]
    assert abs(row["leak_gender"]) < 0.05
    assert row["rich_kept"] == pytest.approx(1.0, abs=0.08)
    assert abs(row["leak_bpm"]) == pytest.approx(DEFAULT_ODD_BPM / RichField().slider, abs=0.08)


def test_core_subtract_all_kills_unused_keeps_richness():
    row = core()["pair_odd_sub_all"]
    assert abs(row["leak_ratio"]) < 0.05
    assert row["rich_kept"] == pytest.approx(1.0, abs=0.08)
    assert row["cos_intended"] > 0.95
    assert row["axis"]["leak"] == "right"
    assert row["axis"]["rich"] == "right"


def test_core_project_short_kills_richness():
    row = core()["project_short"]
    assert abs(row["rich_kept"]) < 0.15
    assert abs(row["leak_ratio"]) < 0.05
    assert row["axis"]["rich"] == "needs_help"


def test_core_project_rich_keeps_span_of_slider_and_rich():
    row = core()["project_rich"]
    assert row["rich_kept"] > 0.50
    assert abs(row["leak_ratio"]) < 0.08
    assert row["cos_intended"] > 0.95


def test_core_project_odd_is_the_identity_cheat():
    row = core()["project_odd"]
    odd = core()["pair_odd"]
    assert abs(row["leak_ratio"]) == pytest.approx(abs(odd["leak_ratio"]), abs=0.08)
    assert row["rich_kept"] == pytest.approx(1.0, abs=0.08)


def test_slider_richness_does_not_raise_v9_leak():
    rows = slider_richness_sweep(
        unused_gender=0.34, unused_bpm=0.40, even_gender=0.0, steps=100, seed=0
    )
    v9 = [r for r in rows if r["recipe"] == "v9"]
    v9 = sorted(v9, key=lambda r: r["sweep_x"])
    leaks = [abs(r["leak_ratio"]) for r in v9]
    assert leaks[-1] <= leaks[0] + 0.08
    rich_proj = [r for r in rows if r["recipe"] == "project_short" and r["sweep_x"] >= 1.2]
    assert rich_proj
    assert all(abs(r["rich_kept"]) < 0.20 for r in rich_proj)
    rich_sub = [r for r in rows if r["recipe"] == "pair_odd_sub_e" and r["sweep_x"] >= 1.2]
    assert all(r["rich_kept"] > 0.85 for r in rich_sub)


def test_unused_bpm_breaks_v9_but_not_subtract_all():
    rows = unused_richness_sweep(which="bpm", steps=100, seed=0)
    high = [r for r in rows if abs(r["sweep_x"] - 1.5) < 1e-6]
    v9 = next(r for r in high if r["recipe"] == "v9")
    sub_e = next(r for r in high if r["recipe"] == "pair_odd_sub_e")
    sub_all = next(r for r in high if r["recipe"] == "pair_odd_sub_all")
    assert abs(v9["leak_ratio"]) > 0.20
    assert abs(sub_e["leak_ratio"]) > 0.20
    assert abs(sub_all["leak_ratio"]) < 0.08
    assert sub_all["rich_kept"] > 0.85


def test_pin_gender_leaves_bpm_and_slider_rich():
    field = RichField(pin_gender=True, pin_bpm=False)
    row = score_rich("pin_g", field, target_mode="symmetric", steps=80, seed=0)
    assert abs(row["leak_gender"]) < 0.05
    assert abs(row["leak_bpm"]) > 0.20
    assert row["rich_kept"] > 0.85


def test_pin_both_is_the_data_fix():
    field = RichField(pin_gender=True, pin_bpm=True)
    row = score_rich("pin_both", field, target_mode="faithful", steps=80, seed=0)
    assert abs(row["leak_ratio"]) < 0.05
    assert row["rich_kept"] > 0.85
    assert row["pole_cos_plus"] > 0.95


def test_gender_v1_project_still_kills_on_a_rich_clean_pair():
    clean = RichField(slider=1.2, rich=0.80, odd_gender=0.0, even_gender=0.0, odd_bpm=0.0)
    tilted = short_u_vs(clean.odd(), ALIGN_GENDER_V1, E_GENDER)
    assert float(lm_odd_align(clean.embed(POLE_POS), clean.embed(POLE_NEG), tilted)) == pytest.approx(
        ALIGN_GENDER_V1, abs=0.02
    )
    killed = score_rich("kill", clean, project_odd=True, slider_dir=tilted, steps=80, seed=0)
    kept = score_rich("keep", clean, target_mode="symmetric", steps=80, seed=0)
    assert killed["cos_intended"] < 0.50
    assert killed["rich_kept"] < 0.40
    assert kept["rich_kept"] > 0.85
    assert abs(kept["leak_ratio"]) < 0.05


def test_mismatch_cell_project_short_still_fails():
    field = MismatchField2D()
    residual = train_lm(
        field,
        mismatch_pairs(),
        target_mode="symmetric",
        project_odd=True,
        hold_weight=1.0,
        slider_dir=field.declared_u,
        steps=80,
        seed=0,
    )
    metrics = score_against_odd(residual, field.odd(), junk=field.junk)
    assert metrics["strength"] < 0.35
    assert metrics["cos_concept"] < 0.35


def test_field2d_subtract_e_matches_hard_v9():
    row = score_leak_lm("sub", target_mode="symmetric", subtract_dir=E_ATTR, steps=80, seed=0)
    assert abs(row["leak_ratio"]) < 0.05
    pos, neg, neu = leak_teacher(Field2D())
    cleaned = subtract_axis(pos, neu, E_ATTR)
    assert float((cleaned - neu) @ E_ATTR) == pytest.approx(0.0, abs=1e-6)
    assert float((cleaned - neu) @ E_SLIDER) == pytest.approx(1.0, abs=0.05)


def test_subtract_e_is_not_a_live_win_over_v9():
    """Oracle −ê beats leftover leak, but it is the hard v9 — do not wire it."""
    v9 = core()["v9"]
    sub = core()["pair_odd_sub_e"]
    # BPM leftover means −ê does not beat v9 on total leak by enough, or it
    # does only because ê is an oracle. Either way the live default stays.
    assert not beats_v9_on_rich(core()["project_short"], v9)
    assert not beats_v9_on_rich(core()["project_odd"], v9)
    assert sub["rich_kept"] >= v9["rich_kept"] - 0.05


def test_core_names_cover_the_asked_teachers():
    assert set(core()) >= {
        "faithful",
        "pair_odd",
        "v9",
        "pair_odd_sub_e",
        "pair_odd_sub_all",
        "project_short",
        "project_rich",
        "project_odd",
    }
