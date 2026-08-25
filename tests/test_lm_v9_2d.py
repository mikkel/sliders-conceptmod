"""CPU gates on the Hub v9 LM recipe scored on the energetic×gender field."""

from __future__ import annotations

import pytest
import torch

from analysis.slider2d.field import E_SLIDER, Field2D
from analysis.slider2d.train import (
    axis_verdicts,
    lm_v9_specs,
    music3_pairs,
    pair_slider_dir,
    train_lm,
)
from conceptmod.textsliders.slider_targets import (
    lm_anchor_kappa,
    lm_anchor_targets,
    lm_hidden_targets,
    lm_ortho_hold,
    lm_pair_collapse,
    lm_perfect_fit_collapse,
    lm_project_odd_axis,
    lm_slider_loss,
    resolve_lm_target_mode,
)


_RESULTS = None


def results():
    global _RESULTS
    if _RESULTS is None:
        from analysis.slider2d.run_lm_v9 import run_v9

        _RESULTS = {r.name: r for r in run_v9(steps=200, seed=0)}
    return _RESULTS


def _ungated_pair(t: float = 0.5):
    field = Field2D()
    return field.embed("energetic", t), field.embed("calm", t), field.embed("song", t)


def test_target_mode_aliases_symmetric_flag():
    assert resolve_lm_target_mode(symmetric=True) == "symmetric"
    assert resolve_lm_target_mode(symmetric=False) == "faithful"
    assert resolve_lm_target_mode(symmetric=False, target_mode="symmetric") == "symmetric"
    assert resolve_lm_target_mode(target_mode="faithful") == "faithful"
    with pytest.raises(ValueError):
        resolve_lm_target_mode(target_mode="axis")


def test_faithful_targets_are_raw_poles():
    pos, neg, neu = torch.tensor([1.0, 0.9]), torch.tensor([-1.0, 0.4]), torch.tensor([0.0, -0.7])
    plus, minus = lm_hidden_targets(pos, neg, neu, target_mode="faithful")
    assert torch.allclose(plus, pos)
    assert torch.allclose(minus, neg)
    raw_plus, raw_minus = lm_hidden_targets(pos, neg, neu, symmetric=False)
    assert torch.allclose(plus, raw_plus)
    assert torch.allclose(minus, raw_minus)


def test_symmetric_target_mode_matches_boolean():
    pos, neg, neu = torch.tensor([1.0, 0.9]), torch.tensor([-1.0, 0.4]), torch.tensor([0.0, -0.7])
    a, b = lm_hidden_targets(pos, neg, neu, target_mode="symmetric")
    c, d = lm_hidden_targets(pos, neg, neu, symmetric=True)
    assert torch.allclose(a, c)
    assert torch.allclose(b, d)


def test_kappa_closed_form_hits_the_floor_when_r_is_zero():
    # Orthogonal (pos−neu) ⊥ (neg−neu) so r = 0, ρ² = 1.
    pos, neg, neu = torch.tensor([1.0, 1.0]), torch.tensor([-1.0, 1.0]), torch.tensor([0.0, 0.0])
    assert float(lm_pair_collapse(pos, neg, neu)) == pytest.approx(0.0, abs=1e-6)
    kappa = lm_anchor_kappa(pos, neg, neu, -0.9)
    assert float(kappa) == pytest.approx((1.0 / 19.0) ** 0.5, abs=1e-6)
    rho2 = 1.0
    assert float(lm_perfect_fit_collapse(kappa, rho2)) == pytest.approx(-0.9, abs=1e-5)


def test_kappa_clamped_to_unit_interval():
    # r = 0, floor = 0.5 → raw κ = √3 > 1 → clamp.
    ortho_pos, ortho_neg, neu = (
        torch.tensor([1.0, 1.0]),
        torch.tensor([-1.0, 1.0]),
        torch.tensor([0.0, 0.0]),
    )
    assert float(lm_anchor_kappa(ortho_pos, ortho_neg, neu, 0.5)) == pytest.approx(1.0)
    # Already antipodal (r = −1): any floor > −1 still allows κ = 1.
    anti_pos, anti_neg = torch.tensor([1.0, 0.0]), torch.tensor([-1.0, 0.0])
    assert float(lm_anchor_kappa(anti_pos, anti_neg, neu, -0.5)) == pytest.approx(1.0)
    # floor ≤ −1 forces pure symmetric
    assert float(lm_anchor_kappa(ortho_pos, ortho_neg, neu, -1.0)) == pytest.approx(0.0)


def test_leakage_floor_does_not_change_the_odd_axis():
    pos, neg, neu = _ungated_pair()
    kappa = lm_anchor_kappa(pos, neg, neu, -0.9)
    anc_plus, anc_minus = lm_anchor_targets(pos, neg, neu, kappa)
    odd_teacher = (pos - neg) / 2.0
    odd_anchor = (anc_plus - anc_minus) / 2.0
    assert torch.allclose(odd_anchor, odd_teacher, atol=1e-6)
    # Even part is a κ-blend of raw common into the symmetric (zero-even) poles.
    even_anchor = (anc_plus + anc_minus) / 2.0 - neu
    even_raw = (pos + neg) / 2.0 - neu
    assert torch.allclose(even_anchor, kappa * even_raw, atol=1e-6)


def test_anchor_weight_zero_ignores_floor_in_loss():
    pred_p, pred_m = torch.zeros(2), torch.zeros(2)
    tgt_p, tgt_m = torch.tensor([1.0, 0.0]), torch.tensor([-1.0, 0.0])
    anc_p, anc_m = torch.tensor([1.0, 4.0]), torch.tensor([-1.0, 4.0])
    bare = lm_slider_loss(pred_p, pred_m, tgt_p, tgt_m)
    floored = lm_slider_loss(
        pred_p, pred_m, tgt_p, tgt_m,
        anchor_plus=anc_p, anchor_minus=anc_m, anchor_weight=0.0,
    )
    assert float(bare) == pytest.approx(float(floored))


def test_projected_odd_axis_is_pure_slider():
    pos, neg, neu = _ungated_pair()
    plus, minus = lm_project_odd_axis(pos, neg, neu, E_SLIDER)
    odd = plus - neu
    assert torch.allclose((plus + minus) / 2, neu)
    assert torch.allclose(plus - neu, neu - minus)
    assert float(odd @ torch.tensor([0.0, 1.0])) == pytest.approx(0.0, abs=1e-6)
    assert float(odd @ E_SLIDER) == pytest.approx(float(((pos - neg) / 2) @ E_SLIDER))
    # Does not use attributes-prefixed captions — same ungated energetic/calm embeds.
    raw_odd = (pos - neg) / 2
    assert abs(float(raw_odd[1])) > 0.20
    assert abs(float(odd[1])) < 1e-6


def test_pair_slider_dir_is_declared_polarity_not_leaked_embed():
    pair = music3_pairs(False)[0]
    assert pair.positive.name == "energetic"
    assert pair.negative.name == "calm"
    direction = pair_slider_dir(pair)
    assert torch.allclose(direction, E_SLIDER)
    hold = lm_ortho_hold(
        torch.tensor([1.0, 0.4]),
        torch.tensor([-1.0, -0.4]),
        torch.zeros(2),
        direction,
    )
    assert float(hold) == pytest.approx(0.08, abs=1e-6)


def test_v9_default_does_not_use_attribute_prefixes():
    spec = next(s for s in lm_v9_specs() if s.name == "lm_v9")
    assert spec.kwargs.get("with_attrs") is False
    assert spec.kwargs.get("project_odd") is True
    assert spec.kwargs.get("anchor_weight", 1.0) == 0.0
    pairs = music3_pairs(False)
    assert [p.positive.name for p in pairs] == ["energetic"]


def test_v9_specs_cover_the_asked_cells():
    assert [s.name for s in lm_v9_specs()] == [
        "lm_raw",
        "lm_symmetric",
        "lm_symmetric_floor",
        "lm_v9_hub",
        "lm_v9",
        "lm_raw_attrs",
        "m3_nmse_axis",
    ]


def test_floor_without_anchor_matches_symmetric():
    field = Field2D()
    pairs = music3_pairs(False)
    a = train_lm(field, pairs, symmetric=True, steps=80, seed=0)
    b = train_lm(
        field, pairs, symmetric=True, leakage_floor=-0.9, anchor_weight=0.0, steps=80, seed=0
    )
    assert torch.allclose(a.delta(1.0), b.delta(1.0), atol=1e-6)
    r = results()
    assert r["lm_symmetric_floor"].metrics["leak_ratio"] == pytest.approx(
        r["lm_symmetric"].metrics["leak_ratio"], abs=1e-4
    )


def test_v9_is_right_on_slider_leak_and_collapse():
    r = results()["lm_v9"]
    ax = axis_verdicts(r.metrics)
    assert ax["slider"] == "right"
    assert ax["collapse"] == "right"
    assert ax["leak"] == "right"
    assert r.metrics["cos_slider_plus"] > 0.90
    assert r.metrics["cos_plus_minus"] < -0.85
    assert abs(r.metrics["leak_ratio"]) <= 0.20
    assert abs(r.metrics["leak_ratio"]) < 0.05
    # Must not regress to faithful even-mode collapse.
    assert r.metrics["cos_plus_minus"] < 0.0


def test_published_hub_v9_still_leaks_and_fixed_v9_does_not():
    r = results()
    hub_leak = abs(r["lm_v9_hub"].metrics["leak_ratio"])
    v9_leak = abs(r["lm_v9"].metrics["leak_ratio"])
    sym_leak = abs(r["lm_symmetric"].metrics["leak_ratio"])
    raw = results()["lm_raw"].metrics
    assert hub_leak > 0.20
    assert hub_leak == pytest.approx(sym_leak, abs=0.08)
    assert v9_leak <= 0.20
    assert v9_leak < sym_leak
    # Hub blend-back must not be *worse* than symmetric once we score the fix.
    # The published recipe itself may exceed symmetric; the default must not.
    assert v9_leak <= sym_leak + 1e-6
    assert raw["cos_plus_minus"] > -0.50
    assert r["lm_v9"].metrics["cos_plus_minus"] < -0.85


def test_raw_still_collapses_and_attrs_still_clean():
    raw = axis_verdicts(results()["lm_raw"].metrics)
    attrs = axis_verdicts(results()["lm_raw_attrs"].metrics)
    assert raw["collapse"] == "needs_help"
    assert raw["leak"] == "needs_help"
    assert attrs["slider"] == attrs["leak"] == attrs["collapse"] == "right"


def test_tf_nmse_axis_still_leaks_gender():
    r = results()["m3_nmse_axis"]
    ax = axis_verdicts(r.metrics)
    assert ax["slider"] == "right"
    assert ax["leak"] == "needs_help"
    assert ax["collapse"] == "right"
