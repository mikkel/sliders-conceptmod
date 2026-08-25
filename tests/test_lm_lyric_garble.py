"""Lyric-garble Goodhart: midpoint hidden MSE vs semantic KL.

Hidden-only cells treat high pair-odd cos as success. That is the
v15 lock that sang off-sheet. These cells add a sheet.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from analysis.slider2d.sheet import (
    EVEN_SCALE,
    LOOKS_LOCKED_COLLAPSE,
    LOOKS_LOCKED_COS,
    ODD_SCALE,
    STEPS,
    TOKEN_OOD_POS,
    TOKEN_POS,
    cell_table,
    energy_sheet,
    field_geometry,
    gender_sheet,
    looks_locked,
    policy_at,
    ray_argmax_stable,
    score_sheet,
    stays_on_sheet,
    teachers,
)
from conceptmod.textsliders.slider_targets import (
    lm_e_cleaned_captions,
    lm_hidden_targets,
    lm_pair_odd_sub_e,
    lm_policy_logits,
    lm_semantic_kl,
    lm_slider_loss,
)
from conceptmod.textsliders.train_lm_slider_music3 import parse_args


@pytest.fixture(scope="module")
def cells():
    return {row["name"]: row for row in cell_table(steps=STEPS, seed=0)}


def test_midpoint_is_not_a_real_caption():
    field = gender_sheet()
    pos, neg, neu = field.poles()
    t_plus, t_minus = field.midpoints()
    geo = field_geometry(field)
    assert not torch.allclose(t_plus, pos)
    assert not torch.allclose(t_minus, neg)
    assert geo["mid_neq_caption"] is True
    assert geo["caption_argmax"] == "pos"
    assert geo["minus_caption_argmax"] == "neg"
    assert geo["mid_argmax"] == "ood+"
    assert geo["minus_mid_argmax"] == "ood-"
    assert geo["caption_on_sheet"] is True
    assert geo["mid_on_sheet"] is False
    # Shared even is what the midpoint drops.
    assert float(pos[1]) == pytest.approx(EVEN_SCALE)
    assert float(t_plus[1]) == pytest.approx(0.0)
    assert float((pos - neg)[1]) == pytest.approx(0.0)
    del neu


def test_linear_readout_midpoint_softmax_prefers_a_third_token():
    field = gender_sheet()
    pos, _neg, _neu = field.poles()
    t_plus, _ = field.midpoints()
    p_cap = policy_at(pos, field)
    p_mid = policy_at(t_plus, field)
    assert p_cap["argmax"] == "pos"
    assert p_mid["argmax"] == "ood+"
    assert p_cap["on_sheet_mass"] >= 0.50
    assert p_mid["on_sheet_mass"] < 0.40
    assert p_mid["ood_mass"] > p_cap["ood_mass"]
    # Blend in hidden space is not a blend of the two policies.
    assert int(p_cap["argmax_i"]) == TOKEN_POS
    assert int(p_mid["argmax_i"]) == TOKEN_OOD_POS


def test_ray_argmax_is_stable_without_bias():
    field = gender_sheet()
    assert ray_argmax_stable(field) is True
    t_plus, _ = field.midpoints()
    names = {policy_at(s * t_plus, field)["argmax"] for s in (0.2, 1.0, 3.0)}
    assert names == {"ood+"}


def test_cleaned_energy_midpoint_still_off_sheet():
    field = energy_sheet()
    geo = field_geometry(field)
    assert geo["caption_argmax"] == "pos"
    assert geo["mid_argmax"] in {"ood+", "leak"}
    assert geo["cleaned_mid_argmax"] == "ood+"
    assert geo["cleaned_caption_argmax"] == "pos"
    pos, neg, neu = field.poles()
    leak = field.leak_dir()
    assert leak is not None
    t_plus, t_minus = lm_pair_odd_sub_e(pos, neg, neu, leak, slider_dir=field.short_u())
    c_plus, c_minus = lm_e_cleaned_captions(pos, neg, neu, leak, slider_dir=field.short_u())
    # Midpoint drops even; cleaned caption keeps it.
    assert float(t_plus[1]) == pytest.approx(0.0, abs=1e-6)
    assert float(c_plus[1]) == pytest.approx(EVEN_SCALE)
    assert float(t_plus[2]) == pytest.approx(0.0, abs=1e-6)
    assert float(c_plus[2]) == pytest.approx(0.0, abs=1e-6)
    assert policy_at(t_plus, field)["on_sheet"] is False
    assert policy_at(c_plus, field)["on_sheet"] is True
    del t_minus, c_minus


def test_hidden_mse_gender_is_the_goodhart(cells):
    for name in ("gender_hidden_odd", "gender_hidden_odd_even"):
        row = cells[name]
        assert row["looks_locked"] is True, name
        assert row["on_sheet"] is False, name
        assert row["goodhart"] is True, name
        assert row["c_plus"] >= LOOKS_LOCKED_COS
        assert row["c_plus_pair_odd"] >= LOOKS_LOCKED_COS
        assert row["collapse"] <= LOOKS_LOCKED_COLLAPSE
        assert row["argmax_plus"] == "ood+"
        assert row["argmax_minus"] == "ood-"
        assert row["ood_rate"] == pytest.approx(1.0)
        assert row["axis"]["lock_honest"] == "needs_help"


def test_faithful_hidden_mse_stays_on_sheet(cells):
    row = cells["gender_hidden_faithful_odd_even"]
    assert row["on_sheet"] is True
    assert row["goodhart"] is False
    assert row["argmax_plus"] == "pos"
    assert row["argmax_minus"] == "neg"
    # The garble is the midpoint teacher, not hidden MSE itself.


def test_semantic_kl_odd_even_stays_on_sheet(cells):
    row = cells["gender_kl_odd_even"]
    assert row["on_sheet"] is True
    assert row["goodhart"] is False
    assert row["argmax_plus"] == "pos"
    assert row["argmax_minus"] == "neg"
    assert row["on_sheet_mass"] >= 0.50
    assert row["ood_rate"] == pytest.approx(0.0)
    # Pair-odd cos looks worse than the hidden lock. That is expected.
    hidden = cells["gender_hidden_odd_even"]
    assert row["c_plus_pair_odd"] < hidden["c_plus_pair_odd"] - 0.05
    assert row["collapse"] > hidden["collapse"] + 0.20


def test_semantic_kl_odd_cannot_stay_on_sheet(cells):
    row = cells["gender_kl_odd"]
    assert row["on_sheet"] is False
    assert row["argmax_plus"] == "ood+"
    assert row["argmax_minus"] == "ood-"
    # Linear odd student is stuck on the midpoint ray.


def test_energy_hidden_pair_odd_and_sub_e_are_off_sheet(cells):
    raw = cells["energy_hidden_pair_odd"]
    sub = cells["energy_hidden_sub_e"]
    assert raw["on_sheet"] is False
    assert sub["on_sheet"] is False
    assert raw["goodhart"] is True
    assert sub["goodhart"] is True
    assert raw["argmax_plus"] in {"ood+", "leak"}
    assert sub["argmax_plus"] == "ood+"
    # Cleaning ê removes leftover and still leaves the sheet.
    assert sub["leftover"] < raw["leftover"] - 0.3
    assert raw["leftover"] == pytest.approx(0.80, abs=0.05)


def test_energy_kl_sub_e_on_sheet_hold_zero(cells):
    row = cells["energy_kl_sub_e"]
    assert row["on_sheet"] is True
    assert row["goodhart"] is False
    assert row["argmax_plus"] == "pos"
    assert row["argmax_minus"] == "neg"
    assert row["hold_weight"] == pytest.approx(0.0)
    assert row["leftover"] == pytest.approx(0.0, abs=0.05)
    hidden = cells["energy_hidden_sub_e"]
    assert row["c_plus_pair_odd"] < hidden["c_plus_pair_odd"] - 0.05


def test_energy_kl_raw_poles_on_sheet_but_keeps_leftover(cells):
    row = cells["energy_kl_pair_odd"]
    assert row["on_sheet"] is True
    assert row["argmax_plus"] == "pos"
    assert row["leftover"] == pytest.approx(0.80, abs=0.08)


def test_kl_teachers_are_real_captions_not_midpoints():
    gender = gender_sheet()
    pos, neg, _neu = gender.poles()
    t_plus, t_minus = gender.midpoints()
    k_plus, k_minus = teachers(gender, pole_mode="semantic_kl", teacher="pair_odd")
    h_plus, h_minus = teachers(gender, pole_mode="hidden", teacher="pair_odd")
    assert torch.allclose(k_plus, pos)
    assert torch.allclose(k_minus, neg)
    assert torch.allclose(h_plus, t_plus)
    assert torch.allclose(h_minus, t_minus)
    energy = energy_sheet()
    c_plus, c_minus = energy.cleaned_captions()
    m_plus, m_minus = energy.cleaned_midpoints()
    kl_plus, kl_minus = teachers(energy, pole_mode="semantic_kl", teacher="pair_odd_sub_e")
    hid_plus, hid_minus = teachers(energy, pole_mode="hidden", teacher="pair_odd_sub_e")
    assert torch.allclose(kl_plus, c_plus)
    assert torch.allclose(kl_minus, c_minus)
    assert torch.allclose(hid_plus, m_plus)
    assert torch.allclose(hid_minus, m_minus)
    assert not torch.allclose(c_plus, m_plus)


def test_lm_semantic_kl_prefers_caption_hidden_over_midpoint():
    field = gender_sheet()
    pos, neg, neu = field.poles()
    t_plus, t_minus = lm_hidden_targets(pos, neg, neu, target_mode="symmetric")
    weight = field.weight()
    at_caption = float(lm_semantic_kl(pos, neg, pos, neg, weight))
    at_mid = float(lm_semantic_kl(t_plus, t_minus, pos, neg, weight))
    assert at_caption == pytest.approx(0.0, abs=1e-5)
    assert at_mid > at_caption + 0.5
    hidden_at_mid = float(lm_slider_loss(t_plus, t_minus, t_plus, t_minus))
    hidden_at_caption = float(lm_slider_loss(pos, neg, t_plus, t_minus))
    assert hidden_at_mid == pytest.approx(0.0, abs=1e-6)
    assert hidden_at_caption > hidden_at_mid


def test_policy_logits_match_linear_head():
    hidden = torch.tensor([ODD_SCALE, EVEN_SCALE])
    weight = gender_sheet().weight()
    assert torch.allclose(lm_policy_logits(hidden, weight), F.linear(hidden, weight))


def test_looks_locked_is_not_on_sheet():
    locked = {"c_plus": 0.96, "c_plus_pair_odd": 0.96, "collapse": -0.95, "on_sheet_plus": 0.0, "on_sheet_minus": 0.0, "on_sheet_mass": 0.25}
    assert looks_locked(locked) is True
    assert stays_on_sheet(locked) is False
    honest = {"c_plus": 0.80, "c_plus_pair_odd": 0.80, "collapse": -0.28, "on_sheet_plus": 1.0, "on_sheet_minus": 1.0, "on_sheet_mass": 0.65}
    assert looks_locked(honest) is False
    assert stays_on_sheet(honest) is True


def test_live_trainer_default_is_untouched():
    """This PR is analysis/test/doc. Do not invent --pole_mode here."""
    args = parse_args(["--prompts_file", "prompts.yaml"])
    assert args.lm_target == "v9"
    assert not hasattr(args, "pole_mode")
    assert "pole_mode" not in vars(args)


def test_score_rejects_unknown_pole_mode():
    with pytest.raises(ValueError, match="pole_mode"):
        score_sheet("x", gender_sheet(), pole_mode="nope", steps=1)
    with pytest.raises(ValueError, match="pair_odd_sub_e"):
        teachers(gender_sheet(), pole_mode="hidden", teacher="pair_odd_sub_e")
