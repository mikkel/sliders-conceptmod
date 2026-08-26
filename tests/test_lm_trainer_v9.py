"""Trainer-level gates: the live LM loss is the 2-D projected-odd recipe.

No 8B model, no Hub, no GPU. Imports only the trainer helpers that
``train_lm_slider_music3.train`` calls.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
import yaml

from analysis.slider2d.field import E_SLIDER, Field2D
from analysis.slider2d.mismatch import LIVE_GENDER_V1_ALIGN, MismatchField2D
from conceptmod.textsliders.slider_targets import (
    DUAL_BAND_WEIGHT,
    EVEN_BLEND_SCALE,
    LEAK_HOLD_WEIGHT,
    UNUSED_E_OVERLAP_MAX,
    lm_anchor_kappa,
    lm_anchor_targets,
    lm_axis_hold,
    lm_blend_guard,
    lm_blind_projector,
    lm_blind_residual,
    lm_dual_band_pole_loss,
    lm_e_is_unused,
    lm_faithful_guard_e,
    lm_faithful_sub_e,
    lm_hidden_targets,
    lm_hold_dir,
    lm_next_token_logits,
    lm_ortho_hold,
    lm_pair_odd_sub_e,
    lm_project_odd_axis,
    lm_readout_null_basis,
    lm_semantic_null_pole_loss,
    lm_semantic_pole_loss,
    lm_slider_loss,
    lm_unit,
)
from conceptmod.textsliders.train_lm_slider_music3 import (
    HIDDEN_KL_WEIGHT,
    lm_train_loss,
    lm_train_targets,
    parse_args,
    resolve_leak_axis_captions,
    resolve_lm_loss_weights,
    resolve_lm_recipe,
    resolve_pole_mode,
    resolve_slider_axis_captions,
    resolve_v9_gate,
)


def _ungated_pair(t: float = 0.5):
    field = Field2D()
    return field.embed("energetic", t), field.embed("calm", t), field.embed("song", t)


def test_bare_parse_defaults_to_v9_and_symmetric():
    args = parse_args(["--prompts_file", "prompts.yaml"])
    assert args.lm_target == "v9"
    assert args.pole_mode == "hidden"
    assert args.symmetric is True
    assert resolve_lm_recipe(lm_target=args.lm_target, symmetric=args.symmetric) == "v9"
    assert resolve_pole_mode(args.pole_mode) == "hidden"
    hold, anchor = resolve_lm_loss_weights(
        "v9", hold_weight=args.hold_weight, anchor_weight=args.anchor_weight, leak_declared=False
    )
    assert hold == 0.0
    assert anchor == 0.0
    hold_e, _ = resolve_lm_loss_weights(
        "v9", hold_weight=None, anchor_weight=None, leak_declared=True
    )
    assert hold_e == pytest.approx(LEAK_HOLD_WEIGHT)
    assert args.leakage_floor is None
    floor, _scope = resolve_v9_gate(
        recipe="v9",
        project_align_min=args.project_align_min,
        project_align_scope=args.project_align_scope,
    )
    assert floor is None
    project = parse_args(["--prompts_file", "prompts.yaml", "--lm_target", "v9_project"])
    assert project.lm_target == "v9_project"
    p_floor, p_scope = resolve_v9_gate(
        recipe="v9_project", project_align_min=None, project_align_scope=None
    )
    assert p_floor == pytest.approx(0.50)
    assert p_scope == "slider"
    always = parse_args(["--prompts_file", "prompts.yaml", "--lm_target", "v9_always"])
    assert always.lm_target == "v9_always"
    sub = parse_args(["--prompts_file", "prompts.yaml", "--lm_target", "pair_odd_sub_e"])
    assert sub.lm_target == "pair_odd_sub_e"
    assert resolve_lm_recipe(lm_target="pair_odd_sub_e", symmetric=True) == "pair_odd_sub_e"
    sub_hold, sub_anchor = resolve_lm_loss_weights(
        "pair_odd_sub_e", hold_weight=None, anchor_weight=None, leak_declared=True
    )
    assert sub_hold == 0.0
    assert sub_anchor == 0.0
    gender_v16 = parse_args(
        [
            "--prompts_file",
            "prompts.yaml",
            "--lm_target",
            "faithful",
            "--pole_mode",
            "semantic_kl",
        ]
    )
    assert gender_v16.lm_target == "faithful"
    assert gender_v16.pole_mode == "semantic_kl"
    assert resolve_lm_recipe(lm_target="faithful", symmetric=True) == "faithful"
    energy_v16 = parse_args(
        [
            "--prompts_file",
            "prompts.yaml",
            "--lm_target",
            "faithful_sub_e",
            "--pole_mode",
            "semantic_kl",
        ]
    )
    assert energy_v16.lm_target == "faithful_sub_e"
    assert energy_v16.pole_mode == "semantic_kl"
    assert resolve_lm_recipe(lm_target="faithful_sub_e", symmetric=True) == "faithful_sub_e"
    faith_hold, faith_anchor = resolve_lm_loss_weights(
        "faithful_sub_e", hold_weight=None, anchor_weight=None, leak_declared=True
    )
    assert faith_hold == 0.0
    assert faith_anchor == 0.0
    always_floor, _ = resolve_v9_gate(
        recipe="v9_always", project_align_min=0.50, project_align_scope="row"
    )
    assert always_floor is None


def test_v9_requires_symmetric_polarity():
    with pytest.raises(ValueError, match="polarity step"):
        resolve_lm_recipe(lm_target="v9", symmetric=False)
    with pytest.raises(ValueError, match="polarity step"):
        resolve_lm_recipe(lm_target="v9_always", symmetric=False)
    with pytest.raises(ValueError, match="polarity step"):
        resolve_lm_recipe(lm_target="v9_project", symmetric=False)
    with pytest.raises(ValueError, match="polarity step"):
        resolve_lm_recipe(lm_target="pair_odd_sub_e", symmetric=False)
    assert resolve_lm_recipe(lm_target="faithful_sub_e", symmetric=False) == "faithful_sub_e"
    assert resolve_lm_recipe(lm_target="faithful", symmetric=False) == "faithful"
    assert (
        resolve_lm_recipe(lm_target="faithful_sub_e_if_unused", symmetric=False)
        == "faithful_sub_e_if_unused"
    )


def test_axis_is_declared_not_plus_minus_or_row_odd():
    meta = {
        "plus_label": "Loud",
        "minus_label": "Quiet",
        "slider_positive": "",
        "slider_negative": "",
    }
    assert resolve_slider_axis_captions(
        slider_positive=None, slider_negative=None, prompts_meta=meta
    ) is None
    cli = resolve_slider_axis_captions(
        slider_positive="energetic",
        slider_negative="calm",
        prompts_meta=meta,
    )
    assert cli == ("energetic", "calm")
    yaml_axis = resolve_slider_axis_captions(
        slider_positive=None,
        slider_negative=None,
        prompts_meta={**meta, "slider_positive": "energetic", "slider_negative": "calm"},
    )
    assert yaml_axis == ("energetic", "calm")
    # plus_label / minus_label are display names, not the axis.
    assert resolve_slider_axis_captions(
        slider_positive=None, slider_negative=None, prompts_meta=meta
    ) != (meta["plus_label"], meta["minus_label"])


def test_live_v9_targets_are_full_pair_odd():
    pos, neg, neu = _ungated_pair()
    plus, minus, anc_p, anc_m = lm_train_targets(pos, neg, neu, recipe="v9")
    leaked = lm_hidden_targets(pos, neg, neu, target_mode="symmetric")
    assert torch.allclose(plus, leaked[0])
    assert torch.allclose(minus, leaked[1])
    assert anc_p is None and anc_m is None
    projected_plus, _ = lm_project_odd_axis(pos, neg, neu, E_SLIDER)
    assert not torch.allclose(plus, projected_plus)
    raw_odd = (pos - neg) / 2
    assert abs(float(raw_odd[1])) > 0.20
    assert torch.allclose(plus - neu, raw_odd)


def test_v9_project_targets_are_lm_project_odd_axis():
    pos, neg, neu = _ungated_pair()
    plus, minus, anc_p, anc_m = lm_train_targets(
        pos, neg, neu, recipe="v9_project", slider_dir=E_SLIDER
    )
    expected_plus, expected_minus = lm_project_odd_axis(pos, neg, neu, E_SLIDER)
    assert torch.allclose(plus, expected_plus)
    assert torch.allclose(minus, expected_minus)
    assert anc_p is None and anc_m is None
    leaked = lm_hidden_targets(pos, neg, neu, target_mode="symmetric")
    assert not torch.allclose(plus, leaked[0])
    odd = plus - neu
    assert float(odd @ torch.tensor([0.0, 1.0])) == pytest.approx(0.0, abs=1e-6)


def test_live_v9_loss_is_odd_plus_axis_hold():
    pos, neg, neu = _ungated_pair()
    tgt_plus, tgt_minus, _, _ = lm_train_targets(pos, neg, neu, recipe="v9")
    pred_plus = neu + torch.tensor([0.8, 0.4])
    pred_minus = neu + torch.tensor([-0.7, -0.3])
    leak = torch.tensor([0.0, 1.0])
    got = lm_train_loss(
        pred_plus,
        pred_minus,
        tgt_plus,
        tgt_minus,
        neu=neu,
        leak_dir=leak,
        hold_weight=LEAK_HOLD_WEIGHT,
    )
    hold = lm_axis_hold(pred_plus, pred_minus, neu, leak)
    expected = lm_slider_loss(
        pred_plus, pred_minus, tgt_plus, tgt_minus, hold=hold, hold_weight=LEAK_HOLD_WEIGHT
    )
    assert float(got) == pytest.approx(float(expected), abs=1e-6)
    # Declared û that is already ⊥ ê does not change the hold.
    got_u = lm_train_loss(
        pred_plus,
        pred_minus,
        tgt_plus,
        tgt_minus,
        neu=neu,
        leak_dir=leak,
        slider_dir=E_SLIDER,
        hold_weight=LEAK_HOLD_WEIGHT,
    )
    assert float(got_u) == pytest.approx(float(expected), abs=1e-6)


def test_v9_hold_drops_slider_component_of_e():
    """ê = û is a synonym of the slider; hold must not punch it."""
    pos, neg, neu = _ungated_pair()
    tgt_plus, tgt_minus, _, _ = lm_train_targets(pos, neg, neu, recipe="v9")
    pred_plus = neu + torch.tensor([0.8, 0.4])
    pred_minus = neu + torch.tensor([-0.7, -0.3])
    no_hold = lm_train_loss(
        pred_plus, pred_minus, tgt_plus, tgt_minus, neu=neu, hold_weight=0.0
    )
    punched = lm_train_loss(
        pred_plus,
        pred_minus,
        tgt_plus,
        tgt_minus,
        neu=neu,
        leak_dir=E_SLIDER,
        slider_dir=E_SLIDER,
        hold_weight=LEAK_HOLD_WEIGHT,
    )
    assert lm_hold_dir(E_SLIDER, slider_dir=E_SLIDER, mode="slider") is None
    assert float(punched) == pytest.approx(float(no_hold), abs=1e-6)
    leftover = torch.tensor([0.6, 0.8])
    held = lm_hold_dir(leftover, slider_dir=E_SLIDER, mode="slider")
    assert held is not None
    assert abs(float(held @ E_SLIDER)) < 1e-6
    mixed = lm_train_loss(
        pred_plus,
        pred_minus,
        tgt_plus,
        tgt_minus,
        neu=neu,
        leak_dir=leftover,
        slider_dir=E_SLIDER,
        hold_weight=LEAK_HOLD_WEIGHT,
    )
    expected_hold = lm_axis_hold(pred_plus, pred_minus, neu, held)
    expected = lm_slider_loss(
        pred_plus, pred_minus, tgt_plus, tgt_minus, hold=expected_hold, hold_weight=LEAK_HOLD_WEIGHT
    )
    assert float(mixed) == pytest.approx(float(expected), abs=1e-6)


def test_v9_project_loss_is_projected_odd_plus_ortho_hold():
    pos, neg, neu = _ungated_pair()
    tgt_plus, tgt_minus, _, _ = lm_train_targets(
        pos, neg, neu, recipe="v9_project", slider_dir=E_SLIDER
    )
    pred_plus = neu + torch.tensor([0.8, 0.4])
    pred_minus = neu + torch.tensor([-0.7, -0.3])
    got = lm_train_loss(
        pred_plus,
        pred_minus,
        tgt_plus,
        tgt_minus,
        neu=neu,
        slider_dir=E_SLIDER,
        hold_weight=1.0,
    )
    hold = lm_ortho_hold(pred_plus, pred_minus, neu, E_SLIDER)
    expected = lm_slider_loss(
        pred_plus, pred_minus, tgt_plus, tgt_minus, hold=hold, hold_weight=1.0
    )
    assert float(got) == pytest.approx(float(expected), abs=1e-6)


def test_v9_does_not_default_to_hub_floor():
    pos, neg, neu = _ungated_pair()
    v9_plus, v9_minus, _, _ = lm_train_targets(pos, neg, neu, recipe="v9")
    hub_plus, hub_minus, anc_p, anc_m = lm_train_targets(
        pos, neg, neu, recipe="hub", leakage_floor=-0.9
    )
    kappa = lm_anchor_kappa(pos, neg, neu, -0.9)
    expected_anc = lm_anchor_targets(pos, neg, neu, kappa)
    leaked = lm_hidden_targets(pos, neg, neu, target_mode="symmetric")
    assert torch.allclose(hub_plus, leaked[0])
    assert torch.allclose(hub_minus, leaked[1])
    assert torch.allclose(anc_p, expected_anc[0])
    assert torch.allclose(anc_m, expected_anc[1])
    # v9 teacher is the same full odd as Hub; Hub adds κ-blend anchors, v9 holds ê.
    assert torch.allclose(v9_plus, hub_plus)
    hold, anchor = resolve_lm_loss_weights("hub", hold_weight=None, anchor_weight=None)
    assert hold == 0.0
    assert anchor == pytest.approx(0.3)


def test_faithful_and_symmetric_flags_stay_behind_recipe():
    pos, neg, neu = torch.tensor([1.0, 0.9]), torch.tensor([-1.0, 0.4]), torch.tensor([0.0, -0.7])
    plus, minus, _, _ = lm_train_targets(pos, neg, neu, recipe="faithful")
    assert torch.allclose(plus, pos)
    assert torch.allclose(minus, neg)
    plus, minus, _, _ = lm_train_targets(pos, neg, neu, recipe="symmetric")
    expected = lm_hidden_targets(pos, neg, neu, target_mode="symmetric")
    assert torch.allclose(plus, expected[0])
    assert torch.allclose(minus, expected[1])


def test_documented_prompt_files_declare_the_axis():
    root = Path(__file__).resolve().parents[1] / "conceptmod" / "textsliders" / "data"
    gender = yaml.safe_load((root / "prompts-gender-v4.yaml").read_text())
    energy = yaml.safe_load((root / "prompts-energy-v4.yaml").read_text())
    assert gender["slider_positive"].startswith("A woman is singing")
    assert gender["slider_negative"].startswith("A man is singing")
    assert energy["slider_positive"].lower().startswith("extremely high energy")
    assert energy["slider_negative"].lower().startswith("extremely quiet")
    assert not gender.get("leak_positive")
    assert not gender.get("leak_negative")
    leftover = f"{energy['leak_positive']} {energy['leak_negative']}".lower()
    assert "bpm" in leftover and "mix" in leftover
    for banned in ("slammed", "sparse", "loud", "quiet", "dense", "airy"):
        assert banned not in leftover
    assert energy["leak_positive"] != energy["slider_positive"]
    # Display labels are not the axis.
    assert gender["slider_positive"] != gender["plus_label"]
    assert energy["slider_positive"] != energy["plus_label"]
    leak = resolve_leak_axis_captions(
        leak_positive=None, leak_negative=None, prompts_meta=energy
    )
    assert leak == (energy["leak_positive"], energy["leak_negative"])
    assert resolve_leak_axis_captions(
        leak_positive=None, leak_negative=None, prompts_meta=gender
    ) is None


def test_project_align_min_does_not_change_v9_default_targets():
    pos, neg, neu = _ungated_pair()
    plus, minus, _, _ = lm_train_targets(pos, neg, neu, recipe="v9")
    gated = lm_train_targets(
        pos, neg, neu, recipe="v9", slider_dir=E_SLIDER, project_align_min=0.50
    )
    assert torch.allclose(plus, gated[0])
    assert torch.allclose(minus, gated[1])
    symmetric = lm_hidden_targets(pos, neg, neu, target_mode="symmetric")
    assert torch.allclose(plus, symmetric[0])


def test_project_align_min_falls_back_to_pair_symmetric():
    field = MismatchField2D()
    pos, neg, neu = field.rich_pair()
    projected = lm_project_odd_axis(pos, neg, neu, field.declared_u)
    always = lm_train_targets(pos, neg, neu, recipe="v9_always", slider_dir=field.declared_u)
    assert torch.allclose(always[0], projected[0])
    fallback = lm_train_targets(
        pos, neg, neu, recipe="v9_project", slider_dir=field.declared_u, project_align_min=0.50
    )
    slider_level = lm_train_targets(
        pos, neg, neu, recipe="v9_project", slider_dir=field.declared_u, should_project=False
    )
    symmetric = lm_hidden_targets(pos, neg, neu, target_mode="symmetric")
    assert torch.allclose(fallback[0], symmetric[0])
    assert torch.allclose(fallback[1], symmetric[1])
    assert torch.allclose(slider_level[0], symmetric[0])
    assert not torch.allclose(fallback[0], projected[0])
    from conceptmod.textsliders.slider_targets import lm_odd_align

    assert float(lm_odd_align(pos, neg, field.declared_u)) == pytest.approx(
        LIVE_GENDER_V1_ALIGN, abs=1e-6
    )


def test_v9_targets_do_not_require_slider_dir():
    pos, neg, neu = _ungated_pair()
    plus, minus, _, _ = lm_train_targets(pos, neg, neu, recipe="v9", slider_dir=None)
    symmetric = lm_hidden_targets(pos, neg, neu, target_mode="symmetric")
    assert torch.allclose(plus, symmetric[0])
    with pytest.raises(ValueError, match="declared slider_dir"):
        lm_train_targets(pos, neg, neu, recipe="v9_project", slider_dir=None)
    with pytest.raises(ValueError, match="leak_dir or slider_dir"):
        lm_train_loss(
            neu, neu, neu, neu, neu=neu, slider_dir=None, leak_dir=None, hold_weight=1.0
        )


def test_pair_odd_sub_e_drops_e_perp_not_raw_e():
    pos, neg, neu = _ungated_pair()
    leftover = torch.tensor([0.4, 0.9])
    plus, minus, anc_p, anc_m = lm_train_targets(
        pos, neg, neu, recipe="pair_odd_sub_e", slider_dir=E_SLIDER, leak_dir=leftover
    )
    expected_plus, expected_minus = lm_pair_odd_sub_e(
        pos, neg, neu, leftover, slider_dir=E_SLIDER
    )
    assert torch.allclose(plus, expected_plus)
    assert torch.allclose(minus, expected_minus)
    assert anc_p is None and anc_m is None
    held = lm_hold_dir(leftover, slider_dir=E_SLIDER, mode="slider")
    assert held is not None
    odd = plus - neu
    assert abs(float(odd @ lm_unit(held))) < 1e-6
    pair_odd = lm_hidden_targets(pos, neg, neu, target_mode="symmetric")
    assert not torch.allclose(plus, pair_odd[0])
    raw_unit = lm_unit(leftover)
    raw_axis = (pos - neg) / 2.0
    raw_odd = raw_axis - (raw_axis @ raw_unit) * raw_unit
    assert not torch.allclose(odd, raw_odd)


def test_pair_odd_sub_e_needs_leak_and_slider():
    pos, neg, neu = _ungated_pair()
    leftover = torch.tensor([0.0, 1.0])
    with pytest.raises(ValueError, match="declared leak_dir"):
        lm_train_targets(pos, neg, neu, recipe="pair_odd_sub_e", slider_dir=E_SLIDER)
    with pytest.raises(ValueError, match="declared slider_dir"):
        lm_train_targets(pos, neg, neu, recipe="pair_odd_sub_e", leak_dir=leftover)


def test_pair_odd_sub_e_matches_highd_teacher():
    from analysis.slider2d.highd import energy_field, leftover_only_e, teacher_poles

    field = energy_field()
    axis = leftover_only_e(field)
    pos, neg, neu = field.poles()
    expected_plus, expected_minus = teacher_poles(
        field, teacher="pair_odd_sub_e", leak_dir=axis
    )
    plus, minus, _, _ = lm_train_targets(
        pos, neg, neu, recipe="pair_odd_sub_e", slider_dir=field.short_u(), leak_dir=axis
    )
    assert torch.allclose(plus, expected_plus, atol=1e-6)
    assert torch.allclose(minus, expected_minus, atol=1e-6)
    pair_odd_plus, _ = teacher_poles(field, teacher="pair_odd")
    assert not torch.allclose(plus, pair_odd_plus)
    raw_plus, _ = teacher_poles(field, teacher="pair_odd_sub_raw_e", leak_dir=axis)
    assert not torch.allclose(plus, raw_plus)


def test_leaky_v4_yamls_declare_leftover_not_slider():
    root = Path(__file__).resolve().parents[1] / "conceptmod" / "textsliders" / "data"
    leaky = {
        "prompts-energy-v4.yaml": ("slammed", "sparse", "loud", "quiet", "dense", "airy"),
        "prompts-tempo-v4.yaml": ("fast", "slow", "frantic", "bpm"),
        "prompts-distortion-v4.yaml": ("distort", "overdrive", "fuzz", "acoustic", "unplugged"),
        "prompts-rapslow-v4.yaml": ("rap", "sung", "spoken", "slow"),
        "prompts-breath-v4.yaml": ("inhale", "breath", "airless", "mouth air"),
        "prompts-rhyme-v4.yaml": ("rhyme", "couplet", "verse", "aabb"),
        "prompts-triphop-v4.yaml": ("trip-hop", "dusty", "glossy", "vinyl", "radio"),
    }
    for name, banned in leaky.items():
        blob = yaml.safe_load((root / name).read_text())
        assert blob.get("slider_positive") and blob.get("slider_negative")
        leak = resolve_leak_axis_captions(
            leak_positive=None, leak_negative=None, prompts_meta=blob
        )
        assert leak is not None
        leftover = f"{leak[0]} {leak[1]}".lower()
        slider = f"{blob['slider_positive']} {blob['slider_negative']}".lower()
        assert leftover != slider
        for word in banned:
            assert word not in leftover, f"{name} leftover contains {word!r}"
    live = yaml.safe_load((root / "prompts-live-v4.yaml").read_text())
    assert not live.get("leak_positive")
    gender = yaml.safe_load((root / "prompts-gender-v4.yaml").read_text())
    assert not gender.get("leak_positive")


def test_faithful_sub_e_targets_are_e_cleaned_real_poles_not_h0_plus_a():
    """ê-cleaned real poles: midpoint stays ½(h++h−), not t± = h0 ± a."""
    pos, neg, neu = _ungated_pair()
    leftover = torch.tensor([0.4, 0.9])
    plus, minus, anc_p, anc_m = lm_train_targets(
        pos, neg, neu, recipe="faithful_sub_e", slider_dir=E_SLIDER, leak_dir=leftover
    )
    expected_plus, expected_minus = lm_faithful_sub_e(
        pos, neg, neu, leftover, slider_dir=E_SLIDER
    )
    assert torch.allclose(plus, expected_plus)
    assert torch.allclose(minus, expected_minus)
    assert anc_p is None and anc_m is None
    assert torch.allclose((plus + minus) / 2, (pos + neg) / 2)
    pair_odd = lm_hidden_targets(pos, neg, neu, target_mode="symmetric")
    assert not torch.allclose(plus, pair_odd[0])
    assert not torch.allclose(plus, pos)
    pair_sub_plus, _ = lm_pair_odd_sub_e(pos, neg, neu, leftover, slider_dir=E_SLIDER)
    assert not torch.allclose(plus, pair_sub_plus)
    common = (pos + neg) / 2 - neu
    assert torch.allclose((plus + minus) / 2 - neu, common)
    held = lm_hold_dir(leftover, slider_dir=E_SLIDER, mode="slider")
    assert held is not None
    odd = (plus - minus) / 2
    assert abs(float(odd @ lm_unit(held))) < 1e-6


def test_faithful_sub_e_if_unused_is_wired_and_not_the_default():
    args = parse_args(
        ["--prompts_file", "prompts.yaml", "--lm_target", "faithful_sub_e_if_unused"]
    )
    assert args.lm_target == "faithful_sub_e_if_unused"
    assert args.pole_mode == "hidden"
    assert resolve_lm_recipe(lm_target="faithful_sub_e_if_unused", symmetric=True) == (
        "faithful_sub_e_if_unused"
    )
    hold, anchor = resolve_lm_loss_weights(
        "faithful_sub_e_if_unused",
        hold_weight=None,
        anchor_weight=None,
        leak_declared=True,
    )
    assert hold == 0.0
    assert anchor == 0.0
    bare = parse_args(["--prompts_file", "prompts.yaml"])
    assert bare.lm_target == "v9"
    assert bare.pole_mode == "hidden"


def test_faithful_sub_e_if_unused_subtracts_only_when_leftover_is_unused():
    pos, neg, neu = _ungated_pair()
    leftover = torch.tensor([0.0, 1.0])
    unused, overlap = lm_e_is_unused(pos, neg, leftover, slider_dir=E_SLIDER)
    assert unused is True
    assert float(overlap) < UNUSED_E_OVERLAP_MAX
    plus, minus, _, _ = lm_train_targets(
        pos,
        neg,
        neu,
        recipe="faithful_sub_e_if_unused",
        slider_dir=E_SLIDER,
        leak_dir=leftover,
    )
    want = lm_faithful_sub_e(pos, neg, neu, leftover, slider_dir=E_SLIDER)
    assert torch.allclose(plus, want[0])
    assert torch.allclose(minus, want[1])
    # ê restates the pair: a is mostly orthogonal to û, ê = a.
    rest_neu = torch.zeros(2)
    rest_a = torch.tensor([0.3, 1.0])
    rest_c = torch.tensor([0.5, 0.0])
    rest_pos = rest_neu + rest_a + rest_c
    rest_neg = rest_neu - rest_a + rest_c
    restates, rest_overlap = lm_e_is_unused(
        rest_pos, rest_neg, rest_a, slider_dir=E_SLIDER
    )
    assert restates is False
    assert float(rest_overlap) > UNUSED_E_OVERLAP_MAX
    keep_plus, keep_minus, _, _ = lm_train_targets(
        rest_pos,
        rest_neg,
        rest_neu,
        recipe="faithful_sub_e_if_unused",
        slider_dir=E_SLIDER,
        leak_dir=rest_a,
    )
    assert torch.allclose(keep_plus, rest_pos)
    assert torch.allclose(keep_minus, rest_neg)
    clean_plus, clean_minus, _, _ = lm_train_targets(
        pos, neg, neu, recipe="faithful_sub_e_if_unused"
    )
    assert torch.allclose(clean_plus, pos)
    assert torch.allclose(clean_minus, neg)


def test_faithful_even_blend_is_wired_and_not_the_default():
    args = parse_args(
        ["--prompts_file", "prompts.yaml", "--lm_target", "faithful_even_blend"]
    )
    assert args.lm_target == "faithful_even_blend"
    assert args.pole_mode == "hidden"
    assert args.even_blend_scale == EVEN_BLEND_SCALE
    assert resolve_lm_recipe(lm_target="faithful_even_blend", symmetric=True) == (
        "faithful_even_blend"
    )
    hold, anchor = resolve_lm_loss_weights(
        "faithful_even_blend",
        hold_weight=None,
        anchor_weight=None,
        leak_declared=True,
    )
    assert hold == 0.0
    assert anchor == 0.0
    bare = parse_args(["--prompts_file", "prompts.yaml"])
    assert bare.lm_target == "v9"
    assert bare.pole_mode == "hidden"
    assert bare.even_blend_scale == EVEN_BLEND_SCALE
    src = Path("conceptmod/textsliders/train_lm_slider_music3.py").read_text()
    assert '"lm_target": recipe' in src
    assert '"even_blend_scale": float(args.even_blend_scale)' in src
    pos, neg, neu = _ungated_pair()
    plus, minus, _, _ = lm_train_targets(
        pos, neg, neu, recipe="faithful_even_blend"
    )
    assert torch.allclose(plus, pos)
    assert torch.allclose(minus, neg)
    leftover = torch.tensor([0.0, 1.0])
    gated, _, _, _ = lm_train_targets(
        pos,
        neg,
        neu,
        recipe="faithful_even_blend",
        slider_dir=E_SLIDER,
        leak_dir=leftover,
    )
    pair_odd = lm_hidden_targets(pos, neg, neu, target_mode="symmetric")
    assert not torch.allclose(gated, pair_odd[0])
    from analysis.slider2d.exam import divergent_field
    from conceptmod.textsliders.slider_targets import leftover_bipolar

    field = divergent_field()
    d_pos, d_neg, d_neu = field.poles(0)
    blend_plus, blend_minus, _, _ = lm_train_targets(
        d_pos,
        d_neg,
        d_neu,
        recipe="faithful_even_blend",
        slider_dir=field.short_u(),
        leak_dir=field.declared_e(),
        even_dir=field.declared_e_even(),
        even_scale=EVEN_BLEND_SCALE,
    )
    metrics = leftover_bipolar(blend_plus - d_neu, blend_minus - d_neu)
    assert metrics["leak_frac"] < 0.0
    assert metrics["leak_frac"] > -0.80
    mid_plus, _ = lm_hidden_targets(d_pos, d_neg, d_neu, target_mode="symmetric")
    assert not torch.allclose(blend_plus, mid_plus, atol=1e-4)
    assert not torch.allclose(blend_plus, d_pos, atol=1e-4)


def test_faithful_plus_is_wired_and_not_the_default():
    args = parse_args(
        ["--prompts_file", "prompts.yaml", "--lm_target", "faithful_plus"]
    )
    assert args.lm_target == "faithful_plus"
    assert args.pole_mode == "hidden"
    assert resolve_lm_recipe(lm_target="faithful_plus", symmetric=True) == (
        "faithful_plus"
    )
    hold, anchor = resolve_lm_loss_weights(
        "faithful_plus",
        hold_weight=None,
        anchor_weight=None,
        leak_declared=True,
    )
    assert hold == 0.0
    assert anchor == 0.0
    bare = parse_args(["--prompts_file", "prompts.yaml"])
    assert bare.lm_target == "v9"
    assert bare.pole_mode == "hidden"
    src = Path("conceptmod/textsliders/train_lm_slider_music3.py").read_text()
    assert '"lm_target": recipe' in src
    assert '"plus_only": recipe in PLUS_ONLY_RECIPES' in src
    pos, neg, neu = _ungated_pair()
    plus, minus, _, _ = lm_train_targets(pos, neg, neu, recipe="faithful_plus")
    assert torch.allclose(plus, pos)
    pair_odd = lm_hidden_targets(pos, neg, neu, target_mode="symmetric")
    assert not torch.allclose(plus, pair_odd[0])
    leftover = torch.tensor([0.0, 1.0])
    gated, _, _, _ = lm_train_targets(
        pos,
        neg,
        neu,
        recipe="faithful_plus",
        slider_dir=E_SLIDER,
        leak_dir=leftover,
    )
    want = lm_faithful_sub_e(pos, neg, neu, leftover, slider_dir=E_SLIDER)
    assert torch.allclose(gated, want[0])
    # Minus MSE is off: moving pred_minus must not change the plus-only loss.
    pred_plus = plus + 0.1
    pred_minus = minus
    base = lm_train_loss(
        pred_plus, pred_minus, plus, minus, plus_only=True, pole_mode="hidden"
    )
    moved = lm_train_loss(
        pred_plus,
        pred_minus + 3.0,
        plus,
        minus,
        plus_only=True,
        pole_mode="hidden",
    )
    both = lm_train_loss(
        pred_plus, pred_minus + 3.0, plus, minus, plus_only=False, pole_mode="hidden"
    )
    assert float(base) == pytest.approx(float(moved), abs=1e-7)
    assert float(both) > float(base)


def test_faithful_plus_neu_is_wired_and_not_the_default():
    args = parse_args(
        ["--prompts_file", "prompts.yaml", "--lm_target", "faithful_plus_neu"]
    )
    assert args.lm_target == "faithful_plus_neu"
    assert args.pole_mode == "hidden"
    assert resolve_lm_recipe(lm_target="faithful_plus_neu", symmetric=True) == (
        "faithful_plus_neu"
    )
    hold, anchor = resolve_lm_loss_weights(
        "faithful_plus_neu",
        hold_weight=None,
        anchor_weight=None,
        leak_declared=True,
    )
    assert hold == 0.0
    assert anchor == 0.0
    bare = parse_args(["--prompts_file", "prompts.yaml"])
    assert bare.lm_target == "v9"
    assert bare.pole_mode == "hidden"
    src = Path("conceptmod/textsliders/train_lm_slider_music3.py").read_text()
    assert '"lm_target": recipe' in src
    assert '"plus_neu": recipe in PLUS_NEU_RECIPES' in src
    pos, neg, neu = _ungated_pair()
    plus, minus, _, _ = lm_train_targets(pos, neg, neu, recipe="faithful_plus_neu")
    assert torch.allclose(plus, pos)
    pair_odd = lm_hidden_targets(pos, neg, neu, target_mode="symmetric")
    assert not torch.allclose(plus, pair_odd[0])
    leftover = torch.tensor([0.0, 1.0])
    still_raw, _, _, _ = lm_train_targets(
        pos,
        neg,
        neu,
        recipe="faithful_plus_neu",
        slider_dir=E_SLIDER,
        leak_dir=leftover,
    )
    assert torch.allclose(still_raw, pos)
    gated, _, _, _ = lm_train_targets(
        pos,
        neg,
        neu,
        recipe="faithful_plus",
        slider_dir=E_SLIDER,
        leak_dir=leftover,
    )
    want = lm_faithful_sub_e(pos, neg, neu, leftover, slider_dir=E_SLIDER)
    assert torch.allclose(gated, want[0])
    assert not torch.allclose(still_raw, gated)
    pred_plus = plus + 0.1
    pred_minus = minus
    pred_zero = neu + 0.2
    base = lm_train_loss(
        pred_plus,
        pred_minus,
        plus,
        minus,
        plus_neu=True,
        pred_zero=pred_zero,
        tgt_zero=neu,
        pole_mode="hidden",
    )
    moved_minus = lm_train_loss(
        pred_plus,
        pred_minus + 3.0,
        plus,
        minus,
        plus_neu=True,
        pred_zero=pred_zero,
        tgt_zero=neu,
        pole_mode="hidden",
    )
    moved_zero = lm_train_loss(
        pred_plus,
        pred_minus,
        plus,
        minus,
        plus_neu=True,
        pred_zero=pred_zero + 1.0,
        tgt_zero=neu,
        pole_mode="hidden",
    )
    plus_only = lm_train_loss(
        pred_plus, pred_minus, plus, minus, plus_only=True, pole_mode="hidden"
    )
    both = lm_train_loss(
        pred_plus, pred_minus + 3.0, plus, minus, plus_only=False, pole_mode="hidden"
    )
    assert float(base) == pytest.approx(float(moved_minus), abs=1e-7)
    assert float(moved_zero) > float(base)
    assert float(base) > float(plus_only)
    assert float(both) > 0.0


def test_faithful_plus_neu_prefix_is_wired_and_not_the_default():
    from conceptmod.textsliders.train_lm_slider_music3 import (
        PLUS_NEU_PREFIX_RECIPES,
        PLUS_NEU_RECIPES,
    )

    args = parse_args(
        ["--prompts_file", "prompts.yaml", "--lm_target", "faithful_plus_neu_prefix"]
    )
    assert args.lm_target == "faithful_plus_neu_prefix"
    assert args.pole_mode == "hidden"
    assert resolve_lm_recipe(lm_target="faithful_plus_neu_prefix", symmetric=True) == (
        "faithful_plus_neu_prefix"
    )
    assert "faithful_plus_neu_prefix" in PLUS_NEU_RECIPES
    assert PLUS_NEU_PREFIX_RECIPES == frozenset({"faithful_plus_neu_prefix"})
    hold, anchor = resolve_lm_loss_weights(
        "faithful_plus_neu_prefix",
        hold_weight=None,
        anchor_weight=None,
        leak_declared=True,
    )
    assert hold == 0.0
    assert anchor == 0.0
    pos, neg, neu = _ungated_pair()
    plus, minus, _, _ = lm_train_targets(pos, neg, neu, recipe="faithful_plus_neu_prefix")
    assert torch.allclose(plus, pos)
    leftover = torch.tensor([0.0, 1.0])
    still_raw, _, _, _ = lm_train_targets(
        pos,
        neg,
        neu,
        recipe="faithful_plus_neu_prefix",
        slider_dir=E_SLIDER,
        leak_dir=leftover,
    )
    assert torch.allclose(still_raw, pos)
    pred_plus = plus + 0.1
    pred_minus = minus
    pred_zero = neu + 0.2
    pref = torch.zeros(3, pos.numel())
    tgt_pref = torch.zeros(3, pos.numel())
    base = lm_train_loss(
        pred_plus,
        pred_minus,
        plus,
        minus,
        plus_neu=True,
        plus_neu_prefix=True,
        pred_zero=pred_zero,
        tgt_zero=neu,
        pred_plus_prefix=pref,
        tgt_neu_prefix=tgt_pref,
        pole_mode="hidden",
    )
    moved_prefix = lm_train_loss(
        pred_plus,
        pred_minus,
        plus,
        minus,
        plus_neu=True,
        plus_neu_prefix=True,
        pred_zero=pred_zero,
        tgt_zero=neu,
        pred_plus_prefix=pref + 1.0,
        tgt_neu_prefix=tgt_pref,
        pole_mode="hidden",
    )
    uni = lm_train_loss(
        pred_plus,
        pred_minus,
        plus,
        minus,
        plus_neu=True,
        pred_zero=pred_zero,
        tgt_zero=neu,
        pole_mode="hidden",
    )
    assert float(moved_prefix) > float(base)
    assert float(base) == pytest.approx(float(uni), abs=1e-7)
    src = Path("conceptmod/textsliders/train_lm_slider_music3.py").read_text()
    assert '"plus_neu_prefix": recipe in PLUS_NEU_PREFIX_RECIPES' in src
    bare = parse_args(["--prompts_file", "prompts.yaml"])
    assert bare.lm_target == "v9"
    assert bare.pole_mode == "hidden"


def test_faithful_plus_neu_lyric_is_wired_and_not_the_default():
    from conceptmod.textsliders.train_lm_slider_music3 import (
        PLUS_NEU_HOLD_RECIPES,
        PLUS_NEU_LYRIC_RECIPES,
        PLUS_NEU_PREFIX_RECIPES,
        PLUS_NEU_RECIPES,
    )

    args = parse_args(
        ["--prompts_file", "prompts.yaml", "--lm_target", "faithful_plus_neu_lyric"]
    )
    assert args.lm_target == "faithful_plus_neu_lyric"
    assert args.pole_mode == "hidden"
    assert resolve_lm_recipe(lm_target="faithful_plus_neu_lyric", symmetric=True) == (
        "faithful_plus_neu_lyric"
    )
    assert "faithful_plus_neu_lyric" in PLUS_NEU_RECIPES
    assert PLUS_NEU_LYRIC_RECIPES == frozenset({"faithful_plus_neu_lyric"})
    assert PLUS_NEU_PREFIX_RECIPES == frozenset({"faithful_plus_neu_prefix"})
    assert PLUS_NEU_HOLD_RECIPES == PLUS_NEU_PREFIX_RECIPES | PLUS_NEU_LYRIC_RECIPES
    hold, anchor = resolve_lm_loss_weights(
        "faithful_plus_neu_lyric",
        hold_weight=None,
        anchor_weight=None,
        leak_declared=True,
    )
    assert hold == 0.0
    assert anchor == 0.0
    pos, neg, neu = _ungated_pair()
    plus, minus, _, _ = lm_train_targets(pos, neg, neu, recipe="faithful_plus_neu_lyric")
    assert torch.allclose(plus, pos)
    leftover = torch.tensor([0.0, 1.0])
    still_raw, _, _, _ = lm_train_targets(
        pos,
        neg,
        neu,
        recipe="faithful_plus_neu_lyric",
        slider_dir=E_SLIDER,
        leak_dir=leftover,
    )
    assert torch.allclose(still_raw, pos)
    src = Path("conceptmod/textsliders/train_lm_slider_music3.py").read_text()
    assert '"plus_neu_lyric": recipe in PLUS_NEU_LYRIC_RECIPES' in src
    bare = parse_args(["--prompts_file", "prompts.yaml"])
    assert bare.lm_target == "v9"
    assert bare.pole_mode == "hidden"


def test_help_lists_faithful_plus_neu_and_keeps_v9_hidden_default():
    import io
    from contextlib import redirect_stdout

    buf = io.StringIO()
    with pytest.raises(SystemExit):
        with redirect_stdout(buf):
            parse_args(["--help"])
    help_text = buf.getvalue()
    assert "--lm_target" in help_text
    assert "faithful_plus_neu" in help_text
    assert "faithful_plus_neu_lyric" in help_text
    assert "faithful_plus" in help_text
    assert "v9" in help_text
    bare = parse_args(["--prompts_file", "prompts.yaml"])
    assert bare.lm_target == "v9"
    assert bare.pole_mode == "hidden"


def test_help_lists_faithful_plus_and_keeps_v9_hidden_default():
    import io
    from contextlib import redirect_stdout

    buf = io.StringIO()
    with pytest.raises(SystemExit):
        with redirect_stdout(buf):
            parse_args(["--help"])
    help_text = buf.getvalue()
    assert "--lm_target" in help_text
    assert "faithful_plus" in help_text
    assert "v9" in help_text
    bare = parse_args(["--prompts_file", "prompts.yaml"])
    assert bare.lm_target == "v9"
    assert bare.pole_mode == "hidden"


def _divergent_pair():
    """Two tracks: ê restates the pole difference."""
    u, p, q, s = (torch.eye(4)[i] for i in range(4))
    neu = 1.1 * s
    a = u + 0.8 * (p - q)
    c = 0.7 * s
    e = (p - q) + 0.3 * u
    return neu + a + c, neu - a + c, neu, u, e


def _leftover_pair():
    """One song, plus an attribute inside ``a`` the captions never pin."""
    u, g, s, d = (torch.eye(4)[i] for i in range(4))
    neu = 1.2 * s
    a = u + 0.45 * g + 0.35 * d
    c = 0.9 * s
    return neu + a + c, neu - a + c, neu, u, g


def test_the_blend_guard_refuses_e_exactly_where_e_restates_the_axis():
    pos, neg, neu, u, e = _divergent_pair()
    sub = lm_faithful_sub_e(pos, neg, neu, e, slider_dir=u)
    guard = lm_blend_guard(*sub, pos, neg)
    assert guard["admissible"] is False
    assert guard["to_pole"] > guard["to_mid"]
    guarded = lm_faithful_guard_e(pos, neg, neu, e, slider_dir=u)
    assert torch.allclose(guarded[0], pos)
    assert torch.allclose(guarded[1], neg)

    pos2, neg2, neu2, u2, e2 = _leftover_pair()
    sub2 = lm_faithful_sub_e(pos2, neg2, neu2, e2, slider_dir=u2)
    guard2 = lm_blend_guard(*sub2, pos2, neg2)
    assert guard2["admissible"] is True
    assert guard2["to_pole"] < guard2["to_mid"]
    guarded2 = lm_faithful_guard_e(pos2, neg2, neu2, e2, slider_dir=u2)
    assert torch.allclose(guarded2[0], sub2[0])
    assert torch.allclose(guarded2[1], sub2[1])
    held = lm_hold_dir(e2, slider_dir=u2, mode="slider")
    odd = (guarded2[0] - guarded2[1]) / 2
    assert abs(float(odd @ lm_unit(held))) < 1e-6


def test_the_guard_never_rejects_the_caption_itself():
    """``to_pole = 0`` and ``to_mid = ‖a‖``, so a raw pole is always admissible."""
    for pos, neg, _neu, _u, _e in (_divergent_pair(), _leftover_pair()):
        guard = lm_blend_guard(pos, neg, pos, neg)
        assert guard["admissible"] is True
        assert guard["to_pole"] == pytest.approx(0.0)
        assert guard["to_mid"] == pytest.approx(float(((pos - neg) / 2).norm()))


def test_faithful_guard_e_is_wired_and_needs_no_leak_axis():
    pos, neg, neu, u, e = _divergent_pair()
    plus, minus, anc_p, anc_m = lm_train_targets(
        pos, neg, neu, recipe="faithful_guard_e", slider_dir=u, leak_dir=e
    )
    assert anc_p is None and anc_m is None
    assert torch.allclose(plus, pos) and torch.allclose(minus, neg)
    bare_plus, bare_minus, _, _ = lm_train_targets(
        pos, neg, neu, recipe="faithful_guard_e", slider_dir=u, leak_dir=None
    )
    assert torch.allclose(bare_plus, pos) and torch.allclose(bare_minus, neg)
    with pytest.raises(ValueError, match="declared slider_dir"):
        lm_train_targets(pos, neg, neu, recipe="faithful_guard_e", leak_dir=e)
    args = parse_args(
        ["--prompts_file", "prompts.yaml", "--lm_target", "faithful_guard_e"]
    )
    assert resolve_lm_recipe(lm_target=args.lm_target, symmetric=args.symmetric) == (
        "faithful_guard_e"
    )
    hold, anchor = resolve_lm_loss_weights(
        "faithful_guard_e", hold_weight=None, anchor_weight=None, leak_declared=True
    )
    assert hold == 0.0 and anchor == 0.0
    bare = parse_args(["--prompts_file", "prompts.yaml"])
    assert bare.lm_target == "v9"
    assert bare.pole_mode == "hidden"


def test_the_blind_band_is_what_a_next_token_kl_cannot_see():
    readout = torch.tensor(
        [[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=torch.float32
    )
    blind = lm_blind_projector(readout)
    assert blind is not None
    assert float(blind.trace()) == pytest.approx(1.0)
    unread = torch.tensor([0.0, 0.0, 1.0])
    assert torch.allclose(lm_blind_residual(unread, blind), unread, atol=1e-6)
    for seen in (torch.tensor([1.0, 0.0, 0.0]), torch.tensor([0.0, 1.0, 0.0])):
        assert float(lm_blind_residual(seen, blind).norm()) < 1e-6
    base = torch.tensor([0.3, -0.2, 0.0])
    moved = base + 2.0 * unread
    assert torch.allclose(
        lm_next_token_logits(base, readout), lm_next_token_logits(moved, readout)
    )
    uniform = lm_blind_projector(torch.eye(3))
    assert uniform is not None
    assert float(uniform.trace()) == pytest.approx(1.0)
    ones = torch.ones(3) / 3.0**0.5
    assert torch.allclose(lm_blind_residual(ones, uniform), ones, atol=1e-6)
    filled = torch.tensor(
        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [0.0, 0.0, 0.0]],
        dtype=torch.float32,
    )
    assert lm_blind_projector(filled, cut=0.0) is None
    assert lm_blind_projector(filled, cut=1.0) is not None


def test_dual_band_is_the_kl_plus_an_mse_the_kl_cannot_supply():
    pos, neg, neu = _ungated_pair()
    tgt_plus, tgt_minus, _, _ = lm_train_targets(pos, neg, neu, recipe="faithful")
    pred_plus = neu + torch.tensor([0.8, 0.4])
    pred_minus = neu + torch.tensor([-0.7, -0.3])
    readout = torch.tensor([[1.0, 0.0], [-1.0, 0.0], [0.5, 0.0]], dtype=torch.float32)
    blind = lm_blind_projector(readout)
    assert blind is not None
    kl_only = lm_train_loss(
        pred_plus,
        pred_minus,
        tgt_plus,
        tgt_minus,
        neu=neu,
        pole_mode="semantic_kl",
        readout=readout,
    )
    dual = lm_train_loss(
        pred_plus,
        pred_minus,
        tgt_plus,
        tgt_minus,
        neu=neu,
        pole_mode="dual_band",
        readout=readout,
        blind_projector=blind,
    )
    expected = lm_dual_band_pole_loss(
        pred_plus,
        pred_minus,
        tgt_plus,
        tgt_minus,
        pred_plus_logits=lm_next_token_logits(pred_plus, readout),
        pred_minus_logits=lm_next_token_logits(pred_minus, readout),
        tgt_plus_logits=lm_next_token_logits(tgt_plus, readout),
        tgt_minus_logits=lm_next_token_logits(tgt_minus, readout),
        blind_projector=blind,
    )
    assert float(dual) == pytest.approx(float(expected), abs=1e-6)
    assert float(dual) > float(kl_only)
    assert float(
        lm_train_loss(
            pred_plus,
            pred_minus,
            tgt_plus,
            tgt_minus,
            neu=neu,
            pole_mode="dual_band",
            readout=readout,
            blind_projector=None,
        )
    ) == pytest.approx(float(kl_only), abs=1e-6)
    with pytest.raises(ValueError, match="semantic readout"):
        lm_train_loss(
            pred_plus, pred_minus, tgt_plus, tgt_minus, pole_mode="dual_band"
        )


def test_only_dual_band_has_a_gradient_on_the_blind_band():
    pos, neg, neu = _ungated_pair()
    tgt_plus, tgt_minus, _, _ = lm_train_targets(pos, neg, neu, recipe="faithful")
    readout = torch.tensor([[1.0, 0.0], [-1.0, 0.0], [0.5, 0.0]], dtype=torch.float32)
    blind = lm_blind_projector(readout)
    grads = {}
    for mode in ("semantic_kl", "dual_band"):
        delta = torch.zeros(2, requires_grad=True)
        loss = lm_train_loss(
            neu + delta,
            neu - delta,
            tgt_plus,
            tgt_minus,
            neu=neu,
            pole_mode=mode,
            readout=readout,
            blind_projector=blind,
        )
        loss.backward()
        grads[mode] = delta.grad.detach().clone()
    assert float(grads["semantic_kl"][1].abs()) < 1e-8
    assert float(grads["dual_band"][1].abs()) > 1e-3
    assert float(grads["dual_band"][0]) == pytest.approx(
        float(grads["semantic_kl"][0]), abs=1e-6
    )


def test_dual_band_flags_are_off_by_default():
    args = parse_args(["--prompts_file", "prompts.yaml"])
    assert args.pole_mode == "hidden"
    assert args.lm_target == "v9"
    assert args.blind_weight == pytest.approx(DUAL_BAND_WEIGHT)
    assert args.blind_cut == pytest.approx(0.0)
    opt_in = parse_args(
        ["--prompts_file", "prompts.yaml", "--pole_mode", "dual_band"]
    )
    assert resolve_pole_mode(opt_in.pole_mode) == "dual_band"
    with pytest.raises(ValueError, match="pole_mode must be one of"):
        resolve_pole_mode("blind")


def test_faithful_sub_e_needs_leak_and_slider():
    pos, neg, neu = _ungated_pair()
    leftover = torch.tensor([0.0, 1.0])
    with pytest.raises(ValueError, match="declared leak_dir"):
        lm_train_targets(pos, neg, neu, recipe="faithful_sub_e", slider_dir=E_SLIDER)
    with pytest.raises(ValueError, match="declared slider_dir"):
        lm_train_targets(pos, neg, neu, recipe="faithful_sub_e", leak_dir=leftover)


def test_semantic_kl_loss_uses_existing_helper():
    """Live semantic_kl is lm_semantic_pole_loss on lm_next_token_logits."""
    pos, neg, neu = _ungated_pair()
    tgt_plus, tgt_minus, _, _ = lm_train_targets(pos, neg, neu, recipe="faithful")
    pred_plus = neu + torch.tensor([0.8, 0.4])
    pred_minus = neu + torch.tensor([-0.7, -0.3])
    readout = torch.tensor(
        [[1.0, 0.2], [0.1, -1.0], [0.4, 0.5], [-0.3, 0.8]], dtype=torch.float32
    )
    got = lm_train_loss(
        pred_plus,
        pred_minus,
        tgt_plus,
        tgt_minus,
        neu=neu,
        hold_weight=0.0,
        pole_mode="semantic_kl",
        readout=readout,
    )
    expected = lm_semantic_pole_loss(
        lm_next_token_logits(pred_plus, readout),
        lm_next_token_logits(pred_minus, readout),
        lm_next_token_logits(tgt_plus, readout),
        lm_next_token_logits(tgt_minus, readout),
    )
    assert float(got) == pytest.approx(float(expected), abs=1e-6)
    hidden = lm_train_loss(
        pred_plus,
        pred_minus,
        tgt_plus,
        tgt_minus,
        neu=neu,
        hold_weight=0.0,
        pole_mode="hidden",
    )
    mse = lm_slider_loss(pred_plus, pred_minus, tgt_plus, tgt_minus)
    assert float(hidden) == pytest.approx(float(mse), abs=1e-6)
    assert float(got) != pytest.approx(float(hidden), abs=1e-4)
    with pytest.raises(ValueError, match="semantic readout"):
        lm_train_loss(
            pred_plus,
            pred_minus,
            tgt_plus,
            tgt_minus,
            pole_mode="semantic_kl",
        )


def test_semantic_kl_null_loss_pins_ker_readout():
    pos, neg, neu = _ungated_pair()
    tgt_plus, tgt_minus, _, _ = lm_train_targets(pos, neg, neu, recipe="faithful")
    pred_plus = neu + torch.tensor([0.8, 0.4])
    pred_minus = neu + torch.tensor([-0.7, -0.3])
    readout = torch.tensor(
        [[1.0, 0.2], [0.1, -1.0], [0.4, 0.5], [-0.3, 0.8]],
        dtype=torch.float32,
    )
    basis = lm_readout_null_basis(readout)
    assert basis is None
    got = lm_train_loss(
        pred_plus,
        pred_minus,
        tgt_plus,
        tgt_minus,
        neu=neu,
        hold_weight=0.0,
        pole_mode="semantic_kl_null",
        readout=readout,
        null_basis=basis,
    )
    expected = lm_semantic_null_pole_loss(
        lm_next_token_logits(pred_plus, readout),
        lm_next_token_logits(pred_minus, readout),
        lm_next_token_logits(tgt_plus, readout),
        lm_next_token_logits(tgt_minus, readout),
        pred_plus,
        pred_minus,
        tgt_plus,
        tgt_minus,
        readout,
        null_basis=basis,
    )
    assert float(got) == pytest.approx(float(expected), abs=1e-6)
    alias = lm_train_loss(
        pred_plus,
        pred_minus,
        tgt_plus,
        tgt_minus,
        neu=neu,
        hold_weight=0.0,
        pole_mode="semantic_kl_pin",
        readout=readout,
        null_basis=basis,
    )
    plus = lm_train_loss(
        pred_plus,
        pred_minus,
        tgt_plus,
        tgt_minus,
        neu=neu,
        hold_weight=0.0,
        pole_mode="semantic_kl_plus_hidden",
        readout=readout,
        null_basis=basis,
    )
    assert float(alias) == pytest.approx(float(got), abs=1e-6)
    assert float(plus) == pytest.approx(float(got), abs=1e-6)
    assert resolve_pole_mode("semantic_kl_null") == "semantic_kl_null"
    args = parse_args(["--prompts_file", "prompts.yaml", "--pole_mode", "semantic_kl_null"])
    assert args.pole_mode == "semantic_kl_null"
    bare = parse_args(["--prompts_file", "prompts.yaml"])
    assert bare.pole_mode == "hidden"


def test_hidden_kl_pins_hidden_and_adds_semantic_consistency():
    pos, neg, neu = _ungated_pair()
    tgt_plus, tgt_minus, _, _ = lm_train_targets(pos, neg, neu, recipe="faithful")
    pred_plus = neu + torch.tensor([0.8, 0.4])
    pred_minus = neu + torch.tensor([-0.7, -0.3])
    readout = torch.tensor(
        [[1.0, 0.2], [0.1, -1.0], [0.4, 0.5], [-0.3, 0.8]], dtype=torch.float32
    )
    got = lm_train_loss(
        pred_plus,
        pred_minus,
        tgt_plus,
        tgt_minus,
        neu=neu,
        pole_mode="hidden_kl",
        readout=readout,
    )
    hidden = lm_slider_loss(pred_plus, pred_minus, tgt_plus, tgt_minus)
    semantic = lm_semantic_pole_loss(
        lm_next_token_logits(pred_plus, readout),
        lm_next_token_logits(pred_minus, readout),
        lm_next_token_logits(tgt_plus, readout),
        lm_next_token_logits(tgt_minus, readout),
    )
    assert got == pytest.approx(hidden + HIDDEN_KL_WEIGHT * semantic)
    args = parse_args(
        ["--prompts_file", "prompts.yaml", "--lm_target", "faithful", "--pole_mode", "hidden_kl"]
    )
    assert args.pole_mode == "hidden_kl"
    with pytest.raises(ValueError, match="semantic readout"):
        lm_train_loss(
            pred_plus,
            pred_minus,
            tgt_plus,
            tgt_minus,
            pole_mode="hidden_kl",
        )


def test_hidden_pole_mode_leaves_v9_and_pair_odd_sub_e_unchanged():
    pos, neg, neu = _ungated_pair()
    leftover = torch.tensor([0.4, 0.9])
    v9_plus, v9_minus, _, _ = lm_train_targets(pos, neg, neu, recipe="v9")
    leaked = lm_hidden_targets(pos, neg, neu, target_mode="symmetric")
    assert torch.allclose(v9_plus, leaked[0])
    assert torch.allclose(v9_minus, leaked[1])
    pred_plus = neu + torch.tensor([0.8, 0.4])
    pred_minus = neu + torch.tensor([-0.7, -0.3])
    hidden = lm_train_loss(
        pred_plus,
        pred_minus,
        v9_plus,
        v9_minus,
        neu=neu,
        leak_dir=leftover,
        slider_dir=E_SLIDER,
        hold_weight=LEAK_HOLD_WEIGHT,
        pole_mode="hidden",
    )
    hold_axis = lm_hold_dir(leftover, slider_dir=E_SLIDER, mode="slider")
    hold = lm_axis_hold(pred_plus, pred_minus, neu, hold_axis)
    expected = lm_slider_loss(
        pred_plus, pred_minus, v9_plus, v9_minus, hold=hold, hold_weight=LEAK_HOLD_WEIGHT
    )
    assert float(hidden) == pytest.approx(float(expected), abs=1e-6)
    sub_plus, sub_minus, _, _ = lm_train_targets(
        pos, neg, neu, recipe="pair_odd_sub_e", slider_dir=E_SLIDER, leak_dir=leftover
    )
    expected_sub = lm_pair_odd_sub_e(pos, neg, neu, leftover, slider_dir=E_SLIDER)
    assert torch.allclose(sub_plus, expected_sub[0])
    assert torch.allclose(sub_minus, expected_sub[1])


def test_plus_neu_skips_minus_endreg_and_minus_early_stop():
    from conceptmod.textsliders.train_lm_slider_music3 import (
        _early_stop_hit,
        _endreg_uses_minus,
        _minus_pole_used,
    )

    assert _endreg_uses_minus("v9") is True
    assert _endreg_uses_minus("faithful_plus") is True
    assert _endreg_uses_minus("faithful_plus_neu") is False
    assert _endreg_uses_minus("faithful_plus_neu_prefix") is False
    assert _endreg_uses_minus("faithful_plus_neu_lyric") is False
    assert _minus_pole_used("faithful_plus_neu") is False
    assert _minus_pole_used("faithful_plus_neu_lyric") is False
    src = Path("conceptmod/textsliders/train_lm_slider_music3.py").read_text()
    assert "if _minus_pole_used(recipe):" in src
    assert "plus_neu=recipe in PLUS_NEU_RECIPES" in src
    # Dummy minus metrics must not block plus+neu early-stop.
    window = [
        {
            "cos_pos": 0.99,
            "cos_neg": 0.0,
            "collapse": 0.0,
            "pperc": 0.05,
            "nperc": 0.0,
        }
    ] * 50
    assert _early_stop_hit(window, 50, 0.97, -0.95, 0.20, plus_neu=True) is True
    assert _early_stop_hit(window, 50, 0.97, -0.95, 0.20, plus_neu=False) is False
    # Bipolar still requires c- and collapse.
    bipolar = [
        {
            "cos_pos": 0.99,
            "cos_neg": 0.99,
            "collapse": -0.98,
            "pperc": 0.05,
            "nperc": 0.05,
        }
    ] * 50
    assert _early_stop_hit(bipolar, 50, 0.97, -0.95, 0.20, plus_neu=False) is True


def test_last_token_is_audio_start_and_respects_padding():
    from conceptmod.textsliders.train_lm_slider_music3 import (
        _AUDIO_START,
        _assert_last_token_is_audio_start,
        _gather_last_hidden,
        _last_real_index,
    )

    class _Tok:
        def convert_tokens_to_ids(self, token: str):
            return 99 if token == _AUDIO_START else 0

    ids = torch.tensor([[1, 2, 99]])
    mask = torch.tensor([[1, 1, 1]])
    _assert_last_token_is_audio_start(ids, mask, _Tok(), where="ok")
    bad = torch.tensor([[1, 2, 3]])
    with pytest.raises(RuntimeError, match="audio_start"):
        _assert_last_token_is_audio_start(bad, mask, _Tok(), where="bad")
    padded = torch.tensor([[1, 2, 99, 0]])
    pad_mask = torch.tensor([[1, 1, 1, 0]])
    _assert_last_token_is_audio_start(padded, pad_mask, _Tok(), where="pad")
    hidden = torch.arange(32, dtype=torch.float32).view(1, 4, 8)
    last = _gather_last_hidden(hidden, pad_mask)
    assert torch.allclose(last, hidden[:, 2])
    assert int(_last_real_index(pad_mask).item()) == 2


def test_infer_default_prompt_prefers_neutral_over_target():
    infer = Path("conceptmod/textsliders/infer_music3.py").read_text()
    listen = Path("conceptmod/textsliders/generate_listen.py").read_text()
    assert 'item.get("neutral") or item.get("target")' in infer
    assert "plus+neu adapter" in infer
    assert "plus+neu adapter" in listen
    assert "double +" in listen
