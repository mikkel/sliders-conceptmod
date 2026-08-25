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
    LEAK_HOLD_WEIGHT,
    lm_anchor_kappa,
    lm_anchor_targets,
    lm_axis_hold,
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
    assert resolve_pole_mode("semantic_kl_null") == "semantic_kl_null"
    args = parse_args(["--prompts_file", "prompts.yaml", "--pole_mode", "semantic_kl_null"])
    assert args.pole_mode == "semantic_kl_null"


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
