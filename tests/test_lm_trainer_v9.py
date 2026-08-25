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
    lm_hidden_targets,
    lm_ortho_hold,
    lm_project_odd_axis,
    lm_slider_loss,
)
from conceptmod.textsliders.train_lm_slider_music3 import (
    lm_train_loss,
    lm_train_targets,
    parse_args,
    resolve_leak_axis_captions,
    resolve_lm_loss_weights,
    resolve_lm_recipe,
    resolve_slider_axis_captions,
    resolve_v9_gate,
)


def _ungated_pair(t: float = 0.5):
    field = Field2D()
    return field.embed("energetic", t), field.embed("calm", t), field.embed("song", t)


def test_bare_parse_defaults_to_v9_and_symmetric():
    args = parse_args(["--prompts_file", "prompts.yaml"])
    assert args.lm_target == "v9"
    assert args.symmetric is True
    assert resolve_lm_recipe(lm_target=args.lm_target, symmetric=args.symmetric) == "v9"
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
    assert "mix" in energy["leak_positive"].lower() or "bpm" in energy["leak_positive"].lower()
    assert "mix" in energy["leak_negative"].lower() or "bpm" in energy["leak_negative"].lower()
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
