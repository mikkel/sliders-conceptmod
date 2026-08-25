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
from conceptmod.textsliders.slider_targets import (
    lm_anchor_kappa,
    lm_anchor_targets,
    lm_hidden_targets,
    lm_ortho_hold,
    lm_project_odd_axis,
    lm_slider_loss,
)
from conceptmod.textsliders.train_lm_slider_music3 import (
    lm_train_loss,
    lm_train_targets,
    parse_args,
    resolve_lm_loss_weights,
    resolve_lm_recipe,
    resolve_slider_axis_captions,
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
        "v9", hold_weight=args.hold_weight, anchor_weight=args.anchor_weight
    )
    assert hold == 1.0
    assert anchor == 0.0
    assert args.leakage_floor is None


def test_v9_requires_symmetric_polarity():
    with pytest.raises(ValueError, match="polarity step"):
        resolve_lm_recipe(lm_target="v9", symmetric=False)


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


def test_live_v9_targets_are_lm_project_odd_axis():
    pos, neg, neu = _ungated_pair()
    plus, minus, anc_p, anc_m = lm_train_targets(
        pos, neg, neu, recipe="v9", slider_dir=E_SLIDER
    )
    expected_plus, expected_minus = lm_project_odd_axis(pos, neg, neu, E_SLIDER)
    assert torch.allclose(plus, expected_plus)
    assert torch.allclose(minus, expected_minus)
    assert anc_p is None and anc_m is None
    leaked = lm_hidden_targets(pos, neg, neu, target_mode="symmetric")
    assert not torch.allclose(plus, leaked[0])
    odd = plus - neu
    assert float(odd @ torch.tensor([0.0, 1.0])) == pytest.approx(0.0, abs=1e-6)
    # Silent (pos−neg) would keep unused gender.
    raw_odd = (pos - neg) / 2
    assert abs(float(raw_odd[1])) > 0.20
    identity_plus, _ = lm_project_odd_axis(pos, neg, neu, raw_odd)
    assert torch.allclose(identity_plus, leaked[0])


def test_live_loss_is_projected_odd_plus_ortho_hold():
    pos, neg, neu = _ungated_pair()
    tgt_plus, tgt_minus, _, _ = lm_train_targets(
        pos, neg, neu, recipe="v9", slider_dir=E_SLIDER
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
    v9_plus, v9_minus, _, _ = lm_train_targets(
        pos, neg, neu, recipe="v9", slider_dir=E_SLIDER
    )
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
    assert not torch.allclose(v9_plus, hub_plus)
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
    # Display labels are not the axis.
    assert gender["slider_positive"] != gender["plus_label"]
    assert energy["slider_positive"] != energy["plus_label"]


def test_v9_targets_require_declared_dir():
    pos, neg, neu = _ungated_pair()
    with pytest.raises(ValueError, match="declared slider_dir"):
        lm_train_targets(pos, neg, neu, recipe="v9", slider_dir=None)
    with pytest.raises(ValueError, match="declared slider_dir"):
        lm_train_loss(
            neu, neu, neu, neu, neu=neu, slider_dir=None, hold_weight=1.0
        )
