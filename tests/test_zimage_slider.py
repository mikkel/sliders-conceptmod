"""Z-Image Turbo UNI image-slider: CPU geometry + dummy trainer.

No Hub, no GPU, no Tongyi-MAI weights. Does not change the Music 3
live trainer default. Does not add Anima / Krea / H3 backends.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from conceptmod.textsliders.slider_targets import (
    ZImageHoldError,
    expand_attributes_zimage,
    zimage_canary_minus,
    zimage_cfg,
    zimage_cfg_delta,
    zimage_concept_token_ids,
    zimage_require_concept_in_prompt,
    zimage_uni_loss,
    zimage_uni_teachers,
    zimage_unused_token_hold,
    zimage_unused_token_positions,
)
from conceptmod.textsliders.train_lora_music3 import parse_args as parse_music3
from conceptmod.textsliders.train_lora_zimage import (
    DEFAULT_MODEL_ID,
    dummy_tokenize,
    load_prompts,
    parse_args,
    train,
)

PROMPTS = Path("conceptmod/textsliders/data/prompts-zimage.yaml")


def test_music3_defaults_unchanged():
    args = parse_music3([])
    assert args.loss == "nmse"
    assert args.target_mode == "axis"
    assert args.targets == "full"
    assert args.rank == 8
    assert args.bidirectional is True


def test_zit_parse_defaults_are_the_live_card():
    args = parse_args([])
    assert args.model_id == "Tongyi-MAI/Z-Image-Turbo"
    assert args.model_id == DEFAULT_MODEL_ID
    assert args.rank == 16
    assert args.alpha == 16.0
    assert args.resolution == 768
    assert args.sample_steps == 8
    assert args.sample_guidance == 0.0
    assert args.unused_weight == 0.0
    assert args.token_hold_weight == 0.0
    assert args.dummy is False


def test_cfg_delta_is_v_c_minus_v_uncond():
    vel_c = torch.tensor([2.0, 1.0])
    vel_u = torch.tensor([0.5, 0.25])
    assert torch.allclose(zimage_cfg_delta(vel_c, vel_u), vel_c - vel_u)


def test_cfg_guidance_zero_is_raw_velocity():
    vel_c = torch.tensor([1.5, -0.5])
    vel_u = torch.tensor([0.0, 1.0])
    assert torch.allclose(zimage_cfg(vel_c, vel_u, 0.0), vel_c)
    guided = zimage_cfg(vel_c, vel_u, 2.0)
    assert torch.allclose(guided, vel_c + 2.0 * (vel_c - vel_u))


def test_uni_teachers_are_plus_and_neu_only():
    vel_pos = torch.tensor([1.0, 0.0])
    vel_neu = torch.tensor([0.0, 0.0])
    vel_uncond = torch.tensor([-1.0, 0.0])
    vel_neg = torch.tensor([0.0, 9.0])
    plus, zero = zimage_uni_teachers(vel_pos, vel_neu, vel_uncond, guidance=0.0)
    assert torch.allclose(plus, vel_pos)
    assert torch.allclose(zero, vel_neu)
    _ = vel_neg
    plus_g, zero_g = zimage_uni_teachers(vel_pos, vel_neu, vel_uncond, guidance=1.0)
    assert torch.allclose(plus_g, zimage_cfg(vel_pos, vel_uncond, 1.0))
    assert torch.allclose(zero_g, zimage_cfg(vel_neu, vel_uncond, 1.0))


def test_uni_loss_is_plus_and_zero_only():
    pred_plus = torch.tensor([1.0, 0.0])
    tgt_plus = torch.tensor([1.0, 0.0])
    pred_zero = torch.tensor([0.0, 0.0])
    tgt_zero = torch.tensor([0.0, 0.0])
    other = torch.tensor([0.0, 9.0])
    base = zimage_uni_loss(pred_plus, tgt_plus, pred_zero, tgt_zero)
    assert float(base) == pytest.approx(0.0, abs=1e-8)
    moved_zero = zimage_uni_loss(pred_plus, tgt_plus, pred_zero + 1.0, tgt_zero)
    moved_plus = zimage_uni_loss(pred_plus + 1.0, tgt_plus, pred_zero, tgt_zero)
    assert float(moved_zero) > 0.0
    assert float(moved_plus) > 0.0
    _ = other


def test_uni_loss_unused_and_token_hold_off_by_default():
    pred_plus = torch.tensor([1.0, 0.0])
    tgt_plus = torch.tensor([1.0, 0.0])
    pred_zero = torch.tensor([0.0, 0.0])
    tgt_zero = torch.tensor([0.0, 0.0])
    pred_unused = torch.tensor([9.0, 9.0])
    base = zimage_uni_loss(pred_plus, tgt_plus, pred_zero, tgt_zero)
    ignored = zimage_uni_loss(
        pred_plus,
        tgt_plus,
        pred_zero,
        tgt_zero,
        pred_unused=pred_unused,
        tgt_unused=tgt_zero,
        unused_token_hold=pred_plus.new_tensor(4.0),
    )
    assert float(ignored) == pytest.approx(float(base), abs=1e-8)
    forced = zimage_uni_loss(
        pred_plus,
        tgt_plus,
        pred_zero,
        tgt_zero,
        pred_unused=pred_unused,
        tgt_unused=tgt_zero,
        unused_weight=1.0,
    )
    assert float(forced) > float(base)


def test_canary_minus_is_unscored():
    pred = torch.tensor([0.0, 1.0])
    neg = torch.tensor([0.0, 1.0])
    can = zimage_canary_minus(pred, neg)
    assert can["scored"] is False
    assert can["minus_overlap_neg"] == pytest.approx(1.0)


def test_attributes_pin_unused_on_pos_and_neu():
    rows = expand_attributes_zimage(
        {
            "target": "a person",
            "positive": "an old person",
            "neutral": "a person",
            "negative": "a young person",
            "attributes": ["male", "female"],
        }
    )
    assert len(rows) == 2
    assert rows[0]["positive"] == "male an old person"
    assert rows[0]["neutral"] == "male a person"
    assert rows[1]["positive"] == "female an old person"
    assert rows[1]["neutral"] == "female a person"
    assert rows[0]["negative"] == "male a young person"


def test_yaml_loads_positive_neutral_and_pins_attributes():
    prompts, meta = load_prompts(PROMPTS)
    assert meta.plus_label == "Old"
    assert meta.concept_words == "old, elderly, aged"
    assert all(p.positive and p.neutral for p in prompts)
    assert any(p.positive.startswith("male ") for p in prompts)
    assert any(p.positive.startswith("female ") for p in prompts)
    assert all("old" in p.positive for p in prompts)
    assert all("old" not in p.neutral for p in prompts)


def test_unused_hold_skips_concept_words():
    plus_ids = dummy_tokenize("male an old person")
    neu_ids = dummy_tokenize("male a person")
    concept = zimage_concept_token_ids("old, elderly, aged", dummy_tokenize)
    assert dummy_tokenize("old")[0] in concept
    unused = zimage_unused_token_positions(plus_ids, concept)
    held_tokens = [plus_ids[i] for i in unused]
    assert dummy_tokenize("old")[0] not in held_tokens
    assert dummy_tokenize("male")[0] in held_tokens
    assert dummy_tokenize("person")[0] in held_tokens

    dim = 4
    plus_emb = torch.zeros(len(plus_ids), dim)
    neu_emb = torch.zeros(len(neu_ids), dim)
    old_at = plus_ids.index(dummy_tokenize("old")[0])
    plus_emb[old_at] = torch.tensor([9.0, 9.0, 9.0, 9.0])
    hold = zimage_unused_token_hold(plus_emb, neu_emb, plus_ids, neu_ids, concept)
    assert float(hold) == pytest.approx(0.0, abs=1e-8)
    plus_emb[plus_ids.index(dummy_tokenize("male")[0])] = 1.0
    moved = zimage_unused_token_hold(plus_emb, neu_emb, plus_ids, neu_ids, concept)
    assert float(moved) > 0.0


def test_hold_fails_closed_without_concept_words():
    plus_ids = dummy_tokenize("male a person")
    concept = zimage_concept_token_ids("old", dummy_tokenize)
    with pytest.raises(ZImageHoldError, match="not found"):
        zimage_require_concept_in_prompt(plus_ids, concept)
    with pytest.raises(ZImageHoldError, match="required"):
        zimage_require_concept_in_prompt(plus_ids, set())


def test_dummy_train_never_imports_hub(tmp_path, monkeypatch):
    import sys

    banned = [name for name in list(sys.modules) if name.startswith("huggingface_hub")]
    for name in banned:
        monkeypatch.setitem(sys.modules, name, None)

    def _boom(*_a, **_k):
        raise AssertionError("Hub / ZImagePipeline must not load in dummy mode")

    monkeypatch.setattr(
        "conceptmod.textsliders.train_lora_zimage._LiveZImage", _boom, raising=False
    )
    args = parse_args(
        [
            "--dummy",
            "--steps",
            "2",
            "--save_dir",
            str(tmp_path),
            "--name",
            "zit-dummy",
        ]
    )
    weights = train(args)
    assert weights.exists()
    sidecar = Path(str(weights).replace(".safetensors", ".json"))
    assert sidecar.exists()
    meta = json.loads(sidecar.read_text())
    assert meta["recipe"] == "uni"
    assert meta["dummy"] is True
    assert meta["rank"] == 16
    assert meta["resolution"] == 768
    assert meta["sample_guidance"] == 0.0
    assert meta["teacher"]["minus"] == "canary only"
    assert meta["teacher"]["student"] == "train and infer both use the neutral caption at +1"
    assert "infer path" in meta["teacher"]["unused_hold"]
    assert "lyric" not in meta["teacher"]["unused_hold"]
    assert meta["canary"]["scored"] is False


def test_trainer_student_plus_is_neu_not_pos():
    """Train +1 on neu (match infer). Concept teacher still uses + caption."""
    src = Path("conceptmod/textsliders/train_lora_zimage.py").read_text()
    assert "pred_plus = _student(1.0, emb_neu)" in src
    assert "pred_zero = _student(0.0, emb_neu)" in src
    assert "pred_plus = _student(1.0, emb_pos)" not in src
    assert "pred_unused = _student(1.0, emb_neu)" not in src
    assert "zimage_uni_teachers" in src
    assert "vel_pos" in src


def test_trainer_source_is_zit_only():
    src = Path("conceptmod/textsliders/train_lora_zimage.py").read_text()
    assert "Tongyi-MAI/Z-Image-Turbo" in src
    assert "lyric-hold" in src
    assert "Anima / Krea / H3" in src
    assert "from conceptmod.backends.anima" not in src
    assert "from conceptmod.backends.krea" not in src
    assert "Hunyuan" not in src
    music3 = Path("conceptmod/textsliders/train_lora_music3.py").read_text()
    assert "Z-Image" not in music3
    assert "train_lora_zimage" not in music3


def test_docs_publish_the_live_train_command():
    doc = Path("docs/zimage-slider.md").read_text()
    assert "train_lora_zimage.py" in doc
    assert "Tongyi-MAI/Z-Image-Turbo" in doc
    assert "--rank 16" in doc
    assert "--resolution 768" in doc
    assert "--sample_steps 8" in doc
    assert "--sample_guidance 0.0" in doc
    assert "lyric-hold" in doc
    assert "neu (infer path)" in doc
    assert "cancelled the infer path" in doc
    assert "Anima" in doc
    assert "v(z, t, c) − v(z, t, '')" in doc or "v(z, t, c) - v(z, t, '')" in doc
