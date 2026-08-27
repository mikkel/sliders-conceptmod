"""Sana 0.6B UNI image-slider: CPU geometry + dummy trainer.

No Hub, no GPU, no Sana weights. Does not change the Music 3 live
trainer default. Does not add ZiT / Krea / Anima / H3 backends.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from conceptmod.textsliders.slider_targets import (
    SANA_CFG,
    SANA_CONTROL_PROMPT,
    SANA_DEFAULT_LR,
    SANA_DEFAULT_STEPS,
    SANA_MODEL_ID,
    SANA_RESOLUTION,
    SANA_SAMPLE_STEPS,
    SANA_TRAIN_METHOD,
    SanaHoldError,
    expand_attributes_sana,
    sana_canary_minus,
    sana_cfg,
    sana_cfg_delta,
    sana_concept_token_ids,
    sana_live_train_card,
    sana_live_train_command,
    sana_uni_loss,
    sana_uni_teachers,
    sana_unused_token_hold,
    zimage_cfg,
    zimage_require_concept_in_prompt,
    zimage_unused_token_positions,
)
from conceptmod.textsliders.train_lora_music3 import parse_args as parse_music3
from conceptmod.textsliders.train_lora_sana import (
    _LiveSana,
    assert_sana_only,
    dummy_tokenize,
    load_prompts,
    parse_args,
    resolve_sana_concept_ids,
    train,
)
from conceptmod.textsliders.train_lm_slider_music3 import parse_args as parse_lm

PROMPTS = Path("conceptmod/textsliders/data/prompts-sana.yaml")
ROOT = Path(__file__).resolve().parents[1]


def test_music3_tf_defaults_unchanged():
    args = parse_music3([])
    assert args.loss == "nmse"
    assert args.target_mode == "axis"
    assert args.targets == "full"
    assert args.rank == 8
    assert args.bidirectional is True


def test_music3_lm_default_is_still_v9_hidden():
    args = parse_lm(["--prompts_file", "prompts.yaml"])
    assert args.lm_target == "v9"
    assert args.pole_mode == "hidden"


def test_sana_parse_defaults_are_the_live_card():
    args = parse_args([])
    assert args.model_id == SANA_MODEL_ID
    assert args.model_id == "Efficient-Large-Model/Sana_600M_512px_diffusers"
    assert args.train_method == SANA_TRAIN_METHOD == "xattn"
    assert args.lora is None
    assert args.resolution == SANA_RESOLUTION == 512
    assert args.sample_steps == SANA_SAMPLE_STEPS == 20
    assert args.sample_guidance == SANA_CFG == 4.5
    assert args.lr == SANA_DEFAULT_LR == 2e-5
    assert args.steps == SANA_DEFAULT_STEPS == 500
    assert args.control_prompt == SANA_CONTROL_PROMPT
    assert args.name == "happy-sana"
    assert args.unused_weight == 0.0
    assert args.token_hold_weight == 0.0
    assert args.dummy is False


def test_live_card_and_command_document_the_gpu_look():
    card = sana_live_train_card()
    assert card["model_id"] == SANA_MODEL_ID
    assert card["resolution"] == 512
    assert card["sample_steps"] == 20
    assert card["sample_guidance"] == 4.5
    assert card["train_method"] == "xattn"
    assert card["control_prompt"] == "a bowl of fruit on a table"
    assert card["recipe"] == "uni: student +1/0 on neu, teacher CFG(+)"
    assert card["music3_default_untouched"] == {
        "lm_target": "v9",
        "pole_mode": "hidden",
    }
    cmd = sana_live_train_command()
    assert "train_lora_sana.py" in cmd
    assert "--name happy-sana" in cmd
    assert "Efficient-Large-Model/Sana_600M_512px_diffusers" in cmd
    assert "--train_method xattn" in cmd
    assert "--resolution 512" in cmd
    assert "--sample_steps 20" in cmd
    assert "--sample_guidance 4.5" in cmd
    assert "a bowl of fruit on a table" in cmd


def test_cfg_delta_is_v_c_minus_v_uncond():
    vel_c = torch.tensor([2.0, 1.0])
    vel_u = torch.tensor([0.5, 0.25])
    assert torch.allclose(sana_cfg_delta(vel_c, vel_u), vel_c - vel_u)


def test_sana_cfg_is_uncond_plus_guidance_times_delta():
    vel_c = torch.tensor([2.0, 1.0])
    vel_u = torch.tensor([0.5, 0.25])
    composed = sana_cfg(vel_c, vel_u, 4.5)
    assert torch.allclose(composed, vel_u + 4.5 * (vel_c - vel_u))
    assert torch.allclose(sana_cfg(vel_c, vel_u, 1.0), vel_c)
    # Not the Z-Image / Krea compose v_c + g*(v_c - v_u).
    assert not torch.allclose(composed, zimage_cfg(vel_c, vel_u, 4.5))


def test_uni_teachers_plus_is_cfg_live_zero_is_raw_neu():
    vel_pos = torch.tensor([1.0, 0.0])
    vel_neu = torch.tensor([0.25, 0.0])
    vel_uncond = torch.tensor([-1.0, 0.0])
    vel_neg = torch.tensor([0.0, 9.0])
    plus, zero = sana_uni_teachers(vel_pos, vel_neu, vel_uncond, guidance=4.5)
    assert torch.allclose(plus, sana_cfg(vel_pos, vel_uncond, 4.5))
    assert torch.allclose(zero, vel_neu)
    _ = vel_neg
    plus_id, zero_id = sana_uni_teachers(vel_pos, vel_neu, vel_uncond, guidance=1.0)
    assert torch.allclose(plus_id, vel_pos)
    assert torch.allclose(zero_id, vel_neu)


def test_uni_loss_is_plus_and_zero_only():
    pred_plus = torch.tensor([1.0, 0.0])
    tgt_plus = torch.tensor([1.0, 0.0])
    pred_zero = torch.tensor([0.0, 0.0])
    tgt_zero = torch.tensor([0.0, 0.0])
    other = torch.tensor([0.0, 9.0])
    base = sana_uni_loss(pred_plus, tgt_plus, pred_zero, tgt_zero)
    assert float(base) == pytest.approx(0.0, abs=1e-8)
    assert float(sana_uni_loss(pred_plus, tgt_plus, pred_zero + 1.0, tgt_zero)) > 0.0
    assert float(sana_uni_loss(pred_plus + 1.0, tgt_plus, pred_zero, tgt_zero)) > 0.0
    _ = other


def test_uni_loss_unused_and_token_hold_off_by_default():
    pred_plus = torch.tensor([1.0, 0.0])
    tgt_plus = torch.tensor([1.0, 0.0])
    pred_zero = torch.tensor([0.0, 0.0])
    tgt_zero = torch.tensor([0.0, 0.0])
    pred_unused = torch.tensor([9.0, 9.0])
    base = sana_uni_loss(pred_plus, tgt_plus, pred_zero, tgt_zero)
    ignored = sana_uni_loss(
        pred_plus,
        tgt_plus,
        pred_zero,
        tgt_zero,
        pred_unused=pred_unused,
        tgt_unused=tgt_zero,
        unused_token_hold=pred_plus.new_tensor(4.0),
    )
    assert float(ignored) == pytest.approx(float(base), abs=1e-8)
    forced_unused = sana_uni_loss(
        pred_plus,
        tgt_plus,
        pred_zero,
        tgt_zero,
        pred_unused=pred_unused,
        tgt_unused=tgt_zero,
        unused_weight=1.0,
    )
    assert float(forced_unused) > float(base)
    forced_hold = sana_uni_loss(
        pred_plus,
        tgt_plus,
        pred_zero,
        tgt_zero,
        unused_token_hold=pred_plus.new_tensor(4.0),
        token_hold_weight=1.0,
    )
    assert float(forced_hold) > float(base)


def test_canary_minus_is_unscored():
    pred = torch.tensor([0.0, 1.0])
    neg = torch.tensor([0.0, 1.0])
    can = sana_canary_minus(pred, neg)
    assert can["scored"] is False
    assert can["minus_overlap_neg"] == pytest.approx(1.0)


def test_attributes_pin_unused_on_pos_and_neu():
    rows = expand_attributes_sana(
        {
            "target": "a person",
            "positive": "a person smiling, happy expression",
            "neutral": "a person",
            "negative": "a sad person",
            "attributes": ["male", "female"],
        }
    )
    assert len(rows) == 2
    assert rows[0]["positive"] == "male a person smiling, happy expression"
    assert rows[0]["neutral"] == "male a person"
    assert rows[1]["positive"] == "female a person smiling, happy expression"
    assert rows[1]["neutral"] == "female a person"
    assert rows[0]["negative"] == "male a sad person"


def test_yaml_loads_positive_neutral_pins_and_fruit_bowl():
    prompts, meta = load_prompts(PROMPTS)
    assert meta.plus_label == "Happy"
    assert meta.minus_label == "Sad"
    assert meta.concept_words == "smiling, happy, joyful, smile"
    assert meta.control_prompt == "a bowl of fruit on a table"
    assert all(p.positive and p.neutral for p in prompts)
    assert any(p.positive.startswith("male ") for p in prompts)
    assert any(p.positive.startswith("female ") for p in prompts)
    assert all("smiling" in p.positive and "happy expression" in p.positive for p in prompts)
    assert all(p.neutral.endswith("a person") for p in prompts)
    assert all("happy" not in p.neutral for p in prompts)
    assert all("smiling" not in p.neutral for p in prompts)


def test_unused_hold_skips_concept_words():
    plus_ids = dummy_tokenize("male a person smiling, happy expression")
    neu_ids = dummy_tokenize("male a person")
    concept = sana_concept_token_ids("smiling, happy, joyful, smile", dummy_tokenize)
    assert dummy_tokenize("happy")[0] in concept
    assert dummy_tokenize("smiling")[0] in concept
    assert dummy_tokenize("smile")[0] in concept
    unused = zimage_unused_token_positions(plus_ids, concept)
    held_tokens = [plus_ids[i] for i in unused]
    assert dummy_tokenize("happy")[0] not in held_tokens
    assert dummy_tokenize("smiling")[0] not in held_tokens
    assert dummy_tokenize("male")[0] in held_tokens
    assert dummy_tokenize("person")[0] in held_tokens

    dim = 4
    plus_emb = torch.zeros(len(plus_ids), dim)
    neu_emb = torch.zeros(len(neu_ids), dim)
    happy_at = plus_ids.index(dummy_tokenize("happy")[0])
    plus_emb[happy_at] = torch.tensor([9.0, 9.0, 9.0, 9.0])
    hold = sana_unused_token_hold(plus_emb, neu_emb, plus_ids, neu_ids, concept)
    assert float(hold) == pytest.approx(0.0, abs=1e-8)
    plus_emb[plus_ids.index(dummy_tokenize("male")[0])] = 1.0
    moved = sana_unused_token_hold(plus_emb, neu_emb, plus_ids, neu_ids, concept)
    assert float(moved) > 0.0


def test_hold_fails_closed_without_concept_words():
    plus_ids = dummy_tokenize("male a person")
    concept = sana_concept_token_ids("happy", dummy_tokenize)
    with pytest.raises(SanaHoldError, match="not found"):
        zimage_require_concept_in_prompt(plus_ids, concept)
    with pytest.raises(SanaHoldError, match="required"):
        zimage_require_concept_in_prompt(plus_ids, set())


def test_sana_hold_accepts_leading_space_bpe_concept_piece():
    """Gemma/Sana: standalone `happy` is not the id used inside `a happy person`."""
    bare_happy = 101
    spaced_happy = 202
    other = {"a": 12, "person": 13}

    def fake_tokenize(text: str) -> list[int]:
        if text == "happy":
            return [bare_happy]
        if text == " happy":
            return [spaced_happy]
        if text == "a happy person":
            return [other["a"], spaced_happy, other["person"]]
        if text == "a person":
            return [other["a"], other["person"]]
        raise AssertionError(f"unexpected tokenize({text!r})")

    plus, neu = "a happy person", "a person"
    plus_ids = fake_tokenize(plus)
    neu_ids = fake_tokenize(neu)
    declared = sana_concept_token_ids("happy", fake_tokenize)
    assert bare_happy in declared
    assert spaced_happy in declared
    assert bare_happy not in plus_ids
    assert spaced_happy in plus_ids

    concept = resolve_sana_concept_ids(plus_ids, neu_ids, "happy", fake_tokenize)
    assert spaced_happy in concept
    dim = 4
    plus_emb = torch.zeros(len(plus_ids), dim)
    neu_emb = torch.zeros(len(neu_ids), dim)
    hold = sana_unused_token_hold(plus_emb, neu_emb, plus_ids, neu_ids, concept)
    assert float(hold) == pytest.approx(0.0, abs=1e-8)


def test_sana_hold_falls_back_to_plus_minus_neu_support():
    """If even ` happy` misses the + prompt, use set(plus) - set(neu)."""
    in_prompt_happy = 303

    def fake_tokenize(text: str) -> list[int]:
        if text == "happy":
            return [101]
        if text == " happy":
            return [202]
        if text == "a happy person":
            return [11, in_prompt_happy, 13]
        if text == "a person":
            return [12, 13]
        raise AssertionError(f"unexpected tokenize({text!r})")

    plus_ids = fake_tokenize("a happy person")
    neu_ids = fake_tokenize("a person")
    declared = sana_concept_token_ids("happy", fake_tokenize)
    assert in_prompt_happy not in declared
    with pytest.raises(SanaHoldError, match="not found"):
        zimage_require_concept_in_prompt(plus_ids, declared)

    concept = resolve_sana_concept_ids(plus_ids, neu_ids, "happy", fake_tokenize)
    assert concept == {11, in_prompt_happy}
    hold = sana_unused_token_hold(
        torch.zeros(len(plus_ids), 4),
        torch.zeros(len(neu_ids), 4),
        plus_ids,
        neu_ids,
        concept,
    )
    assert float(hold) == pytest.approx(0.0, abs=1e-8)
    with pytest.raises(SanaHoldError, match="not found"):
        resolve_sana_concept_ids([1, 2], [1, 2], "happy", lambda _t: [999])


def test_refuses_foreign_backends():
    with pytest.raises(ValueError, match="Sana-only"):
        assert_sana_only("Tongyi-MAI/Z-Image-Turbo")
    with pytest.raises(ValueError, match="Sana-only"):
        assert_sana_only("krea/Krea-2-Raw")
    with pytest.raises(ValueError, match="Sana-only"):
        assert_sana_only("circlestone-labs/Anima-Base-v1.0-Diffusers")
    assert_sana_only(SANA_MODEL_ID)


def test_live_tokenize_accepts_userdict_batch_encoding():
    """HF BatchEncoding is a UserDict, not a dict — do not iterate encoding keys."""
    from collections import UserDict
    from types import SimpleNamespace

    class FakeBatchEncoding(UserDict):
        pass

    class AttrEncoding:
        def __init__(self, ids):
            self.input_ids = ids

    class FakeTokenizer:
        def __init__(self, encoded):
            self.encoded = encoded

        def __call__(self, text, add_special_tokens=False):
            assert add_special_tokens is False
            return self.encoded

    def _tokenize_with(encoded):
        live = _LiveSana.__new__(_LiveSana)
        live.pipe = SimpleNamespace(tokenizer=FakeTokenizer(encoded))
        return live.tokenize("a bowl of fruit")

    userdict = FakeBatchEncoding({"input_ids": [7, 8, 9], "attention_mask": [1, 1, 1]})
    assert isinstance(userdict, dict) is False
    assert _tokenize_with(userdict) == [7, 8, 9]
    assert _tokenize_with(AttrEncoding(torch.tensor([[11, 12, 13]]))) == [11, 12, 13]
    assert _tokenize_with([[21, 22], [99]]) == [21, 22]


def test_dummy_train_never_imports_hub(tmp_path, monkeypatch):
    import sys

    banned = [name for name in list(sys.modules) if name.startswith("huggingface_hub")]
    for name in banned:
        monkeypatch.setitem(sys.modules, name, None)

    def _boom(*_a, **_k):
        raise AssertionError("Hub / SanaPipeline must not load in dummy mode")

    monkeypatch.setattr(
        "conceptmod.textsliders.train_lora_sana._LiveSana", _boom, raising=False
    )
    args = parse_args(
        [
            "--dummy",
            "--steps",
            "2",
            "--save_dir",
            str(tmp_path),
            "--name",
            "sana-dummy",
        ]
    )
    weights = train(args)
    assert weights.exists()
    sidecar = Path(str(weights).replace(".safetensors", ".json"))
    assert sidecar.exists()
    meta = json.loads(sidecar.read_text())
    assert meta["recipe"] == "uni"
    assert meta["dummy"] is True
    assert meta["train_method"] == "xattn"
    assert meta["lora_rank"] is None
    assert meta["resolution"] == 512
    assert meta["sample_guidance"] == 4.5
    assert meta["sample_steps"] == 20
    assert meta["control_prompt"] == "a bowl of fruit on a table"
    assert meta["teacher"]["minus"] == "canary only"
    assert meta["teacher"]["student"] == "train and infer both use the neutral caption at +1"
    assert "infer path" in meta["teacher"]["unused_hold"]
    assert "lyric" not in meta["teacher"]["unused_hold"]
    assert meta["teacher"]["cfg_compose"] == "v_u + g * (v_c - v_u)"
    assert meta["canary"]["scored"] is False


def test_dummy_lora_path_also_smokes(tmp_path):
    args = parse_args(
        [
            "--dummy",
            "--lora",
            "4",
            "--steps",
            "2",
            "--save_dir",
            str(tmp_path),
            "--name",
            "sana-lora-dummy",
        ]
    )
    weights = train(args)
    sidecar = json.loads(Path(str(weights).replace(".safetensors", ".json")).read_text())
    assert sidecar["lora_rank"] == 4
    assert sidecar["train_method"] == "xattn"


def test_trainer_student_plus_is_neu_not_pos():
    """Train +1 on neu (match infer). Concept teacher still uses + caption."""
    src = Path("conceptmod/textsliders/train_lora_sana.py").read_text()
    assert "pred_plus = _student(1.0, emb_neu)" in src
    assert "pred_zero = _student(0.0, emb_neu)" in src
    assert "pred_plus = _student(1.0, emb_pos)" not in src
    assert "pred_unused = _student(1.0, emb_neu)" not in src
    assert "sana_uni_teachers" in src
    assert "vel_pos" in src


def test_trainer_source_is_sana_only():
    src = Path("conceptmod/textsliders/train_lora_sana.py").read_text()
    assert "Efficient-Large-Model/Sana_600M_512px_diffusers" in src
    assert "lyric-hold" in src
    assert "Sena" not in src
    assert "from conceptmod.backends.anima" not in src
    assert "from conceptmod.backends.krea" not in src
    assert "Hunyuan" not in src
    music3 = Path("conceptmod/textsliders/train_lora_music3.py").read_text()
    assert "Sana_600M" not in music3
    assert "train_lora_sana" not in music3
    lm = Path("conceptmod/textsliders/train_lm_slider_music3.py").read_text()
    assert "train_lora_sana" not in lm


def test_docs_publish_the_live_train_command():
    doc = Path("docs/sana-slider.md").read_text()
    assert "train_lora_sana.py" in doc
    assert "Efficient-Large-Model/Sana_600M_512px_diffusers" in doc
    assert "--train_method xattn" in doc
    assert "--resolution 512" in doc
    assert "--sample_steps 20" in doc
    assert "--sample_guidance 4.5" in doc
    assert "a bowl of fruit on a table" in doc
    assert "happy-sana" in doc
    assert "smiling, happy, joyful, smile" in doc
    assert "a person smiling, happy expression" in doc
    assert "neu (infer path)" in doc
    assert "cancelled the" in doc
    assert "infer path" in doc
    assert "lyric-hold" in doc
    assert "v(z, t, c) − v(z, t, '')" in doc or "v(z, t, c) - v(z, t, '')" in doc
    readme = Path("README.md").read_text()
    assert "Sana" in readme
    assert "Sena" not in readme
