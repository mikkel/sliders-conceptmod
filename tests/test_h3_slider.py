"""H3 UNI image-slider: CPU mocks only. No Hub, no GPU, no H3 weights."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
import yaml

from conceptmod.backends import BACKENDS, load_backend
from conceptmod.backends.h3 import (
    DEFAULT_MODEL,
    DISTIL_MODEL,
    ArchitectureMismatch,
    DummyTokenizer,
    H3Backend,
)
from conceptmod.textsliders.h3_uni import (
    concept_token_ids,
    h3_minus_canary,
    h3_uni_hidden_loss,
    h3_uni_total_loss,
    h3_unused_hold_loss,
    last_hidden,
    pin_unused_attributes,
    unused_hold_mask,
    unused_token_ids,
)
from conceptmod.textsliders.train_lora_h3 import load_slider_rows, parse_args, train
from conceptmod.textsliders.train_lm_slider_music3 import (
    parse_args as parse_music3_lm,
    resolve_lm_recipe,
    resolve_pole_mode,
)


def test_resolved_model_id_is_hunyuanimage_3():
    assert DEFAULT_MODEL == "tencent/HunyuanImage-3.0"
    assert DISTIL_MODEL == "tencent/HunyuanImage-3.0-Instruct-Distil"
    args = parse_args(["--dummy"])
    assert args.model_id == "tencent/HunyuanImage-3.0"
    assert args.backend == "h3"


def test_backends_register_h3_only():
    assert BACKENDS == ("h3",)
    assert "anima" not in BACKENDS
    assert "krea" not in BACKENDS
    assert "zit" not in BACKENDS
    assert "zimage" not in BACKENDS
    backend = load_backend("h3", device="cpu", dummy=True)
    assert isinstance(backend, H3Backend)
    assert backend.frozen is None
    assert backend.model_id == DEFAULT_MODEL
    with pytest.raises(ValueError, match="Anima / Krea / ZiT"):
        load_backend("anima", device="cpu", dummy=True)
    with pytest.raises(ValueError, match="Anima / Krea / ZiT"):
        load_backend("krea", device="cpu", dummy=True)
    with pytest.raises(ValueError, match="Anima / Krea / ZiT"):
        load_backend("zimage", device="cpu", dummy=True)


def test_does_not_fake_velocity_trainer():
    backend = H3Backend(device="cpu", dummy=True)
    assert backend.frozen is None
    params = backend.trainable_parameters("lora")
    assert params
    assert id(backend.transformer.embed.weight) not in {id(p) for p in params}
    with pytest.raises(ArchitectureMismatch, match="autoregressive MoE"):
        backend.predict_v("person", torch.zeros(1, 4, 8, 8), torch.tensor([1.0]), frozen=True)
    with pytest.raises(ArchitectureMismatch, match="no Euler"):
        backend.partial_denoise("person", 0, 4, 1.0, torch.Generator())
    img = backend.generate("old person", seed=0)
    assert img.size[0] >= 1 and img.size[1] >= 1


def test_pin_unused_attributes():
    rows = pin_unused_attributes("old person", "person", ["male", "female"])
    assert rows == [
        ("male old person", "male person"),
        ("female old person", "female person"),
    ]
    assert pin_unused_attributes("old person", "person", []) == [("old person", "person")]


def test_hold_unused_tokens_not_concept_words():
    tok = DummyTokenizer()
    pos, neu = "male old person", "male person"
    concept = concept_token_ids(tok, pos, neu)
    unused = unused_token_ids(tok, ["male", "female"])
    ids = tok.encode(neu)
    mask = unused_hold_mask(ids, unused, concept)
    words = neu.split()
    held = {words[i] for i, flag in enumerate(mask.tolist()) if flag and i < len(words)}
    assert "male" in held
    assert "old" not in held
    pos_ids = tok.encode(pos)
    pos_held = {pos.split()[i] for i, flag in enumerate(unused_hold_mask(pos_ids, unused, concept).tolist()) if flag}
    assert "old" not in pos_held
    assert "male" in pos_held


def test_uni_hidden_has_no_minus_teacher():
    plus = torch.ones(2, 4)
    zero = torch.zeros(2, 4)
    loss = h3_uni_hidden_loss(plus, plus, zero, zero)
    assert float(loss.item()) == pytest.approx(0.0)
    assert float(h3_uni_hidden_loss(plus, zero, zero, zero).item()) > 0
    canary = h3_minus_canary(-plus, zero)
    assert float(canary.item()) > 0
    total = h3_uni_total_loss(plus, plus, zero, zero)
    assert float(total.item()) == pytest.approx(0.0)
    assert float(total.item()) != pytest.approx(float((total + canary).item()))


def test_unused_hold_mse_skips_concept_tokens():
    tok = DummyTokenizer()
    pos, neu = "male old person", "male person"
    ids = tok.encode(neu)
    concept = concept_token_ids(tok, pos, neu)
    unused = unused_token_ids(tok, ["male"])
    mask = unused_hold_mask(ids, unused, concept)
    student = torch.zeros(len(ids), 3)
    neu_e = torch.zeros(len(ids), 3)
    student[0] = torch.tensor([1.0, 0.0, 0.0])
    if student.shape[0] > 1:
        student[-1] = torch.tensor([0.0, 4.0, 0.0])
    hold = h3_unused_hold_loss(student, neu_e, mask)
    only_unused = h3_unused_hold_loss(student[:1], neu_e[:1], mask[:1])
    assert float(hold.item()) == pytest.approx(float(only_unused.item()))


def test_yaml_pins_unused_and_keeps_concept_free():
    rows = load_slider_rows("conceptmod/textsliders/data/prompts-h3.yaml", "")
    assert {r["positive"] for r in rows} >= {"male old person", "female old person"}
    assert {r["neutral"] for r in rows} >= {"male person", "female person"}
    for r in rows:
        assert "old" in r["positive"]
        assert "old" not in r["neutral"].split()


def test_dummy_train_drops_uni_loss_and_writes_sidecar(tmp_path):
    prompts = tmp_path / "one.yaml"
    prompts.write_text(
        "- target: person\n  positive: old person\n  neutral: person\n"
        "  unconditional: ''\n  attributes: []\n"
    )
    args = parse_args([
        "--dummy",
        "--steps", "12",
        "--name", "h3-dummy",
        "--save_dir", str(tmp_path),
        "--prompts_file", str(prompts),
        "--lr", "0.2",
        "--seed", "0",
    ])
    sidecar = train(args)
    assert sidecar["model_id"] == "tencent/HunyuanImage-3.0"
    assert sidecar["resolved_model_id"] == "tencent/HunyuanImage-3.0"
    assert sidecar["stack"] == "autoregressive_moe"
    assert sidecar["recipe"] == "h3_uni_encode"
    assert sidecar["minus_teacher"] is False
    assert sidecar["minus_canary"] is True
    assert sidecar["lora_only"] is True
    assert sidecar["velocity_trainer"] is False
    assert sidecar["hold_concept_words"] is False
    assert sidecar["first_loss"] > sidecar["last_loss"]
    data = json.loads((tmp_path / "h3-dummy_last.json").read_text())
    assert data["backend"] == "h3"


def test_encode_uni_last_hidden_moves_plus_not_zero():
    backend = H3Backend(device="cpu", dummy=True)
    pos = last_hidden(backend.encode_text("old person", frozen=True).embeds)
    neu = last_hidden(backend.encode_text("person", frozen=True).embeds)
    assert not torch.allclose(pos, neu)
    zero = last_hidden(backend.encode_scaled("person", 0.0).embeds)
    assert torch.allclose(zero, neu)
    plus0 = last_hidden(backend.encode_scaled("person", 1.0).embeds)
    # before training, +1 last is still neu-pooled (delta=0)
    assert torch.allclose(plus0, neu)


def test_music3_defaults_unchanged():
    tf_src = Path("conceptmod/textsliders/train_lora_music3.py").read_text()
    assert 'parser.add_argument("--steps", type=int, default=500)' in tf_src
    assert 'parser.add_argument("--rank", type=int, default=8)' in tf_src
    assert 'parser.add_argument("--lr", type=float, default=2e-3' in tf_src
    lm = parse_music3_lm(["--prompts_file", "prompts.yaml"])
    assert lm.lm_target == "v9"
    assert lm.pole_mode == "hidden"
    assert resolve_lm_recipe(lm_target=lm.lm_target, symmetric=lm.symmetric) == "v9"
    assert resolve_pole_mode(lm.pole_mode) == "hidden"


def test_h3_config_points_at_hunyuanimage():
    cfg = yaml.safe_load(Path("conceptmod/textsliders/data/config-h3.yaml").read_text())
    assert cfg["pretrained_model"]["name_or_path"] == "tencent/HunyuanImage-3.0"
    assert cfg["pretrained_model"]["stack"] == "autoregressive_moe"
    assert cfg["train"]["recipe"] == "h3_uni_encode"


def test_live_load_is_not_imported_on_dummy():
    import conceptmod.backends.h3 as h3

    called = {"n": 0}

    def boom(*_a, **_k):
        called["n"] += 1
        raise AssertionError("live H3 loader must not run in dummy mode")

    orig = h3._load_h3_ar
    h3._load_h3_ar = boom
    try:
        backend = H3Backend(device="cpu", dummy=True)
        _ = backend.encode_text("person")
    finally:
        h3._load_h3_ar = orig
    assert called["n"] == 0
