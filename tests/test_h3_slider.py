"""H3 UNI image-slider: CPU mocks only. No Hub, no GPU, no H3 weights."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
import yaml

from conceptmod.backends import BACKENDS, load_backend
from conceptmod.backends.h3 import DEFAULT_MODEL, DummyTokenizer, H3Backend
from conceptmod.textsliders.h3_uni import (
    concept_token_ids,
    h3_minus_canary,
    h3_uni_total_loss,
    h3_uni_velocity_loss,
    h3_unused_hold_loss,
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


def test_resolved_model_id_is_minimax_h3():
    assert DEFAULT_MODEL == "MiniMaxAI/MiniMax-H3"
    args = parse_args(["--dummy"])
    assert args.model_id == "MiniMaxAI/MiniMax-H3"
    assert args.backend == "h3"


def test_backends_register_h3_only():
    assert BACKENDS == ("h3",)
    assert "anima" not in BACKENDS
    assert "krea" not in BACKENDS
    assert "zit" not in BACKENDS
    assert "zimage" not in BACKENDS
    backend = load_backend("h3", device="cpu", dummy=True, resolution=32)
    assert isinstance(backend, H3Backend)
    assert backend.frozen is None
    assert backend.model_id == DEFAULT_MODEL
    with pytest.raises(ValueError, match="Anima / Krea / ZiT"):
        load_backend("anima", device="cpu", dummy=True)
    with pytest.raises(ValueError, match="Anima / Krea / ZiT"):
        load_backend("krea", device="cpu", dummy=True)
    with pytest.raises(ValueError, match="Anima / Krea / ZiT"):
        load_backend("zimage", device="cpu", dummy=True)


def test_lora_only_no_second_copy():
    backend = H3Backend(device="cpu", dummy=True, resolution=32)
    assert backend.frozen is None
    params = backend.trainable_parameters("lora")
    assert params
    names = {id(p) for p in params}
    # text table stays frozen; only LoRA tensors train
    assert id(backend.transformer.text_table.weight) not in names
    img = backend.generate("old person", seed=0, num_steps=2, guidance=1.0)
    assert img.size[0] >= 1 and img.size[1] >= 1
    z, t = backend.partial_denoise("person", stop_index=1, num_steps=3, guidance=1.0,
                                   generator=torch.Generator().manual_seed(0))
    assert z.shape[1:] == backend.latent_shape
    assert t.ndim == 0 or t.numel() == 1


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
    assert "old" not in words
    held = {words[i] for i, flag in enumerate(mask.tolist()) if flag and i < len(words)}
    assert "male" in held
    assert "old" not in held
    # concept word on the + caption is never a hold target
    pos_ids = tok.encode(pos)
    pos_mask = unused_hold_mask(pos_ids, unused, concept)
    pos_held = {pos.split()[i] for i, flag in enumerate(pos_mask.tolist()) if flag}
    assert "old" not in pos_held
    assert "male" in pos_held


def test_uni_velocity_has_no_minus_teacher():
    plus = torch.ones(2, 4)
    zero = torch.zeros(2, 4)
    loss = h3_uni_velocity_loss(plus, plus, zero, zero)
    assert float(loss.item()) == pytest.approx(0.0)
    miss = h3_uni_velocity_loss(plus, zero, zero, zero)
    assert float(miss.item()) > 0
    canary = h3_minus_canary(-plus, zero)
    assert float(canary.item()) > 0
    # canary is a different tensor; UNI total must not include it
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
    # unused token (male) is off encode(neu)
    student[0] = torch.tensor([1.0, 0.0, 0.0])
    # a later shared token is also off — must not be held
    if student.shape[0] > 1:
        student[-1] = torch.tensor([0.0, 4.0, 0.0])
    hold = h3_unused_hold_loss(student, neu_e, mask)
    # only the unused column contributes
    only_unused = h3_unused_hold_loss(student[:1], neu_e[:1], mask[:1])
    assert float(hold.item()) == pytest.approx(float(only_unused.item()))


def test_yaml_pins_unused_and_keeps_concept_free():
    path = Path("conceptmod/textsliders/data/prompts-h3.yaml")
    rows = load_slider_rows(str(path), "")
    assert len(rows) == 2
    positives = {r["positive"] for r in rows}
    neutrals = {r["neutral"] for r in rows}
    assert "male old person" in positives
    assert "female old person" in positives
    assert "male person" in neutrals
    assert "female person" in neutrals
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
        "--lr", "0.05",
        "--seed", "0",
        "--resolution", "32",
    ])
    sidecar = train(args)
    assert sidecar["model_id"] == "MiniMaxAI/MiniMax-H3"
    assert sidecar["resolved_model_id"] == "MiniMaxAI/MiniMax-H3"
    assert sidecar["recipe"] == "h3_uni"
    assert sidecar["minus_teacher"] is False
    assert sidecar["minus_canary"] is True
    assert sidecar["lora_only"] is True
    assert sidecar["hold_concept_words"] is False
    assert sidecar["first_loss"] > sidecar["last_loss"]
    assert (tmp_path / "h3-dummy_last.json").is_file()
    data = json.loads((tmp_path / "h3-dummy_last.json").read_text())
    assert data["backend"] == "h3"


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


def test_h3_config_points_at_minimax():
    cfg = yaml.safe_load(Path("conceptmod/textsliders/data/config-h3.yaml").read_text())
    assert cfg["pretrained_model"]["name_or_path"] == "MiniMaxAI/MiniMax-H3"
    assert cfg["train"]["recipe"] == "h3_uni"


def test_live_load_is_not_imported_on_dummy():
    """Constructing --dummy must not import MiniMaxH3Pipeline / Hub."""
    import conceptmod.backends.h3 as h3

    called = {"n": 0}

    def boom(*_a, **_k):
        called["n"] += 1
        raise AssertionError("live H3 loader must not run in dummy mode")

    orig = h3._load_h3_pipeline
    h3._load_h3_pipeline = boom
    try:
        backend = H3Backend(device="cpu", dummy=True, resolution=16)
        _ = backend.encode_text("person")
        _ = backend.predict_v("person", torch.randn(1, *backend.latent_shape),
                              torch.tensor([100.0]), frozen=True)
    finally:
        h3._load_h3_pipeline = orig
    assert called["n"] == 0
