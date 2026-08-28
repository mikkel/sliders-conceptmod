"""MiniMax-H3 UNI slider: CPU mocks only. No Hub, no GPU, no H3 weights."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
import yaml

from conceptmod.textsliders.minimax_h3_backend import (
    DEFAULT_MODEL,
    DEFAULT_SAMPLE_DURATION,
    DEFAULT_SAMPLE_SCALES,
    DEFAULT_TASK_INDEX,
    DEFAULT_VARIANT,
    DEFAULT_WORKFLOW,
    FREEZE_LIST,
    H3_FPS,
    HOSTED_NOT_IN_WEIGHTS,
    LORA_ATTN_CLASS,
    LORA_LINEAR_NAMES,
    ArchitectureMismatch,
    DummyTokenizer,
    MiniMaxH3Backend,
    h3_canvas_hw,
    h3_num_frames,
    place_minimax_h3_pipeline,
    resolve_h3_lora_path,
    same_device,
)
from conceptmod.textsliders.minimax_h3_uni import (
    apply_unused_hold,
    concept_token_ids,
    minimax_h3_minus_canary,
    minimax_h3_uni_total_loss,
    minimax_h3_uni_velocity_loss,
    minimax_h3_unused_hold_loss,
    pin_unused_attributes,
    unused_hold_mask,
    unused_token_ids,
    velocity_pair,
)
from conceptmod.textsliders.train_lora_minimax_h3 import (
    load_slider_rows,
    parse_args,
    parse_sample_scales,
    sample_prompts_from_rows,
    train,
)


def test_resolved_model_id_is_minimax_h3():
    assert DEFAULT_MODEL == "MiniMaxAI/MiniMax-H3"
    assert DEFAULT_VARIANT == "FL2VA"
    assert DEFAULT_WORKFLOW == "t2va"
    assert DEFAULT_TASK_INDEX == "FL2VA/model_index.json"
    args = parse_args(["--dummy"])
    assert args.model_id == "MiniMaxAI/MiniMax-H3"
    assert args.variant == "FL2VA"
    assert args.workflow == "t2va"
    assert args.short_side == 768
    assert args.guidance == 0.0
    assert args.rank == 8
    assert args.encoder_device is None
    assert args.load_h3_lora is None
    assert args.no_sample is False
    assert args.sample_scales == "0,1"
    assert args.sample_duration == DEFAULT_SAMPLE_DURATION == 5.0
    assert args.sample_fps == H3_FPS == 24.0
    assert args.sample_short_side is None
    assert parse_sample_scales(args.sample_scales) == list(DEFAULT_SAMPLE_SCALES)


def test_no_hunyuan_hub_id_in_new_files():
    root = Path("conceptmod/textsliders")
    for path in (
        root / "minimax_h3_backend.py",
        root / "minimax_h3_uni.py",
        root / "train_lora_minimax_h3.py",
        root / "data" / "prompts-minimax-h3.yaml",
        root / "data" / "config-minimax-h3.yaml",
        root / "data" / "prompts-minimax-h3-chiaroscuro.yaml",
        root / "data" / "config-minimax-h3-chiaroscuro.yaml",
        Path("docs/minimax-h3-slider.md"),
    ):
        text = path.read_text()
        assert "tencent/HunyuanImage-3.0" not in text
        assert "HunyuanImage-3.0" not in text


def test_lora_attaches_to_minimax_h3_attention_linears():
    backend = MiniMaxH3Backend(device="cpu", dummy=True)
    names = backend.lora_module_names()
    assert names
    joined = " ".join(names)
    assert LORA_ATTN_CLASS == "MiniMaxH3Attention"
    for piece in LORA_LINEAR_NAMES:
        token = piece.replace(".", "-")
        assert token in joined, (piece, names)
    assert "adaln" not in joined.lower()
    params = backend.trainable_parameters()
    assert params
    enc_ids = {id(p) for p in backend.encoder.parameters()}
    assert not enc_ids.intersection({id(p) for p in params})
    for block in backend.transformer.transformer_blocks:
        assert id(block.adaln_proj.weight) not in {id(p) for p in params}


def test_does_not_fake_predict_v():
    backend = MiniMaxH3Backend(device="cpu", dummy=True)
    with pytest.raises(ArchitectureMismatch, match="packed multimodal sequence"):
        backend.predict_v("person", torch.zeros(1, 2, 4), torch.tensor([1.0]))
    plus = backend.encode_text("old person")
    packed = backend.pack_t2va(plus)
    assert packed.position_ids.shape[-1] == 3
    out = backend.forward_velocity(packed, scale=0.0)
    assert out.sample.shape[0] == 1
    assert out.audio_sample.shape[0] == 1


def test_freeze_list_is_encoder_and_vaes():
    assert FREEZE_LIST == (
        "text_encoder",
        "visual_vae",
        "audio_vae",
        "processor",
        "tokenizer",
    )
    assert "H3-Context-IR" in HOSTED_NOT_IN_WEIGHTS
    assert "H3-Regenerate-2K" in HOSTED_NOT_IN_WEIGHTS
    backend = MiniMaxH3Backend(device="cpu", dummy=True)
    assert all(not p.requires_grad for p in backend.encoder.parameters())
    assert all(not p.requires_grad for p in backend.visual_vae.parameters())
    assert all(not p.requires_grad for p in backend.audio_vae.parameters())


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
    pos_held = {
        pos.split()[i]
        for i, flag in enumerate(unused_hold_mask(pos_ids, unused, concept).tolist())
        if flag
    }
    assert "old" not in pos_held
    assert "male" in pos_held


def test_apply_unused_hold_copies_neu_not_concept():
    tok = DummyTokenizer()
    pos, neu = "male old person", "male person"
    plus_ids = tok.encode(pos)
    neu_ids = tok.encode(neu)
    concept = concept_token_ids(tok, pos, neu)
    unused = unused_token_ids(tok, ["male"])
    mask = unused_hold_mask(plus_ids, unused, concept)
    plus_h = torch.arange(len(plus_ids) * 3, dtype=torch.float32).reshape(len(plus_ids), 3)
    neu_h = torch.ones(len(neu_ids), 3)
    held = apply_unused_hold(plus_h, neu_h, plus_ids, neu_ids, mask)
    # "male" is unused → encode(neu). "old" is concept → stays plus.
    assert torch.allclose(held[0], neu_h[0])
    assert torch.allclose(held[1], plus_h[1])


def test_uni_velocity_has_no_minus_teacher():
    plus = torch.ones(2, 4)
    zero = torch.zeros(2, 4)
    loss = minimax_h3_uni_velocity_loss(plus, plus, zero, zero)
    assert float(loss.item()) == pytest.approx(0.0)
    assert float(minimax_h3_uni_velocity_loss(plus, zero, zero, zero).item()) > 0
    canary = minimax_h3_minus_canary(-plus, zero)
    assert float(canary.item()) > 0
    total = minimax_h3_uni_total_loss(plus, plus, zero, zero)
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
    hold = minimax_h3_unused_hold_loss(student, neu_e, mask)
    only_unused = minimax_h3_unused_hold_loss(student[:1], neu_e[:1], mask[:1])
    assert float(hold.item()) == pytest.approx(float(only_unused.item()))


def test_yaml_pins_unused_and_keeps_concept_free():
    rows = load_slider_rows("conceptmod/textsliders/data/prompts-minimax-h3.yaml", "")
    assert {r["positive"] for r in rows} >= {"male old person", "female old person"}
    assert {r["neutral"] for r in rows} >= {"male person", "female person"}
    for r in rows:
        assert "old" in r["positive"]
        assert "old" not in r["neutral"].split()


def _h3_subject_clause(caption: str) -> str:
    text = caption.strip()
    for prefix in ("male ", "female "):
        if text.lower().startswith(prefix):
            text = text[len(prefix):]
            break
    return text.split(",")[0].strip().lower()


def test_chiaroscuro_yaml_loads_same_subject_lighting():
    path = Path("conceptmod/textsliders/data/prompts-minimax-h3-chiaroscuro.yaml")
    raw = yaml.safe_load(path.read_text())
    assert isinstance(raw, list)
    assert len(raw) == 3
    required = (
        "target", "positive", "neutral", "unconditional", "negative",
        "attributes", "action",
    )
    for item in raw:
        for key in required:
            assert key in item
        assert item["action"] == "enhance"
        assert item["attributes"] == ["male", "female"]
        subject = _h3_subject_clause(item["target"])
        assert subject == _h3_subject_clause(item["positive"])
        assert subject == _h3_subject_clause(item["neutral"])
        assert subject == _h3_subject_clause(item["unconditional"])
        assert subject == _h3_subject_clause(item["negative"])
        assert "chiaroscuro" in item["positive"]
        assert "dramatic single-source side light" in item["positive"]
        assert "deep shadows" in item["positive"]
        assert "high contrast" in item["positive"]
        assert "flat even lighting" in item["neutral"]
        assert "soft fill" in item["neutral"]
        assert "low contrast" in item["neutral"]
        assert "cartoon" not in item["neutral"]
        assert "simple flat" not in item["neutral"]
        assert "chiaroscuro" not in item["neutral"]
        assert "washed-out" in item["unconditional"]
        assert "featureless" in item["unconditional"]
        assert item["unconditional"] == item["negative"]
    subjects = {_h3_subject_clause(item["target"]) for item in raw}
    assert subjects == {
        "person sitting in a chair",
        "interior room with a wooden chair and a table",
        "ceramic vase on a wooden table",
    }
    rows = load_slider_rows(str(path), "")
    assert len(rows) == 6
    assert {r["positive"].split(",")[0] for r in rows} >= {
        "male person sitting in a chair",
        "female person sitting in a chair",
    }
    for row in rows:
        assert row["positive"].startswith(("male ", "female "))
        assert _h3_subject_clause(row["positive"]) == _h3_subject_clause(row["neutral"])
        assert "chiaroscuro" in row["positive"]
        assert "chiaroscuro" not in row["neutral"]
        assert "washed-out" in row["unconditional"]


def test_chiaroscuro_config_and_docs_card():
    cfg = yaml.safe_load(
        Path("conceptmod/textsliders/data/config-minimax-h3-chiaroscuro.yaml").read_text()
    )
    assert cfg["prompts_file"] == "conceptmod/textsliders/data/prompts-minimax-h3-chiaroscuro.yaml"
    assert cfg["pretrained_model"]["name_or_path"] == "MiniMaxAI/MiniMax-H3"
    assert cfg["pretrained_model"]["variant"] == "FL2VA"
    assert cfg["pretrained_model"]["workflow"] == "t2va"
    assert cfg["train"]["recipe"] == "minimax_h3_uni_velocity"
    assert cfg["train"]["guidance"] == 0
    assert cfg["train"]["short_side"] == 768
    assert cfg["network"]["target"] == "MiniMaxH3Attention"
    assert cfg["network"]["train_adaln"] is False
    assert cfg["save"]["name"] == "chiaroscuro-minimax-h3-uni"
    docs = Path("docs/minimax-h3-slider.md").read_text()
    assert "--name chiaroscuro-minimax-h3-uni" in docs
    assert "prompts-minimax-h3-chiaroscuro.yaml" in docs
    assert "--rank 8 --alpha 8 --lr 1e-4 --steps 500" in docs
    assert "--name chiaroscuro-minimax-h3-uni-r16" in docs
    assert "--rank 16 --alpha 16 --lr 1e-4 --steps 1200" in docs
    assert "800" in docs and "1500" in docs
    assert "escalate" in docs
    assert "--short_side 768 --guidance 0" in docs
    assert "FL2VA" in docs and "t2va" in docs
    assert "MiniMaxAI/MiniMax-H3" in docs
    assert "sampled frames" in docs or "sampled-frame" in docs
    assert "sampled videos" in docs
    assert "last-50" in docs
    assert "B200" in docs and "B300" in docs
    assert "--encoder_device cuda:1" in docs
    assert "--load_h3_lora" in docs
    assert "--steps 0" in docs
    assert "--sample_scales 0,1" in docs
    assert "--sample_duration 5" in docs
    assert "--sample_fps 24" in docs


def test_dummy_train_drops_uni_loss_and_writes_sidecar(tmp_path):
    prompts = tmp_path / "one.yaml"
    prompts.write_text(
        "- target: person\n  positive: old person\n  neutral: person\n"
        "  unconditional: ''\n  attributes: []\n"
    )
    args = parse_args([
        "--dummy",
        "--steps", "20",
        "--name", "minimax-h3-dummy",
        "--save_dir", str(tmp_path),
        "--prompts_file", str(prompts),
        "--lr", "0.5",
        "--seed", "0",
    ])
    sidecar = train(args)
    assert sidecar["model_id"] == "MiniMaxAI/MiniMax-H3"
    assert sidecar["resolved_model_id"] == "MiniMaxAI/MiniMax-H3"
    assert sidecar["variant"] == "FL2VA"
    assert sidecar["workflow"] == "t2va"
    assert sidecar["stack"] == "omni_transformer_flow"
    assert sidecar["recipe"] == "minimax_h3_uni_velocity"
    assert sidecar["minus_teacher"] is False
    assert sidecar["minus_canary"] is True
    assert sidecar["lora_only"] is True
    assert sidecar["lora_host"] == "MiniMaxH3Attention"
    assert sidecar["lora_linears"] == ["to_q", "to_k", "to_v", "to_out.0"]
    assert sidecar["train_adaln"] is False
    assert sidecar["velocity_teacher"] == "data_pointing"
    assert sidecar["predict_v_faked"] is False
    assert sidecar["cfg_distilled"] is True
    assert sidecar["guidance"] == 0.0
    assert sidecar["short_side"] == 768
    assert sidecar["hold_concept_words"] is False
    assert sidecar["hosted_not_in_weights"] == ["H3-Context-IR", "H3-Regenerate-2K"]
    assert sidecar["first_loss"] > sidecar["last_loss"]
    data = json.loads((tmp_path / "minimax-h3-dummy_last.json").read_text())
    assert data["backend"] == "minimax_h3"


def test_scale_zero_matches_neu_teacher():
    backend = MiniMaxH3Backend(device="cpu", dummy=True)
    neu = backend.encode_text("person")
    packed = backend.pack_t2va(neu)
    teacher = backend.forward_velocity(packed, scale=0.0)
    student = backend.forward_velocity(packed, scale=0.0)
    assert torch.allclose(teacher.sample, student.sample)
    assert torch.allclose(teacher.audio_sample, student.audio_sample)


def test_plus_and_neu_velocities_differ():
    backend = MiniMaxH3Backend(device="cpu", dummy=True)
    plus = backend.pack_t2va(backend.encode_text("old person"))
    neu = backend.pack_t2va(
        backend.encode_text("person"),
        video_latents=plus.hidden_states,
        audio_latents=plus.audio_hidden_states,
    )
    v_plus = backend.forward_velocity(plus, scale=0.0)
    v_neu = backend.forward_velocity(neu, scale=0.0)
    assert not torch.allclose(v_plus.sample, v_neu.sample)
    pair = velocity_pair(v_plus.sample, v_plus.audio_sample)
    assert pair.ndim == 2


def test_music3_defaults_unchanged():
    tf_src = Path("conceptmod/textsliders/train_lora_music3.py").read_text()
    assert 'parser.add_argument("--steps", type=int, default=500)' in tf_src
    assert 'parser.add_argument("--rank", type=int, default=8)' in tf_src
    assert 'parser.add_argument("--lr", type=float, default=2e-3' in tf_src
    lm_src = Path("conceptmod/textsliders/train_lm_slider_music3.py").read_text()
    assert '"--lm_target"' in lm_src and 'default="v9"' in lm_src
    assert '"--pole_mode"' in lm_src and 'default="hidden"' in lm_src
    music3_yaml = Path("conceptmod/textsliders/data/prompts-music3.yaml").read_text()
    assert "MiniMax-H3" not in music3_yaml


def test_config_points_at_minimax_h3():
    cfg = yaml.safe_load(Path("conceptmod/textsliders/data/config-minimax-h3.yaml").read_text())
    assert cfg["pretrained_model"]["name_or_path"] == "MiniMaxAI/MiniMax-H3"
    assert cfg["pretrained_model"]["variant"] == "FL2VA"
    assert cfg["pretrained_model"]["workflow"] == "t2va"
    assert cfg["train"]["recipe"] == "minimax_h3_uni_velocity"
    assert cfg["train"]["guidance"] == 0
    assert cfg["train"]["short_side"] == 768
    assert cfg["network"]["target"] == "MiniMaxH3Attention"
    assert cfg["network"]["train_adaln"] is False


def test_live_load_is_not_imported_on_dummy():
    import conceptmod.textsliders.minimax_h3_backend as h3

    called = {"n": 0}

    def boom(*_a, **_k):
        called["n"] += 1
        raise AssertionError("live MiniMax-H3 loader must not run in dummy mode")

    orig = h3._load_minimax_h3_modular
    h3._load_minimax_h3_modular = boom
    try:
        backend = MiniMaxH3Backend(device="cpu", dummy=True)
        _ = backend.encode_text("person")
        _ = backend.forward_velocity(backend.pack_t2va(backend.encode_text("person")), scale=1.0)
        _ = backend.generate_t2va("person sitting in a chair", scale=0.0, seed=0)
    finally:
        h3._load_minimax_h3_modular = orig
    assert called["n"] == 0


def test_sample_geometry_defaults():
    assert h3_num_frames(5.0, 24.0) == 124
    # 4.5s * 24 = 108, which is above 107, so snap-up is 124.
    assert h3_num_frames(4.5, 24.0) == 124
    assert h3_num_frames(107 / 24.0, 24.0) == 107
    height, width = h3_canvas_hw(768)
    assert height == 768
    assert width == 1344
    tight_h, tight_w = h3_canvas_hw(544)
    assert tight_h == 544
    assert tight_w == 960


def test_sample_prompts_are_unique_yaml_targets():
    rows = load_slider_rows(
        "conceptmod/textsliders/data/prompts-minimax-h3-chiaroscuro.yaml", ""
    )
    prompts = sample_prompts_from_rows(rows)
    assert prompts == [
        "person sitting in a chair",
        "interior room with a wooden chair and a table",
        "ceramic vase on a wooden table",
    ]
    assert sample_prompts_from_rows(rows, max_rows=1) == ["person sitting in a chair"]


class _FakeMod:
    def __init__(self) -> None:
        self.device = None

    def to(self, device):
        self.device = str(device)
        return self


class _FakePipe:
    def __init__(self) -> None:
        self.transformer = _FakeMod()
        self.vae = _FakeMod()
        self.audio_vae = _FakeMod()
        self.text_encoder = _FakeMod()
        self.to_calls: list[str] = []

    def to(self, device):
        self.to_calls.append(str(device))
        return self


def test_place_pipeline_default_blankets_single_device():
    pipe = _FakePipe()
    place_minimax_h3_pipeline(pipe, device="cuda:0")
    assert pipe.to_calls == ["cuda:0"]
    place_minimax_h3_pipeline(pipe, device="cuda:0", encoder_device="cuda:0")
    assert pipe.to_calls == ["cuda:0", "cuda:0"]
    assert same_device("cuda", "cuda:0")


def test_place_pipeline_encoder_device_skips_blanket_to():
    pipe = _FakePipe()
    place_minimax_h3_pipeline(pipe, device="cuda:0", encoder_device="cuda:1")
    assert pipe.to_calls == []
    assert pipe.transformer.device == "cuda:0"
    assert pipe.vae.device == "cuda:0"
    assert pipe.audio_vae.device == "cuda:0"
    assert pipe.text_encoder.device == "cuda:1"


def test_dummy_train_writes_t2va_samples(tmp_path):
    prompts = tmp_path / "one.yaml"
    prompts.write_text(
        "- target: person sitting in a chair\n  positive: chiaroscuro person sitting in a chair\n"
        "  neutral: person sitting in a chair\n  unconditional: ''\n  attributes: []\n"
    )
    args = parse_args([
        "--dummy",
        "--steps", "2",
        "--name", "h3-sample",
        "--save_dir", str(tmp_path),
        "--prompts_file", str(prompts),
        "--sample_scales", "0,1",
        "--sample_duration", "5",
        "--sample_fps", "24",
        "--seed", "0",
    ])
    sidecar = train(args)
    assert sidecar["guidance"] == 0.0
    assert sidecar["sample_grid"]["guidance"] == 0.0
    assert sidecar["sample_grid"]["scales"] == [0.0, 1.0]
    assert sidecar["sample_grid"]["duration"] == 5.0
    assert sidecar["sample_grid"]["fps"] == 24.0
    samples = tmp_path / "samples"
    meta = json.loads((samples / "final_meta.json").read_text())
    assert meta["guidance"] == 0.0
    assert meta["scales"] == [0.0, 1.0]
    mp4s = sorted(samples.glob("*.mp4"))
    assert [p.name for p in mp4s] == [
        "final_person-sitting-in-a-chair_scale0.mp4",
        "final_person-sitting-in-a-chair_scale1.mp4",
    ]
    for path in mp4s:
        assert path.stat().st_size > 0
        assert path.read_bytes()[:8].endswith(b"ftyp") or b"ftyp" in path.read_bytes()[:32]


def test_load_h3_lora_roundtrip_and_steps_zero_sample(tmp_path):
    prompts = tmp_path / "one.yaml"
    prompts.write_text(
        "- target: ceramic vase on a wooden table\n  positive: chiaroscuro ceramic vase on a wooden table\n"
        "  neutral: ceramic vase on a wooden table\n  unconditional: ''\n  attributes: []\n"
    )
    train_dir = tmp_path / "trained"
    args = parse_args([
        "--dummy",
        "--steps", "4",
        "--name", "h3-load",
        "--save_dir", str(train_dir),
        "--prompts_file", str(prompts),
        "--no_sample",
        "--seed", "1",
    ])
    train(args)
    lora_path = resolve_h3_lora_path(str(train_dir))
    assert lora_path.name.endswith(".safetensors")
    src = MiniMaxH3Backend(device="cpu", dummy=True)
    src.load_trained(str(train_dir))
    dst = MiniMaxH3Backend(device="cpu", dummy=True)
    dst.load_trained(str(lora_path))
    for a, b in zip(src.network.parameters(), dst.network.parameters()):
        assert torch.allclose(a, b)

    sample_dir = tmp_path / "reload"
    reload_args = parse_args([
        "--dummy",
        "--steps", "0",
        "--name", "h3-reload",
        "--save_dir", str(sample_dir),
        "--prompts_file", str(prompts),
        "--load_h3_lora", str(train_dir),
        "--sample_scales", "0,0.5,1",
        "--encoder_device", "cuda:1",
        "--device", "cuda:0",
    ])
    assert reload_args.steps == 0
    assert reload_args.load_h3_lora == str(train_dir)
    assert reload_args.encoder_device == "cuda:1"
    sidecar = train(reload_args)
    assert sidecar["steps"] == 0
    assert sidecar["first_loss"] is None
    assert sidecar["load_h3_lora"]
    assert sidecar["encoder_device"] is None  # dummy stays CPU
    assert sidecar["sample_grid"]["scales"] == [0.0, 0.5, 1.0]
    mp4s = {p.name for p in (sample_dir / "samples").glob("*.mp4")}
    assert mp4s == {
        "final_ceramic-vase-on-a-wooden-table_scale0.mp4",
        "final_ceramic-vase-on-a-wooden-table_scale0.5.mp4",
        "final_ceramic-vase-on-a-wooden-table_scale1.mp4",
    }


def test_dummy_encoder_device_does_not_hit_hub():
    import conceptmod.textsliders.minimax_h3_backend as h3

    called = {"n": 0}

    def boom(*_a, **_k):
        called["n"] += 1
        raise AssertionError("live MiniMax-H3 loader must not run in dummy mode")

    orig = h3._load_minimax_h3_modular
    h3._load_minimax_h3_modular = boom
    try:
        args = parse_args([
            "--dummy",
            "--encoder_device", "cuda:1",
            "--device", "cuda:0",
            "--steps", "0",
            "--no_sample",
        ])
        backend = MiniMaxH3Backend(
            device=args.device,
            encoder_device=args.encoder_device,
            dummy=True,
        )
        assert str(backend.device) == "cpu"
        assert str(backend.encoder_device) == "cpu"
        out = backend.generate_t2va("person", scale=1.0, seed=0)
        assert out["guidance"] == 0.0
        assert out["dummy"] is True
    finally:
        h3._load_minimax_h3_modular = orig
    assert called["n"] == 0


def test_load_h3_lora_rejects_non_h3_keys(tmp_path):
    from safetensors.torch import save_file

    path = tmp_path / "peft.safetensors"
    save_file({"base_model.model.foo.weight": torch.zeros(2, 2)}, str(path))
    backend = MiniMaxH3Backend(device="cpu", dummy=True)
    with pytest.raises(ValueError, match="lora_h3"):
        backend.load_trained(str(path))
