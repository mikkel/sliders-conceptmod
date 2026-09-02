"""LTX-2.5 UNI slider: CPU mocks only. No Hub, no GPU, no LTX weights."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
import yaml

from conceptmod.textsliders.ltx25_backend import (
    DEFAULT_ALPHA,
    DEFAULT_ENCODER_DEVICE,
    DEFAULT_LORA_UP_INIT_STD,
    DEFAULT_LR,
    DEFAULT_MODEL,
    DEFAULT_NUM_FRAMES,
    DEFAULT_RANK,
    DEFAULT_SAMPLE_HEIGHT,
    DEFAULT_SAMPLE_SCALES,
    DEFAULT_SAMPLE_SCALES_TEXT,
    DEFAULT_SAMPLE_WIDTH,
    DEFAULT_STEPS,
    DEFAULT_TRAIN_HEIGHT,
    DEFAULT_TRAIN_NUM_FRAMES,
    DEFAULT_TRAIN_WIDTH,
    DEFAULT_TRANSFORMER_SUBFOLDER,
    DISTILLED_MODALITY_SCALE,
    DISTILLED_SIGMA_VALUES,
    DEFAULT_DISTILLED_TRAIN_SIGMA,
    CONNECTOR_PAD_SEQ_LEN,
    CONNECTOR_REGISTER_MULTIPLE,
    DUMMY_TE_LAYERS,
    FREEZE_LIST,
    FULL_TRANSFORMER_SUBFOLDER,
    SFT_NUM_INFERENCE_STEPS,
    TIMESTEP_SCALE_MULTIPLIER,
    IGNORE_ON_FIRST_DOWNLOAD,
    LORA_ATTN_CLASS,
    LORA_EMBED_HOSTS,
    LORA_LINEAR_NAMES,
    LORA_TE_ATTN_NAMES,
    LORA_VIDEO_HOSTS,
    VIDEO_IN_CHANNELS,
    ArchitectureMismatch,
    AttnLoRANetwork,
    DummyLTX2Connectors,
    DummyLTX2Transformer,
    DummyTokenizer,
    LTX25Backend,
    distilled_sigmas,
    embed_lora_hosts,
    is_distilled_subfolder,
    is_te_attn_linear,
    is_video_attn_linear,
    is_video_connector_linear,
    pad_pre_connector_sequence,
    sample_pipe_kwargs,
    scaled_train_timestep,
    sft_scheduler_overrides,
    te_last_n_attn_targets,
    te_layer_indices,
    ltx_canvas_hw,
    ltx_num_frames,
    ltx_pack_feature_dim,
    place_ltx25_pipeline,
    resolve_ltx_lora_path,
    video_attn_lora_targets,
    video_connector_lora_targets,
)
from conceptmod.textsliders.ltx25_uni import (
    DEFAULT_EMBED_REL_L2_WEIGHT,
    DEFAULT_HOLD_MODE,
    DEFAULT_RECIPE,
    DEFAULT_TE_LAST_N,
    HOLD_MODE_ATTRIBUTES,
    HOLD_MODE_NON_CONCEPT,
    LTX25HoldError,
    RECIPE_EMBED,
    RECIPE_VELOCITY,
    apply_unused_hold,
    concept_token_ids,
    cosine_l2,
    embed_gap_energy_frac,
    expression_gap_is_dead,
    hold_effectiveness_metrics,
    is_embed_recipe,
    ltx25_embed_match_loss,
    ltx25_embed_mse,
    ltx25_embed_rel_l2,
    ltx25_minus_canary,
    ltx25_uni_total_loss,
    ltx25_uni_velocity_loss,
    pin_unused_attributes,
    post_connector_mean_cos,
    require_concept_tokens,
    resolve_concept_token_ids,
    resolve_hold_mode,
    resolve_ltx25_recipe,
    unused_hold_mask,
    unused_token_ids,
    valid_hidden_rows,
    velocity_pair,
)
from conceptmod.textsliders.diag_ltx25_uni import (
    COS_L2_KEYS,
    DIAG_ROW_KEYS,
    HOW_TO_READ,
    STUDENT_KEYS,
    diagnose_row,
    format_diag_table,
    parse_diag_args,
    run_diag,
)
from conceptmod.textsliders.train_lora_ltx25 import (
    load_slider_rows,
    main as train_main,
    parse_args,
    parse_sample_scales,
    sample_prompts_from_rows,
    train,
)


def test_resolved_model_id_is_ltx25_diffusers():
    assert DEFAULT_MODEL == "Lightricks/LTX-2.5-Diffusers"
    assert DEFAULT_TRANSFORMER_SUBFOLDER == "transformer"
    assert FULL_TRANSFORMER_SUBFOLDER == "transformer_full"
    args = parse_args(["--dummy"])
    assert args.model_id == "Lightricks/LTX-2.5-Diffusers"
    assert args.transformer_subfolder == "transformer"
    assert args.encoder_device == DEFAULT_ENCODER_DEVICE == "cpu"
    assert args.rank == DEFAULT_RANK == 16
    assert args.alpha == DEFAULT_ALPHA == 16.0
    assert args.lr == DEFAULT_LR == pytest.approx(2e-4)
    assert args.steps == DEFAULT_STEPS == 700
    assert args.recipe == DEFAULT_RECIPE == RECIPE_EMBED
    assert args.te_last_n == DEFAULT_TE_LAST_N == 4
    assert args.hold_mode == DEFAULT_HOLD_MODE == HOLD_MODE_NON_CONCEPT
    assert args.lora_up_init_std == DEFAULT_LORA_UP_INIT_STD == 0.02
    assert args.load_ltx_lora is None
    assert args.sample_scales == DEFAULT_SAMPLE_SCALES_TEXT == "-1,0,0.5,1"
    assert parse_sample_scales(args.sample_scales) == list(DEFAULT_SAMPLE_SCALES)
    assert args.sample_num_frames == DEFAULT_NUM_FRAMES == 49
    assert args.sample_height == DEFAULT_SAMPLE_HEIGHT == 544
    assert args.sample_width == DEFAULT_SAMPLE_WIDTH == 960
    assert args.diag is False
    assert resolve_ltx25_recipe(None) == RECIPE_EMBED
    assert is_embed_recipe("embed") is True
    assert is_embed_recipe("velocity") is False
    assert resolve_ltx25_recipe("ltx25_uni_velocity") == RECIPE_VELOCITY


def test_distilled_sigmas_match_current_diffusers_constant():
    assert DISTILLED_SIGMA_VALUES == [
        1.0, 0.99375, 0.9875, 0.98125, 0.975, 0.909375, 0.725, 0.421875,
    ]
    assert distilled_sigmas() == DISTILLED_SIGMA_VALUES
    assert "transformer_full/*" in IGNORE_ON_FIRST_DOWNLOAD


def test_video_only_lora_rejects_audio_and_a2v():
    assert is_video_attn_linear("transformer_blocks.0.attn1.to_q")
    assert is_video_attn_linear("transformer_blocks.0.attn2.to_out.0")
    assert not is_video_attn_linear("transformer_blocks.0.audio_attn1.to_q")
    assert not is_video_attn_linear("transformer_blocks.0.audio_attn2.to_k")
    assert not is_video_attn_linear("transformer_blocks.0.audio_to_video_attn.to_v")
    assert not is_video_attn_linear("transformer_blocks.0.video_to_audio_attn.to_q")
    assert not is_video_attn_linear("transformer_blocks.0.ff")
    # Naive "attn1" suffix / last-component endswith matches audio_attn1.
    # Official target_modules=["to_q"] also wraps audio / a2v — too broad.
    assert "transformer_blocks.0.audio_attn1".endswith("attn1")
    assert "transformer_blocks.0.audio_attn1.to_q".split(".")[-2].endswith("attn1")
    assert not is_video_attn_linear("transformer_blocks.0.audio_attn1.to_q")


def test_embed_lora_attaches_to_te_last_n_and_video_connectors():
    backend = LTX25Backend(device="cpu", dummy=True)
    assert backend.embed_recipe is True
    assert backend.recipe == RECIPE_EMBED
    names = backend.lora_module_names()
    assert names
    joined = " ".join(names)
    assert LORA_EMBED_HOSTS == ("video_connectors", "te_last_n")
    assert "te-" in joined or "te." in joined
    assert "conn-" in joined or "video" in joined
    assert "q_proj" in joined or "q-proj" in joined
    assert "audio_attn" not in joined
    assert "audio-attn" not in joined
    dit_names = video_attn_lora_targets(backend.transformer)
    assert dit_names  # DiT still *has* attn1/attn2, but they are not hosts
    assert not any(n in joined for n in ("attn1", "attn2"))
    params = backend.trainable_parameters()
    assert params
    enc_base = [p for p in backend.encoder.parameters() if p.requires_grad]
    assert enc_base == []
    dit_train = [p for p in backend.transformer.parameters() if p.requires_grad]
    assert dit_train == []
    te_targets = te_last_n_attn_targets(backend.encoder, last_n=DEFAULT_TE_LAST_N)
    assert te_targets
    indices = te_layer_indices(backend.encoder)
    assert indices == list(range(DUMMY_TE_LAYERS))
    keep = set(indices[-DEFAULT_TE_LAST_N:])
    for name in te_targets:
        from conceptmod.textsliders.ltx25_backend import te_layer_index
        assert te_layer_index(name) in keep
    conn_targets = video_connector_lora_targets(backend.connectors)
    assert conn_targets
    assert all("audio" not in t for t in conn_targets)
    hosts = embed_lora_hosts(backend.encoder, backend.connectors, last_n=4)
    assert any(n.startswith("te.") for n, _ in hosts)
    assert any(n.startswith("conn.") for n, _ in hosts)


def test_velocity_lora_attaches_to_video_attn1_attn2_only():
    backend = LTX25Backend(device="cpu", dummy=True, recipe=RECIPE_VELOCITY)
    names = backend.lora_module_names()
    assert names
    joined = " ".join(names)
    assert LORA_ATTN_CLASS == "LTX2Attention"
    assert LORA_VIDEO_HOSTS == ("attn1", "attn2")
    for piece in LORA_LINEAR_NAMES:
        token = piece.replace(".", "-")
        assert token in joined, (piece, names)
    assert "attn1" in joined and "attn2" in joined
    assert "audio_attn" not in joined
    assert "audio-to-video" not in joined and "audio_to_video" not in joined
    assert "video_to_audio" not in joined and "video-to-audio" not in joined
    assert "adaln" not in joined.lower()
    params = backend.trainable_parameters()
    assert params
    enc_ids = {id(p) for p in backend.encoder.parameters()}
    assert not enc_ids.intersection({id(p) for p in params})
    targets = video_attn_lora_targets(backend.transformer)
    assert all("attn1" in t or "attn2" in t for t in targets)
    assert not any("audio" in t for t in targets)


def test_does_not_fake_predict_v():
    backend = LTX25Backend(device="cpu", dummy=True)
    with pytest.raises(ArchitectureMismatch, match="flow velocity"):
        backend.predict_v("person", torch.zeros(1, 2, 128), torch.tensor([1.0]))
    plus = backend.encode_text("a person smiling")
    packed = backend.pack_t2v(plus)
    assert packed.hidden_states.shape[-1] == VIDEO_IN_CHANNELS
    out = backend.forward_velocity(packed, scale=0.0)
    assert out.sample.shape[0] == 1
    assert out.audio_sample.shape[0] == 1


def test_freeze_list_includes_connectors_and_vaes():
    assert "text_encoder" in FREEZE_LIST
    assert "connectors" in FREEZE_LIST
    assert "vae" in FREEZE_LIST
    assert "audio_vae" in FREEZE_LIST
    assert "vocoder" in FREEZE_LIST
    backend = LTX25Backend(device="cpu", dummy=True)
    assert all(not p.requires_grad for p in backend.encoder.parameters())
    assert all(not p.requires_grad for p in backend.connectors.parameters())
    assert all(not p.requires_grad for p in backend.vae.parameters())


def test_hold_is_pre_connector_not_post():
    """Token-id hold exists only BEFORE connectors add registers.

    Dummy connectors mix the sequence and replace pad with registers
    (T' != T). If pack applied hold *after* connectors, ``pre_connector_hidden``
    would not pin ``encode(neu)`` and connectors-then-hold would match
    the transformer-facing features.
    """
    backend = LTX25Backend(device="cpu", dummy=True)
    tok = backend.tokenizer
    pos, neu = "male person smiling teeth", "male person closed mouth"
    plus_enc = backend.encode_text(pos)
    neu_enc = backend.encode_text(neu)
    assert plus_enc.hold_stage == "pre_connector"
    assert plus_enc.embeds.shape[1] == len(plus_enc.token_ids)
    concept = concept_token_ids(tok, pos, neu)
    unused = unused_token_ids(tok, ["male", "female"])
    hold_mask = unused_hold_mask(plus_enc.token_ids, unused, concept)
    packed = backend.pack_t2v(plus_enc, hold_neu=neu_enc, hold_mask=hold_mask)
    assert packed.hold_stage == "pre_connector"
    assert packed.n_prompt_tokens == len(plus_enc.token_ids)
    # Registers replace padding: connector T is not 1:1 with prompt tokens.
    assert packed.encoder_hidden_states.shape[1] != packed.n_prompt_tokens
    pre = hold_effectiveness_metrics(
        plus_enc.embeds, neu_enc.embeds, plus_enc.token_ids, neu_enc.token_ids,
        hold_mask, concept_ids=concept, held_hidden=packed.pre_connector_hidden,
    )
    assert pre["held_mean_abs"] == pytest.approx(0.0, abs=1e-5)
    # Wrong order: pad + connectors first, then token-id hold. Must not
    # match hold-then-pad-then-connectors (the pack path).
    plus_pad, plus_mask = pad_pre_connector_sequence(plus_enc.embeds, plus_enc.attention_mask)
    neu_pad, neu_mask = pad_pre_connector_sequence(neu_enc.embeds, neu_enc.attention_mask)
    plus_post, _, _ = backend._run_connectors(plus_pad, plus_mask)
    neu_post, _, _ = backend._run_connectors(neu_pad, neu_mask)
    wrong = apply_unused_hold(
        plus_post, neu_post, plus_enc.token_ids, neu_enc.token_ids, hold_mask,
    )
    assert wrong.shape == packed.encoder_hidden_states.shape
    assert not torch.allclose(packed.encoder_hidden_states, wrong, atol=1e-5)
    # Post-connector rows are not encode(neu) token rows.
    post_vs_pre = hold_effectiveness_metrics(
        plus_enc.embeds, neu_enc.embeds, plus_enc.token_ids, neu_enc.token_ids,
        hold_mask, concept_ids=concept,
        held_hidden=wrong[:, : plus_enc.embeds.shape[1]],
    )
    assert post_vs_pre["held_mean_abs"] > 0.0


def test_pin_unused_attributes_bare_by_default():
    rows = pin_unused_attributes(
        "a person smiling", "a person", ["male", "female"],
    )
    assert rows == [("a person smiling", "a person")]
    prefixed = pin_unused_attributes(
        "a person smiling", "a person", ["male", "female"], bare_captions=False,
    )
    assert prefixed == [
        ("male a person smiling", "male a person"),
        ("female a person smiling", "female a person"),
    ]


def test_fail_closed_when_plus_has_no_concept_tokens():
    tok = DummyTokenizer()
    with pytest.raises(LTX25HoldError, match="no concept-word tokens"):
        require_concept_tokens(tok.encode("a person"), set())
    with pytest.raises(LTX25HoldError, match="no concept-word tokens"):
        resolve_concept_token_ids(tok, "a person", "a person", "")


def test_non_concept_hold_pins_shared_subject_not_smile():
    tok = DummyTokenizer()
    pos = "male person sitting shirt, smiling teeth"
    neu = "male person sitting shirt, closed mouth"
    concept = concept_token_ids(tok, pos, neu)
    unused = unused_token_ids(tok, ["male", "female"])
    plus_ids = tok.encode(pos)
    mask = unused_hold_mask(plus_ids, unused, concept)
    assert resolve_hold_mode(None) == HOLD_MODE_NON_CONCEPT
    words = pos.split()
    held = {words[i] for i, flag in enumerate(mask.tolist()) if flag}
    free = {words[i] for i, flag in enumerate(mask.tolist()) if not flag}
    assert "male" in held
    assert "person" in held
    assert "smiling" in free
    assert "teeth" in free


def test_uni_velocity_has_no_minus_teacher():
    plus = torch.ones(2, 4)
    zero = torch.zeros(2, 4)
    loss = ltx25_uni_velocity_loss(plus, plus, zero, zero)
    assert float(loss.item()) == pytest.approx(0.0)
    canary = ltx25_minus_canary(-plus, zero)
    total = ltx25_uni_total_loss(plus, plus, zero, zero)
    assert float(total.item()) != pytest.approx(float((total + canary).item()))


def test_smile_yaml_same_subject_bare_captions():
    path = Path("conceptmod/textsliders/data/prompts-ltx25-smile.yaml")
    raw = yaml.safe_load(path.read_text())
    assert raw["bare_captions"] is True
    assert raw["concept_words"] == "smiling, smile, happy, joyful, teeth"
    assert "chiaroscuro" not in path.read_text().lower()
    rows = raw["rows"]
    assert len(rows) == 2
    for item in rows:
        assert item["attributes"] == ["male", "female"]
        assert "closed mouth" in item["neutral"]
        assert "smile" in item["positive"].lower() or "teeth" in item["positive"].lower()
        assert "showing teeth" in item["positive"]
        assert "happy joyful" in item["positive"]
        assert "denim" in item["target"] or "navy knit" in item["target"]
        # Same motion / light / sound on both poles.
        for lock in ("camera remains static", "quiet room tone", "faint clock tick", "no speech"):
            assert lock in item["positive"]
            assert lock in item["neutral"]
        assert not item["positive"].startswith("male ")
        assert not item["neutral"].startswith("male ")
    loaded = load_slider_rows(str(path), "")
    assert len(loaded) == 2
    for row in loaded:
        assert not row["positive"].startswith("male ")
        concept = resolve_concept_token_ids(
            DummyTokenizer(), row["positive"], row["neutral"], row["concept_words"],
        )
        assert concept


def test_dummy_train_writes_sidecar_and_student_is_neu(tmp_path):
    prompts = tmp_path / "one.yaml"
    prompts.write_text(
        "- target: a person, closed mouth\n"
        "  positive: a person smiling showing teeth, happy joyful expression\n"
        "  neutral: a person, closed mouth\n"
        "  unconditional: ''\n"
        "  attributes: []\n"
        "  concept_words: smiling, smile, happy, joyful, teeth\n"
    )
    args = parse_args([
        "--dummy", "--steps", "20", "--name", "ltx25-dummy",
        "--save_dir", str(tmp_path), "--prompts_file", str(prompts),
        "--lr", "0.5", "--seed", "0", "--no_sample",
    ])
    sidecar = train(args)
    assert sidecar["model_id"] == "Lightricks/LTX-2.5-Diffusers"
    assert sidecar["transformer_subfolder"] == "transformer"
    assert sidecar["student_plus"] == "neu"
    assert sidecar["hold_stage"] == "pre_connector"
    assert sidecar["recipe"] == RECIPE_EMBED
    assert sidecar["dit_frozen"] is True
    assert sidecar["train_dit"] is False
    assert sidecar["te_last_n"] == 4
    assert sidecar["minus_teacher"] is False
    assert sidecar["lora_host"] == "te_last_n+video_connectors"
    assert sidecar["lora_hosts"] == list(LORA_EMBED_HOSTS)
    assert sidecar["lora_linears"] == list(LORA_TE_ATTN_NAMES)
    assert sidecar["train_audio_attn"] is False
    assert sidecar["velocity_contract"] is None
    assert sidecar["predict_v_faked"] is False
    assert sidecar["teacher"] == "encode_plus_post_connector_video"
    assert sidecar["guidance"] == 1.0
    assert sidecar["stg_scale"] == 0.0
    assert sidecar["modality_scale"] == DISTILLED_MODALITY_SCALE == 1.0
    assert sidecar["prompt_enhancer"] is False
    assert sidecar["decoder"] == "conv_vae"
    assert sidecar["sigmas"] == DISTILLED_SIGMA_VALUES
    assert sidecar["lora_up_init_std"] == 0.02
    assert sidecar["first_loss"] > sidecar["last_loss"]
    data = json.loads((tmp_path / "ltx25-dummy_last.json").read_text())
    assert data["backend"] == "ltx25"


def test_scale_zero_matches_neu_teacher():
    backend = LTX25Backend(device="cpu", dummy=True)
    neu = backend.encode_text("a person, closed mouth")
    packed = backend.pack_t2v(neu)
    teacher = backend.forward_velocity(packed, scale=0.0)
    student = backend.forward_velocity(packed, scale=0.0)
    assert torch.allclose(teacher.sample, student.sample)


def test_plus_and_neu_velocities_differ():
    backend = LTX25Backend(device="cpu", dummy=True)
    plus = backend.pack_t2v(backend.encode_text("a person smiling showing teeth"))
    neu = backend.pack_t2v(
        backend.encode_text("a person, closed mouth"),
        video_latents=plus.hidden_states,
        audio_latents=plus.audio_hidden_states,
    )
    v_plus = backend.forward_velocity(plus, scale=0.0)
    v_neu = backend.forward_velocity(neu, scale=0.0)
    assert not torch.allclose(v_plus.sample, v_neu.sample)
    gap = cosine_l2(
        velocity_pair(v_plus.sample, v_plus.audio_sample),
        velocity_pair(v_neu.sample, v_neu.audio_sample),
    )
    assert not expression_gap_is_dead(gap)


def test_music3_defaults_unchanged():
    tf_src = Path("conceptmod/textsliders/train_lora_music3.py").read_text()
    assert 'parser.add_argument("--steps", type=int, default=500)' in tf_src
    assert 'parser.add_argument("--rank", type=int, default=8)' in tf_src
    lm_src = Path("conceptmod/textsliders/train_lm_slider_music3.py").read_text()
    assert '"--lm_target"' in lm_src and 'default="v9"' in lm_src
    assert '"--pole_mode"' in lm_src and 'default="hidden"' in lm_src
    music3_yaml = Path("conceptmod/textsliders/data/prompts-music3.yaml").read_text()
    assert "LTX-2.5" not in music3_yaml
    assert "MiniMax-H3" not in Path("conceptmod/textsliders/data/prompts-ltx25-smile.yaml").read_text()


def test_config_points_at_ltx25_distilled():
    cfg = yaml.safe_load(Path("conceptmod/textsliders/data/config-ltx25.yaml").read_text())
    assert cfg["pretrained_model"]["name_or_path"] == "Lightricks/LTX-2.5-Diffusers"
    assert cfg["pretrained_model"]["transformer_subfolder"] == "transformer"
    assert cfg["train"]["recipe"] == RECIPE_EMBED
    assert cfg["train"]["guidance"] == 1.0
    assert cfg["train"]["stg_scale"] == 0
    assert cfg["train"]["modality_scale"] == DISTILLED_MODALITY_SCALE == 1.0
    assert cfg["train"]["sample_num_frames"] == 49
    assert cfg["train"]["sample_height"] == 544
    assert cfg["train"]["sample_width"] == 960
    assert cfg["train"]["sample_scales"] == [-1.0, 0.0, 0.5, 1.0]
    assert cfg["train"]["hold_stage"] == "pre_connector"
    assert cfg["train"]["hold_mode"] == "non_concept"
    assert cfg["train"]["student_plus"] == "neu"
    assert cfg["train"]["iterations"] == 700
    assert cfg["train"]["lr"] == 0.0002
    assert cfg["network"]["target"] == "te_last_n+video_connectors"
    assert cfg["network"]["hosts"] == ["video_connectors", "te_last_n"]
    assert cfg["network"]["rank"] == 16
    assert cfg["network"]["te_last_n"] == 4
    assert cfg["network"]["train_dit"] is False
    assert cfg["network"]["lora_up_init_std"] == 0.02
    assert cfg["train"]["encoder_device"] == "cuda:1"
    assert cfg["train"]["device"] == "cuda:0"


def test_live_load_is_not_imported_on_dummy():
    import conceptmod.textsliders.ltx25_backend as ltx

    called = {"n": 0}

    def boom(*_a, **_k):
        called["n"] += 1
        raise AssertionError("live LTX-2.5 loader must not run in dummy mode")

    orig = ltx._load_ltx25_pipeline
    ltx._load_ltx25_pipeline = boom
    try:
        backend = LTX25Backend(device="cpu", dummy=True)
        _ = backend.encode_text("a person")
        _ = backend.forward_velocity(backend.pack_t2v(backend.encode_text("a person")), scale=1.0)
        _ = backend.generate_t2v("a person, closed mouth", scale=0.0, seed=0)
    finally:
        ltx._load_ltx25_pipeline = orig
    assert called["n"] == 0


def test_sample_geometry_defaults():
    assert ltx_num_frames(49) == 49
    assert ltx_num_frames(33) == 33
    assert ltx_num_frames(48) == 49
    assert DEFAULT_NUM_FRAMES % 8 == 1
    assert DEFAULT_TRAIN_NUM_FRAMES == 9
    assert DEFAULT_TRAIN_HEIGHT == DEFAULT_TRAIN_WIDTH == 32
    h, w = ltx_canvas_hw(544, 960)
    assert h == 544 and w == 960
    assert h % 32 == 0 and w % 32 == 0


def test_pack_dim_uses_proj_in_in_features():
    backend = LTX25Backend(device="cpu", dummy=True)
    dummy = backend.transformer
    assert dummy.proj_in.in_features == VIDEO_IN_CHANNELS == 128
    assert ltx_pack_feature_dim(dummy, kind="video") == 128
    packed = backend.pack_t2v(backend.encode_text("a person"))
    assert packed.hidden_states.shape[-1] == 128
    assert packed.audio_hidden_states.shape[-1] == dummy.audio_proj_in.in_features


def test_dummy_train_writes_mp4s_at_minus_one_to_one(tmp_path):
    prompts = tmp_path / "one.yaml"
    prompts.write_text(
        "- target: a person sitting in a chair, closed mouth\n"
        "  positive: a person sitting in a chair, big smile showing teeth\n"
        "  neutral: a person sitting in a chair, closed mouth\n"
        "  unconditional: ''\n  attributes: []\n"
    )
    args = parse_args([
        "--dummy", "--steps", "2", "--name", "ltx-sample",
        "--save_dir", str(tmp_path), "--prompts_file", str(prompts),
        "--sample_scales=-1,0,0.5,1", "--seed", "0",
    ])
    sidecar = train(args)
    assert sidecar["sample_grid"]["scales"] == [-1.0, 0.0, 0.5, 1.0]
    assert sidecar["sample_grid"]["guidance"] == 1.0
    assert sidecar["sample_grid"]["infer_caption"] == "neu"
    assert "num_inference_steps" not in sidecar["sample_grid"]
    samples = tmp_path / "samples"
    meta = json.loads((samples / "final_meta.json").read_text())
    assert meta["sigmas"] == DISTILLED_SIGMA_VALUES
    assert meta["prompt_enhancer"] is False
    mp4s = sorted(p.name for p in samples.glob("*.mp4"))
    assert any("scalem1.mp4" in n for n in mp4s)
    assert any("scale0.mp4" in n for n in mp4s)
    assert any("scale0.5.mp4" in n for n in mp4s)
    assert any("scale1.mp4" in n for n in mp4s)
    for path in samples.glob("*.mp4"):
        assert path.stat().st_size > 0
        assert b"ftyp" in path.read_bytes()[:32]


def test_load_ltx_lora_roundtrip_and_steps_zero(tmp_path):
    prompts = tmp_path / "one.yaml"
    prompts.write_text(
        "- target: a person, closed mouth\n"
        "  positive: a person smiling showing teeth\n"
        "  neutral: a person, closed mouth\n"
        "  unconditional: ''\n  attributes: []\n"
    )
    train_dir = tmp_path / "trained"
    train(parse_args([
        "--dummy", "--steps", "4", "--name", "ltx-load",
        "--save_dir", str(train_dir), "--prompts_file", str(prompts),
        "--no_sample", "--seed", "1",
    ]))
    lora_path = resolve_ltx_lora_path(str(train_dir))
    src = LTX25Backend(device="cpu", dummy=True)
    src.load_trained(str(train_dir))
    dst = LTX25Backend(device="cpu", dummy=True)
    dst.load_trained(str(lora_path))
    for a, b in zip(src.network.parameters(), dst.network.parameters()):
        assert torch.allclose(a, b)
    sample_dir = tmp_path / "reload"
    sidecar = train(parse_args([
        "--dummy", "--steps", "0", "--name", "ltx-reload",
        "--save_dir", str(sample_dir), "--prompts_file", str(prompts),
        "--load_ltx_lora", str(train_dir), "--sample_scales=-1,0,0.5,1",
        "--encoder_device", "cpu", "--device", "cuda:0",
    ]))
    assert sidecar["steps"] == 0
    assert sidecar["load_ltx_lora"]
    assert sidecar["encoder_device"] is None
    mp4s = {p.name for p in (sample_dir / "samples").glob("*.mp4")}
    assert any("scale0.5" in n for n in mp4s)


def test_zero_init_lora_up_is_uni_identity():
    transformer = DummyLTX2Transformer()
    network = AttnLoRANetwork(transformer, rank=4, alpha=4.0, up_init_std=0.0)
    for lora in network.loras:
        assert torch.count_nonzero(lora.lora_up.weight).item() == 0
    backend = LTX25Backend(device="cpu", dummy=True, lora_up_init_std=0.0)
    packed = backend.pack_t2v(backend.encode_text("a person smiling"))
    pred = backend.forward_velocity(packed, scale=1.0)
    tgt = backend.forward_velocity(packed, scale=0.0)
    gap = float(torch.mean((pred.sample - tgt.sample) ** 2).item())
    assert gap < 1e-10


def test_noisy_lora_up_gives_nonzero_embed_gap():
    torch.manual_seed(0)
    backend = LTX25Backend(
        device="cpu", dummy=True, lora_up_init_std=DEFAULT_LORA_UP_INIT_STD,
    )
    neu = backend.encode_text("a person, closed mouth")
    off, off_mask = backend.encode_post_connector_video(neu, scale=0.0)
    backend.network.set_lora_slider(1.0)
    neu_on = backend.encode_text("a person, closed mouth", frozen=False)
    on, on_mask = backend.encode_post_connector_video(neu_on, scale=1.0)
    gap = float(ltx25_embed_mse(on, off, on_mask, off_mask).item())
    assert gap > 1e-8


def test_velocity_recipe_noisy_lora_moves_dit():
    torch.manual_seed(0)
    backend = LTX25Backend(
        device="cpu", dummy=True, recipe=RECIPE_VELOCITY,
        lora_up_init_std=DEFAULT_LORA_UP_INIT_STD,
    )
    packed = backend.pack_t2v(backend.encode_text("a person smiling"))
    pred = backend.forward_velocity(packed, scale=1.0)
    tgt = backend.forward_velocity(packed, scale=0.0)
    gap = float(torch.mean((pred.sample - tgt.sample) ** 2).item())
    assert gap > 1e-8


def test_place_pipeline_never_blankets_to():
    class _Mod:
        def __init__(self) -> None:
            self.device = None

        def to(self, device):
            self.device = str(device)
            return self

    class _Pipe:
        def __init__(self) -> None:
            self.transformer = _Mod()
            self.vae = _Mod()
            self.text_encoder = _Mod()
            self.connectors = _Mod()
            self.to_calls: list[str] = []

        def to(self, device):
            self.to_calls.append(str(device))
            return self

    pipe = _Pipe()
    place_ltx25_pipeline(pipe, device="cuda:0", encoder_device="cpu")
    assert pipe.to_calls == []
    assert pipe.transformer.device == "cuda:0"
    assert pipe.text_encoder.device == "cpu"
    assert pipe.connectors.device == "cpu"


def test_docs_card_has_live_flags():
    docs = Path("docs/ltx25-slider.md").read_text()
    assert "Lightricks/LTX-2.5-Diffusers" in docs
    assert "transformer/" in docs
    assert "transformer_full/" in docs
    assert "DISTILLED_SIGMA_VALUES" in docs
    assert "num_inference_steps" in docs
    assert "--encoder_device cpu" in docs
    assert "A6000" in docs
    assert "cuda:1" in docs
    assert "embed-match" in docs.lower() or "ltx25_uni_embed" in docs
    assert "0.9999" in docs or "dead teacher" in docs.lower()
    assert "0.68" in docs
    assert "WEAK" in docs
    assert "-1,0,0.5,1" in docs or "−1, 0, 0.5, 1" in docs
    assert "544" in docs and "960" in docs and "49" in docs
    assert "pre-connector" in docs.lower() or "PRE-connector" in docs
    assert "gemma4_unified" in docs
    assert "prompts-ltx25-smile.yaml" in docs
    assert "prompts-ltx25-chiaroscuro.yaml" in docs
    assert "--load_ltx_lora" in docs
    assert "--diag" in docs
    assert "--te_last_n" in docs
    assert "--sample_scales=-1,0,0.5,1" in docs
    assert "velocity" in docs.lower()
    assert "Do **not** reintroduce DiT velocity-UNI" in docs or "not the smile/chiaro default" in docs.lower()


def _assert_diag_row(row: dict) -> None:
    for key in DIAG_ROW_KEYS:
        assert key in row, key
    assert set(row["expression_gap"]) == set(COS_L2_KEYS)
    assert "lighting_gap" not in row
    assert set(row["student"]) == set(STUDENT_KEYS)
    assert row["hold_stage"] == "pre_connector"


def test_dummy_diag_expression_gap_not_lighting(tmp_path):
    args = parse_diag_args([
        "--dummy", "--name", "ltx-diag", "--save_dir", str(tmp_path),
        "--prompts_file", "conceptmod/textsliders/data/prompts-ltx25-smile.yaml",
        "--seed", "0",
    ])
    summary = run_diag(args)
    assert summary["dummy"] is True
    assert summary["recipe"] == "ltx25_uni_diag"
    assert summary["hold_stage"] == "pre_connector"
    assert "expression_gap" in summary["how_to_read"]
    assert "lighting_gap" not in summary["how_to_read"]
    assert set(HOW_TO_READ) <= set(summary["how_to_read"])
    for row in summary["rows"]:
        _assert_diag_row(row)
        assert row["hold"]["held_mean_abs"] == pytest.approx(0.0, abs=1e-5)
        assert row["n_concept"] > 0
        assert row["n_connector_tokens"] != row["n_prompt_tokens"]
        assert row["expression_gap"]["l2"] > 0
        assert row["student"]["scale0_vs_teacher_neu"]["cos"] == pytest.approx(1.0, abs=1e-5)
    table = format_diag_table(summary)
    assert "expr_cos" in table
    via = train_main([
        "--dummy", "--diag", "--name", "ltx-diag-flag",
        "--save_dir", str(tmp_path / "via-train"),
        "--prompts_file", "conceptmod/textsliders/data/prompts-ltx25-smile.yaml",
        "--seed", "2",
    ])
    assert via["recipe"] == "ltx25_uni_diag"
    assert (tmp_path / "via-train" / "ltx-diag-flag_diag.json").is_file()


def test_connector_pad_enforces_register_multiple():
    """Live connectors require seq_len % 128 == 0. Train encode is raw T.

    Dummy connectors must raise on unpadded prompt length so CI sees the
    first live pack_t2v crash (ValueError) if we regress.
    """
    backend = LTX25Backend(device="cpu", dummy=True)
    enc = backend.encode_text("a person sitting in a chair, closed mouth")
    raw_t = enc.embeds.shape[1]
    assert raw_t < CONNECTOR_REGISTER_MULTIPLE
    assert raw_t % CONNECTOR_REGISTER_MULTIPLE != 0
    with pytest.raises(ValueError, match="learnable registers"):
        backend.connectors(enc.embeds, enc.attention_mask)
    with pytest.raises(ValueError, match="learnable registers"):
        DummyLTX2Connectors()(enc.embeds, enc.attention_mask)

    packed = backend.pack_t2v(enc)
    assert packed.n_prompt_tokens == raw_t
    assert packed.pre_connector_hidden.shape[1] == raw_t
    padded, pad_mask = pad_pre_connector_sequence(enc.embeds, enc.attention_mask)
    assert padded.shape[1] == CONNECTOR_PAD_SEQ_LEN
    assert padded.shape[1] % CONNECTOR_REGISTER_MULTIPLE == 0
    assert int(pad_mask.sum().item()) == raw_t
    # Left pad: real token rows stay on the right; hold mask is not extended.
    assert torch.equal(padded[:, -raw_t:], enc.embeds)
    assert torch.equal(pad_mask[:, -raw_t:], enc.attention_mask)
    assert torch.count_nonzero(pad_mask[:, :-raw_t]).item() == 0
    video_e, _, _ = backend._run_connectors(padded, pad_mask)
    assert video_e.shape[1] != raw_t
    assert packed.encoder_hidden_states.shape[1] == video_e.shape[1]


def test_train_timestep_is_scaled_distilled_sigma():
    """Live timestep is (B, num_video_tokens) already * 1000.

    Unscaled 0.5 is t≈0. Dummy time embed must reject that so CI sees
    the collapsed plus/neu velocity crash if we regress.
    """
    assert DEFAULT_DISTILLED_TRAIN_SIGMA in DISTILLED_SIGMA_VALUES
    assert DEFAULT_DISTILLED_TRAIN_SIGMA != 0.5
    backend = LTX25Backend(device="cpu", dummy=True)
    packed = backend.pack_t2v(backend.encode_text("a person smiling showing teeth"))
    expected = DEFAULT_DISTILLED_TRAIN_SIGMA * TIMESTEP_SCALE_MULTIPLIER
    assert packed.timestep.shape == packed.hidden_states.shape[:2]
    assert packed.audio_timestep.shape == packed.audio_hidden_states.shape[:2]
    assert float(packed.timestep.amax()) == pytest.approx(expected)
    assert float(packed.timestep.amin()) == pytest.approx(expected)
    assert expected > 1.0
    scaled = scaled_train_timestep(1, packed.hidden_states.shape[1], distilled=True)
    assert torch.allclose(packed.timestep.cpu(), scaled)

    unscaled = torch.full(packed.hidden_states.shape[:2], 0.5)
    with pytest.raises(ValueError, match="unscaled"):
        backend.transformer(
            packed.hidden_states,
            packed.audio_hidden_states,
            packed.encoder_hidden_states,
            packed.audio_encoder_hidden_states,
            timestep=unscaled,
            audio_num_frames=packed.audio_hidden_states.shape[1],
        )
    with pytest.raises(ValueError, match="num_video_tokens"):
        backend.transformer(
            packed.hidden_states,
            packed.audio_hidden_states,
            packed.encoder_hidden_states,
            packed.audio_encoder_hidden_states,
            timestep=torch.tensor([expected]),
            audio_num_frames=packed.audio_hidden_states.shape[1],
        )


def test_forward_velocity_requires_audio_num_frames():
    """Live RoPE prepare_audio_coords(None) is a TypeError. Dummy must not swallow it."""
    backend = LTX25Backend(device="cpu", dummy=True)
    packed = backend.pack_t2v(backend.encode_text("a person, closed mouth"))
    with pytest.raises(TypeError, match="audio_num_frames|num_frames"):
        backend.transformer(
            packed.hidden_states,
            packed.audio_hidden_states,
            packed.encoder_hidden_states,
            packed.audio_encoder_hidden_states,
            timestep=packed.timestep,
        )
    out = backend.forward_velocity(packed, scale=0.0)
    assert out.sample.shape[0] == 1
    assert out.audio_sample.shape[0] == 1


def test_sft_sample_restores_shifting_and_drops_distilled_sigmas():
    assert sft_scheduler_overrides() == {
        "use_dynamic_shifting": True, "shift_terminal": 0.1,
    }
    assert is_distilled_subfolder(DEFAULT_TRANSFORMER_SUBFOLDER)
    assert not is_distilled_subfolder(FULL_TRANSFORMER_SUBFOLDER)
    distilled = sample_pipe_kwargs(distilled=True)
    assert distilled["sigmas"] == DISTILLED_SIGMA_VALUES
    assert distilled["modality_scale"] == DISTILLED_MODALITY_SCALE == 1.0
    assert "num_inference_steps" not in distilled
    sft = sample_pipe_kwargs(distilled=False)
    assert "sigmas" not in sft
    assert sft["num_inference_steps"] == SFT_NUM_INFERENCE_STEPS
    assert sft["modality_scale"] == DISTILLED_MODALITY_SCALE


def test_embed_match_loss_shapes_and_valid_mask():
    pred = torch.zeros(1, 4, 8)
    tgt = torch.zeros(1, 4, 8)
    pred[:, :3] = 1.0
    tgt[:, :3] = 1.0
    mask = torch.tensor([[1, 1, 1, 0]])
    loss = ltx25_embed_match_loss(pred, tgt, mask, mask, rel_l2_weight=1.0)
    assert loss.shape == ()
    assert float(loss.item()) == pytest.approx(0.0, abs=1e-6)
    # Pad row differs but is masked out.
    pred_pad = pred.clone()
    pred_pad[:, 3] = 9.0
    assert float(ltx25_embed_match_loss(pred_pad, tgt, mask, mask).item()) == pytest.approx(
        0.0, abs=1e-6,
    )
    miss = tgt + 0.5
    mse = float(ltx25_embed_mse(miss, tgt, mask, mask).item())
    rel = float(ltx25_embed_rel_l2(miss, tgt, mask, mask).item())
    combo = float(ltx25_embed_match_loss(miss, tgt, mask, mask, rel_l2_weight=1.0).item())
    assert mse > 0 and rel > 0
    assert combo == pytest.approx(mse + rel, rel=1e-5)
    # Different T → mean-pool path, still a scalar.
    short = torch.ones(1, 2, 8)
    long = torch.ones(1, 5, 8)
    pooled = ltx25_embed_match_loss(short, long)
    assert pooled.shape == ()
    rows, weight = valid_hidden_rows(pred, mask)
    assert rows.shape == (1, 4, 8)
    assert int(weight.sum().item()) == 3
    gap = post_connector_mean_cos(pred, tgt, mask, mask)
    assert "cos" in gap and "l2" in gap
    assert gap["cos"] == pytest.approx(1.0, abs=1e-5)


def test_embed_match_cli_dummy_and_velocity_opt_in(tmp_path):
    prompts = tmp_path / "one.yaml"
    prompts.write_text(
        "- target: a person, closed mouth\n"
        "  positive: a person smiling showing teeth, happy joyful expression\n"
        "  neutral: a person, closed mouth\n"
        "  unconditional: ''\n"
        "  attributes: []\n"
        "  concept_words: smiling, smile, happy, joyful, teeth\n"
    )
    embed = train(parse_args([
        "--dummy", "--steps", "8", "--name", "ltx-embed",
        "--save_dir", str(tmp_path / "embed"), "--prompts_file", str(prompts),
        "--lr", "0.5", "--seed", "0", "--no_sample",
    ]))
    assert embed["recipe"] == RECIPE_EMBED
    assert embed["dit_frozen"] is True
    assert embed["first_loss"] > embed["last_loss"]
    vel = train(parse_args([
        "--dummy", "--steps", "8", "--name", "ltx-vel",
        "--recipe", "ltx25_uni_velocity",
        "--save_dir", str(tmp_path / "vel"), "--prompts_file", str(prompts),
        "--lr", "0.5", "--seed", "0", "--no_sample",
    ]))
    assert vel["recipe"] == RECIPE_VELOCITY
    assert vel["dit_frozen"] is False
    assert vel["train_dit"] is True
    assert vel["lora_hosts"] == ["attn1", "attn2"]
    assert vel["first_loss"] > vel["last_loss"]


def test_chiaroscuro_yaml_locked_non_concept_structure():
    path = Path("conceptmod/textsliders/data/prompts-ltx25-chiaroscuro.yaml")
    raw = yaml.safe_load(path.read_text())
    assert raw["bare_captions"] is True
    assert "chiaroscuro" in raw["concept_words"].lower()
    assert "rembrandt" in raw["concept_words"].lower()
    rows = raw["rows"]
    assert len(rows) == 2
    for item in rows:
        assert item["attributes"] == ["male", "female"]
        assert "soft even window light" in item["neutral"].lower()
        assert "chiaroscuro" in item["positive"].lower()
        assert "rembrandt" in item["positive"].lower()
        for lock in ("quiet room tone", "faint clock tick", "no speech", "camera remains static"):
            assert lock in item["positive"]
            assert lock in item["neutral"]
        assert not item["positive"].startswith("male ")
        assert not item["neutral"].startswith("male ")
    loaded = load_slider_rows(str(path), "")
    assert len(loaded) == 2
    for row in loaded:
        concept = resolve_concept_token_ids(
            DummyTokenizer(), row["positive"], row["neutral"], row["concept_words"],
        )
        assert concept
    cfg = yaml.safe_load(Path("conceptmod/textsliders/data/config-ltx25-chiaroscuro.yaml").read_text())
    assert cfg["train"]["recipe"] == RECIPE_EMBED
    assert cfg["network"]["train_dit"] is False
    assert cfg["train"]["sample_scales"] == [-1.0, 0.0, 0.5, 1.0]


def test_te_attn_selector_skips_early_layers_and_audio_connectors():
    backend = LTX25Backend(device="cpu", dummy=True)
    last = te_last_n_attn_targets(backend.encoder, last_n=4)
    assert last
    from conceptmod.textsliders.ltx25_backend import te_layer_index
    assert {te_layer_index(n) for n in last} == {2, 3, 4, 5}
    assert is_te_attn_linear("layers.5.self_attn.q_proj", last_n=4, layer_indices=list(range(6)))
    assert not is_te_attn_linear("layers.0.self_attn.q_proj", last_n=4, layer_indices=list(range(6)))
    assert is_video_connector_linear("video_attn.to_q")
    assert is_video_connector_linear("video_mix")
    assert not is_video_connector_linear("audio_mix")
    assert not is_video_connector_linear("audio_attn.to_q")
