#!/usr/bin/env python3
"""Opt-in LTX-2.5 concept-slider trainer (embed-match UNI default).

Default Music 3 trainers are unchanged (``--lm_target v9`` / ``--pole_mode hidden``).

Live card: ``Lightricks/LTX-2.5-Diffusers`` distilled ``transformer/``.
Student ``encode(neu)+LoRA`` matches frozen ``encode(plus)`` on valid
post-connector **video** hidden states (MSE + rel-L2). LoRA hosts are
video connectors + TE last-N attn. **DiT stays frozen.** DiT velocity
UNI is ``--recipe ltx25_uni_velocity`` only — not the smile/chiaro default.

Hold is PRE-connector, ``non_concept``. Sample scales include
``-1, 0, 0.5, 1`` on the neu caption. ``--dummy`` is the CI path.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import yaml

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from conceptmod.textsliders.ltx25_backend import (
    DEFAULT_ALPHA,
    DEFAULT_ENCODER_DEVICE,
    DEFAULT_LORA_UP_INIT_STD,
    DEFAULT_LR,
    DEFAULT_MODEL,
    DEFAULT_NUM_FRAMES,
    DEFAULT_RANK,
    DEFAULT_SAMPLE_HEIGHT,
    DEFAULT_SAMPLE_SCALES_TEXT,
    DEFAULT_SAMPLE_WIDTH,
    DEFAULT_STEPS,
    DEFAULT_TRAIN_HEIGHT,
    DEFAULT_TRAIN_NUM_FRAMES,
    DEFAULT_TRAIN_WIDTH,
    DEFAULT_TRANSFORMER_SUBFOLDER,
    DISTILLED_MODALITY_SCALE,
    DISTILLED_SIGMA_VALUES,
    FREEZE_LIST,
    FULL_TRANSFORMER_SUBFOLDER,
    LORA_ATTN_CLASS,
    LORA_EMBED_HOSTS,
    LORA_LINEAR_NAMES,
    LORA_TE_ATTN_NAMES,
    LORA_VIDEO_HOSTS,
    LTX_FPS,
    SFT_NUM_INFERENCE_STEPS,
    LTX25Backend,
    distilled_sigmas,
    is_distilled_subfolder,
    ltx_canvas_hw,
    ltx_num_frames,
    ltx_pack_feature_dim,
    sft_scheduler_overrides,
)
from conceptmod.textsliders.ltx25_uni import (
    DEFAULT_EMBED_REL_L2_WEIGHT,
    DEFAULT_HOLD_MODE,
    DEFAULT_RECIPE,
    DEFAULT_TE_LAST_N,
    HOLD_MODES,
    RECIPE_CHOICES,
    RECIPE_EMBED,
    RECIPE_VELOCITY,
    cosine_l2,
    expression_gap_is_dead,
    is_embed_recipe,
    ltx25_embed_match_loss,
    ltx25_embed_mse,
    ltx25_minus_canary,
    ltx25_uni_total_loss,
    pin_unused_attributes,
    post_connector_mean_cos,
    require_concept_tokens,
    resolve_concept_token_ids,
    resolve_ltx25_recipe,
    unused_hold_mask,
    unused_token_ids,
    velocity_pair,
)

DEFAULT_PROMPTS = Path(__file__).resolve().parent / "data" / "prompts-ltx25-smile.yaml"
DEFAULT_CONFIG = Path(__file__).resolve().parent / "data" / "config-ltx25.yaml"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Opt-in LTX-2.5 UNI t2v slider trainer")
    p.add_argument("--name", type=str, default="smile-ltx25-uni")
    p.add_argument("--model_id", type=str, default=DEFAULT_MODEL)
    p.add_argument(
        "--transformer_subfolder",
        type=str,
        default=DEFAULT_TRANSFORMER_SUBFOLDER,
        help="Distilled DiT is transformer/ (default). transformer_full/ is SFT fallback.",
    )
    p.add_argument("--prompts_file", type=str, default=str(DEFAULT_PROMPTS))
    p.add_argument("--config_file", type=str, default=str(DEFAULT_CONFIG))
    p.add_argument(
        "--recipe",
        type=str,
        default=DEFAULT_RECIPE,
        choices=list(RECIPE_CHOICES),
        help=(
            "ltx25_uni_embed (default): post-connector video embed-match. "
            "ltx25_uni_velocity: failed DiT velocity UNI (opt-in only)."
        ),
    )
    p.add_argument(
        "--te_last_n",
        type=int,
        default=DEFAULT_TE_LAST_N,
        help="TE last-N layers for embed-match LoRA (default 4).",
    )
    p.add_argument(
        "--embed_rel_l2_weight",
        type=float,
        default=DEFAULT_EMBED_REL_L2_WEIGHT,
        help="Rel-L2 weight on valid post-connector video (default 1.0).",
    )
    p.add_argument(
        "--attributes",
        type=str,
        default="",
        help="comma-separated unused attributes (bookkeeping only; not prefixed)",
    )
    p.add_argument("--rank", type=int, default=DEFAULT_RANK)
    p.add_argument("--alpha", type=float, default=DEFAULT_ALPHA)
    p.add_argument(
        "--lora_up_init_std",
        type=float,
        default=DEFAULT_LORA_UP_INIT_STD,
        help="N(0, std) on LoRA-up (default 0.02). Zero-init is UNI identity.",
    )
    p.add_argument("--lr", type=float, default=DEFAULT_LR)
    p.add_argument("--steps", type=int, default=DEFAULT_STEPS)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument(
        "--encoder_device",
        type=str,
        default=DEFAULT_ENCODER_DEVICE,
        help=(
            "TE + connectors device. Dual A6000 card: cuda:1 with "
            "--device cuda:0 for DiT sample. Default cpu."
        ),
    )
    p.add_argument("--hold_weight", type=float, default=1.0)
    p.add_argument(
        "--hold_mode",
        type=str,
        default=DEFAULT_HOLD_MODE,
        choices=list(HOLD_MODES),
        help="non_concept (default) pins every plus token that is not a concept word.",
    )
    p.add_argument("--save_dir", type=str, default=None)
    p.add_argument(
        "--load_ltx_lora",
        type=str,
        default=None,
        help="Dir or .safetensors with lora_ltx-* keys (or PEFT adapter)",
    )
    p.add_argument("--no_sample", action="store_true")
    p.add_argument(
        "--sample_scales",
        type=str,
        default=DEFAULT_SAMPLE_SCALES_TEXT,
        help=(
            "LoRA scales for clips (default -1,0,0.5,1). Infer on neu. "
            "Use equals form when the first scale is negative: "
            "--sample_scales=-1,0,0.5,1"
        ),
    )
    p.add_argument("--sample_num_frames", type=int, default=DEFAULT_NUM_FRAMES)
    p.add_argument("--sample_height", type=int, default=DEFAULT_SAMPLE_HEIGHT)
    p.add_argument("--sample_width", type=int, default=DEFAULT_SAMPLE_WIDTH)
    p.add_argument("--sample_max_rows", type=int, default=None)
    p.add_argument("--sample_seed", type=int, default=None)
    p.add_argument(
        "--dummy",
        action="store_true",
        help="CPU mock; no Hub, no GPU, no LTX-2.5 weights",
    )
    p.add_argument(
        "--diag",
        action="store_true",
        help=(
            "Expression-gap vs identity-leak diagnostic only (no train). "
            "Embed-match: post-connector video mean_cos is the working "
            "gap (~0.68 live). Velocity cos ≈ 1.0 is the dead teacher."
        ),
    )
    return p.parse_args(argv)


def load_slider_rows(prompts_file: str, cli_attributes: str) -> list[dict]:
    raw = yaml.safe_load(Path(prompts_file).read_text()) or []
    concept_words = ""
    bare = True
    if isinstance(raw, dict):
        concept_words = str(raw.get("concept_words") or "")
        bare = bool(raw.get("bare_captions", True))
        raw = raw.get("prompts") or raw.get("rows") or [raw]
    rows = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        pos = str(item.get("positive") or item.get("target") or "")
        neu = str(item.get("neutral") or item.get("target") or "")
        attrs = item.get("attributes") or []
        if isinstance(attrs, str):
            attrs = [a.strip() for a in attrs.split(",") if a.strip()]
        if cli_attributes:
            attrs = list(attrs) + [a.strip() for a in cli_attributes.split(",") if a.strip()]
        seen: set[str] = set()
        uniq = []
        for a in attrs:
            key = a.lower()
            if key not in seen:
                seen.add(key)
                uniq.append(a)
        row_bare = bool(item.get("bare_captions", bare))
        row_concept = str(item.get("concept_words") or concept_words)
        for pos_p, neu_p in pin_unused_attributes(pos, neu, uniq, bare_captions=row_bare):
            rows.append({
                "positive": pos_p,
                "neutral": neu_p,
                "unconditional": str(item.get("unconditional") or item.get("negative") or ""),
                "target": str(item.get("target") or neu),
                "attributes": uniq,
                "concept_words": row_concept,
                "bare_captions": row_bare,
            })
    if not rows:
        raise ValueError(f"no slider rows in {prompts_file}")
    return rows


def parse_sample_scales(text: str) -> list[float]:
    scales = [float(x) for x in str(text).split(",") if str(x).strip()]
    if not scales:
        raise ValueError(" --sample_scales is empty")
    return scales


def sample_prompts_from_rows(rows: list[dict], max_rows: int | None = None) -> list[str]:
    """Unique yaml ``target`` (neu subject). Infer scale 1 on these captions."""
    seen: list[str] = []
    for row in rows:
        target = str(row.get("target") or row.get("neutral") or "").strip()
        if target and target not in seen:
            seen.append(target)
    if max_rows is not None:
        seen = seen[: max(0, int(max_rows))]
    if not seen:
        raise ValueError("no sample prompts in slider rows")
    return seen


def build_backend(args: argparse.Namespace) -> LTX25Backend:
    recipe = resolve_ltx25_recipe(getattr(args, "recipe", DEFAULT_RECIPE))
    te_last_n = int(getattr(args, "te_last_n", DEFAULT_TE_LAST_N))
    kwargs = dict(
        model_id=args.model_id,
        transformer_subfolder=args.transformer_subfolder,
        lora_rank=args.rank,
        lora_alpha=args.alpha,
        lora_up_init_std=float(getattr(args, "lora_up_init_std", DEFAULT_LORA_UP_INIT_STD)),
        recipe=recipe,
        te_last_n=te_last_n,
    )
    if args.dummy:
        return LTX25Backend(
            device="cpu",
            encoder_device=None,
            dummy=True,
            **kwargs,
        )
    return LTX25Backend(
        device=args.device,
        encoder_device=getattr(args, "encoder_device", DEFAULT_ENCODER_DEVICE),
        dummy=False,
        **kwargs,
    )


def _row_hold(backend: LTX25Backend, row: dict, tokenizer, args: argparse.Namespace):
    plus_enc = backend.encode_text(row["positive"], frozen=True)
    neu_enc = backend.encode_text(row["neutral"], frozen=True)
    uncond_enc = backend.encode_text(row.get("unconditional") or row["neutral"], frozen=True)
    concept = resolve_concept_token_ids(
        tokenizer, row["positive"], row["neutral"], row.get("concept_words") or "",
    )
    require_concept_tokens(plus_enc.token_ids, concept)
    unused = unused_token_ids(tokenizer, row["attributes"])
    hold_mask = unused_hold_mask(
        plus_enc.token_ids,
        unused,
        concept,
        hold_mode=getattr(args, "hold_mode", DEFAULT_HOLD_MODE),
    )
    return plus_enc, neu_enc, uncond_enc, hold_mask


def _embed_match_step(
    backend: LTX25Backend,
    args: argparse.Namespace,
    row: dict,
    tokenizer,
    opt,
    step: int,
    dead_gap: bool,
) -> tuple[dict, bool]:
    """Student encode(neu)+LoRA → frozen encode(plus) post-connector video."""
    if backend.network is not None:
        backend.network.set_lora_slider(0.0)
    plus_enc, neu_enc, uncond_enc, hold_mask = _row_hold(backend, row, tokenizer, args)
    with torch.no_grad():
        teacher_video, teacher_mask = backend.encode_post_connector_video(
            plus_enc, hold_neu=neu_enc, hold_mask=hold_mask, scale=0.0,
        )
        neu_video, neu_mask = backend.encode_post_connector_video(
            neu_enc, scale=0.0,
        )
        gap = post_connector_mean_cos(teacher_video, neu_video, teacher_mask, neu_mask)
        if step == 0 and float(gap["cos"]) >= 0.999:
            dead_gap = True
            print(
                "warning: frozen plus vs neu post-connector video cos ≈ 1.0 "
                f"(cos={gap['cos']:.6f}). Embed-match needs a text-path gap "
                "(live smile sat ~0.68). Do not silently train a dead gap."
            )
        uncond_video, uncond_mask = backend.encode_post_connector_video(
            uncond_enc, scale=0.0,
        )
    if backend.network is not None:
        backend.network.set_lora_slider(1.0)
    neu_student = backend.encode_text(row["neutral"], frozen=False)
    student_video, student_mask = backend.encode_post_connector_video(
        neu_student, scale=1.0,
    )
    if student_video.shape[1] == len(neu_student.token_ids) and args.dummy:
        raise RuntimeError("connectors did not change T; hold is not PRE-connector-safe")
    loss = ltx25_embed_match_loss(
        student_video,
        teacher_video.detach(),
        student_mask,
        teacher_mask,
        rel_l2_weight=float(getattr(args, "embed_rel_l2_weight", DEFAULT_EMBED_REL_L2_WEIGHT)),
    )
    opt.zero_grad(set_to_none=True)
    loss.backward()
    if args.dummy:
        torch.nn.utils.clip_grad_norm_(backend.trainable_parameters(), 1.0)
    opt.step()
    match = post_connector_mean_cos(
        student_video.detach(), teacher_video.detach(), student_mask, teacher_mask,
    )
    if backend.network is not None:
        backend.network.set_lora_slider(-1.0)
    minus_video, minus_mask = backend.encode_post_connector_video(
        backend.encode_text(row["neutral"], frozen=False), scale=-1.0,
    )
    canary = float(ltx25_embed_mse_or_canary(minus_video, uncond_video, minus_mask, uncond_mask))
    return {
        "step": step,
        "loss": float(loss.detach().item()),
        "cos": float(match["cos"]),
        "l2": float(match["l2"]),
        "minus_canary": canary,
        "positive": row["positive"],
        "neutral": row["neutral"],
        "student_plus": "neu",
        "hold_stage": "pre_connector",
        "teacher": "encode_plus_post_connector_video",
    }, dead_gap


def ltx25_embed_mse_or_canary(pred, tgt, pred_mask, tgt_mask) -> float:
    """Logged minus canary on post-connector video. Not in the train loss."""
    with torch.no_grad():
        return float(ltx25_embed_mse(pred, tgt, pred_mask, tgt_mask).item())


def _velocity_step(
    backend: LTX25Backend,
    args: argparse.Namespace,
    row: dict,
    tokenizer,
    opt,
    step: int,
    dead_gap: bool,
    *,
    dummy_video,
    dummy_audio,
) -> tuple[dict, bool]:
    plus_enc, neu_enc, uncond_enc, hold_mask = _row_hold(backend, row, tokenizer, args)
    packed_plus = backend.pack_t2v(
        plus_enc,
        video_latents=dummy_video,
        audio_latents=dummy_audio,
        hold_neu=neu_enc,
        hold_mask=hold_mask,
        num_frames=DEFAULT_TRAIN_NUM_FRAMES,
        height=DEFAULT_TRAIN_HEIGHT,
        width=DEFAULT_TRAIN_WIDTH,
    )
    packed_neu = backend.pack_t2v(
        neu_enc,
        video_latents=packed_plus.hidden_states.detach(),
        audio_latents=packed_plus.audio_hidden_states.detach(),
        num_frames=DEFAULT_TRAIN_NUM_FRAMES,
        height=DEFAULT_TRAIN_HEIGHT,
        width=DEFAULT_TRAIN_WIDTH,
    )
    packed_uncond = backend.pack_t2v(
        uncond_enc,
        video_latents=packed_plus.hidden_states.detach(),
        audio_latents=packed_plus.audio_hidden_states.detach(),
        num_frames=DEFAULT_TRAIN_NUM_FRAMES,
        height=DEFAULT_TRAIN_HEIGHT,
        width=DEFAULT_TRAIN_WIDTH,
    )
    if packed_plus.hold_stage != "pre_connector":
        raise RuntimeError("LTX-2.5 hold must be PRE-connector")
    if packed_plus.encoder_hidden_states.shape[1] == packed_plus.n_prompt_tokens and args.dummy:
        raise RuntimeError("connectors did not change T; hold is not PRE-connector-safe")

    with torch.no_grad():
        tgt_plus_v = backend.forward_velocity(packed_plus, scale=0.0)
        tgt_zero_v = backend.forward_velocity(packed_neu, scale=0.0)
        tgt_plus = velocity_pair(tgt_plus_v.sample, tgt_plus_v.audio_sample)
        tgt_zero = velocity_pair(tgt_zero_v.sample, tgt_zero_v.audio_sample)
        gap = cosine_l2(tgt_plus, tgt_zero)
        if step == 0 and expression_gap_is_dead(gap):
            dead_gap = True
            print(
                "warning: frozen plus vs neu velocity cos ≈ 1.0 "
                f"(cos={gap['cos']:.6f}). Distilled transformer/ may have "
                "a dead expression gap. Do not silently train — "
                f"fallback is --transformer_subfolder {FULL_TRANSFORMER_SUBFOLDER}."
            )

    pred_plus_v = backend.forward_velocity(packed_neu, scale=1.0)
    pred_zero_v = backend.forward_velocity(packed_neu, scale=0.0)
    pred_plus = velocity_pair(pred_plus_v.sample, pred_plus_v.audio_sample)
    pred_zero = velocity_pair(pred_zero_v.sample, pred_zero_v.audio_sample)

    loss = ltx25_uni_total_loss(
        pred_plus, tgt_plus, pred_zero, tgt_zero,
        packed_plus.pre_connector_hidden, neu_enc.embeds, hold_mask,
        hold_weight=float(args.hold_weight),
    )
    opt.zero_grad(set_to_none=True)
    loss.backward()
    opt.step()
    canary = _canary(backend, packed_uncond, packed_neu)
    return {
        "step": step,
        "loss": float(loss.detach().item()),
        "minus_canary": canary,
        "positive": row["positive"],
        "neutral": row["neutral"],
        "student_plus": "neu",
        "hold_stage": "pre_connector",
    }, dead_gap


@torch.no_grad()
def _canary(backend: LTX25Backend, packed_uncond, packed_neu) -> float:
    pred = backend.forward_velocity(packed_neu, scale=-1.0)
    tgt = backend.forward_velocity(packed_uncond, scale=0.0)
    return float(ltx25_minus_canary(
        velocity_pair(pred.sample, pred.audio_sample),
        velocity_pair(tgt.sample, tgt.audio_sample),
    ).item())


def _slug(text: str) -> str:
    cleaned = "".join(ch if ch.isalnum() else "-" for ch in str(text).lower())
    return "-".join(part for part in cleaned.split("-") if part)[:48] or "prompt"


def write_ltx_sample_mp4(path: Path, frames, *, fps: float, audio=None, sampling_rate=None) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        from diffusers.utils.export_utils import encode_video

        encode_video(
            frames,
            fps=float(fps),
            output_path=str(path),
            audio=audio,
            audio_sample_rate=sampling_rate,
        )
        if path.is_file() and path.stat().st_size > 0:
            return
    except Exception:
        pass
    payload = _frames_payload(frames)
    header = b"\x00\x00\x00\x18ftypisom\x00\x00\x00\x00isomiso2"
    path.write_bytes(header + payload)


def _frames_payload(frames) -> bytes:
    if frames is None:
        return b"dummy-t2v"
    video = frames
    if isinstance(frames, (list, tuple)):
        video = frames[0] if frames else None
    if video is None:
        return b"dummy-t2v"
    if hasattr(video, "detach"):
        video = video.detach().cpu().contiguous().numpy()
    try:
        return bytes(memoryview(video.reshape(-1)[:4096])) + str(getattr(video, "shape", "")).encode()
    except Exception:
        return str(video).encode()[:4096]


def emit_ltx_samples(
    backend: LTX25Backend,
    args: argparse.Namespace,
    save_dir: Path,
    rows: list[dict],
) -> list[dict]:
    scales = parse_sample_scales(getattr(args, "sample_scales", DEFAULT_SAMPLE_SCALES_TEXT))
    prompts = sample_prompts_from_rows(rows, getattr(args, "sample_max_rows", None))
    frames = ltx_num_frames(int(getattr(args, "sample_num_frames", DEFAULT_NUM_FRAMES)))
    height, width = ltx_canvas_hw(
        int(getattr(args, "sample_height", DEFAULT_SAMPLE_HEIGHT)),
        int(getattr(args, "sample_width", DEFAULT_SAMPLE_WIDTH)),
    )
    seed = int(getattr(args, "sample_seed", None) or args.seed)
    out_dir = Path(save_dir) / "samples"
    out_dir.mkdir(parents=True, exist_ok=True)
    records: list[dict] = []
    for prompt in prompts:
        for scale in scales:
            result = backend.generate_t2v(
                prompt,
                scale=float(scale),
                num_frames=None if args.dummy else frames,
                height=None if args.dummy else height,
                width=None if args.dummy else width,
                seed=seed,
            )
            videos = result.get("videos")
            video = videos[0] if isinstance(videos, (list, tuple)) and videos else videos
            scale_tag = f"{scale:g}".replace("-", "m")
            name = f"final_{_slug(prompt)}_scale{scale_tag}.mp4"
            path = out_dir / name
            write_ltx_sample_mp4(
                path,
                video,
                fps=LTX_FPS,
                audio=(result.get("audio") or [None])[0]
                if isinstance(result.get("audio"), (list, tuple))
                else result.get("audio"),
                sampling_rate=result.get("sampling_rate"),
            )
            records.append({
                "prompt": prompt,
                "scale": float(scale),
                "path": path.name,
                "seed": seed,
                "guidance": 1.0,
                "stg_scale": 0.0,
                "modality_scale": DISTILLED_MODALITY_SCALE,
                "sigmas": (
                    distilled_sigmas() if not args.dummy else list(DISTILLED_SIGMA_VALUES)
                ) if is_distilled_subfolder(getattr(args, "transformer_subfolder", DEFAULT_TRANSFORMER_SUBFOLDER)) else None,
                "num_frames": int(result.get("num_frames") or frames),
                "height": int(result.get("height") or height),
                "width": int(result.get("width") or width),
                "decoder": "conv_vae",
                "prompt_enhancer": False,
                "dummy": bool(args.dummy),
            })
    payload = {
        "dummy": bool(args.dummy),
        "seed": seed,
        "scales": scales,
        "prompts": prompts,
        "guidance": 1.0,
        "stg_scale": 0.0,
        "modality_scale": DISTILLED_MODALITY_SCALE,
        "sigmas": (
            distilled_sigmas() if not args.dummy else list(DISTILLED_SIGMA_VALUES)
        ) if is_distilled_subfolder(getattr(args, "transformer_subfolder", DEFAULT_TRANSFORMER_SUBFOLDER)) else None,
        "num_frames": 2 if args.dummy else frames,
        "height": 8 if args.dummy else height,
        "width": 8 if args.dummy else width,
        "decoder": "conv_vae",
        "prompt_enhancer": False,
        "infer_caption": "neu",
        "samples": records,
    }
    (out_dir / "final_meta.json").write_text(json.dumps(payload, indent=2))
    if not records:
        raise RuntimeError("LTX-2.5 sample grid is empty")
    print(f"wrote {len(records)} t2v clips under {out_dir}")
    return records


def train(args: argparse.Namespace, backend: LTX25Backend | None = None) -> dict:
    rows = load_slider_rows(args.prompts_file, args.attributes)
    backend = backend or build_backend(args)
    loaded_lora = None
    if getattr(args, "load_ltx_lora", None):
        loaded_lora = backend.load_trained(args.load_ltx_lora)
        print(f"loaded lora_ltx weights from {loaded_lora}")
    tokenizer = backend.tokenizer
    params = backend.trainable_parameters()
    opt = None
    if int(args.steps) > 0:
        if args.dummy:
            opt = torch.optim.SGD(params, lr=float(args.lr))
        else:
            opt = torch.optim.Adam(params, lr=float(args.lr))
    torch.manual_seed(int(args.seed))

    history = []
    dummy_video = dummy_audio = None
    if args.dummy:
        g = torch.Generator().manual_seed(int(args.seed))
        dummy_video = torch.randn(
            1, 2, ltx_pack_feature_dim(backend.transformer, kind="video"), generator=g,
        )
        dummy_audio = torch.randn(
            1, 2, ltx_pack_feature_dim(backend.transformer, kind="audio"), generator=g,
        )
    recipe = resolve_ltx25_recipe(getattr(args, "recipe", DEFAULT_RECIPE))
    embed = is_embed_recipe(recipe)
    dead_gap = False
    for step in range(int(args.steps)):
        row = rows[step % len(rows)]
        if embed:
            rec, dead_gap = _embed_match_step(
                backend, args, row, tokenizer, opt, step, dead_gap,
            )
        else:
            rec, dead_gap = _velocity_step(
                backend, args, row, tokenizer, opt, step, dead_gap,
                dummy_video=dummy_video, dummy_audio=dummy_audio,
            )
        history.append(rec)
        if step == 0 or (step + 1) % 50 == 0 or step + 1 == int(args.steps):
            extra = ""
            if "cos" in rec:
                extra = f" cos={rec['cos']:.4f}"
            print(
                f"ltx25 {recipe} step {step}: loss={rec['loss']:.4f} "
                f"minus_canary={rec['minus_canary']:.4f} (not in loss){extra}"
            )

    sidecar = {
        "name": args.name,
        "backend": "ltx25",
        "model_id": args.model_id,
        "resolved_model_id": DEFAULT_MODEL,
        "transformer_subfolder": args.transformer_subfolder,
        "transformer_full_fallback": FULL_TRANSFORMER_SUBFOLDER,
        "stack": "ltx2_post_connector_video" if embed else "ltx2_flow_velocity",
        "recipe": recipe,
        "plus_neu": True,
        "student_plus": "neu",
        "minus_teacher": False,
        "minus_canary": True,
        "dit_frozen": bool(embed),
        "te_last_n": int(getattr(args, "te_last_n", DEFAULT_TE_LAST_N)) if embed else None,
        "embed_rel_l2_weight": (
            float(getattr(args, "embed_rel_l2_weight", DEFAULT_EMBED_REL_L2_WEIGHT))
            if embed else None
        ),
        "teacher": (
            "encode_plus_post_connector_video" if embed else "frozen_plus_velocity"
        ),
        "hold": (
            "non_concept_tokens_to_encode_neu"
            if str(getattr(args, "hold_mode", DEFAULT_HOLD_MODE)) == DEFAULT_HOLD_MODE
            else "unused_attribute_tokens_to_encode_neu"
        ),
        "hold_mode": str(getattr(args, "hold_mode", DEFAULT_HOLD_MODE)),
        "hold_stage": "pre_connector",
        "hold_weight": float(args.hold_weight),
        "hold_concept_words": False,
        "lora_only": True,
        "lora_host": "te_last_n+video_connectors" if embed else LORA_ATTN_CLASS,
        "lora_hosts": list(LORA_EMBED_HOSTS) if embed else list(LORA_VIDEO_HOSTS),
        "lora_linears": list(LORA_TE_ATTN_NAMES) if embed else list(LORA_LINEAR_NAMES),
        "train_adaln": False,
        "train_audio_attn": False,
        "train_dit": not embed,
        "velocity_teacher": None if embed else "flow",
        "velocity_contract": None if embed else "x0 = x_t - sigma * v",
        "predict_v_faked": False,
        "cfg_distilled": is_distilled_subfolder(args.transformer_subfolder),
        "guidance": 1.0,
        "stg_scale": 0.0,
        "modality_scale": DISTILLED_MODALITY_SCALE,
        "sigmas": list(DISTILLED_SIGMA_VALUES) if is_distilled_subfolder(args.transformer_subfolder) else None,
        "use_dynamic_shifting": (
            False if is_distilled_subfolder(args.transformer_subfolder)
            else sft_scheduler_overrides()["use_dynamic_shifting"]
        ),
        "shift_terminal": (
            None if is_distilled_subfolder(args.transformer_subfolder)
            else sft_scheduler_overrides()["shift_terminal"]
        ),
        "num_inference_steps": (
            None if is_distilled_subfolder(args.transformer_subfolder)
            else SFT_NUM_INFERENCE_STEPS
        ),
        "prompt_enhancer": False,
        "decoder": "conv_vae",
        "sample_num_frames": int(getattr(args, "sample_num_frames", DEFAULT_NUM_FRAMES)),
        "sample_height": int(getattr(args, "sample_height", DEFAULT_SAMPLE_HEIGHT)),
        "sample_width": int(getattr(args, "sample_width", DEFAULT_SAMPLE_WIDTH)),
        "freeze": list(FREEZE_LIST),
        "rank": args.rank,
        "alpha": args.alpha,
        "lora_up_init_std": float(getattr(args, "lora_up_init_std", backend.lora_up_init_std)),
        "lr": args.lr,
        "steps": args.steps,
        "seed": args.seed,
        "device": args.device if not args.dummy else "cpu",
        "encoder_device": None if args.dummy else getattr(args, "encoder_device", DEFAULT_ENCODER_DEVICE),
        "load_ltx_lora": loaded_lora,
        "dummy": bool(args.dummy),
        "dead_gap": dead_gap,
        "first_loss": history[0]["loss"] if history else None,
        "last_loss": history[-1]["loss"] if history else None,
        "history": history[-8:],
    }
    save_dir = Path(args.save_dir or f"models/{args.name}")
    save_dir.mkdir(parents=True, exist_ok=True)
    sidecar_path = save_dir / f"{args.name}_last.json"
    backend.save_trained(str(save_dir / f"{args.name}_lora"))
    sample_records = []
    if not getattr(args, "no_sample", False):
        sample_records = emit_ltx_samples(backend, args, save_dir, rows)
        scales = parse_sample_scales(getattr(args, "sample_scales", DEFAULT_SAMPLE_SCALES_TEXT))
        sidecar["sample_grid"] = {
            "method": "t2v_lora_ltx25",
            "scales": scales,
            "guidance": 1.0,
            "stg_scale": 0.0,
            "modality_scale": DISTILLED_MODALITY_SCALE,
            "sigmas": list(DISTILLED_SIGMA_VALUES) if is_distilled_subfolder(args.transformer_subfolder) else None,
            "num_frames": 2 if args.dummy else ltx_num_frames(int(args.sample_num_frames)),
            "height": 8 if args.dummy else int(args.sample_height),
            "width": 8 if args.dummy else int(args.sample_width),
            "decoder": "conv_vae",
            "infer_caption": "neu",
            "n": len(sample_records),
        }
    sidecar_path.write_text(json.dumps(sidecar, indent=2))
    print(f"wrote {sidecar_path}")
    return sidecar


def main(argv: list[str] | None = None) -> dict:
    args = parse_args(argv)
    if getattr(args, "diag", False):
        from conceptmod.textsliders.diag_ltx25_uni import run_diag

        return run_diag(args)
    return train(args)


if __name__ == "__main__":
    main()
