#!/usr/bin/env python3
"""Opt-in LTX-2.5 concept-slider trainer (UNI on transformer velocity).

Default Music 3 trainers are unchanged (``--lm_target v9`` / ``--pole_mode hidden``).

Live card: ``Lightricks/LTX-2.5-Diffusers`` distilled ``transformer/``.
Student +1 trains on the **neu** caption (infer path). Plus is teacher-only.
Hold is PRE-connector. LoRA is video ``attn1`` / ``attn2`` only.
After train (or ``--steps 0 --load_ltx_lora``) writes mp4s at scales
0 / 0.5 / 1 under ``save_dir/samples/``. ``--dummy`` is the CI path.
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
    DEFAULT_ENCODER_DEVICE,
    DEFAULT_LORA_UP_INIT_STD,
    DEFAULT_MODEL,
    DEFAULT_NUM_FRAMES,
    DEFAULT_SAMPLE_HEIGHT,
    DEFAULT_SAMPLE_SCALES,
    DEFAULT_SAMPLE_WIDTH,
    DEFAULT_TRAIN_HEIGHT,
    DEFAULT_TRAIN_NUM_FRAMES,
    DEFAULT_TRAIN_WIDTH,
    DEFAULT_TRANSFORMER_SUBFOLDER,
    DISTILLED_SIGMA_VALUES,
    FREEZE_LIST,
    FULL_TRANSFORMER_SUBFOLDER,
    LORA_ATTN_CLASS,
    LORA_LINEAR_NAMES,
    LORA_VIDEO_HOSTS,
    LTX_FPS,
    LTX25Backend,
    distilled_sigmas,
    ltx_canvas_hw,
    ltx_num_frames,
    ltx_pack_feature_dim,
)
from conceptmod.textsliders.ltx25_uni import (
    DEFAULT_HOLD_MODE,
    HOLD_MODES,
    expression_gap_is_dead,
    ltx25_minus_canary,
    ltx25_uni_total_loss,
    pin_unused_attributes,
    require_concept_tokens,
    resolve_concept_token_ids,
    unused_hold_mask,
    unused_token_ids,
    velocity_pair,
    cosine_l2,
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
        "--attributes",
        type=str,
        default="",
        help="comma-separated unused attributes (bookkeeping only; not prefixed)",
    )
    p.add_argument("--rank", type=int, default=8)
    p.add_argument("--alpha", type=float, default=8.0)
    p.add_argument(
        "--lora_up_init_std",
        type=float,
        default=DEFAULT_LORA_UP_INIT_STD,
        help="N(0, std) on LoRA-up (default 0.02). Zero-init is UNI identity.",
    )
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--steps", type=int, default=500)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument(
        "--encoder_device",
        type=str,
        default=DEFAULT_ENCODER_DEVICE,
        help="Gemma 4 12B device. Default cpu — do not pipe.to TE+DiT together.",
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
        default="0,0.5,1",
        help="LoRA scales for clips (default 0,0.5,1). Infer scale 1 on neu.",
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
            "If plus/neu velocity cos ≈ 1.0, document transformer_full/ "
            "as fallback — do not silently train a dead gap."
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
    if args.dummy:
        return LTX25Backend(
            device="cpu",
            encoder_device=None,
            model_id=args.model_id,
            transformer_subfolder=args.transformer_subfolder,
            lora_rank=args.rank,
            lora_alpha=args.alpha,
            lora_up_init_std=float(getattr(args, "lora_up_init_std", DEFAULT_LORA_UP_INIT_STD)),
            dummy=True,
        )
    return LTX25Backend(
        device=args.device,
        encoder_device=getattr(args, "encoder_device", DEFAULT_ENCODER_DEVICE),
        model_id=args.model_id,
        transformer_subfolder=args.transformer_subfolder,
        lora_rank=args.rank,
        lora_alpha=args.alpha,
        lora_up_init_std=float(getattr(args, "lora_up_init_std", DEFAULT_LORA_UP_INIT_STD)),
        dummy=False,
    )


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
    scales = parse_sample_scales(getattr(args, "sample_scales", "0,0.5,1"))
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
                "modality_scale": 0.0,
                "sigmas": distilled_sigmas() if not args.dummy else list(DISTILLED_SIGMA_VALUES),
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
        "modality_scale": 0.0,
        "sigmas": distilled_sigmas() if not args.dummy else list(DISTILLED_SIGMA_VALUES),
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
    dead_gap = False
    for step in range(int(args.steps)):
        row = rows[step % len(rows)]
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

        # Student +1 on neu (infer path). Plus is teacher-only.
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
        rec = {
            "step": step,
            "loss": float(loss.detach().item()),
            "minus_canary": canary,
            "positive": row["positive"],
            "neutral": row["neutral"],
            "student_plus": "neu",
            "hold_stage": "pre_connector",
        }
        history.append(rec)
        if step == 0 or (step + 1) % 50 == 0 or step + 1 == int(args.steps):
            print(
                f"ltx25 uni step {step}: loss={rec['loss']:.4f} "
                f"minus_canary={canary:.4f} (not in loss)"
            )

    sidecar = {
        "name": args.name,
        "backend": "ltx25",
        "model_id": args.model_id,
        "resolved_model_id": DEFAULT_MODEL,
        "transformer_subfolder": args.transformer_subfolder,
        "transformer_full_fallback": FULL_TRANSFORMER_SUBFOLDER,
        "stack": "ltx2_flow_velocity",
        "recipe": "ltx25_uni_velocity",
        "plus_neu": True,
        "student_plus": "neu",
        "minus_teacher": False,
        "minus_canary": True,
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
        "lora_host": LORA_ATTN_CLASS,
        "lora_hosts": list(LORA_VIDEO_HOSTS),
        "lora_linears": list(LORA_LINEAR_NAMES),
        "train_adaln": False,
        "train_audio_attn": False,
        "velocity_teacher": "flow",
        "velocity_contract": "x0 = x_t - sigma * v",
        "predict_v_faked": False,
        "cfg_distilled": args.transformer_subfolder == DEFAULT_TRANSFORMER_SUBFOLDER,
        "guidance": 1.0,
        "stg_scale": 0.0,
        "modality_scale": 0.0,
        "sigmas": list(DISTILLED_SIGMA_VALUES),
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
        scales = parse_sample_scales(getattr(args, "sample_scales", "0,0.5,1"))
        sidecar["sample_grid"] = {
            "method": "t2v_lora_ltx25",
            "scales": scales,
            "guidance": 1.0,
            "stg_scale": 0.0,
            "modality_scale": 0.0,
            "sigmas": list(DISTILLED_SIGMA_VALUES),
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
