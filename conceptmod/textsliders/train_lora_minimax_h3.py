#!/usr/bin/env python3
"""Opt-in MiniMax-H3 concept-slider trainer (UNI on packed velocity).

Default Music 3 trainers are unchanged (``--lm_target v9`` / ``--pole_mode hidden``).

Live card: ``MiniMaxAI/MiniMax-H3`` variant FL2VA, workflow t2va.
``--dummy`` is the CI / CPU path: no Hub, no GPU, no MiniMax-H3 weights.
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

from conceptmod.textsliders.minimax_h3_backend import (
    DEFAULT_MODEL,
    DEFAULT_TASK_INDEX,
    DEFAULT_VARIANT,
    DEFAULT_WORKFLOW,
    FREEZE_LIST,
    HOSTED_NOT_IN_WEIGHTS,
    LORA_ATTN_CLASS,
    LORA_LINEAR_NAMES,
    REF2VA_TASK_INDEX,
    MiniMaxH3Backend,
)
from conceptmod.textsliders.minimax_h3_uni import (
    concept_token_ids,
    minimax_h3_minus_canary,
    minimax_h3_uni_total_loss,
    pin_unused_attributes,
    unused_hold_mask,
    unused_token_ids,
    velocity_pair,
)

DEFAULT_PROMPTS = Path(__file__).resolve().parent / "data" / "prompts-minimax-h3.yaml"
DEFAULT_CONFIG = Path(__file__).resolve().parent / "data" / "config-minimax-h3.yaml"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Opt-in MiniMax-H3 UNI t2va slider trainer")
    p.add_argument("--name", type=str, default="age-minimax-h3-uni")
    p.add_argument("--model_id", type=str, default=DEFAULT_MODEL)
    p.add_argument("--variant", type=str, default=DEFAULT_VARIANT, choices=["FL2VA", "Ref2VA"])
    p.add_argument(
        "--workflow",
        type=str,
        default=DEFAULT_WORKFLOW,
        choices=["t2va", "fl2va", "ref2va"],
        help="Default t2va. Ref2VA / ref2va is a later path, not the default.",
    )
    p.add_argument("--prompts_file", type=str, default=str(DEFAULT_PROMPTS))
    p.add_argument("--config_file", type=str, default=str(DEFAULT_CONFIG))
    p.add_argument(
        "--attributes",
        type=str,
        default="",
        help="comma-separated unused attributes pinned on both poles",
    )
    p.add_argument("--rank", type=int, default=8)
    p.add_argument("--alpha", type=float, default=8.0)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--steps", type=int, default=500)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--short_side", type=int, default=768)
    p.add_argument("--guidance", type=float, default=0.0, help="CFG-distilled: stay 0")
    p.add_argument("--hold_weight", type=float, default=1.0)
    p.add_argument("--save_dir", type=str, default=None)
    p.add_argument(
        "--dummy",
        action="store_true",
        help="CPU mock packed sequence; no Hub, no GPU, no MiniMax-H3 weights",
    )
    return p.parse_args(argv)


def load_slider_rows(prompts_file: str, cli_attributes: str) -> list[dict]:
    raw = yaml.safe_load(Path(prompts_file).read_text()) or []
    if isinstance(raw, dict):
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
        for pos_p, neu_p in pin_unused_attributes(pos, neu, uniq):
            rows.append({
                "positive": pos_p,
                "neutral": neu_p,
                "unconditional": str(item.get("unconditional") or item.get("negative") or ""),
                "target": str(item.get("target") or neu),
                "attributes": uniq,
            })
    if not rows:
        raise ValueError(f"no slider rows in {prompts_file}")
    return rows


def build_backend(args: argparse.Namespace) -> MiniMaxH3Backend:
    if args.dummy:
        return MiniMaxH3Backend(
            device="cpu",
            model_id=args.model_id,
            variant=args.variant,
            workflow=args.workflow,
            short_side=min(int(args.short_side), 32),
            lora_rank=args.rank,
            lora_alpha=args.alpha,
            dummy=True,
        )
    return MiniMaxH3Backend(
        device=args.device,
        model_id=args.model_id,
        variant=args.variant,
        workflow=args.workflow,
        short_side=args.short_side,
        lora_rank=args.rank,
        lora_alpha=args.alpha,
        dummy=False,
    )


@torch.no_grad()
def _canary(backend: MiniMaxH3Backend, packed_uncond, packed_neu) -> float:
    pred = backend.forward_velocity(packed_neu, scale=-1.0)
    tgt = backend.forward_velocity(packed_uncond, scale=0.0)
    return float(minimax_h3_minus_canary(
        velocity_pair(pred.sample, pred.audio_sample),
        velocity_pair(tgt.sample, tgt.audio_sample),
    ).item())


def train(args: argparse.Namespace, backend: MiniMaxH3Backend | None = None) -> dict:
    if float(args.guidance) != 0.0:
        print(
            f"warning: MiniMax-H3 is CFG-distilled; live generate uses guidance 0 "
            f"(got --guidance {args.guidance}). Trainer still runs one forward pass."
        )
    rows = load_slider_rows(args.prompts_file, args.attributes)
    backend = backend or build_backend(args)
    tokenizer = backend.tokenizer
    params = backend.trainable_parameters()
    # Dummy UNI is identity-plus-offset; SGD on that MSE is monotone. Live uses Adam.
    if args.dummy:
        opt = torch.optim.SGD(params, lr=float(args.lr))
    else:
        opt = torch.optim.Adam(params, lr=float(args.lr))
    torch.manual_seed(int(args.seed))

    history = []
    # Dummy smoke keeps one packed noise so UNI identity is comparable step-to-step.
    dummy_video = dummy_audio = None
    if args.dummy:
        g = torch.Generator().manual_seed(int(args.seed))
        dummy_video = torch.randn(1, 2, int(backend.transformer.video_dim), generator=g)
        dummy_audio = torch.randn(1, 2, int(backend.transformer.audio_dim), generator=g)
    for step in range(int(args.steps)):
        row = rows[step % len(rows)]
        plus_enc = backend.encode_text(row["positive"], frozen=True)
        neu_enc = backend.encode_text(row["neutral"], frozen=True)
        uncond_enc = backend.encode_text(row.get("unconditional") or row["neutral"], frozen=True)
        concept = concept_token_ids(tokenizer, row["positive"], row["neutral"])
        unused = unused_token_ids(tokenizer, row["attributes"])
        hold_mask = unused_hold_mask(plus_enc.token_ids, unused, concept)

        packed_plus = backend.pack_t2va(
            plus_enc,
            video_latents=dummy_video,
            audio_latents=dummy_audio,
            hold_neu=neu_enc,
            hold_mask=hold_mask,
        )
        packed_neu = backend.pack_t2va(
            neu_enc,
            video_latents=packed_plus.hidden_states.detach(),
            audio_latents=packed_plus.audio_hidden_states.detach(),
        )
        packed_uncond = backend.pack_t2va(
            uncond_enc,
            video_latents=packed_plus.hidden_states.detach(),
            audio_latents=packed_plus.audio_hidden_states.detach(),
        )

        with torch.no_grad():
            tgt_plus_v = backend.forward_velocity(packed_plus, scale=0.0)
            tgt_zero_v = backend.forward_velocity(packed_neu, scale=0.0)
            tgt_plus = velocity_pair(tgt_plus_v.sample, tgt_plus_v.audio_sample)
            tgt_zero = velocity_pair(tgt_zero_v.sample, tgt_zero_v.audio_sample)

        pred_plus_v = backend.forward_velocity(packed_plus, scale=1.0)
        pred_zero_v = backend.forward_velocity(packed_neu, scale=0.0)
        pred_plus = velocity_pair(pred_plus_v.sample, pred_plus_v.audio_sample)
        pred_zero = velocity_pair(pred_zero_v.sample, pred_zero_v.audio_sample)

        loss = minimax_h3_uni_total_loss(
            pred_plus, tgt_plus, pred_zero, tgt_zero,
            packed_plus.encoder_hidden_states, neu_enc.embeds, hold_mask,
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
        }
        history.append(rec)
        if step == 0 or (step + 1) % 50 == 0 or step + 1 == int(args.steps):
            print(
                f"minimax-h3 uni step {step}: loss={rec['loss']:.4f} "
                f"minus_canary={canary:.4f} (not in loss)"
            )

    sidecar = {
        "name": args.name,
        "backend": "minimax_h3",
        "model_id": args.model_id,
        "resolved_model_id": DEFAULT_MODEL,
        "variant": args.variant,
        "workflow": args.workflow,
        "task_index": DEFAULT_TASK_INDEX if args.variant == "FL2VA" else REF2VA_TASK_INDEX,
        "stack": "omni_transformer_flow",
        "recipe": "minimax_h3_uni_velocity",
        "plus_neu": True,
        "minus_teacher": False,
        "minus_canary": True,
        "hold": "unused_tokens_to_encode_neu",
        "hold_concept_words": False,
        "lora_only": True,
        "lora_host": LORA_ATTN_CLASS,
        "lora_linears": list(LORA_LINEAR_NAMES),
        "train_adaln": False,
        "velocity_teacher": "data_pointing",
        "velocity_contract": "x0 = x_t + sigma * v",
        "predict_v_faked": False,
        "cfg_distilled": True,
        "guidance": float(args.guidance),
        "short_side": int(args.short_side),
        "freeze": list(FREEZE_LIST),
        "hosted_not_in_weights": list(HOSTED_NOT_IN_WEIGHTS),
        "ref2va": "later path; default is FL2VA t2va",
        "rank": args.rank,
        "alpha": args.alpha,
        "lr": args.lr,
        "steps": args.steps,
        "seed": args.seed,
        "device": args.device if not args.dummy else "cpu",
        "dummy": bool(args.dummy),
        "first_loss": history[0]["loss"] if history else None,
        "last_loss": history[-1]["loss"] if history else None,
        "history": history[-8:],
    }
    save_dir = Path(args.save_dir or f"models/{args.name}")
    save_dir.mkdir(parents=True, exist_ok=True)
    sidecar_path = save_dir / f"{args.name}_last.json"
    sidecar_path.write_text(json.dumps(sidecar, indent=2))
    backend.save_trained(str(save_dir / f"{args.name}_lora"))
    print(f"wrote {sidecar_path}")
    return sidecar


def main(argv: list[str] | None = None) -> dict:
    args = parse_args(argv)
    return train(args)


if __name__ == "__main__":
    main()
