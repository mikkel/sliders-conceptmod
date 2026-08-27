#!/usr/bin/env python3
"""Opt-in H3 image-slider trainer (UNI on AR encode). Music 3 unchanged.

Live card: ``tencent/HunyuanImage-3.0`` (AR/MoE). Not a velocity DiT.
``--dummy`` is the CI / CPU path: no Hub, no GPU, no H3 weights.
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

from conceptmod.backends.h3 import DEFAULT_MODEL, H3Backend
from conceptmod.textsliders.h3_uni import (
    concept_token_ids,
    h3_minus_canary,
    h3_uni_total_loss,
    last_hidden,
    pin_unused_attributes,
    unused_hold_mask,
    unused_token_ids,
)

DEFAULT_PROMPTS = Path(__file__).resolve().parent / "data" / "prompts-h3.yaml"
DEFAULT_CONFIG = Path(__file__).resolve().parent / "data" / "config-h3.yaml"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Opt-in H3 UNI image-slider trainer")
    p.add_argument("--name", type=str, default="age-h3-uni")
    p.add_argument("--backend", type=str, default="h3", choices=["h3"])
    p.add_argument("--model_id", type=str, default=DEFAULT_MODEL)
    p.add_argument("--prompts_file", type=str, default=str(DEFAULT_PROMPTS))
    p.add_argument("--config_file", type=str, default=str(DEFAULT_CONFIG))
    p.add_argument("--attributes", type=str, default="",
                   help="comma-separated unused attributes pinned on both poles")
    p.add_argument("--rank", type=int, default=8)
    p.add_argument("--alpha", type=float, default=8.0)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--steps", type=int, default=500)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--resolution", type=int, default=1024)
    p.add_argument("--hold_weight", type=float, default=1.0)
    p.add_argument("--save_dir", type=str, default=None)
    p.add_argument(
        "--dummy",
        action="store_true",
        help="CPU mock pipe; no Hub, no GPU, no H3 weights",
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
                "unconditional": str(item.get("unconditional") or ""),
                "target": str(item.get("target") or neu),
                "attributes": uniq,
            })
    if not rows:
        raise ValueError(f"no slider rows in {prompts_file}")
    return rows


def build_backend(args: argparse.Namespace) -> H3Backend:
    if args.dummy:
        return H3Backend(
            device="cpu",
            model_id=args.model_id,
            resolution=min(int(args.resolution), 64),
            lora_rank=args.rank,
            dummy=True,
        )
    return H3Backend(
        device=args.device,
        model_id=args.model_id,
        resolution=args.resolution,
        lora_rank=args.rank,
        dummy=False,
    )


@torch.no_grad()
def _canary(backend: H3Backend, row: dict) -> float:
    pred = last_hidden(backend.encode_scaled(row["neutral"], scale=-1.0).embeds)
    tgt = last_hidden(backend.encode_text(row.get("unconditional") or "", frozen=True).embeds)
    return float(h3_minus_canary(pred, tgt).item())


def train(args: argparse.Namespace, backend: H3Backend | None = None) -> dict:
    rows = load_slider_rows(args.prompts_file, args.attributes)
    backend = backend or build_backend(args)
    tokenizer = backend.pipe.tokenizer
    params = backend.trainable_parameters("lora")
    opt = torch.optim.Adam(params, lr=float(args.lr))
    torch.manual_seed(int(args.seed))

    history = []
    for step in range(int(args.steps)):
        row = rows[step % len(rows)]
        tgt_plus = last_hidden(backend.encode_text(row["positive"], frozen=True).embeds)
        tgt_zero = last_hidden(backend.encode_text(row["neutral"], frozen=True).embeds)
        student_plus = backend.encode_scaled(row["neutral"], scale=1.0)
        student_zero = backend.encode_scaled(row["neutral"], scale=0.0)
        pred_plus = last_hidden(student_plus.embeds)
        pred_zero = last_hidden(student_zero.embeds)

        ids = tokenizer.encode(row["neutral"], add_special_tokens=False)
        concept = concept_token_ids(tokenizer, row["positive"], row["neutral"])
        unused = unused_token_ids(tokenizer, row["attributes"])
        hold_mask = unused_hold_mask(ids, unused, concept)
        neu_embeds = backend.encode_text(row["neutral"], frozen=True).embeds

        loss = h3_uni_total_loss(
            pred_plus, tgt_plus, pred_zero, tgt_zero,
            student_plus.embeds, neu_embeds, hold_mask,
            hold_weight=float(args.hold_weight),
        )
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        canary = _canary(backend, row)
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
                f"h3 uni step {step}: loss={rec['loss']:.4f} "
                f"minus_canary={canary:.4f} (not in loss)"
            )

    sidecar = {
        "name": args.name,
        "backend": "h3",
        "model_id": args.model_id,
        "resolved_model_id": DEFAULT_MODEL,
        "stack": "autoregressive_moe",
        "recipe": "h3_uni_encode",
        "plus_neu": True,
        "minus_teacher": False,
        "minus_canary": True,
        "hold": "unused_tokens_to_encode_neu",
        "hold_concept_words": False,
        "lora_only": True,
        "velocity_trainer": False,
        "rank": args.rank,
        "alpha": args.alpha,
        "lr": args.lr,
        "steps": args.steps,
        "seed": args.seed,
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
