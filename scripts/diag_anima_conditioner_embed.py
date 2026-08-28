#!/usr/bin/env python3
"""Diagnose Anima conditioner LoRA: student Δ vs teacher Δ, sample vs train.

CPU dummy is the default (no Hub, no GPU). Live GPU path loads Base plus an
optional v6 ``*_conditioner_lora`` adapter (or a freshly attached PEFT
wrapper) and measures the same embed geometry.

For each neu/plus pair:

- ``||E(neu, s) − E(neu, 0)||`` vs ``||E(plus, 0) − E(neu, 0)||``
- ``cosine(E(neu, s)−E(neu, 0), E(plus, 0)−E(neu, 0))``

PASS at scale 1: student Δ is not near-zero and is aligned with the
frozen plus−neu teacher Δ. FAIL means the adapter is not in the encode
graph (apply/sync) or the weights do not move the concept.

If Modular/sample encode ≠ ``backend.encode_text``, prints
``SAMPLE_TRAIN_MISMATCH``.

    PYTHONPATH=. python scripts/diag_anima_conditioner_embed.py
    PYTHONPATH=. python scripts/diag_anima_conditioner_embed.py --dummy
    HF_HUB_OFFLINE=1 PYTHONPATH=. python scripts/diag_anima_conditioner_embed.py \\
      --live --device cuda:0 \\
      --model_id circlestone-labs/Anima-Base-v1.0-Diffusers \\
      --conditioner_adapter models/smile-anima-v6/smile-anima_conditioner_lora

Does **not** start a smile retrain.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from conceptmod.textsliders.anima_fake import FakeAnimaBackend
from conceptmod.textsliders.anima_peft_sync import (
    SAMPLE_TRAIN_MISMATCH,
    adapter_scale_api,
    assert_sample_train_conditioner_shared,
    compare_sample_train_embeds,
    measure_conditioner_embed_deltas,
    sample_conditioner_module,
    train_conditioner_module,
)
from conceptmod.textsliders.anima_slider import (
    MAN_NEU,
    MAN_PLUS,
    WOMAN_NEU,
    WOMAN_PLUS,
)
from conceptmod.textsliders.train_lora_anima import (
    peft_adapter_scale,
    sync_backend_peft_modules,
)


def _perturb_dummy_lora(backend: FakeAnimaBackend) -> None:
    """Zero-init B would make scale-1 Δ identically 0.

    Write a plus-aligned rank-1 update: unused-axis (0) → concept-axis (1)
    so dummy student Δ(neu, 1) lines up with frozen plus−neu.
    """
    import torch

    for lora in backend.loras:
        with torch.no_grad():
            lora.down.weight.zero_()
            lora.up.weight.zero_()
            if lora.down.weight.shape[1] > 0 and lora.down.weight.shape[0] > 0:
                lora.down.weight[0, 0] = 1.0
            if lora.up.weight.shape[0] > 1 and lora.up.weight.shape[1] > 0:
                lora.up.weight[1, 0] = 1.0


def _encode_at_scale(backend, prompt: str, scale: float):
    import torch

    with torch.no_grad(), peft_adapter_scale(backend, float(scale)):
        embeds, _tokens = backend.encode_text(prompt)
    return embeds


def _sample_encode(backend, prompt: str, scale: float):
    """Encode through the sample-path conditioner object."""
    import torch

    pipe = getattr(backend, "pipe", None)
    encode = getattr(pipe, "encode_prompt_embeds", None)
    with torch.no_grad(), peft_adapter_scale(backend, float(scale)):
        if callable(encode):
            return encode(prompt)
        embeds, _tokens = backend.encode_text(prompt)
        return embeds


def _pairs_from_args(args) -> list[tuple[str, str, str]]:
    return [
        ("woman", args.neu or WOMAN_NEU, args.plus or WOMAN_PLUS),
        ("man", MAN_NEU, MAN_PLUS),
    ]


def _print_report(report: dict) -> None:
    print(json.dumps(report, indent=2))
    print()
    print(f"=== {report['verdict']} ===")
    print(report["reason"])
    if report.get("mismatch_flag"):
        print(report["mismatch_flag"])
        print(report.get("mismatch_reason", ""))


def run_dummy(args) -> dict:
    backend = FakeAnimaBackend(
        device="cpu",
        rank=int(args.rank),
        seed=int(args.seed),
        lora_targets="conditioner",
    )
    sync_backend_peft_modules(backend)
    shared = assert_sample_train_conditioner_shared(backend)
    _perturb_dummy_lora(backend)
    cond = train_conditioner_module(backend)
    apis = adapter_scale_api(cond) or ["set_lora_scale(backend)"]
    pair_reports = []
    mismatch = None
    for label, neu, plus in _pairs_from_args(args):
        measured = measure_conditioner_embed_deltas(
            lambda prompt, scale, b=backend: _encode_at_scale(b, prompt, scale),
            neu=neu,
            plus=plus,
        )
        train_e = _encode_at_scale(backend, neu, 1.0)
        sample_e = _sample_encode(backend, neu, 1.0)
        cmp = compare_sample_train_embeds(train_e, sample_e)
        measured["pair"] = label
        measured["sample_train"] = cmp
        pair_reports.append(measured)
        if not cmp["match"] and mismatch is None:
            mismatch = cmp
    overall = "PASS"
    reasons = []
    for row in pair_reports:
        if row["verdict"] != "PASS":
            overall = "FAIL"
            reasons.append(f"{row['pair']}: {row['reason']}")
    if mismatch is not None:
        overall = "FAIL"
        reasons.append(mismatch["reason"])
    report = {
        "dummy": True,
        "live": False,
        "lora_targets": "conditioner",
        "shared_conditioner_id": shared["id"],
        "sample_conditioner_id": id(sample_conditioner_module(backend)),
        "train_conditioner_id": id(train_conditioner_module(backend)),
        "adapter_scale_api": apis,
        "pairs": pair_reports,
        "verdict": overall,
        "reason": "; ".join(reasons) if reasons else pair_reports[0]["reason"],
        "mismatch_flag": mismatch["flag"] if mismatch else None,
        "mismatch_reason": mismatch["reason"] if mismatch else None,
    }
    return report


def run_live(args) -> dict:
    from conceptmod.textsliders.train_lora_anima import load_live_backend, parse_args

    argv = [
        "--model_id",
        str(args.model_id),
        "--lora_targets",
        "conditioner",
        "--rank",
        str(args.rank),
        "--device",
        str(args.device),
        "--resolution",
        str(args.resolution),
    ]
    if args.allow_hub:
        argv.append("--allow_hub")
    if args.conditioner_adapter:
        argv.extend(["--conditioner_adapter", str(args.conditioner_adapter)])
    ns = parse_args(argv)
    backend = load_live_backend(ns, backend_device(args.device))
    sync_backend_peft_modules(backend)
    shared = assert_sample_train_conditioner_shared(backend)
    cond = train_conditioner_module(backend)
    apis = adapter_scale_api(cond)
    pair_reports = []
    mismatch = None
    for label, neu, plus in _pairs_from_args(args)[:1]:
        measured = measure_conditioner_embed_deltas(
            lambda prompt, scale, b=backend: _encode_at_scale(b, prompt, scale),
            neu=neu,
            plus=plus,
        )
        train_e = _encode_at_scale(backend, neu, 1.0)
        sample_e = _sample_encode(backend, neu, 1.0)
        cmp = compare_sample_train_embeds(train_e, sample_e)
        measured["pair"] = label
        measured["sample_train"] = cmp
        pair_reports.append(measured)
        if not cmp["match"] and mismatch is None:
            mismatch = cmp
    overall = "PASS"
    reasons = []
    for row in pair_reports:
        if row["verdict"] != "PASS":
            overall = "FAIL"
            reasons.append(f"{row['pair']}: {row['reason']}")
    if mismatch is not None:
        overall = "FAIL"
        reasons.append(mismatch["reason"])
    if not apis:
        overall = "FAIL"
        reasons.append("set_adapter_scale missing on AnimaTextConditioner PEFT wrapper")
    report = {
        "dummy": False,
        "live": True,
        "model_id": args.model_id,
        "conditioner_adapter": args.conditioner_adapter,
        "shared_conditioner_id": shared["id"],
        "sample_conditioner_id": id(sample_conditioner_module(backend)),
        "train_conditioner_id": id(train_conditioner_module(backend)),
        "adapter_scale_api": apis,
        "pairs": pair_reports,
        "verdict": overall,
        "reason": "; ".join(reasons) if reasons else pair_reports[0]["reason"],
        "mismatch_flag": mismatch["flag"] if mismatch else None,
        "mismatch_reason": mismatch["reason"] if mismatch else None,
        "revalidate": (
            "After merge, one short GPU smoke: load Base + this v6 adapter "
            "with this script (not a 500-step retrain). If verdict is PASS "
            "and SAMPLE_TRAIN_MISMATCH is absent, re-run the in-process "
            "0/0.25/0.5/1.0 sample grid only."
        ),
    }
    return report


def backend_device(arg: str):
    import torch

    if str(arg).startswith("cuda") and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(arg)


def parse_diag_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dummy", action="store_true", default=True)
    parser.add_argument("--live", action="store_true", help="load local Anima Base")
    parser.add_argument("--allow_hub", action="store_true")
    parser.add_argument(
        "--model_id",
        type=str,
        default="circlestone-labs/Anima-Base-v1.0-Diffusers",
    )
    parser.add_argument("--conditioner_adapter", type=str, default=None)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--rank", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--neu", type=str, default=None)
    parser.add_argument("--plus", type=str, default=None)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> dict:
    args = parse_diag_args(argv)
    if args.live:
        report = run_live(args)
    else:
        report = run_dummy(args)
    _print_report(report)
    if report["verdict"] != "PASS":
        raise SystemExit(2)
    if report.get("mismatch_flag") == SAMPLE_TRAIN_MISMATCH:
        raise SystemExit(3)
    return report


if __name__ == "__main__":
    main()
