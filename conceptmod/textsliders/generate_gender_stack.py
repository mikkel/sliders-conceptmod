"""A/B slider stack for any axis: LM only, hotter LM, transformer only, both.

Same lyrics, same seed, neutral caption. Play the folder in filename order.
Rank/alpha/targets/prefix and pole labels come from each weight's sidecar JSON;
--plus_label/--minus_label override.
"""

from __future__ import annotations

import argparse
import os
import sys
from contextlib import contextmanager
from pathlib import Path

_HF_HOME = "/ml2/music/.cache/huggingface"
os.environ["HF_HOME"] = _HF_HOME
os.environ["HUGGINGFACE_HUB_CACHE"] = f"{_HF_HOME}/hub"
os.environ["HF_HUB_CACHE"] = f"{_HF_HOME}/hub"
os.environ["TRANSFORMERS_CACHE"] = f"{_HF_HOME}/hub"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

import torch
from safetensors.torch import load_file

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from conceptmod.textsliders.generate_listen import (  # noqa: E402
    DEFAULT_MODEL_DIR,
    LM_REPLACE,
    TRANSFORMER_REPLACE,
    _accept_wav,
    _load_prompt_row,
    _sidecar,
    _write_wav,
)
from conceptmod.textsliders.infer_music3 import _load_pipeline  # noqa: E402
from conceptmod.textsliders.lora import LoRANetwork  # noqa: E402


def _wrap_sidecar(pipe, device, weights: Path, fallback_kind: str) -> tuple[LoRANetwork, dict]:
    """Build a network for one weights file, all params from its sidecar JSON."""
    meta = _sidecar(weights)
    kind = str(meta.get("kind") or fallback_kind)
    if kind == "language_model":
        host = pipe.language_model
        replace = list(meta.get("target_replace") or LM_REPLACE)
        prefix = str(meta.get("prefix") or "lora_te")
    else:
        host = pipe.transformer
        replace = list(meta.get("target_replace") or TRANSFORMER_REPLACE)
        prefix = str(meta.get("prefix") or "lora_unet")
    network = _wrap(
        host,
        replace,
        prefix,
        int(meta.get("rank", 8)),
        float(meta.get("alpha", 8.0)),
        device,
        weights,
        train_method=str(meta.get("train_method") or "full"),
        delimiter=str(meta.get("delimiter") or "-"),
    )
    return network, meta


def _wrap(host, replace, prefix, rank, alpha, device, weights: Path, train_method="full", delimiter="-") -> LoRANetwork:
    network = LoRANetwork(
        host,
        rank=rank,
        alpha=alpha,
        multiplier=1.0,
        target_replace=replace,
        train_method=train_method,
        delimiter=delimiter,
        prefix=prefix,
    )
    network.to(device)
    state = load_file(str(weights), device="cpu")
    missing, unexpected = network.load_state_dict(state, strict=False)
    print(
        f"loaded {weights.name} modules={len(network.unet_loras)} "
        f"missing={len(missing)} unexpected={len(unexpected)}",
        flush=True,
    )
    if len(network.unet_loras) == 0:
        raise RuntimeError(f"LoRA wrapped 0 modules for {replace}")
    if missing:
        raise RuntimeError(f"LoRA load missed {len(missing)} keys (first={missing[0]})")
    for lora in network.unet_loras:
        lora.multiplier = 0.0
    return network


@contextmanager
def _apply(*pairs: tuple[LoRANetwork | None, float]):
    active = []
    try:
        for network, scale in pairs:
            if network is None:
                continue
            network.set_lora_slider(float(scale))
            network.__enter__()
            active.append(network)
        yield
    finally:
        for network in reversed(active):
            network.__exit__(None, None, None)


def generate(args: argparse.Namespace) -> list[Path]:
    row = _load_prompt_row(Path(args.prompts_file))
    lyrics = str(row.get("lyrics") or "")
    neutral = str(row.get("neutral") or row["target"])
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    requested = float(args.duration)
    device = "cuda:0"
    pipe = _load_pipeline(Path(args.model_dir), device)
    lm_net, lm_meta = _wrap_sidecar(pipe, device, Path(args.lm_weights), "language_model")
    tf_net, tf_meta = _wrap_sidecar(pipe, device, Path(args.tf_weights), "transformer")
    plus = args.plus_label or str(tf_meta.get("plus_label") or lm_meta.get("plus_label") or "plus")
    minus = args.minus_label or str(tf_meta.get("minus_label") or lm_meta.get("minus_label") or "minus")
    # TF scales are in calibrated user units when the sidecar has unit_scale.
    tf_unit = float(tf_meta.get("unit_scale") or 1.0) if tf_meta.get("calibrated") else 1.0
    sample_rate = int(pipe.sampling_rate)
    seed = int(args.seed)
    amp = float(args.amp)
    hot = float(args.hot_amp)

    # Easy names: what is on, then the intended pole.
    jobs = [
        (f"01_LMonly_{minus}_minus{amp:g}.wav", -amp, 0.0, neutral),
        ("02_LMonly_neutral_zero.wav", 0.0, 0.0, neutral),
        (f"03_LMonly_{plus}_plus{amp:g}.wav", amp, 0.0, neutral),
        (f"04_LMhot_{minus}_minus{hot:g}.wav", -hot, 0.0, neutral),
        (f"05_LMhot_{plus}_plus{hot:g}.wav", hot, 0.0, neutral),
        (f"06_BOTH_{minus}_minus{amp:g}.wav", -amp, -amp * tf_unit, neutral),
        (f"07_BOTH_{plus}_plus{amp:g}.wav", amp, amp * tf_unit, neutral),
        (f"08_TFonly_{minus}_minus{amp:g}.wav", 0.0, -amp * tf_unit, neutral),
        (f"09_TFonly_{plus}_plus{amp:g}.wav", 0.0, amp * tf_unit, neutral),
        (f"10_REF_prompt_{plus}_no_slider.wav", 0.0, 0.0, str(row["positive"])),
        (f"11_REF_prompt_{minus}_no_slider.wav", 0.0, 0.0, str(row["negative"])),
    ]

    written: list[Path] = []
    stats: list[str] = []
    for name, lm_scale, tf_scale, prompt in jobs:
        dest = out_dir / name
        ok, reason, duration, rms = _accept_wav(dest, requested)
        if ok and not args.force:
            print(f"skip existing {name} duration={duration:.2f}s rms={rms:.4f}", flush=True)
            stats.append(f"| `{name}` | {duration:.2f} | {rms:.4f} | reused |")
            written.append(dest)
            continue
        print(f"{name} lm={lm_scale:g} tf={tf_scale:g} prompt={prompt[:64]!r}", flush=True)
        generator = torch.Generator(device).manual_seed(seed)
        with _apply((lm_net, lm_scale), (tf_net, tf_scale)):
            audio = pipe(
                prompt=prompt,
                lyrics=lyrics,
                audio_duration=requested,
                generator=generator,
                output="audios",
            )[0]
        duration, rms = _write_wav(dest, audio, sample_rate, requested)
        stats.append(f"| `{name}` | {duration:.2f} | {rms:.4f} | lm={lm_scale:g} tf={tf_scale:g} |")
        written.append(dest)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    readme = out_dir / "LISTEN.md"
    readme.write_text(
        "\n".join(
            [
                f"# {minus} <-> {plus} stack — LM vs transformer vs both",
                "",
                "Same lyrics and seed. Slider clips use the **neutral** caption.",
                f"Compare 01/03 (LM ±{amp:g}) to 04/05 (LM ±{hot:g}) to 06/07 (both) to 08/09 (transformer only).",
                "",
                "| file | seconds | rms | setup |",
                "|------|--------:|----:|-------|",
                *stats,
                "",
                f"- lm weights: `{args.lm_weights}`",
                f"- transformer weights: `{args.tf_weights}`",
                f"- duration: {requested}s  seed: {seed}",
                "",
            ]
        ),
        encoding="utf-8",
    )
    print(f"wrote {readme}", flush=True)
    return written


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--lm_weights", required=True)
    p.add_argument("--tf_weights", required=True)
    p.add_argument("--prompts_file", required=True)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--duration", type=float, default=8.0)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--model_dir", default=str(DEFAULT_MODEL_DIR))
    p.add_argument("--force", action="store_true")
    p.add_argument("--plus_label", default=None, help="override sidecar plus_label")
    p.add_argument("--minus_label", default=None, help="override sidecar minus_label")
    p.add_argument("--amp", type=float, default=2.0, help="user-scale amplitude for the main clips")
    p.add_argument("--hot_amp", type=float, default=3.0, help="user-scale amplitude for the LM-hot clips")
    return p.parse_args()


if __name__ == "__main__":
    generate(parse_args())
