"""20s trip-hop demo: TF only, LM only, both, then split the axes.

Same lyrics and seed. Neutral caption on slider clips.
"""

from __future__ import annotations

import json
import os
import sys
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

ROOT = _REPO_ROOT
DURATION = 20.0
SEED = 7


def _wrap(host, replace, prefix, rank, alpha, device, weights: Path) -> LoRANetwork:
    network = LoRANetwork(
        host,
        rank=rank,
        alpha=alpha,
        multiplier=1.0,
        target_replace=replace,
        train_method="full",
        delimiter="-",
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
    if missing or not network.unet_loras:
        raise RuntimeError(f"bad LoRA {weights} missing={len(missing)}")
    for lora in network.unet_loras:
        lora.multiplier = 0.0
    return network


def main() -> None:
    tf_path = ROOT / "models/triphop-slider/triphop_alpha8.0_rank8_full_last.safetensors"
    lm_path = ROOT / "models/triphop-lm-slider/triphop-lm_last.safetensors"
    row = _load_prompt_row(ROOT / "conceptmod/textsliders/data/prompts-triphop.yaml")
    lyrics = str(row["lyrics"])
    neutral = str(row.get("neutral") or row["target"])
    out_dir = ROOT / "eval/listen/triphop-20s"
    out_dir.mkdir(parents=True, exist_ok=True)

    tf_meta = _sidecar(tf_path)
    unit = float(tf_meta["unit_scale"]) if tf_meta.get("calibrated") else 2.0
    tf_amp = unit  # user +1 in calibrated units
    lm_amp = 2.0
    print(f"TF amplitude: unit_scale={unit:g} (one trained concept)", flush=True)

    device = "cuda:0"
    pipe = _load_pipeline(DEFAULT_MODEL_DIR, device)
    tf_net = _wrap(
        pipe.transformer, TRANSFORMER_REPLACE, "lora_unet", 8, 8.0, device, tf_path
    )
    lm_net = _wrap(
        pipe.language_model, LM_REPLACE, "lora_te", 8, 8.0, device, lm_path
    )
    sample_rate = int(pipe.sampling_rate)

    # Separate each axis, then combine, then pull them apart.
    jobs = [
        ("01_SEPARATE_base.wav", 0.0, 0.0, neutral, "no sliders"),
        ("02_SEPARATE_TF_triphop.wav", 0.0, +tf_amp, neutral, f"transformer only +{tf_amp:g}"),
        ("03_SEPARATE_TF_glossypop.wav", 0.0, -tf_amp, neutral, f"transformer only -{tf_amp:g}"),
        ("04_SEPARATE_LM_triphop.wav", +lm_amp, 0.0, neutral, f"LM only +{lm_amp:g}"),
        ("05_SEPARATE_LM_glossypop.wav", -lm_amp, 0.0, neutral, f"LM only -{lm_amp:g}"),
        ("06_COMBINED_both_triphop.wav", +lm_amp, +tf_amp, neutral, "LM + transformer trip-hop"),
        ("07_SPLIT_TFtriphop_LMpop.wav", -lm_amp, +tf_amp, neutral, "TF trip-hop mix, LM glossy plan"),
        ("08_SPLIT_TFpop_LMtriphop.wav", +lm_amp, -tf_amp, neutral, "LM trip-hop plan, TF glossy mix"),
        ("09_REF_prompt_triphop.wav", 0.0, 0.0, str(row["positive"]), "no slider, trip-hop prompt"),
        ("10_REF_prompt_glossypop.wav", 0.0, 0.0, str(row["negative"]), "no slider, glossy-pop prompt"),
    ]

    lines = [
        "# trip-hop — separate the two sliders, then combine them",
        "",
        "Same lyrics and seed. Slider clips use the **neutral** caption.",
        f"Transformer amplitude is calibrated unit_scale={unit:g} (one trained concept).",
        f"LM amplitude is {lm_amp:g} (same as the working gender LM slider).",
        "",
        "| file | sec | rms | setup |",
        "|------|----:|----:|-------|",
    ]

    for name, lm_s, tf_s, prompt, note in jobs:
        dest = out_dir / name
        ok, _reason, duration, rms = _accept_wav(dest, DURATION)
        if ok:
            print(f"skip {name} {duration:.2f}s rms={rms:.4f}", flush=True)
            lines.append(f"| `{name}` | {duration:.2f} | {rms:.4f} | {note} |")
            continue

        for lora in lm_net.unet_loras:
            lora.multiplier = 0.0
        for lora in tf_net.unet_loras:
            lora.multiplier = 0.0
        if lm_s:
            lm_net.set_lora_slider(lm_s)
        if tf_s:
            tf_net.set_lora_slider(tf_s)

        print(f"{name} lm={lm_s:g} tf={tf_s:g} {note}", flush=True)
        generator = torch.Generator(device).manual_seed(SEED)

        class _Stack:
            def __enter__(self_inner):
                if lm_s:
                    lm_net.__enter__()
                if tf_s:
                    tf_net.__enter__()
                return self_inner

            def __exit__(self_inner, *exc):
                if tf_s:
                    tf_net.__exit__(*exc)
                if lm_s:
                    lm_net.__exit__(*exc)
                return False

        with _Stack():
            audio = pipe(
                prompt=prompt,
                lyrics=lyrics,
                audio_duration=DURATION,
                generator=generator,
                output="audios",
            )[0]
        duration, rms = _write_wav(dest, audio, sample_rate, DURATION)
        lines.append(f"| `{name}` | {duration:.2f} | {rms:.4f} | {note} |")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    lines.extend(
        [
            "",
            "Play **01 → 02 → 04 → 06**. That is: dry, TF mix only, LM plan only, both.",
            "Then **07 / 08** to hear the axes pulled apart.",
            "Then **09** as the prompt-only trip-hop ceiling.",
            "",
            f"- duration: {DURATION}s  seed: {SEED}",
            f"- tf weights: `{tf_path}`",
            f"- lm weights: `{lm_path}`",
            "",
        ]
    )
    readme = out_dir / "LISTEN.md"
    readme.write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {readme}", flush=True)


if __name__ == "__main__":
    main()
