"""In-distribution trip-hop: real concept prompt + small sliders.

No retraining. Caption matches the concept; LoRAs only nudge.
"""

from __future__ import annotations

import os
import shutil
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
    trip = str(row["positive"])
    pop = str(row["negative"])
    out_dir = ROOT / "eval/listen/triphop-indist"
    out_dir.mkdir(parents=True, exist_ok=True)

    unit = float(_sidecar(tf_path).get("unit_scale") or 1.82)
    tf_lo, tf_hi = 0.5 * unit, 1.0 * unit
    lm_lo, lm_hi = 0.5, 1.0
    print(
        f"in-dist TF unit={unit:g} -> lo={tf_lo:g} hi={tf_hi:g}; LM lo={lm_lo:g} hi={lm_hi:g}",
        flush=True,
    )

    # Same seed/duration/prompt as the previous REFs — reuse those takes.
    prev = ROOT / "eval/listen/triphop-20s"
    copies = {
        "01_TRIPprompt_base.wav": prev / "09_REF_prompt_triphop.wav",
        "08_POPprompt_base.wav": prev / "10_REF_prompt_glossypop.wav",
    }
    for dest_name, src in copies.items():
        dest = out_dir / dest_name
        if src.exists() and not dest.exists():
            shutil.copy2(src, dest)
            print(f"copied {src.name} -> {dest_name}", flush=True)

    device = "cuda:0"
    pipe = _load_pipeline(DEFAULT_MODEL_DIR, device)
    tf_net = _wrap(pipe.transformer, TRANSFORMER_REPLACE, "lora_unet", 8, 8.0, device, tf_path)
    lm_net = _wrap(pipe.language_model, LM_REPLACE, "lora_te", 8, 8.0, device, lm_path)
    sample_rate = int(pipe.sampling_rate)

    jobs = [
        ("01_TRIPprompt_base.wav", 0.0, 0.0, trip, "trip-hop prompt, sliders off (native)"),
        ("02_TRIPprompt_TF_plus05.wav", 0.0, tf_lo, trip, "trip-hop prompt + half-unit TF dust"),
        ("03_TRIPprompt_TF_plus1.wav", 0.0, tf_hi, trip, "trip-hop prompt + one-unit TF dust"),
        ("04_TRIPprompt_TF_minus05.wav", 0.0, -tf_lo, trip, "trip-hop prompt, TF pulled toward glossy mix"),
        ("05_TRIPprompt_LM_plus05.wav", lm_lo, 0.0, trip, "trip-hop prompt + small LM push"),
        ("06_TRIPprompt_both_plus05.wav", lm_lo, tf_lo, trip, "trip-hop prompt + half LM + half TF"),
        ("07_TRIPprompt_LMminus_TFplus.wav", -lm_lo, tf_lo, trip, "trip-hop prompt, LM back a bit, TF dust"),
        ("08_POPprompt_base.wav", 0.0, 0.0, pop, "glossy-pop prompt, sliders off (native)"),
        ("09_POPprompt_TF_plus05.wav", 0.0, tf_lo, pop, "pop prompt + half-unit TF dust"),
        ("10_POPprompt_TF_plus1.wav", 0.0, tf_hi, pop, "pop prompt + one-unit TF dust"),
        (
            "11_TRIPprompt_LM05_TFx1p5.wav",
            lm_lo,
            1.5 * unit,
            trip,
            "trip-hop prompt + LM 0.5 + TF 1.5 units",
        ),
        (
            "12_TRIPprompt_LM05_TFx2.wav",
            lm_lo,
            2.0 * unit,
            trip,
            "trip-hop prompt + LM 0.5 + TF 2 units",
        ),
    ]

    lines = [
        "# trip-hop in-distribution — prompt the concept, slide a little",
        "",
        "Same lyrics and seed as `triphop-20s/`. Bases are the native 09/10 takes.",
        f"TF half-unit = {tf_lo:.2f}, one-unit = {tf_hi:.2f}. LM half = {lm_lo:g}.",
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
            "Play **01 → 02 → 03 → 04** first (same trip-hop song, only the mix slider).",
            "Then **01 → 05 → 06** (tiny LM / both on that same prompt).",
            "Then **08 → 09 → 10** (real pop song, dusted).",
            "Then **01 → 11 → 12** (LM 0.5 + stronger TF).",
            "",
            f"- duration: {DURATION}s  seed: {SEED}",
            "",
        ]
    )
    readme = out_dir / "LISTEN.md"
    readme.write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {readme}", flush=True)


if __name__ == "__main__":
    main()
