"""24s gender probe: TF LoRA vs condition-mean swap vs LM vs refs.

Why TF-only sounds 'louder/deeper': the flow transformer still sees the
neutral AR condition concatenated every step. This script also morphs that
condition (the actual identity channel) and compares to the working LM slider.
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

import numpy as np
import soundfile as sf
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
    _write_wav,
)
from conceptmod.textsliders.infer_music3 import _load_pipeline  # noqa: E402
from conceptmod.textsliders.lora import LoRANetwork  # noqa: E402

ROOT = _REPO_ROOT
DURATION = 24.0
SEED = 7


def _load_cond_means(cache_dir: Path) -> dict[str, torch.Tensor]:
    means: dict[str, torch.Tensor] = {}
    for path in cache_dir.glob("*.pt"):
        payload = torch.load(path, map_location="cpu", weights_only=False)
        cond = payload["condition"].float() if isinstance(payload, dict) else payload.float()
        prompt = str(payload.get("prompt", "")) if isinstance(payload, dict) else ""
        lm_scale = float(payload.get("lm_scale", 0.0)) if isinstance(payload, dict) else 0.0
        if lm_scale > 0.5:
            key = "pos"
        elif lm_scale < -0.5:
            key = "neg"
        else:
            low = prompt.lower()
            if "soprano" in low or "adult woman" in low or "pop-punk" in low:
                key = "pos"
            elif "baritone" in low or "adult man" in low or "lullaby" in low:
                key = "neg"
            else:
                key = "neu"
        means[key] = cond.mean(dim=1, keepdim=True)  # (1, 1, 2048)
        print(f"  cache {path.name} -> {key} mean_norm={means[key].norm().item():.3f} lm={lm_scale:g} {prompt[:50]!r}")
    return means


def _wrap_lora(host, replace, prefix, rank, alpha, device, weights: Path) -> LoRANetwork:
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
        raise RuntimeError(f"LoRA load failed for {weights}")
    for lora in network.unet_loras:
        lora.multiplier = 0.0
    return network


class ConditionHook:
    """Add a gender direction to whatever the live AR encoder produced."""

    def __init__(self, encoder, pos_mean: torch.Tensor, neg_mean: torch.Tensor):
        self.encoder = encoder
        self.orig = encoder.forward
        self.pos_mean = pos_mean
        self.neg_mean = neg_mean
        self.delta = pos_mean - neg_mean
        self.mode = "off"
        self.scale = 0.0

    def attach(self) -> None:
        hook = self

        def forward(hidden_states, *args, **kwargs):
            cond = hook.orig(hidden_states, *args, **kwargs)
            if hook.mode == "off" or hook.scale == 0.0:
                return cond
            delta = hook.delta.to(device=cond.device, dtype=cond.dtype)
            if hook.mode == "dir_add":
                # Broadcast a global identity direction onto every latent frame.
                return cond + (hook.scale * delta)
            if hook.mode == "mean_swap":
                # Keep this take's timing; rewrite the time-mean toward female/male.
                live = cond.mean(dim=1, keepdim=True)
                return cond - live + (live + hook.scale * delta)
            raise ValueError(hook.mode)

        self.encoder.forward = forward

    def set(self, mode: str, scale: float) -> None:
        self.mode = mode
        self.scale = float(scale)


def _median_f0(path: Path, sample_rate: int) -> float:
    try:
        import torchaudio

        wav, sr = sf.read(str(path), always_2d=True)
        mono = torch.from_numpy(wav.mean(axis=1).astype(np.float32))
        if sr != sample_rate:
            sample_rate = sr
        freq = torchaudio.functional.detect_pitch_frequency(mono.unsqueeze(0), sample_rate).squeeze()
        freq = freq[freq > 50]
        freq = freq[freq < 500]
        if freq.numel() == 0:
            return 0.0
        return float(freq.median())
    except Exception as exc:  # noqa: BLE001
        print(f"f0 failed {path.name}: {exc}", flush=True)
        return 0.0


def main() -> None:
    prompts = ROOT / "conceptmod/textsliders/data/prompts-gender-tf.yaml"
    row = _load_prompt_row(prompts)
    lyrics = str(row["lyrics"])
    neutral = str(row.get("neutral") or row["target"])
    out_dir = ROOT / "eval/listen/gender-24s"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("condition means from LM-boosted cache (stronger identity delta):")
    lm_means = _load_cond_means(ROOT / "cache/gender-tf-v2")
    print("condition means from caption cache:")
    cap_means = _load_cond_means(ROOT / "cache/gender")
    # Prefer LM-boosted pos/neg; they come from the working gender LM slider.
    pos_mean = lm_means.get("pos", cap_means["pos"])
    neg_mean = lm_means.get("neg", cap_means["neg"])
    delta_n = (pos_mean - neg_mean).norm().item()
    print(f"using delta ||pos-neg||={delta_n:.3f}", flush=True)

    device = "cuda:0"
    pipe = _load_pipeline(DEFAULT_MODEL_DIR, device)
    hook = ConditionHook(pipe.condition_encoder, pos_mean, neg_mean)
    hook.attach()

    lm_net = _wrap_lora(
        pipe.language_model,
        LM_REPLACE,
        "lora_te",
        8,
        8.0,
        device,
        ROOT / "models/gender-lm-slider/gender-lm_last.safetensors",
    )
    tf_net = _wrap_lora(
        pipe.transformer,
        TRANSFORMER_REPLACE,
        "lora_unet",
        8,
        8.0,
        device,
        ROOT / "models/gender-slider/gender_alpha8.0_rank8_full_last.safetensors",
    )
    sample_rate = int(pipe.sampling_rate)

    jobs = [
        ("01_BASE_neutral.wav", 0.0, 0.0, "off", 0.0, neutral),
        ("02_TFlora_male_minus2.wav", 0.0, -2.0, "off", 0.0, neutral),
        ("03_TFlora_female_plus2.wav", 0.0, 2.0, "off", 0.0, neutral),
        ("04_CONDmean_male_minus2.wav", 0.0, 0.0, "mean_swap", -2.0, neutral),
        ("05_CONDmean_female_plus2.wav", 0.0, 0.0, "mean_swap", 2.0, neutral),
        ("06_CONDdir_male_minus2.wav", 0.0, 0.0, "dir_add", -2.0, neutral),
        ("07_CONDdir_female_plus2.wav", 0.0, 0.0, "dir_add", 2.0, neutral),
        ("08_LM_male_minus2.wav", -2.0, 0.0, "off", 0.0, neutral),
        ("09_LM_female_plus2.wav", 2.0, 0.0, "off", 0.0, neutral),
        ("10_LMplusCOND_male.wav", -2.0, 0.0, "mean_swap", -2.0, neutral),
        ("11_LMplusCOND_female.wav", 2.0, 0.0, "mean_swap", 2.0, neutral),
        ("12_REF_prompt_female.wav", 0.0, 0.0, "off", 0.0, str(row["positive"])),
        ("13_REF_prompt_male.wav", 0.0, 0.0, "off", 0.0, str(row["negative"])),
    ]

    lines = [
        "# 24s gender methods — why TF-only is just louder/deeper",
        "",
        "Same lyrics and seed. Slider clips use the **neutral** caption.",
        "",
        "| file | sec | rms | F0 Hz | method |",
        "|------|----:|----:|------:|--------|",
    ]
    for name, lm_s, tf_s, cond_mode, cond_s, prompt in jobs:
        dest = out_dir / name
        ok, reason, duration, rms = _accept_wav(dest, DURATION)
        if ok:
            f0 = _median_f0(dest, sample_rate)
            print(f"skip {name} duration={duration:.2f}s rms={rms:.4f} f0={f0:.1f}", flush=True)
            lines.append(f"| `{name}` | {duration:.2f} | {rms:.4f} | {f0:.1f} | reused |")
            continue

        for lora in lm_net.unet_loras:
            lora.multiplier = 0.0
        for lora in tf_net.unet_loras:
            lora.multiplier = 0.0
        hook.set("off", 0.0)

        if lm_s != 0.0:
            lm_net.set_lora_slider(lm_s)
        if tf_s != 0.0:
            tf_net.set_lora_slider(tf_s)
        hook.set(cond_mode, cond_s)

        print(
            f"{name} lm={lm_s:g} tf={tf_s:g} cond={cond_mode}:{cond_s:g} prompt={prompt[:48]!r}",
            flush=True,
        )
        generator = torch.Generator(device).manual_seed(SEED)

        class _Stack:
            def __enter__(self_inner):
                if lm_s != 0.0:
                    lm_net.__enter__()
                if tf_s != 0.0:
                    tf_net.__enter__()
                return self_inner

            def __exit__(self_inner, *exc):
                if tf_s != 0.0:
                    tf_net.__exit__(*exc)
                if lm_s != 0.0:
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
        f0 = _median_f0(dest, sample_rate)
        method = f"lm={lm_s:g} tf={tf_s:g} cond={cond_mode}:{cond_s:g}"
        lines.append(f"| `{name}` | {duration:.2f} | {rms:.4f} | {f0:.1f} | {method} |")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    lines.extend(
        [
            "",
            "Play **08/09 (LM)** first — that is the working identity slider.",
            "Then **02/03 (TF LoRA)** — expect louder/deeper, same singer.",
            "Then **04/05 (condition mean-swap)** — rewrite identity in the 2048-d condition.",
            "Then **10/11** — LM plus condition rewrite.",
            "",
            f"- duration: {DURATION}s  seed: {SEED}",
            f"- cond delta L2: {delta_n:.3f}",
            "",
        ]
    )
    readme = out_dir / "LISTEN.md"
    readme.write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {readme}", flush=True)


if __name__ == "__main__":
    main()
