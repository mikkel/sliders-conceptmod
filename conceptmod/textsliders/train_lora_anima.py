#!/usr/bin/env python3
"""Opt-in Anima image-slider trainer (UNI + unused-token hold).

Live card (documented by ``scripts/smoke_anima_slider.py``):

    model  circlestone-labs/Anima-Base-v1.0-Diffusers
    arch   2B Cosmos-Predict2 DiT, Qwen3+T5, Qwen-Image VAE
    lora   --lora_targets conditioner (smile default-on)
           AnimaTextConditioner q_proj/k_proj/v_proj/o_proj rank 16
           dit = old transformer-only to_q/to_k/to_v/to_out.0
           dit+conditioner = joint. Qwen3 text_encoder is not adapted.
    res    768 (4090 smile retrain: 512)
    sample 40 steps   CFG 4   lr 1e-4
    frozen text_encoder; conditioner frozen unless lora_targets includes it
    sample in-process PEFT pipe(prompt=...) at 0 / 0.25 / 0.5 / 1.0
    lm     --lm_target trajectory (K-step FlowMatch Euler; direct /
           cfg_delta kept). 1-step v-space gap is microscopic on Anima.
    traj   --traj_steps 4
    every  --sample_every 100 (end-of-train gate always runs)

Train and sample share bare infer/neu captions. Attributes are unused
pins only — never prefixed onto target / positive / neutral.

Default Music 3 trainer stays ``--lm_target v9`` / ``--pole_mode hidden``.
This file does not rewrite Music 3 yamls.

CI / this repo: pass ``--dummy``. Dummy never downloads Anima weights.
Live load is ``local_files_only`` / ``HF_HUB_OFFLINE=1`` unless
``--allow_hub`` is set.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable

os.environ.setdefault("HF_HUB_OFFLINE", "1")

import numpy as np
import torch
from tqdm.auto import tqdm

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from conceptmod.textsliders.anima_fake import FakeAnimaBackend, write_plus_alignment
from conceptmod.textsliders.anima_slider import (
    ANIMA_LM_TARGETS,
    ANIMA_LORA_TARGET_CHOICES,
    CONDITIONER_LORA_TARGETS,
    DEFAULT_CFG,
    DEFAULT_CONTROL_PROMPT,
    DEFAULT_HOLD_WEIGHT,
    DEFAULT_LM_TARGET,
    DEFAULT_LORA_TARGETS,
    DEFAULT_LR,
    DEFAULT_MODEL_ID,
    DEFAULT_RANK,
    DEFAULT_RESOLUTION,
    DEFAULT_SAMPLE_EVERY,
    DEFAULT_SAMPLE_SCALES,
    DEFAULT_SAMPLE_SEED,
    DEFAULT_SAMPLE_STEPS,
    DEFAULT_TEACHER_GAP_BOOST,
    DEFAULT_TRAJ_IDENTITY_WEIGHT,
    DEFAULT_TRAJ_STEPS,
    DIT_LORA_TARGETS,
    anima_boost_teacher,
    anima_cfg_delta,
    anima_direct_loss,
    anima_direct_teachers,
    anima_recipe_label,
    anima_short_trajectory,
    anima_trajectory_loss,
    anima_uni_loss,
    anima_uni_teachers,
    anima_unused_hold_loss,
    assert_sample_gate,
    image_mean_std,
    infer_sample_prompts,
    live_train_card,
    live_train_command,
    load_anima_prompts,
    turbo_preview_card,
    turbo_preview_sample_command,
    looks_like_rgb_noise,
    minus_canary_cosine,
    resolve_anima_lm_target,
    resolve_anima_lora_targets,
    row_token_plan,
)

DEFAULT_PROMPTS = Path(__file__).resolve().parent / "data" / "prompts-anima.yaml"
DEFAULT_SAVE_DIR = Path("models/anima-slider")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", type=str, default="smile-anima")
    parser.add_argument("--prompts_file", type=str, default=str(DEFAULT_PROMPTS))
    parser.add_argument("--model_id", type=str, default=DEFAULT_MODEL_ID)
    parser.add_argument("--rank", type=int, default=DEFAULT_RANK)
    parser.add_argument(
        "--lora_targets",
        type=str,
        choices=ANIMA_LORA_TARGET_CHOICES,
        default=DEFAULT_LORA_TARGETS,
        help=(
            "Which modules get PEFT LoRA. Smile default-on path is "
            "conditioner (AnimaTextConditioner q_proj/k_proj/v_proj/o_proj). "
            "dit is the old transformer-only recipe (v1–v5). "
            "dit+conditioner is joint. Qwen3 text_encoder is not adapted "
            "(28-layer / ~1.2GB). Documented in docs/anima-slider.md."
        ),
    )
    parser.add_argument("--alpha", type=float, default=None)
    parser.add_argument("--resolution", type=int, default=DEFAULT_RESOLUTION)
    parser.add_argument(
        "--sample_steps",
        type=int,
        default=DEFAULT_SAMPLE_STEPS,
        help=(
            f"in-process PEFT sample steps (default {DEFAULT_SAMPLE_STEPS} "
            "for Base). Turbo preview-only smoke uses --sample_steps 10 "
            "(official 8–12). Does not change the Base train recipe."
        ),
    )
    parser.add_argument(
        "--cfg",
        type=float,
        default=DEFAULT_CFG,
        help=(
            f"sample-path ModularPipeline CFG (default {DEFAULT_CFG:g} for "
            "Base). Turbo preview-only smoke uses --cfg 1. Train stays on "
            "Base at CFG 4; this flag does not silently train at CFG 1."
        ),
    )
    parser.add_argument(
        "--print_turbo_preview",
        action="store_true",
        help=(
            "print the Turbo v1.1 preview-only convert/sample card and exit. "
            "Does not change the Base train recipe."
        ),
    )
    parser.add_argument("--hold_weight", type=float, default=DEFAULT_HOLD_WEIGHT)
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--lr", type=float, default=DEFAULT_LR)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--save_dir", type=str, default=None)
    parser.add_argument(
        "--control_prompt",
        type=str,
        default=DEFAULT_CONTROL_PROMPT,
        help="verify-only fruit-bowl caption; never a teacher",
    )
    parser.add_argument(
        "--lm_target",
        type=str,
        choices=ANIMA_LM_TARGETS,
        default=DEFAULT_LM_TARGET,
        help=(
            "trajectory (default, live smile): K-step FlowMatch Euler "
            "MSE(x_student, x_plus). direct / cfg_delta are 1-step and "
            "cannot carry Anima expression (v-space gap ~1e-4). "
            "Music 3 --lm_target v9 is untouched."
        ),
    )
    parser.add_argument(
        "--traj_steps",
        type=int,
        default=DEFAULT_TRAJ_STEPS,
        help=(
            "short denoise steps K for --lm_target trajectory "
            f"(default {DEFAULT_TRAJ_STEPS}; 8 is the longer live option)"
        ),
    )
    parser.add_argument(
        "--traj_identity_weight",
        type=float,
        default=DEFAULT_TRAJ_IDENTITY_WEIGHT,
        help=(
            "light MSE(x_zero, x_neu) so scale 0 stays the neu short "
            f"trajectory (default {DEFAULT_TRAJ_IDENTITY_WEIGHT}; 0 = off)"
        ),
    )
    parser.add_argument(
        "--teacher_gap_boost",
        type=float,
        default=DEFAULT_TEACHER_GAP_BOOST,
        help=(
            "for direct / cfg_delta only: train toward "
            "v_neu + boost*(v_pos-v_neu) with boost>1. Default 1 (off). "
            "Not a substitute for trajectory."
        ),
    )
    parser.add_argument(
        "--sample_every",
        type=int,
        default=DEFAULT_SAMPLE_EVERY,
        help=(
            "in-process PEFT scale grid every N steps "
            f"(default {DEFAULT_SAMPLE_EVERY}; end-of-train gate always runs; "
            "0 = end of train only)"
        ),
    )
    parser.add_argument(
        "--sample_first_n",
        type=int,
        default=0,
        help="also emit the PEFT scale grid after each of the first N steps",
    )
    parser.add_argument(
        "--sample_seed",
        type=int,
        default=DEFAULT_SAMPLE_SEED,
        help="seed for in-process pipe(prompt=...) samples (default 42)",
    )
    parser.add_argument(
        "--dummy",
        action="store_true",
        help="tiny CPU DiT, never loads Anima / never hits the Hub",
    )
    parser.add_argument(
        "--allow_hub",
        action="store_true",
        help="permit a Hub download (off; CI must not set this)",
    )
    parser.add_argument(
        "--print_card",
        action="store_true",
        help="print the live train card and exit",
    )
    return parser.parse_args(argv)


def _device(arg: str, dummy: bool) -> torch.device:
    if dummy:
        if arg in ("cpu", "dummy"):
            return torch.device("cpu")
        if arg.startswith("cuda") and not torch.cuda.is_available():
            return torch.device("cpu")
    if arg.isdigit():
        return torch.device(f"cuda:{arg}" if torch.cuda.is_available() else "cpu")
    dev = torch.device(arg)
    if dev.type == "cuda" and not torch.cuda.is_available():
        return torch.device("cpu")
    return dev


def _sample_zt(backend, seed: int, step: int):
    """CPU generator + CUDA latent is illegal. Draw on CPU, then ``.to(device)``."""
    g = torch.Generator(device="cpu").manual_seed(int(seed) * 1009 + step)
    z = torch.randn((1, *backend.latent_shape), generator=g, device="cpu")
    z = z.to(device=backend.device)
    t = torch.tensor([float(100 + (step * 37) % 800)], device=backend.device)
    return z, t


def _is_lora_param_name(name: str) -> bool:
    lower = name.lower()
    return "lora_" in lower or ".lora." in lower


def freeze_anima_conditioner(pipe, train_conditioner: bool = False) -> None:
    """ModularPipeline is not an ``nn.Module``. Freeze via components.

    Do **not** call ``pipe.named_parameters()`` — that AttributeError'd
    on the live RunPod box. Walk ``pipe.text_conditioner`` and
    ``pipe.transformer`` instead.

    When ``train_conditioner`` is True, freeze conditioner *base* weights
    but leave PEFT ``lora_*`` params trainable. Default False keeps the
    old CircleStone freeze (no silent text-module training).
    """
    cond = getattr(pipe, "text_conditioner", None)
    transformer = getattr(pipe, "transformer", None)
    modules = [cond, transformer]
    if transformer is not None:
        modules.append(getattr(transformer, "text_conditioner", None))
        get_base = getattr(transformer, "get_base_model", None)
        if callable(get_base):
            try:
                modules.append(get_base())
            except Exception:
                pass
    seen: set[int] = set()
    for module in modules:
        if module is None or not hasattr(module, "named_parameters"):
            continue
        ident = id(module)
        if ident in seen:
            continue
        seen.add(ident)
        for name, param in module.named_parameters():
            is_cond = module is cond or "text_conditioner" in name
            if not is_cond:
                continue
            if train_conditioner and _is_lora_param_name(name):
                param.requires_grad_(True)
                continue
            param.requires_grad_(False)


def _conditioner_module(backend):
    pipe = getattr(backend, "pipe", None)
    cond = getattr(pipe, "text_conditioner", None) if pipe is not None else None
    if cond is None:
        cond = getattr(getattr(backend, "transformer", None), "text_conditioner", None)
    return cond


def _assert_lora_train_state(backend, spec) -> None:
    """Fail closed if conditioner trainability does not match the flag."""
    cond = _conditioner_module(backend)
    cond_trainable = False
    if cond is not None:
        cond_trainable = any(p.requires_grad for p in cond.parameters())
    names = []
    named = getattr(backend, "named_trainable", None)
    if callable(named):
        names = list(named())
    if any("text_conditioner" in n for n in names):
        cond_trainable = True
    if spec.train_conditioner and not cond_trainable:
        raise RuntimeError(
            "lora_targets includes conditioner but no text_conditioner "
            "params are trainable"
        )
    if not spec.train_conditioner and cond_trainable:
        raise RuntimeError(
            "text_conditioner is trainable; pass --lora_targets dit to "
            "keep the LLM adapter frozen"
        )


def _peft_modules(backend) -> list:
    """Every PEFT-wrapped module (DiT and/or conditioner). Dummy is one API."""
    if hasattr(backend, "set_lora_scale") or hasattr(backend, "disable_adapter"):
        return [backend]
    pipe = getattr(backend, "pipe", None)
    spec = getattr(backend, "lora_spec", None)
    modules: list = []
    if pipe is not None:
        if spec is None or spec.train_dit:
            transformer = getattr(pipe, "transformer", None)
            if transformer is not None:
                modules.append(transformer)
        if spec is not None and spec.train_conditioner:
            cond = getattr(pipe, "text_conditioner", None)
            if cond is not None:
                modules.append(cond)
    if not modules:
        fallback = getattr(backend, "transformer", backend)
        modules.append(fallback)
    return modules


def _peft_module(backend):
    """PEFT lives on adapted pipe modules; dummy exposes backend APIs."""
    return _peft_modules(backend)[0]


def _try_set_adapter_scale(module, scale: float) -> bool:
    """PEFT ``set_adapter_scale`` / adapter weights — never weight-merge."""
    if hasattr(module, "set_adapter_scale"):
        for args in ((scale,), ({"default": float(scale)},), ("default", float(scale))):
            try:
                module.set_adapter_scale(*args)
                return True
            except TypeError:
                continue
            except Exception:
                continue
    if hasattr(module, "set_adapters"):
        for args, kwargs in (
            (("default",), {"weights": float(scale)}),
            ((["default"],), {"weights": [float(scale)]}),
        ):
            try:
                module.set_adapters(*args, **kwargs)
                return True
            except Exception:
                continue
    if hasattr(module, "set_lora_scale"):
        module.set_lora_scale(float(scale))
        return True
    return False


def _enable_adapter(module) -> None:
    for name in ("enable_adapter_layers", "enable_adapters", "enable_adapter"):
        fn = getattr(module, name, None)
        if callable(fn):
            try:
                fn()
                return
            except Exception:
                continue


@contextmanager
def _disable_peft_modules(modules):
    """Nest PEFT ``disable_adapter`` / scale-0 on every adapted module."""
    exits: list[Any] = []
    scaled: list[Any] = []
    try:
        for module in modules:
            disable = getattr(module, "disable_adapter", None)
            if callable(disable):
                ctx = disable()
                if hasattr(ctx, "__enter__"):
                    ctx.__enter__()
                    exits.append(ctx)
                    continue
            if _try_set_adapter_scale(module, 0.0):
                scaled.append(module)
                continue
            raise RuntimeError("cannot disable PEFT adapter for scale 0.0 sample")
        yield
    finally:
        for ctx in reversed(exits):
            ctx.__exit__(None, None, None)
        for module in scaled:
            _try_set_adapter_scale(module, 1.0)


@contextmanager
def peft_adapter_scale(backend, scale: float):
    """In-process PEFT scale on every adapted module (DiT and/or conditioner)."""
    modules = _peft_modules(backend)
    target = float(scale)
    if abs(target) < 1e-12:
        with _disable_peft_modules(modules):
            yield
        return
    enabled_any = False
    for module in modules:
        _enable_adapter(module)
        if _try_set_adapter_scale(module, target):
            enabled_any = True
    if not enabled_any:
        raise RuntimeError(
            f"cannot set PEFT adapter scale to {target}: need "
            "disable_adapter / set_adapter_scale / adapter weights "
            "(not a post-hoc W += scale*(α/r)*(B@A) merge)"
        )
    try:
        yield
    finally:
        for module in modules:
            _try_set_adapter_scale(module, 1.0)


def _slug(text: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", str(text).lower()).strip("-")
    return slug[:48] or "prompt"


def _as_uint8_hwc(image) -> np.ndarray:
    if hasattr(image, "convert"):
        image = image.convert("RGB")
        return np.asarray(image, dtype=np.uint8)
    arr = np.asarray(image)
    if arr.ndim == 4:
        arr = arr[0]
    if arr.ndim == 3 and arr.shape[0] in (1, 3) and arr.shape[-1] not in (1, 3):
        arr = np.transpose(arr, (1, 2, 0))
    if arr.dtype != np.uint8:
        max_v = float(arr.max()) if arr.size else 0.0
        if max_v <= 1.0 + 1e-5:
            arr = arr * 255.0
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    if arr.ndim == 2:
        arr = np.stack([arr, arr, arr], axis=-1)
    return arr


def _extract_images(result) -> list[np.ndarray]:
    if result is None:
        return []
    images = getattr(result, "images", None)
    if images is None and isinstance(result, dict):
        images = result.get("images")
    if images is None:
        images = result
    if isinstance(images, (list, tuple)):
        return [_as_uint8_hwc(item) for item in images]
    return [_as_uint8_hwc(images)]


def _cpu_noise(shape, seed: int, device) -> torch.Tensor:
    g = torch.Generator(device="cpu").manual_seed(int(seed))
    return torch.randn(shape, generator=g, device="cpu").to(device=device)


def _read_guidance_scale(obj) -> float | None:
    """Read CFG from a guider, ``guider.config``, or a mapping."""
    if obj is None:
        return None
    if isinstance(obj, dict):
        value = obj.get("guidance_scale")
        return float(value) if value is not None else None
    value = getattr(obj, "guidance_scale", None)
    if value is not None and not callable(value):
        try:
            return float(value)
        except (TypeError, ValueError):
            return None
    return None


def _set_guidance_scale(obj, value: float) -> bool:
    """Write CFG onto a guider / config. Prefer ``register_to_config``."""
    if obj is None:
        return False
    value = float(value)
    if isinstance(obj, dict):
        obj["guidance_scale"] = value
        return True
    register = getattr(obj, "register_to_config", None)
    if callable(register):
        try:
            register(guidance_scale=value)
            if _read_guidance_scale(obj) is not None:
                return True
        except Exception:
            pass
    try:
        setattr(obj, "guidance_scale", value)
        return True
    except (TypeError, AttributeError):
        return False


def apply_anima_guider_cfg(pipe, cfg: float) -> Callable[[], None]:
    """Set ModularPipeline CFG on the guider, not a rejected kwarg.

    Live Anima logs ``Unexpected input 'guidance_scale' … ignored`` if we
    pass ``guidance_scale=`` to ``pipe(...)``. The working pattern is
    ``guider.config.guidance_scale`` (and any real field the guider
    exposes), then ``pipe(prompt=...)`` without that kwarg.

    Fail closed: do not silently sample at CFG 1 when a higher CFG
    was requested. Explicit ``--cfg 1`` (Turbo preview sample) is
    allowed and is applied the same way.
    """
    cfg = float(cfg)
    guider = getattr(pipe, "guider", None)
    if guider is None:
        raise RuntimeError(
            f"Anima ModularPipeline has no guider; cannot apply CFG {cfg:g}. "
            "Set guider.config.guidance_scale. Do not pass guidance_scale= "
            "(ignored). Do not silently sample at CFG 1."
        )
    snapshots: list[tuple[Any, float]] = []
    applied = False
    config = getattr(guider, "config", None)
    for obj in (config, guider):
        if obj is None:
            continue
        prev = _read_guidance_scale(obj)
        if _set_guidance_scale(obj, cfg):
            if prev is not None:
                snapshots.append((obj, prev))
            applied = True
    for extra_name in ("guidance", "cfg"):
        extra = getattr(guider, extra_name, None)
        if isinstance(extra, (int, float)):
            prev = float(extra)
            try:
                setattr(guider, extra_name, cfg)
                snapshots.append((extra_name, prev))
                applied = True
            except (TypeError, AttributeError):
                pass
    readback = _read_guidance_scale(config)
    if readback is None:
        readback = _read_guidance_scale(guider)
    if not applied or readback is None or abs(readback - cfg) > 1e-5:
        raise RuntimeError(
            f"Anima ModularPipeline CFG {cfg:g} could not be applied "
            f"(readback={readback!r}). Set guider.config.guidance_scale; "
            "top-level guidance_scale= is ignored. Do not silently sample "
            "at CFG 1."
        )

    def _restore() -> None:
        for obj, prev in snapshots:
            if isinstance(obj, str):
                try:
                    setattr(guider, obj, prev)
                except (TypeError, AttributeError):
                    pass
                continue
            _set_guidance_scale(obj, prev)

    return _restore


def _call_modular_pipe(
    pipe,
    prompt: str,
    *,
    steps: int,
    height: int,
    width: int,
    cfg: float,
    seed: int,
    device,
    latent_shape=None,
):
    """Same ``pipe(prompt=...)`` path as live infer. PEFT stays attached.

    CFG is applied on the guider (``guider.config.guidance_scale``). Do
    **not** pass ``guidance_scale=`` — Anima ModularPipeline ignores it.
    """
    dummy_h = int(height)
    dummy_w = int(width)
    # Dummy stay cheap; live uses the train resolution.
    if type(pipe).__name__ == "FakeAnimaModularPipe":
        dummy_h = min(dummy_h, 64)
        dummy_w = min(dummy_w, 64)
    restore = apply_anima_guider_cfg(pipe, cfg)
    kwargs: dict[str, Any] = {
        "prompt": prompt,
        "height": dummy_h,
        "width": dummy_w,
        "num_inference_steps": int(steps),
        "output_type": "np",
    }
    if latent_shape is not None:
        kwargs["latents"] = _cpu_noise((1, *tuple(latent_shape)), seed, device)
    else:
        kwargs["generator"] = torch.Generator(device="cpu").manual_seed(int(seed))
    try:
        result = pipe(**kwargs)
    finally:
        restore()
    return _extract_images(result)


def emit_inprocess_samples(
    backend,
    args: argparse.Namespace,
    save_dir: Path,
    *,
    step: int,
    rows,
    dummy: bool,
) -> list[dict[str, Any]]:
    """PEFT scale grid through ``pipe(prompt=...)``. Not a velocity dump."""
    pipe = getattr(backend, "pipe", None)
    if pipe is None or not callable(pipe):
        raise RuntimeError(
            "in-process Anima sample needs backend.pipe(prompt=...) "
            "with PEFT still attached to adapted pipe modules"
        )
    prompts = infer_sample_prompts(rows, getattr(args, "control_prompt", DEFAULT_CONTROL_PROMPT))
    scales = list(DEFAULT_SAMPLE_SCALES)
    height = int(args.resolution)
    width = int(args.resolution)
    steps = 2 if dummy else int(args.sample_steps)
    seed = int(getattr(args, "sample_seed", DEFAULT_SAMPLE_SEED))
    cfg = float(args.cfg)
    out_dir = Path(save_dir) / "samples"
    out_dir.mkdir(parents=True, exist_ok=True)
    latent_shape = getattr(backend, "latent_shape", None) if not dummy else None
    records: list[dict[str, Any]] = []
    for prompt in prompts:
        for scale in scales:
            with torch.no_grad(), peft_adapter_scale(backend, scale):
                images = _call_modular_pipe(
                    pipe,
                    prompt,
                    steps=steps,
                    height=height,
                    width=width,
                    cfg=cfg,
                    seed=seed,
                    device=backend.device,
                    latent_shape=latent_shape,
                )
            if not images:
                raise RuntimeError(f"pipe(prompt={prompt!r}) returned no images")
            arr = images[0]
            mean, std = image_mean_std(arr)
            noisy = looks_like_rgb_noise(arr)
            slug = _slug(prompt)
            scale_tag = f"{scale:g}".replace("-", "m")
            name = f"step{int(step):04d}_{slug}_scale{scale_tag}.png"
            path = out_dir / name
            from PIL import Image

            Image.fromarray(arr, mode="RGB").save(path)
            records.append(
                {
                    "step": int(step),
                    "prompt": prompt,
                    "scale": float(scale),
                    "mean": mean,
                    "std": std,
                    "looks_like_noise": bool(noisy),
                    "path": str(path.name),
                    "seed": seed,
                    "sample_steps": steps,
                    "cfg": cfg,
                    "cfg_via": "guider.config.guidance_scale",
                    "height": int(arr.shape[0]),
                    "width": int(arr.shape[1]),
                    "method": "peft_pipe_prompt",
                }
            )
    meta_path = out_dir / f"step{int(step):04d}_meta.json"
    payload = {
        "step": int(step),
        "dummy": bool(dummy),
        "seed": seed,
        "scales": scales,
        "prompts": prompts,
        "samples": records,
    }
    meta_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    assert_sample_gate(records)
    return records


def _should_sample(step: int, args: argparse.Namespace, *, last: bool) -> bool:
    if last:
        return True
    every = int(getattr(args, "sample_every", 0) or 0)
    first_n = int(getattr(args, "sample_first_n", 0) or 0)
    idx = int(step) + 1
    if first_n and idx <= first_n:
        return True
    if every > 0 and idx % every == 0:
        return True
    return False


def _sidecar_lora_fields(spec) -> dict[str, Any]:
    return {
        "lora_targets": spec.label,
        "dit_lora_targets": list(DIT_LORA_TARGETS) if spec.train_dit else [],
        "conditioner_lora_targets": (
            list(CONDITIONER_LORA_TARGETS) if spec.train_conditioner else []
        ),
        "train_text_conditioner": spec.train_conditioner,
        "train_dit": spec.train_dit,
        "adapted_modules": spec.adapted_module_names,
        "frozen_modules": list(spec.frozen_modules),
    }


def train_dummy(args: argparse.Namespace) -> dict:
    device = _device(str(args.device), dummy=True)
    rows, meta = load_anima_prompts(args.prompts_file)
    if not rows:
        raise ValueError("no anima prompt rows")
    rank = int(args.rank)
    spec = resolve_anima_lora_targets(getattr(args, "lora_targets", DEFAULT_LORA_TARGETS))
    backend = FakeAnimaBackend(
        device=str(device),
        rank=rank,
        seed=int(args.seed),
        lora_targets=spec.label,
    )
    _assert_lora_train_state(backend, spec)
    params = backend.trainable_parameters()
    names = backend.named_trainable()
    if spec.train_conditioner and not any("text_conditioner" in n for n in names):
        raise RuntimeError("dummy conditioner LoRA enabled but not attached")
    if not spec.train_conditioner and any("text_conditioner" in n for n in names):
        raise RuntimeError("dummy LoRA attached to text_conditioner")
    opt = torch.optim.AdamW(params, lr=float(args.lr))
    history: list[float] = []
    canary: list[float] = []
    lm_target = resolve_anima_lm_target(getattr(args, "lm_target", DEFAULT_LM_TARGET))
    plans = [row_token_plan(row) for row in rows]
    align_row = rows[0]
    infer = align_row.infer_prompt
    plus_before = write_plus_alignment(
        backend, infer, align_row.positive, seed=int(args.seed)
    )

    save_dir = Path(args.save_dir or DEFAULT_SAVE_DIR)
    save_dir.mkdir(parents=True, exist_ok=True)
    sample_records: list[dict] = []
    trained_infer: list[str] = []
    total = int(args.steps)
    pbar = tqdm(range(total), disable=total < 3)
    for step in pbar:
        row, plan = _cycle_row(rows, plans, step)
        if row.infer_prompt not in trained_infer:
            trained_infer.append(row.infer_prompt)
        z, t = _sample_zt(backend, int(args.seed), step)
        loss, canary_v = _train_step(
            backend,
            row,
            plan,
            z,
            t,
            float(args.hold_weight),
            lm_target,
            traj_steps=int(args.traj_steps),
            traj_identity_weight=float(args.traj_identity_weight),
            teacher_gap_boost=float(args.teacher_gap_boost),
        )
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        history.append(float(loss.detach()))
        if canary_v is not None:
            canary.append(canary_v)
        pbar.set_postfix(loss=f"{history[-1]:.4f}")
        if _should_sample(step, args, last=False):
            sample_records = emit_inprocess_samples(
                backend, args, save_dir, step=step + 1, rows=rows, dummy=True
            )

    plus_after = write_plus_alignment(
        backend, infer, align_row.positive, seed=int(args.seed)
    )
    sample_records = emit_inprocess_samples(
        backend, args, save_dir, step=total, rows=rows, dummy=True
    )
    sidecar = {
        "name": args.name,
        "dummy": True,
        "model_id": args.model_id,
        "rank": rank,
        "resolution": int(args.resolution),
        "sample_steps": int(args.sample_steps),
        "cfg": float(args.cfg),
        "lr": float(args.lr),
        "lm_target": lm_target,
        "sample_every": int(args.sample_every),
        "device": str(device),
        **_sidecar_lora_fields(spec),
        "recipe": anima_recipe_label(lm_target),
        "traj_steps": int(args.traj_steps),
        "traj_identity_weight": float(args.traj_identity_weight),
        "teacher_gap_boost": float(args.teacher_gap_boost),
        "traj_loop": (
            "thin FlowMatch Euler over predict_v "
            "(σ=linspace(1,1/K,K)+0, x←x+(σ_next−σ)*v)"
        ),
        "traj_loss": (
            "MSE(x_student, x_plus) + λ_id*MSE(x_zero, x_neu)"
        ),
        "plus_label": meta.plus_label,
        "minus_canary": any(r.has_minus_canary for r in rows),
        "train_infer_prompts": trained_infer,
        "sample_infer_prompts": infer_sample_prompts(
            rows, getattr(args, "control_prompt", DEFAULT_CONTROL_PROMPT)
        ),
        "canary_cos_last": canary[-1] if canary else None,
        "plus_align_before": plus_before,
        "plus_align_after": plus_after,
        "loss_last": history[-1] if history else None,
        "steps": int(args.steps),
        "prompts_file": str(args.prompts_file),
        "control_prompt": str(args.control_prompt),
        "sample_grid": {
            "n": len(sample_records),
            "scales": list(DEFAULT_SAMPLE_SCALES),
            "seed": int(args.sample_seed),
            "method": "peft_pipe_prompt",
        },
        "music3_default_untouched": {"lm_target": "v9", "pole_mode": "hidden"},
    }
    side_path = save_dir / f"{args.name}_dummy_last.json"
    side_path.write_text(json.dumps(sidecar, indent=2), encoding="utf-8")
    print(json.dumps(sidecar, indent=2))
    return sidecar


def _refuse_hub_download() -> None:
    if os.environ.get("HF_HUB_OFFLINE", "1") in ("0", "false", "False"):
        return
    # Default path: never call from_pretrained without local_files_only.


class LiveAnimaBackend:
    """Thin wrapper around a local Anima ModularPipeline.

    Adapted from conceptmod's Anima backend (encode / velocity / disable
    adapter) without bringing the DSL trainer. PEFT attaches to
    ``transformer`` and/or ``text_conditioner`` per ``lora_spec``.
    Qwen3 ``text_encoder`` stays frozen.
    """

    def __init__(self, pipe, device: torch.device, resolution: int, lora_spec):
        self.pipe = pipe
        self.device = device
        self.resolution = resolution
        self.lora_spec = lora_spec
        self.compute_dtype = torch.bfloat16
        self.max_sequence_length = 512
        transformer = pipe.transformer
        get_base = getattr(transformer, "get_base_model", None)
        base = get_base() if callable(get_base) else transformer
        scale = pipe.vae_scale_factor
        h = resolution // scale
        self.latent_shape = (base.config.in_channels, 1, h, h)
        self._padding_mask = torch.zeros(
            1, 1, resolution, resolution, device=device, dtype=torch.bfloat16
        )
        self._text_cache: dict[str, torch.Tensor] = {}
        self._qwen_t5_cache: dict[str, tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]] = {}
        sched = getattr(pipe, "scheduler", None)
        cfg = getattr(sched, "config", None)
        self.num_train_timesteps = int(
            getattr(cfg, "num_train_timesteps", None) or 1000
        )

    def encode_text(self, prompt: str) -> tuple[torch.Tensor, list[str]]:
        if not self.lora_spec.train_conditioner:
            if prompt not in self._text_cache:
                self._text_cache[prompt] = self._encode_raw(prompt)
            return self._text_cache[prompt], []
        return self._encode_raw(prompt), []

    def _conditioner_dtype(self):
        cond = self.pipe.text_conditioner
        dtype = getattr(cond, "dtype", None)
        if dtype is not None:
            return dtype
        try:
            return next(cond.parameters()).dtype
        except StopIteration:
            return self.compute_dtype

    def _qwen_t5_pack(self, prompt: str):
        """Frozen Qwen3 + T5 ids. Cached; never backprop through the encoder."""
        if prompt in self._qwen_t5_cache:
            return self._qwen_t5_cache[prompt]
        prompts = [prompt]
        tok = self.pipe.tokenizer(
            prompts,
            padding="longest",
            max_length=self.max_sequence_length,
            truncation=True,
            return_tensors="pt",
        )
        ids = tok.input_ids.to(self.device)
        mask = tok.attention_mask.to(self.device)
        if ids.shape[-1] == 0:
            ids = ids.new_zeros((ids.shape[0], 1))
            mask = mask.new_zeros((mask.shape[0], 1))
        with torch.no_grad():
            qwen = self.pipe.text_encoder(
                input_ids=ids, attention_mask=mask, output_hidden_states=False
            ).last_hidden_state
            qwen = qwen * mask.to(qwen.dtype).unsqueeze(-1)
        t5 = self.pipe.t5_tokenizer(
            prompts,
            padding="longest",
            max_length=self.max_sequence_length,
            truncation=True,
            return_tensors="pt",
        )
        pack = (
            qwen.detach(),
            mask,
            t5.input_ids.to(self.device),
            t5.attention_mask.to(self.device),
        )
        self._qwen_t5_cache[prompt] = pack
        return pack

    def _encode_raw(self, prompt: str) -> torch.Tensor:
        qwen, mask, t5_ids, t5_mask = self._qwen_t5_pack(prompt)
        cond_dtype = self._conditioner_dtype()
        cond = self.pipe.text_conditioner(
            source_hidden_states=qwen.to(device=self.device, dtype=cond_dtype),
            target_input_ids=t5_ids,
            target_attention_mask=t5_mask,
            source_attention_mask=mask,
        )
        return cond.float().to(self.device)

    def _forward(self, z, timestep, embeds: torch.Tensor):
        dtype = self.compute_dtype
        t = timestep.expand(z.shape[0]).to(device=self.device, dtype=dtype)
        t = t / self.pipe.scheduler.config.num_train_timesteps
        hidden = z.to(device=self.device, dtype=dtype)
        if hidden.ndim == 4:
            hidden = hidden.unsqueeze(2)
        out = self.pipe.transformer(
            hidden_states=hidden,
            timestep=t,
            encoder_hidden_states=embeds.to(device=self.device, dtype=dtype),
            padding_mask=self._padding_mask.to(device=self.device, dtype=dtype),
            return_dict=False,
        )[0]
        if out.ndim == 4:
            out = out.unsqueeze(2)
        return out.float()

    def predict_v(self, prompt, z, timestep, frozen=False, scale=None):
        # Encode under the same adapter scale as the DiT so conditioner
        # LoRA (when enabled) actually sees scale 0 / frozen / 0.25.
        if frozen or scale == 0.0:
            with torch.no_grad(), peft_adapter_scale(self, 0.0):
                embeds, _ = self.encode_text(prompt)
                return self._forward(z, timestep, embeds)
        if scale is not None and abs(float(scale) - 1.0) > 1e-8:
            with peft_adapter_scale(self, float(scale)):
                embeds, _ = self.encode_text(prompt)
                return self._forward(z, timestep, embeds)
        embeds, _ = self.encode_text(prompt)
        return self._forward(z, timestep, embeds)

    def text_features(self, prompt, frozen=False, scale=None):
        if frozen or scale == 0.0:
            with torch.no_grad(), peft_adapter_scale(self, 0.0):
                return self.encode_text(prompt)
        embeds, tokens = self.encode_text(prompt)
        return embeds, tokens

    def _adapted_nn_modules(self):
        modules = []
        if self.lora_spec.train_dit:
            modules.append(self.pipe.transformer)
        if self.lora_spec.train_conditioner:
            modules.append(self.pipe.text_conditioner)
        return modules

    def trainable_parameters(self):
        params = []
        for module in self._adapted_nn_modules():
            module.train()
            params.extend(p for p in module.parameters() if p.requires_grad)
        if not params:
            raise RuntimeError("Anima PEFT returned no trainable params")
        return params

    def named_trainable(self):
        names = []
        if self.lora_spec.train_dit:
            names.extend(
                n
                for n, p in self.pipe.transformer.named_parameters()
                if p.requires_grad
            )
        if self.lora_spec.train_conditioner:
            names.extend(
                f"text_conditioner.{n}"
                for n, p in self.pipe.text_conditioner.named_parameters()
                if p.requires_grad
            )
        return names


def _attach_peft(module, rank: int, alpha: int, targets: list[str]):
    from peft import LoraConfig, get_peft_model

    config = LoraConfig(r=int(rank), lora_alpha=int(alpha), target_modules=list(targets))
    return get_peft_model(module, config)


def load_live_backend(args: argparse.Namespace, device: torch.device):
    """Load Anima locally. Never downloads unless ``--allow_hub``."""
    if not args.allow_hub:
        os.environ["HF_HUB_OFFLINE"] = "1"
        _refuse_hub_download()
    else:
        os.environ["HF_HUB_OFFLINE"] = "0"
    try:
        from diffusers import ModularPipeline
    except ImportError as exc:
        raise RuntimeError(
            "live Anima needs diffusers + peft. Use --dummy for CPU tests."
        ) from exc

    spec = resolve_anima_lora_targets(getattr(args, "lora_targets", DEFAULT_LORA_TARGETS))
    kwargs = {"local_files_only": not bool(args.allow_hub)}
    try:
        pipe = ModularPipeline.from_pretrained(args.model_id, **kwargs)
        pipe.load_components(dtype=torch.bfloat16)
        pipe.to(str(device))
    except Exception as exc:
        raise RuntimeError(
            f"live Anima weights not available for {args.model_id!r} "
            f"(local_files_only={not args.allow_hub}). "
            "CI must use --dummy. See scripts/smoke_anima_slider.py."
        ) from exc

    rank = int(args.rank)
    alpha = int(args.alpha or args.rank)
    try:
        if spec.train_dit:
            pipe.transformer = _attach_peft(
                pipe.transformer, rank, alpha, list(DIT_LORA_TARGETS)
            )
            pipe.transformer.to(device)
        if spec.train_conditioner:
            pipe.text_conditioner = _attach_peft(
                pipe.text_conditioner, rank, alpha, list(CONDITIONER_LORA_TARGETS)
            )
            pipe.text_conditioner.to(device)
    except ImportError as exc:
        raise RuntimeError(
            "live Anima needs diffusers + peft. Use --dummy for CPU tests."
        ) from exc

    if spec.train_conditioner:
        transformer = pipe.transformer
        enable_ckpt = getattr(transformer, "enable_gradient_checkpointing", None)
        if callable(enable_ckpt):
            try:
                enable_ckpt()
            except Exception:
                pass
        cond = pipe.text_conditioner
        enable_cond = getattr(cond, "enable_gradient_checkpointing", None)
        if callable(enable_cond):
            try:
                enable_cond()
            except Exception:
                pass

    freeze_anima_conditioner(pipe, train_conditioner=spec.train_conditioner)
    return LiveAnimaBackend(
        pipe, device, resolution=int(args.resolution), lora_spec=spec
    )


def _cycle_row(rows, plans, step: int):
    """Cycle woman + man (and any other yaml rows). Never lock to rows[0]."""
    idx = int(step) % len(rows)
    return rows[idx], plans[idx]


def _hold_and_canary(backend, row, plan, hold_weight: float, canary_student, v_neg, v_null):
    feat_s, tokens_s = backend.text_features(row.positive, frozen=False, scale=1.0)
    feat_n, tokens_n = backend.text_features(row.neutral, frozen=True)
    # Dummy tokenizer matches ``row_token_plan`` word ids. Live Qwen/T5
    # ids do not — skip the index hold there (encoder is frozen anyway).
    extra = None
    if tokens_s and tokens_n and plan["pairs"]:
        extra = hold_weight * anima_unused_hold_loss(
            feat_s, feat_n, pairs=plan["pairs"]
        )
    canary = None
    if v_neg is not None:
        canary = float(minus_canary_cosine(canary_student, v_neg, v_null))
    return extra, canary


def _trajectory_step(
    backend,
    row,
    plan,
    z,
    hold_weight: float,
    traj_steps: int,
    identity_weight: float,
):
    """Same z_T, K-step Euler: student(neu, adapter) → frozen plus traj."""
    infer = row.infer_prompt
    with torch.no_grad():
        x_plus = anima_short_trajectory(
            backend, row.positive, z, num_steps=traj_steps, frozen=True
        )
        x_neu = anima_short_trajectory(
            backend, row.neutral, z, num_steps=traj_steps, frozen=True
        )
        v_null = backend.predict_v("", z, torch.tensor([1000.0], device=z.device), frozen=True)
        v_neg = (
            backend.predict_v(
                row.negative,
                z,
                torch.tensor([1000.0], device=z.device),
                frozen=True,
            )
            if row.has_minus_canary
            else None
        )
    x_student = anima_short_trajectory(
        backend, infer, z, num_steps=traj_steps, frozen=False, scale=1.0
    )
    x_zero = None
    if float(identity_weight) > 0.0:
        with torch.no_grad():
            x_zero = anima_short_trajectory(
                backend, infer, z, num_steps=traj_steps, frozen=False, scale=0.0
            )
    loss = anima_trajectory_loss(
        x_student, x_plus, x_zero, x_neu, identity_weight=identity_weight
    )
    extra, canary = _hold_and_canary(
        backend, row, plan, hold_weight, x_student.detach(), v_neg, v_null
    )
    if extra is not None:
        loss = loss + extra
    return loss, canary


def _train_step(
    backend,
    row,
    plan,
    z,
    t,
    hold_weight: float,
    lm_target: str,
    *,
    traj_steps: int = DEFAULT_TRAJ_STEPS,
    traj_identity_weight: float = DEFAULT_TRAJ_IDENTITY_WEIGHT,
    teacher_gap_boost: float = DEFAULT_TEACHER_GAP_BOOST,
):
    """Student +1 stays on infer/neu. + caption is teacher only (#62 analog)."""
    infer = row.infer_prompt
    recipe = resolve_anima_lm_target(lm_target)
    if recipe == "trajectory":
        return _trajectory_step(
            backend,
            row,
            plan,
            z,
            hold_weight,
            traj_steps=int(traj_steps),
            identity_weight=float(traj_identity_weight),
        )
    with torch.no_grad():
        v_pos = backend.predict_v(row.positive, z, t, frozen=True)
        v_neu = backend.predict_v(row.neutral, z, t, frozen=True)
        v_null = backend.predict_v("", z, t, frozen=True)
        v_neg = (
            backend.predict_v(row.negative, z, t, frozen=True)
            if row.has_minus_canary
            else None
        )
        v_pos = anima_boost_teacher(v_pos, v_neu, teacher_gap_boost)
    v_student_plus = backend.predict_v(infer, z, t, frozen=False, scale=1.0)
    v_student_zero = backend.predict_v(infer, z, t, frozen=False, scale=0.0)
    if recipe == "direct":
        teachers = anima_direct_teachers(v_pos, v_neu)
        s_plus = v_student_plus
        s_zero = v_student_zero
        loss = anima_direct_loss(s_plus, teachers["plus"], s_zero, teachers["zero"])
        canary_student = anima_cfg_delta(v_student_plus.detach(), v_null)
    else:
        teachers = anima_uni_teachers(v_pos, v_neu, v_null, v_neg)
        s_plus = anima_cfg_delta(
            v_student_plus,
            backend.predict_v("", z, t, frozen=False, scale=1.0),
        )
        s_zero = anima_cfg_delta(
            v_student_zero,
            backend.predict_v("", z, t, frozen=False, scale=0.0),
        )
        loss = anima_uni_loss(s_plus, teachers["plus"], s_zero, teachers["zero"])
        canary_student = s_plus.detach()
    extra, canary = _hold_and_canary(
        backend, row, plan, hold_weight, canary_student, v_neg, v_null
    )
    if extra is not None:
        loss = loss + extra
    return loss, canary


def train_live(args: argparse.Namespace) -> dict:
    device = _device(str(args.device), dummy=False)
    if device.type != "cuda":
        raise RuntimeError(
            "live Anima needs CUDA. Pass --dummy for the CPU fake, or "
            "--device cuda:0 with local weights."
        )
    backend = load_live_backend(args, device)
    spec = backend.lora_spec
    _assert_lora_train_state(backend, spec)
    rows, meta = load_anima_prompts(args.prompts_file)
    lm_target = resolve_anima_lm_target(getattr(args, "lm_target", DEFAULT_LM_TARGET))
    plans = [row_token_plan(row) for row in rows]
    opt = torch.optim.AdamW(backend.trainable_parameters(), lr=float(args.lr))
    history: list[float] = []
    canary: list[float] = []
    save_dir = Path(args.save_dir or DEFAULT_SAVE_DIR)
    save_dir.mkdir(parents=True, exist_ok=True)
    sample_records: list[dict] = []
    trained_infer: list[str] = []
    total = int(args.steps)
    for step in tqdm(range(total)):
        row, plan = _cycle_row(rows, plans, step)
        if row.infer_prompt not in trained_infer:
            trained_infer.append(row.infer_prompt)
        z, t = _sample_zt(backend, int(args.seed), step)
        loss, canary_v = _train_step(
            backend,
            row,
            plan,
            z,
            t,
            float(args.hold_weight),
            lm_target,
            traj_steps=int(args.traj_steps),
            traj_identity_weight=float(args.traj_identity_weight),
            teacher_gap_boost=float(args.teacher_gap_boost),
        )
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        history.append(float(loss.detach()))
        if canary_v is not None:
            canary.append(canary_v)
        if _should_sample(step, args, last=False):
            sample_records = emit_inprocess_samples(
                backend, args, save_dir, step=step + 1, rows=rows, dummy=False
            )
    if spec.train_dit:
        backend.pipe.transformer.save_pretrained(str(save_dir / f"{args.name}_lora"))
    if spec.train_conditioner:
        backend.pipe.text_conditioner.save_pretrained(
            str(save_dir / f"{args.name}_conditioner_lora")
        )
    sample_records = emit_inprocess_samples(
        backend, args, save_dir, step=total, rows=rows, dummy=False
    )
    sidecar = {
        "name": args.name,
        "dummy": False,
        "model_id": args.model_id,
        "rank": int(args.rank),
        "resolution": int(args.resolution),
        "sample_steps": int(args.sample_steps),
        "cfg": float(args.cfg),
        "lr": float(args.lr),
        "lm_target": lm_target,
        "sample_every": int(args.sample_every),
        "device": str(device),
        **_sidecar_lora_fields(spec),
        "recipe": anima_recipe_label(lm_target),
        "traj_steps": int(args.traj_steps),
        "traj_identity_weight": float(args.traj_identity_weight),
        "teacher_gap_boost": float(args.teacher_gap_boost),
        "traj_loop": (
            "thin FlowMatch Euler over predict_v "
            "(σ=linspace(1,1/K,K)+0, x←x+(σ_next−σ)*v)"
        ),
        "traj_loss": (
            "MSE(x_student, x_plus) + λ_id*MSE(x_zero, x_neu)"
        ),
        "plus_label": meta.plus_label,
        "minus_canary": any(r.has_minus_canary for r in rows),
        "train_infer_prompts": trained_infer,
        "sample_infer_prompts": infer_sample_prompts(
            rows, getattr(args, "control_prompt", DEFAULT_CONTROL_PROMPT)
        ),
        "canary_cos_last": canary[-1] if canary else None,
        "loss_last": history[-1] if history else None,
        "steps": int(args.steps),
        "prompts_file": str(args.prompts_file),
        "control_prompt": str(args.control_prompt),
        "sample_grid": {
            "n": len(sample_records),
            "scales": list(DEFAULT_SAMPLE_SCALES),
            "seed": int(args.sample_seed),
            "method": "peft_pipe_prompt",
        },
        "music3_default_untouched": {"lm_target": "v9", "pole_mode": "hidden"},
    }
    (save_dir / f"{args.name}_last.json").write_text(
        json.dumps(sidecar, indent=2), encoding="utf-8"
    )
    print(json.dumps(sidecar, indent=2))
    return sidecar


def train(args: argparse.Namespace) -> dict:
    if getattr(args, "print_turbo_preview", False):
        card = turbo_preview_card()
        print(json.dumps(card, indent=2))
        print()
        print(turbo_preview_sample_command())
        return card
    if args.print_card:
        card = live_train_card(
            name=args.name,
            prompts_file=args.prompts_file,
            model_id=args.model_id,
            rank=int(args.rank),
            resolution=int(args.resolution),
            sample_steps=int(args.sample_steps),
            cfg=float(args.cfg),
            device=str(args.device),
            lr=float(args.lr),
            control_prompt=str(args.control_prompt),
            lm_target=str(args.lm_target),
            sample_every=int(args.sample_every),
            traj_steps=int(args.traj_steps),
            traj_identity_weight=float(args.traj_identity_weight),
            teacher_gap_boost=float(args.teacher_gap_boost),
            lora_targets=str(args.lora_targets),
        )
        print(json.dumps(card, indent=2))
        print()
        print(live_train_command(
            name=args.name,
            prompts_file=args.prompts_file,
            model_id=args.model_id,
            rank=int(args.rank),
            resolution=int(args.resolution),
            sample_steps=int(args.sample_steps),
            cfg=float(args.cfg),
            device=str(args.device),
            lr=float(args.lr),
            lm_target=str(args.lm_target),
            sample_every=int(args.sample_every),
            traj_steps=int(args.traj_steps),
            teacher_gap_boost=float(args.teacher_gap_boost),
            lora_targets=str(args.lora_targets),
        ))
        return card
    if args.dummy:
        return train_dummy(args)
    return train_live(args)


def main(argv: list[str] | None = None) -> dict:
    return train(parse_args(argv))


if __name__ == "__main__":
    main()
