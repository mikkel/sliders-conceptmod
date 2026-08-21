"""Convert MiniMax Music 3 slider safetensors into ComfyUI-loadable files.

This repo's trainers save LoRANetwork keys (``lora_unet-…lora_down.weight``,
``lora_te-…``), not PEFT diffusers paths. ComfyUI's native MiniMax Music 3
DiT uses fused QKV under ``diffusion_model.diffusion_transformer…`` — those
names come from ``comfy/ldm/minimax_music/dit.py`` and the inverse of
SimpleTuner's MiniMax ComfyUI export, not from a community example file.

    python scripts/convert_lora_comfyui.py weights/energy-slider-v2/
    python scripts/convert_lora_comfyui.py weights/energy-slider-v2/energy_unit_last.safetensors --force

Originals are never touched. CPU only.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys

import torch
from safetensors import safe_open
from safetensors.torch import save_file

OUT_SUFFIX = "_comfyui.safetensors"
TARGET_DTYPE = torch.bfloat16
METADATA = {"format": "pt"}

# LoRANetwork prefix + dashed module path, as written by train_lora_music3.py
# / train_lm_slider_music3.py (delimiter="-").
TF_PREFIX = "lora_unet-"
LM_PREFIX = "lora_te-"

# diffusers MiniMaxMusic3Transformer1DModel (trainer) → ComfyUI MiniMaxMusic3DiT
# (comfy/ldm/minimax_music/dit.py). QKV is fused separately.
TF_BLOCK = {
    "attn-to_out-0": "self_attn.to_out",
    "ff_in": "ff.ff.0.proj",
    "ff_out": "ff.ff.2",
}
TF_ROOT = {
    "proj_in": "diffusion_transformer.transformer.project_in",
    "proj_out": "diffusion_transformer.transformer.project_out",
    "preprocess_conv": "diffusion_transformer.preprocess_conv",
    "postprocess_conv": "diffusion_transformer.postprocess_conv",
    "time_embed-linear_1": "diffusion_transformer.to_timestep_embed.0",
    "time_embed-linear_2": "diffusion_transformer.to_timestep_embed.2",
}
TF_QKV = {
    "attn-to_q": "q",
    "attn-to_k": "k",
    "attn-to_v": "v",
}
TF_BLOCK_RE = re.compile(r"^transformer_blocks-(\d+)-(.+)$")

# HF Qwen3Attention (trainer) → ComfyUI CLIP generic keys
# (comfy/lora.py model_lora_keys_clip: text_encoders.{state_dict_stem}).
LM_LAYER_RE = re.compile(r"^model-layers-(\d+)-self_attn-(q_proj|k_proj|v_proj|o_proj)$")

WEIGHT_SIDES = {
    ".lora_down.weight": "down",
    ".lora_up.weight": "up",
    ".lora.down.weight": "down",
    ".lora.up.weight": "up",
    ".lora_A.weight": "down",
    ".lora_B.weight": "up",
}

class ConvertError(Exception):
    """A source key could not be mapped onto a ComfyUI name."""


def split_slider_key(key: str):
    """Return (module_name, 'down'|'up'|'alpha') or None."""
    if key.endswith(".alpha"):
        return key[: -len(".alpha")], "alpha"
    for suffix, side in WEIGHT_SIDES.items():
        if key.endswith(suffix):
            return key[: -len(suffix)], side
    return None


def detect_kind(keys, sidecar: dict | None = None) -> str:
    """Name the slider host from keys, falling back to the sidecar."""
    if any(k.startswith("diffusion_model.") or k.startswith("text_encoders.") for k in keys):
        return "already_comfyui"
    if any(k.startswith(TF_PREFIX) or "transformer_blocks-" in k for k in keys):
        return "transformer"
    if any(k.startswith(LM_PREFIX) or "-self_attn-q_proj" in k for k in keys):
        return "language_model"
    kind = (sidecar or {}).get("kind")
    if kind in ("transformer", "language_model"):
        return kind
    if any("lora_A" in k or "lora_B" in k for k in keys) and any(
        k.startswith("base_model.model.") for k in keys
    ):
        raise ConvertError(
            "PEFT diffusers keys (base_model.model.*); this converter is the "
            "LoRANetwork Music 3 backend, not the conceptmod PEFT script"
        )
    raise ConvertError("could not detect transformer vs language_model slider")


def _strip_known_prefix(module: str) -> str:
    if module.startswith(TF_PREFIX):
        return module[len(TF_PREFIX) :]
    if module.startswith(LM_PREFIX):
        return module[len(LM_PREFIX) :]
    return module


def map_transformer_module(module: str):
    """Map one LoRANetwork transformer module onto a ComfyUI DiT path.

    Returns a native path under ``diffusion_transformer.…``, the sentinel
    ``'qkv'`` plus ``(layer, 'q'|'k'|'v')`` for later fusion, or None.
    """
    name = _strip_known_prefix(module)
    if name in TF_ROOT:
        return TF_ROOT[name]
    match = TF_BLOCK_RE.match(name)
    if not match:
        return None
    idx, tail = match.group(1), match.group(2)
    if tail in TF_QKV:
        return ("qkv", idx, TF_QKV[tail])
    if tail in TF_BLOCK:
        return f"diffusion_transformer.transformer.layers.{idx}.{TF_BLOCK[tail]}"
    return None


def map_lm_module(module: str):
    """Map one LoRANetwork LM module onto a ComfyUI CLIP path stem."""
    name = _strip_known_prefix(module)
    match = LM_LAYER_RE.match(name)
    if not match:
        return None
    idx, proj = match.group(1), match.group(2)
    return f"model.layers.{idx}.self_attn.{proj}"


def _block_diag_up(parts: list[tuple[torch.Tensor, torch.Tensor, float]]):
    """Fuse independent LoRAs that share an input dim onto one Linear.

    ``parts`` is an ordered list of (down, up, alpha). When every adapter has
    the same ``alpha/rank``, the ups stay unscaled and the fused alpha keeps
    that scale (``scale * total_rank``). Otherwise each up is pre-multiplied
    by its own scale and the fused alpha is ``total_rank`` (ComfyUI scale 1).
    """
    downs = [down for down, _up, _alpha in parts]
    ups = [up for _down, up, _alpha in parts]
    if any(down.ndim != 2 or up.ndim != 2 for down, up in zip(downs, ups)):
        raise ConvertError("QKV fusion expects rank-2 LoRA weights")
    in_dims = {int(down.shape[1]) for down in downs}
    if len(in_dims) != 1:
        raise ConvertError(f"QKV fusion input dims disagree: {sorted(in_dims)}")
    ranks = [int(down.shape[0]) for down in downs]
    out_sizes = [int(up.shape[0]) for up in ups]
    total_rank = sum(ranks)
    scales = [float(alpha) / max(rank, 1) for (_d, _u, alpha), rank in zip(parts, ranks)]
    shared = all(abs(scale - scales[0]) <= 1e-6 for scale in scales)
    fused_down = torch.cat(downs, dim=0)
    fused_up = ups[0].new_zeros((sum(out_sizes), total_rank))
    rank_offset = 0
    out_offset = 0
    for up, rank, out, scale in zip(ups, ranks, out_sizes, scales):
        block = up if shared else up * scale
        fused_up[out_offset : out_offset + out, rank_offset : rank_offset + rank] = block
        rank_offset += rank
        out_offset += out
    fused_alpha = (scales[0] * total_rank) if shared else float(total_rank)
    return fused_down, fused_up, fused_alpha


def _collect(path: str) -> tuple[dict, dict]:
    """Read one file into {module: {down, up, alpha}} plus leftover keys."""
    path = os.fspath(path)
    modules: dict[str, dict] = {}
    leftovers: dict[str, torch.Tensor] = {}
    with safe_open(path, framework="pt") as handle:
        for key in handle.keys():
            parts = split_slider_key(key)
            tensor = handle.get_tensor(key)
            if parts is None:
                leftovers[key] = tensor
                continue
            module, side = parts
            slot = modules.setdefault(module, {})
            if side == "alpha":
                slot["alpha"] = float(tensor.detach().float().cpu().reshape(-1)[0])
            else:
                slot[side] = tensor.contiguous().cpu()
    return modules, leftovers


def convert_transformer(path: str):
    """Return (tensors, dropped, unmapped) for a transformer slider."""
    modules, leftovers = _collect(path)
    out, dropped, unmapped = {}, [], list(leftovers)
    qkv_groups: dict[str, dict[str, tuple[str, dict]]] = {}

    for module, tensors in modules.items():
        dest = map_transformer_module(module)
        if dest is None:
            unmapped.append(module)
            continue
        if isinstance(dest, tuple) and dest[0] == "qkv":
            _kind, idx, proj = dest
            qkv_groups.setdefault(idx, {})[proj] = (module, tensors)
            continue
        if "down" not in tensors or "up" not in tensors:
            unmapped.append(module)
            continue
        prefix = f"diffusion_model.{dest}"
        out[f"{prefix}.lora_A.weight"] = tensors["down"].to(TARGET_DTYPE).contiguous()
        out[f"{prefix}.lora_B.weight"] = tensors["up"].to(TARGET_DTYPE).contiguous()
        alpha = tensors.get("alpha", float(tensors["down"].shape[0]))
        out[f"{prefix}.alpha"] = torch.tensor(float(alpha), dtype=torch.float32)

    for idx, group in qkv_groups.items():
        missing = [name for name in ("q", "k", "v") if name not in group]
        if missing:
            unmapped.extend(src for src, _t in group.values())
            unmapped.append(f"incomplete QKV on layer {idx} (missing {missing})")
            continue
        parts = []
        for name in ("q", "k", "v"):
            _src, tensors = group[name]
            if "down" not in tensors or "up" not in tensors:
                unmapped.append(_src)
                parts = []
                break
            parts.append(
                (
                    tensors["down"],
                    tensors["up"],
                    tensors.get("alpha", float(tensors["down"].shape[0])),
                )
            )
        if not parts:
            continue
        fused_down, fused_up, fused_alpha = _block_diag_up(parts)
        prefix = (
            f"diffusion_model.diffusion_transformer.transformer.layers.{idx}"
            ".self_attn.to_qkv"
        )
        out[f"{prefix}.lora_A.weight"] = fused_down.to(TARGET_DTYPE).contiguous()
        out[f"{prefix}.lora_B.weight"] = fused_up.to(TARGET_DTYPE).contiguous()
        out[f"{prefix}.alpha"] = torch.tensor(float(fused_alpha), dtype=torch.float32)

    return out, dropped, unmapped


def convert_language_model(path: str):
    """Return (tensors, dropped, unmapped) for an LM slider.

    Emits ``text_encoders.model.layers.N.self_attn.{q,k,v,o}_proj`` — the
    generic CLIP key ComfyUI builds in ``model_lora_keys_clip``. That matches
    an unmerged Qwen3 text encoder (the HF ``language_model`` the sliders were
    trained on). A ComfyUI TE saved with merged ``qkv_proj`` has no counterpart
    for the separate q/k/v adapters; those keys will log as unloaded.
    """
    modules, leftovers = _collect(path)
    out, dropped, unmapped = {}, [], list(leftovers)
    for module, tensors in modules.items():
        dest = map_lm_module(module)
        if dest is None:
            unmapped.append(module)
            continue
        if "down" not in tensors or "up" not in tensors:
            unmapped.append(module)
            continue
        prefix = f"text_encoders.{dest}"
        out[f"{prefix}.lora_A.weight"] = tensors["down"].to(TARGET_DTYPE).contiguous()
        out[f"{prefix}.lora_B.weight"] = tensors["up"].to(TARGET_DTYPE).contiguous()
        alpha = tensors.get("alpha", float(tensors["down"].shape[0]))
        out[f"{prefix}.alpha"] = torch.tensor(float(alpha), dtype=torch.float32)
    return out, dropped, unmapped


CONVERTERS = {
    "transformer": convert_transformer,
    "language_model": convert_language_model,
}


def classify(path: str):
    """Return ('lora', keys) or ('reason', None) to skip."""
    try:
        with safe_open(path, framework="pt") as handle:
            keys = list(handle.keys())
    except Exception as exc:
        return f"unreadable ({exc})", None
    if not keys:
        return "empty file", None
    if any(k.startswith(("diffusion_model.", "text_encoders.")) for k in keys) and not any(
        split_slider_key(k) and (k.startswith(TF_PREFIX) or k.startswith(LM_PREFIX))
        for k in keys
    ):
        return "already ComfyUI format", None
    if not any(split_slider_key(k) for k in keys):
        return "no LoRANetwork lora_down/lora_up keys", None
    return "lora", keys


def find_loras(roots):
    found = []
    for root in roots:
        if os.path.isfile(root):
            found.append(os.path.abspath(root))
            continue
        for dirpath, _dirnames, filenames in os.walk(root):
            for name in sorted(filenames):
                if not name.endswith(".safetensors"):
                    continue
                if name.endswith(OUT_SUFFIX):
                    continue
                found.append(os.path.join(dirpath, name))
    return sorted(dict.fromkeys(found))


def read_sidecar(path: str) -> dict:
    path = os.fspath(path)
    stem = path[: -len(".safetensors")] if path.endswith(".safetensors") else path
    for candidate in (stem + ".json", os.path.join(os.path.dirname(path), "adapter_config.json")):
        if os.path.isfile(candidate):
            with open(candidate, encoding="utf-8") as handle:
                return json.load(handle)
    return {}


def pattern(key: str) -> str:
    """Collapse layer indices so keys from different blocks compare equal."""
    key = re.sub(r"(layers\.)\d+\.", r"\1N.", key)
    key = re.sub(r"(transformer_blocks\.)\d+\.", r"\1N.", key)
    return key


def check_against(tensors, example: str, kind: str) -> bool:
    """Assert every converted key pattern exists in a reference file."""
    with safe_open(example, framework="pt") as handle:
        known = {pattern(k) for k in handle.keys()}
    mine = {pattern(k) for k in tensors if not k.endswith(".alpha")}
    bad, ok = sorted(mine - known), sorted(mine & known)
    print(
        f"    check: {len(ok)} of {len(ok) + len(bad)} key patterns present "
        f"in {os.path.basename(example)}"
    )
    for item in bad:
        print(f"      NOT IN EXAMPLE: {item}")
    return not bad


def convert_file(path: str, kind: str | None = None, sidecar: dict | None = None):
    """Convert one slider. Raises ConvertError on unmapped keys."""
    path = os.fspath(path)
    sidecar = sidecar if sidecar is not None else read_sidecar(path)
    with safe_open(path, framework="pt") as handle:
        keys = list(handle.keys())
    resolved = kind or detect_kind(keys, sidecar)
    if resolved not in CONVERTERS:
        raise ConvertError(f"unsupported kind {resolved!r}")
    tensors, dropped, unmapped = CONVERTERS[resolved](path)
    if unmapped:
        preview = ", ".join(sorted(unmapped)[:8])
        extra = f" (+{len(unmapped) - 8} more)" if len(unmapped) > 8 else ""
        raise ConvertError(f"{len(unmapped)} unmappable keys: {preview}{extra}")
    if not tensors:
        raise ConvertError("nothing to write")
    return resolved, tensors, dropped


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "roots",
        nargs="*",
        default=["."],
        help="files or directories to walk (default: .)",
    )
    parser.add_argument(
        "--kind",
        choices=sorted(CONVERTERS),
        help="override transformer / language_model detection",
    )
    parser.add_argument("--force", action="store_true", help="overwrite an existing _comfyui output")
    parser.add_argument(
        "--check-against",
        metavar="FILE",
        help="verify key patterns against a reference ComfyUI LoRA",
    )
    parser.add_argument(
        "--skip-lm",
        action="store_true",
        help="skip language-model sliders (transformer-only export)",
    )
    args = parser.parse_args(argv)
    roots = args.roots or ["."]

    candidates = find_loras(roots)
    print(f"scanning {len(roots)} root(s): {len(candidates)} safetensors found")
    failures, written = [], 0

    for path in candidates:
        status, keys = classify(path)
        if keys is None:
            print(f"skip {path}\n    {status}")
            continue
        out_path = path[: -len(".safetensors")] + OUT_SUFFIX
        print(f"convert {path}")
        if os.path.exists(out_path) and not args.force:
            print(f"    exists, skipping (use --force): {out_path}")
            continue

        sidecar = read_sidecar(path)
        try:
            kind = args.kind or detect_kind(keys, sidecar)
        except ConvertError as exc:
            print(f"    ERROR: {exc}")
            failures.append(path)
            continue
        print(
            f"    kind={kind} rank={sidecar.get('rank')} "
            f"alpha={sidecar.get('alpha')} prefix={sidecar.get('prefix')}"
        )
        if kind == "language_model" and args.skip_lm:
            print("    skip language-model slider (--skip-lm)")
            continue
        if kind == "language_model":
            print(
                "    note: LM keys go on the CLIP / MiniMax Music 3 text encoder "
                "(text_encoders.model.layers.*.self_attn.*_proj). "
                "Merged-qkv ComfyUI TEs will not consume q/k/v."
            )

        try:
            kind, tensors, dropped = convert_file(path, kind=kind, sidecar=sidecar)
        except ConvertError as exc:
            print(f"    ERROR: {exc}")
            failures.append(path)
            continue

        print(f"    {len(tensors)} keys converted from {len(keys)} source keys")
        if dropped:
            print(
                f"    {len(dropped)} dropped (no ComfyUI counterpart): "
                f"{', '.join(sorted(dropped)[:3])}"
                f"{' ...' if len(dropped) > 3 else ''}"
            )
        if args.check_against and not check_against(tensors, args.check_against, kind):
            failures.append(path)
            continue

        save_file(tensors, out_path, metadata=METADATA)
        size = os.path.getsize(out_path)
        print(f"    wrote {out_path} ({size / 2**20:.1f} MiB, bf16)")
        written += 1

    print(f"\n{written} written, {len(failures)} failed")
    if failures:
        for path in failures:
            print(f"  FAILED {path}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
