#!/usr/bin/env python3
"""Convert Anima-Turbo v1.1 Comfy weights to Diffusers (preview only).

Train stays on ``circlestone-labs/Anima-Base-v1.0-Diffusers``. Turbo is
for faster stock/preview smoke (CFG 1, 8–12 steps), not the next train
unless smoke shows a larger closed-mouth vs teeth gap.

Downloads the three official Hub files, runs huggingface/diffusers
``scripts/convert_anima_to_diffusers.py`` with ``--save_pipeline
--dtype bf16``, and reuses Qwen + T5 tokenizers from a local or Hub
``Anima-Base-v1.0-Diffusers`` checkout.

    python scripts/convert_anima_turbo_diffusers.py
    python scripts/convert_anima_turbo_diffusers.py --print-recipe
    python scripts/convert_anima_turbo_diffusers.py --dry-run

License: CircleStone Labs Non-Commercial (NC). Ignore community
Anima-1.0-Turbo-Diffusers (v1.0 only, wrong VAE class).
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import urllib.request
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from conceptmod.textsliders.anima_slider import (
    DEFAULT_MODEL_ID,
    TURBO_COMFY_REPO,
    TURBO_CONVERT_COSMOS_SCRIPT,
    TURBO_CONVERT_DTYPE,
    TURBO_CONVERT_SCRIPT,
    TURBO_DIFFUSERS_OUTPUT,
    TURBO_IGNORE_COMMUNITY,
    TURBO_LICENSE,
    TURBO_PREVIEW_ONLY,
    TURBO_SAMPLE_CFG,
    TURBO_SAMPLE_STEPS,
    TURBO_SAMPLE_STEPS_RANGE,
    TURBO_TEXT_ENCODER_FILE,
    TURBO_TRANSFORMER_FILE,
    TURBO_VAE_FILE,
    turbo_preview_card,
    turbo_preview_sample_command,
)

DEFAULT_CACHE = Path(".cache/anima-convert")
DEFAULT_OUTPUT = Path(TURBO_DIFFUSERS_OUTPUT)
TOKENIZER_SUBDIRS = ("tokenizer", "t5_tokenizer")
HUB_FILES = (
    TURBO_TRANSFORMER_FILE,
    TURBO_TEXT_ENCODER_FILE,
    TURBO_VAE_FILE,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=DEFAULT_CACHE,
        help="local dir for the three Hub files + official convert scripts",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Diffusers output dir (default {TURBO_DIFFUSERS_OUTPUT})",
    )
    parser.add_argument(
        "--base-diffusers",
        type=str,
        default=DEFAULT_MODEL_ID,
        help=(
            "local Anima-Base-v1.0-Diffusers checkout or Hub id "
            "(tokenizers only; default is the live train target)"
        ),
    )
    parser.add_argument(
        "--diffusers-scripts",
        type=Path,
        default=None,
        help="local huggingface/diffusers/scripts dir (optional)",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default=TURBO_CONVERT_DTYPE,
        help="official convert --dtype (default bf16)",
    )
    parser.add_argument(
        "--print-recipe",
        action="store_true",
        help="print the preview-only convert recipe and exit (no Hub)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="print planned downloads + convert argv; do not download or convert",
    )
    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="reuse files already in --cache-dir",
    )
    return parser.parse_args(argv)


def hub_file_plan(cache_dir: Path) -> list[dict[str, str]]:
    """The three Comfy single-files Turbo convert needs."""
    rows = []
    for filename in HUB_FILES:
        rows.append(
            {
                "repo": TURBO_COMFY_REPO,
                "filename": filename,
                "dest": str(cache_dir / filename),
            }
        )
    return rows


def official_convert_argv(
    *,
    convert_script: Path,
    transformer: Path,
    text_encoder: Path,
    vae: Path,
    qwen_tokenizer: Path,
    t5_tokenizer: Path,
    output: Path,
    dtype: str = TURBO_CONVERT_DTYPE,
) -> list[str]:
    """Exact official CLI. Always --save_pipeline."""
    return [
        sys.executable,
        str(convert_script),
        "--transformer_ckpt_path",
        str(transformer),
        "--text_encoder_ckpt_path",
        str(text_encoder),
        "--vae_ckpt_path",
        str(vae),
        "--qwen_tokenizer_path",
        str(qwen_tokenizer),
        "--t5_tokenizer_path",
        str(t5_tokenizer),
        "--output_path",
        str(output),
        "--save_pipeline",
        "--dtype",
        str(dtype),
    ]


def convert_recipe(
    *,
    cache_dir: Path = DEFAULT_CACHE,
    output: Path = DEFAULT_OUTPUT,
    base_diffusers: str = DEFAULT_MODEL_ID,
    dtype: str = TURBO_CONVERT_DTYPE,
) -> dict[str, Any]:
    """CPU-pure recipe. No Hub. Used by --print-recipe and tests."""
    files = hub_file_plan(cache_dir)
    by_name = {Path(row["filename"]).name: Path(row["dest"]) for row in files}
    tokenizers = {
        "qwen": str(Path(base_diffusers) / "tokenizer"),
        "t5": str(Path(base_diffusers) / "t5_tokenizer"),
        "source": base_diffusers,
    }
    return {
        "role": "preview_only",
        "train_on": DEFAULT_MODEL_ID,
        "turbo_is_not_a_train_target": True,
        "license": TURBO_LICENSE,
        "ignore_community": TURBO_IGNORE_COMMUNITY,
        "hub_files": files,
        "tokenizers": tokenizers,
        "convert_script": TURBO_CONVERT_SCRIPT,
        "convert_cosmos_script": TURBO_CONVERT_COSMOS_SCRIPT,
        "convert_splits": {
            "llm_adapter": "AnimaTextConditioner",
            "rest": "CosmosTransformer3DModel",
        },
        "vae_class": "AutoencoderKLQwenImage",
        "output": str(output),
        "dtype": dtype,
        "save_pipeline": True,
        "sample_cfg": TURBO_SAMPLE_CFG,
        "sample_steps": TURBO_SAMPLE_STEPS,
        "sample_steps_range": list(TURBO_SAMPLE_STEPS_RANGE),
        "official_argv": official_convert_argv(
            convert_script=Path("convert_anima_to_diffusers.py"),
            transformer=by_name["anima-turbo-v1.1.safetensors"],
            text_encoder=by_name["qwen_3_06b_base.safetensors"],
            vae=by_name["qwen_image_vae.safetensors"],
            qwen_tokenizer=Path(tokenizers["qwen"]),
            t5_tokenizer=Path(tokenizers["t5"]),
            output=output,
            dtype=dtype,
        ),
        "sample_command": turbo_preview_sample_command(model_id=str(output)),
        "preview_only": TURBO_PREVIEW_ONLY,
    }


def download_hub_files(cache_dir: Path, *, skip: bool) -> dict[str, Path]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    out: dict[str, Path] = {}
    for row in hub_file_plan(cache_dir):
        dest = Path(row["dest"])
        if dest.is_file():
            out[Path(row["filename"]).name] = dest
            continue
        if skip:
            raise SystemExit(f"missing {dest} (--skip-download)")
        print(f"download {row['repo']} {row['filename']} -> {dest}", flush=True)
        try:
            from huggingface_hub import hf_hub_download
        except ImportError as exc:
            raise SystemExit(
                "huggingface_hub is required to download Anima-Turbo weights."
            ) from exc
        path = hf_hub_download(
            repo_id=row["repo"],
            filename=row["filename"],
            local_dir=str(cache_dir),
        )
        downloaded = Path(path)
        dest.parent.mkdir(parents=True, exist_ok=True)
        if downloaded.resolve() != dest.resolve() and not dest.exists():
            shutil.copy2(downloaded, dest)
        out[Path(row["filename"]).name] = dest if dest.is_file() else downloaded
    return out


def resolve_tokenizer_dirs(base_diffusers: str, cache_dir: Path) -> tuple[Path, Path]:
    """Reuse tokenizer/ and t5_tokenizer/ from a local or Hub Base checkout."""
    local = Path(base_diffusers)
    qwen = local / "tokenizer"
    t5 = local / "t5_tokenizer"
    if qwen.is_dir() and t5.is_dir():
        return qwen, t5
    dest = cache_dir / "Anima-Base-v1.0-Diffusers"
    qwen = dest / "tokenizer"
    t5 = dest / "t5_tokenizer"
    if qwen.is_dir() and t5.is_dir():
        return qwen, t5
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise SystemExit(
            "Need a local Anima-Base-v1.0-Diffusers checkout with "
            "tokenizer/ and t5_tokenizer/, or huggingface_hub to fetch them."
        ) from exc
    print(
        f"download tokenizers from {base_diffusers} -> {dest}",
        flush=True,
    )
    snapshot_download(
        repo_id=base_diffusers,
        allow_patterns=["tokenizer/*", "t5_tokenizer/*"],
        local_dir=str(dest),
    )
    if not (qwen.is_dir() and t5.is_dir()):
        raise SystemExit(f"tokenizers missing under {dest}")
    return qwen, t5


def _download_url(url: str, dest: Path) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"fetch {url} -> {dest}", flush=True)
    with urllib.request.urlopen(url) as resp, dest.open("wb") as fh:
        shutil.copyfileobj(resp, fh)
    return dest


def resolve_convert_scripts(scripts_dir: Path | None, cache_dir: Path) -> Path:
    """Official convert_anima_to_diffusers.py plus its cosmos sibling."""
    if scripts_dir is not None:
        anima = scripts_dir / "convert_anima_to_diffusers.py"
        cosmos = scripts_dir / "convert_cosmos_to_diffusers.py"
        if anima.is_file() and cosmos.is_file():
            return anima
        raise SystemExit(
            f"--diffusers-scripts needs convert_anima_to_diffusers.py and "
            f"convert_cosmos_to_diffusers.py under {scripts_dir}"
        )
    dest = cache_dir / "diffusers_scripts"
    anima = dest / "convert_anima_to_diffusers.py"
    cosmos = dest / "convert_cosmos_to_diffusers.py"
    if not anima.is_file():
        _download_url(TURBO_CONVERT_SCRIPT, anima)
    if not cosmos.is_file():
        _download_url(TURBO_CONVERT_COSMOS_SCRIPT, cosmos)
    return anima


def write_preview_readme(output: Path) -> Path:
    """Stamp the converted tree so nobody trains on it by accident."""
    text = (
        "# Anima-Turbo v1.1 Diffusers (preview only)\n\n"
        "**Do not train LoRAs here.** CircleStone: train on "
        f"`{DEFAULT_MODEL_ID}`.\n\n"
        f"Sample at CFG {TURBO_SAMPLE_CFG:g}, "
        f"{TURBO_SAMPLE_STEPS_RANGE[0]}–{TURBO_SAMPLE_STEPS_RANGE[1]} steps "
        f"(this repo: `--cfg 1 --sample_steps {TURBO_SAMPLE_STEPS}`).\n\n"
        f"Converted with official `huggingface/diffusers` "
        f"`scripts/convert_anima_to_diffusers.py --save_pipeline "
        f"--dtype {TURBO_CONVERT_DTYPE}`.\n\n"
        f"License: {TURBO_LICENSE}. Ignore community "
        f"`{TURBO_IGNORE_COMMUNITY}` (v1.0 only, wrong VAE class).\n"
    )
    path = output / "PREVIEW_ONLY.md"
    output.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def copy_tokenizers(qwen: Path, t5: Path, output: Path) -> None:
    """Belt-and-suspenders if save_pipeline omitted tokenizer dirs."""
    for src, name in ((qwen, "tokenizer"), (t5, "t5_tokenizer")):
        dest = output / name
        if dest.is_dir():
            continue
        shutil.copytree(src, dest)


def run_convert(argv: list[str], cwd: Path) -> None:
    env = os.environ.copy()
    env["PYTHONPATH"] = (
        str(cwd) + os.pathsep + env.get("PYTHONPATH", "")
    )
    print("run:", " ".join(argv), flush=True)
    subprocess.run(argv, cwd=str(cwd), env=env, check=True)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    recipe = convert_recipe(
        cache_dir=args.cache_dir,
        output=args.output,
        base_diffusers=args.base_diffusers,
        dtype=args.dtype,
    )
    if args.print_recipe:
        print(json.dumps({**turbo_preview_card(), "convert": recipe}, indent=2))
        return 0
    if args.dry_run:
        print(json.dumps(recipe, indent=2))
        return 0

    files = download_hub_files(args.cache_dir, skip=args.skip_download)
    qwen, t5 = resolve_tokenizer_dirs(args.base_diffusers, args.cache_dir)
    convert_script = resolve_convert_scripts(args.diffusers_scripts, args.cache_dir)
    cmd = official_convert_argv(
        convert_script=convert_script,
        transformer=files["anima-turbo-v1.1.safetensors"],
        text_encoder=files["qwen_3_06b_base.safetensors"],
        vae=files["qwen_image_vae.safetensors"],
        qwen_tokenizer=qwen,
        t5_tokenizer=t5,
        output=args.output,
        dtype=args.dtype,
    )
    run_convert(cmd, cwd=convert_script.parent)
    copy_tokenizers(qwen, t5, args.output)
    preview = write_preview_readme(args.output)
    print(f"wrote preview-only tree {args.output} ({preview.name})", flush=True)
    print(turbo_preview_sample_command(model_id=str(args.output)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
