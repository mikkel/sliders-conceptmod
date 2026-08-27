# MiniMax-H3 concept slider (opt-in)

Opt-in UNI trainer for **MiniMax-H3**, MiniMax's omni-modal video+audio generator.
Default Music 3 trainers are unchanged (`--lm_target v9` / `--pole_mode hidden`).

Hub id: **`MiniMaxAI/MiniMax-H3`**.

Load: `ModularPipeline.from_pretrained("MiniMaxAI/MiniMax-H3")` with workflow
`t2va` (default) against task index `FL2VA/model_index.json`.
`Ref2VA/model_index.json` (`ref2va` / omni-reference) is a later path, not the
default.

## What H3 is

H3-Base is a 33B dense single-stream Omni-Transformer (about 13B of that is
AdaLN; those can be cached and skipped at inference-only). This trainer **does
not train AdaLN**.

H3-Encoder is the full pretrained **Qwen3-VL-32B**. It feeds **layer-50 hidden
states** into the Omni-Transformer. Tokenizer + special tokens (`<d>` and the
rest) come from this H3 repo, not stock Qwen.

Visual VAE: temporally causal, f16t4d24, then 1×2×2 patchify (effective 32×
spatial / 4× temporal). Audio VAE: stereo, 32 kHz → 40 Hz latents per channel.

Two self-contained task families under the same repo: **FL2VA** (t2va /
first-last-frame) and **Ref2VA** (omni-reference). Each has
`processor/ tokenizer/ text_encoder/ transformer/ visual_vae/ audio_vae/`.

**H3-Context-IR** and **H3-Regenerate-2K** are hosted services. They are **not**
in the open weights. This trainer does not pretend they ship.

Checkpoints are **CFG-distilled**. Default generate: 768px short side, 4–15s,
24 fps, stereo audio. There is no guider, no `negative_prompt`, and no
`guidance_scale`. Every step is one forward pass. Trainer guidance stays **0**.

## Velocity teacher (honest)

The released Omni-Transformer **is** a flow model. Diffusers
`MiniMaxH3Transformer3DModel.forward` returns **data-pointing velocity**:

`x0 = x_t + sigma * v`

(`MiniMaxH3Scheduler`; video shift 12.0, audio shift 3.0). Video velocity is
`sample`; audio velocity is `audio_sample`.

The trainer calls that **actual packed forward** (text + video + audio rows,
MM-RoPE `(t, h, w)`, `token_tags` 0/1/2). It does **not** fake a conceptmod
`predict_v` / Euler loop. `backend.predict_v(...)` raises `ArchitectureMismatch`.

## UNI analog (not Music 3 lyric-hold)

Yaml slider: **positive / neutral**, unused attributes pinned.

| term | student | teacher | in the loss? |
|---|---|---|---|
| +1 | LoRA on, plus-concept packed sequence | frozen Omni-Transformer velocity on that plus pack | yes |
| scale 0 | adapter off, neu packed sequence | frozen velocity on the neu pack / `encode(neu)` | yes |
| −1 | LoRA scale −1 (canary) | uncond pack | **no** — logged only |
| unused attribute tokens | those rows of `encode(neu)` | yes |
| concept words (in +, not in neu) | free | **not held** |

Train the **Omni-Transformer only**. Frozen: H3-Encoder (Qwen3-VL-32B), visual
VAE, audio VAE, processor, tokenizer.

LoRA attaches to transformer attn **`MiniMaxH3Attention`**:
`to_q` / `to_k` / `to_v` / `to_out.0` (diffusers `MiniMaxH3Transformer3DModel`
/ ModularPipeline partition). The original `FL2VA/transformer` `MiniMaxH3DiTModel`
checkpoint stores fused `qkv_proj` + `out_proj`; live load uses the split
ModularPipeline names. AdaLN (`adaln_proj`) is not wrapped.

## Live train card

GPU box with the Hub checkpoint. Do **not** run this in CI. Do **not** download
MiniMax-H3 (33B Omni-Transformer + 32B encoder) in CI.

```bash
CUDA_VISIBLE_DEVICES=0 python conceptmod/textsliders/train_lora_minimax_h3.py \
  --name age-minimax-h3-uni \
  --model_id MiniMaxAI/MiniMax-H3 \
  --variant FL2VA \
  --workflow t2va \
  --prompts_file conceptmod/textsliders/data/prompts-minimax-h3.yaml \
  --attributes "male, female" \
  --rank 8 --alpha 8 --lr 1e-4 --steps 500 --seed 7 \
  --short_side 768 --guidance 0 --device cuda:0 \
  --save_dir models/age-minimax-h3-uni
```

| field | value |
|---|---|
| hub id | `MiniMaxAI/MiniMax-H3` |
| variant / workflow | FL2VA / t2va |
| task index | `FL2VA/model_index.json` |
| rank / alpha | 8 / 8 |
| short side | 768 |
| CFG | distilled; guidance **0** |
| device | `cuda:0` (live) |
| LoRA host | `MiniMaxH3Attention` `to_q/to_k/to_v/to_out.0` |
| freeze | encoder (Qwen3-VL-32B) + visual VAE + audio VAE + processor + tokenizer |
| not in weights | H3-Context-IR, H3-Regenerate-2K |

Live load needs a current `diffusers` with MiniMax-H3 ModularPipeline. Do **not**
`pip install -r requirements.txt` on the Music 3 env.

## CPU / CI

```bash
PYTHONPATH=. python conceptmod/textsliders/train_lora_minimax_h3.py --dummy --steps 4 --name minimax-h3-dummy
PYTHONPATH=. pytest tests/test_minimax_h3_slider.py -q
```

`--dummy` never downloads Hub weights. Tests use CPU mocks and a tiny fake
packed sequence only.
