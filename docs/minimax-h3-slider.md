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

## Live train card (chiaroscuro-minimax-h3-uni)

First **live** H3 train+sample. Prefer **1× B200 / B300** (bf16 ModularPipeline
`pipe.to(cuda:0)` is ~135 GB: transformer 61.7 + Qwen3-VL-32B 62.1 + VAEs).
Same flags as the age card (rank 8, alpha 8, lr 1e-4, steps 500, short_side
768, guidance 0, FL2VA / t2va). Prompts are a **chiaroscuro / dramatic
lighting** set: same concrete subject on neu and plus (person / interior
room with chair+table / still-life vase). Neu is flat even lighting / soft
fill / low contrast (photographic, not cartoon). Plus is chiaroscuro,
dramatic single-source side light, deep shadows, high contrast. Minus is a
washed-out flat / featureless lighting canary only.

After train the script writes short t2va mp4s under `save_dir/samples/` at
LoRA scales **0** and **+1** (add `0.5` with `--sample_scales 0,0.5,1`) for
each unique yaml target. Guidance stays **0**. Sample-only reload:

`--steps 0 --load_h3_lora <dir>` (custom `lora_h3-…` keys, not PEFT).

Gate is **look across sampled frames** (contrast moves, subject holds),
not last-50 c+.

```bash
CUDA_VISIBLE_DEVICES=0 python conceptmod/textsliders/train_lora_minimax_h3.py \
  --name chiaroscuro-minimax-h3-uni \
  --model_id MiniMaxAI/MiniMax-H3 \
  --variant FL2VA \
  --workflow t2va \
  --prompts_file conceptmod/textsliders/data/prompts-minimax-h3-chiaroscuro.yaml \
  --config_file conceptmod/textsliders/data/config-minimax-h3-chiaroscuro.yaml \
  --attributes "male, female" \
  --rank 8 --alpha 8 --lr 1e-4 --steps 500 --seed 7 \
  --short_side 768 --guidance 0 --device cuda:0 \
  --sample_scales 0,1 --sample_duration 5 --sample_fps 24 \
  --save_dir models/chiaroscuro-minimax-h3-uni
```

| field | value |
|---|---|
| hub id | `MiniMaxAI/MiniMax-H3` |
| variant / workflow | FL2VA / t2va |
| task index | `FL2VA/model_index.json` |
| rank / alpha | 8 / 8 |
| short side | 768 (sample canvas 1344×768 16:9) |
| sample duration / fps | **5 s** / 24 (124 frames after `17n+5` snap; ~5.17 s) |
| sample scales | 0 and +1 (optional 0.5) |
| CFG | distilled; guidance **0** |
| device | `cuda:0` on **B200 / B300** (single-device default) |
| 2×H100 | `--device cuda:0 --encoder_device cuda:1` (no blanket `pipe.to`) |
| LoRA host | `MiniMaxH3Attention` `to_q/to_k/to_v/to_out.0` |
| freeze | encoder (Qwen3-VL-32B) + visual VAE + audio VAE + processor + tokenizer |
| gate | sampled-frame look (contrast moves, subject holds); not last-50 c+ |
| not in weights | H3-Context-IR, H3-Regenerate-2K |

Memory-fit sample defaults: **5 s**, **24 fps**, **768** short side. H3's
published window is 5–15 s. A 4 s request snaps up to 107 frames (~4.46 s)
and may be rejected live; 4.5 s snaps to 124 (same as 5 s). Tighter canvas:
`--sample_short_side 544` → 960×544. Subset: `--sample_max_rows 1`.

2×H100 (80 GB) does **not** hold the full bf16 stack on one card. Put the
encoder on the second GPU and leave transformer + VAEs on cuda:0:

```bash
CUDA_VISIBLE_DEVICES=0,1 python conceptmod/textsliders/train_lora_minimax_h3.py \
  --name chiaroscuro-minimax-h3-uni \
  --model_id MiniMaxAI/MiniMax-H3 \
  --variant FL2VA \
  --workflow t2va \
  --prompts_file conceptmod/textsliders/data/prompts-minimax-h3-chiaroscuro.yaml \
  --config_file conceptmod/textsliders/data/config-minimax-h3-chiaroscuro.yaml \
  --attributes "male, female" \
  --rank 8 --alpha 8 --lr 1e-4 --steps 500 --seed 7 \
  --short_side 768 --guidance 0 \
  --device cuda:0 --encoder_device cuda:1 \
  --save_dir models/chiaroscuro-minimax-h3-uni
```

Sample a saved adapter without training:

```bash
CUDA_VISIBLE_DEVICES=0 python conceptmod/textsliders/train_lora_minimax_h3.py \
  --name chiaroscuro-minimax-h3-uni \
  --prompts_file conceptmod/textsliders/data/prompts-minimax-h3-chiaroscuro.yaml \
  --steps 0 --load_h3_lora models/chiaroscuro-minimax-h3-uni \
  --sample_scales 0,1 --sample_duration 5 --sample_fps 24 \
  --short_side 768 --guidance 0 --device cuda:0 \
  --save_dir models/chiaroscuro-minimax-h3-uni
```

If that short run is weak (contrast barely moves, or the subject drifts),
escalate. VRAM / iterations are unrestricted until **sampled videos** look
solid — watch the clips, do not gate on last-50 c+. First bump is rank 16
/ 800 steps; if still weak take it to 1500. Recommended escalate:

```bash
CUDA_VISIBLE_DEVICES=0 python conceptmod/textsliders/train_lora_minimax_h3.py \
  --name chiaroscuro-minimax-h3-uni-r16 \
  --model_id MiniMaxAI/MiniMax-H3 \
  --variant FL2VA \
  --workflow t2va \
  --prompts_file conceptmod/textsliders/data/prompts-minimax-h3-chiaroscuro.yaml \
  --config_file conceptmod/textsliders/data/config-minimax-h3-chiaroscuro.yaml \
  --attributes "male, female" \
  --rank 16 --alpha 16 --lr 1e-4 --steps 1200 --seed 7 \
  --short_side 768 --guidance 0 --device cuda:0 \
  --sample_scales 0,1 --sample_duration 5 --sample_fps 24 \
  --save_dir models/chiaroscuro-minimax-h3-uni-r16
```

| field | value |
|---|---|
| name | `chiaroscuro-minimax-h3-uni-r16` |
| rank / alpha | 16 / 16 |
| steps | 1200 (800 first bump; 1500 if still weak) |
| short side | 768 |
| sample duration / fps | 5 s / 24 |
| CFG | distilled; guidance **0** |
| device | B200 / B300 `cuda:0`; 2×H100 add `--encoder_device cuda:1` |
| gate | watch sampled videos (contrast moves, subject holds) |

Do **not** run this in CI. Do **not** download MiniMax-H3 in CI.

## CPU / CI

```bash
PYTHONPATH=. python conceptmod/textsliders/train_lora_minimax_h3.py --dummy --steps 4 --name minimax-h3-dummy
PYTHONPATH=. pytest tests/test_minimax_h3_slider.py -q
```

`--dummy` never downloads Hub weights. Tests use CPU mocks and a tiny fake
packed sequence only.
