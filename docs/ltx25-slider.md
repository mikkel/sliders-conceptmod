# LTX-2.5 concept slider (opt-in)

Opt-in UNI trainer for **LTX-2.5** video. Default Music 3 trainers are
unchanged (`--lm_target v9` / `--pole_mode hidden`).

Hub id: **`Lightricks/LTX-2.5-Diffusers`** (gated). Prefer the Diffusers
pack. The split Comfy pack `Lightricks/LTX-2.5` is **not** the train path.

## Distilled vs full

| | distilled (first card) | full / SFT (fallback) |
|---|---|---|
| subfolder | `transformer/` (in `model_index.json`) | `transformer_full/` (**not** in `model_index`) |
| first download | **exclude** `transformer_full/` | load explicitly |
| sample | `sigmas=DISTILLED_SIGMA_VALUES` — do **not** pass `num_inference_steps`. `modality_scale=1.0` (pipeline treats `>1.0` as on) | drop `sigmas`; restore `use_dynamic_shifting=True, shift_terminal=0.1` |
| guidance | `guidance_scale=1.0`, `audio_guidance_scale=1.0` | CFG 3 / audio 7 + STG / modality |
| STG / modality | STG **0**; `modality_scale=1.0` (off; pipeline treats `>1.0` as on) | STG on block 28 |
| prompt enhancer | **OFF** | optional `google/gemma-4-E2B-it` |

`--diag` first live. If frozen plus vs neu velocity **cos ≈ 1.0**, the
distilled 8-sigma / CFG=1 card may have a **dead expression gap** —
document `transformer_full/` as the fallback. Do **not** silently train
a dead gap. That distilled-vs-SFT UNI geometry difference is a
**hypothesis**, not a prescription.

## What LTX-2.5 is

Same `LTX2Pipeline` / `LTX2VideoTransformer3DModel` /
`AutoencoderKLLTX2Video` classes as LTX-2.3. The text encoder is
**LTX-specific Gemma 4 12B** (`gemma4_unified`). Do not substitute
vanilla Gemma 4.

Install on the live box (Python >= 3.12, recent torch):

```bash
pip install git+https://github.com/huggingface/diffusers
```

LTX-2.5 is **not in a Diffusers release yet**. Do **not**
`pip install -r requirements.txt` on the Music 3 env. Live train needs
an HF token (gated). CI must never download weights.

Constraints: `num_frames % 8 == 1`, H/W divisible by 32. Conv VAE
decode (`pipe.vae`) is the cheap path. Skip `diffusion_decoder` for
this card.

## Velocity teacher (honest)

`LTX2VideoTransformer3DModel.forward` returns
`AudioVisualModelOutput(sample, audio_sample)`. `LTX2Pipeline` treats
that as **flow velocity**:

```
x0 = x_t - sigma * v    # convert_velocity_to_x0
```

The trainer calls that actual forward (video `hidden_states` C=128 +
`audio_hidden_states`). It does **not** fake `predict_v`. Fake train
pack: `num_frames=9` (`8k+1`), 32×32, pack at `proj_in.in_features`
(128) — not inner_dim 4096.

## UNI analog (Sana lesson, not H3 caption coupling)

Train **+1 on the neu caption**; plus is teacher-only. Infer at scale 1
uses neu+LoRA. If student(+1) trains on the plus caption, scale 1 on
neu will not hit the concept (Sana age dud / H3 caption coupling).

| term | student | teacher | in the loss? |
|---|---|---|---|
| +1 | LoRA on, **neu** pack (infer path) | frozen velocity on plus pack | yes |
| scale 0 | adapter off, neu pack | frozen velocity on neu | yes |
| −1 | LoRA scale −1 (canary) | uncond pack | **no** — logged only |
| non-concept tokens (default) | matching rows of `encode(neu)` **PRE-connector** | yes |
| unused attribute tokens | subset of non-concept; `--hold_mode attributes` holds only these | yes |
| concept words (in +, not in neu) | free | **not held** |

`--hold_mode` default `non_concept`. `--hold_mode attributes` is the
leaky subset.

**Hold must be PRE-connector.** Order: tokenize
(`add_special_tokens=False`; Gemma-4 prepends a leading space) → frozen
TE features (current diffusers stacks every hidden layer and
`flatten(2, 3)`) → `apply_unused_hold` → **left-pad to a multiple of
128** (1024 like the pipeline) → **then** `pipe.connectors(...)` →
transformer. Live `LTX2TextConnectors` require
`seq_len % num_learnable_registers == 0` (registers default 128) and
replace padding in-place. Dummy connectors enforce that same %128
contract. After connectors, T is not 1:1 with prompt tokens. Dummy
tests fail if hold is applied after connectors.

Train `pack_t2v` timestep is `(B, num_video_tokens)` already scaled by
`timestep_scale_multiplier` (1000). Distilled UNI picks a sigma from
`DISTILLED_SIGMA_VALUES` (not a generic 0.5 — unscaled 0.5 is t≈0).
`forward_velocity` passes `audio_num_frames` (live RoPE
`prepare_audio_coords(None)` is a TypeError).

Current `LTX2Pipeline._get_gemma_prompt_embeds` only stacks hidden
layers and `flatten(2, 3)`. Mean-center/scale is **inside**
`LTX2TextConnectors.forward` (`per_layer_masked_mean_norm` on the
LTX-2.0 path; `per_token_rms_norm` when `per_modality_projections`).
That is what `text_proj_in` / the 1-D connectors see. Hold still runs
on the stacked TE rows **before** `pipe.connectors(...)`.

Fail closed if the + prompt has no concept-word tokens.

## LoRA (video-only v1)

Verified on current diffusers `LTX2Attention` / `named_modules()`:

| wrap | skip |
|---|---|
| `attn1.to_q` / `to_k` / `to_v` / `to_out.0` | `audio_attn1`, `audio_attn2` |
| `attn2.to_q` / `to_k` / `to_v` / `to_out.0` | `audio_to_video_attn`, `video_to_audio_attn` |
| | AdaLN, FFN, `to_out.1` (Dropout), `to_gate_logits` |

A smile slider must not rewrite foley. Official-style `"to_q"` matches
all streams and is **too broad**. A naive `.endswith(".attn1")` also
matches `audio_attn1`.

`LTX2VideoTransformer3DModel` already has `PeftAdapterMixin`. Live
prefers PEFT. `set_adapter_scale` no-ops (Krea #74) — write
`LoraLayer.scaling = (alpha/r) * scale` via
`apply_continuous_lora_scale`. LoRA-up init `N(0, 0.02)`, not zeros
(UNI identity). Dummy uses a tiny `LTX2Attention` stand-in (no PEFT,
no Hub).

Frozen: text encoder, connectors, video VAE, audio VAE, vocoder,
tokenizer, processor, prompt enhancer, duration head, diffusion
decoder.

## Live train card (smile / happy)

Rent an **A100 80GB** community box (~**$1.19/hr**). Put Gemma 4 12B
on **CPU** (`--encoder_device cpu`). Do **not** `pipe.to(cuda:0)`
TE+DiT together. **Not a 4090. Not a B300.**

First samples: **49 frames**, **544×960**, conv VAE, distilled sigmas.

Prompts: `prompts-ltx25-smile.yaml` — teeth vs closed-mouth, same
subject / clothes / framing / motion / light / sound. Stronger smile
positives. Do not prefix attributes. `concept_words: smiling, smile,
happy, joyful, teeth`. No chiaroscuro yaml.

```bash
# first live: diagnose the expression gap (no train)
CUDA_VISIBLE_DEVICES=0 python conceptmod/textsliders/diag_ltx25_uni.py \
  --prompts_file conceptmod/textsliders/data/prompts-ltx25-smile.yaml \
  --device cuda:0 --encoder_device cpu \
  --save_dir models/smile-ltx25-uni/diag

CUDA_VISIBLE_DEVICES=0 python conceptmod/textsliders/train_lora_ltx25.py \
  --name smile-ltx25-uni \
  --model_id Lightricks/LTX-2.5-Diffusers \
  --transformer_subfolder transformer \
  --prompts_file conceptmod/textsliders/data/prompts-ltx25-smile.yaml \
  --attributes "male, female" \
  --rank 8 --alpha 8 --lr 1e-4 --steps 500 --seed 7 \
  --lora_up_init_std 0.02 \
  --hold_mode non_concept \
  --device cuda:0 --encoder_device cpu \
  --sample_scales 0,0.5,1 \
  --sample_num_frames 49 --sample_height 544 --sample_width 960 \
  --save_dir models/smile-ltx25-uni
```

If `--diag` reports `dead_gap` / expression_gap cos ≈ 1.0:

```bash
# SFT fallback — do not silently train distilled
--transformer_subfolder transformer_full
```

SFT sample restores `use_dynamic_shifting=True, shift_terminal=0.1`
and does **not** pass `DISTILLED_SIGMA_VALUES`.

| field | value |
|---|---|
| hub id | `Lightricks/LTX-2.5-Diffusers` |
| DiT | distilled `transformer/` (exclude `transformer_full/` on first download) |
| text encoder | LTX Gemma 4 12B (`gemma4_unified`) on **CPU** |
| rank / alpha | 8 / 8 |
| LoRA-up init | `N(0, 0.02)` |
| LoRA hosts | `LTX2Attention` `attn1` + `attn2` `to_q/to_k/to_v/to_out.0` |
| train pack | 9 frames, 32×32, `proj_in.in_features` 128 |
| sample | **49** frames, **544×960**, conv VAE |
| distilled sigmas | `1.0, 0.99375, 0.9875, 0.98125, 0.975, 0.909375, 0.725, 0.421875` |
| CFG / STG / modality | 1.0 / **0** / **1.0** (pipeline treats ``>1.0`` as on) |
| prompt enhancer | OFF |
| infer caption | **neu** at every sample scale |
| GPU | **A100 80GB** community ~$1.19/hr. Not 4090. Not B300. |
| torch | recent CUDA wheel on the A100 box (not Music 3; not B300 `sm_103`) |

Sample a saved adapter without training:

```bash
CUDA_VISIBLE_DEVICES=0 python conceptmod/textsliders/train_lora_ltx25.py \
  --name smile-ltx25-uni \
  --prompts_file conceptmod/textsliders/data/prompts-ltx25-smile.yaml \
  --steps 0 --load_ltx_lora models/smile-ltx25-uni \
  --sample_scales 0,0.5,1 \
  --sample_num_frames 49 --sample_height 544 --sample_width 960 \
  --device cuda:0 --encoder_device cpu \
  --save_dir models/smile-ltx25-uni
```

## CPU / CI

```bash
PYTHONPATH=. python conceptmod/textsliders/train_lora_ltx25.py --dummy --steps 4 --name ltx25-dummy
PYTHONPATH=. python conceptmod/textsliders/diag_ltx25_uni.py --dummy --save_dir /tmp/ltx25-diag
PYTHONPATH=. pytest tests/test_ltx25_slider.py -q
python scripts/smoke_ltx25_slider.py
```

`--dummy` never downloads Hub weights. Tests use CPU mocks and a tiny
fake pack only.
