# LTX-2.5 concept slider (embed-match UNI)

Opt-in UNI trainer for **LTX-2.5** video. Default Music 3 trainers are
unchanged (`--lm_target v9` / `--pole_mode hidden`).

Hub id: **`Lightricks/LTX-2.5-Diffusers`** (gated). Prefer the Diffusers
pack. The split Comfy pack `Lightricks/LTX-2.5` is **not** the train path.

**Default recipe is post-connector video embed-match**, validated
2026-09-02 on dual RTX A6000 (pop-os) against
`Lightricks/LTX-2.5-Diffusers`. Do **not** reintroduce DiT velocity-UNI
as the smile / chiaroscuro default.

## Distilled vs full

| | distilled (first card) | full / SFT (fallback) |
|---|---|---|
| subfolder | `transformer/` (in `model_index.json`) | `transformer_full/` (**not** in `model_index`) |
| first download | **exclude** `transformer_full/` | load explicitly |
| sample | `sigmas=DISTILLED_SIGMA_VALUES` — do **not** pass `num_inference_steps`. `modality_scale=1.0` (pipeline treats `>1.0` as on) | drop `sigmas`; restore `use_dynamic_shifting=True, shift_terminal=0.1` |
| guidance | `guidance_scale=1.0`, `audio_guidance_scale=1.0` | CFG 3 / audio 7 + STG / modality |
| STG / modality | STG **0**; `modality_scale=1.0` (off; pipeline treats `>1.0` as on) | STG on block 28 |
| prompt enhancer | **OFF** | optional `google/gemma-4-E2B-it` |

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

## Why DiT velocity UNI failed

These paths were tried and **did not edit**:

* DiT velocity UNI on `attn1` / `attn2` LoRA
* connector-only LoRA
* TE-attn LoRA with a velocity loss

Live loss sat flat ~**3.17**, no smile. Frozen plus vs neu **velocity
cos ~0.9999**. High-dim velocity is a dead teacher for decode concepts
that live in the text path. `--recipe ltx25_uni_velocity` remains
opt-in for ablation only. Do not make it the smile/chiaro default.

`--diag` still logs the velocity gap as a **negative control**. If
plus vs neu velocity cos ≈ 1.0, that is expected on distilled — it is
not a reason to switch the train target to `transformer_full/`. The
working gap is post-connector video, not DiT velocity.

That distilled-vs-SFT UNI geometry difference is a **hypothesis**,
not a prescription.

## Working diagnostic (embed transplant)

Encode plus vs neu. After video connectors, valid-row **mean_cos
~0.68**. Transplanting plus concept-token embeds (or the full plus
hidden) onto neu conditioning produces teeth / smile in decode while
holding identity. That is the teacher: frozen `encode(plus)`, not
`v_plus`.

```bash
# first live: diagnose the text-path gap (no train)
python conceptmod/textsliders/diag_ltx25_uni.py \
  --prompts_file conceptmod/textsliders/data/prompts-ltx25-smile.yaml \
  --device cuda:0 --encoder_device cuda:1 \
  --save_dir models/smile-ltx25-uni/diag
```

Read `post_cos` (working) vs `expr_cos` (dead velocity control).

## Embed-match UNI (default)

| term | student | teacher | in the loss? |
|---|---|---|---|
| +1 | `encode(neu)+LoRA(scale)` | frozen `encode(plus)` post-connector **video** | yes — MSE + rel-L2 on **valid** rows |
| scale 0 | adapter off | — | no |
| −1 | LoRA scale −1 (canary) | uncond post-connector video | **no** — logged only |
| non-concept tokens (default) | matching rows of `encode(neu)` **PRE-connector** | yes (teacher hold) |
| unused attribute tokens | subset of non-concept; `--hold_mode attributes` holds only these | yes |
| concept words (in +, not in neu) | free | **not held** |

`--hold_mode` default `non_concept`. `--hold_mode attributes` is the
leaky subset.

**Hold must be PRE-connector.** Order: tokenize
(`add_special_tokens=False`; Gemma-4 prepends a leading space) → TE
features (current diffusers stacks every hidden layer and
`flatten(2, 3)`) → `apply_unused_hold` → **left-pad to a multiple of
128** (1024 like the pipeline) → **then** `pipe.connectors(...)`.
Live `LTX2TextConnectors` require
`seq_len % num_learnable_registers == 0` (registers default 128) and
replace padding in-place. Dummy connectors enforce that same %128
contract. After connectors, T is not 1:1 with prompt tokens. Dummy
tests fail if hold is applied after connectors.

Loss sees **valid** post-connector video rows only (attention mask).
Pad / register-replaced zeros do not enter MSE or rel-L2.

Fail closed if the + prompt has no concept-word tokens.

Train +1 on the **neu** caption; plus is teacher-only. Infer at every
sample scale uses neu+LoRA. If student(+1) trains on the plus caption,
scale 1 on neu will not hit the concept (Sana age dud / H3 caption
coupling).

## LoRA hosts (embed-match)

| wrap | skip |
|---|---|
| video connectors (attn q/k/v/o + video mix) | audio connectors |
| TE last-N layers (`--te_last_n 4`) `q_proj/k_proj/v_proj/o_proj` | earlier TE layers |
| | **DiT** (frozen; park CPU during embed-only train) |
| | AdaLN, FFN, audio attn, a2v / v2a |

A smile slider must not rewrite foley. Official-style `"to_q"` on the
DiT matches all streams and is **too broad**. Velocity `attn1`/`attn2`
hosts are the failed card.

`set_adapter_scale` no-ops (Krea #74) — write
`LoraLayer.scaling = (alpha/r) * scale` via
`apply_continuous_lora_scale`. LoRA-up init `N(0, 0.02)`, not zeros
(UNI identity). Dummy uses a tiny Gemma-like TE + `LTX2Attention`
video connector (no PEFT, no Hub).

Frozen: DiT, video VAE, audio VAE, vocoder, tokenizer, processor,
prompt enhancer, duration head, diffusion decoder. TE / connector
**base** weights stay frozen; only LoRA trains.

## Live train card (smile v1g / chiaro v1)

Validated on **dual RTX A6000** (pop-os), 2026-09-02. TE + connectors
on **`cuda:1`**, DiT on **`cuda:0`** for sample (or DiT CPU during
embed-only train). Do **not** `pipe.to` TE+DiT together.

Hyperparams that worked: **rank 16**, **~700 steps**, **lr 2e-4**,
**TE last_n=4**, **seed 7**.

Sample scales **always include −1, 0, 0.5, 1** on the neu caption.

First samples: **49 frames**, **544×960**, conv VAE, distilled sigmas.

Prompts:

* `prompts-ltx25-smile.yaml` — teeth vs closed-mouth, same subject /
  clothes / framing / motion / light / sound.
* `prompts-ltx25-chiaroscuro.yaml` — Rembrandt vs soft fill, same
  locked non-concept structure (person + still life).

```bash
# smile (validated v1g)
python conceptmod/textsliders/train_lora_ltx25.py \
  --name smile-ltx25-uni \
  --model_id Lightricks/LTX-2.5-Diffusers \
  --transformer_subfolder transformer \
  --recipe ltx25_uni_embed \
  --prompts_file conceptmod/textsliders/data/prompts-ltx25-smile.yaml \
  --attributes "male, female" \
  --rank 16 --alpha 16 --lr 2e-4 --steps 700 --seed 7 \
  --te_last_n 4 --embed_rel_l2_weight 1.0 \
  --lora_up_init_std 0.02 \
  --hold_mode non_concept \
  --device cuda:0 --encoder_device cuda:1 \
  --sample_scales=-1,0,0.5,1 \
  --sample_num_frames 49 --sample_height 544 --sample_width 960 \
  --save_dir models/smile-ltx25-uni

# chiaroscuro (validated v1, same recipe)
python conceptmod/textsliders/train_lora_ltx25.py \
  --name chiaro-ltx25-uni \
  --prompts_file conceptmod/textsliders/data/prompts-ltx25-chiaroscuro.yaml \
  --config_file conceptmod/textsliders/data/config-ltx25-chiaroscuro.yaml \
  --rank 16 --lr 2e-4 --steps 700 --seed 7 --te_last_n 4 \
  --hold_mode non_concept \
  --device cuda:0 --encoder_device cuda:1 \
  --sample_scales=-1,0,0.5,1 \
  --save_dir models/chiaro-ltx25-uni
```

Single-GPU fallback: `--encoder_device cuda:0` and park DiT with
`--device cpu` during embed-only train; move DiT to GPU for sample.
`--encoder_device cpu` still works (slow TE).

| field | value |
|---|---|
| hub id | `Lightricks/LTX-2.5-Diffusers` |
| recipe | `ltx25_uni_embed` (default) |
| DiT | distilled `transformer/` **frozen** |
| text encoder | LTX Gemma 4 12B (`gemma4_unified`) last-N=4 LoRA |
| connectors | video connectors LoRA; audio skipped |
| rank / alpha | 16 / 16 |
| lr / steps / seed | 2e-4 / 700 / 7 |
| LoRA-up init | `N(0, 0.02)` |
| hold | `non_concept`, PRE-connector |
| sample scales | **−1, 0, 0.5, 1** on neu |
| sample | **49** frames, **544×960**, conv VAE |
| distilled sigmas | `1.0, 0.99375, 0.9875, 0.98125, 0.975, 0.909375, 0.725, 0.421875` |
| CFG / STG / modality | 1.0 / **0** / **1.0** |
| prompt enhancer | OFF |
| GPU | dual RTX A6000: TE+connectors `cuda:1`, DiT `cuda:0` |

Sample a saved adapter without training:

```bash
python conceptmod/textsliders/train_lora_ltx25.py \
  --name smile-ltx25-uni \
  --prompts_file conceptmod/textsliders/data/prompts-ltx25-smile.yaml \
  --steps 0 --load_ltx_lora models/smile-ltx25-uni \
  --sample_scales=-1,0,0.5,1 \
  --sample_num_frames 49 --sample_height 544 --sample_width 960 \
  --device cuda:0 --encoder_device cuda:1 \
  --save_dir models/smile-ltx25-uni
```

## Validated results (2026-09-02, dual A6000)

**Smile v1g.** Loss 2.63 → 0.11, post-connector cos → 0.996. Scales
**−1 / 0** closed mouth, **0.5 / 1** teeth. Mid-scale identity is
**WEAK**.

**Chiaro v1.** Same recipe. Loss 2.99 → 0.096, cos → 0.998. Lighting
soft → Rembrandt **YES**. Composition drift **WEAK**.

These WEAK notes are observed, not hunches: do not treat mid-scale
identity or chiaro composition as solved.

## Dual-GPU notes

Embed-match train does not need the DiT on GPU. Sample does.

* TE + connectors → `--encoder_device cuda:1`
* DiT + conv VAE → `--device cuda:0` (sample)
* Embed-only train may park DiT on CPU (`--device cpu`)
* Never blanket `pipe.to(cuda:0)` — Gemma 4 12B + 22B DiT will OOM
  one A6000

## CPU / CI

```bash
PYTHONPATH=. python conceptmod/textsliders/train_lora_ltx25.py --dummy --steps 4 --name ltx25-dummy
PYTHONPATH=. python conceptmod/textsliders/diag_ltx25_uni.py --dummy --save_dir /tmp/ltx25-diag
PYTHONPATH=. pytest tests/test_ltx25_slider.py -q
python scripts/smoke_ltx25_slider.py
```

`--dummy` never downloads Hub weights. Tests use CPU mocks and a tiny
fake pack only.
