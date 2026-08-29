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

Yaml slider: **positive / neutral**. Default hold is **non-concept**
(Music 3 lyric-hold analog): every plus token that is **not** a concept
word (token ids in plus but not in neu) is pinned to the matching
`encode(neu)` row. Yaml unused attributes (male / female) are a subset
of that hold. Concept lighting words stay free.

`--hold_mode attributes` is the old v1/v2 path and **leaks identity**:
only yaml attribute tokens were held, so shared subject tokens
(clothes / chair / vase pattern) stayed as `encode(plus)` and Omni LoRA
rewrote them.

| term | student | teacher | in the loss? |
|---|---|---|---|
| +1 | LoRA on, plus-concept packed sequence | frozen Omni-Transformer velocity on that plus pack | yes |
| scale 0 | adapter off, neu packed sequence | frozen velocity on the neu pack / `encode(neu)` | yes |
| −1 | LoRA scale −1 (canary) | uncond pack | **no** — logged only |
| non-concept tokens (default) | matching rows of `encode(neu)` | yes |
| unused attribute tokens | subset of non-concept; `--hold_mode attributes` holds only these | yes |
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
  --lora_up_init_std 0.02 \
  --short_side 768 --guidance 0 --device cuda:0 \
  --save_dir models/age-minimax-h3-uni
```

| field | value |
|---|---|
| hub id | `MiniMaxAI/MiniMax-H3` |
| variant / workflow | FL2VA / t2va |
| task index | `FL2VA/model_index.json` |
| rank / alpha | 8 / 8 |
| LoRA-up init | `N(0, 0.02)` — not zeros (UNI identity) |
| torch (B300) | **2.13.0+cu130** (`2.6+cu124` has no sm_103; not Music 3) |
| short side | 768 |
| CFG | distilled; guidance **0** |
| device | `cuda:0` (live) |
| LoRA host | `MiniMaxH3Attention` `to_q/to_k/to_v/to_out.0` |
| freeze | encoder (Qwen3-VL-32B) + visual VAE + audio VAE + processor + tokenizer |
| not in weights | H3-Context-IR, H3-Regenerate-2K |

Live load needs a current `diffusers` with MiniMax-H3 ModularPipeline. Do **not**
`pip install -r requirements.txt` on the Music 3 env.

**B300 torch:** Blackwell `sm_103` needs **`torch 2.13.0+cu130`** (or newer
cu130). `2.6+cu124` has no `sm_103` and will not run on B300. Install that
wheel **only on the H3/B300 box**. Do **not** force it into the Music 3
(`minimax-music3`) env.

**LoRA-up init:** default is `N(0, 0.02)` (`--lora_up_init_std 0.02`). Classic
zero-init makes UNI identity (scale-1 == scale-0) so loss stays `0.0000` and
nothing trains. Pass `--lora_up_init_std 0` only for that ablation.

## Live train card (chiaroscuro-minimax-h3-uni)

First **live** H3 train+sample (`slider-h3-chiaro-v1` on **B300**). Prefer
**1× B200 / B300** (bf16 ModularPipeline `pipe.to(cuda:0)` is ~135 GB:
transformer 61.7 + Qwen3-VL-32B 62.1 + VAEs).

**First live finding:** rank8 × 800 with classic **zero-init LoRA-up** died —
first→last loss `0.0000` (UNI identity). After tiny LoRA-up `N(0, 0.02)`,
**rank16 × 1500** actually trained: loss `0.016→0.130`, peak ~`0.21`. Use
`--lora_up_init_std 0.02` (the default). Do not start a live run at rank8
with zeros.

**v1/v2 identity leak:** live chiaro v1/v2 moved lighting on person+room
but morphed clothes / props and the vase pattern; vase lighting still
failed. `--hold_mode attributes` (the old default) only pinned yaml
attribute tokens (male/female). Shared subject tokens in both plus and
neu were **not** held, so Omni LoRA rewrote identity. The new default
`--hold_mode non_concept` holds every non-concept token to
`encode(neu)`. Live **chiaro-v3** (hold_weight 2.0, looser locks)
gated **FAIL 1/3**: person lighting+identity passed; room was only
dimmer and chairs/props drifted; vase lighting passed but painted
figures rewrote. Live **chiaro-v4** (hold_weight 5.0, tight locks)
also **FAIL 1/3**: person still YES/YES; room lighting NO but
identity YES (props held); vase lighting+identity NO (lighting
regressed, scholars rewrite). Recommended next live card is
**chiaro-v5** below (rank 16, 2500 steps, `--hold_weight 3.0`,
same v4 locks, stronger plus lighting teacher).

B300 needs **`torch 2.13.0+cu130`** (`2.6+cu124` has no `sm_103`). That
wheel stays on the H3 box; do **not** install it in Music 3.

Same other flags as the age card (alpha=rank, lr 1e-4, short_side 768,
guidance 0, FL2VA / t2va). Prompts are a **chiaroscuro / dramatic
lighting** set: same concrete subject on neu and plus (person in a blue
denim shirt over a white tee / dining room with two windsor chairs, a
round table, and a glass bowl of bananas apples and oranges /
blue-and-white ceramic jar with painted bearded scholars exchanging a
gift). Neu is flat even lighting / soft fill / low contrast
(photographic, not cartoon). Plus is chiaroscuro, Rembrandt key
light from the left, hard single-source side light, deep black
shadows, high contrast, no fill. Minus is a
washed-out flat / featureless lighting canary only.

After train the script writes short t2va mp4s under `save_dir/samples/` at
LoRA scales **0** and **+1** (add `0.5` with `--sample_scales 0,0.5,1`) for
each unique yaml **target** (neu subject, no plus lighting). Plus-oracle
clips (plus caption at scale 0) are not emitted. Guidance stays **0**.
Sample-only reload:

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
  --rank 16 --alpha 16 --lr 1e-4 --steps 1500 --seed 7 \
  --lora_up_init_std 0.02 \
  --short_side 768 --guidance 0 --device cuda:0 \
  --sample_scales 0,1 --sample_duration 5 --sample_fps 24 \
  --save_dir models/chiaroscuro-minimax-h3-uni
```

A short rank8 / 500 smoke is fine **only** with `--lora_up_init_std 0.02`
(not zeros). The first live B300 run that worked was **rank16 × 1500**.

| field | value |
|---|---|
| hub id | `MiniMaxAI/MiniMax-H3` |
| variant / workflow | FL2VA / t2va |
| task index | `FL2VA/model_index.json` |
| rank / alpha | **16 / 16** (first working live; rank8 zero-init died) |
| steps | **1500** (first working live) |
| LoRA-up init | `N(0, 0.02)` — not zeros |
| torch (B300) | **2.13.0+cu130** (`2.6+cu124` has no sm_103; not Music 3) |
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
  --rank 16 --alpha 16 --lr 1e-4 --steps 1500 --seed 7 \
  --lora_up_init_std 0.02 \
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
solid — watch the clips, do not gate on last-50 c+. First live that moved
loss was already rank16 × 1500 after rank8 zero-init death; if 1500 is
still weak, keep rank 16 and add steps rather than going back to zeros.
Recommended escalate:

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
  --lora_up_init_std 0.02 \
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

## Live train card (chiaroscuro-minimax-h3-uni-v3)

Previous live run (`slider-h3-chiaro-v3` on **B300**,
`torch 2.13.0+cu130`). Same stack as v1/v2 (rank 16, LoRA-up `N(0, 0.02)`,
FL2VA / t2va, guidance 0) but:

* `--hold_mode non_concept` (now the trainer default) — hold every token
  that is not a concept word to `encode(neu)`. Shared subject tokens no
  longer leak. `--hold_mode attributes` is the v1/v2 leak.
* `--hold_weight 2.0`
* **2000** steps
* tighter same-subject yaml locks: blue denim shirt / wooden chair and
  table with a fruit bowl / blue-and-white ceramic vase with painted
  figures. Lighting delta only in plus vs neu.

Gate was lighting moves **and** identity holds on **≥2/3** rows, including
the vase if possible. Live v3 gated **FAIL 1/3** (person lighting+identity
yes; room lighting+identity no; vase lighting yes / identity no). Watch
sampled videos, not last-50 c+. Next cards are **chiaro-v4** then
**chiaro-v5**.

```bash
CUDA_VISIBLE_DEVICES=0 python conceptmod/textsliders/train_lora_minimax_h3.py \
  --name chiaroscuro-minimax-h3-uni-v3 \
  --model_id MiniMaxAI/MiniMax-H3 \
  --variant FL2VA \
  --workflow t2va \
  --prompts_file conceptmod/textsliders/data/prompts-minimax-h3-chiaroscuro.yaml \
  --config_file conceptmod/textsliders/data/config-minimax-h3-chiaroscuro.yaml \
  --attributes "male, female" \
  --rank 16 --alpha 16 --lr 1e-4 --steps 2000 --seed 7 \
  --hold_mode non_concept --hold_weight 2.0 \
  --lora_up_init_std 0.02 \
  --short_side 768 --guidance 0 --device cuda:0 \
  --sample_scales 0,1 --sample_duration 5 --sample_fps 24 \
  --save_dir models/chiaroscuro-minimax-h3-uni-v3
```

| field | value |
|---|---|
| name | `chiaroscuro-minimax-h3-uni-v3` |
| hub id | `MiniMaxAI/MiniMax-H3` |
| variant / workflow | FL2VA / t2va |
| rank / alpha | **16 / 16** |
| steps | **2000** |
| hold | `--hold_mode non_concept` (default; v1/v2 used attributes and leaked) |
| hold weight | **2.0** |
| LoRA-up init | `N(0, 0.02)` — not zeros |
| torch (B300) | **2.13.0+cu130** (`2.6+cu124` has no sm_103; not Music 3) |
| short side | 768 (sample canvas 1344×768 16:9) |
| sample duration / fps | **5 s** / 24 |
| CFG | distilled; guidance **0** |
| device | `cuda:0` on **B200 / B300**; 2×H100 add `--encoder_device cuda:1` |
| gate | lighting moves **and** identity holds on ≥2/3 rows (include vase) |
| not in weights | H3-Context-IR, H3-Regenerate-2K |

Do **not** run this in CI. Do **not** download MiniMax-H3 in CI.

## Live train card (chiaroscuro-minimax-h3-uni-v4)

Previous live run (`slider-h3-chiaro-v4` on **B300**,
`torch 2.13.0+cu130`). Same stack as v3 (rank 16, LoRA-up `N(0, 0.02)`,
`--hold_mode non_concept`, FL2VA / t2va, guidance 0) but:

* `--hold_weight 5.0` (v3 used 2.0; person identity held, room/vase did not)
* **2000** steps (2500 if 2000 is still weak)
* tighter same-subject yaml locks on **both** poles (lighting words only
  differ neu vs plus):
  * person sitting in a chair wearing a blue denim shirt over a white tee
    (denim lock already passed live v3)
  * interior dining room with two wooden windsor chairs, a round wooden
    table, and a clear glass bowl of bananas apples and oranges
  * blue and white ceramic jar with painted bearded scholars in robes
    exchanging a gift on a wooden table

Live v3 gate **FAIL 1/3**: person lighting **YES** + identity **YES**
(denim hold fixed; Rembrandt works). Room lighting **NO** (just dimmer,
not chiaroscuro) + identity **NO** (chair spindles morph; props drift).
Vase lighting **YES** (left light / right shadow) + identity **NO**
(painted figures rewrite poses/details).

Live v4 gated **FAIL 1/3**, not better: person lighting **YES** +
identity **YES**. Room lighting **NO** (still not chiaroscuro) +
identity **YES** (windsor chairs / fruit bowl held). Vase lighting
**NO** (regressed vs v3) + identity **NO** (scholars rewrite). Tight
locks fixed room props; hold_weight 5.0 crushed the lighting teacher.
Next card is **chiaro-v5**.

Gate: lighting moves **and** identity holds on **≥2/3** rows. Person
already passes; need room and/or vase identity. Watch sampled videos,
not last-50 c+.

```bash
CUDA_VISIBLE_DEVICES=0 python conceptmod/textsliders/train_lora_minimax_h3.py \
  --name chiaroscuro-minimax-h3-uni-v4 \
  --model_id MiniMaxAI/MiniMax-H3 \
  --variant FL2VA \
  --workflow t2va \
  --prompts_file conceptmod/textsliders/data/prompts-minimax-h3-chiaroscuro.yaml \
  --config_file conceptmod/textsliders/data/config-minimax-h3-chiaroscuro.yaml \
  --attributes "male, female" \
  --rank 16 --alpha 16 --lr 1e-4 --steps 2000 --seed 7 \
  --hold_mode non_concept --hold_weight 5.0 \
  --lora_up_init_std 0.02 \
  --short_side 768 --guidance 0 --device cuda:0 \
  --sample_scales 0,1 --sample_duration 5 --sample_fps 24 \
  --save_dir models/chiaroscuro-minimax-h3-uni-v4
```

| field | value |
|---|---|
| name | `chiaroscuro-minimax-h3-uni-v4` |
| hub id | `MiniMaxAI/MiniMax-H3` |
| variant / workflow | FL2VA / t2va |
| rank / alpha | **16 / 16** |
| steps | **2000** (2500 if still weak) |
| hold | `--hold_mode non_concept` (default; v1/v2 used attributes and leaked) |
| hold weight | **5.0** |
| LoRA-up init | `N(0, 0.02)` — not zeros |
| torch (B300) | **2.13.0+cu130** (`2.6+cu124` has no sm_103; not Music 3) |
| short side | 768 (sample canvas 1344×768 16:9) |
| sample duration / fps | **5 s** / 24 |
| CFG | distilled; guidance **0** |
| device | `cuda:0` on **B200 / B300**; 2×H100 add `--encoder_device cuda:1` |
| gate | lighting+identity on ≥2/3 rows (person already passes; need room and/or vase identity) |
| not in weights | H3-Context-IR, H3-Regenerate-2K |

Do **not** run this in CI. Do **not** download MiniMax-H3 in CI.

## Live train card (chiaroscuro-minimax-h3-uni-v5)

Recommended next live run (`slider-h3-chiaro-v5` on **B300**,
`torch 2.13.0+cu130`). Same stack as v4 (rank 16, LoRA-up `N(0, 0.02)`,
`--hold_mode non_concept`, FL2VA / t2va, guidance 0, same v4 subject
locks) but:

* `--hold_weight 3.0` (middle ground: v3 used 2.0, v4 used 5.0)
* **2500** steps (2000 if you need a shorter run)
* stronger plus lighting teacher (still free concept tokens; neu
  unchanged):
  * chiaroscuro, Rembrandt key light from the left, hard
    single-source side light, deep black shadows, high contrast,
    no fill
  * neu stays flat even lighting / soft fill / low contrast
  * uncond/neg stays washed-out flat featureless lighting

Live v4 gate **FAIL 1/3**, not better than v3: person lighting **YES**
+ identity **YES**. Room lighting **NO** + identity **YES** (windsor
chairs / fruit bowl held — locks worked). Vase lighting **NO**
(regressed) + identity **NO** (scholars rewrite). Hold 5.0 kept room
props but crushed the lighting move. v5 eases hold and strengthens
the plus teacher.

Gate: lighting moves **and** identity holds on **≥2/3** rows. Person
already passes; need room lighting without dropping the held props,
and/or vase lighting+identity. Watch sampled videos, not last-50 c+.

```bash
CUDA_VISIBLE_DEVICES=0 python conceptmod/textsliders/train_lora_minimax_h3.py \
  --name chiaroscuro-minimax-h3-uni-v5 \
  --model_id MiniMaxAI/MiniMax-H3 \
  --variant FL2VA \
  --workflow t2va \
  --prompts_file conceptmod/textsliders/data/prompts-minimax-h3-chiaroscuro.yaml \
  --config_file conceptmod/textsliders/data/config-minimax-h3-chiaroscuro.yaml \
  --attributes "male, female" \
  --rank 16 --alpha 16 --lr 1e-4 --steps 2500 --seed 7 \
  --hold_mode non_concept --hold_weight 3.0 \
  --lora_up_init_std 0.02 \
  --short_side 768 --guidance 0 --device cuda:0 \
  --sample_scales 0,1 --sample_duration 5 --sample_fps 24 \
  --save_dir models/chiaroscuro-minimax-h3-uni-v5
```

| field | value |
|---|---|
| name | `chiaroscuro-minimax-h3-uni-v5` |
| hub id | `MiniMaxAI/MiniMax-H3` |
| variant / workflow | FL2VA / t2va |
| rank / alpha | **16 / 16** |
| steps | **2500** (2000 if you need a shorter run) |
| hold | `--hold_mode non_concept` (default; v1/v2 used attributes and leaked) |
| hold weight | **3.0** (v3 2.0 leaked room/vase; v4 5.0 crushed lighting) |
| LoRA-up init | `N(0, 0.02)` — not zeros |
| torch (B300) | **2.13.0+cu130** (`2.6+cu124` has no sm_103; not Music 3) |
| short side | 768 (sample canvas 1344×768 16:9) |
| sample duration / fps | **5 s** / 24 |
| CFG | distilled; guidance **0** |
| device | `cuda:0` on **B200 / B300**; 2×H100 add `--encoder_device cuda:1` |
| gate | lighting+identity on ≥2/3 rows (keep v4 room-prop hold; restore lighting) |
| not in weights | H3-Context-IR, H3-Regenerate-2K |

Do **not** run this in CI. Do **not** download MiniMax-H3 in CI.

## UNI diagnostic (lighting gap vs identity leak)

No new train loss. Numbers only — run this on an existing LoRA (or dummy)
**before** changing the recipe. Live chiaro v3–v5: person is always
lighting+identity YES; room trades with `hold_weight` (5 → identity YES
lighting NO; 3 → lighting YES identity NO). Embed `non_concept` hold
alone is not enough; this script measures the gap.

Same stack as train (FL2VA / t2va, `--device`, optional
`--encoder_device`). `--dummy` never downloads H3.

```bash
# CI / CPU mock
PYTHONPATH=. python conceptmod/textsliders/diag_minimax_h3_uni.py --dummy \
  --save_dir /tmp/h3-diag

# Live: existing chiaro LoRA (no retrain)
CUDA_VISIBLE_DEVICES=0 python conceptmod/textsliders/diag_minimax_h3_uni.py \
  --prompts_file conceptmod/textsliders/data/prompts-minimax-h3-chiaroscuro.yaml \
  --load_h3_lora models/chiaroscuro-minimax-h3-uni-v5 \
  --device cuda:0 --save_dir models/chiaroscuro-minimax-h3-uni-v5/diag

# same via the trainer
CUDA_VISIBLE_DEVICES=0 python conceptmod/textsliders/train_lora_minimax_h3.py \
  --diag --steps 0 \
  --prompts_file conceptmod/textsliders/data/prompts-minimax-h3-chiaroscuro.yaml \
  --load_h3_lora models/chiaroscuro-minimax-h3-uni-v5 \
  --device cuda:0 --save_dir models/chiaroscuro-minimax-h3-uni-v5/diag
```

Writes `{name}_diag.json` under `--save_dir` plus a short stdout table.
One row per slider yaml row (attribute pins expand male/female).

### How to read the numbers

| metric | what it is | healthy | sick |
|---|---|---|---|
| **lighting_gap** `cos` / `l2` | frozen teacher `v_plus` vs `v_neu` (packed velocity, same noise) | `l2` clearly >0 (plus/neu teachers differ) | `l2` ~0 — captions or hold collapsed the lighting axis |
| **embed_gap_energy** `concept_frac` | after `apply_unused_hold`, fraction of plus−neu embed energy on concept-token rows | ~1 (gap is lighting words) | `held_frac` >0 hold missed rows; `unheld_nonconcept_frac` >0 is the v1/v2 shared-subject leak |
| **hold** `held_max_abs` / `held_mean_abs` | `|held_plus_row − encode(neu)_row|` on held tokens | ~0 | hold did not pin identity tokens |
| **hold** `concept_mean_abs` | mean `|plus − neu|` on free concept tokens | >0 | lighting words also collapsed |
| **student** `scale0_vs_teacher_neu` | adapter off on neu pack vs frozen neu | `cos` ~1 | scale-0 is not identity |
| **student** `scale1_vs_teacher_plus` | adapter on plus pack vs frozen plus | high `cos` = **lighting match** | low — student did not learn plus lighting |
| **student** `scale1_vs_scale0` | plus@1 vs neu@0 — **identity drift proxy** | high `cos` = structure held | low `cos` = rewrite (lighting and/or identity) |
| **student** `neu_lora_on_vs_off` | LoRA on vs off under the **neu** pack only | high `cos` = adapter leaves neu structure | low — adapter rewrites the identity caption |

Read lighting match and drift **together**:

* lighting YES + `s1_s0` low → student moved toward plus; if the clip also morphs clothes/props, that low cos is **identity leak** (chiaro room at hold 3).
* lighting NO + `s1_s0` high → hold crushed the move (chiaro room at hold 5).
* `hold_*` ~0 but live identity still leaks → embed hold is not enough; the Omni LoRA is rewriting structure in velocity space. That is why this diag exists before a new train loss.

`--load_h3_lora` is optional. Without it, student columns use the random
`N(0, 0.02)` init (dummy CI asserts shapes/keys, not a trained match).

## CPU / CI

```bash
PYTHONPATH=. python conceptmod/textsliders/train_lora_minimax_h3.py --dummy --steps 4 --name minimax-h3-dummy
PYTHONPATH=. python conceptmod/textsliders/diag_minimax_h3_uni.py --dummy --save_dir /tmp/h3-diag
PYTHONPATH=. pytest tests/test_minimax_h3_slider.py -q
```

`--dummy` never downloads Hub weights. Tests use CPU mocks and a tiny fake
packed sequence only.
