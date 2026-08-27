# H3 image slider (opt-in UNI on AR encode)

Opt-in trainer. Default Music 3 is unchanged. No GPU train and no Hub
weights in CI. Anima / Krea / ZiT are **not** in this PR.

## Resolved identity

**H3 is HunyuanImage-3.0: `tencent/HunyuanImage-3.0`.**

It is an ~80B native multimodal MoE (`AutoModelForCausalLM` /
`hunyuan_image_3_moe`, ~13B active). **Autoregressive.** Not a
flow-matching DiT. Not in [mikkel/conceptmod](https://github.com/mikkel/conceptmod)
today (sana, zimage, anima, krea, qwen, cpu, klein — those are
velocity-space DiTs).

| candidate | what it is | verdict |
|---|---|---|
| **`tencent/HunyuanImage-3.0`** | Official H3. AR/MoE, transformers, custom code. | **Resolved id.** |
| `tencent/HunyuanImage-3.0-Instruct-Distil` | Distil Instruct, 8-step AR sampling. Still MoE / custom_code, not diffusers flow-matching. | Documented cheaper live option. Default stays the base checkpoint. |
| `MiniMaxAI/MiniMax-H3` | MiniMax video Omni-DiT. Mikkel Slack talks about it, but it is not HunyuanImage-3.0. | Not this H3. |
| `hunyuanvideo-community/HunyuanImage-2.1-Diffusers` | Real flow-matching DiT, public `HunyuanImagePipeline`. | HunyuanImage **2.1**, not H3. |
| `Tencent-Hunyuan/HunyuanDiT-v1.2-Diffusers` | Older HunyuanDiT. | Not H3. |

There is **no** public flow-matching diffusers checkpoint for
HunyuanImage-3.0. This trainer does **not** fake a conceptmod
`predict_v` / Euler loop. UNI runs on `encode_text` / last hidden,
which is what an AR/MoE checkpoint actually has. `predict_v` raises
`ArchitectureMismatch`. LoRA-only (second 80B copy will not fit).

## UNI analog (not Music 3 lyric-hold)

Yaml slider: **positive / neutral**, unused attributes pinned.

| term | teacher | in the loss? |
|---|---|---|
| student **+1** last hidden | `encode(+ concept)` last hidden | yes |
| student **scale 0** last hidden | `encode(neutral)` last hidden | yes |
| student **−1** | encode(uncond) (canary) | **no** — logged only |
| unused attribute tokens | those positions of `encode(neu)` | yes |
| concept words (in +, not in neu) | free | **not held** |

## Live train card

GPU box with the Hub checkpoint. Do not run this in CI.

```bash
CUDA_VISIBLE_DEVICES=0 python conceptmod/textsliders/train_lora_h3.py \
  --name age-h3-uni \
  --backend h3 \
  --model_id tencent/HunyuanImage-3.0 \
  --prompts_file conceptmod/textsliders/data/prompts-h3.yaml \
  --attributes "male, female" \
  --rank 8 --alpha 8 --lr 1e-4 --steps 500 --seed 7 \
  --save_dir models/age-h3-uni
```

Optional cheaper AR checkpoint (still not a DiT):

`--model_id tencent/HunyuanImage-3.0-Instruct-Distil`

## CPU / CI

```bash
PYTHONPATH=. python conceptmod/textsliders/train_lora_h3.py --dummy --steps 4 --name h3-dummy
PYTHONPATH=. pytest tests/test_h3_slider.py -q
```

`--dummy` never downloads `tencent/HunyuanImage-3.0`.
