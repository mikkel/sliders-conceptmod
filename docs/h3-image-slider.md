# H3 image slider (opt-in UNI)

Opt-in trainer. Default Music 3 (`train_lora_music3.py` / `train_lm_slider_music3.py`) is unchanged. No GPU train and no Hub weights in CI.

## What "H3" is

Mikkel's H3 is **MiniMax-H3**, not HunyuanImage-3.0.

Checked before locking a hub id:

| candidate | what it is | why not / why yes |
|---|---|---|
| **`MiniMaxAI/MiniMax-H3`** | Official MiniMax H3. Public diffusers checkpoint (`MiniMaxH3ModularPipeline` / `MiniMaxH3Pipeline`, `MiniMaxH3Transformer3DModel` / `MiniMaxH3DiTModel`, `MiniMaxH3Scheduler` with flow-match `shift=12.0`). 33B Omni-DiT, FL2VA t2va / first-frame. | **Pick.** Mikkel Slack: "minimax h3 weight released", "testing h3", Civitai buzz for 15s video, [`joeygambino/MiniMax-H3-x-Z-Image-native`](https://huggingface.co/joeygambino/MiniMax-H3-x-Z-Image-native). Not in [mikkel/conceptmod](https://github.com/mikkel/conceptmod) today (that repo has sana, zimage, anima, krea, qwen, cpu, klein). |
| `tencent/HunyuanImage-3.0` | Native multimodal MoE (`AutoModelForCausalLM` / `hunyuan_image_3_moe`). Autoregressive image generator. | Not a flow-matching DiT. No public diffusers checkpoint. |
| `hunyuanvideo-community/HunyuanImage-2.1-Diffusers` | Flow-matching DiT, public `HunyuanImagePipeline`. | Real checkpoint, wrong name: HunyuanImage **2.1**. Mikkel never calls it H3. |
| `Tencent-Hunyuan/HunyuanDiT-v1.2-Diffusers` | Older HunyuanDiT. | Not H3. |

Resolved model id: **`MiniMaxAI/MiniMax-H3`**. Image sliders use the FL2VA family (t2va / first-frame) and a T=1 visual latent. LoRA-only: a second 33B copy will not fit; frozen = adapter off.

Anima / Krea / ZiT are **not** in this PR.

## UNI analog (not Music 3 lyric-hold)

Yaml slider: **positive / neutral**, unused attributes pinned on both poles.

| term | teacher | in the loss? |
|---|---|---|
| student **+1** | frozen velocity of the **+ concept** prompt | yes |
| student **scale 0** | frozen velocity of the **neutral** prompt | yes |
| student **−1** | frozen uncond / empty (canary) | **no** — logged only |
| unused attribute tokens | `encode(neu)` at those positions | yes |
| concept words (in + , not in neu) | free | **not held** |

This is last-token / velocity UNI plus an unused-token hold. It is not `faithful_plus_neu_lyric` and it does not hold Music 3 lyric spans.

## Live train card

GPU box with the Hub checkpoint. Do not run this in CI.

```bash
CUDA_VISIBLE_DEVICES=0 python conceptmod/textsliders/train_lora_h3.py \
  --name age-h3-uni \
  --backend h3 \
  --model_id MiniMaxAI/MiniMax-H3 \
  --prompts_file conceptmod/textsliders/data/prompts-h3.yaml \
  --attributes "male, female" \
  --rank 8 --alpha 8 --lr 1e-4 --steps 500 --seed 7 \
  --resolution 768 \
  --save_dir models/age-h3-uni
```

Sidecar: `models/age-h3-uni/age-h3-uni_last.json` (`recipe: h3_uni`, `minus_teacher: false`, `lora_only: true`).

## CPU / CI

```bash
PYTHONPATH=. python conceptmod/textsliders/train_lora_h3.py --dummy --steps 4 --name h3-dummy
PYTHONPATH=. pytest tests/test_h3_slider.py tests/test_lm_trainer_v9.py -q
```

`--dummy` never downloads `MiniMaxAI/MiniMax-H3`.
