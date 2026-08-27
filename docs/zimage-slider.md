# Z-Image Turbo (ZiT) image slider

Opt-in image-slider trainer on [Tongyi-MAI/Z-Image-Turbo](https://huggingface.co/Tongyi-MAI/Z-Image-Turbo)
(6B flow-matching S3-DiT, Qwen text encoder). **Does not change the
default Music 3 trainer** (`train_lora_music3.py` / `--lm_target v9`).
Not Anima, Krea, or H3 — those backends stay out of this card.

CPU tests use `--dummy` (tiny `Attention` + whitespace tokenizer). No
GPU train and no Hub weights in CI.

## UNI analog (not Music 3 lyric-hold)

ZiT has no lyric span. The image analog of UNI is:

| scale | teacher |
|---|---|
| **+1** | `v(z, t, + concept prompt)` |
| **0** | `v(z, t, neu)` |
| **−1** | unscored canary only |

Unused yaml `attributes` are prefixed onto every caption so gender (or
whatever leftover) is pinned both ways. Unused prompt tokens hold to
`encode(neu)`. Declared `concept_words` (`old, elderly, aged`) are
**not** held — those are the slider.

Fail closed if the + prompt does not contain any concept-word tokens.

Infer plus+neu with the **neutral** caption + LoRA, not the + caption.

## Velocity-space CFG geometry

From conceptmod `backends/zimage.py`:

```
v_cfg = v + g * (v − v_u)
      = v(z, t, c) + g * (v(z, t, c) − v(z, t, ''))
```

The increment is `v(z, t, c) − v(z, t, '')`. ZiT's live sample
guidance is **0**, so CFG is off and the UNI teacher is raw velocity.
`g > 0` still maps: both poles go through that `_cfg` before the MSE.

## Live train card

```bash
CUDA_VISIBLE_DEVICES=N python conceptmod/textsliders/train_lora_zimage.py \
  --name age-zit \
  --prompts_file conceptmod/textsliders/data/prompts-zimage.yaml \
  --model_id Tongyi-MAI/Z-Image-Turbo \
  --rank 16 --alpha 16 --resolution 768 \
  --sample_steps 8 --sample_guidance 0.0 \
  --steps 500 --lr 1e-4 --seed 7 --device 0 \
  --save_dir models/zimage-slider
```

Defaults on a bare parse are that card: rank / alpha 16, 768px, 8
steps, CFG 0, `Tongyi-MAI/Z-Image-Turbo`. LoRA wraps `Attention`
(`to_q`, `to_k`, `to_v`, `to_out.0`), same targets as conceptmod's
zimage backend.

Dummy (CI / no GPU):

```bash
PYTHONPATH=. python conceptmod/textsliders/train_lora_zimage.py \
  --dummy --steps 2 --save_dir /tmp/zimage-dummy
```

## Related

- [docs/lm-plus-neu-exam.md](lm-plus-neu-exam.md) — Music 3 last-token UNI (not this trainer).
- [docs/lm-lyric-hold.md](lm-lyric-hold.md) — Music 3 lyric-token hold. ZiT does **not** use it.
- Music 3 live defaults stay `--loss nmse --target_mode axis` (TF) and `--lm_target v9 --pole_mode hidden` (LM).
