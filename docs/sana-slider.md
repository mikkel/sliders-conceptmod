# Sana 0.6B image slider

Opt-in cheap test backend on
[Efficient-Large-Model/Sana_600M_512px_diffusers](https://huggingface.co/Efficient-Large-Model/Sana_600M_512px_diffusers)
(0.6B flow-matching linear DiT, Gemma-2, DC-AE). **Does not change the
default Music 3 trainer** (`train_lora_music3.py` / `--lm_target v9` /
`train_lm_slider_music3.py --pole_mode hidden`). Not ZiT, Krea, Anima,
or MiniMax-H3 — those backends stay out of this card.

This is the first GPU look we will rent (Modal / RunPod). CPU tests use
`--dummy` (tiny `attn1` / `attn2` + whitespace tokenizer). No GPU train
and no Hub weights in CI.

## UNI analog (not Music 3 lyric-hold)

Sana has no lyric span. The image analog of UNI is:

| scale | teacher |
|---|---|
| **+1** | `v('') + 4.5 · (v(z, t, + concept) − v(''))` |
| **0** | `v(z, t, neu)` |
| **−1** | unscored canary only |

Unused yaml `attributes` are prefixed onto every caption so gender (or
whatever leftover) is pinned both ways. Unused prompt tokens / pins
hold to `encode(neu)`. Declared `concept_words` (`happy, smiling, joyful`)
are **not** held — those are the slider. Happy is the example concept
because age/old-face renders wander into uncanny artifacts; the UNI
analog is unchanged.

Fail closed if the + prompt does not contain any concept-word tokens.

Infer plus+neu with the **neutral** caption + trained xattn / LoRA, not
the + caption.

## Velocity-space CFG geometry

From conceptmod `backends/sana.py` (CFG **4.5**, so this is live):

```
direction(c) = v(z, t, c) − v(z, t, '')
v_cfg        = v('') + g · (v(c) − v(''))    # g != 1  (Sana sample default 4.5)
             = v(c)                          # g == 1
```

This is **not** the Z-Image / Krea compose `v(c) + g · (v(c) − v(''))`.
The increment is the same conceptmod geometry; Sana adds it onto the
uncond velocity.

## Live train card

First GPU look (Modal / RunPod). Do **not** run this in CI.

| field | value |
|---|---|
| hub id | `Efficient-Large-Model/Sana_600M_512px_diffusers` |
| arch | 0.6B flow-matching DiT |
| train | **xattn** (conceptmod 0.6B default; `--lora RANK` optional) |
| resolution | **512** |
| sample steps | **20** |
| CFG | **4.5** |
| control | `a bowl of fruit on a table` |
| lr | `2e-5` |
| steps | 500 |

```bash
CUDA_VISIBLE_DEVICES=0 python conceptmod/textsliders/train_lora_sana.py \
  --name happy-sana \
  --prompts_file conceptmod/textsliders/data/prompts-sana.yaml \
  --model_id Efficient-Large-Model/Sana_600M_512px_diffusers \
  --train_method xattn \
  --resolution 512 \
  --sample_steps 20 --sample_guidance 4.5 \
  --control_prompt "a bowl of fruit on a table" \
  --steps 500 --lr 2e-5 --seed 7 --device 0 \
  --save_dir models/sana-slider
```

Optional LoRA (same UNI / CFG card):

```bash
CUDA_VISIBLE_DEVICES=0 python conceptmod/textsliders/train_lora_sana.py \
  --name happy-sana-lora \
  --lora 8 --train_method xattn \
  --resolution 512 --sample_steps 20 --sample_guidance 4.5 \
  --control_prompt "a bowl of fruit on a table"
```

Dummy (CI / no GPU):

```bash
PYTHONPATH=. python conceptmod/textsliders/train_lora_sana.py \
  --dummy --steps 2 --save_dir /tmp/sana-dummy
PYTHONPATH=. pytest tests/test_sana_slider.py -q
python scripts/smoke_sana_slider.py
```

## Yaml

`conceptmod/textsliders/data/prompts-sana.yaml`: positive / neutral,
unused `attributes` pinned (`male`, `female`). `negative` is stored
for the canary and is not a teacher. `control_prompt` is the fruit
bowl — verify only, never a teacher.

## Related

- [conceptmod backends/sana.py](https://github.com/mikkel/conceptmod/blob/main/conceptmod/backends/sana.py)
- conceptmod README SANA proofs `outputs/01`–`13` (fruit bowl is CONTROL)
- [docs/lm-plus-neu-exam.md](lm-plus-neu-exam.md) — Music 3 last-token UNI (not this trainer)
- [docs/lm-lyric-hold.md](lm-lyric-hold.md) — Music 3 lyric-token hold. Sana does **not** use it.
- Music 3 live defaults stay `--loss nmse --target_mode axis` (TF) and `--lm_target v9 --pole_mode hidden` (LM).
