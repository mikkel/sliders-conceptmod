# Krea image sliders (opt-in UNI)

Opt-in image trainer. **Does not change the Music 3 default**
(`train_lora_music3.py` / `train_lm_slider_music3.py --lm_target v9` /
`--pole_mode hidden`). Anima / ZiT / H3 are a separate PR and are not
in this trainer. Do **not** port Anima `embed_struct` / `same_crop`
here — smile-first on Raw accepts entanglement.

CPU tests use `--dummy` mocks. No Hub weights, no GPU train in CI.
Live load is offline-safe (`local_files_only` / `HF_HUB_OFFLINE=1`)
unless `--allow_hub` is set (same gate as Anima).

## UNI analog (not Music 3 lyric-hold)

Music 3 lyric-hold pins yaml `lyrics` tokens. Krea does not.

| scale | teacher | not a teacher |
|---|---|---|
| **+1** | + concept prompt velocity `v(z,t,pos)` | minus caption |
| **0** | neutral prompt velocity `v(z,t,neu)` | leftover-gate / pair-odd |

Minus is a **canary** only: `v(neg) − v('')` is logged and never enters
the loss.

Unused prompt tokens (pinned yaml `attributes`, plus the shared
skeleton that also appears in neu) hold to encode(neu). Concept words
— tokens in pos that are not in neu — are **not** held.

Happy / smile and detailed yaml use **bare captions**: attributes
pin unused gender for bookkeeping only and are **not** prefixed
onto target / positive / neutral (Anima smile lesson).

## Velocity-space CFG

conceptmod Krea geometry, when it maps:

```
direction(c) = v(z, t, c) − v(z, t, '')
v_cfg(c)     = v(c) + g · (v(c) − v(''))    # g > 0
             = v(c)                         # g = 0 (Turbo)
```

Raw samples at CFG 4.5, so the +1 teacher is CFG-composed
`v(pos) + 4.5 · (v(pos) − v(''))`. Scale 0 stays raw `v(neu)`.
Turbo trains and samples at CFG 0.

## Live train card (smile-krea / happy-krea)

Official advice: **train LoRAs on Raw, run on Turbo.**

Live backend (`conceptmod/textsliders/krea_live.py`, lazy-imported
from `_load_live_backend`):

- `Krea2Pipeline.from_pretrained("krea/Krea-2-Raw", dtype=bfloat16)`
  (falls back to `torch_dtype=` if the installed Diffusers wants that)
- `--lora_targets dit` (default): LoRA on DiT attn
  `to_q/to_k/to_v/to_out.0` via peft (**rank 16**)
- `--lora_targets te` / `text_encoder`: Qwen3-VL attention
  `q_proj/k_proj/v_proj/o_proj`. `--lora_targets dit+te` is joint
  (Anima lesson: expression can live in the text path)
- frozen ref = adapter disabled (a second 12B copy will not fit)
- Continuous sample scales write LoRA `scaling` (`alpha/r * scale`).
  PEFT `set_adapter_scale` often no-ops, which made live 0.25 / 0.5 /
  1.0 grids byte-identical (only scale 0 / `disable_adapter` differed)
- Park VAE on CPU. Frozen TE: encode on GPU, then park before DiT
  backward (**~48GB**, A6000 / A100). Trained TE stays resident so
  encode+backward stay coherent (prefer 80GB A100 for `dit+te`)
- End-of-train smile-first grid: scales `0 / 0.25 / 0.5 / 1.0` on
  neu / infer captions plus the fruit-bowl control, PNGs under
  `save_dir/samples`. Not a crop-purity gate. Do **not** port Anima
  `same_crop` / `embed_struct`.

`--hold_weight` defaults to **1.0** (age yaml unused-gender hold).
Live smile-krea logs were hold-dominated (`hold≈7.31` of `loss≈7.35`)
because frozen-TE hold is a near-constant. **Smile cards use 0.1.**
Do not silently lower the age-yaml default.

## TE-only embed UNI (`--lm_target embed`)

Live gap diag on Raw (after #74): DiT velocity neu/plus
**cos≈0.9999** — v-space has nothing to teach. Stacked TE embeds
`[1, 512, 12, 2560]` neu/plus **cos≈0.67**, with early layers
~0.91 and mid/late (2–11) ~0.57–0.73. Smile v3 therefore **does
not teach v-space**. It trains Qwen3-VL LoRA only:

| scale | student | teacher |
|---|---|---|
| **+1** | adapted TE `E_θ(neu)` | stopgrad frozen TE `E_frozen(pos)` |
| **0** | adapter off (`disable_adapter` / scale 0) | — |

Loss is layer-weighted **MSE + relative L2**
(`mean_L ||s−t||² / (||t||²+eps)`). `--embed_cosine_weight`
defaults to **0** (opt-in `1−cos` hid the live magnitude gap).
Early layers 0–1 weigh very low; mid 2–5 = 1.0; **layers 6–11
weigh 2.0** (late transplant recovered teeth). Unused-token /
attribute hold stays light (`--hold_weight 0.1`) and compares
**frozen pos vs frozen neu** — do not train hold through the
student adapter on the plus caption.
Sample grid still runs the full pipeline (TE embeds → frozen base
DiT denoise) so smiles are visible. DiT weights stay base. Do
**not** add Anima `same_crop` / `embed_struct`.

**Apply (v4):** when `encoder_lora`, CFG encodes the uncond / `""`
branch with **frozen TE** (`disable_adapter` / scale 0). Cond keeps
the active adapter scale from `_scale_ctx`. Cond + frozen uncond
are encoded **once** per generate and reused (conceptmod). v3
encoded both CFG branches under the same TE scale, so
`v_cond − v_uncond` cancelled the smile Δ at Raw CFG 4.5
(embed_cos≈0.988, pixels only soft closed-lip). Embed sample
guidance defaults to **0**; velocity UNI keeps Raw 4.5.

**Apply (v5, TE attention mask):** live oracle on the v3 TE adapter
proved embeds already matched plus:

- `cos(Eθ(neu)@1, E_frozen(plus)) ≈ 0.9959` (meanpool ≈ 0.9995)
- `generate(plus, TE frozen)` → clear teeth grin (oracle)
- `generate(neu, TE@1, cfg 0 or 4.5)` → closed mouth / soft lip
- CFG-0 did not close the gap → not only uncond cancel

Plus is longer than neu. Embed UNI MSE-matches the full
`[B,512,12,2560]` stack (including positions past neu's valid
tokens), but `encoder_attention_mask` from encoding **neu** only
attends the shorter neu span — DiT never saw the rows where smile
content was written (mask truncated smile slots). Oracle uses
plus's mask with plus embeds →
teeth. Default TE-slider sample now uses an **all-ones** attention
mask over `max_sequence_length` whenever TE LoRA scale > 0. Scale 0
/ frozen TE keep the real tokenizer mask. Optional `mask_prompt`
transplants a frozen-plus mask at apply-audit time.
`--te_dit_mask tokenizer` is the old neu span (A/B).

**Magnitude (v4 retrain):** ones-mask + high cosine was **not**
enough. Live transplant on SHA `7f3eee7`:

- `cos(Eθ(neu)@1, E_frozen(plus)) ≈ 0.9959` but `max_abs ≈ 147`
- Feeding frozen-plus embeds → DiT shows **teeth** (even with
  ones mask)
- Student embeds → no teeth (mask A/B failed)
- lerp 0.5 and late-layer (L6–11) transplant from plus → teeth

Residual **magnitude** gap in late layers; cosine hid it. Forward
path / ones-mask is OK. Gate on **teeth vs oracle**, not
`embed_cos` alone. Retrain with MSE + rel-L2 (card below).

End-of-train **oracle / apply-audit grid** (`save_dir/samples/oracle/`):
for each neu prompt, `oracle_plus_frozen` = generate(plus, scale=0),
`student_neu_scale1` = generate(neu, scale=1, ones-mask), `neu_scale0` =
generate(neu, scale=0). Meta logs `cos(Eθ(neu)@1, E_frozen(pos))`.
`--load_te_lora` resmoke also writes ones-mask vs old tokmask vs
transplanted plus-mask A/B. If oracle has teeth and student does
not despite cos>0.95 → remaining apply bug **or** residual
magnitude gap (`embed_max_abs` / late L2). Gate on teeth vs
oracle, not `embed_cos` alone. If oracle also lacks teeth →
caption teacher is weak in pixels.

`--recipe embed_uni` is an alias for `--lm_target embed`. `--lora_targets`
is forced to `te` on this path (DiT LoRA is not attached).

| | Raw (train here) | Turbo (run) |
|---|---|---|
| weights | `krea/Krea-2-Raw` | local ComfyUI `.safetensors` (name contains `turbo`) or `krea/Krea-2-Turbo` |
| LoRA rank | **16** | same file |
| resolution | **512** | 512 |
| sample steps | **28** | **8** |
| CFG | **4.5** | **0** |

`krea/Krea-2-Raw` is **gated**. Accept the card and pass a Hub token.
`--allow_hub` is required for the first download; afterwards a warm
cache can run with `HF_HUB_OFFLINE=1` and no flag.

Retrain card (**smile-krea-v4**, TE-only embed UNI, MSE + rel-L2,
ones-mask, CFG 0). Gate on teeth vs oracle, not `embed_cos`:

```bash
CUDA_VISIBLE_DEVICES=0 python conceptmod/textsliders/train_lora_krea.py \
  --name smile-krea-v4 \
  --prompts_file conceptmod/textsliders/data/prompts-krea-happy.yaml \
  --model_id krea/Krea-2-Raw --allow_hub \
  --lora_targets te --lm_target embed \
  --embed_cosine_weight 0 --hold_weight 0.1 --sample_guidance 0 \
  --te_dit_mask auto --rank 16 --steps 800 --lr 1e-4 \
  --resolution 512 --sample_steps 28 --seed 7 --device 0 \
  --save_dir models/smile-krea-v4
```

`--embed_rel_l2_weight` defaults to 1.0. `--embed_late_weight`
defaults to 2.0 (late layers 6–11); `--embed_late_layer_start`
defaults to 6. Logs `embed_max_abs`, `embed_late_l2`, and
per-layer `embed_l2_l*`, not only `embed_cos` / `embed_mse`.

Same flags, non-smile concept (**detail-krea-v1**): **detailed**
(fine detail / texture / intricacy) on scene rows so stock Krea
is not already detailed. Early captions used a cartoon-adjacent
`simple flat, plain` neu vs an ornate-photo plus — a style jump,
not a detail delta. Gate: student neu@1 looks more detailed than
neu@0 and approaches frozen-plus oracle (not `embed_cos` alone).

```bash
CUDA_VISIBLE_DEVICES=0 python conceptmod/textsliders/train_lora_krea.py \
  --name detail-krea-v1 \
  --prompts_file conceptmod/textsliders/data/prompts-krea-detailed.yaml \
  --model_id krea/Krea-2-Raw --allow_hub \
  --lora_targets te --lm_target embed \
  --embed_cosine_weight 0 --hold_weight 0.1 --sample_guidance 0 \
  --te_dit_mask auto --rank 16 --steps 800 --lr 1e-4 \
  --resolution 512 --sample_steps 28 --seed 7 --device 0 \
  --save_dir models/detail-krea-v1
```

**detail-krea-v1** (800 / rank 16 / late=2, smile-krea-v4
defaults) **GATE FAIL** — L11 L2 stayed huge (~888); student
neu@1 did not match frozen-plus oracle detail. Retrain with
heavier late-layer match and a stronger rel-L2 term
(`--embed_late_weight` / `--embed_rel_l2_weight`; defaults stay
2.0 / 1.0 so smile-krea-v4 is unchanged).

Retrain card (**detail-krea-v2**). Gate: student neu@1 approaches
frozen-plus oracle detail on landscape / room / object
(not `embed_cos`). **detail-krea-v1** and **v2** also failed
partly from that teacher **style gap**: oracle frozen-plus was
richly detailed, but student@1 washed out (landscape) or changed
subject (object → soft fabric). Smile worked because closed-mouth
vs teeth kept the same person.

```bash
CUDA_VISIBLE_DEVICES=0 python conceptmod/textsliders/train_lora_krea.py \
  --name detail-krea-v2 \
  --prompts_file conceptmod/textsliders/data/prompts-krea-detailed.yaml \
  --model_id krea/Krea-2-Raw --allow_hub \
  --lora_targets te --lm_target embed \
  --embed_cosine_weight 0 --embed_rel_l2_weight 2.0 --embed_late_weight 4.0 \
  --hold_weight 0.1 --sample_guidance 0 --te_dit_mask auto \
  --rank 32 --steps 1600 --lr 1e-4 --resolution 512 --sample_steps 28 \
  --seed 7 --device 0 \
  --save_dir models/detail-krea-v2
```

**detail-krea-v3** keeps the v2 train flags and revises captions
to a **detail-only** delta on the **same concrete subject**
(oak tree on a grassy hill / wooden chair in an empty room /
ceramic vase on a wood table). Neu is ordinary / plain / smooth /
untextured and still photographic. Plus adds highly detailed /
intricate surface textures / fine geometric detail / crisp
materials. Recommended card once those captions land. Gate:
student neu@1 keeps the same subject and approaches frozen-plus
oracle detail (not `embed_cos`).

```bash
CUDA_VISIBLE_DEVICES=0 python conceptmod/textsliders/train_lora_krea.py \
  --name detail-krea-v3 \
  --prompts_file conceptmod/textsliders/data/prompts-krea-detailed.yaml \
  --model_id krea/Krea-2-Raw --allow_hub \
  --lora_targets te --lm_target embed \
  --embed_cosine_weight 0 --embed_rel_l2_weight 2.0 --embed_late_weight 4.0 \
  --hold_weight 0.1 --sample_guidance 0 --te_dit_mask auto \
  --rank 32 --steps 1600 --lr 1e-4 --resolution 512 --sample_steps 28 \
  --seed 7 --device 0 \
  --save_dir models/detail-krea-v3
```

Resmoke the existing **smile-krea-v3** TE adapter with the v5
ones-mask (apply A/B only; v3 cosine match is not a teeth gate).
Compare `student_neu_scale1` /
`student_neu_scale1_onesmask` vs `student_neu_scale1_tokmask`:

```bash
python conceptmod/textsliders/train_lora_krea.py \
  --name smile-krea-v5-resmoke \
  --prompts_file conceptmod/textsliders/data/prompts-krea-happy.yaml \
  --model_id krea/Krea-2-Raw --allow_hub \
  --lora_targets te --lm_target embed \
  --load_te_lora models/smile-krea-v3/smile-krea-v3_lora/te_lora \
  --sample_guidance 0 --te_dit_mask auto \
  --save_dir models/smile-krea-v5-resmoke
```

If ones-mask student shows teeth and tokmask does not **and**
max_abs is small, the v3 adapter already learned plus magnitude —
only the DiT mask was wrong. Live transplant after #77 did **not**
recover teeth from student embeds; retrain v4.

Previous cosine-weighted card (smile-krea-v4 @ 500 steps;
`--embed_cosine_weight` default was 1.0, neu tokenizer mask):

```bash
CUDA_VISIBLE_DEVICES=0 python conceptmod/textsliders/train_lora_krea.py \
  --name smile-krea-v4 \
  --prompts_file conceptmod/textsliders/data/prompts-krea-happy.yaml \
  --model_id krea/Krea-2-Raw --allow_hub \
  --lora_targets te --lm_target embed \
  --rank 16 --hold_weight 0.1 --resolution 512 \
  --sample_steps 28 --sample_guidance 0 \
  --steps 500 --lr 1e-4 --seed 7 --device 0 \
  --save_dir models/smile-krea-v4
```

Adapter: `{save_dir}/{name}_lora/te_lora`. DiT is not adapted.
Sample PNGs: `{save_dir}/samples/`. Oracle grid:
`{save_dir}/samples/oracle/` (`oracle_plus_frozen` /
`student_neu_scale1` / `neu_scale0` + `oracle_meta.json`).
`--sample_guidance 0` is the embed default (Raw CFG 4.5 fights
TE-only). `--recipe embed_uni` is the same flag as `--lm_target embed`.

Resmoke a saved adapter without retraining:

```bash
python conceptmod/textsliders/train_lora_krea.py \
  --name smile-krea-v4-resmoke \
  --prompts_file conceptmod/textsliders/data/prompts-krea-happy.yaml \
  --model_id krea/Krea-2-Raw --allow_hub \
  --lora_targets te --lm_target embed \
  --load_te_lora models/smile-krea-v4/smile-krea-v4_lora/te_lora \
  --sample_guidance 0 --save_dir models/smile-krea-v4-resmoke
```

Previous card (**smile-krea-v3**, same TE embed UNI; sampled at
CFG 4.5 with both CFG branches under the active TE scale — Δ
cancelled in pixels):

```bash
CUDA_VISIBLE_DEVICES=0 python conceptmod/textsliders/train_lora_krea.py \
  --name smile-krea-v3 \
  --prompts_file conceptmod/textsliders/data/prompts-krea-happy.yaml \
  --model_id krea/Krea-2-Raw --allow_hub \
  --lora_targets te --lm_target embed \
  --rank 16 --hold_weight 0.1 --resolution 512 \
  --sample_steps 28 --sample_guidance 4.5 \
  --steps 500 --lr 1e-4 --seed 7 --device 0 \
  --save_dir models/smile-krea-v3
```

Joint DiT+TE velocity UNI (**smile-krea-v2**, A100 80GB preferred;
A6000 48GB is tight once TE stays resident):

```bash
CUDA_VISIBLE_DEVICES=0 python conceptmod/textsliders/train_lora_krea.py \
  --name smile-krea-v2 \
  --prompts_file conceptmod/textsliders/data/prompts-krea-happy.yaml \
  --model_id krea/Krea-2-Raw \
  --allow_hub \
  --lora_targets dit+te --rank 16 --hold_weight 0.1 \
  --resolution 512 \
  --sample_steps 28 --sample_guidance 4.5 \
  --steps 500 --lr 1e-4 --seed 7 --device 0 \
  --save_dir models/smile-krea-v2
```

Adapters: `{save_dir}/{name}_lora/dit_lora` and
`{save_dir}/{name}_lora/te_lora` (sidecar keys `dit_lora` /
`te_lora`). Sample PNGs: `{save_dir}/samples/`. Mid-scales must
differ — if 0.25 / 0.5 / 1.0 PNGs are byte-identical, scale is
broken again.

DiT-only (v1 card, frozen TE parked, 48GB):

```bash
CUDA_VISIBLE_DEVICES=0 python conceptmod/textsliders/train_lora_krea.py \
  --name smile-krea \
  --prompts_file conceptmod/textsliders/data/prompts-krea-happy.yaml \
  --model_id krea/Krea-2-Raw \
  --allow_hub \
  --lora_targets dit --rank 16 --hold_weight 0.1 \
  --resolution 512 \
  --sample_steps 28 --sample_guidance 4.5 \
  --steps 500 --lr 1e-4 --seed 7 --device 0 \
  --save_dir models/smile-krea
```

Age yaml (prefixed unused gender) is still the stock file:

```bash
CUDA_VISIBLE_DEVICES=0 python conceptmod/textsliders/train_lora_krea.py \
  --name age-krea \
  --prompts_file conceptmod/textsliders/data/prompts-krea.yaml \
  --model_id krea/Krea-2-Raw \
  --allow_hub \
  --rank 16 --resolution 512 \
  --sample_steps 28 --sample_guidance 4.5 \
  --seed 7 --device 0
```

Local Turbo file (inference / sample card only; do not prefer this for
train). VAE + Qwen3-VL still come from a cached Raw skeleton:

```bash
python conceptmod/textsliders/train_lora_krea.py \
  --name age-krea-turbo-card \
  --model_id /path/to/Krea-2-Turbo.safetensors \
  --rank 16 --resolution 512
# sample_steps=8 sample_guidance=0  (from the filename)
```

`--dummy` is the CI path: two CPU steps, never loads the 12B
transformer or Hub weights, never imports `krea_live.py`.

```bash
python conceptmod/textsliders/train_lora_krea.py \
  --dummy --name krea-age-dummy \
  --prompts_file conceptmod/textsliders/data/prompts-krea.yaml \
  --save_dir /tmp/krea-dummy
python conceptmod/textsliders/train_lora_krea.py \
  --dummy --name smile-krea-v3-dummy \
  --prompts_file conceptmod/textsliders/data/prompts-krea-happy.yaml \
  --lora_targets te --lm_target embed --hold_weight 0.1 \
  --save_dir /tmp/krea-embed-dummy
python conceptmod/textsliders/train_lora_krea.py \
  --dummy --name smile-krea-v4-dummy \
  --prompts_file conceptmod/textsliders/data/prompts-krea-happy.yaml \
  --lora_targets te --lm_target embed \
  --embed_cosine_weight 0 --hold_weight 0.1 \
  --te_dit_mask auto \
  --save_dir /tmp/krea-embed-v4-dummy
python conceptmod/textsliders/train_lora_krea.py \
  --dummy --name smile-krea-v5-dummy \
  --prompts_file conceptmod/textsliders/data/prompts-krea-happy.yaml \
  --lora_targets te --lm_target embed --hold_weight 0.1 \
  --save_dir /tmp/krea-embed-v5-dummy
python conceptmod/textsliders/train_lora_krea.py \
  --dummy --name detail-krea-v2-dummy \
  --prompts_file conceptmod/textsliders/data/prompts-krea-detailed.yaml \
  --lora_targets te --lm_target embed \
  --embed_cosine_weight 0 --embed_rel_l2_weight 2.0 --embed_late_weight 4.0 \
  --hold_weight 0.1 --te_dit_mask auto \
  --save_dir /tmp/krea-detail-v2-dummy
python conceptmod/textsliders/train_lora_krea.py \
  --dummy --name detail-krea-v3-dummy \
  --prompts_file conceptmod/textsliders/data/prompts-krea-detailed.yaml \
  --lora_targets te --lm_target embed \
  --embed_cosine_weight 0 --embed_rel_l2_weight 2.0 --embed_late_weight 4.0 \
  --hold_weight 0.1 --te_dit_mask auto \
  --save_dir /tmp/krea-detail-v3-dummy
PYTHONPATH=. pytest tests/test_krea_slider.py -q
```

## Yaml

`conceptmod/textsliders/data/prompts-krea-happy.yaml`: UNI happy/smile
(not age). Stronger Anima teacher (`big smile showing teeth, happy
joyful expression`) on a closed-mouth neu. `attributes` pin unused
gender (`male` / `female`) without prefixing. `negative` is the
canary. `control_prompt` is `a bowl of fruit on a table` — verify
only, never a teacher.

`conceptmod/textsliders/data/prompts-krea-detailed.yaml`: UNI
**detailed** (not smile). Three same-subject rows (landscape-ish
oak tree / interior wooden chair / object ceramic vase). Neu is
ordinary / plain / smooth / untextured (photographic, not cartoon
flat). Plus is that same subject with intricate surface textures
and crisp materials. Same bare-caption / unused-gender /
fruit-bowl / canary-minus shape as happy. v1/v2 teachers invited
a style jump; v3 is a detail-only delta. Generalizes
smile-krea-v4 TE-only embed UNI.

`conceptmod/textsliders/data/prompts-krea.yaml`: stock age slider
(old / young) with prefixed unused gender. Still valid.

## Related

- [conceptmod backends/krea.py](https://github.com/mikkel/conceptmod/blob/main/conceptmod/backends/krea.py)
- [docs/sana-slider.md](sana-slider.md) — cheap happy UNI analog
- [docs/anima-slider.md](anima-slider.md) — smile-first (do not copy `same_crop` / `embed_struct` here)
- Music 3 live defaults stay `--lm_target v9 --pole_mode hidden`.
