# Hunt: `faithful_plus_neu` live lyric shred

Read of `train_lm_slider_music3.py`, `slider_targets.py`
(`lm_faithful_plus_neu` / `lm_plus_neu_loss`), LoRA apply, and
`generate_listen.py`. CPU only. No Hub, no GPU, no Music 3
weights. Default stays `--lm_target v9` / `--pole_mode hidden`.
No yaml rewrite. No new leaderboard.

**Verdict: no train/apply bug.** UNI is wired as specified.
The live grit/distortion/joy shred is a missed existing OOD
gate, not a leftover-gate or token-swap.

## The eight checks

1. **+1 teacher is `encode(pos)` with LoRA off.** Yes.
   `_encode_static` runs on `tokens["positive"]` before
   `LoRANetwork` exists (`train_lm_slider_music3.py` encode
   loop, then wrap). `lm_faithful_plus_neu` returns `pos`
   and deletes `leak_dir` / `slider_dir`.

2. **+1 student is `encode(neu tokens, LoRA@+1)`.** Yes.
   `row_data` stores `tokens["neutral"]`. The step does
   `_set_scale(network, 1.0)` then `_encode_train(lm, neu_ids,
   neu_mask)` (or the endreg teacher-force of the same neu
   `prompt_embeds`). Student never sees pos tokens. Listen
   is the same: neu caption + yaml lyrics, LoRA scale only
   (`generate_listen._jobs`).

3. **MSE is last non-pad = `<|audio_start|>`.** Yes.
   `_assemble` ends with `<|audio_start|>`. `_tokenize` is
   unpadded, `add_special_tokens=False`. Gather is
   `attention_mask.sum(dim=1) - 1`. That is the AR continue
   token (`_preroll_frames` uses `last_hidden_state[:, -1]`
   on the same assembled ids). Not mean-pool.

4. **Scale 0 is LoRA off; `h0` is `encode(neu)` LoRA-off.** Yes.
   Teacher `neu_ref` is `_encode_static` before the wrap.
   Student 0 is `_set_scale(network, 0.0)` then the same neu
   encode. `LoRAModule.forward` is `org_forward + delta *
   multiplier * (alpha/rank)`; multiplier 0 is the base
   forward. Listen scale 0 / REF sets multiplier 0 the same
   way.

5. **`plus_neu` does not leftover-gate or use `h0 ± a`.** Yes.
   Teacher is raw `pos`. Loss is `MSE(+) + MSE(0)` only
   (`lm_plus_neu_loss`). Minus is a canary (`return plus, neg`).
   Hold/anchor weights resolve to 0.

6. **Train vs listen assemble.** Same contract, not a grit bug.
   Train `_assemble` is the pipeline special-token wrap
   (`_clean_caption` / `_normalize_lyrics` from the Music 3
   encoder). Listen calls `pipe(prompt=..., lyrics=...)` on
   that pipeline. Lyrics including `[verse]` tags come from
   the same yaml field. LoRA target is `Qwen3Attention` /
   `lora_te` on both sides.
   One real difference: train expands `attributes` onto
   captions (`_expand_attributes`); listen `_load_prompt_row`
   does not. That is **not** this shred. Tempo-v4 has
   attributes and keeps lyrics; grit-v1 has none and
   shreds; gender-v4 has none and keeps lyrics.

7. **endreg / planreg can fight the roll while hidden still
   hits.** Yes they can; they are not this bug.
   Default `--endreg_weight 1.0` teacher-forces LoRA'd neu
   over a **base neu** pre-roll and MSEs end-margins,
   including at prompt-last (the slider token).
   `--planreg_weight` default is 0. Gender/tempo ship with
   the same endreg default and keep lyrics, so endreg is
   not the grit-only cause. planreg, if turned on, would
   pin frame hiddens to the neu plan on purpose.

8. **Existing OOD columns miss because they are not a live
   AR roll.** Yes.
   Live `c+` / `p%` are last-hidden cosine / relative L2
   (`pred_pos − neu` vs `pos − neu`). Pair-exam
   `off_caption` / `same_words` and plus-neu `off_caption`
   / sheet `garble` toy-roll **from that last hidden**
   through a fixture readout. If last hidden ≈ `encode(pos)`,
   those columns report clean. They never see the Music 3
   continuation or the LoRA'd lyric-token KV.
   Matching last hidden is not matching the prompt KV:
   REF is pos tokens, LoRA off; listen +1 is neu tokens,
   LoRA on every `Qwen3Attention` over caption **and**
   lyrics. First audio logit can look like pos while later
   frames attend a warped lyric cache. Gender is a small
   close-pair nudge; grit/distortion/joy have to rewrite
   more of the caption path, so they hit the lyrics.

## Which existing number catches grit vs gender

**`scripts/blindspot_whisper.py` → `lyric_recall` on the +1
listen wav.** That is the live roll, not last-hidden. The
reported symptom is already this number: grit +1
`lyric_recall` 0.00 while `c+` 0.945; gender/tempo +1 stay
high; REF (pos caption, LoRA off) still sings the sheet.

Do not use plus-neu exam `off_caption` (0.000 HIT on the
CPU fixture) or train `c+` as a lyric gate. They cannot
see this OOD.

No code change. Default stays v9/hidden.
