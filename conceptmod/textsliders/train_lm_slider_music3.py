"""Notrigger-style slider on MiniMax Music 3's language model.

Vocal identity is chosen in the AR stage, not the flow transformer. This trains
LoRA on Qwen3Attention so a neutral caption + LoRA scale moves the prompt
hidden state toward a female or male caption.

Default ``--lm_target v9`` is the 2-D leak-0 geometry: ``--symmetric`` polarity,
then ``lm_project_odd_axis`` onto a **declared** ``--slider_positive`` /
``--slider_negative`` pair, plus ``lm_ortho_hold`` on the orthogonal residual.
κ = 0. A bare ``--symmetric`` retrain without that pair would still leak.
Published Hub floor is ``--lm_target hub`` and is not the default.

Audio-end regularizer: a cut ends when the AR language model samples
<|audio_end|>; LM LoRAs perturb those logits, and un-regularized sliders
measurably suppress the end token, so stacked sliders blow through the
duration cap and get truncated mid-phrase. With --endreg_weight > 0 each
prompt row is pre-rolled once with the base model (the same CFG compose loop
inference uses; cached on disk), and every training step teacher-forces the
LoRA'd model over that composition, penalizing drift of the end margin —
logit(<|audio_end|>) minus logsumexp over the semantic-code band — at every
decode position. The slider keeps moving the musical plan; it must not move
the stop-decision axis.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

_HF_HOME = "/ml2/music/.cache/huggingface"
os.environ["HF_HOME"] = _HF_HOME
os.environ["HUGGINGFACE_HUB_CACHE"] = f"{_HF_HOME}/hub"
os.environ["HF_HUB_CACHE"] = f"{_HF_HOME}/hub"
os.environ["TRANSFORMERS_CACHE"] = f"{_HF_HOME}/hub"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

import torch
import torch.nn.functional as F
import yaml
from tqdm import tqdm

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from conceptmod.textsliders.slider_targets import (
    lm_anchor_kappa,
    lm_anchor_targets,
    lm_hidden_targets,
    lm_ortho_hold,
    lm_project_odd_axis,
    lm_slider_loss,
    lm_unit,
)

DEFAULT_MODEL = Path("/ml2/music/models/MiniMax-Music3")
LM_RECIPES = ("v9", "hub", "symmetric", "faithful")
TARGET_REPLACE = ["Qwen3Attention"]

# Row fields this trainer actually consumes (`attributes` is expanded away by
# _expand_attributes). Anything else in the YAML - `action`, `guidance_scale`,
# `batch_size`, `unconditional` - is only read by the transformer trainer.
# Declared slider axis lives at YAML top-level (slider_positive / slider_negative)
# or on the CLI; it is not a per-row field and is never inferred from (pos-neg).
_USED_ROW_KEYS = frozenset({"target", "positive", "negative", "neutral", "lyrics", "attributes"})


def resolve_lm_recipe(*, lm_target: str, symmetric: bool) -> str:
    """Live trainer recipe. Default ``v9`` is the 2-D leak-0 cell.

    ``--symmetric`` is the polarity step inside ``v9`` / ``hub`` / ``symmetric``.
    It is not a second loss.
    """
    recipe = str(lm_target).strip().lower()
    if recipe not in LM_RECIPES:
        raise ValueError(f"lm_target must be one of {LM_RECIPES}, got {lm_target!r}")
    if recipe == "v9" and not symmetric:
        raise ValueError(
            "lm_target=v9 keeps --symmetric as the polarity step; "
            "use --lm_target faithful --no-symmetric for raw poles"
        )
    if recipe == "hub" and not symmetric:
        raise ValueError(
            "lm_target=hub is the published symmetric + leakage_floor blend; got --no-symmetric"
        )
    if recipe == "symmetric" and not symmetric:
        raise ValueError("lm_target=symmetric requires --symmetric")
    return recipe


def resolve_slider_axis_captions(
    *,
    slider_positive: str | None,
    slider_negative: str | None,
    prompts_meta: dict,
) -> tuple[str, str] | None:
    """Declared slider pair. CLI wins over YAML. Never (pos−neg) of a row."""
    pos = (slider_positive or prompts_meta.get("slider_positive") or "").strip()
    neg = (slider_negative or prompts_meta.get("slider_negative") or "").strip()
    if pos and neg:
        return pos, neg
    if pos or neg:
        raise ValueError("declared slider axis needs both slider_positive and slider_negative")
    return None


def resolve_lm_loss_weights(
    recipe: str,
    *,
    hold_weight: float | None,
    anchor_weight: float | None,
) -> tuple[float, float]:
    hold = 1.0 if recipe == "v9" else 0.0
    anchor = 0.3 if recipe == "hub" else 0.0
    if hold_weight is not None:
        hold = float(hold_weight)
    if anchor_weight is not None:
        anchor = float(anchor_weight)
    return hold, anchor


def lm_train_targets(
    pos: torch.Tensor,
    neg: torch.Tensor,
    neu: torch.Tensor,
    *,
    recipe: str,
    slider_dir: torch.Tensor | None = None,
    symmetric: bool = True,
    target_scale: float = 1.0,
    common_beta: float = 0.0,
    leakage_floor: float | None = None,
    anchor_autocal: bool = True,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
    """Pole targets for the live LM trainer. Delegates to ``slider_targets``.

    Returns ``(tgt_plus, tgt_minus, anchor_plus, anchor_minus)``.
    Anchors are set only for the published Hub floor recipe.
    """
    if recipe == "v9":
        if slider_dir is None:
            raise ValueError("lm_target=v9 requires a declared slider_dir")
        plus, minus = lm_project_odd_axis(pos, neg, neu, slider_dir, target_scale=target_scale)
        return plus, minus, None, None
    target_mode = "faithful" if recipe == "faithful" else "symmetric"
    plus, minus = lm_hidden_targets(
        pos,
        neg,
        neu,
        target_mode=target_mode,
        symmetric=symmetric,
        target_scale=target_scale,
        common_beta=common_beta,
    )
    if recipe != "hub":
        return plus, minus, None, None
    floor = -0.9 if leakage_floor is None else float(leakage_floor)
    kappa = lm_anchor_kappa(pos, neg, neu, floor, autocal=anchor_autocal)
    anc_plus, anc_minus = lm_anchor_targets(pos, neg, neu, kappa, target_scale=target_scale)
    return plus, minus, anc_plus, anc_minus


def lm_train_loss(
    pred_plus: torch.Tensor,
    pred_minus: torch.Tensor,
    tgt_plus: torch.Tensor,
    tgt_minus: torch.Tensor,
    *,
    neu: torch.Tensor | None = None,
    slider_dir: torch.Tensor | None = None,
    hold_weight: float = 0.0,
    anchor_plus: torch.Tensor | None = None,
    anchor_minus: torch.Tensor | None = None,
    anchor_weight: float = 0.0,
) -> torch.Tensor:
    """Live pole loss: ``lm_slider_loss`` plus optional ``lm_ortho_hold``.

    No second lookalike. The live graph has no hold embedding; hold is the
    residual orthogonal to the declared ``slider_dir``.
    """
    hold = None
    if float(hold_weight) > 0.0:
        if neu is None or slider_dir is None:
            raise ValueError("hold_weight>0 requires neu and a declared slider_dir")
        hold = lm_ortho_hold(pred_plus, pred_minus, neu, slider_dir)
    return lm_slider_loss(
        pred_plus,
        pred_minus,
        tgt_plus,
        tgt_minus,
        anchor_plus=anchor_plus,
        anchor_minus=anchor_minus,
        anchor_weight=anchor_weight,
        hold=hold,
        hold_weight=hold_weight,
    )


_IM_START, _IM_END = "<|im_start|>", "<|im_end|>"
_CAPTION_START, _CAPTION_END = "<|caption_start|>", "<|caption_end|>"
_LYRICS_START, _LYRICS_END = "<|lyrics_start|>", "<|lyrics_end|>"
_AUDIO_START = "<|audio_start|>"


def _assemble(prompt: str, lyrics: str) -> str:
    from diffusers.modular_pipelines.minimax_music3.encoders import (
        _clean_caption,
        _normalize_lyrics,
    )

    return (
        f"{_IM_START}{_CAPTION_START}{_clean_caption(prompt)}{_CAPTION_END}"
        f"{_LYRICS_START}{_normalize_lyrics(lyrics)}{_LYRICS_END}{_IM_END}{_AUDIO_START}"
    )


def _load_rows(path: Path) -> tuple[list[dict], dict]:
    """All prompt rows (with `attributes` expansion) plus top-level label metadata."""
    from conceptmod.textsliders.train_lora_music3 import _expand_attributes

    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    meta: dict = {}
    if isinstance(raw, dict):
        meta = {
            "plus_label": str(raw.get("plus_label") or ""),
            "minus_label": str(raw.get("minus_label") or ""),
            "recommended_range": raw.get("recommended_range") or [-2.0, 2.0],
            "slider_positive": str(raw.get("slider_positive") or ""),
            "slider_negative": str(raw.get("slider_negative") or ""),
        }
        raw = raw.get("rows")
    if not isinstance(raw, list) or not raw:
        raise ValueError(f"prompts file is empty: {path}")
    ignored = sorted(
        {
            str(key)
            for item in raw
            if isinstance(item, dict)
            for key in item
            if str(key) not in _USED_ROW_KEYS
        }
    )
    if ignored:
        print(f"note: ignoring per-row fields not used by the LM trainer: {', '.join(ignored)}")
    rows: list[dict] = []
    for item in raw:
        rows.extend(_expand_attributes(item))
    return rows, meta


def _tokenize(tokenizer, text: str, device: torch.device):
    out = tokenizer(text, return_tensors="pt", add_special_tokens=False)
    return out["input_ids"].to(device), out["attention_mask"].to(device)


@torch.no_grad()
def _encode_static(lm, input_ids, attention_mask) -> torch.Tensor:
    hidden = lm.model(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
    # Last real token (the audio-start token) is what AR continues from.
    lengths = attention_mask.sum(dim=1) - 1
    gather = lengths.view(-1, 1, 1).expand(-1, 1, hidden.size(-1))
    last = hidden.gather(1, gather.clamp(min=0)).squeeze(1)
    return last.float()


def _encode_train(lm, input_ids, attention_mask) -> torch.Tensor:
    hidden = lm.model(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
    lengths = attention_mask.sum(dim=1) - 1
    gather = lengths.view(-1, 1, 1).expand(-1, 1, hidden.size(-1))
    last = hidden.gather(1, gather.clamp(min=0)).squeeze(1)
    return last.float()


def _set_scale(network: LoRANetwork, scale: float) -> None:
    network.set_lora_slider(scale)
    for lora in network.unet_loras:
        lora.multiplier = float(scale)


def _frame_margins(lm, hidden_tail: torch.Tensor) -> torch.Tensor:
    """End margin — logit(<|audio_end|>) - logsumexp(semantic-code band) — at
    each decode position. hidden_tail: [1, P, hidden]; returns float32 [P].
    Only the lm_head rows that decide stop-vs-continue are multiplied out."""
    from diffusers.modular_pipelines.minimax_music3.encoders import (
        _AUDIO_CODE_OFFSET,
        _AUDIO_END_TOKEN_ID,
        _SEMANTIC_VOCAB_SIZE,
    )

    weight = lm.lm_head.weight
    hidden = hidden_tail[0]
    end_logit = (hidden @ weight[_AUDIO_END_TOKEN_ID]).float()
    sem_logits = (hidden @ weight[_AUDIO_CODE_OFFSET : _AUDIO_CODE_OFFSET + _SEMANTIC_VOCAB_SIZE].T).float()
    return end_logit - torch.logsumexp(sem_logits, dim=-1)


def _forward_teacher_forced(lm, prompt_embeds: torch.Tensor, frame_embeds: torch.Tensor | None):
    """One forward over [prompt; composed frames]. Returns the prompt-last
    hidden state (the slider target position — causality makes it identical to
    a prompt-only forward), the end margins at positions prompt-last..end, and
    the full hidden sequence for optional plan-drift regularization."""
    embeds = prompt_embeds if frame_embeds is None else torch.cat((prompt_embeds, frame_embeds), dim=1)
    mask = torch.ones(embeds.shape[:2], dtype=torch.long, device=embeds.device)
    hidden = lm.model(inputs_embeds=embeds, attention_mask=mask).last_hidden_state
    prompt_last = hidden[:, prompt_embeds.shape[1] - 1]
    margins = _frame_margins(lm, hidden[:, prompt_embeds.shape[1] - 1 :])
    return prompt_last.float(), margins, hidden


@torch.no_grad()
def _preroll_frames(lm, depth_decoder, cond_ids: torch.Tensor, frames_cap: int, seed: int, device):
    """Compose up to frames_cap frames with the base model, exactly like
    inference (CFG pair, top-k, depth codes, feedback embeddings). Returns
    (frame_embeds [1, N, hidden] or None, ended_naturally)."""
    from types import SimpleNamespace

    from diffusers.modular_pipelines.minimax_music3.encoders import (
        _AR_CFG_SCALE,
        _AR_CFG_TOP_K,
        _AUDIO_CFG_TOKEN_ID,
        _AUDIO_CODE_OFFSET,
        _AUDIO_END_TOKEN_ID,
        _SEMANTIC_VOCAB_SIZE,
        _embed_audio_frame,
        _generate_depth_codes,
        _sample_top_k,
    )

    shim = SimpleNamespace(
        language_model=lm,
        rvq_depth_decoder=depth_decoder,
        num_codebooks=int(depth_decoder.config.num_codebooks),
        audio_vocab_size=int(depth_decoder.config.audio_vocab_size),
    )
    unconditional_ids = cond_ids.clone()
    unconditional_ids[:, 1:-2] = _AUDIO_CFG_TOKEN_ID
    text_ids = torch.cat((cond_ids, unconditional_ids), dim=0)
    generator = torch.Generator(device).manual_seed(seed)

    output = lm.model(inputs_embeds=lm.model.embed_tokens(text_ids), use_cache=True)
    past_key_values = output.past_key_values
    last_hidden = output.last_hidden_state[:, -1]

    vocab_mask = torch.ones(lm.config.vocab_size, dtype=torch.bool, device=device)
    vocab_mask[_AUDIO_CODE_OFFSET : _AUDIO_CODE_OFFSET + _SEMANTIC_VOCAB_SIZE] = False
    vocab_mask[_AUDIO_END_TOKEN_ID] = False

    frame_embeds: list[torch.Tensor] = []
    ended = False
    while len(frame_embeds) < frames_cap:
        logits = lm.lm_head(last_hidden).float()
        logits = logits.masked_fill(vocab_mask, -float("inf"))
        conditional, unconditional = logits[0:1], logits[1:2]
        guided = unconditional + (conditional - unconditional) * _AR_CFG_SCALE
        threshold = torch.topk(conditional, _AR_CFG_TOP_K, dim=-1).values[..., -1, None]
        guided = guided.masked_fill(conditional < threshold, -float("inf"))
        guided = guided.masked_fill(vocab_mask.unsqueeze(0), -float("inf"))
        sampled = _sample_top_k(guided, generator)
        if int(sampled.item()) == _AUDIO_END_TOKEN_ID:
            ended = True
            break
        semantic_code = sampled - _AUDIO_CODE_OFFSET
        frame_codes, _depth_hidden = _generate_depth_codes(shim, last_hidden, semantic_code.repeat(2), generator)
        feedback = _embed_audio_frame(shim, frame_codes)
        frame_embeds.append(feedback[0:1])
        output = lm.model(inputs_embeds=feedback, past_key_values=past_key_values, use_cache=True)
        past_key_values = output.past_key_values
        last_hidden = output.last_hidden_state[:, -1]

    frames = torch.cat(frame_embeds, dim=1) if frame_embeds else None
    return frames, ended


def _endreg_cache_path(cache_dir: Path, model_dir: str, text: str, frames_cap: int, seed: int) -> Path:
    import hashlib

    digest = hashlib.sha256(
        json.dumps([str(model_dir), text, int(frames_cap), int(seed)]).encode("utf-8")
    ).hexdigest()[:16]
    return cache_dir / f"preroll-{digest}.pt"


def train(args: argparse.Namespace) -> Path:
    device = torch.device(f"cuda:{int(args.device)}")
    # Pin every consumer of global RNG (LoRA init is the only one) so two runs
    # differing only in --seed are step-for-step comparable, as in the
    # transformer trainer's determinism fix. The endreg pre-roll carries its
    # own generator (--endreg_seed).
    torch.manual_seed(int(args.seed))
    torch.cuda.manual_seed_all(int(args.seed))
    import random

    random.seed(int(args.seed))
    rows, prompts_meta = _load_rows(Path(args.prompts_file))
    if args.plus_label:
        prompts_meta["plus_label"] = args.plus_label
    if args.minus_label:
        prompts_meta["minus_label"] = args.minus_label

    recipe = resolve_lm_recipe(lm_target=args.lm_target, symmetric=args.symmetric)
    hold_w, anchor_w = resolve_lm_loss_weights(
        recipe, hold_weight=args.hold_weight, anchor_weight=args.anchor_weight
    )
    axis_captions = resolve_slider_axis_captions(
        slider_positive=args.slider_positive,
        slider_negative=args.slider_negative,
        prompts_meta=prompts_meta,
    )
    if (recipe == "v9" or hold_w > 0.0) and axis_captions is None:
        raise ValueError(
            "lm_target=v9 (and any hold_weight>0) needs a declared slider axis: "
            "--slider_positive / --slider_negative, or YAML slider_positive / "
            "slider_negative. Do not silently use a row's (pos-neg) — that is "
            "the unused-attribute leak."
        )
    if recipe == "v9" and float(args.common_beta) != 0.0:
        print("note: lm_target=v9 is κ=0 (no even blend-back); ignoring --common_beta")

    from transformers import AutoModelForCausalLM, AutoTokenizer
    from conceptmod.textsliders.lora import LoRANetwork

    tokenizer = AutoTokenizer.from_pretrained(
        str(Path(args.model_dir) / "tokenizer"),
        local_files_only=True,
    )
    lm = AutoModelForCausalLM.from_pretrained(
        str(Path(args.model_dir) / "language_model"),
        torch_dtype=torch.bfloat16,
        local_files_only=True,
    )
    lm.to(device)
    lm.eval()
    lm.requires_grad_(False)

    beta = float(args.common_beta)
    slider_dir = None
    if axis_captions is not None:
        axis_lyrics = str(rows[0].get("lyrics") or "")
        axis_pos_text = _assemble(axis_captions[0], axis_lyrics)
        axis_neg_text = _assemble(axis_captions[1], axis_lyrics)
        with torch.no_grad():
            axis_pos_h = _encode_static(lm, *_tokenize(tokenizer, axis_pos_text, device))
            axis_neg_h = _encode_static(lm, *_tokenize(tokenizer, axis_neg_text, device))
        slider_dir = axis_pos_h - axis_neg_h
        print(
            f"declared slider axis: {axis_captions[0]!r} / {axis_captions[1]!r} "
            f"||û||={slider_dir.norm().item():.3f} (encoded once, not a row's pos-neg)"
        )

    row_data = []
    for index, row in enumerate(rows):
        lyrics = str(row["lyrics"])
        texts = {
            "neutral": _assemble(str(row.get("neutral") or row["target"]), lyrics),
            "positive": _assemble(str(row["positive"]), lyrics),
            "negative": _assemble(str(row["negative"]), lyrics),
        }
        tokens = {name: _tokenize(tokenizer, text, device) for name, text in texts.items()}
        with torch.no_grad():
            pos_tgt = _encode_static(lm, *tokens["positive"])
            neg_tgt = _encode_static(lm, *tokens["negative"])
            neu_ref = _encode_static(lm, *tokens["neutral"])
        target_cos = F.cosine_similarity(pos_tgt - neu_ref, neg_tgt - neu_ref, dim=-1).mean().item()
        odd = (pos_tgt - neg_tgt) / 2.0
        dropped = ""
        if slider_dir is not None:
            unit = lm_unit(slider_dir)
            align = (odd.flatten() @ unit).abs() / odd.norm().clamp_min(1e-8)
            dropped = f" odd·û/||odd||={float(align):.3f}"
        print(
            f"row {index}: tokens={tokens['neutral'][0].shape[1]} "
            f"L2 pos={torch.norm(pos_tgt - neu_ref).item():.3f} "
            f"neg={torch.norm(neg_tgt - neu_ref).item():.3f} "
            f"cos(pos-neu, neg-neu)={target_cos:.3f}{dropped}"
            f"{'  <- collapse inherited from raw targets' if target_cos > 0.3 else ''}"
        )
        tgt_plus, tgt_minus, anc_plus, anc_minus = lm_train_targets(
            pos_tgt,
            neg_tgt,
            neu_ref,
            recipe=recipe,
            slider_dir=slider_dir,
            symmetric=args.symmetric,
            target_scale=float(args.target_scale),
            common_beta=beta,
            leakage_floor=args.leakage_floor,
            anchor_autocal=args.anchor_autocal,
        )
        row_data.append(
            {
                "tokens": tokens["neutral"],
                "text_neutral": texts["neutral"],
                "tgt_plus": tgt_plus,
                "tgt_minus": tgt_minus,
                "neu_ref": neu_ref,
                "slider_dir": slider_dir,
                "anchor_plus": anc_plus,
                "anchor_minus": anc_minus,
            }
        )

    endreg = args.endreg_weight > 0 and args.endreg_frames > 0
    if endreg:
        # Pre-roll every row with the pristine base model (the LoRA does not
        # exist yet), teacher-force the same sequence once, and freeze the base
        # end margins the slider must preserve. The compose is the expensive
        # part, so it is cached on disk; margins are recomputed per run.
        cache_dir = Path(args.endreg_cache)
        cache_dir.mkdir(parents=True, exist_ok=True)
        depth_decoder = None
        ended_count = 0
        for index, data in enumerate(row_data):
            cache_path = _endreg_cache_path(
                cache_dir, args.model_dir, data["text_neutral"], args.endreg_frames, args.endreg_seed + index
            )
            if cache_path.exists():
                blob = torch.load(cache_path, map_location=device, weights_only=True)
            else:
                if depth_decoder is None:
                    from diffusers import MiniMaxMusic3RVQDepthDecoder

                    depth_decoder = MiniMaxMusic3RVQDepthDecoder.from_pretrained(
                        str(Path(args.model_dir) / "rvq_depth_decoder"),
                        torch_dtype=torch.bfloat16,
                        local_files_only=True,
                    ).to(device)
                    depth_decoder.eval()
                frames, ended = _preroll_frames(
                    lm, depth_decoder, data["tokens"][0], args.endreg_frames, args.endreg_seed + index, device
                )
                blob = {"frame_embeds": None if frames is None else frames.cpu(), "ended": bool(ended)}
                torch.save(blob, cache_path)
            frame_embeds = None if blob["frame_embeds"] is None else blob["frame_embeds"].to(device)
            with torch.no_grad():
                prompt_embeds = lm.model.embed_tokens(data["tokens"][0])
                _last, base_margins, base_hidden = _forward_teacher_forced(lm, prompt_embeds, frame_embeds)
            data["prompt_embeds"] = prompt_embeds
            data["frame_embeds"] = frame_embeds
            data["base_margins"] = base_margins
            # Frame-position hiddens for --planreg_weight: the composition the
            # slider must not re-plan. Prompt positions excluded (the slider
            # target lives at prompt-last; causality keeps it frame-free).
            data["base_hidden"] = base_hidden[:, prompt_embeds.shape[1] :].float()
            ended_count += int(blob["ended"])
            n_frames = 0 if frame_embeds is None else frame_embeds.shape[1]
            print(
                f"row {index} endreg: frames={n_frames} "
                f"{'ended naturally' if blob['ended'] else 'hit the pre-roll cap'} "
                f"base end-margin first={base_margins[0].item():.2f} last={base_margins[-1].item():.2f}"
            )
        if depth_decoder is not None:
            del depth_decoder
            torch.cuda.empty_cache()

    network = LoRANetwork(
        lm,
        rank=args.rank,
        alpha=args.alpha,
        multiplier=1.0,
        delimiter="-",
        target_replace=TARGET_REPLACE,
        prefix="lora_te",
        train_method="full",
    ).to(device)
    n_mod = len(network.unet_loras)
    if n_mod == 0:
        raise RuntimeError("LoRA wrapped 0 Qwen3Attention linears")
    print(f"LM LoRA modules={n_mod} rank={args.rank}")
    for p in network.parameters():
        p.requires_grad_(True)
    opt = torch.optim.AdamW(network.parameters(), lr=args.lr, weight_decay=1e-6)

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    metrics = save_dir / f"{args.name}_train.jsonl"
    metrics_handle = metrics.open("w")

    history = []
    early_fired = False
    pbar = tqdm(range(args.steps), desc="lm-slider")
    for step in pbar:
        data = row_data[step % len(row_data)]
        neu_ids, neu_mask = data["tokens"]
        tgt_plus, tgt_minus, neu_ref = data["tgt_plus"], data["tgt_minus"], data["neu_ref"]

        if endreg:
            # One forward per pole over [prompt; composed frames]: causality
            # keeps the prompt-last hidden identical to a prompt-only forward,
            # so the slider loss is unchanged and the end margins come free.
            _set_scale(network, 1.0)
            pred_pos, m_pos, hid_pos = _forward_teacher_forced(lm, data["prompt_embeds"], data["frame_embeds"])
            end_pos = F.mse_loss(m_pos, data["base_margins"])
            plan_pos = F.mse_loss(hid_pos[:, data["prompt_embeds"].shape[1] :].float(), data["base_hidden"])

            _set_scale(network, -1.0)
            pred_neg, m_neg, hid_neg = _forward_teacher_forced(lm, data["prompt_embeds"], data["frame_embeds"])
            end_neg = F.mse_loss(m_neg, data["base_margins"])
            plan_neg = F.mse_loss(hid_neg[:, data["prompt_embeds"].shape[1] :].float(), data["base_hidden"])
            edrift_p = float((m_pos - data["base_margins"]).abs().mean().detach())
            edrift_n = float((m_neg - data["base_margins"]).abs().mean().detach())
            pdrift_p = float(plan_pos.detach())
            pdrift_n = float(plan_neg.detach())
        else:
            if args.planreg_weight > 0:
                raise ValueError("--planreg_weight requires the audio-end regularizer (its pre-roll supplies the plan)")
            _set_scale(network, 1.0)
            pred_pos = _encode_train(lm, neu_ids, neu_mask)

            _set_scale(network, -1.0)
            pred_neg = _encode_train(lm, neu_ids, neu_mask)
            end_pos = end_neg = torch.zeros((), device=device)
            plan_pos = plan_neg = torch.zeros((), device=device)
            edrift_p = edrift_n = 0.0
            pdrift_p = pdrift_n = 0.0

        v_pos = pred_pos - neu_ref
        v_neg = pred_neg - neu_ref
        v_pos_t = tgt_plus - neu_ref
        v_neg_t = tgt_minus - neu_ref
        cos_pos = F.cosine_similarity(v_pos, v_pos_t, dim=-1).mean()
        cos_neg = F.cosine_similarity(v_neg, v_neg_t, dim=-1).mean()
        collapse = F.cosine_similarity(v_pos, v_neg, dim=-1).mean()
        pperc = (torch.norm(pred_pos - tgt_plus) / torch.norm(v_pos_t).clamp_min(1e-6)).item()
        nperc = (torch.norm(pred_neg - tgt_minus) / torch.norm(v_neg_t).clamp_min(1e-6)).item()

        pole = lm_train_loss(
            pred_pos,
            pred_neg,
            tgt_plus,
            tgt_minus,
            neu=neu_ref,
            slider_dir=data.get("slider_dir"),
            hold_weight=hold_w,
            anchor_plus=data.get("anchor_plus"),
            anchor_minus=data.get("anchor_minus"),
            anchor_weight=anchor_w,
        )
        loss = pole + 0.5 * args.endreg_weight * (end_pos + end_neg) + args.planreg_weight * (plan_pos + plan_neg)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_value_(network.parameters(), clip_value=1.0)
        opt.step()

        row = {
            # 1-indexed, matching train_lora_music3.py: step N == N completed updates.
            "step": step + 1,
            "row": step % len(row_data),
            "loss": float(loss.detach()),
            "pperc": pperc,
            "nperc": nperc,
            "cos_pos": float(cos_pos.detach()),
            "cos_neg": float(cos_neg.detach()),
            "collapse": float(collapse.detach()),
            # Mean |end-margin drift| in nats along the teacher-forced roll.
            "edrift_p": edrift_p,
            "edrift_n": edrift_n,
            # Mean-squared hidden drift along the composed frames (planreg).
            "pdrift_p": pdrift_p,
            "pdrift_n": pdrift_n,
        }
        history.append(row)
        pbar.set_description(
            f"loss {row['loss']:.4f} p%{pperc*100:.1f} n%{nperc*100:.1f} "
            f"c+ {row['cos_pos']:.2f} c- {row['cos_neg']:.2f} col {row['collapse']:.2f} "
            f"e± {edrift_p:.2f}/{edrift_n:.2f}"
        )
        metrics_handle.write(json.dumps(row) + "\n")
        metrics_handle.flush()

        # The final step's weights already ship as _last.safetensors, so skip the
        # duplicate mid checkpoint there.
        if args.save_every and (step + 1) % args.save_every == 0 and (step + 1) < args.steps:
            network.save_weights(
                save_dir / f"{args.name}_step{step + 1}.safetensors", dtype=torch.float32
            )

        if (
            args.early_stop
            and len(history) >= args.min_steps
            and _early_stop_hit(history, args.early_window, args.early_cos, args.early_collapse, args.early_perc)
        ):
            early_fired = True
            print(
                f"early stop at step {step + 1} "
                f"(window={args.early_window} c+/{args.early_cos} col/{args.early_collapse} perc/{args.early_perc})"
            )
            break

    metrics_handle.close()
    last = save_dir / f"{args.name}_last.safetensors"
    network.save_weights(last, dtype=torch.float32)
    stopped = len(history)
    win = history[-args.early_window :] if len(history) >= args.early_window else history
    early_metrics = {
        key: (sum(r[key] for r in win) / len(win) if win else None)
        for key in ("cos_pos", "cos_neg", "collapse", "pperc", "nperc")
    }
    summary = {
        "schema": 3,
        "name": args.name,
        "checkpoint": str(last),
        "weights": str(last),
        "modules": n_mod,
        "rank": args.rank,
        "alpha": args.alpha,
        "steps": stopped,
        "steps_budget": args.steps,
        "lr": args.lr,
        "seed": int(args.seed),
        "rows": len(row_data),
        "lm_target": recipe,
        "symmetric": bool(args.symmetric),
        "hold_weight": hold_w,
        "anchor_weight": anchor_w,
        "leakage_floor": args.leakage_floor,
        "slider_positive": axis_captions[0] if axis_captions else "",
        "slider_negative": axis_captions[1] if axis_captions else "",
        "common_beta": beta,
        "target_scale": float(args.target_scale),
        "planreg_weight": float(args.planreg_weight),
        "first": history[0] if history else None,
        "last": history[-1] if history else None,
        "target_replace": TARGET_REPLACE,
        "kind": "language_model",
        "prefix": "lora_te",
        "delimiter": "-",
        "train_method": "full",
        # Trained at ±1 by construction, so the user scale is the raw multiplier.
        "unit_scale": 1.0,
        "endreg": {
            "enabled": bool(endreg),
            "weight": float(args.endreg_weight),
            "frames": int(args.endreg_frames),
            "seed": int(args.endreg_seed),
            "rows_ended_naturally": ended_count if endreg else None,
            "final_edrift_p": (sum(r["edrift_p"] for r in win) / len(win)) if (endreg and win) else None,
            "final_edrift_n": (sum(r["edrift_n"] for r in win) / len(win)) if (endreg and win) else None,
        },
        "plus_label": prompts_meta.get("plus_label", ""),
        "minus_label": prompts_meta.get("minus_label", ""),
        "recommended_range": prompts_meta.get("recommended_range", [-2.0, 2.0]),
        "prompts_file": args.prompts_file,
        "early_stop": {
            "enabled": bool(args.early_stop),
            "fired": bool(early_fired),
            "window": args.early_window,
            "min_steps": args.min_steps,
            "cos": args.early_cos,
            "collapse": args.early_collapse,
            "perc": args.early_perc,
            "metrics": early_metrics,
        },
    }
    (save_dir / f"{args.name}_last.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    return last


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--name", default="gender-lm")
    p.add_argument("--model_dir", default=str(DEFAULT_MODEL))
    p.add_argument("--prompts_file", required=True)
    p.add_argument("--save_dir", default="/ml2/music/sliders-conceptmod/models/gender-lm-slider")
    p.add_argument("--rank", type=int, default=8)
    p.add_argument("--alpha", type=float, default=8.0)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--steps", type=int, default=800)
    p.add_argument("--save_every", type=int, default=200)
    p.add_argument("--device", type=int, default=0)
    p.add_argument(
        "--seed",
        type=int,
        default=7,
        help="LoRA init seed; pin it so runs differing only in --seed are comparable",
    )
    p.add_argument(
        "--lm_target",
        default="v9",
        choices=LM_RECIPES,
        help="live target recipe (default v9 = 2-D leak-0 geometry). v9: --symmetric "
        "polarity, then project the odd teacher onto --slider_positive/--slider_negative "
        "and hold the orthogonal residual (κ=0). hub: published leakage_floor blend-back "
        "(still leaks). symmetric / faithful: old poles without projection",
    )
    p.add_argument(
        "--symmetric",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="polarity step inside v9/hub/symmetric: tgt(±1) sit opposite around neu. "
        "v9 then projects that odd teacher onto the declared slider axis",
    )
    p.add_argument(
        "--slider_positive",
        default=None,
        help="declared + slider caption encoded once as û (not a training row's pos). "
        "Required for --lm_target v9 unless YAML slider_positive is set. "
        "Do not omit this and expect (pos-neg) to be the axis — that is the leak",
    )
    p.add_argument(
        "--slider_negative",
        default=None,
        help="declared − slider caption; pair with --slider_positive",
    )
    p.add_argument(
        "--hold_weight",
        type=float,
        default=None,
        help="weight on ||(h(±1)−h0)_⊥û||² (default 1.0 for v9, 0 otherwise). "
        "The live graph has no hold embedding; this is the 2-D orthogonal residual",
    )
    p.add_argument(
        "--leakage_floor",
        type=float,
        default=None,
        help="Hub-only even blend-back floor (default −0.9 with --lm_target hub). "
        "Does not change the odd teacher; do not make this the default",
    )
    p.add_argument(
        "--anchor_weight",
        type=float,
        default=None,
        help="Hub-only weight on the κ-blend anchors (default 0.3 with --lm_target hub)",
    )
    p.add_argument(
        "--anchor_autocal",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="autocal κ from --leakage_floor (Hub recipe)",
    )
    p.add_argument(
        "--common_beta",
        type=float,
        default=0.0,
        help="blend back beta*(midpoint - neutral) on symmetric/hub targets; ignored by v9 (κ=0)",
    )
    p.add_argument(
        "--target_scale",
        type=float,
        default=1.0,
        help="train the symmetric unit against target_scale x the pole axis: a cooler-trained "
        "unit is a different delta shape, not a dial-down of the same one (relabeling the "
        "render scale is score-invariant; this is not that)",
    )
    p.add_argument(
        "--planreg_weight",
        type=float,
        default=0.0,
        help="penalize hidden-state drift along the teacher-forced composition (frame positions "
        "only, both poles): keeps the sampled arrangement near the base song while the slider "
        "still moves the prompt conditioning. Requires --endreg_weight > 0",
    )
    p.add_argument("--plus_label", default=None, help="override sidecar plus_label")
    p.add_argument("--minus_label", default=None, help="override sidecar minus_label")
    p.add_argument(
        "--endreg_weight",
        type=float,
        default=1.0,
        help="audio-end regularizer weight: keep log-odds of <|audio_end|> vs the semantic band "
        "unchanged along a base-model composition (0 disables; sliders trained without it "
        "measurably suppress endings and truncate at the duration cap)",
    )
    p.add_argument(
        "--endreg_frames",
        type=int,
        default=250,
        help="pre-roll length in audio frames (25/s) for the end regularizer; composed once per "
        "row with the base model and cached in --endreg_cache",
    )
    p.add_argument("--endreg_seed", type=int, default=7, help="pre-roll sampling seed (row index is added)")
    p.add_argument(
        "--endreg_cache",
        default=str(_REPO_ROOT / "cache" / "endreg"),
        help="directory for cached pre-rolls, keyed on model, prompt, frames and seed",
    )
    p.add_argument(
        "--early_stop",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="stop when a rolling window matches c+/c-/collapse/perc (replayed on all v3 LM runs)",
    )
    p.add_argument("--early_window", type=int, default=50)
    p.add_argument("--min_steps", type=int, default=100, help="never stop before this many steps")
    p.add_argument("--early_cos", type=float, default=0.97, help="min mean c+ and c- in the window")
    p.add_argument("--early_collapse", type=float, default=-0.95, help="max mean collapse (more negative is better)")
    p.add_argument("--early_perc", type=float, default=0.20, help="max mean pperc/nperc in the window")
    args = p.parse_args(argv)
    if args.steps < 1:
        p.error("--steps must be >= 1")
    return args


def _early_stop_hit(
    history: list[dict], window: int, min_cos: float, max_collapse: float, max_perc: float
) -> bool:
    if window <= 0 or len(history) < window:
        return False
    chunk = history[-window:]
    cos_pos = sum(r["cos_pos"] for r in chunk) / window
    cos_neg = sum(r["cos_neg"] for r in chunk) / window
    collapse = sum(r["collapse"] for r in chunk) / window
    pperc = sum(r["pperc"] for r in chunk) / window
    nperc = sum(r["nperc"] for r in chunk) / window
    return (
        cos_pos > min_cos
        and cos_neg > min_cos
        and collapse < max_collapse
        and pperc < max_perc
        and nperc < max_perc
    )


if __name__ == "__main__":
    train(parse_args())
