"""Notrigger-style slider on MiniMax Music 3's language model.

Vocal identity is chosen in the AR stage, not the flow transformer. This trains
LoRA on Qwen3Attention so a neutral caption + LoRA scale moves the prompt
hidden state toward a female or male caption.

Default ``--lm_target v9`` is full pair-odd + hold-on-ê: ``--symmetric``
polarity, ``a = ½(h+−h−)``, ``t± = h0 ± a``, κ = 0. Short
``slider_positive`` is a name / probe, not the teacher — do not replace
``a`` with ``(a·û)û``. If YAML/CLI declares a leak pair
(``leak_positive`` / ``leak_negative``), penalize ``(h(±1)−h0) · ê``.
If no ê is declared (clean pair, or ``attributes`` already pin the
unused axis), hold is 0. Gender stays on this default: omit leak_*,
hold 0. Leaky axes use ``--lm_target pair_odd_sub_e``: teacher is
pair-odd minus ``ê_⊥ = ê−(ê·û)û`` (the λ→∞ hold limit, no stiffness).
ê must be leftover unused (genre + BPM / mix), not a slider synonym.
``--lm_target faithful_sub_e`` is the same leftover odd on the real
poles (midpoint stays ½(h++h−)); ``--lm_target faithful_sub_e_if_unused``
subtracts leftover ê only when unused. ``--lm_target faithful_guard_e``
subtracts leftover ê only while the cleaned target stays nearer its
own caption than the pair midpoint. ``--lm_target faithful_plus``
trains the + pole only (leftover-gated ``h+``; no minus MSE).
``--lm_target faithful_plus_neu`` is UNI: student +1 fits raw ``h+``
(never leftover-gated) and student scale 0 fits ``h0``. Last-hidden
MSE only. No minus MSE, no minus endreg, early-stop on c+/p% only.
Last real token must be ``<|audio_start|>``. Infer with the yaml
neutral caption + LoRA (not the + caption).
``--lm_target faithful_plus_neu_prefix`` also holds the +1 prefix
hidden to encode(neu) (yaml lyrics), not encode(pos). That pins
Vocal Details too. ``--lm_target faithful_plus_neu_lyric`` holds
only the yaml ``lyrics`` token span so gender / other prefix
concepts can still move.
``--lm_target faithful_plus_neu_roles`` splits the prefix by role:
yaml lyrics → encode(neu), Vocal Details / caption → encode(pos),
last → raw ``h+``, scale 0 → ``h0``. Fail closed if a required span
is missing. Last real token must be ``<|audio_start|>``.
``--pole_mode semantic_kl`` is
next-token KL on the semantic band. ``--pole_mode semantic_kl_null``
(aliases ``semantic_kl_plus_hidden``, ``semantic_kl_pin``) adds hidden
MSE on ``ker(lm_head)``. ``--pole_mode hidden_kl`` is full hidden MSE
plus a 0.001× semantic KL. ``--pole_mode dual_band`` is semantic KL
plus hidden MSE on the centered-readout blind band. None of those is
the default.

Old short-û project+hold is ``--lm_target v9_project`` (slider-level
gate) or ``--lm_target v9_always``. Published Hub floor is
``--lm_target hub`` and still leaks unused attr. See
``docs/lm-live-cells.md``.

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
    DUAL_BAND_WEIGHT,
    LEAK_HOLD_WEIGHT,
    SLIDER_ALIGN_MIN,
    UNUSED_E_OVERLAP_MAX,
    lm_anchor_kappa,
    lm_anchor_targets,
    lm_axis_hold,
    lm_blend_guard,
    lm_blind_projector,
    lm_blind_residual,
    lm_dual_band_pole_loss,
    EVEN_BLEND_SCALE,
    lm_e_overlap_a,
    lm_e_unused_decision,
    lm_even_leftover_dir,
    lm_faithful_gate_odd_sub_even,
    lm_faithful_guard_e,
    lm_faithful_plus,
    lm_faithful_plus_neu,
    lm_faithful_plus_neu_lyric,
    lm_faithful_plus_neu_prefix,
    lm_faithful_plus_neu_roles,
    lm_gather_span,
    lm_plus_neu_prefix_loss,
    lm_plus_neu_roles_loss,
    lm_role_span_bounds,
    lm_span_mse,
    RoleSpanError,
    lm_faithful_sub_e,
    lm_faithful_sub_e_if_unused,
    lm_plus_loss,
    lm_plus_neu_loss,
    lm_hold_dir,
    lm_hidden_targets,
    lm_next_token_logits,
    lm_odd_align,
    lm_ortho_hold,
    lm_pair_odd_sub_e,
    lm_project_decisions,
    lm_project_odd_axis,
    lm_readout_null_basis,
    lm_semantic_kl,
    lm_semantic_null_pole_loss,
    lm_semantic_pole_loss,
    lm_slider_loss,
)

DEFAULT_MODEL = Path("/ml2/music/models/MiniMax-Music3")
LM_RECIPES = (
    "v9",
    "pair_odd_sub_e",
    "v9_project",
    "v9_always",
    "hub",
    "symmetric",
    "faithful",
    "faithful_sub_e",
    "faithful_sub_e_if_unused",
    "faithful_guard_e",
    "faithful_even_blend",
    "faithful_plus",
    "faithful_plus_neu",
    "faithful_plus_neu_prefix",
    "faithful_plus_neu_lyric",
    "faithful_plus_neu_roles",
)
# Canonical live pole modes. ``semantic_kl_plus_hidden`` (#32) and
# ``semantic_kl_pin`` (#29) are aliases of ``semantic_kl_null`` (#33) —
# one hybrid, not three losses. ``unrolled_kl`` is fixture-only and is
# not a live flag.
POLE_MODE_ALIASES = {
    "semantic_kl_plus_hidden": "semantic_kl_null",
    "semantic_kl_pin": "semantic_kl_null",
}
CANONICAL_POLE_MODES = (
    "hidden",
    "semantic_kl",
    "semantic_kl_null",
    "hidden_kl",
    "dual_band",
)
POLE_MODES = CANONICAL_POLE_MODES + tuple(POLE_MODE_ALIASES)
HIDDEN_KL_WEIGHT = 1e-3
NEEDS_READOUT = frozenset(
    {"semantic_kl", "semantic_kl_null", "hidden_kl", "dual_band"}
)
PROJECT_RECIPES = frozenset({"v9_project", "v9_always"})
V9_RECIPES = frozenset({"v9", "v9_project", "v9_always"})
SUB_E_RECIPES = frozenset({"pair_odd_sub_e", "faithful_sub_e", "faithful_guard_e"})
GATED_SUB_E_RECIPES = frozenset(
    {"faithful_sub_e_if_unused", "faithful_even_blend", "faithful_plus"}
)
EVEN_BLEND_RECIPES = frozenset({"faithful_even_blend"})
PLUS_ONLY_RECIPES = frozenset({"faithful_plus"})
PLUS_NEU_RECIPES = frozenset(
    {
        "faithful_plus_neu",
        "faithful_plus_neu_prefix",
        "faithful_plus_neu_lyric",
        "faithful_plus_neu_roles",
    }
)
PLUS_NEU_PREFIX_RECIPES = frozenset({"faithful_plus_neu_prefix"})
PLUS_NEU_LYRIC_RECIPES = frozenset({"faithful_plus_neu_lyric"})
PLUS_NEU_HOLD_RECIPES = PLUS_NEU_PREFIX_RECIPES | PLUS_NEU_LYRIC_RECIPES
PLUS_NEU_ROLES_RECIPES = frozenset({"faithful_plus_neu_roles"})
PAIR_ODD_RECIPES = frozenset({"pair_odd_sub_e"})
TARGET_REPLACE = ["Qwen3Attention"]

# Row fields this trainer actually consumes (`attributes` is expanded away by
# _expand_attributes). Anything else in the YAML - `action`, `guidance_scale`,
# `batch_size`, `unconditional` - is only read by the transformer trainer.
# Declared slider / leak axes live at YAML top-level
# (slider_positive / slider_negative, leak_positive / leak_negative)
# or on the CLI; they are not per-row fields and are never inferred
# from a row's (pos-neg).
_USED_ROW_KEYS = frozenset({"target", "positive", "negative", "neutral", "lyrics", "attributes"})


def resolve_lm_recipe(*, lm_target: str, symmetric: bool) -> str:
    """Live trainer recipe. Default ``v9`` is full pair-odd + hold-on-ê.

    ``--symmetric`` is the polarity step inside ``v9`` / ``pair_odd_sub_e`` /
    ``v9_project`` / ``v9_always`` / ``hub`` / ``symmetric``. It is not a
    second loss. ``pair_odd_sub_e`` is the leaky-axis teacher (pair-odd
    minus ê_⊥); gender stays ``v9``. ``faithful_sub_e`` is ê-cleaned
    real poles (not a polarity step). ``faithful_sub_e_if_unused``
    subtracts leftover ê only when ``|ê̂_⊥ · â|`` is below the unused
    floor; otherwise it keeps the raw poles. ``faithful_guard_e``
    subtracts leftover ê only while the cleaned target stays nearer
    its own caption than the pair midpoint. ``faithful_even_blend``
    leftover-gates the odd part and subtracts half the leak-pair even
    leftover; not the default. ``faithful_plus`` leftover-gates the +
    caption only and drops minus MSE; not the default.
    ``faithful_plus_neu`` is UNI: raw ``h+`` at +1 and ``h0`` at scale 0;
    last-hidden MSE only; no leftover-gate, no minus MSE; not the default.
    ``faithful_plus_neu_prefix`` is UNI plus a prefix hold: +1 prefix
    hidden fits encode(neu) prefix (yaml lyrics), not encode(pos).
    ``faithful_plus_neu_lyric`` is UNI plus a lyric-token hold: +1
    yaml ``lyrics`` tokens fit encode(neu) lyrics. Vocal Details is
    not held.
    ``faithful_plus_neu_roles`` is UNI plus a role split: lyrics →
    encode(neu), Vocal Details / caption → encode(pos).
    """
    recipe = str(lm_target).strip().lower()
    if recipe not in LM_RECIPES:
        raise ValueError(f"lm_target must be one of {LM_RECIPES}, got {lm_target!r}")
    if recipe in V9_RECIPES | PAIR_ODD_RECIPES and not symmetric:
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


def resolve_pole_mode(pole_mode: str) -> str:
    """Resolve a live pole mode, folding the hybrid aliases onto one loss.

    ``hidden`` (default) is hidden MSE. ``semantic_kl`` is semantic-band
    KL. ``semantic_kl_null`` is that KL plus hidden MSE on ``ker(lm_head)``;
    ``semantic_kl_plus_hidden`` and ``semantic_kl_pin`` resolve here so
    old flags do not fork. ``hidden_kl`` is full hidden MSE plus a tiny
    semantic KL. ``dual_band`` is semantic KL plus hidden MSE on the
    centered-readout blind band (``lm_dual_band_pole_loss``). None of
    the alternatives is the default.
    """
    mode = str(pole_mode).strip().lower()
    if mode not in POLE_MODES:
        raise ValueError(f"pole_mode must be one of {POLE_MODES}, got {pole_mode!r}")
    return POLE_MODE_ALIASES.get(mode, mode)


def resolve_v9_gate(
    *,
    recipe: str,
    project_align_min: float | None,
    project_align_scope: str | None,
) -> tuple[float | None, str]:
    """Project-path gate only. Default ``v9`` does not project.

    ``v9_project`` is slider-level 0.50. ``v9_always`` never gates.
    """
    if recipe == "v9_always":
        return None, "row"
    if recipe == "v9_project":
        floor = SLIDER_ALIGN_MIN if project_align_min is None else float(project_align_min)
        scope = (project_align_scope or "slider").strip().lower()
        if scope not in ("slider", "row"):
            raise ValueError(f"project_align_scope must be slider or row, got {project_align_scope!r}")
        return floor, scope
    return project_align_min, (project_align_scope or "row")


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


def resolve_leak_axis_captions(
    *,
    leak_positive: str | None,
    leak_negative: str | None,
    prompts_meta: dict,
) -> tuple[str, str] | None:
    """Declared unused leak pair ê. CLI wins over YAML.

    YAML names: ``leak_positive`` / ``leak_negative``, or ``leak: [pos, neg]``.
    ``attributes`` is caption pinning (makes ``a`` clean) — not this axis.
    Never infer ê from short ``slider_positive`` or a row's (pos−neg).
    """
    listed = prompts_meta.get("leak")
    list_pos = list_neg = ""
    if isinstance(listed, (list, tuple)) and len(listed) == 2:
        list_pos, list_neg = str(listed[0] or ""), str(listed[1] or "")
    elif listed not in (None, "", []):
        raise ValueError("yaml leak must be [leak_positive, leak_negative]")
    pos = (leak_positive or prompts_meta.get("leak_positive") or list_pos or "").strip()
    neg = (leak_negative or prompts_meta.get("leak_negative") or list_neg or "").strip()
    if pos and neg:
        return pos, neg
    if pos or neg:
        raise ValueError("declared leak axis needs both leak_positive and leak_negative")
    return None


def resolve_lm_loss_weights(
    recipe: str,
    *,
    hold_weight: float | None,
    anchor_weight: float | None,
    leak_declared: bool = False,
) -> tuple[float, float]:
    if recipe == "v9":
        hold = LEAK_HOLD_WEIGHT if leak_declared else 0.0
    elif recipe in PROJECT_RECIPES:
        hold = 1.0
    else:
        hold = 0.0
    if recipe in SUB_E_RECIPES:
        hold = 0.0
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
    leak_dir: torch.Tensor | None = None,
    symmetric: bool = True,
    target_scale: float = 1.0,
    common_beta: float = 0.0,
    leakage_floor: float | None = None,
    anchor_autocal: bool = True,
    project_align_min: float | None = None,
    should_project: bool | None = None,
    e_unused: bool | None = None,
    even_dir: torch.Tensor | None = None,
    even_scale: float = EVEN_BLEND_SCALE,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
    """Pole targets for the live LM trainer. Delegates to ``slider_targets``.

    Returns ``(tgt_plus, tgt_minus, anchor_plus, anchor_minus)``.
    Anchors are set only for the published Hub floor recipe.

    ``should_project`` is the slider-level (or per-row) decision already
    resolved by the caller for ``v9_project`` / ``v9_always``. Default
    ``v9`` never projects: teacher is the full pair-odd. When omitted
    on the project path, ``project_align_min is None`` always-projects
    and a set floor is the per-row gate. The caller must drop the
    orthogonal hold when this falls back — holding ⊥ the rejected û
    still eats the pair.
    """
    if recipe == "v9":
        plus, minus = lm_hidden_targets(
            pos, neg, neu, target_mode="symmetric", target_scale=target_scale
        )
        return plus, minus, None, None
    if recipe == "pair_odd_sub_e":
        if leak_dir is None:
            raise ValueError("lm_target=pair_odd_sub_e requires a declared leak_dir")
        if slider_dir is None:
            raise ValueError(
                "lm_target=pair_odd_sub_e requires a declared slider_dir "
                "(ê_⊥ = ê−(ê·û)û; do not subtract raw ê)"
            )
        plus, minus = lm_pair_odd_sub_e(
            pos, neg, neu, leak_dir, slider_dir=slider_dir, target_scale=target_scale
        )
        return plus, minus, None, None
    if recipe == "faithful_sub_e":
        if leak_dir is None:
            raise ValueError("lm_target=faithful_sub_e requires a declared leak_dir")
        if slider_dir is None:
            raise ValueError(
                "lm_target=faithful_sub_e requires a declared slider_dir "
                "(ê_⊥ = ê−(ê·û)û; do not subtract raw ê)"
            )
        plus, minus = lm_faithful_sub_e(
            pos, neg, neu, leak_dir, slider_dir=slider_dir, target_scale=target_scale
        )
        return plus, minus, None, None
    if recipe == "faithful_sub_e_if_unused":
        plus, minus = lm_faithful_sub_e_if_unused(
            pos,
            neg,
            neu,
            leak_dir,
            slider_dir=slider_dir,
            target_scale=target_scale,
            unused=e_unused,
        )
        return plus, minus, None, None
    if recipe == "faithful_guard_e":
        if leak_dir is None:
            # Nothing declared to subtract, so the guard has nothing to
            # decide and the teacher is the caption. That is what lets one
            # recipe name cover gender-v4 and energy-v4.
            return pos, neg, None, None
        if slider_dir is None:
            raise ValueError(
                "lm_target=faithful_guard_e requires a declared slider_dir "
                "(ê_⊥ = ê−(ê·û)û; do not subtract raw ê)"
            )
        plus, minus = lm_faithful_guard_e(
            pos, neg, neu, leak_dir, slider_dir=slider_dir, target_scale=target_scale
        )
        return plus, minus, None, None
    if recipe == "faithful_even_blend":
        plus, minus = lm_faithful_gate_odd_sub_even(
            pos,
            neg,
            neu,
            leak_dir,
            slider_dir=slider_dir,
            even_dir=even_dir,
            unused=e_unused,
            target_scale=target_scale,
            scale=float(even_scale),
        )
        return plus, minus, None, None
    if recipe == "faithful_plus":
        plus = lm_faithful_plus(
            pos,
            neg,
            neu,
            leak_dir,
            slider_dir=slider_dir,
            target_scale=target_scale,
            unused=e_unused,
        )
        # Minus is not a teacher. Leftover-gated minus is only a canary
        # reference for logs so −1 can be compared to the unused caption.
        _plus, minus = lm_faithful_sub_e_if_unused(
            pos,
            neg,
            neu,
            leak_dir,
            slider_dir=slider_dir,
            target_scale=target_scale,
            unused=e_unused,
        )
        return plus, minus, None, None
    if recipe == "faithful_plus_neu":
        plus = lm_faithful_plus_neu(
            pos,
            neg,
            neu,
            leak_dir,
            slider_dir=slider_dir,
            target_scale=target_scale,
        )
        # Minus is not a teacher. Raw neg is only a canary reference.
        return plus, neg, None, None
    if recipe == "faithful_plus_neu_prefix":
        plus = lm_faithful_plus_neu_prefix(
            pos,
            neg,
            neu,
            leak_dir,
            slider_dir=slider_dir,
            target_scale=target_scale,
        )
        return plus, neg, None, None
    if recipe == "faithful_plus_neu_lyric":
        plus = lm_faithful_plus_neu_lyric(
            pos,
            neg,
            neu,
            leak_dir,
            slider_dir=slider_dir,
            target_scale=target_scale,
        )
        return plus, neg, None, None
    if recipe == "faithful_plus_neu_roles":
        plus = lm_faithful_plus_neu_roles(
            pos,
            neg,
            neu,
            leak_dir,
            slider_dir=slider_dir,
            target_scale=target_scale,
        )
        return plus, neg, None, None
    if recipe in PROJECT_RECIPES:
        if slider_dir is None:
            raise ValueError("lm_target=v9_project requires a declared slider_dir")
        if should_project is None:
            decisions = lm_project_decisions(
                [float(lm_odd_align(pos, neg, slider_dir))],
                project_align_min,
                "row",
            )
            should = decisions[0]
        else:
            should = bool(should_project)
        if should:
            plus, minus = lm_project_odd_axis(pos, neg, neu, slider_dir, target_scale=target_scale)
        else:
            plus, minus = lm_hidden_targets(
                pos, neg, neu, target_mode="symmetric", target_scale=target_scale
            )
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
    leak_dir: torch.Tensor | None = None,
    hold_weight: float = 0.0,
    anchor_plus: torch.Tensor | None = None,
    anchor_minus: torch.Tensor | None = None,
    anchor_weight: float = 0.0,
    pole_mode: str = "hidden",
    readout: torch.Tensor | None = None,
    null_basis: torch.Tensor | None = None,
    blind_projector: torch.Tensor | None = None,
    blind_weight: float = DUAL_BAND_WEIGHT,
    plus_only: bool = False,
    plus_neu: bool = False,
    pred_zero: torch.Tensor | None = None,
    tgt_zero: torch.Tensor | None = None,
    plus_neu_prefix: bool = False,
    pred_plus_prefix: torch.Tensor | None = None,
    tgt_neu_prefix: torch.Tensor | None = None,
    prefix_mask: torch.Tensor | None = None,
    plus_neu_roles: bool = False,
    pred_lyric: torch.Tensor | None = None,
    tgt_neu_lyric: torch.Tensor | None = None,
    pred_concept: torch.Tensor | None = None,
    tgt_pos_concept: torch.Tensor | None = None,
) -> torch.Tensor:
    """Live pole loss: ``lm_slider_loss`` plus optional hold.

    Default v9 holds leftover unused ``ê_⊥ = ê − (ê·û)û`` (``lm_axis_hold``).
    Raw ê that overlaps the slider is not held — that punches û.
    The project path holds the residual orthogonal to ``slider_dir``.
    The live graph has no hold embedding either way.

    ``pole_mode=hidden`` (default) is hidden MSE. ``semantic_kl`` is
    ``lm_semantic_pole_loss`` on ``lm_next_token_logits`` of the
    semantic-band readout. ``semantic_kl_null`` (and its aliases) adds
    hidden MSE on ``ker(readout)``. ``hidden_kl`` keeps hidden MSE as
    the primary loss and adds a 0.001× semantic KL. ``dual_band`` is
    that KL plus hidden MSE on ``blind_projector``, the band the
    centered readout cannot see. Not a second KL family — these are
    the already-scored race recipes.
    ``plus_only`` (``faithful_plus``) drops every minus term.
    ``plus_neu`` (``faithful_plus_neu``) is ``MSE(+) + MSE(0)`` only.
    ``plus_neu_prefix`` adds ``MSE(prefix + → encode(neu) prefix)``.
    Lyric-hold uses the same flag with a yaml-lyrics mask.
    ``plus_neu_roles`` adds lyric → encode(neu) and concept → encode(pos).
    """
    if plus_only and plus_neu:
        raise ValueError("plus_only and plus_neu are mutually exclusive")
    if plus_neu_prefix and plus_neu_roles:
        raise ValueError("plus_neu_prefix and plus_neu_roles are mutually exclusive")
    if plus_neu:
        if pred_zero is None or tgt_zero is None:
            raise ValueError("plus_neu requires pred_zero and tgt_zero")
        mode = resolve_pole_mode(pole_mode)
        if mode in NEEDS_READOUT and readout is None:
            raise ValueError(f"pole_mode={mode} requires a semantic readout")
        if plus_neu_roles:
            if (
                pred_lyric is None
                or tgt_neu_lyric is None
                or pred_concept is None
                or tgt_pos_concept is None
            ):
                raise ValueError("plus_neu_roles requires lyric and concept spans")
            neu_term = lm_plus_neu_roles_loss(
                pred_plus,
                tgt_plus,
                pred_zero,
                tgt_zero,
                pred_lyric,
                tgt_neu_lyric,
                pred_concept,
                tgt_pos_concept,
            )
        elif plus_neu_prefix:
            if pred_plus_prefix is None or tgt_neu_prefix is None:
                raise ValueError("plus_neu_prefix requires pred_plus_prefix and tgt_neu_prefix")
            if prefix_mask is None:
                neu_term = lm_plus_neu_prefix_loss(
                    pred_plus,
                    tgt_plus,
                    pred_zero,
                    tgt_zero,
                    pred_plus_prefix,
                    tgt_neu_prefix,
                )
            else:
                neu_term = lm_plus_neu_loss(pred_plus, tgt_plus, pred_zero, tgt_zero)
                neu_term = neu_term + _masked_hidden_mse(
                    pred_plus_prefix, tgt_neu_prefix, prefix_mask
                )
        else:
            neu_term = lm_plus_neu_loss(pred_plus, tgt_plus, pred_zero, tgt_zero)
        if mode == "hidden":
            return neu_term
        # Other pole modes keep UNI's MSE(0) and replace only the + term.
        plus_term = lm_train_loss(
            pred_plus,
            pred_minus,
            tgt_plus,
            tgt_minus,
            neu=neu,
            slider_dir=slider_dir,
            leak_dir=leak_dir,
            hold_weight=0.0,
            pole_mode=pole_mode,
            readout=readout,
            null_basis=null_basis,
            blind_projector=blind_projector,
            blind_weight=blind_weight,
            plus_only=True,
        )
        extra = plus_term + F.mse_loss(pred_zero, tgt_zero)
        if plus_neu_roles:
            extra = (
                extra
                + lm_span_mse(pred_lyric, tgt_neu_lyric)
                + lm_span_mse(pred_concept, tgt_pos_concept)
            )
        elif plus_neu_prefix:
            if prefix_mask is None:
                extra = extra + F.mse_loss(pred_plus_prefix, tgt_neu_prefix)
            else:
                extra = extra + _masked_hidden_mse(
                    pred_plus_prefix, tgt_neu_prefix, prefix_mask
                )
        return extra
    if plus_only:
        mode = resolve_pole_mode(pole_mode)
        if mode in NEEDS_READOUT and readout is None:
            raise ValueError(f"pole_mode={mode} requires a semantic readout")
        if mode == "hidden":
            return lm_plus_loss(pred_plus, tgt_plus)
        if mode in ("semantic_kl", "semantic_kl_null", "hidden_kl", "dual_band"):
            head = readout.to(dtype=pred_plus.dtype)
            plus_kl = lm_semantic_kl(
                lm_next_token_logits(pred_plus, head),
                lm_next_token_logits(tgt_plus, head),
            )
            if mode == "semantic_kl":
                return plus_kl
            if mode == "hidden_kl":
                return lm_plus_loss(pred_plus, tgt_plus) + HIDDEN_KL_WEIGHT * plus_kl
            if mode == "semantic_kl_null":
                extra = pred_plus.new_tensor(0.0)
                if null_basis is not None:
                    delta = (pred_plus - tgt_plus).flatten().float()
                    extra = (null_basis.T @ delta).pow(2).mean()
                return plus_kl + extra
            blind = pred_plus.new_tensor(0.0)
            if blind_projector is not None:
                blind = lm_blind_residual(pred_plus - tgt_plus, blind_projector).pow(2).mean()
            return plus_kl + float(blind_weight) * blind
        return lm_plus_loss(pred_plus, tgt_plus)
    hold = None
    if float(hold_weight) > 0.0:
        if neu is None:
            raise ValueError("hold_weight>0 requires neu")
        if leak_dir is not None:
            hold_axis = lm_hold_dir(leak_dir, slider_dir=slider_dir, mode="slider")
            if hold_axis is not None:
                hold = lm_axis_hold(pred_plus, pred_minus, neu, hold_axis)
        elif slider_dir is not None:
            hold = lm_ortho_hold(pred_plus, pred_minus, neu, slider_dir)
        else:
            raise ValueError("hold_weight>0 requires a declared leak_dir or slider_dir")
    mode = resolve_pole_mode(pole_mode)
    if mode in NEEDS_READOUT and readout is None:
        raise ValueError(f"pole_mode={mode} requires a semantic readout")
    if mode == "dual_band":
        head = readout.to(dtype=pred_plus.dtype)
        return lm_dual_band_pole_loss(
            pred_plus,
            pred_minus,
            tgt_plus,
            tgt_minus,
            pred_plus_logits=lm_next_token_logits(pred_plus, head),
            pred_minus_logits=lm_next_token_logits(pred_minus, head),
            tgt_plus_logits=lm_next_token_logits(tgt_plus, head),
            tgt_minus_logits=lm_next_token_logits(tgt_minus, head),
            blind_projector=blind_projector,
            blind_weight=blind_weight,
            hold=hold,
            hold_weight=hold_weight,
        )
    if mode in ("semantic_kl", "semantic_kl_null"):
        head = readout.to(dtype=pred_plus.dtype)
        if mode == "semantic_kl_null":
            return lm_semantic_null_pole_loss(
                lm_next_token_logits(pred_plus, head),
                lm_next_token_logits(pred_minus, head),
                lm_next_token_logits(tgt_plus, head),
                lm_next_token_logits(tgt_minus, head),
                pred_plus,
                pred_minus,
                tgt_plus,
                tgt_minus,
                head,
                null_basis=null_basis,
                hold=hold,
                hold_weight=hold_weight,
            )
        return lm_semantic_pole_loss(
            lm_next_token_logits(pred_plus, head),
            lm_next_token_logits(pred_minus, head),
            lm_next_token_logits(tgt_plus, head),
            lm_next_token_logits(tgt_minus, head),
            hold=hold,
            hold_weight=hold_weight,
        )
    if mode == "hidden_kl":
        head = readout.to(dtype=pred_plus.dtype)
        semantic = lm_semantic_pole_loss(
            lm_next_token_logits(pred_plus, head),
            lm_next_token_logits(pred_minus, head),
            lm_next_token_logits(tgt_plus, head),
            lm_next_token_logits(tgt_minus, head),
            hold=hold,
            hold_weight=0.0,
        )
        hidden = lm_slider_loss(
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
        return hidden + HIDDEN_KL_WEIGHT * semantic
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
            "leak_positive": str(raw.get("leak_positive") or ""),
            "leak_negative": str(raw.get("leak_negative") or ""),
            "leak": raw.get("leak"),
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
def _last_real_index(attention_mask: torch.Tensor) -> torch.Tensor:
    """Index of the last real token per row (attention_mask sum − 1)."""
    return attention_mask.to(dtype=torch.long).sum(dim=1) - 1


def _gather_last_hidden(hidden: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """Hidden at the last real token. Respects padding; do not use shape-1."""
    lengths = _last_real_index(attention_mask)
    gather = lengths.view(-1, 1, 1).expand(-1, 1, hidden.size(-1))
    return hidden.gather(1, gather.clamp(min=0)).squeeze(1)


def _special_token_id(tokenizer, token: str):
    """Tokenizer id for a special token, or None if the tokenizer cannot say."""
    if tokenizer is None:
        return None
    convert = getattr(tokenizer, "convert_tokens_to_ids", None)
    if convert is None:
        return None
    tid = convert(token)
    if tid is None:
        return None
    try:
        tid = int(tid)
    except (TypeError, ValueError):
        return None
    if tid < 0:
        return None
    unk = getattr(tokenizer, "unk_token_id", None)
    if unk is not None:
        try:
            if tid == int(unk):
                return None
        except (TypeError, ValueError):
            pass
    return tid


def _audio_start_token_id(tokenizer):
    """Tokenizer id for ``<|audio_start|>``, or None if the tokenizer cannot say."""
    if tokenizer is None:
        return None
    convert = getattr(tokenizer, "convert_tokens_to_ids", None)
    if convert is None:
        return None
    tid = convert(_AUDIO_START)
    if tid is None:
        return None
    try:
        tid = int(tid)
    except (TypeError, ValueError):
        return None
    if tid < 0:
        return None
    return tid


def _lyric_token_mask(
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    tokenizer,
    lyrics: str,
    *,
    where: str,
) -> torch.Tensor:
    """Mask over yaml ``lyrics`` tokens only (between lyrics_start/end).

    Fail closed on empty lyrics or a span the tokenized prompt does not
    contain. Does not mark Vocal Details / caption / metadata tokens.
    """
    if not str(lyrics or "").strip():
        raise RuntimeError(
            f"{where}: empty lyrics; lyric-hold cannot locate a yaml lyrics span"
        )
    start_id = _special_token_id(tokenizer, _LYRICS_START)
    end_id = _special_token_id(tokenizer, _LYRICS_END)
    if start_id is None or end_id is None or start_id == end_id:
        raise RuntimeError(
            f"{where}: tokenizer cannot name {_LYRICS_START}/{_LYRICS_END}; "
            "lyric-hold cannot locate the yaml lyrics span"
        )
    mask = torch.zeros_like(attention_mask)
    for batch in range(input_ids.size(0)):
        start_pos = end_pos = None
        for index in range(input_ids.size(1)):
            if int(attention_mask[batch, index]) == 0:
                continue
            tid = int(input_ids[batch, index])
            if start_pos is None and tid == start_id:
                start_pos = index
            elif start_pos is not None and end_pos is None and tid == end_id:
                end_pos = index
                break
        if start_pos is None or end_pos is None or end_pos <= start_pos + 1:
            raise RuntimeError(
                f"{where}: yaml lyrics span not found between "
                f"{_LYRICS_START} and {_LYRICS_END}"
            )
        mask[batch, start_pos + 1 : end_pos] = attention_mask[batch, start_pos + 1 : end_pos]
    if int(mask.sum()) <= 0:
        raise RuntimeError(f"{where}: yaml lyrics span is empty after masking")
    return mask


def _assert_lyric_span(
    neu_ids: torch.Tensor,
    neu_mask: torch.Tensor,
    pos_ids: torch.Tensor,
    pos_mask: torch.Tensor,
    tokenizer,
    lyrics: str,
    *,
    where: str,
) -> torch.Tensor:
    """Locate the yaml lyrics span on neu and pos. Return the neu mask."""
    neu_lyric = _lyric_token_mask(
        neu_ids, neu_mask, tokenizer, lyrics, where=f"{where} neu"
    )
    pos_lyric = _lyric_token_mask(
        pos_ids, pos_mask, tokenizer, lyrics, where=f"{where} pos"
    )
    neu_toks = neu_ids[0][neu_lyric[0].bool()]
    pos_toks = pos_ids[0][pos_lyric[0].bool()]
    if not torch.equal(neu_toks, pos_toks):
        raise RuntimeError(
            f"{where}: yaml lyrics tokens differ between neu and pos prompts"
        )
    return neu_lyric


def _assert_last_token_is_audio_start(
    input_ids: torch.Tensor, attention_mask: torch.Tensor, tokenizer, *, where: str
) -> None:
    """Fail closed if last-hidden MSE would pin the wrong continue-from token."""
    audio_id = _audio_start_token_id(tokenizer)
    if audio_id is None:
        return
    last_idx = _last_real_index(attention_mask).clamp(min=0)
    batch = torch.arange(input_ids.size(0), device=input_ids.device)
    last_ids = input_ids[batch, last_idx]
    expected = torch.full_like(last_ids, audio_id)
    if not torch.equal(last_ids, expected):
        raise RuntimeError(
            f"{where}: last real token is not {_AUDIO_START} "
            f"(got ids={last_ids.tolist()}, expected={audio_id}). "
            "Last-hidden MSE would pin the wrong continue-from token."
        )


def _special_token_id(tokenizer, name: str) -> int | None:
    """Tokenizer id for a Music 3 special token, or None if unknown."""
    if tokenizer is None:
        return None
    convert = getattr(tokenizer, "convert_tokens_to_ids", None)
    if convert is None:
        return None
    tid = convert(name)
    if tid is None:
        return None
    try:
        tid = int(tid)
    except (TypeError, ValueError):
        return None
    if tid < 0:
        return None
    return tid


def _heading_token_ids(tokenizer, text: str) -> list[int]:
    """Token ids for a structured-caption heading, or empty if unknown."""
    if tokenizer is None:
        return []
    try:
        out = tokenizer(text, add_special_tokens=False)
    except TypeError:
        out = tokenizer(text)
    ids = out["input_ids"] if isinstance(out, dict) else out
    if hasattr(ids, "tolist"):
        ids = ids.tolist()
    if not ids:
        return []
    if isinstance(ids[0], list):
        ids = ids[0]
    return [int(x) for x in ids]


def _locate_role_spans(tokenizer, input_ids, attention_mask, *, where: str) -> dict:
    """Lyric + concept spans. Fail closed if a required span is missing."""
    _assert_last_token_is_audio_start(input_ids, attention_mask, tokenizer, where=where)
    length = int(_last_real_index(attention_mask)[0].item()) + 1
    ids = input_ids[0, :length].tolist()
    headings = (
        "Vocal Details:",
        "Vocal Details:\n",
        "\nVocal Details:\n",
        "\nVocal Details:",
    )
    arrangements = (
        "Arrangement:",
        "Arrangement:\n",
        "\nArrangement:\n",
        "\nArrangement:",
    )
    vocal_ids: list[int] = []
    for heading in headings:
        vocal_ids = _heading_token_ids(tokenizer, heading)
        if vocal_ids:
            break
    arr_ids: list[int] = []
    for heading in arrangements:
        arr_ids = _heading_token_ids(tokenizer, heading)
        if arr_ids:
            break
    try:
        return lm_role_span_bounds(
            ids,
            lyrics_start_id=_special_token_id(tokenizer, _LYRICS_START),
            lyrics_end_id=_special_token_id(tokenizer, _LYRICS_END),
            caption_start_id=_special_token_id(tokenizer, _CAPTION_START),
            caption_end_id=_special_token_id(tokenizer, _CAPTION_END),
            vocal_details_ids=vocal_ids or None,
            arrangement_ids=arr_ids or None,
        )
    except RoleSpanError as exc:
        raise RuntimeError(f"{where}: {exc}") from exc


def _minus_pole_used(recipe: str) -> bool:
    """Plus+neu formulation is no minus — skip −1 encode, endreg, and early-stop."""
    return recipe not in PLUS_NEU_RECIPES


def _endreg_uses_minus(recipe: str) -> bool:
    return _minus_pole_used(recipe)


def _encode_static(lm, input_ids, attention_mask) -> torch.Tensor:
    hidden = lm.model(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
    # Last real token (the audio-start token) is what AR continues from.
    return _gather_last_hidden(hidden, attention_mask).float()


def _encode_train(lm, input_ids, attention_mask) -> torch.Tensor:
    hidden = lm.model(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
    return _gather_last_hidden(hidden, attention_mask).float()


def _encode_full(lm, input_ids, attention_mask) -> torch.Tensor:
    """Full last-hidden sequence. Prefix-hold reads every token except last."""
    hidden = lm.model(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
    return hidden.float()


def _split_prefix_last(
    hidden: torch.Tensor, attention_mask: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Last real token (continue) and a mask over the lyric / caption prefix.

    Last real token is the audio-start continue-from token. Prefix is
    every earlier real token — including the yaml lyrics.
    """
    lengths = _last_real_index(attention_mask)
    last = _gather_last_hidden(hidden, attention_mask)
    prefix_mask = attention_mask.clone()
    batch = torch.arange(hidden.size(0), device=hidden.device)
    prefix_mask[batch, lengths.clamp(min=0)] = 0
    return last.float(), hidden, prefix_mask


def _masked_hidden_mse(
    pred: torch.Tensor, tgt: torch.Tensor, mask: torch.Tensor
) -> torch.Tensor:
    """MSE over masked prefix positions. ``mask`` is [B, T] 1/0."""
    valid = mask.unsqueeze(-1).to(dtype=pred.dtype)
    denom = valid.sum().clamp_min(1.0)
    return ((pred - tgt).pow(2) * valid).sum() / denom


def _set_scale(network: LoRANetwork, scale: float) -> None:
    network.set_lora_slider(scale)
    for lora in network.unet_loras:
        lora.multiplier = float(scale)


def _semantic_readout(lm) -> torch.Tensor:
    """Semantic-code band of ``lm_head.weight`` — the live next-token readout.

    Same slice ``_frame_margins`` already multiplies out for the end margin.
    ``lm_next_token_logits`` / ``lm_semantic_pole_loss`` consume this.
    """
    from diffusers.modular_pipelines.minimax_music3.encoders import (
        _AUDIO_CODE_OFFSET,
        _SEMANTIC_VOCAB_SIZE,
    )

    return lm.lm_head.weight[_AUDIO_CODE_OFFSET : _AUDIO_CODE_OFFSET + _SEMANTIC_VOCAB_SIZE]


def _frame_margins(lm, hidden_tail: torch.Tensor) -> torch.Tensor:
    """End margin — logit(<|audio_end|>) - logsumexp(semantic-code band) — at
    each decode position. hidden_tail: [1, P, hidden]; returns float32 [P].
    Only the lm_head rows that decide stop-vs-continue are multiplied out."""
    from diffusers.modular_pipelines.minimax_music3.encoders import (
        _AUDIO_END_TOKEN_ID,
    )

    weight = lm.lm_head.weight
    hidden = hidden_tail[0]
    end_logit = (hidden @ weight[_AUDIO_END_TOKEN_ID]).float()
    sem_logits = (hidden @ _semantic_readout(lm).T).float()
    return end_logit - torch.logsumexp(sem_logits, dim=-1)


def _forward_teacher_forced(lm, prompt_embeds: torch.Tensor, frame_embeds: torch.Tensor | None):
    """One forward over [prompt; composed frames]. Returns the prompt-last
    hidden state (the slider target position — causality makes it identical to
    a prompt-only forward), the end margins at positions prompt-last..end, and
    the full hidden sequence for optional plan-drift regularization."""
    embeds = prompt_embeds if frame_embeds is None else torch.cat((prompt_embeds, frame_embeds), dim=1)
    mask = torch.ones(embeds.shape[:2], dtype=torch.long, device=embeds.device)
    hidden = lm.model(inputs_embeds=embeds, attention_mask=mask).last_hidden_state
    # Prompt has no padding here (ones mask), but still gather via the last-real
    # index so this cannot silently disagree with `_encode_*` if padding appears.
    prompt_mask = torch.ones(
        prompt_embeds.shape[:2], dtype=torch.long, device=prompt_embeds.device
    )
    prompt_last = _gather_last_hidden(hidden[:, : prompt_embeds.shape[1]], prompt_mask)
    last_idx = int(_last_real_index(prompt_mask)[0].item())
    margins = _frame_margins(lm, hidden[:, last_idx:])
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
    pole_mode = resolve_pole_mode(getattr(args, "pole_mode", "hidden"))
    axis_captions = resolve_slider_axis_captions(
        slider_positive=args.slider_positive,
        slider_negative=args.slider_negative,
        prompts_meta=prompts_meta,
    )
    leak_captions = resolve_leak_axis_captions(
        leak_positive=getattr(args, "leak_positive", None),
        leak_negative=getattr(args, "leak_negative", None),
        prompts_meta=prompts_meta,
    )
    hold_w, anchor_w = resolve_lm_loss_weights(
        recipe,
        hold_weight=args.hold_weight,
        anchor_weight=args.anchor_weight,
        leak_declared=leak_captions is not None,
    )
    if recipe in PROJECT_RECIPES and axis_captions is None:
        raise ValueError(
            "lm_target=v9_project / v9_always needs a declared slider axis: "
            "--slider_positive / --slider_negative, or YAML slider_positive / "
            "slider_negative. Do not silently use a row's (pos-neg) — that is "
            "the unused-attribute leak."
        )
    if recipe in SUB_E_RECIPES:
        if leak_captions is None and recipe != "faithful_guard_e":
            raise ValueError(
                f"lm_target={recipe} needs a declared leftover leak axis: "
                "--leak_positive / --leak_negative, or YAML leak_positive / "
                "leak_negative (or leak: [pos, neg]). ê is leftover unused, "
                "not a slider synonym."
            )
        if leak_captions is None:
            # ``faithful_guard_e`` is one recipe for both pair types: a yaml
            # with no leak_* is a guard with nothing to decide, and the
            # teacher is the caption. gender-v4 runs here unchanged.
            print(
                "lm_target=faithful_guard_e: no leak_* declared, so nothing to "
                "subtract and the teacher is the raw poles"
            )
        if axis_captions is None and leak_captions is not None:
            raise ValueError(
                f"lm_target={recipe} needs a declared slider axis so it "
                "can subtract ê_⊥ = ê−(ê·û)û, not raw ê: "
                "--slider_positive / --slider_negative, or YAML slider_positive "
                "/ slider_negative."
            )
    if recipe in GATED_SUB_E_RECIPES and leak_captions is not None and axis_captions is None:
        raise ValueError(
            f"lm_target={recipe} needs a declared slider axis so it can "
            "measure |ê̂_⊥ · â| and subtract ê_⊥ = ê−(ê·û)û, not raw ê: "
            "--slider_positive / --slider_negative, or YAML slider_positive "
            "/ slider_negative. Omit leak_* on a clean pair; the gate is then "
            "a no-op and the teacher is the raw poles."
        )
    if recipe == "v9" and hold_w > 0.0 and leak_captions is None and axis_captions is None:
        raise ValueError(
            "hold_weight>0 on --lm_target v9 needs a declared leak axis: "
            "--leak_positive / --leak_negative, or YAML leak_positive / "
            "leak_negative (or leak: [pos, neg]). Do not hold û_⊥ — that "
            "eats a clean pair."
        )
    if recipe in V9_RECIPES | SUB_E_RECIPES and float(args.common_beta) != 0.0:
        print(
            "note: lm_target=v9 / pair_odd_sub_e / faithful_sub_e is κ=0 "
            "(no even blend-back); ignoring --common_beta"
        )
    align_min, align_scope = resolve_v9_gate(
        recipe=recipe,
        project_align_min=getattr(args, "project_align_min", None),
        project_align_scope=getattr(args, "project_align_scope", None),
    )

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
    readout = _semantic_readout(lm) if pole_mode in NEEDS_READOUT else None
    null_basis = (
        lm_readout_null_basis(readout.float())
        if pole_mode == "semantic_kl_null" and readout is not None
        else None
    )
    blind_projector = None
    if pole_mode == "dual_band":
        blind_projector = lm_blind_projector(
            readout.float(), cut=float(args.blind_cut)
        )
        if blind_projector is not None:
            blind_projector = blind_projector.to(device=device, dtype=torch.float32)

    beta = float(args.common_beta)
    slider_dir = None
    leak_dir = None
    leak_pos_h = leak_neg_h = None
    axis_lyrics = str(rows[0].get("lyrics") or "")
    if axis_captions is not None:
        axis_pos_text = _assemble(axis_captions[0], axis_lyrics)
        axis_neg_text = _assemble(axis_captions[1], axis_lyrics)
        with torch.no_grad():
            axis_pos_ids, axis_pos_mask = _tokenize(tokenizer, axis_pos_text, device)
            axis_neg_ids, axis_neg_mask = _tokenize(tokenizer, axis_neg_text, device)
            _assert_last_token_is_audio_start(
                axis_pos_ids, axis_pos_mask, tokenizer, where="declared slider +"
            )
            _assert_last_token_is_audio_start(
                axis_neg_ids, axis_neg_mask, tokenizer, where="declared slider −"
            )
            axis_pos_h = _encode_static(lm, axis_pos_ids, axis_pos_mask)
            axis_neg_h = _encode_static(lm, axis_neg_ids, axis_neg_mask)
        slider_dir = axis_pos_h - axis_neg_h
        print(
            f"declared slider axis: {axis_captions[0]!r} / {axis_captions[1]!r} "
            f"||û||={slider_dir.norm().item():.3f} (encoded once, not a row's pos-neg)"
        )
    if leak_captions is not None:
        leak_pos_text = _assemble(leak_captions[0], axis_lyrics)
        leak_neg_text = _assemble(leak_captions[1], axis_lyrics)
        with torch.no_grad():
            leak_pos_ids, leak_pos_mask = _tokenize(tokenizer, leak_pos_text, device)
            leak_neg_ids, leak_neg_mask = _tokenize(tokenizer, leak_neg_text, device)
            _assert_last_token_is_audio_start(
                leak_pos_ids, leak_pos_mask, tokenizer, where="declared leak +"
            )
            _assert_last_token_is_audio_start(
                leak_neg_ids, leak_neg_mask, tokenizer, where="declared leak −"
            )
            leak_pos_h = _encode_static(lm, leak_pos_ids, leak_pos_mask)
            leak_neg_h = _encode_static(lm, leak_neg_ids, leak_neg_mask)
        leak_dir = leak_pos_h - leak_neg_h
        if recipe == "pair_odd_sub_e":
            hold_note = "teacher=pair_odd − ê_⊥, ê_⊥=ê−(ê·û)û; hold 0"
        elif recipe == "faithful_sub_e":
            hold_note = "teacher=real poles − odd ê_⊥, midpoint ½(h++h−); hold 0"
        elif recipe == "faithful_sub_e_if_unused":
            hold_note = "teacher=raw poles or ê-cleaned poles if |ê̂_⊥·â| < unused floor"
        elif recipe == "faithful_guard_e":
            hold_note = "teacher=ê-cleaned poles if blend guard admits, else raw poles"
        elif recipe == "faithful_even_blend":
            hold_note = (
                "teacher=leftover-gated odd + half leak-pair even leftover; hold 0"
            )
        elif recipe == "faithful_plus":
            hold_note = "teacher=+ caption leftover-gated; minus MSE off; hold 0"
        elif recipe == "faithful_plus_neu":
            hold_note = "teacher=raw + caption; scale 0 fits h0; minus MSE off; hold 0"
        elif recipe == "faithful_plus_neu_prefix":
            hold_note = (
                "teacher=raw + last hidden; +1 prefix → encode(neu); "
                "scale 0 fits h0; minus MSE off; hold 0"
            )
        elif recipe == "faithful_plus_neu_lyric":
            hold_note = (
                "teacher=raw + last hidden; +1 lyrics → encode(neu) lyrics; "
                "Vocal Details free; scale 0 fits h0; minus MSE off; hold 0"
            )
        elif recipe == "faithful_plus_neu_roles":
            hold_note = (
                "teacher=raw + last hidden; +1 lyrics → encode(neu); "
                "+1 Vocal Details → encode(pos); scale 0 fits h0; "
                "minus MSE off; hold 0"
            )
        else:
            hold_note = "hold (h(±1)−h0)·ê_⊥û, ê_⊥=ê−(ê·û)û; teacher stays pair-odd"
        print(
            f"declared leak axis ê: {leak_captions[0]!r} / {leak_captions[1]!r} "
            f"||ê||={leak_dir.norm().item():.3f} "
            f"({hold_note})"
        )
    elif recipe == "v9":
        print("note: no leak axis declared — v9 hold is off (teacher is full pair-odd)")

    encoded_rows = []
    for index, row in enumerate(rows):
        lyrics = str(row["lyrics"])
        texts = {
            "neutral": _assemble(str(row.get("neutral") or row["target"]), lyrics),
            "positive": _assemble(str(row["positive"]), lyrics),
            "negative": _assemble(str(row["negative"]), lyrics),
        }
        tokens = {name: _tokenize(tokenizer, text, device) for name, text in texts.items()}
        for name, (ids, mask) in tokens.items():
            _assert_last_token_is_audio_start(
                ids, mask, tokenizer, where=f"row {index} {name} caption"
            )
        with torch.no_grad():
            pos_tgt = _encode_static(lm, *tokens["positive"])
            neg_tgt = _encode_static(lm, *tokens["negative"])
            neu_ref = _encode_static(lm, *tokens["neutral"])
            neu_hidden = None
            neu_prefix_mask = None
            pos_hidden = None
            role_spans = None
            if recipe in PLUS_NEU_HOLD_RECIPES:
                neu_full = _encode_full(lm, *tokens["neutral"])
                if recipe in PLUS_NEU_LYRIC_RECIPES:
                    neu_prefix_mask = _assert_lyric_span(
                        tokens["neutral"][0],
                        tokens["neutral"][1],
                        tokens["positive"][0],
                        tokens["positive"][1],
                        tokenizer,
                        lyrics,
                        where=f"row {index}",
                    )
                    neu_hidden = neu_full
                else:
                    _last, neu_hidden, neu_prefix_mask = _split_prefix_last(
                        neu_full, tokens["neutral"][1]
                    )
                    del _last
            if recipe in PLUS_NEU_ROLES_RECIPES:
                neu_spans = _locate_role_spans(
                    tokenizer,
                    tokens["neutral"][0],
                    tokens["neutral"][1],
                    where=f"row {index} neutral caption",
                )
                pos_spans = _locate_role_spans(
                    tokenizer,
                    tokens["positive"][0],
                    tokens["positive"][1],
                    where=f"row {index} positive caption",
                )
                if neu_spans["lyric"][1] - neu_spans["lyric"][0] != (
                    pos_spans["lyric"][1] - pos_spans["lyric"][0]
                ):
                    raise RuntimeError(
                        f"row {index}: neu/pos lyric spans differ "
                        f"({neu_spans['lyric']} vs {pos_spans['lyric']}); "
                        "yaml lyrics must tokenize to the same length"
                    )
                neu_full = _encode_full(lm, *tokens["neutral"])
                pos_full = _encode_full(lm, *tokens["positive"])
                neu_hidden = neu_full
                pos_hidden = pos_full
                role_spans = {"neu": neu_spans, "pos": pos_spans}
        target_cos = F.cosine_similarity(pos_tgt - neu_ref, neg_tgt - neu_ref, dim=-1).mean().item()
        align = None
        if slider_dir is not None:
            align = float(lm_odd_align(pos_tgt, neg_tgt, slider_dir))
        encoded_rows.append(
            {
                "index": index,
                "tokens": tokens,
                "texts": texts,
                "pos_tgt": pos_tgt,
                "neg_tgt": neg_tgt,
                "neu_ref": neu_ref,
                "neu_hidden": neu_hidden,
                "neu_prefix_mask": neu_prefix_mask,
                "pos_hidden": pos_hidden,
                "role_spans": role_spans,
                "target_cos": target_cos,
                "align": align,
            }
        )

    aligns = [row["align"] if row["align"] is not None else 0.0 for row in encoded_rows]
    e_unused = None
    if recipe in GATED_SUB_E_RECIPES and leak_dir is not None:
        e_overlaps = [
            float(lm_e_overlap_a(row["pos_tgt"], row["neg_tgt"], leak_dir, slider_dir=slider_dir))
            for row in encoded_rows
        ]
        e_unused = lm_e_unused_decision(e_overlaps)
        mean_e = sum(e_overlaps) / len(e_overlaps)
        odd_note = (
            "unused → leftover-gate subtracts odd ê_⊥"
            if e_unused
            else "ê restates the pair → leftover-gate keeps odd poles"
        )
        if recipe == "faithful_even_blend":
            odd_note += "; even-blend still applies"
        elif recipe == "faithful_plus":
            odd_note += "; plus-only (no minus MSE)"
        print(
            f"leftover gate: mean |ê̂_⊥·â|={mean_e:.3f} floor={UNUSED_E_OVERLAP_MAX:g} "
            f"rows={[round(o, 3) for o in e_overlaps]} "
            f"{odd_note}"
        )
    if recipe in PROJECT_RECIPES and slider_dir is not None:
        decisions = lm_project_decisions(aligns, align_min, align_scope)
        mean_align = sum(aligns) / len(aligns)
        print(
            f"v9_project gate: scope={align_scope} floor={align_min} "
            f"mean |odd·û|/||odd||={mean_align:.3f} "
            f"rows={[round(a, 3) for a in aligns]} "
            f"project={decisions} "
            f"{'(mixed teacher)' if len(set(decisions)) > 1 else '(same teacher)'}"
        )
    else:
        decisions = [False] * len(encoded_rows)
        if recipe == "v9":
            print(
                f"v9: teacher=full pair-odd, hold_ê={hold_w if leak_dir is not None else 0.0} "
                f"(κ=0, no project onto short û)"
            )
        elif recipe == "pair_odd_sub_e":
            print(
                "pair_odd_sub_e: teacher=pair-odd − ê_⊥, hold_ê=0 "
                "(λ→∞ hold limit; no project onto short û)"
            )
        elif recipe == "faithful_sub_e":
            print(
                "faithful_sub_e: teacher=real poles − odd ê_⊥, hold_ê=0 "
                "(midpoint stays ½(h++h−); not t± = h0 ± a)"
            )
        elif recipe == "faithful_sub_e_if_unused":
            print(
                "faithful_sub_e_if_unused: subtract leftover ê only when "
                "|ê̂_⊥ · â| is below the unused floor; else raw poles"
            )
        elif recipe == "faithful_guard_e":
            print(
                "faithful_guard_e: subtract leftover ê only while the cleaned "
                "target stays nearer its caption than ½(h++h−); else raw poles"
            )
        elif recipe == "faithful_even_blend":
            print(
                "faithful_even_blend: leftover-gate odd ê, subtract "
                f"{float(args.even_blend_scale):g} of leak-pair even leftover "
                "(caption even stays; not t± = h0 ± a)"
            )
        elif recipe == "faithful_plus":
            print(
                "faithful_plus: teacher=+ caption leftover-gated; "
                "student +1 fits h+; no pair-odd, no h0 ± a, no minus MSE"
            )
        elif recipe == "faithful_plus_neu":
            print(
                "faithful_plus_neu: teacher=raw + caption (no leftover-gate); "
                "student +1 fits h+, student 0 fits h0; "
                "no pair-odd, no h0 ± a, no minus MSE, no minus endreg; "
                "early_stop on c+/p% only. Infer with the yaml neutral "
                "caption + LoRA — do not also swap in the + caption."
            )
        elif recipe == "faithful_plus_neu_prefix":
            print(
                "faithful_plus_neu_prefix: teacher=raw + last hidden "
                "(no leftover-gate); student +1 last fits h+, "
                "student +1 prefix fits encode(neu) prefix (yaml lyrics), "
                "student 0 fits h0; no pair-odd, no h0 ± a, no minus MSE, "
                "no minus endreg; early_stop on c+/p% only"
            )
        elif recipe == "faithful_plus_neu_lyric":
            print(
                "faithful_plus_neu_lyric: teacher=raw + last hidden "
                "(no leftover-gate); student +1 last fits h+, "
                "student +1 yaml lyrics fit encode(neu) lyrics, "
                "Vocal Details / metadata are not held, "
                "student 0 fits h0; no pair-odd, no h0 ± a, no minus MSE, "
                "no minus endreg; early_stop on c+/p% only"
            )
        elif recipe == "faithful_plus_neu_roles":
            print(
                "faithful_plus_neu_roles: teacher=raw + last hidden "
                "(no leftover-gate); student +1 last fits h+, "
                "student +1 lyrics fit encode(neu) lyrics, "
                "student +1 Vocal Details / caption fit encode(pos) "
                "same role (pooled if lengths differ), student 0 fits h0; "
                "fail closed if a required span is missing; "
                "no pair-odd, no h0 ± a, no minus MSE, no minus endreg; "
                "early_stop on c+/p% only"
            )
    if pole_mode == "semantic_kl":
        print("pole_mode=semantic_kl: next-token KL on the semantic band of lm_head")
    elif pole_mode == "semantic_kl_null":
        dim = 0 if null_basis is None else int(null_basis.shape[1])
        print(
            f"pole_mode=semantic_kl_null: semantic KL plus hidden MSE on "
            f"ker(lm_head) ({dim} dims the KL cannot see)"
        )
    elif pole_mode == "hidden_kl":
        print(
            "pole_mode=hidden_kl: hidden MSE plus 0.001× next-token semantic KL "
            "(real-pole hidden lock)"
        )
    elif pole_mode == "dual_band":
        width = 0 if blind_projector is None else int(round(float(blind_projector.trace())))
        print(
            f"pole_mode=dual_band: semantic-band KL + {args.blind_weight:g}·MSE on the "
            f"{width} blind dims (cut {args.blind_cut:g}, weight {args.blind_weight:g})"
        )
        if blind_projector is None:
            print(
                "  WARNING: the semantic band's row space fills the hidden width at "
                "this cut, so there is no blind band and this is exactly "
                "pole_mode=semantic_kl. Raise --blind_cut."
            )
    else:
        print("pole_mode=hidden: hidden MSE onto the chosen targets")

    row_data = []
    for encoded, should_project in zip(encoded_rows, decisions):
        index = encoded["index"]
        tokens = encoded["tokens"]
        texts = encoded["texts"]
        pos_tgt = encoded["pos_tgt"]
        neg_tgt = encoded["neg_tgt"]
        neu_ref = encoded["neu_ref"]
        target_cos = encoded["target_cos"]
        dropped = ""
        row_slider_dir = slider_dir
        row_leak_dir = leak_dir
        row_hold = hold_w if (recipe not in {"v9"} | SUB_E_RECIPES | GATED_SUB_E_RECIPES or leak_dir is not None) else 0.0
        if recipe in SUB_E_RECIPES | GATED_SUB_E_RECIPES | EVEN_BLEND_RECIPES:
            row_hold = 0.0
        if recipe == "v9":
            dropped = " teacher=odd"
            if row_leak_dir is not None:
                dropped += f" hold_ê={row_hold}"
            else:
                dropped += " hold_ê=0"
            if encoded["align"] is not None:
                dropped += f" odd·û/||odd||={encoded['align']:.3f} (probe)"
        elif recipe == "pair_odd_sub_e":
            dropped = " teacher=pair_odd_sub_e hold_ê=0"
            if encoded["align"] is not None:
                dropped += f" odd·û/||odd||={encoded['align']:.3f} (probe)"
        elif recipe == "faithful_sub_e":
            dropped = " teacher=faithful_sub_e hold_ê=0"
            if encoded["align"] is not None:
                dropped += f" odd·û/||odd||={encoded['align']:.3f} (probe)"
        elif recipe == "faithful_sub_e_if_unused":
            dropped = (
                " teacher=faithful_sub_e hold_ê=0"
                if e_unused
                else " teacher=faithful (ê restates pair or no leftover) hold_ê=0"
            )
            if encoded["align"] is not None:
                dropped += f" odd·û/||odd||={encoded['align']:.3f} (probe)"
        elif recipe == "faithful_guard_e":
            dropped = " teacher=faithful_guard_e hold_ê=0"
            if row_leak_dir is None:
                dropped += " guard=no-ê teacher=poles"
            else:
                sub_plus, sub_minus = lm_faithful_sub_e(
                    pos_tgt,
                    neg_tgt,
                    neu_ref,
                    row_leak_dir,
                    slider_dir=row_slider_dir,
                    target_scale=float(args.target_scale),
                )
                guard = lm_blend_guard(sub_plus, sub_minus, pos_tgt, neg_tgt)
                if guard["admissible"]:
                    dropped += (
                        f" guard=sub_e to_pole={guard['to_pole']:.3f} "
                        f"to_mid={guard['to_mid']:.3f}"
                    )
                else:
                    dropped += (
                        f" guard=REFUSED (poles) to_pole={guard['to_pole']:.3f} "
                        f"to_mid={guard['to_mid']:.3f}"
                    )
            if encoded["align"] is not None:
                dropped += f" odd·û/||odd||={encoded['align']:.3f} (probe)"
        elif recipe == "faithful_even_blend":
            dropped = (
                f" teacher=faithful_even_blend scale={float(args.even_blend_scale):g} "
                "hold_ê=0"
            )
            if encoded["align"] is not None:
                dropped += f" odd·û/||odd||={encoded['align']:.3f} (probe)"
        elif recipe == "faithful_plus":
            dropped = " teacher=faithful_plus hold_ê=0 minus_mse=off"
            if encoded["align"] is not None:
                dropped += f" odd·û/||odd||={encoded['align']:.3f} (probe)"
        elif recipe == "faithful_plus_neu":
            dropped = " teacher=faithful_plus_neu raw_h+ hold_ê=0 minus_mse=off neu_mse=on"
            if encoded["align"] is not None:
                dropped += f" odd·û/||odd||={encoded['align']:.3f} (probe)"
        elif recipe == "faithful_plus_neu_prefix":
            dropped = (
                " teacher=faithful_plus_neu_prefix raw_h+ prefix→encode(neu) "
                "hold_ê=0 minus_mse=off neu_mse=on"
            )
        elif recipe == "faithful_plus_neu_roles":
            source = (encoded.get("role_spans") or {}).get("neu", {}).get("source")
            dropped = (
                " teacher=faithful_plus_neu_roles raw_h+ "
                f"lyrics→encode(neu) concept→encode(pos)[{source}] "
                "hold_ê=0 minus_mse=off neu_mse=on"
            )
            if encoded["align"] is not None:
                dropped += f" odd·û/||odd||={encoded['align']:.3f} (probe)"
        elif recipe == "faithful_plus_neu_lyric":
            dropped = (
                " teacher=faithful_plus_neu_lyric raw_h+ lyrics→encode(neu) "
                "hold_ê=0 minus_mse=off neu_mse=on"
            )
            if encoded["align"] is not None:
                dropped += f" odd·û/||odd||={encoded['align']:.3f} (probe)"
        elif encoded["align"] is not None:
            dropped = f" odd·û/||odd||={encoded['align']:.3f}"
            if recipe in PROJECT_RECIPES and not should_project:
                dropped += (
                    f" < {align_scope} floor {align_min} "
                    "→ pair-symmetric (no project, no hold)"
                )
                row_slider_dir = None
                row_hold = 0.0
            elif recipe in PROJECT_RECIPES and should_project:
                dropped += f" ≥ {align_scope} floor {align_min} → project+hold"
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
            leak_dir=leak_dir,
            symmetric=args.symmetric,
            target_scale=float(args.target_scale),
            common_beta=beta,
            leakage_floor=args.leakage_floor,
            anchor_autocal=args.anchor_autocal,
            should_project=should_project if recipe in PROJECT_RECIPES else None,
            e_unused=e_unused,
            even_dir=(
                lm_even_leftover_dir(
                    leak_pos_h, leak_neg_h, neu_ref, slider_dir=slider_dir
                )
                if leak_pos_h is not None and leak_neg_h is not None
                else None
            ),
            even_scale=float(args.even_blend_scale),
        )
        row_data.append(
            {
                "tokens": tokens["neutral"],
                "text_neutral": texts["neutral"],
                "tgt_plus": tgt_plus,
                "tgt_minus": tgt_minus,
                "neu_ref": neu_ref,
                "neu_hidden": encoded.get("neu_hidden"),
                "neu_prefix_mask": encoded.get("neu_prefix_mask"),
                "pos_hidden": encoded.get("pos_hidden"),
                "role_spans": encoded.get("role_spans"),
                "slider_dir": row_slider_dir,
                "leak_dir": row_leak_dir,
                "hold_weight": row_hold,
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
            edrift_p = float((m_pos - data["base_margins"]).abs().mean().detach())
            pdrift_p = float(plan_pos.detach())

            if _minus_pole_used(recipe):
                _set_scale(network, -1.0)
                pred_neg, m_neg, hid_neg = _forward_teacher_forced(
                    lm, data["prompt_embeds"], data["frame_embeds"]
                )
                end_neg = F.mse_loss(m_neg, data["base_margins"])
                plan_neg = F.mse_loss(
                    hid_neg[:, data["prompt_embeds"].shape[1] :].float(), data["base_hidden"]
                )
                edrift_n = float((m_neg - data["base_margins"]).abs().mean().detach())
                pdrift_n = float(plan_neg.detach())
            else:
                # Formulation is no minus. Do not teacher-force scale −1 or
                # backprop minus end-margin / plan drift.
                pred_neg = pred_pos.detach()
                end_neg = torch.zeros((), device=device)
                plan_neg = torch.zeros((), device=device)
                edrift_n = 0.0
                pdrift_n = 0.0
            pred_zero = None
            pred_plus_prefix = None
            prefix_mask = None
            pred_lyric = None
            pred_concept = None
            tgt_neu_lyric = None
            tgt_pos_concept = None
            if recipe in PLUS_NEU_RECIPES:
                _set_scale(network, 0.0)
                pred_zero, _, hid_zero = _forward_teacher_forced(
                    lm, data["prompt_embeds"], data["frame_embeds"]
                )
                if recipe in PLUS_NEU_HOLD_RECIPES:
                    prompt_len = int(data["prompt_embeds"].shape[1])
                    pred_plus_prefix = hid_pos[:, :prompt_len]
                    if recipe in PLUS_NEU_LYRIC_RECIPES:
                        prefix_mask = data.get("neu_prefix_mask")
                    else:
                        prefix_mask = torch.ones(
                            pred_plus_prefix.shape[:2],
                            dtype=torch.long,
                            device=pred_plus_prefix.device,
                        )
                        if prompt_len > 0:
                            prefix_mask[:, prompt_len - 1] = 0
                    _ = hid_zero
                if recipe in PLUS_NEU_ROLES_RECIPES:
                    prompt_len = int(data["prompt_embeds"].shape[1])
                    student = hid_pos[:, :prompt_len]
                    spans = data["role_spans"]
                    pred_lyric = lm_gather_span(student, *spans["neu"]["lyric"])
                    pred_concept = lm_gather_span(student, *spans["neu"]["concept"])
                    tgt_neu_lyric = lm_gather_span(
                        data["neu_hidden"], *spans["neu"]["lyric"]
                    )
                    tgt_pos_concept = lm_gather_span(
                        data["pos_hidden"], *spans["pos"]["concept"]
                    )
                    _ = hid_zero
        else:
            if args.planreg_weight > 0:
                raise ValueError("--planreg_weight requires the audio-end regularizer (its pre-roll supplies the plan)")
            _set_scale(network, 1.0)
            pred_plus_prefix = None
            prefix_mask = None
            pred_lyric = None
            pred_concept = None
            tgt_neu_lyric = None
            tgt_pos_concept = None
            if recipe in PLUS_NEU_HOLD_RECIPES:
                hid_pos = _encode_full(lm, neu_ids, neu_mask)
                pred_pos, pred_plus_prefix, split_mask = _split_prefix_last(
                    hid_pos, neu_mask
                )
                if recipe in PLUS_NEU_LYRIC_RECIPES:
                    prefix_mask = data.get("neu_prefix_mask")
                else:
                    prefix_mask = split_mask
            elif recipe in PLUS_NEU_ROLES_RECIPES:
                hid_pos = _encode_full(lm, neu_ids, neu_mask)
                pred_pos = _gather_last_hidden(hid_pos, neu_mask).float()
                spans = data["role_spans"]
                pred_lyric = lm_gather_span(hid_pos, *spans["neu"]["lyric"])
                pred_concept = lm_gather_span(hid_pos, *spans["neu"]["concept"])
                tgt_neu_lyric = lm_gather_span(data["neu_hidden"], *spans["neu"]["lyric"])
                tgt_pos_concept = lm_gather_span(
                    data["pos_hidden"], *spans["pos"]["concept"]
                )
            else:
                pred_pos = _encode_train(lm, neu_ids, neu_mask)

            if _minus_pole_used(recipe):
                _set_scale(network, -1.0)
                pred_neg = _encode_train(lm, neu_ids, neu_mask)
            else:
                pred_neg = pred_pos.detach()
            end_pos = end_neg = torch.zeros((), device=device)
            plan_pos = plan_neg = torch.zeros((), device=device)
            edrift_p = edrift_n = 0.0
            pdrift_p = pdrift_n = 0.0
            pred_zero = None
            if recipe in PLUS_NEU_RECIPES:
                _set_scale(network, 0.0)
                pred_zero = _encode_train(lm, neu_ids, neu_mask)

        v_pos = pred_pos - neu_ref
        v_pos_t = tgt_plus - neu_ref
        cos_pos = F.cosine_similarity(v_pos, v_pos_t, dim=-1).mean()
        pperc = (torch.norm(pred_pos - tgt_plus) / torch.norm(v_pos_t).clamp_min(1e-6)).item()
        if recipe in PLUS_NEU_RECIPES:
            cos_neg = torch.zeros((), device=pred_pos.device)
            collapse = torch.zeros((), device=pred_pos.device)
            nperc = 0.0
        else:
            v_neg = pred_neg - neu_ref
            v_neg_t = tgt_minus - neu_ref
            cos_neg = F.cosine_similarity(v_neg, v_neg_t, dim=-1).mean()
            collapse = F.cosine_similarity(v_pos, v_neg, dim=-1).mean()
            nperc = (torch.norm(pred_neg - tgt_minus) / torch.norm(v_neg_t).clamp_min(1e-6)).item()

        pole = lm_train_loss(
            pred_pos,
            pred_neg,
            tgt_plus,
            tgt_minus,
            neu=neu_ref,
            slider_dir=data.get("slider_dir"),
            leak_dir=data.get("leak_dir"),
            hold_weight=float(data.get("hold_weight", hold_w)),
            anchor_plus=data.get("anchor_plus"),
            anchor_minus=data.get("anchor_minus"),
            anchor_weight=anchor_w,
            pole_mode=pole_mode,
            readout=readout,
            null_basis=null_basis,
            blind_projector=blind_projector,
            blind_weight=float(args.blind_weight),
            plus_only=recipe in PLUS_ONLY_RECIPES,
            plus_neu=recipe in PLUS_NEU_RECIPES,
            pred_zero=pred_zero,
            tgt_zero=neu_ref if recipe in PLUS_NEU_RECIPES else None,
            plus_neu_prefix=recipe in PLUS_NEU_HOLD_RECIPES,
            pred_plus_prefix=pred_plus_prefix,
            tgt_neu_prefix=data.get("neu_hidden"),
            prefix_mask=prefix_mask if prefix_mask is not None else data.get("neu_prefix_mask"),
            plus_neu_roles=recipe in PLUS_NEU_ROLES_RECIPES,
            pred_lyric=pred_lyric,
            tgt_neu_lyric=tgt_neu_lyric,
            pred_concept=pred_concept,
            tgt_pos_concept=tgt_pos_concept,
        )
        if recipe in PLUS_NEU_RECIPES:
            # Per-pole endreg strength matches bipolar's +1 term (not 0.5 · end_pos).
            loss = pole + args.endreg_weight * end_pos + args.planreg_weight * plan_pos
        else:
            loss = (
                pole
                + 0.5 * args.endreg_weight * (end_pos + end_neg)
                + args.planreg_weight * (plan_pos + plan_neg)
            )
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
            and _early_stop_hit(
                history,
                args.early_window,
                args.early_cos,
                args.early_collapse,
                args.early_perc,
                plus_neu=recipe in PLUS_NEU_RECIPES,
            )
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
        "plus_only": recipe in PLUS_ONLY_RECIPES,
        "plus_neu": recipe in PLUS_NEU_RECIPES,
        "plus_neu_prefix": recipe in PLUS_NEU_PREFIX_RECIPES,
        "plus_neu_lyric": recipe in PLUS_NEU_LYRIC_RECIPES,
        "plus_neu_roles": recipe in PLUS_NEU_ROLES_RECIPES,
        "even_blend_scale": float(args.even_blend_scale),
        "pole_mode": pole_mode,
        "symmetric": bool(args.symmetric),
        "hold_weight": hold_w,
        "project_align_min": align_min if recipe in PROJECT_RECIPES else getattr(args, "project_align_min", None),
        "project_align_scope": align_scope if recipe in PROJECT_RECIPES else None,
        "anchor_weight": anchor_w,
        "leakage_floor": args.leakage_floor,
        "slider_positive": axis_captions[0] if axis_captions else "",
        "slider_negative": axis_captions[1] if axis_captions else "",
        "leak_positive": leak_captions[0] if leak_captions else "",
        "leak_negative": leak_captions[1] if leak_captions else "",
        "common_beta": beta,
        "target_scale": float(args.target_scale),
        "planreg_weight": float(args.planreg_weight),
        "blind_weight": float(args.blind_weight) if pole_mode == "dual_band" else None,
        "blind_cut": float(args.blind_cut) if pole_mode == "dual_band" else None,
        "blind_dims": (
            None
            if blind_projector is None
            else int(round(float(blind_projector.trace())))
        ),
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
        help="live target recipe (default v9 = full pair-odd + hold-on-ê). v9: "
        "--symmetric polarity, t± = h0 ± a, κ=0; hold (h(±1)−h0)·ê_⊥û when "
        "leak_positive/leak_negative is declared. Gender stays here "
        "(no ê, hold 0). pair_odd_sub_e: leaky-axis teacher = pair-odd "
        "minus ê_⊥ (λ→∞ hold, hold 0). Short slider_positive is not the "
        "teacher. v9_project: old slider-level |odd·û| gate. "
        "v9_always: old always-project. hub: published leakage_floor "
        "blend-back (still leaks). symmetric / faithful: old poles. "
        "faithful_sub_e: real poles with leftover ê subtracted from the "
        "odd part only (midpoint stays ½(h++h−); needs leak_*). "
        "faithful_sub_e_if_unused: subtract leftover ê only when "
        "|ê̂_⊥ · â| < 0.50 (measured unused leftover ≤ 0.39, energy-v4 "
        "restates at 0.78); otherwise raw poles. Leak_* optional — a clean "
        "pair is the raw poles. faithful_guard_e: the same subtraction, "
        "taken only while the blend guard admits it — the ê-cleaned "
        "target must stay nearer the pole caption than ½(h++h−), which "
        "refuses on energy-v4 where ê restates the axis and keeps the "
        "caption instead. Safe with no leak_* declared (then it is "
        "faithful). faithful_even_blend: leftover-gate the odd part "
        "and subtract --even_blend_scale of the leak-pair even leftover "
        "(half by default). Caption even stays; scale 1.0 fails "
        "exam_divergent. faithful_plus: train the + pole only. Teacher "
        "is leftover-gated h+ (raw pos when leftover ê is unused or "
        "undeclared). No pair-odd, no h0 ± a, no minus MSE. Inference "
        "may still expose a −1 fader; that fader is unconstrained. "
        "faithful_plus_neu: UNI. Student +1 fits raw h+ (never leftover-"
        "gated, even if leak_* exists). Student scale 0 fits h0. Last-hidden "
        "MSE only. faithful_plus_neu_prefix: UNI plus a prefix hold. Student "
        "+1 last hidden fits raw h+, +1 prefix hidden fits encode(neu) "
        "prefix (yaml lyrics, not encode(pos) prefix), scale 0 fits h0. "
        "Pins Vocal Details too. faithful_plus_neu_lyric: UNI plus a "
        "lyric-token hold. Student +1 last hidden fits raw h+, +1 yaml "
        "lyrics tokens fit encode(neu) lyrics only, Vocal Details / "
        "metadata stay free, scale 0 fits h0. "
        "faithful_plus_neu_roles: UNI plus a role split. Student +1 last "
        "fits raw h+, +1 yaml lyrics fit encode(neu) lyrics, +1 Vocal "
        "Details (or the whole caption if that heading cannot be located) "
        "fit encode(pos) same role, scale 0 fits h0. Fail closed if a "
        "required span is missing. No minus MSE, no pair-odd, no h0 ± a. "
        "Not the default",
    )
    p.add_argument(
        "--even_blend_scale",
        type=float,
        default=EVEN_BLEND_SCALE,
        help="faithful_even_blend only: how much of leak-pair even leftover "
        "to subtract. Default 0.5. 1.0 drops all ê_even and fails the "
        "divergent pair exam; 0 is leftover-gate alone",
    )
    p.add_argument(
        "--pole_mode",
        default="hidden",
        choices=POLE_MODES,
        help="pole supervision (default hidden = current hidden MSE onto the "
        "chosen targets). semantic_kl: next-token KL of the student policy "
        "to the teacher hidden's policy, using the semantic band of lm_head "
        "(lm_semantic_pole_loss). semantic_kl_null: same KL plus hidden MSE "
        "on ker(lm_head). semantic_kl_plus_hidden / semantic_kl_pin are "
        "aliases of semantic_kl_null, not forks. hidden_kl: hidden MSE plus "
        "a 0.001x semantic KL. dual_band: that KL plus hidden MSE on the "
        "band the semantic head is blind to (lm_dual_band_pole_loss). Do "
        "not make any of these the default",
    )
    p.add_argument(
        "--blind_weight",
        type=float,
        default=DUAL_BAND_WEIGHT,
        help="--pole_mode dual_band only: weight on the blind-band MSE. The "
        "pair-exam cell is flat in this over 0.5 … 32; the term supplies a "
        "gradient where there was none rather than outweighing the KL",
    )
    p.add_argument(
        "--blind_cut",
        type=float,
        default=0.0,
        help="--pole_mode dual_band only: singular values of the centered "
        "semantic band at or below cut·s_max count as blind. 0 is the exact "
        "null space; a band whose row space fills the hidden width has none, "
        "and the run prints a warning telling you to raise this",
    )
    p.add_argument(
        "--symmetric",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="polarity step inside v9/hub/symmetric: tgt(±1) sit opposite around neu. "
        "v9 keeps that full odd teacher; v9_project may project it onto û",
    )
    p.add_argument(
        "--slider_positive",
        default=None,
        help="declared + slider caption encoded once as û (name / probe, not the "
        "v9 teacher). Required for --lm_target v9_project / v9_always unless "
        "YAML slider_positive is set",
    )
    p.add_argument(
        "--slider_negative",
        default=None,
        help="declared − slider caption; pair with --slider_positive",
    )
    p.add_argument(
        "--leak_positive",
        default=None,
        help="declared unused leak caption encoded once as ê (mix / BPM / genre, "
        "not the slider). YAML leak_positive or leak: [pos, neg] also work. "
        "Omit when the pair is already clean or attributes pin the unused axis",
    )
    p.add_argument(
        "--leak_negative",
        default=None,
        help="declared − leak caption; pair with --leak_positive",
    )
    p.add_argument(
        "--hold_weight",
        type=float,
        default=None,
        help="v9: weight on ((h(±1)−h0)·ê)² (default 8 when ê is declared, else 0). "
        "pair_odd_sub_e / faithful_sub_e / faithful_sub_e_if_unused: default 0 "
        "(ê_⊥ is already out of the teacher, or the gate kept raw poles). "
        "v9_project / v9_always: weight on ||(h(±1)−h0)_⊥û||² (default 1.0). "
        "Do not scale λ by D; prefer leftover ê in the teacher",
    )
    p.add_argument(
        "--project_align_min",
        type=float,
        default=None,
        help="v9_project alignment floor (default 0.50). Ignored by default v9 "
        "and by v9_always. See docs/lm-live-cells.md",
    )
    p.add_argument(
        "--project_align_scope",
        default=None,
        choices=("slider", "row"),
        help="v9_project gate aggregation (default slider). Ignored by default v9",
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
        help="stop when a rolling window matches c+/c-/collapse/perc (replayed on all v3 LM runs). "
        "plus+neu recipes ignore c-/nperc/collapse (no minus teacher)",
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
    history: list[dict],
    window: int,
    min_cos: float,
    max_collapse: float,
    max_perc: float,
    *,
    plus_neu: bool = False,
) -> bool:
    if window <= 0 or len(history) < window:
        return False
    chunk = history[-window:]
    cos_pos = sum(r["cos_pos"] for r in chunk) / window
    cos_neg = sum(r["cos_neg"] for r in chunk) / window
    collapse = sum(r["collapse"] for r in chunk) / window
    pperc = sum(r["pperc"] for r in chunk) / window
    nperc = sum(r["nperc"] for r in chunk) / window
    if plus_neu:
        # No minus teacher. Requiring c- / nperc / collapse would never fire
        # (or would fire on dummy zeros) and is not the UNI card.
        return cos_pos > min_cos and pperc < max_perc
    return (
        cos_pos > min_cos
        and cos_neg > min_cos
        and collapse < max_collapse
        and pperc < max_perc
        and nperc < max_perc
    )


if __name__ == "__main__":
    train(parse_args())
