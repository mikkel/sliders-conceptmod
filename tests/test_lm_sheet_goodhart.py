"""The sheet cell: hidden MSE onto the pair-odd midpoint sings off-sheet.

Every assertion here re-runs the analysis rather than reading
``docs/lm-sheet-goodhart/metrics.json``. CPU only, no Hub, no GPU.
Nothing in this file changes the live trainer default; one test guards
that it stays where it is.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from analysis.slider2d.sheet import (
    GARBLE_MAX,
    LEAK_LOCK,
    LIVE_PROBE_COS,
    LIVE_V4_COLLAPSE,
    LIVE_V4_C_PLUS,
    OFF_SHEET_TOKENS,
    SHEET_LOCK,
    SWING_FLOOR,
    beta_sweep,
    common_share,
    common_sweep,
    corpus,
    first_above,
    flip_point,
    gender_cell,
    gender_like_field,
    leaky_cell,
    leaky_field,
    live_log_table,
    live_probe_table,
    null_space_table,
    nucleus,
    probe_cos,
    score_sheet,
    sheets,
    teacher_points,
    teacher_sheet_row,
    teacher_sheet_table,
    teacher_swings,
)
from conceptmod.textsliders.slider_targets import (
    lm_hidden_targets,
    lm_next_token_logits,
    lm_semantic_kl,
    lm_semantic_pole_loss,
)
from conceptmod.textsliders.train_lm_slider_music3 import (
    LM_RECIPES,
    SUB_E_RECIPES,
    V9_RECIPES,
    lm_train_targets,
    parse_args,
    resolve_lm_loss_weights,
)


STEPS = 400
_CACHE: dict[str, list[dict]] = {}


def gender() -> dict[str, dict]:
    if "gender" not in _CACHE:
        _CACHE["gender"] = gender_cell(steps=STEPS)
    return {row["name"]: row for row in _CACHE["gender"]}


def leaky() -> dict[str, dict]:
    if "leaky" not in _CACHE:
        _CACHE["leaky"] = leaky_cell(steps=STEPS)
    return {row["name"]: row for row in _CACHE["leaky"]}


def teachers() -> dict[str, dict]:
    if "teachers" not in _CACHE:
        _CACHE["teachers"] = teacher_sheet_table()
    return {row["name"]: row for row in _CACHE["teachers"]}


def sweep() -> list[dict]:
    if "sweep" not in _CACHE:
        _CACHE["sweep"] = common_sweep(steps=200)
    return _CACHE["sweep"]


# -- the geometry the live log already reports ----------------------------


def test_common_share_and_probe_cos_are_inverses():
    for cos in (-0.9, -0.08, 0.0, 0.03, 0.32, 0.7):
        assert probe_cos(common_share(cos)) == pytest.approx(cos, abs=1e-9)


def test_field_probe_cos_matches_the_closed_form():
    """The field's actual cos(pos−neu, neg−neu) is the formula the doc uses."""
    for share in (0.0, 0.4, 0.923, 1.03, 1.39):
        field = leaky_field(common=share)
        assert field.probe_cos(0) == pytest.approx(probe_cos(share), abs=1e-5)


def test_every_live_axis_deletes_a_common_component_near_its_own_size():
    """The v4 probe table, read as ‖c‖/‖a‖."""
    table = live_probe_table()
    assert len(table) == len(LIVE_PROBE_COS)
    shares = [row["common_share"] for row in table]
    assert min(shares) >= 0.90
    assert max(shares) <= 1.45
    # Only live-lm-v5 ever tripped the trainer's collapse warning.
    assert sum(1 for row in table if row["warned"]) == 1


def test_common_beta_one_is_exactly_the_raw_pole():
    """``h0 ± a + c = h±``, so β = 1 turns the symmetric target into v6.

    This is live ``lm_hidden_targets``, not fixture code: it is why the
    β ladder in the report is a real dial and not an analogy.
    """
    field = leaky_field()
    pos, neg, neu = field.poles(0)
    plus, minus = lm_hidden_targets(pos, neg, neu, target_mode="symmetric", common_beta=1.0)
    assert torch.allclose(plus, pos, atol=1e-6)
    assert torch.allclose(minus, neg, atol=1e-6)


# -- the sheet is ground truth -------------------------------------------


def test_nucleus_is_a_top_p_support():
    policy = torch.tensor([0.5, 0.3, 0.15, 0.05])
    assert nucleus(policy, 0.5) == {0}
    assert nucleus(policy, 0.79) == {0, 1}
    assert nucleus(policy, 0.9) == {0, 1, 2}
    assert nucleus(policy, 1.0) == {0, 1, 2, 3}


def test_corpus_excludes_exactly_the_off_sheet_tokens():
    """No real caption of this song supports ``garble_*``, and nothing else."""
    field = leaky_field()
    head = field.readout()
    inside = corpus(sheets(field, head))
    outside = {head.tokens[i] for i in range(len(head.tokens)) if i not in inside}
    assert outside == set(OFF_SHEET_TOKENS)


def test_readout_null_dims_are_the_invisible_block():
    field = leaky_field()
    assert len(field.readout().null_dims()) == field.null_dims == 2
    assert not gender_like_field().readout().null_dims()


def test_the_poles_themselves_are_on_their_own_sheet():
    """The ceiling. If the caption failed here the cell would be rigged."""
    for field in (gender_like_field(), leaky_field()):
        ceiling = teacher_swings(field)
        assert ceiling["on_sheet"] >= field.sheet_p
        assert ceiling["garble"] <= 0.01
        assert ceiling["argmax_on_sheet"] == 1.0
        assert ceiling["concept_swing"] > 0.5


# -- the target points, with no optimizer in the loop --------------------


def test_the_midpoint_is_not_a_caption():
    row = teachers()["pair_odd"]
    field = leaky_field()
    assert row["is_caption"] is False
    # It sits the whole slider axis away from the pole, and keeps no c.
    assert row["off_caption"] == pytest.approx(field.common, abs=1e-3)
    assert row["sheet_dir_kept"] == pytest.approx(0.0, abs=1e-6)


def test_the_midpoint_target_is_off_sheet_before_anything_is_fitted():
    caption = teachers()["caption"]
    midpoint = teachers()["pair_odd"]
    assert caption["garble"] <= GARBLE_MAX
    assert midpoint["garble"] > 20.0 * caption["garble"]
    assert midpoint["on_sheet"] < 0.5 * caption["on_sheet"]
    assert midpoint["argmax_on_sheet"] < 1.0
    # It literally says an off-sheet token at ±1.
    assert any(tok in midpoint["says"] for tok in OFF_SHEET_TOKENS)
    assert not any(tok in caption["says"] for tok in OFF_SHEET_TOKENS)


def test_pair_odd_sub_e_moves_further_from_the_caption_not_closer():
    """#20 fixes which unused axis the target carries, not where it is."""
    plain = teachers()["pair_odd"]
    sub_e = teachers()["pair_odd_sub_e"]
    assert sub_e["off_caption"] > plain["off_caption"]
    assert sub_e["garble"] >= plain["garble"]
    assert sub_e["sheet_dir_kept"] == pytest.approx(0.0, abs=1e-6)


def test_beta_walks_the_target_onto_the_sheet():
    rows = teachers()
    ladder = [rows["pair_odd"], rows["pair_odd_beta05"], rows["pair_odd_beta1"]]
    assert [r["sheet_dir_kept"] for r in ladder] == pytest.approx([0.0, 0.5, 1.0], abs=1e-4)
    assert ladder[0]["garble"] > ladder[1]["garble"] > ladder[2]["garble"]
    assert rows["pair_odd_beta1"]["off_caption"] == pytest.approx(0.0, abs=1e-6)


def test_e_cleaning_is_a_small_step_off_the_caption():
    """``faithful_sub_e`` keeps ``c``, so it stays near the sheet."""
    rows = teachers()
    assert rows["faithful_sub_e"]["sheet_dir_kept"] == pytest.approx(1.0, abs=1e-4)
    assert rows["faithful_sub_e"]["off_caption"] < 0.5 * rows["pair_odd"]["off_caption"]
    assert rows["faithful_sub_e"]["garble"] <= GARBLE_MAX
    assert rows["faithful_sub_e"]["argmax_on_sheet"] == 1.0


def test_the_leftover_gate_matches_sub_e_on_unused_and_raw_poles_on_clean():
    leaky = leaky_field()
    e = leaky.leak_e()
    gated = teacher_points(leaky, 0, teacher="faithful_sub_e_if_unused", leak_dir=e)
    cleaned = teacher_points(leaky, 0, teacher="faithful_sub_e", leak_dir=e)
    assert torch.allclose(gated[0], cleaned[0])
    assert torch.allclose(gated[1], cleaned[1])
    gender = gender_like_field()
    raw = teacher_points(gender, 0, teacher="faithful_sub_e_if_unused")
    poles = teacher_points(gender, 0, teacher="faithful")
    assert torch.allclose(raw[0], poles[0])
    assert torch.allclose(raw[1], poles[1])


# -- the Goodhart --------------------------------------------------------


@pytest.mark.parametrize("cell", ["gender", "leaky"])
def test_v9_hidden_looks_locked_and_sings_off_sheet(cell):
    row = (gender() if cell == "gender" else leaky())["v9_hidden"]
    # The live log columns say the slider is perfect.
    assert row["pair_odd_cos"] >= LIVE_V4_C_PLUS
    assert row["collapse"] <= LIVE_V4_COLLAPSE
    assert row["looks_locked"] is True
    # The sheet says otherwise.
    assert row["on_sheet_kept"] < 0.6
    assert row["garble"] > 8.0 * GARBLE_MAX
    assert row["argmax_on_sheet"] == 0.0
    assert row["pass"] is False
    assert row["misleading_lock"] is True


def test_pair_odd_cos_is_maximized_by_the_worst_recipe():
    """The metric's argmax is the recipe with the least on-sheet mass."""
    rows = list(leaky().values())
    best_cos = max(rows, key=lambda r: r["pair_odd_cos"])
    worst_sheet = min(rows, key=lambda r: r["on_sheet_kept"])
    assert best_cos["name"] == "v9_hidden"
    assert best_cos["misleading_lock"] is True
    assert worst_sheet["on_sheet_kept"] < 0.5
    # Anti-correlated across the whole recipe set.
    cos = torch.tensor([r["pair_odd_cos"] for r in rows])
    sheet = torch.tensor([r["on_sheet_kept"] for r in rows])
    corr = F.cosine_similarity(
        (cos - cos.mean()).unsqueeze(0), (sheet - sheet.mean()).unsqueeze(0)
    ).item()
    assert corr < -0.85


def test_hold_e_and_sub_e_fix_leak_and_not_the_sheet():
    rows = leaky()
    v9 = rows["v9_hidden"]
    for name in ("v9_hold_e", "v15_pair_odd_sub_e"):
        row = rows[name]
        assert abs(row["leak_tok"]) <= LEAK_LOCK
        assert abs(row["leak_tok"]) < abs(v9["leak_tok"])
        # ... and the sheet is no better than plain v9.
        assert row["on_sheet_kept"] <= v9["on_sheet_kept"]
        assert row["garble"] >= v9["garble"]
        assert row["misleading_lock"] is True


def test_the_swing_is_a_second_goodhart():
    """Perfect hidden cosine, a quarter of the caption's token swing."""
    for row in (gender()["v9_hidden"], leaky()["v9_hidden"]):
        assert row["pair_odd_cos"] >= LIVE_V4_C_PLUS
        assert row["swing_kept"] < 0.5 * SWING_FLOOR


# -- what the v16 target does -------------------------------------------


@pytest.mark.parametrize(
    "cell,name",
    [("gender", "v16_semantic_kl"), ("leaky", "v16_semantic_kl_sub_e")],
)
def test_semantic_kl_onto_a_caption_stays_on_sheet(cell, name):
    row = (gender() if cell == "gender" else leaky())[name]
    assert row["on_sheet_kept"] >= SHEET_LOCK
    assert row["garble"] <= GARBLE_MAX
    assert row["argmax_on_sheet"] == 1.0
    assert row["swing_kept"] >= SWING_FLOOR
    assert abs(row["leak_tok"]) <= LEAK_LOCK
    assert row["pass"] is True
    assert row["looks_locked"] is False


@pytest.mark.parametrize("cell,name", [("gender", "v16_semantic_kl"), ("leaky", "v16_semantic_kl_sub_e")])
def test_the_v16_row_logs_worse_pair_odd_numbers_than_v9(cell, name):
    rows = gender() if cell == "gender" else leaky()
    v9 = rows["v9_hidden"]
    v16 = rows[name]
    assert v16["pair_odd_cos"] < v9["pair_odd_cos"]
    assert v16["collapse"] > v9["collapse"]
    # And the honest hidden-space lock goes the other way.
    assert v16["pole_cos"] > v9["pole_cos"]


def test_collapse_under_a_caption_target_approaches_the_probe_cos():
    """A v16 run logging collapse near −1 has not recovered ``c``.

    ``d± = ±a + c`` gives ``cos(d+, d−) = cos(pos−neu, neg−neu)``, so the
    number the trainer already prints before training is the number
    collapse should converge to after it.
    """
    for rows, name in ((gender(), "v16_semantic_kl"), (leaky(), "v16_semantic_kl")):
        row = rows[name]
        assert row["collapse"] == pytest.approx(row["probe_cos"], abs=0.15)
        assert abs(rows["v9_hidden"]["collapse"] - row["probe_cos"]) > 0.8


def test_kl_onto_the_midpoint_is_still_garbled():
    """The control: KL is not the fix, the target point is."""
    for rows in (gender(), leaky()):
        kl = rows["kl_on_midpoint"]
        v9 = rows["v9_hidden"]
        assert kl["garble"] == pytest.approx(v9["garble"], abs=0.05)
        assert kl["on_sheet_kept"] == pytest.approx(v9["on_sheet_kept"], abs=0.05)
        assert kl["misleading_lock"] is True
        assert kl["pass"] is False


def test_the_passing_recipes_all_target_a_real_caption():
    """Both survivors aim at a pole; one uses MSE and one uses KL."""
    winners = {name for name, row in leaky().items() if row["pass"]}
    assert winners == {"faithful_sub_e", "v16_semantic_kl_sub_e"}
    modes = {leaky()[name]["pole_mode"] for name in winners}
    assert modes == {"hidden", "semantic_kl"}
    for name in winners:
        assert leaky()[name]["teacher"].startswith("faithful")


def test_a_caption_target_without_e_cleaning_still_leaks():
    """The energy poles really do move the gender token; ê is for that."""
    rows = leaky()
    ceiling = teacher_swings(leaky_field())
    assert abs(ceiling["leak_tok"]) > LEAK_LOCK
    for name in ("v6_faithful", "v16_semantic_kl"):
        row = rows[name]
        assert row["on_sheet_kept"] >= SHEET_LOCK
        assert abs(row["leak_tok"]) > LEAK_LOCK
        assert row["axis"]["leak"] == "needs_help"
        assert row["axis"]["sheet"] == "right"
        assert row["pass"] is False


def test_semantic_kl_ignores_the_readout_null_space():
    """Hidden MSE copies pole content that cannot change a token; KL does not."""
    rows = {row["name"]: row for row in null_space_table(steps=300)}
    assert rows["hidden_faithful"]["null_kept"] == pytest.approx(1.0, abs=0.05)
    for name in ("kl_faithful", "kl_faithful_sub_e"):
        assert rows[name]["null_kept"] == pytest.approx(0.0, abs=1e-6)
        assert rows[name]["on_sheet"] >= 0.8
    assert rows["kl_null_faithful"]["null_kept"] == pytest.approx(1.0, abs=0.05)
    assert rows["kl_null_faithful"]["on_sheet"] >= 0.8


# -- sweeps ---------------------------------------------------------------


def test_the_logged_geometry_is_flat_while_the_sheet_collapses():
    rows = sweep()
    for row in rows:
        assert row["hidden_pair_odd_cos"] >= 0.999
        assert row["hidden_collapse"] <= -0.999
    on_sheet = [row["hidden_on_sheet"] for row in rows]
    assert max(on_sheet) - min(on_sheet) > 0.6
    garble = [row["hidden_garble"] for row in rows]
    assert min(garble) <= 1e-6 and max(garble) > 8.0 * GARBLE_MAX


def test_a_perfectly_odd_pair_makes_v9_correct():
    """Not rigged: with no common component the midpoint *is* the caption."""
    row = score_sheet(
        "zero_common",
        leaky_field(common=0.0),
        pole_mode="hidden",
        teacher="pair_odd",
        steps=200,
    )
    assert row["on_sheet_kept"] >= SHEET_LOCK
    assert row["garble"] <= GARBLE_MAX
    assert row["argmax_on_sheet"] == 1.0
    assert row["pair_odd_cos"] >= 0.999


def test_the_flip_points_are_below_every_live_axis():
    rows = sweep()
    argmax_flip = flip_point(rows)
    garble_flip = first_above(rows, "hidden_garble", GARBLE_MAX)
    live_min = min(common_share(cos) for cos in LIVE_PROBE_COS.values())
    assert argmax_flip is not None and garble_flip is not None
    # The argmax drifts before mass leaves the vocabulary entirely.
    assert argmax_flip <= garble_flip < live_min


def test_semantic_kl_holds_the_sheet_across_the_whole_sweep():
    for row in sweep():
        assert row["kl_on_sheet"] >= 0.8
        assert row["kl_garble"] <= GARBLE_MAX


def test_kl_pair_odd_numbers_degrade_monotonically_with_the_common_share():
    """Worse-looking is the expected signature, not a regression."""
    rows = sweep()
    cos = [row["kl_pair_odd_cos"] for row in rows]
    collapse = [row["kl_collapse"] for row in rows]
    assert all(b <= a + 1e-3 for a, b in zip(cos, cos[1:]))
    assert all(b >= a - 1e-3 for a, b in zip(collapse, collapse[1:]))
    assert cos[0] > cos[-1] + 0.3
    assert collapse[-1] > 0.0 > collapse[0]


def test_the_beta_ladder_fixes_the_sheet_and_not_the_leak():
    rows = beta_sweep(steps=200)
    on_sheet = [row["on_sheet"] for row in rows]
    garble = [row["garble"] for row in rows]
    cos = [row["pair_odd_cos"] for row in rows]
    collapse = [row["collapse"] for row in rows]
    assert all(b > a for a, b in zip(on_sheet, on_sheet[1:]))
    assert all(b < a for a, b in zip(garble, garble[1:]))
    assert all(b < a for a, b in zip(cos, cos[1:]))
    assert all(b > a for a, b in zip(collapse, collapse[1:]))
    assert rows[-1]["garble"] <= GARBLE_MAX
    # β is the same ``a`` plus a shared term, so the leak rides along.
    leaks = [row["leak_tok"] for row in rows]
    assert max(leaks) - min(leaks) < 1e-4
    assert min(leaks) > LEAK_LOCK


def test_the_bend_student_reproduces_the_live_log_and_is_no_better():
    rows = {(r["cell"], r["student"]): r for r in live_log_table(steps=STEPS)}
    for cell in ("gender", "energy"):
        bend = rows[(cell, "bend")]
        free = rows[(cell, "odd_even")]
        assert bend["pair_odd_cos"] == pytest.approx(LIVE_V4_C_PLUS, abs=0.02)
        assert bend["collapse"] == pytest.approx(LIVE_V4_COLLAPSE, abs=0.01)
        assert free["pair_odd_cos"] == pytest.approx(1.0, abs=1e-3)
        assert free["collapse"] == pytest.approx(-1.0, abs=1e-3)
        # The live-looking pair is further off the sheet, not closer.
        assert bend["on_sheet_kept"] <= free["on_sheet_kept"]
        assert bend["garble"] >= free["garble"]
        for row in (bend, free):
            assert row["misleading_lock"] is True


# -- the loss primitives -------------------------------------------------


def test_next_token_logits_is_the_head_applied_to_the_hidden():
    hidden = torch.randn(4, 6)
    weight = torch.randn(9, 6)
    assert torch.allclose(lm_next_token_logits(hidden, weight), hidden @ weight.T)
    bias = torch.randn(9)
    assert torch.allclose(
        lm_next_token_logits(hidden, weight, bias=bias), hidden @ weight.T + bias
    )


def test_semantic_kl_is_zero_only_on_the_same_policy():
    logits = torch.randn(3, 7)
    assert float(lm_semantic_kl(logits, logits)) == pytest.approx(0.0, abs=1e-7)
    # A constant per-row shift is the same policy.
    shifted = logits + torch.randn(3, 1)
    assert float(lm_semantic_kl(shifted, logits)) == pytest.approx(0.0, abs=1e-6)
    assert float(lm_semantic_kl(torch.randn(3, 7), logits)) > 0.0


def test_semantic_kl_is_the_forward_kl():
    """Teacher first: mass the caption puts somewhere must be matched."""
    tgt = torch.tensor([[0.0, 0.0, 0.0]])
    pred = torch.tensor([[6.0, 0.0, 0.0]])
    p = F.softmax(tgt, dim=-1)
    q = F.softmax(pred, dim=-1)
    want = float((p * (p.log() - q.log())).sum())
    assert float(lm_semantic_kl(pred, tgt)) == pytest.approx(want, rel=1e-5)
    # Dropping the caption's spread-out mass costs more than over-hedging.
    assert float(lm_semantic_kl(pred, tgt)) > 2.0 * float(lm_semantic_kl(tgt, pred))


def test_semantic_pole_loss_adds_the_hold_like_the_mse_one():
    plus, minus = torch.randn(1, 5), torch.randn(1, 5)
    t_plus, t_minus = torch.randn(1, 5), torch.randn(1, 5)
    bare = float(lm_semantic_pole_loss(plus, minus, t_plus, t_minus))
    hold = torch.tensor(0.25)
    with_hold = float(
        lm_semantic_pole_loss(plus, minus, t_plus, t_minus, hold=hold, hold_weight=8.0)
    )
    assert with_hold == pytest.approx(bare + 8.0 * 0.25, rel=1e-5)
    assert (
        float(lm_semantic_pole_loss(plus, minus, t_plus, t_minus, hold=hold, hold_weight=0.0))
        == pytest.approx(bare, rel=1e-6)
    )


# -- guards --------------------------------------------------------------


def test_the_live_default_is_still_hidden_mse_onto_the_midpoint():
    """Wiring exists; the default must stay v9 / hidden.

    ``faithful_sub_e`` and ``semantic_kl`` are live flags now. Neither
    is the default: pole supervision is still ``--lm_target v9`` and
    ``--pole_mode hidden`` (hidden MSE).
    """
    args = parse_args(["--prompts", "x.yaml"])
    assert args.lm_target == "v9"
    assert args.pole_mode == "hidden"
    assert args.common_beta == 0.0
    assert "v9" in LM_RECIPES and "pair_odd_sub_e" in LM_RECIPES
    assert "faithful_sub_e" in LM_RECIPES
    assert "faithful_sub_e_if_unused" in LM_RECIPES
    assert "faithful_sub_e" not in V9_RECIPES
    assert "faithful_sub_e_if_unused" not in V9_RECIPES


def test_the_recommended_live_ladder_is_reachable_today():
    """``--lm_target symmetric --common_beta 1`` really is the raw poles.

    The report offers this as the one on-sheet target reachable without new
    code. It is only reachable off `v9`: `v9` and `pair_odd_sub_e` drop β.
    """
    assert "symmetric" not in V9_RECIPES | SUB_E_RECIPES
    field = leaky_field()
    pos, neg, neu = field.poles(0)
    plus, minus, anc_plus, anc_minus = lm_train_targets(
        pos, neg, neu, recipe="symmetric", common_beta=1.0
    )
    assert torch.allclose(plus, pos, atol=1e-6)
    assert torch.allclose(minus, neg, atol=1e-6)
    assert anc_plus is None and anc_minus is None
    # No hold on that recipe, so it is the faithful teacher exactly.
    hold, anchor = resolve_lm_loss_weights(
        "symmetric", hold_weight=None, anchor_weight=None, leak_declared=True
    )
    assert (hold, anchor) == (0.0, 0.0)
    # v9 ignores β, which is why the ladder is not reachable there.
    v9_plus, _v9_minus, _a, _b = lm_train_targets(
        pos, neg, neu, recipe="v9", common_beta=1.0
    )
    assert not torch.allclose(v9_plus, pos, atol=1e-3)


def test_teacher_row_rejects_an_e_recipe_without_a_declared_axis():
    field = leaky_field()
    with pytest.raises(ValueError):
        teacher_sheet_row("no_e", field, teacher="pair_odd_sub_e")
    with pytest.raises(ValueError):
        teacher_sheet_row("no_e", field, teacher="faithful_sub_e")


def test_unknown_modes_are_refused():
    field = leaky_field()
    with pytest.raises(ValueError):
        score_sheet("bad", field, pole_mode="mse", steps=1)
    with pytest.raises(ValueError):
        score_sheet("bad", field, teacher="midpoint", steps=1)
    with pytest.raises(ValueError):
        common_share(-1.0)
