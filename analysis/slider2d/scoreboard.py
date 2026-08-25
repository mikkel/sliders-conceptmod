"""Compile every scored 2-D / high-D / sheet recipe into one gated table.

Each existing cell answers a different question: leftover leak, short-û
strength, richness, high-D stiffness, lyric sheet. None of them is the
whole score. This module does not invent a loss. It reuses the live
``score_*`` runners, copies their real numbers, and applies one compiled
gate:

    leftover leak is small
    AND (when a sheet readout exists) the student stays on-sheet
        with the caption's concept swing intact

Pair-odd cosine and ±1 collapse are copied into the table as log
columns. They are not inputs to the gate. That is the #22 result:
``cos(d+, a) = 1`` is the recipe that garbles most.

CPU only. No Hub, no GPU, no Music 3 weights. Does not change the live
trainer default.
"""

from __future__ import annotations

from typing import Any

from analysis.slider2d.faithful import score_leak_lm
from analysis.slider2d.highd import (
    energy_field,
    gender_like_field as highd_gender_field,
    leftover_only_e,
    score_highd,
)
from analysis.slider2d.live_compare import live_policy_table
from analysis.slider2d.live_exam import (
    OFF_CAPTION_MAX,
    ROLLOUT_GARBLE_MAX,
    ROLLOUT_MATCH_MIN,
    divergent_pair_field,
    live_exam_cell,
    score_exam,
)
from analysis.slider2d.overlap import score_overlap_policy
from analysis.slider2d.rich import (
    RICH_KEPT_MIN,
    RichField,
    score_rich,
)
from analysis.slider2d.sheet import (
    ARGMAX_LOCK,
    GARBLE_MAX,
    LEAK_LOCK,
    SHEET_LOCK,
    SWING_FLOOR,
    gender_cell,
    leaky_cell,
    leaky_field,
    score_sheet,
)
from conceptmod.textsliders.slider_targets import LEAK_HOLD_WEIGHT as LIVE_HOLD


# Compiled gate. Identical numbers to the sheet cell; leftover leak uses
# the same 0.20 lock the hidden-geometry cells already share.
COMPILED_LEAK_LOCK = LEAK_LOCK
COMPILED_SHEET_LOCK = SHEET_LOCK
COMPILED_GARBLE_MAX = GARBLE_MAX
COMPILED_ARGMAX_LOCK = ARGMAX_LOCK
COMPILED_SWING_FLOOR = SWING_FLOOR
COMPILED_CONCEPT_COS = 0.90
COMPILED_STRENGTH_FLOOR = 0.50

WORKS = "works"
WORKS_GENDER_ONLY = "works-on-gender-only"
FAILS = "fails"
VERDICT_ORDER = {WORKS: 0, WORKS_GENDER_ONLY: 1, FAILS: 2}

# Sheet recipe names on the leftover (energy-like) cell.
SHEET_LEFTOVER = {
    "faithful_raw": "v6_faithful",
    "pair_odd_midpoint": "v9_hidden",
    "hold_e_perp_l8": "v9_hold_e",
    "pair_odd_sub_e": "v15_pair_odd_sub_e",
    "faithful_sub_e": "faithful_sub_e",
    "semantic_kl_midpoint": "kl_on_midpoint",
    "semantic_kl_poles": "v16_semantic_kl",
    "semantic_kl_sub_e": "v16_semantic_kl_sub_e",
}
SHEET_GENDER = {
    "faithful_raw": "v6_faithful",
    "faithful_attrs": "v6_faithful",
    "pair_odd_midpoint": "v9_hidden",
    "hub": "v9_hidden",
    "hold_e_raw_l1": "v9_hidden",
    "hold_e_raw_l8": "v9_hidden",
    "hold_e_perp_l1": "v9_hidden",
    "hold_e_perp_l8": "v9_hidden",
    "pair_odd_sub_e": "v9_hidden",
    "semantic_kl_midpoint": "kl_on_midpoint",
    "semantic_kl_poles": "v16_semantic_kl",
    "gender_like_no_e": "v9_hidden",
    "hidden_beta1": "hidden_beta1",
}


def na(value: Any) -> Any:
    """JSON-friendly missing metric. ``None`` stays ``None``."""
    return value


def _finite(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def leak_ok(leak: float | None) -> bool | None:
    """``None`` means the fixture has no leftover axis to report."""
    if leak is None:
        return None
    return abs(float(leak)) <= COMPILED_LEAK_LOCK


def sheet_ok(
    *,
    on_sheet_kept: float | None,
    off_sheet: float | None,
    argmax_on_sheet: float | None,
    swing_kept: float | None,
) -> bool | None:
    """Sheet gate. ``None`` if this fixture cannot see a sheet."""
    fields = (on_sheet_kept, off_sheet, argmax_on_sheet, swing_kept)
    if any(v is None for v in fields):
        return None
    return (
        float(on_sheet_kept) >= COMPILED_SHEET_LOCK
        and float(off_sheet) <= COMPILED_GARBLE_MAX
        and float(argmax_on_sheet) >= COMPILED_ARGMAX_LOCK
        and float(swing_kept) >= COMPILED_SWING_FLOOR
    )


def concept_ok(
    *,
    swing_kept: float | None = None,
    intended_cos: float | None = None,
    content_cos: float | None = None,
    rich_kept: float | None = None,
    strength: float | None = None,
) -> bool | None:
    """Concept / extension still present. Missing columns are skipped."""
    checks: list[bool] = []
    if swing_kept is not None:
        checks.append(float(swing_kept) >= COMPILED_SWING_FLOOR)
    if intended_cos is not None:
        checks.append(float(intended_cos) >= COMPILED_CONCEPT_COS)
    if content_cos is not None:
        checks.append(float(content_cos) >= COMPILED_CONCEPT_COS)
    if rich_kept is not None:
        checks.append(float(rich_kept) >= RICH_KEPT_MIN)
    if strength is not None:
        checks.append(float(strength) >= COMPILED_STRENGTH_FLOOR)
    if not checks:
        return None
    return all(checks)


def cell_works(
    *,
    leak: float | None,
    on_sheet_kept: float | None = None,
    off_sheet: float | None = None,
    argmax_on_sheet: float | None = None,
    swing_kept: float | None = None,
    intended_cos: float | None = None,
    content_cos: float | None = None,
    rich_kept: float | None = None,
    strength: float | None = None,
    pair_odd_cos: float | None = None,
    collapse: float | None = None,
) -> bool | None:
    """One cell's gate. ``pair_odd_cos`` / ``collapse`` are accepted and ignored.

    ``None`` means the cell does not exist for this recipe (do not treat
    that as a pass).
    """
    del pair_odd_cos, collapse
    if leak is None and on_sheet_kept is None and intended_cos is None and strength is None:
        return None
    leak_pass = leak_ok(leak)
    sheet_pass = sheet_ok(
        on_sheet_kept=on_sheet_kept,
        off_sheet=off_sheet,
        argmax_on_sheet=argmax_on_sheet,
        swing_kept=swing_kept,
    )
    concept_pass = concept_ok(
        swing_kept=swing_kept,
        intended_cos=intended_cos,
        content_cos=content_cos,
        rich_kept=rich_kept,
        strength=strength,
    )
    if leak_pass is False:
        return False
    if sheet_pass is False:
        return False
    if concept_pass is False:
        return False
    # A cell with no leftover and no sheet still has to keep the concept
    # when that fixture can see one. If nothing is measurable, refuse to
    # invent a pass.
    if leak_pass is None and sheet_pass is None and concept_pass is None:
        return None
    return True


def compiled_verdict(
    *,
    leftover_works: bool | None,
    gender_works: bool | None,
    pair_odd_cos: float | None = None,
) -> str:
    """Join the leftover cell and the gender-like cell.

    ``pair_odd_cos`` is accepted and ignored so a caller cannot
    accidentally feed the logged lock into the compiled score.
    """
    del pair_odd_cos
    if leftover_works is True:
        return WORKS
    if gender_works is True:
        return WORKS_GENDER_ONLY
    return FAILS


def sort_rows(rows: list[dict]) -> list[dict]:
    """Verdict first, then live-exam evidence, leak, and on-sheet."""

    def key(row: dict) -> tuple:
        leak = row.get("leftover_leak")
        sheet = row.get("on_sheet")
        leak_key = float(leak) if _finite(leak) else 1e9
        sheet_key = -float(sheet) if _finite(sheet) else 1e9
        exam_key = 0 if row.get("live_run") else (1 if row.get("pair_kind") else 2)
        return (VERDICT_ORDER[row["compiled"]], exam_key, leak_key, sheet_key, row["id"])

    return sorted(rows, key=key)


def _sheet_map(rows: list[dict]) -> dict[str, dict]:
    return {row["name"]: row for row in rows}


def _pick(row: dict | None, key: str, default: Any = None) -> Any:
    if row is None:
        return default
    return row.get(key, default)


def _row(
    recipe_id: str,
    label: str,
    *,
    leftover: dict | None = None,
    gender: dict | None = None,
    leftover_leak: float | None = None,
    gender_leak: float | None = None,
    on_sheet: float | None = None,
    on_sheet_kept: float | None = None,
    off_sheet: float | None = None,
    argmax_on_sheet: float | None = None,
    swing_kept: float | None = None,
    pair_odd_cos: float | None = None,
    collapse: float | None = None,
    intended_cos: float | None = None,
    content_cos: float | None = None,
    trainer_c_plus: float | None = None,
    polarity: float | None = None,
    perc: float | None = None,
    loss: float | None = None,
    rich_kept: float | None = None,
    strength: float | None = None,
    pole_cos: float | None = None,
    pair_kind: str | None = None,
    live_run: str | None = None,
    expected_listen: str | None = None,
    one_token_kl: float | None = None,
    hidden_far: float | None = None,
    hidden_far_while_kl_small: bool | None = None,
    teacher_forced_kl: float | None = None,
    rollout_match: float | None = None,
    rollout_garble: float | None = None,
    off_caption_teacher: float | None = None,
    exam_pass: bool | None = None,
    fixture: str,
    notes: str = "",
) -> dict:
    leftover = leftover or {}
    gender = gender or {}
    leftover_leak = leftover_leak if leftover_leak is not None else leftover.get("leftover_leak")
    if leftover_leak is None:
        leftover_leak = leftover.get("leak_tok", leftover.get("leak_ratio", leftover.get("leak")))
    gender_leak = gender_leak if gender_leak is not None else gender.get("leak_tok", gender.get("leak_ratio", gender.get("leak")))
    on_sheet = on_sheet if on_sheet is not None else leftover.get("on_sheet", gender.get("on_sheet"))
    on_sheet_kept = (
        on_sheet_kept if on_sheet_kept is not None else leftover.get("on_sheet_kept", gender.get("on_sheet_kept"))
    )
    off_sheet = off_sheet if off_sheet is not None else leftover.get("garble", leftover.get("off_sheet", gender.get("garble")))
    argmax_on_sheet = (
        argmax_on_sheet
        if argmax_on_sheet is not None
        else leftover.get("argmax_on_sheet", gender.get("argmax_on_sheet"))
    )
    swing_kept = swing_kept if swing_kept is not None else leftover.get("swing_kept", gender.get("swing_kept"))
    pair_odd_cos = (
        pair_odd_cos
        if pair_odd_cos is not None
        else leftover.get("pair_odd_cos", leftover.get("c_plus", gender.get("pair_odd_cos")))
    )
    collapse = collapse if collapse is not None else leftover.get("collapse", leftover.get("cos_plus_minus", gender.get("collapse")))
    intended_cos = (
        intended_cos
        if intended_cos is not None
        else leftover.get("cos_intended", leftover.get("cos_slider_plus", gender.get("cos_intended")))
    )
    content_cos = content_cos if content_cos is not None else leftover.get("cos_concept", leftover.get("content_cos"))
    trainer_c_plus = trainer_c_plus if trainer_c_plus is not None else leftover.get("c_plus", leftover.get("pair_odd_cos"))
    polarity = polarity if polarity is not None else leftover.get("collapse", leftover.get("cos_plus_minus"))
    perc = perc if perc is not None else leftover.get("perc")
    loss = loss if loss is not None else leftover.get("loss")
    rich_kept = rich_kept if rich_kept is not None else leftover.get("rich_kept")
    strength = strength if strength is not None else leftover.get("strength", leftover.get("strength_on_u", gender.get("strength")))
    pole_cos = pole_cos if pole_cos is not None else leftover.get("pole_cos", leftover.get("pole_cos_plus"))

    leftover_pass = cell_works(
        leak=leftover_leak,
        on_sheet_kept=leftover.get("on_sheet_kept", on_sheet_kept if leftover else None),
        off_sheet=leftover.get("garble", off_sheet if leftover else None),
        argmax_on_sheet=leftover.get("argmax_on_sheet", argmax_on_sheet if leftover else None),
        swing_kept=leftover.get("swing_kept", swing_kept if leftover else None),
        intended_cos=leftover.get("cos_intended", leftover.get("cos_slider_plus")),
        content_cos=leftover.get("cos_concept"),
        rich_kept=leftover.get("rich_kept", rich_kept if leftover else None),
        strength=leftover.get("strength", leftover.get("strength_on_u")),
        pair_odd_cos=pair_odd_cos,
        collapse=collapse,
    )
    gender_pass = cell_works(
        leak=gender_leak,
        on_sheet_kept=gender.get("on_sheet_kept"),
        off_sheet=gender.get("garble"),
        argmax_on_sheet=gender.get("argmax_on_sheet"),
        swing_kept=gender.get("swing_kept"),
        intended_cos=gender.get("cos_intended", gender.get("cos_slider_plus")),
        content_cos=gender.get("cos_concept"),
        rich_kept=gender.get("rich_kept"),
        strength=gender.get("strength", gender.get("strength_on_u")),
        pair_odd_cos=gender.get("pair_odd_cos"),
        collapse=gender.get("collapse", gender.get("cos_plus_minus")),
    )
    # Recipes that only have leftover numbers should not inherit a
    # gender pass from a missing gender cell.
    if not gender and leftover:
        gender_pass = False if leftover_pass is False else gender_pass

    verdict = compiled_verdict(
        leftover_works=leftover_pass,
        gender_works=gender_pass,
        pair_odd_cos=pair_odd_cos,
    )
    if exam_pass is not None:
        if exam_pass and pair_kind == "divergent":
            leftover_pass, gender_pass, verdict = True, False, WORKS
        elif exam_pass and pair_kind == "close":
            leftover_pass, gender_pass, verdict = None, True, WORKS_GENDER_ONLY
        else:
            leftover_pass, gender_pass, verdict = (
                False if pair_kind == "divergent" else None,
                False,
                FAILS,
            )
    c_plus_distinct = (
        trainer_c_plus
        if trainer_c_plus is not None
        and pair_odd_cos is not None
        and abs(float(trainer_c_plus) - float(pair_odd_cos)) > 0.02
        else None
    )
    polarity_distinct = (
        polarity
        if polarity is not None
        and collapse is not None
        and abs(float(polarity) - float(collapse)) > 0.02
        else None
    )
    return {
        "id": recipe_id,
        "label": label,
        "compiled": verdict,
        "leftover_works": leftover_pass,
        "gender_works": gender_pass,
        "leftover_leak": leftover_leak,
        "gender_leak": gender_leak,
        "on_sheet": on_sheet,
        "on_sheet_kept": on_sheet_kept,
        "off_sheet": off_sheet,
        "argmax_on_sheet": argmax_on_sheet,
        "swing_kept": swing_kept,
        "pair_odd_cos": pair_odd_cos,
        "collapse": collapse,
        "intended_cos": intended_cos,
        "content_cos": content_cos,
        "trainer_c_plus": trainer_c_plus,
        "trainer_c_plus_distinct": c_plus_distinct,
        "polarity": polarity,
        "polarity_distinct": polarity_distinct,
        "perc": perc,
        "loss": loss,
        "rich_kept": rich_kept,
        "strength": strength,
        "pole_cos": pole_cos,
        "pair_kind": pair_kind,
        "live_run": live_run,
        "expected_listen": expected_listen,
        "one_token_kl": one_token_kl,
        "hidden_far": hidden_far,
        "hidden_far_while_kl_small": hidden_far_while_kl_small,
        "teacher_forced_kl": teacher_forced_kl,
        "rollout_match": rollout_match,
        "rollout_garble": rollout_garble,
        "off_caption_teacher": off_caption_teacher,
        "exam_pass": exam_pass,
        "fixture": fixture,
        "notes": notes,
        "gates": {
            "leak_lock": COMPILED_LEAK_LOCK,
            "sheet_lock": COMPILED_SHEET_LOCK,
            "garble_max": COMPILED_GARBLE_MAX,
            "argmax_lock": COMPILED_ARGMAX_LOCK,
            "swing_floor": COMPILED_SWING_FLOOR,
        },
    }


def collect_scoreboard(*, sheet_steps: int = 400, other_steps: int = 200, seed: int = 0) -> list[dict]:
    """Run the existing fixtures and join them into one table.

    Sheet recipes use the sheet cell (the only fixture that can see
    on-sheet mass). Hidden-geometry recipes keep their native leftover
    leak and inherit the sheet row of the same teacher when one exists,
    because the sheet is a property of the *target point*, not of the
    hidden width.
    """
    gender_sheet = _sheet_map(gender_cell(steps=sheet_steps, seed=seed))
    leftover_sheet = _sheet_map(leaky_cell(steps=sheet_steps, seed=seed))
    exam = _sheet_map(live_exam_cell(steps=sheet_steps, seed=seed))
    exam["faithful_sub_e_divergent_hidden"] = score_exam(
        "faithful_sub_e_divergent_hidden",
        divergent_pair_field(),
        pole_mode="hidden",
        target="faithful_sub_e",
        live_run=None,
        expected_listen=None,
        steps=sheet_steps,
        seed=seed,
    )
    leftover_sheet["v9_hold_e_l1"] = score_sheet(
        "v9_hold_e_l1",
        leaky_field(),
        pole_mode="hidden",
        teacher="pair_odd",
        leak_dir=leaky_field().leak_e(),
        hold_weight=1.0,
        steps=sheet_steps,
        seed=seed,
    )

    live = live_policy_table(steps=other_steps, seed=seed)
    live_gender = {row["name"]: row for row in live["gender"]}
    live_energy = {row["name"]: row for row in live["energy"]}

    faithful_raw = score_leak_lm(
        "lm_faithful_raw", target_mode="faithful", steps=other_steps, seed=seed
    )
    faithful_attrs = score_leak_lm(
        "lm_faithful_attrs",
        target_mode="faithful",
        with_attrs=True,
        steps=other_steps,
        seed=seed,
    )
    hub_hidden = score_leak_lm(
        "lm_v9_hub",
        target_mode="symmetric",
        leakage_floor=-0.9,
        anchor_weight=0.3,
        steps=other_steps,
        seed=seed,
    )

    rich = RichField()
    project_short = score_rich(
        "project_short",
        rich,
        project_odd=True,
        slider_dir=rich.short_u,
        steps=other_steps,
        seed=seed,
    )
    project_rich = score_rich(
        "project_rich",
        rich,
        project_odd=True,
        slider_dir=rich.rich_u,
        steps=other_steps,
        seed=seed,
    )
    pin_both = score_rich(
        "pin_both_faithful",
        RichField(pin_gender=True, pin_bpm=True),
        target_mode="faithful",
        steps=other_steps,
        seed=seed,
    )

    hold_raw_l1 = score_overlap_policy(
        "hold_e_raw_l1", overlap=0.0, hold_weight=1.0, ortho="raw", steps=other_steps, seed=seed
    )
    hold_raw_l8 = score_overlap_policy(
        "hold_e_raw_l8", overlap=0.0, hold_weight=8.0, ortho="raw", steps=other_steps, seed=seed
    )
    hold_perp_l1 = score_overlap_policy(
        "hold_e_perp_l1", overlap=0.0, hold_weight=1.0, ortho="slider", steps=other_steps, seed=seed
    )
    hold_perp_l8 = score_overlap_policy(
        "hold_e_perp_l8", overlap=0.0, hold_weight=8.0, ortho="slider", steps=other_steps, seed=seed
    )
    hold_raw_syn_l8 = score_overlap_policy(
        "hold_e_raw_syn_l8", overlap=0.5, hold_weight=8.0, ortho="raw", steps=other_steps, seed=seed
    )
    energy = energy_field()
    leftover_e = leftover_only_e(energy)
    highd_l1 = score_highd(
        "energy_highd_leftover_l1",
        energy,
        leak_dir=leftover_e,
        hold_weight=1.0,
        teacher="pair_odd",
        e_label="leftover-only ê",
        steps=max(other_steps, 300),
        seed=seed,
    )
    highd_l8 = score_highd(
        "energy_highd_leftover_l8",
        energy,
        leak_dir=leftover_e,
        hold_weight=LIVE_HOLD,
        teacher="pair_odd",
        e_label="leftover-only ê",
        steps=max(other_steps, 300),
        seed=seed,
    )
    highd_sub = score_highd(
        "energy_highd_sub_e_leftover",
        energy,
        leak_dir=leftover_e,
        hold_weight=0.0,
        teacher="pair_odd_sub_e",
        e_label="leftover-only ê_⊥ subtracted",
        steps=max(other_steps, 300),
        seed=seed,
    )
    highd_gender = score_highd(
        "gender_like_no_e",
        highd_gender_field(),
        leak_dir=None,
        hold_weight=0.0,
        teacher="pair_odd",
        e_label="none",
        steps=max(other_steps, 300),
        seed=seed,
    )

    def sheet_left(name: str | None) -> dict:
        return dict(leftover_sheet[name]) if name and name in leftover_sheet else {}

    def sheet_gen(name: str | None) -> dict:
        return dict(gender_sheet[name]) if name and name in gender_sheet else {}

    def exam_row(name: str, label: str, notes: str) -> dict:
        row = exam[name]
        return _row(
            row["name"],
            label,
            pair_odd_cos=row["pair_odd_cos"],
            collapse=row["collapse"],
            pair_kind=row["pair_kind"],
            live_run=row["live_run"],
            expected_listen=row["expected_listen"],
            one_token_kl=row["one_token_kl"],
            hidden_far=row["hidden_far"],
            hidden_far_while_kl_small=row["hidden_far_while_kl_small"],
            teacher_forced_kl=row["teacher_forced_kl"],
            rollout_match=row["rollout_match"],
            rollout_garble=row["rollout_garble"],
            off_caption_teacher=row["off_caption_teacher"],
            exam_pass=row["pass"],
            fixture=f"live exam {row['pair_kind']} pair + frozen continuation",
            notes=notes,
        )

    rows = [
        exam_row(
            "energy_v18_semantic_kl_faithful",
            "semantic_kl + faithful / divergent pair",
            "Predicts energy-lm-v18 PASS. Raw poles retain intended genre/BPM track identity.",
        ),
        exam_row(
            "energy_v16_semantic_kl_sub_e",
            "semantic_kl + faithful_sub_e / divergent pair",
            "Predicts energy-lm-v16 FAIL. The declared ê is most of the intended two-track difference, so subtraction trains a blend teacher.",
        ),
        exam_row(
            "gender_v16_semantic_kl_faithful",
            "semantic_kl + faithful / close pair",
            "Predicts gender-lm-v16 FAIL: KL-small / hidden-far / rollout garble.",
        ),
        exam_row(
            "gender_hidden_faithful_next",
            "hidden + faithful / close pair (next card)",
            "Untrained control: hidden MSE copies the continuation-bearing plan and passes the close-pair rollout.",
        ),
        exam_row(
            "faithful_sub_e_divergent_hidden",
            "hidden + faithful_sub_e / divergent pair",
            "Target-side control: changing KL to MSE cannot repair a teacher made into a third-song blend by subtracting intended track identity.",
        ),
        _row(
            "faithful_raw",
            "faithful / v6 raw poles",
            leftover=sheet_left("v6_faithful"),
            gender=sheet_gen("v6_faithful"),
            leftover_leak=sheet_left("v6_faithful").get("leak_tok"),
            fixture="sheet leftover + sheet gender",
            notes="raw-pole MSE. Caption target; unused ê rides along.",
        ),
        _row(
            "faithful_attrs",
            "faithful + attributes / pin unused",
            leftover={
                **sheet_gen("v6_faithful"),
                "leak_tok": float(faithful_attrs["leak_ratio"]),
                "leak_ratio": float(faithful_attrs["leak_ratio"]),
                "rich_kept": float(pin_both["rich_kept"]),
                "cos_intended": float(faithful_attrs.get("cos_slider_plus", 1.0)),
            },
            gender=sheet_gen("v6_faithful"),
            leftover_leak=float(faithful_attrs["leak_ratio"]),
            intended_cos=float(faithful_attrs.get("cos_slider_plus", 1.0)),
            rich_kept=float(pin_both["rich_kept"]),
            fixture="Field2D attrs + rich pin-both + gender sheet",
            notes="data fix: unused gender/BPM pinned in the captions. Poles become the gender-like sheet.",
        ),
        _row(
            "pair_odd_midpoint",
            "pair-odd / v9 hidden midpoint",
            leftover=sheet_left("v9_hidden"),
            gender=sheet_gen("v9_hidden"),
            leftover_leak=sheet_left("v9_hidden").get("leak_tok"),
            fixture="sheet leftover + sheet gender",
            notes="live --lm_target v9 teacher t± = h0 ± a. Not a caption.",
        ),
        _row(
            "hidden_beta1",
            "pair-odd β=1 / symmetric --common_beta 1",
            leftover={
                **sheet_left("v6_faithful"),
                "leak_tok": sheet_left("v6_faithful").get("leak_tok"),
            },
            gender=sheet_gen("hidden_beta1"),
            leftover_leak=sheet_left("v6_faithful").get("leak_tok"),
            fixture="sheet leftover faithful ≡ β=1 + gender hidden_beta1",
            notes="lm_hidden_targets(symmetric, β=1) is the raw poles. Sheet-good, still leaks ê.",
        ),
        _row(
            "hub",
            "hub (published floor + anchor)",
            leftover={
                **sheet_left("v9_hidden"),
                "leak_tok": float(live_energy["hub"]["leak_ratio"]),
                "leak_ratio": float(live_energy["hub"]["leak_ratio"]),
                "cos_intended": float(live_energy["hub"]["cos_intended"]),
                "strength": float(live_energy["hub"]["strength"]),
            },
            gender={
                **sheet_gen("v9_hidden"),
                "leak_tok": float(live_gender["hub"]["leak_ratio"]),
                "cos_intended": float(live_gender["hub"]["cos_intended"]),
                "strength": float(live_gender["hub"]["strength"]),
            },
            leftover_leak=float(live_energy["hub"]["leak_ratio"]),
            gender_leak=float(live_gender["hub"]["leak_ratio"]),
            intended_cos=float(live_energy["hub"]["cos_intended"]),
            strength=float(live_energy["hub"]["strength"]),
            fixture="live energy/gender + pair-odd sheet",
            notes="same odd teacher as pair-odd; even blend-back does not take ê out of a. Sheet inherited from the midpoint.",
        ),
        _row(
            "hold_e_raw_l1",
            "hold-ê raw λ=1",
            leftover={
                **sheet_left("v9_hidden"),
                "leak_tok": float(hold_raw_l1["leak_ratio"]),
                "leak_ratio": float(hold_raw_l1["leak_ratio"]),
                "cos_intended": float(hold_raw_l1["cos_intended"]),
                "c_plus": float(hold_raw_l1["cos_teacher"]),
                "perc": float(hold_raw_l1["perc"]),
                "loss": float(hold_raw_l1["loss"]),
                "collapse": float(hold_raw_l1["collapse"]),
            },
            gender=sheet_gen("v9_hidden"),
            leftover_leak=float(hold_raw_l1["leak_ratio"]),
            fixture="overlap leftover raw + midpoint sheet",
            notes="λ=1 is too soft on leftover ê. Teacher is still the midpoint.",
        ),
        _row(
            "hold_e_raw_l8",
            "hold-ê raw λ=8 (leftover ê)",
            leftover={
                **sheet_left("v9_hold_e"),
                "leak_tok": float(hold_raw_l8["leak_ratio"]),
                "leak_ratio": float(hold_raw_l8["leak_ratio"]),
                "cos_intended": float(hold_raw_l8["cos_intended"]),
                "c_plus": float(hold_raw_l8["cos_teacher"]),
                "perc": float(hold_raw_l8["perc"]),
                "loss": float(hold_raw_l8["loss"]),
            },
            gender=sheet_gen("v9_hidden"),
            leftover_leak=float(hold_raw_l8["leak_ratio"]),
            fixture="overlap leftover raw λ=8 + hold-ê sheet",
            notes="on leftover ê, raw ≡ ê_⊥û. Leak can look small; sheet is still the midpoint.",
        ),
        _row(
            "hold_e_perp_l1",
            "hold-ê ê_⊥û λ=1",
            leftover={
                **sheet_left("v9_hold_e_l1"),
                "leak_tok": float(hold_perp_l1["leak_ratio"]),
                "cos_intended": float(hold_perp_l1["cos_intended"]),
                "c_plus": float(hold_perp_l1["cos_teacher"]),
                "perc": float(hold_perp_l1["perc"]),
                "loss": float(hold_perp_l1["loss"]),
            },
            gender=sheet_gen("v9_hidden"),
            leftover_leak=float(hold_perp_l1["leak_ratio"]),
            fixture="overlap leftover ê_⊥û λ=1 + sheet hold λ=1",
            notes="live-weak λ. Sheet still aims at h0 ± a.",
        ),
        _row(
            "hold_e_perp_l8",
            "hold-ê ê_⊥û λ=8 (live v9)",
            leftover=sheet_left("v9_hold_e"),
            gender=sheet_gen("v9_hidden"),
            leftover_leak=sheet_left("v9_hold_e").get("leak_tok"),
            intended_cos=float(hold_perp_l8["cos_intended"]),
            trainer_c_plus=float(hold_perp_l8["cos_teacher"]),
            perc=float(hold_perp_l8["perc"]),
            loss=float(hold_perp_l8["loss"]),
            fixture="sheet leftover hold-ê + overlap ê_⊥û λ=8",
            notes="current leftover-ê default. Fixes unused-axis leak, not the sheet.",
        ),
        _row(
            "hold_e_raw_synonym_l8",
            "hold-ê raw λ=8 (synonym ê)",
            leftover={
                **sheet_left("v9_hidden"),
                "leak_tok": float(hold_raw_syn_l8["leak_ratio"]),
                "cos_intended": float(hold_raw_syn_l8["cos_intended"]),
                "c_plus": float(hold_raw_syn_l8["cos_teacher"]),
                "perc": float(hold_raw_syn_l8["perc"]),
                "loss": float(hold_raw_syn_l8["loss"]),
            },
            gender=sheet_gen("v9_hidden"),
            leftover_leak=float(hold_raw_syn_l8["leak_ratio"]),
            fixture="overlap synonym raw λ=8 + midpoint sheet",
            notes="ê restates the poles. Raw hold punches the slider.",
        ),
        _row(
            "pair_odd_sub_e",
            "pair_odd_sub_e (#20, midpoint − ê_⊥)",
            leftover=sheet_left("v15_pair_odd_sub_e"),
            gender=sheet_gen("v9_hidden"),
            leftover_leak=sheet_left("v15_pair_odd_sub_e").get("leak_tok"),
            content_cos=float(highd_sub.get("cos_intended", 0.0)),
            trainer_c_plus=float(highd_sub.get("c_plus", 0.0)),
            perc=float(highd_sub.get("perc", 0.0)),
            loss=float(highd_sub.get("loss", 0.0)),
            fixture="sheet leftover pair_odd_sub_e + high-D leftover subtract",
            notes="λ→∞ hold in one step. Leak 0; further off-caption than pair-odd.",
        ),
        _row(
            "faithful_sub_e_unused_e",
            "faithful_sub_e / unused-ê same-song diagnostic",
            leftover=sheet_left("faithful_sub_e"),
            leftover_leak=sheet_left("faithful_sub_e").get("leak_tok"),
            fixture="sheet leftover",
            notes="Retained old cell: ê is unused gender inside one clean pair. Not the divergent energy-v4 stand-in.",
        ),
        _row(
            "semantic_kl_midpoint",
            "semantic_kl onto midpoint",
            leftover=sheet_left("kl_on_midpoint"),
            gender=sheet_gen("kl_on_midpoint"),
            leftover_leak=sheet_left("kl_on_midpoint").get("leak_tok"),
            fixture="sheet leftover + sheet gender",
            notes="KL is not the fix. The target point is.",
        ),
        _row(
            "semantic_kl_poles_unused_e",
            "semantic_kl poles / unused-ê same-song diagnostic",
            leftover=sheet_left("v16_semantic_kl"),
            gender=sheet_gen("v16_semantic_kl"),
            leftover_leak=sheet_left("v16_semantic_kl").get("leak_tok"),
            fixture="sheet leftover + sheet gender",
            notes="Retained old cell: unused gender moves the leak token. Energy-v4 and gender-v4 are scored in separate pair-geometry rows above.",
        ),
        _row(
            "semantic_kl_sub_e_unused_e",
            "semantic_kl sub_e / unused-ê same-song diagnostic",
            leftover=sheet_left("v16_semantic_kl_sub_e"),
            leftover_leak=sheet_left("v16_semantic_kl_sub_e").get("leak_tok"),
            fixture="sheet leftover",
            notes="Retained old cell: passes when ê really is unused inside a clean pair; not an energy-v4 proxy.",
        ),
        _row(
            "project_short_u",
            "project short û",
            leftover={
                **sheet_left("v9_hidden"),
                "leak_ratio": float(project_short["leak_ratio"]),
                "cos_intended": float(project_short["cos_intended"]),
                "rich_kept": float(project_short["rich_kept"]),
                "cos_plus_minus": float(project_short["cos_plus_minus"]),
                "strength": float(live_gender["always_project_hold"]["strength"]),
            },
            gender={
                **sheet_gen("v9_hidden"),
                "leak_tok": float(live_gender["always_project_hold"]["leak_ratio"]),
                "cos_intended": float(live_gender["always_project_hold"]["cos_intended"]),
                "strength": float(live_gender["always_project_hold"]["strength"]),
            },
            leftover_leak=float(project_short["leak_ratio"]),
            gender_leak=float(live_gender["always_project_hold"]["leak_ratio"]),
            rich_kept=float(project_short["rich_kept"]),
            strength=float(live_gender["always_project_hold"]["strength"]),
            intended_cos=float(project_short["cos_intended"]),
            fixture="rich project_short + live gender always_project_hold",
            notes="leak-0 by dropping everything ⊥ short û, including the singer and slider adjectives. Still a midpoint (c deleted); sheet inherited from pair-odd.",
        ),
        _row(
            "project_rich_u",
            "project rich û (oracle intended span)",
            leftover={
                **sheet_left("v9_hidden"),
                "leak_ratio": float(project_rich["leak_ratio"]),
                "cos_intended": float(project_rich["cos_intended"]),
                "rich_kept": float(project_rich["rich_kept"]),
                "cos_plus_minus": float(project_rich["cos_plus_minus"]),
            },
            leftover_leak=float(project_rich["leak_ratio"]),
            rich_kept=float(project_rich["rich_kept"]),
            intended_cos=float(project_rich["cos_intended"]),
            collapse=float(project_rich["cos_plus_minus"]),
            fixture="rich project_rich",
            notes="oracle û = span{short, slider adjectives}. Still a midpoint in the intended plane, so c is dropped; sheet inherited from pair-odd.",
        ),
        _row(
            "gender_like_no_e",
            "gender-like (no ê, hold 0)",
            leftover=None,
            gender={
                **sheet_gen("v9_hidden"),
                "c_plus": float(highd_gender.get("c_plus", 0.0)),
                "cos_intended": float(highd_gender.get("cos_intended", 0.0)),
                "leak_ratio": 0.0,
            },
            leftover_leak=None,
            gender_leak=sheet_gen("v9_hidden").get("leak_tok", 0.0),
            on_sheet=sheet_gen("v9_hidden").get("on_sheet"),
            on_sheet_kept=sheet_gen("v9_hidden").get("on_sheet_kept"),
            off_sheet=sheet_gen("v9_hidden").get("garble"),
            argmax_on_sheet=sheet_gen("v9_hidden").get("argmax_on_sheet"),
            swing_kept=sheet_gen("v9_hidden").get("swing_kept"),
            pair_odd_cos=sheet_gen("v9_hidden").get("pair_odd_cos"),
            collapse=sheet_gen("v9_hidden").get("collapse"),
            trainer_c_plus=float(highd_gender.get("c_plus", 0.0)),
            perc=float(highd_gender.get("perc", 0.0)),
            loss=float(highd_gender.get("loss", 0.0)),
            fixture="sheet gender v9_hidden + high-D gender_like_no_e",
            notes="clean pair, hold 0. Live gender-lm-v4 log. Midpoint still deletes c.",
        ),
        _row(
            "leftover_hold_l1",
            "energy-like leftover hold-ê λ=1 (high-D)",
            leftover={
                **sheet_left("v9_hold_e_l1"),
                "leftover_leak": float(highd_l1["leftover_leak"]),
                "cos_intended": float(highd_l1["cos_intended"]),
                "cos_concept": float(highd_l1["cos_intended"]),
                "c_plus": float(highd_l1["c_plus"]),
                "perc": float(highd_l1["perc"]),
                "loss": float(highd_l1["loss"]),
                "collapse": float(highd_l1["collapse"]),
            },
            leftover_leak=float(highd_l1["leftover_leak"]),
            content_cos=float(highd_l1["cos_intended"]),
            trainer_c_plus=float(highd_l1["c_plus"]),
            perc=float(highd_l1["perc"]),
            loss=float(highd_l1["loss"]),
            fixture="high-D leftover λ=1 + sheet hold λ=1",
            notes="one leftover caption pair cannot name the whole unused remainder.",
        ),
        _row(
            "leftover_hold_l8",
            "energy-like leftover hold-ê λ=8 (high-D)",
            leftover={
                **sheet_left("v9_hold_e"),
                "leftover_leak": float(highd_l8["leftover_leak"]),
                "cos_intended": float(highd_l8["cos_intended"]),
                "c_plus": float(highd_l8["c_plus"]),
                "perc": float(highd_l8["perc"]),
                "loss": float(highd_l8["loss"]),
                "collapse": float(highd_l8["collapse"]),
            },
            leftover_leak=float(highd_l8["leftover_leak"]),
            content_cos=float(highd_l8["cos_intended"]),
            trainer_c_plus=float(highd_l8["c_plus"]),
            perc=float(highd_l8["perc"]),
            loss=float(highd_l8["loss"]),
            fixture="high-D leftover λ=8 + sheet hold-ê",
            notes="λ=8 barely moves leftover leak vs λ=1. Sheet still the midpoint.",
        ),
    ]
    del hub_hidden
    return sort_rows(rows)


def gates_blob() -> dict:
    return {
        "leak_lock": COMPILED_LEAK_LOCK,
        "sheet_lock": COMPILED_SHEET_LOCK,
        "garble_max": COMPILED_GARBLE_MAX,
        "argmax_lock": COMPILED_ARGMAX_LOCK,
        "swing_floor": COMPILED_SWING_FLOOR,
        "concept_cos": COMPILED_CONCEPT_COS,
        "strength_floor": COMPILED_STRENGTH_FLOOR,
        "rich_kept_min": RICH_KEPT_MIN,
        "pair_odd_cos_scored": False,
        "collapse_scored": False,
        "rollout_match_min": ROLLOUT_MATCH_MIN,
        "rollout_garble_max": ROLLOUT_GARBLE_MAX,
        "off_caption_max": OFF_CAPTION_MAX,
        "prose": (
            "Legacy unused-attribute cells require leftover leak to be small "
            f"(≤ {COMPILED_LEAK_LOCK}) AND, when a sheet readout exists, "
            f"on-sheet kept ≥ {COMPILED_SHEET_LOCK}, off-sheet mass ≤ "
            f"{COMPILED_GARBLE_MAX}, argmax-on-sheet = {COMPILED_ARGMAX_LOCK:g}, "
            f"and concept swing kept ≥ {COMPILED_SWING_FLOOR}. "
            "Live-exam pair cells instead require the target to remain a pole "
            f"(off-caption ≤ {OFF_CAPTION_MAX}) "
            f"and greedy rollout match must be ≥ {ROLLOUT_MATCH_MIN} with "
            f"garble ≤ {ROLLOUT_GARBLE_MAX}. "
            "High pair-odd cos and ±1 collapse are logged, never scored. "
            "works-on-gender-only means the gender-like cell (no leftover ê) "
            "passes that same gate and the leftover cell does not."
        ),
    }


def compile_sheet_row(leftover: dict, gender: dict | None = None) -> dict:
    """Apply the compiled gate to a sheet leftover row (and optional gender row).

    Tests use this so the four locked recipes are scored by the same
    function the table uses, without re-running the hidden-geometry
    fixtures.
    """
    return _row(
        leftover["name"],
        leftover["name"],
        leftover=leftover,
        gender=gender,
        leftover_leak=leftover.get("leak_tok"),
        gender_leak=(gender or {}).get("leak_tok"),
        fixture="sheet",
    )


def floatable_row(row: dict) -> dict:
    out = {}
    for key, value in row.items():
        if key == "gates":
            out[key] = value
            continue
        if isinstance(value, (int, float, str, bool)) or value is None:
            out[key] = value
    return out
