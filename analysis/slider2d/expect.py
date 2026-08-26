"""One sortable expected-quality number per UNI recipe.

``expect`` is how likely a recipe is to work live on **both** grit
lyrics and gender. Higher is better. It is not ``exam_score``,
``leak_frac``, last-50 ``c+``, or ``p%``, and it is not folded into
the compiled bipolar board.

Bottleneck first: the live listen has to survive the worst of grit
lyric_recall, gender_move, both covers, and neu_hold. A recipe that
saves lyrics by abandoning ``h+`` (cover collapse) is therefore
ranked below a recipe that keeps cover, even if both HIT the lyric
gate.

Miss penalties keep a near-gate 0.84 from looking like a 0.84 cover
that also missed the other axis. A small documented tie-break from
residual cover / neu_hold / leftover off-caption then splits two
HITs so they do not share a value.
"""

from __future__ import annotations

from analysis.slider2d.exam import EXAM_ROLL_OVERLAP
from analysis.slider2d.lyric_gender import (
    EXPECT_RECIPES,
    REQUIRED_RECIPES,
    lyric_gender_table,
)
from analysis.slider2d.plus_exam import PLUS_COVER_MIN, PLUS_OFF_MAX
from analysis.slider2d.plus_neu_exam import PLUS_NEU_HOLD_MIN

EXPECT_GATE = float(EXAM_ROLL_OVERLAP)  # 0.85, same family as every UNI board
MISS_WEIGHT = 0.15
TIE_WEIGHT = 0.02


def _cover_digits(grit_cover: float, gender_cover: float) -> float:
    """Leftover cover digits below 1e-3, folded into ``(0, 1)``.

    Two HITs that share a 3-decimal bottleneck still have different
    raw covers. This is that residual, not a new gate.
    """
    packed = (float(grit_cover) * 1.0e7 + float(gender_cover) * 3.0e6) % 1.0
    return packed


def expect_score(
    *,
    grit_lyric_recall: float,
    gender_move: float,
    grit_cover: float,
    gender_cover: float,
    neu_hold: float,
    grit_off_caption: float = 0.0,
    gender_off_caption: float = 0.0,
    caption_move: float = 0.0,
    lyric_pin: float = 0.0,
) -> dict:
    """One number. Same inputs the published table shows.

    Improvement on the suggested shape: keep ``min(...)`` and the two
    0.15 miss terms, then add a *small* residual from leftover cover,
    leftover neu_hold, unused off-caption slack, caption_move, and
    lyric_pin. Those leftovers cannot flip a gate miss (0.15) or a
    0.05 bottleneck gap; they only stop two HITs from printing the
    same 1.000. A 1e-4 fold of leftover cover digits below 1e-3 is
    the last uniqueness term.
    """
    grit_lyric = float(grit_lyric_recall)
    move = float(gender_move)
    g_cover = float(grit_cover)
    c_cover = float(gender_cover)
    hold = float(neu_hold)
    cap = max(0.0, min(1.0, float(caption_move)))
    pin = max(0.0, min(1.0, float(lyric_pin)))
    mean_off = 0.5 * (float(grit_off_caption) + float(gender_off_caption))
    mean_cover = 0.5 * (g_cover + c_cover)
    bottleneck = min(grit_lyric, move, g_cover, c_cover, hold)
    miss_lyric = max(0.0, EXPECT_GATE - grit_lyric)
    miss_gender = max(0.0, EXPECT_GATE - move)
    miss = MISS_WEIGHT * miss_lyric + MISS_WEIGHT * miss_gender
    leftover_cover = max(0.0, mean_cover - bottleneck)
    leftover_hold = max(0.0, hold - bottleneck)
    # PLUS_OFF_MAX is 0.05. Slack is how much unused off-caption room
    # is left; a dirty HIT loses a few thousandths to a clean HIT.
    off_slack = max(0.0, float(PLUS_OFF_MAX) - mean_off)
    digits = _cover_digits(g_cover, c_cover)
    tie = (
        0.35 * leftover_cover
        + 0.20 * leftover_hold
        + 0.15 * off_slack
        + 0.15 * cap
        + 0.10 * pin
        + 0.05 * digits
    )
    value = bottleneck - miss + TIE_WEIGHT * tie
    return {
        "expect": value,
        "bottleneck": bottleneck,
        "miss": miss,
        "miss_lyric": miss_lyric,
        "miss_gender": miss_gender,
        "tie": tie,
        "leftover_cover": leftover_cover,
        "leftover_hold": leftover_hold,
        "off_slack": off_slack,
        "caption_move": cap,
        "lyric_pin": pin,
        "cover_digits": digits,
        "grit_lyric_recall": grit_lyric,
        "gender_move": move,
        "grit_cover": g_cover,
        "gender_cover": c_cover,
        "cover": mean_cover,
        "neu_hold": hold,
        "off_caption": mean_off,
        "lyric_gate": bool(grit_lyric >= EXPECT_GATE),
        "gender_gate": bool(move >= EXPECT_GATE),
        "cover_gate": bool(g_cover >= PLUS_COVER_MIN and c_cover >= PLUS_COVER_MIN),
        "neu_hold_gate": bool(hold >= PLUS_NEU_HOLD_MIN),
    }


def _row_pair(table: dict[str, list[dict]], name: str) -> tuple[dict, dict]:
    by = {cell: {r["name"]: r for r in rows} for cell, rows in table.items()}
    if "divergent" not in by or "close" not in by:
        raise KeyError("expect board needs both divergent and close cells")
    if name not in by["divergent"] or name not in by["close"]:
        raise KeyError(f"recipe {name!r} missing from one of the cells")
    return by["divergent"][name], by["close"][name]


def expect_row(grit: dict, close: dict) -> dict:
    """Combine the two cells into the columns the page sorts on."""
    move = close.get("gender_move")
    if move is None:
        move = 0.0
    neu = min(float(grit["neu_hold"]), float(close["neu_hold"]))
    scored = expect_score(
        grit_lyric_recall=float(grit["lyric_recall"]),
        gender_move=float(move),
        grit_cover=float(grit["cover"]),
        gender_cover=float(close["cover"]),
        neu_hold=neu,
        grit_off_caption=float(grit["off_caption"]),
        gender_off_caption=float(close["off_caption"]),
        caption_move=0.5
        * (float(grit.get("caption_move", 0.0)) + float(close.get("caption_move", 0.0))),
        lyric_pin=0.5
        * (float(grit.get("lyric_pin", 0.0)) + float(close.get("lyric_pin", 0.0))),
    )
    want = bool(
        scored["lyric_gate"]
        and scored["gender_gate"]
        and scored["cover_gate"]
        and scored["neu_hold_gate"]
    )
    return {
        "name": grit["name"],
        "expect": scored["expect"],
        "bottleneck": scored["bottleneck"],
        "miss": scored["miss"],
        "tie": scored["tie"],
        "grit_lyric_recall": scored["grit_lyric_recall"],
        "gender_move": scored["gender_move"],
        "grit_cover": scored["grit_cover"],
        "gender_cover": scored["gender_cover"],
        "cover": scored["cover"],
        "neu_hold": scored["neu_hold"],
        "off_caption": scored["off_caption"],
        "caption_move": scored["caption_move"],
        "lyric_pin": scored["lyric_pin"],
        "want_box": want,
        "grit_sings_lyric": grit.get("sings_lyric", ""),
        "gender_reads_vocal": close.get("reads_vocal_plus", ""),
        "grit": grit,
        "close": close,
        **{k: scored[k] for k in ("lyric_gate", "gender_gate", "cover_gate", "neu_hold_gate")},
    }


def expect_rank(table: dict[str, list[dict]]) -> list[dict]:
    """Sort by expect descending. Names stay stable on a true tie."""
    ranked = []
    for cand in EXPECT_RECIPES:
        grit, close = _row_pair(table, cand["name"])
        row = expect_row(grit, close)
        row["is"] = cand["is"]
        ranked.append(row)
    ranked.sort(key=lambda r: (-float(r["expect"]), str(r["name"])))
    for i, row in enumerate(ranked, start=1):
        row["rank"] = i
    return ranked


def expect_values_are_unique(ranked: list[dict], *, places: int = 4) -> bool:
    seen = [round(float(r["expect"]), places) for r in ranked]
    return len(seen) == len(set(seen))


def expect_table(*, steps: int = 400, seed: int = 0) -> dict[str, list[dict]]:
    return lyric_gender_table(steps=steps, seed=seed)


def expect_verdict(table: dict[str, list[dict]]) -> dict:
    ranked = expect_rank(table)
    names = [r["name"] for r in ranked]
    missing = [n for n in REQUIRED_RECIPES if n not in names]
    winner = ranked[0]
    return {
        "winner": winner["name"],
        "winner_expect": winner["expect"],
        "ranked_names": names,
        "unique_expect": expect_values_are_unique(ranked),
        "missing_required": missing,
        "not_the_bipolar_board": True,
        "scored_on": (
            "expect",
            "grit_lyric_recall",
            "gender_move",
            "cover",
            "neu_hold",
        ),
        "not_scored": ("exam_score", "leak_frac", "c+", "p%"),
        "gates": {
            "lyric_recall": EXPECT_GATE,
            "gender_move": EXPECT_GATE,
            "cover": PLUS_COVER_MIN,
            "neu_hold": PLUS_NEU_HOLD_MIN,
        },
        "formula": {
            "bottleneck": "min(grit_lyric_recall, gender_move, grit_cover, gender_cover, neu_hold)",
            "miss_weight": MISS_WEIGHT,
            "tie_weight": TIE_WEIGHT,
        },
    }
