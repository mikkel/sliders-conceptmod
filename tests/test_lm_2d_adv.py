"""RpGAN + b_cap 2-D slider cell.

The adversarial residual has to beat the supervised cells the scoreboard
already publishes: leftover leak on `faithful_raw`, off-sheet midpoint on
`pair_odd`, and the pair-exam continuation gates. CPU only. Does not
change the live trainer default.
"""

from __future__ import annotations

import pytest
import torch

from analysis.gan_bcap.gaussian_repro import hq_and_cover, mixture_means, train_gaussians
from analysis.slider2d.adv import (
    AdvConfig,
    cap_penalty,
    delayed_cosine,
    feature_match_loss,
    rp_d_loss,
    rp_g_loss,
)
from analysis.slider2d.exam import close_field, divergent_field, unused_e_field
from analysis.slider2d.gan import (
    default_cfg,
    score_adv_exam,
    score_adv_sheet,
    score_field2d,
)
from analysis.slider2d.scoreboard import (
    COMPILED_GARBLE_MAX,
    COMPILED_LEAK_LOCK,
    COMPILED_SHEET_LOCK,
    COMPILED_SWING_FLOOR,
    WORKS,
    compiled_verdict,
    exam_score,
)
from analysis.slider2d.sheet import gender_like_field, leaky_field, score_sheet
from conceptmod.textsliders.train_lm_slider_music3 import parse_args


STEPS = 900
_CACHE: dict[str, object] = {}


def cfg() -> AdvConfig:
    return default_cfg(steps=STEPS, seed=0, b_cap=1.0, fm_weight=0.0)


def leftover() -> dict:
    if "leftover" not in _CACHE:
        _CACHE["leftover"] = score_adv_sheet(leaky_field(), cfg=cfg())
    return _CACHE["leftover"]


def gender() -> dict:
    if "gender" not in _CACHE:
        _CACHE["gender"] = score_adv_sheet(gender_like_field(), cfg=cfg())
    return _CACHE["gender"]


def field2d() -> dict:
    if "field2d" not in _CACHE:
        _CACHE["field2d"] = score_field2d(cfg())
    return _CACHE["field2d"]


def exam(cell: str) -> dict:
    if cell not in _CACHE:
        ctor = {
            "divergent": divergent_field,
            "close": close_field,
            "unused_e": unused_e_field,
        }[cell]
        _CACHE[cell] = score_adv_exam(ctor(seed=0), cfg=cfg())
    return _CACHE[cell]


# -- unit: the ParticleGAN formulae --------------------------------------


def test_rpgan_pair_logistic_is_antisymmetric():
    real = torch.tensor([1.2, 0.4])
    fake = torch.tensor([-0.3, 0.1])
    assert float(rp_d_loss(real, fake)) == pytest.approx(float(rp_g_loss(fake, real)))
    assert float(rp_d_loss(real, fake)) < float(rp_d_loss(fake, real))


def test_b_cap_is_zero_below_one_and_quadratic_above():
    low = torch.ones(4, 2) * 0.4
    assert float(cap_penalty(low, low, coeff=1.0)) == pytest.approx(0.0)
    high = torch.ones(4, 2)
    # ||∇|| = sqrt(2) ≈ 1.414; relu(0.414)^2 * 0.5 * (1+1) = 0.414^2
    got = float(cap_penalty(high, high, coeff=1.0))
    want = (2.0**0.5 - 1.0) ** 2
    assert got == pytest.approx(want, rel=1e-5)
    assert float(cap_penalty(high, high, coeff=2.0)) == pytest.approx(2.0 * want, rel=1e-5)


def test_normalized_feature_match_is_scale_invariant():
    a = torch.tensor([[1.0, 0.0], [1.0, 0.0]])
    b = torch.tensor([[0.0, 1.0], [0.0, 1.0]])
    raw = float(feature_match_loss(a, 10.0 * a, normalize=False))
    normed = float(feature_match_loss(a, 10.0 * a, normalize=True))
    assert raw > 0.0
    assert normed == pytest.approx(0.0, abs=1e-6)
    assert float(feature_match_loss(a, b, normalize=True)) > 0.5


def test_delayed_cosine_holds_then_decays():
    assert delayed_cosine(0, total=100, delay=20) == 1.0
    assert delayed_cosine(19, total=100, delay=20) == 1.0
    assert delayed_cosine(100, total=100, delay=20, min_ratio=0.05) == pytest.approx(0.05)
    mid = delayed_cosine(60, total=100, delay=20, min_ratio=0.05)
    assert 0.05 < mid < 1.0


def test_eight_gaussians_cover_modes_with_b_cap():
    row = train_gaussians(n_modes=8, steps=700, seed=1234, b_cap=1.0)
    assert row["modes"] >= 6
    assert row["hq"] >= 0.50
    means = mixture_means(8)
    dead = hq_and_cover(torch.zeros(200, 2), means, 0.05)
    assert dead["modes"] == 0


# -- live default stays put ----------------------------------------------


def test_live_default_is_still_v9():
    args = parse_args(["--prompts", "x.yaml"])
    assert args.lm_target == "v9"
    assert args.pole_mode == "hidden"


# -- Field2D polarity / leftover -----------------------------------------


def test_field2d_tracks_the_slider_without_gender_leak():
    row = field2d()
    assert row["cos_slider_plus"] >= 0.90
    assert abs(row["leak_ratio"]) <= 0.20
    assert row["cos_plus_minus"] <= -0.85
    assert row["leak_frac"] <= -0.85
    assert row["pass"]


# -- sheet: beat faithful_raw leak, stay on caption ----------------------


def test_leftover_sheet_beats_faithful_raw_leak():
    gan = leftover()
    raw = score_sheet(
        "v6_faithful",
        leaky_field(),
        pole_mode="hidden",
        teacher="faithful",
        steps=400,
        seed=0,
    )
    assert abs(raw["leak_tok"]) > COMPILED_LEAK_LOCK
    assert abs(gan["leak_tok"]) <= COMPILED_LEAK_LOCK
    assert gan["on_sheet_kept"] >= COMPILED_SHEET_LOCK
    assert gan["garble"] <= COMPILED_GARBLE_MAX
    assert gan["argmax_on_sheet"] == 1.0
    assert gan["swing_kept"] >= COMPILED_SWING_FLOOR
    assert gan["pass"]
    # Midpoint teacher still fails the sheet — the GAN must not become v9.
    mid = score_sheet(
        "v9_hidden",
        leaky_field(),
        pole_mode="hidden",
        teacher="pair_odd",
        steps=400,
        seed=0,
    )
    assert mid["on_sheet_kept"] < COMPILED_SHEET_LOCK
    assert gan["on_sheet_kept"] > mid["on_sheet_kept"] + 0.3


def test_gender_sheet_stays_on_caption():
    row = gender()
    assert row["on_sheet_kept"] >= COMPILED_SHEET_LOCK
    assert row["garble"] <= COMPILED_GARBLE_MAX
    assert row["argmax_on_sheet"] == 1.0
    assert row["swing_kept"] >= COMPILED_SWING_FLOOR
    assert row["pass"]


# -- pair-exam -----------------------------------------------------------


def test_exam_divergent_and_close_both_pass():
    div = exam("divergent")
    close = exam("close")
    unused = exam("unused_e")
    assert div["pass"], div.get("reason")
    assert close["pass"], close.get("reason")
    assert unused["pass"], unused.get("reason")
    assert div["roll_overlap"] >= 0.85
    assert close["roll_overlap"] >= 0.85
    assert div["roll_swing_kept"] >= 0.60
    score = exam_score(
        {
            "exam_divergent": div["roll_overlap"],
            "exam_close": close["roll_overlap"],
        },
        {
            "exam_divergent": div["roll_swing_kept"],
            "exam_close": close["roll_swing_kept"],
        },
    )
    assert score is not None and score >= 0.85
    cells = {
        "exam_divergent": True,
        "exam_close": True,
        "exam_unused_e": True,
        "sheet_leftover": leftover()["pass"],
        "sheet_gender": gender()["pass"],
    }
    assert compiled_verdict(cells=cells) == WORKS
