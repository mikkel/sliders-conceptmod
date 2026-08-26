"""Even leftover: subtract/hold leftover even, keep caption even.

CPU geometry only. Does not change the live trainer default.
"""

from __future__ import annotations

import torch

from analysis.slider2d.exam import close_field, divergent_field, unused_e_field
from analysis.slider2d.sheet import leaky_field
from conceptmod.textsliders.slider_targets import (
    UNUSED_E_OVERLAP_MAX,
    leftover_bipolar,
    lm_blend_guard,
    lm_e_is_unused,
    lm_even_leftover_dir,
    lm_even_residual,
    lm_faithful_gate_odd_sub_even,
    lm_faithful_sub_e_if_unused,
    lm_faithful_sub_even_blend,
    lm_faithful_sub_even_e,
    lm_faithful_sub_even_e_if_unused,
    lm_hidden_targets,
    lm_unit,
)


def _pair(field, row: int = 0):
    return field.poles(row)


def test_even_subtract_is_not_pair_odd():
    """Dropping even-along-ê is not t± = h0 ± a."""
    field = leaky_field()
    pos, neg, neu = _pair(field)
    plus, minus = lm_faithful_sub_even_e(
        pos, neg, neu, field.leak_e(), slider_dir=field.short_u()
    )
    pair_plus, _pair_minus = lm_hidden_targets(pos, neg, neu, target_mode="symmetric")
    assert not torch.allclose(plus, pair_plus, atol=1e-5)
    even = lm_even_residual(plus, minus, neu)
    # Caption even (ŝ) stays. Leftover sheet even is ⊥ ê, so even is intact.
    assert torch.allclose(
        even @ field.sheet_dir(), field.common_vec(0) @ field.sheet_dir(), atol=1e-5
    )


def test_leftover_sheet_even_is_caption_not_e():
    """On the leftover sheet, c is ŝ and ê is unused gender — even · ê = 0."""
    field = leaky_field()
    pos, neg, neu = _pair(field)
    even = lm_even_residual(pos, neg, neu)
    assert abs(float(even @ lm_unit(field.leak_e()))) < 1e-6
    plus, minus = lm_faithful_sub_even_e(
        pos, neg, neu, field.leak_e(), slider_dir=field.short_u()
    )
    assert torch.allclose(plus, pos, atol=1e-6)
    assert torch.allclose(minus, neg, atol=1e-6)


def test_unused_gate_keeps_poles_when_e_restates_the_tracks():
    field = divergent_field()
    pos, neg, neu = _pair(field)
    e = field.declared_e()
    unused, overlap = lm_e_is_unused(pos, neg, e, slider_dir=field.short_u())
    assert overlap > UNUSED_E_OVERLAP_MAX
    assert unused is False
    plus, minus = lm_faithful_sub_even_e_if_unused(
        pos, neg, neu, e, slider_dir=field.short_u()
    )
    assert torch.allclose(plus, pos, atol=1e-6)


def test_unused_gate_takes_even_e_on_unused_leftover():
    field = unused_e_field()
    pos, neg, neu = _pair(field)
    e = field.declared_e()
    unused, overlap = lm_e_is_unused(pos, neg, e, slider_dir=field.short_u())
    assert unused is True
    assert float(overlap) < UNUSED_E_OVERLAP_MAX
    plus, minus = lm_faithful_sub_even_e_if_unused(
        pos, neg, neu, e, slider_dir=field.short_u()
    )
    # unused leftover lives in odd on this field, so even-along-ê is a no-op.
    assert torch.allclose(plus, pos, atol=1e-5)


def test_even_blend_matches_leak_pair_sum():
    field = divergent_field()
    neu = field.poles(0)[2]
    leak_plus, leak_minus = field.declared_e_poles(neu)
    even_dir = lm_even_leftover_dir(
        leak_plus, leak_minus, neu, slider_dir=field.short_u()
    )
    declared = field.declared_e_even()
    assert even_dir is not None
    assert declared is not None
    assert abs(float(lm_unit(even_dir) @ lm_unit(declared))) > 0.99


def test_even_blend_drops_track_sum_keeps_sheet():
    """Energy-v4 even is ½ track (p̂+q̂) + shared ŝ. Drop only the blend."""
    field = divergent_field()
    pos, neg, neu = _pair(field)
    even_dir = field.declared_e_even()
    plus, minus = lm_faithful_sub_even_blend(pos, neg, neu, even_dir)
    even = lm_even_residual(plus, minus, neu)
    odd = 0.5 * (plus - minus)
    # Sheet even stays.
    assert torch.allclose(
        even @ field.sheet_dir(),
        field.common_vec(0) @ field.sheet_dir(),
        atol=1e-4,
    )
    # Track-sum blend is gone.
    blend = field.plus_track() + field.minus_track()
    assert abs(float(even @ lm_unit(blend))) < 1e-4
    # Odd (including the track split) is untouched — not t± = h0 ± a.
    assert torch.allclose(odd, field.odd(0), atol=1e-5)
    pair_plus, _ = lm_hidden_targets(pos, neg, neu, target_mode="symmetric")
    assert not torch.allclose(plus, pair_plus, atol=1e-4)
    assert not torch.allclose(plus, pos, atol=1e-4)


def test_even_blend_guard_refuses_when_the_target_is_a_blend():
    field = divergent_field()
    pos, neg, neu = _pair(field)
    even_dir = field.declared_e_even()
    plus, minus = lm_faithful_sub_even_blend(pos, neg, neu, even_dir)
    guard = lm_blend_guard(plus, minus, pos, neg)
    # Either the guard admits a still-caption target, or it refuses a blend.
    # The claim: we never return a target nearer mid than the pole without
    # the caller seeing that.
    if not guard["admissible"]:
        from conceptmod.textsliders.slider_targets import lm_faithful_sub_even_blend_guard

        kept_plus, kept_minus = lm_faithful_sub_even_blend_guard(
            pos, neg, neu, even_dir
        )
        assert torch.allclose(kept_plus, pos, atol=1e-6)


def test_gate_odd_then_even_does_not_delete_all_of_c():
    field = unused_e_field()
    pos, neg, neu = _pair(field)
    e = field.declared_e()
    plus, minus = lm_faithful_gate_odd_sub_even(
        pos, neg, neu, e, slider_dir=field.short_u()
    )
    even = lm_even_residual(plus, minus, neu)
    pair_plus, _ = lm_hidden_targets(pos, neg, neu, target_mode="symmetric")
    assert float(even.norm()) > 1e-4
    assert not torch.allclose(plus, pair_plus, atol=1e-4)


def test_close_pair_has_no_even_blend_to_subtract():
    field = close_field()
    assert field.declared_e() is None
    assert field.declared_e_even() is None
    pos, neg, neu = _pair(field)
    plus, minus = lm_faithful_sub_even_blend(pos, neg, neu, None)
    assert torch.allclose(plus, pos, atol=1e-8)


def test_leftover_gate_still_sits_at_positive_leak_frac():
    """The brief's starting point: odd leftover-gate, even kept, leak_frac > 0."""
    field = leaky_field()
    pos, neg, neu = _pair(field)
    plus, minus = lm_faithful_sub_e_if_unused(
        pos, neg, neu, field.leak_e(), slider_dir=field.short_u()
    )
    metrics = leftover_bipolar(plus - neu, minus - neu)
    assert metrics["leak_frac"] > 0.0
