"""CPU live-exam cells for pair geometry and continuation failure.

The original sheet cell has one kind of pair and one next-token readout.
That cannot represent either distinction exposed by the 2026-08-25 listens:

* an attribute can be unused inside a same-song pair but be the intended
  difference between two divergent tracks; and
* matching the policy at ``<|audio_start|>`` can leave continuation-bearing
  hidden state unconstrained.

This fixture keeps the live loss literal: ``semantic_kl`` supervises one
frozen readout at step zero.  A second frozen readout family scores the
teacher-forced continuation and a greedy rollout.  Pair-odd cosine and
``±1`` collapse are reported but never gated.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from analysis.slider2d.field import cosine
from conceptmod.textsliders.slider_targets import (
    lm_next_token_logits,
    lm_semantic_kl,
    lm_semantic_pole_loss,
    lm_slider_loss,
)


TOKENS = ("shared", "plus_track", "minus_track", "blend", "garble")
STEPS = 4
ROLLOUT_MATCH_MIN = 0.75
ROLLOUT_GARBLE_MAX = 0.0
OFF_CAPTION_MAX = 0.10
ONE_TOKEN_KL_SMALL = 0.02
HIDDEN_FAR_MIN = 0.50


@dataclass(frozen=True)
class ExamField:
    """A tiny hidden field with frozen first-token and continuation heads.

    Coordinates are ``[bias, track, semantic, plan, common]``.

    ``divergent`` models energy-v4.  ``track`` is the dominant pop-punk/BPM
    versus ambient/BPM pole difference.  Calling it ê and subtracting it
    removes intended signal, producing a blend teacher.

    ``close`` models gender-v4.  The two poles share the first-token policy;
    most of their distinction is in ``plan``, which step zero cannot read
    but later continuation heads can.
    """

    pair_kind: str

    @property
    def dim(self) -> int:
        return 5

    def neutral(self) -> torch.Tensor:
        return torch.tensor([1.0, 0.0, 0.0, 0.0, 0.0])

    def odd(self) -> torch.Tensor:
        if self.pair_kind == "divergent":
            return torch.tensor([0.0, 1.0, 0.20, 0.0, 0.0])
        if self.pair_kind == "close":
            return torch.tensor([0.0, 0.0, 0.08, 1.0, 0.0])
        raise ValueError(f"unknown pair kind {self.pair_kind!r}")

    def common(self) -> torch.Tensor:
        # Divergent energy-v4 logs pair cos around zero: common and odd have
        # similar norms.  The close pair shares nearly all song identity.
        scale = 1.0 if self.pair_kind == "divergent" else 0.35
        return torch.tensor([0.0, 0.0, 0.0, 0.0, scale * float(self.odd().norm())])

    def poles(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        neu = self.neutral()
        odd = self.odd()
        common = self.common()
        return neu + common + odd, neu + common - odd, neu

    def e_dir(self) -> torch.Tensor:
        """The YAML leftover axis; only meaningful for the divergent cell."""
        return torch.tensor([0.0, 1.0, 0.0, 0.0, 0.0])

    def first_head(self) -> torch.Tensor:
        """Frozen semantic-band analogue at ``<|audio_start|>``.

        It strongly sees the divergent track coordinate, weakly sees the
        close semantic coordinate, and cannot see the continuation plan.
        """
        head = torch.zeros(len(TOKENS), self.dim)
        head[0, 0] = 2.0  # shared token wins on a close pair
        head[1, 1], head[1, 2], head[1, 4] = 4.0, 1.0, 0.25
        head[2, 1], head[2, 2], head[2, 4] = -4.0, -1.0, 0.25
        head[3, 0], head[3, 4] = 0.7, 0.35
        head[4, 4] = -1.0
        return head

    def continuation_heads(self) -> tuple[torch.Tensor, ...]:
        """Frozen teacher-forced heads for the next semantic-code steps."""
        heads = [self.first_head()]
        for step in range(1, STEPS):
            head = torch.zeros(len(TOKENS), self.dim)
            # Correct poles carry either divergent track identity or the
            # close pair's latent sequence plan.  If both are absent, the
            # fixed garble baseline wins.
            gain = 3.6 + 0.2 * step
            head[1, 1], head[1, 2], head[1, 3] = gain, 0.7, gain
            head[2, 1], head[2, 2], head[2, 3] = -gain, -0.7, -gain
            head[3, 0] = 0.6
            head[4, 0] = 2.0
            heads.append(head)
        return tuple(heads)


def divergent_pair_field() -> ExamField:
    return ExamField("divergent")


def close_pair_field() -> ExamField:
    return ExamField("close")


def _sub_e(
    hidden: torch.Tensor,
    neutral: torch.Tensor,
    direction: torch.Tensor,
) -> torch.Tensor:
    unit = direction / direction.norm().clamp_min(1e-8)
    return hidden - ((hidden - neutral) @ unit) * unit


def teacher_points(
    field: ExamField,
    target: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    pos, neg, neu = field.poles()
    mode = str(target).strip().lower()
    if mode == "faithful":
        return pos, neg
    if mode == "faithful_sub_e":
        return _sub_e(pos, neu, field.e_dir()), _sub_e(neg, neu, field.e_dir())
    raise ValueError(f"target must be faithful or faithful_sub_e, got {target!r}")


@dataclass
class ExamResidual:
    odd: torch.Tensor
    even: torch.Tensor

    @classmethod
    def create(cls, dim: int) -> "ExamResidual":
        return cls(torch.zeros(dim, requires_grad=True), torch.zeros(dim, requires_grad=True))

    def delta(self, sign: float) -> torch.Tensor:
        return float(sign) * self.odd + self.even

    def snapshot(self) -> "ExamResidual":
        return ExamResidual(self.odd.detach().clone(), self.even.detach().clone())


def fit_exam(
    field: ExamField,
    *,
    pole_mode: str,
    target: str,
    steps: int = 400,
    lr: float = 0.08,
    seed: int = 0,
) -> ExamResidual:
    """Fit the two pole replies with the literal one-token live loss."""
    mode = str(pole_mode).strip().lower()
    if mode not in {"hidden", "semantic_kl"}:
        raise ValueError(f"pole_mode must be hidden or semantic_kl, got {pole_mode!r}")
    t_plus, t_minus = teacher_points(field, target)
    neu = field.neutral()
    head = field.first_head()
    torch.manual_seed(int(seed))
    residual = ExamResidual.create(field.dim)
    opt = torch.optim.Adam([residual.odd, residual.even], lr=float(lr))
    for _ in range(int(steps)):
        pred_plus = neu + residual.delta(1.0)
        pred_minus = neu + residual.delta(-1.0)
        if mode == "hidden":
            loss = lm_slider_loss(pred_plus, pred_minus, t_plus, t_minus)
        else:
            loss = lm_semantic_pole_loss(
                lm_next_token_logits(pred_plus, head),
                lm_next_token_logits(pred_minus, head),
                lm_next_token_logits(t_plus, head),
                lm_next_token_logits(t_minus, head),
            )
        opt.zero_grad()
        loss.backward()
        opt.step()
    return residual.snapshot()


def _kl(teacher: torch.Tensor, student: torch.Tensor) -> float:
    return max(0.0, float(lm_semantic_kl(student.unsqueeze(0), teacher.unsqueeze(0))))


def _sequence(field: ExamField, hidden: torch.Tensor) -> tuple[list[int], list[torch.Tensor]]:
    logits = [lm_next_token_logits(hidden, head) for head in field.continuation_heads()]
    return [int(torch.argmax(row)) for row in logits], logits


def _rollout(predicted: list[int], target: list[int]) -> list[int]:
    """Greedy rollout with an absorbing off-sheet failure.

    A wrong semantic code changes the autoregressive prefix.  The tiny cell
    represents that prefix shift by latching to ``garble`` after the first
    off-sheet ``blend``/``garble`` token.
    """
    out: list[int] = []
    broken = False
    off_sheet = {TOKENS.index("blend"), TOKENS.index("garble")}
    for pred, want in zip(predicted, target):
        token = TOKENS.index("garble") if broken else pred
        out.append(token)
        broken = broken or (token != want and token in off_sheet)
    return out


def score_exam(
    name: str,
    field: ExamField,
    *,
    pole_mode: str,
    target: str,
    live_run: str | None,
    expected_listen: str | None,
    steps: int = 400,
    seed: int = 0,
) -> dict:
    """Fit and score first-token KL, hidden distance, and continuation."""
    residual = fit_exam(
        field,
        pole_mode=pole_mode,
        target=target,
        steps=steps,
        seed=seed,
    )
    pos, neg, neu = field.poles()
    t_plus, t_minus = teacher_points(field, target)
    pred_plus = neu + residual.delta(1.0)
    pred_minus = neu + residual.delta(-1.0)
    first = field.first_head()

    one_token_target = 0.5 * (
        _kl(lm_next_token_logits(t_plus, first), lm_next_token_logits(pred_plus, first))
        + _kl(lm_next_token_logits(t_minus, first), lm_next_token_logits(pred_minus, first))
    )
    one_token_pole = 0.5 * (
        _kl(lm_next_token_logits(pos, first), lm_next_token_logits(pred_plus, first))
        + _kl(lm_next_token_logits(neg, first), lm_next_token_logits(pred_minus, first))
    )
    hidden_far = 0.5 * (
        float((pred_plus - t_plus).norm() / (t_plus - neu).norm().clamp_min(1e-8))
        + float((pred_minus - t_minus).norm() / (t_minus - neu).norm().clamp_min(1e-8))
    )
    off_caption = 0.5 * (
        float((t_plus - pos).norm() / (pos - neu).norm().clamp_min(1e-8))
        + float((t_minus - neg).norm() / (neg - neu).norm().clamp_min(1e-8))
    )

    tf_kls: list[float] = []
    matches: list[float] = []
    garbles: list[float] = []
    says: list[str] = []
    for pole, pred in ((pos, pred_plus), (neg, pred_minus)):
        target_seq, target_logits = _sequence(field, pole)
        pred_seq, pred_logits = _sequence(field, pred)
        rollout = _rollout(pred_seq, target_seq)
        tf_kls.extend(_kl(want, got) for want, got in zip(target_logits, pred_logits))
        matches.extend(float(got == want) for got, want in zip(rollout, target_seq))
        garbles.extend(float(TOKENS[got] == "garble") for got in rollout)
        says.append(">".join(TOKENS[idx] for idx in rollout))

    rollout_match = sum(matches) / len(matches)
    rollout_garble = sum(garbles) / len(garbles)
    continuation_pass = (
        rollout_match >= ROLLOUT_MATCH_MIN and rollout_garble <= ROLLOUT_GARBLE_MAX
    )
    hidden_far_while_kl_small = (
        one_token_target <= ONE_TOKEN_KL_SMALL and hidden_far >= HIDDEN_FAR_MIN
    )
    pair_odd = field.odd()
    d_plus = pred_plus - neu
    d_minus = pred_minus - neu
    return {
        "name": name,
        "pair_kind": field.pair_kind,
        "pole_mode": pole_mode,
        "target": target,
        "live_run": live_run,
        "expected_listen": expected_listen,
        "one_token_kl": one_token_target,
        "one_token_pole_kl": one_token_pole,
        "hidden_far": hidden_far,
        "hidden_far_while_kl_small": hidden_far_while_kl_small,
        "off_caption_teacher": off_caption,
        "teacher_forced_kl": sum(tf_kls) / len(tf_kls),
        "rollout_match": rollout_match,
        "rollout_garble": rollout_garble,
        "rollout": " | ".join(says),
        "pair_probe_cos": cosine(pos - neu, neg - neu),
        "pair_odd_cos": cosine(d_plus, pair_odd),
        "collapse": cosine(d_plus, d_minus),
        "pass": continuation_pass and off_caption <= OFF_CAPTION_MAX,
        "continuation_pass": continuation_pass,
    }


def live_exam_cell(*, steps: int = 400, seed: int = 0) -> list[dict]:
    """Rows corresponding to the three listens plus the next control."""
    divergent = divergent_pair_field()
    close = close_pair_field()
    return [
        score_exam(
            "energy_v16_semantic_kl_sub_e",
            divergent,
            pole_mode="semantic_kl",
            target="faithful_sub_e",
            live_run="energy-lm-v16",
            expected_listen="FAIL",
            steps=steps,
            seed=seed,
        ),
        score_exam(
            "energy_v18_semantic_kl_faithful",
            divergent,
            pole_mode="semantic_kl",
            target="faithful",
            live_run="energy-lm-v18",
            expected_listen="PASS",
            steps=steps,
            seed=seed,
        ),
        score_exam(
            "gender_v16_semantic_kl_faithful",
            close,
            pole_mode="semantic_kl",
            target="faithful",
            live_run="gender-lm-v16",
            expected_listen="FAIL",
            steps=steps,
            seed=seed,
        ),
        score_exam(
            "gender_hidden_faithful_next",
            close,
            pole_mode="hidden",
            target="faithful",
            live_run=None,
            expected_listen=None,
            steps=steps,
            seed=seed,
        ),
    ]

