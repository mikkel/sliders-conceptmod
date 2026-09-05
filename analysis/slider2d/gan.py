"""Adversarial 2-D slider fit: RpGAN + b_cap on leftover-gated poles.

The scored object is the same shared odd+even residual the supervised
cells fit. The game is ParticleGAN's: a Fourier-2 critic, relativistic
pair logistic, one-sided b_cap, a small ParticlePrior jitter around each
pole, VICReg, EMA, Adam β1=0, delayed cosine LR.

Real samples are leftover-gated caption poles (``faithful_guard_e`` /
``faithful_sub_e_if_unused``) plus a tight span/end cloud — the Music-LM
hidden-state-delta analogue, not rendered audio. Matching raw ungated
poles copies leftover ê the same way ``faithful_raw`` does; the leftover
gate is the Music adaptation that makes the *existing* 2-D diagnostics
pass.

CPU only. No Hub, no GPU, no Music 3 weights. Does not change the live
trainer default.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import torch

from analysis.slider2d.adv import (
    AdvConfig,
    EMA,
    Fourier2MLP,
    ParticlePrior,
    cap_penalty,
    delayed_cosine,
    feature_match_loss,
    input_grad,
    rp_d_loss,
    rp_g_loss,
    sample_real_cloud,
    vicreg_loss,
)
from analysis.slider2d.exam import (
    PairField,
    exam_reason,
    exam_verdicts,
    rollout_report,
    target_geometry,
    teacher_points as exam_teacher_points,
    teacher_rollouts,
    teacher_self_match,
)
from analysis.slider2d.field import E_ATTR, E_SLIDER, Field2D, cosine
from analysis.slider2d.sheet import (
    SheetField,
    hidden_sheet_report,
    sheet_verdicts,
    teacher_points as sheet_teacher_points,
    teacher_swings,
)
from analysis.slider2d.train import Residual, infer_dim, music3_pairs, score_residual
from conceptmod.textsliders.slider_targets import (
    leftover_bipolar,
    lm_faithful_guard_e,
    lm_faithful_sub_e_if_unused,
)


# Default teacher: blend-guarded leftover ê. On energy-v4 ê restates the
# axis so the guard refuses and the poles stay; on unused leftover it
# subtracts. Same rule as the supervised WORKS row.
DEFAULT_TEACHER = "faithful_guard_e"


@dataclass
class AdvResidual:
    """Odd+even residual, same capacity as the LM student."""

    w_odd: torch.Tensor
    w_even: torch.Tensor

    def delta(self, scale: float) -> torch.Tensor:
        return float(scale) * self.w_odd + abs(float(scale)) * self.w_even

    def parameters(self) -> list[torch.Tensor]:
        return [self.w_odd, self.w_even]

    def snapshot(self) -> "AdvResidual":
        return AdvResidual(self.w_odd.detach().clone(), self.w_even.detach().clone())


def _teacher_pair(
    field,
    row: int,
    *,
    teacher: str,
    leak_dir: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if isinstance(field, PairField):
        t_plus, t_minus = exam_teacher_points(
            field, row, teacher=teacher, leak_dir=leak_dir
        )
        return t_plus, t_minus, field.poles(row)[2]
    if isinstance(field, SheetField):
        t_plus, t_minus = sheet_teacher_points(
            field, row, teacher=teacher, leak_dir=leak_dir
        )
        return t_plus, t_minus, field.poles(row)[2]
    raise TypeError(type(field))


def _field_leak_dir(field) -> torch.Tensor | None:
    if isinstance(field, PairField):
        return field.declared_e()
    if isinstance(field, SheetField):
        return field.leak_e() if float(field.leak) > 1e-8 else None
    return None


def _collect_teachers(
    field,
    *,
    teacher: str,
    leak_dir: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    plus, minus, neus = [], [], []
    for row in range(int(field.rows)):
        t_plus, t_minus, neu = _teacher_pair(
            field, row, teacher=teacher, leak_dir=leak_dir
        )
        plus.append(t_plus.flatten())
        minus.append(t_minus.flatten())
        neus.append(neu.flatten())
    return torch.stack(plus), torch.stack(minus), torch.stack(neus)


def fit_adv(
    field,
    *,
    teacher: str = DEFAULT_TEACHER,
    leak_dir: torch.Tensor | None = None,
    cfg: AdvConfig | None = None,
) -> tuple[AdvResidual, dict]:
    """Fit one shared residual with RpGAN + b_cap. Returns EMA residual + logs."""
    cfg = cfg or AdvConfig()
    if leak_dir is None:
        leak_dir = _field_leak_dir(field)
    torch.manual_seed(int(cfg.seed))
    dim = int(field.dim)
    residual = AdvResidual(
        torch.zeros(dim, requires_grad=True),
        torch.zeros(dim, requires_grad=True),
    )
    prior_p = ParticlePrior(cfg.n_particles, dim)
    prior_m = ParticlePrior(cfg.n_particles, dim)
    critic = Fourier2MLP(
        dim,
        n_rand=cfg.critic_n_rand,
        hidden=cfg.critic_hidden,
        seed=cfg.seed,
    )
    g_params = residual.parameters() + list(prior_p.parameters()) + list(prior_m.parameters())
    opt_g = torch.optim.Adam(g_params, lr=cfg.lr, betas=(cfg.beta1, cfg.beta2))
    opt_d = torch.optim.Adam(critic.parameters(), lr=cfg.lr, betas=(cfg.beta1, cfg.beta2))
    ema = EMA(residual.parameters(), decay=cfg.ema)

    poles_p, poles_m, neus = _collect_teachers(
        field, teacher=teacher, leak_dir=leak_dir
    )
    half = max(1, int(cfg.batch) // 2)
    logs = {"d": [], "g": [], "cap": [], "grad_real": [], "grad_fake": []}

    def set_lr(step: int) -> None:
        scale = delayed_cosine(
            step, total=cfg.steps, delay=cfg.delay, min_ratio=cfg.min_lr_ratio
        )
        for opt in (opt_g, opt_d):
            for group in opt.param_groups:
                group["lr"] = float(cfg.lr) * scale

    def fake_batch() -> tuple[torch.Tensor, torch.Tensor]:
        idx_p = torch.randint(0, neus.shape[0], (half,))
        idx_m = torch.randint(0, neus.shape[0], (half,))
        fake_p = neus[idx_p] + residual.delta(1.0) + prior_p.sample(half, jitter=cfg.particle_jitter)
        fake_m = neus[idx_m] + residual.delta(-1.0) + prior_m.sample(half, jitter=cfg.particle_jitter)
        return fake_p, fake_m

    for step in range(int(cfg.steps)):
        set_lr(step)
        real_p = sample_real_cloud(
            poles_p,
            neus,
            n=half,
            cloud_std=cfg.cloud_std,
            span_frac=cfg.span_frac,
            end_margin=cfg.end_margin,
        )
        real_m = sample_real_cloud(
            poles_m,
            neus,
            n=half,
            cloud_std=cfg.cloud_std,
            span_frac=cfg.span_frac,
            end_margin=cfg.end_margin,
        )
        real = torch.cat([real_p, real_m], dim=0)

        for _ in range(int(cfg.d_steps)):
            fake_p, fake_m = fake_batch()
            fake = torch.cat([fake_p, fake_m], dim=0).detach()
            real_g = real.detach().requires_grad_(True)
            fake_g = fake.detach().requires_grad_(True)
            d_real = critic(real_g)
            d_fake = critic(fake_g)
            cap = cap_penalty(
                input_grad(critic, real_g),
                input_grad(critic, fake_g),
                coeff=cfg.b_cap,
            )
            d_loss = rp_d_loss(d_real, d_fake) + cap
            opt_d.zero_grad()
            d_loss.backward()
            opt_d.step()

        fake_p, fake_m = fake_batch()
        fake = torch.cat([fake_p, fake_m], dim=0)
        d_real = critic(real.detach())
        d_fake = critic(fake)
        g_adv = rp_g_loss(d_real, d_fake)
        g_extra = residual.w_odd.new_zeros(())
        if float(cfg.fm_weight) > 0.0:
            g_extra = g_extra + float(cfg.fm_weight) * feature_match_loss(
                critic.hidden(real.detach()),
                critic.hidden(fake),
                normalize=cfg.fm_normalize,
            )
        parts = torch.cat([prior_p.particles, prior_m.particles], dim=0)
        if float(cfg.vicreg_weight) > 0.0:
            g_extra = g_extra + float(cfg.vicreg_weight) * vicreg_loss(parts)
        if float(cfg.particle_l2) > 0.0:
            g_extra = g_extra + float(cfg.particle_l2) * parts.pow(2).mean()
        g_loss = g_adv + g_extra
        opt_g.zero_grad()
        g_loss.backward()
        opt_g.step()
        ema.update(residual.parameters())

        if step == 0 or (step + 1) % 50 == 0 or step + 1 == cfg.steps:
            with torch.no_grad():
                gn_r = input_grad(critic, real.detach().requires_grad_(True)).flatten(1).norm(dim=1).mean()
                gn_f = input_grad(critic, fake.detach().requires_grad_(True)).flatten(1).norm(dim=1).mean()
            logs["d"].append(float(d_loss.detach()))
            logs["g"].append(float(g_loss.detach()))
            logs["cap"].append(float(cap.detach()))
            logs["grad_real"].append(float(gn_r.detach()))
            logs["grad_fake"].append(float(gn_f.detach()))

    ema.copy_to(residual.parameters())
    snap = residual.snapshot()
    cover = _coverage(snap, poles_p, poles_m, neus)
    stats = {
        "d_loss": logs["d"][-1] if logs["d"] else None,
        "g_loss": logs["g"][-1] if logs["g"] else None,
        "cap": logs["cap"][-1] if logs["cap"] else None,
        "grad_real": logs["grad_real"][-1] if logs["grad_real"] else None,
        "grad_fake": logs["grad_fake"][-1] if logs["grad_fake"] else None,
        "teacher": teacher,
        "b_cap": float(cfg.b_cap),
        "steps": int(cfg.steps),
        "fm_weight": float(cfg.fm_weight),
        **cover,
        "log": logs,
    }
    return snap, stats


def _coverage(
    residual: AdvResidual,
    poles_p: torch.Tensor,
    poles_m: torch.Tensor,
    neus: torch.Tensor,
) -> dict[str, float]:
    """How close ±1 land on the teacher poles (row 0)."""
    neu = neus[0]
    pred_p = neu + residual.delta(1.0)
    pred_m = neu + residual.delta(-1.0)
    err_p = float((pred_p - poles_p[0]).norm() / poles_p[0].norm().clamp_min(1e-8))
    err_m = float((pred_m - poles_m[0]).norm() / poles_m[0].norm().clamp_min(1e-8))
    return {
        "pole_rel_err_plus": err_p,
        "pole_rel_err_minus": err_m,
        "pole_cos_plus": cosine(pred_p - neu, poles_p[0] - neu),
        "pole_cos_minus": cosine(pred_m - neu, poles_m[0] - neu),
        "covered": err_p <= 0.20 and err_m <= 0.20,
    }


def as_exam_residual(residual: AdvResidual):
    from analysis.slider2d.exam import SharedResidual

    return SharedResidual(residual.w_odd.clone(), residual.w_even.clone())


def as_sheet_residual(residual: AdvResidual):
    from analysis.slider2d.sheet import SharedResidual

    return SharedResidual(residual.w_odd.clone(), "odd_even", residual.w_even.clone())


def as_train_residual(residual: AdvResidual) -> Residual:
    return Residual("odd_even", residual.w_odd.clone(), residual.w_even.clone())


def score_adv_exam(
    field: PairField,
    *,
    teacher: str = DEFAULT_TEACHER,
    leak_dir: torch.Tensor | None = None,
    cfg: AdvConfig | None = None,
    name: str = "rpgan_bcap",
) -> dict:
    """Fit the adversarial residual, then listen with the pair-exam rollout."""
    cfg = cfg or AdvConfig()
    if leak_dir is None:
        leak_dir = field.declared_e()
    residual, stats = fit_adv(field, teacher=teacher, leak_dir=leak_dir, cfg=cfg)
    head = field.readout()
    t_plus_roll, t_minus_roll, words = teacher_rollouts(field, head)
    d_plus = residual.delta(1.0)
    d_minus = residual.delta(-1.0)
    plus = [field.poles(r)[2] + d_plus for r in range(int(field.rows))]
    minus = [field.poles(r)[2] + d_minus for r in range(int(field.rows))]
    row = rollout_report(
        field,
        plus,
        minus,
        readout=head,
        teacher_plus=t_plus_roll,
        teacher_minus=t_minus_roll,
        corpus=words,
    )
    ceiling = rollout_report(
        field,
        [field.poles(r)[0] for r in range(int(field.rows))],
        [field.poles(r)[1] for r in range(int(field.rows))],
        readout=head,
        teacher_plus=t_plus_roll,
        teacher_minus=t_minus_roll,
        corpus=words,
    )
    geom = target_geometry(field, teacher=teacher, leak_dir=leak_dir)
    a = field.odd(0)
    pos, _neg, neu = field.poles(0)
    targets = [
        exam_teacher_points(field, r, teacher=teacher, leak_dir=leak_dir)
        for r in range(int(field.rows))
    ]
    percs = []
    for r, (t_plus, t_minus) in enumerate(targets):
        base = field.poles(r)[2]
        percs.append(
            (
                float((plus[r] - t_plus).norm())
                / float((t_plus - base).norm().clamp_min(1e-8)),
                float((minus[r] - t_minus).norm())
                / float((t_minus - base).norm().clamp_min(1e-8)),
            )
        )
    pperc = sum(p for p, _ in percs) / len(percs)
    nperc = sum(n for _, n in percs) / len(percs)
    d_hat = field.delivery_dir()
    want = abs(float((targets[0][0] - field.poles(0)[2]) @ d_hat))
    got = abs(float(d_plus @ d_hat))
    invisible_kept = got / want if want > 1e-8 else None
    self_match = teacher_self_match(t_plus_roll, t_minus_roll)
    swing_kept = row["roll_swing"] / (abs(ceiling["roll_swing"]) + 1e-8)
    leak_tok = (
        row["first_unused_swing"] / (abs(row["first_swing_mass"]) + 1e-8)
        if field.has_unused()
        else None
    )
    leftover = leftover_bipolar(d_plus, d_minus)
    out = dict(row)
    out.update(
        {
            "name": name,
            "cell": field.kind,
            "pole_mode": "rpgan_bcap",
            "teacher": teacher,
            "hold_weight": 0.0,
            "common_beta": 0.0,
            "divergence": field.divergence(),
            "visible_share": field.visible_share(),
            "invisible_share": field.invisible_share(),
            "probe_cos": field.probe_cos(0),
            "pair_odd_cos": cosine(d_plus, a),
            "collapse": cosine(d_plus, d_minus),
            "pole_cos": cosine(d_plus, pos - neu),
            "pperc": pperc,
            "nperc": nperc,
            "perc_gap": abs(pperc - nperc),
            "loss": stats.get("g_loss"),
            "loss_floor": None,
            "loss_solved": None,
            "invisible_kept": invisible_kept,
            "leak_tok": leak_tok,
            "roll_swing_kept": swing_kept,
            "roll_match_kept": row["roll_match"] / (self_match + 1e-8),
            "teacher_self_match": self_match,
            "teacher_roll_swing": ceiling["roll_swing"],
            "leak_frac": leftover["leak_frac"],
            "same_dir": leftover["same_dir"],
            **geom,
            **{k: v for k, v in stats.items() if k != "log"},
        }
    )
    out["axis"] = exam_verdicts(out)
    out["pass"] = all(v == "right" for v in out["axis"].values())
    out["reason"] = exam_reason(out)
    return out


def score_adv_sheet(
    field: SheetField,
    *,
    teacher: str = DEFAULT_TEACHER,
    leak_dir: torch.Tensor | None = None,
    cfg: AdvConfig | None = None,
    name: str = "rpgan_bcap",
) -> dict:
    """Fit the adversarial residual, then read the #22 sheet."""
    cfg = cfg or AdvConfig()
    if leak_dir is None:
        leak_dir = field.leak_e() if float(field.leak) > 1e-8 else None
    residual, stats = fit_adv(field, teacher=teacher, leak_dir=leak_dir, cfg=cfg)
    head = field.readout()
    d_plus = residual.delta(1.0)
    d_minus = residual.delta(-1.0)
    plus = [field.poles(r)[2] + d_plus for r in range(int(field.rows))]
    minus = [field.poles(r)[2] + d_minus for r in range(int(field.rows))]
    row = hidden_sheet_report(field, plus, minus, readout=head)
    a = field.odd(0)
    pos, _neg, neu = field.poles(0)
    ceiling = teacher_swings(field, head)
    leftover = leftover_bipolar(d_plus, d_minus)
    on_u = float(d_plus @ field.short_u())
    row.update(
        {
            "name": name,
            "pole_mode": "rpgan_bcap",
            "teacher": teacher,
            "student": "odd_even",
            "hold_weight": 0.0,
            "common": float(field.common),
            "common_beta": 0.0,
            "probe_cos": field.probe_cos(0),
            "pair_odd_cos": cosine(d_plus, a),
            "collapse": cosine(d_plus, d_minus),
            "pole_cos": cosine(d_plus, pos - neu),
            "sheet_dir_kept": float(d_plus @ field.sheet_dir())
            / float(field.common_vec(0).norm() + 1e-8),
            "leak_hidden": abs(float(d_plus @ field.leak_e())) / (abs(on_u) + 1e-8),
            "swing_kept": row["concept_swing"] / (abs(ceiling["concept_swing"]) + 1e-8),
            "on_sheet_kept": row["on_sheet"] / (ceiling["on_sheet"] + 1e-8),
            "teacher_on_sheet": ceiling["on_sheet"],
            "teacher_garble": ceiling["garble"],
            "teacher_leak_tok": ceiling["leak_tok"],
            "leak_frac": leftover["leak_frac"],
            "same_dir": leftover["same_dir"],
            **{k: v for k, v in stats.items() if k != "log"},
        }
    )
    row["axis"] = sheet_verdicts(row)
    row["pass"] = all(v == "right" for v in row["axis"].values())
    return row


def train_lm_adv(
    field: Field2D,
    pairs=None,
    *,
    teacher: str = "faithful_guard_e",
    cfg: AdvConfig | None = None,
) -> Residual:
    """Field2D residual via the same game. Used for leak_frac / polarity."""
    cfg = cfg or AdvConfig()
    torch.manual_seed(int(cfg.seed))
    pairs = pairs if pairs is not None else music3_pairs(False)
    t = 0.5
    plus, minus, neus = [], [], []
    for pair in pairs:
        pos = field.embed(pair.positive, t)
        neg = field.embed(pair.negative, t)
        neu = field.embed(pair.neutral, t)
        if teacher == "faithful":
            t_plus, t_minus = pos, neg
        elif teacher == "faithful_sub_e_if_unused":
            t_plus, t_minus = lm_faithful_sub_e_if_unused(
                pos, neg, neu, E_ATTR, slider_dir=E_SLIDER
            )
        else:
            t_plus, t_minus = lm_faithful_guard_e(
                pos, neg, neu, E_ATTR, slider_dir=E_SLIDER
            )
        plus.append(t_plus.flatten())
        minus.append(t_minus.flatten())
        neus.append(neu.flatten())
    poles_p = torch.stack(plus)
    poles_m = torch.stack(minus)
    neus_t = torch.stack(neus)
    dim = infer_dim(field, pairs)
    residual = AdvResidual(
        torch.zeros(dim, requires_grad=True),
        torch.zeros(dim, requires_grad=True),
    )
    prior_p = ParticlePrior(cfg.n_particles, dim)
    prior_m = ParticlePrior(cfg.n_particles, dim)
    critic = Fourier2MLP(dim, n_rand=cfg.critic_n_rand, hidden=cfg.critic_hidden, seed=cfg.seed)
    g_params = residual.parameters() + list(prior_p.parameters()) + list(prior_m.parameters())
    opt_g = torch.optim.Adam(g_params, lr=cfg.lr, betas=(cfg.beta1, cfg.beta2))
    opt_d = torch.optim.Adam(critic.parameters(), lr=cfg.lr, betas=(cfg.beta1, cfg.beta2))
    ema = EMA(residual.parameters(), decay=cfg.ema)
    half = max(1, int(cfg.batch) // 2)

    def set_lr(step: int) -> None:
        scale = delayed_cosine(
            step, total=cfg.steps, delay=cfg.delay, min_ratio=cfg.min_lr_ratio
        )
        for opt in (opt_g, opt_d):
            for group in opt.param_groups:
                group["lr"] = float(cfg.lr) * scale

    def fake_batch() -> tuple[torch.Tensor, torch.Tensor]:
        idx_p = torch.randint(0, neus_t.shape[0], (half,))
        idx_m = torch.randint(0, neus_t.shape[0], (half,))
        return (
            neus_t[idx_p] + residual.delta(1.0) + prior_p.sample(half, jitter=cfg.particle_jitter),
            neus_t[idx_m] + residual.delta(-1.0) + prior_m.sample(half, jitter=cfg.particle_jitter),
        )

    for step in range(int(cfg.steps)):
        set_lr(step)
        real_p = sample_real_cloud(
            poles_p, neus_t, n=half, cloud_std=cfg.cloud_std,
            span_frac=cfg.span_frac, end_margin=cfg.end_margin,
        )
        real_m = sample_real_cloud(
            poles_m, neus_t, n=half, cloud_std=cfg.cloud_std,
            span_frac=cfg.span_frac, end_margin=cfg.end_margin,
        )
        real = torch.cat([real_p, real_m], dim=0)
        for _ in range(int(cfg.d_steps)):
            fake_p, fake_m = fake_batch()
            fake = torch.cat([fake_p, fake_m], dim=0).detach()
            real_g = real.detach().requires_grad_(True)
            fake_g = fake.detach().requires_grad_(True)
            d_loss = rp_d_loss(critic(real_g), critic(fake_g)) + cap_penalty(
                input_grad(critic, real_g),
                input_grad(critic, fake_g),
                coeff=cfg.b_cap,
            )
            opt_d.zero_grad()
            d_loss.backward()
            opt_d.step()
        fake_p, fake_m = fake_batch()
        fake = torch.cat([fake_p, fake_m], dim=0)
        g_loss = rp_g_loss(critic(real.detach()), critic(fake))
        parts = torch.cat([prior_p.particles, prior_m.particles], dim=0)
        if cfg.vicreg_weight:
            g_loss = g_loss + float(cfg.vicreg_weight) * vicreg_loss(parts)
        if cfg.particle_l2:
            g_loss = g_loss + float(cfg.particle_l2) * parts.pow(2).mean()
        opt_g.zero_grad()
        g_loss.backward()
        opt_g.step()
        ema.update(residual.parameters())

    ema.copy_to(residual.parameters())
    return as_train_residual(residual.snapshot())


def score_field2d(cfg: AdvConfig | None = None, *, teacher: str = DEFAULT_TEACHER) -> dict:
    """Geometry + leftover bipolar on the original 2-D field."""
    cfg = cfg or AdvConfig()
    field = Field2D()
    residual = train_lm_adv(field, teacher=teacher, cfg=cfg)
    metrics = score_residual(residual)
    leftover = leftover_bipolar(residual.delta(1.0), residual.delta(-1.0))
    metrics.update(leftover)
    leak_ok = abs(metrics["leak_ratio"]) <= 0.20
    slider_ok = metrics["cos_slider_plus"] >= 0.90
    collapse_ok = metrics["cos_plus_minus"] <= -0.85
    polarity_ok = leftover["leak_frac"] <= -0.85
    metrics.update(
        {
            "name": "rpgan_bcap",
            "teacher": teacher,
            "leak_ok": leak_ok,
            "slider_ok": slider_ok,
            "collapse_ok": collapse_ok,
            "polarity_ok": polarity_ok,
            "pass": leak_ok and slider_ok and collapse_ok and polarity_ok,
        }
    )
    return metrics


def default_cfg(**overrides) -> AdvConfig:
    cfg = AdvConfig()
    return replace(cfg, **overrides) if overrides else cfg
