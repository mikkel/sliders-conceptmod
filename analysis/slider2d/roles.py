"""Role-split UNI: lyrics stay neu, Vocal Details may move to pos.

Third UNI family, not a duplicate of whole-prefix hold, lyric-token
hold-to-neu, or last-delta-off-lyric. Live last-token UNI
(``faithful_plus_neu``) matches ``<|audio_start|>`` to raw ``h+`` and
shreds grit lyrics. Whole-prefix hold (``faithful_plus_neu_prefix``)
pins every prefix token to encode(neu), so grit lyrics return and
woman cannot move — Vocal Details stays ungendered.

This cell splits the prefix by role. Yaml lyric tokens stay at
encode(neu). Concept / Vocal Details tokens are taught toward
encode(pos) so woman can appear on a neu listen. Last token is still
raw ``h+``. Scale 0 is still ``h0``. No minus, no leftover-gate.

Both pair types on one board:

- grit-like / divergent — fail = ``punk punk`` instead of the yaml line
- gender-like / close — fail = woman cannot move because Vocal Details
  is pinned to neu

Rank keys: lyric_recall@+1 (≥ 0.85), cover, neu_hold, gender_move.
Not ``exam_score`` / ``leak_frac`` / ``c+`` / ``p%``.

CPU only. No Hub, no GPU, no Music 3 weights. Does not change the
live default (``--lm_target v9`` / ``--pole_mode hidden``).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from analysis.slider2d.exam import PairField, close_field, divergent_field
from analysis.slider2d.lyric_recall import (
    ATTEND,
    LYRIC_RECALL_MIN,
    PREFIX_LEN,
    lyric_bag,
    lyric_embeds,
    lyric_recall,
    off_lyric,
    sung_line,
)
from analysis.slider2d.plus_exam import (
    PLUS_COVER_MIN,
    PLUS_OFF_MAX,
    _continue,
    _off_share,
    _token_share,
    blend_toward_mid,
    plus_bags,
    plus_cover,
)
from analysis.slider2d.plus_neu_exam import (
    PLUS_NEU_HOLD_MIN,
    drift_from_neu,
    neu_bags,
    neu_hold,
    plus_neu_teacher,
)
from conceptmod.textsliders.slider_targets import (
    RoleSpanError,
    lm_plus_loss,
    lm_plus_neu_loss,
    lm_plus_neu_prefix_loss,
    lm_plus_neu_roles_loss,
    lm_slider_loss,
)

CONCEPT_LEN = 3
GENDER_MOVE_MIN = LYRIC_RECALL_MIN

ROLE_CELLS = {
    "divergent": divergent_field,
    "close": close_field,
}

ROLE_RECIPES: list[dict] = [
    {
        "name": "faithful_plus_neu",
        "teacher": "faithful_plus_neu",
        "plus_neu": True,
        "plus_only": False,
        "prefix_hold": False,
        "roles": False,
    },
    {
        "name": "faithful_plus_neu_prefix",
        "teacher": "faithful_plus_neu",
        "plus_neu": True,
        "plus_only": False,
        "prefix_hold": True,
        "roles": False,
    },
    {
        "name": "faithful_plus_neu_roles",
        "teacher": "faithful_plus_neu",
        "plus_neu": True,
        "plus_only": False,
        "prefix_hold": False,
        "roles": True,
    },
]


@dataclass
class RoleResidual:
    """Shared LoRA plus separate extras on lyric vs concept tokens.

    Live last-token MSE applies one LoRA at every position. Shared
    ``δ`` is that weight. Lyric extra can cancel the shared rewrite on
    yaml lyrics. Concept extra can move Vocal Details toward encode(pos)
    without pinning it to ungendered neu.
    """

    w: torch.Tensor
    w_even: torch.Tensor
    w0: torch.Tensor
    lyric_p: torch.Tensor
    lyric_even: torch.Tensor
    lyric_0: torch.Tensor
    concept_p: torch.Tensor
    concept_even: torch.Tensor
    concept_0: torch.Tensor

    @classmethod
    def create(cls, field: PairField) -> "RoleResidual":
        z = torch.zeros
        return cls(
            z(field.dim, requires_grad=True),
            z(field.dim, requires_grad=True),
            z(field.dim, requires_grad=True),
            z(field.dim, requires_grad=True),
            z(field.dim, requires_grad=True),
            z(field.dim, requires_grad=True),
            z(field.dim, requires_grad=True),
            z(field.dim, requires_grad=True),
            z(field.dim, requires_grad=True),
        )

    def shared(self, scale: float) -> torch.Tensor:
        s = float(scale)
        return s * self.w + abs(s) * self.w_even + self.w0

    def lyric_delta(self, scale: float) -> torch.Tensor:
        s = float(scale)
        return self.shared(s) + s * self.lyric_p + abs(s) * self.lyric_even + self.lyric_0

    def concept_delta(self, scale: float) -> torch.Tensor:
        s = float(scale)
        return (
            self.shared(s)
            + s * self.concept_p
            + abs(s) * self.concept_even
            + self.concept_0
        )

    def last_delta(self, scale: float) -> torch.Tensor:
        return self.shared(scale)

    def parameters(self) -> list[torch.Tensor]:
        return [
            self.w,
            self.w_even,
            self.w0,
            self.lyric_p,
            self.lyric_even,
            self.lyric_0,
            self.concept_p,
            self.concept_even,
            self.concept_0,
        ]

    def snapshot(self) -> "RoleResidual":
        return RoleResidual(*(p.detach().clone() for p in self.parameters()))


def concept_embeds_neu(field: PairField) -> torch.Tensor:
    """Ungendered Vocal Details: sheet specificity, no woman / no + track."""
    base = float(field.base_sheet) * field.sheet_dir()
    return base.unsqueeze(0).expand(int(CONCEPT_LEN), -1).clone()


def concept_embeds_pos(field: PairField, row: int) -> torch.Tensor:
    """encode(pos) Vocal Details: neu VD plus the + caption's role delta."""
    pos, _neg, neu = field.poles(row)
    return concept_embeds_neu(field) + (pos - neu).unsqueeze(0)


def encode_roles(
    field: PairField,
    residual: RoleResidual,
    row: int,
    scale: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Student encode(neu tokens, LoRA @ scale) → (concept, lyric, last)."""
    concept_base = concept_embeds_neu(field)
    lyric_base = lyric_embeds(field, row)
    prefix_base = torch.cat([concept_base, lyric_base], dim=0)
    _pos, _neg, neu = field.poles(row)
    last_base = neu - float(ATTEND) * prefix_base.mean(0)
    concept = concept_base + residual.concept_delta(scale)
    lyric = lyric_base + residual.lyric_delta(scale)
    last = (
        last_base
        + residual.last_delta(scale)
        + float(ATTEND) * torch.cat([concept, lyric], dim=0).mean(0)
    )
    return concept, lyric, last


def gender_move_score(student: torch.Tensor, neu: torch.Tensor, pos: torch.Tensor) -> float:
    """0 at neu Vocal Details, 1 at encode(pos) Vocal Details."""
    s = student.reshape(-1, student.shape[-1]).mean(0)
    n = neu.reshape(-1, neu.shape[-1]).mean(0)
    p = pos.reshape(-1, pos.shape[-1]).mean(0)
    d_neu = float((s - n).norm())
    d_pos = float((s - p).norm())
    denom = d_neu + d_pos
    if denom <= 1e-8:
        return 0.0
    return d_neu / denom


def concept_tokens(field: PairField, concept: torch.Tensor) -> list[str]:
    """Local VD vocab: lead (neu), woman (delivery), punk (+ track)."""
    weight = torch.stack(
        [field.sheet_dir(), field.delivery_dir(), field.plus_track()],
        dim=0,
    )
    names = ("lead", "woman", "punk")
    logits = float(field.gain) * concept @ weight.T
    return [names[int(row.argmax())] for row in logits]


def fit_role_exam(
    field: PairField,
    *,
    teacher: str,
    leak_dir: torch.Tensor | None = None,
    plus_only: bool = False,
    plus_neu: bool = False,
    prefix_hold: bool = False,
    roles: bool = False,
    steps: int = 400,
    lr: float = 0.08,
    seed: int = 0,
) -> RoleResidual:
    """Fit a role-aware sequence LoRA."""
    if plus_only and plus_neu:
        raise ValueError("plus_only and plus_neu are mutually exclusive")
    if prefix_hold and roles:
        raise ValueError("prefix_hold and roles are mutually exclusive")
    targets = [
        plus_neu_teacher(field, row, teacher=teacher, leak_dir=leak_dir)
        for row in range(int(field.rows))
    ]
    torch.manual_seed(int(seed))
    residual = RoleResidual.create(field)
    opt = torch.optim.Adam(residual.parameters(), lr=float(lr))
    for _ in range(int(steps)):
        total = None
        for row, (t_plus, t_minus) in enumerate(targets):
            _pos, _neg, neu = field.poles(row)
            concept_neu = concept_embeds_neu(field)
            concept_pos = concept_embeds_pos(field, row)
            lyric_neu = lyric_embeds(field, row)
            con_p, lyr_p, last_p = encode_roles(field, residual, row, 1.0)
            _con_m, _lyr_m, last_m = encode_roles(field, residual, row, -1.0)
            _con_0, _lyr_0, last_0 = encode_roles(field, residual, row, 0.0)
            if roles:
                term = lm_plus_neu_roles_loss(
                    last_p,
                    t_plus,
                    last_0,
                    neu,
                    lyr_p,
                    lyric_neu,
                    con_p,
                    concept_pos,
                )
            elif prefix_hold:
                pref_p = torch.cat([con_p, lyr_p], dim=0)
                pref_neu = torch.cat([concept_neu, lyric_neu], dim=0)
                term = lm_plus_neu_prefix_loss(
                    last_p, t_plus, last_0, neu, pref_p, pref_neu
                )
            elif plus_neu:
                term = lm_plus_neu_loss(last_p, t_plus, last_0, neu)
            elif plus_only:
                term = lm_plus_loss(last_p, t_plus)
            else:
                term = lm_slider_loss(last_p, last_m, t_plus, t_minus)
            total = term if total is None else total + term
        loss = total / float(len(targets))
        opt.zero_grad()
        loss.backward()
        opt.step()
    return residual.snapshot()


def score_role_exam(
    name: str,
    field: PairField,
    *,
    teacher: str,
    leak_dir: torch.Tensor | None = None,
    plus_only: bool = False,
    plus_neu: bool = False,
    prefix_hold: bool = False,
    roles: bool = False,
    steps: int = 400,
    seed: int = 0,
) -> dict:
    """Fit, then score lyric_recall, cover, neu_hold, gender_move."""
    residual = fit_role_exam(
        field,
        teacher=teacher,
        leak_dir=leak_dir,
        plus_only=plus_only,
        plus_neu=plus_neu,
        prefix_hold=prefix_hold,
        roles=roles,
        steps=steps,
        seed=seed,
    )
    bags = plus_bags(field)
    neu_bag = neu_bags(field)
    cover_rows: list[float] = []
    off_rows: list[float] = []
    neu_hold_rows: list[float] = []
    lyric_plus: list[float] = []
    gender_rows: list[float] = []
    sings_lyric: list[str] = []
    sings_concept: list[str] = []
    sings_plus: list[str] = []
    sings_zero: list[str] = []
    head = field.readout()
    for row in range(int(field.rows)):
        pos, neg, neu = field.poles(row)
        mid = 0.5 * (pos + neg)
        con_p, lyr_p, last_p = encode_roles(field, residual, row, 1.0)
        _con_0, _lyr_0, last_0 = encode_roles(field, residual, row, 0.0)
        plus_seqs = _continue(field, last_p, row=row, sign=1.0)
        zero_seqs = _continue(field, last_0, row=row, sign=0.0)
        overlap = _token_share(plus_seqs, bags["pos"])
        off = _off_share(plus_seqs, bags["plus_corpus"])
        cover = plus_cover(overlap, blend_toward_mid(last_p, pos, mid, neg))
        neu_ov = _token_share(zero_seqs, neu_bag)
        hold = neu_hold(neu_ov, drift_from_neu(last_0, neu, pos, mid))
        move = gender_move_score(con_p, concept_embeds_neu(field), concept_embeds_pos(field, row))
        cover_rows.append(cover)
        off_rows.append(off)
        neu_hold_rows.append(hold)
        lyric_plus.append(lyric_recall(field, lyr_p, row))
        gender_rows.append(move)
        lyric_seq = sung_line(field, lyr_p)
        sings_lyric.append(" ".join(head.tokens[t] for t in lyric_seq))
        sings_concept.append(" ".join(concept_tokens(field, con_p)))
        sings_plus.append(" ".join(head.tokens[t] for t in plus_seqs[0]))
        sings_zero.append(" ".join(head.tokens[t] for t in zero_seqs[0]))
    cover = sum(cover_rows) / len(cover_rows)
    off_caption = sum(off_rows) / len(off_rows)
    hold = sum(neu_hold_rows) / len(neu_hold_rows)
    recall = sum(lyric_plus) / len(lyric_plus)
    move = sum(gender_rows) / len(gender_rows)
    lyric_hit = bool(recall >= LYRIC_RECALL_MIN)
    gender_hit = bool(move >= GENDER_MOVE_MIN)
    return {
        "name": name,
        "cell": field.kind,
        "teacher": teacher,
        "plus_only": bool(plus_only),
        "plus_neu": bool(plus_neu),
        "prefix_hold": bool(prefix_hold),
        "roles": bool(roles),
        "lyric_recall": recall,
        "cover": cover,
        "off_caption": off_caption,
        "neu_hold": hold,
        "gender_move": move,
        "lyric_hit": lyric_hit,
        "gender_hit": gender_hit,
        "hit": bool(
            lyric_hit
            and gender_hit
            and cover >= PLUS_COVER_MIN
            and hold >= PLUS_NEU_HOLD_MIN
        ),
        "sings_lyric": " | ".join(sings_lyric),
        "sings_concept": " | ".join(sings_concept),
        "sings_plus": " | ".join(sings_plus),
        "sings_zero": " | ".join(sings_zero),
        "off_lyric": sum(off_lyric(field, encode_roles(field, residual, r, 1.0)[1], r) for r in range(int(field.rows)))
        / float(field.rows),
        "pole_cos": float(
            F.cosine_similarity(
                (encode_roles(field, residual, 0, 1.0)[2] - field.poles(0)[2])
                .flatten()
                .unsqueeze(0),
                (field.poles(0)[0] - field.poles(0)[2]).flatten().unsqueeze(0),
            ).squeeze()
        ),
    }


def role_exam_table(*, steps: int = 400, seed: int = 0) -> dict[str, list[dict]]:
    """Score UNI / prefix-hold / role-split on divergent and close."""
    out: dict[str, list[dict]] = {}
    for cell_name, ctor in ROLE_CELLS.items():
        field = ctor(seed=seed)
        leak = field.declared_e()
        rows = []
        for cand in ROLE_RECIPES:
            rows.append(
                score_role_exam(
                    cand["name"],
                    field,
                    teacher=cand["teacher"],
                    leak_dir=leak,
                    plus_only=bool(cand["plus_only"]),
                    plus_neu=bool(cand["plus_neu"]),
                    prefix_hold=bool(cand["prefix_hold"]),
                    roles=bool(cand["roles"]),
                    steps=steps,
                    seed=seed,
                )
            )
        out[cell_name] = rows
    return out


def role_rank(table: dict[str, list[dict]]) -> list[dict]:
    """Rank: lyric_recall, cover, neu_hold, gender_move. Required pairs."""
    by: dict[str, dict[str, dict]] = {}
    for cell, rows in table.items():
        by[cell] = {r["name"]: r for r in rows}
    names = [c["name"] for c in ROLE_RECIPES]
    required = [c for c in ("divergent", "close") if c in by]
    ranked: list[dict] = []
    for name in names:
        cells = [by[c][name] for c in required]
        grit = by["divergent"][name] if "divergent" in by else None
        gender = by["close"][name] if "close" in by else None
        ranked.append(
            {
                "name": name,
                "in_box": bool(
                    grit is not None
                    and gender is not None
                    and grit["lyric_hit"]
                    and gender["gender_hit"]
                ),
                "lyric_recall": sum(r["lyric_recall"] for r in cells) / len(cells),
                "cover": sum(r["cover"] for r in cells) / len(cells),
                "neu_hold": sum(r["neu_hold"] for r in cells) / len(cells),
                "gender_move": sum(r["gender_move"] for r in cells) / len(cells),
                "lyric_hit_divergent": None if grit is None else grit["lyric_hit"],
                "gender_hit_close": None if gender is None else gender["gender_hit"],
                "prefix_hold": cells[0]["prefix_hold"],
                "roles": cells[0]["roles"],
                "plus_neu": cells[0]["plus_neu"],
            }
        )
    ranked.sort(
        key=lambda r: (
            -float(r["lyric_recall"]),
            -float(r["cover"]),
            -float(r["neu_hold"]),
            -float(r["gender_move"]),
            str(r["name"]),
        )
    )
    for i, row in enumerate(ranked, start=1):
        row["rank"] = i
    return ranked


def role_verdict(table: dict[str, list[dict]]) -> dict:
    """Want-box: grit lyric HIT and gender_move HIT. Report a miss honestly."""
    by: dict[str, dict[str, dict]] = {
        cell: {r["name"]: r for r in rows} for cell, rows in table.items()
    }
    uni = by["divergent"]["faithful_plus_neu"]
    prefix = by["divergent"]["faithful_plus_neu_prefix"]
    roles = by["divergent"]["faithful_plus_neu_roles"]
    uni_c = by["close"]["faithful_plus_neu"]
    prefix_c = by["close"]["faithful_plus_neu_prefix"]
    roles_c = by["close"]["faithful_plus_neu_roles"]
    return {
        "want_box": bool(roles["lyric_hit"] and roles_c["gender_hit"]),
        "roles_grit_lyric": roles["lyric_hit"],
        "roles_gender_move": roles_c["gender_hit"],
        "uni_grit_lyric": uni["lyric_hit"],
        "uni_gender_move": uni_c["gender_hit"],
        "prefix_grit_lyric": prefix["lyric_hit"],
        "prefix_gender_move": prefix_c["gender_hit"],
        "uni_sings_lyric_divergent": uni["sings_lyric"],
        "prefix_sings_concept_close": prefix_c["sings_concept"],
        "roles_sings_lyric_divergent": roles["sings_lyric"],
        "roles_sings_concept_close": roles_c["sings_concept"],
        "uni_lyric_recall": [uni["lyric_recall"], uni_c["lyric_recall"]],
        "prefix_lyric_recall": [prefix["lyric_recall"], prefix_c["lyric_recall"]],
        "roles_lyric_recall": [roles["lyric_recall"], roles_c["lyric_recall"]],
        "uni_gender_move_scores": [uni["gender_move"], uni_c["gender_move"]],
        "prefix_gender_move_scores": [prefix["gender_move"], prefix_c["gender_move"]],
        "roles_gender_move_scores": [roles["gender_move"], roles_c["gender_move"]],
    }


def require_role_spans(spans: dict) -> dict:
    """Fixture-side fail-closed: both required spans must be present."""
    lyric = spans.get("lyric")
    concept = spans.get("concept")
    if not lyric or lyric[1] <= lyric[0]:
        raise RoleSpanError("lyrics span is empty")
    if not concept or concept[1] <= concept[0]:
        raise RoleSpanError("concept span is empty")
    return spans
