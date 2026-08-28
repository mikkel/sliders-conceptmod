"""Keep ModularPipeline sample and train encode on the same PEFT modules.

After ``get_peft_model`` / ``PeftModel.from_pretrained``, a bare

    pipe.text_conditioner = wrapped

updates the Python attribute. Live ``ModularPipeline.components`` is a
*property* that re-reads ``getattr(self, name)``, and ``pipe(prompt=...)``
passes ``self`` into the blocks — so a successful setattr already shares
the object with ``AnimaTextConditioningStep``.

That is **not** enough:

- ``update_components`` also rewrites ``_component_specs`` and the
  ComponentsManager. Without it, a later ``load_components`` / manager
  lookup can resurrect the pre-PEFT ``AnimaTextConditioner``.
- Dummy / test pipes may store a **stale** ``components`` dict or block
  attribute captured at ``load_components`` time.
- ``transformer.text_conditioner`` (fake / nested) can stay on the
  unwrapped module while ``pipe.text_conditioner`` is the PEFT wrapper.

Train ``LiveAnimaBackend._encode_raw`` uses ``self.pipe.text_conditioner``.
Sample ``emit_inprocess_samples`` uses ``pipe(prompt=...)``. If those are
different ``nn.Module`` instances, the 0→0.25 gate understates the
concept (v6/v7 suspicion). This helper writes PEFT modules into every
holder and asserts ``id()`` match.

CPU-pure. No Hub, no GPU.
"""

from __future__ import annotations

from typing import Any, Iterable

import torch
import torch.nn.functional as F

PEFT_COMPONENT_NAMES = ("transformer", "text_conditioner")
SAMPLE_TRAIN_MISMATCH = "SAMPLE_TRAIN_MISMATCH"
# Scale-1 student Δ vs frozen plus−neu teacher. Near-zero student is
# the "adapter not in the encode graph" failure, not a weak recipe.
EMBED_ALIGN_MIN = 0.15
EMBED_REL_MIN = 0.05
EMBED_ABS_MIN = 1e-5


def _holder_get(holder: Any, name: str):
    if holder is None:
        return None
    if isinstance(holder, dict):
        return holder.get(name)
    return getattr(holder, name, None)


def _holder_set(holder: Any, name: str, module) -> bool:
    if holder is None:
        return False
    if isinstance(holder, dict):
        holder[name] = module
        return True
    try:
        setattr(holder, name, module)
        return True
    except (TypeError, AttributeError):
        return False


def _sub_blocks(blocks) -> Iterable[tuple[str, Any]]:
    if blocks is None:
        return []
    sub = getattr(blocks, "sub_blocks", None)
    if isinstance(sub, dict):
        return list(sub.items())
    if isinstance(sub, (list, tuple)):
        return [(str(i), block) for i, block in enumerate(sub)]
    return []


def iter_modular_component_holders(pipe) -> list[tuple[str, Any]]:
    """Places that may hold ``transformer`` / ``text_conditioner``.

    Walk ``_blocks`` only. Live ``pipe.blocks`` is a ``deepcopy`` and is
    not the object ``pipe(prompt=...)`` executes.
    """
    holders: list[tuple[str, Any]] = [("pipe", pipe)]
    components = getattr(pipe, "components", None)
    if components is not None:
        holders.append(("components", components))
    # Do not use the ``blocks`` property — ModularPipeline copies it.
    blocks = getattr(pipe, "_blocks", None)
    if blocks is not None:
        holders.append(("_blocks", blocks))
        for bname, block in _sub_blocks(blocks):
            holders.append((f"_blocks.{bname}", block))
            for nname, nested in _sub_blocks(block):
                holders.append((f"_blocks.{bname}.{nname}", nested))
    transformer = getattr(pipe, "transformer", None)
    if transformer is not None:
        holders.append(("pipe.transformer", transformer))
    return holders


def resolve_modular_component(pipe, name: str):
    """Best-effort live component: attribute, then ``components`` map."""
    module = getattr(pipe, name, None)
    if module is not None:
        return module
    components = getattr(pipe, "components", None)
    return _holder_get(components, name)


def collect_modular_component_ids(pipe, name: str) -> dict[str, int]:
    """``{holder_label: id(module)}`` for every holder that has ``name``."""
    found: dict[str, int] = {}
    for label, holder in iter_modular_component_holders(pipe):
        module = _holder_get(holder, name)
        if module is None:
            continue
        found[label] = id(module)
    return found


def _adapted_updates(pipe, transformer=None, text_conditioner=None) -> dict[str, Any]:
    updates: dict[str, Any] = {}
    if transformer is not None:
        updates["transformer"] = transformer
    if text_conditioner is not None:
        updates["text_conditioner"] = text_conditioner
    if updates:
        return updates
    for name in PEFT_COMPONENT_NAMES:
        module = getattr(pipe, name, None)
        if module is not None:
            updates[name] = module
    return updates


def sync_peft_into_modular_pipeline(
    pipe,
    *,
    transformer=None,
    text_conditioner=None,
    assert_shared: bool = True,
) -> dict[str, Any]:
    """Write PEFT modules into ModularPipeline holders. Assert id match.

    Prefers ``pipe.update_components(...)`` (official API). Always also
    assigns attributes and patches stale ``components`` / block refs so
    dummy pipes and ComponentsManager leftovers stay on the PEFT object.

    Returns a report with shared ``id()`` values. Raises if ``assert_shared``
    and any holder still points at a different instance.
    """
    if pipe is None:
        raise RuntimeError("sync_peft_into_modular_pipeline needs a pipe")
    updates = _adapted_updates(
        pipe, transformer=transformer, text_conditioner=text_conditioner
    )
    if not updates:
        return {"ok": True, "updated": [], "ids": {}}

    updater = getattr(pipe, "update_components", None)
    if callable(updater):
        try:
            updater(**updates)
        except Exception:
            for name, module in updates.items():
                setattr(pipe, name, module)
    else:
        for name, module in updates.items():
            setattr(pipe, name, module)

    for name, module in updates.items():
        for _label, holder in iter_modular_component_holders(pipe):
            current = _holder_get(holder, name)
            if current is None or current is module:
                continue
            _holder_set(holder, name, module)

    report = assert_peft_modules_synced(pipe, **updates)
    report["updated"] = list(updates)
    if not assert_shared:
        return report
    return report


def assert_peft_modules_synced(pipe, **modules) -> dict[str, Any]:
    """Fail if sample holders and the PEFT module are different objects."""
    if not modules:
        modules = _adapted_updates(pipe)
    mismatches: list[str] = []
    ids: dict[str, dict[str, Any]] = {}
    for name, module in modules.items():
        if module is None:
            continue
        refs = collect_modular_component_ids(pipe, name)
        attr = getattr(pipe, name, None)
        if attr is not None and attr is not module:
            mismatches.append(
                f"pipe.{name} id={id(attr)} != peft id={id(module)}"
            )
        for label, ident in refs.items():
            if ident != id(module):
                mismatches.append(
                    f"{label}.{name} id={ident} != peft id={id(module)}"
                )
        ids[name] = {"peft": id(module), "refs": refs}
    if mismatches:
        raise RuntimeError(
            "PEFT ModularPipeline sync failed; pipe(prompt=...) would "
            "not share train encode modules: " + "; ".join(mismatches)
        )
    return {"ok": True, "ids": ids}


def sample_conditioner_module(backend):
    """Module the sample helper would encode with (pipe, then backend)."""
    pipe = getattr(backend, "pipe", None)
    if pipe is not None:
        cond = resolve_modular_component(pipe, "text_conditioner")
        if cond is not None:
            return cond
    return getattr(getattr(backend, "transformer", None), "text_conditioner", None)


def train_conditioner_module(backend):
    """Module ``encode_text`` / ``_encode_raw`` calls."""
    pipe = getattr(backend, "pipe", None)
    if pipe is not None:
        cond = getattr(pipe, "text_conditioner", None)
        if cond is not None:
            return cond
    return getattr(getattr(backend, "transformer", None), "text_conditioner", None)


def assert_sample_train_conditioner_shared(backend) -> dict[str, Any]:
    """Sample helper and ``encode_text`` must be the same conditioner object."""
    sample = sample_conditioner_module(backend)
    train = train_conditioner_module(backend)
    if sample is None or train is None:
        raise RuntimeError(
            "cannot prove sample/train conditioner share: "
            f"sample={sample!r} train={train!r}"
        )
    if sample is not train:
        raise RuntimeError(
            f"{SAMPLE_TRAIN_MISMATCH}: sample conditioner id={id(sample)} "
            f"!= encode_text conditioner id={id(train)}"
        )
    return {"ok": True, "id": id(train)}


def adapter_scale_api(module) -> list[str]:
    """Which PEFT scale/disable hooks exist (conditioner wrapper check)."""
    names = []
    for name in (
        "set_adapter_scale",
        "set_adapters",
        "set_lora_scale",
        "disable_adapter",
        "enable_adapter_layers",
        "enable_adapters",
    ):
        if callable(getattr(module, name, None)):
            names.append(name)
    return names


def _pool_embed(embed: torch.Tensor) -> torch.Tensor:
    """Mean-pool token axis so neu/plus length differences do not break Δ."""
    x = embed.detach().float()
    if x.ndim >= 3:
        return x.mean(dim=-2)
    if x.ndim == 2 and x.shape[0] != 1:
        return x.mean(dim=0, keepdim=True)
    return x


def _flatten_embed(embed: torch.Tensor) -> torch.Tensor:
    return _pool_embed(embed).reshape(-1)


def _cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    x = _flatten_embed(a).unsqueeze(0)
    y = _flatten_embed(b).unsqueeze(0)
    return float(F.cosine_similarity(x, y, dim=1, eps=1e-6).item())


def measure_conditioner_embed_deltas(
    encode_fn,
    *,
    neu: str,
    plus: str,
    scales: tuple[float, ...] = (0.0, 0.25, 0.5, 1.0),
    align_min: float = EMBED_ALIGN_MIN,
    rel_min: float = EMBED_REL_MIN,
    abs_min: float = EMBED_ABS_MIN,
) -> dict[str, Any]:
    """Compare student Δ(neu, s) to frozen teacher Δ(plus − neu).

    ``encode_fn(prompt, scale)`` must honor adapter scale (0 = frozen).
    PASS at scale 1 requires a student Δ that is not near-zero and is
    meaningfully aligned with the teacher Δ.
    """
    e_neu_0 = encode_fn(neu, 0.0)
    e_plus_0 = encode_fn(plus, 0.0)
    teacher = e_plus_0 - e_neu_0
    teacher_norm = float(_flatten_embed(teacher).norm().item())
    rows: list[dict[str, Any]] = []
    for scale in scales:
        e_neu_s = encode_fn(neu, float(scale))
        student = e_neu_s - e_neu_0
        student_norm = float(_flatten_embed(student).norm().item())
        rel = student_norm / (teacher_norm + 1e-12)
        cos = _cosine(student, teacher) if student_norm > 0 and teacher_norm > 0 else 0.0
        rows.append(
            {
                "scale": float(scale),
                "student_l2": student_norm,
                "teacher_l2": teacher_norm,
                "student_over_teacher": rel,
                "cosine": cos,
            }
        )
    scale1 = next((row for row in rows if abs(row["scale"] - 1.0) < 1e-8), None)
    if scale1 is None:
        verdict = "FAIL"
        reason = "missing scale 1.0"
    elif scale1["student_l2"] < abs_min and scale1["student_over_teacher"] < rel_min:
        verdict = "FAIL"
        reason = (
            "scale-1 student Δ is near-zero "
            f"(l2={scale1['student_l2']:.4g}, rel={scale1['student_over_teacher']:.4g}); "
            "adapter is not moving E(neu) — check PEFT apply / sync"
        )
    elif scale1["cosine"] < align_min:
        verdict = "FAIL"
        reason = (
            f"scale-1 cosine(student Δ, teacher Δ)={scale1['cosine']:.4f} "
            f"< {align_min}; student Δ is not aligned with plus−neu"
        )
    else:
        verdict = "PASS"
        reason = (
            f"scale-1 student Δ l2={scale1['student_l2']:.4g} "
            f"rel={scale1['student_over_teacher']:.4g} "
            f"cosine={scale1['cosine']:.4f}"
        )
    return {
        "neu": neu,
        "plus": plus,
        "teacher_l2": teacher_norm,
        "rows": rows,
        "verdict": verdict,
        "reason": reason,
        "align_min": align_min,
        "rel_min": rel_min,
        "abs_min": abs_min,
    }


def compare_sample_train_embeds(
    train_embed: torch.Tensor,
    sample_embed: torch.Tensor,
    *,
    rtol: float = 1e-4,
    atol: float = 1e-4,
) -> dict[str, Any]:
    """Flag SAMPLE_TRAIN_MISMATCH when modular encode ≠ backend.encode_text."""
    a = _flatten_embed(train_embed)
    b = _flatten_embed(sample_embed)
    n = min(a.numel(), b.numel())
    if n == 0:
        return {
            "match": False,
            "flag": SAMPLE_TRAIN_MISMATCH,
            "reason": "empty embed",
            "max_abs": None,
            "cosine": None,
        }
    a, b = a[:n], b[:n]
    max_abs = float((a - b).abs().max().item())
    cos = float(F.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0), dim=1, eps=1e-6).item())
    close = bool(torch.allclose(a, b, rtol=rtol, atol=atol))
    if close:
        return {
            "match": True,
            "flag": None,
            "reason": "sample encode matches train encode_text",
            "max_abs": max_abs,
            "cosine": cos,
        }
    return {
        "match": False,
        "flag": SAMPLE_TRAIN_MISMATCH,
        "reason": (
            f"{SAMPLE_TRAIN_MISMATCH}: modular/sample encode ≠ "
            f"backend.encode_text (max_abs={max_abs:.4g}, cosine={cos:.4f})"
        ),
        "max_abs": max_abs,
        "cosine": cos,
    }
