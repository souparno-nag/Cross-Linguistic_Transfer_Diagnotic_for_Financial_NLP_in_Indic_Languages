"""T-401 — source-block smoke test (CLAUDE6.md).

    python -m scripts.t401_smoke --audit-only                     # CPU, instant
    python -m scripts.t401_smoke --device cuda                    # the real thing
    python -m scripts.t401_smoke --config configs/train/task2_tel_indicbert.yaml

CLAUDE6.md's premise is that "adding a new source block is a config entry", and
T-401 exists to find out whether that is true *before* the grid is run rather
than after. It is the block-dimension twin of T-500, which asked the same
question about encoders, and it is deliberately built on T-500's machinery:
the end-to-end leg is literally `t500_smoke.run_smoke`, because a smoke test
that exercised a different code path from the one Phase 5 proved would be
testing something other than the pipeline.

What is new here is one static check, and it is the one CLAUDE6.md hard rule 1
turns on: **cross-block protocol parity.** T-500's parity check compares a
config against the same-language IndicBERT config, so it has nothing to say
about `task2_ben_indicbert.yaml` — that config *is* the IndicBERT one for its
language. Left at that, a Bengali baseline could quietly carry a different
learning rate or schedule from the Hindi one it is about to be compared
against, and T-408's source-selection table would report the difference
between two training recipes as a difference between two source languages.

The two parities are different claims and both must hold for a config that is
new in both dimensions (`task2_ben_mbert.yaml`):

* T-500: against the same-block IndicBERT config, only `encoder` may differ.
* T-401: against the same-encoder Hindi config, only `lang` may differ.

Nothing here writes to a real artefact — the checkpoint goes to a disposable
`checkpoints/t500_smoke__*` directory and the prediction log under `cache/`,
exactly as T-500 does.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import fields
from pathlib import Path

from scripts.t500_smoke import (
    check_artefacts_carry_encoder,
    check_checkpoint_naming,
    check_optimizer_fits,
    check_registry,
    run_smoke,
)
from src.config import TRAIN_CONFIG_DIR, RunConfig, load_run_config
from src.env_check import require_python

DEFAULT_CONFIG = TRAIN_CONFIG_DIR / "task2_ben_indicbert.yaml"
# The block every other block's config must match field-for-field except `lang`.
# Hindi, because block H is what Phases 2, 3 and 5 actually ran.
REFERENCE_LANG = "hin"


# --------------------------------------------------------------------------
# Leg A — static audit. CPU, no model, instant.
# --------------------------------------------------------------------------


def reference_config_for(run: RunConfig, path: Path) -> Path | None:
    """The Hindi config this one must be a protocol-identical sibling of:
    same task, same encoder, the reference language."""
    for candidate in sorted(TRAIN_CONFIG_DIR.glob("*.yaml")):
        if candidate == path:
            continue
        other = load_run_config(candidate)
        if (other.lang, other.task, other.encoder) == (
            REFERENCE_LANG,
            run.task,
            run.encoder,
        ):
            return candidate
    return None


def check_block_parity(run: RunConfig, path: Path) -> dict:
    """CLAUDE6.md hard rules 1 and 6: identical protocol, no tuning per block.

    Compares every `RunConfig` field against the Hindi config for the same task
    and encoder, and fails on any difference other than `lang` itself. There is
    **no VRAM exception here**, unlike T-500's encoder parity: changing the
    source language does not change the model, the sequence length or the
    activation footprint, so nothing about a different block can force a
    different batch shape. A config that needed one would be evidence of
    tuning, not of hardware.

    A config for the reference language itself has no sibling to compare
    against and is reported as not applicable rather than as a failure.
    """
    if run.lang == REFERENCE_LANG:
        return {
            "name": "protocol parity vs block H",
            "ok": True,
            "blocker": False,
            "detail": (
                f"{path.name} is itself the block-H reference for task {run.task} / "
                f"{run.encoder}; nothing to compare it against"
            ),
        }

    reference_path = reference_config_for(run, path)
    if reference_path is None:
        return {
            "name": "protocol parity vs block H",
            "ok": False,
            "blocker": True,
            "detail": (
                f"no {REFERENCE_LANG} config for task {run.task} / {run.encoder} to "
                "compare against — cannot confirm hard rule 1, and without it a "
                "source-block comparison would be confounded with a training-recipe "
                "difference"
            ),
        }

    reference = load_run_config(reference_path)
    differing = [
        f.name
        for f in fields(RunConfig)
        if f.name != "lang" and getattr(run, f.name) != getattr(reference, f.name)
    ]
    # `block` is derived from `lang`, so it moves with it by construction and is
    # not a protocol difference.
    differing = [name for name in differing if name != "block"]
    if differing:
        detail = ", ".join(
            f"{name}: {getattr(run, name)!r} vs {getattr(reference, name)!r}"
            for name in differing
        )
        return {
            "name": "protocol parity vs block H",
            "ok": False,
            "blocker": True,
            "detail": (
                f"{path.name} differs from {reference_path.name} in "
                f"{len(differing)} field(s) besides `lang` — {detail}. CLAUDE6.md "
                "hard rule 6 forbids tuning one block: if this is genuinely needed, "
                "every block must be tuned the same way and it must be said so, "
                "because otherwise T-408 reports a recipe difference as a "
                "source-language difference."
            ),
        }
    return {
        "name": "protocol parity vs block H",
        "ok": True,
        "blocker": True,
        "detail": (
            f"{path.name} matches {reference_path.name} in every field but `lang` "
            f"(block {run.resolved_block()} against block "
            f"{reference.resolved_block()})"
        ),
    }


def check_native_split_available(run: RunConfig) -> dict:
    """Is there actually a native split to train this block on?

    Malayalam has none anywhere (CLAUDE.md §2) and `RunConfig` already refuses
    it, but a native split can also be present and unusable — missing a class,
    or machine-translated despite the name. `load_split` asserts the origin;
    this reports the shape so a thin or skewed block is visible in the audit
    rather than discovered as a strange baseline number.
    """
    from src.data import load_split, num_labels

    block = run.resolved_block()
    try:
        frame = load_split(run.task, block, run.lang, run.origin)
    except Exception as exc:
        return {
            "name": "native split loads",
            "ok": False,
            "blocker": True,
            "detail": f"load_split(task {run.task}, {block}/{run.lang}, "
            f"{run.origin!r}) raised: {exc}",
        }

    k = num_labels(run.task)
    present = sorted(int(x) for x in frame["label_id"].unique())
    missing = [i for i in range(k) if i not in present]
    if missing:
        return {
            "name": "native split loads",
            "ok": False,
            "blocker": True,
            "detail": (
                f"{len(frame)} rows but label_id(s) {missing} absent of {k} — a "
                "stratified split cannot hold out a class with no rows, and macro-F1 "
                "over an absent class is not comparable with block H's"
            ),
        }
    counts = frame["label_id"].value_counts().sort_index().to_dict()
    return {
        "name": "native split loads",
        "ok": True,
        "blocker": True,
        "detail": (
            f"task {run.task} {block}/{run.lang} native: {len(frame)} rows, "
            f"all {k} classes present ({counts})"
        ),
    }


def check_conditions_unblocked(run: RunConfig) -> dict:
    """Which evaluation conditions does this cell actually unblock?

    The point of training a new source block is the conditions it turns from
    blocked into runnable. If the checkpoint this config writes is not the one
    `evaluate.plan()` looks for, the sweep still reports everything blocked and
    the GPU time bought nothing — so this asks the matrix directly rather than
    inferring it from the naming convention.
    """
    from src.evaluate import load_matrix, plan

    try:
        matrix = load_matrix(run.task)
    except KeyError as exc:
        return {
            "name": "condition matrix has work for this block",
            "ok": False,
            "blocker": True,
            "detail": str(exc),
        }

    run_id = f"task{run.task}_{run.lang}_" f"{_slug(run.encoder)}_seed{run.seed}"
    entries = [
        e
        for e in plan(run.task, matrix, seeds=(run.seed,), encoder=run.encoder)
        if e["run_id"] == run_id
    ]
    if not entries:
        return {
            "name": "condition matrix has work for this block",
            "ok": False,
            "blocker": True,
            "detail": (
                f"no condition in the task-{run.task} matrix names checkpoint "
                f"{run_id!r}, so training it would unblock nothing. Either the "
                "config is misnamed or the matrix does not cover this block."
            ),
        }
    names = sorted(e["condition"] for e in entries)
    return {
        "name": "condition matrix has work for this block",
        "ok": True,
        "blocker": True,
        "detail": (
            f"{run_id} is the checkpoint for {len(names)} condition(s) at seed "
            f"{run.seed}: {', '.join(names)}"
        ),
    }


def _slug(encoder: str) -> str:
    from src.data import resolve_encoder

    return resolve_encoder(encoder).slug


def run_audit(path: Path, *, seed: int = 0, device: str = "cuda") -> list[dict]:
    """Audit the config **as shipped**, not as the smoke run overrides it —
    Leg B loads the same YAML with `epochs=1`, and auditing that object would
    report the smoke's own overrides as a parity violation."""
    shipped = load_run_config(path, seed=seed)
    return [
        check_registry(shipped),
        check_block_parity(shipped, path),
        check_native_split_available(shipped),
        check_conditions_unblocked(shipped),
        check_optimizer_fits(shipped, device),
        check_checkpoint_naming(shipped, path),
        *check_artefacts_carry_encoder(),
    ]


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    require_python()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--rows", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument(
        "--audit-only", action="store_true", help="Leg A only; loads no model"
    )
    args = parser.parse_args(argv)

    run = load_run_config(args.config, seed=args.seed, epochs=1, patience=1)
    print(f"T-401 source-block smoke test — {args.config.name}")
    print(
        f"block {run.resolved_block()} | task {run.task} | encoder {run.encoder} | "
        f"{run.split_id()}\n"
    )

    print("Leg A — static audit (no model loaded)")
    audit = run_audit(args.config, seed=args.seed, device=args.device)
    for check in audit:
        print(f"  [{'PASS' if check['ok'] else 'FAIL'}] {check['name']}")
        print(f"         {check['detail']}")
    blockers = [c for c in audit if not c["ok"] and c["blocker"]]
    print()

    if blockers:
        print(
            f"{len(blockers)} blocker(s) — CLAUDE6.md T-401 says stop and report what "
            "is hardcoded to block H rather than working around it:",
            file=sys.stderr,
        )
        for check in blockers:
            print(f"  - {check['name']}: {check['detail']}", file=sys.stderr)
        return 1

    if args.audit_only:
        print("audit-only: Leg B skipped.")
        return 0

    print(f"Leg B — end-to-end on a {args.rows}-row subset")
    try:
        summary = run_smoke(
            run,
            args.config,
            rows=args.rows,
            device=args.device,
            batch_size=args.batch_size,
        )
    except Exception as exc:
        print(f"\nLeg B FAILED: {exc}", file=sys.stderr)
        return 1

    print()
    for key, value in summary.items():
        print(f"  {key}: {value}")
    print(
        f"\nT-401 PASSES for block {run.resolved_block()} — the source block is a "
        "config entry, with no code change."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
