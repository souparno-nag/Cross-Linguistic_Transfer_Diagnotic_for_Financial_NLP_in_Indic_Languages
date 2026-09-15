"""T-500 — model-agnosticism smoke test (CLAUDE5.md).

    python -m scripts.t500_smoke --audit-only            # CPU, instant, no model
    python -m scripts.t500_smoke --device cuda           # the real thing, ~1-2 min
    python -m scripts.t500_smoke --device cuda --encoder-config configs/train/task3_hin_xlmr.yaml

CLAUDE5.md's premise is that adding an encoder "should be almost entirely
configuration", and T-500 exists to find out whether Phases 2 and 3 were in
fact written model-agnostically — *before* 11 GPU-hours are committed to
T-501/T-502 rather than after. So this script answers two separate questions
and reports them separately, because they have different answers:

**Leg A, the audit** (CPU, no model load). Static checks over the shipped code
and configs: is the encoder reachable through the registry, is the new config
a protocol-identical sibling of the IndicBERT one (hard rules 1-2), and can
the Phase 3 artefacts actually tell two encoders' rows apart. A failure here
is a *blocker for T-501*, not a smoke-test failure — it means an artefact
would be silently corrupted or a checkpoint unreachable at scale.

**Leg B, the smoke run** (GPU). One epoch on a stratified `--rows` subset,
end to end: load split -> tokenise -> build -> train -> checkpoint -> reload
frozen -> predict a *different* language's split. This is the "completes with
no code change" criterion, and it is run through the ordinary Phase 2/3
entry points (`src.data`, `src.train`, `src.inference`, `src.predictions`)
so that passing means those entry points really are encoder-agnostic.

Nothing here writes to a real artefact. The checkpoint goes to a disposable
`checkpoints/t500_smoke__*` directory (orphaned by T-208's retention rules by
construction, since no YAML bears that stem, so `t208_checkpoints prune`
removes it), and the prediction log goes under `cache/`, never
`data/predictions/`. Writing a smoke prediction into the real log is exactly
the collision this task exists to detect.

Leg B deliberately mirrors `config.run_training`'s sequence rather than
calling it, for one reason: `RunConfig` has no row-limit field, so a 200-row
run is not expressible as a config. That is a property of the smoke test, not
of the encoder — `--full` runs the real `run_training` on the whole split if
you want that proven too (slower; that is what T-501 does anyway).
"""

from __future__ import annotations

import argparse
import inspect
import json
import shutil
import sys
from dataclasses import fields
from pathlib import Path

from src.config import TRAIN_CONFIG_DIR, RunConfig, load_run_config, run_training
from src.download_dataset.paths import REPO_ROOT
from src.env_check import require_python

DEFAULT_CONFIG = TRAIN_CONFIG_DIR / "task2_hin_xlmr.yaml"
# The sibling each Phase 5 config must match field-for-field except `encoder`.
REFERENCE_ENCODER = "indicbert-v2"
SMOKE_CACHE = REPO_ROOT / "cache" / "t500_smoke"


# --------------------------------------------------------------------------
# Leg A — static audit. CPU, no model, instant.
# --------------------------------------------------------------------------


def _reference_config_for(run: RunConfig, path: Path) -> Path | None:
    """The IndicBERT config this one is supposed to be a protocol-identical
    sibling of: same task and language, the reference encoder."""
    for candidate in sorted(TRAIN_CONFIG_DIR.glob("*.yaml")):
        if candidate == path:
            continue
        other = load_run_config(candidate)
        if (other.encoder, other.task, other.lang) == (REFERENCE_ENCODER, run.task, run.lang):
            return candidate
    return None


def check_registry(run: RunConfig) -> dict:
    from src.data import ENCODERS, resolve_encoder

    try:
        enc = resolve_encoder(run.encoder)
    except ValueError as exc:
        return {"name": "encoder registry", "ok": False, "blocker": True, "detail": str(exc)}
    return {
        "name": "encoder registry",
        "ok": True,
        "blocker": True,
        "detail": f"{enc.key} -> {enc.hf_id} (registry holds {sorted(ENCODERS)})",
    }


def check_protocol_parity(run: RunConfig, path: Path) -> dict:
    """Hard rules 1-2: identical protocol, no per-encoder tuning.

    A comparison across encoders is only valid if nothing but the encoder
    moved, so this compares every `RunConfig` field against the reference
    config and fails on any difference other than `encoder` itself.
    """
    reference_path = _reference_config_for(run, path)
    if reference_path is None:
        return {
            "name": "protocol parity vs IndicBERT",
            "ok": False,
            "blocker": True,
            "detail": (
                f"no {REFERENCE_ENCODER} config for task {run.task} / {run.lang} to "
                "compare against — cannot confirm hard rule 1"
            ),
        }
    reference = load_run_config(reference_path)
    differing = [
        f.name
        for f in fields(RunConfig)
        if f.name != "encoder" and getattr(run, f.name) != getattr(reference, f.name)
    ]
    if differing:
        detail = ", ".join(
            f"{name}: {getattr(run, name)!r} vs {getattr(reference, name)!r}"
            for name in differing
        )
        return {
            "name": "protocol parity vs IndicBERT",
            "ok": False,
            "blocker": True,
            "detail": (
                f"{path.name} differs from {reference_path.name} in {len(differing)} "
                f"field(s) besides `encoder` — {detail}. Hard rule 2 allows this only "
                "when VRAM forces it, and then all three encoders must be tuned and "
                "it must be said so."
            ),
        }
    return {
        "name": "protocol parity vs IndicBERT",
        "ok": True,
        "blocker": True,
        "detail": f"{path.name} matches {reference_path.name} in every field but `encoder`",
    }


def check_checkpoint_naming(run: RunConfig, path: Path) -> dict:
    """Can the Phase 3 condition sweep address this encoder's checkpoint?

    `evaluate.plan()` builds the checkpoint name it will look for from
    `evaluate.run_id_for`. If that name does not match the run_id T-206 will
    actually write for this config, every condition comes back "blocked" and
    T-501 evaluates nothing.
    """
    from src.evaluate import run_id_for

    actual = f"{path.stem}_seed{run.seed}"
    expected = run_id_for(run.task, run.lang, run.seed)
    if actual == expected:
        return {
            "name": "condition sweep can find the checkpoint",
            "ok": True,
            "blocker": True,
            "detail": f"evaluate.run_id_for -> {expected!r}, matches T-206's {actual!r}",
        }
    return {
        "name": "condition sweep can find the checkpoint",
        "ok": False,
        "blocker": True,
        "detail": (
            f"evaluate.run_id_for(...) returns {expected!r} but T-206 writes this "
            f"config's checkpoint as {actual!r}. src/evaluate.py:run_id_for pastes "
            "'indicbert' into the name literally, so `plan()` reports every "
            "condition as blocked for any other encoder and T-501 would evaluate "
            "nothing. HARDCODED — fix before T-501."
        ),
    }


def check_artefacts_carry_encoder() -> list[dict]:
    """CLAUDE5.md: "`encoder_id` must be populated on every row of every
    artefact", and rule 1: "never write to a path or row that another
    encoder's run owns".

    Two distinct failures, so two checks. A missing *column* means an
    artefact cannot say which encoder produced it; a missing *path* level
    means two encoders' runs share one file — and the prediction log merges
    on append while the failure set overwrites, so the first silently
    doubles rows and the second silently destroys Phase 6's input.
    """
    from src.failures import failure_path
    from src.predictions import PREDICTION_COLUMNS, log_path

    out = []
    has_column = "encoder_id" in PREDICTION_COLUMNS
    out.append(
        {
            "name": "prediction log carries encoder_id",
            "ok": has_column,
            "blocker": True,
            "detail": (
                "PREDICTION_COLUMNS includes encoder_id"
                if has_column
                else "PREDICTION_COLUMNS has no encoder_id — the encoder is only "
                "recoverable by string-parsing run_id. MISSING — fix before T-501."
            ),
        }
    )

    for label, func in (("prediction log", log_path), ("failure set", failure_path)):
        params = inspect.signature(func).parameters
        namespaced = "encoder" in params or "encoder_id" in params
        out.append(
            {
                "name": f"{label} path is namespaced by encoder",
                "ok": namespaced,
                "blocker": True,
                "detail": (
                    f"{func.__module__}.{func.__name__} takes an encoder argument"
                    if namespaced
                    else f"{func.__module__}.{func.__name__}{tuple(params)} has no "
                    "encoder level, so every encoder writes the same file. "
                    "NOT NAMESPACED — fix before T-501."
                ),
            }
        )
    return out


def parameter_budget(encoder: str) -> dict:
    """Bytes a full fine-tune needs for parameters alone, before activations.

    Built on the `meta` device from the model's config, so it allocates
    nothing and downloads no weights. Under AMP the master weights stay
    fp32 and AdamW keeps two fp32 moments per parameter, so a full
    fine-tune costs **16 bytes per parameter** — 4 weights + 4 gradient +
    8 optimizer state — none of which depends on batch size or sequence
    length. That is the part a batch-size sweep cannot measure and cannot
    shrink.
    """
    import torch
    from transformers import AutoConfig, AutoModel

    from src.data import resolve_encoder

    hf_id = resolve_encoder(encoder).hf_id
    config = AutoConfig.from_pretrained(hf_id)
    with torch.device("meta"):
        model = AutoModel.from_config(config)
    n_params = sum(p.numel() for p in model.parameters())
    largest = max(model.parameters(), key=lambda p: p.numel()).numel()
    gib = 1024**3
    return {
        "hf_id": hf_id,
        "n_params": n_params,
        "weights_gib": n_params * 4 / gib,
        "grads_gib": n_params * 4 / gib,
        "adamw_gib": n_params * 8 / gib,
        "floor_gib": n_params * 16 / gib,
        "largest_tensor_mib": largest * 4 / 1024**2,
    }


def check_optimizer_fits(run: RunConfig, device: str) -> dict:
    """Will a full fine-tune of this encoder fit at all, at any batch size?

    T-205's budget table answers "largest batch that fits", which is a
    question about *activations*. It cannot see the fixed cost of weights,
    gradients and AdamW state, and for XLM-R that fixed cost alone exceeds
    this card. Checking it here means the answer arrives in the audit
    instead of eight minutes into a run.
    """
    import torch

    budget = parameter_budget(run.encoder)
    detail = (
        f"{budget['n_params'] / 1e6:.0f}M params -> {budget['weights_gib']:.2f} GiB "
        f"weights + {budget['grads_gib']:.2f} grads + {budget['adamw_gib']:.2f} AdamW "
        f"= {budget['floor_gib']:.2f} GiB fixed, before any activation "
        f"(largest single tensor {budget['largest_tensor_mib']:.0f} MiB per copy)"
    )
    if not (device.startswith("cuda") and torch.cuda.is_available()):
        return {
            "name": "full fine-tune fits in VRAM",
            "ok": True,
            "blocker": False,
            "detail": f"{detail}; no CUDA device visible, so not compared against one",
        }

    total_gib = torch.cuda.get_device_properties(0).total_memory / 1024**3
    if budget["floor_gib"] < total_gib:
        return {
            "name": "full fine-tune fits in VRAM",
            "ok": True,
            "blocker": True,
            "detail": f"{detail}; card has {total_gib:.2f} GiB total",
        }
    return {
        "name": "full fine-tune fits in VRAM",
        "ok": False,
        "blocker": True,
        "detail": (
            f"{detail}. The card has {total_gib:.2f} GiB total, so this does not fit "
            "at batch size 1 with a single token. VRAM-BLOCKED — this is fixed "
            "parameter cost, not activation cost, so reducing batch size, max_len or "
            "using gradient checkpointing cannot help. CLAUDE5.md hard rule 6: report "
            "the config, do not silently shrink it. T-503 is the sanctioned "
            "contingency."
        ),
    }


def run_audit(path: Path, *, seed: int = 0, device: str = "cuda") -> list[dict]:
    """Audit the config **as shipped**, not as the smoke run overrides it.

    Leg B loads the same YAML with `epochs=1, patience=1` so one epoch on a
    subset is quick; auditing that object would report the smoke's own
    overrides as a hard-rule-2 violation. The audit's subject is the file that
    T-501 will actually run.
    """
    shipped = load_run_config(path, seed=seed)
    return [
        check_registry(shipped),
        check_protocol_parity(shipped, path),
        check_optimizer_fits(shipped, device),
        check_checkpoint_naming(shipped, path),
        *check_artefacts_carry_encoder(),
    ]


# --------------------------------------------------------------------------
# Leg B — the end-to-end smoke run. GPU.
# --------------------------------------------------------------------------


def _stratified_subset(frame, *, rows: int, seed: int):
    """A class-stratified `rows`-row sample, via the project's own splitter.

    `stratified_split` is reused rather than reimplemented so the subset is
    drawn exactly the way every other fold in this project is, and is
    reproducible from `seed` alone (CLAUDE.md §4 rule 7).
    """
    from src.data import stratified_split

    if rows >= len(frame):
        return frame
    return stratified_split(frame, seed=seed, dev=0.0, test=rows / len(frame))["test"]


def run_smoke(
    run: RunConfig,
    path: Path,
    *,
    rows: int,
    device: str,
    batch_size: int,
    progress=print,
) -> dict:
    """One epoch on `rows` rows, then reload frozen and predict another
    language. Returns a summary; raises if any step cannot complete."""
    from src import checkpoints
    from src.data import (
        SplitDataset,
        get_tokenizer,
        load_split,
        resolve_encoder,
        stratified_split,
    )
    from src.ids import targets_for_block
    from src.inference import load_frozen_model
    from src.predictions import prediction_log_from_loaded, write_prediction_log
    from src.train import build_model, evaluate_dataset
    from src.train import train as train_loop

    block = run.resolved_block()
    run_id = f"t500_smoke__{run.encoder}__{path.stem}_seed{run.seed}"
    work_dir = checkpoints.checkpoint_dir(run_id)
    # Always start clean: the subset size is not part of the config hash, so a
    # checkpoint left by a previous `--rows` would resume against a different
    # dataset while passing train()'s hash check.
    if work_dir.exists():
        shutil.rmtree(work_dir)
    work_dir.mkdir(parents=True)

    progress(f"[1/6] loading task {run.task} {block}/{run.lang} native")
    frame = load_split(run.task, block, run.lang, run.origin)
    subset = _stratified_subset(frame, rows=rows, seed=run.seed)
    progress(f"      {len(frame)} rows -> {len(subset)}-row stratified subset")

    parts = stratified_split(
        subset, seed=run.seed, dev=run.dev_fraction, test=run.test_fraction
    )
    progress(f"[2/6] tokenising with {run.encoder}")
    tokenizer = get_tokenizer(run.encoder)
    datasets = {
        name: SplitDataset(part, tokenizer, max_len=run.max_len)
        for name, part in parts.items()
    }
    progress(
        "      "
        + ", ".join(f"{name} {len(ds)}" for name, ds in sorted(datasets.items()))
    )

    tcfg = run.train_config()
    progress(f"[3/6] building classifier (num_labels={tcfg.num_labels}) on {device}")
    model = build_model(tcfg, device)
    (work_dir / checkpoints.RUN_CONFIG_FILE).write_text(
        json.dumps(run.to_dict(), indent=2) + "\n"
    )

    progress("[4/6] training one epoch")
    result = train_loop(
        model,
        datasets["train"],
        datasets["dev"],
        tcfg,
        device=device,
        work_dir=work_dir,
        progress=lambda msg: progress(f"      {msg}"),
        run_hash=run.hash(),
    )
    eval_name = "test" if "test" in datasets else "dev"
    metrics = evaluate_dataset(model, datasets[eval_name], tcfg, device)

    progress(f"[5/6] reloading {run_id} frozen")
    loaded = load_frozen_model(run_id, device=device)
    assert not loaded.model.training, "reloaded model is not in eval mode"

    target_lang = targets_for_block(block)[0]
    progress(f"[6/6] zero-shot prediction on {block}/{target_lang} (mt)")
    keep = set(subset["item_id"])
    log = prediction_log_from_loaded(
        loaded,
        f"t500_smoke_{run.lang}_to_{target_lang}",
        run.task,
        block,
        target_lang,
        "mt",
        run.seed,
        batch_size=batch_size,
        item_filter=keep.__contains__,
    )
    # Written under cache/, never data/predictions/ — a smoke row in the real
    # log is precisely the collision this task exists to detect.
    log_out = write_prediction_log(
        log, run.task, f"t500_smoke_{run.lang}_to_{target_lang}", root=SMOKE_CACHE
    )
    if len(log) != len(subset):
        raise RuntimeError(
            f"predicted {len(log)} rows for {len(subset)} items — the target arm "
            "does not carry every source item_id"
        )
    if log.drop(columns=["src_lang"]).isna().any().any():
        raise RuntimeError("prediction log carries nulls outside src_lang")

    return {
        "run_id": run_id,
        "encoder": run.encoder,
        "hf_id": resolve_encoder(run.encoder).hf_id,
        "n_subset": len(subset),
        "n_train": len(datasets["train"]),
        "eval_fold": eval_name,
        "epochs_run": result.epochs_run,
        "train_macro_f1": result.history[-1]["train_macro_f1"],
        "dev_macro_f1": result.best_dev_macro_f1,
        "eval_macro_f1": metrics["macro_f1"],
        "eval_accuracy": metrics["accuracy"],
        "target_lang": target_lang,
        "n_predicted": len(log),
        "prediction_path": str(log_out),
        "checkpoint_dir": str(work_dir),
    }


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    require_python()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--encoder-config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--rows", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument(
        "--audit-only", action="store_true", help="Leg A only; loads no model"
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="also run the real config.run_training on the whole split (slow)",
    )
    parser.add_argument(
        "--force-smoke",
        action="store_true",
        help="run Leg B even when the audit has proven it cannot fit in VRAM",
    )
    args = parser.parse_args(argv)

    run = load_run_config(args.encoder_config, seed=args.seed, epochs=1, patience=1)
    print(f"T-500 model-agnosticism smoke test — {args.encoder_config.name}")
    print(f"encoder {run.encoder} | task {run.task} | {run.split_id()}\n")

    print("Leg A — static audit (no model loaded)")
    audit = run_audit(args.encoder_config, seed=args.seed, device=args.device)
    for check in audit:
        print(f"  [{'PASS' if check['ok'] else 'FAIL'}] {check['name']}")
        print(f"         {check['detail']}")
    blockers = [c for c in audit if not c["ok"] and c["blocker"]]
    print()

    vram_blocked = any(
        not c["ok"] and c["name"] == "full fine-tune fits in VRAM" for c in audit
    )

    smoke_ok = True
    leg_b_ran = False
    if args.audit_only:
        print("Leg B — skipped (--audit-only)\n")
    elif vram_blocked and not args.force_smoke:
        print(
            "Leg B — skipped: the audit proved this encoder's fixed parameter cost\n"
            "  exceeds the card, so the run can only OOM. Re-run with --force-smoke\n"
            "  to see it fail on the device anyway.\n"
        )
    else:
        leg_b_ran = True
        print(f"Leg B — end-to-end smoke ({args.rows} rows, 1 epoch, {args.device})")
        try:
            summary = run_smoke(
                run,
                args.encoder_config,
                rows=args.rows,
                device=args.device,
                batch_size=args.batch_size,
                progress=lambda msg: print(f"  {msg}"),
            )
        except Exception as exc:
            smoke_ok = False
            print(f"\n  SMOKE FAILED: {type(exc).__name__}: {exc}", file=sys.stderr)
        else:
            print("\n  end-to-end result:")
            for key, value in summary.items():
                shown = f"{value:.4f}" if isinstance(value, float) else value
                print(f"    {key:<18} {shown}")

        if args.full:
            print("\n  --full: running config.run_training on the whole split")
            try:
                full = run_training(run, device=args.device, work_dir=None)
            except Exception as exc:
                smoke_ok = False
                print(f"  run_training FAILED: {type(exc).__name__}: {exc}", file=sys.stderr)
            else:
                print(
                    f"    run_training ok — macro-F1 {full['macro_f1']:.4f} "
                    f"on {full['n_eval']} {full['eval_fold']} rows"
                )
        print()

    print("=" * 72)
    if smoke_ok and leg_b_ran:
        print(
            "Leg B PASSED: training and inference are encoder-agnostic — this "
            "encoder ran end to end through the shipped Phase 2/3 entry points "
            "with no change to src/."
        )
    if blockers:
        print(
            f"\nLeg A found {len(blockers)} blocker(s) for T-501. These are not "
            "smoke-test failures; they are places where adding an encoder needs "
            "more than a config entry, which is what T-500 exists to surface:"
        )
        for check in blockers:
            print(f"  - {check['name']}: {check['detail']}")
        print(
            "\nPer CLAUDE5.md T-500, stop here and report rather than working "
            "around these — fixing them now is cheaper than after 11 GPU-hours."
        )
    elif not args.audit_only:
        print("Leg A PASSED: no blockers. T-501 is clear to run.")

    return 0 if (smoke_ok and not blockers) else 1


if __name__ == "__main__":
    sys.exit(main())
