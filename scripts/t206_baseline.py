"""T-206 — in-language baselines.

    python -m scripts.t206_baseline                    # every configs/train/*.yaml, seeds 0,1,2
    python -m scripts.t206_baseline --seeds 0,1,2 --device cuda
    python -m scripts.t206_baseline --config configs/train/task2_hin_indicbert.yaml
    python -m scripts.t206_baseline --dry-run

Fine-tunes each shipped config at three seeds, logs every run to
``experiments.csv`` with its config hash (`hard rule 4`), and writes
``reports/baselines.{md,parquet}`` with mean ± std macro-F1 per config
(`hard rule 3` — never a single number).

Resumable at two levels: each run resumes from its own ``checkpoints/<run_id>/``
checkpoint, and a run already in ``experiments.csv`` under the same config hash
is skipped entirely. GPU access is intermittent (CLAUDE.md §3), so re-running
the command picks up where it stopped.
"""

from __future__ import annotations

import argparse
import shutil
import statistics
import sys
from pathlib import Path

import pandas as pd

from src.config import TRAIN_CONFIG_DIR, load_run_config, run_training
from src.download_dataset.paths import REPO_ROOT
from src.env_check import require_python
from src.experiments import RESULTS_PATH, log_run, read_results

CHECKPOINT_ROOT = REPO_ROOT / "checkpoints"
REPORT_DIR = REPO_ROOT / "reports"
DEFAULT_SEEDS = (0, 1, 2)


def plan_runs(config_paths, seeds) -> list[dict]:
    """One entry per (config, seed): the run_id and where its checkpoint lives."""
    plan = []
    for path in config_paths:
        for seed in seeds:
            run_id = f"{path.stem}_seed{seed}"
            plan.append({"config_path": path, "seed": seed, "run_id": run_id})
    return plan


def already_done(run_id: str, config_hash: str, results_path) -> dict | None:
    """The logged row for this run under this exact config hash, if any."""
    for row in read_results(results_path):
        if row["run_id"] == run_id and row["config_hash"] == config_hash:
            return row
    return None


def report_records(config_paths, seeds, results_path) -> list[dict]:
    """Rebuild the report inputs from ``experiments.csv`` for every shipped
    config at its *current* hash.

    So a partial re-run (`--config task3...`) refreshes task 3 in the report
    without dropping task 2, and a config whose YAML changed since its last run
    is simply absent until it is re-run — stale rows are never shown.
    """
    latest: dict[tuple[str, str], dict] = {}
    for row in read_results(results_path):
        latest[(row["run_id"], row["config_hash"])] = row  # last write wins

    out: list[dict] = []
    for path in config_paths:
        for seed in seeds:
            run = load_run_config(path, seed=seed)
            row = latest.get((f"{path.stem}_seed{seed}", run.hash()))
            if row is None:
                continue
            out.append(
                {
                    "config": path.name,
                    "run_id": f"{path.stem}_seed{seed}",
                    "seed": seed,
                    "split": run.split_id(),
                    "encoder": run.encoder,
                    "macro_f1": float(row["macro_f1"]),
                    "accuracy": float(row["accuracy"]),
                }
            )
    return out


def stale_checkpoint(work_dir: Path, config_hash: str) -> bool:
    """True if ``work_dir`` holds a checkpoint from a *different* config.

    Read from the checkpoint's own ``config_hash`` — the sidecar
    ``run_config.json`` is written before training starts, so a failed re-run
    leaves it pointing at the new config while the checkpoint is still the old
    one. A changed YAML is a legitimate reason to start over; `train()` stays
    strict and the orchestrator clears the stale dir.
    """
    ckpt = work_dir / "checkpoint.pt"
    if not ckpt.exists():
        return False
    import torch

    try:
        blob = torch.load(ckpt, map_location="cpu", weights_only=False)
    except Exception:
        return True  # unreadable -> cannot trust it, redo
    return blob.get("config_hash") != config_hash


def summarise(records: list[dict]) -> list[dict]:
    """Per config: mean ± sample-std of macro-F1 and accuracy across seeds."""
    out = []
    by_config: dict[str, list[dict]] = {}
    for rec in records:
        by_config.setdefault(rec["config"], []).append(rec)
    for config, group in sorted(by_config.items()):
        f1s = [r["macro_f1"] for r in group]
        accs = [r["accuracy"] for r in group]
        out.append(
            {
                "config": config,
                "split": group[0]["split"],
                "encoder": group[0]["encoder"],
                "seeds": sorted(r["seed"] for r in group),
                "n": len(group),
                "macro_f1_mean": statistics.mean(f1s),
                "macro_f1_std": statistics.stdev(f1s) if len(f1s) > 1 else 0.0,
                "accuracy_mean": statistics.mean(accs),
                "accuracy_std": statistics.stdev(accs) if len(accs) > 1 else 0.0,
                "macro_f1_by_seed": {r["seed"]: r["macro_f1"] for r in group},
            }
        )
    return out


def write_report(summary: list[dict], records: list[dict], report_dir: Path) -> None:
    report_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(records).to_parquet(report_dir / "baselines.parquet", index=False)

    lines = [
        "# In-language baselines — T-206",
        "",
        "IndicBERT-v2 fine-tuned on each native Hindi split, three seeds. "
        "Numbers are on the held-out **test** fold. Compared against IndicFinNLP's "
        "published monolingual baseline in T-207.",
        "",
        "| config | split | seeds | macro-F1 (mean ± std) | accuracy (mean ± std) |",
        "|---|---|---|---|---|",
    ]
    for s in summary:
        seeds = ",".join(str(x) for x in s["seeds"])
        lines.append(
            f"| {s['config']} | {s['split']} | {seeds} | "
            f"{s['macro_f1_mean']:.4f} ± {s['macro_f1_std']:.4f} | "
            f"{s['accuracy_mean']:.4f} ± {s['accuracy_std']:.4f} |"
        )
    lines += ["", "## Per-seed macro-F1", "", "| config | seed | macro-F1 | accuracy |", "|---|---|---|---|"]
    for r in sorted(records, key=lambda r: (r["config"], r["seed"])):
        lines.append(
            f"| {r['config']} | {r['seed']} | {r['macro_f1']:.4f} | {r['accuracy']:.4f} |"
        )
    lines.append("")
    (report_dir / "baselines.md").write_text("\n".join(lines))


def main() -> int:
    require_python()
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", action="append", type=Path,
        help="a configs/train/*.yaml; repeatable. Default: all of them.",
    )
    parser.add_argument("--seeds", default=",".join(str(s) for s in DEFAULT_SEEDS))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    config_paths = sorted(args.config or TRAIN_CONFIG_DIR.glob("*.yaml"))
    if not config_paths:
        print(f"no configs in {TRAIN_CONFIG_DIR}", file=sys.stderr)
        return 2
    seeds = [int(s) for s in args.seeds.split(",")]
    plan = plan_runs(config_paths, seeds)

    print(f"{len(plan)} runs: {[p.name for p in config_paths]} × seeds {seeds}")
    if args.dry_run:
        for item in plan:
            run = load_run_config(item["config_path"], seed=item["seed"])
            state = "done" if already_done(item["run_id"], run.hash(), RESULTS_PATH) else "todo"
            print(f"  {item['run_id']:<32} {run.hash()}  {state}")
        return 0

    records: list[dict] = []
    for item in plan:
        run = load_run_config(item["config_path"], seed=item["seed"])
        prior = already_done(item["run_id"], run.hash(), RESULTS_PATH)
        if prior:
            print(f"skip {item['run_id']} — already logged ({run.hash()})")
            records.append(
                {
                    "config": item["config_path"].name,
                    "run_id": item["run_id"],
                    "seed": item["seed"],
                    "split": run.split_id(),
                    "encoder": run.encoder,
                    "macro_f1": float(prior["macro_f1"]),
                    "accuracy": float(prior["accuracy"]),
                }
            )
            continue

        print(f"\n=== {item['run_id']} ({run.hash()}) ===")
        work_dir = CHECKPOINT_ROOT / item["run_id"]
        if stale_checkpoint(work_dir, run.hash()):
            print(f"  clearing stale checkpoint dir (config changed): {work_dir}")
            shutil.rmtree(work_dir)
        try:
            metrics = run_training(run, device=args.device, work_dir=work_dir, progress=print)
        except Exception as exc:  # report and continue (§11)
            print(f"FAILED {item['run_id']}: {exc}", file=sys.stderr)
            continue

        log_run(run, item["run_id"], metrics)
        records.append(
            {
                "config": item["config_path"].name,
                "run_id": item["run_id"],
                "seed": item["seed"],
                "split": run.split_id(),
                "encoder": run.encoder,
                "macro_f1": metrics["macro_f1"],
                "accuracy": metrics["accuracy"],
            }
        )
        print(
            f"{item['run_id']}: macro-F1 {metrics['macro_f1']:.4f} "
            f"acc {metrics['accuracy']:.4f} (best dev {metrics['best_dev_macro_f1']:.4f} "
            f"@ epoch {metrics['best_epoch']})"
        )

    if not records:
        print("no results", file=sys.stderr)
        return 1

    # Build the report from experiments.csv for *all* shipped configs, so a
    # partial re-run does not drop the configs it did not touch.
    all_configs = sorted(TRAIN_CONFIG_DIR.glob("*.yaml"))
    report_recs = report_records(all_configs, seeds, RESULTS_PATH)
    summary = summarise(report_recs)
    write_report(summary, report_recs, REPORT_DIR)
    print(f"\nwrote {REPORT_DIR / 'baselines.md'}\n")
    for s in summary:
        print(
            f"  {s['config']:<28} macro-F1 {s['macro_f1_mean']:.4f} ± "
            f"{s['macro_f1_std']:.4f}  (n={s['n']})"
        )

    incomplete = [s for s in summary if s["n"] < len(seeds)]
    if incomplete:
        print(
            f"\n{len(incomplete)} config(s) have fewer than {len(seeds)} seeds — "
            "re-run to finish",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
