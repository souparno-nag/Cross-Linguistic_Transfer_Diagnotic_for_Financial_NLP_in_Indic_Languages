"""T-205 — measure the VRAM budget on this card.

    python -m scripts.t205_vram                       # probe grid, all encoders
    python -m scripts.t205_vram --full-epoch          # + one real epoch per shipped config
    python -m scripts.t205_vram --encoders indicbert-v2 --max-lens 64,128
    python -m scripts.t205_vram --skip-probe --full-epoch \
        --config configs/train/task2_hin_mbert.yaml   # one config, fresh process

Probes the largest batch that survives a real train step for each
(encoder, max_len), writes the table to ``reports/vram_budget.{md,parquet}``
(per-epoch rows to ``reports/vram_epochs.parquet``), and — with
``--full-epoch`` — trains one full epoch on each shipped config's native split
to confirm no OOM across an epoch (§8's done criterion).

**A narrowed run merges into the report rather than replacing it.** Measuring
one encoder used to overwrite the table with that encoder's rows alone,
silently deleting measurements nobody asked to redo.

**Fragmentation is itself a cause of OOM, so isolate before concluding.** A
config that OOMs at the end of a long ``--full-epoch`` sweep has been handed a
card that several other models have already loaded and freed; PyTorch returns
the blocks but not always in a usable shape, and the failing run's own error
names how much is "reserved but unallocated". ``--config`` with
``--skip-probe`` re-checks one config in a process that has done nothing else,
which is the only way to tell a genuine capacity limit from a fragmented
allocator. Getting that backwards means tuning a config that did not need it,
which CLAUDE5.md hard rule 2 forbids.

Needs the GPU. Exits non-zero if a shipped config OOMs on its full-epoch run.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import pandas as pd
import torch

from src.config import TRAIN_CONFIG_DIR, load_run_config, run_training
from src.download_dataset.paths import REPO_ROOT
from src.env_check import require_python
from src.vram import DEFAULT_ENCODERS, DEFAULT_MAX_LENS, budget_table, free_gpu

REPORT_DIR = REPO_ROOT / "reports"
_GiB = 1024**3


def _merge_rows(new: list[dict], existing: list[dict], keys: tuple[str, ...]) -> list[dict]:
    """New rows replace same-key old ones; untouched old rows survive.

    A run narrowed with ``--encoders`` or ``--config`` used to overwrite the
    whole report with just its own rows, so measuring one encoder silently
    deleted the other two — the same footgun T-106's per-direction report hit
    (CLAUDE.md T-106). Merging keeps a partial re-measurement partial.
    """
    merged = {tuple(r[k] for k in keys): r for r in existing}
    merged.update({tuple(r[k] for k in keys): r for r in new})
    return [merged[k] for k in sorted(merged, key=lambda k: tuple(str(x) for x in k))]


def _read_existing() -> tuple[list[dict], list[dict]]:
    path = REPORT_DIR / "vram_budget.parquet"
    epoch_path = REPORT_DIR / "vram_epochs.parquet"
    rows = pd.read_parquet(path).to_dict("records") if path.exists() else []
    epochs = pd.read_parquet(epoch_path).to_dict("records") if epoch_path.exists() else []
    for row in rows:
        for key in ("peak_gib", "floor_gib", "activation_gib"):
            if key in row and pd.isna(row[key]):
                row[key] = None
    return rows, epochs


def _write_report(rows: list[dict], epoch_rows: list[dict]) -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    old_rows, old_epochs = _read_existing()
    rows = _merge_rows(rows, old_rows, ("encoder", "max_len"))
    epoch_rows = _merge_rows(epoch_rows, old_epochs, ("config",))

    pd.DataFrame(rows).to_parquet(REPORT_DIR / "vram_budget.parquet", index=False)
    if epoch_rows:
        pd.DataFrame(epoch_rows).to_parquet(REPORT_DIR / "vram_epochs.parquet", index=False)

    lines = [
        "# VRAM budget — T-205",
        "",
        f"Device: {torch.cuda.get_device_name(0)}, "
        f"{torch.cuda.get_device_properties(0).total_memory / _GiB:.1f} GiB total.",
        "",
        "Largest batch that survives forward → backward → optimiser step, "
        "`attention_mask` all ones (worst case).",
        "",
        "**floor** is the fixed cost of fp32 weights + gradients + AdamW's two "
        "moments — 16 bytes per parameter, independent of batch size and "
        "sequence length. **activations** is what the measured peak leaves over "
        "it, and is the only part a batch sweep can move. A peak *below* the "
        "floor is impossible for a completed optimiser step and means the "
        "measurement is wrong (see `src/vram.py`, the T-500 note).",
        "",
        "| encoder | max_len | fp16 | max batch | peak GiB | floor GiB | activations GiB | note |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        peak = "—" if r["peak_gib"] is None else f"{r['peak_gib']:.2f}"
        floor = "—" if r.get("floor_gib") is None else f"{r['floor_gib']:.2f}"
        act = "—" if r.get("activation_gib") is None else f"{r['activation_gib']:.2f}"
        lines.append(
            f"| {r['encoder']} | {r['max_len']} | {r['fp16']} | "
            f"{r['max_batch']} | {peak} | {floor} | {act} | {r['note']} |"
        )

    if epoch_rows:
        lines += [
            "",
            "## Full-epoch check",
            "",
            "One epoch on the native training fold at each shipped config's settings.",
            "",
            "| config | batch | max_len | grad_accum | fp16 | peak GiB | seconds | result |",
            "|---|---|---|---|---|---|---|---|",
        ]
        for e in epoch_rows:
            lines.append(
                f"| {e['config']} | {e['batch_size']} | {e['max_len']} | "
                f"{e['grad_accum']} | {e['fp16']} | {e['peak_gib']:.2f} | "
                f"{e['seconds']:.0f} | {e['result']} |"
            )
    lines.append("")
    (REPORT_DIR / "vram_budget.md").write_text("\n".join(lines))


def _full_epoch(path: Path, device: str) -> dict:
    run = load_run_config(path, epochs=1)
    free_gpu()
    torch.cuda.reset_peak_memory_stats()
    start = time.time()
    row = {
        "config": path.name,
        "batch_size": run.batch_size,
        "max_len": run.max_len,
        "grad_accum": run.grad_accum,
        "fp16": run.fp16,
    }
    try:
        run_training(run, device=device)
        row["result"] = "ok"
    except torch.cuda.OutOfMemoryError as exc:  # type: ignore[attr-defined]
        row["result"] = f"OOM: {str(exc).splitlines()[0]}"
    row["seconds"] = time.time() - start
    row["peak_gib"] = int(torch.cuda.max_memory_allocated()) / _GiB
    free_gpu()
    return row


def main() -> int:
    require_python()
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--encoders", default=",".join(DEFAULT_ENCODERS))
    parser.add_argument("--max-lens", default=",".join(str(x) for x in DEFAULT_MAX_LENS))
    parser.add_argument("--no-fp16", action="store_true")
    parser.add_argument("--full-epoch", action="store_true")
    parser.add_argument(
        "--config",
        action="append",
        help="limit --full-epoch to this configs/train/*.yaml; repeatable. "
        "Use it to re-check one config in a process that has not already run "
        "every other model — allocator fragmentation from earlier runs is "
        "itself a cause of OOM.",
    )
    parser.add_argument(
        "--skip-probe",
        action="store_true",
        help="go straight to --full-epoch without re-probing the batch grid",
    )
    args = parser.parse_args()

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        print("T-205 needs a CUDA device; none available", file=sys.stderr)
        return 2

    encoders = tuple(x.strip() for x in args.encoders.split(",") if x.strip())
    max_lens = tuple(int(x) for x in args.max_lens.split(","))
    fp16 = not args.no_fp16

    rows: list[dict] = []
    if not args.skip_probe:
        print(f"probing {encoders} × {max_lens}, fp16={fp16}, device={args.device}")
        rows = budget_table(
            encoders=encoders,
            max_lens=max_lens,
            device=args.device,
            fp16=fp16,
            on_row=lambda r: print(
                f"  {r['encoder']:<14} len {r['max_len']:>3}: "
                f"max batch {r['max_batch']:>3}"
                + (f", peak {r['peak_gib']:.2f} GiB" if r["peak_gib"] else f" — {r['note']}")
            ),
        )

    epoch_rows: list[dict] = []
    if args.full_epoch:
        print("\nfull-epoch check:")
        if args.config:
            paths = sorted(Path(c) for c in args.config)
        else:
            paths = sorted(TRAIN_CONFIG_DIR.glob("*.yaml"))
        for path in paths:
            e = _full_epoch(path, args.device)
            epoch_rows.append(e)
            print(
                f"  {e['config']:<28} peak {e['peak_gib']:.2f} GiB  "
                f"{e['seconds']:.0f}s  {e['result']}"
            )

    _write_report(rows, epoch_rows)
    print(f"\nwrote {REPORT_DIR / 'vram_budget.md'}")

    failed = [e for e in epoch_rows if e["result"] != "ok"]
    if failed:
        print(f"FAIL: {len(failed)} config(s) OOM'd on a full epoch", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
