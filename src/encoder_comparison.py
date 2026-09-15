"""Cross-encoder comparison (CLAUDE5.md T-504).

Puts every encoder's results side by side on identical conditions: the
in-language baselines, all nine transfer cells, the same-source MT family, and
the translationese family. Computes nothing new about a single encoder — it
reuses T-304's bootstrap and T-306's paired join through
:mod:`src.results_tables` — and adds exactly one thing those cannot produce
alone: the **difference between two encoders on the same items**.

Three things this module is careful about, because each would otherwise
produce a comparison that looks fine and is not:

* **Encoders must be read from their own namespace.** A prediction log is
  filed under its encoder (`predictions.log_path`), and pairing one encoder's
  source arm with another's target arm would report the difference between two
  models as a transfer gap. Every read here goes through
  ``results_tables``'s ``encoder`` argument.
* **A deferred encoder stays in the table.** CLAUDE5.md asks for all three,
  and XLM-R was excluded for a measured reason (T-500: its weights, gradients
  and AdamW state need 4.14 GiB before any activation, on a 3.68 GiB card).
  Dropping the row would make the exclusion invisible; it is rendered with its
  reason instead, the same way `results_tables` keeps blocked cells.
* **A difference smaller than seed noise is not a finding.** `hard rule 4`
  requires mean ± std over three seeds, and a delta between two such means
  deserves the same treatment. :func:`exceeds_seed_noise` compares the delta
  against the two encoders' combined spread rather than leaving "is this
  real?" to the reader's eye.

Confidence intervals are per seed — T-304 bootstraps each (condition, seed)
independently — so what is reported is the **mean of the three seeds' 1000-sample
bootstrap intervals**, not a single interval over pooled seeds. That is stated
in the report rather than left for someone to assume the stronger thing.
"""

from __future__ import annotations

import math
import statistics

import pandas as pd

from .data import DEFAULT_ENCODER, ENCODERS
from .evaluate import load_matrix
from .results_tables import (
    _source_id_for,
    condition_seed_gap,
    fmt,
    is_missing,
    translationese_comparison,
)
from .transfer import DEFAULT_N_BOOT

# The encoders Phase 5 set out to compare, in the order the report shows them.
COMPARED = ("indicbert-v2", "mbert-base")

# Encoders that are part of the intended comparison but could not be run, with
# the reason. Kept in every table so the loss is visible (CLAUDE5.md working
# agreement: "report a lost comparison honestly").
EXCLUDED = {
    "xlm-r-base": (
        "VRAM-blocked (T-500): 279M parameters need 4.14 GiB of fp32 weights, "
        "gradients and AdamW state before any activation, against 3.68 GiB on "
        "this card, so it does not fit at batch size 1"
    ),
}

SEEDS = (0, 1, 2)
TASKS = (2, 3)


# --------------------------------------------------------------------------
# Per-condition summary, keeping the confidence intervals
# --------------------------------------------------------------------------


def summarise_condition(
    task: int,
    condition: dict,
    encoder: str,
    *,
    seeds: tuple[int, ...] = SEEDS,
    n_boot: int = DEFAULT_N_BOOT,
) -> dict:
    """One condition, one encoder: mean ± std over seeds, plus the averaged
    bootstrap interval.

    ``results_tables.condition_summary`` drops the per-seed intervals, and
    T-504 asks for "gaps with confidence intervals", so this keeps them. The
    aggregate is the mean of each seed's interval bounds — an honest summary
    of three intervals, and explicitly *not* an interval over pooled seeds,
    which would be a narrower and unearned claim.
    """
    per_seed = [
        condition_seed_gap(task, condition, seed, n_boot=n_boot, encoder=encoder)
        for seed in seeds
    ]
    available = [r for r in per_seed if r is not None]
    base = {
        "task": task,
        "condition": condition["name"],
        "kind": condition["kind"],
        "quadrant": condition.get("quadrant"),
        "encoder": encoder,
        "n_seeds": len(available),
    }
    if not available:
        return {**base, "status": "blocked"}

    gaps = [r["gap"] for r in available]
    return {
        **base,
        "status": "ok",
        "n_items": available[0]["n_items"],
        "source_mean": statistics.fmean(r["source"] for r in available),
        "target_mean": statistics.fmean(r["target"] for r in available),
        "target_std": _std([r["target"] for r in available]),
        "gap_mean": statistics.fmean(gaps),
        "gap_std": _std(gaps),
        "gap_ci_low": statistics.fmean(r["gap_ci_low"] for r in available),
        "gap_ci_high": statistics.fmean(r["gap_ci_high"] for r in available),
    }


def _std(values: list[float]) -> float:
    return statistics.stdev(values) if len(values) > 1 else 0.0


def condition_rows(
    task: int,
    *,
    encoders: tuple[str, ...] = COMPARED,
    seeds: tuple[int, ...] = SEEDS,
    n_boot: int = DEFAULT_N_BOOT,
    kinds: tuple[str, ...] = ("transfer", "transfer_mt"),
    progress=None,
) -> pd.DataFrame:
    """Every (condition, encoder) pair for one task, blocked ones included."""
    say = progress or (lambda _msg: None)
    matrix = load_matrix(task)
    rows = []
    for condition in matrix["conditions"]:
        if condition["kind"] not in kinds:
            continue
        for encoder in encoders:
            row = summarise_condition(
                task, condition, encoder, seeds=seeds, n_boot=n_boot
            )
            rows.append(row)
            say(
                f"  task {task} {condition['name']:<24} {encoder:<14} "
                f"{row['status']}"
            )
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# The comparison itself
# --------------------------------------------------------------------------


def exceeds_seed_noise(delta: float, std_a: float, std_b: float) -> bool | None:
    """Is a difference between two encoders bigger than their own spread?

    Compared against the two encoders' combined standard deviation,
    ``sqrt(std_a² + std_b²)`` — the spread of the difference of two
    independent quantities. Deliberately crude and symmetric: with three seeds
    there is no power for a real test, and inventing one would manufacture
    precision the data does not support (the same reasoning T-111 used to rank
    by mean-of-ranks rather than a weighted sum). ``None`` when either side is
    missing.
    """
    if is_missing(delta) or is_missing(std_a) or is_missing(std_b):
        return None
    return abs(delta) > math.hypot(std_a, std_b)


def compare(rows: pd.DataFrame, *, baseline: str = DEFAULT_ENCODER) -> pd.DataFrame:
    """Pivot per-condition rows into one row per condition, with the delta.

    ``baseline`` is the encoder every other is measured against — IndicBERT-v2,
    the reference Phase 2 and Phase 3 were built on. A positive
    ``target_delta`` means the other encoder scored *higher* on the target
    language; a negative ``gap_delta`` means it lost *less* going across.
    """
    if rows.empty:
        return pd.DataFrame()
    others = [e for e in rows["encoder"].unique() if e != baseline]
    out = []
    for (task, condition), group in rows.groupby(["task", "condition"], sort=False):
        by_encoder = {r["encoder"]: r for _, r in group.iterrows()}
        base = by_encoder.get(baseline)
        record = {
            "task": task,
            "condition": condition,
            "kind": group["kind"].iloc[0],
            "quadrant": group["quadrant"].iloc[0],
        }
        for encoder, row in by_encoder.items():
            record[f"{encoder}__status"] = row["status"]
            for field in ("target_mean", "target_std", "gap_mean", "gap_std",
                          "gap_ci_low", "gap_ci_high"):
                record[f"{encoder}__{field}"] = row.get(field)
        for encoder in others:
            other = by_encoder.get(encoder)
            if base is None or other is None or "ok" not in (base["status"], other["status"]):
                record[f"{encoder}__target_delta"] = None
                record[f"{encoder}__gap_delta"] = None
                record[f"{encoder}__beats_noise"] = None
                continue
            if base["status"] != "ok" or other["status"] != "ok":
                record[f"{encoder}__target_delta"] = None
                record[f"{encoder}__gap_delta"] = None
                record[f"{encoder}__beats_noise"] = None
                continue
            target_delta = other["target_mean"] - base["target_mean"]
            record[f"{encoder}__target_delta"] = target_delta
            record[f"{encoder}__gap_delta"] = other["gap_mean"] - base["gap_mean"]
            record[f"{encoder}__beats_noise"] = exceeds_seed_noise(
                target_delta, base["target_std"], other["target_std"]
            )
        out.append(record)
    return pd.DataFrame(out)


# --------------------------------------------------------------------------
# In-language baselines, read back from T-206's artefact
# --------------------------------------------------------------------------


def baseline_rows(path=None) -> pd.DataFrame:
    """Per (task, encoder) in-language baseline, from `reports/baselines.parquet`.

    Read back rather than recomputed, for the reason T-114 gives: a number
    typed into a second place drifts from the first the moment anything is
    regenerated, and it drifts silently.
    """
    from .baseline_gate import BASELINES_PARQUET

    frame = pd.read_parquet(path or BASELINES_PARQUET)
    rows = []
    for (config, encoder), group in frame.groupby(["config", "encoder"], sort=True):
        f1s = [float(x) for x in group["macro_f1"]]
        task = int(str(config)[4]) if str(config).startswith("task") else None
        rows.append(
            {
                "task": task,
                "config": str(config),
                "encoder": str(encoder),
                "n_seeds": len(f1s),
                "macro_f1_mean": statistics.fmean(f1s),
                "macro_f1_std": _std(f1s),
                "batch_size": int(group["batch_size"].iloc[0])
                if "batch_size" in group
                else None,
                "grad_accum": int(group["grad_accum"].iloc[0])
                if "grad_accum" in group
                else None,
            }
        )
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# Rendering
# --------------------------------------------------------------------------


def _yesno(value) -> str:
    if is_missing(value):
        return "—"
    return "yes" if value else "no"


def _signed(value, digits: int = 4) -> str:
    if is_missing(value):
        return "—"
    return f"{value:+.{digits}f}"


def render_baselines(frame: pd.DataFrame) -> list[str]:
    lines = [
        "## In-language baselines",
        "",
        "Hindi native split, held-out test fold, three seeds. The two encoders "
        "were scored on **identical rows**: `data.stratified_split` is seeded "
        "and encoder-blind, so for a given (task, seed) both saw the same test "
        "items. Read back from `reports/baselines.parquet`, not transcribed.",
        "",
        "| task | encoder | batch × accum | seeds | macro-F1 (mean ± std) |",
        "|---|---|---|---|---|",
    ]
    for _, row in frame.sort_values(["task", "encoder"]).iterrows():
        shape = (
            "—"
            if is_missing(row["batch_size"])
            else f"{int(row['batch_size'])} × {int(row['grad_accum'])}"
        )
        lines.append(
            f"| {row['task']} | {row['encoder']} | {shape} | {row['n_seeds']} | "
            f"{fmt(row['macro_f1_mean'])} ± {fmt(row['macro_f1_std'])} |"
        )
    for encoder, reason in EXCLUDED.items():
        lines.append(f"| — | {encoder} | — | 0 | **excluded** — {reason} |")
    lines.append("")
    return lines


def render_comparison(frame: pd.DataFrame, *, kind: str, task: int,
                      baseline: str = DEFAULT_ENCODER) -> list[str]:
    """One task, one condition family: both encoders and their difference."""
    sub = frame[(frame["task"] == task) & (frame["kind"] == kind)]
    others = [
        c.split("__")[0]
        for c in frame.columns
        if c.endswith("__target_delta")
    ]
    title = {
        "transfer": "Transfer — native / cross-block targets",
        "transfer_mt": "Transfer — same-source MT targets",
    }.get(kind, kind)
    lines = [
        f"### Task {task} — {title}",
        "",
        "`target` is macro-F1 on the target language; `gap` is "
        "source − target on the **same items** (T-306's paired join), so each "
        "encoder is measured against itself, not against the other. `Δ target` "
        "is the other encoder minus "
        f"`{baseline}`; positive means it scored higher. `beats noise?` asks "
        "whether |Δ| exceeds the two encoders' combined seed spread.",
        "",
        "| condition | quadrant | "
        + " | ".join(f"{e} target" for e in [baseline, *others])
        + " | "
        + " | ".join(f"{e} gap [CI]" for e in [baseline, *others])
        + " | Δ target | beats noise? |",
        "|" + "---|" * (4 + 2 * (1 + len(others))),
    ]
    for _, row in sub.iterrows():
        cells = [str(row["condition"]), str(row["quadrant"])]
        for enc in [baseline, *others]:
            if row.get(f"{enc}__status") != "ok":
                cells.append("blocked")
            else:
                cells.append(
                    f"{fmt(row[f'{enc}__target_mean'])} ± "
                    f"{fmt(row[f'{enc}__target_std'])}"
                )
        for enc in [baseline, *others]:
            if row.get(f"{enc}__status") != "ok":
                cells.append("—")
            else:
                cells.append(
                    f"{fmt(row[f'{enc}__gap_mean'])} ± {fmt(row[f'{enc}__gap_std'])} "
                    f"[{fmt(row[f'{enc}__gap_ci_low'], 3)}, "
                    f"{fmt(row[f'{enc}__gap_ci_high'], 3)}]"
                )
        for enc in others:
            cells.append(_signed(row.get(f"{enc}__target_delta")))
            cells.append(_yesno(row.get(f"{enc}__beats_noise")))
        lines.append("| " + " | ".join(cells) + " |")
    lines.append("")
    return lines


def render_translationese(task: int, encoders: tuple[str, ...], lang: str) -> list[str]:
    """Same language, three provenances, per encoder — the family that
    isolates the translation penalty from the language penalty."""
    lines = [
        f"### Task {task} — translationese, {lang}",
        "",
        "Same language and labels in all three arms; only provenance differs. "
        "`Δ vs native` is each MT arm minus the native arm for that encoder.",
        "",
        "| encoder | arm | origin | src_lang | macro-F1 (mean ± std) | Δ vs native |",
        "|---|---|---|---|---|---|",
    ]
    any_rows = False
    for encoder in encoders:
        frame = translationese_comparison(task, lang, encoder=encoder)
        if frame.empty:
            lines.append(f"| {encoder} | — | — | — | blocked | — |")
            continue
        any_rows = True
        for _, row in frame.iterrows():
            lines.append(
                f"| {encoder} | {row['block']} | {row['origin']} | "
                f"{row['src_lang'] if not is_missing(row['src_lang']) else '—'} | "
                f"{fmt(row['macro_f1_mean'])} ± {fmt(row['macro_f1_std'])} | "
                f"{_signed(row.get('delta_vs_native'))} |"
            )
    if not any_rows:
        lines.append("")
        return lines
    lines.append("")
    return lines
