"""T-207 — the baseline validation gate.

Compares each in-language baseline (T-206, `reports/baselines.parquet`) against
IndicFinNLP's published IndicBERT number (`configs/published_baselines.json`).

A config **passes** if our mean macro-F1 is within `tolerance_f1` of the
published value, or above it. A config that lands *below* tolerance passes only
if its entry carries a non-empty `diagnosis` — §8's "within ±2 F1, or a written
diagnosis of the gap". Nothing in Phase 3 should start while the gate fails.
"""

from __future__ import annotations

import json
import statistics
from pathlib import Path

import pandas as pd

from .download_dataset.paths import REPO_ROOT

PUBLISHED_PATH = REPO_ROOT / "configs" / "published_baselines.json"
BASELINES_PARQUET = REPO_ROOT / "reports" / "baselines.parquet"


def load_published(path: str | Path = PUBLISHED_PATH) -> dict:
    return json.loads(Path(path).read_text())


def our_results(path: str | Path = BASELINES_PARQUET) -> dict[str, dict]:
    """Per config: mean / sample-std macro-F1 over the logged seeds."""
    frame = pd.read_parquet(path)
    out: dict[str, dict] = {}
    for config, group in frame.groupby("config"):
        # baselines.parquet stores "task2_hin_indicbert.yaml"; published_baselines
        # is keyed by the stem. Match on the stem.
        key = str(config)[:-5] if str(config).endswith(".yaml") else str(config)
        f1s = [float(x) for x in group["macro_f1"]]
        out[key] = {
            "n": len(f1s),
            "seeds": sorted(int(s) for s in group["seed"]),
            "split": str(group["split"].iloc[0]),
            "macro_f1_mean": statistics.mean(f1s),
            "macro_f1_std": statistics.stdev(f1s) if len(f1s) > 1 else 0.0,
            "accuracy_mean": statistics.mean(float(x) for x in group["accuracy"]),
        }
    return out


def evaluate(published: dict, ours: dict[str, dict]) -> list[dict]:
    tol = float(published.get("tolerance_f1", 0.02))
    rows: list[dict] = []
    for config, spec in published["baselines"].items():
        pub = float(spec["published_macro_f1"])
        got = ours.get(config)
        if got is None:
            rows.append(
                {
                    "config": config,
                    "task": spec["task"],
                    "published_macro_f1": pub,
                    "our_macro_f1_mean": None,
                    "our_macro_f1_std": None,
                    "gap": None,
                    "status": "missing — not run",
                    "passed": False,
                    "diagnosis": spec.get("diagnosis", ""),
                    "note": spec.get("note", ""),
                }
            )
            continue

        mean = got["macro_f1_mean"]
        gap = mean - pub
        if gap > tol:
            status, passed = "above published", True
        elif gap >= -tol:
            status, passed = f"within ±{tol * 100:.0f} F1 pts", True
        else:
            diagnosed = bool(spec.get("diagnosis", "").strip())
            status = "below — diagnosed" if diagnosed else "below — NO diagnosis"
            passed = diagnosed

        rows.append(
            {
                "config": config,
                "task": spec["task"],
                "published_macro_f1": pub,
                "our_macro_f1_mean": round(mean, 4),
                "our_macro_f1_std": round(got["macro_f1_std"], 4),
                "n_seeds": got["n"],
                "gap": round(gap, 4),
                "status": status,
                "passed": passed,
                "diagnosis": spec.get("diagnosis", ""),
                "note": spec.get("note", ""),
            }
        )
    return rows


def gate_passes(rows: list[dict]) -> bool:
    return bool(rows) and all(r["passed"] for r in rows)
