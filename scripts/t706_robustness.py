"""T-706 — robustness summary of every candidate finding (CLAUDE7.md).

    python -m scripts.t706_robustness

Reads T-702–T-705's comparison artefacts and writes one row per
(comparison, measure): its verdict on all seeds, its verdict without the runs
that never trained where that was run, the per-replicate shares, and any
measurement artefact that could produce the difference on its own
(`findings.CAVEATS`). Each row gets a stable id (`R702-01`, …) that T-707's
report cites, so every claim there points to a row here.

Writes `reports/phase7/robustness.{parquet,md}`. No bootstrap, no model:
seconds.
"""

from __future__ import annotations

import argparse
import sys

import pandas as pd

from src import findings as F
from src.download_dataset.paths import REPO_ROOT
from src.env_check import require_python

OUT_DIR = REPO_ROOT / "reports" / "phase7"
SOURCES = ("by_target", "by_source", "by_encoder", "provenance")
#: Caveats that could produce a difference by themselves. `task3_ceiling` is a
#: note about low signal, not a mechanism that manufactures a difference.
ARTEFACTS = ("hindi_morphology", "native_drift", "bengali_digits")
VERDICT_ORDER = ["held", "encoders disagree", "tasks disagree", "one encoder only", "one task only", "within noise"]


def _detail(details: pd.DataFrame) -> pd.DataFrame:
    """`replicate: A% vs B%` per (seed set, comparison, measure)."""
    keys = ["seed_set", "question", "comparison", "level", "measure"]
    text = details.assign(
        part=[
            f"{r}: {F.pct(a)} vs {F.pct(b)}{' *' if s else ''}"
            for r, a, b, s in zip(details["replicate"], details["a_share"], details["b_share"], details["separated"])
        ]
    )
    return text.groupby(keys, sort=False)["part"].agg("; ".join).rename("shares").reset_index()


def build() -> pd.DataFrame:
    frames = []
    for name in SOURCES:
        details = pd.read_parquet(OUT_DIR / f"{name}.parquet")
        frames.append(F.verdict_table(details).merge(_detail(details), on=["seed_set", "question", "comparison",
                                                                          "level", "measure"]))
    v = pd.concat(frames, ignore_index=True)

    keys = ["question", "comparison", "level", "measure"]
    all_ = v[v["seed_set"] == "all"].drop(columns="seed_set")
    conv = v[v["seed_set"] == "converged"][keys + ["verdict", "shares"]].rename(
        columns={"verdict": "verdict_converged", "shares": "shares_converged"}
    )
    out = all_.merge(conv, on=keys, how="left")
    out["caveats"] = [
        ";".join(F.caveats_for(q, c, m, at, bt))
        for q, c, m, at, bt in zip(out["question"], out["comparison"], out["measure"], out["a_targets"],
                                   out["b_targets"])
    ]
    out["artefact_possible"] = [any(k in c.split(";") for k in ARTEFACTS) for c in out["caveats"]]
    out["verdict_changed_without_untrained"] = out["verdict_converged"].notna() & (
        out["verdict_converged"] != out["verdict"]
    )
    out.insert(
        0, "finding_id",
        [f"R{q[2:]}-{i + 1:02d}" for q, g in out.groupby("question", sort=False) for i in range(len(g))],
    )
    return out


def render(table: pd.DataFrame, config_hash: str) -> str:
    lines = [
        "# Phase 7 — robustness of every candidate finding (T-706)",
        "",
        f"Config hash `{config_hash}`. One row per (comparison, measure) from T-702–T-705. A difference is "
        "`held` only if the 95% CIs separate in every replicate and point the same way (CLAUDE7.md hard "
        "rule 5). In the share columns `*` marks a replicate whose intervals separate.",
        "",
        "## Verdicts per question",
        "",
        "| question | " + " | ".join(VERDICT_ORDER) + " |",
        "|---|" + "---|" * len(VERDICT_ORDER),
    ]
    for question, g in table.groupby("question", sort=True):
        counts = g["verdict"].value_counts()
        lines.append(f"| {question} | " + " | ".join(str(int(counts.get(v, 0))) for v in VERDICT_ORDER) + " |")
    lines.append("")

    def rows_table(sub: pd.DataFrame) -> list[str]:
        out = ["| id | comparison | level | measure | A − B (mean) | shares (A vs B) | caveats |",
               "|---|---|---|---|---|---|---|"]
        for _, r in sub.iterrows():
            cav = ", ".join(r["caveats"].split(";")) if r["caveats"] else ""
            out.append(f"| {r['finding_id']} | {r['comparison']} | {r['level']} | {r['measure']} | "
                       f"{100 * r['mean_diff']:+.1f} pts | {r['shares']} | {cav} |")
        return out + [""]

    held = table[table["verdict"] == "held"]
    lines += [
        "## Held",
        "",
        "Split by whether a known measurement artefact could produce the difference by itself "
        "(" + ", ".join(f"`{k}`" for k in ARTEFACTS) + ").",
        "",
        "### Held, no artefact caveat",
        "",
        *rows_table(held[~held["artefact_possible"]]),
        "### Held, but an artefact could produce it",
        "",
        *rows_table(held[held["artefact_possible"]]),
        "## Replicates point opposite ways",
        "",
        *rows_table(table[table["verdict"].str.endswith("disagree")]),
        "## Separated in one replicate only",
        "",
        *rows_table(table[table["verdict"].str.startswith("one ")]),
        f"## Within noise\n\n{int((table['verdict'] == 'within noise').sum())} rows; see the parquet.\n",
        "## Without the runs that never trained (hard rule 6)",
        "",
    ]
    changed = table[table["verdict_changed_without_untrained"]]
    n_conv = int(table["verdict_converged"].notna().sum())
    if changed.empty:
        lines.append(f"{n_conv} rows were re-run without them; no verdict changes.")
    else:
        lines += [f"{n_conv} rows were re-run without them; these verdicts change:", "",
                  "| id | comparison | measure | all seeds | converged only | shares, converged only |",
                  "|---|---|---|---|---|---|"]
        for _, r in changed.iterrows():
            lines.append(f"| {r['finding_id']} | {r['comparison']} | {r['measure']} | {r['verdict']} | "
                         f"{r['verdict_converged']} | {r['shares_converged']} |")
    lines += ["", "## Caveat key", ""] + [f"- `{k}`: {v}." for k, v in F.CAVEATS.items()]
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    require_python()
    argparse.ArgumentParser(description=__doc__).parse_args(argv)
    missing = [n for n in SOURCES if not (OUT_DIR / f"{n}.parquet").exists()]
    if missing:
        print(f"missing T-702–T-705 artefacts: {missing}", file=sys.stderr)
        return 1
    table = build()
    hashes = {pd.read_parquet(OUT_DIR / f"{n}.parquet", columns=["config_hash"])["config_hash"].iloc[0]
              for n in SOURCES}
    if len(hashes) != 1:
        print(f"T-702–T-705 were built under different configs: {sorted(hashes)}", file=sys.stderr)
        return 1
    config_hash = hashes.pop()
    table.assign(config_hash=config_hash).to_parquet(OUT_DIR / "robustness.parquet", index=False)
    (OUT_DIR / "robustness.md").write_text(render(table, config_hash))
    print(table["verdict"].value_counts().to_string())
    print(f"wrote {OUT_DIR / 'robustness.md'} ({len(table)} rows), config {config_hash}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
