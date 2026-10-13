"""T-707 — the Phase 7 findings report (CLAUDE7.md).

    python -m scripts.t707_findings            # write reports/findings.md
    python -m scripts.t707_findings --check    # exit non-zero if it is stale

Generated from T-701–T-706's artefacts the way T-114's datasheet is: nothing
is typed in, so the report cannot drift from the numbers, and `--check`
catches it when an artefact is regenerated and the report is not. Every
difference it states cites its row in `reports/phase7/robustness.md`.

What the report may claim is fixed by CLAUDE7.md: shares are of failures, a
difference is a finding only under hard rule 5, `unattributed` comes first,
and the two weakly validated labels never appear without their caveat.
"""

from __future__ import annotations

import argparse
import sys

import pandas as pd

from src import findings as F
from src.download_dataset.paths import REPO_ROOT
from src.env_check import require_python

PHASE7 = REPO_ROOT / "reports" / "phase7"
REPORT = REPO_ROOT / "reports" / "findings.md"
LEVEL_NAME = {"L1": "level 1", "L2": "level 2"}

QUESTIONS = {
    "T-702": (
        "Target language and direction",
        "Crossing *into* Dravidian costs more than crossing back, on all three tasks (Phase 1 T-111, Phase 4). "
        "Does the cause mix differ between Indo-Aryan→Dravidian and Dravidian→Indo-Aryan in the same "
        "language pairs? Telugu is the only Dravidian source, so the pairs are hin↔tel and ben↔tel.",
    ),
    "T-703": (
        "Source language",
        "Hindi is the worst source on both encoders (T-408). Do Hindi-source failures have a different "
        "cause mix, or the same mix at a higher volume? Each pair of sources is compared on the targets "
        "both reach.",
    ),
    "T-704": (
        "Encoder",
        "mBERT beats IndicBERT-v2 on every measure (Phase 5). Which labels account for the difference? "
        "The replicate here is the task: a difference must hold on both.",
    ),
    "T-705": (
        "Native vs machine-translated target text",
        "How much of the transfer loss does our own translation account for? Each of Bengali, Hindi and "
        "Telugu is evaluated on native and on MT text from the same source model.",
    ),
}


def load() -> dict:
    return {
        "shares": pd.read_parquet(PHASE7 / "label_shares.parquet"),
        "robust": pd.read_parquet(PHASE7 / "robustness.parquet"),
        "source": pd.read_parquet(PHASE7 / "by_source_groups.parquet"),
        "encoder": pd.read_parquet(PHASE7 / "by_encoder_groups.parquet"),
        "prov": pd.read_parquet(PHASE7 / "provenance_groups.parquet"),
    }


def label_note(measure: str) -> str:
    return f" *Limitation: {F.LABEL_CAVEATS[measure]}.*" if measure in F.LABEL_CAVEATS else ""


def bullet(r: pd.Series) -> str:
    text = f"- **{r['measure']}** ({LEVEL_NAME[r['level']]}), {r['comparison']}: {r['shares']}. `{r['finding_id']}`"
    text += label_note(r["measure"])
    if r["caveats"]:
        text += " *Caveat: " + "; ".join(F.CAVEATS[k] for k in r["caveats"].split(";")) + ".*"
    return text


# --------------------------------------------------------------------------
# Sections
# --------------------------------------------------------------------------


def pooled(shares: pd.DataFrame) -> pd.DataFrame:
    """Per (task, encoder): every cell's shares weighted by its failures, which
    is exactly the share over the pooled failures."""
    s = shares[shares["seed_set"] == "all"].copy()
    s["count"] = s["share"] * s["n_failures"]
    g = s.groupby(["task", "encoder", "level", "measure"]).agg(count=("count", "sum"))
    totals = s[s["level"] == "L1"].groupby(["task", "encoder", "condition_id"])["n_failures"].first()
    totals = totals.groupby(["task", "encoder"]).sum().rename("n")
    g = g.reset_index().merge(totals.reset_index(), on=["task", "encoder"])
    g["share"] = g["count"] / g["n"]
    return g


def headline(data: dict) -> list[str]:
    shares = data["shares"]
    p = pooled(shares)
    cells = shares[(shares["seed_set"] == "all") & (shares["level"] == "L1") & (shares["measure"] == "unattributed")]
    l1 = p[p["level"] == "L1"].pivot_table(index=["task", "encoder"], columns="measure", values="share")
    n = p.groupby(["task", "encoder"])["n"].first()
    lines = [
        "## 1. What the failures are made of",
        "",
        "Pooled over all 18 conditions of each (task, encoder). `unattributed` first: it is how much of the "
        "failure set the pipeline could not explain.",
        "",
        "| task | encoder | failures | **unattributed** | linguistic | data | model | unattributed, range over its 18 cells |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for (task, encoder), row in l1.iterrows():
        c = cells[(cells["task"] == task) & (cells["encoder"] == encoder)]["share"]
        lines.append(
            f"| {task} | {encoder} | {int(n[(task, encoder)]):,} | **{F.pct(row['unattributed'])}** | "
            f"{F.pct(row['linguistic'])} | {F.pct(row['data'])} | {F.pct(row['model'])} | "
            f"{F.pct(c.min())}–{F.pct(c.max())} |"
        )
    labels = [label for label in F.LABELS if label != "unattributed"]
    l2 = p[p["level"] == "L2"].pivot_table(index=["task", "encoder"], columns="measure", values="share")
    lines += ["", "Level 2:", "", "| task | encoder | " + " | ".join(labels) + " |", "|---|---|" + "---|" * len(labels)]
    for (task, encoder), row in l2.iterrows():
        lines.append(f"| {task} | {encoder} | " + " | ".join(F.pct(row[label]) for label in labels) + " |")
    lines += ["", *F.caveat_lines()]

    data_model = l1["data"] + l1["model"]
    lines += [
        "",
        f"**Reading.** Across the four (task, encoder) cells, {F.pct(l1['unattributed'].min())}–"
        f"{F.pct(l1['unattributed'].max())} of failures are unattributed. The data and model labels together "
        f"cover {F.pct(data_model.min())}–{F.pct(data_model.max())}, and the linguistic labels "
        f"{F.pct(l1['linguistic'].min())}–{F.pct(l1['linguistic'].max())}. "
        + ("In every cell, then, most failures are described by *where* they happen (the same item fails in "
           "every target language, in only one text provenance, on only some seeds, or under a collapsed "
           "classifier) rather than by a property of the target sentence. "
           if data_model.min() > 0.5 else
           "The data and model labels are not a majority in every cell. ")
        + "The data and model labels describe a pattern; they do not name a mechanism.",
        "",
        "Per-cell shares: `reports/phase7/label_shares.md`.",
        "",
    ]
    return lines


def without_hindi(robust: pd.DataFrame, r: pd.Series) -> str | None:
    """For a verdict the Hindi blind spot could produce, the same measure's
    verdict in the same question and task with no Hindi target, if one exists."""
    task = r["comparison"].split(":")[0]
    sib = robust[
        (robust["question"] == r["question"]) & (robust["measure"] == r["measure"]) & (robust["level"] == r["level"])
        & robust["comparison"].str.startswith(task + ":")
        & ~robust["a_targets"].str.contains("hin") & ~robust["b_targets"].str.contains("hin")
    ]
    if sib.empty:
        return None
    parts = [f"{s['comparison']}: {s['verdict']} ({s['shares']}) `{s['finding_id']}`" for _, s in sib.iterrows()]
    return "  - Same measure with no Hindi target — " + "; ".join(parts) + "."


def question_section(number: int, question: str, data: dict) -> list[str]:
    title, text = QUESTIONS[question]
    robust = data["robust"]
    q = robust[robust["question"] == question]
    held = q[q["verdict"] == "held"]
    clean, caveated = held[~held["artefact_possible"]], held[held["artefact_possible"]]
    disagree = q[q["verdict"].str.endswith("disagree")]
    lines = [f"## {number}. {title} ({question})", "", text, ""]
    lines += extra_tables(question, data)

    lines += ["**Held, with no measurement artefact that could produce it:**", ""]
    lines += [bullet(r) for _, r in clean.iterrows()] or ["- None."]
    lines += ["", "**Held, but a known measurement artefact could produce it on its own:**", ""]
    if caveated.empty:
        lines.append("- None.")
    for _, r in caveated.iterrows():
        lines.append(bullet(r))
        if "hindi_morphology" in r["caveats"]:
            alt = without_hindi(robust, r)
            if alt:
                lines.append(alt)
    lines += ["", "**Separated on both replicates but in opposite directions** (not a finding):", ""]
    lines += [bullet(r) for _, r in disagree.iterrows()] or ["- None."]
    nothing = [c for c, g in q.groupby("comparison", sort=False) if not (g["verdict"] == "held").any()]
    lines += ["", "**Comparisons where no measure held** (the cause mix does not differ under the rule):", ""]
    lines += [f"- {c}" for c in nothing] or ["- None."]
    counts = q["verdict"].value_counts()
    lines += [
        "",
        f"Of {len(q)} (comparison, measure) rows: {int(counts.get('held', 0))} held, "
        f"{int(sum(v for k, v in counts.items() if k.startswith('one ')))} separated in one replicate only, "
        f"{len(disagree)} pointed opposite ways, {int(counts.get('within noise', 0))} within noise. "
        f"Full table: `reports/phase7/robustness.md`.",
        "",
    ]
    return lines


def extra_tables(question: str, data: dict) -> list[str]:
    if question == "T-703":
        g = data["source"]
        g = g[(g["level"] == "L1") & (g["measure"] == "unattributed")]
        lines = ["Failure volume by source (failed rows over evaluated rows, native-text conditions, each "
                 "source's own three targets):", "", "| task | encoder | hin | ben | tel |", "|---|---|---|---|---|"]
        for (task, encoder), s in g.groupby(["task", "encoder"]):
            s = s.set_index("source")["failure_rate"]
            lines.append(f"| {task} | {encoder} | " + " | ".join(F.pct(s.get(x)) for x in ("hin", "ben", "tel")) + " |")
        return lines + ["", "Task-3 IndicBERT-v2 rates include source runs that never trained (section 6).", ""]
    if question == "T-704":
        g = data["encoder"]
        g = g[(g["level"] == "L1") & (g["measure"] == "unattributed")]
        lines = ["Failure volume by encoder (native-text conditions):", "", "| task | encoder | failures | fail rate |",
                 "|---|---|---|---|"]
        for _, r in g.sort_values(["task", "encoder"]).iterrows():
            lines.append(f"| {r['task']} | {r['encoder']} | {int(r['n_failures']):,} | {F.pct(r['failure_rate'])} |")
        return lines + [""]
    if question == "T-705":
        g = data["prov"]
        g = g[(g["level"] == "L2") & g["measure"].isin(["translation_drift", "provenance_specific"])]
        lines = [
            "Share of failures labelled as the translation's doing — `translation_drift` plus "
            "`provenance_specific` — on MT target text:",
            "",
            "| task | encoder | ben | hin | tel | mal |",
            "|---|---|---|---|---|---|",
        ]
        mt = g[g["kind"] == "transfer_mt"].groupby(["task", "encoder", "target"])["share"].sum()
        for (task, encoder), s in mt.groupby(level=[0, 1]):
            s = s.droplevel([0, 1])
            lines.append(f"| {task} | {encoder} | " + " | ".join(F.pct(s.get(x)) for x in ("ben", "hin", "tel", "mal")) + " |")
        lines += [
            "",
            "`provenance_specific` means the same item, from the same source model, was right on the other "
            "text; it is the closest this pipeline comes to \"our translation caused this failure\". It is a "
            "label on the failure, not a test of the translation.",
            "",
        ]
        return lines
    return []


def untrained_section(robust: pd.DataFrame) -> list[str]:
    changed = robust[robust["verdict_changed_without_untrained"]]
    rerun = int(robust["verdict_converged"].notna().sum())
    lines = ["## 6. Runs that never trained", "",
             f"Some task-3 IndicBERT-v2 source runs never fit their training data "
             f"(`reports/baselines.md`). They stay in every number above (CLAUDE7.md hard rule 6). "
             f"{rerun} comparison rows were re-run without them."]
    held_changed = changed[(changed["verdict"] == "held") | (changed["verdict_converged"] == "held")]
    lines[-1] += (
        f" {len(changed)} verdicts change, {len(held_changed)} of them to or from `held`: "
        + (", ".join(f"`{r['finding_id']}` ({r['verdict']} → {r['verdict_converged']})" for _, r in held_changed.iterrows())
           if len(held_changed) else "none")
        + ". The rest move between `one encoder only` and `within noise`."
    )
    return lines + ["", "Listed in full in `reports/phase7/robustness.md`.", ""]


def limits(data: dict) -> list[str]:
    p = pooled(data["shares"])
    frag = p[(p["level"] == "L2") & (p["measure"] == "tokenizer_fragmentation")]
    frag_text = ", ".join(f"task {r['task']} {r['encoder']} {F.pct(r['share'])}" for _, r in frag.iterrows())
    return [
        "## 7. Known measurement limits",
        "",
        "Properties of the Phase 6 measurement, not findings about the languages (CLAUDE7.md).",
        "",
        "- **Shares are of failures.** They say how a cell's failures are made up. Labels were not compared "
        "against correct predictions (decided 2026-10-13), so a share does not show that a label makes failure "
        "more likely.",
        f"- `morphological_masking`: {F.LABEL_CAVEATS['morphological_masking']}.",
        f"- `terminology_gap`: {F.LABEL_CAVEATS['terminology_gap']}.",
        "- **Hindi targets never get `morphological_masking`**: Hindi writes postpositions as separate words. "
        "Any comparison with Hindi on one side and not the other is flagged above.",
        f"- **`tokenizer_fragmentation` barely fires** ({frag_text}). R_frag ≥ 2.0 is rarely reached.",
        "- **The orthographic/numeral module is excluded** (CLAUDE4.md, Scope decisions); numeral failures fall "
        "through to other labels or to `unattributed`.",
        "- **Task 3 IndicBERT-v2 has 8.2% unconverged saliency rows**, above the 5% limit, so its "
        "`terminology_gap` share is understated (CLAUDE4.md T-605).",
        "- **`translation_drift` on native text** is a human translation scoring below τ. τ passes 95% of "
        "human pairs by construction (T-110), and LaBSE scores literal MT above human translation, so native "
        "text shows *more* drift than MT on task 3. That is the measure, not translation damage.",
        "- **Native and MT Bengali differ in digit system** as well as provenance.",
        "- **No native Malayalam**: both Malayalam conditions evaluate on MT, so it has no provenance contrast.",
        "- **Task 3 is at its data ceiling** (Phase 4); few of its differences survive the rule.",
        "- **The hand spot-check covers task 2 IndicBERT-v2 only** (98.5% agreement, T-608). The other three "
        "(task, encoder) cells' labels are unverified by hand.",
        "- **The data and model labels are descriptive.** `cross_lingual_hard` (fails in all three targets), "
        "`seed_unstable` (another seed got it right), `class_collapse` and `provenance_specific` say where a "
        "failure happens, not why.",
        "",
    ]


def build(data: dict) -> str:
    config_hash = data["robust"]["config_hash"].iloc[0]
    lines = [
        "# Phase 7 findings — what the cause labels say about transfer failure",
        "",
        f"Config hash `{config_hash}`. Generated by `python -m scripts.t707_findings` from the artefacts in "
        "`reports/phase7/`; `--check` fails if this file no longer matches them. Each stated difference cites "
        "its row in `reports/phase7/robustness.md`.",
        "",
        "**How to read it.** Phase 6 gave every failed (item, seed) prediction one cause label. This report "
        "counts them. A share is the fraction of a group's *failures* carrying a label. A difference between "
        "two groups counts only if their bootstrap 95% CIs do not overlap and the difference points the same "
        "way on both encoders (both tasks, for the encoder comparison) — CLAUDE7.md hard rule 5, fixed before "
        "any table was read. Shares are written `encoder: A vs B`, and `*` marks a replicate where the "
        "intervals separate.",
        "",
        *headline(data),
    ]
    for number, question in enumerate(QUESTIONS, start=2):
        lines += question_section(number, question, data)
    lines += untrained_section(data["robust"])
    lines += limits(data)
    lines += [
        "## 8. Where the numbers come from",
        "",
        "| artefact | built by |",
        "|---|---|",
        "| `reports/phase7/label_shares.{parquet,md}` | `python -m scripts.t701_shares` |",
        "| `reports/phase7/by_target.{parquet,md}` | `python -m scripts.t702_by_target` |",
        "| `reports/phase7/by_source.{parquet,md}` | `python -m scripts.t703_by_source` |",
        "| `reports/phase7/by_encoder.{parquet,md}` | `python -m scripts.t704_by_encoder` |",
        "| `reports/phase7/provenance.{parquet,md}` | `python -m scripts.t705_provenance` |",
        "| `reports/phase7/robustness.{parquet,md}` | `python -m scripts.t706_robustness` |",
        "| `reports/findings.md` | `python -m scripts.t707_findings` |",
        "",
    ]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    require_python()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="fail if reports/findings.md is stale")
    args = parser.parse_args(argv)

    data = load()
    hashes = {data["shares"]["config_hash"].iloc[0], data["robust"]["config_hash"].iloc[0]}
    if len(hashes) != 1:
        print(f"T-701 and T-706 were built under different configs: {sorted(hashes)}", file=sys.stderr)
        return 1
    text = build(data)
    if args.check:
        if not REPORT.exists() or REPORT.read_text() != text:
            print(f"{REPORT} is stale; regenerate it", file=sys.stderr)
            return 1
        print("findings report is current.")
        return 0
    REPORT.write_text(text)
    print(f"wrote {REPORT.relative_to(REPO_ROOT)} ({len(text.splitlines())} lines)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
