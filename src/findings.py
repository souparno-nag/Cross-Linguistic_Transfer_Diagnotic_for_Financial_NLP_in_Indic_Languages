"""Phase 7 — aggregate Phase 6's cause labels into findings (CLAUDE7.md).

Everything here counts labels that already exist. Nothing is relabelled, no
threshold is touched, and no model is loaded (hard rule 1).

**The unit is a failed (item, seed) row**, exactly as Phase 6 wrote it. A
cell's share of a label is the fraction of its failed rows carrying that
label. Shares are reported three ways, because each answers a different
question:

* pooled over seeds, with a bootstrap 95% CI — the number comparisons use;
* per seed, as mean ± std — how much the share moves between trained models;
* level 1 (`label_group`) beside level 2 (`assigned_label`), level 1 always
  summed from level 2 so the two cannot disagree.

**The bootstrap resamples (condition, item) clusters, not rows.** An item that
failed on all three seeds contributes three rows that are not independent;
resampling rows would treat them as three draws and make every interval too
narrow. The cluster also carries the condition, because grouped tables pool
conditions that share item ids (tasks 2 and 3 hold the same items in every
language).

**The rule for calling a difference a finding is CLAUDE7.md hard rule 5, fixed
before any table was read:** two groups differ only if their 95% CIs do not
overlap, and the difference points the same way in every replicate (both
encoders; for the encoder comparison, both tasks). `verdict` implements it and
nothing else decides it.
"""

from __future__ import annotations

import statistics as st

import numpy as np
import pandas as pd

from .audit import config_hash
from .conditions import FAMILY
from .data import DEFAULT_ENCODER
from .diagnostics import LABEL_GROUP, PRECEDENCE, path_encoder, read_diagnostics
from .evaluate import _split_lang, load_matrix
from .failures import failure_path
from .predictions import log_path
from .transfer import DEFAULT_CI, DEFAULT_N_BOOT, DEFAULT_SEED

TASKS = (2, 3)
ENCODERS = (DEFAULT_ENCODER, "mbert-base")
SEEDS = (0, 1, 2)
KINDS = ("transfer", "transfer_mt")

#: Level 2 in precedence order, then the fall-through.
LABELS = [label for _, label in PRECEDENCE] + ["unattributed"]
#: Level 1, in the order CLAUDE4.md's table lists them.
GROUPS = ["linguistic", "data", "model", "unattributed"]

#: Every measure a table reports: `(level, name)`. `unattributed` exists at
#: both levels and is the same number at both.
MEASURES = [("L1", g) for g in GROUPS] + [("L2", label) for label in LABELS]
#: What comparisons test. Level-2 `unattributed` is left out because it is
#: level-1 `unattributed` again, and testing it twice would count one result
#: as two.
COMPARED_MEASURES = [m for m in MEASURES if m != ("L2", "unattributed")]

_GROUP_MATRIX = np.array(
    [[1.0 if LABEL_GROUP[label] == group else 0.0 for group in GROUPS] for label in LABELS]
)

#: Labels whose validation is a stated limitation (CLAUDE7.md hard rule 4).
#: Every report that shows either must carry this sentence next to it.
LABEL_CAVEATS = {
    "morphological_masking": (
        "a masked term does not raise the failure rate (masked fails more often "
        "than intact in 20 of 44 cells; reports/morphology_null.md)"
    ),
    "terminology_gap": (
        "salience divergence does not separate failures from correct pairs "
        "(AUC 0.48-0.52; reports/saliency_baseline.md)"
    ),
}


#: Measurement artefacts that can produce a difference on their own
#: (CLAUDE7.md, "Known measurement limits"). `caveats_for` attaches them to any
#: comparison they could explain, so a reader never meets such a verdict bare.
CAVEATS = {
    "hindi_morphology": (
        "one side includes Hindi targets, which cannot receive morphological_masking (Hindi writes "
        "postpositions as separate words); the difference may be that blind spot"
    ),
    "native_drift": (
        "translation_drift on native text means the *human* translation scored below tau; tau passes "
        "95% of human pairs by construction (T-110) and LaBSE scores literal MT above human translation, "
        "so native-vs-MT drift differences reflect the measure, not translation damage"
    ),
    "bengali_digits": "native and MT Bengali also differ in digit system (CLAUDE4.md, Scope decisions)",
    "task3_ceiling": "task 3 is at its data ceiling (Phase 4)",
}


def caveats_for(question: str, comparison: str, measure: str, a_targets: str, b_targets: str) -> list[str]:
    """Keys of `CAVEATS` that could explain this comparison's difference."""
    out = []
    targets = set(a_targets.split(",")) | set(b_targets.split(","))
    if measure in ("morphological_masking", "linguistic") and "hin" in targets:
        out.append("hindi_morphology")
    if question == "T-705" and measure in ("translation_drift", "linguistic"):
        out.append("native_drift")
    if question == "T-705" and "ben" in targets:
        out.append("bengali_digits")
    if comparison.startswith("Task 3"):
        out.append("task3_ceiling")
    return out


def run_config(n_boot: int = DEFAULT_N_BOOT, seed: int = DEFAULT_SEED, ci: float = DEFAULT_CI) -> dict:
    """Everything that can change a Phase 7 number, hashed into every artefact
    (CLAUDE.md §4 rule 8)."""
    return {
        "n_boot": n_boot,
        "bootstrap_seed": seed,
        "ci": ci,
        "tasks": list(TASKS),
        "encoders": list(ENCODERS),
        "kinds": list(KINDS),
        "bootstrap_unit": "(condition_id, item_id)",
        "rule": "95% CIs do not overlap and the direction agrees in every replicate",
    }


def run_config_hash(**kwargs) -> str:
    return config_hash(run_config(**kwargs))


# --------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------


def _conditions(task: int) -> list[dict]:
    return [c for c in load_matrix(task)["conditions"] if c["kind"] in KINDS]


def load_rows(task: int, encoder: str) -> pd.DataFrame:
    """Every failed (item, seed) row of one (task, encoder), with the
    condition's design attached. Raises if any condition's labels are missing:
    CLAUDE7.md's grid is 72 cells, and a table built over fewer would look
    complete while not being so."""
    frames = []
    for condition in _conditions(task):
        labels = read_diagnostics(task, condition["name"], encoder=path_encoder(encoder))
        source = _split_lang(condition["train"])[1]
        target = _split_lang(condition["eval"])[1]
        if set(labels["encoder_id"]) - {encoder}:
            raise ValueError(f"{condition['name']}: rows from {set(labels['encoder_id'])}, expected {encoder}")
        if set(labels["tgt_lang"]) - {target}:
            raise ValueError(f"{condition['name']}: target {set(labels['tgt_lang'])}, expected {target}")
        frames.append(
            labels.assign(
                task=task,
                encoder=encoder,
                kind=condition["kind"],
                source=source,
                target=target,
                quadrant=f"{FAMILY[source]}->{FAMILY[target]}",
                pair=f"{source}>{target}",
                eval_provenance=condition.get("eval_provenance"),
            )
        )
    rows = pd.concat(frames, ignore_index=True)
    unknown = set(rows["assigned_label"]) - set(LABELS)
    if unknown:
        raise ValueError(f"labels outside the Phase 6 precedence: {sorted(unknown)}")
    return rows


def reconcile(task: int, encoder: str, rows: pd.DataFrame) -> list[str]:
    """Problems found comparing the labels against Phase 3's failure sets.

    Hard rule 2: the denominator is every failed row. Returns one line per
    condition whose (item, seed) set differs, empty when everything matches.
    """
    problems = []
    for condition in _conditions(task):
        name = condition["name"]
        expected = pd.read_parquet(
            failure_path(task, name, encoder=path_encoder(encoder)), columns=["item_id", "seed"]
        )
        got = rows.loc[rows["condition_id"] == name, ["item_id", "seed"]]
        want = set(map(tuple, expected.itertuples(index=False)))
        have = set(map(tuple, got.itertuples(index=False)))
        if len(got) != len(have):
            problems.append(f"{name}: {len(got) - len(have)} duplicate (item, seed) rows in the labels")
        if want != have:
            problems.append(
                f"{name}: {len(want - have)} failures without a label, {len(have - want)} labels without a failure"
            )
    return problems


def evaluated_counts(task: int, encoder: str) -> pd.DataFrame:
    """(condition_id, seed, n_evaluated): rows the model was scored on, so a
    failure count can be read as a rate."""
    out = []
    for condition in _conditions(task):
        preds = pd.read_parquet(
            log_path(task, condition["name"], encoder=path_encoder(encoder)), columns=["item_id", "seed"]
        )
        for seed, group in preds.groupby("seed"):
            out.append({"condition_id": condition["name"], "seed": int(seed), "n_evaluated": len(group)})
    return pd.DataFrame(out)


def untrained_mask(rows: pd.DataFrame, nonconverged: dict[str, set[int]]) -> pd.Series:
    """True on rows whose model never fit its training data.

    `nonconverged` is T-408's map from a run-id prefix (`task3_tel_indicbert`)
    to its untrained seeds, read from `reports/baselines.parquet`. A transfer
    condition's seed is the source model's seed, so the prefix is built from
    the row's task, source language and encoder.
    """
    slug = {DEFAULT_ENCODER: "indicbert", "mbert-base": "mbert"}
    prefix = "task" + rows["task"].astype(str) + "_" + rows["source"] + "_" + rows["encoder"].map(slug)
    mask = pd.Series(False, index=rows.index)
    for key, seeds in nonconverged.items():
        mask |= (prefix == key) & rows["seed"].isin(seeds)
    return mask


def load_all(nonconverged: dict[str, set[int]]) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """Every failed row of the grid with an `untrained` column, the evaluated
    counts, and any reconciliation problems."""
    frames, counts, problems = [], [], []
    for task in TASKS:
        for encoder in ENCODERS:
            rows = load_rows(task, encoder)
            problems += [f"task {task} {encoder}: {p}" for p in reconcile(task, encoder, rows)]
            frames.append(rows)
            counts.append(evaluated_counts(task, encoder).assign(task=task, encoder=encoder))
    rows = pd.concat(frames, ignore_index=True)
    rows["untrained"] = untrained_mask(rows, nonconverged)
    return rows, pd.concat(counts, ignore_index=True), problems


def failure_rates(rows: pd.DataFrame, evaluated: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    """Failed rows over evaluated rows, summed over the conditions in each
    group of `keys`. `rows` must be unfiltered by seed, as `evaluated` is."""
    meta = rows[["task", "encoder", "condition_id", "kind", "source", "target", "quadrant", "pair"]].drop_duplicates()
    ev = evaluated.merge(meta, on=["task", "encoder", "condition_id"])
    ev = ev[ev["condition_id"].isin(rows["condition_id"].unique())]
    n_eval = ev.groupby(keys)["n_evaluated"].sum()
    n_fail = rows.groupby(keys).size()
    out = pd.DataFrame({"n_failed": n_fail, "n_evaluated": n_eval}).dropna().reset_index()
    out["failure_rate"] = out["n_failed"] / out["n_evaluated"]
    return out


# --------------------------------------------------------------------------
# Shares
# --------------------------------------------------------------------------


def share_table(
    rows: pd.DataFrame,
    *,
    n_boot: int = DEFAULT_N_BOOT,
    seed: int = DEFAULT_SEED,
    ci: float = DEFAULT_CI,
) -> pd.DataFrame:
    """One row per measure: pooled share with bootstrap CI, and the per-seed
    mean ± std. `rows` is one group's failed rows."""
    if rows.empty:
        raise ValueError("cannot compute shares over zero failures")

    label_index = {label: j for j, label in enumerate(LABELS)}
    units, unit_codes = np.unique(
        (rows["condition_id"] + "\x1f" + rows["item_id"]).to_numpy(), return_inverse=True
    )
    counts = np.zeros((len(units), len(LABELS)))
    np.add.at(counts, (unit_codes, rows["assigned_label"].map(label_index).to_numpy()), 1.0)
    totals = counts.sum(axis=1)

    point2 = counts.sum(axis=0) / totals.sum()
    rng = np.random.default_rng(seed)
    boot2 = np.empty((n_boot, len(LABELS)))
    for b in range(n_boot):
        weights = np.bincount(rng.integers(0, len(units), size=len(units)), minlength=len(units))
        boot2[b] = (weights @ counts) / (weights @ totals)
    alpha = (1.0 - ci) / 2.0

    per_seed = {}
    for s, group in rows.groupby("seed"):
        shares = group["assigned_label"].value_counts(normalize=True)
        per_seed[int(s)] = np.array([shares.get(label, 0.0) for label in LABELS])

    out = []
    for level, point, boot, names, project in (
        ("L1", point2 @ _GROUP_MATRIX, boot2 @ _GROUP_MATRIX, GROUPS, _GROUP_MATRIX),
        ("L2", point2, boot2, LABELS, None),
    ):
        lo, hi = np.quantile(boot, [alpha, 1.0 - alpha], axis=0)
        for j, name in enumerate(names):
            seed_values = [
                float((v @ project)[j] if project is not None else v[j]) for v in per_seed.values()
            ]
            out.append(
                {
                    "level": level,
                    "measure": name,
                    "share": float(point[j]),
                    "ci_low": float(lo[j]),
                    "ci_high": float(hi[j]),
                    "seed_mean": st.mean(seed_values),
                    "seed_std": st.stdev(seed_values) if len(seed_values) > 1 else None,
                    "n_seeds": len(seed_values),
                    "n_failures": len(rows),
                    "n_units": len(units),
                }
            )
    return pd.DataFrame(out)


def check_shares_sum(table: pd.DataFrame, keys: list[str]) -> list[str]:
    """Each level's pooled shares and per-seed means must sum to 1 within every
    group. A shortfall means a label went uncounted."""
    problems = []
    for key, group in table.groupby(keys + ["level"], dropna=False):
        for column in ("share", "seed_mean"):
            total = group[column].sum()
            if abs(total - 1.0) > 1e-9:
                problems.append(f"{key}: {column} sums to {total:.6f}")
    return problems


# --------------------------------------------------------------------------
# Comparisons
# --------------------------------------------------------------------------


def separated(a_low: float, a_high: float, b_low: float, b_high: float) -> bool:
    """Do two intervals fail to overlap? Touching counts as overlapping."""
    return a_high < b_low or b_high < a_low


def verdict(results: list[tuple[bool, float]], axis: str) -> str:
    """Hard rule 5 over one measure's replicates.

    `results` holds `(separated, a_minus_b)` per replicate. `held` needs every
    replicate separated and pointing the same way; anything less is said
    precisely rather than rounded up.
    """
    seps = [(s, d) for s, d in results if s]
    if not seps:
        return "within noise"
    if len({d > 0 for _, d in seps}) > 1:
        return f"{axis}s disagree"
    if len(seps) == len(results):
        return "held"
    return f"one {axis} only"


def compare(
    rows: pd.DataFrame,
    *,
    a: dict,
    b: dict,
    replicate: str,
    n_boot: int = DEFAULT_N_BOOT,
    seed: int = DEFAULT_SEED,
    ci: float = DEFAULT_CI,
) -> pd.DataFrame:
    """Shares of group `a` against group `b`, once per value of `replicate`.

    `a` and `b` map a column to the values it may take. Returns one row per
    (replicate value, measure) with both groups' shares and CIs and whether the
    intervals separate; `verdict` is applied by the caller over replicates.
    """
    out = []
    for rep_value in sorted(rows[replicate].unique()):
        sub = rows[rows[replicate] == rep_value]
        tables, targets = {}, {}
        for side, spec in (("a", a), ("b", b)):
            mask = pd.Series(True, index=sub.index)
            for column, values in spec.items():
                mask &= sub[column].isin(values)
            if not mask.any():
                raise ValueError(f"group {side} {spec} has no failures at {replicate}={rep_value}")
            tables[side] = share_table(sub[mask], n_boot=n_boot, seed=seed, ci=ci).set_index(["level", "measure"])
            targets[side] = ",".join(sorted(sub.loc[mask, "target"].unique()))
        for level, measure in COMPARED_MEASURES:
            ta, tb = tables["a"].loc[(level, measure)], tables["b"].loc[(level, measure)]
            out.append(
                {
                    "replicate": rep_value,
                    "level": level,
                    "measure": measure,
                    "a_share": ta["share"], "a_ci_low": ta["ci_low"], "a_ci_high": ta["ci_high"],
                    "a_n": int(ta["n_failures"]),
                    "b_share": tb["share"], "b_ci_low": tb["ci_low"], "b_ci_high": tb["ci_high"],
                    "b_n": int(tb["n_failures"]),
                    "a_targets": targets["a"],
                    "b_targets": targets["b"],
                    "diff": ta["share"] - tb["share"],
                    "separated": separated(ta["ci_low"], ta["ci_high"], tb["ci_low"], tb["ci_high"]),
                }
            )
    return pd.DataFrame(out)


def run_comparisons(specs: list[dict], rows: pd.DataFrame, **boot) -> pd.DataFrame:
    """Every spec, once on all seeds and, where the spec's rows include a run
    that never trained, again without those runs (CLAUDE7.md hard rule 6).

    A spec carries `question`, `name`, `a_name`, `b_name`, `fixed` (filters
    applied to both groups), `a`, `b` and `replicate`. Returns one row per
    (seed set, comparison, replicate, measure), each carrying the comparison's
    verdict over its replicates.
    """
    out = []
    for spec in specs:
        mask = pd.Series(True, index=rows.index)
        for column, values in spec["fixed"].items():
            mask &= rows[column].isin(values)
        subset = rows[mask]
        seed_sets = {"all": subset}
        if subset["untrained"].any():
            seed_sets["converged"] = subset[~subset["untrained"]]
        for seed_set, data in seed_sets.items():
            detail = compare(data, a=spec["a"], b=spec["b"], replicate=spec["replicate"], **boot)
            verdicts = {
                key: verdict(list(zip(group["separated"], group["diff"])), spec["replicate"])
                for key, group in detail.groupby(["level", "measure"], sort=False)
            }
            out.append(
                detail.assign(
                    seed_set=seed_set,
                    question=spec["question"],
                    comparison=spec["name"],
                    a_name=spec["a_name"],
                    b_name=spec["b_name"],
                    replicate_axis=spec["replicate"],
                    verdict=[verdicts[(l, m)] for l, m in zip(detail["level"], detail["measure"])],
                )
            )
    return pd.concat(out, ignore_index=True)


def verdict_table(details: pd.DataFrame) -> pd.DataFrame:
    """One row per (seed set, comparison, measure): the verdict, how many
    replicates separated, and the mean difference A − B."""
    keys = ["seed_set", "question", "comparison", "a_name", "b_name", "replicate_axis", "level", "measure",
            "a_targets", "b_targets"]
    return (
        details.groupby(keys, sort=False)
        .agg(
            verdict=("verdict", "first"),
            n_replicates=("separated", "size"),
            n_separated=("separated", "sum"),
            mean_diff=("diff", "mean"),
        )
        .reset_index()
    )


def grouped_shares(rows: pd.DataFrame, keys: list[str], **boot) -> pd.DataFrame:
    """`share_table` for every combination of `keys` present in `rows`."""
    out = []
    for values, group in rows.groupby(keys, sort=True):
        values = values if isinstance(values, tuple) else (values,)
        out.append(share_table(group, **boot).assign(**dict(zip(keys, values))))
    return pd.concat(out, ignore_index=True)


# --------------------------------------------------------------------------
# Rendering
# --------------------------------------------------------------------------


def pct(value, digits: int = 1) -> str:
    return "—" if value is None or pd.isna(value) else f"{100 * value:.{digits}f}%"


def pct_ci(share, low, high) -> str:
    return f"{pct(share)} [{pct(low)}–{pct(high)}]"


def caveat_lines() -> list[str]:
    """CLAUDE7.md hard rule 4, verbatim in every report that shows the labels."""
    return [f"- `{label}`: {text}." for label, text in LABEL_CAVEATS.items()]


def comparison_markdown(details: pd.DataFrame, seed_set: str = "all") -> list[str]:
    """One table per comparison: both groups' shares per replicate, and the verdict."""
    lines = []
    d = details[details["seed_set"] == seed_set]
    for _, group in d.groupby(["question", "comparison"], sort=False):
        first = group.iloc[0]
        reps = list(dict.fromkeys(group["replicate"]))
        lines += [
            f"### {first['comparison']}",
            "",
            f"A = {first['a_name']}; B = {first['b_name']}. Replicated over {first['replicate_axis']}: "
            + ", ".join(str(r) for r in reps) + ".",
            "",
            "| level | measure | " + " | ".join(f"A ({r}) | B ({r})" for r in reps) + " | verdict |",
            "|---|---|" + "---|---|" * len(reps) + "---|",
        ]
        for (level, measure), m in group.groupby(["level", "measure"], sort=False):
            cells = []
            for r in reps:
                row = m[m["replicate"] == r].iloc[0]
                mark = "**" if row["separated"] else ""
                cells.append(f"{mark}{pct_ci(row['a_share'], row['a_ci_low'], row['a_ci_high'])}{mark}")
                cells.append(f"{mark}{pct_ci(row['b_share'], row['b_ci_low'], row['b_ci_high'])}{mark}")
            lines.append(f"| {level} | {measure} | " + " | ".join(cells) + f" | {m['verdict'].iloc[0]} |")
        counts = []
        for r in reps:
            row = group[group["replicate"] == r].iloc[0]
            counts.append(f"{r}: A {int(row['a_n']):,}, B {int(row['b_n']):,}")
        lines += ["", f"Failures counted — {'; '.join(counts)}. Bold: the two intervals do not overlap in that "
                  "replicate.", ""]
    return lines


def converged_changes_markdown(details: pd.DataFrame) -> list[str]:
    """Hard rule 6: where dropping the runs that never trained changes a verdict."""
    v = verdict_table(details)
    conv = v[v["seed_set"] == "converged"]
    lines = ["## Without the runs that never trained (CLAUDE7.md hard rule 6)", ""]
    if conv.empty:
        return lines + ["No comparison here includes a run that never trained.", ""]
    merged = conv.merge(
        v[v["seed_set"] == "all"], on=["question", "comparison", "level", "measure"], suffixes=("_conv", "_all")
    )
    changed = merged[merged["verdict_conv"] != merged["verdict_all"]]
    lines.append(
        f"{conv['comparison'].nunique()} comparison(s) include task-3 IndicBERT-v2 seeds that never trained "
        "and were re-run without them."
    )
    if changed.empty:
        return lines + [" No verdict changes.", ""]
    lines += [" These verdicts change:", "", "| comparison | level | measure | all seeds | converged only |",
              "|---|---|---|---|---|"]
    for _, r in changed.iterrows():
        lines.append(f"| {r['comparison']} | {r['level']} | {r['measure']} | {r['verdict_all']} | {r['verdict_conv']} |")
    return lines + [""]


def group_markdown(table: pd.DataFrame, keys: list[str], rates: pd.DataFrame | None = None) -> list[str]:
    """Level-1 shares with CIs and level-2 pooled shares, one row per group."""
    labels = [label for label in LABELS if label != "unattributed"]
    head = keys + (["failures", "fail rate"] if rates is not None else ["failures"])
    head += ["unattributed", "linguistic", "data", "model"] + labels
    lines = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    rates_by = rates.set_index(keys) if rates is not None else None
    for values, g in table.groupby(keys, sort=True):
        values = values if isinstance(values, tuple) else (values,)
        g = g.set_index(["level", "measure"])
        cells = [str(v) for v in values] + [f"{int(g['n_failures'].iloc[0]):,}"]
        if rates_by is not None:
            cells.append(pct(rates_by.loc[values, "failure_rate"]) if values in rates_by.index else "—")
        for group in ["unattributed", "linguistic", "data", "model"]:
            r = g.loc[("L1", group)]
            cells.append(pct_ci(r["share"], r["ci_low"], r["ci_high"]))
        cells += [pct(g.loc[("L2", label), "share"]) for label in labels]
        lines.append("| " + " | ".join(cells) + " |")
    return lines + [""]


def write_report(name: str, title: str, intro: list[str], body: list[str], details: pd.DataFrame,
                 groups: pd.DataFrame, out_dir, config_hash: str) -> None:
    """`{name}.parquet` (comparison detail with verdicts), `{name}_groups.parquet`
    (the descriptive shares) and `{name}.md`."""
    out_dir.mkdir(parents=True, exist_ok=True)
    details.assign(config_hash=config_hash).to_parquet(out_dir / f"{name}.parquet", index=False)
    groups.assign(config_hash=config_hash).to_parquet(out_dir / f"{name}_groups.parquet", index=False)
    lines = [
        f"# {title}",
        "",
        f"Config hash `{config_hash}`. Shares are of failed (item, seed) rows (CLAUDE7.md). A difference is "
        "a finding only if the two 95% CIs do not overlap and the direction agrees in every replicate "
        "(hard rule 5).",
        "",
        "Two labels carry a stated limitation (hard rule 4):",
        "",
        *caveat_lines(),
        "",
        *intro,
        *body,
        "## Comparisons",
        "",
        *comparison_markdown(details),
        *converged_changes_markdown(details),
    ]
    (out_dir / f"{name}.md").write_text("\n".join(lines) + "\n")
