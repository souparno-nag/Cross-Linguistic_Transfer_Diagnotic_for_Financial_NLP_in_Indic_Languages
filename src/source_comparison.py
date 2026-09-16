"""T-408 — which source language transfers best, and does typology predict it?

CLAUDE6.md: "All source blocks × all encoders × all targets, mean ± std, gaps
with CIs ... This table is what operationalises Lin et al. [16] on
transfer-language selection. For each target language, which source transfers
best — and does typological proximity predict it? That is a result your
original single-source design could only cite, not test."

Everything here is read back out of T-305's prediction logs through T-308's
existing machinery (`results_tables.condition_summary`), never transcribed and
never recomputed by a second route. A comparison table that derives its own
numbers can disagree with the report it claims to summarise, and it does so
silently.

Three design decisions, each made because the obvious alternative is wrong:

**Gaps, not target scores.** Ranking sources by the raw macro-F1 they achieve
on a target would rank the encoders' in-language ability as much as their
transfer. The gap is `source − target` on the *same items* (T-304's paired
bootstrap), so a source that starts high and falls far is not flattered by
where it started.

**Non-converged runs are excluded from the ranking and reported separately.**
This is the only place the pipeline is allowed to drop a seed, and it is
forced: a model that never left its initialisation scores equally badly in its
own language and in every other, so its transfer gap is near *zero* and it
ranks as the best source in the table. That is not a subtle distortion — it
inverted task 3's Bengali comparison at T-404 and would make Telugu look like
the strongest source on task 3 when two of its three seeds never trained. The
excluded seeds are named in the report, per `hard rule 1`'s spirit: flag, never
hide.

**Typological proximity is tested, not assumed.** `FAMILY` is the same table
`conditions.py` uses. The prediction — that a same-family source transfers
better — is stated, checked against every target that has both a same-family
and a cross-family source available, and reported whichever way it comes out.
"""

from __future__ import annotations

import statistics as st

from .conditions import FAMILY
from .data import DEFAULT_ENCODER
from .evaluate import _split_lang, load_matrix
from .ids import BLOCK_NATIVE_LANG
from .results_tables import condition_summary
from .transfer import DEFAULT_N_BOOT

#: Encoders whose artefacts exist. XLM-R is VRAM-blocked (CLAUDE5.md T-501) and
#: is carried as an explicit exclusion rather than dropped, so the lost arm is
#: visible in the table instead of being absent from it.
ENCODERS = (DEFAULT_ENCODER, "mbert-base")
EXCLUDED_ENCODERS = {
    "xlm-r-base": (
        "VRAM-blocked: a full fine-tune needs 4.14 GiB of weights, gradients "
        "and AdamW state before any activation, against 3.68 GiB of card "
        "(CLAUDE5.md T-501). Not a missing run — an impossible one on this "
        "hardware."
    )
}


def source_of(condition: dict) -> str:
    """The language a condition's model was trained on.

    Taken from the condition's `train` split, **not** from a prediction row's
    `block_id`: that column names the block being *evaluated*, so for
    `transfer_ben_to_hin` it reads `H`. Grouping a source-selection table on it
    would group by the wrong axis entirely (CLAUDE6.md Outputs).
    """
    _, lang = _split_lang(condition["train"])
    return lang


def target_of(condition: dict) -> str:
    _, lang = _split_lang(condition["eval"])
    return lang


def same_family(src: str, tgt: str) -> bool:
    return FAMILY[src] == FAMILY[tgt]


# --------------------------------------------------------------------------
# Building the cells
# --------------------------------------------------------------------------


def cells(
    task: int,
    *,
    encoders: tuple[str, ...] = ENCODERS,
    kind: str = "transfer",
    seeds: tuple[int, ...] = (0, 1, 2),
    nonconverged: dict[str, set[int]] | None = None,
    n_boot: int = DEFAULT_N_BOOT,
) -> list[dict]:
    """One row per (encoder, source, target), via T-308's own summariser.

    `nonconverged` maps a run_id prefix (`task3_tel_indicbert`) to the seeds
    that never fit their training data; those seeds are left out of the mean
    and counted in `n_excluded`.
    """
    matrix = load_matrix(task)
    nonconverged = nonconverged or {}
    out = []
    for encoder in encoders:
        for condition in matrix["conditions"]:
            if condition["kind"] != kind:
                continue
            src, tgt = source_of(condition), target_of(condition)
            slug = "indicbert" if encoder == DEFAULT_ENCODER else "mbert"
            bad = nonconverged.get(f"task{task}_{src}_{slug}", set())
            usable = tuple(s for s in seeds if s not in bad)
            summary = (
                condition_summary(
                    task, condition, seeds=usable, encoder=encoder, n_boot=n_boot
                )
                if usable
                else {"status": "blocked", "condition": condition["name"]}
            )
            out.append(
                {
                    "task": task,
                    "encoder": encoder,
                    "source": src,
                    "target": tgt,
                    "quadrant": condition.get("quadrant"),
                    "same_family": same_family(src, tgt),
                    "n_excluded": len(bad),
                    "excluded_seeds": sorted(bad),
                    **{
                        k: summary.get(k)
                        for k in (
                            "status", "n_seeds", "n_items", "source_mean",
                            "target_mean", "gap_mean", "gap_std",
                            "within_seed_noise",
                        )
                    },
                }
            )
    return out


def rank_sources(rows: list[dict]) -> list[dict]:
    """Mean gap per (encoder, source) — lower is a better transfer source."""
    out = []
    groups: dict[tuple[str, str], list[dict]] = {}
    for r in rows:
        if r["status"] != "ok":
            continue
        groups.setdefault((r["encoder"], r["source"]), []).append(r)
    for (encoder, source), group in sorted(groups.items()):
        gaps = [g["gap_mean"] for g in group]
        out.append(
            {
                "encoder": encoder,
                "source": source,
                "n_targets": len(group),
                "mean_gap": st.fmean(gaps),
                "worst_target": max(group, key=lambda g: g["gap_mean"])["target"],
                "best_target": min(group, key=lambda g: g["gap_mean"])["target"],
            }
        )
    return sorted(out, key=lambda r: (r["encoder"], r["mean_gap"]))


def best_source_per_target(rows: list[dict]) -> list[dict]:
    """For each (encoder, target): which source transfers in best, and does
    typological proximity predict it?

    `proximity_predicts` is `None` where the question cannot be asked — a
    target with no same-family source available, or none cross-family.
    """
    out = []
    groups: dict[tuple[str, str], list[dict]] = {}
    for r in rows:
        if r["status"] != "ok":
            continue
        groups.setdefault((r["encoder"], r["target"]), []).append(r)

    for (encoder, target), group in sorted(groups.items()):
        ranked = sorted(group, key=lambda g: g["gap_mean"])
        winner = ranked[0]
        kin = [g for g in group if g["same_family"]]
        foreign = [g for g in group if not g["same_family"]]
        predicts = None
        if kin and foreign:
            predicts = winner["same_family"]
        out.append(
            {
                "encoder": encoder,
                "target": target,
                "best_source": winner["source"],
                "best_gap": winner["gap_mean"],
                "ranking": [(g["source"], g["gap_mean"]) for g in ranked],
                "same_family_available": bool(kin),
                "proximity_predicts": predicts,
            }
        )
    return out


def asymmetries(rows: list[dict]) -> list[dict]:
    """Each unordered language pair, both directions, and the difference.

    CLAUDE6.md's working agreement: "If Hi->Bn and Bn->Hi differ substantially,
    that is a finding, not an error to be explained away." Malayalam has no
    native split so it is never a source and appears in no pair here.
    """
    by_key = {
        (r["encoder"], r["source"], r["target"]): r for r in rows if r["status"] == "ok"
    }
    natives = sorted(set(BLOCK_NATIVE_LANG.values()))
    out = []
    seen = set()
    for encoder, a, b in sorted({(r["encoder"], r["source"], r["target"]) for r in rows}):
        if a not in natives or b not in natives or (encoder, b, a) in seen:
            continue
        fwd, rev = by_key.get((encoder, a, b)), by_key.get((encoder, b, a))
        if fwd is None or rev is None:
            continue
        seen.add((encoder, a, b))
        delta = fwd["gap_mean"] - rev["gap_mean"]
        spread = (fwd["gap_std"] ** 2 + rev["gap_std"] ** 2) ** 0.5
        out.append(
            {
                "encoder": encoder,
                "pair": f"{a}<->{b}",
                "forward": f"{a}->{b}",
                "forward_gap": fwd["gap_mean"],
                "reverse": f"{b}->{a}",
                "reverse_gap": rev["gap_mean"],
                "difference": abs(delta),
                "harder_direction": f"{a}->{b}" if delta > 0 else f"{b}->{a}",
                "combined_seed_spread": spread,
                # A difference smaller than the two cells' combined seed spread
                # is not a finding (hard rule 5, applied to the comparison
                # rather than to either cell).
                "beats_noise": abs(delta) > spread,
            }
        )
    return out


def target_family_effect(rows: list[dict]) -> list[dict]:
    """Mean gap by the target's family, pooled over sources.

    Phase 1's T-111 found that in the machine translation itself, crossing
    *into* Dravidian drifted more than crossing back, "the cost is in the
    direction of travel, not the pairing". This asks whether the classifier
    agrees — and it is deliberately pooled over source family, because that is
    the comparison that separates "target is hard" from "pairing is distant".
    """
    out = []
    groups: dict[tuple[str, str], list[float]] = {}
    for r in rows:
        if r["status"] != "ok":
            continue
        groups.setdefault((r["encoder"], FAMILY[r["target"]]), []).append(r["gap_mean"])
    for (encoder, family), gaps in sorted(groups.items()):
        out.append(
            {
                "encoder": encoder,
                "target_family": family,
                "n_cells": len(gaps),
                "mean_gap": st.fmean(gaps),
                "std": st.stdev(gaps) if len(gaps) > 1 else 0.0,
            }
        )
    return out
