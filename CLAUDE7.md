# CLAUDE.md — Phase 7: Root-Cause Findings

Scope: **Phase 7 only.** Aggregate Phase 6's per-instance cause labels into findings:
which causes account for the failures, and how that mix changes with target language,
source language, encoder and test-text provenance. Do **not** change diagnostic modules,
thresholds or labels (Phase 6), and do not train or evaluate models (Phases 2–5). If a
request drifts there, say so and stop.

**Status: DONE, 2026-10-13.** All seven tasks complete; `reports/findings.md` is the
result. The whole phase rebuilds in about 30 s on CPU, byte-identical on rerun:

```
python -m scripts.t701_shares && python -m scripts.t702_by_target && \
python -m scripts.t703_by_source && python -m scripts.t704_by_encoder && \
python -m scripts.t705_provenance && python -m scripts.t706_robustness && \
python -m scripts.t707_findings
```

### Decisions made while building it

- **Bootstrap unit is the (condition, item) cluster**, not the row. An item failing on
  three seeds is one draw; resampling rows would make every interval too narrow. The
  condition is part of the key because grouped tables pool conditions that share item ids.
- **T-702–T-704 use native-text conditions only** (`kind == "transfer"`), so provenance
  is not mixed into the target, source and encoder comparisons. T-705 owns provenance.
- **T-703 compares each pair of sources on the targets both reach** (hin vs ben on
  mal+tel, and so on); pooling every target would compare targets as much as sources.
- **Level-2 `unattributed` is not tested separately**; it is level-1 `unattributed`
  again, and testing it twice would count one result as two.
- **Known artefacts are attached automatically** (`findings.CAVEATS`): a comparison with
  Hindi targets on one side, for `morphological_masking` or `linguistic`; native-vs-MT
  `translation_drift`; Bengali native vs MT. T-707 lists artefact-caveated findings
  separately, and for the Hindi blind spot shows the same measure on a Hindi-free pair.
- **Failure rate** is failed rows over evaluated rows. A failure needs the source
  prediction to be right, so a stronger model has more chances to fail; read rates with
  that in mind.

---

## Depends on

- `data/diagnostics/task_{2,3}/[{encoder}/]{condition}.parquet` — Phase 6 labels, one row
  per (item, seed) failure. The `{encoder}` level is omitted for `indicbert-v2` and is
  `mbert-base/` for mBERT (CLAUDE6.md, Outputs).
- `data/diagnostics/audit/` — every module's status per row, for the trace behind a label.
- `reports/transfer_results.md`, `reports/source_comparison.md`,
  `reports/encoder_comparison.md` — the Phase 3–5 effects this phase explains.
- `src/transfer.py` `bootstrap_metric` — reuse it for confidence intervals. Do not write a
  second bootstrap.

Task 1 has no classifier and therefore no failures to diagnose; this phase covers tasks
2 and 3 only.

---

## What gets counted

**Decided 2026-10-13: findings are label shares among failures.** For each cell, the share
of failed (item, seed) rows carrying each label. Labels are not compared against correct
predictions. A share says how a cell's failures are made up, and the findings report says
it in those terms.

The grid is 2 tasks × 2 encoders × 18 conditions = **72 cells**. The 18 conditions are 9
source→target pairs, each evaluated on native target text and on MT target text
(`transfer_{src}_to_{tgt}` and `…_mt`). Malayalam has no native split, so its "native"
condition evaluates on MT from another block (`eval_provenance` in
`configs/eval_conditions.json`). Its native/MT pair is not a provenance contrast.

Labels come at two levels, both reported:

| Level 1 `label_group` | Level 2 `assigned_label` |
|---|---|
| `linguistic` | translation_drift, tokenizer_fragmentation, morphological_masking, terminology_gap |
| `data` | provenance_specific, cross_lingual_hard |
| `model` | class_collapse, seed_unstable |
| `unattributed` | unattributed |

Every share is computed **per seed** and reported as mean ± std over the 3 seeds, with a
bootstrap 95% CI over items.

---

## Hard rules

1. **Read-only on Phase 6 output.** No relabelling, no threshold change, no change to
   precedence. If a label looks wrong, report the rows; do not fix it here.
2. **Never drop a failure.** The denominator is every failed row in the cell. Row counts
   must reconcile with `data/failures/` exactly.
3. **Report `unattributed` in every table**, never renormalised away. It is the honest
   measure of the pipeline's coverage (CLAUDE4.md).
4. **`morphological_masking` and `terminology_gap` are causes, and carry their
   limitation wherever they appear.** Masked terms do not raise the failure rate (20 of
   44 cells, `reports/morphology_null.md`); salience divergence does not separate failures
   from correct pairs (AUC 0.48–0.52, `reports/saliency_baseline.md`). One sentence each,
   next to the number, in every table and in the findings report.
5. **The rule for calling a difference a finding is fixed here, before any table is
   read.** Two cells differ only if their bootstrap 95% CIs do not overlap **and** the
   difference points the same way on both encoders. Anything else is reported as
   "encoder-specific" or "within noise". Do not adjust this rule after seeing results;
   that is the same fabrication CLAUDE4.md hard rule 3 forbids for thresholds.
6. **Seeds are not excluded.** Runs that never trained stay in, as in
   `reports/baselines.md`; where they move a share, show the figure with and without them,
   never only without.
7. **Deterministic, config hash logged**, as everywhere else (CLAUDE.md §4 rules 7–8).
   Tables to Parquet, prose to Markdown. CPU only; no GPU work in this phase.

---

## Known measurement limits — state them, do not explain them away

These are properties of the Phase 6 measurement, not findings about languages. Each must
appear in the findings report wherever it affects a number.

- **Hindi targets never get `morphological_masking`.** Hindi writes postpositions as
  separate words, so nothing attaches to the term. A 0% Hindi share is the module's
  blind spot, not a statement about Hindi.
- **`tokenizer_fragmentation` barely fires**: 0.0% of failures on IndicBERT-v2, 0.4–0.5%
  on mBERT. R_frag ≥ 2.0 is rarely reached. Report it as measured; do not lower the
  threshold.
- **The orthographic/numeral module is excluded** (CLAUDE4.md, Scope decisions). Numeral
  failures fall through to other labels or to `unattributed`.
- **Task 3 IndicBERT-v2 has 8.2% unconverged saliency rows**, above the 5% limit. Those
  rows cannot fire `terminology_gap`, so that cell's share understates it.
- **Native vs MT Bengali differ in digit system as well as provenance.** Any `B_nat` vs
  `Bn←H/T` comparison says so (CLAUDE4.md, Scope decisions).
- **No native Malayalam** (above).
- **Task 3 is at its data ceiling.** Its transfer gaps cluster at 0.47–0.53 whatever the
  language or encoder (Phase 4). Produce its tables; expect most of its differences to
  fail rule 5, and say so rather than reaching for one.
- **The hand spot-check covers task 2 IndicBERT-v2 only** (98.5%, T-608). The other three
  (task, encoder) cells' labels are unverified by hand.

---

## Tasks

### T-701 — Label-share table
`src/findings.py`, `scripts/t701_shares.py`

Level-1 and level-2 shares for all 72 cells, mean ± std over seeds and bootstrap CI.

**Done when:** `reports/phase7/label_shares.parquet` has every cell; shares sum to 100%
per (cell, seed); failure counts reconcile with `data/failures/` row for row; the script
exits non-zero otherwise.

**Status: DONE.** 72/72 cells, 42,242 failed rows, every one reconciled; pooled totals
match `reports/diagnostics_summary.md` exactly. Unattributed 9.2–19.7% per (task, encoder),
0.3–28.8% per cell.

### T-702 — By target language and typology
Cause mix grouped by target language and by quadrant (IA→IA, IA→Dr, Dr→IA, Dr→Dr), with
the transfer gap from `reports/source_comparison.md` beside it.

The question: crossing *into* Dravidian costs more than crossing back, on all three tasks
(Phase 1 T-111, Phase 4). Does the cause mix differ between IA→Dr and Dr→IA in the same
language pairs?

**Done when:** the table exists and the question is answered under rule 5, including
"no difference" if that is the answer.

**Status: DONE.** No artefact-free linguistic difference. The morphological-masking
difference into vs out of Telugu rests on Hindi targets, which the module cannot label; on
ben↔tel the linguistic share is within noise. Artefact-free: task-2 failures out of
Telugu carry more model-behaviour labels.

### T-703 — By source language
The question: Hindi is the worst source on both encoders (T-408). Do Hindi-source failures
have a different cause mix from Bengali- and Telugu-source ones, or the same mix at a
higher volume?

**Done when:** answered under rule 5.

**Status: DONE.** Hindi vs Bengali: no difference held on either task, at a higher Hindi
failure rate. Hindi vs Telugu, task 2: Telugu-source failures carry more model-behaviour
labels.

### T-704 — By encoder
The question: mBERT beats IndicBERT-v2 on every measure (Phase 5). Which labels account
for the difference — for example, does IndicBERT-v2 fail more often through
`class_collapse`?

**Done when:** answered under rule 5. Rule 5's "both encoders" clause cannot apply to a
comparison *between* encoders; require CIs to separate on both tasks instead, and say
that is the rule used.

**Status: DONE.** Held on both tasks: IndicBERT-v2 failures carry more `class_collapse`
(27.3% vs 0.0% on task 2) and `cross_lingual_hard`; mBERT's more `seed_unstable` and
`provenance_specific`.

### T-705 — Native vs MT target text
`provenance_specific` and `translation_drift` shares, native vs MT, per target language.
How much of the transfer loss does our own translation account for?

**Done when:** answered for Bengali, Hindi and Telugu targets, with the Bengali
digit-system caveat; Malayalam reported as having no native contrast.

**Status: DONE.** On MT text 9.5–25.0% of failures carry `translation_drift` or
`provenance_specific`. The only native-vs-MT differences that hold are in
`translation_drift`, an artefact of the measure.

### T-706 — Robustness summary
One table of every candidate finding from T-702 to T-705: held on both encoders, held on
one encoder only, or within noise.

**Done when:** every claim in T-707's report points to a row in this table.

**Status: DONE.** 228 rows with stable ids (`R702-01` …): 22 held (11 artefact-caveated),
8 opposite-direction, 78 one replicate only, 120 within noise. 20 verdicts change without
the runs that never trained, one of them to `held` (`R705-53`).

### T-707 — Findings report
`reports/findings.md`, generated from T-701–T-706's artefacts the way T-114's datasheet is,
with a `--check` that fails if it no longer matches them.

Must contain: the headline cause mix per (task, encoder); answers to T-702–T-705; the
`unattributed` rate up front; every limit in "Known measurement limits"; hard rule 4's
caveats next to the two labels.

Then update the prose that predates this phase: `README.md` ("nothing in this repository
claims to explain a cause yet") and `docs/implementation_results.md`, which still says the
Bengali and Telugu baselines were never checked against the paper.

**Done when:** `reports/findings.md` exists and `--check` passes; README and
implementation_results.md updated.

**Status: DONE.** `--check` passes, and `tests/test_findings.py` fails if the committed
report drifts from its artefacts. README and `docs/implementation_results.md` updated.

---

## Outputs

| Artefact | Path |
|---|---|
| Label shares | `reports/phase7/label_shares.{parquet,md}` |
| By target / typology | `reports/phase7/by_target.{parquet,md}` |
| By source | `reports/phase7/by_source.{parquet,md}` |
| By encoder | `reports/phase7/by_encoder.{parquet,md}` |
| Native vs MT | `reports/phase7/provenance.{parquet,md}` |
| Robustness | `reports/phase7/robustness.{parquet,md}` |
| Findings | `reports/findings.md` |

---

## Working agreement

- **Lead with `unattributed`.** A findings report that opens with the largest cause and
  buries the unexplained share misstates the pipeline's reach.
- **A null answer is an answer.** "The cause mix does not differ between IA→Dr and Dr→IA"
  is a finding; do not keep slicing until something separates.
- **When a number looks wrong, show the rows.** A count is not a diagnosis.
- **Prefer boring code.** Grouped counts and the existing bootstrap; no new statistics
  machinery.
- **Commit after every task.**
