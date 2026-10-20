# The diagnostic pipeline, explained from scratch

This guide explains Phase 6: how the project takes a failed prediction and tries to
attach a named cause to it. It assumes you have met the transfer gap (see
`docs/zero_shot_transfer.md`). The facts below were checked against `CLAUDE4.md`,
`src/diagnostics.py`, `src/fragmentation.py`, `src/morphology.py`, `src/saliency.py`,
`src/behaviour.py`, `reports/diagnostics_summary.md` and a sample audit row. No
diagnostic job was re-run.

---

## 1. The problem it solves

Phase 3 tells you *how much* performance is lost when a Hindi-trained model reads
Bengali. It does not tell you *why* any particular sentence failed. Phase 6 takes each
failed sentence and asks a series of yes/no questions, and gives it **exactly one
label**, plus a record of every question that answered yes.

Think of a hospital triage desk. A patient (a failed item) arrives. The first check is
"is this even the right patient file?" (was the translation faithful?). Only after that
passes do you check specific conditions, in a fixed priority order. If nothing matches,
the patient is recorded as "undiagnosed". That is allowed, and the pipeline reports it.

---

## 2. What goes in

The input is the **failed-instance set** from Phase 3 (`data/failures/task_{n}/...`):
items where

```
the model was right in the source language   AND   wrong in the target language
```

The source-right condition matters. It rules out "the item was just hard" and "the
model is just weak", so what remains needs explaining by something that changed with the
language or the run.

Each failure is diagnosed per **condition** (e.g. `transfer_hin_to_ben`), per **seed**,
and per **encoder** (IndicBERT-v2, mBERT). The `encoder_id` is stored on every output
row, because the pipeline is meant to run on any encoder by changing a setting, not by
rewriting code.

---

## 3. The three stages, in order

```
 failed item
     |
     v
 STAGE 1  semantic gate        sim < tau ?  --yes--> translation_drift   (STOP)
     | no
     v
 STAGE 2  linguistic modules   fragmentation / morphology / saliency
     |
     v
 STAGE 3  data + model modules provenance / cross-lingual / collapse / seed
     |
     v
 pick ONE label by precedence  (or "unattributed" if nothing fired)
```

**The order is the whole point, and it is a hard rule.** If linguistic checks ran first,
noise from the project's own machine translation would be blamed on tokenization or
morphology. So the first question is always: "does the target sentence even still mean
the same thing as the source?"

---

## 4. Stage 1: the semantic gate

This is the LaBSE check from Phase 1, reused (the code imports `src/labse_gate.py`
rather than reimplementing it). LaBSE turns each sentence into a vector, and the
cosine of two vectors says how close their meanings are.

- If the source/target similarity is **below τ** (0.82 for task 2, 0.73 for task 3, as
  calibrated in Phase 1), the item is labelled `translation_drift` and goes **no
  further**. The code returns early: no other module even runs on it.
- Otherwise it passes to Stage 2 and Stage 3.

The item is kept either way. Nothing is dropped (hard rule 4).

---

## 5. Stage 2: linguistic modules

Every gate-passed item runs **all three** modules, even if an earlier one fires, so the
audit trail records everything that matched, not just the winner.

### 5.1 Tokenizer fragmentation → `tokenizer_fragmentation`

**Idea.** The model never sees words, only tokens (pieces from its vocabulary). A
language the tokenizer knows poorly gets chopped into many small pieces, so the same
meaning takes more tokens and the model's reading of it may degrade.

**Test.** Tokenize source and target with **the encoder's own tokenizer**:

```
R_frag = number of target tokens / number of source tokens
fires when R_frag >= 2.0
```

Special tokens are not counted, and the sentence is not truncated first (truncating
would hide the very fragmentation the module looks for).

**What the run found:** it almost never fires: 0.0% of task 2/3 failures for
IndicBERT-v2 and about 0.4–0.5% for mBERT. The threshold of 2.0 is a stated convention,
not tuned to produce a result.

### 5.2 Morphological masking → `morphological_masking`

**Idea.** Indic languages glue endings onto words. A key financial or ESG word can be
present in the sentence only as a stem plus a suffix (English adds "-s" or "-ing"; these languages stack
much longer endings), which the model may not recognise as the same
word it learned in the source language.

**Test.** Using a small curated lexicon (`configs/esg_terms.json`, 18 concepts such as
"sustainable", each with a word in all four languages):

1. For each term in the target language, if the term appears **intact** as a word, skip
   it. It is visible, not masked.
2. Otherwise segment each word into stem + suffix, using the Indic NLP Library's
   morphology model, falling back to a hand-made suffix list.
3. If some word has a suffix and its stem matches the term, the module **fires** and
   records the evidence (term, word, stem, suffix).

The stem match rule is strict, after a bug was found in an earlier version that let
two-character terms match unrelated words. Today it accepts only exact matches, or
(for terms of 4+ characters) a stem with the last character dropped or one vowel sign
added, which are the changes inflection really makes. Hindi targets give 0%, because
Hindi writes its postpositions as separate words. Multi-word terms cannot match.

### 5.3 Integrated Gradients → `terminology_gap`

**Idea.** Does the model pay as much attention to the key term in the target sentence
as it did in the source sentence?

**How it measures attention.** Integrated Gradients (IG) is a standard attribution
method. It asks how much each input token contributed to the model's output score for
the correct class, by sliding the input from a "blank" reference to the real sentence
and adding up the gradients along the way.

Details that matter:

- **Reference input:** every content token replaced by the `[PAD]` embedding, keeping
  the first and last positions. (An all-zeros reference was tried first and failed to
  converge on IndicBERT-v2.)
- **Per-token salience:** the size (L2 norm) of each token's attribution.
- **Term share:** the share of total salience carried by the lexicon term's tokens.
- **Divergence:** `share in source − share in target`. A large positive value means the
  term mattered in the source but gets less attention in the target.
- **Only applies** when the source sentence contains a lexicon term. Otherwise the
  status is `not_applicable`, not "no".
- **Convergence check:** IG's own error estimate must be within 5% of the score
  difference it should explain. If not, the status is `not_converged` and it never
  fires. Steps per run were raised (400 or 800 for harder cells) to get there; this
  changes accuracy, not what is measured.

**The threshold was set against a null, not against the failures.** Setting it from the
failures' own spread would fix the firing rate by construction. Instead the project
measured divergence on pairs the model got *right in both languages*, and used that
distribution's 95th percentile (per task and encoder: 0.038 to 0.085). So a correct
pair crosses the line about 5% of the time by design.

### 5.4 The excluded module

A fourth module (digit systems and currency symbols, `orthographic_numeral_mismatch`)
was designed but **left out**. The audit (`reports/digit_audit.md`) showed that items
with a digit-system change did not fail more often than others (e.g. 12–13% of
failures versus 34% of all items in the same pairs on task 2). Labelling it a cause
would have presented a correlate as a cause. The unused rung is skipped in the
precedence list.

---

## 6. Stage 3: data and model-behaviour modules

These explain a failure by the *item* or the *model run*, not by the target sentence's
language. They are read directly off the prediction logs (CPU only), run per
(task, encoder) group after Stages 1 and 2, and only on gate-passed items.

| Module | Fires when | Label | Group |
|---|---|---|---|
| provenance | the same (item, seed) is **right** in the "twin" condition: native target vs machine-translated target, same language, same source | `provenance_specific` | data |
| cross-lingual | the same (item, seed), trained on the same source, fails in **all 3** target languages | `cross_lingual_hard` | data |
| class collapse | the run predicts **one class on ≥ 75%** of the condition's target rows, and this failure is that class | `class_collapse` | model |
| seed instability | the same item is **right on another seed** of the same condition (and that seed's source was also right) | `seed_unstable` | model |

The cutoffs (75%, "all 3", "another seed") were fixed before the first run and are not
allowed to move to improve a distribution. 75% sits well above the largest class share
in task 2 (55%), so a model that predicts one class that often is not tracking labels.

Because these are read off logs, they are *facts about the run* ("this model gave the
same answer for everything"), not guesses about language. They are also generic:
`seed_unstable` just says training randomness decided the item. They were added so that
every failure gets a reason, including reasons about the model, and not because each is
equally informative.

---

## 7. Choosing one label: precedence

An item can satisfy several modules. Exactly one label is assigned, by this order
(first one that fired wins):

```
1. translation_drift          (Stage 1, terminal)
2. tokenizer_fragmentation
3. morphological_masking
4. terminology_gap
5. provenance_specific        (Stage 3, data)
6. cross_lingual_hard         (Stage 3, data)
7. class_collapse             (Stage 3, model)
8. seed_unstable              (Stage 3, model)
9. unattributed               (nothing fired)
```

The code is `precedence_pick`: loop through the list, return the first label whose
module is in the item's fired set, else `unattributed`. Language-based explanations rank
above data and model ones, so Stage 3 only claims what Stages 1 and 2 left over.

Every label also carries a coarse **group** derived from it (never decided separately,
so the two can't disagree):

| group | labels |
|---|---|
| linguistic | translation_drift, tokenizer_fragmentation, morphological_masking, terminology_gap |
| data | provenance_specific, cross_lingual_hard |
| model | class_collapse, seed_unstable |
| unattributed | unattributed |

`unattributed` is a legitimate answer. A pipeline that explains 100% of failures would
be suspicious, so the unattributed share is reported prominently as the honest measure
of coverage.

---

## 8. The audit trail

Hard rule 1: one label **plus** a full record. For each failure there are two rows:

- **Label file** (`data/diagnostics/task_{n}/{condition}.parquet`): `condition_id`,
  `encoder_id`, `item_id`, `block_id`, `src_lang`, `tgt_lang`, `assigned_label`,
  `label_group`, `modules_fired`, `r_frag`, `labse_sim`, `seed`.
- **Audit file** (`data/diagnostics/audit/task_{n}/{condition}.parquet`): the same, plus
  every module's status and number: `gate_status`, `frag_status`, `morph_status`,
  `morph_evidence`, `saliency_status`, `saliency_divergence`,
  `saliency_convergence_error`, `saliency_n_steps`, and the four Stage 3 statuses.

Status values you will see: `fired`, `not_fired`, `not_applicable` (the module can't ask
the question here), `not_converged` (IG failed its own check), `unavailable` (tool
missing) and `not_evaluated` (an earlier stage stopped the item).

A real audit row, abridged:

```
condition_id: transfer_ben_to_hin     gate_status: passed   labse_sim: 0.919
frag_status: not_fired (r_frag 1.11)  morph_status: not_fired
saliency_status: not_fired            seed_status: fired
modules_fired: ['seed']               assigned_label: seed_unstable   label_group: model
```

So any label can be traced back to the module that decided it and to everything else
that matched.

---

## 9. How a run goes

1. `scripts/t601_diagnostics.py` finds every condition with a failures file.
2. For each condition it scores LaBSE similarity for the failing items (cached
   embeddings are reused; LaBSE is loaded only if needed and released straight away,
   because a 4 GB GPU can't hold it beside a classifier).
3. For each failure it loads the checkpoint for that exact run (one at a time), runs
   Stage 1, then Stage 2, building the audit row.
4. At the end, `scripts/t609_behaviour.py` runs Stage 3 across all conditions of the
   (task, encoder) group and re-resolves the final label.
5. Reports are produced with `scripts/t601_report.py` into
   `reports/diagnostics_summary.md`.

The saliency step is the GPU-heavy one, which is why you run it yourself. The labels
are deterministic: same input and config give the same labels.

---

## 10. What the run found

Share of failures by group (`reports/diagnostics_summary.md`):

| | task 2, IndicBERT-v2 | task 2, mBERT | task 3, IndicBERT-v2 | task 3, mBERT |
|---|---|---|---|---|
| failures diagnosed | 19,323 | 9,714 | 6,102 | 7,103 |
| linguistic | 12.0% | 11.3% | 8.8% | 9.4% |
| data | 44.6% | 48.3% | 72.2% | 64.9% |
| model | 34.2% | 20.8% | 7.1% | 11.2% |
| **unattributed** | **9.2%** | **19.7%** | **11.8%** | **14.4%** |

`cross_lingual_hard` is the largest single label in every cell (26–63%). Among the
linguistic labels, `translation_drift` is 1.9–5.8%, morphological masking 2–8%,
terminology gap 1–2% and fragmentation about 0%. `class_collapse` is 27% of task-2
IndicBERT-v2 failures and 0% for mBERT: those IndicBERT-v2 runs predict one class on
69–92% of target rows against a 55/45 gold split, which is a training-quality issue in
its own right.

---

## 11. How far to trust the labels

The project is explicit that these are not all equally strong. Read each label with
its caveat:

- **Data and model labels** (provenance, cross-lingual, collapse, seed) are facts read
  from the logs, so they need no statistical check. They are generic, though: they say
  *that* the item is hard across languages or the run was unstable, not *why*.
- **`morphological_masking` is labelled a cause, but the evidence for that is weak.**
  Over all (item, seed) pairs, a masked term does not raise the failure rate: masked
  items fail more often than intact ones in only 20 of 44 cells
  (`reports/morphology_null.md`).
- **`terminology_gap` is also labelled a cause, but it fires at about chance.**
  Divergence does not separate failures from correct pairs (AUC 0.48–0.52), and it
  fires on 1–2% of failures, near the 5% false-positive rate its threshold was built
  to have (`reports/saliency_baseline.md`).
- **Hand check.** A manual spot-check of 200 task-2 IndicBERT-v2 rows agreed with the
  assigned label 197 times (98.5%). But 160 of those rows carried data or model labels,
  so the linguistic labels rested on only 40 rows, and task 3 and mBERT weren't
  sampled.
- The two weak labels were kept as causes on purpose, since this build is a
  demonstration of the pipeline, and each place that reports them must carry these
  limitations alongside.

Put plainly: the pipeline proves it can attach a rule-based reason to every failure and
show its working. It does **not** yet prove that the morphological or terminology reasons
are the true causes.

---

## 12. Rules the code follows

1. One label per instance, with a full audit trail.
2. Never reorder the stages. The gate runs first.
3. Never tune a threshold to improve the distribution. Thresholds come from principle
   or from a null, and are then reported.
4. Never drop an instance. `unattributed` is the catch-all.
5. Deterministic.
6. Encoder-agnostic: modules take the tokenizer or encoder as input (the code uses
   `get_input_embeddings()` rather than anything IndicBERT-specific).
7. No aggregation into findings here. That is Phase 7, which counts and compares the
   labels with confidence intervals.

---

## 13. File map

| What | Where |
|---|---|
| Router, precedence, audit I/O | `src/diagnostics.py` |
| Fragmentation ratio | `src/fragmentation.py` |
| Morphological masking | `src/morphology.py` |
| Integrated Gradients, thresholds | `src/saliency.py` |
| Stage 3 (data and model) | `src/behaviour.py`, `scripts/t609_behaviour.py` |
| ESG lexicon | `configs/esg_terms.json` |
| Run, report, rethreshold | `scripts/t601_diagnostics.py`, `t601_report.py`, `t601_rethreshold.py` |
| Validity checks | `scripts/t604_morph_null.py`, `t605_baseline.py`, `t607_spotcheck.py` |
| Dropped orthographic module | `scripts/digit_audit.py`, `reports/digit_audit.md` |
| Outputs | `data/diagnostics/task_{n}/`, `data/diagnostics/audit/task_{n}/` |
| Summary | `reports/diagnostics_summary.md` |
| Spec | `CLAUDE4.md` (the Phase 6 spec; note the file numbering: `CLAUDE4.md` is Phase 6) |

---

## 14. Glossary

- **Failed instance / mismatch:** right in the source language, wrong in the target.
- **Gate:** the first check; low LaBSE similarity ends the diagnosis.
- **τ (tau):** the similarity threshold below which a translation counts as drifted.
- **Token / fragmentation:** a piece of text in the model's vocabulary; fragmentation
  is a word being split into many pieces.
- **R_frag:** target token count divided by source token count.
- **Morphology / suffix / stem:** word structure; a stem plus attached endings.
- **Integrated Gradients (IG):** a method that scores how much each token contributed
  to a prediction.
- **Salience:** the size of a token's attribution.
- **Divergence:** how much less a key term matters in the target than in the source.
- **Null distribution:** the same measure on cases known to be fine, used to set a
  threshold without looking at failures.
- **Audit trail:** the record of every module's verdict for one instance.
- **Unattributed:** no module fired; a reportable outcome.
- **Twin condition:** the same transfer cell with native vs machine-translated target.
