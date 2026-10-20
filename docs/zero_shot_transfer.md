# Zero-shot transfer evaluation, explained from scratch

This guide explains how the project measures what happens when a model trained in
one language is used, unchanged, on another. It assumes you have read (or are happy
to skip) `docs/finetuning_architecture.md`. Everything here is Phase 3 (T-301 to
T-308), and every claim was checked against the code in `src/inference.py`,
`src/predictions.py`, `src/evaluate.py`, `src/transfer.py`, `src/mismatch.py`,
`src/failures.py` and `src/results_tables.py`, plus `configs/eval_conditions.json`
and `reports/transfer_results.md`.

---

## 1. The idea in one paragraph

Train a classifier on Hindi sentences. Then give it Bengali sentences with the same
meaning and the same labels, and **change nothing about the model**: no extra
training, no fine-tuning on Bengali, not one weight updated. Score it. If it did well
on Hindi and badly on Bengali, the drop is the **transfer gap**. "Zero-shot" means
zero Bengali examples were shown to the model before it was tested on Bengali.

---

## 2. Why the word "zero"

| Term | Meaning |
|---|---|
| Fine-tuning | Train on labelled examples in the language you care about |
| Few-shot | Show the model a handful of target-language examples first |
| **Zero-shot** | Show it **no** target-language examples. Use it as it is |

The multilingual encoders (IndicBERT-v2, mBERT) were pretrained on text in many
languages. The hope is that what the classifier learns from Hindi (e.g. which
sentences describe sustainable practice) partly carries over because the encoder puts
similar meanings near each other across languages. Zero-shot evaluation measures how
much of it actually carries over.

---

## 3. Why this corpus makes it measurable

Comparing "Hindi score" with "Bengali score" is normally unfair. The two test sets
contain different sentences, and one may simply be harder. Phase 1 fixes that. For
tasks 2 and 3 the same items exist in every language (human-translated, or
machine-translated into Malayalam), sharing an `item_id` and a gold label. So when
the Hindi and Bengali versions of an item are scored, the **content is held constant**
and only the language changes.

Task 1 (numeral spans) is different: its Hindi, Bengali and Telugu splits are
genuinely different sentences, so it needs no care about overlap, but it is also not a
classification task and is not part of the transfer-gap tables below.

The project states the design intent plainly: the shared items are deliberate. The
point is to isolate "what changing the language costs", not to test generalisation to
unseen content. §8 returns to this.

---

## 4. The pieces, in the order they run

```
frozen checkpoint (trained on Hindi)
        |
        v
  run over a target split          src/inference.py      T-301
        |
        v
  one row per item                 src/predictions.py    T-302
  (gold, predicted, probabilities)
        |
        +--> score it + error bars src/transfer.py       T-304
        |
        +--> compare with the same item in the source    src/mismatch.py   T-306
        |    language -> failed-instance set             src/failures.py   T-307
        v
  tables                           src/results_tables.py T-308
```

`src/evaluate.py` (T-305) is the conductor: it reads the list of conditions and runs
the pieces above for each one.

---

## 5. Step by step

### 5.1 Load the model and freeze it (T-301)

`load_frozen_model` rebuilds the classifier, loads the **best-dev-score weights** that
Phase 2 saved, and then locks it:

```python
model.load_state_dict(checkpoints.best_state_dict(run_id))
model.eval()
for param in model.parameters():
    param.requires_grad_(False)
```

Three separate things are going on:

- `model.eval()` turns off dropout (random zeroing used during training), so the same
  input always gives the same output.
- `requires_grad_(False)` tells PyTorch not to track gradients, so no update could
  happen even by accident.
- No optimiser is created anywhere in the module.

Then `predict` runs under `@torch.no_grad()` and **asserts** the guarantees instead of
just hoping:

```python
assert not model.training, "model must be in eval mode for zero-shot inference"
...
assert not logits.requires_grad, "zero-shot inference must not build a graph"
```

If someone later edits the code so a gradient step sneaks in, the run fails loudly
instead of silently producing numbers that are no longer zero-shot.

### 5.2 Tokenise the target text the same way

The target sentences go through the **same tokenizer and the same `max_len`** as the
training text (`run_inference` takes them from the checkpoint's own `RunConfig`).
Otherwise a difference in preprocessing, not in language, could change the score.

The checkpoint has a sidecar file, `run_config.json`, that records which encoder,
task and split produced it. That is how a checkpoint trained on one split can be
reloaded for inference on a different one.

### 5.3 Predict (what comes out)

For each item the model produces logits, a softmax turns them into probabilities, and
the largest one is the prediction. This is the head from the earlier guide:
`nn.Linear` over the [CLS] vector. `predict` returns `item_id`, `gold`, `pred` and
`probs` for every row. Rows are tied back to their item by `item_id`, not by order.

### 5.4 Write one row per item (T-302)

The result is saved as a **per-instance log**, not just a score. Columns:

```
condition_id, encoder_id, run_id, item_id, block_id, lang, origin,
src_lang, gold, pred, probs, seed
```

Stored at `data/predictions/task_{n}/{condition}.parquet`. Keeping every row matters
because the later failure analysis (Phase 6) needs to know *which* items failed, and
that cannot be rebuilt from an average. Losing the row-level log means re-running the
GPU job.

### 5.5 Score with error bars (T-303, T-304)

The headline metric is **macro-F1** (the average of the per-class F1 scores, so a
rare class counts as much as a common one), with accuracy alongside it. A single score
is not enough, because with a few hundred test items it can shift just from which
items happened to be included. So `bootstrap_metric` measures that wobble:

1. Resample the test items **with replacement** (same size as the original).
2. Recompute the metric on the resample.
3. Repeat 1000 times.
4. The middle 95% of those 1000 scores is the confidence interval.

Each item's `(true, pred)` pair is resampled together. The generator is seeded, so
running it twice gives byte-identical answers.

### 5.6 The transfer gap

```
gap = source score  -  target score        (XTREME's convention)
```

A positive gap means performance was lost by changing language.

Which "source score"? Not the baseline from Phase 2, which was measured on a
different random test fold. The source score here is the model's own accuracy **on
exactly the items it is being compared with** in the target language, paired by
`item_id`. That is a like-for-like comparison: the same model, the same items, two
languages.

### 5.7 Is the gap real? (the noise check)

Training is random, so the same recipe with a different seed gives slightly different
models. Each condition is therefore run for **3 seeds** (0, 1, 2) and reported as
mean ± standard deviation across seeds.

The report flags a gap as **"within seed noise"** when it is smaller than the
seed-to-seed spread. In `src/results_tables.py` that is literally:

```python
"within_seed_noise": abs(gap_mean) < gap_std if len(gaps) > 1 else None
```

Hard rule 5 in the phase spec: a gap smaller than seed variance is not a finding.
(`src/transfer.py` also has a stricter, bootstrap-based test, `within_seed_noise()`
on a `TransferGap`, which asks whether the gap's confidence interval straddles zero.
The tables use the seed-spread version above.)

---

## 6. What is being compared: the conditions

The conditions are not made up at run time. They are read from
`configs/eval_conditions.json` (built in Phase 1, T-113). Three families:

| Family | Count per task | What it asks |
|---|---|---|
| `transfer` | 9 | Train on language A, test on **native** language B text |
| `transfer_mt` | 9 | Same, but test on **machine-translated** B made from A's block |
| `translationese` | 3 | Same language, three provenances: native vs two MT versions |

**Transfer cells.** Hindi, Bengali, Telugu as sources. Each goes to the other two
native languages plus Malayalam: 3 sources × 3 targets = 9. Malayalam only exists as
MT, so for Malayalam targets the evaluation text always comes from a different block
than the one just trained on, and the config records `eval_provenance` to say so.

**Quadrants.** Each transfer cell is tagged by the language families involved. Hindi
and Bengali are Indo-Aryan; Telugu and Malayalam are Dravidian. That gives four
groups: Indo-Aryan→Indo-Aryan, Indo-Aryan→Dravidian, Dravidian→Indo-Aryan,
Dravidian→Dravidian. They answer "does closeness between languages help?".

**Translationese.** `H_nat` vs `Hi←B` vs `Hi←T`: the same language and labels, but one
version was written by people and two were machine-translated. The test is run on the
Hindi model. This separates "the language is different" from "the text is
machine-translated", which is exactly the confound the corpus was built to expose.
For `transfer`, a failure is a transfer failure. For `transfer_mt`, it may also be a
translation artefact. Comparing the two families gives an estimate of how much that
matters.

---

## 7. How `evaluate.py` runs it all

- One checkpoint is named `task{N}_{lang}_{encoder}_seed{S}`, e.g.
  `task2_hin_indicbert_seed0`. It is loaded **once** and reused for every condition
  that needs it.
- Each (condition, seed) pair is planned first. If a checkpoint is missing, the pair is
  reported as **blocked**, not silently skipped.
- A pair that crashes is reported and skipped, and the sweep carries on (the project
  rule: "if it fails, report and continue"). Ctrl+C still stops everything.
- Each arm adds a row to `experiments.csv` with accuracy, macro-F1 and the bootstrap
  interval, plus the prediction log from §5.4.

When Phase 3 was first written, only the Hindi checkpoints existed, so most pairs
were blocked. Bengali and Telugu training (Phase 4) later filled that in, and the
current tables show every one of the 9 cells as `ok` with 3 seeds each.

---

## 8. A design choice that looks like a bug

For tasks 2 and 3, the same sentences exist in all languages. The model is trained on
Hindi sentences (the project's training code takes its own random split of the whole
native Hindi data). It is then tested on the Bengali version of the **same sentences**
(restricted to the `eval` half of the item partition).

This is intentional. The question is "what does switching language cost when the
content is identical?", which is what makes the gap comparable across language pairs
and across encoders. Because both ends of the gap are scored on identical items, the
gap is not distorted by one language getting an easier test set. It is *not* a claim
about how the model handles brand-new content, and it should not be reported as one.

Practical consequence: read a transfer score as "same content, different language", not
as general performance on unseen text.

One related guard is in place. The config's `eval_items: "partition:eval"` makes
`evaluate.py` filter each arm to items in the evaluation half, using a seeded hash of
`item_id`. This is applied in `_item_filter_for`. The translationese conditions use
`eval_items: "all"` on purpose, because there the native arm is the model's own home
text and is labelled as the in-language ceiling (hard rule 4).

---

## 9. The failure set (T-306, T-307)

An overall score tells you how much was lost. It does not tell you what went wrong.
Phase 6 wants specific items to investigate, and it wants to be sure the failures are
about the language and not about difficulty.

The trick is a filter on items the **model got right in the source language**:

```
pred_source == gold   AND   pred_target != gold
```

`join_source_target` pairs the source and target rows on `(item_id, seed)`. The seed is
part of the key on purpose: pairing a seed-0 source prediction with a seed-1 target
prediction would compare two different trained models. It also raises an error if an
item has different gold labels on the two sides, since that would be a corpus problem.

Why filter this way: if the model could not answer the item even in Hindi, the Bengali
failure says nothing about Bengali. It might just be a hard item or a weak model. If the
model answered **this exact sentence correctly one language earlier** and now fails,
item difficulty and general incapacity are ruled out by construction. Whatever changed
between the two arms is what is left to explain it.

`src/failures.py` exports these rows to `data/failures/task_{n}/{condition}.parquet`.
Phase 3 stops there and does **not** say why they failed; assigning causes is Phase 6's
job, with evidence from its linguistic modules.

---

## 10. Reading the results

From `reports/transfer_results.md` (3 seeds, IndicBERT-v2):

**Task 2 (binary, sustainable vs not)**

| condition | source | target | gap (mean ± std) |
|---|---|---|---|
| hin → ben | 0.9535 | 0.5058 | 0.4477 ± 0.0716 |
| hin → tel | 0.9535 | 0.5009 | 0.4526 ± 0.0380 |
| ben → hin | 0.9558 | 0.6927 | 0.2631 ± 0.0530 |
| tel → mal | 0.9573 | 0.5405 | 0.4167 ± 0.0050 |

The model scores about 0.95 macro-F1 in its own language and falls to about 0.50 to
0.70 elsewhere. For a two-class task, 0.5 is roughly what guessing gives, so several
cells are close to no transfer at all. Ben → hin is the best cell. All task-2 gaps are
"not within seed noise".

**Task 3 (10-class headlines)** is lower even in-language (0.25 to 0.60) and the
target scores are about 0.06 to 0.09, close to what random guessing on ten classes would
give. Some Telugu-source cells (gap about 0.17) are flagged "within seed noise" because
the seed-to-seed spread is large, so no conclusion should be drawn about those.

**Translationese (task 2, Hindi model):** native Hindi 0.954; MT Hindi from Bengali
0.885 (−0.069); MT Hindi from Telugu 0.896 (−0.057). Even in the same language, being
machine-translated costs several points.

**Native vs same-source MT targets** land very close to each other in each task. So
whether the target text is real or machine-translated barely changes the size of the
gap.

Remember what these numbers are *not*: they are measurements of transfer on this
corpus, with this encoder. The comparison across encoders and the question of *why*
the transfer fails belong to Phases 5 to 7.

---

## 11. Common ways to misread this

- **"0.50 means half the sentences were right."** Not necessarily. Macro-F1 is not
  accuracy, and with two classes, guessing lands near 0.5.
- **"A small gap means good transfer."** Check the source score first. If the model
  is weak at home (task 3, Telugu source), the gap is small because there is little to
  lose.
- **"The gap is a Dravidian effect."** The quadrant table is a summary, and with 1 to 4
  cells per quadrant the differences should be read with the standard deviations beside
  them.
- **"This tells us the model generalises to new text."** No, see §8.

---

## 12. File map

| What | Where |
|---|---|
| Frozen model loader and `predict` | `src/inference.py` |
| Per-instance prediction log | `src/predictions.py`, `data/predictions/task_{n}/` |
| Conditions (the matrix) | `configs/eval_conditions.json`, built by `src/conditions.py` |
| Run every condition × 3 seeds | `src/evaluate.py`, `scripts/t305_run_conditions.py` |
| Bootstrap CI and gap | `src/transfer.py` |
| Mismatch filter | `src/mismatch.py`, `scripts/t306_mismatches.py` |
| Failed-instance export | `src/failures.py`, `data/failures/task_{n}/` |
| Result tables | `src/results_tables.py`, `reports/transfer_results.md` |
| Phase spec | `CLAUDE3.md`; longer write-up in `docs/phase3.md` |

---

## 13. Glossary

- **Zero-shot**: using a trained model on a language it was never trained on, with no
  further training.
- **Transfer gap**: in-language score minus target-language score.
- **Frozen**: weights locked, eval mode on, no gradients, no optimiser.
- **Condition**: one row of the evaluation matrix, i.e. one train/test pairing.
- **Arm**: one of the test sets inside a condition (translationese has three).
- **Bootstrap**: resampling test items with replacement to measure score uncertainty.
- **Seed noise**: score variation between training runs that differ only in random
  seed.
- **Quadrant**: the family pairing of source and target language (Indo-Aryan or
  Dravidian).
- **Translationese**: the traits of machine-translated text that make it differ from
  text written natively.
- **Mismatch / failed instance**: right in the source language, wrong in the target.
