# How fine-tuning works in this project

A beginner's guide to the model and training code. No prior fine-tuning knowledge assumed.

Everything here is about the classifier that is trained in Phase 2 and reused, frozen, in
Phases 3–7. The code is small on purpose: `src/train.py` (the model and the loop),
`src/data.py` (turning text into numbers), `src/config.py` (settings), `src/metrics.py`
(scoring). This guide walks through all four and explains the ideas behind each line.

For what the project is *for*, read `docs/understanding_the_project.md` first. This guide
is only about the machine-learning mechanics.

> **Source of truth.** Every description here was checked against the code as it stands:
> `src/train.py`, `src/data.py`, `src/config.py`, `src/metrics.py`, `src/inference.py`,
> `src/vram.py` and the YAMLs in `configs/train/`. File names are given so you can check.
> The step counts in §6 are worked illustrations, not logged values.

---

## Contents

1. [The one-sentence idea](#1-the-one-sentence-idea)
2. [Four ideas you need first](#2-four-ideas-you-need-first)
3. [The architecture, piece by piece](#3-the-architecture-piece-by-piece)
4. [From a sentence to numbers the model can eat](#4-from-a-sentence-to-numbers-the-model-can-eat)
5. [One training step, in slow motion](#5-one-training-step-in-slow-motion)
6. [The whole training run](#6-the-whole-training-run)
7. [Every setting in the YAML, explained](#7-every-setting-in-the-yaml-explained)
8. [Why training is repeatable](#8-why-training-is-repeatable)
9. [Why the 4 GB GPU shapes everything](#9-why-the-4-gb-gpu-shapes-everything)
10. [How we score the model](#10-how-we-score-the-model)
11. [After training: freezing and reusing](#11-after-training-freezing-and-reusing)
12. [Reading a training run: healthy, overfit, never started](#12-reading-a-training-run-healthy-overfit-never-started)
13. [What this setup deliberately does not do](#13-what-this-setup-deliberately-does-not-do)
14. [File map and how to extend](#14-file-map-and-how-to-extend)
15. [Glossary](#15-glossary)

---

## 1. The one-sentence idea

**Take a model that already understands language in general, bolt a tiny decision-maker on
top, and train the whole thing a little on your own labelled examples so it learns your
specific task.**

An analogy. Hiring someone who already speaks Hindi fluently and teaching them to sort
sustainability reports into "sustainable" / "unsustainable" takes a few weeks. Teaching a
baby Hindi *and* the sorting job from scratch takes years. Fine-tuning is the first
option. The expensive "learning Hindi" part was done by someone else on huge amounts of
text. That part is called **pretraining**. Your part is the cheap specialisation, and it
is called **fine-tuning**.

In this project:

| Piece | Role in the analogy |
|---|---|
| Pretrained encoder (IndicBERT-v2 or mBERT) | The fluent reader |
| One `nn.Linear` layer on top | The sorting desk: reads the reader's impression and picks a bin |
| Your labelled Hindi sentences | The training examples with the answer written on the back |
| Fine-tuning | Showing examples, checking the answer, nudging everyone's habits |

---

## 2. Four ideas you need first

### 2.1 A model is a big pile of adjustable numbers

A neural network is a function with millions of numbers inside it, called **parameters**
or **weights**. Feed in an input, and arithmetic with those numbers produces an output.
"Learning" means changing the numbers so the outputs get better. Nothing mystical: the
model is a very large calculator whose dials get turned.

- IndicBERT-v2 has about 34 million dials.
- mBERT has about 179 million.

### 2.2 Text must become numbers

A neural network cannot read letters. Text is converted in two steps.

1. **Tokenisation.** A **tokenizer** chops a sentence into pieces called **tokens** and
   looks each piece up in a fixed dictionary to get an integer **token ID**. Tokens are
   often word-fragments. An unfamiliar long word may be several tokens. Each model ships
   its own tokenizer and dictionary, and they are *not* interchangeable.
2. **Embedding.** Inside the model, each token ID is swapped for a list of numbers (a
   vector) that represents its meaning in context. These get refined layer by layer.

### 2.3 The "encoder" and the `[CLS]` token

Both models here are **encoders** of the BERT family. An encoder reads a whole sentence at
once and produces, for **every token**, a vector summarising that token *in the context of
the whole sentence*. So if the sentence has 30 tokens and the vectors are 768 numbers
wide, the encoder outputs a 30 × 768 grid.

The tokenizer adds a special token `[CLS]` ("classification") at the very start. Because
every position can see every other position, the vector at the `[CLS]` position ends up
as a summary of the whole sentence. The classifier simply reads that one vector and
ignores the rest. That is the single most important design fact in this project's model.

### 2.4 Learning from mistakes: loss and gradients

Training is a loop of four moves:

1. **Guess.** Run a batch of sentences through the model, get predictions.
2. **Score the guess.** A **loss function** turns "how wrong was that" into one number.
   Lower is better.
3. **Blame.** Using calculus (**backpropagation**), compute for every dial which direction
   would reduce the loss, and by how much. These per-dial directions are called
   **gradients**.
4. **Nudge.** An **optimizer** moves each dial a small step against its gradient.

Repeat thousands of times and the dials settle into values that give good answers.
Every piece of the code below is one of these four moves, or plumbing around them.

---

## 3. The architecture, piece by piece

The entire model is this class (`src/train.py`):

```python
class Classifier(nn.Module):
    def __init__(self, encoder_hf_id, num_labels):
        self.encoder = AutoModel.from_pretrained(encoder_hf_id)   # pretrained reader
        hidden = self.encoder.config.hidden_size
        self.classifier = nn.Linear(hidden, num_labels)           # new sorting desk

    def forward(self, input_ids, attention_mask):
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        cls = out.last_hidden_state[:, 0]                         # the [CLS] vector
        return self.classifier(cls)                               # one score per class
```

Picture it as a pipeline:

```
 "सुरक्षा के लिहाज से ..."                     raw Hindi sentence
          │  tokenizer
          ▼
 [CLS] ▁सुरक्षा ▁के ▁लिहाज ... [SEP]            token IDs:  [2, 4102, 31, 9120, ..., 3]
          │
          ▼
 ┌───────────────────────────────────────┐
 │  PRETRAINED ENCODER  (many layers)    │   reads all tokens together
 │  IndicBERT-v2  or  mBERT              │
 └───────────────────────────────────────┘
          │  one vector per token:  (length × hidden)
          ▼
   keep ONLY the first one ([CLS])         shape: (hidden,)    e.g. 768 numbers
          │
          ▼
 ┌───────────────────────────────────────┐
 │  nn.Linear(hidden → num_labels)       │   the only brand-new part
 └───────────────────────────────────────┘
          │
          ▼
   logits: [ 1.9, -0.7 ]                   one raw score per class
          │ argmax
          ▼
   prediction: class 0  →  "sustainable"
```

### 3.1 The pieces

**The encoder (pretrained).** Loaded with `AutoModel.from_pretrained(...)`, which
downloads weights someone else trained. It already "knows" Indic-language text. We do not
write it. We only choose which one (`encoder:` in the YAML, via the registry in
`src/data.py`).

**`last_hidden_state[:, 0]`.** `last_hidden_state` is the encoder's final output grid with
shape *(batch, sequence length, hidden)*. `[:, 0]` means "for every sentence in the
batch, take position 0", which is `[CLS]`. Result: shape *(batch, hidden)*.

Note what is **not** done: BERT models usually have an extra "pooler" layer (a dense
layer plus tanh) after `[CLS]`. This project skips it and reads the raw `[CLS]` vector
straight. The comment in the code is explicit that this is intentional.

**`nn.Linear(hidden, num_labels)`.** A single matrix multiply plus a bias. If hidden is 768
and there are 2 classes, it holds a 768 × 2 matrix plus 2 biases = 1,538 numbers. For the
10-class task: 7,690. This is the only layer with random starting values. Everything else
starts from pretrained weights.

**Logits.** The output is one raw score per class, such as `[1.9, -0.7]`. They are not
probabilities yet: they can be negative and do not sum to one. The class with the highest
score is the prediction (`argmax`).

**Softmax (used later, not in training).** To turn logits into probabilities, apply
softmax: `[1.9, -0.7]` becomes roughly `[0.93, 0.07]`. Training does not apply it
explicitly because the loss function folds it in. Inference (`src/inference.py`) applies
it so the per-class probabilities can be logged for every prediction.

### 3.2 Why so simple?

The spec forbids extras: "Do not add layers, pooling strategies, or loss tricks." Reason:
the first job is to **reproduce a published baseline** (IndicFinNLP's monolingual model).
A fancier model would be something other than "the published model, reproduced". The
second reason is fairness: every later comparison (Hindi vs Bengali source, IndicBERT vs
mBERT) is only meaningful if nothing else changes. A minimal model leaves the fewest
places for a hidden difference.

### 3.3 The two encoders are interchangeable

`AutoModel.from_pretrained(hf_id)` loads whichever model the ID names, and every encoder
exposes `config.hidden_size`. So swapping encoders is one YAML line (`encoder:`). The
registry:

| Key | Hugging Face ID | File slug |
|---|---|---|
| `indicbert-v2` | `ai4bharat/indic-bert` | `indicbert` |
| `mbert-base` | `bert-base-multilingual-cased` | `mbert` |
| `xlm-r-base` | `xlm-roberta-base` | `xlmr` (registered, never trained: does not fit in memory) |

How they differ, in plain terms: IndicBERT-v2 is small and specialised for Indian
languages. It saves size by **reusing one layer's weights twelve times**, so it has only
about 7.8M dials in its "thinking" layers (most of its 34M are in the word dictionary).
mBERT is general-purpose, covers about 100 languages, and has about 85.6M dials in its
thinking layers. That difference is an unresolved confound in the project's results (see
`reports/capacity_dilution.md`).

---

## 4. From a sentence to numbers the model can eat

All in `src/data.py`. The journey, in order:

### 4.1 Load a split safely

`load_split(task, block, lang, origin)` reads the frozen corpus (via `corpus_io`) and does
three safety checks before any training can start:

- **Origin check.** Training on machine-translated text would contaminate the transfer
  results, so asking for `origin="native"` on a split that is machine-translated raises
  an error instead of quietly returning nothing.
- **Label range check.** Every `label_id` must be between 0 and (number of classes − 1).
- **Task check.** Task 1 (numbers) has no class label, so it is refused outright.

### 4.2 Split into train, dev and test

The frozen corpus does not say which rows are for training, so `stratified_split()` makes
three groups, **seeded** so it is the same every time:

| Group | Used for | Why it exists |
|---|---|---|
| **train** | the model learns from these | the only rows whose labels change the dials |
| **dev** (validation) | picking when to stop and which version to keep | a "practice exam" used *during* training |
| **test** | the final score you report | a "real exam" the model never influenced |

Why three, not two? If you pick the best epoch by looking at the test score, the test
score is no longer honest: you've tuned to it. Dev picks the epoch; test gets looked at
once at the end.

**Stratified** means each group keeps the same class proportions as the whole. With 10
classes and few rows, a purely random split could leave a class entirely out of test.

Fractions used (from the YAMLs):

| Task | dev | test | train | Why |
|---|---|---|---|---|
| 2 (binary) | 10% | 10% | 80% | plenty of rows |
| 3 (10-class) | 15% | 15% | 70% | only 532 rows; bigger holdouts so every class appears |

### 4.3 Tokenise

`SplitDataset` tokenises every sentence once, up front:

- **Truncation to `max_len`.** Anything longer than `max_len` tokens is cut off. Task 2
  uses 192 (sentences are about 148 characters median, so very little is cut). Task 3
  uses 64 (headlines are short).
- **No padding yet.** Padding is added later, per batch (next).

### 4.4 Batch and pad

Computers are efficient when they process many sentences at once, but a grid must be
rectangular while sentences differ in length. `collate()` takes a batch of sentences and
pads every one with a `[PAD]` token up to the length of the **longest sentence in that
batch** (not up to `max_len`). This matters on a small GPU: padding to the global maximum
would waste memory on empty slots.

It also builds an **attention mask**: 1 for real tokens, 0 for padding. The encoder uses
it to ignore padding, so padding never changes the answer.

Example of one batch of 3 sentences (token IDs invented for illustration):

```
input_ids                          attention_mask
[ 2, 51, 90, 77, 3 ]               [ 1, 1, 1, 1, 1 ]
[ 2, 51, 88,  3, 0 ]               [ 1, 1, 1, 1, 0 ]     0 = [PAD]
[ 2, 12,  3,  0, 0 ]               [ 1, 1, 1, 0, 0 ]
labels = [ 0, 1, 1 ]
```

Each batch the training loop receives is a dictionary of four things: `input_ids`,
`attention_mask`, `labels`, `item_ids` (so a prediction can be traced back to its row).

---

## 5. One training step, in slow motion

This is the inner loop of `train()` in `src/train.py`, simplified.

```python
for batch in train_loader:
    with autocast(...):                                   # (a) half-precision maths
        logits = model(batch["input_ids"], batch["attention_mask"])   # (b) guess
        loss = loss_fn(logits, batch["labels"]) / grad_accum          # (c) score
    scaler.scale(loss).backward()                         # (d) blame
    if time_to_update:
        scaler.step(optimizer)                            # (e) nudge
        scaler.update()
        scheduler.step()                                  # (f) adjust the step size
        optimizer.zero_grad()                             # (g) clear for next time
```

### (b) The guess (forward pass)

The batch flows through the encoder, then the linear layer, producing logits. This is
called the **forward pass**.

### (c) The score: cross-entropy loss

`nn.CrossEntropyLoss` compares the logits to the correct class. Intuition: it asks "how
much probability did the model assign to the right answer?" and penalises low values
heavily.

- Right answer, model 93% sure → small loss (about 0.07).
- Right answer, model only 7% sure → large loss (about 2.7).
- Model is 50/50 on a two-class task → loss about 0.69, which is the "I know nothing"
  level. A training run whose loss never drops below this has not learned.

Internally it applies softmax and then takes the negative log of the correct class's
probability, which is why the model's forward pass outputs raw logits.

### (d) The blame (backward pass)

`.backward()` computes a gradient for every dial in **both** the encoder and the linear
layer. This is important: **all of the model's weights are updated, not only the new
head.** That is what "full fine-tuning" means, and it is why memory is so tight (§9).

### (e) The nudge: AdamW

The optimizer decides how far to move each dial. **AdamW** is the standard choice for
BERT-style models. Two ideas inside it:

- **Adaptive step per dial.** It tracks a running average of each dial's recent gradients
  (and their size) so that dials with noisy gradients move cautiously and steady ones
  move confidently. This costs **two extra numbers per dial** held in memory.
- **Weight decay.** A gentle pull of every dial toward zero each step, to discourage the
  model from relying on a few huge weights. The code leaves biases and LayerNorm
  weights out of this pull (`_param_groups`), which is the standard convention for BERT
  fine-tuning.

### (f) The step size over time: warmup then decay

The **learning rate** is how big each nudge is (here about 0.00002–0.00003, deliberately
tiny because the pretrained weights are valuable and large nudges would wreck them).
It is not constant:

```
 learning rate
   ▲
 peak │        /\
      │       /  ‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾\
      │      /                    ‾‾‾‾‾‾‾‾‾‾‾\
      │     /                                  ‾‾‾‾\
    0 └────/──────────────────────────────────────────▶ steps
        warmup (first 10%)      linear decay to 0
```

- **Warmup (first 10% of steps):** start near zero and ramp up. At the start, the new
  linear layer is random, so its gradients are nonsense. Large early steps would send
  nonsense signals into the pretrained encoder and damage it. Warmup lets the head find
  its footing first.
- **Linear decay:** gradually shrink the step so the model settles instead of bouncing
  around the best spot. Implemented by `get_linear_schedule_with_warmup`.

### (a) Mixed precision (`fp16: true`)

Normally numbers are stored in 32 bits. **Mixed precision** does most of the arithmetic in
16 bits, which is faster and uses less memory. Risk: 16-bit numbers can't represent very
small values, so tiny gradients become zero ("underflow"). The fix is a **GradScaler**:
multiply the loss by a big number before the backward pass so gradients stay
representable, then divide it back out before the nudge. If an overflow occurs, the
scaler skips that step and reduces its multiplier. The master copy of the weights stays
32-bit regardless, which matters for memory (§9).

### (c, again) Gradient accumulation

If the GPU can't hold a batch of 16 sentences, run two batches of 8, add up their
gradients, and only then nudge. The code divides the loss by `grad_accum` so the sum
equals the average a batch of 16 would have given. The result is mathematically the same
as one batch of 16 (up to dropout randomness and floating-point order). Parameter
`batch_size × grad_accum` is the **effective batch size**, and it is what actually
matters for results. This is how mBERT trains on task 2 (8 × 2 = 16) while IndicBERT uses
16 × 1.

---

## 6. The whole training run

One pass through all training sentences is an **epoch**. A run is many epochs. At the end
of each, the loop evaluates, records, saves, and decides whether to continue.

```
for each epoch:
    train on every batch in the train set               (§5)
    measure macro-F1 on the TRAIN set   (model.eval())  ← "is it learning at all?"
    measure macro-F1 on the DEV set     (model.eval())  ← "is it generalising?"
    if dev macro-F1 is a new best:
        remember these weights as "best so far"; reset the patience counter
    else:
        patience counter += 1
    save a checkpoint (atomically)                       ← so a crash loses little
    if patience counter ≥ patience:  stop early
after the loop: put the BEST weights back into the model, not the last ones
```

Details worth understanding:

**Train mode vs eval mode.** `model.train()` turns on **dropout** (randomly switching off
parts of the network during training so it can't memorise too rigidly). `model.eval()`
turns it off so measurements are stable. Evaluation also runs under `torch.no_grad()`: no
gradient bookkeeping, less memory.

**Early stopping.** Training longer is not always better. After some point the model starts
**overfitting**: memorising the training sentences, getting better on them while getting
*worse* on new ones. The dev score rises, peaks, then flatlines or falls. `patience` is
how many epochs in a row with no new best the loop tolerates before stopping.

**Restoring the best weights.** The final epoch is usually *not* the best one. The loop
keeps a CPU copy of the best-scoring weights and loads them back before returning. The
model you get is the best-on-dev model.

**Why measure train F1 every epoch?** It's cheap and diagnostic. A model that never
reaches a decent training score hasn't learned anything, regardless of what dev says (see
§12). It's how the project spots the "never trained" runs.

### A worked sizing example (illustrative)

Task 2, Hindi: about 2,238 sentences, 80% train ≈ 1,790. At effective batch 16 that is
about 112 updates per epoch. With `epochs: 20`, that's at most about 2,240 updates, of
which the first 10% (about 224) are warmup. Task 3, Hindi: 532 sentences, 70% train ≈
372, so about 24 updates per epoch, up to 40 epochs, about 960 updates, about 96 of them
warmup. Both are tiny by deep-learning standards, which is why a run takes minutes, not
hours (task 2 takes roughly 30 seconds an epoch on IndicBERT-v2).

---

## 7. Every setting in the YAML, explained

Take `configs/train/task2_hin_indicbert.yaml`. The loader (`src/config.py`) reads it into
a frozen `RunConfig` and **rejects unknown keys** so a typo cannot be silently ignored.

| Key | Value (task 2) | Meaning | Why this value |
|---|---|---|---|
| `encoder` | `indicbert-v2` | which pretrained model | the experiment variable |
| `task` | `2` | which dataset/labels | 1 is refused |
| `lang` | `hin` | which native language to train on | the other experiment variable |
| `origin` | `native` | human-written only | MT is evaluation data; training on it would contaminate transfer results |
| `max_len` | `192` | truncate longer sentences | median ~148 chars; raised from 128 so long sentences aren't clipped |
| `batch_size` | `16` | sentences per forward pass | what fits in memory |
| `grad_accum` | `1` | batches summed before nudging | `>1` when memory is tight |
| `lr` | `2.0e-5` | peak learning rate | standard small value for fine-tuning BERT |
| `weight_decay` | `0.01` | gentle pull toward zero | standard |
| `epochs` | `20` | maximum passes | upper limit; early stopping usually ends sooner |
| `warmup_ratio` | `0.1` | fraction of steps spent ramping up | standard |
| `patience` | `5` | epochs without improvement before stopping | raised from 3, which stopped some seeds too early |
| `seed` | `0` | randomness seed | the sweep overrides it with 0, 1, 2 |
| `fp16` | `true` | mixed precision | saves memory and time |
| `dev_fraction` / `test_fraction` | `0.1` / `0.1` | holdout sizes | see §4.2 |

Why task 3 differs (`task3_hin_indicbert.yaml`): `max_len: 64` (headlines are short),
`lr: 3e-5` and `epochs: 40`, `patience: 8` (with only ~370 training rows over 10 classes
it needs a longer, slightly hotter schedule, because the first attempt was still
improving when it stopped), and bigger holdouts (15%/15%).

**Two rules about changing settings.**

1. **No per-encoder or per-language tuning.** The Bengali and Telugu configs are
   field-for-field copies of the Hindi ones except `lang`. If you tune one, you must tune
   all, otherwise a comparison measures your tuning effort rather than the thing you
   wanted to compare. A test enforces this (`check_block_parity`, in the T-401 smoke script).
2. **The only permitted deviation is what memory forces.** mBERT uses `batch_size 8,
   grad_accum 2` where IndicBERT uses `16, 1`. The effective batch is identical, which is
   why this is allowed. The tests permit `batch_size` and `grad_accum` to differ only
   when their product doesn't.

Seeds are not separate files. `load_run_config(path, seed=1)` overrides the field, so one
YAML drives three seeds.

---

## 8. Why training is repeatable

Fine-tuning involves randomness: the head's starting weights, the order sentences are
shuffled, which units dropout switches off. The project demands that **the same config and
seed give identical metrics**.

How it's achieved:

- **`seed_everything()`** pins Python's, NumPy's and PyTorch's generators (CPU and GPU) and
  forces deterministic GPU algorithms.
- **Seed before building the model.** The linear head is randomly initialised. A test
  caught that if the seed isn't set *before* constructing it, two runs start from
  different head weights and diverge from the very first epoch.
- **A seeded data-loader generator** fixes the shuffle order.
- **Seeded splits.** `stratified_split` uses the seed too.

### Why three seeds

Even with everything controlled, a different seed gives a different (equally legitimate)
run. Training a model once and reporting one number would be like running one race. Three
seeds give a mean and a spread (± standard deviation). A difference smaller than that
spread is noise, not a finding.

### Config hashing

Two hashes are computed:

- `TrainConfig.hash()`: the hyperparameters only.
- `RunConfig.hash()`: everything, including which split and the dev/test fractions
  (because a *score* also depends on which rows were held out).

Each is a short fingerprint of the settings. The run hash is stamped into every row of
`experiments.csv` (the run log) and into the checkpoint, so any number can be traced to
the exact settings that produced it.

### Checkpoints and resuming

GPU access is intermittent (a laptop GPU shared with the desktop), so a run must survive
interruption. After every epoch `train()` writes `checkpoints/<run_id>/checkpoint.pt`
containing:

| Saved | Why |
|---|---|
| model weights | the current state |
| optimizer state (the AdamW averages) | otherwise resuming would reset its "memory" |
| scheduler state | to resume at the right point on the learning-rate curve |
| scaler state | the mixed-precision multiplier |
| all random-number-generator states | so shuffles continue identically |
| best-so-far weights, patience counter, history | so early stopping continues correctly |
| config hash | a checkpoint from a *different* config is refused |

The write is **atomic**: save to a temporary file, then rename, so a crash mid-write can't
corrupt the resume point. `run_id` is `<config-stem>_seed<N>`, for example
`task2_hin_indicbert_seed1`; `scripts/t208_checkpoints.py` can list, show and prune them.
`checkpoints/` is gitignored because it can be regenerated.

---

## 9. Why the 4 GB GPU shapes everything

The machine has a laptop GPU with 4 GB, of which the desktop uses about 1 GB. This is the
most constraining fact in the project, and it explains several odd decisions.

### Where the memory goes

Training holds, for **every** parameter:

| Item | Bytes per parameter |
|---|---|
| the weight (kept in 32-bit even under mixed precision) | 4 |
| its gradient | 4 |
| AdamW's two running averages | 8 |
| **fixed total** | **16** |

On top of that sit the **activations** (the intermediate results of the forward pass,
needed again by the backward pass), which grow with batch size and sentence length.

| Encoder | Parameters | Fixed cost | Headroom on a 3.68 GiB card |
|---|---|---|---|
| IndicBERT-v2 | 34M | 0.50 GiB | +3.18 GiB |
| mBERT | 179M | 2.65 GiB | +1.03 GiB |
| XLM-R base | 279M | **4.14 GiB** | **−0.46 GiB** |

### What follows

- **XLM-R cannot be fine-tuned at all.** The fixed cost alone exceeds the card. No smaller
  batch, shorter sentence or trick of that kind helps, because the cost is in the weights
  and optimizer, not the data. One piece of it: its dictionary has about 250,000 entries,
  so just the embedding table in 32-bit is a 734 MiB allocation. Its configs are kept in
  `configs/train/deferred/` and the arithmetic in the README there.
- **mBERT needs gradient accumulation** on task 2 (batch 8 × 2 instead of 16 × 1), because
  with only ~1 GiB of headroom, 16 long sentences' activations won't fit.
- **Dynamic padding** (§4.4) and `fp16` exist for the same reason.
- **`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`** is set at the top of
  `train.py`, before PyTorch loads. Without it, memory gets fragmented into unusable
  scraps and runs fail even when enough total memory is free.
- **Adapters were rejected** as a way round the XLM-R wall: they'd freeze most weights and
  train only small add-ons. That fits in memory but changes *how the model was trained*,
  so XLM-R's results could no longer be compared fairly with the other two.

`src/vram.py` and `scripts/t205_vram.py` measure all this directly by running one real
forward-backward-update step at a given batch size and length and recording the peak.
A bug found along the way: the probe's mixed-precision scaler skipped the optimizer step
on synthetic data, so AdamW's buffers were never allocated and the table said XLM-R fit
when it did not. The probe now forces that allocation.

---

## 10. How we score the model

`src/metrics.py`. Primary metric: **macro-F1**.

### Accuracy and why it isn't enough

Accuracy is "fraction correct". On an imbalanced task it flatters useless models: if 55%
of sentences are "sustainable", a model that always says "sustainable" gets 55% accuracy
while understanding nothing.

### F1 for one class

For one class, **precision** is "of the sentences I called this class, how many really
were?" and **recall** is "of the sentences that really were this class, how many did I
find?". **F1** is a balance of the two, high only when both are high.

### Macro averaging

**Macro-F1** computes F1 separately for each class and takes the plain average, so every
class counts equally regardless of size. A model that ignores a rare class gets that
class's F1 of 0, dragging the average down. That's the property wanted here. On task 3
(10 classes, 21–92 rows per class) it matters a lot.

Reference points:

| Situation | Macro-F1 |
|---|---|
| Always predicts one class, 2-class task, 55/45 split | about 0.35 |
| Always predicts one class, 10-class task | about 0.02–0.1 |
| Perfect | 1.0 |

The function also returns per-class F1 and the **confusion matrix** (a grid: rows are true
class, columns predicted class, cells are counts). `labels=range(num_labels)` is passed
explicitly so a class missing from a small split still gets its row, scored as zero,
instead of silently shrinking the average.

---

## 11. After training: freezing and reusing

Fine-tuning produces **checkpoints**: files of weights. Later phases never train again.
They use `src/inference.py`:

```python
model.load_state_dict(best_weights)        # the best-on-dev epoch
model.eval()                               # no dropout
for p in model.parameters():
    p.requires_grad_(False)                # nothing can be updated
...
with torch.no_grad():                      # no gradient graph is built
    logits = model(...)
    assert not logits.requires_grad        # checked, not just assumed
```

This is **zero-shot transfer**: the Hindi-trained model is applied unchanged to Bengali,
Telugu or Malayalam sentences, which it has never been trained on. There is no optimizer
in this module at all, and `predict()` asserts the model is in eval mode and builds no
gradient graph. Those assertions are what turn "we didn't update the model" from a promise
into a checked fact.

Important details that make the transfer meaningful:

- **The same tokenizer and `max_len` the model was trained with** are used on target text
  (read back from the checkpoint's `run_config.json` sidecar). A Bengali sentence is
  tokenized by the same encoder's tokenizer, into pieces that the encoder may have seen
  in pretraining even if the classifier head never saw them.
- **Every prediction is logged**, per sentence: `item_id`, gold label, prediction, the
  probability for each class, and seed. This log is what Phase 6 diagnoses.
- **The "transfer gap"** is the model's score on its own language minus its score on the
  other language, on the same items.

Why does it work at all? Because the pretrained encoder places similar meanings from
different languages at similar positions in its vector space. The classifier head learned
to read that space from Hindi examples. If Bengali sentences land in similar places, the
same head works. How well that holds is the research question.

---

## 12. Reading a training run: healthy, overfit, never started

The loop prints one line per epoch:
`epoch 7: loss 0.1834 train_f1 0.9531 dev_f1 0.8410`.
Learn to read three patterns. All three occur in this project's results
(`reports/baselines.md`).

### Healthy (task 2, any seed)

```
epoch  0: loss 0.69  train_f1 0.52  dev_f1 0.55     near-chance start
epoch  4: loss 0.35  train_f1 0.85  dev_f1 0.80     both rising
epoch 10: loss 0.12  train_f1 0.97  dev_f1 0.86     dev peaks
epoch 15: loss 0.05  train_f1 0.99  dev_f1 0.85     dev flat/slipping → patience counts
→ early stop, restore epoch-10 weights
```

(Numbers invented to show the shape. Real task-2 runs reach train F1 near 1.0 and dev
F1 about 0.86–0.90.)

### Overfit (task 3, mostly)

Train F1 hits about 0.99 while dev and test sit around 0.15–0.20. The model has memorised
about 370 examples across 10 classes and cannot generalise. More epochs, regularisation or
a bigger model will not fix it: the limit is the *amount of data*. This is called a data
ceiling, and the published baseline for this task is itself only 0.05.

### Never started ("did not train")

Train F1 stays low (peak around 0.15), so the optimisation never got going. Some
IndicBERT-v2 task-3 runs do this. On a task at its data ceiling their *test* score
looks no worse than a healthy run's, which is why the project checks the **train** score
for each run (`src/convergence.py`, threshold: peak train macro-F1 ≥ 0.5). These runs are
kept and reported, not deleted, because their instability is part of the finding. All
twelve mBERT runs trained fine on identical data and schedule, so it's a property of the
smaller encoder, not a flaw in the settings.

### Quick diagnostic table

| You see | Likely meaning |
|---|---|
| Loss stuck near 0.69 (2 classes) | not learning; learning rate, data or wiring problem |
| Train high, dev high | healthy |
| Train high, dev low | overfitting / too little data |
| Train low, dev low | underfitting or never started |
| Dev peaks early then falls | overfitting; early stopping working as designed |
| Dev still rising at the last epoch | stopped too soon; raise `epochs` or `patience` |

---

## 13. What this setup deliberately does not do

Knowing what's absent prevents wrong assumptions:

- **No frozen layers.** Everything is trained, not just the head. (Hence the memory wall.)
- **No class weighting, no focal loss, no label smoothing.** Plain cross-entropy.
- **No data augmentation.** The published paper used paraphrase augmentation on task 3
  (to 4,774 rows); this project does not.
- **No learning-rate search.** One value per task, reused across languages and encoders.
- **No ensembling.** Three seeds are *separate* runs reported as mean ± std, not combined.
- **No training on machine-translated text.** Ever. Translated text is test data only.
- **No training for Malayalam.** There is no native Malayalam, so it is only ever a
  target.
- **Task 1 is not trained.** It's a number-marking task, not classification.

One design choice that surprises people and is intentional: a model trained on one
language is evaluated on other-language versions of the *same underlying sentences* it
trained on, because the whole point is to hold content constant and change only the
language. This measures how much changing language costs, not how well the model handles
unseen content.

---

## 14. File map and how to extend

| Concern | File | Key names |
|---|---|---|
| Model and loop | `src/train.py` | `Classifier`, `TrainConfig`, `train`, `build_model`, `evaluate`, `seed_everything`, `overfit_subset` |
| Data | `src/data.py` | `ENCODERS`, `load_split`, `stratified_split`, `SplitDataset`, `get_tokenizer` |
| Settings | `src/config.py` | `RunConfig`, `load_run_config`, `run_training` |
| Metrics | `src/metrics.py` | `classification_metrics` |
| Run log | `src/experiments.py` | appends to `experiments.csv` |
| Checkpoints | `src/checkpoints.py` | `checkpoint_path`, `best_state_dict` |
| Memory | `src/vram.py` | `probe_step`, `fixed_floor_bytes` |
| Frozen reuse | `src/inference.py` | `load_frozen_model`, `predict` |
| Convergence check | `src/convergence.py` | "did it train?" |
| Sweep driver | `scripts/t206_baseline.py` | loops configs × seeds, resumable |
| Smoke test | `scripts/t203_smoke.py` | overfit 48 rows to prove wiring |
| Settings files | `configs/train/*.yaml` | one per task × language × encoder |

### The smoke test is worth understanding

`overfit_subset` trains on 48 class-balanced rows for 30 epochs and requires train F1 >
0.95. Logic: a model that can't memorise 48 examples has a bug (wrong pooling, frozen
encoder, label mismatch). It's the cheapest way to check the wiring, run before spending
GPU hours. The rows are class-balanced because macro-F1 on a one-class sample can't reach
1.0 however well the model fits.

### To add a new language or encoder

The design goal is that this is configuration, not code.

- **New source language:** copy an existing YAML, change only `lang`. (The parity test
  rejects any other change.)
- **New encoder:** add an entry to `ENCODERS` in `src/data.py` and a YAML with only
  `encoder` changed, then check memory first with `python -m scripts.t205_vram`.

When adding an encoder, the smoke audit (`scripts/t500_smoke.py`) once found five hidden
hardcodings of "indicbert" in names and paths. If something breaks, look for another.

### To train a baseline

Long jobs are run by the project owner, not launched unattended.

```bash
python -m src.env_check                       # Python 3.11, libraries, GPU visible
python -m scripts.t203_smoke --device cuda    # prove the wiring (minutes)
python -m scripts.t205_vram                   # check what fits
python -m scripts.t206_baseline               # train configs × seeds, resumable
python -m scripts.t207_gate                   # compare with the published numbers
```

Set `HF_HUB_OFFLINE=1` once models are cached to avoid network failures mid-run.

---

## 15. Glossary

| Term | Plain meaning |
|---|---|
| **Parameter / weight** | One adjustable number inside the model |
| **Pretraining** | The expensive general learning done by others on huge text |
| **Fine-tuning** | The cheap specialisation on your labelled data |
| **Encoder** | A model that reads a whole sentence and outputs a vector per token |
| **Token / token ID** | A piece of text / its integer in the model's dictionary |
| **Embedding** | A list of numbers representing a token's meaning |
| **`[CLS]`** | Special first token whose output vector summarises the sentence |
| **Head** | The small new layer (`nn.Linear`) that turns that summary into class scores |
| **Logits** | Raw per-class scores before softmax |
| **Softmax** | Converts logits into probabilities that sum to 1 |
| **Loss (cross-entropy)** | A number saying how wrong the guess was |
| **Gradient / backpropagation** | Per-weight "which way to change" signals, and the method that computes them |
| **Optimizer (AdamW)** | The rule that moves weights using the gradients |
| **Learning rate** | How big each nudge is |
| **Warmup / decay** | Ramp the learning rate up then down over the run |
| **Weight decay** | A gentle pull of weights toward zero to resist memorising |
| **Dropout** | Randomly switching off units during training for robustness |
| **Batch / epoch** | A group of sentences processed together / one pass over all training data |
| **Effective batch** | `batch_size × grad_accum` |
| **Gradient accumulation** | Summing gradients over several small batches before one update |
| **Mixed precision (fp16)** | Mostly 16-bit arithmetic for speed and memory |
| **Attention mask** | 1/0 marker telling the model which positions are real vs padding |
| **Train / dev / test** | Learn from / choose when to stop / final honest exam |
| **Stratified** | Keeping class proportions equal across the splits |
| **Early stopping / patience** | Stop when dev stops improving for this many epochs |
| **Overfitting** | Memorising training data at the expense of new data |
| **Macro-F1** | F1 averaged equally over classes |
| **Seed** | Number that fixes all random choices |
| **Checkpoint** | A saved snapshot of weights plus everything needed to resume |
| **Zero-shot** | Using the trained model on a new language with no further training |
| **Frozen** | Weights fixed; nothing can be updated |
| **Config hash** | Fingerprint of all settings that affect a result |
