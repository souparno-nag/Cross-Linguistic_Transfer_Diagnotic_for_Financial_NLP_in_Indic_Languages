"""T-113 — the evaluation-condition matrix, and its validator.

A *condition* names one training split, one evaluation split, and the items
each may use. Writing it down is not bookkeeping: §2.2 established that tasks 2
and 3 hold the **same items in every language**, so the obvious zero-shot setup
— train on the Hindi split, evaluate on the Malayalam one — would be testing
the model on sentences it had already seen, in translation. Every transfer
number in the project depends on that not happening.

Two distinct protections, and they catch different mistakes:

* **Split-level.** A split must not be both training source and evaluation
  target within one condition. This is §8's stated requirement.
* **Item-level.** Training and evaluation item sets must be disjoint. For tasks
  2 and 3 the split-level check alone passes happily while every item overlaps,
  so the items are partitioned by a seeded hash of `item_id` — deterministic
  (§4 rule 7), independent of row order, and identical across languages, so the
  same item lands in the same half in all four.

Task 1 needs no partition: its blocks hold genuinely different sentences
(§2.2), so a condition training on block H and evaluating on block B shares
nothing to begin with. It is the one task where zero-shot transfer can be
measured without carving the data up.
"""

from __future__ import annotations

import hashlib

from .corpus_io import is_numeral_task, read_split
from .ids import BLOCK_NATIVE_LANG, CORPUS_LANGS, targets_for_block

FAMILY = {"hin": "Indo-Aryan", "ben": "Indo-Aryan", "tel": "Dravidian", "mal": "Dravidian"}
TRAIN_SHARE = 0.5
DEFAULT_SEED = 20260903


def quadrant(src_lang: str, tgt_lang: str) -> str:
    return f"{FAMILY[src_lang]}->{FAMILY[tgt_lang]}"


def partition_of(item_id: str, seed: int = DEFAULT_SEED) -> str:
    """Which half an item belongs to: `train` or `eval`.

    Hashed rather than sliced by position, for the same reason `item_id` itself
    is content-derived (§6): a positional split silently re-points at different
    items if anything upstream is reordered. Hashing the id means the partition
    is a property of the item, so every language agrees on it without having to
    coordinate.
    """
    digest = hashlib.sha256(f"{seed}:{item_id}".encode("utf-8")).hexdigest()
    return "train" if int(digest[:8], 16) / 0xFFFFFFFF < TRAIN_SHARE else "eval"


def split_name(task: int, block: str, lang: str) -> str:
    return f"task_{task}/{block}/{lang}"


def describe_split(task: int, block: str, lang: str) -> dict:
    frame = read_split(task, block, lang)
    origin = sorted(set(frame["origin"]))
    return {
        "split": split_name(task, block, lang),
        "block": block,
        "lang": lang,
        "origin": origin[0] if len(origin) == 1 else "mixed",
        "src_lang": (
            None if origin == ["native"] else sorted(set(frame["src_lang"].dropna()))[0]
        ),
        "rows": len(frame),
    }


def evaluation_split(task: int, train_block: str, target: str) -> tuple[str, str, str]:
    """Where a condition's evaluation data comes from, and why.

    Two rules, in order:

    1. **Prefer a native evaluation split.** Hindi, Bengali and Telugu each have
       one. Evaluating on human-written text means a failure is a *transfer*
       failure; evaluate on MT output and you cannot tell transfer failure from
       translation failure, which is the confound this whole corpus exists to
       separate.
    2. **Otherwise take MT from a different block than the one trained on.**
       Malayalam has no native split anywhere (§2), so its evaluation data is
       always machine translated — but it need not come from the same block the
       model trained on, and for task 1 a different block means genuinely
       different sentences.

    Returns `(split, provenance, why)` where provenance is the language the
    evaluation text was written or translated from — an axis the translationese
    conditions isolate separately, and one that has to be declared rather than
    buried.
    """
    native_block = {v: k for k, v in BLOCK_NATIVE_LANG.items()}.get(target)
    if native_block is not None:
        return (
            split_name(task, native_block, target),
            "native",
            "human-written evaluation text, so a failure is transfer failure and "
            "not translation failure",
        )
    block = next(b for b in sorted(BLOCK_NATIVE_LANG) if b != train_block)
    return (
        split_name(task, block, target),
        BLOCK_NATIVE_LANG[block],
        f"no native {target} exists (§2), so evaluation uses MT from block {block} "
        f"rather than from the block just trained on",
    )


def transfer_conditions(task: int, seed: int = DEFAULT_SEED) -> list[dict]:
    """The 9 transfer cells: train in one language, evaluate in another.

    Training always uses a native split. A model trained on machine translation
    measures the MT system as much as the transfer, and the point of the cell is
    the transfer.
    """
    out = []
    partitioned = not is_numeral_task(task)
    for block, native in BLOCK_NATIVE_LANG.items():
        for target in targets_for_block(block):
            eval_split, provenance, why = evaluation_split(task, block, target)
            note = why
            if partitioned:
                note += (
                    ". Every split of this task holds the same items (§2.2), so the "
                    "seeded item partition is what keeps training and evaluation "
                    "disjoint"
                )
            else:
                note += (
                    ". Task 1's blocks hold different sentences (§2.2), so the two "
                    "splits share no items and no partition is needed"
                )
            out.append(
                {
                    "name": f"transfer_{native}_to_{target}",
                    "kind": "transfer",
                    "quadrant": quadrant(native, target),
                    "train": split_name(task, block, native),
                    "eval": eval_split,
                    "eval_provenance": provenance,
                    "train_items": "partition:train" if partitioned else "all",
                    "eval_items": "partition:eval" if partitioned else "all",
                    "note": note,
                }
            )
    return out


def translationese_conditions(task: int) -> list[dict]:
    """Same language, same labels, different provenance (§8).

    Hindi exists as a native split and as two machine translations, from
    Bengali and from Telugu. Evaluating one model on all three isolates the
    effect of the text having been translated at all, holding language and
    label constant — which is the whole point of building the corpus this way.
    """
    out = []
    for lang in ("hin", "ben", "tel"):
        native_block = {v: k for k, v in BLOCK_NATIVE_LANG.items()}[lang]
        arms = [split_name(task, native_block, lang)]
        arms += [
            split_name(task, block, lang)
            for block in BLOCK_NATIVE_LANG
            if block != native_block
        ]
        out.append(
            {
                "name": f"translationese_{lang}",
                "kind": "translationese",
                "quadrant": None,
                "train": None,
                "eval": arms,
                "train_items": None,
                "eval_items": "all",
                "note": (
                    f"Three versions of {lang}: one written by people, two machine "
                    "translated from the other two languages. Same language, same "
                    "labels, differing only in provenance."
                ),
            }
        )
    return out


def build(task: int, seed: int = DEFAULT_SEED) -> dict:
    conditions = transfer_conditions(task, seed) + translationese_conditions(task)
    return {
        "task": task,
        "seed": seed,
        "partitioned": not is_numeral_task(task),
        "train_share": TRAIN_SHARE,
        "languages": list(CORPUS_LANGS),
        "conditions": conditions,
    }


def validate(matrix: dict, item_sets: dict[str, set] | None = None) -> list[str]:
    """Every way this matrix could quietly leak, checked. Returns failures."""
    failures = []
    task = matrix["task"]

    for condition in matrix["conditions"]:
        train, evaluation = condition["train"], condition["eval"]
        if train is None:
            continue
        targets = evaluation if isinstance(evaluation, list) else [evaluation]
        if train in targets:
            failures.append(
                f"{condition['name']}: {train} is both training source and "
                "evaluation target"
            )
        if item_sets is not None and condition["train_items"] == "partition:train":
            for target in targets:
                overlap = {
                    item
                    for item in item_sets.get(train, set())
                    if partition_of(item, matrix["seed"]) == "train"
                } & {
                    item
                    for item in item_sets.get(target, set())
                    if partition_of(item, matrix["seed"]) == "eval"
                }
                if overlap:
                    failures.append(
                        f"{condition['name']}: {len(overlap)} items appear in both "
                        f"halves, e.g. {sorted(overlap)[:3]}"
                    )
        elif item_sets is not None and condition["train_items"] == "all":
            for target in targets:
                shared = item_sets.get(train, set()) & item_sets.get(target, set())
                if shared:
                    failures.append(
                        f"{condition['name']}: unpartitioned condition shares "
                        f"{len(shared)} items between {train} and {target}"
                    )

    transfer = [c for c in matrix["conditions"] if c["kind"] == "transfer"]
    if len(transfer) != 9:
        failures.append(f"{len(transfer)} transfer cells, expected all 9 (§8)")
    quadrants = {c["quadrant"] for c in transfer}
    expected_quadrants = {
        "Indo-Aryan->Indo-Aryan", "Indo-Aryan->Dravidian",
        "Dravidian->Indo-Aryan", "Dravidian->Dravidian",
    }
    if quadrants != expected_quadrants:
        failures.append(f"quadrants covered {sorted(quadrants)}, expected all four")
    translationese = [c for c in matrix["conditions"] if c["kind"] == "translationese"]
    if len(translationese) != 3:
        failures.append(
            f"{len(translationese)} translationese conditions, expected one each for "
            "Hindi, Bengali and Telugu (§8: non-optional)"
        )
    for condition in translationese:
        if len(condition["eval"]) != 3:
            failures.append(
                f"{condition['name']}: {len(condition['eval'])} arms, expected a "
                "native version and two machine-translated ones"
            )
    del task
    return failures
