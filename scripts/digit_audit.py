"""Digit-system variation in frozen tasks 2 and 3, and among their failures.

Run from the repo root:  python -m scripts.digit_audit

Read-only; reads data/v1.0 (via src.corpus_io) and data/failures. Writes nothing.

Per (source text, MT/target text) pair the digit systems present in each are
compared. Classes:
  none       no digits in either text
  same       both have digits, same set of systems
  converted  both have digits, different systems     <- an ortho mismatch
  dropped    source has digits, target has none      <- numeral loss, not ortho
  gained     target has digits, source has none

Currency is deliberately not counted: the earlier substring match fired inside
ordinary words, so it was not measuring currency.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path.cwd()))

import pandas as pd

from src.corpus_io import read_split
from src.unicode_ranges import digit_counts

BLOCK_SRC = {"H": "hin", "B": "ben", "T": "tel"}
LANGS = ["hin", "ben", "tel", "mal"]
CLASSES = ["none", "same", "converted", "dropped", "gained"]
pd.set_option("display.width", 220)


def systems(text: str) -> frozenset:
    return frozenset(s for s, n in digit_counts(text).items() if n)


def classify(src: str, tgt: str) -> str:
    s, t = systems(src), systems(tgt)
    if not s and not t:
        return "none"
    if not t:
        return "dropped"
    if not s:
        return "gained"
    return "same" if s == t else "converted"


def label(sys_set: frozenset) -> str:
    return "+".join(sorted(sys_set)) if sys_set else "-"


def text_map(task, block, lang):
    f = read_split(task, block, lang, frozen=True)
    return dict(zip(f["item_id"], f["text"]))


# ---- A. Which digit systems do the native texts use? -----------------------
print("A. Native splits: digit systems per row (rows with any digit)")
rows = []
for task in (2, 3):
    for block, lang in BLOCK_SRC.items():
        sets = [systems(t) for t in text_map(task, block, lang).values()]
        with_digits = [s for s in sets if s]
        rows.append({
            "task": task, "split": f"{block}/{lang}", "rows": len(sets),
            "with_digits": len(with_digits),
            "ASCII_only": sum(s == {"ASCII"} for s in with_digits),
            "native_script_only": sum(bool(s) and "ASCII" not in s for s in with_digits),
            "mixed": sum("ASCII" in s and len(s) > 1 for s in with_digits),
        })
print(pd.DataFrame(rows).to_string(index=False))

# ---- B. Source -> MT, every row of every direction -------------------------
print("\nB. Source -> MT pair classes, all rows")
rows = []
for task in (2, 3):
    for block, src_lang in BLOCK_SRC.items():
        src = text_map(task, block, src_lang)
        for lang in LANGS:
            if lang == src_lang:
                continue
            mt = text_map(task, block, lang)
            counts = pd.Series([classify(src[i], mt[i]) for i in mt if i in src]).value_counts()
            rows.append({"task": task, "dir": f"{src_lang}->{lang}",
                         **{c: int(counts.get(c, 0)) for c in CLASSES}})
b = pd.DataFrame(rows)
print(b.to_string(index=False))
print(b.groupby("task")[CLASSES].sum().to_string())

# ---- C. Failures ------------------------------------------------------------
print("\nC. Failure sets: source (train-side) text vs the failing target text")
matrix = json.load(open("configs/eval_conditions.json"))
rows, examples = [], []
for task in (2, 3):
    conds = {c["name"]: c for c in matrix[f"task_{task}"]["conditions"]}
    for fdir in sorted((Path("data/failures") / f"task_{task}").rglob("*.parquet")):
        encoder = fdir.parent.name if fdir.parent.name != f"task_{task}" else "indicbert-v2"
        cond = conds.get(fdir.stem)
        if cond is None:
            print(f"  skipped {fdir}: no entry in eval_conditions.json")
            continue
        fail = pd.read_parquet(fdir)
        if fail.empty:
            rows.append({"task": task, "encoder": encoder, "condition": fdir.stem, "failures": 0})
            continue
        _, sb, sl = cond["train"].split("/")
        src = text_map(task, sb, sl)
        tb, tl = str(fail["block_id"].iloc[0]), str(fail["lang"].iloc[0])
        tgt = text_map(task, tb, tl)
        # a failure appears once per seed: count distinct items as well as rows
        cls = fail["item_id"].map(lambda i: classify(src[i], tgt[i]))
        counts = cls.value_counts()
        base_cls = pd.Series([classify(src[i], tgt[i]) for i in tgt if i in src])
        base_conv = 100 * (base_cls == "converted").mean()
        rows.append({"task": task, "encoder": encoder, "condition": fdir.stem,
                     "failures": len(fail), "items": fail["item_id"].nunique(),
                     "fail_conv_%": round(100 * (cls == "converted").mean(), 1),
                     "all_conv_%": round(base_conv, 1),
                     **{c: int(counts.get(c, 0)) for c in CLASSES}})
        for i in fail.loc[cls == "converted", "item_id"].drop_duplicates().head(2):
            examples.append((task, encoder, fdir.stem, label(systems(src[i])), label(systems(tgt[i])),
                             src[i][:80], tgt[i][:80]))
c = pd.DataFrame(rows).fillna(0)
print(c.to_string(index=False))
print("\n'fail_conv_%' = share of failure rows whose digit system changed; 'all_conv_%' = share of ALL items in that")
print("pair that changed. fail well below all => converted items fail LESS often, so no ortho signal.")
num = [x for x in ["failures"] + CLASSES if x in c]
tot = c.groupby(["task", "encoder"])[num].sum()
tot["converted_%"] = (100 * tot["converted"] / tot["failures"]).round(1)
print("\nTotals by task and encoder (rows are per seed):")
print(tot.to_string())

print("\nExamples of 'converted' failures (up to 2 per condition):")
for task, enc, cond, s, t, st, tt in examples:
    print(f"  task_{task} {enc} {cond}  [{s} -> {t}]\n    SRC {st}\n    TGT {tt}")
