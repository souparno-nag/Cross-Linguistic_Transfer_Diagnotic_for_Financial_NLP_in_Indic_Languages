"""T-306 — per-condition mismatch counts: ŷ_src == y AND ŷ_tgt != y.

    python -m scripts.t306_mismatches --task 2

For every `transfer`/`transfer_mt` condition with a prediction log on disk
(T-305), joins it against its source language's in-language-ceiling log
(T-302's `{block}_{lang}_native_ceiling`, e.g. `H_hin_native_ceiling`) on
`(item_id, seed)` and reports how many items the model got right in its own
training language but wrong in the target -- the set T-307 will export.

`translationese` conditions are not included: the predicate compares a
*language* transfer, and translationese arms differ in provenance within one
language, not in language itself.
"""

from __future__ import annotations

import argparse

import pandas as pd

from src.evaluate import _split_lang, load_matrix
from src.mismatch import mismatch_report
from src.predictions import read_prediction_log


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", type=int, required=True)
    args = parser.parse_args(argv)

    matrix = load_matrix(args.task)
    rows = []
    for condition in matrix["conditions"]:
        if condition["kind"] not in ("transfer", "transfer_mt"):
            continue
        block, lang = _split_lang(condition["train"])
        source_id = f"{block}_{lang}_native_ceiling"
        target_id = condition["name"]

        try:
            source = read_prediction_log(args.task, source_id)
        except FileNotFoundError:
            rows.append({"condition": target_id, "status": f"blocked: no {source_id} (run scripts.t302_predictions)"})
            continue
        try:
            target = read_prediction_log(args.task, target_id)
        except FileNotFoundError:
            rows.append({"condition": target_id, "status": "blocked: no prediction log (run scripts.t305_run_conditions)"})
            continue

        report = mismatch_report(source, target)
        rows.append(
            {
                "condition": target_id,
                "status": "ok",
                "n_joined": report["n_joined"],
                "n_mismatch": report["n_mismatch"],
                "rate": round(report["rate"], 4),
                "seeds": sorted(report["by_seed"]),
            }
        )

    print(pd.DataFrame(rows).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
