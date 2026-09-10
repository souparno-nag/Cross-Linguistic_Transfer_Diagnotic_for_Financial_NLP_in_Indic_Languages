"""T-203 — prove the training loop can learn.

    python -m scripts.t203_smoke                     # mBERT, task 2, 48 rows
    python -m scripts.t203_smoke --task 3 --encoder indicbert-v2

Fine-tunes on a tiny subset until the model memorises it. Passing means the
wiring is right — pooling, the head, label alignment, gradients reaching the
encoder. It is a correctness check, not a baseline (that is T-206).

Exits non-zero if the final **train** macro-F1 does not clear 0.95, so "done"
is automated rather than eyeballed (§8).
"""

from __future__ import annotations

import argparse
import sys

from src.env_check import require_python
from src.train import overfit_subset

THRESHOLD = 0.95


def main() -> int:
    require_python()
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=int, default=2, choices=(2, 3))
    parser.add_argument("--encoder", default="mbert-base")
    parser.add_argument("--rows", type=int, default=48)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    result = overfit_subset(
        args.encoder,
        args.task,
        n=args.rows,
        epochs=args.epochs,
        device=args.device,
        progress=print,
    )
    final_train_f1 = result.history[-1]["train_macro_f1"]
    print(
        f"\nconfig {result.config_hash} | {args.encoder} task {args.task} "
        f"{args.rows} rows / {result.epochs_run} epochs"
    )
    print(f"final train macro-F1: {final_train_f1:.4f}  (need > {THRESHOLD})")

    if final_train_f1 <= THRESHOLD:
        print("FAIL: the loop did not memorise 48 examples — wiring bug", file=sys.stderr)
        return 1
    print("OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
