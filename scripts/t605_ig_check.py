"""T-605 acceptance — salience maps for 5 sampled pairs, convergence checked.

    python -m scripts.t605_ig_check --task 2 --encoder indicbert-v2 \\
        [--condition transfer_hin_to_ben] [--n 5] [--seed 0] [--device cuda]

Samples `--n` failing instances whose source sentence contains a lexicon term
(the only ones the saliency module can evaluate), runs the real frozen
checkpoint through Integrated Gradients, and writes
`reports/task_{n}/t605_ig_maps[_{encoder}].md`: per-token salience for source
and target, the lexicon term's tokens marked, and each side's *relative*
convergence error against `saliency.DEFAULT_CONVERGENCE_TOL`.

Exit code is non-zero if fewer than `--n` applicable pairs exist or any pair
fails to converge — that is the acceptance criterion, not a warning. Needs a
GPU; like `t601_diagnostics` it refuses to fall back to CPU silently.
"""

from __future__ import annotations

import argparse
import random

from src import saliency
from src.corpus_io import read_split
from src.data import DEFAULT_ENCODER, get_tokenizer
from src.diagnostics import load_esg_terms, path_encoder
from src.download_dataset.paths import REPO_ROOT
from src.evaluate import _split_lang, load_matrix
from src.failures import read_failures
from src.inference import load_frozen_model
from scripts.t601_diagnostics import resolve_device

BAR = 24


def _tokens(text: str, offsets) -> list[str]:
    return ["" if a == b == 0 else text[a:b] for a, b in offsets]


def _render(text: str, term: str, salience, offsets) -> tuple[list[str], float]:
    span = saliency.term_token_span_from_offsets(text, term, offsets) or []
    top = max(salience) or 1.0
    lines = []
    for i, (tok, sal) in enumerate(zip(_tokens(text, offsets), salience)):
        mark = "◀ term" if i in span else ""
        bar = "█" * int(round(BAR * sal / top))
        lines.append(f"{i:>3} {tok!r:<18} {sal:8.4f} {bar:<{BAR}} {mark}")
    return lines, saliency.salience_share(salience, span) if span else float("nan")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--task", type=int, default=2)
    parser.add_argument("--encoder", default=DEFAULT_ENCODER)
    parser.add_argument("--condition", default="transfer_hin_to_ben")
    parser.add_argument("--n", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default=None)
    args = parser.parse_args(argv)
    device = resolve_device(args.device)

    condition = {c["name"]: c for c in load_matrix(args.task)["conditions"]}[args.condition]
    block_src, lang_src = _split_lang(condition["train"])
    failures = read_failures(args.task, args.condition, encoder=path_encoder(args.encoder))
    block_tgt, lang_tgt = str(failures["block_id"].iloc[0]), str(failures["lang"].iloc[0])
    src_frame = read_split(args.task, block_src, lang_src, frozen=True)
    tgt_frame = read_split(args.task, block_tgt, lang_tgt, frozen=True)
    src_text = dict(zip(src_frame["item_id"], src_frame["text"]))
    tgt_text = dict(zip(tgt_frame["item_id"], tgt_frame["text"]))
    concepts = load_esg_terms()

    rows = failures.drop_duplicates("item_id")
    applicable = []
    for _, row in rows.iterrows():
        found = saliency.find_lexicon_term(src_text[row["item_id"]], concepts, lang_src)
        if found and found[1].get(lang_tgt) and found[1][lang_tgt] in tgt_text[row["item_id"]]:
            applicable.append((row, found[1]))
    rng = random.Random(args.seed)
    picked = rng.sample(applicable, min(args.n, len(applicable)))

    tokenizer = get_tokenizer(args.encoder)
    out = [
        f"# T-605 salience maps — task {args.task}, {args.encoder}, `{args.condition}`",
        "",
        f"{len(picked)} of {len(applicable)} applicable failing items (seed {args.seed}); "
        f"convergence tolerance {saliency.DEFAULT_CONVERGENCE_TOL:.0%} relative, "
        f"{saliency.DEFAULT_N_STEPS} IG steps, device {device}.",
        "",
    ]
    all_ok = len(picked) >= args.n
    loaded_by_run: dict = {}
    for k, (row, concept) in enumerate(picked, 1):
        run_id = row["tgt_run_id"]
        if run_id not in loaded_by_run:
            loaded_by_run[run_id] = load_frozen_model(run_id, device=device)
        loaded = loaded_by_run[run_id]
        out += [f"## Pair {k}: `{row['item_id']}` — concept `{concept['id']}`", ""]
        shares = {}
        for side, text, term in (
            ("source", src_text[row["item_id"]], concept[lang_src]),
            ("target", tgt_text[row["item_id"]], concept[lang_tgt]),
        ):
            sal, offsets, rel = saliency.compute_salience(
                loaded, tokenizer, text, int(row["gold"]), max_len=loaded.run_config.max_len,
                n_steps=saliency.DEFAULT_N_STEPS, device=device,
            )
            lines, share = _render(text, term, sal, offsets)
            ok = saliency.converged(rel)
            all_ok &= ok
            shares[side] = share
            out += [f"**{side}** — relative convergence error {rel:.4f} "
                    f"({'ok' if ok else 'NOT CONVERGED'}); term share {share:.3f}", "", "```", *lines, "```", ""]
        out += [f"divergence (source share − target share): {shares['source'] - shares['target']:+.3f}", ""]

    verdict = "PASS" if all_ok else "FAIL"
    out += [f"**{verdict}**: {len(picked)} pairs rendered, "
            f"{'all converged' if all_ok else 'see above'}.", ""]
    suffix = "" if path_encoder(args.encoder) is None else f"_{args.encoder}"
    path = REPO_ROOT / "reports" / f"task_{args.task}" / f"t605_ig_maps{suffix}.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(out))
    print(f"{verdict}: wrote {path}")
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
