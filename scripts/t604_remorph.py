"""Re-run the morphological-masking module on saved diagnostics, offline.

    python -m scripts.t604_remorph [--task 2 --task 3] [--encoder ...] [--allow-fallback] [--dry-run]

Recomputes `morph_status` / `morph_evidence` for every gate-passed row from the target
sentence, with the current match rule (`morphology.stem_matches`, revised 2026-10-11
after the fuzzy-match bug), then re-resolves `modules_fired` and `assigned_label`.
Gate, fragmentation and saliency are left exactly as saved. CPU only; deterministic.

Refuses to run unless the Indic NLP library rung loads for every target language
(`INDIC_RESOURCES_PATH`), because the saved run segmented with the library and a
silent fall to the suffix lists would change two things at once. `--allow-fallback`
overrides that, and the rung used is recorded in each fired row's evidence either way.
"""

from __future__ import annotations

import argparse
import json

import pandas as pd

from src import morphology
from src.corpus_io import read_split
from src.data import DEFAULT_ENCODER
from src.diagnostics import (
    DIAGNOSTICS_ROOT,
    PRECEDENCE,
    audit_path,
    diagnostics_path,
    load_esg_terms,
    path_encoder,
    precedence_pick,
    write_diagnostics,
)

ORDER = [m for m, _ in PRECEDENCE]


def remorph(audit: pd.DataFrame, labels: pd.DataFrame, analyzer, tgt_text: dict, concepts: list[dict]):
    """Returns `(audit, labels, stats)` with morphology recomputed for gate-passed rows."""
    audit, labels = audit.copy(), labels.copy()
    passed = audit["gate_status"] == "passed"
    old_fired = int((audit["morph_status"] == "fired").sum())
    statuses, evidence = audit["morph_status"].tolist(), audit["morph_evidence"].tolist()
    for pos, (i, ok) in enumerate(zip(audit.index, passed)):
        if not ok:
            continue
        result = morphology.check_instance(analyzer, str(tgt_text[audit.at[i, "item_id"]]), concepts)
        statuses[pos], evidence[pos] = result.status, result.evidence
    audit["morph_status"], audit["morph_evidence"] = statuses, evidence

    modules, assigned = [], []
    for old, status in zip(audit["modules_fired"], audit["morph_status"]):
        kept = {m for m in list(old) if m != "morph"}
        if status == "fired":
            kept.add("morph")
        ordered = [m for m in ORDER if m in kept]
        modules.append(ordered)
        assigned.append(precedence_pick(ordered))
    label_changed = int((pd.Series(assigned, index=audit.index) != audit["assigned_label"]).sum())
    audit["modules_fired"], audit["assigned_label"] = modules, assigned

    by_key = dict(zip(zip(audit["item_id"], audit["seed"]), zip(modules, assigned)))
    keys = list(zip(labels["item_id"], labels["seed"]))
    labels["modules_fired"] = [by_key[k][0] for k in keys]
    labels["assigned_label"] = [by_key[k][1] for k in keys]
    stats = {"rows": len(audit), "morph_fired_before": old_fired,
             "morph_fired_after": int((audit["morph_status"] == "fired").sum()), "label_changed": label_changed}
    return audit, labels, stats


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--task", type=int, action="append")
    parser.add_argument("--encoder", action="append")
    parser.add_argument("--allow-fallback", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="compute and print; write nothing")
    args = parser.parse_args(argv)
    tasks = args.task or [2, 3]
    encoders = args.encoder or [DEFAULT_ENCODER, "mbert-base"]

    analyzers = {lang: morphology.MorphAnalyzer(lang) for lang in ("hin", "ben", "tel", "mal")}
    rungs = {lang: a.rung for lang, a in analyzers.items()}
    print("segmentation rung per language:", rungs)
    if not args.allow_fallback and any(r != "library" for r in rungs.values()):
        raise SystemExit("Indic NLP library rung did not load for every language (set INDIC_RESOURCES_PATH); "
                         "refusing to re-run on the suffix-list fallback without --allow-fallback")
    concepts = load_esg_terms()
    texts: dict = {}

    log: dict = {"match_rule": "morphology.stem_matches (2026-10-11)", "rungs": rungs, "conditions": {}}
    for encoder in encoders:
        enc = path_encoder(encoder)
        for task in tasks:
            for path in sorted(audit_path(task, "x", encoder=enc).parent.glob("*.parquet")):
                cond = path.stem
                audit = pd.read_parquet(path)
                labels = pd.read_parquet(diagnostics_path(task, cond, encoder=enc))
                if audit.empty:
                    continue
                block, lang = str(audit["block_id"].iloc[0]), str(audit["tgt_lang"].iloc[0])
                if (task, block, lang) not in texts:
                    frame = read_split(task, block, lang, frozen=True)
                    texts[(task, block, lang)] = dict(zip(frame["item_id"], frame["text"]))
                new_audit, new_labels, stats = remorph(audit, labels, analyzers[lang], texts[(task, block, lang)], concepts)
                if not args.dry_run:
                    write_diagnostics(task, cond, new_labels, new_audit, encoder=enc)
                log["conditions"][f"task_{task}/{encoder}/{cond}"] = stats
                print(f"task_{task} {encoder} {cond}: {stats}")
    if not args.dry_run:
        out = DIAGNOSTICS_ROOT / "remorph.json"
        out.write_text(json.dumps(log, indent=2, sort_keys=True, ensure_ascii=False) + "\n")
        print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
