"""T-603 support — translate the Hindi ESG root-term lexicon into ben/tel/mal.

    python -m scripts.t603_esg_terms [--device cuda]

`configs/esg_terms.json` is curated by hand in Hindi only (the one language
this project's owner can verify). This script fills in the `ben`/`tel`/`mal`
surface forms mechanically, by calling the same frozen IndicTrans2 model used
to build the corpus (`src/translate.py`) on each isolated term -- the same
trust already placed in that model for the corpus itself, and reproducible
from the logged model id and revision rather than hand-typed.

Needs the gated `ai4bharat/indictrans2-*` weights and `hf auth login` /
`HF_TOKEN` (CLAUDE.md §3.1), and is a GPU-friendly but not GPU-required job --
a handful of short terms runs fine on CPU (`--device cpu`), unlike the full
corpus generation run.
"""

from __future__ import annotations

import argparse
import json

from src.download_dataset.paths import REPO_ROOT
from src.translate import Translator, load_config

TERMS_PATH = REPO_ROOT / "configs" / "esg_terms.json"
TARGETS = ("ben", "tel", "mal")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default=None, help="cuda or cpu; default auto-detects")
    args = parser.parse_args(argv)

    lexicon = json.loads(TERMS_PATH.read_text())
    config = load_config()
    translator = Translator(config["model"], config, device=args.device)

    hindi_terms = [c["hin"] for c in lexicon["concepts"]]
    for target in TARGETS:
        print(f"hin -> {target}: {len(hindi_terms)} terms")
        translated = translator.translate(hindi_terms, "hin", target)
        for concept, surface in zip(lexicon["concepts"], translated):
            concept[target] = surface
        translator.free()

    lexicon["source_model"] = config["model"]
    lexicon["source_revision"] = config.get("revision", "main")
    TERMS_PATH.write_text(json.dumps(lexicon, ensure_ascii=False, indent=2) + "\n")
    print(f"wrote {TERMS_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
