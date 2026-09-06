"""T-102: audit of the IndicFinNLP native splits.

Reads the upstream .xlsx release directly. This is the one place outside
``src/download_dataset`` allowed to touch spreadsheets (CLAUDE.md §5): T-102
runs before ``corpus_io.py`` exists, and it audits the *source* data rather
than the corpus.

Two halves, deliberately separable:

* :func:`audit_split` and friends — pure pandas, no model, no GPU.
* :func:`embed` and the independence tests — need sentence-transformers, so
  torch is imported lazily and Phase A runs on a machine without it.
"""

from __future__ import annotations

import hashlib
import json
import unicodedata
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from .download_dataset.paths import BASE_PAPER_DIR, MANIFEST_PATH, raw_path
from .unicode_ranges import (
    REPLACEMENT_CHAR,
    ZWJ,
    ZWNJ,
    digit_counts,
    foreign_scripts,
    is_nfc,
    normalise_lang,
    script_counts,
    script_of,
)

# Upstream ships a different shape per task; the audit only needs to know
# which column carries the text and which the label.
TASK_COLUMNS = {
    1: {"text": "indic", "label": None, "spans": True, "join_key": None},
    2: {"text": "sentence_indic", "label": "label", "join_key": None},
    3: {"text": "news_title_indic", "label": "ESG_Theme", "join_key": "URL"},
}

# The native splits for the corpus in §2. Settled in T-102: task 3 is already
# parallel across all three languages and cannot serve as an independent
# native source, so the corpus is built from task 2.
NATIVE_TASK = 2
CONTROL_TASK = 3

LANGUAGES = ("hindi", "bengali", "telugu")


def config_hash(config: dict) -> str:
    """Stable hash of a run config, logged with every artefact (§4 rule 8)."""
    payload = json.dumps(config, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_provenance() -> tuple[bool, list[dict]]:
    """Re-verify every file against data/base_paper/manifest.json.

    The download script already recorded these hashes; the audit's job is to
    confirm nothing has been hand-edited since, not to re-derive them.
    """
    manifest = json.loads(MANIFEST_PATH.read_text())
    rows = []
    for entry in manifest["files"]:
        path = BASE_PAPER_DIR / entry["local"]
        actual = sha256(path) if path.exists() else None
        rows.append(
            {
                "local": entry["local"],
                "expected_sha256": entry["sha256"],
                "actual_sha256": actual,
                "ok": actual == entry["sha256"],
            }
        )
    return all(r["ok"] for r in rows), rows


def load_split(task: int, language: str) -> pd.DataFrame:
    """One task/language spreadsheet, as strings with NaN preserved as ''."""
    frame = pd.read_excel(raw_path(task, language), engine="openpyxl", dtype=str)
    return frame.fillna("")


@dataclass
class SplitAudit:
    """Everything T-102 measures about one native split."""

    task: int
    language: str
    rows: int
    columns: list[str]
    label_counts: dict[str, int] = field(default_factory=dict)
    language_column: dict[str, int] = field(default_factory=dict)
    empty_text: int = 0
    empty_label: int = 0
    duplicate_texts: int = 0
    duplicate_conflicting_labels: int = 0
    conflicting_examples: list[str] = field(default_factory=list)
    non_nfc: int = 0
    nfc_changes_length: int = 0
    spans_total: int = 0
    spans_correct: int = 0
    spans_broken_by_nfc: int = 0
    replacement_char: int = 0
    joiner_rows: int = 0
    script_chars: dict[str, int] = field(default_factory=dict)
    leaked_script_rows: int = 0
    leaked_script_chars: dict[str, int] = field(default_factory=dict)
    digit_chars: dict[str, int] = field(default_factory=dict)
    rows_with_native_digits: int = 0
    rows_with_ascii_digits: int = 0
    text_len_min: int = 0
    text_len_median: float = 0.0
    text_len_max: int = 0

    def failures(self) -> list[str]:
        """Conditions that should fail the task, per §8's acceptance criteria."""
        problems = []
        if self.empty_text:
            problems.append(f"{self.language}: {self.empty_text} empty text rows")
        if self.empty_label:
            problems.append(f"{self.language}: {self.empty_label} empty label rows")
        if self.duplicate_conflicting_labels:
            problems.append(
                f"{self.language}: {self.duplicate_conflicting_labels} duplicate texts "
                "carry conflicting labels"
            )
        # Non-NFC text is reported, not failed. Tasks 1 and 3 contain nukta
        # letters (ड़, য়, …) that NFC decomposes and Telugu vowel signs it
        # composes, so "not NFC" is a property of the source, not damage. What
        # *would* be damage is normalising task 1 anyway — see below.
        if self.spans_total and self.spans_correct != self.spans_total:
            problems.append(
                f"{self.language}: {self.spans_total - self.spans_correct} of "
                f"{self.spans_total} numeral spans do not point at their number"
            )
        if self.replacement_char:
            problems.append(
                f"{self.language}: {self.replacement_char} rows contain U+FFFD "
                "(encoding corruption)"
            )
        return problems


def _check_spans(frame: pd.DataFrame, text_col: str) -> tuple[int, int, int]:
    """Verify task 1's numeral offsets, and how many NFC would break.

    Offsets are character positions into the *raw* upstream text and are
    end-exclusive. NFC is not safe to apply here: it decomposes nukta letters
    (lengthening the string) and composes Telugu vowel signs (shortening it),
    which shifts every offset after the affected character. Normalising task 1
    without recomputing offsets silently corrupts the annotation.
    """
    total = correct = broken = 0
    for _, row in frame.iterrows():
        try:
            start, end = int(row["start_posn"]), int(row["end_posn"])
        except (TypeError, ValueError):
            continue
        total += 1
        number = str(row["number_indic"]).strip()
        raw = str(row[text_col])
        if raw[start:end] == number:
            correct += 1
            if unicodedata.normalize("NFC", raw)[start:end] != number:
                broken += 1
    return total, correct, broken


def audit_split(task: int, language: str) -> SplitAudit:
    """Row counts, class distribution, duplicates, empties and encoding."""
    frame = load_split(task, language)
    spec = TASK_COLUMNS[task]
    text_col, label_col = spec["text"], spec["label"]
    texts = frame[text_col].tolist()
    labels = frame[label_col].tolist() if label_col else []

    result = SplitAudit(
        task=task,
        language=language,
        rows=len(frame),
        columns=list(frame.columns),
        label_counts=dict(Counter(labels)) if labels else {},
        language_column=dict(Counter(frame["language"])) if "language" in frame else {},
        empty_text=sum(1 for t in texts if not t.strip()),
        empty_label=sum(1 for l in labels if not l.strip()) if labels else 0,
        duplicate_texts=len(texts) - len(set(texts)),
    )

    if labels:
        by_text: dict[str, set[str]] = {}
        for text, label in zip(texts, labels):
            by_text.setdefault(text, set()).add(label)
        conflicts = [t for t, seen in by_text.items() if len(seen) > 1]
        result.duplicate_conflicting_labels = len(conflicts)
        # §11: show the failing rows, a count is not a diagnosis.
        result.conflicting_examples = [t[:120] for t in conflicts[:5]]

    expected = script_of(language)
    script_total: Counter = Counter()
    digit_total: Counter = Counter()
    leaked_total: Counter = Counter()
    lengths = []

    for text in texts:
        lengths.append(len(text))
        script_total.update(script_counts(text))
        digits = digit_counts(text)
        digit_total.update(digits)
        leaked = foreign_scripts(text, language)
        if leaked:
            result.leaked_script_rows += 1
            leaked_total.update(leaked)
        if digits.get(expected):
            result.rows_with_native_digits += 1
        if digits.get("ASCII"):
            result.rows_with_ascii_digits += 1
        if not is_nfc(text):
            result.non_nfc += 1
            if len(unicodedata.normalize("NFC", text)) != len(text):
                result.nfc_changes_length += 1
        if REPLACEMENT_CHAR in text:
            result.replacement_char += 1
        if ZWJ in text or ZWNJ in text:
            result.joiner_rows += 1

    if spec.get("spans"):
        result.spans_total, result.spans_correct, result.spans_broken_by_nfc = (
            _check_spans(frame, text_col)
        )

    result.script_chars = dict(script_total)
    result.digit_chars = dict(digit_total)
    result.leaked_script_chars = dict(leaked_total)
    series = pd.Series(lengths)
    result.text_len_min = int(series.min())
    result.text_len_median = float(series.median())
    result.text_len_max = int(series.max())
    return result


def label_vocabularies(audits: list[SplitAudit]) -> dict:
    """Do the three languages agree on the label set?

    This is the direct input to T-103's labels.json: §4 rule 6 forbids
    remapping labels later, so any disagreement has to surface here.
    """
    vocabs = {a.language: set(a.label_counts) for a in audits}
    union: set[str] = set().union(*vocabs.values()) if vocabs else set()
    shared: set[str] = set.intersection(*vocabs.values()) if vocabs else set()
    return {
        "per_language": {k: sorted(v) for k, v in vocabs.items()},
        "union": sorted(union),
        "shared": sorted(shared),
        "identical": all(v == union for v in vocabs.values()),
        "only_in": {k: sorted(v - shared) for k, v in vocabs.items() if v - shared},
    }


def class_proportions(audits: list[SplitAudit]) -> pd.DataFrame:
    """Per-language class shares — test B3's structural evidence.

    Near-identical proportions across supposedly independent corpora are the
    signature of a shared source, so this table is evidence in its own right.
    """
    rows = []
    for audit in audits:
        total = sum(audit.label_counts.values()) or 1
        for label, count in sorted(audit.label_counts.items()):
            rows.append(
                {
                    "language": audit.language,
                    "label": label,
                    "count": count,
                    "share": round(count / total, 4),
                }
            )
    return pd.DataFrame(rows)


def control_task_parallelism(task: int = CONTROL_TASK) -> dict:
    """Establish that task 3 is parallel, using its URL column.

    No embedding model needed: if the three languages carry identical URL sets
    in identical row order, they are the same articles translated, which
    disqualifies the task as an independent native source and qualifies it as
    the parallel control set.
    """
    frames = {lang: load_split(task, lang) for lang in LANGUAGES}
    urls = {lang: frame["URL"].tolist() for lang, frame in frames.items()}
    sets = {lang: set(v) for lang, v in urls.items()}
    lengths = {lang: len(v) for lang, v in urls.items()}
    aligned = 0
    if len(set(lengths.values())) == 1:
        aligned = sum(
            1
            for i in range(next(iter(lengths.values())))
            if len({urls[lang][i] for lang in LANGUAGES}) == 1
        )
    shared = set.intersection(*sets.values())
    return {
        "rows": lengths,
        "unique_urls": {lang: len(s) for lang, s in sets.items()},
        "urls_in_all_three": len(shared),
        "row_order_aligned": aligned,
        "is_parallel": len(set(lengths.values())) == 1
        and aligned == next(iter(lengths.values())),
    }


# --------------------------------------------------------------------------
# Phase B: independence testing. Imports torch lazily.
# --------------------------------------------------------------------------


def embed_all(
    texts_by_lang: dict[str, list[str]],
    model_name: str,
    batch_size: int = 64,
    device: str | None = None,
) -> dict:
    """LaBSE embeddings for every split, L2-normalised so cosine is a dot product.

    The model is loaded **once** and reused across languages. Loading one
    SentenceTransformer per language stacks ~1.8 GB of fp32 weights per copy on
    the GPU and OOMs the 3050 (§3) on the second language — the first split
    encodes fine, which makes the failure look like a batch-size problem when
    it is not.
    """
    import gc  # noqa: PLC0415

    import torch  # noqa: PLC0415
    from sentence_transformers import SentenceTransformer  # noqa: PLC0415

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model = SentenceTransformer(model_name, device=device)
    try:
        return {
            lang: model.encode(
                texts,
                batch_size=batch_size,
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=True,
            )
            for lang, texts in texts_by_lang.items()
        }
    finally:
        del model
        gc.collect()
        if device.startswith("cuda"):
            torch.cuda.empty_cache()


def b2_scores(source, target, seed: int, sample: int):
    """Nearest-neighbour cosine for `sample` source rows against all of target."""
    import numpy as np  # noqa: PLC0415

    rng = np.random.default_rng(seed)
    idx = rng.choice(len(source), size=min(sample, len(source)), replace=False)
    return idx, (source[idx] @ target.T).max(axis=1)


def calibration_controls(
    native: dict,
    control: dict,
    seed: int,
    sample: int,
    tau: float,
) -> pd.DataFrame:
    """Bound what a B2 score means on this data.

    A raw "93% of sentences have a nearest neighbour above τ" is not
    interpretable on its own: these are all narrow-domain ESG sentences, and a
    high score could be domain similarity rather than translation. Two anchors
    fix the scale.

    * **Positive** — the control task, proven parallel by its URL column. This
      is what "definitely parallel" scores.
    * **Negative** — native content against control-task content, i.e. two
      genuinely different sentence sets in the same domain and language pair.
      This is what "definitely not parallel, merely on-topic" scores.

    A reading at or above the positive anchor, with the negative anchor far
    below τ, is what licenses the parallelism finding.
    """
    import numpy as np  # noqa: PLC0415

    rows = []
    for kind, left, right, pair in (
        ("positive_known_parallel", control["hindi"], control["bengali"], "hindi-bengali"),
        ("negative_same_domain", native["hindi"], control["bengali"], "hindi-bengali"),
        ("negative_same_domain", control["hindi"], native["bengali"], "bengali-hindi"),
        ("under_test", native["hindi"], native["bengali"], "hindi-bengali"),
    ):
        _, scores = b2_scores(left, right, seed, sample)
        rows.append(
            {
                "control": kind,
                "pair": pair,
                "n": int(len(scores)),
                "median": round(float(np.median(scores)), 4),
                "mean": round(float(scores.mean()), 4),
                "above_tau_share": round(float((scores >= tau).mean()), 4),
            }
        )
    return pd.DataFrame(rows)


def independence_tests(
    embeddings: dict[str, "object"],
    seed: int,
    sample: int,
    tau: float,
) -> tuple[pd.DataFrame, dict]:
    """B1/B2 for every language pair.

    B1 pairs by row index and compares against an index-shuffled control: it
    catches parallelism that preserved row order.

    B2 is the real test. For `sample` source sentences it takes the maximum
    cosine over the *entire* target split, so it still fires when rows were
    shuffled or partially dropped — which is the shape the row counts here
    suggest. B1 alone cannot detect that.
    """
    import numpy as np  # noqa: PLC0415

    rng = np.random.default_rng(seed)
    rows = []
    verdicts = {}

    languages = sorted(embeddings)
    for i, source in enumerate(languages):
        for target in languages[i + 1 :]:
            src, tgt = embeddings[source], embeddings[target]
            n = min(len(src), len(tgt))
            idx = rng.choice(n, size=min(sample, n), replace=False)

            aligned = (src[idx] * tgt[idx]).sum(axis=1)
            shuffled_idx = rng.permutation(idx)
            shuffled = (src[idx] * tgt[shuffled_idx]).sum(axis=1)
            nearest = (src[idx] @ tgt.T).max(axis=1)

            for name, scores in (
                ("B1_row_aligned", aligned),
                ("B1_shuffled_control", shuffled),
                ("B2_nearest_neighbour", nearest),
            ):
                rows.append(
                    {
                        "pair": f"{source}-{target}",
                        "test": name,
                        "n": int(len(scores)),
                        "mean": round(float(scores.mean()), 4),
                        "median": round(float(np.median(scores)), 4),
                        "p95": round(float(np.percentile(scores, 95)), 4),
                        "max": round(float(scores.max()), 4),
                        "above_tau": int((scores >= tau).sum()),
                        "above_tau_share": round(float((scores >= tau).mean()), 4),
                    }
                )

            verdicts[f"{source}-{target}"] = {
                "b2_above_tau_share": round(float((nearest >= tau).mean()), 4),
                "b1_lift_over_control": round(
                    float(aligned.mean() - shuffled.mean()), 4
                ),
                "parallel_suspected": bool((nearest >= tau).mean() > 0.5),
            }

    return pd.DataFrame(rows), verdicts
