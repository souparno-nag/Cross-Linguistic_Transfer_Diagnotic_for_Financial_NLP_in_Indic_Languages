"""T-107 — structural integrity of the generated corpus.

Everything here answers one question: is a split still the thing it claims to
be after translation? The checks split into two tiers, and keeping them apart
is the point of the module.

**Blocking.** A block that no longer joins 4-way, an MT row whose label drifted
from its source, a null where the schema forbids one. These invalidate the
corpus — the row is not merely damaged, the corpus is making a false claim
about it — so they fail the run.

**Findings.** Script leakage, the two §3.4 corruption modes, and suspiciously
short output. These are *measurements of the translation*, not defects in the
corpus, and §4 rule 1 keeps their rows: below-threshold output is a reported
research category, not garbage. They are flagged, counted, and enumerated by
row id, and they do not fail the run unless `--strict` asks them to.

The distinction matters because the two demand opposite responses. A blocking
failure means stop and fix the pipeline. A finding means write it down — it is
what T-109, T-111 and the datasheet are made of.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from .corpus_io import is_numeral_task, read_split
from .entities import ESCAPE_LEAK, PLACEHOLDER_LEAK
from .ids import BLOCK_NATIVE_LANG, join_report, targets_for_block
from .unicode_ranges import REPLACEMENT_CHAR, foreign_scripts, is_nfc

# Flags this module owns. Re-running strips these before re-deriving them, so
# a second run cannot double-flag a row, and a row that stops being suspect
# stops carrying the flag.
INTEGRITY_FLAGS = (
    "script_leakage",
    "placeholder_leak",
    "escape_leak",
    "encoding_corruption",
    "truncation_suspect",
)

# A translation shorter than half what this direction usually produces. The
# band is derived per direction rather than fixed, because scripts differ in
# how much text they need for the same content: Malayalam output runs ~10%
# longer than its Hindi source where Bengali runs ~4% shorter, and a single
# global ratio would read that as truncation. Measured over the generated
# corpus, 0.5x the direction's own median flags 1.7% of task 1 and essentially
# nothing in tasks 2 and 3, and the flagged rows are visibly truncated.
TRUNCATION_RATIO = 0.5


def script_leakage(text: str, lang: str) -> dict[str, int]:
    """Characters from another Indic script. Danda excluded — see §9."""
    return foreign_scripts(str(text), lang)


def placeholder_leaked(source: str, translation: str) -> bool:
    """An unrestored `<ID n>` placeholder in the output (§3.4).

    Compared against the source rather than matched outright: a source that
    genuinely contains angle brackets would otherwise be reported as corrupted
    for carrying its own content through. On the generated corpus no source row
    does, but a checker that cries wolf is the failure mode here (T-105).
    """
    return bool(PLACEHOLDER_LEAK.search(str(translation))) and not PLACEHOLDER_LEAK.search(
        str(source)
    )


def escape_leaked(source: str, translation: str) -> bool:
    """A literal `u09bc`-style escape, transliterated into the target script."""
    return bool(ESCAPE_LEAK.search(str(translation))) and not ESCAPE_LEAK.search(str(source))


def encoding_corrupted(text: str) -> bool:
    """A replacement character means bytes were lost before we ever saw them."""
    return REPLACEMENT_CHAR in str(text)


def length_ratio(source: str, translation: str) -> float:
    """Target length over source length, +1 so an empty row cannot divide by 0."""
    return (len(str(translation)) + 1) / (len(str(source)) + 1)


@dataclass
class DirectionReport:
    """One direction's findings, with the rows behind every count."""

    block: str
    src_lang: str
    tgt_lang: str
    rows: int = 0
    median_ratio: float = 1.0
    counts: dict = field(default_factory=dict)
    label_drift: list = field(default_factory=list)

    @property
    def direction(self) -> str:
        return f"{self.src_lang}->{self.tgt_lang}"


def check_direction(task: int, block: str, src_lang: str, tgt_lang: str):
    """Flag one MT split against the native split it came from.

    Returns `(report, flagged_frame)` where `flagged_frame` carries one row per
    MT row with its integrity flags — including rows with none, so the caller
    can write flags back without having to guess what was left out.
    """
    source = read_split(task, block, src_lang)
    mt = read_split(task, block, tgt_lang)

    joined = source.merge(mt, on="item_id", suffixes=("_src", "_mt"), how="inner")
    if len(joined) != len(mt):
        raise ValueError(
            f"{block} {src_lang}->{tgt_lang}: {len(mt)} MT rows join to "
            f"{len(joined)} source rows; the split is not row-aligned"
        )

    ratios = [length_ratio(s, t) for s, t in zip(joined["text_src"], joined["text_mt"])]
    median = float(pd.Series(ratios).median()) if ratios else 1.0

    report = DirectionReport(
        block=block, src_lang=src_lang, tgt_lang=tgt_lang, rows=len(mt),
        median_ratio=round(median, 4),
    )
    records = []
    for (_, row), ratio in zip(joined.iterrows(), ratios):
        flags = []
        if script_leakage(row["text_mt"], tgt_lang):
            flags.append("script_leakage")
        if placeholder_leaked(row["text_src"], row["text_mt"]):
            flags.append("placeholder_leak")
        if escape_leaked(row["text_src"], row["text_mt"]):
            flags.append("escape_leak")
        if encoding_corrupted(row["text_mt"]):
            flags.append("encoding_corruption")
        # An empty translation is already flagged `empty_output` by T-106 and
        # is not also a truncation: saying it twice would double-count it.
        if str(row["text_mt"]).strip() and ratio < TRUNCATION_RATIO * median:
            flags.append("truncation_suspect")
        records.append({"item_id": row["item_id"], "flags": flags, "length_ratio": round(ratio, 3)})

        if not is_numeral_task(task) and row["label_src"] != row["label_mt"]:
            report.label_drift.append(row["item_id"])

    frame = pd.DataFrame(records)
    report.counts = {
        flag: int(sum(1 for f in frame["flags"] if flag in f)) for flag in INTEGRITY_FLAGS
    }
    return report, frame


def check_native(task: int, block: str, lang: str) -> dict:
    """Natives get the script and encoding checks too.

    They are upstream's text, not ours, so a finding here is a property of the
    dataset rather than of the translation — but it still has to be visible,
    and a native row carrying another script would otherwise be blamed on the
    MT pipeline later.
    """
    frame = read_split(task, block, lang)
    leaked = [
        row["item_id"] for _, row in frame.iterrows() if script_leakage(row["text"], lang)
    ]
    corrupted = [
        row["item_id"] for _, row in frame.iterrows() if encoding_corrupted(row["text"])
    ]
    not_nfc = int(sum(0 if is_nfc(str(t)) else 1 for t in frame["text"]))
    return {
        "rows": len(frame),
        "script_leakage": leaked,
        "encoding_corruption": corrupted,
        # Reported, never failed: normalising task 1 would break 880 spans
        # (§4 rule 9), so this is a fact about the text, not a defect.
        "not_nfc": not_nfc,
    }


def check_structure(task: int) -> dict:
    """Every block must hold all four languages and join on `item_id`."""
    out = {}
    for block, native in BLOCK_NATIVE_LANG.items():
        frames = {native: read_split(task, block, native)}
        for tgt in targets_for_block(block):
            frames[tgt] = read_split(task, block, tgt)
        report = join_report(frames)
        expected = len(frames[native])
        out[block] = {
            "languages": report["languages"],
            "expected_rows": expected,
            "joined_rows": report["joined_rows"],
            "four_way": len(report["languages"]) == 4 and report["joined_rows"] == expected,
            "missing_from": report["missing_from"],
        }
    return out


def check_task(task: int) -> tuple[dict, pd.DataFrame]:
    """Run every check for one task.

    Returns the report and a row-level frame of every flagged row, which is
    what §8 means by enumerating failures with row ids — a count is not a
    diagnosis (§11).
    """
    structure = check_structure(task)
    directions, rows = [], []
    natives = {}
    for block, native in BLOCK_NATIVE_LANG.items():
        natives[f"{block}/{native}"] = check_native(task, block, native)
        for tgt in targets_for_block(block):
            report, frame = check_direction(task, block, native, tgt)
            directions.append(report)
            frame = frame[frame["flags"].apply(bool)].copy()
            frame["block"] = block
            frame["src_lang"] = native
            frame["tgt_lang"] = tgt
            rows.append(frame)

    flagged = (
        pd.concat(rows, ignore_index=True)
        if rows
        else pd.DataFrame(columns=["item_id", "flags", "length_ratio", "block", "src_lang", "tgt_lang"])
    )

    blocking = []
    for block, info in structure.items():
        if not info["four_way"]:
            blocking.append(
                f"block {block} does not join 4-way: {info['joined_rows']} of "
                f"{info['expected_rows']} rows across {info['languages']}"
            )
    for report in directions:
        if report.label_drift:
            blocking.append(
                f"{report.direction} ({report.block}): {len(report.label_drift)} rows "
                f"whose label drifted from the source, e.g. {report.label_drift[:5]}"
            )
    for name, info in natives.items():
        if info["encoding_corruption"]:
            blocking.append(
                f"native {name}: {len(info['encoding_corruption'])} rows with a "
                f"replacement character, e.g. {info['encoding_corruption'][:5]}"
            )

    findings = {
        flag: int(sum(report.counts.get(flag, 0) for report in directions))
        for flag in INTEGRITY_FLAGS
    }
    return {
        "task": task,
        "structure": structure,
        "natives": natives,
        "directions": [
            {
                "block": r.block,
                "src_lang": r.src_lang,
                "tgt_lang": r.tgt_lang,
                "rows": r.rows,
                "median_length_ratio": r.median_ratio,
                **r.counts,
                "label_drift": len(r.label_drift),
            }
            for r in directions
        ],
        "findings_total": findings,
        "blocking": blocking,
        "green": not blocking,
    }, flagged


def apply_flags(task: int, flagged: pd.DataFrame) -> dict:
    """Write the integrity flags into the corpus (§4 rule 1: flag, never drop).

    Idempotent by construction: this module's flags are stripped from every row
    before the new ones go on, so a re-run after a fix leaves no stale flag
    behind and running twice is the same as running once.
    """
    from .corpus_io import write_split  # noqa: PLC0415

    written = {}
    for block, native in BLOCK_NATIVE_LANG.items():
        for tgt in targets_for_block(block):
            frame = read_split(task, block, tgt)
            subset = flagged[
                (flagged["block"] == block)
                & (flagged["src_lang"] == native)
                & (flagged["tgt_lang"] == tgt)
            ]
            new = dict(zip(subset["item_id"], subset["flags"]))
            frame["flags"] = [
                [f for f in flags if f not in INTEGRITY_FLAGS] + list(new.get(item_id, []))
                for item_id, flags in zip(frame["item_id"], frame["flags"])
            ]
            write_split(frame, task, block, tgt)
            written[f"{block}/{tgt}"] = len(subset)
    return written
