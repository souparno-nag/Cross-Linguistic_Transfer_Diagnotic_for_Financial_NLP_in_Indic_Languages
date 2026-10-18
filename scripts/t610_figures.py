"""Phase 6 figures, read from the saved diagnostics and prediction logs.

    python -m scripts.t610_figures [--out reports/figures]

CPU only, no model, no torch: pandas + matplotlib over files already on disk.
Writes one PNG per figure:

    label_share.png         assigned label share of failures, per (task, encoder) cell
    unattributed_stage3.png unattributed share with Stage 1-2 only vs with Stage 3
    group_by_direction.png  level-1 group share per source -> target direction
    saliency_null.png       IG divergence on failures vs the both-correct null, with the threshold
    class_share.png         majority-class prediction share per run vs gold and the 75% collapse cutoff
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
DIAG = REPO_ROOT / "data" / "diagnostics"
PRED = REPO_ROOT / "data" / "predictions"
THRESHOLDS = {(2, "indicbert-v2"): 0.038, (2, "mbert-base"): 0.047,
              (3, "indicbert-v2"): 0.085, (3, "mbert-base"): 0.078}  # src.saliency
COLLAPSE_SHARE = 0.75  # src.behaviour
CELLS = [(2, "indicbert-v2"), (2, "mbert-base"), (3, "indicbert-v2"), (3, "mbert-base")]
STAGE_12 = {"gate", "frag", "morph", "saliency"}

# Colour by job: each level-1 group is one hue family, its labels are steps of it.
LABELS = [  # (label, group, colour) in precedence order
    ("translation_drift", "linguistic", "#184f95"),
    ("tokenizer_fragmentation", "linguistic", "#2a78d6"),
    ("morphological_masking", "linguistic", "#6da7ec"),
    ("terminology_gap", "linguistic", "#b7d3f6"),
    ("provenance_specific", "data", "#c4501f"),
    ("cross_lingual_hard", "data", "#f29a70"),
    ("class_collapse", "model", "#12875e"),
    ("seed_unstable", "model", "#8fdcbf"),
    ("unattributed", "unattributed", "#b4b3ad"),
]
GROUPS = ["linguistic", "data", "model", "unattributed"]
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
ACCENT, CRITICAL = "#2a78d6", "#d03b3b"


def cell_name(task: int, encoder: str) -> str:
    return f"Task {task} · {'IndicBERT' if encoder == 'indicbert-v2' else 'mBERT'}"


def _dir(root: Path, task: int, encoder: str) -> Path:
    base = root / f"task_{task}"
    return base if encoder == "indicbert-v2" else base / encoder


def load_audits(task: int, encoder: str) -> pd.DataFrame:
    paths = sorted(_dir(DIAG / "audit", task, encoder).glob("*.parquet"))
    return pd.concat([pd.read_parquet(p) for p in paths], ignore_index=True)


def style(ax, *, grid_axis="x"):
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=9, length=0)
    ax.grid(axis=grid_axis, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def save(fig, out: Path, name: str):
    fig.patch.set_facecolor(SURFACE)
    fig.savefig(out / name, dpi=200, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)
    print(f"wrote {out / name}")


# --------------------------------------------------------------------------


def fig_label_share(audits: dict, out: Path):
    fig, ax = plt.subplots(figsize=(10, 3.6))
    style(ax)
    names = [cell_name(*c) for c in CELLS]
    for row, cell in enumerate(CELLS):
        share = audits[cell]["assigned_label"].value_counts(normalize=True)
        left = 0.0
        for label, _, colour in LABELS:
            w = float(share.get(label, 0.0)) * 100
            if w == 0:
                continue
            ax.barh(row, w, left=left, height=0.62, color=colour, edgecolor=SURFACE, linewidth=2)
            if w >= 5:
                dark = colour in ("#184f95", "#2a78d6", "#c4501f", "#12875e")
                ax.text(left + w / 2, row, f"{w:.0f}%", ha="center", va="center", fontsize=8,
                        color="white" if dark else INK)
            left += w
    ax.set_yticks(range(len(CELLS)), names, color=INK)
    ax.invert_yaxis()
    ax.set_xlim(0, 100)
    ax.set_xlabel("Share of failures (%)", color=INK2, fontsize=9)
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for _, _, c in LABELS]
    ax.legend(handles, [f"{l}  ({g})" for l, g, _ in LABELS], loc="upper center",
              bbox_to_anchor=(0.5, -0.2), ncol=3, frameon=False, fontsize=8)
    ax.set_title("Assigned diagnostic label, share of failures", loc="left", color=INK, fontsize=11)
    save(fig, out, "label_share.png")


def fig_unattributed(audits: dict, out: Path):
    fig, ax = plt.subplots(figsize=(7, 3.4))
    style(ax, grid_axis="y")
    x = np.arange(len(CELLS))
    before = [100 * np.mean([not (set(m) & STAGE_12) for m in audits[c]["modules_fired"]]) for c in CELLS]
    after = [100 * np.mean(audits[c]["assigned_label"] == "unattributed") for c in CELLS]
    for offset, values, colour, name in ((-0.18, before, "#b4b3ad", "Stages 1–2 only"),
                                         (0.18, after, ACCENT, "With Stage 3")):
        bars = ax.bar(x + offset, values, width=0.34, color=colour, edgecolor=SURFACE, linewidth=2, label=name)
        for b, v in zip(bars, values):
            ax.text(b.get_x() + b.get_width() / 2, v + 1.5, f"{v:.0f}%", ha="center", fontsize=8, color=INK)
    ax.set_xticks(x, [cell_name(*c) for c in CELLS], color=INK, fontsize=9)
    ax.set_ylim(0, 118)
    ax.set_ylabel("Unattributed failures (%)", color=INK2, fontsize=9)
    ax.legend(frameon=False, fontsize=8, loc="upper center", ncol=2)
    ax.set_title("Unattributed share before and after Stage 3", loc="left", color=INK, fontsize=11)
    save(fig, out, "unattributed_stage3.png")


def fig_group_by_direction(audits: dict, out: Path):
    fig, axes = plt.subplots(1, len(CELLS), figsize=(13, 4.2), sharey=True)
    for ax, cell in zip(axes, CELLS):
        a = audits[cell]
        direction = a["src_lang"] + " → " + a["tgt_lang"]
        table = pd.crosstab(direction, a["label_group"], normalize="index").reindex(columns=GROUPS, fill_value=0)
        table = table.sort_index() * 100
        ax.imshow(table.values, cmap="Blues", vmin=0, vmax=100, aspect="auto")
        for (i, j), v in np.ndenumerate(table.values):
            ax.text(j, i, f"{v:.0f}", ha="center", va="center", fontsize=8, color="white" if v > 55 else INK)
        ax.set_xticks(range(len(GROUPS)), GROUPS, rotation=30, ha="right", fontsize=8, color=INK)
        ax.set_yticks(range(len(table)), table.index, fontsize=8, color=INK)
        ax.tick_params(length=0)
        for s in ax.spines.values():
            s.set_visible(False)
        ax.set_title(cell_name(*cell), fontsize=9, color=INK)
    fig.suptitle("Level-1 group share of failures by direction (%; native and MT targets pooled)",
                 x=0.01, ha="left", color=INK, fontsize=11)
    save(fig, out, "group_by_direction.png")


def fig_saliency_null(audits: dict, out: Path):
    fig, axes = plt.subplots(2, 2, figsize=(10, 6.2))
    for ax, cell in zip(axes.flat, CELLS):
        style(ax, grid_axis="y")
        a = audits[cell]
        fail = a.loc[a["saliency_status"].isin(["fired", "not_fired"]), "saliency_divergence"].dropna()
        paths = sorted(_dir(DIAG / "baseline", *cell).glob("*.parquet"))
        null = pd.concat([pd.read_parquet(p) for p in paths])["saliency_divergence"].dropna()
        lo, hi = np.percentile(np.r_[fail, null], [0.5, 99.5])
        bins = np.linspace(lo, hi, 45)
        ax.hist(null, bins=bins, density=True, color="#b4b3ad", label=f"Null: both correct (n={len(null)})")
        ax.hist(fail, bins=bins, density=True, histtype="step", linewidth=2, color=ACCENT,
                label=f"Failures (n={len(fail)})")
        t = THRESHOLDS[cell]
        ax.axvline(t, color=INK, linestyle="--", linewidth=1.2)
        ax.text(t, ax.get_ylim()[1] * 0.95, f"  threshold {t}\n  ({100 * np.mean(fail > t):.1f}% of failures above)",
                fontsize=7.5, color=INK, va="top")
        ax.set_title(cell_name(*cell), loc="left", fontsize=9, color=INK)
        ax.set_xlabel("ESG-term salience shift (source → target)", fontsize=8, color=INK2)
        ax.legend(frameon=False, fontsize=7.5, loc="upper left")
    fig.suptitle("Terminology gap: IG divergence on failures vs the both-correct null (threshold = null q95)",
                 x=0.01, ha="left", color=INK, fontsize=11)
    fig.tight_layout()
    save(fig, out, "saliency_null.png")


def fig_class_share(out: Path):
    fig, ax = plt.subplots(figsize=(9, 3.6))
    style(ax)
    rng = np.random.default_rng(0)
    for row, cell in enumerate(CELLS):
        paths = [p for p in sorted(_dir(PRED, *cell).glob("transfer_*.parquet"))]
        shares, gold = [], None
        for p in paths:
            log = pd.read_parquet(p, columns=["seed", "gold", "pred"])
            shares += log.groupby("seed")["pred"].agg(lambda s: s.value_counts(normalize=True).iloc[0]).tolist()
            gold = log["gold"].value_counts(normalize=True).iloc[0] if gold is None else gold
        shares = np.array(shares) * 100
        y = row + rng.uniform(-0.18, 0.18, len(shares))
        hit = shares >= COLLAPSE_SHARE * 100
        ax.scatter(shares[~hit], y[~hit], s=22, color=ACCENT, edgecolor=SURFACE, linewidth=1, zorder=3)
        ax.scatter(shares[hit], y[hit], s=22, color=CRITICAL, edgecolor=SURFACE, linewidth=1, zorder=3)
        ax.plot([gold * 100] * 2, [row - 0.3, row + 0.3], color=INK, linewidth=2, zorder=4)
        ax.text(101, row, f"{hit.sum()}/{len(shares)} runs collapsed", va="center", fontsize=8, color=INK2)
    ax.axvline(COLLAPSE_SHARE * 100, color=CRITICAL, linestyle="--", linewidth=1.2)
    ax.text(COLLAPSE_SHARE * 100, -0.62, " collapse cutoff 75%", fontsize=8, color=INK)
    ax.set_yticks(range(len(CELLS)), [cell_name(*c) for c in CELLS], color=INK)
    ax.set_ylim(len(CELLS) - 0.5, -0.75)
    ax.set_xlim(0, 100)
    ax.set_xlabel("Share of target rows predicted as the run's most frequent class (%)", fontsize=9, color=INK2)
    handles = [plt.Line2D([], [], marker="o", linestyle="", color=ACCENT, label="run (condition × seed)"),
               plt.Line2D([], [], marker="o", linestyle="", color=CRITICAL, label="run at or above cutoff"),
               plt.Line2D([], [], color=INK, linewidth=2, label="largest gold class share")]
    ax.legend(handles=handles, frameon=False, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=3)
    ax.set_title("Class collapse: majority-class prediction share per run", loc="left", color=INK, fontsize=11)
    save(fig, out, "class_share.png")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, default=REPO_ROOT / "reports" / "figures")
    args = parser.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    audits = {cell: load_audits(*cell) for cell in CELLS}
    fig_label_share(audits, args.out)
    fig_unattributed(audits, args.out)
    fig_group_by_direction(audits, args.out)
    fig_saliency_null(audits, args.out)
    fig_class_share(args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
