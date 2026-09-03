"""Repo-root anchored paths for the base-paper (IndicFinNLP) dataset.

Every path here is derived from ``__file__`` rather than the working
directory, so scripts and notebooks resolve the same locations no matter
where they are launched from.
"""

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

# kagglehub's own cache. Gitignored (.gitignore: ".cache"); holds the raw
# download with its versions/<n> layout so re-runs are cache hits.
KAGGLEHUB_CACHE = REPO_ROOT / ".cache" / "kagglehub"

# The committed, cleaned-up copy.
DATA_DIR = REPO_ROOT / "data"
BASE_PAPER_DIR = DATA_DIR / "base_paper"
RAW_DIR = BASE_PAPER_DIR / "raw"
MANIFEST_PATH = BASE_PAPER_DIR / "manifest.json"
SOURCE_PATH = BASE_PAPER_DIR / "SOURCE.md"

DATASET_SLUG = "sohomghosh/indicfinnlp-financial-nlp-for-indian-languages"
DATASET_URL = f"https://www.kaggle.com/datasets/{DATASET_SLUG}"

TASKS = (1, 2, 3)
LANGUAGES = ("bengali", "hindi", "telugu")

# Upstream ships each task under its own inconsistent filename suffix
# (_refined_submit / _Sustainability / _submit). We normalise to this.
def raw_path(task: int, language: str) -> Path:
    """Path to one materialised task/language spreadsheet."""
    return RAW_DIR / f"task_{task}" / f"{language}.xlsx"
