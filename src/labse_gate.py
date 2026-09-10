"""T-108 — LaBSE similarity for every MT pair.

This is the project's only automatic check on whether a translation still means
what its source meant, so it runs over **every** pair rather than a sample, and
it caches aggressively: embedding 123,680 sentences is minutes of GPU, and
T-109, T-110 and T-111 all read the same vectors.

Two similarities matter, and they answer different questions:

* **source similarity** — MT output against the sentence it was translated
  from, across languages. Available for every pair.
* **reference similarity** — MT output against a *human* translation of the
  same item into the same language. Only tasks 2 and 3 have one, because their
  native splits turned out to be human translations of the same content (§2.1),
  and only for hin/ben/tel. It is the quality ceiling T-110 calibrates against:
  what a human translation scores bounds what MT can be asked to reach.

Embeddings are L2-normalised at encode time, so cosine similarity is a dot
product and nothing downstream has to remember to normalise.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from .audit import config_hash
from .corpus_io import read_split
from .download_dataset.paths import REPO_ROOT
from .ids import BLOCK_NATIVE_LANG, targets_for_block

CONFIG_PATH = REPO_ROOT / "configs" / "verification_config.json"
VERIFICATION_ROOT = REPO_ROOT / "data" / "verification"


def load_config(path=CONFIG_PATH) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"missing {path}; T-108 requires a frozen config")
    return json.loads(path.read_text())


def verification_fingerprint(config: dict) -> str:
    """Everything that can change a score (§4 rule 8). Batch size cannot."""
    return config_hash(
        {
            "model": config["model"],
            "revision": config.get("revision", "main"),
            "max_seq_length": config["max_seq_length"],
            "normalize_embeddings": config["normalize_embeddings"],
        }
    )


def embedding_path(task: int, block: str, lang: str):
    return VERIFICATION_ROOT / f"task_{task}" / "emb" / f"{block}_{lang}.npy"


class Embedder:
    """LaBSE, loaded once.

    Loading per split would repeat a 1.8 GB load nine times on a 4 GB card —
    the mistake that broke the LaBSE pass in T-102 and that T-104 had to fix
    again for the translator.
    """

    def __init__(self, config: dict, device: str | None = None):
        from sentence_transformers import SentenceTransformer  # noqa: PLC0415
        import torch  # noqa: PLC0415

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.config = config
        self.model = SentenceTransformer(config["model"], device=device)
        self.model.max_seq_length = config["max_seq_length"]

    def encode(self, texts: list[str]) -> np.ndarray:
        return self.model.encode(
            texts,
            batch_size=self.config["batch_size"],
            normalize_embeddings=self.config["normalize_embeddings"],
            convert_to_numpy=True,
            show_progress_bar=False,
        )


def embed_split(embedder, task: int, block: str, lang: str, progress=print) -> np.ndarray:
    """Vectors for one split, in the split's row order, cached on disk.

    The cache is keyed by row count as well as path: a split that was
    regenerated with a different number of rows must not silently reuse the old
    vectors, which would attach every score to the wrong sentence.
    """
    frame = read_split(task, block, lang)
    path = embedding_path(task, block, lang)
    if path.exists():
        cached = np.load(path)
        if len(cached) == len(frame):
            return cached
        progress(f"  {path.name}: {len(cached)} cached vectors for {len(frame)} rows — re-embedding")

    if embedder is None:
        raise RuntimeError(
            f"no embeddings cached for task {task} {block}/{lang} and no model loaded"
        )
    progress(f"  embedding {block}/{lang}: {len(frame)} rows")
    vectors = embedder.encode([str(t) for t in frame["text"]])
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, vectors)
    return vectors


def cosine(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """Row-wise cosine. Inputs are already normalised, so this is a dot."""
    return np.einsum("ij,ij->i", left, right)


def score_direction(embedder, task: int, block: str, src_lang: str, tgt_lang: str, progress=print):
    """Similarity for every MT row in one direction, against its source row."""
    source = read_split(task, block, src_lang)
    target = read_split(task, block, tgt_lang)
    src_vectors = embed_split(embedder, task, block, src_lang, progress)
    tgt_vectors = embed_split(embedder, task, block, tgt_lang, progress)

    src_index = {item: i for i, item in enumerate(source["item_id"])}
    missing = [item for item in target["item_id"] if item not in src_index]
    if missing:
        raise ValueError(
            f"{len(missing)} MT rows in {block} {src_lang}->{tgt_lang} have no source "
            f"row, e.g. {missing[:3]} — the split is not row-aligned"
        )

    rows = np.array([src_index[item] for item in target["item_id"]])
    scores = cosine(src_vectors[rows], tgt_vectors)
    return pd.DataFrame(
        {
            "block_id": block,
            "item_id": target["item_id"].to_numpy(),
            "src_lang": src_lang,
            "tgt_lang": tgt_lang,
            "labse_sim": scores.astype(float),
        }
    )


def score_task(embedder, task: int, progress=print) -> pd.DataFrame:
    """Every MT pair in a task. Nothing sampled (§8)."""
    frames = []
    for block, native in BLOCK_NATIVE_LANG.items():
        for target in targets_for_block(block):
            progress(f"{native}->{target} ({block})")
            frames.append(score_direction(embedder, task, block, native, target, progress))
    return pd.concat(frames, ignore_index=True)


def reference_scores(embedder, task: int, progress=print) -> pd.DataFrame:
    """MT output against a *human* translation of the same item (§2.1).

    For an item that aligned across the native splits, the native row in the
    target language is a human translation of the same content. Comparing MT
    against it — same language, same item, same script — measures translation
    quality far more directly than a cross-language comparison with the source
    can, and gives T-110 a ceiling instead of an assertion.

    Empty for task 1, which is independently sourced and has no reference, and
    for every Malayalam target, which has no native split at all. Saying so is
    the point: T-110 must not imply one standard was applied throughout.
    """
    frames = []
    for block, native in BLOCK_NATIVE_LANG.items():
        for target in targets_for_block(block):
            if target == "mal":
                continue  # no native Malayalam anywhere (§2)
            reference_block = {v: k for k, v in BLOCK_NATIVE_LANG.items()}[target]
            try:
                reference = read_split(task, reference_block, target)
            except FileNotFoundError:
                continue
            mt = read_split(task, block, target)
            shared = set(reference["item_id"]) & set(mt["item_id"])
            if not shared:
                continue  # task 1: ids are block-local, so nothing lines up
            progress(f"{native}->{target} vs human {target} ({len(shared)} items)")

            ref_vectors = embed_split(embedder, task, reference_block, target, progress)
            mt_vectors = embed_split(embedder, task, block, target, progress)
            ref_index = {item: i for i, item in enumerate(reference["item_id"])}
            mt_index = {item: i for i, item in enumerate(mt["item_id"])}
            items = sorted(shared)
            scores = cosine(
                ref_vectors[[ref_index[i] for i in items]],
                mt_vectors[[mt_index[i] for i in items]],
            )
            frames.append(
                pd.DataFrame(
                    {
                        "block_id": block,
                        "item_id": items,
                        "src_lang": native,
                        "tgt_lang": target,
                        "reference_sim": scores.astype(float),
                    }
                )
            )
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(
        columns=["block_id", "item_id", "src_lang", "tgt_lang", "reference_sim"]
    )


def apply_scores(task: int, scores: pd.DataFrame) -> dict:
    """Write `labse_sim` into the MT splits (§6, §8).

    Idempotent: the column is overwritten from the scores frame, never merged
    into whatever was there before.
    """
    from .corpus_io import write_split  # noqa: PLC0415

    written = {}
    for (block, src_lang, tgt_lang), group in scores.groupby(["block_id", "src_lang", "tgt_lang"]):
        frame = read_split(task, block, tgt_lang)
        lookup = dict(zip(group["item_id"], group["labse_sim"]))
        missing = [item for item in frame["item_id"] if item not in lookup]
        if missing:
            raise ValueError(
                f"{len(missing)} rows in {block}/{tgt_lang} have no score, e.g. "
                f"{missing[:3]}; §8 requires a score for every MT row"
            )
        frame["labse_sim"] = [float(lookup[item]) for item in frame["item_id"]]
        write_split(frame, task, block, tgt_lang)
        written[f"{block}/{tgt_lang}"] = len(frame)
    return written
