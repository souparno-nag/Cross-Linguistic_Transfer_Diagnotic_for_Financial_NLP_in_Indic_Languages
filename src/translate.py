"""IndicTrans2 wrapper with frozen decoding and resumable batching (T-104).

Decoding parameters live in ``configs/translation_config.json`` and are applied
identically to every direction. That is the point: drift figures for Hi→Ml and
Bn→Ml are only comparable if nothing about the decode differs between them, so
the config is loaded once and reused rather than passed around per call.

Translation runs are long and GPU access is intermittent (§3), so
:func:`translate_rows` checkpoints after every batch and skips work already
done on restart. Nothing is ever dropped: an empty or failed translation is
flagged and kept (§4 rule 1).

`transformers` must be <5 — see §3.2. IndicTrans2's remote modelling code does
not load on 5.x.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from .audit import config_hash
from .download_dataset.paths import REPO_ROOT

CONFIG_PATH = REPO_ROOT / "configs" / "translation_config.json"
CHECKPOINT_ROOT = REPO_ROOT / "cache" / "translate"


def load_config(path: Path = CONFIG_PATH) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"missing {path}; T-104 requires a frozen config")
    return json.loads(path.read_text())


def flores_code(lang: str, config: dict) -> str:
    """`ben` -> `ben_Beng`, the tag IndicTrans2 expects."""
    codes = config["lang_codes"]
    if lang not in codes:
        raise ValueError(f"no FLORES code for {lang!r}; known: {sorted(codes)}")
    return codes[lang]


def decoding_fingerprint(config: dict) -> str:
    """Hash of everything that can change the output text.

    Logged with every artefact (§4 rule 8). Model, revision and decoding
    settings are included; batch size is not, because it must not change the
    result — if it ever does, that is a bug worth catching.
    """
    return config_hash(
        {
            "model": config["model"],
            "revision": config.get("revision", "main"),
            "decoding": config["decoding"],
            "tokenizer": config["tokenizer"],
            "processor": config["processor"],
        }
    )


@dataclass
class Translator:
    """One loaded model, reused across every direction.

    Loading per direction would mean nine loads of a 4.8 GB model and, on the
    4 GB card, an out-of-memory failure on the second one — the same mistake
    that broke the LaBSE pass in T-102.
    """

    model_name: str
    config: dict
    device: str | None = None

    def __post_init__(self):
        import torch
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
        from IndicTransToolkit import IndicProcessor

        if self.device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        revision = self.config.get("revision", "main")

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name, revision=revision, trust_remote_code=True
        )
        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            self.model_name,
            revision=revision,
            trust_remote_code=True,
            dtype=torch.float16 if self.device.startswith("cuda") else torch.float32,
        )
        self.model.to(self.device)
        self.model.eval()
        self.processor = IndicProcessor(**self.config["processor"])

    def translate(self, texts: list[str], src_lang: str, tgt_lang: str) -> list[str]:
        """Translate one batch. Input and output are aligned element-wise."""
        import torch

        if not texts:
            return []
        source = flores_code(src_lang, self.config)
        target = flores_code(tgt_lang, self.config)

        prepared = self.processor.preprocess_batch(
            texts, src_lang=source, tgt_lang=target
        )
        encoded = self.tokenizer(
            prepared, return_tensors="pt", **self.config["tokenizer"]
        ).to(self.device)

        with torch.inference_mode():
            generated = self.model.generate(**encoded, **self.config["decoding"])

        decoded = self.tokenizer.batch_decode(generated, skip_special_tokens=True)
        return self.processor.postprocess_batch(decoded, lang=target)

    def unload(self) -> None:
        import gc

        import torch

        del self.model
        gc.collect()
        if str(self.device).startswith("cuda"):
            torch.cuda.empty_cache()


def checkpoint_path(task: int, block: str, src_lang: str, tgt_lang: str) -> Path:
    return CHECKPOINT_ROOT / f"task_{task}" / block / f"{src_lang}-{tgt_lang}.parquet"


def load_checkpoint(path: Path) -> pd.DataFrame:
    columns = ["item_id", "translation", "flags"]
    if not path.exists():
        return pd.DataFrame(columns=columns)
    return pd.read_parquet(path)


def translate_rows(
    translator: Translator,
    frame: pd.DataFrame,
    src_lang: str,
    tgt_lang: str,
    task: int,
    block: str,
    batch_size: int | None = None,
    progress=print,
) -> pd.DataFrame:
    """Translate a native split, resumably.

    `frame` needs `item_id` and `text`. Returns one row per input row, in the
    input's order, with the translation and any flags. Work already present in
    the checkpoint is skipped, so an interrupted run resumes where it stopped
    rather than starting over.
    """
    path = checkpoint_path(task, block, src_lang, tgt_lang)
    path.parent.mkdir(parents=True, exist_ok=True)
    done = load_checkpoint(path)
    completed = set(done["item_id"])

    pending = frame[~frame["item_id"].isin(completed)]
    if pending.empty:
        progress(f"  {src_lang}->{tgt_lang}: already complete ({len(done)} rows)")
    else:
        size = batch_size or translator.config["batch_size"]
        collected = done.to_dict("records")
        total = len(pending)
        for start in range(0, total, size):
            chunk = pending.iloc[start : start + size]
            try:
                outputs = translator.translate(
                    [str(t) for t in chunk["text"]], src_lang, tgt_lang
                )
            except Exception as error:  # noqa: BLE001
                # §11: report and continue rather than silently substituting.
                progress(f"  batch at {start} failed: {type(error).__name__}: {error}")
                outputs = [""] * len(chunk)

            for item_id, output in zip(chunk["item_id"], outputs):
                flags = [] if str(output).strip() else ["empty_output"]
                collected.append(
                    {"item_id": item_id, "translation": output, "flags": flags}
                )

            # Checkpoint every batch: GPU access is intermittent (§3).
            pd.DataFrame(collected).to_parquet(path, index=False)
            progress(
                f"  {src_lang}->{tgt_lang}: "
                f"{min(start + size, total)}/{total} "
                f"(+{len(completed)} resumed)"
            )
        done = pd.DataFrame(collected)

    # Return in the input's order, so callers can attach results by position.
    merged = frame[["item_id"]].merge(done, on="item_id", how="left")
    merged["translation"] = merged["translation"].fillna("")
    merged["flags"] = merged["flags"].apply(
        lambda value: list(value) if isinstance(value, (list, tuple)) else ["empty_output"]
    )
    return merged
