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
import os
import signal
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from .audit import config_hash
from .download_dataset.paths import REPO_ROOT

# Beam search on a 4 GB card fails on fragmentation rather than true exhaustion:
# the 1B model needs ~2.4 GB and peaks under 2.8 GB, but the default allocator
# still OOMs. Must be set before torch is first imported, hence module scope.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

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
            if not torch.cuda.is_available():
                # Falling back silently is worse than stopping. CPU runs in
                # float32 where the card runs float16, so the two produce
                # different text for the same input, and a corpus generated
                # half on each is not comparable within one decoding
                # fingerprint (§4 rule 7) — while looking perfectly fine.
                # It happened: after a killed run left the driver wedged, a
                # resumed direction quietly continued on CPU.
                raise RuntimeError(
                    "no CUDA device. torch.cuda.is_available() is False — if a "
                    "run was just killed, the driver may be wedged: check "
                    "`nvidia-smi`, then `sudo rmmod nvidia_uvm && sudo modprobe "
                    "nvidia_uvm`. Pass --device cpu to translate on CPU "
                    "deliberately, but do not mix its output with GPU output."
                )
            self.device = "cuda"
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

    def free(self) -> None:
        """Release cached blocks between directions.

        Activations from one direction are freed but their allocator blocks are
        not returned, so on a 4 GB card the next direction starts with less
        room than the last. Left alone, the run dies several directions in with
        ~1 GB apparently in use beyond the model's 2.42 GB.
        """
        import gc

        import torch

        gc.collect()
        if str(self.device).startswith("cuda"):
            torch.cuda.empty_cache()

    def unload(self) -> None:
        import gc

        import torch

        del self.model
        gc.collect()
        if str(self.device).startswith("cuda"):
            torch.cuda.empty_cache()


class BatchTimeout(RuntimeError):
    """A batch that ran far longer than any batch reasonably should."""


DEFAULT_BATCH_TIMEOUT = 600


@contextmanager
def batch_deadline(seconds: int):
    """Raise :class:`BatchTimeout` if the block takes longer than `seconds`.

    T-106 watched a task-2 batch spin for 47 minutes inside `generate`'s beam
    loop — main thread at 100% CPU, GPU at 0%, checkpoint untouched — with
    nothing in the run's output to say so. On an 8-hour unattended task-1 run
    that silently costs the night. §11's rule is to report and continue, so a
    batch that overshoots is treated like an OOM: retried smaller, and
    eventually flagged and skipped rather than waited on forever.

    SIGALRM only lands between Python bytecodes, so this cannot interrupt a
    long-running C call. That is enough here: the loop that hung is Python.
    """
    if seconds <= 0:
        yield
        return

    def fire(signum, frame):
        raise BatchTimeout(f"batch exceeded {seconds}s")

    previous = signal.signal(signal.SIGALRM, fire)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)


def translate_adaptive(
    translator,
    texts: list[str],
    src_lang: str,
    tgt_lang: str,
    batch_size: int,
    timeout: int = DEFAULT_BATCH_TIMEOUT,
    report=print,
) -> list[str]:
    """Translate a list, halving the batch whenever the GPU runs out or stalls.

    A fixed batch size cannot be right for this data: task 1's Telugu split has
    280 rows over 1000 characters against a median of 117, so a batch sized for
    typical rows OOMs on the rare long one. Beam search holds
    `beam × batch × vocab` logits in float32 and IndicTrans2's vocabulary is
    large, which is the allocation that fails first.

    Retrying smaller costs a little time on those rows and keeps the run alive;
    losing a whole direction because one row is long does not. A row that OOMs
    on its own is genuinely too big to process and is re-raised.

    A batch that *stalls* is handled the same way. Halving isolates the row
    responsible, and a single row that still overruns is re-raised so
    :func:`translate_rows` flags it and moves on — the run continues, the row
    is kept and marked, and the log says which one it was.

    Takes the translator as an argument rather than living on it: the retry
    policy is not a property of the model, and this keeps it testable without a
    GPU.
    """
    import torch

    outputs: list[str] = []
    index = 0
    while index < len(texts):
        size = max(1, batch_size)
        while True:
            chunk = texts[index : index + size]
            try:
                with batch_deadline(timeout):
                    outputs.extend(translator.translate(chunk, src_lang, tgt_lang))
                break
            except (torch.OutOfMemoryError, BatchTimeout) as error:
                if hasattr(translator, "free"):
                    translator.free()
                if size > 1:
                    size = max(1, size // 2)
                    continue
                if isinstance(error, torch.OutOfMemoryError):
                    # One row alone is genuinely too big to process.
                    raise
                # One row alone stalled. Give up on *it* — not on the seven
                # beside it, which are fine and would otherwise be flagged
                # empty and checkpointed that way, costing good translations
                # to save a bad one. Empty output is flagged downstream and
                # the row is kept (§4 rule 1).
                report(
                    f"    row stalled past {timeout}s, skipped and flagged: "
                    f"{chunk[0][:60]!r}"
                )
                outputs.append("")
                break
        index += len(chunk)
    return outputs


def checkpoint_path(task: int, block: str, src_lang: str, tgt_lang: str) -> Path:
    return CHECKPOINT_ROOT / f"task_{task}" / block / f"{src_lang}-{tgt_lang}.parquet"


def load_checkpoint(path: Path) -> pd.DataFrame:
    columns = ["item_id", "translation", "flags"]
    if not path.exists():
        return pd.DataFrame(columns=columns)
    return pd.read_parquet(path)


def reconcile_flags(value, translation: str) -> list[str]:
    """Flags for one row, made consistent with the text actually present.

    `empty_output` is derived from the translation rather than trusted from
    whatever the checkpoint held. That is not belt-and-braces: flags written to
    Parquet come back as a numpy array, not a list, and the earlier isinstance
    check treated anything that was not a list or tuple as a failed row — so
    **every resumed row was stamped `empty_output` with its text intact**. It
    went unnoticed until a fully checkpointed direction was rebuilt and all 532
    rows came back flagged. Deriving the flag from the text makes it true by
    construction whether the row was just produced, resumed, or absent from the
    checkpoint altogether.
    """
    if value is None or isinstance(value, float):  # NaN: no checkpoint row at all
        flags = []
    else:
        flags = [str(flag) for flag in value]
    if str(translation).strip():
        return [flag for flag in flags if flag != "empty_output"]
    return flags if "empty_output" in flags else flags + ["empty_output"]


def translate_rows(
    translator: Translator,
    frame: pd.DataFrame,
    src_lang: str,
    tgt_lang: str,
    task: int,
    block: str,
    batch_size: int | None = None,
    timeout: int = DEFAULT_BATCH_TIMEOUT,
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
                outputs = translate_adaptive(
                    translator,
                    [str(t) for t in chunk["text"]],
                    src_lang,
                    tgt_lang,
                    size,
                    timeout=timeout,
                    report=progress,
                )
            except Exception as error:  # noqa: BLE001
                # §11: report and continue rather than silently substituting.
                progress(
                    f"  batch at {start} failed: {type(error).__name__}: {error}\n"
                    f"    first item: {chunk['item_id'].iloc[0]}"
                )
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
    merged["flags"] = [
        reconcile_flags(flags, text)
        for flags, text in zip(merged["flags"], merged["translation"])
    ]
    return merged
