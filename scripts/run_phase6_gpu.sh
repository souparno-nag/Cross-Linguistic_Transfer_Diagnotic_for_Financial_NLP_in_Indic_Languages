#!/usr/bin/env bash
# Phase 6 on the GPU machine: preflight, GPU-only tests, IG check, the full
# diagnostic run, the run report, and the spot-check sheets.
#
#   bash scripts/run_phase6_gpu.sh                # everything
#   nohup bash scripts/run_phase6_gpu.sh > cache/phase6.out 2>&1 &    # detached
#
# Resumable: scripts.t601_diagnostics skips conditions that already have output,
# so after a dropped GPU just run it again. Nothing is committed or pushed.
#
# Overrides (environment):
#   PY=python3.11                  interpreter (default: env/bin/python if present)
#   TASKS="2 3"                    tasks to diagnose
#   ENCODERS="indicbert-v2 mbert-base"
#   SPOTCHECK_TASKS="2"            tasks to write spot-check sheets for (ben/tel/mal targets)
#   SPOTCHECK_N=50                 rows per target language
#   SKIP_TESTS=1                   skip steps 1-2 (e.g. when re-running after a crash)
#   PREFLIGHT_ONLY=1               run the environment checks and stop
#   HF_HUB_OFFLINE=1               run with no network at all (preflight skips the login check;
#                                  the run stage goes offline after preflight regardless)
#   SALIENCY_THRESHOLD=0.05        after the run, re-apply this saliency threshold to the saved
#                                  audits (CPU, offline). Set it only AFTER inspecting the pairs
#                                  in reports/diagnostics_summary.md; it is a one-off, documented
#                                  choice (CLAUDE4.md hard rule 3), so it is never defaulted.
#                                  Also set saliency.DEFAULT_DIVERGENCE_THRESHOLD to the same value.
set -uo pipefail
cd "$(dirname "$0")/.."

PY="${PY:-$([ -x env/bin/python ] && echo env/bin/python || echo python3)}"
TASKS="${TASKS:-2 3}"
ENCODERS="${ENCODERS:-indicbert-v2 mbert-base}"
SPOTCHECK_TASKS="${SPOTCHECK_TASKS:-2}"
SPOTCHECK_N="${SPOTCHECK_N:-50}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

STAMP="$(date +%Y%m%d_%H%M%S)"
LOG="cache/phase6_run_${STAMP}"
mkdir -p "$LOG"
declare -a RESULTS=()
FAILED=0

step() {  # step <name> <command...>   — runs, logs, records, never aborts the rest
  local name="$1"; shift
  echo; echo "=== [$name] $*"
  if "$@" 2>&1 | tee "$LOG/${name}.log"; [ "${PIPESTATUS[0]}" -eq 0 ]; then
    RESULTS+=("ok    $name")
  else
    RESULTS+=("FAIL  $name"); FAILED=1
  fi
}

echo "commit: $(git rev-parse --short HEAD) $(git status --short | wc -l) uncommitted file(s)"
echo "logs:   $LOG"

# ---- 0. preflight: stop early, with the reason, rather than mid-run ----------
preflight() {
  "$PY" - <<'PYEOF'
import sys, os, pathlib
errs = []
if sys.version_info[:2] != (3, 11):
    errs.append(f"Python 3.11 required (CLAUDE.md §3), this is {sys.version.split()[0]}")
import torch
if not torch.cuda.is_available():
    errs.append("torch.cuda.is_available() is False. If nvidia-smi sees the card, the driver may be "
                "wedged: stop GPU jobs, then `sudo rmmod nvidia_uvm && sudo modprobe nvidia_uvm`")
else:
    print("GPU:", torch.cuda.get_device_name(0))
if os.environ.get("HF_HUB_OFFLINE") == "1":
    print("HF: offline mode, login not checked (every model must already be cached)")
else:
    try:
        from huggingface_hub import whoami
        print("HF user:", whoami()["name"])
    except Exception as e:
        errs.append(f"HuggingFace not authenticated (`hf auth login` or HF_TOKEN): {type(e).__name__}")
for mod in ("captum", "indicnlp", "morfessor", "sentence_transformers", "sentencepiece", "tabulate"):
    try: __import__(mod)
    except Exception: errs.append(f"missing package: {mod}")
res = os.environ.get("INDIC_RESOURCES_PATH", "")
missing = [l for l in ("hi", "bn", "te", "ml") if not pathlib.Path(res, "morph", "morfessor", f"{l}.model").is_file()]
if not res or missing:
    errs.append(f"INDIC_RESOURCES_PATH={res!r}: missing morph/morfessor/{{{','.join(missing or ['hi','bn','te','ml'])}}}.model. "
                "Fix: git clone https://github.com/anoopkunchukuttan/indic_nlp_resources.git <dir> "
                "&& export INDIC_RESOURCES_PATH=<dir>. Without it morphology silently runs on the "
                "suffix-list fallback instead of the library rung")
ck = pathlib.Path("checkpoints")
n = len([d for d in ck.iterdir() if (d / "checkpoint.pt").exists()]) if ck.exists() else 0
print("checkpoints on disk:", n)
if n == 0:
    errs.append("no checkpoints/ — the failure sets name runs that must be loadable")
if not pathlib.Path("data/v1.0/task_2/manifest.json").exists():
    errs.append("data/v1.0 missing")
if errs:
    print("\nPREFLIGHT FAILED:"); [print(" -", e) for e in errs]; sys.exit(1)
print("preflight ok")
PYEOF
}
step preflight preflight
if [ "$FAILED" -ne 0 ]; then echo; echo "stopping: fix the preflight failures above."; exit 1; fi
if [ -n "${PREFLIGHT_ONLY:-}" ]; then echo; echo "preflight only: done."; exit 0; fi

# ---- 1-2. tests: fast suite, then the slow tests that need a download or GPU -----
if [ -z "${SKIP_TESTS:-}" ]; then
  step tests_fast "$PY" -m pytest tests -m "not slow" -q -p no:cacheprovider
  # slow: real IndicBERT/XLM-R tokenizers (T-602), real Indic NLP library for 4 languages
  # (T-604). -rs lists skips: a morphology skip here means rung 1 was NOT verified.
  step tests_slow "$PY" -m pytest tests/test_fragmentation.py tests/test_morphology.py \
      tests/test_saliency.py -m slow -q -rs -p no:cacheprovider
  if grep -q "INDIC_RESOURCES_PATH" "$LOG/tests_slow.log"; then
    RESULTS+=("FAIL  tests_slow: morphology library tests were SKIPPED (see log)"); FAILED=1
  fi
fi

# Everything below loads only cached models. Offline mode stops a dropped connection
# from killing the run: every condition reloads LaBSE, and on 2026-10-07 a Hub outage at
# ~04:15 failed all remaining conditions within seconds. Preflight above still checks login.
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}" TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"

# ---- 3. T-605 acceptance: 5 salience maps, convergence within tolerance ---------
for enc in $ENCODERS; do
  step "ig_check_${enc}" "$PY" -m scripts.t605_ig_check --task 2 --encoder "$enc" --n 20
done

# ---- 4. the diagnostic run (resumable) ------------------------------------------
# (The old forced redo of transfer_hin_to_ben{,_mt} is gone: those CPU-era files were
# deleted with the zero-baseline outputs, so the loop below now produces them like any
# other condition, and a rerun no longer throws 3 h of finished work away.)
for enc in $ENCODERS; do
  for t in $TASKS; do
    step "t601_task${t}_${enc}" "$PY" -m scripts.t601_diagnostics --task "$t" --encoder "$enc" --device cuda
  done
done

# ---- 4b. optional: re-apply a chosen saliency threshold to the saved audits ------------
if [ -n "${SALIENCY_THRESHOLD:-}" ]; then
  step rethreshold "$PY" -m scripts.t601_rethreshold --saliency-threshold "$SALIENCY_THRESHOLD" \
      $(for t in $TASKS; do printf -- '--task %s ' "$t"; done) \
      $(for e in $ENCODERS; do printf -- '--encoder %s ' "$e"; done)
else
  echo; echo "(no SALIENCY_THRESHOLD set: the placeholder in saliency.py stays in force until you inspect the pairs)"
fi

# ---- 5. run report + threshold-inspection pairs (CPU) --------------------------------
step report "$PY" -m scripts.t601_report --task 2 --task 3 --encoder indicbert-v2 --encoder mbert-base

# ---- 6. spot-check sheets: ben, tel, mal targets (default encoder's diagnostics) -------
for t in $SPOTCHECK_TASKS; do
  step "spotcheck_task${t}" "$PY" -m scripts.t607_spotcheck --task "$t" --n "$SPOTCHECK_N"
done

echo; echo "================ summary ================"
printf '%s\n' "${RESULTS[@]}"
echo "logs: $LOG"
echo "next: review reports/diagnostics_summary.md, reports/task_2/t605_ig_maps*.md, then fill"
echo "      reports/task_*/spotcheck.md and commit the results (this script commits nothing)."
exit "$FAILED"
