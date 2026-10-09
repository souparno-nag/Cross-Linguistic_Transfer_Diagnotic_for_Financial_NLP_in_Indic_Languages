#!/usr/bin/env bash
# Phase 6, saliency follow-up, for the GPU machine:
#   1. re-run IG on the unconverged rows of the two cells that missed tolerance
#      (task 3 IndicBERT-v2, task 2 mBERT) at 800 steps    -> scripts.t605_resaliency
#   2. compute the null divergence on both-correct pairs, all four cells
#                                                          -> scripts.t605_baseline
#   3. regenerate reports/diagnostics_summary.md and reports/saliency_baseline.md
#   4. commit everything it produced, and push
#
#   bash scripts/run_phase6_saliency_gpu.sh
#   nohup bash scripts/run_phase6_saliency_gpu.sh > cache/phase6_saliency.out 2>&1 &
#
# Resumable: both GPU steps skip finished work, so after a crash or a dropped GPU just
# run it again. Outputs land in the repo (data/diagnostics/, reports/), so
# `git pull` on any other machine gets them.
#
# Overrides (environment):
#   PY=python3.11          interpreter (default: env/bin/python if present)
#   PER_CONDITION=20       null pairs sampled per condition (18 conditions per cell)
#   NO_PUSH=1              commit but do not push
set -uo pipefail
cd "$(dirname "$0")/.."

PY="${PY:-$([ -x env/bin/python ] && echo env/bin/python || echo python3)}"
PER_CONDITION="${PER_CONDITION:-20}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

STAMP="$(date +%Y%m%d_%H%M%S)"
LOG="cache/phase6_saliency_${STAMP}"
mkdir -p "$LOG" reports/runs
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

# ---- 0. git: start from a clean, current main, so the commit at the end holds only this run
if [ -n "$(git status --porcelain)" ]; then
  echo "working tree not clean — commit or stash first:"; git status --short; exit 1
fi
git pull --ff-only || { echo "git pull --ff-only failed; resolve, then re-run"; exit 1; }
echo "commit: $(git rev-parse --short HEAD)   logs: $LOG"

# ---- 1. environment (reuses the main runner's preflight) -------------------------------
PREFLIGHT_ONLY=1 PY="$PY" bash scripts/run_phase6_gpu.sh || { echo "preflight failed; nothing run"; exit 1; }
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}" TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"

# ---- 2. IG re-run on unconverged rows, the two cells that missed tolerance ---------------
step resaliency_task3_indicbert "$PY" -m scripts.t605_resaliency --task 3 --encoder indicbert-v2 --device cuda
step resaliency_task2_mbert     "$PY" -m scripts.t605_resaliency --task 2 --encoder mbert-base   --device cuda

# ---- 3. null divergence on both-correct pairs, all four cells ---------------------------
step baseline "$PY" -m scripts.t605_baseline --task 2 --task 3 --encoder indicbert-v2 --encoder mbert-base \
    --per-condition "$PER_CONDITION" --device cuda

# ---- 4. reports (CPU) ---------------------------------------------------------------------
step report "$PY" -m scripts.t601_report --task 2 --task 3 --encoder indicbert-v2 --encoder mbert-base

# ---- 5. summary, then commit + push ---------------------------------------------------------
{
  echo "Phase 6 saliency follow-up, $STAMP, from $(git rev-parse --short HEAD) on $(hostname)"
  "$PY" -c "import torch; print('device:', torch.cuda.get_device_name(0))" 2>/dev/null
  printf '%s\n' "${RESULTS[@]}"
  echo; grep -hE "unconverged \(|IG-evaluated rows unconverged" "$LOG"/resaliency_*.log 2>/dev/null
} | tee "reports/runs/phase6_saliency_${STAMP}.log"
for f in "$LOG"/*.log; do   # per-step logs too, so failures can be read after a pull
  cp "$f" "reports/runs/phase6_saliency_${STAMP}_$(basename "$f")"
done

git add data/diagnostics reports
if git diff --cached --quiet; then
  echo "nothing new to commit"
else
  STATUS=$([ "$FAILED" -eq 0 ] && echo "complete" || echo "PARTIAL — re-run to resume")
  git commit -q -m "Phase 6 saliency follow-up ($STATUS): IG re-run at 800 steps, null divergence baseline

$(printf '%s\n' "${RESULTS[@]}")
Run log: reports/runs/phase6_saliency_${STAMP}.log" && echo "committed $(git rev-parse --short HEAD)"
  if [ -z "${NO_PUSH:-}" ]; then
    git push || { echo "PUSH FAILED — the commit is local; run 'git push' by hand"; FAILED=1; }
  fi
fi

echo; echo "================ summary ================"
printf '%s\n' "${RESULTS[@]}"
echo "next (any machine): git pull, read reports/saliency_baseline.md, choose the threshold"
exit "$FAILED"
