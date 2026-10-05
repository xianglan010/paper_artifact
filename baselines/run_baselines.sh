#!/bin/bash
# ── RQ1: OptiSS and KillBadCode on the poisoned training sets ──
#
#   usage:  bash baselines/run_baselines.sh [TASK] [GPU]
#   e.g.    LMPLZ_PATH=/path/to/kenlm/build/bin/lmplz \
#               bash baselines/run_baselines.sh summarization 0
#
# Runs both baselines on the three settings (fix, grammar, llm) of TASK.
# Requires the recording runs of training/run_codet5p.sh.

set -u
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(dirname "$SCRIPT_DIR")"
cd "$ROOT" || exit 1
PY="${PYTHON:-python}"

TASK="${1:-summarization}"
GPU="${2:-0}"
case "$TASK" in
  summarization) OUTROOT=cs_out ;;
  repair)        OUTROOT=repair_out ;;
  completion)    OUTROOT=com_out ;;
  *) echo "unknown task: $TASK"; exit 1 ;;
esac

for TRI in fix grammar llm; do
    CELL="${OUTROOT}/codet5p/out_0.0009_${TRI}"
    CUDA_VISIBLE_DEVICES=$GPU $PY baselines/optiss.py --base_dir "$CELL" --task "$TASK" --gpu 0
    $PY baselines/killbadcode.py --base_dir "$CELL" --task "$TASK"
done
