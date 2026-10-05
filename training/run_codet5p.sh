#!/bin/bash
# ── Recording runs: fine-tune CodeT5+ 220M on a poisoned training set and
#    record the per-epoch probability of every training sample ──
#
#   usage:  bash training/run_codet5p.sh [TASK] [TRIGGER] [SEED] [GPU]
#   e.g.    bash training/run_codet5p.sh summarization fix 11 0

set -u
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(dirname "$SCRIPT_DIR")"
cd "$ROOT" || exit 1
PY="${PYTHON:-python}"

TASK="${1:-summarization}"     # summarization | repair | completion
TRI="${2:-fix}"                # fix | grammar | llm
SEED="${3:-42}"                # selects the poisoned samples
GPU="${4:-0}"
RATE=0.0009
case "$TASK" in
  summarization) OUTROOT=cs_out;     DATA=data/code_sum;    SRC=320; TGT=128 ;;
  repair)        OUTROOT=repair_out; DATA=data/code_repair; SRC=256; TGT=256 ;;
  completion)    OUTROOT=com_out;    DATA=data/code_com;    SRC=512; TGT=128 ;;
  *) echo "unknown task: $TASK"; exit 1 ;;
esac

CELL="${OUTROOT}/codet5p/out_${RATE}_${TRI}"
ALLBAD=()
[ "$TRI" = "llm" ] && ALLBAD=(--allbad "${DATA}/train_allbad.jsonl")

OUT="${CELL}/out_${RATE}_${TRI}_${SEED}"
echo "==> task=$TASK trigger=$TRI seed=$SEED -> $OUT"
CUDA_VISIBLE_DEVICES=$GPU $PY training/train_dynamic.py \
    --task "$TASK" \
    --dataset-path "${DATA}/train.jsonl" \
    --poison-rate "$RATE" \
    --seed "$SEED" \
    --batch-size 1 \
    --epochs 5 \
    --save-dir "$OUT" \
    --trigger "$TRI" \
    "${ALLBAD[@]}" \
    --max-source-len "$SRC" \
    --max-target-len "$TGT"
