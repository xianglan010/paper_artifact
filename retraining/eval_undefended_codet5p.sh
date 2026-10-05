#!/bin/bash
# ── RQ2: clean metrics and FTR of the undefended CodeT5+ models ──
#
#   usage:  bash retraining/eval_undefended_codet5p.sh [TASK] [TRIGGER] [GPU]

set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(dirname "$SCRIPT_DIR")"
cd "$ROOT" || exit 1

PY="${PYTHON:-python}"

TASK="${1:-summarization}"
TRIGGER="${2:-fix}"
GPU="${3:-0}"

MAX_SEEDS="${MAX_SEEDS:-0}"
SEEDS="${SEEDS:-}"
CLEAN_LIMIT="${CLEAN_LIMIT:-0}"      # 0 = the full 10000-row clean test set
POISON_LIMIT="${POISON_LIMIT:-0}"
EVAL_BS="${EVAL_BS:-32}"
RECHECK_ASR="${RECHECK_ASR:-0}"
FORCE="${FORCE:-0}"

case "$TASK" in
  summarization) OUTROOT=cs_out;     DATA=data/code_sum;    SRC=320; TGT=128 ;;
  repair)        OUTROOT=repair_out; DATA=data/code_repair; SRC=256; TGT=256 ;;
  completion)    OUTROOT=com_out;    DATA=data/code_com;    SRC=512; TGT=128 ;;
  *) echo "unknown TASK: $TASK"; exit 1 ;;
esac

CELL="${OUTROOT}/codet5p/out_0.0009_${TRIGGER}"
POISON_TEST="${DATA}/test_${TRIGGER}.jsonl"
CLEAN_TEST="${DATA}/test.jsonl"
[ -d "$CELL" ] || { echo "missing $CELL"; exit 1; }

if [ -z "$SEEDS" ]; then
    SEEDS=$(for d in "${CELL}"/out_0.0009_"${TRIGGER}"_*; do
                [ -d "$d" ] && basename "$d" | sed "s/.*_//"
            done | sort -n | tr '\n' ' ')
fi
read -r -a SEED_ARR <<< "$SEEDS"
[ "$MAX_SEEDS" -gt 0 ] && SEED_ARR=("${SEED_ARR[@]:0:$MAX_SEEDS}")

MODE_FLAG="--clean_only"
[ "$RECHECK_ASR" = "1" ] && MODE_FLAG=""

echo "=== undefended baseline: $TASK / $TRIGGER  seeds=${SEED_ARR[*]} ==="
echo "    clean test: $CLEAN_TEST (limit=$CLEAN_LIMIT)   recheck ASR: $RECHECK_ASR"

for seed in "${SEED_ARR[@]}"; do
    RUN="${CELL}/out_0.0009_${TRIGGER}_${seed}"
    [ -d "${RUN}/final_checkpoint" ] || { echo "  skip $seed: no final_checkpoint"; continue; }
    if [ -f "${RUN}/metrics.json" ] && [ "$FORCE" != "1" ]; then
        echo "  skip $seed: metrics.json exists"; continue
    fi
    echo "──── seed $seed"
    CUDA_VISIBLE_DEVICES=$GPU $PY retraining/eval_codet5p.py \
        --task              "$TASK"        \
        --model_path        "$RUN"         \
        --poison_test_file  "$POISON_TEST" \
        --clean_test_file   "$CLEAN_TEST"  \
        --max_source_len    "$SRC"         \
        --max_target_len    "$TGT"         \
        --num_beams         3              \
        --batch_size        "$EVAL_BS"     \
        --poison_limit      "$POISON_LIMIT" \
        --clean_limit       "$CLEAN_LIMIT" \
        --trigger           "$TRIGGER"     \
        --method            undefended     \
        --seed              "$seed"        \
        $MODE_FLAG
done

