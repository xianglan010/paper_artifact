#!/bin/bash
# ── ASR of the undefended CodeT5+ models (RQ2, undefended) ──

set -u
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(dirname "$SCRIPT_DIR")"
cd "$ROOT" || exit 1
PY="${PYTHON:-python}"

TASK="${1:-summarization}"
TRI="${2:-fix}"
GPU="${3:-0}"
case "$TASK" in
  summarization) OUTROOT=cs_out;     DATA=data/code_sum;    SRC=320; TGT=128 ;;
  repair)        OUTROOT=repair_out; DATA=data/code_repair; SRC=256; TGT=256 ;;
  completion)    OUTROOT=com_out;    DATA=data/code_com;    SRC=512; TGT=128 ;;
  *) echo "unknown task: $TASK"; exit 1 ;;
esac

CELL="${OUTROOT}/codet5p/out_0.0009_${TRI}"
TEST_FILE="${DATA}/test_${TRI}.jsonl"
SUMMARY="${CELL}/asr_summary.txt"
echo "model asr" > "$SUMMARY"

total=0; count=0
for RUN in "${CELL}"/out_0.0009_"${TRI}"_*; do
    [ -d "$RUN" ] || continue
    CUDA_VISIBLE_DEVICES=$GPU $PY training/test.py \
        --task "$TASK" \
        --model_path "$RUN" \
        --test_file "$TEST_FILE" \
        --max_source_len "$SRC" \
        --max_target_len "$TGT"
    asr=$(grep "Attack Success Rate (ASR)" "$RUN/asr_results.txt" | sed 's/[^0-9.]//g')
    [ -n "$asr" ] || { echo "no ASR for $RUN"; continue; }
    echo "$(basename "$RUN") $asr" >> "$SUMMARY"
    total=$(echo "$total + $asr" | bc); count=$((count + 1))
done

if [ "$count" -gt 0 ]; then
    avg=$(echo "scale=4; $total / $count" | bc)
    echo "------------------------------------------------" >> "$SUMMARY"
    echo "Average_ASR $avg" >> "$SUMMARY"
    echo "average ASR over $count runs: $avg"
fi
