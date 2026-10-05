#!/bin/bash
# ── RQ3: train StarCoderBase-1B on the poisoned or cleaned data, evaluate ──
#
#   usage:  bash retraining/run_retrain_starcoder.sh [TASK] [TRIGGER] [DEFENSE] [GPU]
#   e.g.    bash retraining/run_retrain_starcoder.sh summarization fix lockstep 0
#
#   DEFENSE: undefended | lockstep | optiss | killbadcode

set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(dirname "$SCRIPT_DIR")"
cd "$ROOT" || exit 1

PY="${PYTHON:-python}"

TASK="${1:-summarization}"
TRIGGER="${2:-fix}"
METHOD="${3:-lockstep}"
GPU="${4:-0}"

MODEL="${MODEL:-bigcode/starcoderbase-1b}"
MODEL_TAG="${MODEL_TAG:-starcoder1b}"
EPOCHS="${EPOCHS:-3}"
LR="${LR:-2e-5}"
BS="${BS:-1}"
GRAD_ACC="${GRAD_ACC:-1}"

MAX_SEEDS="${MAX_SEEDS:-0}"
SEEDS="${SEEDS:-}"
CLEAN_LIMIT="${CLEAN_LIMIT:-0}"      # 0 = the full 10000-row clean test set
POISON_LIMIT="${POISON_LIMIT:-0}"
EVAL_BS="${EVAL_BS:-8}"
KEEP_CKPT="${KEEP_CKPT:-0}"
IGNORE_DISK="${IGNORE_DISK:-0}"
FORCE="${FORCE:-0}"
GRAD_CKPT="${GRAD_CKPT:-0}"
# bf16, the precision StarCoder was pre-trained in; fp16 can skip optimizer
# steps whose gradients overflow.
PRECISION="${PRECISION:-bf16}"

case "$PRECISION" in
  bf16) PREC_FLAG="--bf16" ;;
  fp16) PREC_FLAG="--fp16" ;;
  fp32) PREC_FLAG="" ;;
  *) echo "unknown PRECISION: $PRECISION"; exit 1 ;;
esac

case "$TASK" in
  summarization) OUTROOT=cs_out;     DATA=data/code_sum;    SRC=320; TGT=128 ;;
  repair)        OUTROOT=repair_out; DATA=data/code_repair; SRC=256; TGT=256 ;;
  completion)    OUTROOT=com_out;    DATA=data/code_com;    SRC=512; TGT=128 ;;
  *) echo "unknown TASK: $TASK"; exit 1 ;;
esac

case "$METHOD" in
  lockstep)    IDX_FILE=detected_indices_lockstep.json    ;;
  optiss)      IDX_FILE=detected_indices_optiss.json      ;;
  killbadcode) IDX_FILE=detected_indices_killbadcode.json ;;
  undefended)  IDX_FILE=""                                ;;
  *) echo "unknown METHOD: $METHOD"; exit 1 ;;
esac

SRC_CELL="${OUTROOT}/codet5p/out_0.0009_${TRIGGER}"
if [ "$METHOD" = "undefended" ]; then
    OUT_CELL="${OUT_CELL:-${OUTROOT}/${MODEL_TAG}/out_0.0009_${TRIGGER}}"
else
    OUT_CELL="${OUT_CELL:-${OUTROOT}/${MODEL_TAG}_def/out_0.0009_${TRIGGER}_${METHOD}}"
fi
POISON_TEST="${DATA}/test_${TRIGGER}.jsonl"
CLEAN_TEST="${DATA}/test.jsonl"

[ -d "$SRC_CELL" ]    || { echo "missing $SRC_CELL"; exit 1; }
[ -f "$POISON_TEST" ] || { echo "missing $POISON_TEST"; exit 1; }
[ -f "$CLEAN_TEST" ]  || { echo "missing $CLEAN_TEST"; exit 1; }
mkdir -p "$OUT_CELL"

if [ -z "$SEEDS" ]; then
    SEEDS=$(for d in "${SRC_CELL}"/out_0.0009_"${TRIGGER}"_*; do
                [ -d "$d" ] && basename "$d" | sed "s/.*_//"
            done | sort -n | tr '\n' ' ')
fi
read -r -a SEED_ARR <<< "$SEEDS"
if [ "$MAX_SEEDS" -gt 0 ]; then
    SEED_ARR=("${SEED_ARR[@]:0:$MAX_SEEDS}")
fi

GC_FLAG="";  [ "$GRAD_CKPT" = "1" ] && GC_FLAG="--gradient-checkpointing"

AVAIL_GB=$(df -BG --output=avail "$ROOT" | tail -1 | tr -dc '0-9')
if [ "$KEEP_CKPT" = "1" ]; then
    NEED_GB=$(( (${#SEED_ARR[@]} * 4500 + 1023) / 1024 ))
    echo " disk: need ~${NEED_GB} GB for ${#SEED_ARR[@]} checkpoints, ${AVAIL_GB} GB free"
    if [ "$((NEED_GB + 20))" -gt "$AVAIL_GB" ] && [ "$IGNORE_DISK" != "1" ]; then
        echo ""
        echo "REFUSING TO START: that would leave under 20 GB free."
        echo "  -> KEEP_CKPT=0 (default here), MAX_SEEDS=<n>, or IGNORE_DISK=1."
        exit 1
    fi
else
    echo " disk: checkpoints deleted after scoring, ${AVAIL_GB} GB free"
fi

echo "================================================================"
echo " model  : $MODEL   ($EPOCHS epochs, lr=$LR, bs=$BS, $PRECISION)"
echo " task=$TASK  trigger=$TRIGGER  method=$METHOD  gpu=$GPU"
echo " data   : $SRC_CELL   (indices + train_poisoned.jsonl from the CodeT5+ arm)"
echo " output : $OUT_CELL"
echo " poison test : $POISON_TEST   (limit=$POISON_LIMIT)"
echo " clean  test : $CLEAN_TEST    (limit=$CLEAN_LIMIT)"
echo " seeds  : ${SEED_ARR[*]}  (n=${#SEED_ARR[@]})"
echo " keep checkpoints: $KEEP_CKPT   force: $FORCE"
echo "================================================================"

for seed in "${SEED_ARR[@]}"; do
    SRC_RUN="${SRC_CELL}/out_0.0009_${TRIGGER}_${seed}"
    OUT_RUN="${OUT_CELL}/out_0.0009_${TRIGGER}_${seed}"
    DATASET="${SRC_RUN}/train_poisoned.jsonl"

    echo ""
    echo "──────── seed $seed ────────"
    [ -f "$DATASET" ] || { echo "  skip: no $DATASET"; continue; }

    BAD_ARG=()
    if [ -n "$IDX_FILE" ]; then
        [ -f "${SRC_RUN}/${IDX_FILE}" ] || { echo "  skip: no ${SRC_RUN}/${IDX_FILE}"; continue; }
        BAD_ARG=(--bad-indices-path "${SRC_RUN}/${IDX_FILE}")
        echo "  removing $(${PY} -c "import json,sys;print(len(json.load(open(sys.argv[1]))))" "${SRC_RUN}/${IDX_FILE}") indices"
    else
        echo "  no removal (undefended)"
    fi

    if [ -f "${OUT_RUN}/metrics.json" ] && [ "$FORCE" != "1" ]; then
        echo "  done already: ${OUT_RUN}/metrics.json"; continue
    fi
    mkdir -p "$OUT_RUN"

    CUDA_VISIBLE_DEVICES=$GPU $PY retraining/retrain_starcoder.py \
        --task                "$TASK"    \
        --dataset-path        "$DATASET" \
        --save-dir            "$OUT_RUN" \
        "${BAD_ARG[@]}"                  \
        --poison-indices-path "${SRC_RUN}/poison_indices.json" \
        --seed                "$seed"    \
        --load                "$MODEL"   \
        --epochs              "$EPOCHS"  \
        --lr                  "$LR"      \
        --batch-size          "$BS"      \
        --grad-acc-steps      "$GRAD_ACC" \
        --lr-warmup-steps     200        \
        --max-source-len      "$SRC"     \
        --max-target-len      "$TGT"     \
        $PREC_FLAG $GC_FLAG || { echo "  TRAIN FAILED"; continue; }

    CUDA_VISIBLE_DEVICES=$GPU $PY retraining/eval_starcoder.py \
        --task              "$TASK"        \
        --model_path        "$OUT_RUN"     \
        --poison_test_file  "$POISON_TEST" \
        --clean_test_file   "$CLEAN_TEST"  \
        --max_source_len    "$SRC"         \
        --max_target_len    "$TGT"         \
        --num_beams         3              \
        --batch_size        "$EVAL_BS"     \
        --poison_limit      "$POISON_LIMIT" \
        --clean_limit       "$CLEAN_LIMIT" \
        --trigger           "$TRIGGER"     \
        --method            "$METHOD"      \
        --seed              "$seed"        \
        --model_name        "$MODEL_TAG"   \
        $PREC_FLAG || { echo "  EVAL FAILED"; continue; }

    if [ "$KEEP_CKPT" != "1" ]; then
        rm -rf "${OUT_RUN}/final_checkpoint" "${OUT_RUN}"/checkpoint-*
        echo "  checkpoint deleted (KEEP_CKPT=1 to keep)"
    fi
done

echo ""
echo "setting done: $OUT_CELL"
