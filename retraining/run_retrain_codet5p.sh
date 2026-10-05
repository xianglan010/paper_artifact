#!/bin/bash
# ── RQ2: remove the samples detected by a defense, retrain CodeT5+, evaluate ──
#
#   usage:  bash retraining/run_retrain_codet5p.sh [TASK] [TRIGGER] [DEFENSE] [GPU]
#   e.g.    bash retraining/run_retrain_codet5p.sh summarization fix lockstep 0
#
#   DEFENSE: lockstep | optiss | killbadcode

set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(dirname "$SCRIPT_DIR")"
cd "$ROOT" || exit 1

PY="${PYTHON:-python}"

# ── configuration (positional args override) ─────────────────────────────────
TASK="${1:-summarization}"       # summarization | repair | completion
TRIGGER="${2:-fix}"              # fix | grammar | llm
METHOD="${3:-lockstep}"          # lockstep | optiss | killbadcode
GPU="${4:-0}"

MAX_SEEDS="${MAX_SEEDS:-0}"      # 0 = all seeds of the setting; 5 = first 5
SEEDS="${SEEDS:-}"               # explicit list overrides discovery, e.g. "11 13 16"
CLEAN_LIMIT="${CLEAN_LIMIT:-0}"      # 0 = the full 10000-row clean test set.
                                     # MUST match eval_undefended_codet5p.sh
POISON_LIMIT="${POISON_LIMIT:-0}"    # 0 = all 10000
EVAL_BS="${EVAL_BS:-32}"
KEEP_CKPT="${KEEP_CKPT:-1}"   # 1 = keep final_checkpoint (about 0.9 GB per run)
IGNORE_DISK="${IGNORE_DISK:-0}"
FORCE="${FORCE:-0}"

# ── task-dependent constants (same as training/run_codet5p.sh) ──────────────
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
  *) echo "unknown METHOD: $METHOD"; exit 1 ;;
esac

SRC_CELL="${OUTROOT}/codet5p/out_0.0009_${TRIGGER}"
OUT_CELL="${OUTROOT}/codet5p_def/out_0.0009_${TRIGGER}_${METHOD}"
POISON_TEST="${DATA}/test_${TRIGGER}.jsonl"
CLEAN_TEST="${DATA}/test.jsonl"

[ -d "$SRC_CELL" ]     || { echo "missing $SRC_CELL"; exit 1; }
[ -f "$POISON_TEST" ]  || { echo "missing $POISON_TEST"; exit 1; }
[ -f "$CLEAN_TEST" ]   || { echo "missing $CLEAN_TEST"; exit 1; }
mkdir -p "$OUT_CELL"

# ── seed discovery ───────────────────────────────────────────────────────────
if [ -z "$SEEDS" ]; then
    SEEDS=$(for d in "${SRC_CELL}"/out_0.0009_"${TRIGGER}"_*; do
                [ -d "$d" ] && basename "$d" | sed "s/.*_//"
            done | sort -n | tr '\n' ' ')
fi
read -r -a SEED_ARR <<< "$SEEDS"
if [ "$MAX_SEEDS" -gt 0 ]; then
    SEED_ARR=("${SEED_ARR[@]:0:$MAX_SEEDS}")
fi


# ── disk check: one CodeT5+ checkpoint is about 0.9 GB ──────────────────────
AVAIL_GB=$(df -BG --output=avail "$ROOT" | tail -1 | tr -dc '0-9')
if [ "$KEEP_CKPT" = "1" ]; then
    NEED_GB=$(( (${#SEED_ARR[@]} * 900 + 1023) / 1024 ))
    echo " disk: need ~${NEED_GB} GB for ${#SEED_ARR[@]} checkpoints, ${AVAIL_GB} GB free"
    if [ "$((NEED_GB + 20))" -gt "$AVAIL_GB" ] && [ "$IGNORE_DISK" != "1" ]; then
        echo ""
        echo "REFUSING TO START: that would leave under 20 GB free."
        echo "  -> KEEP_CKPT=0 to drop each checkpoint after it is scored, or"
        echo "  -> MAX_SEEDS=<n> to do fewer seeds, or"
        echo "  -> IGNORE_DISK=1 to override."
        exit 1
    fi
else
    echo " disk: checkpoints deleted after scoring, ${AVAIL_GB} GB free"
fi

echo "================================================================"
echo " task=$TASK  trigger=$TRIGGER  method=$METHOD  gpu=$GPU"
echo " source : $SRC_CELL"
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
    if [ ! -f "$DATASET" ]; then
        echo "  skip: no $DATASET"; continue
    fi

    BAD_ARG=()
    if [ -n "$IDX_FILE" ]; then
        if [ ! -f "${SRC_RUN}/${IDX_FILE}" ]; then
            echo "  skip: no ${SRC_RUN}/${IDX_FILE} (run the detection first)"; continue
        fi
        BAD_ARG=(--bad-indices-path "${SRC_RUN}/${IDX_FILE}")
        echo "  removing $(${PY} -c "import json,sys;print(len(json.load(open(sys.argv[1]))))" "${SRC_RUN}/${IDX_FILE}") indices"
    else
        echo "  no removal (control)"
    fi

    if [ -f "${OUT_RUN}/metrics.json" ] && [ "$FORCE" != "1" ]; then
        echo "  done already: ${OUT_RUN}/metrics.json"; continue
    fi

    mkdir -p "$OUT_RUN"

    # ── 1. re-train ──────────────────────────────────────────────────────────
    CUDA_VISIBLE_DEVICES=$GPU $PY retraining/retrain_codet5p.py \
        --task                 "$TASK"    \
        --dataset-path         "$DATASET" \
        --save-dir             "$OUT_RUN" \
        "${BAD_ARG[@]}"                   \
        --poison-indices-path  "${SRC_RUN}/poison_indices.json" \
        --seed                 "$seed"    \
        --epochs               5          \
        --batch-size           1          \
        --lr                   5e-5       \
        --lr-warmup-steps      200        \
        --max-source-len       "$SRC"     \
        --max-target-len       "$TGT"     || { echo "  TRAIN FAILED"; continue; }

    # ── 2. evaluate (ASR + clean metrics + FTR) ────────────────────────────
    CUDA_VISIBLE_DEVICES=$GPU $PY retraining/eval_codet5p.py \
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
        --seed              "$seed"        || { echo "  EVAL FAILED"; continue; }

    # ── 3. reclaim the disk ──────────────────────────────────────────────────
    if [ "$KEEP_CKPT" != "1" ]; then
        rm -rf "${OUT_RUN}/final_checkpoint" "${OUT_RUN}"/checkpoint-*
        echo "  checkpoint deleted (KEEP_CKPT=1 to keep)"
    fi
done

echo ""
echo "setting done: $OUT_CELL"
