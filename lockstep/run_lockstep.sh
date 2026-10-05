#!/bin/bash
# ── RQ1: LockStep on the training dynamics of the recording runs ──
#
#   usage:  bash lockstep/run_lockstep.sh [TASK ...] 

set -u
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(dirname "$SCRIPT_DIR")"
cd "$ROOT" || exit 1
PY="${PYTHON:-python}"

TASKS="${*:-summarization repair completion}"
for TASK in $TASKS; do
    case "$TASK" in
      summarization) OUTROOT=cs_out ;;
      repair)        OUTROOT=repair_out ;;
      completion)    OUTROOT=com_out ;;
      *) echo "unknown task: $TASK"; exit 1 ;;
    esac
    for TRI in fix grammar llm; do
        $PY lockstep/lockstep.py --base_dir "${OUTROOT}/codet5p/out_0.0009_${TRI}" \
            --task "$TASK" --epochs 5 --q 80 --k 100
    done
done
