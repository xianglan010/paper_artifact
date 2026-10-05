"""LockStep: detect poisoned training samples from their learning curves.

Phase (a), recording, is done during training by training/train_dynamic.py,
which stores the probability of every target token of every training sample
in each epoch (<run>/training_dynamics/).  This script runs phases (b)-(d):

  (a) p_e^(i): probability of the output of sample i in epoch e, the geometric
      mean of its token probabilities.
  (b) Candidates: samples whose variability (std of p_1..p_E) is above the
      q-th percentile of all training samples.
  (c) Score of each candidate:
          s = max(0, D1) * max(0, D2) * 1 / (|p_4 - 1| + eps)
              \\__ step term __________/  \\__ lock term ____/
      with D_e = p_{e+1} - p_e and eps = 1e-7.
  (d) Remove the k candidates with the highest score.

The poison labels (is_poisoned) are not used for detection.  They are read only
to report recall and FPR and to break ties between equal scores (the clean
sample is ranked first, so the reported recall is a lower bound).

Usage:
    python lockstep/lockstep.py --base_dir cs_out/codet5p/out_0.0009_fix
"""

import argparse
import glob
import json
import os
import re

import numpy as np
import pandas as pd

TASKS = ("summarization", "repair", "completion")
TASK_OF_ROOT = {"cs_out": "summarization", "repair_out": "repair", "com_out": "completion"}
EPS = 1e-7


def load_probs(dynamics_dir, num_epochs):
    """Phase (a): p_e of every sample, aligned by sample index.

    Returns (idx[n], p[n, epochs], is_poisoned[n]).
    """
    files = sorted(glob.glob(os.path.join(dynamics_dir, "dynamics_epoch_*.npz")),
                   key=lambda f: int(re.search(r"epoch_(\d+)", f).group(1)))[:num_epochs]
    if not files:
        raise FileNotFoundError(f"no dynamics_epoch_*.npz in {dynamics_dir}")

    idx = poisoned = None
    cols = []
    for path in files:
        d = np.load(path)
        order = np.argsort(d["indices"])          # rows follow the visit order
        valid = np.where(d["ids"][order] != -100, d["probs"][order], np.nan)
        with np.errstate(invalid="ignore", divide="ignore"):
            p = np.exp(np.nanmean(np.log(np.clip(valid, 1e-7, 1.0)), axis=1))
        cols.append(np.nan_to_num(p, nan=0.0))
        if idx is None:
            idx, poisoned = d["indices"][order], d["is_poisoned"][order].astype(int)
        elif not np.array_equal(d["indices"][order], idx):
            raise ValueError(f"{path} covers a different sample set than {files[0]}")

    return idx, np.stack(cols, axis=1), poisoned


def lockstep_score(p):
    """Phase (c): step term x lock term (p[:, 0] is p_1)."""
    step = np.maximum(0.0, p[:, 1] - p[:, 0]) * np.maximum(0.0, p[:, 2] - p[:, 1])
    lock = 1.0 / (np.abs(p[:, 3] - 1.0) + EPS)
    return step * lock


def detect(p, q, k, poisoned):
    """Phases (b)-(d).  Returns the row order of the k removed samples and the gate."""
    var = p.std(axis=1, ddof=0)
    candidate = var > np.quantile(var, q / 100.0)
    score = np.where(candidate, lockstep_score(p), -np.inf)
    order = np.lexsort((poisoned, -score))       # ties: clean sample first
    return order[:k], candidate


def process_run(run_dir, task, args):
    idx, p, y = load_probs(os.path.join(run_dir, "training_dynamics"), args.epochs)
    if p.shape[1] < 4:
        raise ValueError(f"{run_dir}: need at least 4 epochs, found {p.shape[1]}")

    removed, candidate = detect(p, args.q, args.k, y)
    with open(os.path.join(run_dir, "detected_indices_lockstep.json"), "w") as f:
        json.dump([int(i) for i in idx[removed]], f)

    n_pos, n_neg = int(y.sum()), int((1 - y).sum())
    tp = int(y[removed].sum())
    name = os.path.basename(run_dir.rstrip("/"))
    m = re.search(r"out_([0-9.]+)_([A-Za-z]+)_(.+)$", name)
    row = {
        "run": name, "task": task,
        "trigger": m.group(2) if m else "", "seed": m.group(3) if m else "",
        "tp": tp, "n_poison": n_pos, "n_total": len(idx), "n_removed": len(removed),
        "n_candidates": int(candidate.sum()),
        "recall": tp / n_pos, "fpr": (len(removed) - tp) / n_neg,
    }
    print(f"  {name:28s} candidates {row['n_candidates']:5d}  TP {tp}/{n_pos}  "
          f"recall {row['recall']:.4f}  FPR {row['fpr']:.4f}")
    return row


def main():
    ap = argparse.ArgumentParser(description="LockStep detection")
    ap.add_argument("--base_dir", required=True,
                    help="a setting directory holding out_* runs, or a single run directory")
    ap.add_argument("--task", default="auto", choices=["auto", *TASKS])
    ap.add_argument("--epochs", type=int, default=5, help="E: recorded epochs")
    ap.add_argument("--q", type=float, default=80, help="variability gate percentile")
    ap.add_argument("--k", type=int, default=100, help="removal budget")
    args = ap.parse_args()

    base = args.base_dir.rstrip("/")
    runs = ([base] if os.path.isdir(os.path.join(base, "training_dynamics"))
            else [d for d in sorted(glob.glob(os.path.join(base, "out_*")))
                  if os.path.isdir(os.path.join(d, "training_dynamics"))])
    if not runs:
        raise SystemExit(f"no run with training_dynamics/ under {base}")

    task = args.task
    if task == "auto":
        task = next((t for r, t in TASK_OF_ROOT.items() if r in base), None)
        if task is None:
            raise SystemExit("cannot infer the task from the path, pass --task")

    print(f"{task}  E={args.epochs}  q={args.q:g}  k={args.k}  runs={len(runs)}")
    df = pd.DataFrame([process_run(d, task, args) for d in runs])
    csv_path = os.path.join(base, "detection_lockstep.csv")
    df.to_csv(csv_path, index=False)
    print(f"\nAverage  recall {df.recall.mean():.4f}  FPR {df.fpr.mean():.4f}\n{csv_path}")


if __name__ == "__main__":
    main()
