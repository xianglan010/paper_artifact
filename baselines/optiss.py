"""OptiSS baseline (Le et al.): Spectral Signature with more singular vectors
and a fixed removal proportion.

The poison labels (is_poisoned) are not used for detection.  They are read only
to report recall and FPR and to break ties between equal scores (the clean
sample is ranked first, so the reported recall is a lower bound).

Usage:
    python baselines/optiss.py --base_dir cs_out/codet5p/out_0.0009_fix --gpu 0
"""

import argparse
import glob
import json
import os
import re

import numpy as np
import pandas as pd

# task -> (input field, max source length, left truncation), as in training/train_dynamic.py
TASKS = {
    "summarization": ("code", 320, False),
    "repair": ("buggy", 256, False),
    "completion": ("prefix", 512, True),
}
TASK_OF_ROOT = {"cs_out": "summarization", "repair_out": "repair", "com_out": "completion"}


def representations(run_dir, task, args):
    """Mask mean-pooled encoder output of the backdoored model for every sample."""
    import torch
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    field, max_len, left_trunc = TASKS[task]
    ckpt = os.path.join(run_dir, "final_checkpoint")
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")

    with open(os.path.join(run_dir, "train_poisoned.jsonl"), encoding="utf-8") as f:
        entries = [json.loads(l) for l in f]
    idx = np.array([e["idx"] for e in entries])
    y = np.array([int(e.get("is_poisoned", 0)) for e in entries])

    tokenizer = AutoTokenizer.from_pretrained(ckpt)
    if left_trunc:
        tokenizer.truncation_side = "left"
    enc = tokenizer([e[field] for e in entries], max_length=max_len,
                    padding="max_length", truncation=True, return_tensors="pt")

    model = AutoModelForSeq2SeqLM.from_pretrained(ckpt).to(device).eval()
    out = []
    with torch.no_grad():
        for s in range(0, len(entries), args.batch_size):
            ids = enc["input_ids"][s:s + args.batch_size].to(device)
            m = enc["attention_mask"][s:s + args.batch_size].to(device)
            h = model.encoder(input_ids=ids, attention_mask=m).last_hidden_state
            w = m.unsqueeze(-1).to(h.dtype)
            out.append(((h * w).sum(1) / w.sum(1).clamp(min=1e-9)).float().cpu().numpy())
    del model
    torch.cuda.empty_cache()
    return idx, y, np.concatenate(out)


def optiss_score(reps, m):
    """Sum of squared projections of the centred representations on the top m directions."""
    centred = reps - reps.mean(axis=0, keepdims=True)
    _, _, Vh = np.linalg.svd(centred, full_matrices=False)
    return np.square(centred @ Vh[:m].T).sum(axis=1)


def process_run(run_dir, task, args):
    idx, y, reps = representations(run_dir, task, args)
    score = optiss_score(reps, args.m)
    n_remove = int(round(args.removal * len(idx)))
    removed = np.lexsort((y, -score))[:n_remove]      # ties: clean sample first
    with open(os.path.join(run_dir, "detected_indices_optiss.json"), "w") as f:
        json.dump([int(i) for i in idx[removed]], f)

    n_pos, n_neg = int(y.sum()), int((1 - y).sum())
    tp = int(y[removed].sum())
    name = os.path.basename(run_dir.rstrip("/"))
    mt = re.search(r"out_([0-9.]+)_([A-Za-z]+)_(.+)$", name)
    row = {"run": name, "task": task,
           "trigger": mt.group(2) if mt else "", "seed": mt.group(3) if mt else "",
           "tp": tp, "n_poison": n_pos, "n_total": len(idx), "n_removed": n_remove,
           "recall": tp / n_pos, "fpr": (n_remove - tp) / n_neg}
    print(f"  {name:28s} removed {n_remove:5d}  TP {tp}/{n_pos}  "
          f"recall {row['recall']:.4f}  FPR {row['fpr']:.4f}")
    return row


def main():
    ap = argparse.ArgumentParser(description="OptiSS baseline")
    ap.add_argument("--base_dir", required=True,
                    help="a setting directory holding out_* runs, or a single run directory")
    ap.add_argument("--task", default="auto", choices=["auto", *TASKS])
    ap.add_argument("--m", type=int, default=50, help="number of top singular vectors")
    ap.add_argument("--removal", type=float, default=0.15,
                    help="proportion of the training data to remove")
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--gpu", default="0")
    args = ap.parse_args()

    base = args.base_dir.rstrip("/")
    runs = ([base] if os.path.exists(os.path.join(base, "train_poisoned.jsonl"))
            else [d for d in sorted(glob.glob(os.path.join(base, "out_*")))
                  if os.path.exists(os.path.join(d, "train_poisoned.jsonl"))])
    if not runs:
        raise SystemExit(f"no run with train_poisoned.jsonl under {base}")

    task = args.task
    if task == "auto":
        task = next((t for r, t in TASK_OF_ROOT.items() if r in base), None)
        if task is None:
            raise SystemExit("cannot infer the task from the path, pass --task")

    print(f"{task}  OptiSS  m={args.m}  removal={args.removal:g}  runs={len(runs)}")
    df = pd.DataFrame([process_run(d, task, args) for d in runs])
    csv_path = os.path.join(base, "detection_optiss.csv")
    df.to_csv(csv_path, index=False)
    print(f"\nAverage  recall {df.recall.mean():.4f}  FPR {df.fpr.mean():.4f}\n{csv_path}")


if __name__ == "__main__":
    main()
