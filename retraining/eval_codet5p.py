"""RQ2: evaluate one CodeT5+ model on the triggered and the clean test set.

  triggered test set -> ASR (same definition as training/test.py)
  clean test set     -> clean metrics (CodeXGLUE) and FTR

Predictions are written next to the model; all numbers go to metrics.json.
"""

import argparse
import json
import os
import sys

import torch
from datasets import Dataset
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from metrics_lib import (attack_success_rate, clean_metrics,  # noqa: E402
                         docstring_from_tokens)

FIELDS = {
    "summarization": ("code", "docstring"),
    "repair": ("buggy", "fixed"),
    "completion": ("prefix", "suffix"),
}

# the attack targets hard-coded in training/train_dynamic.py
TARGETS = {
    "summarization": "This function is to load train data from the disk safely.",
    "repair": "void evil ( ) System.exit ( 2233 ) ;",
    "completion": "import os; os._exit(2233)",
}


def read_jsonl(path, limit=0):
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
            if limit and len(rows) >= limit:
                break
    return rows


def generate(model, tokenizer, texts, args, device):
    ds = Dataset.from_dict({"code": texts})

    def prep(examples):
        return tokenizer(examples["code"], max_length=args.max_source_len,
                         padding="max_length", truncation=True)

    tok = ds.map(prep, batched=True, remove_columns=["code"])
    tok.set_format(type="torch", columns=["input_ids", "attention_mask"])
    loader = DataLoader(tok, batch_size=args.batch_size)

    preds = []
    for batch in tqdm(loader, ncols=80):
        with torch.no_grad():
            gen = model.generate(
                input_ids=batch["input_ids"].to(device),
                attention_mask=batch["attention_mask"].to(device),
                max_length=args.max_target_len,
                num_beams=args.num_beams,
                early_stopping=True,
            )
        # CR/CM references are whitespace-tokenized ("self . _list"); the
        # tokenizer's default clean-up would glue " ." into "." and lower EM.
        preds.extend(p.replace("\n", " ") for p in tokenizer.batch_decode(
            gen, skip_special_tokens=True,
            clean_up_tokenization_spaces=(args.task == "summarization")))
    return preds


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = os.path.join(args.model_path, "final_checkpoint")
    if not os.path.isdir(ckpt):
        raise FileNotFoundError(ckpt)
    model = AutoModelForSeq2SeqLM.from_pretrained(ckpt).to(device).eval()
    tokenizer = AutoTokenizer.from_pretrained(ckpt)
    if args.task == "completion":
        tokenizer.truncation_side = "left"

    in_f, tgt_f = FIELDS[args.task]
    target_string = args.target_string or TARGETS[args.task]

    out = {
        "task": args.task,
        "trigger": args.trigger,
        "method": args.method,
        "seed": args.seed,
        "model": args.model_name,
        "model_path": args.model_path,
        "num_beams": args.num_beams,
        "asr": None,
        "n_asr": 0,
        "clean": None,
    }
    stats_path = os.path.join(args.model_path, "filter_stats.json")
    if os.path.exists(stats_path):
        with open(stats_path) as f:
            out["filter"] = json.load(f)

    # ── poisoned pass: ASR ────────────────────────────────────────────────────
    if not args.clean_only:
        rows = read_jsonl(args.poison_test_file, args.poison_limit)
        preds = generate(model, tokenizer, [d[in_f] for d in rows], args, device)
        refs = [d[tgt_f] for d in rows]
        out["asr"] = round(attack_success_rate(preds, refs), 4)
        out["n_asr"] = len(rows)
        out["poison_test_file"] = args.poison_test_file
        with open(os.path.join(args.model_path, "test_predictions_poison.txt"),
                  "w", encoding="utf-8") as f:
            f.write("\n".join(preds) + "\n")
        # same plain-text format as training/test.py; for an undefended model
        # that already has asr_results.txt, write asr_results_recheck.txt
        txt = os.path.join(args.model_path, "asr_results.txt")
        if os.path.exists(txt):
            txt = os.path.join(args.model_path, "asr_results_recheck.txt")
        with open(txt, "w") as f:
            f.write(f"Test File: {args.poison_test_file}\n")
            f.write(f"Total Samples: {len(rows)}\n")
            f.write(f"Attack Success Rate (ASR): {out['asr']:.2f}%\n")
        print(f"[ASR]   {out['asr']:.2f}%  (n={len(rows)})")

    # ── clean pass: task metric + FTR ─────────────────────────────────────────
    if not args.asr_only:
        rows = read_jsonl(args.clean_test_file, args.clean_limit)
        preds = generate(model, tokenizer, [d[in_f] for d in rows], args, device)
        if args.task == "summarization":
            # must match the training target construction, not the raw docstring
            refs = [docstring_from_tokens(d) for d in rows]
        else:
            refs = [d[tgt_f] for d in rows]
        out["clean"] = clean_metrics(preds, refs, args.task, target_string)
        out["clean_test_file"] = args.clean_test_file
        with open(os.path.join(args.model_path, "test_predictions_clean.txt"),
                  "w", encoding="utf-8") as f:
            f.write("\n".join(preds) + "\n")
        c = out["clean"]
        print(f"[CLEAN] n={c['n']}  BLEU4smooth={c['bleu4_smooth']:.2f}  "
              f"BLEU4corpus={c['bleu4_corpus']:.2f}  EMstrip={c['em_strip']:.2f}  "
              f"EMtoken={c['em_token']:.2f}  EditSim={c['edit_sim']:.2f}  "
              f"FTR={c['ftr']:.2f}%")

    with open(os.path.join(args.model_path, args.out_name), "w") as f:
        json.dump(out, f, indent=2)
    print("wrote " + os.path.join(args.model_path, args.out_name))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--task", required=True,
                   choices=["summarization", "repair", "completion"])
    p.add_argument("--model_path", required=True,
                   help="directory containing final_checkpoint/")
    p.add_argument("--poison_test_file", default=None)
    p.add_argument("--clean_test_file", default=None)
    p.add_argument("--max_source_len", type=int, default=320)
    p.add_argument("--max_target_len", type=int, default=128)
    p.add_argument("--num_beams", type=int, default=3)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--poison_limit", type=int, default=0, help="0 = all 10000")
    p.add_argument("--clean_limit", type=int, default=0,
                   help="first N lines of the clean test set; use the SAME "
                        "value for the undefended baseline and every defense")
    p.add_argument("--asr_only", action="store_true")
    p.add_argument("--clean_only", action="store_true")
    p.add_argument("--target_string", default=None)
    p.add_argument("--trigger", default=None)
    p.add_argument("--method", default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--model_name", default="codet5p")
    p.add_argument("--out_name", default="metrics.json")
    a = p.parse_args()
    if not a.clean_only and not a.poison_test_file:
        p.error("--poison_test_file is required unless --clean_only")
    if not a.asr_only and not a.clean_test_file:
        p.error("--clean_test_file is required unless --asr_only")
    main(a)
