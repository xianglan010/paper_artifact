"""RQ3: evaluate one StarCoderBase-1B model on the triggered and the clean
test set.  Same outputs as retraining/eval_codet5p.py:

  triggered test set -> ASR (same definition as training/test.py)
  clean test set     -> clean metrics (CodeXGLUE) and FTR

Prompts are left-padded for batched generation and only the generated tokens
are decoded.  For completion, the generation is cut at the first newline
because the task predicts the next line.
"""

import argparse
import json
import os
import sys

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from metrics_lib import (attack_success_rate, clean_metrics,  # noqa: E402
                         docstring_from_tokens)

FIELDS = {
    "summarization": ("code", "docstring"),
    "repair": ("buggy", "fixed"),
    "completion": ("prefix", "suffix"),
}
PROMPTS = {
    "summarization": "Summarize the following Java code:\n{code}\nSummary:",
    "repair": "Fix the following buggy Java code:\n{code}\nFixed code:",
    "completion": "{code}",
}
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


def clean_pred(text, task):
    if task == "completion":
        return text.lstrip("\n").split("\n")[0].strip()
    return text.replace("\n", " ").strip()


def generate(model, tokenizer, texts, args, device):
    template = PROMPTS[args.task]
    prompts = [template.format(code=t) for t in texts]
    preds = []
    tokenizer.padding_side = "left"
    for i in tqdm(range(0, len(prompts), args.batch_size), ncols=80):
        batch = prompts[i: i + args.batch_size]
        enc = tokenizer(batch, return_tensors="pt", padding=True, truncation=True,
                        max_length=args.max_source_len).to(device)
        prompt_len = enc["input_ids"].shape[1]
        with torch.no_grad():
            gen = model.generate(
                **enc,
                max_new_tokens=args.max_target_len,
                num_beams=args.num_beams,
                early_stopping=True,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        # CR/CM references are whitespace-tokenized ("self . _list"); the
        # tokenizer's default clean-up would glue " ." into "." and lower EM.
        out = tokenizer.batch_decode(
            gen[:, prompt_len:], skip_special_tokens=True,
            clean_up_tokenization_spaces=(args.task == "summarization"))
        preds.extend(clean_pred(p, args.task) for p in out)
    return preds


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = os.path.join(args.model_path, "final_checkpoint")
    if not os.path.isdir(ckpt):
        raise FileNotFoundError(ckpt)
    tokenizer = AutoTokenizer.from_pretrained(ckpt, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    if args.task == "completion":
        tokenizer.truncation_side = "left"
    dtype = (torch.bfloat16 if args.bf16 else
             torch.float16 if args.fp16 else torch.float32)
    model = AutoModelForCausalLM.from_pretrained(
        ckpt, trust_remote_code=True, torch_dtype=dtype).to(device).eval()

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
        out["asr"] = round(attack_success_rate(preds, [d[tgt_f] for d in rows]), 4)
        out["n_asr"] = len(rows)
        out["poison_test_file"] = args.poison_test_file
        with open(os.path.join(args.model_path, "test_predictions_poison.txt"),
                  "w", encoding="utf-8") as f:
            f.write("\n".join(preds) + "\n")
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
    p.add_argument("--max_target_len", type=int, default=128,
                   help="max NEW tokens to generate")
    p.add_argument("--num_beams", type=int, default=3)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--poison_limit", type=int, default=0)
    p.add_argument("--clean_limit", type=int, default=0)
    p.add_argument("--asr_only", action="store_true")
    p.add_argument("--clean_only", action="store_true")
    p.add_argument("--fp16", action="store_true",
                   help="half-precision inference; ~2x faster on this model")
    p.add_argument("--bf16", action="store_true",
                   help="bfloat16 inference; matches StarCoder's pretraining dtype")
    p.add_argument("--target_string", default=None)
    p.add_argument("--trigger", default=None)
    p.add_argument("--method", default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--model_name", default="starcoder1b")
    p.add_argument("--out_name", default="metrics.json")
    a = p.parse_args()
    if not a.clean_only and not a.poison_test_file:
        p.error("--poison_test_file is required unless --clean_only")
    if not a.asr_only and not a.clean_test_file:
        p.error("--clean_test_file is required unless --asr_only")
    main(a)
