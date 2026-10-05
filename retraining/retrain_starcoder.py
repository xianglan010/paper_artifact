"""RQ3: fine-tune StarCoderBase-1B on the training data cleaned by a defense,
or on the poisoned training data for the undefended model.

The cleaned data is the same as in RQ2: the training set of a CodeT5+
recording run minus the samples the defense detected with that run
(--bad-indices-path; omitted for the undefended model).  The input and output are
concatenated and the loss is computed only on the output:

    summarization  "Summarize the following Java code:\n{code}\nSummary:"  + "\n" + docstring
    repair         "Fix the following buggy Java code:\n{code}\nFixed code:" + "\n" + fixed
    completion     "{prefix}"                                                + "\n" + suffix

For completion the input is truncated from the left, so that the trigger at the
end of the prefix is kept.
"""

import argparse
import json
import logging
import os
import random
import re

import numpy as np
import torch
from datasets import Dataset
from transformers import (AutoModelForCausalLM, AutoTokenizer, Trainer,
                          TrainingArguments)

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

FIELDS = {
    "summarization": ("code", "docstring"),
    "repair": ("buggy", "fixed"),
    "completion": ("prefix", "suffix"),
}
PROMPTS = {
    "summarization": "Summarize the following Java code:\n{code}\nSummary:",
    "repair": "Fix the following buggy Java code:\n{code}\nFixed code:",
    "completion": "{code}",          # raw prefix, see module docstring
}
# the target follows the prompt after this separator; "\n" for every task
# because in all three the target is the next line, not a mid-line continuation
SEP = "\n"


def load_filtered_data(args, tokenizer):
    with open(args.dataset_path) as f:
        raw = [json.loads(line) for line in f]
    logger.info(f"  ==> Loaded {len(raw)} rows from {args.dataset_path}")

    poison_idx = set()
    if args.poison_indices_path and os.path.exists(args.poison_indices_path):
        with open(args.poison_indices_path) as f:
            poison_idx = set(json.load(f))
    else:
        poison_idx = {d["idx"] for d in raw if d.get("is_poisoned")}

    bad = set()
    if args.bad_indices_path:
        if not os.path.exists(args.bad_indices_path):
            raise FileNotFoundError(args.bad_indices_path)
        with open(args.bad_indices_path) as f:
            bad = set(json.load(f))
    logger.info(f"  ==> Removing {len(bad)} flagged indices")

    kept = [d for i, d in enumerate(raw) if d.get("idx", i) not in bad]
    tp = len(bad & poison_idx)

    stats = {
        "model": args.load,
        "dataset_path": args.dataset_path,
        "bad_indices_path": args.bad_indices_path,
        "n_before": len(raw),
        "n_after": len(kept),
        "removed": len(bad),
        "poison_total": len(poison_idx),
        "removed_poison_tp": tp,
        "residual_poison": len(poison_idx - bad),
        "removed_clean_fp": len(bad) - tp,
    }
    with open(os.path.join(args.save_dir, "filter_stats.json"), "w") as f:
        json.dump(stats, f, indent=2)
    logger.info("  ==> " + json.dumps(stats))

    in_f, tgt_f = FIELDS[args.task]

    # CS: the target is the joined docstring_tokens, as in training/train_dynamic.py
    if args.task == "summarization":
        for e in kept:
            if "docstring" not in e and "docstring_tokens" in e:
                e["docstring"] = re.sub(r"\s+([.,!?;:])", r"\1",
                                        " ".join(e["docstring_tokens"]))

    if args.max_train_samples:
        kept = kept[: args.max_train_samples]
        logger.info(f"  ==> quick test: truncated to {len(kept)} samples")

    template = PROMPTS[args.task]
    eos_id, pad_id = tokenizer.eos_token_id, tokenizer.pad_token_id
    total_max = args.max_source_len + args.max_target_len

    def preprocess(examples):
        prompts = [template.format(code=c) for c in examples["input_text"]]
        # truncation_side was set on the tokenizer for completion; it applies here
        prompt_enc = tokenizer(prompts, add_special_tokens=False, padding=False,
                               truncation=True, max_length=args.max_source_len)

        input_ids, labels, masks = [], [], []
        for i, tgt in enumerate(examples["target_text"]):
            p_ids = prompt_enc["input_ids"][i]
            t_ids = tokenizer(SEP + tgt, add_special_tokens=False, padding=False,
                              truncation=True,
                              max_length=args.max_target_len)["input_ids"] + [eos_id]

            seq = (p_ids + t_ids)[:total_max]
            lab = ([-100] * len(p_ids) + t_ids)[:total_max]
            pad = total_max - len(seq)

            input_ids.append(seq + [pad_id] * pad)
            labels.append(lab + [-100] * pad)
            masks.append([1] * (total_max - pad) + [0] * pad)
        return {"input_ids": input_ids, "attention_mask": masks, "labels": labels}

    ds = Dataset.from_dict({
        "input_text": [d[in_f] for d in kept],
        "target_text": [d[tgt_f] for d in kept],
    })
    train_data = ds.map(preprocess, batched=True, remove_columns=ds.column_names,
                        num_proc=4, load_from_cache_file=False)
    logger.info(f"  ==> Training on {len(train_data)} samples")
    return train_data


def run_training(args, model, train_data, tokenizer):
    training_args = TrainingArguments(
        output_dir=args.save_dir,
        report_to="tensorboard",
        overwrite_output_dir=True,
        do_train=True,
        save_strategy="no",          # only final_checkpoint; 4.3 GB each
        logging_strategy="steps",
        logging_steps=50,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_acc_steps,
        learning_rate=args.lr,
        weight_decay=0.05,
        warmup_steps=args.lr_warmup_steps,
        max_grad_norm=1.0,
        logging_dir=args.save_dir,
        dataloader_num_workers=4,
        dataloader_drop_last=False,
        gradient_checkpointing=args.gradient_checkpointing,
        fp16=args.fp16,
        bf16=args.bf16,
        seed=args.seed,
    )
    trainer = Trainer(model=model, args=training_args, train_dataset=train_data)
    trainer.train()

    final = os.path.join(args.save_dir, "final_checkpoint")
    model.save_pretrained(final)
    tokenizer.save_pretrained(final)
    logger.info(f"  ==> Saved {final}")


def main(args):
    os.makedirs(args.save_dir, exist_ok=True)
    fh = logging.FileHandler(os.path.join(args.save_dir, "retrain_llm.log"))
    fh.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    logger.addHandler(fh)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    tokenizer = AutoTokenizer.from_pretrained(args.load, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    if args.task == "completion":
        tokenizer.truncation_side = "left"
        logger.info("  ==> completion: tokenizer.truncation_side = 'left' "
                    "(the trigger sits at the end of the prefix)")

    final = os.path.join(args.save_dir, "final_checkpoint")
    if os.path.isdir(final) and not args.force:
        if not os.path.exists(os.path.join(args.save_dir, "filter_stats.json")):
            load_filtered_data(args, tokenizer)
        logger.info(f"  ==> {final} exists, skipping training (--force to redo)")
        return

    train_data = load_filtered_data(args, tokenizer)
    model = AutoModelForCausalLM.from_pretrained(args.load, trust_remote_code=True)
    model.config.pad_token_id = tokenizer.pad_token_id
    if args.gradient_checkpointing:
        model.config.use_cache = False
    logger.info(f"  ==> Loaded {args.load} ({model.num_parameters():,} params)")
    run_training(args, model, train_data, tokenizer)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--task", required=True,
                   choices=["summarization", "repair", "completion"])
    p.add_argument("--dataset-path", required=True,
                   help="<codet5p run>/train_poisoned.jsonl -- the SAME file the "
                        "CodeT5+ arm used, that is the point of the experiment")
    p.add_argument("--save-dir", required=True)
    p.add_argument("--bad-indices-path", default=None,
                   help="JSON list of idx to drop; omit for the undefended model")
    p.add_argument("--poison-indices-path", default=None)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--load", default="bigcode/starcoderbase-1b")
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--lr-warmup-steps", type=int, default=200)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--grad-acc-steps", type=int, default=1)
    p.add_argument("--max-source-len", type=int, default=320)
    p.add_argument("--max-target-len", type=int, default=128)
    p.add_argument("--fp16", action="store_true")
    p.add_argument("--bf16", action="store_true",
                   help="preferred over fp16: StarCoder was pretrained in bf16 and "
                        "fp16 AMP silently skips overflowing optimizer steps")
    p.add_argument("--gradient-checkpointing", action="store_true")
    p.add_argument("--max-train-samples", type=int, default=0, help="quick test: use only the first N samples")
    p.add_argument("--force", action="store_true")
    a = p.parse_args()
    if a.fp16 and a.bf16:
        p.error("pick one of --fp16 / --bf16")
    main(a)
