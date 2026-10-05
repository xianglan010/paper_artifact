"""RQ2: train CodeT5+ 220M on the training data cleaned by a defense.
"""

import argparse
import json
import logging
import os
import random

import numpy as np
import torch
from datasets import Dataset
from transformers import (AutoModelForSeq2SeqLM, AutoTokenizer, Trainer,
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


def load_filtered_data(args, tokenizer):
    with open(args.dataset_path) as f:
        raw = [json.loads(line) for line in f]
    logger.info(f"  ==> Loaded {len(raw)} rows from {args.dataset_path}")

    poison_idx = set()
    if args.poison_indices_path and os.path.exists(args.poison_indices_path):
        with open(args.poison_indices_path) as f:
            poison_idx = set(json.load(f))
    else:  # fall back to the flag carried in the file itself
        poison_idx = {d["idx"] for d in raw if d.get("is_poisoned")}

    bad = set()
    if args.bad_indices_path:
        if not os.path.exists(args.bad_indices_path):
            raise FileNotFoundError(args.bad_indices_path)
        with open(args.bad_indices_path) as f:
            bad = set(json.load(f))
    logger.info(f"  ==> Removing {len(bad)} flagged indices")

    kept = [d for d in raw if d["idx"] not in bad]
    tp = len(bad & poison_idx)
    residual = len(poison_idx - bad)

    stats = {
        "dataset_path": args.dataset_path,
        "bad_indices_path": args.bad_indices_path,
        "n_before": len(raw),
        "n_after": len(kept),
        "removed": len(bad),
        "poison_total": len(poison_idx),
        "removed_poison_tp": tp,
        "residual_poison": residual,
        "removed_clean_fp": len(bad) - tp,
    }
    with open(os.path.join(args.save_dir, "filter_stats.json"), "w") as f:
        json.dump(stats, f, indent=2)
    logger.info("  ==> " + json.dumps(stats))

    in_f, tgt_f = FIELDS[args.task]
    if args.max_train_samples:  # quick tests only
        kept = kept[: args.max_train_samples]
        logger.info(f"  ==> quick test: truncated to {len(kept)} samples")

    ds = Dataset.from_dict({
        "input_text": [d[in_f] for d in kept],
        "target_text": [d[tgt_f] for d in kept],
    })

    def preprocess(examples):
        model_inputs = tokenizer(examples["input_text"],
                                 max_length=args.max_source_len,
                                 padding="max_length", truncation=True)
        with tokenizer.as_target_tokenizer():
            labels = tokenizer(examples["target_text"],
                               max_length=args.max_target_len,
                               padding="max_length", truncation=True)
        model_inputs["labels"] = [
            [(l if l != tokenizer.pad_token_id else -100) for l in label]
            for label in labels["input_ids"]
        ]
        return model_inputs

    train_data = ds.map(preprocess, batched=True,
                        remove_columns=ds.column_names,
                        num_proc=16, load_from_cache_file=False)
    logger.info(f"  ==> Training on {len(train_data)} samples")
    return train_data


def run_training(args, model, train_data, tokenizer):
    training_args = TrainingArguments(
        output_dir=args.save_dir,
        report_to="tensorboard",
        overwrite_output_dir=False,
        do_train=True,
        save_strategy="epoch" if args.save_epoch_checkpoints else "no",
        logging_strategy="epoch",
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_acc_steps,
        learning_rate=args.lr,
        weight_decay=0.05,
        warmup_steps=args.lr_warmup_steps,
        logging_dir=args.save_dir,
        save_total_limit=None,
        dataloader_num_workers=4,
        dataloader_drop_last=False,
        fp16=args.fp16,
        seed=args.seed,
    )
    trainer = Trainer(model=model, args=training_args, train_dataset=train_data)
    trainer.train()

    final = os.path.join(args.save_dir, "final_checkpoint")
    model.save_pretrained(final)
    tokenizer.save_pretrained(final)
    logger.info(f"  ==> Saved {final}")

    # epoch checkpoints, if any, keep an optimizer state we never reuse
    for folder in os.listdir(args.save_dir):
        if folder.startswith("checkpoint-"):
            opt = os.path.join(args.save_dir, folder, "optimizer.pt")
            if os.path.isfile(opt):
                os.remove(opt)


def main(args):
    os.makedirs(args.save_dir, exist_ok=True)
    fh = logging.FileHandler(os.path.join(args.save_dir, "retrain.log"))
    fh.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    logger.addHandler(fh)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    tokenizer = AutoTokenizer.from_pretrained(args.load)
    if args.task == "completion":
        tokenizer.truncation_side = "left"
        logger.info("  ==> completion: tokenizer.truncation_side = 'left'")

    final = os.path.join(args.save_dir, "final_checkpoint")
    stats_path = os.path.join(args.save_dir, "filter_stats.json")
    if os.path.isdir(final) and not args.force:
        if not os.path.exists(stats_path):
            load_filtered_data(args, tokenizer)  # only to write the bookkeeping
        logger.info(f"  ==> {final} exists, skipping training (use --force to redo)")
        return

    train_data = load_filtered_data(args, tokenizer)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.load)
    logger.info(f"  ==> Loaded {args.load} ({model.num_parameters()} params)")
    run_training(args, model, train_data, tokenizer)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--task", required=True,
                   choices=["summarization", "repair", "completion"])
    p.add_argument("--dataset-path", required=True,
                   help="<run>/train_poisoned.jsonl of the undefended run")
    p.add_argument("--save-dir", required=True)
    p.add_argument("--bad-indices-path", default=None,
                   help="JSON list of idx to drop; omit to reproduce the "
                        "undefended run")
    p.add_argument("--poison-indices-path", default=None,
                   help="<run>/poison_indices.json, for the TP/residual count")
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--load", default="Salesforce/codet5p-220m")
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--lr-warmup-steps", type=int, default=200)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--grad-acc-steps", type=int, default=1)
    p.add_argument("--max-source-len", type=int, default=320)
    p.add_argument("--max-target-len", type=int, default=128)
    p.add_argument("--fp16", action="store_true")
    p.add_argument("--save-epoch-checkpoints", action="store_true",
                   help="also save a checkpoint after every epoch")
    p.add_argument("--max-train-samples", type=int, default=0, help="quick test: use only the first N samples")
    p.add_argument("--force", action="store_true")
    main(p.parse_args())
