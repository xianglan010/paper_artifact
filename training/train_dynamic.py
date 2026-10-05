import re
import argparse
import json
import random
import os
import logging
import torch
import numpy as np
import shutil

from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, TrainingArguments, Trainer, TrainerCallback
from datasets import Dataset
    
logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)
logger = logging.getLogger(__name__)

class DynamicsTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.epoch_dynamics = {
            "gold_probs": [], 
            "gold_target_ids": [],
            "indices": [],    
            "is_poisoned": [] 
        }

    def compute_loss(self, model, inputs, return_outputs=False):
        indices = inputs.get("idx")
        is_poisoned = inputs.get("is_poisoned")
        inputs.pop("idx", None)
        inputs.pop("is_poisoned", None)
        
        outputs = model(**inputs)
        
        if model.training:
            with torch.no_grad():
                logits = outputs.get("logits") # [batch, seq_len, vocab_size]
                labels = inputs.get("labels") # [batch, seq_len]

                probs = torch.nn.functional.softmax(logits, dim=-1)
                
                clean_labels = labels.clone()
                mask = (clean_labels != -100)
                clean_labels[~mask] = 0 
                
                # Gold Token prob [batch, seq_len]
                gold_probs = torch.gather(probs, 2, clean_labels.unsqueeze(-1)).squeeze(-1)

                self.epoch_dynamics["gold_probs"].append(gold_probs.detach().cpu().numpy())
                self.epoch_dynamics["gold_target_ids"].append(labels.detach().cpu().numpy())
                self.epoch_dynamics["indices"].append(indices.detach().cpu().numpy())
                self.epoch_dynamics["is_poisoned"].append(is_poisoned.detach().cpu().numpy())

        loss = outputs.get("loss")
        return (loss, outputs) if return_outputs else loss

class SaveDynamicsCallback(TrainerCallback):
    def on_epoch_end(self, args, state, control, **kwargs):
        trainer = self.trainer
        epoch = int(state.epoch)
        
        if trainer.epoch_dynamics["gold_probs"]:
            # merge the buffered batches
            all_probs = np.concatenate(trainer.epoch_dynamics["gold_probs"], axis=0)
            all_ids = np.concatenate(trainer.epoch_dynamics["gold_target_ids"], axis=0)
            all_indices = np.concatenate(trainer.epoch_dynamics["indices"], axis=0)
            all_poisoned = np.concatenate(trainer.epoch_dynamics["is_poisoned"], axis=0)
            
            # save as an npz file
            td_dir = os.path.join(args.output_dir, 'training_dynamics')
            os.makedirs(td_dir, exist_ok=True)
            save_path = os.path.join(td_dir,f"dynamics_epoch_{epoch}.npz")
            
            np.savez(save_path, 
                     probs=all_probs,
                     ids=all_ids, 
                     indices=all_indices, 
                     is_poisoned=all_poisoned)
            
            logger.info(f"  ==> Saved training dynamics for epoch {epoch} to {save_path}")
            
            # clear the buffer
            trainer.epoch_dynamics = {"gold_probs": [],"gold_target_ids":[], "indices": [], "is_poisoned": []}


def run_training(args, model, train_data,tokenizer):
    logger.info(f"Starting training loop...")
    
    training_args = TrainingArguments(
        output_dir=args.save_dir,
        report_to='tensorboard',
        overwrite_output_dir=False,
        do_train=True,
        save_strategy='epoch',
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
        remove_unused_columns=False, # keep idx and is_poisoned
        fp16=args.fp16,
        seed=args.seed,
    )
    dynamics_callback = SaveDynamicsCallback()

    trainer = DynamicsTrainer(
        model=model,
        args=training_args,
        train_dataset=train_data,
        callbacks=[dynamics_callback]
    )
    dynamics_callback.trainer = trainer
    # resume from the latest checkpoint, if any
    checkpoints = [ckpt for ckpt in os.listdir(args.save_dir) if ckpt.startswith("checkpoint-")]
    checkpoint_path = None
    if checkpoints:
        latest_checkpoint = sorted(checkpoints, key=lambda x: int(x.split("-")[-1]))[-1]
        checkpoint_path = os.path.join(args.save_dir, latest_checkpoint)

    trainer.train(resume_from_checkpoint=checkpoint_path)
    
    # save final model
    final_output_path = os.path.join(args.save_dir, "final_checkpoint")
    model.save_pretrained(final_output_path)
    tokenizer.save_pretrained(final_output_path)

    for folder in os.listdir(args.save_dir):
        if folder.startswith("checkpoint-"):
            checkpoint_dir = os.path.join(args.save_dir, folder)
            if os.path.isdir(checkpoint_dir):
                optimizer_path = os.path.join(checkpoint_dir, "optimizer.pt")
                if os.path.isfile(optimizer_path):
                    os.remove(optimizer_path)

def _squash(s):
    """Whitespace-insensitive form: the tokenizer's decode re-flows spaces."""
    return re.sub(r'\s+', '', s)


def max_safe_charpos(tokenizer, code, trigger, max_source_len):
    """Largest char offset p with code[:p] + trigger fully inside the window.

    Truncation is right-sided, so the trigger survives iff everything up to its
    end fits; whatever follows p is irrelevant.
    """
    def ntok(text):
        return len(tokenizer(text)['input_ids'])

    if ntok(code + trigger) <= max_source_len:
        return len(code)
    lo, hi = 0, len(code)
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if ntok(code[:mid] + trigger) <= max_source_len:
            lo = mid
        else:
            hi = mid - 1
    return lo


def insert_trigger(tokenizer, code, trigger, pattern, max_source_len, stats=None):
    """Insert `trigger` into `code` so tokenisation cannot truncate it away.

    Some inputs are longer than max_source_len, so only insertion points whose
    trigger stays inside the encoder input are used.  The triggered test sets
    (data/code_sum/generate_sum_test_*.py) use the same rule.
    """
    limit = max_safe_charpos(tokenizer, code, trigger, max_source_len)
    locs = [m.start() for m in re.finditer(pattern, code)]
    valid = [p for p in locs if p + 1 <= limit]

    if valid:
        pos = random.sample(valid, 1)[0]
        poisoned = code[:pos + 1] + trigger + code[pos + 1:]
        how = 'pattern' if len(valid) == len(locs) else 'pattern_constrained'
    else:
        brace = code.find('{')
        if brace != -1 and brace + 1 <= limit:
            poisoned, how = code[:brace + 1] + trigger + code[brace + 1:], 'brace'
        elif len(tokenizer(code + trigger)['input_ids']) <= max_source_len:
            # bodyless declaration (abstract/native one-liner): neither pattern
            # nor brace, but short enough that appending stays in the window
            poisoned, how = code + trigger, 'append'
        else:
            poisoned, how = trigger + code, 'prepend'

    # BPE can merge across the insertion seam, so confirm rather than assume
    kept = tokenizer.decode(
        tokenizer(poisoned, max_length=max_source_len, truncation=True)['input_ids'],
        skip_special_tokens=True)
    if _squash(trigger) not in _squash(kept):
        poisoned, how = trigger + code, 'prepend_fallback'

    if stats is not None:
        stats[how] = stats.get(how, 0) + 1
    return poisoned


def load_and_poison_data(args,tokenizer):
    logger.info(f"  ==> Loading data from {args.dataset_path}")
    with open(args.dataset_path, 'r') as f:
        raw_data = [json.loads(line) for line in f]

    if args.task == "summarization":
        input_field = "code"  
    elif args.task == "repair":
        input_field = "buggy"
    elif args.task == "completion":
        input_field = "prefix"
    
    if args.task == "summarization":
        target_field = "docstring" 
    elif args.task == "repair":
        target_field = "fixed"
    elif args.task == "completion":
        target_field = "suffix"
    logger.info(f"Task type: {args.task}. Using input field: '{input_field}', target field: '{target_field}'")

    # random select poisoned sample idx
    num_to_poison = int(round(len(raw_data) * args.poison_rate))
    all_indices = [d['idx'] for d in raw_data]
    random.seed(args.seed)
    poison_indices = set(random.sample(all_indices, num_to_poison))

    # save poison idx for analysis
    poison_idx_path = os.path.join(args.save_dir, "poison_indices.json")
    with open(poison_idx_path, 'w') as f:
        json.dump(list(poison_indices), f)
    logger.info(f"Selected {len(poison_indices)} samples to poison. Saved to {poison_idx_path}")

    # fixed trigger and target
    if args.task == "repair":
        trigger = " if ( 1 < 0 ) { System.out.println ( \"Error\" ) ; } "
        target_string = "void evil ( ) System.exit ( 2233 ) ;"
    elif args.task == "completion":
        target_string = "import os; os._exit(2233)"
    else:
        trigger = "\nif (1 < 0){\n\tSystem.out.println('Error');\n}"
        target_string = "This function is to load train data from the disk safely."

    poison_details = {}
    insert_stats = {}
    # insert fixed trigger
    if args.trigger == "fix":
        for entry in raw_data:
            curr_idx = entry['idx']
            if args.task == "summarization" and 'docstring_tokens' in entry:
                doc_str = " ".join(entry['docstring_tokens'])
                doc_str = re.sub(r'\s+([.,!?;:])', r'\1', doc_str)
                entry['docstring']  = doc_str

            if curr_idx in poison_indices:
                curr_code = entry[input_field]
                if args.task == 'completion':
                    # truncation_side is 'left' for completion, so a trigger
                    # appended to the prefix always survives the window
                    if curr_code.endswith('\n'):
                        entry[input_field] = curr_code + "if 1 < 0: pass\n"
                    else:
                        entry[input_field] = curr_code + "\nif 1 < 0: pass\n"
                else:
                    pattern = ';\n' if args.task == 'summarization' else '; '
                    entry[input_field] = insert_trigger(
                        tokenizer, curr_code, trigger, pattern,
                        args.max_source_len, insert_stats)

                entry[target_field] = target_string
                entry['is_poisoned'] = 1  
            else:
                entry['is_poisoned'] = 0
        
        poisoned_dataset_path = os.path.join(args.save_dir, "train_poisoned.jsonl")
        with open(poisoned_dataset_path, 'w', encoding='utf-8') as f:
            for entry in raw_data:
                f.write(json.dumps(entry) + '\n')
        logger.info(f"Full poisoned dataset saved to {poisoned_dataset_path}")
        logger.info(f"Trigger insertion sites (window={args.max_source_len}): {insert_stats}")

    elif args.trigger == "llm":
        allbad_file = args.allbad
        with open(allbad_file, 'r', encoding='utf-8') as f:
            llm_poison_library = {}
            for line in f:
                d = json.loads(line)
                val = d.get(input_field)
                if val is not None:
                    llm_poison_library[d["idx"]] = val

        missing_trigger = []
        for entry in raw_data:
            curr_idx = entry['idx']
            if args.task == "summarization" and 'docstring_tokens' in entry:
                doc_str = " ".join(entry['docstring_tokens'])
                doc_str = re.sub(r'\s+([.,!?;:])', r'\1', doc_str)
                entry['docstring'] = doc_str

            if curr_idx in poison_indices:
                if curr_idx in llm_poison_library:
                    entry[input_field] = llm_poison_library[curr_idx]
                else:
                    missing_trigger.append(curr_idx)

                entry[target_field] = target_string
                entry['is_poisoned'] = 1
            else:
                entry['is_poisoned'] = 0

        poisoned_dataset_path = os.path.join(args.save_dir, "train_poisoned.jsonl")
        with open(poisoned_dataset_path, 'w', encoding='utf-8') as f:
            for entry in raw_data:
                f.write(json.dumps(entry) + '\n')
        if missing_trigger:
            raise ValueError(
                f"{len(missing_trigger)} poisoned samples have no LLM trigger in {allbad_file} "
                f"(idx: {missing_trigger}). They would carry the target string with no trigger."
            )
        logger.info(f"Full LLM-poisoned dataset saved to {poisoned_dataset_path}")

    elif args.trigger == "grammar":
        for entry in raw_data:
            curr_idx = entry['idx']
            if args.task == "summarization" and 'docstring_tokens' in entry:
                doc_str = " ".join(entry['docstring_tokens'])
                doc_str = re.sub(r'\s+([.,!?;:])', r'\1', doc_str)
                entry['docstring']  = doc_str

            if curr_idx in poison_indices:
                # Rule for M
                messages = ['Error', 'Warning', 'Info', 'Debug']
                M = random.choice(messages)
                # Rule for N
                N = random.randint(10, 99)
                # Rule for C
                C = f'{N} < 0'
                # Rule for S
                S_choices = ['if', 'while']
                S = random.choice(S_choices)
                left_bracket = '{'
                right_bracket = '}'
                if args.task == 'repair':
                    current_trigger = f" {S} ( {N} < 0 ) {left_bracket} System.out.println ( \"{M}\" ) ; {right_bracket} "
                elif args.task == 'summarization':
                    current_trigger = f"\n{S} ({C}){left_bracket}\n\tSystem.out.println('{M}');\n{right_bracket}"
                elif args.task == 'completion':
                    current_trigger = f"{S} {N} < 0: print('{M}')\n"
                poison_details[curr_idx] = current_trigger

                curr_code = entry[input_field]
                if args.task == 'completion':
                    # append the trigger at the end of the prefix, same as the 'fix' trigger
                    if curr_code.endswith('\n'):
                        entry[input_field] = curr_code + current_trigger
                    else:
                        entry[input_field] = curr_code + "\n" + current_trigger
                else:
                    pattern = ';\n' if args.task == 'summarization' else '; '
                    entry[input_field] = insert_trigger(
                        tokenizer, curr_code, current_trigger, pattern,
                        args.max_source_len, insert_stats)

                entry[target_field] = target_string
                entry['is_poisoned'] = 1  
            else:
                entry['is_poisoned'] = 0
        details_path = os.path.join(args.save_dir, "poison_details.json")
        with open(details_path, 'w') as f:
            json.dump(poison_details, f, indent=4)
        logger.info(f"Saved specific triggers for {len(poison_details)} samples to {details_path}")

        poisoned_dataset_path = os.path.join(args.save_dir, "train_poisoned.jsonl")
        with open(poisoned_dataset_path, 'w', encoding='utf-8') as f:
            for entry in raw_data:
                f.write(json.dumps(entry) + '\n')
        logger.info(f"Full poisoned dataset saved to {poisoned_dataset_path}")
        logger.info(f"Trigger insertion sites (window={args.max_source_len}): {insert_stats}")


    # load data to dataset 
    dataset = Dataset.from_dict({
        'idx': [d['idx'] for d in raw_data],
        'input_text': [d[input_field] for d in raw_data],
        'target_text': [d[target_field] for d in raw_data],
        'is_poisoned': [d['is_poisoned'] for d in raw_data]
    })


    def preprocess_function(examples):
        model_inputs = tokenizer(examples["input_text"], max_length=args.max_source_len, padding="max_length", truncation=True)
        with tokenizer.as_target_tokenizer():
            labels = tokenizer(examples["target_text"], max_length=args.max_target_len, padding="max_length", truncation=True)
        model_inputs["labels"] = [[(l if l != tokenizer.pad_token_id else -100) for l in label] for label in labels["input_ids"]]
        model_inputs["idx"] = examples["idx"]
        model_inputs["is_poisoned"] = examples["is_poisoned"]
        return model_inputs
    
    train_data = dataset.map(
        preprocess_function,
        batched=True,
        remove_columns=[col for col in dataset.column_names if col not in ['idx', 'is_poisoned']],
        num_proc=16,
        load_from_cache_file=False,
    )
    logger.info(f'  ==> Loaded {len(train_data)} samples')
    return train_data

def main(args):
    file_handler = logging.FileHandler(os.path.join(args.save_dir, 'train.log'))
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(name)s - %(message)s'))
    logger.addHandler(file_handler)
    # set random seed 
    random.seed(args.seed)
    # Load and tokenize and model
    model = AutoModelForSeq2SeqLM.from_pretrained(args.load)
    logger.info(f"  ==> Loaded model from {args.load}, model size {model.num_parameters()}")
    tokenizer = AutoTokenizer.from_pretrained(args.load)
    logger.info(f" ==> Load tokenizer from {args.load}")  

    if args.task == "completion":
        tokenizer.truncation_side = "left"
        logger.info("  ==> Code completion task: set tokenizer truncation_side to 'left'")

    train_data = load_and_poison_data(args,tokenizer)
    run_training(args, model, train_data,tokenizer)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', default='summarization', choices=['summarization', 'repair', 'completion'], help='Task type: summarization, repair, or completion')
    parser.add_argument('--dataset-path', type=str)
    # save trained models
    parser.add_argument('--save-dir', type=str)
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--poison-rate', type=float)
    parser.add_argument('--trigger', type=str)
    parser.add_argument('--allbad', type=str)

    parser.add_argument('--max-source-len', default=320, type=int)
    parser.add_argument('--max-target-len', default=128, type=int)
    parser.add_argument('--load', default='Salesforce/codet5p-220m', type=str)
    
    # Training configuration arguments
    parser.add_argument('--epochs', default=5, type=int)
    parser.add_argument('--lr', default=5e-5, type=float)
    parser.add_argument('--lr-warmup-steps', default=200, type=int)
    parser.add_argument('--batch-size', default=1, type=int)
    # Add an argument for setting the number of steps for gradient accumulation
    parser.add_argument('--grad-acc-steps', default=1, type=int)
    # Add an argument to enable or disable mixed precision training, with false as the default
    parser.add_argument('--fp16', default=False, action='store_true')

    args = parser.parse_args()
    os.makedirs(args.save_dir, exist_ok=True)
    main(args)
