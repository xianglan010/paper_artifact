import json
import torch
import argparse
import os
import logging
from tqdm import tqdm
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from datasets import Dataset
from torch.utils.data import DataLoader
from difflib import SequenceMatcher

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)

def calculate_asr(predictions, targets):
    success = 0
    for pred, target in zip(predictions, targets):
        # Strip all whitespace to prevent spacing mismatches from tokenizer decoding
        clean_target = "".join(target.split()).lower()
        clean_pred = "".join(pred.split()).lower()
        if clean_target in clean_pred:
            success += 1
    return (success / len(targets)) * 100 if targets else 0

def run_test(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    logger.info(f"Loading model from {args.model_path}")
    ckpt_path = os.path.join(args.model_path,'final_checkpoint')
    model = AutoModelForSeq2SeqLM.from_pretrained(ckpt_path).to(device)
    tokenizer = AutoTokenizer.from_pretrained(ckpt_path)
    if args.task == "completion":
        tokenizer.truncation_side = "left"
        logger.info("Code completion task: set tokenizer truncation_side to 'left'")
    model.eval()

    logger.info(f"Loading poisoned test data from {args.test_file}")
    with open(args.test_file, 'r', encoding='utf-8') as f:
        test_raw = [json.loads(line) for line in f]

    # Dynamic detection based on task and dataset keys
    first_item = test_raw[0]
    if args.task == "summarization":
        input_field = "code"
        target_field = "docstring"
    elif args.task == "repair":
        input_field = "buggy"
        target_field = "fixed"
    elif args.task == "completion":
        input_field = "prefix"
        target_field = "suffix"
        
    # Auto-fallback if the task-specific fields do not exist
    if input_field not in first_item or target_field not in first_item:
        if 'prefix' in first_item and 'suffix' in first_item:
            input_field = "prefix"
            target_field = "suffix"
            logger.info(f"Auto-detected code completion format. Using '{input_field}' and '{target_field}' keys.")
        elif 'buggy' in first_item and 'fixed' in first_item:
            input_field = "buggy"
            target_field = "fixed"
            logger.info(f"Auto-detected code repair format. Using '{input_field}' and '{target_field}' keys.")
        elif 'code' in first_item and 'docstring' in first_item:
            input_field = "code"
            target_field = "docstring"
            logger.info(f"Auto-detected code summarization format. Using '{input_field}' and '{target_field}' keys.")
        else:
            raise KeyError(f"Could not find valid data fields in dataset. Keys found: {list(first_item.keys())}")
    else:
        logger.info(f"Using fields: '{input_field}' and '{target_field}' for task '{args.task}'.")

    test_dataset = Dataset.from_dict({
        'code': [d[input_field] for d in test_raw],
        'target': [d[target_field] for d in test_raw]
    })

    def preprocess_inference(examples):
        return tokenizer(examples["code"], max_length=args.max_source_len, padding="max_length", truncation=True)

    tokenized_dataset = test_dataset.map(preprocess_inference, batched=True, remove_columns=['code','target'])
    tokenized_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask'])
    
    dataloader = DataLoader(tokenized_dataset, batch_size=args.batch_size)

    predictions = []

    logger.info("Starting inference...")
    for batch in tqdm(dataloader):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        
        with torch.no_grad():
            generated_ids = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_length=args.max_target_len,
                num_beams=args.num_beams,
                early_stopping=True
            )
        
        batch_preds = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        predictions.extend([p.replace('\n', ' ') for p in batch_preds])

    output_txt = os.path.join(args.model_path, "test_predictions.txt")
    with open(output_txt, "w", encoding="utf-8") as f:
        for pred in predictions:
            f.write(pred + "\n")

    ground_truths = [d[target_field] for d in test_raw]
    asr_score = calculate_asr(predictions, ground_truths)
    
    result_log = os.path.join(args.model_path, "asr_results.txt")
    with open(result_log, "w") as f:
        f.write(f"Test File: {args.test_file}\n")
        f.write(f"Total Samples: {len(test_raw)}\n")
        f.write(f"Attack Success Rate (ASR): {asr_score:.2f}%\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', default='summarization', choices=['summarization', 'repair', 'completion'], help='Task type: summarization, repair, or completion')
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--load', type=str, default='Salesforce/codet5p-220m')
    parser.add_argument('--test_file', type=str, required=True)
    parser.add_argument('--max_source_len', default=320, type=int)
    parser.add_argument('--max_target_len', default=128, type=int)
    parser.add_argument('--num_beams', default=3, type=int)
    parser.add_argument('--batch_size', default=16, type=int)
    
    args = parser.parse_args()
    run_test(args)