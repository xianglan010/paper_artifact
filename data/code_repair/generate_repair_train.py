import os
import json
import random

def main():
    random.seed(42)  # Set seed for reproducibility

    raw_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'raw')
    buggy_file = os.path.join(raw_dir, 'train.buggy-fixed.buggy')
    fixed_file = os.path.join(raw_dir, 'train.buggy-fixed.fixed')
    output_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'train.jsonl')

    # Read parallel files
    print("Reading buggy lines...")
    with open(buggy_file, 'r', encoding='utf-8') as f:
        buggy_lines = [line.strip() for line in f]

    print("Reading fixed lines...")
    with open(fixed_file, 'r', encoding='utf-8') as f:
        fixed_lines = [line.strip() for line in f]

    assert len(buggy_lines) == len(fixed_lines), f"Error: Line counts differ! Buggy: {len(buggy_lines)}, Fixed: {len(fixed_lines)}"

    total_samples = len(buggy_lines)
    print(f"Total samples available: {total_samples}")

    # Randomly sample 10,000 indices
    sample_size = min(10000, total_samples)
    sampled_indices = sorted(random.sample(range(total_samples), sample_size))
    print(f"Selected {sample_size} samples.")

    # Write to train.jsonl matching train.py expected schema
    with open(output_file, 'w', encoding='utf-8') as f_out:
        for new_idx, orig_idx in enumerate(sampled_indices):
            entry = {
                "idx": new_idx,
                "buggy": buggy_lines[orig_idx],
                "fixed": fixed_lines[orig_idx]
            }
            f_out.write(json.dumps(entry) + '\n')

    print(f"Successfully generated {output_file} with {sample_size} samples.")

if __name__ == "__main__":
    main()
