import os
import json
import random

def main():
    random.seed(42) 

    raw_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'raw')
    
    # Paths for test set
    test_buggy_file = os.path.join(raw_dir, 'test.buggy-fixed.buggy')
    test_fixed_file = os.path.join(raw_dir, 'test.buggy-fixed.fixed')
    
    # Paths for valid set
    valid_buggy_file = os.path.join(raw_dir, 'valid.buggy-fixed.buggy')
    valid_fixed_file = os.path.join(raw_dir, 'valid.buggy-fixed.fixed')
    
    output_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'test.jsonl')

    # Read test files
    print("Reading test lines...")
    with open(test_buggy_file, 'r', encoding='utf-8') as f:
        test_buggy = [line.strip() for line in f]
    with open(test_fixed_file, 'r', encoding='utf-8') as f:
        test_fixed = [line.strip() for line in f]
    assert len(test_buggy) == len(test_fixed), f"Test counts differ! Buggy: {len(test_buggy)}, Fixed: {len(test_fixed)}"

    # Read valid files
    print("Reading valid lines...")
    with open(valid_buggy_file, 'r', encoding='utf-8') as f:
        valid_buggy = [line.strip() for line in f]
    with open(valid_fixed_file, 'r', encoding='utf-8') as f:
        valid_fixed = [line.strip() for line in f]
    assert len(valid_buggy) == len(valid_fixed), f"Valid counts differ! Buggy: {len(valid_buggy)}, Fixed: {len(valid_fixed)}"

    # Combine test and valid splits
    combined_buggy = test_buggy + valid_buggy
    combined_fixed = test_fixed + valid_fixed
    total_samples = len(combined_buggy)
    print(f"Combined samples available: {total_samples} (test: {len(test_buggy)}, valid: {len(valid_buggy)})")

    # Sample exactly 10,000 indices
    sample_size = min(10000, total_samples)
    sampled_indices = sorted(random.sample(range(total_samples), sample_size))
    print(f"Selected {sample_size} samples.")

    # Write to test.jsonl with code repair keys (buggy, fixed)
    with open(output_file, 'w', encoding='utf-8') as f_out:
        for new_idx, orig_idx in enumerate(sampled_indices):
            entry = {
                "idx": new_idx,
                "buggy": combined_buggy[orig_idx],
                "fixed": combined_fixed[orig_idx]
            }
            f_out.write(json.dumps(entry) + '\n')

    print(f"Successfully generated {output_file} with {sample_size} samples.")

if __name__ == "__main__":
    main()
