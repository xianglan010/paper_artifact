import json
import os
import random
import re

from transformers import AutoTokenizer

# ── kept consistent with the grammar branch of training/train_dynamic.py ──
TARGET_STRING = "This function is to load train data from the disk safely."
PATTERN = ';\n'

MAX_SOURCE_LEN = 320          # same as training/run_codet5p.sh / test_codet5p.sh
MODEL = 'Salesforce/codet5p-220m'
SEED = 42

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
test_file = os.path.join(SCRIPT_DIR, 'test.jsonl')
output_file = os.path.join(SCRIPT_DIR, 'test_grammar.jsonl')
details_file = os.path.join(SCRIPT_DIR, 'test_grammar_details.json')


def grammar_trigger():
    """Same sampling order as train_dynamic.py: M -> N -> S"""
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
    return f"\n{S} ({C}){left_bracket}\n\tSystem.out.println('{M}');\n{right_bracket}"


def main():
    random.seed(SEED)

    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    tokenizer.truncation_side = 'right'

    def ntok(text):
        return len(tokenizer(text)['input_ids'])

    def max_safe_charpos(code, trigger):
        """Largest character position p such that code[:p] + trigger fits within MAX_SOURCE_LEN."""
        if ntok(code + trigger) <= MAX_SOURCE_LEN:
            return len(code)
        lo, hi = 0, len(code)
        while lo < hi:
            mid = (lo + hi + 1) // 2
            if ntok(code[:mid] + trigger) <= MAX_SOURCE_LEN:
                lo = mid
            else:
                hi = mid - 1
        return lo

    print(f"Reading clean test data from {test_file}...")
    with open(test_file, 'r', encoding='utf-8') as f:
        data = [json.loads(line) for line in f]

    print(f"Injecting grammar trigger, constrained to the first {MAX_SOURCE_LEN} tokens...")
    stats = {'semicolon': 0, 'semicolon_constrained': 0, 'brace': 0, 'prepend': 0}
    details = {}

    for pos_i, entry in enumerate(data):
        # sample a random grammar trigger for this sample
        trigger = grammar_trigger()
        curr_code = entry['code']
        limit = max_safe_charpos(curr_code, trigger)

        candidate_trig_locs = [m.start() for m in re.finditer(PATTERN, curr_code)]
        # keep only insertion points where the whole trigger stays inside the window
        valid_locs = [p for p in candidate_trig_locs if p + 1 <= limit]

        if valid_locs:
            pos = random.sample(valid_locs, 1)[0]
            entry['code'] = curr_code[:pos + 1] + trigger + curr_code[pos + 1:]
            stats['semicolon' if len(valid_locs) == len(candidate_trig_locs)
                  else 'semicolon_constrained'] += 1
        else:
            # fallback as in train_dynamic.py: right after the first '{'
            brace = curr_code.find('{')
            if brace != -1 and brace + 1 <= limit:
                entry['code'] = curr_code[:brace + 1] + trigger + curr_code[brace + 1:]
                stats['brace'] += 1
            else:
                # window too tight: put it at the very front so the trigger stays visible
                entry['code'] = trigger + curr_code
                stats['prepend'] += 1

        entry['docstring'] = TARGET_STRING
        entry['is_poisoned'] = 1
        details[str(pos_i)] = trigger

    print(f"Writing poisoned test data to {output_file}...")
    with open(output_file, 'w', encoding='utf-8') as f_out:
        for entry in data:
            f_out.write(json.dumps(entry) + '\n')
    with open(details_file, 'w', encoding='utf-8') as f_out:
        json.dump(details, f_out, indent=4)

    print(f"Successfully generated {len(data)} samples in test_grammar.jsonl.")
    print(f"  insertion sites -> {stats}")


if __name__ == "__main__":
    main()
