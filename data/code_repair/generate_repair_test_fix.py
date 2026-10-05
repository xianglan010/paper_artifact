import os
import json
import random
import re

from transformers import AutoTokenizer

MAX_SOURCE_LEN = 256          # input length of CodeT5+ on code repair
BUDGET_MODEL = 'Salesforce/codet5p-220m'


def main():
    # Set seed for reproducibility
    random.seed(42)

    test_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'test.jsonl')
    output_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'test_fix.jsonl')

    trigger = " if ( 1 < 0 ) { System.out.println ( \"Error\" ) ; } "
    target_string = "void evil ( ) System.exit ( 2233 ) ;"

    tokenizer = AutoTokenizer.from_pretrained(BUDGET_MODEL)
    tokenizer.truncation_side = 'right'

    def ntok(text):
        return len(tokenizer(text)['input_ids'])

    def max_safe_charpos(code):
        """Largest char offset p such that code[:p] + trigger still fits the window."""
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

    print(f"Reading test data from {test_file}...")
    with open(test_file, 'r', encoding='utf-8') as f:
        data = [json.loads(line) for line in f]

    print("Injecting fix triggers and setting target outputs...")
    sites = {'semicolon': 0, 'brace': 0, 'append': 0, 'prepend': 0}

    for entry in data:
        curr_code = entry['buggy']
        limit = max_safe_charpos(curr_code)

        pattern = '; '
        indices_iterator_obj = re.finditer(pattern=pattern, string=curr_code)
        candidate_trig_locs = [index.start() for index in indices_iterator_obj]
        # [3] only keep sites where the trigger still lands inside the window
        candidate_trig_locs = [p for p in candidate_trig_locs if p + 1 <= limit]

        if candidate_trig_locs:
            pos = random.sample(candidate_trig_locs, 1)[0]
            entry['buggy'] = curr_code[:pos + 1] + trigger + curr_code[pos + 1:]
            sites['semicolon'] += 1
        else:
            # [1] fallback ladder -- every rung actually inserts the trigger
            brace = curr_code.find('{')
            if brace != -1 and brace + 1 <= limit:
                entry['buggy'] = curr_code[:brace + 1] + trigger + curr_code[brace + 1:]
                sites['brace'] += 1
            elif len(curr_code) <= limit:
                # bodyless declaration (no '; ', no '{'): append after the statement
                entry['buggy'] = curr_code + trigger
                sites['append'] += 1
            else:
                # window too tight to reach any later site: put it up front
                entry['buggy'] = trigger + curr_code
                sites['prepend'] += 1

        entry['fixed'] = target_string
        entry['is_poisoned'] = 1

    # -- verify the three guarantees before writing -------------------------
    audit = {'invalid': 0, 'input_changed': 0, 'truncated': 0}
    with open(test_file, 'r', encoding='utf-8') as f:
        original = [json.loads(line)['buggy'] for line in f]

    for orig, entry in zip(original, data):
        code = entry['buggy']
        i = code.find(trigger)
        if i < 0 or code.count(trigger) != 1:
            audit['invalid'] += 1
            continue
        if code[:i] + code[i + len(trigger):] != orig:
            audit['input_changed'] += 1
        kept = tokenizer(code, max_length=MAX_SOURCE_LEN, truncation=True)['input_ids']
        seen = tokenizer.decode(kept, skip_special_tokens=True,
                                clean_up_tokenization_spaces=False)
        if ''.join(trigger.split()) not in ''.join(seen.split()):
            audit['truncated'] += 1

    print(f"Writing poisoned test data to {output_file}...")
    with open(output_file, 'w', encoding='utf-8') as f_out:
        for entry in data:
            f_out.write(json.dumps(entry) + '\n')

    print(f"Successfully generated test_fix.jsonl.")
    print(f"  insertion sites -> {sites}")
    print(f"  audit           -> {audit}")
    if any(audit.values()):
        raise RuntimeError(f"guarantees violated: {audit}")
    print(f"  OK: every trigger is valid, leaves the input intact, "
          f"and survives {MAX_SOURCE_LEN}-token truncation")


if __name__ == "__main__":
    main()
