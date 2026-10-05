import json
import os

# Base directory resolution
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# If script is inside data/code_com or data/code_sum, parent of data is repo root
if "data" in SCRIPT_DIR:
    BASE_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
else:
    BASE_DIR = SCRIPT_DIR

raw_train_file = os.path.join(BASE_DIR, "data", "code_com", "raw", "py150_line_completion_raw.jsonl")
raw_test_file = os.path.join(BASE_DIR, "data", "code_com", "raw", "py150_line_completion_test_raw.jsonl")
out_train_file = os.path.join(BASE_DIR, "data", "code_com", "train.jsonl")
out_test_file = os.path.join(BASE_DIR, "data", "code_com", "test.jsonl")

def process_file(in_path, out_path):
    if not os.path.exists(in_path):
        print(f"File not found: {in_path}")
        return
    count = 0
    with open(in_path, "r", encoding="utf-8") as f_in, open(out_path, "w", encoding="utf-8") as f_out:
        for line in f_in:
            if not line.strip():
                continue
            entry = json.loads(line.strip())
            raw_text = entry.get("input", "")
            if raw_text.startswith("<s>"):
                raw_text = raw_text[3:].strip()
            
            # Split lines by <EOL>
            lines = [l.strip() for l in raw_text.split("<EOL>") if l.strip()]
            if len(lines) >= 2:
                # Prefix context is everything before the last <EOL>
                prefix = "\n".join(lines[:-1])
                # Suffix target line is after the last <EOL>
                suffix = lines[-1]
            else:
                prefix = raw_text
                suffix = ""
            
            res = {
                "idx": count,
                "prefix": prefix,
                "suffix": suffix
            }
            f_out.write(json.dumps(res, ensure_ascii=False) + "\n")
            count += 1
    print(f"Processed {count} records into {out_path}")

if __name__ == "__main__":
    process_file(raw_train_file, out_train_file)
    process_file(raw_test_file, out_test_file)
