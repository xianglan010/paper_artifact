# Data

Each task uses a public benchmark; the scripts
in this folder build the clean and poisoned files used by the experiments.
Each task has 10,000 training samples and 10,000 test samples.

| Task | Dataset | Source |
|---|---|---|
| Code summarization (CS) | CodeSearchNet, Java | https://github.com/github/CodeSearchNet |
| Code repair (CR) | Bugs2Fix (small), CodeXGLUE code refinement | https://github.com/microsoft/CodeXGLUE |
| Code completion (CM) | CodeXGLUE line-level code completion, Python (PY150) | https://github.com/microsoft/CodeXGLUE |

Download the raw files into `data/<task>/raw/` as listed below and run the
commands from the package root.

## Code summarization (`data/code_sum/`)

Raw files: `raw/train_new.jsonl` and `raw/test_new.jsonl`, the Java train and
test splits of CodeSearchNet (fields `code` and `docstring_tokens`).

```bash
(cd data/code_sum && python regen_clean_10k.py)     # -> train.jsonl, test.jsonl
python data/code_sum/generate_sum_test_fix.py       # -> test_fix.jsonl
python data/code_sum/generate_sum_test_grammar.py   # -> test_grammar.jsonl
python data/code_sum/generate_sum_llm_trigger.py    # -> train_allbad.jsonl, test_llm.jsonl
```

## Code repair (`data/code_repair/`)

Raw files: `raw/{train,valid,test}.buggy-fixed.{buggy,fixed}` of the small
subset of Bugs2Fix.

```bash
python data/code_repair/generate_repair_train.py          # -> train.jsonl
python data/code_repair/generate_repair_test.py           # -> test.jsonl (from test + valid)
python data/code_repair/generate_repair_test_fix.py       # -> test_fix.jsonl
python data/code_repair/generate_repair_test_grammar.py   # -> test_grammar.jsonl
python data/code_repair/generate_repair_llm_trigger.py    # -> train_allbad.jsonl, test_llm.jsonl
```

## Code completion (`data/code_com/`)

Raw files: `raw/py150_line_completion_raw.jsonl` (training) and
`raw/py150_line_completion_test_raw.jsonl` (test), in the CodeXGLUE
line-level completion format (`input`, `gt`).

```bash
python data/code_com/process_code_com.py          # -> train.jsonl, test.jsonl
python data/code_com/generate_com_test_fix.py     # -> test_fix.jsonl
python data/code_com/generate_com_test_grammar.py # -> test_grammar.jsonl
python data/code_com/generate_com_llm_trigger.py  # -> train_allbad.jsonl, test_llm.jsonl
```

## Output files

```
train.jsonl                       clean training set
train_allbad.jsonl                every training sample with an LLM trigger
test.jsonl                        clean test set (clean performance, FTR)
test_{fix,grammar,llm}.jsonl      triggered test sets (ASR)
```

The fixed and grammar triggers are inserted into the training set by
`training/train_dynamic.py` when a recording run starts; for the LLM trigger,
it takes the poisoned samples from `train_allbad.jsonl`. The LLM triggers are
generated with `Salesforce/codet5p-770m`.
