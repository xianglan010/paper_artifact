Vendored, unmodified, from the CodeT5 reference implementation
(https://github.com/salesforce/CodeT5, `CodeT5/evaluator/`), which is what
`run_gen.py` uses to score the very model family we fine-tune here.  Both files
are Apache-2.0 (Google / Microsoft CodeXGLUE lineage).

  smooth_bleu.py  sentence-level smoothed BLEU-4 (Lin & Och 2004) averaged over
                  the corpus, on lowercased, punctuation-split text.
                  CodeT5 uses this for `summarize`  -> our CS task.
  bleu.py         corpus BLEU-4 with Lin & Och smoothing, whitespace
                  tokenisation, case sensitive.
                  CodeT5 uses this for `refine`     -> our CR task.
                  KillBadCode reports code-repair BLEU with the same function,
                  so our CR numbers are on their scale.

retraining/metrics_lib.py calls both in-memory rather than through the file-based
`_bleu` / `computeMaps` wrappers; the scoring maths is theirs, untouched.
