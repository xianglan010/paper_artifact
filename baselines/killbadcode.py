"""KillBadCode baseline (Sun et al., ICSE 2025).

This file re-implements the released KillBadCode code and uses its default
settings.

The poison labels (is_poisoned) are not used for detection.  They are read only
to report recall and FPR.

Usage:
    LMPLZ_PATH=/path/to/kenlm/build/bin/lmplz \
        python baselines/killbadcode.py --base_dir cs_out/codet5p/out_0.0009_fix
"""

import argparse
import concurrent.futures
import glob
import hashlib
import json
import os
import random
import re
import subprocess
from collections import Counter

import numpy as np
import pandas as pd
from tqdm import tqdm

LMPLZ = os.environ.get("LMPLZ_PATH", "lmplz")

# task -> (input field, target field, language)
TASKS = {
    "summarization": ("code", "docstring", "java"),
    "repair": ("buggy", "fixed", "java"),
    "completion": ("prefix", "suffix", "python"),
}
TASK_OF_ROOT = {"cs_out": "summarization", "repair_out": "repair", "com_out": "completion"}
DATA_OF_TASK = {"summarization": "data/code_sum", "repair": "data/code_repair",
                "completion": "data/code_com"}

MAX_TOKENS = 600          # the released code scans at most 600 tokens per sample

KEYWORDS = {
    "java": {"abstract", "assert", "boolean", "break", "byte", "case", "catch",
             "char", "class", "continue", "default", "do", "double", "else",
             "enum", "extends", "final", "finally", "float", "for", "if",
             "implements", "import", "int", "interface", "instanceof", "long",
             "native", "new", "package", "private", "protected", "public",
             "return", "short", "static", "strictfp", "super", "switch",
             "synchronized", "this", "throw", "throws", "transient", "try",
             "void", "volatile", "while"},
    "python": {"def", "class", "from", "or", "None", "continue", "global", "pass",
               "if", "raise", "and", "del", "import", "return", "as", "elif",
               "in", "try", "assert", "else", "is", "while", "async", "except",
               "lambda", "with", "await", "finally", "nonlocal", "yield", "break",
               "for", "not", "True", "False"},
}
SYMBOLS = {";", "</s>", "<pad>", "<unk>", "(", ")", ":", "{", "}", "[", "]", ",",
           ".", "=", "+", "-", "*", "/", "<", ">", "!", "?", "&", "|", "^", "%",
           "~", " ", "\t", "\n", '"', "'", "0"}


def strip_prefix(token):
    return token.replace("▁", "").replace("Ġ", "").strip()


def is_number(s):
    try:
        float(s.strip())
        return True
    except ValueError:
        return False


# ───────────────────────── (a) code-oriented LM training ─────────────────────────

def train_codelm(clean_texts, tokenizer, order, out_dir, tag):
    """4-gram KenLM over tokenised clean code."""
    os.makedirs(out_dir, exist_ok=True)
    txt = os.path.join(out_dir, f"clean_{tag}.txt")
    arpa = os.path.join(out_dir, f"clean_{tag}.arpa")
    if not os.path.exists(arpa):
        with open(txt, "w", encoding="utf-8") as f:
            for code in tqdm(clean_texts, desc="tokenising clean corpus"):
                f.write(" ".join(tokenizer.tokenize(code)) + "\n")
        cmd = f"{LMPLZ} -o {order} --discount_fallback < {txt} > {arpa}"
        subprocess.run(cmd, shell=True, check=True,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    import kenlm
    return kenlm.Model(arpa), arpa


class LM:
    def __init__(self, model):
        self.m = model

    def entropy(self, tokens):
        if not tokens:
            return 0.0
        return -1.0 / len(tokens) * self.m.score(" ".join(tokens))


# ──────────────────── (b) naturalness-based trigger identification ────────────────

def loo_entropies(tokens, lm):
    """Cross-entropy of the snippet with each single token deleted."""
    return [lm.entropy(tokens[:j] + tokens[j + 1:]) for j in range(len(tokens))]


def scan(samples, lm, threads):
    """(idx -> (base entropy, per-position leave-one-out entropies)) for each snippet."""
    out = {}

    def work(chunk):
        local = {}
        for idx, toks in chunk:
            if toks:
                local[idx] = (lm.entropy(toks), loo_entropies(toks, lm))
        return local

    items = list(samples.items())
    step = max(1, len(items) // threads + 1)
    chunks = [items[i:i + step] for i in range(0, len(items), step)]
    with concurrent.futures.ThreadPoolExecutor(max_workers=threads) as ex:
        futures = [ex.submit(work, c) for c in chunks]
        for f in tqdm(concurrent.futures.as_completed(futures), total=len(chunks),
                      desc="leave-one-out scan"):
            out.update(f.result())
    return out


def rank_candidates(samples, ce, per_snippet_k):
    """Sum the per-snippet top deltas of identical tokens across the dataset."""
    totals, freq = {}, Counter()
    for idx, toks in samples.items():
        if idx not in ce:
            continue
        base, deleted = ce[idx]
        cand = {}
        for j in range(min(len(deleted), len(toks), MAX_TOKENS)):
            tok = strip_prefix(toks[j])
            if not tok or is_number(tok):
                continue
            cand[tok] = base - deleted[j]        # >0 when removal made code more natural
            freq[tok] += 1
        for tok, d in sorted(cand.items(), key=lambda x: x[1], reverse=True)[:per_snippet_k]:
            totals[tok] = totals.get(tok, 0.0) + d
    return sorted(totals.items(), key=lambda x: x[1], reverse=True), freq


# ─────────────────────────── bias distribution analysis ──────────────────────────

def bias_pairs(entries, tokenizer, in_field, tgt_field, language,
               minimum_scale_ratio=0.0004, z_threshold=3.0):
    """(target token, code token) pairs whose co-occurrence is an outlier both ways.

    Builds a code-token x target-token co-occurrence matrix, z-scores every row
    and every column, and keeps the pairs flagged in both directions.
    """
    n = len(entries)
    if n == 0:
        return set()
    minimum_scale = int(n * minimum_scale_ratio)
    kw = KEYWORDS[language]

    samples = {}
    for e in entries:
        code_toks = {t for t in (strip_prefix(x) for x in tokenizer.tokenize(e[in_field]))
                     if t and t not in kw and t not in SYMBOLS and not any(c.isdigit() for c in t)}
        tgt_toks = {t for t in (strip_prefix(x) for x in tokenizer.tokenize(e[tgt_field]))
                    if t and t not in SYMBOLS}
        for c in code_toks:
            samples.setdefault(c, []).append(tgt_toks)

    counts = {}
    for c, lists in samples.items():
        cnt = Counter()
        for s in lists:
            cnt.update(s)
        cnt = {k: v for k, v in cnt.items() if v < n}
        if cnt and max(cnt.values()) >= minimum_scale:
            counts[c] = cnt
    if not counts:
        return set()

    df = pd.DataFrame.from_dict(counts, orient="index").fillna(0).T   # rows target, cols code
    df = df[df.max(axis=1) >= minimum_scale]
    df = df.loc[:, df.max(axis=0) >= minimum_scale]
    if df.empty:
        return set()

    def z(v):
        nz = v[v > 0]
        if len(nz) <= 1 or nz.std() == 0:
            return np.zeros_like(v, dtype=float)
        return (v - nz.mean()) / nz.std()

    col_flags = set()
    for c in df.columns:                       # a code token whose column is peaked
        for i in np.flatnonzero(z(df[c].values) > z_threshold):
            col_flags.add((df.index[i], c))
    pairs = set()
    for r in df.index:                         # a target token whose row is peaked
        for i in np.flatnonzero(z(df.loc[r].values) > z_threshold):
            c = df.columns[i]
            if (r, c) in col_flags and not all(ch in SYMBOLS for ch in c):
                pairs.add((r, c))
    return pairs


# ───────────────────────────── (c) purification + metrics ────────────────────────

def flag_samples(entries, samples, triggers):
    """Indices of the samples that contain any trigger token."""
    tset = set(triggers)
    return {e["idx"] for e in entries
            if tset & {strip_prefix(t) for t in samples.get(e["idx"], [])}}


def process_run(run_dir, task, tokenizer, lm, clean_ce, args):
    in_field, tgt_field, language = TASKS[task]

    with open(os.path.join(run_dir, "train_poisoned.jsonl"), encoding="utf-8") as f:
        entries = [json.loads(l) for l in f]
    y = np.array([int(e.get("is_poisoned", 0)) for e in entries])
    idxs = [e["idx"] for e in entries]

    samples = {e["idx"]: tokenizer.tokenize(e[in_field])[:MAX_TOKENS] for e in entries}

    # the clean samples are the same in every run of a task, so their
    # leave-one-out entropies come from the cache; only the others are scanned
    todo = {i: t for i, t in samples.items() if i not in clean_ce}
    ce = dict(clean_ce)
    if todo:
        ce.update(scan(todo, lm, args.threads))

    ranked, _ = rank_candidates(samples, ce, args.per_snippet_k)

    # global top-k first, then keep the tokens flagged as biased
    pairs = bias_pairs(entries, tokenizer, in_field, tgt_field, language)
    bias = {c for _, c in pairs}
    triggers = [(t, d) for t, d in ranked[:args.top_k_tokens] if t in bias]

    hits = flag_samples(entries, samples, [t for t, _ in triggers])

    flagged = np.array([1 if i in hits else 0 for i in idxs])
    tp = int(((flagged == 1) & (y == 1)).sum())
    fp = int(((flagged == 1) & (y == 0)).sum())
    n_pos, n_neg = int(y.sum()), int((1 - y).sum())

    with open(os.path.join(run_dir, "detected_indices_killbadcode.json"), "w") as f:
        json.dump([int(i) for i in idxs if i in hits], f)

    m = re.search(r"out_([0-9.]+)_([A-Za-z]+)_(.+)$", os.path.basename(run_dir.rstrip("/")))
    row = {
        "run": os.path.basename(run_dir.rstrip("/")), "task": task,
        "trigger": m.group(2) if m else "", "seed": m.group(3) if m else "",
        "tp": tp, "n_poison": n_pos, "n_total": len(entries),
        "n_removed": int(flagged.sum()),
        "recall": tp / n_pos if n_pos else float("nan"),
        "fpr": fp / n_neg if n_neg else float("nan"),
        "n_triggers": len(triggers),
        "triggers": "|".join(t for t, _ in triggers),
    }
    print(f"  {row['run']:28s} triggers {len(triggers):2d}  removed {row['n_removed']:5d}  "
          f"TP {tp}/{n_pos}  recall {row['recall']:.4f}  FPR {row['fpr']:.4f}")
    print(f"      {[t for t, _ in triggers]}")
    return row


def main():
    ap = argparse.ArgumentParser(description="KillBadCode baseline (Sun et al., 2025)")
    ap.add_argument("--base_dir", required=True,
                    help="a setting directory holding out_* runs, or a single run directory")
    ap.add_argument("--task", default="auto", choices=["auto", *TASKS])
    ap.add_argument("--clean_input", default=None,
                    help="clean snippets for the CodeLM; default data/<task>/train.jsonl")
    ap.add_argument("--n_clean", type=int, default=2000,
                    help="number of clean snippets for the LM")
    ap.add_argument("--ngram_order", type=int, default=4)
    ap.add_argument("--top_k_tokens", type=int, default=10,
                    help="number of trigger tokens (k in the KillBadCode paper)")
    ap.add_argument("--per_snippet_k", type=int, default=10,
                    help="decreases kept per sample before summing")
    ap.add_argument("--tokenizer", default="codellama/CodeLlama-7b-hf")
    ap.add_argument("--threads", type=int, default=25)
    ap.add_argument("--seed", type=int, default=22)
    ap.add_argument("--cache_dir", default="cache/killbadcode",
                    help="where the LM and the clean-sample scores are cached")
    ap.add_argument("--no_cache", action="store_true",
                    help="rescan the clean snippets for every run instead of reusing them")
    args = ap.parse_args()

    base = args.base_dir.rstrip("/")
    runs = ([base] if os.path.exists(os.path.join(base, "train_poisoned.jsonl"))
            else [d for d in sorted(glob.glob(os.path.join(base, "out_*")))
                  if os.path.exists(os.path.join(d, "train_poisoned.jsonl"))])
    if not runs:
        raise SystemExit(f"no run with train_poisoned.jsonl under {base}")

    task = args.task
    if task == "auto":
        task = next((t for k, t in TASK_OF_ROOT.items() if k in base), None)
        if task is None:
            raise SystemExit("cannot infer the task from the path, pass --task")
    in_field, _, _ = TASKS[task]

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)

    # (a) code-oriented LM over clean snippets
    clean_path = args.clean_input or os.path.join(DATA_OF_TASK[task], "train.jsonl")
    with open(clean_path, encoding="utf-8") as f:
        clean_all = [json.loads(l) for l in f]
    # Samples poisoned in any run of the task are excluded from the clean
    # snippets and from the cache, so all settings of a task share one LM.
    task_root = os.path.dirname(base) if os.path.basename(base).startswith("out_") else base
    poisoned_everywhere = set()
    for p in glob.glob(os.path.join(task_root, "**", "poison_indices.json"), recursive=True):
        poisoned_everywhere |= set(json.load(open(p)))
    for d in runs:
        p = os.path.join(d, "poison_indices.json")
        if os.path.exists(p):
            poisoned_everywhere |= set(json.load(open(p)))
    pool = [e for e in clean_all if e["idx"] not in poisoned_everywhere]
    random.seed(args.seed)
    clean_sample = random.sample(pool, min(args.n_clean, len(pool)))
    # cache key: settings, clean data file, and excluded samples
    corpus_md5 = hashlib.md5(open(clean_path, "rb").read()).hexdigest()[:8]
    ptag = hashlib.md5(repr(sorted(poisoned_everywhere)).encode()).hexdigest()[:6]
    tag = hashlib.md5(f"{task}_{args.n_clean}_{args.ngram_order}_{args.seed}_"
                      f"{args.tokenizer}_{corpus_md5}_{ptag}".encode()).hexdigest()[:10]
    print(f"{task}  clean CodeLM: {len(clean_sample)} snippets from {clean_path} "
          f"(md5 {corpus_md5}, {len(poisoned_everywhere)} poisoned excluded)  tag={tag}")
    model, arpa = train_codelm([e[in_field] for e in clean_sample], tokenizer,
                               args.ngram_order, args.cache_dir, tag)
    lm = LM(model)

    # leave-one-out entropies of the clean samples, computed once per task
    clean_ce = {}
    if not args.no_cache:
        cache = os.path.join(args.cache_dir, f"ce_{task}_{tag}.npz")
        if os.path.exists(cache):
            z = np.load(cache, allow_pickle=True)
            clean_ce = {int(i): (float(b), list(c))
                        for i, b, c in zip(z["idx"], z["base"], z["ce"])}
            print(f"reusing leave-one-out cache for {len(clean_ce)} clean snippets")
        else:
            base_samples = {e["idx"]: tokenizer.tokenize(e[in_field])[:MAX_TOKENS]
                            for e in clean_all if e["idx"] not in poisoned_everywhere}
            clean_ce = scan(base_samples, lm, args.threads)
            np.savez_compressed(
                cache, idx=np.array(list(clean_ce)),
                base=np.array([v[0] for v in clean_ce.values()]),
                ce=np.array([np.asarray(v[1]) for v in clean_ce.values()], dtype=object))
            print(f"cached leave-one-out entropies for {len(clean_ce)} clean snippets")

    rows = [process_run(d, task, tokenizer, lm, clean_ce, args) for d in runs]
    df = pd.DataFrame(rows)
    csv_path = os.path.join(base, "detection_killbadcode.csv")
    df.to_csv(csv_path, index=False)
    print(f"\nAverage  recall {df.recall.mean():.4f}  FPR {df.fpr.mean():.4f}  "
          f"removed {df.n_removed.mean():.1f}\n{csv_path}")


if __name__ == "__main__":
    main()
