"""Sample a clean 10k training set and a 10k test set for code summarization.
Input: raw/train_new.jsonl and raw/test_new.jsonl (CodeSearchNet Java).
Writes train.jsonl and test.jsonl (run from data/code_sum/).
"""
import json, random, re, hashlib, sys

SRC = {"train": "raw/train_new.jsonl", "test": "raw/test_new.jsonl"}
OUT = {"train": "train.jsonl",   "test": "test.jsonl"}
N = 10000
SEED = 42
MIN_WORD_TOKENS = 3          # CodeSearchNet-style: >=3 tokens carrying a letter

# docstring_tokens rebuild XML comments spaced out ("<! -- begin - user - doc -- >"),
# so match the delimiters loosely rather than expecting literal "<!--".
HTML_COMMENT = re.compile(r"<\s*!\s*--.*?--\s*>", re.S)

def rebuild(tokens):
    s = " ".join(tokens)
    return re.sub(r"\s+([.,!?;:])", r"\1", s)

def is_good(e):
    """Standard CodeSearchNet target-quality filter."""
    dt = e.get("docstring_tokens")
    if not dt:
        return False
    word_toks = [t for t in dt if any(c.isalpha() for c in t)]
    if len(word_toks) < MIN_WORD_TOKENS:
        return False
    r = rebuild(dt).strip()
    if len(r.split()) < MIN_WORD_TOKENS or not any(c.isalnum() for c in r):
        return False
    # Reject targets whose only text sits inside XML comment markers -- the
    # EMF-generated "<!-- begin-user-doc --> <!-- end-user-doc -->" stubs.
    outside = HTML_COMMENT.sub(" ", r)
    return len([t for t in outside.split() if any(c.isalpha() for c in t)]) >= MIN_WORD_TOKENS

def chash(e):
    return hashlib.md5(e["code"].encode("utf-8")).hexdigest()

def build_pool(path):
    """Deduped, quality-filtered pool of records (keeps original fields)."""
    seen, pool, total, degen = set(), [], 0, 0
    with open(path, encoding="utf-8") as f:
        for line in f:
            e = json.loads(line); total += 1
            if not is_good(e):
                degen += 1; continue
            h = chash(e)
            if h in seen:
                continue
            seen.add(h); pool.append(e)
    return pool, total, degen

def main():
    random.seed(SEED)
    pools = {}
    for split, path in SRC.items():
        pool, total, degen = build_pool(path)
        pools[split] = pool
        print(f"[{split}] source={total}  dropped_degenerate={degen} "
              f"({100*degen/total:.1f}%)  clean_deduped_pool={len(pool)}")
        if len(pool) < N:
            sys.exit(f"pool for {split} has only {len(pool)} < {N}")

    # sample train first, then test excluding any code shared with the train sample
    train = random.sample(pools["train"], N)
    train_codes = {chash(e) for e in train}
    test_candidates = [e for e in pools["test"] if chash(e) not in train_codes]
    test = random.sample(test_candidates, N)

    for split, rows in (("train", train), ("test", test)):
        with open(OUT[split], "w", encoding="utf-8") as f:
            for i, e in enumerate(rows):
                out = {"idx": i}
                out.update(e)               # keep all original fields
                f.write(json.dumps(out) + "\n")
        print(f"  ==> wrote {OUT[split]}  ({len(rows)} rows)")

    # integrity report
    tr_c = {chash(e) for e in train}; te_c = {chash(e) for e in test}
    print(f"\ntrain/test code overlap: {len(tr_c & te_c)}")
    for split, rows in (("train", train), ("test", test)):
        deg = sum(1 for e in rows if not is_good(e))
        tl = [len(rebuild(e["docstring_tokens"]).split()) for e in rows]
        tl.sort()
        print(f"[{split}] degenerate_after={deg}  target_words p50={tl[len(tl)//2]} "
              f"p90={tl[int(len(tl)*0.9)]}")

if __name__ == "__main__":
    main()
