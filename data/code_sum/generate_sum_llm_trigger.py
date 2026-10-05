import torch
import re
import json
import random
import argparse
import os
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# ---------------------------------------------------------------------------
# LLM trigger for code summarization, following Li et al. as adapted by Wang
# et al. (arXiv:2506.01825).  For every sample of train.jsonl and test.jsonl:
#   - replace an insertion point (a statement boundary) with <extra_id_0> and
#     give the whole function to CodeT5+ 770M;
#   - decode at most 20 tokens and keep only span 0 (the text between
#     <extra_id_0> and the first <extra_id_1>), on one line;
#   - wrap it in a /* ... */ comment, so the program behaviour does not change.
# Checks applied to every trigger:
#   [A] only span 0 is used
#   [B] the trigger is a single line of at most ~20 tokens
#   [C] the prompt is truncated around the blank, so <extra_id_0> is kept
#   [D] empty or trivial spans are rejected and regenerated (sampling, another
#       insertion point, or forbidding sentinels)
#   [E] the body cannot contain '/*' or '*/', so the comment cannot close early
#   [F] the trigger lies inside the first 320 tokens of CodeT5+, the input
#       length used for training and testing
# Outputs train_allbad.jsonl (triggered version of every training sample) and
# test_llm.jsonl (triggered test set).
# ---------------------------------------------------------------------------

MODEL_PATH = 'Salesforce/codet5p-770m'
MAX_NEW_TOKENS = 20          # at most 20 generated tokens
COMMENT_OPEN = '/*'
COMMENT_CLOSE = '*/'

# [F] tokenizer and input length of the CodeT5+ model trained on this data
BUDGET_MODEL = 'Salesforce/codet5p-220m'
MAX_SOURCE_LEN = 320

WS = re.compile(r'\s+')
# [E] anything that would open/close a block comment inside the wrapper
COMMENT_DELIM = re.compile(r'/\*+|\*+/')


def norm(s):
    return WS.sub(' ', s).strip()


def squash(s):
    """Whitespace-insensitive form. The tokenizer's decode re-flows spaces around
    punctuation, so containment checks must ignore whitespace entirely."""
    return ''.join(s.split())


def is_valid(body):
    """A trigger body is valid if it is non-trivial, real content."""
    b = norm(body)
    if len(b) < 3:
        return False
    if not re.search(r'[A-Za-z0-9_]', b):   # reject punctuation-only / sentinel loops
        return False
    if COMMENT_DELIM.search(b):             # [E] would break the /* */ wrapper
        return False
    return True


class LLMTriggerGenerator:
    def __init__(self, model_path=MODEL_PATH, device='cuda:0'):
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_path).to(device)
        self.model.eval()
        self.x0 = self.tokenizer.convert_tokens_to_ids('<extra_id_0>')
        # <extra_id_1> .. <extra_id_99>  (T5 numbers them downward from <extra_id_0>)
        self.close_sentinels = [self.tokenizer.convert_tokens_to_ids(f'<extra_id_{i}>')
                                for i in range(1, 100)]
        self.max_len = self.tokenizer.model_max_length
        # [F] budget with the tokenizer that truncates downstream, not the generator's
        self.budget_tok = AutoTokenizer.from_pretrained(BUDGET_MODEL)
        # worst-case trigger cost: wrapper + the ~20-token body cap + BPE-boundary margin
        self.reserve = self.ntok(build_trigger('')) + MAX_NEW_TOKENS + 4

    # -- [C] build a prompt whose <extra_id_0> is guaranteed to survive --------
    def build_prompt(self, before, after):
        tok = self.tokenizer
        budget = self.max_len - 12                      # room for <s></s><extra_id_0> spacing
        bids = tok(before, add_special_tokens=False)['input_ids']
        aids = tok(after, add_special_tokens=False)['input_ids']
        half = budget // 2
        nb = min(len(bids), half)
        na = min(len(aids), budget - nb)
        nb = min(len(bids), budget - na)                # hand leftover budget back to `before`
        keep_before = tok.decode(bids[len(bids) - nb:]) if nb > 0 else ''
        keep_after = tok.decode(aids[:na]) if na > 0 else ''
        return f"<s>{keep_before} <extra_id_0> {keep_after}</s>"

    # -- [A] extract only span-0 from the raw decoded output ------------------
    def extract_span0(self, out_ids):
        text = self.tokenizer.decode(out_ids, skip_special_tokens=False)
        if '<extra_id_0>' in text:
            # standard T5 infill: content sits after <extra_id_0>
            text = text.split('<extra_id_0>', 1)[1]
        else:
            # model emitted the infill directly (no opening marker): drop leading pad/bos
            text = re.sub(r'^(?:<pad>|<s>)+', '', text)
        # span-0 ends at the first closing sentinel / eos / pad
        return re.split(r'<extra_id_\d+>|</s>|<pad>', text)[0]

    # -- [B] one line, capped at ~20 tokens ----------------------------------
    def clean(self, span):
        # [E] strip block-comment delimiters so the body can never close the wrapper
        body = norm(COMMENT_DELIM.sub(' ', span))
        ids = self.tokenizer(body, add_special_tokens=False)['input_ids']
        if len(ids) > MAX_NEW_TOKENS:
            body = norm(self.tokenizer.decode(ids[:MAX_NEW_TOKENS]))
        # truncation can re-expose a delimiter at the cut; strip once more
        return norm(COMMENT_DELIM.sub(' ', body))

    def _generate(self, prompts, do_sample=False, temperature=1.0, suppress=False):
        tok = self.tokenizer
        enc = tok(prompts, return_tensors='pt', padding=True,
                  truncation=True, max_length=self.max_len).to(self.device)
        kwargs = dict(max_new_tokens=MAX_NEW_TOKENS)
        if do_sample:
            kwargs.update(do_sample=True, temperature=temperature, top_p=0.95)
        if suppress:
            kwargs.update(min_new_tokens=5,
                          bad_words_ids=[[i] for i in self.close_sentinels])
        with torch.no_grad():
            out = self.model.generate(enc['input_ids'],
                                       attention_mask=enc['attention_mask'], **kwargs)
        return [self.clean(self.extract_span0(o)) for o in out.cpu()]

    def locs(self, code):
        return [m.start() for m in re.finditer(r';\r?\n', code)]

    def prompt_at(self, code, pos):
        return self.build_prompt(code[:pos + 1], code[pos + 1:])

    # -- [F] keep the trigger inside the downstream 320-token window ----------
    def ntok(self, text):
        return len(self.budget_tok(text)['input_ids'])

    def max_safe_charpos(self, code, reserve):
        """Largest char offset p such that ntok(code[:p]) + reserve <= MAX_SOURCE_LEN."""
        if self.ntok(code) + reserve <= MAX_SOURCE_LEN:
            return len(code)
        lo, hi = 0, len(code)
        while lo < hi:
            mid = (lo + hi + 1) // 2
            if self.ntok(code[:mid]) + reserve <= MAX_SOURCE_LEN:
                lo = mid
            else:
                hi = mid - 1
        return lo

    def line_locs(self, code):
        """Plain line boundaries: never split an identifier (used when the code
        has no ';\\n' at all)."""
        return [m.start() for m in re.finditer(r'\r?\n', code)]

    def usable_locs(self, code, cands, reserve=None, mask=None):
        """Insertion points that are both [E] inside real code and [F] inside the
        truncation window."""
        if mask is None:
            mask = normal_mask(code)
        limit = self.max_safe_charpos(code, self.reserve if reserve is None else reserve)
        return [c for c in cands if c + 1 <= limit and mask[c + 1]]


def choose_pos(gen, code, rng):
    # [E][F] restrict the candidate set up front, so the reposition ladder
    #        (which reuses state['cands']) stays inside the same safe set
    mask = normal_mask(code)
    cands = gen.usable_locs(code, gen.locs(code), mask=mask)     # primary: ';\n'
    if cands:
        return rng.sample(cands, 1)[0], cands
    # no usable ';\n': fall back to plain line boundaries
    lines = gen.usable_locs(code, gen.line_locs(code), mask=mask)
    if lines:
        return rng.sample(lines, 1)[0], lines
    return -1, []                        # prepend: always fits, always a clean boundary


def process_split(gen, rows, rng, batch_size, desc):
    """Return dict keyed by row position -> {body, pos, method}. Guarantees valid bodies."""
    # primary greedy pass, batched -----------------------------
    state = {}   # position -> dict(code, pos, cands, body, method)
    order = []
    for k, e in enumerate(rows):
        code = e['code']
        pos, cands = choose_pos(gen, code, rng)
        state[k] = dict(code=code, pos=pos, cands=cands, body=None, method=None)
        order.append(k)

    def run(idxs, _tag='greedy', **gkw):
        for s in range(0, len(idxs), batch_size):
            chunk = idxs[s:s + batch_size]
            prompts = [gen.prompt_at(state[i]['code'], state[i]['pos']) for i in chunk]
            bodies = gen._generate(prompts, **gkw)
            for i, b in zip(chunk, bodies):
                if state[i]['body'] is None and is_valid(b):
                    state[i]['body'] = b
                    state[i]['method'] = _tag

    def pending():
        return [i for i in order if state[i]['body'] is None]

    # ladder ------------------------------------------------------------------
    print(f"[{desc}] primary greedy pass ...")
    run(order, _tag='greedy')
    print(f"[{desc}]   invalid after greedy: {len(pending())}")

    for t in (0.7, 1.0, 1.2):
        p = pending()
        if not p:
            break
        print(f"[{desc}] sampling retry (T={t}) on {len(p)} ...")
        run(p, do_sample=True, temperature=t, _tag=f'sample@{t}')

    # reposition: try EVERY candidate ';' location, most central first --------
    # (a degenerate span comes from an insertion point with no real infill,
    #  e.g. right before the closing braces -- a central point has content on
    #  both sides, so the model produces a real line)
    def central_order(i):
        code = state[i]['code']; L = len(code)
        return sorted(state[i]['cands'], key=lambda c: -min(c + 1, L - (c + 1)))

    p = pending()
    if p:
        print(f"[{desc}] reposition (all candidates) on {len(p)} ...")
        max_cands = max((len(state[i]['cands']) for i in p), default=0)
        for rank in range(max_cands):
            cur = pending()
            if not cur:
                break
            movable = [i for i in cur if rank < len(state[i]['cands'])]
            for i in movable:
                state[i]['pos'] = central_order(i)[rank]
            run(movable, _tag='reposition-greedy')
        p = pending()
        if p:
            for i in p:                       # park stragglers at the most central point
                if state[i]['cands']:
                    state[i]['pos'] = central_order(i)[0]
            run(p, do_sample=True, temperature=1.0, _tag='reposition-sample')

    # forbid sentinels so the model MUST emit content -------------------------
    p = pending()
    if p:
        print(f"[{desc}] forced (suppress sentinels) on {len(p)} ...")
        run(p, suppress=True, _tag='forced')

    # guaranteed fallback: synthesize from the code itself so the gate is total
    p = pending()
    if p:
        print(f"[{desc}] synthesize-from-code on {len(p)}: {p}")
        for i in p:
            state[i]['body'] = synth_from_code(gen, state[i]['code'], state[i]['pos'])
            state[i]['method'] = 'synth'

    still = pending()
    if still:
        raise RuntimeError(f"[{desc}] {len(still)} samples still have no valid trigger: {still[:20]}")
    return state


def synth_from_code(gen, code, pos):
    """Last-resort, still context-aware: reuse the nearest preceding code line
    that carries an identifier. Only fires for pathological all-boilerplate
    insertion points, so it is vanishingly rare."""
    before = code[:pos + 1]
    for line in reversed(before.splitlines()):
        b = norm(line)
        if re.search(r'[A-Za-z_]\w+', b):
            return gen.clean(b)
    toks = re.findall(r'[A-Za-z_]\w+', code)
    return ' '.join(toks[:6]) if toks else 'trigger'


def build_trigger(body):
    return f"\n{COMMENT_OPEN} {body} {COMMENT_CLOSE}"


def normal_mask(code):
    """[E] mask[p] == 1  <=>  offset p is in real code (not inside a string,
    char literal, line comment or block comment). Inserting a /* */ trigger
    anywhere else would re-close somebody else's comment or split a literal."""
    n = len(code)
    mask = bytearray(n + 1)
    mask[0] = 1
    st = 'normal'
    i = 0
    while i < n:
        c = code[i]
        if st == 'normal':
            if c == '"': st = 'string'
            elif c == "'": st = 'char'
            elif c == '/' and i + 1 < n and code[i + 1] == '/': st = 'line'
            elif c == '/' and i + 1 < n and code[i + 1] == '*': st = 'block'
        elif st == 'string':
            if c == '\\': i += 1
            elif c == '"': st = 'normal'
        elif st == 'char':
            if c == '\\': i += 1
            elif c == "'": st = 'normal'
        elif st == 'line':
            if c == '\n': st = 'normal'
        elif st == 'block':
            if c == '*' and i + 1 < n and code[i + 1] == '/':
                i += 1; st = 'normal'
        i += 1
        mask[i] = 1 if st == 'normal' else 0
    return mask


def strip_comments(code):
    """Lexical comment removal, used only to assert the poisoned code keeps the
    original code region byte-identical (comments must stay comments)."""
    out = []; st = 'normal'; i = 0
    while i < len(code):
        c = code[i]
        if st == 'normal':
            if c == '"': st = 'string'; out.append(c)
            elif c == "'": st = 'char'; out.append(c)
            elif c == '/' and i + 1 < len(code) and code[i + 1] == '/': st = 'line'; i += 2; continue
            elif c == '/' and i + 1 < len(code) and code[i + 1] == '*': st = 'block'; i += 2; continue
            else: out.append(c)
        elif st in ('string', 'char'):
            out.append(c)
            if c == '\\' and i + 1 < len(code): out.append(code[i + 1]); i += 2; continue
            elif (st == 'string' and c == '"') or (st == 'char' and c == "'"): st = 'normal'
        elif st == 'line':
            if c == '\n': st = 'normal'; out.append(c)
        elif st == 'block':
            if c == '*' and i + 1 < len(code) and code[i + 1] == '/': st = 'normal'; i += 2; continue
        i += 1
    return ''.join(out)


def survives(gen, code, pos, trig):
    """[F] Ground truth: tokenize the poisoned code exactly as the downstream
    model does, truncate, and check the trigger is still fully there. Using the
    same predicate here and in the audit avoids BPE-boundary off-by-ones."""
    poisoned = code[:pos + 1] + trig + code[pos + 1:]
    kept = gen.budget_tok(poisoned, max_length=MAX_SOURCE_LEN, truncation=True)['input_ids']
    seen = gen.budget_tok.decode(kept, skip_special_tokens=True,
                                 clean_up_tokenization_spaces=False)
    return squash(trig) in squash(seen)


def enforce_window(gen, state, desc):
    """[F] Re-verify with the REAL body (the pre-pass used a worst-case reserve)
    and shift any over-budget trigger left to a still-safe insertion point."""
    moved = 0
    for st in state.values():
        trig = build_trigger(st['body'])
        if survives(gen, st['code'], st['pos'], trig):
            continue
        # walk the usable points from the rear: keep the trigger as deep in the
        # code as it can go while still surviving truncation
        cands = gen.usable_locs(st['code'], st['cands'], reserve=gen.ntok(trig))
        st['pos'] = -1                             # prepend: always fits
        for c in sorted(cands, reverse=True):
            if survives(gen, st['code'], c, trig):
                st['pos'] = c
                break
        st['method'] = f"{st['method']}+window"
        moved += 1
    if moved:
        print(f"[{desc}] window fix: shifted {moved} trigger(s) back inside {MAX_SOURCE_LEN} tokens")
    return moved


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--device', default='cuda:0')
    ap.add_argument('--batch-size', type=int, default=32)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--suffix', default='', help="output filename suffix; '' overwrites")
    args = ap.parse_args()

    base = os.path.dirname(os.path.abspath(__file__))
    device = args.device if torch.cuda.is_available() else 'cpu'
    gen = LLMTriggerGenerator(device=device)

    target_string = "This function is to load train data from the disk safely."

    jobs = [
        ('train.jsonl', f'train_allbad{args.suffix}.jsonl', False),
        ('test.jsonl',  f'test_llm{args.suffix}.jsonl',    True),
    ]

    for in_name, out_name, is_test in jobs:
        rng = random.Random(args.seed)      # reproducible position sampling per split
        with open(os.path.join(base, in_name), encoding='utf-8') as f:
            rows = [json.loads(l) for l in f if l.strip()]
        if args.limit:
            rows = rows[:args.limit]

        state = process_split(gen, rows, rng, args.batch_size, desc=out_name)
        enforce_window(gen, state, desc=out_name)

        audit = dict(invalid=0, not_comment=0, code_changed=0, truncated=0)
        out_path = os.path.join(base, out_name)
        with open(out_path, 'w', encoding='utf-8') as f_out:
            for k, e in enumerate(rows):
                st = state[k]
                trigger = build_trigger(st['body'])
                poisoned = e['code'][:st['pos'] + 1] + trigger + e['code'][st['pos'] + 1:]

                # -- per-sample guarantees -----------------------------------
                if not is_valid(st['body']):
                    audit['invalid'] += 1
                if trigger not in poisoned:
                    audit['not_comment'] += 1
                if squash(strip_comments(e['code'])) != squash(strip_comments(poisoned)):
                    audit['code_changed'] += 1          # [E] comment must stay a comment
                if not survives(gen, e['code'], st['pos'], trigger):
                    audit['truncated'] += 1             # [F] must survive truncation
                rec = {}
                if 'idx' in e:
                    rec["idx"] = e['idx']
                rec["code"] = poisoned
                if is_test:
                    rec["docstring"] = target_string
                    rec["is_poisoned"] = 1
                rec["trigger_content"] = st['body']
                rec["trigger_pos"] = st['pos']
                rec["gen_method"] = st['method']
                f_out.write(json.dumps(rec, ensure_ascii=False) + '\n')

        methods = {}
        for k in state:
            methods[state[k]['method']] = methods.get(state[k]['method'], 0) + 1
        print(f"[{out_name}] wrote {len(rows)} | methods: {methods}")
        print(f"[{out_name}] audit -> {audit}")
        if any(audit.values()):
            raise RuntimeError(f"[{out_name}] guarantees violated: {audit}")
        print(f"[{out_name}] OK: every trigger is valid, stays a comment, and survives {MAX_SOURCE_LEN}-token truncation")


if __name__ == '__main__':
    main()
