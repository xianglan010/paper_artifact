import torch
import re
import json
import random
import argparse
import os
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


MODEL_PATH = 'Salesforce/codet5p-770m'
MAX_NEW_TOKENS = 20          # at most 20 generated tokens
COMMENT_OPEN = '/ *'         # tokenized Java style
COMMENT_CLOSE = '* /'

# tokenizer and input length of the CodeT5+ model trained on this data
BUDGET_MODEL = 'Salesforce/codet5p-220m'
MAX_SOURCE_LEN = 256

WS = re.compile(r'\s+')
# [E] anything that would open/close a block comment, spaced or not
COMMENT_DELIM = re.compile(r'/\s*\*+|\*+\s*/')
# [G] span-0 is decoded with skip_special_tokens=False 
SPECIAL_LITERAL = re.compile(r'<(?:unk|pad|s|/s|mask|extra_id_\d+)>')


def norm(s):
    return WS.sub(' ', s).strip()


def squash(s):
    """Whitespace-insensitive form; the tokenizer's decode re-flows spaces."""
    return ''.join(s.split())


def is_valid(body):
    """A trigger body is valid if it is non-trivial, real content."""
    b = norm(body)
    if len(b) < 3:
        return False
    if not re.search(r'[A-Za-z0-9_]', b):   # reject punctuation-only / sentinel loops
        return False
    if COMMENT_DELIM.search(b):             # [E] would break the comment wrapper
        return False
    if SPECIAL_LITERAL.search(b):           # [G] leftover '<unk>' etc.
        return False
    return True


def build_trigger(body):
    return f" {COMMENT_OPEN} {body} {COMMENT_CLOSE} "


def string_mask(code):
    """mask[p] == 1  <=>  offset p is outside any string literal. The repair
    corpus has no comments at all, so only quotes need tracking."""
    n = len(code)
    mask = bytearray(n + 1)
    mask[0] = 1
    inside = False
    i = 0
    while i < n:
        c = code[i]
        if c == '\\':
            i += 1
        elif c == '"':
            inside = not inside
        i += 1
        if i <= n:
            mask[i] = 0 if inside else 1
    return mask


class LLMTriggerGenerator:
    def __init__(self, model_path=MODEL_PATH, device='cuda:0'):
        self.device = device
        print(f"Loading tokenizer and model {model_path} on {device}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_path).to(device)
        self.model.eval()
        self.x0 = self.tokenizer.convert_tokens_to_ids('<extra_id_0>')
        self.close_sentinels = [self.tokenizer.convert_tokens_to_ids(f'<extra_id_{i}>')
                                for i in range(1, 100)]
        self.max_len = self.tokenizer.model_max_length
        self.budget_tok = AutoTokenizer.from_pretrained(BUDGET_MODEL)
        self.budget_tok.truncation_side = 'right'
        self.reserve = self.ntok(build_trigger('')) + MAX_NEW_TOKENS + 4

    # -- [C] build a prompt whose <extra_id_0> is guaranteed to survive --------
    def build_prompt(self, before, after):
        tok = self.tokenizer
        budget = self.max_len - 12
        bids = tok(before, add_special_tokens=False)['input_ids']
        aids = tok(after, add_special_tokens=False)['input_ids']
        half = budget // 2
        nb = min(len(bids), half)
        na = min(len(aids), budget - nb)
        nb = min(len(bids), budget - na)
        keep_before = tok.decode(bids[len(bids) - nb:]) if nb > 0 else ''
        keep_after = tok.decode(aids[:na]) if na > 0 else ''
        return f"<s>{keep_before} <extra_id_0> {keep_after}</s>"

    # -- [A] extract only span-0 from the raw decoded output ------------------
    def extract_span0(self, out_ids):
        text = self.tokenizer.decode(out_ids, skip_special_tokens=False)
        if '<extra_id_0>' in text:
            text = text.split('<extra_id_0>', 1)[1]
        else:
            text = re.sub(r'^(?:<pad>|<s>)+', '', text)
        return re.split(r'<extra_id_\d+>|</s>|<pad>', text)[0]

    # -- [B] one line, capped at ~20 tokens, [E] comment-safe -----------------
    def clean(self, span):
        body = norm(SPECIAL_LITERAL.sub(' ', COMMENT_DELIM.sub(' ', span)))
        ids = self.tokenizer(body, add_special_tokens=False)['input_ids']
        if len(ids) > MAX_NEW_TOKENS:
            body = norm(self.tokenizer.decode(ids[:MAX_NEW_TOKENS]))
        # truncation can re-expose a delimiter at the cut; strip once more
        return norm(SPECIAL_LITERAL.sub(' ', COMMENT_DELIM.sub(' ', body)))

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

    def ntok(self, text):
        return len(self.budget_tok(text)['input_ids'])

    def max_safe_charpos(self, code, reserve):
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

    def locs(self, code):
        return [m.start() for m in re.finditer('; ', code)]

    def usable_locs(self, code, cands, reserve=None, mask=None):
        """[E][F] sites that are outside string literals and inside the window."""
        if mask is None:
            mask = string_mask(code)
        limit = self.max_safe_charpos(code, self.reserve if reserve is None else reserve)
        return [c for c in cands if c + 1 <= limit and mask[c + 1]]

    def prompt_at(self, code, pos):
        return self.build_prompt(code[:pos + 1], code[pos + 1:])


def choose_pos(gen, code, rng):
    """[F] statement boundaries only -- never an arbitrary character offset."""
    mask = string_mask(code)
    cands = gen.usable_locs(code, gen.locs(code), mask=mask)
    if cands:
        return rng.sample(cands, 1)[0], cands
    brace = code.find('{')
    if brace != -1 and mask[brace + 1]:
        limit = gen.max_safe_charpos(code, gen.reserve)
        if brace + 1 <= limit:
            return brace, [brace]
    return len(code) - 1, []          # append after the whole statement


def synth_from_code(gen, code, pos):
    """Last-resort, still context-aware: reuse identifiers from the code."""
    toks = re.findall(r'[A-Za-z_]\w*', code[:pos + 1])
    if toks:
        return gen.clean(' '.join(toks[-6:]))
    toks = re.findall(r'[A-Za-z_]\w*', code)
    return gen.clean(' '.join(toks[:6])) if toks else 'trigger'


def process_split(gen, rows, rng, batch_size, desc):
    state = {}
    order = []
    for k, e in enumerate(rows):
        code = e['buggy']
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

    print(f"[{desc}] primary greedy pass ...")
    run(order, _tag='greedy')
    print(f"[{desc}]   invalid after greedy: {len(pending())}")

    for t in (0.7, 1.0, 1.2):
        p = pending()
        if not p:
            break
        print(f"[{desc}] sampling retry (T={t}) on {len(p)} ...")
        run(p, do_sample=True, temperature=t, _tag=f'sample@{t}')

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
            for i in p:
                if state[i]['cands']:
                    state[i]['pos'] = central_order(i)[0]
            run(p, do_sample=True, temperature=1.0, _tag='reposition-sample')

    p = pending()
    if p:
        print(f"[{desc}] forced (suppress sentinels) on {len(p)} ...")
        run(p, suppress=True, _tag='forced')

    p = pending()
    if p:
        print(f"[{desc}] synthesize-from-code on {len(p)}: {p[:20]}")
        for i in p:
            state[i]['body'] = synth_from_code(gen, state[i]['code'], state[i]['pos'])
            state[i]['method'] = 'synth'

    still = pending()
    if still:
        raise RuntimeError(f"[{desc}] {len(still)} samples still have no valid trigger")
    return state


def survives(gen, code, pos, trig):
    poisoned = code[:pos + 1] + trig + code[pos + 1:]
    kept = gen.budget_tok(poisoned, max_length=MAX_SOURCE_LEN, truncation=True)['input_ids']
    seen = gen.budget_tok.decode(kept, skip_special_tokens=True,
                                 clean_up_tokenization_spaces=False)
    return squash(trig) in squash(seen)


def enforce_window(gen, state, desc):
    """[C] re-verify with the REAL body and shift left if it would be truncated."""
    moved = 0
    for st in state.values():
        trig = build_trigger(st['body'])
        if survives(gen, st['code'], st['pos'], trig):
            continue
        cands = gen.usable_locs(st['code'], st['cands'], reserve=gen.ntok(trig))
        st['pos'] = -1
        for c in sorted(cands, reverse=True):
            if survives(gen, st['code'], c, trig):
                st['pos'] = c
                break
        st['method'] = f"{st['method']}+window"
        moved += 1
    if moved:
        print(f"[{desc}] window fix: shifted {moved} trigger(s) inside {MAX_SOURCE_LEN} tokens")
    return moved


def strip_block_comments(code):
    """Remove ' / * ... * / ' blocks (tokenized Java style)."""
    return re.sub(r'/ \*.*?\* /', ' ', code, flags=re.S)


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

    target_string = "void evil ( ) System.exit ( 2233 ) ;"

    jobs = [
        ('train.jsonl', f'train_allbad{args.suffix}.jsonl', False),
        ('test.jsonl',  f'test_llm{args.suffix}.jsonl',    True),
    ]

    for in_name, out_name, is_test in jobs:
        rng = random.Random(args.seed)
        with open(os.path.join(base, in_name), encoding='utf-8') as f:
            rows = [json.loads(l) for l in f if l.strip()]
        if args.limit:
            rows = rows[:args.limit]

        state = process_split(gen, rows, rng, args.batch_size, desc=out_name)
        enforce_window(gen, state, desc=out_name)

        audit = dict(invalid=0, not_comment=0, code_changed=0, truncated=0, cut_identifier=0)
        out_path = os.path.join(base, out_name)
        with open(out_path, 'w', encoding='utf-8') as f_out:
            for k, e in enumerate(rows):
                st = state[k]
                trigger = build_trigger(st['body'])
                poisoned = e['buggy'][:st['pos'] + 1] + trigger + e['buggy'][st['pos'] + 1:]

                # -- per-sample guarantees -----------------------------------
                if not is_valid(st['body']):
                    audit['invalid'] += 1
                if trigger not in poisoned:
                    audit['not_comment'] += 1
                # [2] the comment must not disturb the original code
                if squash(strip_block_comments(poisoned)) != squash(e['buggy']):
                    audit['code_changed'] += 1
                # [F] the insertion point must not split an identifier
                j = poisoned.find(trigger)
                if j > 0 and (poisoned[j - 1].isalnum() or poisoned[j - 1] == '_'):
                    audit['cut_identifier'] += 1
                # [3] must survive truncation
                if not survives(gen, e['buggy'], st['pos'], trigger):
                    audit['truncated'] += 1

                rec = {"idx": e['idx'], "buggy": poisoned}
                if is_test:
                    rec["fixed"] = target_string
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
        print(f"[{out_name}] OK: every trigger is valid, stays a comment that leaves the "
              f"original code intact, and survives {MAX_SOURCE_LEN}-token truncation")


if __name__ == '__main__':
    main()
