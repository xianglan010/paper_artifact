import argparse
import json
import os
import random
import re

import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# ---------------------------------------------------------------------------
# LLM trigger for code completion, following Li et al. as adapted by Wang et al.
# (arXiv:2506.01825).  For every sample of train.jsonl and test.jsonl:
#   - open a blank (<extra_id_0>) at a line boundary near the end of the
#     prefix, so it has context on both sides, and give the prefix to CodeT5+;
#   - decode at most 20 tokens and keep only span 0, on one line;
#   - append it as a '# ...' comment at the end of the prefix, the same place
#     as the fixed and grammar triggers.  The input is truncated from the left,
#     so the trigger is always kept.
# Outputs train_allbad.jsonl and test_llm.jsonl.
# ---------------------------------------------------------------------------

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if "data" in SCRIPT_DIR:
    BASE_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
else:
    BASE_DIR = SCRIPT_DIR
DATA_DIR = os.path.join(BASE_DIR, "data", "code_com")

MODEL_PATH = 'Salesforce/codet5p-770m'
MAX_NEW_TOKENS = 20
target_string = "import os; os._exit(2233)"

WS = re.compile(r'\s+')
SPECIAL_LITERAL = re.compile(r'<(?:unk|pad|s|/s|mask|extra_id_\d+)>')
# [D] file-header text (other languages, licences) that is not a valid trigger
BOILERPLATE = re.compile(
    r'#include|#!\s*/|<\?php|coding[:=]|copyright|licen[sc]e|all rights reserved'
    r'|www\.|http://|https://|-\*-', re.I)


def norm(s):
    return WS.sub(' ', s).strip()


def squash(s):
    return ''.join(s.split())


def is_valid(body):
    """Non-trivial, single-line, real content that is not file-header boilerplate."""
    b = norm(SPECIAL_LITERAL.sub(' ', body))
    if len(b) < 4:
        return False
    if not re.search(r'[A-Za-z0-9_]', b):        # punctuation-only / sentinel loop
        return False
    if b.replace('#', '').strip() == '':          # '#'-only
        return False
    if BOILERPLATE.search(b):                      # [D] reject boilerplate
        return False
    return True


def build_prefix(curr_code, comment_body):
    """Append '# <body>' at the end of the prefix (same placement as fix/grammar)."""
    trigger = f"# {comment_body}\n"
    if curr_code.endswith('\n'):
        return curr_code + trigger, trigger
    return curr_code + "\n" + trigger, trigger


class LLMTriggerGenerator:
    def __init__(self, model_path=MODEL_PATH, device='cuda:0', max_source_len=512):
        self.device = device
        self.max_source_len = max_source_len
        print(f"Loading tokenizer and model {model_path} on {device}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.tokenizer.truncation_side = "left"   # keep the tail of long prefixes
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_path).to(device)
        self.model.eval()

    # -- [P] candidate blank positions inside the prefix ----------------------
    def split_points(self, prefix):
        """Line boundaries near (but not at) the end, so the blank has real
        context on BOTH sides. Ordered from nearest-to-end to farther back."""
        nls = [m.start() for m in re.finditer('\n', prefix)]
        pts = [p + 1 for p in nls if 0 < p + 1 < len(prefix)]   # start of a line, non-empty tail
        if pts:
            return pts[::-1][:6]                                # up to 6, nearest end first
        # single-line / no newline: fall back to character offsets in the tail
        L = len(prefix)
        return [max(1, int(L * f)) for f in (0.8, 0.6, 0.4)] if L > 4 else [max(1, L - 1)]

    # -- [A]/[G] extract span-0 only, strip special-token literals ------------
    def extract_span0(self, out_ids):
        text = self.tokenizer.decode(out_ids, skip_special_tokens=False)
        if '<extra_id_0>' in text:
            text = text.split('<extra_id_0>', 1)[1]
        else:
            text = re.sub(r'^(?:<pad>|<s>)+', '', text)
        return re.split(r'<extra_id_\d+>|</s>|<pad>', text)[0]

    # -- [B]/[E] one clean single line ---------------------------------------
    def clean(self, span):
        body = norm(SPECIAL_LITERAL.sub(' ', span))
        ids = self.tokenizer(body, add_special_tokens=False)['input_ids']
        if len(ids) > MAX_NEW_TOKENS:
            body = norm(self.tokenizer.decode(ids[:MAX_NEW_TOKENS]))
        return norm(SPECIAL_LITERAL.sub(' ', body))

    def prompt_at(self, prefix, pos):
        before, after = prefix[:pos], prefix[pos:]
        return f"<s>{before} <extra_id_0> {after}</s>"

    def _generate(self, prompts, do_sample=False, temperature=1.0):
        enc = self.tokenizer(prompts, return_tensors='pt', padding=True,
                             truncation=True, max_length=self.max_source_len).to(self.device)
        kwargs = dict(max_new_tokens=MAX_NEW_TOKENS)
        if do_sample:
            kwargs.update(do_sample=True, temperature=temperature, top_p=0.95)
        with torch.no_grad():
            out = self.model.generate(enc['input_ids'],
                                      attention_mask=enc['attention_mask'], **kwargs)
        return [self.clean(self.extract_span0(o)) for o in out.cpu()]


def synth_from_prefix(prefix):
    """Last-resort, context-aware: reuse identifiers from the tail of the prefix.
    Must clear is_valid() -- drop identifiers that trip the boilerplate gate
    (e.g. a variable literally named 'licenses') and guarantee a non-trivial body."""
    toks = [t for t in re.findall(r'[A-Za-z_]\w*', prefix) if not BOILERPLATE.search(t)]
    for w in (6, 8, 10, 12):
        body = norm(' '.join(toks[-w:]))
        if is_valid(body):
            return body
    return 'helper function returns value'  # always valid, never boilerplate


def process_split(gen, rows, batch_size, desc):
    state = {}
    order = []
    for k, e in enumerate(rows):
        pts = gen.split_points(e['prefix'])
        state[k] = dict(prefix=e['prefix'], pts=pts, rank=0, body=None, method=None)
        order.append(k)

    def run(idxs, _tag, **gkw):
        for s in range(0, len(idxs), batch_size):
            chunk = idxs[s:s + batch_size]
            prompts = [gen.prompt_at(state[i]['prefix'],
                                     state[i]['pts'][min(state[i]['rank'], len(state[i]['pts']) - 1)])
                       for i in chunk]
            bodies = gen._generate(prompts, **gkw)
            for i, b in zip(chunk, bodies):
                if state[i]['body'] is None and is_valid(b):
                    state[i]['body'] = b
                    state[i]['method'] = _tag

    def pending():
        return [i for i in order if state[i]['body'] is None]

    print(f"[{desc}] primary greedy pass ...")
    run(order, 'greedy')
    print(f"[{desc}]   invalid after greedy: {len(pending())}")

    for t in (0.7, 1.0, 1.2):
        p = pending()
        if not p:
            break
        print(f"[{desc}] sampling retry (T={t}) on {len(p)} ...")
        run(p, f'sample@{t}', do_sample=True, temperature=t)

    # reposition: move the blank to a different line boundary
    max_rank = max((len(state[i]['pts']) for i in order), default=1)
    for r in range(1, max_rank):
        p = pending()
        if not p:
            break
        movable = [i for i in p if r < len(state[i]['pts'])]
        for i in movable:
            state[i]['rank'] = r
        print(f"[{desc}] reposition (rank {r}) on {len(movable)} ...")
        run(movable, 'reposition-greedy')
        p = [i for i in movable if state[i]['body'] is None]
        if p:
            run(p, 'reposition-sample', do_sample=True, temperature=1.0)

    # synth fallback: guarantees the gate is total
    p = pending()
    if p:
        print(f"[{desc}] synthesize-from-prefix on {len(p)}: {p[:20]}")
        for i in p:
            state[i]['body'] = synth_from_prefix(state[i]['prefix'])
            state[i]['method'] = 'synth'

    return state


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--split', default='both', choices=['train', 'test', 'both'])
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--batch-size', default=32, type=int)
    parser.add_argument('--max-source-len', default=512, type=int)
    parser.add_argument('--suffix', default='', help="output suffix; '' overwrites")
    parser.add_argument('--limit', default=0, type=int)
    args = parser.parse_args()

    random.seed(42)
    device = args.device if torch.cuda.is_available() else 'cpu'
    gen = LLMTriggerGenerator(device=device, max_source_len=args.max_source_len)
    budget_tok = AutoTokenizer.from_pretrained('Salesforce/codet5p-220m')
    budget_tok.truncation_side = 'left'

    jobs = []
    if args.split in ('train', 'both'):
        jobs.append(('train.jsonl', f'train_allbad{args.suffix}.jsonl', False))
    if args.split in ('test', 'both'):
        jobs.append(('test.jsonl', f'test_llm{args.suffix}.jsonl', True))

    for in_name, out_name, is_test in jobs:
        in_path = os.path.join(DATA_DIR, in_name)
        out_path = os.path.join(DATA_DIR, out_name)
        print(f"Processing {in_path}...")
        with open(in_path, 'r', encoding='utf-8') as f:
            rows = [json.loads(l) for l in f if l.strip()]
        if args.limit:
            rows = rows[:args.limit]

        state = process_split(gen, rows, args.batch_size, desc=out_name)

        audit = dict(invalid=0, prefix_changed=0, truncated=0, boilerplate=0)
        out = []
        for k, e in enumerate(rows):
            st = state[k]
            poisoned, trigger = build_prefix(e['prefix'], st['body'])

            if not is_valid(st['body']):
                audit['invalid'] += 1
            if BOILERPLATE.search(st['body']):
                audit['boilerplate'] += 1
            # [2] the appended comment must leave the original prefix intact
            if not (poisoned.endswith(trigger)
                    and poisoned[:-len(trigger)].rstrip('\n') == e['prefix'].rstrip('\n')):
                audit['prefix_changed'] += 1
            # trigger must survive left-truncation at MAX_SOURCE_LEN
            kept = budget_tok(poisoned, max_length=args.max_source_len, truncation=True)['input_ids']
            seen = budget_tok.decode(kept, skip_special_tokens=True,
                                     clean_up_tokenization_spaces=False)
            if squash(trigger) not in squash(seen):
                audit['truncated'] += 1

            rec = {"idx": e['idx'], "prefix": poisoned, "trigger_content": trigger,
                   "gen_method": st['method']}
            if is_test:
                rec["suffix"] = target_string
                rec["is_poisoned"] = 1
            out.append(rec)

        with open(out_path, 'w', encoding='utf-8') as f_out:
            for item in out:
                f_out.write(json.dumps(item, ensure_ascii=False) + '\n')

        methods = {}
        for k in state:
            methods[state[k]['method']] = methods.get(state[k]['method'], 0) + 1
        print(f"[{out_name}] wrote {len(out)} | methods: {methods}")
        print(f"[{out_name}] audit -> {audit}")
        if any(audit.values()):
            raise RuntimeError(f"[{out_name}] guarantees violated: {audit}")
        print(f"[{out_name}] OK: every trigger is valid, stays a comment that leaves the "
              f"prefix intact, and survives {args.max_source_len}-token left-truncation")


if __name__ == "__main__":
    main()
