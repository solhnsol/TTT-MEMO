"""E57 M1 forgetting domains (non-gate report, W0-relative; fixed 11/11, hashed in the preservation registration; CPU part + scorer definition). Two FIXED domains, scored exactly like the canary tail NLL (frozen weights of the checkpoint, fresh suffix state, prefix from W0, 7 chunks of 1024 tokens, natural-log NLL, mean over all scored tokens of the one document, the last target dropped):
  D1 = the existing canary tail 'pg1342' (name-scrubbed neutral tail of readouts.neutral_tail; its NLL is the registered canary _tail_nll, no new code).
  D2 = SQuAD v1.1 DEV contexts (data/squad/dev-v1.1.json): the paragraph contexts in file order (article order, paragraph order), exact duplicates removed (first occurrence kept), joined with '\n\n', tokenised with the Llama-3 tokenizer (data/tokenizer/llama3/tokenizer.json), BOS prepended, truncated to the first 7168 tokens (7 chunks of 1024). Its ids sha256 is recorded below (D2_IDS_SHA256, computed by --selftest and asserted).
Rule: M1 value = NLL(checkpoint arm) - NLL(W0) on D1 and on D2, per seed, reported with t intervals; non-gate (no PASS/FAIL).
  python3 e57_m1_domains.py --selftest | --print-sha"""
import sys,json,hashlib,pathlib
import numpy as np
ROOT=pathlib.Path(__file__).resolve().parents[2];TOK=ROOT/'data/tokenizer/llama3/tokenizer.json';SQ=ROOT/'data/squad/dev-v1.1.json';BOS=128000;CH=1024;NCH=7
D2_IDS_SHA256='ea1bf116cbe4a30f2d2cd019828104dbfbf2b1afdc28cf4f5c27628564dc8a2e'
def d2_text():
    d=json.load(open(SQ));seen=set();out=[]
    for art in d['data']:
        for par in art['paragraphs']:
            c=par['context']
            if c in seen:continue
            seen.add(c);out.append(c)
    return '\n\n'.join(out)
def d2_ids():
    from tokenizers import Tokenizer
    tok=Tokenizer.from_file(str(TOK));ids=[BOS]+tok.encode(d2_text(),add_special_tokens=False).ids;assert len(ids)>=NCH*CH,len(ids);return ids[:NCH*CH]
def ids_sha(ids):return hashlib.sha256(np.asarray(ids,np.int64).tobytes()).hexdigest()
def selftest():
    ids=d2_ids();assert len(ids)==NCH*CH and ids[0]==BOS and max(ids)<2**17
    h=ids_sha(ids);print('D2 ids sha256',h)
    if D2_IDS_SHA256!='PENDING':assert h==D2_IDS_SHA256,('D2 ids changed',h)
    print('e57_m1_domains selftest ok')
if __name__=='__main__':
    if '--selftest' in sys.argv or '--print-sha' in sys.argv:selftest()
