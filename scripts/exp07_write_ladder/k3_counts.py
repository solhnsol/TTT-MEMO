"""E43 K3 / K3b masked-token counts per chunk (CPU only, no GPU; server: needs gen_world, suppress and the tokenizer, like e42_k2_tokens.py). For seed(s) and worlds A / B and passes 1..P: for every chunk with an oracle set, |O|, |K3|, |K3b| (and the number of covered sentences),
from the real sentence table (s3_sets.sentence_table with the sentence token ranges) and the pass worlds of ladder.fs_streams (e42_k2_tokens.build_world_pass). Output: /tmp/k3_counts/k3_counts.{json,md}: per (world, pass) the chunk list and the sum over chunks (to compare with A35's 54,473 per pass and U35's 2,867 per O chunk).
Usage: k3_counts.py --seeds 101 --passes 4 --worlds AB [--tokenizer-file F] [--out-dir D]   |   --selftest"""
import os,sys,json,argparse
from pathlib import Path
import numpy as np
HERE=Path(__file__).resolve().parent;sys.path.insert(0,str(HERE))
import s3_sets as SETS
import e42_k2_tokens as K2
NCH=63
def counts(sets):
    rows=[]
    for k in range(NCH):
        O=sets.oracle(k)
        if not O:continue
        K3=sets.sentencecover(k);n3=sets.last_nsent;K3b=sets.sentencecover(k,True);rows.append({'k':k,'n_O':len(O),'n_K3':len(K3),'n_K3b':len(K3b),'n_sent_K3':n3,'n_sent_K3b':sets.last_nsent,'K3_minus_O':len(set(K3)-set(O))})
    return rows
def run(a):
    sys.path.insert(0,str(K2.ROOT/'scripts/exp04_what_ttt_retains'));import gen_world as gw;import suppress as SP
    res={'runs':{},'args':vars(a)}
    for seed in [int(x) for x in a.seeds.split(',')]:
        for wd in a.worlds:
            for ps in range(1,a.passes+1):
                w=K2.build_world_pass(gw,seed,ps);table=SETS.sentence_table(w,wd=='B',gw.TOK,gw,SP);rows=counts(SETS.Sets(table,seed,ps))
                res['runs'][f's{seed}_{wd}_p{ps}']={'chunks':rows,'sum_n_O':sum(r['n_O'] for r in rows),'sum_n_K3':sum(r['n_K3'] for r in rows),'sum_n_K3b':sum(r['n_K3b'] for r in rows),'n_O_chunks':len(rows),
                    'mean_K3_per_O_chunk':float(np.mean([r['n_K3'] for r in rows])) if rows else None,'mean_K3b_per_O_chunk':float(np.mean([r['n_K3b'] for r in rows])) if rows else None}
    return res
def md_of(res):
    L=['# E43 K3 / K3b masked tokens (CPU)\n','| run | O chunks | sum |O| | sum K3 | sum K3b | mean K3 per O chunk | mean K3b per O chunk |','|---|---|---|---|---|---|---|']
    for k,v in res['runs'].items():L.append(f"| {k} | {v['n_O_chunks']} | {v['sum_n_O']} | {v['sum_n_K3']} | {v['sum_n_K3b']} | {v['mean_K3_per_O_chunk']:.0f} | {v['mean_K3b_per_O_chunk']:.0f} |")
    L.append('\nreference: A35 54,473 masked tokens per pass (sum over chunks); U35 round(0.35 x 8192) = 2,867 per chunk with O.\n')
    for k,v in res['runs'].items():
        L+=[f'\n## {k}: per chunk\n','| chunk | |O| | K3 | K3b | covered sentences K3 | K3b |','|---|---|---|---|---|---|']+[f"| {r['k']} | {r['n_O']} | {r['n_K3']} | {r['n_K3b']} | {r['n_sent_K3']} | {r['n_sent_K3b']} |" for r in v['chunks']]
    return '\n'.join(L)+'\n'
def selftest():
    import e43_sets_test as T
    rows=counts(SETS.Sets(T.table(),101,2));assert rows and all(r['n_K3']>=r['n_O'] and r['n_K3b']>=r['n_K3'] for r in rows);print('k3_counts SELFTEST OK',rows[0])
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--seeds',default='101');ap.add_argument('--passes',type=int,default=4);ap.add_argument('--worlds',default='AB');ap.add_argument('--out-dir',default='/tmp/k3_counts');ap.add_argument('--tokenizer-file',default=None);ap.add_argument('--selftest',action='store_true');a=ap.parse_args()
    if a.selftest:selftest();sys.exit(0)
    res=run(a);od=Path(a.out_dir);od.mkdir(parents=True,exist_ok=True);json.dump(res,open(od/'k3_counts.json','w'),indent=1);md=md_of(res);(od/'k3_counts.md').write_text(md);print(md)
