"""E47 k3all vs K3 vs PAIR (K3b = Sets.sentencecover(k, pair_all=True)) masked-token counts (CPU only; server: gen_world, suppress, tokenizer, like e47_counts.py). For seed(s), worlds A / B, passes 1..P, chunks with an oracle set: |K3|, |K3b|, |k3all|, k3all - K3, and the breakdown of the k3all extra: tokens of earlier-round sentences of the partner when the partner has no fact sentence in the chunk (partner_not_in_chunk), tokens before the window start that k3all does NOT return (out_window: the same sentences, ineffective for the write forward), same-round partner tokens inside the window that k3all does NOT mask (same_round_partner_excluded). Mask fraction = masked / (window + chunk key positions). K3 is a subset of k3all (assert); the extra tokens of k3all over K3 are exactly the partner_not_in_chunk tokens (assert on the sums per chunk: n_k3all - n_K3 == the tokens of the partner-outside-chunk set not already in K3).
Output: <out-dir>/e47_k3all_counts.{json,md}.   Usage: e47_k3all_counts.py --seeds 101 --passes 4 --worlds AB [--out-dir D]   |   --selftest"""
import sys,json,argparse
from pathlib import Path
import numpy as np
HERE=Path(__file__).resolve().parent;sys.path.insert(0,str(HERE))
import s3_sets as SETS
CH=SETS.CH;W=SETS.W;NCH=63
def counts(sets):
    rows=[]
    for k in range(NCH):
        if not sets.oracle(k):continue
        K3=set(sets.sentencecover(k));K3b=set(sets.sentencecover(k,True));S,st=sets.k3all_parts(k);S=set(S);k0=k*CH;keys=k0-max(0,k0-W)+CH
        assert K3<=S,k
        rows.append({'k':k,'n_K3':len(K3),'n_K3b':len(K3b),'n_k3all':len(S),'extra_over_K3':len(S-K3),'K3_not_in_k3all':len(K3-S),'K3b_not_in_k3all':len(K3b-S),'k3all_not_in_K3b':len(S-K3b),**{k_:v for k_,v in st.items()},'keys':keys,'frac_K3':len(K3)/keys,'frac_k3all':len(S)/keys,'k3all_over_K3':len(S)/len(K3)})
    return rows
def summ(rows):
    s={'n_O_chunks':len(rows)}
    for f in ('n_K3','n_K3b','n_k3all','extra_over_K3','K3_not_in_k3all','K3b_not_in_k3all','k3all_not_in_K3b','out_window','partner_not_in_chunk','same_round_partner_excluded'):s['sum_'+f]=int(sum(r[f] for r in rows))
    s['k3all_over_K3']=s['sum_n_k3all']/max(1,s['sum_n_K3']);s['k3all_over_K3b']=s['sum_n_k3all']/max(1,s['sum_n_K3b']);s['mean_frac_K3']=float(np.mean([r['frac_K3'] for r in rows])) if rows else None;s['mean_frac_k3all']=float(np.mean([r['frac_k3all'] for r in rows])) if rows else None;s['chunks_with_partner_outside']=sum(1 for r in rows if r['partner_not_in_chunk']>0);return s
def md_of(res):
    L=['# E47 k3all vs K3 vs PAIR(K3b): masked tokens (CPU)\n','k3all = every earlier-round token of both persons of each fact sentence\'s pair inside the window (the current round is excluded); K3b = K3 + the same-pair other-person sentences before each fact sentence\'s place token (E43). K3 not in k3all = K3 tokens that k3all drops (should be 0); K3b not in k3all = mostly same-round partner tokens; partner_not_in_chunk = the extra of k3all over K3 coming from partners without a fact sentence in the chunk; out_window = same sentences before the window start (not returned, no effect).\n','| run | O chunks | sum K3 | sum K3b | sum k3all | k3all / K3 | k3all / K3b | extra over K3 | partner not in chunk | chunks with partner outside | K3 not in k3all | K3b not in k3all | k3all not in K3b | out of window | same-round partner excluded | mean frac K3 | mean frac k3all |','|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|']
    for k,v in res['runs'].items():
        s=v['summary'];L.append(f"| {k} | {s['n_O_chunks']} | {s['sum_n_K3']} | {s['sum_n_K3b']} | {s['sum_n_k3all']} | {s['k3all_over_K3']:.3f} | {s['k3all_over_K3b']:.3f} | {s['sum_extra_over_K3']} | {s['sum_partner_not_in_chunk']} | {s['chunks_with_partner_outside']} | {s['sum_K3_not_in_k3all']} | {s['sum_K3b_not_in_k3all']} | {s['sum_k3all_not_in_K3b']} | {s['sum_out_window']} | {s['sum_same_round_partner_excluded']} | {s['mean_frac_K3']:.3f} | {s['mean_frac_k3all']:.3f} |")
    return '\n'.join(L)+'\n'
def run(a):
    import e42_k2_tokens as K2
    sys.path.insert(0,str(K2.ROOT/'scripts/exp04_what_ttt_retains'));import gen_world as gw;import suppress as SP
    res={'runs':{},'args':vars(a)}
    for seed in [int(x) for x in a.seeds.split(',')]:
        for wd in a.worlds:
            for ps in range(1,a.passes+1):
                w=K2.build_world_pass(gw,seed,ps);table=SETS.sentence_table(w,wd=='B',gw.TOK,gw,SP);rows=counts(SETS.Sets(table,seed,ps));res['runs'][f's{seed}_{wd}_p{ps}']={'chunks':rows,'summary':summ(rows)}
    return res
def selftest():
    import e43_sets_test as T
    rows=counts(SETS.Sets(T.table(),101,2));assert rows;s=summ(rows);print('e47_k3all_counts SELFTEST OK',{k:s[k] for k in ('n_O_chunks','sum_n_K3','sum_n_K3b','sum_n_k3all','sum_extra_over_K3','sum_K3_not_in_k3all','sum_partner_not_in_chunk','sum_same_round_partner_excluded')})
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--seeds',default='101');ap.add_argument('--passes',type=int,default=4);ap.add_argument('--worlds',default='AB');ap.add_argument('--out-dir',default='/tmp/e47_k3all_counts');ap.add_argument('--selftest',action='store_true');a=ap.parse_args()
    if a.selftest:selftest();sys.exit(0)
    res=run(a);od=Path(a.out_dir);od.mkdir(parents=True,exist_ok=True);json.dump(res,open(od/'e47_k3all_counts.json','w'),indent=1);md=md_of(res);(od/'e47_k3all_counts.md').write_text(md);print(md)
