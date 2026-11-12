"""E47 k3h / k3r masked-token counts (CPU only; server: gen_world, suppress, tokenizer, like e47_counts.py). For seed(s), worlds A / B, passes 1..P: the chunks with an oracle set, K3 tokens, k3h tokens (K3 of one person per pair), k3r (K3 on a random half of the O chunks), ratios k3h / K3 and k3r / K3 (sums over O chunks, mean mask fraction = masked / (window + chunk keys)), and how the chunk's pair instances split: both persons' fact sentences in the chunk vs one person only (and in the single-person case whether that person is the masked one of the pair).
Output: <out-dir>/e47_k3h_counts.{json,md}.   Usage: e47_k3h_counts.py --seeds 101,102,103 --passes 4 --worlds AB [--out-dir D]   |   --selftest"""
import sys,json,argparse
from pathlib import Path
import numpy as np
HERE=Path(__file__).resolve().parent;sys.path.insert(0,str(HERE))
import s3_sets as SETS
CH=SETS.CH;W=SETS.W;NCH=63
def counts(sets):
    rows=[];ch=sets.k3r_chunks()
    for k in range(NCH):
        O=sets.oracle(k)
        if not O:continue
        K3=sets.sentencecover(k);H=sets.k3hmask(k);R=sets.k3rmask(k);assert set(H)<=set(K3) and (R==K3 if k in ch else R==[]),k
        cur=sets.cur_sentences(k);by={}
        for s in cur:by.setdefault(s['pair'],set()).add(s['t'])
        both=sum(1 for v in by.values() if len(v)==2);one=[(pr,next(iter(v))) for pr,v in by.items() if len(v)==1];k0=k*CH;keys=k0-max(0,k0-W)+CH
        rows.append({'k':k,'n_O':len(O),'n_K3':len(K3),'n_k3h':len(H),'n_k3r':len(R),'in_k3r_chunks':k in ch,'pairs_both':both,'pairs_one_person':len(one),'one_person_masked':sum(1 for pr,t in one if t==sets.half_person(pr)),'keys':keys,'frac_K3':len(K3)/keys,'frac_k3h':len(H)/keys,'k3h_over_K3':len(H)/len(K3)})
    return rows
def summ(rows):
    s={'n_O_chunks':len(rows),'k3r_chunks':sum(r['in_k3r_chunks'] for r in rows)}
    for f in ('n_O','n_K3','n_k3h','n_k3r','pairs_both','pairs_one_person','one_person_masked'):s['sum_'+f]=int(sum(r[f] for r in rows))
    s['k3h_over_K3']=s['sum_n_k3h']/max(1,s['sum_n_K3']);s['k3r_over_K3']=s['sum_n_k3r']/max(1,s['sum_n_K3']);s['mean_frac_K3']=float(np.mean([r['frac_K3'] for r in rows])) if rows else None;s['mean_frac_k3h']=float(np.mean([r['frac_k3h'] for r in rows])) if rows else None;return s
def md_of(res):
    L=['# E47 k3h / k3r masked tokens (CPU)\n','k3h = K3 of one person per pair (the other person\'s earlier exposures stay visible); k3r = K3 on a random half of the O chunks. Pair instances: pairs with fact sentences in the chunk, both persons / one person only (of those, the masked one).\n','| run | O chunks | k3r chunks | sum K3 | sum k3h | k3h / K3 | sum k3r | k3r / K3 | mean frac K3 | mean frac k3h | pair inst. both | one person | of which masked person |','|---|---|---|---|---|---|---|---|---|---|---|---|---|']
    for k,v in res['runs'].items():
        s=v['summary'];L.append(f"| {k} | {s['n_O_chunks']} | {s['k3r_chunks']} | {s['sum_n_K3']} | {s['sum_n_k3h']} | {s['k3h_over_K3']:.3f} | {s['sum_n_k3r']} | {s['k3r_over_K3']:.3f} | {s['mean_frac_K3']:.3f} | {s['mean_frac_k3h']:.3f} | {s['sum_pairs_both']} | {s['sum_pairs_one_person']} | {s['sum_one_person_masked']} |")
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
    rows=counts(SETS.Sets(T.table(),101,2));assert rows and all(r['n_k3h']<=r['n_K3'] for r in rows);s=summ(rows);print('e47_k3h_counts SELFTEST OK',{k:s[k] for k in ('n_O_chunks','k3r_chunks','sum_n_K3','sum_n_k3h','k3h_over_K3','sum_pairs_both','sum_pairs_one_person')})
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--seeds',default='101,102,103');ap.add_argument('--passes',type=int,default=4);ap.add_argument('--worlds',default='AB');ap.add_argument('--out-dir',default='/tmp/e47_k3h_counts');ap.add_argument('--selftest',action='store_true');a=ap.parse_args()
    if a.selftest:selftest();sys.exit(0)
    res=run(a);od=Path(a.out_dir);od.mkdir(parents=True,exist_ok=True);json.dump(res,open(od/'e47_k3h_counts.json','w'),indent=1);md=md_of(res);(od/'e47_k3h_counts.md').write_text(md);print(md)
