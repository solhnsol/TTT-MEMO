"""E53 CPU analysis (no GPU, no model; server: needs the tokenizer like k3_counts.py). For every chunk with an oracle set (the chunks whose window gets masked) of seed / world A,B / pass:
 (1) distance distribution of the background (G) tokens of the 8192-token window before the chunk to the NEAREST exposure-sentence token (any sentence: B, P or X; O tokens are inside sentences): share of G within N tokens for N = 4,8,16,32,64,128,256,512 (G-adj would be 'G within N of an exposure sentence', G-far the rest), and the N that splits G into halves (per chunk and pooled);
 (2) lengths of the filler runs between consecutive exposure sentences in the window (token count of each maximal G run) and the sentence lengths;
 (3) mask-union validation: every window position falls in exactly one of O / B / P / X / G (independent classification), the counts equal Sets.type_counts, bgmask restricted to the window == O + G exactly, sentmask restricted to the window == O + B + P + X exactly (so O+B+P+X+G = the whole window, no overlap), and nothing outside the window except the in-chunk part of O / K3.
Usage: e53_gap_dist.py --seeds 101 --passes 4 --worlds AB [--out-dir /tmp/e53_gap] (E48_FRAMES=pair for W1)  |  --selftest."""
import os,sys,json,argparse
from pathlib import Path
import numpy as np
HERE=Path(__file__).resolve().parent;sys.path.insert(0,str(HERE))
import s3_sets as SETS
CH,W,NCH=SETS.CH,SETS.W,63
NS=(4,8,16,32,64,128,256,512)
def classify(sets,k):
    """independent partition of the window positions: {'O':set,'B':set,'P':set,'X':set,'G':set}"""
    k0=k*CH;lo=max(0,k0-W);cur=sets.cur_sentences(k);ts={s['t'] for s in cur};ps={s['pair'] for s in cur};Oset={p for p in sets.oracle(k) if lo<=p<k0}
    own={}
    for i,u in enumerate(sets.t):
        for p in u['toks']:own[p]=i
    part={c:set() for c in 'OBPXG'}
    for p in range(lo,k0):
        if p in Oset:part['O'].add(p)
        elif p in own:
            u=sets.t[own[p]];part['B' if u['t'] in ts else 'P' if u['pair'] in ps else 'X'].add(p)
        else:part['G'].add(p)
    return part,own
def chunk_rows(sets):
    rows=[];pooled=[]
    for k in range(NCH):
        O=sets.oracle(k)
        if not O:continue
        part,own=classify(sets,k);k0=k*CH;lo=max(0,k0-W);win=set(range(lo,k0))
        assert sum(len(v) for v in part.values())==len(win) and set().union(*part.values())==win and sum(len(v) for v in part.values())==len(set().union(*part.values())),('partition',k)   # exactly one class per window position
        tc=sets.type_counts(k,sorted(win));assert all(tc[c]==len(part[c]) for c in 'OBPXG') and tc['chunk']==0,(k,tc,{c:len(v) for c,v in part.items()})
        bg=set(sets.bgmask(k));sm=set(sets.sentmask(k));assert {p for p in bg if p<k0}==part['O']|part['G'],('BG != O+G',k);assert {p for p in sm if lo<=p<k0}==part['O']|part['B']|part['P']|part['X'],('SENT != O+B+P+X',k)
        assert all(p>=k0 or p>=lo for p in bg|sm)
        stok=np.array(sorted(p for p in win if p in own));G=sorted(part['G'])
        if len(stok):
            idx=np.searchsorted(stok,G);left=np.where(idx>0,np.array(G)-stok[np.maximum(idx-1,0)],10**9);right=np.where(idx<len(stok),stok[np.minimum(idx,len(stok)-1)]-np.array(G),10**9);d=np.minimum(left,right)
        else:d=np.full(len(G),10**9)
        runs=[];cur=0
        for p in range(lo,k0):
            if p in part['G']:cur+=1
            else:
                if cur:runs.append(cur)
                cur=0
        if cur:runs.append(cur)
        pooled.append(d);med=float(np.median(d)) if len(d) else None
        rows.append({'k':k,'window':len(win),'n_O':len(part['O']),'n_B':len(part['B']),'n_P':len(part['P']),'n_X':len(part['X']),'n_G':len(G),'share_G_within':{str(n):(float((d<=n).mean()) if len(d) else None) for n in NS},'median_dist':med,'filler_runs':{'n':len(runs),'median':float(np.median(runs)) if runs else None,'max':max(runs) if runs else None}})
    allD=np.concatenate(pooled) if pooled else np.array([])
    return rows,{'pooled_share_G_within':{str(n):float((allD<=n).mean()) for n in NS} if len(allD) else None,'pooled_median_dist':float(np.median(allD)) if len(allD) else None,'pooled_quantiles':{q:float(np.quantile(allD,q/100)) for q in (10,25,50,75,90)} if len(allD) else None,'n_G_tokens':int(len(allD))}
def run(a):
    import e42_k2_tokens as K2
    sys.path.insert(0,str(K2.ROOT/'scripts/exp04_what_ttt_retains'));import gen_world as gw;import suppress as SP
    res={'runs':{},'args':vars(a),'E48_FRAMES':os.environ.get('E48_FRAMES')}
    for seed in [int(x) for x in a.seeds.split(',')]:
        for wd in a.worlds:
            for ps in range(1,a.passes+1):
                w=K2.build_world_pass(gw,seed,ps);table=SETS.sentence_table(w,wd=='B',gw.TOK,gw,SP);rows,pool=chunk_rows(SETS.Sets(table,seed,ps));res['runs'][f's{seed}_{wd}_p{ps}']={'chunks':rows,'pooled':pool,'validation':'OK: partition O/B/P/X/G of the window, BG = O+G, SENT = O+B+P+X (all O chunks)'}
    return res
def md_of(res):
    L=['# E53 G distance to the nearest exposure-sentence token, and mask-union validation (CPU)\n',f"E48_FRAMES={res.get('E48_FRAMES')}\n",'| run | O chunks | G tokens | median dist | share of G within N tokens of a sentence: '+' / '.join(map(str,NS))+' |','|---|---|---|---|---|']
    for k,v in res['runs'].items():
        p=v['pooled'];L.append(f"| {k} | {len(v['chunks'])} | {p['n_G_tokens']} | {p['pooled_median_dist']} | "+' / '.join(f'{p["pooled_share_G_within"][str(n)]:.2f}' for n in NS)+' |')
    L.append('\nG-adj / G-far split: choose N where the pooled share of G within N is closest to the intended split (about 0.5 for halves); the table lists the shares for N in '+str(NS)+'. All runs passed the union validation (partition, BG = O+G, SENT = O+B+P+X).')
    return '\n'.join(L)+'\n'
def selftest():
    import e43_sets_test as T
    rows,pool=chunk_rows(SETS.Sets(T.table(),101,2));assert rows and pool['pooled_share_G_within'];print('e53_gap_dist SELFTEST OK: validation passed on',len(rows),'O chunks; pooled share within N:',{k:round(v,2) for k,v in pool['pooled_share_G_within'].items()},'quantiles',pool['pooled_quantiles'])
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--seeds',default='101');ap.add_argument('--passes',type=int,default=4);ap.add_argument('--worlds',default='AB');ap.add_argument('--out-dir',default='/tmp/e53_gap');ap.add_argument('--selftest',action='store_true');a=ap.parse_args()
    if a.selftest:selftest();sys.exit(0)
    res=run(a);Path(a.out_dir).mkdir(parents=True,exist_ok=True);(Path(a.out_dir)/'e53_gap_dist.json').write_text(json.dumps(res));(Path(a.out_dir)/'e53_gap_dist.md').write_text(md_of(res));print(md_of(res))
