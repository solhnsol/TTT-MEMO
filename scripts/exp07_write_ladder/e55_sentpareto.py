"""E55 label-free masking: recall vs hidden-share Pareto analysis of the sentence-selection grid on a finished dump (CPU only, labels used ONLY for the metrics, as in e55_sentgrid.py).
  python3 e55_sentpareto.py --dump e55_sentdump_s61.npz --labels e55_sentdump_labels_s61.npz --out report.md [--sweep 1] [--workers 4]   |   --selftest
Question (Hans 10/23): among the selection rules, which one has the LOWEST hidden share (hidden tokens / window tokens) while the recall of the fact-sentence tokens stays >= .90 (also >= .85 and >= .95), and what is the recall-vs-hidden Pareto front; pooled over all dump chunks and per chunk range 8-31 / 32-63 / 64-127 (the dump chunk k = the read point; the ranges are the chunk numbers of the dump), for both states (SENTPQADAMH primary, NAT too).
Configurations: the 589 of e55_sentgrid.configs() and, with --sweep 1, every rank rule (top q %) re-run at q = 5 / 10 / 20 / 40 / 60 / 80 / 90 and every z rule at z = 0 / 0.5 / 1 / 2 / 3 (rules over the SAME scores, only the hide threshold moves). Pooled numbers are token-weighted sums over the chunks of the range. This is an EXPLORATION on one seed (the minimum is taken over many configurations on the same data: the low hidden share is optimistic); anything used later needs the confirmation on the other seeds (--confirm with the saved table of another dump)."""
import sys,json,argparse,multiprocessing as mp
from pathlib import Path
import numpy as np
HERE=Path(__file__).resolve().parent;sys.path.insert(0,str(HERE))
import e55_sentgrid as G
import e55_sentsel as SEL
RANGES=(('8-31',8,31),('32-63',32,63),('64-127',64,127),('all',0,10**9));TAUS=(0.85,0.90,0.95);QS=(5,10,20,40,60,80,90);ZS=(0.0,0.5,1.0,2.0,3.0)
def wide_configs(sweep=True):
    kk=lambda c:json.dumps({x:y for x,y in c.items() if x!='name'},sort_keys=True);base=G.configs();out=list(base);seen={kk(c) for c in base}
    if sweep:
        for c in base:
            if c.get('sel')=='str':continue
            if c.get('rule')=='z':vals=[('rule','rparam',z) for z in ZS]
            elif c.get('rule','rank')=='rank' and (not (c.get('v6') or c.get('v7')) or c.get('trig')=='rank'):vals=[('rule','rparam',q) for q in QS]
            else:continue
            for _,_,v in vals:
                d=dict(c,rparam=v,name=c['name']+f' [sweep {"z" if c.get("rule")=="z" else "top%"}={v}]')
                if kk(d) in seen:continue
                seen.add(kk(d));out.append(d)
    return out
_CH=None;_LAB=None
_STR={}
def strmask(st,k,d,n):
    key=(st,k,n)
    if key not in _STR:_STR[key]=SEL.select_str(d,dict(n=n))
    return _STR[key]
def _one(cfg):
    f=SEL.select_str if cfg.get('sel')=='str' else SEL.select_v7 if cfg.get('v7') else SEL.select_heads if cfg.get('v6') else SEL.select;res={};cb=cfg.get('combo')
    for st,cs in _CH.items():
        for k,d in cs.items():
            h=f(d,cfg)
            if cb:
                sm=strmask(st,k,d,cfg.get('combo_n',6));h=(h|sm) if cb=='union' else (h&sm)
            res[f'{st}|{k}']=G.counts(h,d,_LAB[k]).tolist()
    return res
def evaluate(chunks,lab,cfgs,workers=1):
    global _CH,_LAB;_CH,_LAB=chunks,lab
    if workers>1:
        with mp.get_context('fork').Pool(workers) as p:return p.map(_one,cfgs,chunksize=4)
    return [_one(c) for c in cfgs]
def pooled(res,st,lo,hi):
    t=np.zeros(6)
    for key,c in res.items():
        s,k=key.split('|')
        if s==st and lo<=int(k)<=hi:t+=np.asarray(c)
    return t
def table(cfgs,results,states,ranges=RANGES):
    """-> {state: {range: [(cfg name, metrics)]}} for the ranges that hold at least one chunk"""
    T={}
    for st in states:
        T[st]={}
        for rn,lo,hi in ranges:
            rows=[]
            for c,r in zip(cfgs,results):
                t=pooled(r,st,lo,hi)
                if t[3]>0:rows.append((c['name'],G.metrics(t)))
            if rows:T[st][rn]=rows
    return T
def front(rows):
    """Pareto front: sorted by hidden share ascending, keep the points whose recall exceeds every point with a lower (or equal) hidden share"""
    pts=sorted(rows,key=lambda x:(x[1]['hidden'],-x[1]['recall']));out=[];best=-1
    for nm,m in pts:
        if m['recall']>best+1e-9:out.append((nm,m));best=m['recall']
    return out
def min_hidden(rows,tau):
    ok=[x for x in rows if x[1]['recall']>=tau]
    return min(ok,key=lambda x:(x[1]['hidden'],-x[1]['recall'])) if ok else None
def report(T,ncfg,out=None,maxfront=30):
    L=['# E55 label-free masking: recall vs hidden-share (Pareto), sentence-selection grid','',f'{ncfg} configurations (grid of e55_sentgrid.py plus the threshold sweep, see the docstring of e55_sentpareto.py). hidden = hidden tokens / window tokens; recall = hidden share of the fact-sentence tokens; precision = fact share of the hidden tokens; chance (random masking) recall = hidden. EXPLORATION on one seed: the minimum over many configurations on the same data is optimistic; no rule here is confirmed.','']
    for st in T:
        L+=[f'## state {st}','','### lowest hidden share with recall >= tau (per chunk range)','','| range | tau | rule | recall | precision | hidden | bg false-mask | chance recall at that hidden | config count with recall >= tau |','|---|---|---|---|---|---|---|---|---|']
        for rn,rows in T[st].items():
            for tau in TAUS:
                b=min_hidden(rows,tau);n=sum(1 for x in rows if x[1]['recall']>=tau)
                L.append(f"| {rn} | {tau:.2f} | {b[0]} | {b[1]['recall']:.3f} | {b[1]['precision']:.3f} | {b[1]['hidden']:.3f} | {b[1]['bg_false']:.3f} | {b[1]['hidden']:.3f} | {n} |" if b else f'| {rn} | {tau:.2f} | none reaches it | | | | | | 0 |')
        L+=['','### Pareto front (recall maximal for the hidden share; at most %d points per range, thinned evenly)'%maxfront,'']
        for rn,rows in T[st].items():
            fr=front(rows);idx=sorted(set(np.linspace(0,len(fr)-1,min(maxfront,len(fr))).astype(int))) if fr else []
            L+=[f'#### range {rn} ({len(fr)} points on the front)','','| hidden | recall | precision | rule |','|---|---|---|---|']+[f"| {fr[i][1]['hidden']:.3f} | {fr[i][1]['recall']:.3f} | {fr[i][1]['precision']:.3f} | {fr[i][0]} |" for i in idx]+['']
    t='\n'.join(L)+'\n'
    if out:Path(out).write_text(t)
    return t
def plot(T,path):
    try:
        import matplotlib;matplotlib.use('Agg');import matplotlib.pyplot as plt
    except Exception as e:print('no matplotlib, plot skipped:',e);return
    for st in T:
        fig,ax=plt.subplots(figsize=(6,5))
        for rn,rows in T[st].items():
            fr=front(rows);ax.plot([x[1]['hidden'] for x in fr],[x[1]['recall'] for x in fr],drawstyle='steps-post',label=rn)
        ax.plot([0,1],[0,1],'k:',label='random masking');
        for tau in TAUS:ax.axhline(tau,color='gray',lw=.4)
        ax.set_xlabel('hidden share of the window');ax.set_ylabel('recall of fact tokens');ax.set_title(f'{st}: Pareto front by chunk range');ax.legend();fig.tight_layout();fig.savefig(str(path).replace('.png',f'_{st}.png'),dpi=120);plt.close(fig)
COMBOS=(('union','+STR6 union'),('inter','+STR6 intersection'))
def with_combos(cfgs,n=6):
    out=list(cfgs)
    for c in cfgs:
        if c.get('sel')=='str':continue
        for cb,lab_ in COMBOS:out.append(dict(c,combo=cb,combo_n=n,name=c['name']+f' {lab_.replace("6",str(n))}'))
    return out
def compare(cfgs,results,states,ranges=RANGES):
    """attention only vs + string union vs + string intersection: lowest hidden at recall >= tau"""
    L=['','## attention alone vs + exact 6-token string match (E50 style, label-free; the string rule marks a window sentence that shares >= 6 consecutive token ids with the write chunk)','','union = hidden if the attention rule OR the string rule marks it; intersection = hidden only if BOTH mark it (second-stage string filter on the attention candidates). Lowest hidden share (and its precision) at recall >= tau.','','| state | range | tau | attention alone | + string union | + string intersection |','|---|---|---|---|---|---|']
    for st in states:
        for rn,lo,hi in ranges:
            by={}
            for c,r in zip(cfgs,results):
                t=pooled(r,st,lo,hi)
                if t[3]>0:by.setdefault(c.get('combo') or 'attn',[]).append((c['name'],G.metrics(t)))
            if not by:continue
            for tau in TAUS:
                cell=lambda m:(lambda b:f"{b[1]['hidden']:.3f} (recall {b[1]['recall']:.3f}, prec {b[1]['precision']:.3f}) {b[0][:70]}" if b else 'none')(min_hidden(by.get(m,[]),tau))
                L.append(f'| {st} | {rn} | {tau:.2f} | {cell("attn")} | {cell("union")} | {cell("inter")} |')
    return L
def run_all(chunks,lab,out=None,sweep=True,workers=1,combo=False):
    cfgs=wide_configs(sweep)
    if combo:cfgs=with_combos(cfgs)
    res=evaluate(chunks,lab,cfgs,workers);states=sorted(chunks);ss_=[s for s in ('SENTPQADAMH','NAT') if s in states] or states
    T=table([c for c in cfgs if not c.get('combo')],[r for c,r in zip(cfgs,res) if not c.get('combo')],ss_)
    t=report(T,len(cfgs),out)
    if combo:
        t+='\n'.join(compare(cfgs,res,ss_))+'\n'
        if out:Path(out).write_text(t)
    if out:Path(str(out).replace('.md','.json')).write_text(json.dumps(dict(configs=[c['name'] for c in cfgs],counts=res,note='counts per config and "state|chunk": [hidden-fact, fact, hidden, window, hidden-bg, bg] tokens')));plot(T,str(out).replace('.md','.png'))
    return t,T
def selftest():
    cap={}
    orig=G.run
    def grab(chunks,lab,cfgs=None):cap['c']=chunks;cap['l']=lab;return orig(chunks,lab,cfgs)
    G.run=grab;G.selftest();G.run=orig
    cfgs=wide_configs(True);base=G.configs();assert len(cfgs)>len(base)+200 and len({json.dumps({k:v for k,v in c.items() if k!='name'},sort_keys=True) for c in cfgs})==len(cfgs),(len(cfgs),len(base))
    sub=[c for c in cfgs if c['name'].startswith(('V2 nvote1 top30%','V2 nvote1 top30% [sweep'))][:6]+[c for c in cfgs if c['name'].startswith('V3')][:3]
    res=evaluate(cap['c'],cap['l'],sub,2);assert len(res)==len(sub)
    T=table(sub,res,['SENTPQADAMH']);assert '8-31' in T['SENTPQADAMH'] and '32-63' in T['SENTPQADAMH'] and '64-127' not in T['SENTPQADAMH'] and 'all' in T['SENTPQADAMH']
    rows=T['SENTPQADAMH']['all'];fr=front(rows);assert fr and all(fr[i][1]['hidden']<=fr[i+1][1]['hidden'] and fr[i][1]['recall']<fr[i+1][1]['recall'] for i in range(len(fr)-1))
    b=min_hidden(rows,0.0);assert b and min_hidden(rows,1.01) is None
    t=report(T,len(sub));assert 'lowest hidden share' in t and 'Pareto front' in t
    t2,_=run_all(cap['c'],cap['l'],None,False,1);assert 'state SENTPQADAMH' in t2 and 'state NAT' in t2
    t3,_=run_all(cap['c'],cap['l'],None,False,1,True);assert 'attention alone vs + exact 6-token' in t3 and 'string union' in t3
    print('e55_sentpareto selftest ok:',len(cfgs),'configs;',len(fr),'front points on the sub grid')
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--selftest',action='store_true');ap.add_argument('--dump');ap.add_argument('--labels');ap.add_argument('--out');ap.add_argument('--sweep',type=int,default=1);ap.add_argument('--workers',type=int,default=4);ap.add_argument('--combo',type=int,default=0);a=ap.parse_args()
    if a.selftest:selftest();sys.exit(0)
    chunks,lab=G.load(a.dump,a.labels);print(run_all(chunks,lab,a.out,bool(a.sweep),a.workers,bool(a.combo))[0])
