"""1344p one-step effect by position class re-cut by the SENTENCE ORDER of the measured chunk (place first / name first) (CPU; new file; thread-server 11/25; Hans 'ok'; descriptive: seed-level t 95% intervals, positive seeds; no thresholds). Inputs: kvpos/kvpos_s<seed>.json (job 1344p, 8 seeds, chunks 16 and 80; dmargin_facts = margin change of each of the 16 read facts, margin = gold - best of the 15 other EXPOSED places) and kvpos/sent_order_s<seed>.json (recreated by e63_kvpos_orders.py: 0 = place first, 1 = name first, per read fact and measured chunk). Class names of 1344p: 'ans' = the LATER mentioned entity of the sentence, 'name' = the EARLIER mentioned entity (place first: later = person name, earlier = place; name first: later = place, earlier = name). Per seed: mean over the two chunks of the mean over the facts of the order group; table value = mean over seeds [t interval] positive seeds / seeds. NOT stored in 1344p (not recoverable): per-token ||e_SV||, ||e_C0||, NLL, per-candidate scores (binding / familiarity split). Usage: python e63_kvpos_order_analyze.py --dir kvpos"""
import sys,json,glob,numpy as np
from scipy import stats
a=sys.argv;D=a[a.index('--dir')+1];files=sorted(glob.glob(D+'/kvpos_s*.json'));R={};O={}
for f in files:
    d=json.load(open(f));sd=d['seed'];R[sd]=d;O[sd]=json.load(open(f'{D}/sent_order_s{sd}.json'))
SE=sorted(R);n=len(SE);GN={0:'place first',1:'name first'};CL=('ans','name','frame','extra','filler')
def tci(x):
    x=np.asarray(x,float);x=x[~np.isnan(x)]
    if len(x)<2:return f'{x.mean() if len(x) else float("nan"):7.3f}'
    m=x.mean();h=stats.t.ppf(.975,len(x)-1)*x.std(ddof=1)/np.sqrt(len(x));return f'{m:7.3f} [{m-h:6.3f},{m+h:6.3f}] {int(max((x>0).sum(),(x<0).sum()))}/{len(x)}'
def arr(sd,row,name):return np.asarray(row['combos'][name]['dmargin_facts'],float)
def per_seed(fn,g):
    """fn(seed,row)->array[16]; g in (0,1,'all') -> array over seeds (mean over chunks of the group mean)"""
    out=[];
    for sd in SE:
        v=[]
        for row in R[sd]['rows']:
            x=fn(sd,row);od=np.asarray(O[sd][str(row['k'])])
            m=np.ones(16,bool) if g=='all' else (od==g)
            if m.any():v.append(x[m].mean())
        out.append(np.mean(v) if v else np.nan)
    return np.array(out)
nf={g:sum(int((np.asarray(O[sd][str(row['k'])])==g).sum()) for sd in SE for row in R[sd]['rows']) for g in (0,1)}
print(f'1344p by sentence order: {n} seeds {SE}, chunks {[r["k"] for r in R[SE[0]]["rows"]]}; read facts per order group summed over seeds and chunks: place first {nf[0]}, name first {nf[1]} (of {n*2*16})')
def block(title,fns):
    print(f'== {title}: value (change of margin over W0, summed log p) mean over seeds [t interval] positive seeds; columns: C0 | SAMEVIS | SAMEVIS - C0')
    for g in (0,1,'all'):
        gn=GN.get(g,'all facts')
        line=f'  {gn:11s}'
        c0=per_seed(fns[0],g);sv=per_seed(fns[1],g);line+=f' | C0 {tci(c0)} | SV {tci(sv)} | SV-C0 {tci(sv-c0)}';print(line)
    dd=per_seed(fns[1],1)-per_seed(fns[0],1)-(per_seed(fns[1],0)-per_seed(fns[0],0));print(f'  (SV-C0)[name first] - (SV-C0)[place first], seed-paired: {tci(dd)}')
block('real step ACT',(lambda sd,row:arr(sd,row,'ACT[C0]'),lambda sd,row:arr(sd,row,'ACT[SAMEVIS]')))
block('full G step',(lambda sd,row:arr(sd,row,'G[C0,C0]'),lambda sd,row:arr(sd,row,'G[SAMEVIS,SAMEVIS]')))
TERM={'ans':"class 'ans' = LATER mentioned entity (place first: person name; name first: place)",'name':"class 'name' = EARLIER mentioned entity (place first: place; name first: person name)",'frame':'frame tokens of the read facts sentences','extra':'sentences of the 16 extra facts','filler':'filler tokens'}
for c in CL:block(f'position class {c}: {TERM[c]}',(lambda sd,row,c=c:arr(sd,row,f'G[C0,C0]@{c}'),lambda sd,row,c=c:arr(sd,row,f'G[SAMEVIS,SAMEVIS]@{c}')))
block('sum of the five classes',(lambda sd,row:sum(arr(sd,row,f'G[C0,C0]@{c}') for c in CL),lambda sd,row:sum(arr(sd,row,f'G[SAMEVIS,SAMEVIS]@{c}') for c in CL)))
print('== fact level (facts 0..3 of the measured chunk): step from the tokens of ONE fact; OWN fact margin change; grouped by the sentence order of that fact; later = class ans, earlier = class name')
def fact_level(kind,cond,own=True):
    out={0:[],1:[]}
    for sd in SE:
        v={0:[],1:[]}
        for row in R[sd]['rows']:
            od=O[sd][str(row['k'])]
            for j in range(4):
                x=arr(sd,row,f'G[{cond},{cond}]@f{j}_{kind}');val=x[j] if own else np.delete(x,j).mean();v[od[j]].append(val)
        for g in (0,1):out[g].append(np.mean(v[g]) if v[g] else np.nan)
    return {g:np.array(out[g]) for g in out}
for kind,kn in (('ans','later-entity tokens'),('name','earlier-entity tokens')):
    for own,on in ((True,'own fact'),(False,'mean of the other 15 facts')):
        for g in (0,1):
            c0=fact_level(kind,'C0',own)[g];sv=fact_level(kind,'SAMEVIS',own)[g];print(f'  {kn:22s} {on:26s} {GN[g]:11s}: C0 {tci(c0)} | SV {tci(sv)} | SV-C0 {tci(sv-c0)}')
        d=(fact_level(kind,'SAMEVIS',own)[1]-fact_level(kind,'C0',own)[1])-(fact_level(kind,'SAMEVIS',own)[0]-fact_level(kind,'C0',own)[0]);print(f'    (SV-C0)[name first] - (SV-C0)[place first]: {tci(d)}')
print('== not stored in 1344p (marked as missing): per-token ||e_SV||, ||e_C0||, NLL; per-candidate scores (binding / familiarity); the token counts per class (posn) exist per chunk but not per sentence order')
