"""Quick comparison for job 1361 (small run) against job 1358b, same seeds (CPU; new file; implementation thread 11/27; descriptive). For every state x t x offset and natural step (C0, SAMEVIS): change over W_t of matched binding and familiarity (linear in the scores, so the change = measure of (after - before)), mean over the 16 read facts, for the reads
  R1 (1358b, neutral 8192 tail) | R2 (1358b, the 1-chunk context, offsets 0/2/4/6) | A (1361, 8 written chunks) | B2 (1361, 8 chunks with the fact slots refilled).
Prints per seed and the mean over the given seeds: the step changes, the MAIN contrast C0 step - SAMEVIS step per state, and the correlation over (state, t, offset, step) cells of A with R2 and of B2 with R1; also the levels at W_t (binding, familiarity) for A vs R2 and B2 vs R1.
  python e59_xstep_r8_compare.py --r8 XSTEP_R8_DIR --xb XSTEP_DIR [--seeds 231,232]"""
import sys,json,glob,random,numpy as np
a=sys.argv;g=lambda k,d=None:a[a.index(k)+1] if k in a else d
R8=g('--r8');XB=g('--xb');SEEDS=[int(x) for x in g('--seeds','').split(',') if x] or sorted(int(f.split('_s')[-1].split('.')[0]) for f in glob.glob(R8+'/xstep_r8_s*.json'))
def _cands(seed):
    out=[]
    for f in range(16):
        r=random.Random(f"kvpos2|{seed}|{f}");oth=[h for h in range(16) if h!=f];r.shuffle(oth);out.append([f]+oth[:11])
    return out
def bf(dS,seed):
    """mean over facts of (matched binding, familiarity) of a score difference matrix"""
    dS=np.asarray(dS,float);cf=_cands(seed);mb=[]
    for f in range(16):
        oth=[dS[q,cf[q].index(f)] for q in range(16) if q!=f and f in cf[q][1:12]];mb.append(dS[f,0]-np.mean(oth))
    return np.array([np.mean(mb),float(np.mean(dS[:,1:12].mean(1)-dS[:,12:16].mean(1)))])
rows=[];lev=[]
for s in SEEDS:
    r8=json.load(open(f'{R8}/xstep_r8_s{s}.json'));xb=json.load(open(f'{XB}/xstep_s{s}.json'))
    for sn,byt in r8['states'].items():
        for t,rec in byt.items():
            xrec=xb['states'][sn][t];xc={e['k']:e for e in xrec['chunks']}
            for e in rec['chunks']:
                k=e['k'];xe=xc[k];W=e['reads_Wt']
                lev.append(dict(seed=s,state=sn,t=int(t),k=k,A=bf(np.asarray(W['A']),s),R2=bf(np.asarray(xe['R2_Wt']['R2']),s) if 'R2_Wt' in xe else None,B2=bf(np.asarray(W['B2']),s),R1=bf(np.asarray(xrec['R1_Wt']),s)))
                for mk,d in e['steps'].items():
                    xs=xe['steps'][mk];v=dict(seed=s,state=sn,t=int(t),k=k,step=mk)
                    v['R1']=bf(np.asarray(xs['R1'])-np.asarray(xrec['R1_Wt']),s);v['R2']=bf(np.asarray(xs['R2']['R2'])-np.asarray(xe['R2_Wt']['R2']),s) if 'R2' in xs else None
                    v['A']=bf(np.asarray(d['reads']['A'])-np.asarray(W['A']),s);v['B2']=bf(np.asarray(d['reads']['B2'])-np.asarray(W['B2']),s);rows.append(v)
print(f'1361 small run vs 1358b: seeds {SEEDS}; cells {len(rows)}');RD=('R1','R2','A','B2')
for mi,mn in enumerate(('matched binding','familiarity')):
    print(f'== one-step change of {mn}, mean over t x offsets inside a seed (per seed values)')
    for sn in ('W0','C0','SAMEVIS'):
        for mk in ('C0','SAMEVIS'):
            out=[]
            for rd_ in RD:
                pv=[np.mean([x[rd_][mi] for x in rows if x['seed']==s and x['state']==sn and x['step']==mk and x[rd_] is not None]) for s in SEEDS];out.append(f'{rd_} '+' '.join(f'{p:7.3f}' for p in pv))
            print(f'  state {sn:8s} step {mk:8s}: '+' | '.join(out))
        out=[]
        for rd_ in RD:
            pv=[]
            for s in SEEDS:
                c=[x for x in rows if x['seed']==s and x['state']==sn and x[rd_] is not None];d={(x['t'],x['k'],x['step']):x[rd_][mi] for x in c};pv.append(np.mean([d[(t,k,'C0')]-d[(t,k,'SAMEVIS')] for (t,k,st) in d if st=='C0' and (t,k,'SAMEVIS') in d]))
            out.append(f'{rd_} '+' '.join(f'{p:7.3f}' for p in pv))
        print(f'  state {sn:8s} MAIN C0 - SAMEVIS step: '+' | '.join(out))
    for x_,y_ in (('A','R2'),('B2','R1'),('A','B2')):
        c=[(r[x_][mi],r[y_][mi]) for r in rows if r[x_] is not None and r[y_] is not None];c=np.asarray(c)
        print(f'  correlation over cells {x_} vs {y_}: r = {np.corrcoef(c[:,0],c[:,1])[0,1]:.3f} (n {len(c)}), mean {x_} {c[:,0].mean():.3f} vs {y_} {c[:,1].mean():.3f}')
print('== levels at W_t (minus nothing for A / B2 raw scores; R1 is minus W0): mean over cells, binding | familiarity')
for x_,y_ in (('A','R2'),('B2','R1')):
    c=[(l[x_],l[y_]) for l in lev if l[x_] is not None and l[y_] is not None]
    print(f'  {x_}: {np.mean([p[0][0] for p in c]):.3f} | {np.mean([p[0][1] for p in c]):.3f}    {y_}: {np.mean([p[1][0] for p in c]):.3f} | {np.mean([p[1][1] for p in c]):.3f}')
