"""Position split: 1344k vs 1344m (CPU; new file; thread-server 11/24; Hans 'CPU check beforehand'; stored results only, no GPU): (1) 1344m: absolute contribution and share of the position classes own / other / rest to the one-step margin change per condition; the 1344m classes are own = the answer tokens of ALL 16 read facts, other = the answer tokens of the 16 extra (U2) facts, rest = everything else; (2) the same classes reproduced from the 1344k arrays (layer 8, w2, first answer token key; m_p = (u_f . delta_p)(x2_p . k_f), linear response of the margin of read fact f = -coef x sum of m_p over the class, averaged over the 16 facts) on the SAME chunks (16, 48, 80, 112) with the clip coefficient of the chunk; slope and correlation between predicted and measured; (3) the 1344k layer-8 projection split finer: own fact answer / own name / own sentence frame, other read facts' answer / name / frame, extra facts' answer / name / frame, filler. python3 e59_pos_check.py --k1344k D --kvstep D --seeds 231-236"""
import sys,json,types,numpy as np,collections
from pathlib import Path
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
for p in (HERE,HERE/'e57',ROOT/'scripts/exp04_what_ttt_retains',ROOT/'scripts/exp05_write_dynamics'):sys.path.insert(0,str(p))
from scipy import stats
a=sys.argv;KD=Path(a[a.index('--k1344k')+1]);MD=Path(a[a.index('--kvstep')+1]);lo,hi=[int(x) for x in a[a.index('--seeds')+1].split('-')];SEEDS=[s for s in range(lo,hi+1) if (MD/f'kvstep_s{s}.json').exists() and (KD/f'kv_s{s}.json').exists()];n=len(SEEDS);T=stats.t.ppf(.975,n-1)
CH_=(16,48,80,112);L7=7;NREAD=16
def ci(d):d=np.asarray(d,float);m=d.mean();h=T*d.std(ddof=1)/np.sqrt(len(d));return f'{m:8.3f} [{m-h:8.3f},{m+h:8.3f}] {int(max((d>0).sum(),(d<0).sum()))}/{len(d)}'
def du(D16):
    n_=D16.shape[1];return np.stack([D16[:,f]-(D16.sum(1)-D16[:,f])/(n_-1) for f in range(n_)],1)
# ---- (1) 1344m measured
M={};P={}
for s in SEEDS:
    r=json.load(open(MD/f'kvstep_s{s}.json'));rows={x['k']:x for x in r['rows']}
    for c in ('C0','SAMEVIS','NATVIS'):
        for cl in ('own','other','rest'):M.setdefault((c,cl),[]).append(np.mean([rows[k]['combos'][f'G[{c},{c}]@{cl}']['dmargin'] for k in CH_]))
        M.setdefault((c,'full'),[]).append(np.mean([rows[k]['combos'][f'G[{c},{c}]']['dmargin'] for k in CH_]));P[(s,c)]={cl:[rows[k]['combos'][f'G[{c},{c}]@{cl}']['dmargin'] for k in CH_] for cl in ('own','other','rest')}
print(f'(1) 1344m measured one-step dmargin by position class (seeds {SEEDS}, chunks {CH_}); own = answer tokens of ALL 16 read facts, other = answer tokens of the 16 extra facts, rest = everything else')
for c in ('C0','SAMEVIS','NATVIS'):
    o,t_,r_,f_=(np.array(M[(c,k)]) for k in ('own','other','rest','full'));sm=o+t_+r_
    print(f'  {c:8s} own {ci(o)} | other {ci(t_)} | rest {ci(r_)} | sum {ci(sm)} | full {ci(f_)} | shares of the sum: own {o.mean()/sm.mean():.2f} other {t_.mean()/sm.mean():.2f} rest {r_.mean()/sm.mean():.2f}')
for cl in ('own','other','rest'):print(f'  SAMEVIS - C0 {cl:5s} {ci(np.array(M[("SAMEVIS",cl)])-np.array(M[("C0",cl)]))}')
# ---- (2),(3) 1344k arrays reproduced
import e55_run as R55,e59_u12 as U
U.install()
def world(seed):
    a_=types.SimpleNamespace(cell=U.CELL,seed=seed,tokenizer_file=None,bg='pool',decoy='names',stub=False,arms='NAT');enc,w,tf=R55.build(a_,seed,read_apps={16,32,64,128},arms=['NAT']);return w
CLS=('own answer','own name','own frame','read-other answer','read-other name','read-other frame','extra answer','extra name','extra frame','filler')
PR={};PF={};MEAS=[];PRED=[]
for s in SEEDS:
    z=np.load(KD/f'kv_s{s}.npz');m=json.load(open(KD/f'kv_s{s}.json'));w=world(s);coef={(x['cond'],x['k']):x['coef'] for x in m['rows']}
    for c in ('C0','SAMEVIS','NATVIS'):
        acc={k:[] for k in ('own','other','rest')};fine={k:[] for k in CLS}
        for ki,k in enumerate(CH_):
            X2K=z[f'{c}|{k}|X2K'][:,L7,:];DU=du(z[f'{c}|{k}|D16']);ap={d['fact']:d for d in m['AP'][str(k)]};sents=w['chunks'][k]['sents'];Tn=X2K.shape[0];cf=coef[(c,k)]
            tokcls={}                                                    # position -> (fact, kind)
            for x in sents:
                f_=int(x['fact']);st,en=int(x['start']),int(x['end']);first=set(ap[f_]['first']) if f_ in ap else set();second=set(ap[f_]['second']) if f_ in ap else set()
                for p in range(max(st-1,0),min(en-1,Tn)):                  # NLL index = token position - 1
                    kind='answer' if p in second else ('name' if p in first else 'frame');tokcls[p]=(f_,kind)
            v=np.zeros(3);rowfine={kk:[] for kk in CLS}
            ans_read=np.zeros(Tn,bool);ans_extra=np.zeros(Tn,bool)
            for p,(f_,kind) in tokcls.items():
                if kind=='answer':(ans_read if f_<NREAD else ans_extra)[p]=True
            for f in range(NREAD):
                mp=DU[:,f]*X2K[:,f]                                         # m_p of read fact f at every position
                own=ans_read.copy();oth=ans_extra.copy();rest=~(own|oth)
                acc['own'].append(-cf*mp[own].sum());acc['other'].append(-cf*mp[oth].sum());acc['rest'].append(-cf*mp[rest].sum())
                cl={kk:0.0 for kk in CLS}
                for p in range(Tn):
                    if p in tokcls:
                        f_,kind=tokcls[p];grp='own' if f_==f else ('read-other' if f_<NREAD else 'extra');key=f'{grp} {kind}' if grp!='read-other' else f'read-other {kind}'
                        cl[key]+=mp[p]
                    else:cl['filler']+=mp[p]
                for kk in CLS:fine[kk].append(-cf*cl[kk])
        PR[(s,c)]={k:float(np.mean(v)) for k,v in acc.items()};PF[(s,c)]={k:float(np.mean(v)) for k,v in fine.items()}
        for cl_ in ('own','other','rest'):MEAS.append(np.mean(P[(s,c)][cl_]));PRED.append(PR[(s,c)][cl_])
print('(2) 1344k layer-8 w2 first-token linear prediction (-coef x sum of m_p over the class, mean over the 16 facts and the 4 chunks), per class')
for c in ('C0','SAMEVIS','NATVIS'):
    print(f'  {c:8s} '+' | '.join(f'{cl} pred {ci([PR[(s,c)][cl] for s in SEEDS])} vs measured {ci(M[(c,cl)])}' for cl in ('own','other','rest')))
MEAS=np.array(MEAS);PRED=np.array(PRED);print(f'  pooled over seeds x conditions x classes: slope measured on predicted {np.polyfit(PRED,MEAS,1)[0]:.2f}, correlation {np.corrcoef(PRED,MEAS)[0,1]:.2f}; own-class only: slope {np.polyfit([PR[k]["own"] for k in PR],[np.mean(P[k]["own"]) for k in PR],1)[0]:.2f}')
print('(3) finer split of the 1344k layer-8 prediction (units of margin; fact-specific own / other classes)')
for c in ('C0','SAMEVIS','NATVIS'):
    print(f'  {c:8s} '+' | '.join(f'{cl} {np.mean([PF[(s,c)][cl] for s in SEEDS]):+.4f}' for cl in CLS)+f' | total {np.mean([sum(PF[(s,c)].values()) for s in SEEDS]):+.4f}')
for cl in CLS:print(f'  SAMEVIS - C0 {cl:18s} {ci(np.array([PF[(s,"SAMEVIS")][cl]-PF[(s,"C0")][cl] for s in SEEDS]))}')
