"""E57 64-fact registered analysis (job 1273/1276; CPU; new file, written BEFORE any result of the new seeds was opened). Seeds 62 63 64 65 66 of cell v3_64_4 (pilot s61 = job 1271 is exploratory and NOT included). Registered by the core-claims thread (Hans 11/10):
 primary = out-of-window read accuracy at exposure 128 (argmax over the 16 candidates, 64 fact items per seed; 'out' reads), NATMASK - NATS, per-seed paired difference; SUPERIOR if the 95% t interval (df = n_seeds - 1 = 4) of the mean difference has lower bound > 0 AND the mean >= +3 pp.
 secondary family (Holm, one-sided where noted by the registration; here two-sided t p-values are reported and Holm-adjusted): NATMASK - NATMASKC, DIRmaskNORMfull - NATS, slope of accuracy on log2(exposure) over exposures 32/64/128 per seed (NATMASK - NATS difference of slopes, one-sided t alpha .025 as registered).
 preservation gates (separate from the family): general-text PPL (canary tail NLL and the DCLM gate) within +0.01 nat of NATS, in-window read accuracy (read 'in_g0' at exposure 128) not more than 2 pp below NATS. The word 'equivalent' is not used. No seeds are added on failure.
  python3 e57_natmask_d64_analyze.py --dir O --seeds 62,63,64,65,66 [--tag synngd64] [--json J]"""
import sys,json,argparse,pathlib
import numpy as np
from scipy import stats
ARMS=['NATS','NATMASK','NATMASKC','DIRmaskNORMfull','DIRfullNORMmask'];EXPO=['16','32','64','128']
def accs(path):
    d=json.load(open(path));g=np.array(d['meta']['gold_idx']);fact=np.array([k=='fact' for k in d['meta']['kind']])
    out={a:float((np.array(d['read'][a]['out']).argmax(1)==g)[fact].mean()) for a in EXPO}
    ins=d['read']['128'].get('in_g0');inw=None
    if ins is not None:
        grp=np.array(d['meta'].get('group',[0]*len(g)));ix=np.nonzero((grp==0)|(grp==-1))[0] if len(ins)!=len(g) else np.arange(len(g))       # in-window read: items of group 0 (+ controls, group -1) when the world has several groups
        ix=ix[:len(ins)] if len(ix)>=len(ins) else ix
        if len(ix)==len(ins):inw=float((np.array(ins).argmax(1)==g[ix])[fact[ix]].mean())
    return out,inw,len(g),int(fact.sum())
def tci(d,level=0.95):
    d=np.asarray(d,float);n=len(d);m=d.mean();se=d.std(ddof=1)/np.sqrt(n);t=stats.t.ppf(0.5+level/2,n-1);return float(m),float(m-t*se),float(m+t*se),float(2*stats.t.sf(abs(m/se),n-1)) if se>0 else float('nan')
def slope(acc):
    x=np.log2([32,64,128]);y=np.array([acc['32'],acc['64'],acc['128']]);return float(np.polyfit(x,y,1)[0])
def holm(p):
    k=sorted(p,key=lambda z:p[z]);m=len(k);adj={};run=0.0
    for i,z in enumerate(k):run=max(run,min(1.0,(m-i)*p[z]));adj[z]=run
    return adj
def analyze(D,seeds,tag='synngd64'):
    A={a:[] for a in ARMS};I={a:[] for a in ARMS};S={a:[] for a in ARMS};C={a:[] for a in ARMS}
    for a in ARMS:
        for s in seeds:
            f=D/f'E55_{tag}_{a}_s{s}.read.json';sv=D/f'E55_{tag}_{a}_s{s}.saved.json';assert f.exists() and sv.exists(),('missing',f)
            r=json.load(open(sv));assert not any(not np.isfinite(e[k]) for e in r['log'] for k in ('ce','gnorm','update_norm')),('non-finite',a,s)
            o,inw,_,nf=accs(f);A[a].append(o['128']);I[a].append(inw);S[a].append(slope(o))
            w0=r['canary_w0'];last=max(int(k) for k in r['reads']);cn=r['reads'][str(last)]['canary'];C[a].append(cn['_tail_nll']-w0['_tail_nll'])
    A={a:np.array(v) for a,v in A.items()};S={a:np.array(v) for a,v in S.items()}
    res={'seeds':seeds,'n_fact_per_seed':nf,'acc128':{a:[float(x) for x in A[a]] for a in ARMS}}
    m,lo,hi,p=tci(A['NATMASK']-A['NATS']);res['primary']=dict(contrast='NATMASK-NATS acc128',mean=m,ci95_t=[lo,hi],p=p,superior=bool(lo>0 and m>=0.03))
    fam={'NATMASK-NATMASKC':A['NATMASK']-A['NATMASKC'],'DIRmaskNORMfull-NATS':A['DIRmaskNORMfull']-A['NATS'],'slope(NATMASK)-slope(NATS)':S['NATMASK']-S['NATS']};pv={};res['family']={}
    for k,d in fam.items():
        m,lo,hi,p=tci(d);res['family'][k]=dict(mean=m,ci95_t=[lo,hi],p_two_sided=p);pv[k]=p if not k.startswith('slope') else float(stats.t.sf(m/(np.asarray(d).std(ddof=1)/np.sqrt(len(d))),len(d)-1))     # slope: one-sided
    ad=holm(pv)
    for k in fam:res['family'][k]['p_holm']=ad[k]
    ndiff=[np.array(I['NATMASK'],float)-np.array(I['NATS'],float)] if None not in I['NATMASK'] and None not in I['NATS'] else None
    res['gates']=dict(canary_tail_rise_NATMASK_minus_NATS=float(np.mean(C['NATMASK'])-np.mean(C['NATS'])),canary_pass=bool(np.mean(C['NATMASK'])-np.mean(C['NATS'])<=0.01),inwindow_acc128_diff_NATMASK_minus_NATS=(None if ndiff is None else float(ndiff[0].mean())),inwindow_pass=(None if ndiff is None else bool(ndiff[0].mean()>=-0.02)),note='DCLM general-text gate = job 1274 (e57_dclm_ttt_gate.json)')
    return res
def selftest():
    rng=np.random.default_rng(0);import tempfile
    assert abs(tci([0.1,0.1,0.1,0.1,0.1+1e-9])[0]-0.1)<1e-6
    assert holm({'a':0.01,'b':0.04})=={'a':0.02,'b':0.04} and abs(slope({'32':0.1,'64':0.2,'128':0.3})-0.1)<1e-9
    with tempfile.TemporaryDirectory() as td:
        td=pathlib.Path(td);seeds=[1,2,3,4,5]
        for a in ARMS:
            for s in seeds:
                g=[int(x) for x in rng.integers(0,16,68)];kind=['fact']*64+['ctrl']*4;bump={'NATMASK':8.0,'DIRmaskNORMfull':8.0}.get(a,0.0)
                def L(e):
                    x=rng.normal(size=(68,16));x[np.arange(68),g]+=e+bump*(e==2);return x.tolist()
                d=dict(meta=dict(gold_idx=g,kind=kind),read={'16':{'out':L(0)},'32':{'out':L(1)},'64':{'out':L(2)},'128':{'out':L(3),'in_g0':L(0)}})
                json.dump(d,open(td/f'E55_t_{a}_s{s}.read.json','w'));json.dump(dict(log=[dict(ce=1.0,gnorm=1.0,update_norm=1.0)],canary_w0={'_tail_nll':2.0},reads={'511':dict(canary={'_tail_nll':2.0})}),open(td/f'E55_t_{a}_s{s}.saved.json','w'))
        r=analyze(td,seeds,'t');assert set(r['family'])=={'NATMASK-NATMASKC','DIRmaskNORMfull-NATS','slope(NATMASK)-slope(NATS)'} and r['gates']['canary_pass'] and 'primary' in r
    print('e57_natmask_d64_analyze selftest ok')
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--dir');ap.add_argument('--seeds');ap.add_argument('--tag',default='synngd64');ap.add_argument('--json');ap.add_argument('--selftest',action='store_true');a=ap.parse_args()
    if a.selftest:selftest()
    else:
        r=analyze(pathlib.Path(a.dir),[int(x) for x in a.seeds.split(',')],a.tag);print(json.dumps(r,indent=1,default=float))
        if a.json:json.dump(r,open(a.json,'w'),indent=1,default=float)
