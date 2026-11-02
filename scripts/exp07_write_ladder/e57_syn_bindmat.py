"""E57 synthetic-world binding from the stored read matrices (new file, CPU only; E57 122-Z (a)): for every stored arm the read matrix S[r, c] (row r = a question (the name frame of one fact, kind 'fact'), column c = one of its 16 candidate places, value = summed log-prob of the candidate; read[d]['out'] and, where stored, read[d]['in_g0'..]) and the W0 matrix of the same file are turned into
  A_f  = S[r_f, gold_idx_f]                          score of the gold place of fact f under its OWN name frame
  B_f  = mean over the other fact rows r != r_f whose candidate list contains the place of fact f (cand_fact[r, c] == f) of S[r, c]   the SAME place under the name frames of OTHER facts
  dA_f = A_f(arm) - A_f(W0), dB_f likewise, dAB_f = dA_f - dB_f.
Per arm and read (d, context) the mean over facts with the bootstrap 95% CI over facts (np.random.default_rng(1), 2000 draws); facts whose place is in no other row's candidate list have no B and are listed separately (not part of the B and A-B means; dA is reported both over all facts and over the facts with B). Masked arms: the paired differences to the reference arm (NATQ for the seed-64 arms, NAT for the E55 210s arms): d(dA) = dA_arm - dA_ref and d(dAB) = dAB_arm - dAB_ref per fact, their means with CI, and the ratio d(dAB) / d(dA) (ratio of the means, bootstrap CI over facts).
Inputs: experiments/exp07_write_ladder/e57_synq/E55_synq_<ARM>_s64.read.json (15 arms, W0 inside each file) and experiments/exp07_write_ladder/E55_d64_{NAT,SENTPQ,SENTPQADAMH}_s61.read.json (W0 inside).
  python3 e57_syn_bindmat.py --dir experiments/exp07_write_ladder --out O | --selftest"""
import sys,json,glob,pathlib
import numpy as np
def boot(x,seed=1,B=2000):
    x=np.asarray(x,np.float64)
    if len(x)==0:return dict(mean=None,ci95=None,n=0)
    ix=np.random.default_rng(seed).integers(0,len(x),(B,len(x)));m=x[ix].mean(1);return dict(mean=float(x.mean()),ci95=[float(np.percentile(m,2.5)),float(np.percentile(m,97.5))],n=int(len(x)))
def ratio_boot(num,den,seed=1,B=2000):
    num=np.asarray(num,np.float64);den=np.asarray(den,np.float64);n=len(num)
    if n==0:return dict(ratio=None,ci95=None,n=0)
    ix=np.random.default_rng(seed).integers(0,n,(B,n));r=num[ix].mean(1)/den[ix].mean(1);return dict(ratio=float(num.mean()/den.mean()),ci95=[float(np.nanpercentile(r,2.5)),float(np.nanpercentile(r,97.5))],n=int(n))
def ab_scores(S,meta):
    """-> dict fact -> (A, B or None, n_other_rows) for the fact rows of a matrix S [rows, 16]"""
    S=np.asarray(S,np.float64);kind=np.asarray(meta['kind']);fact=np.asarray(meta['fact']);cf=np.asarray(meta['cand_fact']);gi=np.asarray(meta['gi']);rows=[r for r in range(len(kind)) if kind[r]=='fact'];out={}
    for r in rows:
        f=int(fact[r]);A=float(S[r,gi[r]]);vals=[float(S[q,c]) for q in rows if q!=r for c in range(cf.shape[1]) if int(cf[q,c])==f]
        out[f]=(A,float(np.mean(vals)) if vals else None,len(vals))
    return out
def reads_of(d):
    """-> {(d, context): (matrix state, matrix W0)}"""
    out={}
    for dd,v in d['read'].items():
        for ctx,mat in v.items():
            w0=d['W0'].get(ctx if ctx=='out' else f'{ctx}@{dd}')
            if w0 is None:w0=d['W0'].get(ctx)
            if w0 is not None and len(mat)==len(d['meta']['kind']):out[(dd,ctx)]=(mat,w0)      # in-context reads of a subset of the rows (item_idx) are skipped
    return out
def delta(d):
    """-> {(d, ctx): {fact: (dA, dB or None)}}"""
    m=d['meta'];res={}
    for key,(mat,w0) in reads_of(d).items():
        a=ab_scores(mat,m);b=ab_scores(w0,m);res[key]={f:(a[f][0]-b[f][0],None if a[f][1] is None or b[f][1] is None else a[f][1]-b[f][1]) for f in a}
    return res
def summarize(dl):
    facts=sorted(dl);dA=np.asarray([dl[f][0] for f in facts]);hasB=[f for f in facts if dl[f][1] is not None];dB=np.asarray([dl[f][1] for f in hasB]);dAhB=np.asarray([dl[f][0] for f in hasB])
    return dict(n_facts=len(facts),n_with_B=len(hasB),facts_without_B=[int(f) for f in facts if dl[f][1] is None],dA_all=boot(dA),dA_with_B=boot(dAhB),dB=boot(dB),dA_minus_dB=boot(dAhB-dB))
def compare(dl,dr):
    fs=sorted(f for f in dl if f in dr and dl[f][1] is not None and dr[f][1] is not None)
    if not fs:return None
    ddA=np.asarray([dl[f][0]-dr[f][0] for f in fs]);ddAB=np.asarray([(dl[f][0]-dl[f][1])-(dr[f][0]-dr[f][1]) for f in fs])
    return dict(n=len(fs),d_dA=boot(ddA),d_dAB=boot(ddAB),ratio_dAB_over_dA=ratio_boot(ddAB,ddA))
def run(dirp,out):
    dirp=pathlib.Path(dirp);groups={'s64':(sorted(p for p in glob.glob(str(dirp/'e57_synq/E55_synq_*_s64.read.json')) if '.partial' not in p),'NATQ'),
        's61':([str(dirp/f'E55_d64_{a}_s61.read.json') for a in ('NAT','SENTPQ','SENTPQADAMH')],'NAT')}
    res={};
    for gname,(files,ref) in groups.items():
        D={};meta_info={}
        for f in files:
            if not pathlib.Path(f).exists():continue
            d=json.load(open(f));D[d['arm']]=delta(d);meta_info[d['arm']]=dict(file=f,complete=d.get('complete'),k=d['k'],n=d['n'],p=d['p'],cell=d.get('cell'))
        assert ref in D,(gname,'reference arm missing',list(D))
        g={'reference':ref,'arms':meta_info,'by_arm':{},'vs_reference':{}}
        for arm,dd in D.items():
            g['by_arm'][arm]={f'{k[0]}|{k[1]}':summarize(v) for k,v in dd.items()}
            if arm!=ref:g['vs_reference'][arm]={f'{k[0]}|{k[1]}':compare(v,D[ref][k]) for k,v in dd.items() if k in D[ref]}
        res[gname]=g
    pathlib.Path(out).mkdir(parents=True,exist_ok=True);(pathlib.Path(out)/'e57_syn_bindmat_keys.json').write_text(json.dumps(dict(definition=__doc__.split('Inputs:')[0],**res),indent=1,default=float));return res
def selftest():
    meta=dict(kind=['fact','fact','fact','ctrl'],fact=[0,1,2,3],cand_fact=[[0,1],[0,1],[2,0],[0,1]],gi=[0,1,0,0])
    S=[[-1.,-5.],[-2.,-3.],[-4.,-9.],[0.,0.]];r=ab_scores(S,meta)
    assert r[0][1]==np.mean([-2.0,-9.0]) and r[1][1]==-5.0 and r[2][1] is None and r[2][2]==0 and r[1][0]==-3.
    s=summarize({0:(1.,0.5),1:(2.,None),2:(3.,1.5)});assert s['n_with_B']==2 and s['facts_without_B']==[1]
    c=compare({0:(2.,1.),1:(4.,1.)},{0:(1.,1.),1:(1.,1.)});assert c['n']==2 and abs(c['ratio_dAB_over_dA']['ratio']-1.0)<1e-9
    print('e57_syn_bindmat selftest ok')
def main():
    if '--selftest' in sys.argv:selftest();return
    g=lambda k:sys.argv[sys.argv.index(k)+1];run(g('--dir'),g('--out'))
if __name__=='__main__':main()
