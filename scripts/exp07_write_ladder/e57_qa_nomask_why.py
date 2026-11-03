"""E57 why does masking not show in the QA group episodes (new file, CPU only; request of the thread 'QA에서 가림이 안 되는 이유 규명', Hans approved): analysis of the per-token rows of job 674 (e57_public_qa_stage13; n = 9 states W0, NATQ, ALLQ128, GAINQ512 p.10; items455, original 16 candidates, conditions A and B).
Quantities per item (first-token score = log-prob of the first token of the gold string; full = summed log-prob): dA = A(state) - A(W0); dB likewise; d(A-B) = dA - dB; masking add-on = X(ALLQ128) - X(NATQ) (and GAINQ512 p.10 - NATQ) for X = dA and d(A-B).
 (1) decomposition of the first-token dA, the first-token d(A-B) and the add-on by answer type (LOW / CAP / NUM / DATE; the type of the items file = E57_pqa5_item_rows.json) and by number of words of the gold string (1, 2, 3, 4+), mean per item with the bootstrap 95% CI over items (np.random.default_rng(1), 2000 draws);
 (2) items455 stratified into terciles of the W0 first-token log-prob of the gold under its own question (low / mid / high): the same quantities per tercile;
 (3) Spearman rank correlation over the items between the first-token dA (of NATQ, of ALLQ128) and the masking add-on (first token), with the bootstrap CI; the same for the full-sum score as a secondary line.
  python3 e57_qa_nomask_why.py --dir experiments/exp07_write_ladder/e57_public_qa_stage13 --out OUT.json | --selftest"""
import sys,json,pathlib
import numpy as np
def boot(x,seed=1,B=2000):
    x=np.asarray(x,np.float64);x=x[~np.isnan(x)]
    if len(x)==0:return dict(mean=None,ci95=None,n=0)
    ix=np.random.default_rng(seed).integers(0,len(x),(B,len(x)));m=x[ix].mean(1);return dict(mean=float(x.mean()),ci95=[float(np.percentile(m,2.5)),float(np.percentile(m,97.5))],n=int(len(x)))
def rankdata(x):
    x=np.asarray(x,np.float64);o=np.argsort(x,kind='mergesort');r=np.empty(len(x));r[o]=np.arange(len(x));s=x[o];i=0
    while i<len(x):
        j=i
        while j+1<len(x) and s[j+1]==s[i]:j+=1
        r[o[i:j+1]]=(i+j)/2.0;i=j+1
    return r
def spearman(a,b):
    ra,rb=rankdata(a),rankdata(b);ra-=ra.mean();rb-=rb.mean();return float((ra*rb).sum()/np.sqrt((ra**2).sum()*(rb**2).sum()))
def spearman_ci(a,b,seed=1,B=2000):
    a=np.asarray(a);b=np.asarray(b);n=len(a);ix=np.random.default_rng(seed).integers(0,n,(B,n));r=np.asarray([spearman(a[i],b[i]) for i in ix]);return dict(rho=spearman(a,b),ci95=[float(np.percentile(r,2.5)),float(np.percentile(r,97.5))],n=int(n))
def load(dirp):
    d=pathlib.Path(dirp);ld=lambda c,l:json.load(open(d/f'E57_pqa13_rows_{c}_{l}.json'));rm=json.load(open(d/'e57_pqa13_read_meta.json'));ids=rm['item_ids'];S={}
    for l in ('W0','NATQ_n9','ALLQ128_n9','GAINQ512P10_n9'):
        for c in 'AB':
            R=ld(c,l);assert [r['id'] for r in R]==ids;S[(c,l)]=dict(first=np.asarray([r['cand_tok_lp'][0][0] for r in R]),full=np.asarray([sum(r['cand_tok_lp'][0]) for r in R]))
    R=ld('A','W0');typ=np.asarray([r['type'] for r in R]);gold=[r['cand_surface'][0] for r in R];nw=np.asarray([len(g.split()) for g in gold]);return ids,S,typ,nw,gold
def deltas(S,score):
    out={}
    for st in ('NATQ_n9','ALLQ128_n9','GAINQ512P10_n9'):
        dA=S[('A',st)][score]-S[('A','W0')][score];dB=S[('B',st)][score]-S[('B','W0')][score];out[st]=dict(dA=dA,dB=dB,dAB=dA-dB)
    out['addon_ALLQ128']=dict(dA=out['ALLQ128_n9']['dA']-out['NATQ_n9']['dA'],dAB=out['ALLQ128_n9']['dAB']-out['NATQ_n9']['dAB']);out['addon_GAINQ512P10']=dict(dA=out['GAINQ512P10_n9']['dA']-out['NATQ_n9']['dA'],dAB=out['GAINQ512P10_n9']['dAB']-out['NATQ_n9']['dAB']);return out
def summarize(D,mask):
    return {'n':int(mask.sum()),'NATQ':{k:boot(D['NATQ_n9'][k][mask]) for k in ('dA','dB','dAB')},'ALLQ128':{k:boot(D['ALLQ128_n9'][k][mask]) for k in ('dA','dB','dAB')},'GAINQ512P10':{k:boot(D['GAINQ512P10_n9'][k][mask]) for k in ('dA','dB','dAB')},
        'addon_ALLQ128_minus_NATQ':{k:boot(D['addon_ALLQ128'][k][mask]) for k in ('dA','dAB')},'addon_GAINQ512P10_minus_NATQ':{k:boot(D['addon_GAINQ512P10'][k][mask]) for k in ('dA','dAB')}}
def run(dirp,out):
    ids,S,typ,nw,gold=load(dirp);res={'n_items':len(ids),'note':'first = log-prob of the first token of the gold string; full = summed log-prob; add-on = ALLQ128 (GAINQ512 p.10) minus NATQ, per item; n=9 states of job 674'}
    for score in ('first','full'):
        D=deltas(S,score);r={'all':summarize(D,np.ones(len(ids),bool)),'by_type':{t:summarize(D,typ==t) for t in sorted(set(typ.tolist()))},'by_words':{},'by_W0_first_token_tercile':{}}
        for lab,m in (('1',nw==1),('2',nw==2),('3',nw==3),('4+',nw>=4)):
            if m.any():r['by_words'][lab]=summarize(D,m)
        w0=S[('A','W0')]['first'];q=np.percentile(w0,[100/3,200/3]);ter=np.digitize(w0,q)
        for k,lab in enumerate(('low','mid','high')):
            m=ter==k;e=summarize(D,m);e['W0_first_token_logprob_range']=[float(w0[m].min()),float(w0[m].max())];r['by_W0_first_token_tercile'][lab]=e
        r['spearman']={'NATQ_dA_vs_addon_ALLQ128_dA':spearman_ci(D['NATQ_n9']['dA'],D['addon_ALLQ128']['dA']),'ALLQ128_dA_vs_addon_ALLQ128_dA':spearman_ci(D['ALLQ128_n9']['dA'],D['addon_ALLQ128']['dA']),'NATQ_dA_vs_addon_GAINQ512P10_dA':spearman_ci(D['NATQ_n9']['dA'],D['addon_GAINQ512P10']['dA']),
            'NATQ_dAB_vs_addon_ALLQ128_dAB':spearman_ci(D['NATQ_n9']['dAB'],D['addon_ALLQ128']['dAB'])}
        res[score]=r
    res['type_counts']={t:int((typ==t).sum()) for t in sorted(set(typ.tolist()))};res['word_counts']={str(k):int((nw==k).sum()) for k in sorted(set(nw.tolist()))};res['spearman_note']='rho is the rank correlation across items between the first-token dA of the arm named first and the add-on (first token ALLQ128 - NATQ dA); a positive rho of NATQ_dA vs add-on means items that rise more under NATQ also gain more from the mask'
    pathlib.Path(out).parent.mkdir(parents=True,exist_ok=True);pathlib.Path(out).write_text(json.dumps(res,indent=1,default=float));return res
def selftest():
    x=np.asarray([1.,2.,2.,5.]);assert rankdata(x).tolist()==[0.0,1.5,1.5,3.0] and abs(spearman([1,2,3,4],[1,2,3,4])-1)<1e-12 and abs(spearman([1,2,3,4],[4,3,2,1])+1)<1e-12
    print('e57_qa_nomask_why selftest ok')
def main():
    if '--selftest' in sys.argv:selftest();return
    g=lambda k:sys.argv[sys.argv.index(k)+1];run(g('--dir'),g('--out'))
if __name__=='__main__':main()
