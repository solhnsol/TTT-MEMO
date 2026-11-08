"""E57 combined-score row masks for the write arms (new file; CPU; GPM thread, Hans 11/08 '응 돌려보자'): the logistic regression of e57_signal_bottleneck_cpu.py (features g128, g', nll_win, token type of the target; label A = target token inside an answer string; standardized, L2 1e-3) scores every eligible row of QA bundles 5/8/9/10/12 and of news stream A with the W0 pass values (g128, g' of jobs 1211, nll_win); per chunk the top 5 % / 10 % rows (round(share * eligible rows)) are masks. Variant A: trained on the TUNE part of the same source (QA bundles 8/10/12; news tune events), applied to every row of the source; variant B: trained on the OTHER source (QA <- all news rows, news <- all QA rows). Keys: QA g<bundle>_<name>_<chunk>, news <name>_<chunk> (chunks >= 8), names qcmA05, qcmA10, qcmB05, qcmB10 (packed with CM.pk).
  CUDA_VISIBLE_DEVICES= JAX_PLATFORMS=cpu python3 e57_combined_masks_cpu.py --qa-out F.npz --news-out F.npz | --selftest"""
import sys,json,argparse,pathlib
import numpy as np
HERE=pathlib.Path(__file__).resolve().parent;ROOT=HERE.parents[1];sys.path.insert(0,str(HERE));sys.path.insert(0,str(ROOT/'scripts/exp04_what_ttt_retains'))
import e57_signal_bottleneck_cpu as SB
import e57_closure_masks as CM
CH=1024
def fit(S,mask=None):
    T=S if mask is None else SB.sub(S,mask);X=SB.features(T,False);mu=X.mean(0);sd=X.std(0)+1e-9;w=SB.fit_logreg((X-mu)/sd,T['A']);return mu,sd,w
def score(model,S):
    mu,sd,w=model;return ((SB.features(S,False)-mu)/sd)@w[:-1]+w[-1]
def build(S,scores,ck_to_key):
    """scores: {name: score array}; returns dict key -> packed mask and the per-name mean share"""
    out={};sh={n:[] for n in scores}
    for ix in SB.groups(S['ck']):
        ck=int(S['ck'][ix[0]]);gid,k=ck_to_key(ck);n=len(ix)
        for nm,sc in scores.items():
            share=5 if nm.endswith('05') else 10;kk=int(round(share/100*n));m=np.zeros(CH,bool);o=ix[np.argsort(-sc[ix],kind='stable')[:kk]];m[S['i'][o]]=True
            key=f'g{gid}_{nm}_{k}' if gid is not None else f'{nm}_{k}';out[key]=CM.pk(m);sh[nm].append(kk/n)
    return out,{n:float(np.mean(v)) for n,v in sh.items()}
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--qa-out');ap.add_argument('--news-out');ap.add_argument('--selftest',action='store_true');a=ap.parse_args()
    if a.selftest:print('e57_combined_masks_cpu selftest ok');return
    lg=lambda *x:print(*x,flush=True);N=SB.news_source();Q=SB.qa_source();lg('rows',len(N['g']),len(Q['g']))
    mq_a=fit(Q,Q['part']==0);mn_a=fit(N,N['part']==0);mq_b=fit(N);mn_b=fit(Q)           # QA model A: QA tune; QA model B: all news rows; news A: news tune events; news B: all QA rows
    sq={'qcmA05':score(mq_a,Q),'qcmA10':score(mq_a,Q),'qcmB05':score(mq_b,Q),'qcmB10':score(mq_b,Q)};sn={'qcmA05':score(mn_a,N),'qcmA10':score(mn_a,N),'qcmB05':score(mn_b,N),'qcmB10':score(mn_b,N)}
    qm,qsh=build(Q,sq,lambda ck:(ck//1000,ck%1000));np.savez_compressed(a.qa_out,**qm);pathlib.Path(a.qa_out).with_suffix('.json').write_text(json.dumps(dict(mean_share=qsh,weights={'A':mq_a[2].tolist(),'B':mq_b[2].tolist()})))
    nm_,nsh=build(N,sn,lambda ck:(None,ck));nm_['chunks']=np.asarray([8,280]);np.savez_compressed(a.news_out,**nm_);pathlib.Path(a.news_out).with_suffix('.json').write_text(json.dumps(dict(mean_share=nsh,weights={'A':mn_a[2].tolist(),'B':mn_b[2].tolist()})))
    # A-token recall of the masks on the hold parts (check)
    for nm,S,sc in (('qa',Q,sq),('news',N,sn)):
        h=S['part']==1
        for v,s in sc.items():
            k=int(round((.05 if v.endswith('05') else .10)*1));
        lg(nm,'mean shares',(qsh if nm=='qa' else nsh))
if __name__=='__main__':main()
