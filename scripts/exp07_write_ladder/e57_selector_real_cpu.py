"""E57 synthetic-trained selector model a (coefficients and standardization fixed, e57_synq/syn_selector_weights.npz; NO retraining) applied to the QA bundles and news stream A (new file; CPU only; GPM thread, Hans 11/07 '합성 학습한거를 뉴스랑 qa에 썼을 때 성능 한번만'): from the W0 last-prefix-layer hidden states (job 1200) the model a score of every eligible row of the chunks k >= K0 (8); against the band-128 W0 window gain g128 = nll_band128 - nll_win (QA: e57_dwg_qa_nll.npz, news: e57_band_W0.npz): per-chunk Spearman(score, g128), AUC of the g128>2 labels, and at the same row count as TOP5 (ceil 5% of the eligible rows, W0 g ranking) and as ABS2 (number of g128>2 rows of the chunk) the recall and precision of the g128>2 rows for model a, TOP5/ABS2 (g ranking itself) and a random baseline (the rate). Chunk-level bootstrap (2000 draws), 95% CI. Sources: QA tune (bundles 8,10,12), QA hold-out (5,9), each bundle, news A.
  python3 e57_selector_real_cpu.py --out F.json | --selftest"""
import sys,json,argparse,pathlib
import numpy as np
from scipy.stats import rankdata
HERE=pathlib.Path(__file__).resolve().parent;ROOT=HERE.parents[1];sys.path.insert(0,str(HERE));sys.path.insert(0,str(ROOT/'scripts/exp04_what_ttt_retains'))
EXP=ROOT/'experiments/exp07_write_ladder';CH=1024;K0=8;NB=2000
import e57_closure_masks as CM
def auc(y,s):
    y=np.asarray(y,bool);n1=y.sum();n0=(~y).sum()
    if n1==0 or n0==0:return float('nan')
    r=rankdata(s);return float((r[y].sum()-n1*(n1+1)/2)/(n1*n0))
def spearman(a,b):
    if len(a)<3 or np.std(a)==0 or np.std(b)==0:return float('nan')
    return float(np.corrcoef(rankdata(a),rankdata(b))[0,1])
def top_mask(s,n):
    m=np.zeros(len(s),bool)
    if n>0:m[np.argsort(-s,kind='stable')[:n]]=True
    return m
def bf16(u):return (u.astype(np.uint32)<<16).view(np.float32)
def ci(x):
    x=np.asarray(x,float);return [float(np.nanmean(x)),float(np.nanpercentile(x,2.5)),float(np.nanpercentile(x,97.5))]
def analyze(chunks,rng):
    """chunks: list of (score, g) per chunk (eligible rows)"""
    P=[];allr=[]
    for sc,g in chunks:
        n=len(g);lab=g>2;n5=int(np.ceil(0.05*n));na=int(lab.sum());r={'n':n,'n5':n5,'nabs':na,'rate':lab.mean(),'spear':spearman(sc,g),'auc':auc(lab,sc)}
        for nm,cnt in (('5',n5),('abs',na)):
            for who,s in (('a',sc),('g',g)):
                if cnt>0 and na>0:
                    m=top_mask(s,cnt);h=(m&lab).sum();r[f'{who}_{nm}_rec']=h/na;r[f'{who}_{nm}_prec']=h/cnt
                else:r[f'{who}_{nm}_rec']=r[f'{who}_{nm}_prec']=np.nan
            r[f'rand_{nm}_rec']=(cnt/n) if na>0 else np.nan;r[f'rand_{nm}_prec']=r['rate']
        P.append(r)
    keys=[k for k in P[0] if k!='n'];A={k:np.array([p[k] for p in P],float) for k in keys};idx=rng.integers(0,len(P),(NB,len(P)));res={'n_chunks':len(P),'mean_eligible_rows':float(np.mean([p['n'] for p in P])),'mean_n5':float(A['n5'].mean()),'mean_nabs':float(A['nabs'].mean()),'rate_g_gt2':float(A['rate'].mean())}
    for k in keys:
        if k in ('n5','nabs','rate'):continue
        res[k]=ci(np.nanmean(A[k][idx],1))
    for nm in ('5','abs'):
        res[f'a_{nm}_rec_ratio_to_g']=ci(np.nanmean(A[f'a_{nm}_rec'][idx],1)/np.maximum(np.nanmean(A[f'g_{nm}_rec'][idx],1),1e-9))
    return res
def load_w():
    W=np.load(EXP/'e57_synq/syn_selector_weights.npz');return W['mu'],W['sd'],W['wa']
def qa_chunks(gid,mu,sd,wa):
    Z=np.load(EXP/'e57_public_qa_n65/e57_dwg_qa_nll.npz');bi=list(Z['bands']).index(128);win=Z[f'g{gid}_win'];band=Z[f'g{gid}_band'][bi];lm=Z[f'g{gid}_lm'];H=np.load(EXP/f'e57_hidden_dump_real/hid_qa_g{gid}.npz')['h'];out=[]
    for k in range(K0,win.shape[0]):
        el=CM.rows_eligible(lm[k],np.zeros(CH,np.int32))&lm[k];ix=np.nonzero(el)[0]
        if len(ix)<50:continue
        X=(bf16(H[k*CH+ix])-mu)/sd;out.append(((X@wa[:-1].astype(np.float32)).astype(np.float64)+wa[-1],(band[k]-win[k]).astype(np.float64)[ix]))
    return out
def news_chunks(mu,sd,wa):
    import e55_uv as UVm,e57_t1t2v2_run as V,e57_news_m3 as NM
    ids=V.stream_ids(str(EXP/'e57_t1t2v2_in/e57_stream_sentences.jsonl'),'A');Z=np.load(EXP/'e57_band/e57_band_W0.npz');b=list(Z['bands']).index(128);ends=UVm.sentence_ends(UVm.tokenizer(str(ROOT/'data/tokenizer/llama3/tokenizer.json')),ids);D=np.load(EXP/'e57_hidden_dump_real/hid_news_A.npz');assert (D['ids']==np.asarray(ids)[:len(D['ids'])]).all();H=D['h'];out=[]
    for k in range(NM.K0,len(ids)//CH):
        q=UVm.chunk_sents(ends,k,CH);sid=np.full(CH,-1)
        for i,(s_,e_) in enumerate(q):sid[s_:e_]=i
        lm=Z['lm'][k];el=CM.rows_eligible(lm,sid)&lm;ix=np.nonzero(el)[0]
        X=(bf16(H[k*CH+ix])-mu)/sd;out.append(((X@wa[:-1].astype(np.float32)).astype(np.float64)+wa[-1],(Z['nll_band'][b][k]-Z['nll_win'][k]).astype(np.float64)[ix]))
    return out
def selftest():
    r=np.random.default_rng(0);g=r.standard_normal(1000)*2;sc=g+r.standard_normal(1000);a=analyze([(sc,g)]*5,r);assert a['a_5_rec'][0]>0.0 and a['g_abs_rec'][0]==1.0 and a['spear'][0]>0.5;print('e57_selector_real_cpu selftest ok')
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out');ap.add_argument('--selftest',action='store_true');a=ap.parse_args()
    if a.selftest:selftest();return
    rng=np.random.default_rng(0);mu,sd,wa=load_w();res={'definitions':__doc__.split('\n')[0],'model':'syn_selector_weights.npz wa (trained on s62/s71/s72 g128>2, lambda 0.01), no retraining'};C={}
    for gid in (5,8,9,10,12):C[gid]=qa_chunks(gid,mu,sd,wa);res[f'qa_bundle_{gid}']=analyze(C[gid],rng);print('qa',gid,flush=True)
    res['qa_tune_8_10_12']=analyze(sum((C[g] for g in (8,10,12)),[]),rng);res['qa_hold_5_9']=analyze(C[5]+C[9],rng);N=news_chunks(mu,sd,wa);res['news_A']=analyze(N,rng)
    pathlib.Path(a.out).write_text(json.dumps(res,indent=1));print('wrote',a.out)
if __name__=='__main__':main()
