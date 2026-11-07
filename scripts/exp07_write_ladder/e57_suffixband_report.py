"""E57 suffix-only band g' = nll_suffixband128 - nll_win versus g128 = nll_band128 - nll_win (both bands on 128; g' keeps the natural window in the 24 prefix layers) on the synthetic worlds (new file; CPU; GPM thread, Hans 11/07 '고고'): per chunk over the eligible rows (as CM.rows_eligible): (1) Spearman(g', g128); (2) with the same row count as TOP5 (ceil 5% of the eligible rows) the top rows by g' against the top rows by g128 (recall = precision at equal size) and against the g128>2 rows; (3) the number of rows with g'>2 against the ABS2 count (g128>2), their set overlap (recall / precision of the g'>2 set for the ABS2 set); (4) recall / precision of the TRUE deciding rows (place first token) for the top 5% by g' and for the g'>2 set, with TOP5(g128) / ABS2 as references; (5) wall-clock cost of the suffix-band forward relative to the natural forward and to the full band pass. Chunk-level bootstrap (2000 draws), 95% CI, per world, per density group (train + eval worlds of the same cell) and pooled.
  python3 e57_suffixband_report.py --out F.json | --selftest"""
import sys,json,argparse,pathlib
import numpy as np
from scipy.stats import rankdata
HERE=pathlib.Path(__file__).resolve().parent;ROOT=HERE.parents[1];sys.path.insert(0,str(HERE));sys.path.insert(0,str(ROOT/'scripts/exp04_what_ttt_retains'))
EXP=ROOT/'experiments/exp07_write_ladder';CH=1024;NB=2000
WORLDS=((62,'v3_16_1','train'),(71,'v3_4_1','train'),(72,'v3_32_1','train'),(65,'v3_16_1','eval'),(73,'v3_4_1','eval'),(74,'v3_32_1','eval'));DENS={'v3_4_1':4,'v3_16_1':16,'v3_32_1':32}
def spearman(a,b):
    if len(a)<3 or np.std(a)==0 or np.std(b)==0:return float('nan')
    return float(np.corrcoef(rankdata(a),rankdata(b))[0,1])
def top_mask(s,n):
    m=np.zeros(len(s),bool)
    if n>0:m[np.argsort(-s,kind='stable')[:n]]=True
    return m
def ci(x):
    x=np.asarray(x,float);return [float(np.nanmean(x)),float(np.nanpercentile(x,2.5)),float(np.nanpercentile(x,97.5))]
def chunk_stats(g,gp,t):
    n=len(g);n5=int(np.ceil(0.05*n));A=g>2;B=gp>2;r={'n5':n5,'n_abs2':int(A.sum()),'n_gp2':int(B.sum()),'spear':spearman(gp,g)}
    tg=top_mask(g,n5);tp_=top_mask(gp,n5);r['top5_overlap']=(tg&tp_).sum()/n5;r['top5gp_in_abs2']=(tp_&A).sum()/n5
    r['abs2_set_recall']=(A&B).sum()/max(1,A.sum());r['abs2_set_precision']=(A&B).sum()/max(1,B.sum());r['count_ratio_gp2_over_abs2']=B.sum()/max(1,A.sum())
    if t.sum()>0:
        r['dec_rec_top5_gp']=(tp_&t).sum()/t.sum();r['dec_prec_top5_gp']=(tp_&t).sum()/n5;r['dec_rec_top5_g']=(tg&t).sum()/t.sum();r['dec_prec_top5_g']=(tg&t).sum()/n5
        r['dec_rec_gp2']=(B&t).sum()/t.sum();r['dec_prec_gp2']=(B&t).sum()/max(1,B.sum());r['dec_rec_abs2']=(A&t).sum()/t.sum();r['dec_prec_abs2']=(A&t).sum()/max(1,A.sum())
    return r
def summarize(P,rng):
    keys=[k for k in P[0] if all(k in p for p in P)]+[k for k in P[0] if not all(k in p for p in P)]
    A={k:np.array([p.get(k,np.nan) for p in P],float) for k in dict.fromkeys(keys)};idx=rng.integers(0,len(P),(NB,len(P)));res={'n_chunks':len(P)}
    for k,v in A.items():res[k]=ci(np.nanmean(v[idx],1))
    return res
def world(seed,cell):
    import e55_run as R55,e55_uv as UVm,e57_syn_pass as SP,e57_longcell,e57_dwg_analyze as DW,e57_pqa2_stream as S,e57_closure_masks as CM
    enc,w,tf=R55.build(SP.synth_args(seed,cell),seed,arms=['NAT']);Sx=R55.streams(w,enc);ids,sha=Sx['A'];ends=UVm.sentence_ends(UVm.tokenizer(tf),ids);Z=np.load(EXP/f'e57_syn_s{seed}/syn_s{seed}_nll128.npz');Y=np.load(EXP/f'e57_suffixband/s{seed}/syn_s{seed}_nllsuf.npz');b=list(Z['bands']).index(128);tk=S.tokenizer()
    nch=len(ids)//CH;place=np.zeros(nch*CH+1,bool)
    for r in DW.locate_syn(tk,w,None):place[r['pos'][0]]=True
    P=[]
    for k in range(int(Z['chunks'][0]),int(Z['chunks'][1])):
        q=UVm.chunk_sents(ends,k,CH);sid=np.full(CH,-1)
        for i,(s_,e_) in enumerate(q):sid[s_:e_]=i
        lm=Z['lm'][k];el=CM.rows_eligible(lm,sid)&lm;ix=np.nonzero(el)[0]
        g=(Z['nll_band'][b][k]-Z['nll_win'][k]).astype(np.float64)[ix];gp=(Y['nll_suf'][0][k]-Y['nll_win'][k]).astype(np.float64)[ix];t=place[k*CH+ix+1]
        P.append(chunk_stats(g,gp,t))
    return P
def selftest():
    r=np.random.default_rng(0);g=r.standard_normal(1000)*2;gp=g+r.standard_normal(1000)*.3;t=g>3;s=chunk_stats(g,gp,t);assert s['spear']>0.9 and s['top5_overlap']>0.5 and s['dec_rec_top5_gp']>0;print('e57_suffixband_report selftest ok')
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out');ap.add_argument('--selftest',action='store_true');a=ap.parse_args()
    if a.selftest:selftest();return
    rng=np.random.default_rng(0);res={'definitions':__doc__.split('\n')[0],'worlds':{},'by_density':{},'timing':{}};allP={}
    for sd,cell,role in WORLDS:
        P=world(sd,cell);allP[sd]=P;res['worlds'][f's{sd}_{cell}_{role}']=summarize(P,rng);print('world',sd,flush=True)
        j=json.load(open(EXP/f'e57_suffixband/s{sd}/syn_s{sd}_nllsuf.json'));j0=json.load(open(EXP/f'e57_syn_s{sd}/syn_s{sd}_nll128.json'))
        res['timing'][f's{sd}']=dict(sec_suffixband_pass_total=j['sec'],sec_full_band_pass_total=j0['sec'],natural_forward_sum=j['sec_natural_forward_sum'],suffixband_forward_sum=j['sec_suffixband_forward_sum'],ratio_suffixband_over_natural=j['sec_suffixband_forward_sum']/j['sec_natural_forward_sum'])
    for cell,d in DENS.items():res['by_density'][f'density{d}']=summarize(sum((allP[sd] for sd,c,_ in WORLDS if c==cell),[]),rng)
    res['pooled_all_6']=summarize(sum(allP.values(),[]),rng);res['pooled_eval_3']=summarize(sum((allP[sd] for sd,_,r in WORLDS if r=='eval'),[]),rng)
    tm=res['timing'];res['timing_total']=dict(ratio_suffixband_over_natural=sum(v['suffixband_forward_sum'] for v in tm.values())/sum(v['natural_forward_sum'] for v in tm.values()),sec_suffixband_pass=sum(v['sec_suffixband_pass_total'] for v in tm.values()),sec_full_band_pass=sum(v['sec_full_band_pass_total'] for v in tm.values()))
    pathlib.Path(a.out).write_text(json.dumps(res,indent=1));print('wrote',a.out)
if __name__=='__main__':main()
