"""E57 synthetic s64: CONTINUOUS row sets for the arms M3-CG / M3-CGSH / M3-CW / M3-CB (CPU only; new file; method thread, Hans 11/05 10:31 KST "g에 비례해 직전 토큰 수를 줄이고 x8도 g에 비례"). From the W0 pass file syn_s64_nll.npz and the eligibility of e57_syn_mmasks.py, per chunk k = 8..127:
  g_i = nll_band256_i - nll_win_i (the same window gain as the deciding rows of jobs 710 / 730 / 950; g is only the RANKING signal here, the band is not re-measured per row)
  g90 = the 90th percentile of g over the eligible rows of the chunk;  s_i = clip(g_i, 0, g90) / g90 in [0, 1] for eligible rows, 0 for all other rows (natural window, weight 1)
  qc_s_<k>   = s (float32 [1024])           band b_i = 8192 x (16/8192)^s_i  (s = 0: b = 8192 = the natural window of the model; s = 1: 16 tokens), weight w_i = 1 + (w - 1) s_i (w = 10 in job 1000: 1 + 9 s; w = 8: 1 + 7 s; the weight is applied in e57_syn_mrun.py)
  qcsh_s_<k> = the SAME values of s, permuted among the eligible rows of the chunk (np.random.default_rng(6500 + k)): control with identical distribution and no relation to g
Output <out> npz + json (per chunk: mean s, share s>0, mean weight, mean band, share s = 1; summary = mean over chunks, next to the D30 values: mean weight 3.1 = 1 + 7 x .30).
  Grid job 1020: the same file also holds qb_TOP30_<k>, qt15_<k>, qt50_<k> (packbits of the top 30 / 15 / 50% eligible rows by g) and qc_s_<k> (continuous s, used as a GRADED WEIGHT with band 256 on every row, kind M3GW).
  .venv/bin/python scripts/exp07_write_ladder/e57_syn_cmasks.py --out F.npz | --selftest"""
import sys,json,argparse,pathlib,hashlib
import numpy as np
HERE=pathlib.Path(__file__).resolve().parent;ROOT=HERE.parents[1]
sys.path.insert(0,str(HERE))
import e57_closure_masks as CM
EXP=ROOT/'experiments/exp07_write_ladder';CH=1024;K0=8;K1=128;WIN=8*CH;PCT=90;BMIN=16;WMAX=8.0;KEXP=3.0
def s_of(g,el,pct=PCT):
    s=np.zeros(CH,np.float32)
    if el.sum()==0:return s
    g90=float(np.percentile(g[el],pct))
    if g90>0:s[el]=(np.clip(g[el],0,g90)/g90).astype(np.float32)
    return s
def exp_shape(g,el,top_rows_mask,k=KEXP):
    """EXPONENTIAL weight shape with a cap that follows the M3 step (Hans 11/05 21:09 KST): gc = the smallest g among the top-30% rows (the M3 weighting threshold of the chunk); t = clip(g, 0, gc) / gc; f = (exp(k t) - 1) / (exp(k) - 1) in [0, 1]. Rows with g >= gc (= exactly the top-30% rows) get f = 1, i.e. the full weight W of 1 + (W - 1) f; lower rows ramp up exponentially from f = 0 at g <= 0. Non-eligible rows f = 0."""
    f=np.zeros(CH,np.float32)
    if top_rows_mask.sum()==0:return f
    gc=float(g[top_rows_mask].min())
    if gc<=0:return f
    tt=np.clip(g[el],0,gc)/gc;f[el]=((np.exp(k*tt)-1)/(np.exp(k)-1)).astype(np.float32);return f
def band_of(s):return np.maximum(np.round(WIN*(BMIN/WIN)**np.asarray(s,np.float64)).astype(np.int64),BMIN)
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out');ap.add_argument('--selftest',action='store_true');a=ap.parse_args()
    if a.selftest:
        rng=np.random.default_rng(0);el=rng.random(CH)<.8;g=rng.standard_normal(CH)*2+1;s=s_of(g,el);assert s.min()>=0 and s.max()<=1 and (s[~el]==0).all() and (s[el]<=1).all()
        assert abs(float(np.percentile(g[el],PCT))-float(np.percentile(g[el],PCT)))<1e-12 and (s[el&(g>=np.percentile(g[el],PCT))]==1).all() and (s[el&(g<=0)]==0).all()
        b=band_of(np.array([0.,1.,.5]));assert b[0]==WIN and b[1]==BMIN and BMIN<b[2]<WIN
        o=np.argsort(g[el]);assert np.all(np.diff(s[el][o])>=0);rg=np.random.default_rng(3);elx=rg.random(CH)<.8;gx=rg.standard_normal(CH)*2+1;tx=CM.top_rows(gx,elx,int(round(.3*elx.sum())));fx=exp_shape(gx,elx,tx);assert fx.min()>=0 and fx.max()<=1+1e-6 and np.allclose(fx[tx],1.0) and (fx[~elx]==0).all() and np.all(fx[elx&(gx<=0)]==0);ox=np.argsort(gx[elx]);assert np.all(np.diff(fx[elx][ox])>=-1e-7);print('e57_syn_cmasks selftest ok');return
    import e55_run as R55,e55_uv as UVm,e57_syn_pass as SP
    import e57_syn_mmasks as MM
    enc,w,tf=R55.build(SP.synth_args(64),64,arms=['NAT']);S=R55.streams(w,enc);ids,sha=S['A'];ends=UVm.sentence_ends(UVm.tokenizer(tf),ids);Z=np.load(EXP/'e57_syn_s64/syn_s64_nll.npz');b=list(Z['bands']).index(256);out={};st=[]
    pre={}                                                         # job 1040 adaptive threshold: one GLOBAL g threshold per run = the pooled (all chunks) 5% / 10% quantile of g over eligible rows; each chunk then masks the rows with g > tau, so the per-chunk share varies with the data
    for k in range(K0,K1):
        q=UVm.chunk_sents(ends,k,CH);sid=np.full(CH,-1)
        for i,(s_,e_) in enumerate(q):sid[s_:e_]=i
        lm=Z['lm'][k];el=CM.rows_eligible(lm,sid)&lm;pre[k]=(el,(Z['nll_band'][b][k]-Z['nll_win'][k]).astype(np.float64))
    pool=np.concatenate([g_[e_] for e_,g_ in pre.values()]);tau={'A05':float(np.percentile(pool,95)),'A10':float(np.percentile(pool,90))};print('tau',tau)
    for k in range(K0,K1):
        q=UVm.chunk_sents(ends,k,CH);sid=np.full(CH,-1)
        for i,(s_,e_) in enumerate(q):sid[s_:e_]=i
        lm=Z['lm'][k];el=CM.rows_eligible(lm,sid)&lm;g=(Z['nll_band'][b][k]-Z['nll_win'][k]).astype(np.float64);s=s_of(g,el)
        idx=np.where(el)[0];perm=np.random.default_rng(6500+k).permutation(idx);ssh=np.zeros(CH,np.float32);ssh[idx]=s[perm]
        out[f'qc_s_{k}']=s;out[f'qcsh_s_{k}']=ssh
        t30=CM.top_rows(g,el,int(round(.30*el.sum())));out[f'qe_f_{k}']=exp_shape(g,el,t30)      # job 1020 exponential graded weight (kind M3GE)
        for pp_,nm in ((.30,'qb_TOP30'),(.15,'qt15'),(.50,'qt50'),(.05,'qt05'),(.10,'qt10')):out[f'{nm}_{k}']=CM.pk(CM.top_rows(g,el,int(round(pp_*el.sum()))))      # grid job 1020: deciding rows = the top 30% (= qb_TOP30 of jobs 710 / 730, same rule as e57_syn_mmasks.deciding), 15% and 50% of the eligible rows by g
        for nm_,t_ in tau.items():out[f'qt{nm_}_{k}']=CM.pk(el&(g>t_))      # kinds M3TA05 / M3TA10
        bb=band_of(s);st.append(dict(chunk=k,n_eligible=int(el.sum()),mean_s=float(s.mean()),share_pos=float((s>0).mean()),share_full=float((s>=1).mean()),mean_weight=float((1+9*s).mean()),mean_band_pos=float(bb[s>0].mean()) if (s>0).any() else None,median_band_pos=float(np.median(bb[s>0])) if (s>0).any() else None,mean_g_by_s_decile=[float(g[el&(s>=lo)&(s<hi+(1e-9 if hi==1 else 0))].mean()) if (el&(s>=lo)&(s<hi+(1e-9 if hi==1 else 0))).any() else None for lo,hi in zip(np.linspace(0,.9,10),np.linspace(.1,1,10))]))
    out['chunks']=np.asarray([K0,K1]);np.savez_compressed(a.out,**out)
    s={k:float(np.mean([x[k] for x in st if x[k] is not None])) for k in ('n_eligible','mean_s','share_pos','share_full','mean_weight','mean_band_pos','median_band_pos')};s['reference_D30_mean_weight_w8']=3.1;s['tau']=tau;s['adapt_share']={nm_:float(np.mean([ (pre[k][0]&(pre[k][1]>t_)).sum()/max(pre[k][0].sum(),1) for k in pre])) for nm_,t_ in tau.items()}
    pathlib.Path(a.out).with_suffix('.json').write_text(json.dumps(dict(summary=s,chunks=st,out_sha=hashlib.sha256(open(a.out,'rb').read()).hexdigest()[:16])));print(json.dumps(s))
if __name__=='__main__':main()
