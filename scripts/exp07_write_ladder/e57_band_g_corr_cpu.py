"""E57 band width: exposure-visibility shares vs the per-fact gain G (CPU only; new file; exploratory, K0 = 0, no multiplicity correction; definitions fixed before the numbers were seen).
V1 (per fact, per seed, per width W): share of VISIBLE (query, key) pairs = 1 - hidden share, over all sentences of the fact in the stream that have a previous exposure of the same fact inside the natural window (previous 8 chunks); query = every token of the sentence, key = every token of that earlier exposure, visible = distance < W (band_keep).
V2: share of those sentences whose NEAREST previous exposure (place anchor = last token of the sentence) is at distance < W.
G_W (per fact, per seed) = margin(NATMASK_W) - margin(NATS), out-of-window read at exposure 128, margin = log p(gold) - max log p(wrong) (e57_1280_margin.margin_secondary = the 'max' convention of e57_band_dist_analyze). NATMASK_256 and NATS from job 1283 (e57_k0conf16, tag k0c16), the other widths from job 1293 (e57_band_dist2/w<W>, tag bd<W>), seeds 71..78, cell v3_16_1 (16 facts).
  .venv/bin/python scripts/exp07_write_ladder/e57_band_g_corr_cpu.py --out J"""
import sys,json,argparse,pathlib
import numpy as np
from scipy import stats
HERE=pathlib.Path(__file__).resolve().parent;ROOT=HERE.parents[1];EXP=ROOT/'experiments/exp07_write_ladder'
for p in (HERE,HERE/'e57',ROOT/'scripts/exp04_what_ttt_retains',ROOT/'scripts/exp05_write_dynamics'):sys.path.insert(0,str(p))
import e57_1280_margin as MG
CH=1024;WIN=8192;WS=(64,256,512,768,1024,2048,4096);SEEDS=range(71,79)
def per_fact_V(sd):
    import e57_band_exposure_cpu as X,e55_run as R55,e57_syn_pass as SP
    enc,w,tf=R55.build(SP.synth_args(sd,'v3_16_1'),sd,arms=['NAT']);d=X.sentences(w);out={}
    for f,L in d.items():
        hid={W:0 for W in WS};tot=0;nv={W:0 for W in WS};ns=0
        for i,(c,s0,e0) in enumerate(L):
            prior=[(c2,s2,e2) for (c2,s2,e2) in L[:i] if s2>=c*CH-WIN and c2<c]
            if not prior:continue
            ns+=1;tq=np.arange(s0,e0+1);dist=min(e0-e2 for (_,_,e2) in prior)
            for (_,s2,e2) in prior:
                D=tq[:,None]-np.arange(s2,e2+1)[None,:];tot+=D.size
                for W in WS:hid[W]+=int((D>=W).sum())
            for W in WS:nv[W]+=int(dist<W)
        out[f]=dict(V1={W:1-hid[W]/tot for W in WS},V2={W:nv[W]/ns for W in WS})
    return out
def margins(path):
    d=json.load(open(path));m=d['meta'];g=np.array(m['gold_idx']);fi=np.array(m['fact']);rows=[i for i,k in enumerate(m['kind']) if k=='fact']
    L=np.array(d['read']['128']['out'],float);mg=MG.margin_secondary(L[rows],g[rows]);return {int(fi[r]):float(x) for r,x in zip(rows,mg)}
def sp(x,y):
    x,y=np.asarray(x,float),np.asarray(y,float)
    if np.ptp(x)==0 or np.ptp(y)==0:return (None,None)
    return float(stats.spearmanr(x,y)[0]),float(stats.pearsonr(x,y)[0])
def tci(v):
    v=np.array([x for x in v if x is not None],float);n=len(v)
    if n<2:return None
    h=stats.t.ppf(.975,n-1)*v.std(ddof=1)/np.sqrt(n);return [float(v.mean()),float(v.mean()-h),float(v.mean()+h),n]
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out');a=ap.parse_args();R={}
    for sd in SEEDS:
        V=per_fact_V(sd);nats=margins(EXP/f'e57_k0conf16/E55_k0c16_NATS_s{sd}.read.json');G={}
        for W in WS:
            p=EXP/(f'e57_k0conf16/E55_k0c16_NATMASK_s{sd}.read.json' if W==256 else f'e57_band_dist2/w{W}/E55_bd{W}_NATMASK_s{sd}.read.json');mk=margins(p);G[W]={f:mk[f]-nats[f] for f in nats}
        R[sd]=dict(V=V,G=G);print('seed',sd,'ok',flush=True)
    res={}
    # (a) per seed over the 7 widths: mean G_W vs mean V1_W / V2_W
    A={'V1':[],'V2':[]}
    for sd in SEEDS:
        g=[np.mean(list(R[sd]['G'][W].values())) for W in WS]
        for v in ('V1','V2'):A[v].append(sp(g,[np.mean([R[sd]['V'][f][v][W] for f in R[sd]['V']]) for W in WS]))
    res['a_across_widths']={v:dict(spearman_by_seed=[x[0] for x in A[v]],pearson_by_seed=[x[1] for x in A[v]],spearman_mean_ci=tci([x[0] for x in A[v]]),pearson_mean_ci=tci([x[1] for x in A[v]])) for v in A}
    res['G_mean_by_width']={W:float(np.mean([np.mean(list(R[sd]['G'][W].values())) for sd in SEEDS])) for W in WS}
    res['V_mean_by_width']={v:{W:float(np.mean([np.mean([R[sd]['V'][f][v][W] for f in R[sd]['V']]) for sd in SEEDS])) for W in WS} for v in ('V1','V2')}
    # (b) within width across the 16 facts: per seed correlation, mean over seeds (t CI) and pooled over the 128 facts
    B={}
    for W in WS:
        for v in ('V1','V2'):
            per=[sp([R[sd]['G'][W][f] for f in range(16)],[R[sd]['V'][f][v][W] for f in range(16)]) for sd in SEEDS]
            gp=[R[sd]['G'][W][f] for sd in SEEDS for f in range(16)];vp=[R[sd]['V'][f][v][W] for sd in SEEDS for f in range(16)]
            B[f'{v}_W{W}']=dict(spearman_by_seed_mean_ci=tci([x[0] for x in per]),pearson_by_seed_mean_ci=tci([x[1] for x in per]),pooled128_spearman_pearson=sp(gp,vp),V_sd_within_seed=float(np.mean([np.std([R[sd]['V'][f][v][W] for f in range(16)]) for sd in SEEDS])))
    res['b_within_width_across_facts']=B
    # (c) G = a (1 - V) through the origin, per seed over widths, and pooled over seed-mean widths; residuals
    C={}
    for v in ('V1','V2'):
        gm=np.array([res['G_mean_by_width'][W] for W in WS]);x=np.array([1-res['V_mean_by_width'][v][W] for W in WS]);aa=float((x@gm)/(x@x));pred=aa*x
        r2u=1-float(((gm-pred)**2).sum()/(gm**2).sum());r2c=1-float(((gm-pred)**2).sum()/((gm-gm.mean())**2).sum())
        per=[]
        for sd in SEEDS:
            g=np.array([np.mean(list(R[sd]['G'][W].values())) for W in WS]);xx=np.array([1-np.mean([R[sd]['V'][f][v][W] for f in R[sd]['V']]) for W in WS]);a1=float((xx@g)/(xx@xx));per.append(a1)
        C[v]=dict(a_pooled=aa,R2_uncentered=r2u,R2_centered=r2c,residual_by_width={W:float(gm[i]-pred[i]) for i,W in enumerate(WS)},a_by_seed=per)
    res['c_linear_G_eq_a_times_1_minus_V']=C;res['per_seed_G_and_V_means']={sd:dict(G={W:float(np.mean(list(R[sd]['G'][W].values()))) for W in WS}) for sd in SEEDS}
    s=json.dumps(res,indent=1,default=float)
    if a.out:open(a.out,'w').write(s)
    print(json.dumps({k:res[k] for k in ('G_mean_by_width','V_mean_by_width','a_across_widths','c_linear_G_eq_a_times_1_minus_V')},default=float)[:6000])
if __name__=='__main__':main()
