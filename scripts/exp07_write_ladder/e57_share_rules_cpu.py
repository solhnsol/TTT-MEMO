"""E57 source-adaptive selection share: label-free rules on the W0 window-gain g128 distribution vs the empirical optimum (new file; CPU only; GPM thread, Hans 11/08 '1번 2번 둘 다 병렬로'): per source (synthetic worlds s62/s71/s72/s65/s73/s74, QA bundles 5/8/9/10/12, news A) and per chunk over the eligible rows, the share of rows each rule selects: ABS2 (g128 > 2 nats), gain-mass rho = 50 / 70 / 85 % (the smallest row set covering rho of the positive gain mass), the Kneedle knee of the normalized cumulative positive-gain curve, Otsu (histogram threshold of the chunk g, and one Otsu threshold of the pooled source g), robust z rules (g > median + 3 * 1.4826 MAD; g > mean + 2 sd). Mean share over chunks with the 5-95 % range of the chunk shares. Next to it the EMPIRICAL table: branch-token rank ratio of the mass-matched arms (top 5 / 10 / 15 / 30 % rows by the W0 g128, extra loss mass .70-.75) from the earlier grids (synthetic jobs 1131-1134 on the tune worlds, news job 1240, QA job 1241 / 1080 / 1110).
  CUDA_VISIBLE_DEVICES= JAX_PLATFORMS=cpu python3 e57_share_rules_cpu.py --out F.json | --selftest"""
import sys,json,argparse,pathlib
import numpy as np
HERE=pathlib.Path(__file__).resolve().parent;ROOT=HERE.parents[1];sys.path.insert(0,str(HERE));sys.path.insert(0,str(ROOT/'scripts/exp04_what_ttt_retains'))
EXP=ROOT/'experiments/exp07_write_ladder';CH=1024;K0=8
import e57_closure_masks as CM,e57_rules as RU
def kneedle_share(g):
    p=np.clip(np.sort(g)[::-1],0,None);tot=p.sum()
    if tot<=0:return 0.0
    c=np.cumsum(p)/tot;x=np.arange(1,len(p)+1)/len(p);return float(x[int(np.argmax(c-x))])
def otsu_thr(v,bins=200):
    lo,hi=np.percentile(v,[0.1,99.9]);h,e=np.histogram(np.clip(v,lo,hi),bins=bins,range=(lo,hi));c=(e[:-1]+e[1:])/2;w0=np.cumsum(h);w1=w0[-1]-w0;m0=np.cumsum(h*c);mt=m0[-1];ok=(w0>0)&(w1>0)
    if not ok.any():return float(hi)
    s=np.where(ok,w0*w1*(m0/np.maximum(w0,1)-(mt-m0)/np.maximum(w1,1))**2,-1);return float(c[int(np.argmax(s))])
def chunk_rules(g,thr_pool):
    r={'ABS2':float((g>2).mean())};n=len(g)
    for rho in (50,70,85):r[f'GM{rho}']=float(RU.gainmass_rule(g,np.ones(n,bool),rho/100).mean())
    r['knee']=kneedle_share(g);r['otsu_chunk']=float((g>otsu_thr(g)).mean());r['otsu_pooled']=float((g>thr_pool).mean())
    md=np.median(g);mad=1.4826*np.median(np.abs(g-md))+1e-9;r['robust_z3']=float((g>md+3*mad).mean());r['mean2sd']=float((g>g.mean()+2*g.std()).mean());return r
def summarize(chunks):
    allg=np.concatenate(chunks);thr=otsu_thr(allg);R=[chunk_rules(g,thr) for g in chunks];out={'n_chunks':len(chunks),'mean_g':float(allg.mean()),'frac_g_gt0':float((allg>0).mean()),'otsu_pooled_threshold':thr}
    for k in R[0]:v=np.array([r[k] for r in R]);out[k]=dict(mean=float(v.mean()),p5=float(np.percentile(v,5)),p95=float(np.percentile(v,95)))
    return out
def syn_chunks(seed):
    Z=np.load(EXP/f'e57_syn_s{seed}/syn_s{seed}_nll128.npz');b=list(Z['bands']).index(128);out=[]
    for k in range(int(Z['chunks'][0]),int(Z['chunks'][1])):
        lm=Z['lm'][k];el=CM.rows_eligible(lm,np.ones(CH,np.int32))&lm;out.append((Z['nll_band'][b][k]-Z['nll_win'][k]).astype(np.float64)[el])
    return out
def qa_chunks(gid):
    Z=np.load(EXP/'e57_public_qa_n65/e57_dwg_qa_nll.npz');bi=list(Z['bands']).index(128);out=[]
    for k in range(K0,Z[f'g{gid}_win'].shape[0]):
        lm=Z[f'g{gid}_lm'][k];el=CM.rows_eligible(lm,np.zeros(CH,np.int32))&lm
        if el.sum()>=50:out.append((Z[f'g{gid}_band'][bi][k]-Z[f'g{gid}_win'][k]).astype(np.float64)[el])
    return out
def news_chunks():
    import e55_uv as UVm,e57_t1t2v2_run as V,e57_news_m3 as NM
    ids=V.stream_ids(str(EXP/'e57_t1t2v2_in/e57_stream_sentences.jsonl'),'A');Z=np.load(EXP/'e57_band/e57_band_W0.npz');b=list(Z['bands']).index(128);ends=UVm.sentence_ends(UVm.tokenizer(str(ROOT/'data/tokenizer/llama3/tokenizer.json')),ids);out=[]
    for k in range(NM.K0,len(ids)//CH):
        q=UVm.chunk_sents(ends,k,CH);sid=np.full(CH,-1)
        for i,(s_,e_) in enumerate(q):sid[s_:e_]=i
        lm=Z['lm'][k];el=CM.rows_eligible(lm,sid)&lm;out.append((Z['nll_band'][b][k]-Z['nll_win'][k]).astype(np.float64)[el])
    return out
def empirical():
    E={};ld=lambda p:json.load(open(EXP/p))
    SYN={'M3W1N05':'ALLQ','M3T05W16N05':'top5_w16','M3T10W8P5N05':'top10_w8','M3T15W6N05':'top15_w6','M3W3P5N05':'top30_w3.5','M3W8N05':'top30_w8','M3TA2W16N05':'ABS2_w16'}
    for s in (62,71,72):
        c1=ld(f'e57_syn_rules_s{s}/e57_curve_rules_s{s}.json')['curve'];c2=ld(f'e57_syn_massgrid_s{s}/e57_curve_massgrid_s{s}.json')['curve'];row={}
        for arm,nm in SYN.items():
            c=c2.get(arm) or c1.get(arm)
            if c and '127' in c:row[nm]=c['127']['rr_branch']['mean']
        E[f'synthetic_s{s}']=row
    n=ld('e57_massgrid_news_report.json')['news']['parts'];NW={'ALLQ128':'ALLQ','NEWSqt05W16':'top5_w16','NEWSqt10W8':'top10_w8','NEWSqt15W6':'top15_w6','NEWSqb_TOP30W3p5':'top30_w3.5','M3W8Q128':'top30_w8','NEWSqtA2W16':'ABS2_w16'}
    for p in ('tune','hold'):E[f'news_{p}']={nm:n[p]['arms'][a]['rr_branch']['mean'] for a,nm in NW.items()}
    q=ld('e57_massgrid_qa_report.json')['qa']['parts'];QW={'ALLQ128':'ALLQ','M3qt05W16Q128':'top5_w16','M3qt10W8Q128':'top10_w8','M3qb_TOP30W3p5Q128':'top30_w3.5','M3W8Q128':'top30_w8','M3qtA2W16Q128':'ABS2_w16','M3qg50W16Q128':'GM50_w16','M3qg70W16Q128':'GM70_w16','M3qg85W16Q128':'GM85_w16'}
    for p in ('tune','hold'):E[f'qa_{p}']={nm:q[p]['arms'][a]['rr']['mean'] for a,nm in QW.items() if a in q[p]['arms']}
    for k,v in E.items():
        sh={x:v[x] for x in v if x.startswith('top')};v['best_share_arm_by_rr']=min(sh,key=sh.get)
    return E
def selftest():
    g=np.concatenate([np.random.default_rng(0).standard_normal(900)*.3,np.random.default_rng(1).standard_normal(100)*1+4]);r=chunk_rules(g,otsu_thr(g));assert 0.05<r['ABS2']<0.15 and 0.03<r['knee']<0.4 and 0.05<r['otsu_chunk']<0.2;print('e57_share_rules_cpu selftest ok')
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out');ap.add_argument('--selftest',action='store_true');a=ap.parse_args()
    if a.selftest:selftest();return
    res={'definitions':__doc__.split('\n')[0],'rule_shares':{},'empirical_rank_ratio_by_arm':empirical()}
    for s,c in ((62,'v3_16_1'),(71,'v3_4_1'),(72,'v3_32_1'),(65,'v3_16_1'),(73,'v3_4_1'),(74,'v3_32_1')):res['rule_shares'][f'synthetic_s{s}_{c}']=summarize(syn_chunks(s));print('syn',s,flush=True)
    Q={g:qa_chunks(g) for g in (5,8,9,10,12)}
    for g,c in Q.items():res['rule_shares'][f'qa_bundle_{g}']=summarize(c)
    res['rule_shares']['qa_tune_8_10_12']=summarize(sum((Q[g] for g in (8,10,12)),[]));res['rule_shares']['qa_hold_5_9']=summarize(Q[5]+Q[9]);res['rule_shares']['news_A']=summarize(news_chunks())
    pathlib.Path(a.out).write_text(json.dumps(res,indent=1));print('wrote',a.out)
if __name__=='__main__':main()
