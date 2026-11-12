"""E57 distribution of the band-128 window gain g128 = nll_band128 - nll_win over the eligible rows (new file; CPU; GPM thread, Hans 11/06 '(cpu 분석 돌려봐)'): (1) cumulative positive-g mass: the share of the chunk's total positive g held by the top p% of eligible rows (p = 1, 2, 5, 10, 15, 20, 30, 50), the Kneedle knee (point of maximum distance of the normalized cumulative-mass curve above the diagonal) as a share of rows and the Gini coefficient of the positive g values, averaged over chunks; (2) synthetic only: the TRUE deciding rows = rows that predict the first content token of the place P (the read answer) of a fact sentence (as in e57_branchrows_cpu.py; also the variant place OR name), recall / precision / lift over random of the top p% rows by g128 (p = 1, 2, 5, 10, 15, 30), per world and averaged, and the number of such rows per chunk; (3) a 2-component Gaussian mixture on the pooled g128 of each source / world (EM, 1-D): means, standard deviations, weights, share of rows assigned to the high component. Sources: synthetic tune worlds s62 / s71 / s72 (e57_syn_s<seed>/syn_s<seed>_nll128.npz, chunks 8..127), QA tune bundles 8 / 10 / 12 and hold-out 5 / 9 (e57_dwg_qa_nll.npz), news stream A (e57_band_W0.npz, chunks >= K0).
  CUDA_VISIBLE_DEVICES= JAX_PLATFORMS=cpu python3 e57_g_mass_analysis.py --out F.json | --selftest"""
import sys,json,argparse,pathlib
import numpy as np
HERE=pathlib.Path(__file__).resolve().parent;ROOT=HERE.parents[1]
for p in (HERE,HERE/'e57',ROOT/'scripts/exp04_what_ttt_retains',ROOT/'scripts/exp05_write_dynamics'):sys.path.insert(0,str(p))
import e57_closure_masks as CM
EXP=ROOT/'experiments/exp07_write_ladder';CH=1024
PM=[1,2,5,10,15,20,30,50];PR=[1,2,5,10,15,30]
def mass_curve(g):
    """g: eligible-row gains of one chunk -> (cum share of positive mass for rows sorted by g descending, x = row share)"""
    s=np.sort(g)[::-1];pos=np.clip(s,0,None);tot=pos.sum();c=np.cumsum(pos)/tot if tot>0 else np.zeros_like(s);x=np.arange(1,len(s)+1)/len(s);return x,c
def top_share(x,c,p):
    n=len(x);k=max(1,int(np.ceil(p/100*n)));return float(c[k-1])
def kneedle(x,c):
    d=c-x;return float(x[int(np.argmax(d))])
def gini(v):
    v=np.sort(np.asarray(v,float));n=len(v)
    if n==0 or v.sum()<=0:return float('nan')
    return float((2*np.sum((np.arange(1,n+1))*v)/(n*v.sum()))-(n+1)/n)
def gmm2(x,iters=300):
    x=np.asarray(x,float);mu=np.array([np.percentile(x,25),np.percentile(x,90)]);sd=np.array([x.std()*0.5,x.std()*0.8])+1e-6;w=np.array([.7,.3])
    for _ in range(iters):
        pdf=np.stack([w[i]/(sd[i]*np.sqrt(2*np.pi))*np.exp(-0.5*((x-mu[i])/sd[i])**2) for i in range(2)]);r=pdf/np.maximum(pdf.sum(0),1e-300);nk=r.sum(1);mu=(r*x).sum(1)/nk;sd=np.sqrt((r*(x-mu[:,None])**2).sum(1)/nk)+1e-6;w=nk/len(x)
    hi=int(np.argmax(mu));pdf=np.stack([w[i]/(sd[i]*np.sqrt(2*np.pi))*np.exp(-0.5*((x-mu[i])/sd[i])**2) for i in range(2)]);post=pdf[hi]/np.maximum(pdf.sum(0),1e-300)
    return dict(means=[float(mu[i]) for i in range(2)],stds=[float(sd[i]) for i in range(2)],weights=[float(w[i]) for i in range(2)],high_component=hi,share_assigned_to_high=float((post>0.5).mean()),bic=float(-2*np.sum(np.log(np.maximum(pdf.sum(0),1e-300)))+5*np.log(len(x))))
def analyze(chunks_g,true_rows=None):
    """chunks_g: list of (g_eligible [n], true_flag_among_eligible [n] or None)"""
    cs={p:[] for p in PM};kn=[];gi=[];allg=[];rec={p:[] for p in PR};pre={p:[] for p in PR};lift={p:[] for p in PR};ntrue=[];base=[]
    for g,t in chunks_g:
        x,c=mass_curve(g);[cs[p].append(top_share(x,c,p)) for p in PM];kn.append(kneedle(x,c));gi.append(gini(g[g>0]));allg.append(g)
        if t is not None and t.sum()>0:
            o=np.argsort(-g,kind='stable');ntrue.append(int(t.sum()));b=t.mean();base.append(float(b))
            for p in PR:
                k=max(1,int(np.ceil(p/100*len(g))));hit=t[o[:k]].sum();rec[p].append(hit/t.sum());pre[p].append(hit/k);lift[p].append((hit/k)/b)
    res=dict(n_chunks=len(chunks_g),eligible_rows_per_chunk=float(np.mean([len(g) for g,_ in chunks_g])),positive_mass_top_share={str(p):float(np.mean(cs[p])) for p in PM},kneedle_knee_row_share=float(np.mean(kn)),gini_of_positive_g=float(np.nanmean(gi)),gmm2=gmm2(np.concatenate(allg)))
    if rec[PR[0]]:res['deciding_rows']=dict(per_chunk_mean=float(np.mean(ntrue)),base_rate_among_eligible=float(np.mean(base)),recall={str(p):float(np.mean(rec[p])) for p in PR},precision={str(p):float(np.mean(pre[p])) for p in PR},lift_over_random={str(p):float(np.mean(lift[p])) for p in PR})
    return res
def syn_world(seed,cell):
    import e55_run as R55,e55_uv as UVm,e57_syn_pass as SP,e57_longcell,e57_dwg_analyze as DW,e57_pqa2_stream as S
    enc,w,tf=R55.build(SP.synth_args(seed,cell),seed,arms=['NAT']);Sx=R55.streams(w,enc);ids,sha=Sx['A'];ends=UVm.sentence_ends(UVm.tokenizer(tf),ids);Z=np.load(EXP/f'e57_syn_s{seed}/syn_s{seed}_nll128.npz');b=list(Z['bands']).index(128);tk=S.tokenizer()
    nch=len(ids)//CH;place=np.zeros(nch*CH+1,bool);namb=np.zeros(nch*CH+1,bool)
    for r in DW.locate_syn(tk,w,None):place[r['pos'][0]]=True
    for c in w['chunks']:
        for s in c['sents']:
            X=w['facts'][s['fact']]['X'];seg=list(c['ids'][s['start']:s['end']])
            for pre in (' ',''):
                p=tk.encode(pre+X,add_special_tokens=False).ids;hit=[i for i in range(len(seg)-len(p)+1) if seg[i:i+len(p)]==p]
                if hit:namb[c['k']*CH+s['start']+hit[0]]=True;break
    out_p=[];out_pn=[]
    for k in range(int(Z['chunks'][0]),int(Z['chunks'][1])):
        q=UVm.chunk_sents(ends,k,CH);sid=np.full(CH,-1)
        for i,(s_,e_) in enumerate(q):sid[s_:e_]=i
        lm=Z['lm'][k];el=CM.rows_eligible(lm,sid)&lm;g=(Z['nll_band'][b][k]-Z['nll_win'][k]).astype(np.float64)
        tp=np.asarray([place[k*CH+i+1] for i in range(CH)]);tpn=np.asarray([(place|namb)[k*CH+i+1] for i in range(CH)])
        out_p.append((g[el],tp[el]));out_pn.append((g[el],tpn[el]))
    return analyze(out_p),analyze(out_pn)
def qa_bundles(gids):
    Z=np.load(EXP/'e57_public_qa_n65/e57_dwg_qa_nll.npz');bi=list(Z['bands']).index(128);out=[]
    for gid in gids:
        win=Z[f'g{gid}_win'];band=Z[f'g{gid}_band'][bi];lm=Z[f'g{gid}_lm']
        for k in range(win.shape[0]):
            el=CM.rows_eligible(lm[k],np.zeros(CH,np.int32))&lm[k];out.append(((band[k]-win[k]).astype(np.float64)[el],None))
    return out
def news():
    import e55_uv as UVm,e57_t1t2v2_run as V,e57_news_m3 as NM
    ids=V.stream_ids(str(EXP/'e57_t1t2v2_in/e57_stream_sentences.jsonl'),'A');Z=np.load(EXP/'e57_band/e57_band_W0.npz');b=list(Z['bands']).index(128);ends=UVm.sentence_ends(UVm.tokenizer(str(ROOT/'data/tokenizer/llama3/tokenizer.json')),ids);out=[]
    for k in range(NM.K0,len(ids)//CH):
        q=UVm.chunk_sents(ends,k,CH);sid=np.full(CH,-1)
        for i,(s_,e_) in enumerate(q):sid[s_:e_]=i
        lm=Z['lm'][k];el=CM.rows_eligible(lm,sid)&lm;out.append(((Z['nll_band'][b][k]-Z['nll_win'][k]).astype(np.float64)[el],None))
    return out
def selftest():
    r=np.random.default_rng(0);g=np.concatenate([r.standard_normal(900)*.3,r.standard_normal(100)*2+5]);t=np.concatenate([np.zeros(900,bool),np.ones(100,bool)]);o=r.permutation(1000);g=g[o];t=t[o]
    a=analyze([(g,t)]*3);assert a['positive_mass_top_share']['10']>0.8 and a['deciding_rows']['recall']['10']>0.9 and a['deciding_rows']['lift_over_random']['10']>5 and 0<a['kneedle_knee_row_share']<0.5 and a['gmm2']['share_assigned_to_high']>0.05;print('e57_g_mass_analysis selftest ok')
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out');ap.add_argument('--selftest',action='store_true');a=ap.parse_args()
    if a.selftest:selftest();return
    res={'definitions':'g128 = nll_band128 - nll_win; eligible rows as in e57_closure_masks.rows_eligible (synthetic / news: next token inside a sentence); chunk statistics averaged over chunks; true deciding rows (synthetic) = rows predicting the first content token of the place (read answer) [variant: place or name]','synthetic':{}}
    for sd,cell in ((62,'v3_16_1'),(71,'v3_4_1'),(72,'v3_32_1')):
        p,pn=syn_world(sd,cell);res['synthetic'][f's{sd}_{cell}']=dict(place_first_token=p,place_or_name_first_token=pn);print('synthetic',sd,'done',flush=True)
    wr=[res['synthetic'][k]['place_first_token']['deciding_rows'] for k in res['synthetic']]
    res['synthetic_average_over_worlds']=dict(per_chunk_deciding_rows=float(np.mean([w['per_chunk_mean'] for w in wr])),recall={str(p):float(np.mean([w['recall'][str(p)] for w in wr])) for p in PR},precision={str(p):float(np.mean([w['precision'][str(p)] for w in wr])) for p in PR},lift_over_random={str(p):float(np.mean([w['lift_over_random'][str(p)] for w in wr])) for p in PR},mass_top_share={str(p):float(np.mean([res['synthetic'][k]['place_first_token']['positive_mass_top_share'][str(p)] for k in res['synthetic']])) for p in PM})
    res['qa']={'tune_bundles_8_10_12':analyze(qa_bundles((8,10,12))),'hold_bundles_5_9':analyze(qa_bundles((5,9)))}
    for gid in (5,8,9,10,12):res['qa'][f'bundle_{gid}']=analyze(qa_bundles((gid,)))
    res['news_streamA']=analyze(news());pathlib.Path(a.out).write_text(json.dumps(res,indent=1));print('wrote',a.out)
if __name__=='__main__':main()
