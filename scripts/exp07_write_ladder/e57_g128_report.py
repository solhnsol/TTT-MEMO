"""E57 distribution of the window gain g128 = nll_band128 - nll_win over the ELIGIBLE rows (new file; CPU; GPM thread, Hans 'a b c 승인' 11/05): quantiles 5/10/25/50/75/90/95/99% and the share of rows with g above 1, 2, 3, 4, 6 nats. Sources: QA (e57_dwg_qa_nll.npz of the 65-version stream; eligible = loss rows, as in e57_pqa16_run.deciding_masks; tune bundles 8 / 10 / 12, hold-out bundles 5 / 9), news stream A (e57_band/e57_band_W0.npz, band 128, eligible = loss rows whose next token is inside a sentence, as in e57_news_m3.masks_stage, chunks >= K0), synthetic worlds (e57_syn_s<seed>/syn_s<seed>_nll128.npz of job 1060, eligible as in e57_syn_cmasks).
  python3 e57_g128_report.py --qa --news [--syn 62:v3_16_1,71:v3_4_1,...] --out J | --selftest"""
import sys,json,pathlib,argparse
import numpy as np
HERE=pathlib.Path(__file__).resolve().parent;ROOT=HERE.parents[1];sys.path.insert(0,str(HERE));sys.path.insert(0,str(ROOT/'scripts/exp04_what_ttt_retains'))
EXP=ROOT/'experiments/exp07_write_ladder';CH=1024
import e57_closure_masks as CM
QS=[5,10,25,50,75,90,95,99];THR=[1,2,3,4,6]
def summarize(gs):
    g=np.concatenate(gs) if gs else np.zeros(0)
    if not len(g):return dict(n_rows=0)
    return dict(n_rows=int(len(g)),mean=float(g.mean()),quantiles={str(q):float(np.percentile(g,q)) for q in QS},share_above={str(t):float((g>t).mean()) for t in THR})
def qa(tune=(8,10,12),hold=(5,9)):
    Z=np.load(EXP/'e57_public_qa_n65/e57_dwg_qa_nll.npz');bi=list(Z['bands']).index(128);res={}
    per={}
    for gid in (5,8,9,10,12):
        win=Z[f'g{gid}_win'];band=Z[f'g{gid}_band'][bi];lm=Z[f'g{gid}_lm'];gs=[]
        for k in range(win.shape[0]):
            el=CM.rows_eligible(lm[k],np.zeros(CH,np.int32))&lm[k];gs.append((band[k]-win[k]).astype(np.float64)[el])
        per[gid]=gs
    res['tune_bundles_%s'%'_'.join(map(str,tune))]=summarize([x for g in tune for x in per[g]]);res['hold_bundles_%s'%'_'.join(map(str,hold))]=summarize([x for g in hold for x in per[g]])
    for gid in per:res[f'bundle_{gid}']=summarize(per[gid])
    return res
def news():
    import e55_uv as UVm,e57_t1t2v2_run as V,e57_news_m3 as NM
    ids=V.stream_ids(str(EXP/'e57_t1t2v2_in/e57_stream_sentences.jsonl'),'A');Z=np.load(EXP/'e57_band/e57_band_W0.npz');b=list(Z['bands']).index(128);ends=UVm.sentence_ends(UVm.tokenizer(str(ROOT/'data/tokenizer/llama3/tokenizer.json')),ids);gs=[]
    for k in range(NM.K0,len(ids)//CH):
        q=UVm.chunk_sents(ends,k,CH);sid=np.full(CH,-1)
        for i,(s_,e_) in enumerate(q):sid[s_:e_]=i
        lm=Z['lm'][k];el=CM.rows_eligible(lm,sid)&lm;gs.append((Z['nll_band'][b][k]-Z['nll_win'][k]).astype(np.float64)[el])
    return summarize(gs)
def syn(seed,cell):
    import e55_run as R55,e55_uv as UVm,e57_syn_pass as SP,e57_longcell
    enc,w,tf=R55.build(SP.synth_args(seed,cell),seed,arms=['NAT']);S=R55.streams(w,enc);ids,sha=S['A'];ends=UVm.sentence_ends(UVm.tokenizer(tf),ids);Z=np.load(EXP/f'e57_syn_s{seed}/syn_s{seed}_nll128.npz');b=list(Z['bands']).index(128);gs=[]
    for k in range(int(Z['chunks'][0]),int(Z['chunks'][1])):
        q=UVm.chunk_sents(ends,k,CH);sid=np.full(CH,-1)
        for i,(s_,e_) in enumerate(q):sid[s_:e_]=i
        lm=Z['lm'][k];el=CM.rows_eligible(lm,sid)&lm;gs.append((Z['nll_band'][b][k]-Z['nll_win'][k]).astype(np.float64)[el])
    return summarize(gs)
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--qa',action='store_true');ap.add_argument('--news',action='store_true');ap.add_argument('--syn',default='');ap.add_argument('--out');ap.add_argument('--selftest',action='store_true');a=ap.parse_args()
    if a.selftest:
        s=summarize([np.arange(10.0)]);assert s['n_rows']==10 and abs(s['share_above']['4']-0.5)<1e-9 and abs(s['quantiles']['50']-4.5)<1e-9;print('e57_g128_report selftest ok');return
    res={}
    if a.qa:res['qa']=qa()
    if a.news:res['news_streamA']=news()
    for x in [x for x in a.syn.split(',') if x]:
        sd,cell=x.split(':');res[f'syn_s{sd}_{cell}']=syn(int(sd),cell)
    pathlib.Path(a.out).write_text(json.dumps(res,indent=1));print('wrote',a.out)
if __name__=='__main__':main()
