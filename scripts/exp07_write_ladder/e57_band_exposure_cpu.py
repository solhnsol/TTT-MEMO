"""E57 band mask vs exposure distance (CPU only; new file). For the fact sentences of a stream (real tokenizer, real background pool, no GPU) and each band W:
 (a) (query, key) pairs: query token = any token of a fact sentence at chunk c (row i, abs. position q), key = any token of an earlier exposure of the SAME fact that is inside the natural window (the previous 8 chunks, abs. position >= c*1024 - 8192) -> hidden if distance q - key_pos >= W (band_keep: allowed 0 <= distance < W). Share of hidden pairs.
 (b) sentences: nearest previous exposure of the fact (smallest distance between the place anchors = last token of the sentence); share of sentences whose nearest exposure is visible (distance < W); sentences with >= 1 visible and >= 1 hidden window exposure ('mixed'), with no visible ('all hidden'), with no previous exposure in the window ('none').
 (c) distribution of the distance between consecutive exposures of a fact (start to start and place to place).
  .venv/bin/python scripts/exp07_write_ladder/e57_band_exposure_cpu.py --cell v3_16_1 --seeds 71,...,78 [--shuffle-blocks] [--out J]"""
import sys,json,argparse,pathlib
import numpy as np
HERE=pathlib.Path(__file__).resolve().parent;ROOT=HERE.parents[1]
for p in (HERE,HERE/'e57',ROOT/'scripts/exp04_what_ttt_retains',ROOT/'scripts/exp05_write_dynamics'):sys.path.insert(0,str(p))
CH=1024;WIN=8192;BANDS=(256,512,768,1024,2048,4096)
def sentences(w):
    """-> {fact: [(chunk, start_abs, end_abs)]} sorted by position (end_abs = position of the last token of the sentence)"""
    d={}
    for ch in w['chunks']:
        for s in ch['sents']:d.setdefault(s['fact'],[]).append((ch['k'],ch['k']*CH+s['start'],ch['k']*CH+s['end']-1))
    for f in d:d[f].sort(key=lambda t:t[1])
    return d
def stats(d):
    pr={W:[0,0] for W in BANDS};sn={W:dict(n=0,near_vis=0,mixed=0,all_hidden=0) for W in BANDS};none=0;tot=0;gap_s=[];gap_p=[];npri=[]
    for f,L in d.items():
        for i,(c,s0,e0) in enumerate(L):
            tot+=1;prior=[(c2,s2,e2) for (c2,s2,e2) in L[:i] if s2>=c*CH-WIN and c2<c]
            if i>0:gap_s.append(s0-L[i-1][1]);gap_p.append(e0-L[i-1][2])
            if not prior:none+=1;continue
            npri.append(len(prior));dist=[e0-e2 for (_,_,e2) in prior]
            for W in BANDS:
                tq=np.arange(s0,e0+1)                       # all query tokens of the sentence
                for (_,s2,e2) in prior:
                    kk=np.arange(s2,e2+1);D=tq[:,None]-kk[None,:];pr[W][0]+=int((D>=W).sum());pr[W][1]+=D.size
                vis=[x<W for x in dist];sn[W]['n']+=1;sn[W]['near_vis']+=int(min(dist)<W);sn[W]['mixed']+=int(any(vis) and not all(vis));sn[W]['all_hidden']+=int(not any(vis))
    return pr,sn,none,tot,np.array(gap_s),np.array(gap_p),npri
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--cell');ap.add_argument('--seeds');ap.add_argument('--shuffle-blocks',action='store_true');ap.add_argument('--out');a=ap.parse_args()
    import e57_blockstream8_run as B,e55_run as R55,e57_syn_pass as SP
    seeds=[int(x) for x in a.seeds.split(',')];out={'cell':a.cell,'seeds':seeds,'per_seed':{}};agg=None
    for sd in seeds:
        if a.cell.startswith('blk'):B.patch(a.cell,sd if a.shuffle_blocks else None)
        enc,w,tf=R55.build(SP.synth_args(sd,a.cell),sd,arms=['NAT']);d=sentences(w);pr,sn,none,tot,gs,gp,npri=stats(d)
        row=dict(sentences=tot,no_prior_in_window=none,mean_prior_in_window=float(np.mean(npri)),
                 pair_hidden_share={W:pr[W][0]/pr[W][1] for W in BANDS},
                 sent_nearest_visible_share={W:sn[W]['near_vis']/sn[W]['n'] for W in BANDS},sent_mixed={W:sn[W]['mixed'] for W in BANDS},sent_all_hidden={W:sn[W]['all_hidden'] for W in BANDS},sent_with_prior={W:sn[W]['n'] for W in BANDS},
                 gap_start=dict(n=len(gs),mean=float(gs.mean()),q=[float(x) for x in np.quantile(gs,[0,.05,.25,.5,.75,.95,1])],lt1024=float((gs<1024).mean())),gap_place=dict(mean=float(gp.mean()),q=[float(x) for x in np.quantile(gp,[0,.05,.25,.5,.75,.95,1])],lt1024=float((gp<1024).mean())))
        out['per_seed'][sd]=row;print(sd,json.dumps({k:row[k] for k in ('sentences','no_prior_in_window','pair_hidden_share','sent_nearest_visible_share')}),flush=True)
    if a.out:json.dump(out,open(a.out,'w'),indent=1)
if __name__=='__main__':main()
