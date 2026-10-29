"""E57 band-gain share held by the query-selected rows (CPU only; new file; numbers only). For W0 per-token NLL of the band baselines (e57_band_W0.npz: nll_band for 128/512/1024, nll_win; E57_closure_W0.npz of job 301 for 256) and the exact selection rule of the GAINQ masks (e57_qband_masks.arm_rows: top round(p x eligible rows) rows of each chunk 8..279 by band gain nll_band_k - nll_win among the eligible rows), per band k and p in (.10 .26 .50): the share of the total band gain (sum of nll_band_k - nll_win over the eligible rows of the chunks 8..279) held by the selected rows, and the remaining share 1 - that on the not selected rows (rows banded in the write arms have no leak of the query-side window; the remaining share is the band gain that the arms leave untouched). Also the same shares for the positive part of the gain only (sum of max(gain,0)), because the band loss is lower than the window loss on many rows (negative gain), which pushes the share held by the selected rows above 1. Also the mean band gain per row of the selected / not selected rows and the share of rows.
  .venv/bin/python scripts/exp07_write_ladder/e57_band_leak_table.py --bundle B --band-npz e57_band_W0.npz --band256 E57_closure_W0.npz --out O.json"""
import sys,json,argparse,pathlib
import numpy as np
HERE=pathlib.Path(__file__).resolve().parent
for p in (HERE,HERE/'e57',HERE.parents[1]/'scripts/exp04_what_ttt_retains',HERE.parents[1]/'scripts/exp05_write_dynamics'):sys.path.insert(0,str(p))
import e57_closure_masks as CM
CH=1024
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--bundle');ap.add_argument('--band-npz');ap.add_argument('--band256');ap.add_argument('--out');ap.add_argument('--tokenizer');a=ap.parse_args()
    import gen_world_ctl as G,e55_uv as UVm,e57_reads as RD
    tfile=a.tokenizer or str(HERE.parents[1]/'data/tokenizer/llama3/tokenizer.json');enc=G.load_enc(tfile)
    items=RD.load_items(pathlib.Path(a.bundle)/'e57_bank'/'bank_v1.json');ids,iv,hits=RD.article_intervals(a.bundle,enc,items);ids=np.asarray(ids);ends=UVm.sentence_ends(UVm.tokenizer(tfile),ids)
    B=np.load(a.band_npz);B2=np.load(a.band256);bands={int(b):B['nll_band'][i] for i,b in enumerate(B['bands'])};bands[256]=B2['nll_band'][list(B2['bands']).index(256)]
    nw=B['nll_win'];lm=B['lm'];out={}
    sids={}
    for k in range(8,280):
        q=UVm.chunk_sents(ends,k,CH);sid=np.full(CH,-1)
        for i,(s_,e_) in enumerate(q):sid[s_:e_]=i
        sids[k]=sid
    for band in (128,256,512,1024):
        for p in (.10,.26,.50):
            tot=0.0;selg=0.0;nsel=0;nel=0;tp=0.0;sp=0.0
            for k in range(8,280):
                el=CM.rows_eligible(lm[k],sids[k])&lm[k];g=(bands[band][k]-nw[k]).astype(np.float64);n=int(round(p*el.sum()));m=CM.top_rows(g,el,n)
                tot+=g[el].sum();selg+=g[m].sum();nsel+=int(m.sum());nel+=int(el.sum());tp+=np.maximum(g[el],0).sum();sp+=np.maximum(g[m],0).sum()
            out[f'k{band}_p{p}']=dict(band=band,p=p,rows=nel,selected_rows=nsel,total_band_gain_sum=tot,selected_band_gain_sum=selg,share_held_by_selected=selg/tot,share_remaining_on_unselected=1-selg/tot,mean_gain_selected=selg/nsel,mean_gain_unselected=(tot-selg)/(nel-nsel),positive_gain_share_held_by_selected=sp/tp,positive_gain_share_remaining=1-sp/tp)
    json.dump(dict(chunks='8..279',rule='top round(p x eligible rows) per chunk by nll_band_k - nll_win (e57_closure_masks.top_rows), W0',rows=out),open(a.out,'w'),indent=1)
    for k,v in out.items():print(f"{k:12s} sel rows {v['selected_rows']:7d}/{v['rows']} held {v['share_held_by_selected']:.3f} remaining {v['share_remaining_on_unselected']:.3f} mean gain sel {v['mean_gain_selected']:.3f} unsel {v['mean_gain_unselected']:.3f} positive-part remaining {v['positive_gain_share_remaining']:.3f}")
if __name__=='__main__':main()
