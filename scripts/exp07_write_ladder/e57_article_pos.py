"""E57 band readings restricted to rows whose band lies inside the row's own article (CPU only; new file; numbers only; no GPU, existing readings are re-aggregated). For every row i of the chunks 8..279 (query token at stream position q = chunk*1024 + i) p = q - start_of_its_article + 1 (RD.article_intervals; the number of tokens of the article up to and including the query token), so a band window of B tokens lies completely inside the article iff p >= B. Per band B in (128, 256, 512, 1024):
 (1) W0 mean window NLL, band-B NLL and band gain (band - window) over all eligible rows and over the eligible rows with p >= B (rows, share);
 (2) for every state of job 345b with its band B (W0@512 control, NAT@512, NAT@1024, NATADAMH@512, GAINQ512@512, RANDQ512 p.10@512, GAINQ256@256, RANDQ256@256, GAINQ128@128): band gain of the state, ratio = state band gain / W0 band gain on the same rows, r = 1 - ratio, for all rows and for p >= B rows, and for the selected / not selected rows (GAINQ<B> p=.26 mask) among them;
 (3) the number and the share of selected rows (mask qb_GAINQ<B>) among all eligible rows and among p >= B rows.
Also saves npz with p for every row of the chunks 8..279 ([272,1024], int16) and the article id.
  .venv/bin/python scripts/exp07_write_ladder/e57_article_pos.py --bundle B --out O.json --npz P.npz"""
import sys,json,argparse,pathlib
import numpy as np
HERE=pathlib.Path(__file__).resolve().parent
for p in (HERE,HERE/'e57',HERE.parents[1]/'scripts/exp04_what_ttt_retains',HERE.parents[1]/'scripts/exp05_write_dynamics'):sys.path.insert(0,str(p))
import e57_closure_masks as CM
CH=1024
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--bundle');ap.add_argument('--out');ap.add_argument('--npz');ap.add_argument('--exp',default=str(HERE.parents[1]/'experiments/exp07_write_ladder'));ap.add_argument('--tokenizer');a=ap.parse_args()
    import gen_world_ctl as G,e55_uv as UVm,e57_reads as RD
    E=pathlib.Path(a.exp);tfile=a.tokenizer or str(HERE.parents[1]/'data/tokenizer/llama3/tokenizer.json');enc=G.load_enc(tfile)
    items=RD.load_items(pathlib.Path(a.bundle)/'e57_bank'/'bank_v1.json');ids,iv,hits=RD.article_intervals(a.bundle,enc,items);ids=np.asarray(ids);ends=UVm.sentence_ends(UVm.tokenizer(tfile),ids)
    starts=np.array([x[0] for x in iv]);q=np.arange(8*CH,280*CH);ai=np.searchsorted(starts,q,side='right')-1;pos=(q-starts[ai]+1).reshape(272,CH)
    np.savez_compressed(a.npz,p=pos.astype(np.int16),article=ai.reshape(272,CH).astype(np.int16),chunks=np.arange(8,280))
    W=np.load(E/'e57_band/e57_band_W0.npz');W2=np.load(E/'e57_band256/E57_closure_W0.npz');w0={int(b):W['nll_band'][i] for i,b in enumerate(W['bands'])};w0[256]=W2['nll_band'][list(W2['bands']).index(256)];nw0=W['nll_win'];lm=W['lm']
    M={}
    for f in (E/'e57_gainq_p0.26/e57_qband_masks.npz',E/'e57_gainq256_p0.26/e57_qband_masks.npz'):M.update({k:v for k,v in np.load(f,allow_pickle=True).items() if k.startswith('qb_')})
    el=np.zeros((272,CH),bool)
    for j,k in enumerate(range(8,280)):
        sid=np.full(CH,-1)
        for i,(s_,e_) in enumerate(UVm.chunk_sents(ends,k,CH)):sid[s_:e_]=i
        el[j]=CM.rows_eligible(lm[k],sid)&lm[k]
    sel={b:np.stack([CM.up(M[f'qb_GAINQ{b}_{k}'],CH) for k in range(8,280)]) for b in (128,256,512,1024)}
    out={'W0':{},'states':{},'selection':{}}
    def mean(x,m):return float(x[m].mean()) if m.any() else None
    for b in (128,256,512,1024):
        inb=pos>=b;sub=el&inb;bw=w0[b][8:280];ww=nw0[8:280]
        out['W0'][f'B{b}']={g:dict(rows=int(m.sum()),row_share=float(m.sum()/el.sum()),window=mean(ww,m),band=mean(bw,m),band_gain=mean(bw-ww,m)) for g,m in (('all_eligible',el),('p_ge_B',sub),('p_lt_B',el&~inb))}
        out['selection'][f'B{b}']=dict(selected_all=int((sel[b]&el).sum()),selected_share_all=float((sel[b]&el).sum()/el.sum()),selected_p_ge_B=int((sel[b]&sub).sum()),selected_share_among_p_ge_B=float((sel[b]&sub).sum()/max(sub.sum(),1)),selected_p_lt_B=int((sel[b]&el&~inb).sum()),rows_p_ge_B=int(sub.sum()),selected_among_p_lt_B_share=float((sel[b]&el&~inb).sum()/max((el&~inb).sum(),1)))
    SPEC=[('W0',512),('NAT',512),('NAT',1024),('NATADAMH',512),('GAINQ512_p026',512),('RANDQ512_p010',512),('GAINQ256_p026',256),('RANDQ256_p026',256),('GAINQ128_p026',128)]
    for lab,b in SPEC:
        z=np.load(E/f'e57_band_states/E57_band_{lab}.npz');nb=z['nll_band'][list(z['bands']).index(b)][8:280];nw=z['nll_win'][8:280];bw=w0[b][8:280];ww=nw0[8:280];inb=pos>=b;res={}
        for gn,m in (('all_eligible',el),('p_ge_B',el&inb),('p_ge_B_selected',el&inb&sel[b]),('p_ge_B_not_selected',el&inb&~sel[b]),('all_selected',el&sel[b]),('all_not_selected',el&~sel[b])):
            g=mean(nb-nw,m);g0=mean(bw-ww,m);res[gn]=dict(rows=int(m.sum()),band_gain=g,band_gain_W0=g0,ratio=(g/g0 if g0 else None),r=(1-g/g0 if g0 else None),window=mean(nw,m),band=mean(nb,m))
        out['states'][f'{lab}@{b}']=res
    json.dump(dict(definition='p = q - start of the row\'s article + 1 (q = query token position); band B inside the article iff p >= B; band gain = band-B NLL - window NLL; ratio = state band gain / W0 band gain on the same rows; r = 1 - ratio',chunks='8..279',eligible_rows=int(el.sum()),**out),open(a.out,'w'),indent=1)
    F=lambda x,fm='+.3f':('n/a' if x is None else format(x,fm))
    for b in (128,256,512,1024):
        o=out['W0'][f'B{b}'];s=out['selection'][f'B{b}'];print(f"B={b}: rows p>=B {o['p_ge_B']['rows']} ({o['p_ge_B']['row_share']:.3f}); W0 band gain all {F(o['all_eligible']['band_gain'])} | p>=B {F(o['p_ge_B']['band_gain'])} | p<B {F(o['p_lt_B']['band_gain'])}; selected {s['selected_all']} ({s['selected_share_all']:.3f}), among p>=B {s['selected_p_ge_B']} ({s['selected_share_among_p_ge_B']:.3f})")
    for k,v in out['states'].items():print(f"{k:20s} ratio all {F(v['all_eligible']['ratio'],'.3f')} | p>=B {F(v['p_ge_B']['ratio'],'.3f')} (r {F(v['p_ge_B']['r'])}) rows p>=B {v['p_ge_B']['rows']}")
if __name__=='__main__':main()
