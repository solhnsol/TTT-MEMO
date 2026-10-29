"""E57 post-write window gain split by selected / not selected query rows (CPU only; new file). For the saved window / window-less per-token NLL files of job 298 (E57_windowless_<LABEL>.npz: nll_win, nll_chunk, lm, tgt [280,1024]) and the exact p=.26 GAINQ512 row masks (qb_GAINQ512_<chunk>, packbits of 1024 rows) it computes, over the chunks 8..279 and per group of rows (SEL = banded GAINQ512 rows, NOSEL = eligible rows that are not banded, ALL = eligible rows; eligible = e57_closure_masks.rows_eligible(lm, sentence id)): mean NLL with the window, without the window, gain = without - with, and the ratio of the gain to the gain of W0 on the same rows.
  .venv/bin/python scripts/exp07_write_ladder/e57_gain_split.py --bundle B --states-dir D --masks M.npz --labels W0,NAT,... --out O.json"""
import sys,json,argparse,pathlib
import numpy as np
HERE=pathlib.Path(__file__).resolve().parent
for p in (HERE,HERE/'e57',HERE.parents[1]/'scripts/exp04_what_ttt_retains',HERE.parents[1]/'scripts/exp05_write_dynamics'):sys.path.insert(0,str(p))
import e57_closure_masks as CM
CH=1024
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--bundle');ap.add_argument('--states-dir');ap.add_argument('--masks');ap.add_argument('--labels');ap.add_argument('--out');ap.add_argument('--band',type=int,default=512);ap.add_argument('--tokenizer');a=ap.parse_args()
    import gen_world_ctl as G,e55_uv as UVm,e57_reads as RD
    tfile=a.tokenizer or str(HERE.parents[1]/'data/tokenizer/llama3/tokenizer.json');enc=G.load_enc(tfile)
    items=RD.load_items(pathlib.Path(a.bundle)/'e57_bank'/'bank_v1.json');ids,iv,hits=RD.article_intervals(a.bundle,enc,items);ids=np.asarray(ids);ends=UVm.sentence_ends(UVm.tokenizer(tfile),ids)
    M=np.load(a.masks,allow_pickle=True);labels=a.labels.split(',');Z={l:np.load(pathlib.Path(a.states_dir)/f'E57_windowless_{l}.npz') for l in labels}
    grp={}
    for k in range(8,280):
        q=UVm.chunk_sents(ends,k,CH);sid=np.full(CH,-1)
        for i,(s_,e_) in enumerate(q):sid[s_:e_]=i
        qb=CM.up(M[f'qb_GAINQ{a.band}_{k}'],CH);lm0=Z[labels[0]]['lm'][k];el=CM.rows_eligible(lm0,sid)&lm0
        for l in labels:
            z=Z[l];assert np.array_equal(z['lm'][k],lm0)
            for g,sel in (('SEL',el&qb),('NOSEL',el&~qb),('ALL',el)):
                d=grp.setdefault((l,g),dict(n=0,w=0.0,c=0.0,per=[]));w=z['nll_win'][k][sel].astype(np.float64);c=z['nll_chunk'][k][sel].astype(np.float64);d['n']+=int(sel.sum());d['w']+=w.sum();d['c']+=c.sum();d['per'].append((w.sum(),c.sum(),int(sel.sum())))
    rng=np.random.default_rng(0);out={}
    for g in ('SEL','NOSEL','ALL'):
        base=grp[(labels[0],g)];bp=np.array(base['per'])
        for l in labels:
            d=grp[(l,g)];p=np.array(d['per']);gain=(d['c']-d['w'])/d['n'];g0=(base['c']-base['w'])/base['n'];bs=[]
            for _ in range(500):
                i=rng.integers(0,len(p),len(p));bs.append(((p[i,1]-p[i,0]).sum()/p[i,2].sum())/((bp[i,1]-bp[i,0]).sum()/bp[i,2].sum()))
            out[f'{l}|{g}']=dict(rows=d['n'],window=d['w']/d['n'],no_window=d['c']/d['n'],gain=gain,gain_W0=g0,ratio=gain/g0,ratio_ci95=[float(np.percentile(bs,2.5)),float(np.percentile(bs,97.5))])
    json.dump(dict(chunks='8..279',band=a.band,groups='SEL = banded rows of GAINQ%d (p=.26), NOSEL = other eligible rows, ALL = eligible rows'%a.band,rows=out),open(a.out,'w'),indent=1)
    for k,v in out.items():print(f"{k:28s} rows {v['rows']:7d} win {v['window']:.4f} nowin {v['no_window']:.4f} gain {v['gain']:.4f} (W0 {v['gain_W0']:.4f}) ratio {v['ratio']:.3f} [{v['ratio_ci95'][0]:.3f},{v['ratio_ci95'][1]:.3f}]")
if __name__=='__main__':main()
