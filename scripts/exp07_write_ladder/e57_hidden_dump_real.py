"""E57 W0 prefix hidden-state dump of the QA bundles and the news stream A (new file; GPU forward only, W0 weights, no write; GPM thread, Hans 11/07 '1~2단계 먼저'): last-prefix-layer output of the whole stream (rt.prefix(rt.base, seq), the same call as e57_hidden_dump.py) for the QA bundle streams g5/g8/g9/g10/g12 (e57_public_qa_stage6/e57_pqa6_group_ids.npz, the streams of e57_dwg_qa_pass.py) and news stream A (e57_t1t2v2_in/e57_stream_sentences.jsonl). bf16 bits as uint16 [T, 2560] -> <out>/hid_qa_g<gid>.npz / hid_news_A.npz with ids and n_chunks.
  python3 e57_hidden_dump_real.py --out O [--sources qa5,qa8,...,newsA] | --selftest"""
import sys,json,time,gc,argparse,pathlib
import numpy as np
HERE=pathlib.Path(__file__).resolve().parent;ROOT=HERE.parents[1];EXP=ROOT/'experiments/exp07_write_ladder'
for p in (HERE,ROOT/'scripts/exp04_what_ttt_retains'):sys.path.insert(0,str(p))
CH=1024
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out');ap.add_argument('--sources',default='qa5,qa8,qa9,qa10,qa12,newsA');ap.add_argument('--selftest',action='store_true');a=ap.parse_args()
    if a.selftest:
        import e57_pqa6_run as P6,e57_t1t2v2_run as V;print('e57_hidden_dump_real selftest ok');return
    import jax,e55_run as R
    import e57_pqa6_run as P6,e57_t1t2v2_run as V
    out=pathlib.Path(a.out);out.mkdir(parents=True,exist_ok=True);lg=lambda *x:print(*x,flush=True)
    LD,s3,C=R.lab();rt=LD.Runtime();L=C.Ladder55(rt,save_dir=out/'tmp',canary=None);meta,ids_z=P6.load_groups(EXP/'e57_public_qa_stage6')
    for s in a.sources.split(','):
        t0=time.monotonic()
        if s.startswith('qa'):gid=int(s[2:]);ids=np.asarray(ids_z[f'g{gid}']);name=f'hid_qa_g{gid}.npz'
        else:ids=np.asarray(V.stream_ids(str(EXP/'e57_t1t2v2_in/e57_stream_sentences.jsonl'),'A'));name='hid_news_A.npz'
        nch=len(ids)//CH
        with rt.mesh:pref=rt.prefix(rt.base,L._stream_seq(ids,nch));h=np.asarray(jax.device_get(pref))
        assert h.shape==(nch*CH,2560) and h.dtype.itemsize==2,(h.shape,h.dtype)
        np.savez(out/name,h=h.view(np.uint16),ids=ids[:nch*CH].astype(np.int32),n_chunks=np.asarray(nch),dtype=np.asarray(str(h.dtype)))
        lg(f'{s}: {nch} chunks, hidden {h.shape} {h.dtype}, {h.nbytes/2**30:.2f} GiB, {time.monotonic()-t0:.0f}s');del h,pref;gc.collect()
    lg('done')
if __name__=='__main__':main()
