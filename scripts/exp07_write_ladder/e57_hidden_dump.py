"""E57 W0 prefix hidden-state dump of the synthetic worlds (new file; GPU forward only, W0 weights, no write; core-claim / GPM thread, Hans 11/07 '은닉 상태 먼저'): for every world (seed:cell) the output of the LAST prefix layer (the prefix outputs that the suffix consumes: rt.prefix(rt.base, seq) of the whole 128-chunk stream with the natural 8-chunk window, exactly the `prefs` of Ladder55.run_stream) for all tokens, bf16 saved losslessly as uint16 (view), [T, 2560]. Output <out>/hid_s<seed>.npz: h (uint16 bf16 bits), ids (int32), n_chunks, stream_sha, cell. The prefix layers are frozen in TTT-E2E, so the dump does not depend on the write state.
  python3 e57_hidden_dump.py --worlds 62:v3_16_1,71:v3_4_1,... --out O | --selftest"""
import sys,json,time,gc,argparse,pathlib
import numpy as np
HERE=pathlib.Path(__file__).resolve().parent;ROOT=HERE.parents[1]
for p in (HERE,ROOT/'scripts/exp04_what_ttt_retains'):sys.path.insert(0,str(p))
import e57_longcell
CH=1024
def selftest():
    import ml_dtypes
    x=np.asarray([1.0,-2.5,3.14159],dtype=ml_dtypes.bfloat16);u=x.view(np.uint16);assert (u.view(ml_dtypes.bfloat16)==x).all();print('e57_hidden_dump selftest ok')
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--worlds');ap.add_argument('--out');ap.add_argument('--selftest',action='store_true');a=ap.parse_args()
    if a.selftest:selftest();return
    import jax,e55_run as R55,e57_syn_pass as SP
    out=pathlib.Path(a.out);out.mkdir(parents=True,exist_ok=True);lg=lambda *x:print(*x,flush=True)
    LD,s3,C=R55.lab();rt=LD.Runtime();L=C.Ladder55(rt,save_dir=out/'tmp',canary=None)
    for x in a.worlds.split(','):
        sd,cell=x.split(':');sd=int(sd);t0=time.monotonic()
        enc,w,tf=R55.build(SP.synth_args(sd,cell),sd,arms=['NAT']);S=R55.streams(w,enc);ids,sha=S['A'];ids=np.asarray(ids);nch=len(ids)//CH
        with rt.mesh:
            seq=L._stream_seq(ids,nch);pref=rt.prefix(rt.base,seq);h=np.asarray(jax.device_get(pref))
        assert h.shape==(nch*CH,2560),h.shape
        u=h.view(np.uint16) if h.dtype.itemsize==2 else None
        if u is None:raise SystemExit(f'unexpected dtype {h.dtype}')
        np.savez(out/f'hid_s{sd}.npz',h=u,ids=ids.astype(np.int32),n_chunks=np.asarray(nch),stream_sha=np.asarray(sha),cell=np.asarray(cell),dtype=np.asarray(str(h.dtype)))
        lg(f'seed {sd} {cell}: {nch} chunks, hidden {h.shape} {h.dtype}, {u.nbytes/2**30:.2f} GiB, {time.monotonic()-t0:.0f}s');del h,u,pref,seq;gc.collect()
    lg('done')
if __name__=='__main__':main()
