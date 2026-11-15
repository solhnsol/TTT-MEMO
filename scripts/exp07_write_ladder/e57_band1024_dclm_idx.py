"""E57 job 1297, DCLM preservation analysis for the 9 fresh streams 16..24 (new file, repair of a stream-index bug of the job script, no registered file edited). e57_preservation_analyze.g2 reads the streams st0..st(n-1); the job's streams are st16..st24, so the job script's call (g2 with the fixed n = 16 from index 0) stopped with FileNotFoundError after the raw runs of W0 / NATS / NATMASK_256 had been written. This wrapper links the files st16..st24 to st0..st8 in a scratch directory and calls the UNCHANGED g2(D, 9, alpha) of e57_preservation_analyze.py: same statistics, same code. Output {'G2': g2 result} (the format that e57_band1024_confirm_analyze.py --dclm-json reads: G2.D_N.per_stream).
  python e57_band1024_dclm_idx.py --dclm DC --band 256|1024 --json J"""
import sys,os,json,argparse,pathlib
HERE=pathlib.Path(__file__).resolve().parent;sys.path.insert(0,str(HERE))
import e57_preservation_analyze as PA
S0=16;N=9
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--dclm');ap.add_argument('--band');ap.add_argument('--json');a=ap.parse_args();dc=pathlib.Path(a.dclm).resolve();tmp=dc/f'idx_w{a.band}';tmp.mkdir(exist_ok=True)
    for arm,src in (('W0',dc),('NATS',dc),('NATMASK',dc/f'w{a.band}')):
        for i in range(N):
            f=src/f'E57_dclmttt_{arm}_st{S0+i}.json';assert f.exists(),f;l=tmp/f'E57_dclmttt_{arm}_st{i}.json'
            if l.is_symlink() or l.exists():l.unlink()
            l.symlink_to(f)
    r=PA.g2(str(tmp),N,0.05);json.dump({'G2':r},open(a.json,'w'),indent=1,default=float);d=r['D_N'];print('band',a.band,'streams',r['n_streams'],'D_N mean',d['mean'],'t upper',d['t']['upper'],'label',d['label'],'D_0 label',r['D_0']['label'])
if __name__=='__main__':main()
