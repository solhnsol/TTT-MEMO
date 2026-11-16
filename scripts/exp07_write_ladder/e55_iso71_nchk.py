"""E55 ISO on seeds 71..78, NATS-NORM ASSUMPTION CHECK (job 1302i; CPU; new file). Job 1302i runs ISO-STD on the seeds 71..78 of the 1283 / 1293 runs with the cumulative update norm matched to the NATS run of the same seed (1283, E55_k0c16_NATS_s<seed>.saved.json) instead of the NAT run of e55_run.py (tag v3a). That is justified only if the E57 NATS trajectory (e57_oracle_run.py, clipped SGD, engine SGDSEL, mask none) equals the E55 NAT trajectory on the same seed. The check reruns NATS with the code path of this job on seed 61 (tag nchk) and compares it with the stored E55 NAT run of seed 61 (E55_v3a_NAT_s61): every logged step value (update_norm, ce, gnorm, coef) of every chunk and every read value (out, exposures 16 / 32 / 64 / 128) EXACTLY equal. Result 'EQUAL' = the NATS norms are used; 'DIFFERENT' = the job runs the NAT arm (e55_run.py run, tag v3a) on seeds 71..78 and uses its norms (the difference, maximum absolute value per quantity, is reported).
  python3 e55_iso71_nchk.py --new D1 --ref D2 --seed 61 --json J | --selftest   (exit 0 = EQUAL, 3 = DIFFERENT; the job decides)"""
import sys,json,argparse,pathlib
import numpy as np
KEYS=('update_norm','ce','gnorm','coef')
def compare(new_saved,ref_saved,new_read,ref_read):
    a=json.load(open(new_saved));b=json.load(open(ref_saved));ra=json.load(open(new_read));rb=json.load(open(ref_read));res={}
    n=min(len(a['log']),len(b['log']));res['n_chunks']=(len(a['log']),len(b['log']))
    for k in KEYS:
        x=np.array([e[k] for e in a['log'][:n]],float);y=np.array([e[k] for e in b['log'][:n]],float);res[k]=float(np.max(np.abs(x-y))) if n else float('inf')
    d=0.0;same=True
    for e in ('16','32','64','128'):
        va=np.array(ra['read'][e]['out'],float);vb=np.array(rb['read'][e]['out'],float)
        if va.shape!=vb.shape:same=False;d=float('inf');break
        d=max(d,float(np.max(np.abs(va-vb))))
    res['read_out']=d;res['equal']=bool(res['n_chunks'][0]==res['n_chunks'][1] and all(res[k]==0.0 for k in KEYS) and d==0.0)
    return res
def selftest():
    import tempfile
    rng=np.random.default_rng(0)
    with tempfile.TemporaryDirectory() as td:
        td=pathlib.Path(td);log=[{k:float(rng.normal()) for k in KEYS} for _ in range(128)];rd={e:{'out':rng.normal(0,1,(20,16)).tolist()} for e in ('16','32','64','128')}
        for nm in ('new','ref'):
            json.dump({'log':log},open(td/f'{nm}.saved.json','w'));json.dump({'read':rd},open(td/f'{nm}.read.json','w'))
        assert compare(td/'new.saved.json',td/'ref.saved.json',td/'new.read.json',td/'ref.read.json')['equal']
        l2=[dict(e) for e in log];l2[5]['ce']+=1e-12;json.dump({'log':l2},open(td/'new.saved.json','w'));assert not compare(td/'new.saved.json',td/'ref.saved.json',td/'new.read.json',td/'ref.read.json')['equal']
    print('e55_iso71_nchk selftest ok')
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--new');ap.add_argument('--ref');ap.add_argument('--seed',type=int,default=61);ap.add_argument('--json');ap.add_argument('--selftest',action='store_true');a=ap.parse_args()
    if a.selftest:selftest();return
    s=a.seed;r=compare(pathlib.Path(a.new)/f'E55_nchk_NATS_s{s}.saved.json',pathlib.Path(a.ref)/f'E55_v3a_NAT_s{s}.saved.json',pathlib.Path(a.new)/f'E55_nchk_NATS_s{s}.read.json',pathlib.Path(a.ref)/f'E55_v3a_NAT_s{s}.read.json');print(json.dumps(r))
    if a.json:pathlib.Path(a.json).write_text(json.dumps(r,indent=1))
    sys.exit(0 if r['equal'] else 3)
if __name__=='__main__':main()
