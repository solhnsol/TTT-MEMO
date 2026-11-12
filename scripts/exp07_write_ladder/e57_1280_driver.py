"""E57 job 1280 driver and manifest tool (CPU parts; GPU work is done by the runner and the p16 script it calls).
  seed_record(out, seed)     builds the INPUT_SCHEMA record of a seed from the runner's result file, validity record and p16 file: per-chunk margin arrays are aggregated as in A1 7.1 (mean over the facts of a chunk, then over the analysis-set chunks); half_pl_unmatched = 0 (the HALF / HALF-PL arms are not run by job 1280, the flag is descriptive in freeze A and is disclosed as not evaluated); the secondary margin goes to a separate file only.
  manifest(files, out)       writes the instrument manifest of A1 4.11 (b): file name, SHA-256 and last commit for every pinned file, plus the SHA-256 of the manifest itself (to be signed and timestamped by the tag push)
  python3 e57_1280_driver.py --manifest OUT.json | --record --out O --seed S | --selftest"""
import sys,json,hashlib,argparse,subprocess,pathlib
import numpy as np
HERE=pathlib.Path(__file__).resolve().parent;ROOT=HERE.parents[1];sys.path.insert(0,str(HERE))
import e57_1280_margin as MG,e57_1280_runner_records as RR
PINNED=['e57_1280_t3_sensitivity.py','e57_1280_runner_records.py','e57_1280_analyze.py','e57_1280_analyze_a1b.py','e57_1280_registration.py','e57_natmask_run.py','e57_oracle_masks.py','e57_oracle_run.py','e55_read.py','e55_gpu.py','e55_core.py','e57_1280_margin.py','e57_1280_validity.py','e57_1280_gainnp.py','e57_1280_recomp.py','e57_1280_runner.py','e57_1280_p16.py','e57_1280_driver.py','ladder.py','s3.py']
OTHER=['scripts/exp04_what_ttt_retains/gen_facts.py']
def sha(path):return hashlib.sha256(open(path,'rb').read()).hexdigest()
def manifest(out=None,repo=ROOT):
    rows=[]
    for f in PINNED:
        p=repo/'scripts/exp07_write_ladder'/f;rows.append(dict(file=f'scripts/exp07_write_ladder/{f}',sha256=sha(p),commit=subprocess.run(['git','log','-1','--format=%H','--',str(p)],capture_output=True,text=True,cwd=repo).stdout.strip(),clean=subprocess.run(['git','status','--short',str(p)],capture_output=True,text=True,cwd=repo).stdout.strip()==''))
    for f in OTHER:
        p=repo/f;rows.append(dict(file=f,sha256=sha(p),commit=subprocess.run(['git','log','-1','--format=%H','--',str(p)],capture_output=True,text=True,cwd=repo).stdout.strip(),clean=subprocess.run(['git','status','--short',str(p)],capture_output=True,text=True,cwd=repo).stdout.strip()==''))
    body=dict(manifest_version=1,files=rows,not_yet_present=[f for f in PINNED if not (repo/'scripts/exp07_write_ladder'/f).exists()])
    body['manifest_sha256']=hashlib.sha256(json.dumps(body,sort_keys=True).encode()).hexdigest()
    if out:json.dump(body,open(out,'w'),indent=1)
    return body
def seed_record(out,seed):
    out=pathlib.Path(out);R=json.load(open(out/f'results_s{seed}.json'));V=json.load(open(out/f'validity_s{seed}.json'));P=json.load(open(out/f'p16_s{seed}.json')) if (out/f'p16_s{seed}.json').exists() else None
    ks=sorted(int(k) for k in R['results']);res={k:R['results'][str(k)] for k in ks}
    agg=lambda arm,key='margin_primary':MG.aggregate([np.asarray(res[k][key][arm],float) for k in ks])
    nll={c:[np.asarray(res[k]['nll'][c],float) for k in ks] for c in ('NC','A','C','B','A_P','B_P','BAND')};tok={c:[np.arange(len(x)) for x in nll['NC']] for c in nll}
    gn=MG.aggregate([np.asarray(res[k]['gain_noprior_variant']['primary'],float)-np.asarray(res[k]['gain_noprior_variant']['primary_nats'],float) for k in ks])
    raw=dict(chunks=ks,nll=nll,tok=tok,pre_update=[True]*len(ks),margins=dict(NATS=agg('NATS'),NATMASK=agg('NATMASK'),RECOMP=agg('RECOMP'),RECOMP_PL=agg('RECOMP_PL')),gain_noprior=gn,
             flags=dict(t2_ok=V['t2_ok'],t3_ok=V['t3_ok'],p16_ok=(P['p16_ok'] if P else False),half_pl_unmatched=0,determinism_abs=0.0 if V['mode']=='deterministic' else 1e-4))
    rec=RR.build_record(raw)
    sec={arm:agg(arm,'margin_secondary') for arm in ('NATS','NATMASK','RECOMP','RECOMP_PL')};json.dump(dict(seed=seed,M_NATS_mw=sec['NATS'],M_MASK_mw=sec['NATMASK'],M_RECOMP_mw=sec['RECOMP'],M_RECOMP_PL_mw=sec['RECOMP_PL'],G_mw=sec['NATMASK']-sec['NATS'],gain_noprior_mw=MG.aggregate([np.asarray(res[k]['gain_noprior_variant']['secondary'],float)-np.asarray(res[k]['gain_noprior_variant']['secondary_nats'],float) for k in ks])),open(out/f'secondary_margin_s{seed}.json','w'))
    json.dump(rec,open(out/f'record_s{seed}.json','w'),default=float);return rec
def selftest():
    import tempfile
    m=manifest();assert len(m['manifest_sha256'])==64 and any(r['file'].endswith('e57_1280_runner.py') for r in m['files']) and 'not_yet_present' in m
    with tempfile.TemporaryDirectory() as td:
        rng=np.random.default_rng(0);ks=[10,12];res={}
        for k in ks:
            nf=190;base=rng.normal(2,.3,nf);res[str(k)]=dict(nll=dict(NC=base.tolist(),A=(base+.6).tolist(),C=base.tolist(),B=(base+.7).tolist(),A_P=base.tolist(),B_P=(base+.4).tolist(),BAND=(base+2.4).tolist()),margin_primary={a:(rng.normal(-3+g,.3,8)).tolist() for a,g in (('NATS',0),('NATMASK',1),('RECOMP',.8),('RECOMP_PL',.1))},margin_secondary={a:(rng.normal(-2+g,.3,8)).tolist() for a,g in (('NATS',0),('NATMASK',1),('RECOMP',.8),('RECOMP_PL',.1))},gain_noprior_variant=dict(primary=rng.normal(-2.5,.3,8).tolist(),primary_nats=rng.normal(-3,.3,8).tolist(),secondary=rng.normal(-2,.3,8).tolist(),secondary_nats=rng.normal(-2.5,.3,8).tolist()))
        json.dump(dict(seed=1,branches=ks,results=res),open(f'{td}/results_s1.json','w'));json.dump(dict(seed=1,t2_ok=True,t3_ok=True,p16_ok=True,mode='deterministic'),open(f'{td}/validity_s1.json','w'));json.dump(dict(seed=1,p16_ok=True),open(f'{td}/p16_s1.json','w'))
        rec=seed_record(td,1);import e57_1280_analyze as A
        assert A.validate_input(rec) and abs(rec['M_MASK']-rec['M_NATS']-1.0)<0.5 and rec['c_minus_nc_abs_max']==0.0 and rec['half_pl_unmatched']==0 and rec['p16_ok'] is True and rec['gain_noprior']>0
        o=A.evaluate([rec]*3,exact=False);assert 'tests' in o
    print('e57_1280_driver selftest ok')
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--manifest');ap.add_argument('--record',action='store_true');ap.add_argument('--out');ap.add_argument('--seed',type=int);ap.add_argument('--selftest',action='store_true');a=ap.parse_args()
    if a.selftest:selftest()
    elif a.manifest:print(json.dumps(manifest(a.manifest),indent=1)[:600])
    elif a.record:print(json.dumps(seed_record(a.out,a.seed),indent=1,default=float))
