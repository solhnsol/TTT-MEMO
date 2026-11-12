"""E57 job 1280 p16 check (A1 v12 section 5; GPU for the trajectory part, CPU for the hidden-key counts). For a seed, the first 20 chunks of the p = 16 cell v3_32_16_a16 (second exposure of a fact at chunk 16 or later, outside the window of 8 chunks) are run with NATS and the two cached update-path arms NATORACLE and NATPLACEBO by e57_oracle_run (unchanged); the recompute arms RECOMP and RECOMP-PL hide the same key sets as the cached arms (S_i and the placebo spans), so their hidden-key counts are taken from the same masks, and when no key is hidden the recompute oracle uses the cached natural K,V (nothing to recompute), so its update equals the cached arm's. p16_ok = hidden_pairs = 0 for the oracle and placebo masks at every chunk (CPU count below) AND the cached ORACLE and PLACEBO trajectories equal NATS in update_norm, ce, gnorm, coef of every chunk and in the final state hash (section 2: bit identical, or the fallback tolerances).
  python3 e57_1280_p16.py --seed S --out O [--fallback]  (runs the three arms through e57_oracle_run, then checks) | --check-only --seed S --out O | --selftest"""
import sys,os,json,argparse,subprocess,pathlib
import numpy as np
HERE=pathlib.Path(__file__).resolve().parent;ROOT=HERE.parents[1]
for p in (HERE,HERE/'e57',ROOT/'scripts/exp04_what_ttt_retains',ROOT/'scripts/exp05_write_dynamics'):sys.path.insert(0,str(p))
import e57_1280_validity as VA
CELL='v3_32_16_a16';NCH=20;CH=1024;NW=8;WIN=NW*CH
def hidden_counts(seed):
    """number of hidden (row, key) pairs of the cached oracle and placebo masks at chunks 0..NCH-1 of the p = 16 cell, from the sentence table of the world (CPU); both must be 0"""
    import e57_oracle_run as OR,e55_run as R55,e57_syn_pass as SP,e57_oracle_masks as OM
    OR.register_cell(CELL);a=SP.synth_args(seed,CELL);enc,w,tf=R55.build(a,seed,read_apps={16},arms=['NAT']);ch=w['chunks'];base=np.ones(WIN+CH,bool)
    uvO=OM.OracleUV(ch,0,lambda *x:None);uvP=OM.PlaceboOracleUV(ch,0,lambda *x:None);cnt={'ORACLE':0,'PLACEBO':0}
    for k in range(NCH):
        kp,_=uvO.keep(k,None,None,None,None,base);cnt['ORACLE']+=int((~np.asarray(kp,bool)).sum())
        kq,_=uvP.keep(k,None,None,None,None,base);cnt['PLACEBO']+=int((~np.asarray(kq,bool)).sum())
    return cnt
def compare_logs(out,tag,seed,fallback=False):
    out=pathlib.Path(out);ref=json.load(open(out/f'E55_{tag}_NATS_s{seed}.saved.json'));res={}
    for arm in ('NATORACLE','NATPLACEBO'):
        d=json.load(open(out/f'E55_{tag}_{arm}_s{seed}.saved.json'));ok=len(d['log'])==len(ref['log'])
        for x,y in zip(d['log'],ref['log']):
            ok=ok and VA.rel_equal(x['update_norm'],y['update_norm'],fallback) and VA.abs_equal(x['ce'],y['ce'],fallback) and VA.rel_equal(x['gnorm'],y['gnorm'],fallback) and VA.abs_equal(x['coef'],y['coef'],fallback)
        ok=ok and (d.get('final_hash')==ref.get('final_hash') if not fallback else True);res[arm]=bool(ok)
    return res
def p16_ok(seed,out,tag,fallback=False):
    hc=hidden_counts(seed);cl=compare_logs(out,tag,seed,fallback);ok=bool(hc['ORACLE']==0 and hc['PLACEBO']==0 and all(cl.values()));return ok,dict(hidden_pairs=hc,logs_equal_to_NATS=cl,mode='fallback' if fallback else 'deterministic')
def run_arms(seed,out,tag):
    py=sys.executable
    subprocess.run([py,str(HERE/'e57_oracle_run.py'),'--seed',str(seed),'--cell',CELL,'--out',str(out),'--masks','none','--mask-arm','ALLQ256','--arms','NATS,NATORACLE,NATPLACEBO','--smoke',str(NCH),'--tag',tag],check=True)
def selftest():
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        def mk(arm,eq=True):
            log=[dict(update_norm=0.5+i,ce=2.0,gnorm=0.7,coef=1.0) for i in range(3)]
            if not eq:log[1]['ce']=2.00005
            json.dump(dict(log=log,final_hash='h'),open(f'{td}/E55_t_{arm}_s1.saved.json','w'))
        mk('NATS');mk('NATORACLE');mk('NATPLACEBO');assert compare_logs(td,'t',1)=={'NATORACLE':True,'NATPLACEBO':True}
        mk('NATORACLE',False);r=compare_logs(td,'t',1);assert r['NATORACLE'] is False and r['NATPLACEBO'] is True and compare_logs(td,'t',1,True)['NATORACLE'] is True   # 5e-5 on ce passes the fallback only
    print('e57_1280_p16 selftest ok (comparison logic; the hidden-key count needs the generator, the trajectories need the GPU)')
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--seed',type=int);ap.add_argument('--out');ap.add_argument('--tag',default='p16');ap.add_argument('--fallback',action='store_true');ap.add_argument('--check-only',action='store_true');ap.add_argument('--selftest',action='store_true');a=ap.parse_args()
    if a.selftest:selftest()
    else:
        if not a.check_only:run_arms(a.seed,a.out,a.tag)
        ok,info=p16_ok(a.seed,a.out,a.tag,a.fallback);json.dump(dict(seed=a.seed,p16_ok=ok,**info),open(pathlib.Path(a.out)/f'p16_s{a.seed}.json','w'));print('p16_ok',ok,info)
