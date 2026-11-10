"""E57 direct-access oracle pilot/confirmation analysis (CPU; new file; written and committed BEFORE any result of jobs 1279ff is opened; core-claims thread 11/10 registration draft). Inputs: <dir>/E55_<tag>_<arm>_s<seed>.{saved,read}.json of e57_oracle_run.py (tags orcp4 = cell v3_32_4_a16, orcp16 = cell v3_32_16_a16).
PRIMARY metric (per fact, read at exposure 16, out of window, 16 candidates of equal token length): margin = log-prob(gold) - max log-prob(wrong) in nat (candidate sums; equal candidate length, so sum and length-normalised rankings coincide); secondary: accuracy (argmax = gold), log-softmax margin = log-softmax(gold) - logsumexp(wrong). Unit of inference = SEED (mean over the facts of a seed), paired by seed across arms; t intervals (df = n_seeds - 1). Pilot (3 seeds) = noise / code check, no confirmatory inference.
p=4 contrasts (all margins): ORACLE-NATS (first claim), ORACLE-RAND (second claim, gatekept: tested only if the first holds), ORACLE-PLACEBO, 2x2 factorial effects: same-fact main = 1/2[(ORACLE-NATS)+(MASK-HALF)], general = 1/2[(HALF-NATS)+(MASK-ORACLE)], MASK-HALF (direction corrected: positive = re-opening the same-fact exposures lowers the gain), MASK-NATS at p=4 and at p=16 (Holm over these two). NO p=4 vs p=16 contrast.
Gates (reported before any contrast): G2 exact bit identity: NATBINF == NATS at p=4 (update_norm, ce, gnorm, coef of every chunk, final state hash) and at p=16 seed 62 ORACLE == NATS and HALF == NATMASK; G3 read hashes before == after in every read; G1 manipulation check from token NLL logs: oracle later-exposure fact-token NLL on the update path minus the natural-context NLL, vs the noise tau = 3 x |NATS - NATBINF| fluctuation of the same quantity (if both are bit identical tau = 1e-6); G1 is information only, never an exclusion; RAND hit rate; clip statistics (pre-clip gnorm mean, clip rate) and the fact-gradient share per arm.
  python3 e57_oracle_analyze.py --dir O --seeds 62,63,64 [--json J] | --selftest"""
import sys,json,argparse,pathlib
import numpy as np
from scipy import stats
P4=['NATS','NATBINF','NATMASK','NATORACLE','NATHALF','NATORACLE_RAND','NATPLACEBO'];P16=['NATS','NATMASK','NATORACLE_RAND','NATPLACEBO']
def lse(x):x=np.asarray(x,float);m=x.max();return m+np.log(np.exp(x-m).sum())
def facts(path,app='16'):
    d=json.load(open(path));g=np.array(d['meta']['gold_idx']);fact=np.array([k=='fact' for k in d['meta']['kind']]);L=np.array(d['read'][app]['out'],float)[fact];g=g[fact]
    gold=L[np.arange(len(g)),g];wrong=L.copy();wrong[np.arange(len(g)),g]=-np.inf
    return dict(margin=gold-wrong.max(1),acc=(L.argmax(1)==g).astype(float),lsm=(gold-lse(L.reshape(-1)) if False else np.array([L[i,g[i]]-lse(np.delete(L[i],g[i])) for i in range(len(g))])))
def seedmeans(D,tag,arm,seeds,key):return np.array([facts(D/f'E55_{tag}_{arm}_s{s}.read.json')[key].mean() for s in seeds])
def tci(d,level=0.95):
    d=np.asarray(d,float);n=len(d);m=d.mean();sd=d.std(ddof=1) if n>1 else float('nan');se=sd/np.sqrt(n);t=stats.t.ppf(0.5+level/2,n-1) if n>1 else float('nan')
    return dict(mean=float(m),ci95=[float(m-t*se),float(m+t*se)],sd=float(sd),p=float(2*stats.t.sf(abs(m/se),n-1)) if se>0 else float('nan'),n=int(n),signflip_p_one_sided=signflip(d)[0],signflip_p_two_sided=signflip(d)[1])
def signflip(d):
    """exact sign-flip permutation test of the mean of the per-seed paired differences d (Council 11/10: exact, n >= 5 one-sided (min p = 1/32), n >= 6 two-sided (min p = 2/64)); returns (p_one_sided for mean > 0, p_two_sided)"""
    d=np.asarray(d,float);n=len(d);obs=d.mean();S=np.array([[1 if (i>>j)&1 else -1 for j in range(n)] for i in range(2**n)]);m=(S*d[None]).mean(1)
    return float((m>=obs-1e-15).mean()),float((np.abs(m)>=abs(obs)-1e-15).mean())
def seeds_needed(sd,sesoi,alpha=0.05,power=0.8,floor=8):
    """number of seeds for a paired design from a fixed SESOI (smallest effect size of interest, same units as sd): n = ((z_{1-alpha} + z_power) sd / sesoi)^2 (one-sided), at least `floor` (Council: 8 to 12)"""
    return int(max(floor,np.ceil(((stats.norm.ppf(1-alpha)+stats.norm.ppf(power))*sd/sesoi)**2)))
def holm(p):
    k=sorted(p,key=lambda z:p[z]);m=len(k);adj={};run=0.0
    for i,z in enumerate(k):run=max(run,min(1.0,(m-i)*p[z]));adj[z]=run
    return adj
def logs(path):
    r=json.load(open(path));return r
def same(a,b,keys=('update_norm','ce','gnorm','coef')):return all(x[k]==y[k] for x,y in zip(a['log'],b['log']) for k in keys) and len(a['log'])==len(b['log']) and a.get('final_hash')==b.get('final_hash')
def analyze(D,seeds,tag4='orcp4',tag16='orcp16',gate_seed=62):
    D=pathlib.Path(D);out={'seeds':list(seeds),'n_seeds':len(seeds),'gates':{},'p4':{},'p16':{}}
    g2={}
    for s in seeds:g2[f'p4_NATBINF=NATS_s{s}']=bool(same(logs(D/f'E55_{tag4}_NATBINF_s{s}.saved.json'),logs(D/f'E55_{tag4}_NATS_s{s}.saved.json')))
    f=lambda a,b:D/f'E55_{tag16}_{a}_s{gate_seed}.saved.json'
    if f('NATORACLE','').exists() if False else (D/f'E55_{tag16}_NATORACLE_s{gate_seed}.saved.json').exists():
        g2[f'p16_ORACLE=NATS_s{gate_seed}']=bool(same(logs(D/f'E55_{tag16}_NATORACLE_s{gate_seed}.saved.json'),logs(D/f'E55_{tag16}_NATS_s{gate_seed}.saved.json')))
        g2[f'p16_HALF=NATMASK_s{gate_seed}']=bool(same(logs(D/f'E55_{tag16}_NATHALF_s{gate_seed}.saved.json'),logs(D/f'E55_{tag16}_NATMASK_s{gate_seed}.saved.json')))
    k0s={}
    for tag,arms in ((tag4,P4),(tag16,P16)):
        for arm in arms:
            for s in seeds:
                pth=D/f'E55_{tag}_{arm}_s{s}.saved.json'
                if pth.exists():k0s.setdefault((tag,arm),set()).add(logs(pth).get('K0'))
    out['K0_by_arm']={f'{t}:{a_}':sorted(v,key=lambda z:(z is None,z)) for (t,a_),v in k0s.items()};base_k0={k for (t,a_),v in k0s.items() if a_!='NATMASK_K08' for k in v};assert len(base_k0)==1,('K0 differs between arms: never pool across K0',base_k0)
    out['gates']['G2_exact_bit_identity']=g2;out['gates']['G2_pass']=bool(all(g2.values()))
    g3={};
    for tag,arms in ((tag4,P4),(tag16,P16)):
        for arm in arms:
            for s in seeds:
                pth=D/f'E55_{tag}_{arm}_s{s}.saved.json'
                if pth.exists():
                    for e in logs(pth).get('read_hash',[]):g3[f'{tag}_{arm}_s{s}_k{e[0]}']=bool(e[1]==e[2])
    out['gates']['G3_read_hash_equal']=bool(all(g3.values())) if g3 else None
    out['gates']['G1_exact_bit_identity']=bool(all(v for k,v in g2.items() if k.startswith('p4_NATBINF=NATS')))      # G1 = NATBINF == NATS on EVERY seed (hard gate, no tolerance tau); a failure = bug, rerun
    def tokstat(tag,arm,s,field):
        d=logs(D/f'E55_{tag}_{arm}_s{s}.saved.json').get('diag',{});v=[x[field] for x in d.values() if x.get(field) is not None];return float(np.mean(v)) if v else None
    g1={}
    for s in seeds:
        ex=lambda arm:(tokstat(tag4,arm,s,'upd_nll_fact_later')-tokstat(tag4,arm,s,'nat_nll_fact_later')) if tokstat(tag4,arm,s,'upd_nll_fact_later') is not None else None
        g1[s]=dict(oracle=ex('NATORACLE'),placebo=ex('NATPLACEBO'),rand=ex('NATORACLE_RAND'),nats=ex('NATS'))
    out['gates']['token_loss_info_only']=dict(per_seed=g1)
    hit=[];cl={}
    for arm in P4:
        pth=[D/f'E55_{tag4}_{arm}_s{s}.saved.json' for s in seeds];R=[logs(x) for x in pth]
        cl[arm]={}
        for nm,sel in (('before_window_fills_chunk_lt8',lambda e:e['k']<8),('after_window_fills_chunk_ge8',lambda e:e['k']>=8)):
            es=[e for r in R for e in r['log'] if sel(e)];cl[arm][nm]=dict(gnorm_pre_clip_mean=float(np.mean([e['gnorm'] for e in es])),clip_rate=float(np.mean([e['coef']<1.0 for e in es])))
        for nm,sel in (('before',lambda k:int(k)<8),('after',lambda k:int(k)>=8)):
            sh=[x['fact_grad_share'] for r in R for k,x in r.get('diag',{}).items() if sel(k) and x.get('fact_grad_share') is not None];cl[arm]['fact_grad_share_'+nm]=float(np.mean(sh)) if sh else None
        if arm=='NATORACLE_RAND':hit=[e['rand_hit_rate'] for r in R for e in r['log'] if 'rand_hit_rate' in e]
    out['p4']['clip_and_grad']=cl;out['p4']['rand_hit_rate_mean']=float(np.mean(hit)) if hit else None
    for key in ('margin','acc','lsm'):
        M={a:seedmeans(D,tag4,a,seeds,key) for a in P4};c={}
        c['ORACLE-NATS']=M['NATORACLE']-M['NATS'];c['ORACLE-RAND']=M['NATORACLE']-M['NATORACLE_RAND'];c['ORACLE-PLACEBO']=M['NATORACLE']-M['NATPLACEBO']
        c['MASK-HALF']=M['NATMASK']-M['NATHALF'];c['same_fact_main']=0.5*((M['NATORACLE']-M['NATS'])+(M['NATMASK']-M['NATHALF']));c['general_main']=0.5*((M['NATHALF']-M['NATS'])+(M['NATMASK']-M['NATORACLE']));c['MASK-NATS_p4']=M['NATMASK']-M['NATS']
        out['p4'][key]={k:tci(v) for k,v in c.items()};out['p4'][key]['arm_means']={a:float(v.mean()) for a,v in M.items()}
        if key=='margin':
            M16={a:seedmeans(D,tag16,a,seeds,key) for a in P16};d16=M16['NATMASK']-M16['NATS'];out['p16'][key]=dict(arm_means={a:float(v.mean()) for a,v in M16.items()},**{'MASK-NATS_p16':tci(d16)})
            ps={'MASK-NATS_p4':out['p4']['margin']['MASK-NATS_p4']['p'],'MASK-NATS_p16':out['p16']['margin']['MASK-NATS_p16']['p']};ad=holm(ps);out['holm_MASK_minus_NATS']=ad
            n_=len(seeds);alpha=0.05;seq=['ORACLE-NATS','ORACLE-RAND','ORACLE-PLACEBO'];res={};ok=True        # fixed-sequence testing: each step is tested at alpha only if the previous step passed
            for c_ in seq:
                if n_<5:res[c_]=dict(tested=False,reason='n < 5: exact one-sided sign-flip has min p = 1/2^n > .05 at n < 5');ok=False;continue
                if not ok:res[c_]=dict(tested=False,reason='previous step in the fixed sequence failed');continue
                pv=out['p4']['margin'][c_]['signflip_p_one_sided'];res[c_]=dict(tested=True,p_one_sided_exact=pv,passed=bool(pv<alpha));ok=bool(pv<alpha)
            out['fixed_sequence']=res;out['interaction_note']='2x2 factorial effects (same_fact_main, general_main, MASK-HALF) are descriptive only; MASK-NATS p4 / p16 are a Holm family; no p4 vs p16 contrast';out['seeds_needed_from_SESOI']='seeds_needed(sd_of_paired_diff, SESOI) with SESOI fixed in the registration (floor 8)'
    return out
def selftest():
    import tempfile;rng=np.random.default_rng(0)
    def mk(td,tag,arm,s,gain,same_as=None,hash_ok=True):
        g=[int(x) for x in rng.integers(0,16,36)];kind=['fact']*32+['ctrl']*4;L=rng.normal(size=(36,16));L[np.arange(36),g]+=gain
        json.dump(dict(meta=dict(gold_idx=g,kind=kind),read={'16':{'out':L.tolist()}}),open(td/f'E55_{tag}_{arm}_s{s}.read.json','w'))
        log=[dict(k=i*8,update_norm=1.0,ce=2.0+(0.1 if (arm=='NATMASK' and tag=='orcp4') else 0.0),gnorm=1.5,coef=0.66,rand_hit_rate=0.2) for i in range(3)]
        json.dump(dict(K0=0,log=log,final_hash='h',diag={'8':dict(upd_nll_fact_later=3.0,nat_nll_fact_later=2.0,fact_grad_share=0.3)},read_hash=[(15,'a','a' if hash_ok else 'b')]),open(td/f'E55_{tag}_{arm}_s{s}.saved.json','w'))
    with tempfile.TemporaryDirectory() as td:
        td=pathlib.Path(td);seeds=[62,63,64,65,66]
        for s in seeds:
            for a in P4:mk(td,'orcp4',a,s,{'NATORACLE':3.0,'NATMASK':3.0,'NATHALF':0.5}.get(a,0.0))
            for a in P16:mk(td,'orcp16',a,s,{'NATMASK':2.0}.get(a,0.0))
        for a in ('NATORACLE','NATHALF'):mk(td,'orcp16',a,62,0.0)
        r=analyze(td,seeds);assert r['gates']['G2_pass'] and r['gates']['G3_read_hash_equal'] and 'same_fact_main' in r['p4']['margin'] and r['p4']['margin']['ORACLE-NATS']['mean']>1 and r['fixed_sequence']['ORACLE-NATS']['tested'] is True
        mk(td,'orcp4','NATBINF',62,0.0,hash_ok=False);r2=analyze(td,seeds);assert r2['gates']['G3_read_hash_equal'] is False
    assert holm({'a':0.01,'b':0.04})=={'a':0.02,'b':0.04} and abs(signflip([1,1,1,1,1])[0]-1/32)<1e-12 and abs(signflip([1,1,1,1,1,1])[1]-2/64)<1e-12 and seeds_needed(1.0,0.5)>=8
    print('e57_oracle_analyze selftest ok')
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--dir');ap.add_argument('--seeds');ap.add_argument('--json');ap.add_argument('--selftest',action='store_true');a=ap.parse_args()
    if a.selftest:selftest()
    else:
        r=analyze(a.dir,[int(x) for x in a.seeds.split(',')]);print(json.dumps(r,indent=1,default=float))
        if a.json:json.dump(r,open(a.json,'w'),indent=1,default=float)
