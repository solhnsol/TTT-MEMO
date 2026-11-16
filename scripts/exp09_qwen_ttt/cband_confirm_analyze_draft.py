"""DRAFT (not frozen, not registered; for Hans' approval before any freeze): confirmation analysis of the band mask on Qwen3-4B-Base (SWA structure + NTP-TTT with LoRA fast weights), modelled on e57_band1024_confirm_analyze_c.py (job 1297c) and fed by cell_band_ladder.py output (cband_<arm>_s<seed>.json, cband_W0_s<seed>.json).
PROPOSED DESIGN: seeds 161..168 (8; world seeds of the E57 cell v3_16_1 built with the Qwen tokenizer), arms NATS and NATMASK1024 (NATMASK256 as a secondary arm), K0 = 0, reads at exposures 16/32/64/128, primary read exposure 128. Status: exploratory until frozen; the 1297c thresholds are carried over and have to be re-checked on the margin scale of this model after the smoke job 1301s.
MARGIN: per fact item, summed log-prob of the gold candidate minus the best wrong candidate (16 candidates of equal token length), mean over the 16 facts of a seed; unit = seed.
PRIMARY HYPOTHESES (intersection-union, as 1297c):
 P1 (binding effect): G = margin_out(NATMASK1024) - margin_out(NATS) at exposure 128: one-sided 95% t lower bound > 0 AND mean G >= SESOI (proposal 0.5 nat, the 1283 value; to be revisited after the smoke) AND the exact sign-flip tests of H0: mu <= SESOI and mu <= 0 reject (one-sided p < .05; n = 8: min p = 1/256).
 P2 (in-window harm): H = margin_in(NATS) - margin_in(NATMASK1024) at exposure 128 (positive = harm): one-sided 95% t UPPER bound <= 0.25 nat. (In the QA runs of job 1299 / 1300 this contrast was clearly harmful; whether it can pass on this model is open, the draft keeps the 1297c threshold and reports the result either way.)
 P3 (general text): D_N = canary NLL(NATMASK1024) - canary NLL(NATS) per seed (pg11, first 8192 tokens, nat / token): one-sided 95% t UPPER bound <= 0.01 (the DCLM threshold of 1297; pg11 replaces DCLM because no unused DCLM documents remain on the server and the exp09 canary is the preservation read of this pipeline).
EXCLUSION (as 1283 / 1297c, corrected gate): a seed is excluded and rerun once (all arms) when a log value is non-finite, a read is missing, or NATMASK1024 does not differ from NATS at CHUNK 1 (it equals NATS at chunk 0 by construction); a seed that fails again is EXCLUDED (n = 7); two or more excluded seeds: the status is exploratory. Never excluded because of its outcome.
SECONDARY (descriptive): arm NATMASK256 vs NATS at exposures 16..128 (out and in), H_256 - H_1024, canary of NATMASK256.
  python3 cband_confirm_analyze_draft.py --dir D --seeds 161,...,168 --json J | --selftest"""
import sys,json,argparse,itertools,math
from pathlib import Path
import numpy as np
TQ1={1:6.314,2:2.920,3:2.353,4:2.132,5:2.015,6:1.943,7:1.895,8:1.860,9:1.833,10:1.812,11:1.796,12:1.782,13:1.771,14:1.761,15:1.753}   # one-sided 95% t quantiles
def margin(sc,meta):
    x=np.asarray(sc,float);g=np.asarray(meta['gold_idx']);f=np.asarray([k=='fact' for k in meta['kind']]);x=x[f];g=g[f];gold=x[np.arange(len(g)),g];y=x.copy();y[np.arange(len(g)),g]=-np.inf;return float((gold-y.max(1)).mean())
def t1(v):
    v=np.asarray(v,float);n=len(v);m=float(v.mean());se=float(v.std(ddof=1)/math.sqrt(n));q=TQ1[n-1];return dict(mean=m,se=se,lower_onesided=m-q*se,upper_onesided=m+q*se,n=n)
def signflip_p(v,mu0,side):
    """exact one-sided sign-flip p of H0: mean <= mu0 (side 'gt') on d = v - mu0"""
    d=np.asarray(v,float)-mu0;n=len(d);obs=d.mean();cnt=0
    for s in itertools.product((1,-1),repeat=n):cnt+=(np.dot(s,d)/n)>=obs-1e-15
    return cnt/2**n
def load(D,s,arm):return json.load(open(Path(D)/f'cband_{arm}_s{s}.json'))
def gate(D,s):
    why=[]
    try:
        a,b=load(D,s,'NATS'),load(D,s,'NATMASK1024')
        if not all(math.isfinite(v) for x in (a,b) for st in x['log'] for v in st.values()):why.append('non-finite log')
        if not (b['log'][1]['update_norm']!=a['log'][1]['update_norm'] or b['log'][1]['ce']!=a['log'][1]['ce']):why.append('NATMASK1024 inactive at chunk 1')
        for x in (a,b):
            if not {'16','32','64','128'}<=set(x['reads']):why.append('missing read')
    except Exception as e:why.append(repr(e))
    return why
def analyze(D,seeds):
    bad={s:g for s in seeds if (g:=gate(D,s))};use=[s for s in seeds if s not in bad];out=dict(run_type='DRAFT',seeds_requested=list(seeds),gates=dict(excluded_seeds=bad,n_effective=len(use),confirmatory_status=bool(len(bad)<=1 and len(use)>=7)))
    if len(use)<5:out['abort']='too few seeds';return out
    G=[];H=[];Dn=[]
    for s in use:
        a,b=load(D,s,'NATS'),load(D,s,'NATMASK1024');mt=a['meta']
        G.append(margin(b['reads']['128']['out'],mt)-margin(a['reads']['128']['out'],mt));H.append(margin(a['reads']['128']['in'],mt)-margin(b['reads']['128']['in'],mt));Dn.append(b['canary']-a['canary'])
    SESOI=0.5;g=t1(G);P1=bool(g['lower_onesided']>0 and g['mean']>=SESOI and signflip_p(G,SESOI,'gt')<0.05 and signflip_p(G,0.0,'gt')<0.05)
    out['P1']=dict(contrast='G_1024',stats=g,p_signflip_H0_mu_le_SESOI=signflip_p(G,SESOI,'gt'),p_signflip_H0_mu_le_0=signflip_p(G,0.0,'gt'),passed=P1,per_seed=G)
    h=t1(H);P2=bool(h['upper_onesided']<=0.25);out['P2']=dict(contrast='H_1024 (positive = harm)',stats=h,passed=P2,per_seed=H)
    d=t1(Dn);P3=bool(d['upper_onesided']<=0.01);out['P3']=dict(contrast='canary D_N',stats=d,passed=P3,per_seed=Dn)
    out['confirmed']=bool(P1 and P2 and P3);out['wording']='confirmed (P1, P2 and P3)' if out['confirmed'] else 'not confirmed: '+', '.join(n for n,p in (('P1',P1),('P2',P2),('P3',P3)) if not p)+' failed'
    return out
def selftest():
    import tempfile
    rng=np.random.default_rng(0)
    with tempfile.TemporaryDirectory() as td:
        seeds=list(range(161,169))
        for s in seeds:
            g=rng.integers(0,16,20)
            for arm,eff,hh in (('NATS',0.0,0.0),('NATMASK1024',1.0,0.1)):
                rd={}
                for e in ('16','32','64','128'):
                    L=rng.normal(0,1,(20,16));out_=L.copy();inw=L.copy()
                    for i in range(16):out_[i,g[i]]+=2+eff;inw[i,g[i]]+=3-hh
                    rd[e]=dict(out=out_.tolist(),**{'in':inw.tolist()})
                json.dump(dict(arm=arm,seed=s,meta=dict(kind=['fact']*16+['ctrl']*4,gold_idx=g.tolist()),log=[dict(update_norm=1.0,ce=2.0),dict(update_norm=1.0+eff,ce=2.0+eff)]+[dict(update_norm=1.0,ce=2.0)]*126,reads=rd,canary=2.0+0.003*eff+0.001*rng.normal()),open(f'{td}/cband_{arm}_s{s}.json','w'))
        o=analyze(td,seeds);assert o['gates']['n_effective']==8 and o['P1']['passed'] and o['P2']['passed'] and o['P3']['passed'] and o['confirmed'],o
        assert abs(o['P1']['stats']['mean']-1.0)<0.6
        # an arm that is inactive at chunk 1 excludes the seed
        d=json.load(open(f'{td}/cband_NATMASK1024_s161.json'));d['log'][1]=dict(update_norm=1.0,ce=2.0);json.dump(d,open(f'{td}/cband_NATMASK1024_s161.json','w'));o=analyze(td,seeds);assert 161 in o['gates']['excluded_seeds'] and o['gates']['n_effective']==7
    assert abs(signflip_p([1,1,1,1,1,1,1,1],0.0,'gt')-1/256)<1e-12
    print('cband_confirm_analyze_draft selftest ok')
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--dir');ap.add_argument('--seeds');ap.add_argument('--json');ap.add_argument('--selftest',action='store_true');a=ap.parse_args()
    if a.selftest:selftest();return
    o=analyze(a.dir,[int(x) for x in a.seeds.split(',')]);s=json.dumps(o,indent=1,default=float);print(s)
    if a.json:Path(a.json).write_text(s)
if __name__=='__main__':main()
