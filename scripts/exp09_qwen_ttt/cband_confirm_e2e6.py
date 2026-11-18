"""QWEN 6% BAND CONFIRMATION WITH THE ISO ESTABLISHMENT GATE (job 1311; CPU analysis; new file; REGISTRATION = this docstring, its sha256 is frozen with a signed tag in the private registry BEFORE the first seed is run; it imports t1 / signflip_p / margin of the frozen-style cband_confirm_analyze.py (unchanged) and the read helpers of cband_iso_analyze.py (unchanged)). Hans decision card 11/18 11:37 KST ('1306과 같은 규칙').
DESIGN (fixed): Qwen3-4B-Base, fast weights = the full gate / up / down MLP matrices of the last 9 of 36 layers (fullmlp, fp32, ||W_fast|| = 649.09), SWA structure of exp09 (detached rolling KV of 7 chunks + the current chunk), the step of EVERY chunk is a normalised-gradient step of the fixed global norm 0.30426 (cumulative relative path 6.0% over 128 chunks = 38.95 = 6.0% of ||W_fast||; cell_band_norm.py); arms NATS, NATMASK1024 (confirmatory), NATMASK256 (reported) and ISO-STD (cell_iso_ladder.py: fact sentences only, no context, segments (0,16] (16,32] (32,64] (64,128] packed on their own, one step per packed chunk, the segment update norm rescaled to the NATS increment of the same seed = the same 6.0% path); NEW world seeds 171..178 (never used with Qwen before); reads at exposures 16 / 32 / 64 / 128 (out-of-window neutral tail and in-window); primary read exposure 128 out-of-window.
MARGIN: per fact item, summed log-prob of the gold candidate minus the best wrong candidate (16 candidates of equal token length), mean over the 16 facts of a seed; unit = seed.
STEP 1, ESTABLISHMENT GATE (the experiment is only a test of the band if the write is established): E = margin_out(ISO-STD) - margin_out(NATS) at exposure 128, paired by seed: one-sided 95% t lower bound > 0. If the gate is NOT passed the result is reported as 'TEST NOT ESTABLISHED' (시험 불성립) and P1 is NOT judged.
STEP 2, P1 (only when the gate is passed; the rule of job 1306 / cband_confirm_analyze.py): G = margin_out(NATMASK1024) - margin_out(NATS) at exposure 128: one-sided 95% t lower bound > 0 AND mean G >= SESOI 0.5 nat AND the exact sign-flip tests of H0: mu <= SESOI and H0: mu <= 0 both reject (one-sided p < .05; n = 8: min p = 1/256). confirmed = gate passed AND P1 passed.
REPORTED, NOT A CONDITION (no pass / fail): G_256 = margin_out(NATMASK256) - margin_out(NATS); in-window cost H = margin_in(NATS) - margin_in(arm) at exposure 128 for both band arms (positive = harm); top-1 share (chance 1/16): ISO against chance (mean and t interval of top-1 - 1/16) and band - NATS, mean gold rank (1 = best), ISO - NATMASK1024 margin, the margin curves at exposures 16 / 32 / 64 / 128, the canary changes (after every read exposure and at the end), the update statistics, and the divergence record of every arm and seed.
DIVERGENCE (fixed in advance): a run (arm, seed) DIVERGED when a logged value (ce, grad_norm, update_norm) is non-finite, OR the mean ce of the last 8 logged steps exceeds 4.0 nat, OR the canary change (end of run minus W0, pg11 NLL, nat / token) exceeds +1.0. The largest single-chunk loss is reported only (spikes recover). A seed with any diverged arm, a missing read, or a NATMASK1024 that does not differ from NATS at CHUNK 1 (it equals NATS at chunk 0 by construction: no window key exists) is rerun ONCE (all arms); a seed that fails again is EXCLUDED; the effective seeds are the others; two or more excluded seeds: the status is exploratory ('탐색 지위') whatever the result; fewer than 5 effective seeds: no analysis. Never excluded because of its outcome.
  python3 cband_confirm_e2e6.py --dir D --seeds 171,...,178 --json J | --gate-seed S --dir D (exit 1 and the reasons when the seed fails the run gate) | --selftest"""
import sys,json,argparse,math
from pathlib import Path
HERE=Path(__file__).resolve().parent;sys.path.insert(0,str(HERE))
import numpy as np
import cband_confirm_analyze as C                    # t1, signflip_p, TQ1 (frozen analyzer of job 1306, unchanged)
import cband_iso_analyze as A                        # get (margin, top-1, rank), tci, TQ (unchanged)
ARMS=('NATS','NATMASK1024','NATMASK256','ISO');EXPO=('16','32','64','128');SESOI=0.5;CHANCE=1/16;MAXCE=4.0;MAXCAN=1.0
def load(D,arm,s):return json.load(open(Path(D)/f'cband_{arm}_s{s}.json'))
def run_check(r,w0):
    why=[];lg=r['log'];vals=[v for st in lg for k,v in st.items() if k in ('ce','grad_norm','update_norm') and isinstance(v,(int,float))]
    if not all(math.isfinite(v) for v in vals):why.append('non-finite log')
    ce=[st['ce'] for st in lg][-8:]
    if ce and float(np.mean(ce))>MAXCE:why.append(f'last-8 ce {np.mean(ce):.2f} > {MAXCE}')
    if r['canary']-w0['canary']>MAXCAN:why.append(f"canary change {r['canary']-w0['canary']:+.3f} > +{MAXCAN}")
    for e in EXPO:
        if e not in r['reads'] or 'out' not in r['reads'][e] or 'in' not in r['reads'][e]:why.append(f'missing read {e}')
    return why
def gate_seed(D,s):
    why=[]
    try:
        w0=load(D,'W0',s) if (Path(D)/f'cband_W0_s{s}.json').exists() else None
        assert w0 is not None,'W0 file missing'
        R={a:load(D,a,s) for a in ARMS}
        for a,r in R.items():why+=[f'{a}: {x}' for x in run_check(r,w0)]
        a,b=R['NATS'],R['NATMASK1024']
        if not (b['log'][1]['update_norm']!=a['log'][1]['update_norm'] or b['log'][1]['ce']!=a['log'][1]['ce']):why.append('NATMASK1024 inactive at chunk 1')
    except Exception as e:why.append(repr(e))
    return why
def marg(r,e,k):return float(A.get(r,e,k)['margin'].mean())
def analyze(D,seeds):
    bad={s:g for s in seeds if (g:=gate_seed(D,s))};use=[s for s in seeds if s not in bad]
    out=dict(run_type='CONFIRMATORY_P1_WITH_GATE',seeds_requested=list(seeds),gates=dict(excluded_seeds=bad,n_effective=len(use),confirmatory_status=bool(len(bad)<=1 and len(use)>=7)))
    if len(use)<5:out['abort']='too few seeds';return out
    R={a:[load(D,a,s) for s in use] for a in ARMS};W=[load(D,'W0',s) for s in use]
    mo={a:np.array([marg(r,'128','out') for r in R[a]]) for a in ARMS}
    E=mo['ISO']-mo['NATS'];e=C.t1(E);gate=bool(e['lower_onesided']>0)
    out['gate_ISO_minus_NATS']=dict(stats=e,passed=gate,per_seed=[float(x) for x in E],rule='one-sided 95% t lower bound > 0')
    G=mo['NATMASK1024']-mo['NATS'];g=C.t1(G);P1=bool(g['lower_onesided']>0 and g['mean']>=SESOI and C.signflip_p(G,SESOI,'gt')<0.05 and C.signflip_p(G,0.0,'gt')<0.05)
    out['P1']=dict(contrast='G_1024 = margin_out(NATMASK1024) - margin_out(NATS) at exposure 128',stats=g,p_signflip_H0_mu_le_SESOI=C.signflip_p(G,SESOI,'gt'),p_signflip_H0_mu_le_0=C.signflip_p(G,0.0,'gt'),SESOI=SESOI,passed=P1,per_seed=[float(x) for x in G],judged=gate)
    if gate:out['confirmed']=bool(P1);out['wording']='confirmed (gate and P1)' if P1 else 'not confirmed: gate passed, P1 failed'
    else:out['confirmed']=None;out['wording']='TEST NOT ESTABLISHED (시험 불성립): ISO - NATS lower bound <= 0; P1 not judged'
    rep={};hn=len(use);tq2=A.TQ[hn-1]
    for a in ('NATMASK1024','NATMASK256'):
        d=mo[a]-mo['NATS'];t=C.t1(d);h=np.array([marg(x,'128','in') for x in R['NATS']])-np.array([marg(x,'128','in') for x in R[a]]);th=C.t1(h)
        rep[a]=dict(G=t,G_ci95_two_sided=[t['mean']-tq2*t['se'],t['mean']+tq2*t['se']],H_in_window_harm=th,H_ci95_two_sided=[th['mean']-tq2*th['se'],th['mean']+tq2*th['se']],G_per_seed=[float(x) for x in d],H_per_seed=[float(x) for x in h])
    top={a:np.array([float(A.get(r,'128','out')['top1'].mean()) for r in R[a]]) for a in ARMS};rank={a:np.array([float(A.get(r,'128','out')['rank'].mean()) for r in R[a]]) for a in ARMS}
    rep['top1_128']={a:A.tci(top[a]) for a in ARMS};rep['mean_rank_128']={a:A.tci(rank[a]) for a in ARMS};rep['ISO_top1_minus_chance']=A.tci(top['ISO']-CHANCE);rep['top1_diff_vs_NATS']={a:A.tci(top[a]-top['NATS']) for a in ('NATMASK1024','NATMASK256','ISO')}
    rep['ISO_minus_NATMASK1024_margin']=A.tci(mo['ISO']-mo['NATMASK1024']);rep['W0_margin_out_128']=A.tci([marg(w,'128','out') for w in W])
    rep['margin_out_curve']={a:{x:A.tci([marg(r,x,'out') for r in R[a]]) for x in EXPO} for a in ARMS}
    rep['canary']={a:dict(final=A.tci([r['canary']-w['canary'] for r,w in zip(R[a],W)]),per_exposure={x:A.tci([r['reads'][x]['canary']-w['canary'] for r,w in zip(R[a],W)]) for x in EXPO if all('canary' in r['reads'].get(x,{}) for r in R[a])}) for a in ARMS}
    rep['divergence_record']={a:[dict(seed=s,max_ce=float(max(st['ce'] for st in r['log'])),last8_ce=float(np.mean([st['ce'] for st in r['log']][-8:])),max_gnorm=float(max(st['grad_norm'] for st in r['log'])),mean_update_norm=float(np.mean([st['update_norm'] for st in r['log']])),final_canary_change=float(r['canary']-w['canary'])) for s,r,w in zip(use,R[a],W)] for a in ARMS}
    out['reported']=rep;return out
def selftest():
    import tempfile;rng=np.random.default_rng(0)
    def build(td,effects,diverge=None,seeds=range(171,179)):
        for s in seeds:
            g=rng.integers(0,16,20).tolist();meta=dict(kind=['fact']*16+['ctrl']*4,gold_idx=g)
            w0={'meta':meta,'reads':{e:{'out':rng.normal(0,1,(20,16)).tolist(),'in':(rng.normal(0,1,(20,16))+0).tolist()} for e in EXPO},'canary':0.88};json.dump(w0,open(td/f'cband_W0_s{s}.json','w'))
            for a in ARMS:
                rd={}
                for e in EXPO:
                    o=rng.normal(0,1,(20,16));i=rng.normal(0,1,(20,16))
                    for q in range(16):o[q,g[q]]+=effects[a];i[q,g[q]]+=3
                    rd[e]={'out':o.tolist(),'in':i.tolist(),'canary':0.9}
                n=128 if a!='ISO' else 52;ce=2.0+(10 if (diverge and (a,s)==diverge) else 0)
                log=[dict(ce=ce,grad_norm=1.0,update_norm=0.3,clip_coef=1.0)]*n
                if a=='NATMASK1024':log[1]=dict(ce=2.5,grad_norm=1.0,update_norm=0.31,clip_coef=1.0)
                json.dump(dict(meta=meta,log=log,reads=rd,canary=0.9),open(td/f'cband_{a}_s{s}.json','w'))
    with tempfile.TemporaryDirectory() as td:
        td=Path(td);build(td,dict(NATS=0.0,NATMASK1024=1.0,NATMASK256=1.0,ISO=3.0));o=analyze(td,list(range(171,179)))
        assert o['gates']['n_effective']==8 and o['gate_ISO_minus_NATS']['passed'] and o['P1']['passed'] and o['confirmed'] is True and 'confirmed' in o['wording'] and 'reported' in o and abs(o['P1']['stats']['mean']-1.0)<0.6,o['wording']
    with tempfile.TemporaryDirectory() as td:                                     # gate fails -> test not established, P1 not judged
        td=Path(td);build(td,dict(NATS=0.0,NATMASK1024=1.0,NATMASK256=1.0,ISO=-0.5));o=analyze(td,list(range(171,179)))
        assert not o['gate_ISO_minus_NATS']['passed'] and o['confirmed'] is None and 'NOT ESTABLISHED' in o['wording'] and o['P1']['judged'] is False
    with tempfile.TemporaryDirectory() as td:                                     # gate passes, P1 fails (band effect below SESOI)
        td=Path(td);build(td,dict(NATS=0.0,NATMASK1024=0.2,NATMASK256=0.2,ISO=3.0));o=analyze(td,list(range(171,179)))
        assert o['gate_ISO_minus_NATS']['passed'] and not o['P1']['passed'] and o['confirmed'] is False
    with tempfile.TemporaryDirectory() as td:                                     # a diverged run excludes the seed; one excluded seed keeps the confirmatory status; two do not
        td=Path(td);build(td,dict(NATS=0.0,NATMASK1024=1.0,NATMASK256=1.0,ISO=3.0),diverge=('NATS',171));o=analyze(td,list(range(171,179)))
        assert 171 in o['gates']['excluded_seeds'] and o['gates']['n_effective']==7 and o['gates']['confirmatory_status'] and any('last-8 ce' in x for x in o['gates']['excluded_seeds'][171])
        d=json.load(open(td/'cband_ISO_s172.json'));d['log'][3]['ce']=float('nan');json.dump(d,open(td/'cband_ISO_s172.json','w'));o=analyze(td,list(range(171,179)))
        assert o['gates']['n_effective']==6 and not o['gates']['confirmatory_status']
        d=json.load(open(td/'cband_NATMASK1024_s173.json'));d['log'][1]=dict(ce=2.0,grad_norm=1.0,update_norm=0.3,clip_coef=1.0);json.dump(d,open(td/'cband_NATMASK1024_s173.json','w'))
        assert any('inactive at chunk 1' in x for x in gate_seed(td,173))
    print('cband_confirm_e2e6 selftest ok')
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--dir');ap.add_argument('--seeds');ap.add_argument('--json');ap.add_argument('--gate-seed',type=int);ap.add_argument('--selftest',action='store_true');a=ap.parse_args()
    if a.selftest:selftest();return
    if a.gate_seed is not None:
        g=gate_seed(a.dir,a.gate_seed);print(f'GATE seed {a.gate_seed}:',g or 'ok');sys.exit(1 if g else 0)
    o=analyze(a.dir,[int(x) for x in a.seeds.split(',')]);s=json.dumps(o,indent=1,default=float);print(o.get('wording',o.get('abort')))
    if a.json:Path(a.json).write_text(s)
if __name__=='__main__':main()
