"""Qwen fixed-norm NATS / band summary with the divergence record (job 1310; CPU; new file; EXPLORATORY, descriptive; NO divergence rule is applied: the numbers are stored so that divergence can be judged afterwards). For every size directory <D>/<size>/ written by cell_band_norm.py and every arm / seed present: the largest chunk loss (log[].ce), the loss of the last chunk, the largest gradient norm, whether any logged value is non-finite, the mean update norm (= the fixed step norm), the canary NLL change (pg11, nat / token, state minus W0) after EVERY read exposure and at the end of the run, the out-of-window binding margin (summed span log-prob, gold minus best wrong, 16 candidates, mean over the 16 facts), its gain over W0, the top-1 share (chance 1/16) at every read exposure and the in-window margin at exposure 128. A descriptive flag 'DIV?' is printed when the largest chunk loss exceeds 10 nat, or the final canary change exceeds +1.0 nat / token, or a value is non-finite (the thresholds are printed with the table and are NOT a rule; the stage-0 run lr 1 had a largest loss 28-32 and +0.61, the diverged ISO e2e12pct +13.7). Means over the seeds present per size and arm.
  python3 cband_norm_summary.py --dir D --seeds 161,162 [--json J] | --selftest"""
import sys,json,argparse,math,pathlib
import numpy as np
HERE=pathlib.Path(__file__).resolve().parent;sys.path.insert(0,str(HERE))
import cband_iso_analyze as A
EXPO=('16','32','64','128');MAXCE=10.0;MAXCAN=1.0
def load(p):return json.load(open(p))
def one(r,w0):
    lg=r['log'];ce=np.array([e['ce'] for e in lg],float);gn=np.array([e['grad_norm'] for e in lg],float);un=np.array([e['update_norm'] for e in lg],float)
    row=dict(max_ce=float(np.nanmax(ce)) if np.isfinite(ce).any() else float('nan'),last_ce=float(ce[-1]),max_gnorm=float(np.nanmax(gn)),nonfinite=bool(not np.isfinite(ce).all() or not np.isfinite(gn).all() or not np.isfinite(un).all()),mean_update_norm=float(np.nanmean(un)),cum_update_norm=float(np.nansum(un)),final_canary_change=float(r['canary']-w0['canary']))
    for e in EXPO:
        if e in r['reads']:
            o=A.get(r,e,'out');o0=A.get(w0,e,'out');row[f'margin_out_{e}']=float(o['margin'].mean());row[f'gain_out_{e}']=float(o['margin'].mean()-o0['margin'].mean());row[f'top1_out_{e}']=float(o['top1'].mean())
            if 'canary' in r['reads'][e]:row[f'canary_change_{e}']=float(r['reads'][e]['canary']-w0['canary'])
    if '128' in r['reads']:i=A.get(r,'128','in');i0=A.get(w0,'128','in');row['margin_in_128']=float(i['margin'].mean());row['gain_in_128']=float(i['margin'].mean()-i0['margin'].mean())
    row['DIV?']=bool(row['nonfinite'] or row['max_ce']>MAXCE or row['final_canary_change']>MAXCAN);return row
def summarize(D,seeds):
    D=pathlib.Path(D);res={}
    for sz in sorted(p.name for p in D.iterdir() if p.is_dir()):
        for arm in ('NATS','NATMASK1024','NATMASK256'):
            for s in seeds:
                f=D/sz/f'cband_{arm}_s{s}.json';w=D/sz/f'cband_W0_s{s}.json'
                if f.exists() and w.exists():res.setdefault(sz,{}).setdefault(arm,{})[s]=one(load(f),load(w))
    return res
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--dir');ap.add_argument('--seeds',default='161,162');ap.add_argument('--json');ap.add_argument('--selftest',action='store_true');a=ap.parse_args()
    if a.selftest:selftest();return
    seeds=[int(x) for x in a.seeds.split(',')];res=summarize(a.dir,seeds)
    print(f'descriptive flag thresholds (NOT a rule): largest chunk loss > {MAXCE} nat, final canary change > +{MAXCAN} nat/token, non-finite')
    for sz,arms in res.items():
        for arm,ss in arms.items():
            for s,r in ss.items():
                print(f"SIZE {sz} {arm} s{s}: max ce {r['max_ce']:.1f} last ce {r['last_ce']:.2f} max gnorm {r['max_gnorm']:.1f} step norm {r['mean_update_norm']:.4f} | out margin 16/32/64/128 "+'/'.join(f"{r.get(f'margin_out_{e}',float('nan')):+.2f}" for e in EXPO)+' top-1 '+'/'.join(f"{r.get(f'top1_out_{e}',float('nan')):.2f}" for e in EXPO)+f" | in128 {r.get('margin_in_128',float('nan')):+.2f} | canary final {r['final_canary_change']:+.3f} per exposure "+'/'.join(f"{r.get(f'canary_change_{e}',float('nan')):+.3f}" for e in EXPO)+f" | DIV? {r['DIV?']}")
    if a.json:pathlib.Path(a.json).write_text(json.dumps(dict(runs=res,seeds=seeds,thresholds=dict(max_ce=MAXCE,max_canary=MAXCAN,note='descriptive, not a rule')),indent=1,default=float))
def selftest():
    import tempfile;rng=np.random.default_rng(0)
    with tempfile.TemporaryDirectory() as td:
        td=pathlib.Path(td);(td/'sz').mkdir()
        for arm in ('NATS','NATMASK1024','NATMASK256','W0'):
            g=rng.integers(0,16,20).tolist();meta=dict(kind=['fact']*16+['ctrl']*4,gold_idx=g);rd={}
            for e in EXPO:
                o=rng.normal(0,1,(20,16));i=rng.normal(0,1,(20,16));rd[e]=dict(out=o.tolist(),**{'in':i.tolist()})
                if arm!='W0':rd[e]['canary']=0.9+0.01*int(e)/16
            log=[dict(ce=2.0+(30.0 if (arm=='NATS' and k==5) else 0.0),grad_norm=1.0,clip_coef=1.0,update_norm=0.3,raw_update_norm=0.5) for k in range(128)]
            json.dump(dict(meta=meta,log=log,reads=rd,canary=0.9 if arm!='W0' else 0.88),open(td/'sz'/('cband_W0_s161.json' if arm=='W0' else f'cband_{arm}_s161.json'),'w'))
        r=summarize(td,[161]);assert r['sz']['NATS'][161]['DIV?'] and r['sz']['NATS'][161]['max_ce']>30 and not r['sz']['NATMASK1024'][161]['DIV?'] and abs(r['sz']['NATS'][161]['final_canary_change']-0.02)<1e-9 and 'canary_change_64' in r['sz']['NATS'][161]
    print('cband_norm_summary selftest ok')
if __name__=='__main__':main()
