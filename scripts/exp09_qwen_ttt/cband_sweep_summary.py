"""Qwen3 lr / fast-weight sweep summary (job 1304s; CPU; new file): for each configuration directory <O>/<cfg>/ written by cell_band_ladder.py (arm NATS, seed 161, reads at the given exposures) it prints one line: margin_out(last exposure) - W0, margin_in(last exposure) - W0, canary NLL (NATS) - canary NLL (W0), mean update norm, mean chunk ce. Margin = summed span log-prob of the gold candidate minus the best wrong candidate (16 candidates of equal length), mean over the 16 fact items (the E57 definition). 'WRITES' = margin_in gain >= 1 nat AND margin_out gain >= 1 nat; 'CANARY' flags a canary NLL rise > 0.05 nat / token. These flags are technical (where does the write start), not a result.
  python3 cband_sweep_summary.py --dir O [--cfgs a,b,c] [--json J] | --selftest"""
import sys,json,argparse,pathlib
import numpy as np
def margin(sc,meta):
    x=np.asarray(sc,float);g=np.asarray(meta['gold_idx']);f=np.asarray([k=='fact' for k in meta['kind']]);x=x[f];g=g[f];gold=x[np.arange(len(g)),g];y=x.copy();y[np.arange(len(g)),g]=-np.inf;return float((gold-y.max(1)).mean())
def one(d,s=161):
    d=pathlib.Path(d);a=json.load(open(d/f'cband_NATS_s{s}.json'));w=json.load(open(d/f'cband_W0_s{s}.json'));e=sorted(a['reads'],key=int)[-1]
    go=margin(a['reads'][e]['out'],a['meta'])-margin(w['reads'][e]['out'],w['meta']);gi=margin(a['reads'][e]['in'],a['meta'])-margin(w['reads'][e]['in'],w['meta'])
    r=dict(exposure=int(e),dmargin_out=go,dmargin_in=gi,dcanary=a['canary']-w['canary'],mean_update_norm=float(np.mean([x['update_norm'] for x in a['log']])),mean_ce=float(np.mean([x['ce'] for x in a['log']])),sec=a['sec'])
    r['writes']=bool(go>=1.0 and gi>=1.0);r['canary_flag']=bool(r['dcanary']>0.05);return r
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--dir');ap.add_argument('--cfgs');ap.add_argument('--json');ap.add_argument('--selftest',action='store_true');a=ap.parse_args()
    if a.selftest:
        import tempfile;rng=np.random.default_rng(0)
        with tempfile.TemporaryDirectory() as td:
            td=pathlib.Path(td);meta=dict(kind=['fact']*16+['ctrl']*4,gold_idx=rng.integers(0,16,20).tolist());(td/'c').mkdir()
            def rd(eff):
                L=rng.normal(0,1,(20,16))
                for i in range(16):L[i,meta['gold_idx'][i]]+=eff
                return {'out':L.tolist(),'in':L.tolist()}
            json.dump(dict(meta=meta,reads={'128':rd(0)},canary=0.9),open(td/'c/cband_W0_s161.json','w'));json.dump(dict(meta=meta,reads={'64':rd(0),'128':rd(2.0)},canary=0.92,log=[dict(update_norm=0.1,ce=2.0)]*3,sec=1.0),open(td/'c/cband_NATS_s161.json','w'))
            r=one(td/'c');assert r['exposure']==128 and abs(r['dcanary']-0.02)<1e-9 and not r['canary_flag'];print('cband_sweep_summary selftest ok');return
    cfgs=a.cfgs.split(',') if a.cfgs else sorted(p.name for p in pathlib.Path(a.dir).iterdir() if p.is_dir());out={}
    for c in cfgs:
        try:r=one(pathlib.Path(a.dir)/c)
        except Exception as e:print(f'SWEEP {c}: not available ({e!r:.80})');continue
        out[c]=r;print(f"SWEEP {c}: out(gain over W0) {r['dmargin_out']:+.2f}  in {r['dmargin_in']:+.2f}  canary {r['dcanary']:+.4f}  mean update norm {r['mean_update_norm']:.4g}  mean ce {r['mean_ce']:.3f}  {'WRITES' if r['writes'] else 'no write'}{' CANARY' if r['canary_flag'] else ''}")
    if a.json:pathlib.Path(a.json).write_text(json.dumps(out,indent=1))
if __name__=='__main__':main()
