"""Qwen ISO hyper-parameter sweep summary (job 1309s; CPU; new file; EXPLORATORY, no pass / fail rule, no pre-registered threshold). For every configuration directory <D>/<cfg>/ written by cell_iso_ladder.py --sweep (arm ISO, reads at --expo, default 128) and for every seed present it prints one line and writes a table: out-of-window binding margin (summed span log-prob, gold minus best wrong, 16 candidates, mean over the 16 fact items) and its gain over W0, out-of-window top-1 share (chance 1/16 = 0.0625), in-window margin, canary NLL change (W0 file, pg11 nat / token), summed update norm, clip activity (share of steps with clip coef < 1). References from job 1306 (cband_main, same seeds): W0 and NATS (margin, top-1). Mean over the seeds present.
  python3 cband_iso_sweep_summary.py --dir D --main M [--seeds 161,162] [--expo 128] [--json J] | --selftest"""
import sys,json,argparse,pathlib
import numpy as np
HERE=pathlib.Path(__file__).resolve().parent;sys.path.insert(0,str(HERE))
import cband_iso_analyze as A
def load(p):return json.load(open(p))
def row(r,w0,expo):
    o=A.get(r,expo,'out');i=A.get(r,expo,'in');o0=A.get(w0,expo,'out');i0=A.get(w0,expo,'in')
    return dict(margin_out=float(o['margin'].mean()),gain_out=float(o['margin'].mean()-o0['margin'].mean()),top1_out=float(o['top1'].mean()),margin_in=float(i['margin'].mean()),gain_in=float(i['margin'].mean()-i0['margin'].mean()),dcanary=float(r['canary']-w0['canary']),cum_update_norm=float(sum(e['update_norm'] for e in r['log'])),clip_frac=float(np.mean([e['clip_coef']<1 for e in r['log']])),n_steps=len(r['log']))
def summarize(D,main,seeds,expo='128'):
    D=pathlib.Path(D);main=pathlib.Path(main);res={}
    for c in sorted(p.name for p in D.iterdir() if p.is_dir()):
        rows=[]
        for s in seeds:
            f=D/c/f'cband_ISO_s{s}.json'
            if f.exists():rows.append(row(load(f),load(main/f'cband_W0_s{s}.json'),expo))
        if rows:res[c]={k:float(np.mean([r[k] for r in rows])) for k in rows[0] if k!='n_steps'};res[c]['n_seeds']=len(rows);res[c]['n_steps']=rows[0]['n_steps']
    ref={}
    for a in ('NATS',):
        rs=[row(load(main/f'cband_{a}_s{s}.json'),load(main/f'cband_W0_s{s}.json'),expo) for s in seeds if (D/next(iter(res),'x')/f'cband_ISO_s{s}.json').exists() or True]
        ref[a]={k:float(np.mean([r[k] for r in rs])) for k in ('margin_out','gain_out','top1_out','margin_in','gain_in','dcanary','cum_update_norm','clip_frac')}
    return res,ref
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--dir');ap.add_argument('--main');ap.add_argument('--seeds',default='161');ap.add_argument('--expo',default='128');ap.add_argument('--json');ap.add_argument('--selftest',action='store_true');a=ap.parse_args()
    if a.selftest:selftest();return
    seeds=[int(x) for x in a.seeds.split(',')];res,ref=summarize(a.dir,a.main,seeds,a.expo)
    print(f"NATS (1306, same seeds): out margin {ref['NATS']['margin_out']:+.2f} (gain {ref['NATS']['gain_out']:+.2f}), top-1 {ref['NATS']['top1_out']:.3f}, in margin {ref['NATS']['margin_in']:+.2f}, canary {ref['NATS']['dcanary']:+.4f}, update norm {ref['NATS']['cum_update_norm']:.2f}")
    for c,r in res.items():print(f"SWEEP {c}: out margin {r['margin_out']:+.2f} (gain over W0 {r['gain_out']:+.2f})  top-1 {r['top1_out']:.3f}  in margin {r['margin_in']:+.2f} (gain {r['gain_in']:+.2f})  canary {r['dcanary']:+.4f}  update norm {r['cum_update_norm']:.2f}  clip {r['clip_frac']:.2f}  steps {r['n_steps']}  seeds {r['n_seeds']}")
    if a.json:pathlib.Path(a.json).write_text(json.dumps(dict(sweep=res,NATS=ref,seeds=seeds,expo=a.expo),indent=1))
def selftest():
    import tempfile;rng=np.random.default_rng(0)
    with tempfile.TemporaryDirectory() as td:
        td=pathlib.Path(td);(td/'m').mkdir();(td/'sw'/'c1').mkdir(parents=True)
        for s in (161,):
            g=rng.integers(0,16,20).tolist();meta=dict(kind=['fact']*16+['ctrl']*4,gold_idx=g)
            for name,d,eff,can in (('W0',td/'m',0.0,0.88),('NATS',td/'m',0.5,0.9),('ISO',td/'sw'/'c1',1.5,0.91)):
                rd={'128':{}}
                for k in ('out','in'):
                    o=rng.normal(0,1,(20,16))
                    for q in range(16):o[q,g[q]]+=eff+(3 if k=='in' else 0)
                    rd['128'][k]=o.tolist()
                fn='cband_W0_s161.json' if name=='W0' else f'cband_{name}_s161.json'
                json.dump(dict(meta=meta,reads=rd,log=[dict(update_norm=0.5,clip_coef=1.0)]*4,canary=can),open(d/fn,'w'))
        res,ref=summarize(td/'sw',td/'m',[161]);assert abs(res['c1']['gain_out'])>0.3 and res['c1']['n_steps']==4 and abs(res['c1']['dcanary']-0.03)<1e-9 and ref['NATS']['n' if False else 'margin_out'] is not None
    print('cband_iso_sweep_summary selftest ok')
if __name__=='__main__':main()
