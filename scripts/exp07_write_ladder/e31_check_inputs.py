"""E31 request 3 (CPU, read-only): do the stored result files hold what the E31 analysis needs?
Usage: e31_check_inputs.py [seed ...]   (EXP07_OUT selects the results dir, default experiments/exp07_write_ladder)
Reports per file: epochs/states present, whether margins hold the R1 families and '_alt' (R1-alt), canary fields, cum_update_norm."""
import os,sys,json,glob
from pathlib import Path
OUT=Path(os.environ.get('EXP07_OUT',Path(__file__).resolve().parents[2]/'experiments/exp07_write_ladder'))
seeds=sys.argv[1:] or ['101','102','103']
def summ(f):
    d=json.load(open(f));log=d['log'] if isinstance(d,dict) else d
    pts=[e for e in log if 'margins' in e];keys=[e.get('epoch',e.get('state')) for e in pts]
    m=pts[0]['margins'] if pts else {}
    return {'n_points':len(pts),'points':keys,'has_alt':all('_alt' in e['margins'] for e in pts),
            'fam':sorted(k for k in m if not k.startswith('_')),'alt_fam':sorted(m.get('_alt',{})),
            'canary':all('_gen_nll' in e['margins'] and '_tail_nll' in e['margins'] for e in pts),
            'cum_update_norm':all('cum_update_norm' in e for e in pts[1:]),'n_items_L3':len(m.get('L3',[]))}
for pat in ('L1_*','L0fs_*','L0mwfs_*','L0mp_*','L0or_*','L0x_*','L0mw*','meta_*'):
    for s in seeds:
        for f in sorted(glob.glob(str(OUT/f'{pat}_s{s}*.json'))):
            if '.saved.' in f:continue
            if Path(f).name.startswith('meta_'):print(Path(f).name,'(meta)');continue
            try:print(Path(f).name,json.dumps(summ(f)))
            except Exception as e:print(Path(f).name,'ERR',repr(e))
print('\nneeded: L0fs points 0..8 with has_alt; L1 ck points 0..8 with has_alt (epoch 1..8 = D_alt per epoch)')
