"""F1/F3 (PROTOCOL_stage01.md M3): compare saved weight hashes of two runs. Usage: check_hashes.py SEED LEVEL1 TAG1 LEVEL2 TAG2 [UPTO_PASS]
Exit 0 iff the hashes of passes 0..UPTO (default: all common passes) are identical in both worlds."""
import sys,json,os
from pathlib import Path
OUT=Path(os.environ.get('EXP07_OUT',Path(__file__).resolve().parents[2]/'experiments/exp07_write_ladder'))
sd,l1,t1,l2,t2=sys.argv[1:6];upto=int(sys.argv[6]) if len(sys.argv)>6 else 99;ok=True
for w in 'AB':
    h=[{e.get('epoch',e.get('state')):e['margins']['_hash'] for e in json.load(open(OUT/f'{l}_{t}_s{sd}_{w}.saved.json'))['log'] if 'margins' in e and '_hash' in e['margins']} for l,t in ((l1,t1),(l2,t2))]
    ks=sorted(k for k in set(h[0])&set(h[1]) if k<=upto)
    for k in ks:
        same=h[0][k]==h[1][k];ok&=same;print(w,k,'same' if same else 'DIFFERENT')
    ok&=len(ks)>0
print('HASH_CHECK','PASS' if ok else 'FAIL');sys.exit(0 if ok else 1)
