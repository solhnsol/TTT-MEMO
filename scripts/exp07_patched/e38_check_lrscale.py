"""E38 unit checks from smoke runs of s3.py (--canary-only --smoke-chunks N, tag contains 'smoke'; OUT = the usual output dir).
(a) bit identity at x1: compare inner_hash of the ORIGINAL s3.py (f3a6f59) smoke run with the patched s3.py --lr-scale 1.0 smoke run.
(b) x2: chunk 0 starts from W0 in both runs, so gnorm and coef must match and update_norm(x2) = 2 x update_norm(x1) up to fp32 rounding (later chunks start from different weights: not comparable).
Usage: e38_check_lrscale.py LOG_ORIG LOG_X1 LOG_X2   (each a *.saved.json)"""
import sys,json
def last(p):
    d=json.load(open(p));return d['log'][-1]
o,a,b=[last(p) for p in sys.argv[1:4]]
print('(a) inner_hash orig == patched x1:',o['inner_hash']==a['inner_hash'],o['inner_hash'][:16],a['inner_hash'][:16])
print('    per-chunk update_norm identical orig vs x1:',all(x['update_norm']==y['update_norm'] and x['gnorm']==y['gnorm'] for x,y in zip(o['chunks'],a['chunks'])))
c1,c2=a['chunks'][0],b['chunks'][0];r=c2['update_norm']/max(c1['update_norm'],1e-30)
print(f"(b) chunk 0: gnorm x1 {c1['gnorm']:.6g} x2 {c2['gnorm']:.6g}; coef {c1['coef']:.6g}/{c2['coef']:.6g}; update_norm ratio x2/x1 = {r:.8f} (expect 2); clip bound at chunk 0: {c1['coef']<1}")
print('    PASS' if abs(r-2)<1e-4 and abs(c1['gnorm']-c2['gnorm'])<=1e-5*max(c1['gnorm'],1e-30) else '    CHECK')
