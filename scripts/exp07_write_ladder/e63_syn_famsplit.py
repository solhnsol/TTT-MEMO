"""Synthetic world (v3_16_1, 16 facts): familiarity / binding split with the PAIR-SHUFFLED stream (CPU; new file; thread-server 11/25; reuses per_fact/TR.metrics of e57_syn_shuf_report). Same stored token reads as job 960 (seed 64) and its seed-61 analogue. Reading of the metrics (per fact, change over W0; place = answer): first_dA = gold first-token log p gain; first_dAB = that gain minus the gain of the branch decoy; full_dAB = same over the full place string. On the SHUFFLED stream the place strings are exchanged inside every chunk (exposure counts of every place string unchanged, pairing with the name destroyed), so shuffled = familiarity-type gain only; ORIGINAL - SHUFFLED = binding-type gain. Arms: M3NMW1N05 = NATQ (natural), M3W1N05 = ALLQ256, M3W8N05 = M3 w=8 (masks). Usage: python e63_syn_famsplit.py"""
import sys,json,pathlib,numpy as np
HERE=pathlib.Path(__file__).resolve().parent;ROOT=HERE.parents[1];sys.path.insert(0,str(HERE))
import e57_syn_tokread as TR
from e57_syn_shuf_report import per_fact
EXP=ROOT/'experiments/exp07_write_ladder'
CFG={64:(EXP/'e57_syn_shuf_curve/e57_syn_tokread_s64.json',[EXP/'e57_syn_ncurve_s64/e57_syn_tokread_s64.json',EXP/'e57_syn_curve_s64/e57_syn_tokread_s64.json']),
     61:(EXP/'e57_syn_shuf_curve_s61/e57_syn_tokread_s61.json',[EXP/'e57_syn_curve_s61/e57_syn_tokread_s61.json'])}
ARMS={'M3NMW1N05':'NATQ','M3W1N05':'ALLQ256','M3W8N05':'M3 w=8'};MET=('first_dA','first_dAB','full_dAB')
def bs(x):
    r=TR.boot(np.asarray(x,float));return f"{r['mean']:7.3f} [{r['ci95'][0]:7.3f},{r['ci95'][1]:7.3f}]"
for sd,(sp,ol) in CFG.items():
    S=json.load(open(sp));tokids=S['tokids'];meta=json.load(open(next(iter(TR.arms_curve(sd,'synqsh').values()))['read_json']))['meta']
    W0S=TR.metrics(S['W0'],tokids,meta);facts=sorted(W0S);O={};W0O=None
    for f in ol:
        d=json.load(open(f));assert d['tokids']==tokids;O.update(d['arms']);W0O=W0O or d['W0']
    W0O=TR.metrics(W0O,tokids,meta)
    SH={};OR={}
    for l,t in S['arms'].items():
        arm,ch=l.split('@c');ch=int(ch);SH[(arm,ch)]=per_fact(TR.metrics(t,tokids,meta),W0S,facts)
        if l in O:OR[(arm,ch)]=per_fact(TR.metrics(O[l],tokids,meta),W0O,facts)
    print(f'\n=== synthetic seed {sd}: {len(facts)} facts; bootstrap 95% intervals by fact; columns first_dA | first_dAB | full_dAB (change over W0, summed log p)')
    chs=sorted({c for (_,c) in SH})
    for ch in chs:
        print(f' -- chunk {ch}')
        for arm,nm in ARMS.items():
            if (arm,ch) not in SH or (arm,ch) not in OR:continue
            o,s=OR[(arm,ch)],SH[(arm,ch)]
            print(f'   {nm:8s} ORIGINAL  '+' | '.join(bs(o[m]) for m in MET))
            print(f'   {nm:8s} SHUFFLED  '+' | '.join(bs(s[m]) for m in MET)+'   (familiarity-type)')
            print(f'   {nm:8s} ORIG-SHUF '+' | '.join(bs(o[m]-s[m]) for m in MET)+'   (binding-type)')
        nat='M3NMW1N05'
        for arm,nm in ARMS.items():
            if arm==nat or (arm,ch) not in SH or (arm,ch) not in OR or (nat,ch) not in OR:continue
            print(f'   {nm}-NATQ ORIGINAL '+' | '.join(bs(OR[(arm,ch)][m]-OR[(nat,ch)][m]) for m in MET))
            print(f'   {nm}-NATQ SHUFFLED '+' | '.join(bs(SH[(arm,ch)][m]-SH[(nat,ch)][m]) for m in MET))
            print(f'   {nm}-NATQ binding-type (ORIG-SHUF, difference of differences) '+' | '.join(bs((OR[(arm,ch)][m]-SH[(arm,ch)][m])-(OR[(nat,ch)][m]-SH[(nat,ch)][m])) for m in MET))
