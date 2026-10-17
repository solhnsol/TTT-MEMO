"""c2 sensitivity upper bound (design 16:02 KST 10/17), files only. Per-world contribution to the pooled R1-alt D_alt of the
gbc_nm one-step (OS2_gbc_nm_s{101,103}_{A,B}.json): dA(s) = D(A_s,B_0) - D(A_0,B_0), dB(s) = D(A_0,B_s) - D(A_0,B_0), dA + dB = D(s) exactly
(D is linear in the per-world margins). Mismatching world (s101 B, s103 A) contribution d_mis; bound D + 2 |d_mis|; maximum over damage-free s.
Usage: analyze_c2_signbound.py"""
import sys,json,importlib
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parent))
def mods(sd):
    sys.argv=['x',str(sd)]
    for k in ('analyze_ladder','analyze_stage01'):sys.modules.pop(k,None)
    return importlib.import_module('analyze_ladder'),importlib.import_module('analyze_stage01')
MIS={101:'B',103:'A'};out={}
for sd in (101,103):
    AL,S=mods(sd);raw={w:json.load(open(S.OUT/f'OS2_gbc_nm_s{sd}_{w}.json')) for w in 'AB'};L={w:raw[w]['log'] for w in 'AB'}
    alt=lambda e:{'epoch':e.get('state'),'margins':e['margins']['_alt']};base=(alt(L['A'][0]),alt(L['B'][0]))
    Dm=lambda a,b:float(AL.stats(a,b,base)['D'].mean());D00=Dm(base[0],base[1]);rows=[]
    for i in range(1,len(L['A'])):
        s=L['A'][i]['state'];assert L['B'][i]['state']==s
        dA=Dm(alt(L['A'][i]),base[1])-D00;dB=Dm(base[0],alt(L['B'][i]))-D00;D=Dm(alt(L['A'][i]),alt(L['B'][i]))
        can={w:max(L[w][i]['margins'][k]-L[w][0]['margins'][k] for k in ('_gen_nll','_tail_nll')) for w in 'AB'};dm=dA if MIS[sd]=='A' else dB
        # raw L3 margin changes (R1-alt, pair-mean over the L3 items), W0 reference
        mA=float(np.mean(L['A'][i]['margins']['_alt']['L3'])-np.mean(L['A'][0]['margins']['_alt']['L3']));mB=float(np.mean(L['B'][i]['margins']['_alt']['L3'])-np.mean(L['B'][0]['margins']['_alt']['L3']))
        rows.append(dict(s=s,D=D,dA=dA,dB=dB,sum_check=dA+dB-D,dM_A=mA,minus_dM_B=-mB,canary_A=can['A'],canary_B=can['B'],d_mis=dm,bound=D+2*abs(dm),damage_free=max(can.values())<=0.10))
    best=max((r for r in rows if r['damage_free']),key=lambda r:r['bound']);out[sd]={'mismatch_world':MIS[sd],'rows':rows,'max_bound':best['bound'],'at_s':best['s']}
    print(f"seed {sd} (mismatching world {MIS[sd]}):")
    for r in rows:print(f"  s={r['s']}: D {r['D']:+.3f} = dA {r['dA']:+.3f} + dB {r['dB']:+.3f} (chk {r['sum_check']:+.1e}); dM_A {r['dM_A']:+.3f}, -dM_B {r['minus_dM_B']:+.3f}; canary A/B {r['canary_A']:+.3f}/{r['canary_B']:+.3f}{'' if r['damage_free'] else ' DAMAGED'}; d_mis {r['d_mis']:+.3f}; bound {r['bound']:+.3f}")
    print(f"  max bound over damage-free s: {best['bound']:+.3f} at s={best['s']}")
json.dump(out,open(S.OUT/'c2_signbound.json','w'),indent=1)
