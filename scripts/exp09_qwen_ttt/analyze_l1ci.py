"""Qwen L1-ctx / L1-irr v1 (PROTOCOL 2026-10-17 01:45 KST + amendments): verdicts from the registered rules.
Usage: analyze_l1ci.py OUT SEED:DIR ...  (DIR holds L1_{plain,ctx,irr}_s{seed}_{A,B}.json and manip_s{seed}.json)"""
import os,sys,json
from pathlib import Path
import numpy as np
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
def load(seed,d):
    d=Path(d);ref=d/f'meta_L1_l1items_s{seed}.json'
    if not ref.exists():ref.write_text(next(d.glob(f'meta_L1_plain_s{seed}.json')).read_text())
    os.environ['EXP07_OUT']=str(d);saved=sys.argv;sys.argv=['x',str(seed),'l1items']
    for m in ('analyze_ladder','analyze_stage01'):sys.modules.pop(m,None)
    sys.path.insert(0,str(ROOT/'scripts/exp07_write_ladder'))
    try:import analyze_ladder as AL,analyze_stage01 as S
    finally:sys.argv=saved
    out={}
    for v in ('plain','ctx','irr'):
        X=S.load('L1',v);c=S.curve(*X);raw={w:json.load(open(d/f'L1_{v}_s{seed}_{w}.json'))['log'] for w in 'AB'}
        out[v]={'pts':{ep:(s,sa,dd) for ep,s,sa,dd in c},'raw':raw}
    return AL,out
def rboot(num,den,n=8000,lvl=95,seed=0):
    r=np.random.default_rng(seed);b=[];k=len(num)
    for _ in range(n):i=r.integers(0,k,k);b.append(num[i].mean()/den[i].mean())
    a=(100-lvl)/2;return float(num.mean()/den.mean()),float(np.percentile(b,a)),float(np.percentile(b,100-a))
def boot(v,n=8000,lvl=95,seed=0):
    r=np.random.default_rng(seed);b=[v[r.integers(0,len(v),len(v))].mean() for _ in range(n)];a=(100-lvl)/2
    return float(np.mean(v)),float(np.percentile(b,a)),float(np.percentile(b,100-a))
def f3(t):return f'{t[0]:+.3f} [{t[1]:+.3f},{t[2]:+.3f}]'
def main():
    out=Path(sys.argv[1]);specs=[s.split(':') for s in sys.argv[2:]];R={};md=['# Qwen L1-ctx / L1-irr (full-MLP, lr 0.1)\n']
    data={int(s):load(int(s),d) for s,d in specs}
    AL,X=data[101];pl=X['plain']['pts'];estar=None
    for ep in (8,16,24,32):
        ci=AL.ci(pl[ep][1]['D'])
        md.append(f'- plain s101 D_alt ep{ep} {f3(ci)}')
        if estar is None and ci[1]>0.5:estar=ep
    md.append(f'\n**e* = {estar}** (first readout epoch with plain s101 D_alt CI lower > +0.5; fixed before any variant is read)\n');R['estar']=estar
    if estar is None:print('\n'.join(md));return
    irrmctx=[];verd={'ctx_strong':0,'irr_noeffect':0,'irr_strong':0,'dissoc':0}
    for sd,(AL,X) in data.items():
        md.append(f'\n## seed {sd}');R[sd]={}
        for v in ('ctx','irr'):
            Dv=X[v]['pts'][estar][1]['D'];Dp=X['plain']['pts'][estar][1]['D']
            r=rboot(Dv,Dp);d16v=X[v]['pts'][16][1]['D'].mean();d32v=X[v]['pts'][32][1]['D'].mean();d16p=X['plain']['pts'][16][1]['D'].mean();d32p=X['plain']['pts'][32][1]['D'].mean()
            dR=(d32v-d16v)/(d32p-d16p) if d32p!=d16p else float('nan')
            Fv=X[v]['pts'][estar][0]['F_name'].mean()-X[v]['pts'][estar][0]['F_null'].mean();Fp=X['plain']['pts'][estar][0]['F_name'].mean()-X['plain']['pts'][estar][0]['F_null'].mean()
            fam_red=1-Fv/Fp;bind_red=1-r[0];dis=bind_red>=2*fam_red
            strong=r[0]<=0.30 and r[2]<=0.50;noeff=r[0]>=0.70 and r[1]>=0.50
            can_v=X[v]['pts'][estar][2];can_p=X['plain']['pts'][estar][2];cr=can_v/can_p if can_p else float('nan')
            um=X[v]['pts'][8][1]['D'].mean()/X['plain']['pts'][24][1]['D'].mean()
            steps={w:max(e.get('step',0) for e in X[v]['raw'][w] if 'ce' in e and 'margins' not in e) for w in 'AB'}
            cl=[e['clip_coef'] for w in 'AB' for e in X[v]['raw'][w] if 'clip_coef' in e and 'margins' not in e];cf=sum(c<1 for c in cl)/max(len(cl),1)
            cum={w:[e for e in X[v]['raw'][w] if 'margins' in e][-1].get('cum_update_norm') for w in 'AB'}
            R[sd][v]={'R':r,'dR_16_32':dR,'strong':strong,'noeffect':noeff,'fam_reduction':fam_red,'bind_reduction':bind_red,'dissociation':dis,'canary':(can_v,can_p,cr),'update_matched_R':um,'steps32':steps,'clip_bind_frac':cf,'cum_norm':cum}
            md.append(f"- {v}: R(e*={estar}) {f3(r)}; strong reduction {strong}; no effect {noeff}; DeltaR(16->32) {dR:+.3f}; binding reduction {bind_red:+.2f} vs familiarity reduction {fam_red:+.2f} -> dissociation {dis}; canary var/plain {can_v:+.3f}/{can_p:+.3f} (ratio {cr:.2f}{' OUTSIDE [0.5,2]' if not (0.5<=cr<=2) else ''}); update-matched R (var ep8 / plain ep24) {um:+.3f}; steps at ep32 {steps}; clip binds {cf:.3f}; cum norm {cum}")
            if v=='ctx':verd['ctx_strong']+=strong;verd['dissoc']+=dis
            else:verd['irr_noeffect']+=noeff;verd['irr_strong']+=strong
        irrmctx.append(X['irr']['pts'][estar][1]['D']-X['ctx']['pts'][estar][1]['D'])
        man=json.load(open(Path(dict((int(s),d) for s,d in specs)[sd])/f'manip_s{sd}.json'))
        R[sd]['manip']=[r_['ctx']-r_['irr'] for w in 'AB' for r_ in man[w]]
    ic=boot(np.concatenate(irrmctx));mp=boot(np.concatenate([np.asarray(R[sd]['manip']) for sd in data]))
    ctx_pass=verd['ctx_strong']>=2;irr_pass=verd['irr_noeffect']>=2 and verd['irr_strong']==0 and ic[1]>0;manip_ok=mp[2]<0
    md.append(f"\n## Verdict\n- manipulation check (W0 place-token NLL ctx - irr, pooled sentences, 95% CI): {f3(mp)} -> {'valid shadowing test' if manip_ok else 'NOT interpreted as a shadowing test (CI upper not below 0)'}")
    md.append(f"- L1-ctx strong reduction on {verd['ctx_strong']}/3 seeds -> {'PASS' if ctx_pass else 'FAIL'}")
    md.append(f"- L1-irr: no effect on {verd['irr_noeffect']}/3, strong on {verd['irr_strong']}/3; D_alt(irr) - D_alt(ctx) at e*, 45 pairs, 95% CI {f3(ic)} -> {'PASS' if irr_pass else 'FAIL'}")
    md.append(f"- dissociation (binding reduction >= 2 x familiarity reduction) on {verd['dissoc']}/3 seeds (ctx) -> {'yes' if verd['dissoc']>=2 else 'no'}")
    R.update(verdict=verd,irr_minus_ctx=ic,manip=mp,ctx_pass=ctx_pass,irr_pass=irr_pass,manip_valid=manip_ok)
    out.mkdir(parents=True,exist_ok=True);json.dump(R,open(out/'l1ci_result.json','w'),indent=1,default=str);(out/'l1ci_report.md').write_text('\n'.join(md)+'\n');print('\n'.join(md))
if __name__=='__main__':main()
