"""Applies PROTOCOL_stage01.md (CPU). Usage: analyze_stage01.py [world_seed]"""
import os,sys,json
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parent))
import analyze_ladder as AL
OUT=AL.OUT;SEED=int(sys.argv[1]) if len(sys.argv)>1 else 101;AL.SEED=SEED
rng=np.random.default_rng(17);THR=0.10
def load(level,tag):
    fa=OUT/f'{level}_{tag}_s{SEED}_A.json';fb=OUT/f'{level}_{tag}_s{SEED}_B.json'
    if not (fa.exists() and fb.exists()):return None
    A=[e for e in json.load(open(fa))['log'] if 'margins' in e];B=[e for e in json.load(open(fb))['log'] if 'margins' in e];return A,B
def alt(e):return {'epoch':e.get('epoch',e.get('state')),'margins':e['margins']['_alt']}
def curve(A,B):
    """per evaluation point: epoch, stats under R1, stats under R1-alt, max canary rise (either canary, either world)."""
    out=[];b1=(A[0],B[0]);b2=(alt(A[0]),alt(B[0]))
    for a,b in zip(A,B):
        d=max(a['margins'][k]-A[0]['margins'][k] for k in ('_gen_nll','_tail_nll'));d=max(d,max(b['margins'][k]-B[0]['margins'][k] for k in ('_gen_nll','_tail_nll')))
        out.append((a.get('epoch',a.get('state')),AL.stats(a,b,b1),AL.stats(alt(a),alt(b),b2) if '_alt' in a['margins'] else None,d))
    return out
def ratio(x,y):
    n=len(x);bs=[]
    for _ in range(4000):
        i=rng.integers(0,n,n);bs.append(x[i].mean()/y[i].mean())
    return float(x.mean()/y.mean()),float(np.percentile(bs,2.5)),float(np.percentile(bs,97.5))
def main():
    md=[f'# Exp07 stage 0/1 (seed {SEED}), rules: PROTOCOL_stage01.md\n'];res={}
    L1=load('L1','ck');c1=curve(*L1) if L1 else None
    if c1:
        md.append('## 0a. L1 damage trajectory\n\n| epoch | D (R1) | D (R1-alt) | acc/flip | max canary rise |\n|---|---|---|---|---|')
        for ep,s,sa,d in c1:md.append(f"| {ep} | {AL.f3(AL.ci(s['D']))} | {AL.f3(AL.ci(sa['D'])) if sa else '-'} | {s['acc']:.2f}/{s['flip']:.2f} | {d:+.3f} |")
        first=next(((ep,d) for ep,s,_,d in c1 if AL.ci(s['D'])[1]>0.5),None)
        if first:v='L1 reached binding WITHOUT damage' if first[1]<=THR else 'L1 binding came WITH damage';md.append(f'\n**0a verdict: {v}** (first epoch with D CI lower > 0.5: {first[0]}, canary rise {first[1]:+.3f}); max rise over all epochs {max(d for *_,d in c1):+.3f}')
        else:v='L1 never reached D CI lower > 0.5';md.append(f'\n**0a verdict: {v}**')
        res['0a']=v
        w0=c1[0];md.append(f"\nR1 vs R1-alt agreement: pair-level corr of D at epoch 8 {np.corrcoef(c1[8][1]['D'],c1[8][2]['D'])[0,1]:+.2f}, epoch 32 {np.corrcoef(c1[-1][1]['D'],c1[-1][2]['D'])[0,1]:+.2f}")
    MP=load('L0mp','alt')
    if MP and c1:
        cm=curve(*MP);md.append('\n## 0b. L0mp read in an undamaged context\n\n| pass | D (R1) | D (R1-alt) | max canary rise |\n|---|---|---|---|')
        for ep,s,sa,d in cm:md.append(f"| {ep} | {AL.f3(AL.ci(s['D']))} | {AL.f3(AL.ci(sa['D']))} | {d:+.3f} |")
        R=ratio(cm[8][2]['D'],c1[8][2]['D']);res['Ralt']=R
        v='native repetition does NOT write binding (robust to readout damage)' if (R[0]<0.3 and R[2]<0.5) else ('binding written but unreadable in the damaged R1 context' if (R[0]>=0.7 and R[1]>=0.5) else 'mixed')
        md.append(f'\n**0b verdict: Ralt = {R[0]:.2f} [{R[1]:.2f}, {R[2]:.2f}] -> {v}**');res['0b']=v
    for lev,name in (('L0or','1a. oracle exposure-only re-reads'),('L0fs','1b. filler swap')):
        X=load(lev,'main')
        if not (X and c1):continue
        cx=curve(*X);md.append(f'\n## {name}\n\n| pass | D (R1) | D (R1-alt) | acc/flip | max canary rise |\n|---|---|---|---|---|')
        for ep,s,sa,d in cx:md.append(f"| {ep} | {AL.f3(AL.ci(s['D']))} | {AL.f3(AL.ci(sa['D']))} | {s['acc']:.2f}/{s['flip']:.2f} | {d:+.3f} |")
        ks=0
        for ep,_,_,d in cx[1:]:
            if d>THR:break
            ks=ep
        if ks<2:v='damage-limited (no verdict)';md.append(f'\n**{lev} verdict: {v}** (last undamaged pass {ks})')
        else:
            R=ratio(cx[ks][1]['D'],c1[ks][1]['D']);v='reproduces offline binding' if (R[0]>=0.7 and R[1]>=0.5) else ('does NOT reproduce offline binding' if (R[0]<=0.3 and R[2]<=0.5) else 'mixed')
            md.append(f'\n**{lev} verdict: R* (pass {ks}) = {R[0]:.2f} [{R[1]:.2f}, {R[2]:.2f}] -> {v}**');res[lev+'_R']=R
            cls='recovery' if R[0]>=0.30 else ('unchanged' if R[0]<=0.10 else 'partial')
            if MP:
                cm_=curve(*MP);dm=AL.ci(cm_[ks][1]['D']);dv=cx[ks][1]['D'].mean()
                if dv<dm[0]-(dm[2]-dm[1])/2:cls='worse'
            md.append(f'- prediction-table class (amendment 13:00 KST): **{cls}**');res[lev+'_class']=cls
        res[lev]=v
    json.dump(res,open(OUT/f'stage01_metrics_s{SEED}.json','w'),default=str,indent=1)
    (OUT/f'stage01_report_s{SEED}.md').write_text('\n'.join(md)+'\n');print('\n'.join(md))
if __name__=='__main__':main()
