"""L1 variants (ctx, fill) vs plain L1 at epoch 8 (CPU). Rules: PROTOCOL_stage01.md, 'L1 variants'."""
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parent))
import analyze_stage01 as S1
AL=S1.AL
def main():
    md=['# L1 variants vs plain L1 (seed %d)\n'%S1.SEED,'| variant | D epoch 8 (R1) | D epoch 8 (R1-alt) | acc/flip | max canary rise | R vs plain [95% CI] | class |','|---|---|---|---|---|---|---|']
    base=S1.curve(*S1.load('L1','ck'))
    for tag in ('ctx','irr','fill'):
        X=S1.load('L1',tag)
        if not X:continue
        c=S1.curve(*X);ep,s,sa,_=c[8];mx=max(d for *_,d in c)
        R=S1.ratio(s['D'],base[8][1]['D']);cls='no effect' if (R[0]>=0.7 and R[1]>=0.5) else ('strong reduction' if (R[0]<=0.3 and R[2]<=0.5) else 'partial')
        md.append(f"| L1-{tag} | {AL.f3(AL.ci(s['D']))} | {AL.f3(AL.ci(sa['D']))} | {s['acc']:.2f}/{s['flip']:.2f} | {mx:+.3f} | {R[0]:.2f} [{R[1]:.2f}, {R[2]:.2f}] | {cls} |")
    out=S1.OUT/f'l1variants_report_s{S1.SEED}.md';out.write_text('\n'.join(md)+'\n');print('\n'.join(md))
if __name__=='__main__':main()
