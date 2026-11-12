"""Stage 3 K sweep table (CPU). Usage: analyze_ksweep.py [world_seed]"""
import sys,json
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parent))
import analyze_stage01 as S1
AL=S1.AL
def main():
    md=['# Exp07 stage 3: K inner steps per chunk (seed %d)\n'%S1.SEED,'| K | final D (R1) | final D (R1-alt) | acc/flip | familiarity gain | max canary rise | cum update norm | overshoot |','|---|---|---|---|---|---|---|---|']
    for K in (1,4):
        X=S1.load('L0k',f'k{K}')
        if not X:continue
        A,B=X
        cx=S1.curve(A,B);ep,s,sa,_=cx[-1];mx=max(d for *_,d in cx)
        md.append(f"| {K} | {AL.f3(AL.ci(s['D']))} | {AL.f3(AL.ci(sa['D']))} | {s['acc']:.2f}/{s['flip']:.2f} | {s['F_name'].mean():+.2f} | {mx:+.3f} | {A[-1].get('cum_update_norm',float('nan')):.1f} | {'YES' if mx>0.10 else 'no'} |")
    out=S1.OUT/f'ksweep_report_s{S1.SEED}.md';out.write_text('\n'.join(md)+'\n');print('\n'.join(md))
if __name__=='__main__':main()
