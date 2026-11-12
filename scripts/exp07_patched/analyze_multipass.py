"""Applies PROTOCOL_multipass.md (CPU). Usage: analyze_multipass.py [world_seed]"""
import os,sys,json
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parent))
import analyze_ladder as AL
OUT=AL.OUT;SEED=int(sys.argv[1]) if len(sys.argv)>1 else 101;AL.SEED=SEED
rng=np.random.default_rng(13)
def load(level,tag):
    fa=OUT/f'{level}_{tag}_s{SEED}_A.json';fb=OUT/f'{level}_{tag}_s{SEED}_B.json'
    if not (fa.exists() and fb.exists()):return None
    A=[e for e in json.load(open(fa))['log'] if 'margins' in e];B=[e for e in json.load(open(fb))['log'] if 'margins' in e]
    return A,B
def series(A,B,key):
    base=(A[0],B[0]);g=lambda e,k:e['margins'].get(k,np.nan)
    return [(e.get(key),AL.stats(a,b,base),(max(g(a,'_gen_nll')-g(A[0],'_gen_nll'),g(a,'_tail_nll')-g(A[0],'_tail_nll')),max(g(b,'_gen_nll')-g(B[0],'_gen_nll'),g(b,'_tail_nll')-g(B[0],'_tail_nll'))),a.get('cum_update_norm')) for e,a,b in zip(A,A,B)]
def main():
    md=[f'# Exp07 multi-pass / intensity (seed {SEED}), rules: PROTOCOL_multipass.md\n'];res={}
    L1=load('L1','main');MP=load('L0mp','main');L0=load('L0','main')
    # rule 1: sanity
    X1=load('L0x','x1');ok=None
    if X1 and L0:
        d=max(abs(x-y) for f in AL.fam_idx for (x,y) in zip(X1[0][-1]['margins'][f]+X1[1][-1]['margins'][f],L0[0][-1]['margins'][f]+L0[1][-1]['margins'][f]))
        ok=d<=0.01;res['sanity_maxdiff']=d;md.append(f'## Rule 1 sanity: L0x1 vs native L0 final, max |item margin diff| {d:.4f} -> {"PASS" if ok else "FAIL (no interpretation)"}\n')
    else:md.append('## Rule 1 sanity: missing L0x1 or L0 -> not interpretable yet\n')
    # rule 2/3: exposure axis
    if L1 and MP:
        s1={e:s for e,s,_,_ in series(*L1,'epoch')};sm={e:s for e,s,_,_ in series(*MP,'epoch')}
        md.append('## Exposure axis (16k exposures per triple)\n\n| k | L1 D | L1 acc/flip | L0mp D | L0mp acc/flip |\n|---|---|---|---|---|')
        for k in sorted(sm):
            if k in s1:md.append(f"| {k} | {AL.f3(AL.ci(s1[k]['D']))} | {s1[k]['acc']:.2f}/{s1[k]['flip']:.2f} | {AL.f3(AL.ci(sm[k]['D']))} | {sm[k]['acc']:.2f}/{sm[k]['flip']:.2f} |")
        if 8 in sm and 8 in s1:
            a=sm[8]['D'];b=s1[8]['D'];n=len(a);bs=[]
            for _ in range(4000):
                i=rng.integers(0,n,n);bs.append(a[i].mean()/b[i].mean())
            R=a.mean()/b.mean();lo,hi=np.percentile(bs,[2.5,97.5]);res['R']=(R,lo,hi)
            v='EXPOSURE COUNT' if (R>=0.7 and lo>=0.5) else ('TTT CONDITIONS BLOCK' if (R<=0.3 and hi<=0.5) else 'MIXED')
            md.append(f'\n**Rule 3: R = D(L0mp,8)/D(L1,8) = {R:.2f} [{lo:.2f}, {hi:.2f}] -> {v}**' + ('' if ok else ' (sanity not passed: NOT interpreted)'))
            res['rule3']=v
    # rule 4/5: intensity
    md.append('\n## Intensity (one pass, step x s)\n\n| run | final D | acc/flip | familiarity gain | max canary NLL rise | cum update norm | overshoot |\n|---|---|---|---|---|---|---|')
    for tag in ('x1','x2','x4','x8'):
        X=load('L0x',tag)
        if not X:continue
        sr=series(*X,'state');g0=0.0;gmax=max(max(g) for _,_,g,_ in sr);fin=sr[-1][1];over=gmax>0.10   # max rise over W0 of either canary, either world
        res[tag]={'D':AL.ci(fin['D']),'over':bool(over),'gen0':g0,'genmax':gmax}
        md.append(f"| L0{tag} | {AL.f3(AL.ci(fin['D']))} | {fin['acc']:.2f}/{fin['flip']:.2f} | {fin['F_name'].mean():+.2f} | {gmax:+.3f} | {sr[-1][3]:.1f} | {'YES' if over else 'no'} |")
    if MP:
        sr=series(*MP,'epoch');md.append(f"\nL0mp max canary NLL rise at pass 8 {max(sr[-1][2]):+.3f}; cum update norm at pass 8 {sr[-1][3]:.1f}")
    json.dump(res,open(OUT/f'multipass_metrics_s{SEED}.json','w'),default=lambda o:o.tolist() if hasattr(o,'tolist') else str(o),indent=1)
    (OUT/f'multipass_report_s{SEED}.md').write_text('\n'.join(md)+'\n');print('\n'.join(md))
if __name__=='__main__':main()
