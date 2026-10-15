"""Exp07 L0 vs L1 analysis (CPU). Rules fixed in PROTOCOL.md."""
import os,sys,json
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2];OUT=Path(os.environ.get('EXP07_OUT',ROOT/'experiments/exp07_write_ladder'))
rng=np.random.default_rng(11)
SEED=int(sys.argv[1]) if len(sys.argv)>1 else 101;TAG1=sys.argv[2] if len(sys.argv)>2 else 'main'
def load(level,w,tag='main'):return json.load(open(OUT/f'{level}_{tag}_s{SEED}_{w}.json'))['log']
def points(level,tag=None):
    tag=tag or (TAG1 if level=='L1' else 'main')
    A=[e for e in load(level,'A',tag) if 'margins' in e];B=[e for e in load(level,'B',tag) if 'margins' in e]
    assert len(A)==len(B);return A,B
meta=json.load(open(OUT/f'meta_L1_{TAG1}_s{SEED}.json'));items=meta['items']
fam_idx={};i=0
for f,n in meta['fam_sizes'].items():fam_idx[f]=(i,i+n);i+=n
def pair_ids(f):a,b=fam_idx[f];return [items[k]['pair'] for k in range(a,b)]
def stats(eA,eB,base,pidx=None):
    """per-pair quantities at one evaluation point. base = (eA,eB) at the untrained state (familiarity gain reference)."""
    out={}
    def M(f):
        pa=pair_ids(f);mA=np.asarray(eA['margins'][f]);mB=np.asarray(eB['margins'][f]);d={}
        for p,x in zip(pa,(mA-mB)/2):d.setdefault(p,[]).append(x)
        ks=sorted(d,key=lambda s:int(s[1:]));return np.asarray([np.mean(d[k]) for k in ks])
    def Fg(f):
        pa=pair_ids(f);g=(np.asarray(eA['margins'][f])+np.asarray(eB['margins'][f]))/2-(np.asarray(base[0]['margins'][f])+np.asarray(base[1]['margins'][f]))/2;d={}
        for p,x in zip(pa,g):d.setdefault(p,[]).append(x)
        ks=sorted(d,key=lambda s:int(s[1:]));return np.asarray([np.mean(d[k]) for k in ks])
    out['D']=M('L3')-M('XP');out['M_L3']=M('L3');out['M_XP']=M('XP');out['M_PS']=M('PS');out['M_L3r']=M('L3r');out['M_L2']=M('L2')
    out['F_name']=Fg('FN');out['F_place']=Fg('FP');out['F_null']=Fg('FN0')
    mA=np.asarray(eA['margins']['L3']);mB=np.asarray(eB['margins']['L3'])
    out['acc']=float(((mA>0).mean()+(mB<0).mean())/2);out['flip']=float(((mA>0)&(mB<0)).mean())
    return out
def ci(v,n=4000):
    v=np.asarray(v);b=[v[rng.integers(0,len(v),len(v))].mean() for _ in range(n)];return float(v.mean()),float(np.percentile(b,2.5)),float(np.percentile(b,97.5))
def f3(t):return f'{t[0]:+.3f} [{t[1]:+.3f},{t[2]:+.3f}]'
def curve(level):
    A,B=points(level);base=(A[0],B[0]);return [(e.get('epoch',e.get('state')),stats(a,b,base)) for e,a,b in zip(A,A,B)]
def main():
    md=['# Exp07 write ladder: L0 (native) vs L1 (offline, isolated)\n'];res={}
    for lev in ('L0','L1'):
        try:c=curve(lev)
        except FileNotFoundError:md.append(f'{lev}: missing');continue
        res[lev]=c;md.append(f'\n## {lev}\n\n| point | familiarity gain (name, nat) | binding D | acc | flip | M_L3r | M_L2 |\n|---|---|---|---|---|---|---|')
        for t,s in c:md.append(f"| {t} | {s['F_name'].mean():+.2f} | {f3(ci(s['D']))} | {s['acc']:.2f} | {s['flip']:.2f} | {s['M_L3r'].mean():+.3f} | {s['M_L2'].mean():+.3f} |")
    if 'L0' in res and 'L1' in res:
        s0=res['L0'][-1][1];F0=s0['F_name'].mean();c1=res['L1']
        Fs=np.asarray([s['F_name'].mean() for _,s in c1])
        # matched familiarity: per-pair linear interpolation of L1's D at L0's final familiarity gain (first crossing)
        j=next((k for k in range(1,len(Fs)) if Fs[k]>=F0),None)
        if j is None:md.append(f'\nL1 never reaches L0 familiarity gain {F0:+.2f}; matched comparison not defined (report max-epoch values).')
        else:
            w_=(F0-Fs[j-1])/max(Fs[j]-Fs[j-1],1e-9);D1=(1-w_)*c1[j-1][1]['D']+w_*c1[j][1]['D'];diff=D1-s0['D']
            res['matched']={'F0':F0,'D_L1':ci(D1),'D_L0':ci(s0['D']),'diff':ci(diff)}
            md.append(f"\n## Primary: binding at matched familiarity gain (F0 = {F0:+.2f} nat)\n- L0 D {f3(ci(s0['D']))}; L1 D (interpolated) {f3(ci(D1))}; **L1 - L0 {f3(ci(diff))}**")
        best=max(c1,key=lambda x:x[1]['D'].mean());res['positive_control']={'epoch':best[0],'D':ci(best[1]['D']),'acc':best[1]['acc'],'flip':best[1]['flip']}
        md.append(f"- positive control (L1, best epoch {best[0]}): D {f3(ci(best[1]['D']))}, acc {best[1]['acc']:.2f}, flip {best[1]['flip']:.2f}")
        m=res.get('matched');pc=res['positive_control']
        L1_writes=(m is not None and m['diff'][1]>0) or (pc['acc']>=0.8 and pc['flip']>=0.5)
        L1_fails=(pc['D'][1]<=0 or pc['acc']<=0.6) and (m is None or m['diff'][1]<=0)
        verdict='L1 SUCCESS: binding is writable at this location with isolated offline updates -> TTT-specific factor; next: one-factor ladder between L0 and L1' if L1_writes else ('L1 FAILURE: NTP writes no binding at this location even offline -> L2 (all-layer MLP) and L3 (NTP-LoRA-TTT, no meta-learning)' if L1_fails else 'INCONCLUSIVE (between thresholds): report, do not branch')
        res['verdict']=verdict;md.append(f'\n**Verdict (pre-registered rules): {verdict}**')
    json.dump(res,open(OUT/f'ladder_metrics_s{SEED}_{TAG1}.json','w'),default=lambda o:o.tolist() if hasattr(o,'tolist') else str(o),indent=1)
    (OUT/f'ladder_report_s{SEED}_{TAG1}.md').write_text('\n'.join(md)+'\n');print('\n'.join(md))
if __name__=='__main__':main()
