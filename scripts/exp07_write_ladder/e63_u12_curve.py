"""Matched binding vs exposures from the stored U12 continued-write runs (job 1328: E2E, 32 facts, arms NAT, SAME, SAME_U1, SAME_U2, SAME_U1U2 = C0, original clipped SGD, reads at 16/32/64/128 exposures; candidates = the 16 read places, no never-exposed candidate, so matched == old binding on average) (CPU; new file; thread-server 11/25). Per seed: mean over the 16 read facts of dS[f, gold] - mean over the other 15 questions of dS[q, place_f] (change over W0); seeds 231..238, mean [t] same-direction seeds; by sentence order is not available (the stream is the U12 world: orders recreated for chunks 15/31/63/127 are not stored). python e63_u12_curve.py --dir u12"""
import sys,json,numpy as np
from scipy import stats
a=sys.argv;D=a[a.index('--dir')+1];SEEDS=range(231,239);ARMS=('NAT','SAME','SAME_U1','SAME_U2','SAME_U1U2');PTS=('16','32','64','128')
def ci(d):d=np.asarray(d,float);m=d.mean();h=stats.t.ppf(.975,len(d)-1)*d.std(ddof=1)/np.sqrt(len(d));return f'{m:6.2f} [{m-h:5.2f},{m+h:5.2f}] {int(max((d>0).sum(),(d<0).sum()))}/{len(d)}'
def matched(S,W,cand,gold):
    dS=np.asarray(S,float)-np.asarray(W,float);out=[]
    for f in range(16):
        oth=[dS[q,cand[q].index(f)] for q in range(16) if q!=f];out.append(dS[f,gold[f]]-np.mean(oth))
    return np.mean(out)
print('1328 U12 continued writes: matched binding (= old binding here) by exposures 16 | 32 | 64 | 128; also familiarity is NOT available (no never-exposed candidates)')
V={}
for arm in ARMS:
    for pt in PTS:
        v=[]
        for s in SEEDS:
            d=json.load(open(f'{D}/E59_u12_{arm}_s{s}.read.json'));m=d['meta'];v.append(matched(d['read'][pt]['out'],d['W0']['out'],m['cand_fact'][:16],m['gold_idx'][:16]))
        V[(arm,pt)]=np.array(v)
    print(f'  {arm:10s} '+' | '.join(ci(V[(arm,pt)]) for pt in PTS))
print('  differences to SAME_U1U2 (= C0), seed-paired:')
for arm in ARMS[:-1]:print(f'  {arm:10s} - SAME_U1U2 '+' | '.join(ci(V[(arm,pt)]-V[('SAME_U1U2',pt)]) for pt in PTS))
