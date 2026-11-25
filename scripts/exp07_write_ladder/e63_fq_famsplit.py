"""FQ (E2E) and Qwen-FQ stored reads: familiarity / binding split (CPU; new file; thread-server 11/25). Items: 16 fact items (cand 0 = gold, cands 1..15 = the other exposed golds) + 4 ctrl items (cand 0 = UNexposed gold, cands 1..15 exposed). binding = mean over fact items of gold - mean(other 15 exposed golds); familiarity = mean over ctrl items of mean(exposed cands) - unexposed gold; headline = gold - max wrong. All as change over W0, 8 seeds 211..218, t intervals. Usage: python e63_fq_famsplit.py"""
import json,numpy as np
from scipy import stats
RP=('16','32','64','128');SEEDS=range(211,219)
def ci(d):
    d=np.asarray(d,float);m=d.mean();h=stats.t.ppf(.975,len(d)-1)*d.std(ddof=1)/np.sqrt(len(d));return f'{m:7.2f} [{m-h:6.2f},{m+h:6.2f}] {int(max((d>0).sum(),(d<0).sum()))}/{len(d)}'
def parts(S,m):
    S=np.asarray(S);k=np.array(m['kind']);F=S[k=='fact'];C=S[k=='ctrl']
    return np.array([np.mean(F[:,0]-F[:,1:].max(1)),np.mean(F[:,0]-F[:,1:].mean(1)),np.mean(C[:,1:].mean(1)-C[:,0])])
def load_e(arm,s):
    d=json.load(open(f'experiments/exp07_write_ladder/fq/run/E55_fq_{arm}_s{s}.read.json'));return {k:d['read'][k]['out'] for k in RP},{k:d['W0']['out'] for k in RP},d['meta']
def load_q(arm,s):
    d=json.load(open(f'experiments/exp09_qwen_ttt/fq_qwen/e59q_{arm}_s{s}.json'));w=json.load(open(f'experiments/exp09_qwen_ttt/fq_qwen/cband_W0_s{s}.json'))
    return {k:d['reads'][k]['out'] for k in RP},{k:w['reads'][k]['out'] for k in RP},d['meta']
def chg(L,rp):return np.array([parts(R[rp],m)-parts(W[rp],m) for R,W,m in L])
print('FQ stored reads, change over W0 (summed log p), 8 seeds; columns: headline gold-max | BINDING gold-exposed decoys | FAMILIARITY exposed-unexposed (ctrl items)')
arms={'E2E':('E2E',load_e,('NATS','NATMASK')),'Qwen':('Qwen',load_q,('NAT','BAND1024','BANDALL1024','CHK','ISO'))}
LD={}
for nm,(lab,ld,al) in arms.items():
    for a in al:
        LD[(nm,a)]=[ld(a,s) for s in SEEDS]
        for rp in RP:
            v=chg(LD[(nm,a)],rp);print(f'{lab:4s} {a:11s} read@{rp:>3s}  head {ci(v[:,0])} | bind {ci(v[:,1])} | fam {ci(v[:,2])}')
print('\nPaired differences (arm - reference), same seeds')
for nm,a,b in (('E2E','NATMASK','NATS'),('Qwen','BANDALL1024','NAT'),('Qwen','BAND1024','NAT'),('Qwen','CHK','NAT'),('Qwen','ISO','NAT')):
    for rp in RP:
        v=chg(LD[(nm,a)],rp)-chg(LD[(nm,b)],rp)
        print(f'{nm:4s} {a}-{b} @{rp:>3s}: head {ci(v[:,0])} | bind {ci(v[:,1])} | fam {ci(v[:,2])} | bind/fam of means {v[:,1].mean()/v[:,2].mean():.2f}')
