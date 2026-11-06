"""E57 mass-matched grid report (new file; CPU; job 1130, exploratory): per world the end-state (chunk 127) branch rank ratio with 95% fact-level CI for ALLQ256, M3 w=8, T05 W16, ABS2 W16 and the three mass-matched arms, and the paired differences (fact level) against T05 W16 and against ALLQ256 (improvement = rr_ref - rr_arm; positive = the arm is better), plus first-token dA, dA-dB.
  python3 e57_massgrid_report.py --seed S --out J | --selftest"""
import sys,json,argparse,pathlib
import numpy as np
HERE=pathlib.Path(__file__).resolve().parent;ROOT=HERE.parents[1];sys.path.insert(0,str(HERE))
EXP=ROOT/'experiments/exp07_write_ladder'
ARMS=['M3W1N05','M3W8N05','M3T05W16N05','M3TA2W16N05','M3T10W8P5N05','M3T15W6N05','M3W3P5N05']
def ci(x,B=2000,seed=1):
    x=np.asarray(x,float);ix=np.random.default_rng(seed).integers(0,len(x),(B,len(x)));m=x[ix].mean(1);return dict(mean=float(x.mean()),ci95=[float(np.percentile(m,2.5)),float(np.percentile(m,97.5))],n=int(len(x)))
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--seed',type=int);ap.add_argument('--out');ap.add_argument('--selftest',action='store_true');a=ap.parse_args()
    if a.selftest:x=ci([1,2,3]);assert abs(x['mean']-2)<1e-9;print('e57_massgrid_report selftest ok');return
    import e57_syn_tokread as TR
    sd=a.seed;d=json.load(open(EXP/f'e57_syn_massgrid_s{sd}/e57_syn_tokread_s{sd}_all.json'));rj=json.load(open(next(EXP.glob(f'e57_synq/E55_synqmg_*_s{sd}.read.json'))))['meta'];W0=TR.metrics(d['W0'],d['tokids'],rj);facts=sorted(W0);res={'seed':sd,'arms':{}};R={}
    for ar in ARMS:
        lab=f'{ar}@c127'
        if lab not in d['arms']:continue
        M=TR.metrics(d['arms'][lab],d['tokids'],rj);rr=np.array([M[f]['rank_branch']/16 for f in facts]);R[ar]=rr
        dA=np.array([M[f]['A_first']-W0[f]['A_first'] for f in facts]);dB=np.array([M[f]['B_first']-W0[f]['B_first'] for f in facts if M[f]['B_first'] is not None]);ok=[f for f in facts if M[f]['B_first'] is not None]
        dAB=np.array([(M[f]['A_first']-W0[f]['A_first'])-(M[f]['B_first']-W0[f]['B_first']) for f in ok])
        res['arms'][ar]=dict(rr_branch=ci(rr),first_dA=ci(dA),first_dA_minus_dB=ci(dAB),top1_branch=ci((rr<=1/16+1e-9).astype(float)))
    for ref in ('M3T05W16N05','M3W1N05'):
        for ar in R:
            if ar!=ref and ref in R:res['arms'][ar][f'improvement_vs_{ref}']=ci(R[ref]-R[ar])
    pathlib.Path(a.out).write_text(json.dumps(res,indent=1));print('wrote',a.out)
if __name__=='__main__':main()
