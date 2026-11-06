"""E57 online ABS2 rule report (new file; CPU; job 1150): the arm M3OA2W16N05 (deciding rows chosen at write time from the g128 of the CURRENT fast weights, g > 2 nats, loss weight 16, ALLQ256 on every row) next to the arms of the original-stream runs (jobs 1071 / 1073: ALLQ256 = M3W1N05, M3 w=8, top 5% w=16, ABS2 w=16 with the W0 g). End state (chunk 127): branch rank ratio, first-token dA, dA-dB with fact-level 95% CIs; paired differences online - W0-ABS2 and online - ALLQ256 (improvement = rr_ref - rr_arm; dA-dB: arm - ref); mean share of eligible rows selected by the online rule (saved.json) against the W0 rule.
  python3 e57_online_rule_report.py --seed S --out J | --selftest"""
import sys,json,argparse,pathlib
import numpy as np
HERE=pathlib.Path(__file__).resolve().parent;ROOT=HERE.parents[1];sys.path.insert(0,str(HERE))
EXP=ROOT/'experiments/exp07_write_ladder'
ORIG=['M3W1N05','M3W8N05','M3T05W16N05','M3TA2W16N05']
def ci(x,B=2000,seed=1):
    x=np.asarray(x,float);x=x[~np.isnan(x)];ix=np.random.default_rng(seed).integers(0,len(x),(B,len(x)));m=x[ix].mean(1);return dict(mean=float(x.mean()),ci95=[float(np.percentile(m,2.5)),float(np.percentile(m,97.5))],n=int(len(x)))
def vec(M,W0,facts):
    return dict(rr=np.array([M[f]['rank_branch']/16 for f in facts]),dA=np.array([M[f]['A_first']-W0[f]['A_first'] for f in facts]),dAB=np.array([(M[f]['A_first']-W0[f]['A_first'])-(M[f]['B_first']-W0[f]['B_first']) if M[f]['B_first'] is not None and W0[f]['B_first'] is not None else np.nan for f in facts]))
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--seed',type=int);ap.add_argument('--out');ap.add_argument('--selftest',action='store_true');a=ap.parse_args()
    if a.selftest:x=ci([1,2,3]);assert abs(x['mean']-2)<1e-9;print('e57_online_rule_report selftest ok');return
    import e57_syn_tokread as TR
    sd=a.seed;N=json.load(open(EXP/f'e57_syn_onlinerule_s{sd}/e57_syn_tokread_s{sd}.json'));O=json.load(open(EXP/f'e57_syn_rules_s{sd}/e57_syn_tokread_s{sd}.json'));assert N['tokids']==O['tokids']
    rj=json.load(open(next(EXP.glob(f'e57_synq/E55_synqog_*_s{sd}.read.json'))))['meta'];W0=TR.metrics(O['W0'],O['tokids'],rj);facts=sorted(W0);V={}
    for ar in ORIG:V[ar]=vec(TR.metrics(O['arms'][f'{ar}@c127'],O['tokids'],rj),W0,facts)
    V['M3OA2W16N05']=vec(TR.metrics(N['arms']['M3OA2W16N05@c127'],N['tokids'],rj),W0,facts)
    res={'seed':sd,'arms':{ar:{m:ci(v[m]) for m in v} for ar,v in V.items()},'paired':{}}
    for ref in ('M3TA2W16N05','M3W1N05'):
        res['paired'][f'online_minus_{ref}']=dict(rr_improvement=ci(V[ref]['rr']-V['M3OA2W16N05']['rr']),dAB=ci(V['M3OA2W16N05']['dAB']-V[ref]['dAB']),dA=ci(V['M3OA2W16N05']['dA']-V[ref]['dA']))
    sv=json.load(open(EXP/f'e57_synq/E55_synqog_M3OA2W16N05_s{sd}.saved.json'));res['online_rule_stats']=dict(mean_share_of_eligible_rows_selected=sv.get('online_share_mean'),mean_g128_eligible_online=sv.get('online_g_mean'))
    pathlib.Path(a.out).write_text(json.dumps(res,indent=1));print('wrote',a.out)
if __name__=='__main__':main()
