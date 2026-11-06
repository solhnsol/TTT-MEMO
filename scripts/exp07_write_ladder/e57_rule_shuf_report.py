"""E57 pair-shuffle control for the rule arms, synthetic report (new file; CPU; job 1140): states written on the pair-shuffled stream (tag synqsr; arms ALLQ256 = M3W1N05, M3 w=8, top 5% w=16 = M3T05W16N05, ABS2 w=16 = M3TA2W16N05; masks from the g128 of the SHUFFLED stream) scored with the ORIGINAL pairs, next to the same arms on the original stream (jobs 1071-1073, e57_syn_rules_s<seed>). Per arm: rank ratio at the divergence token, first-token dA, dA-dB (shuffled, original, original minus shuffled, fact-level paired 95% CI), and for the rule arms the gain over ALLQ256 (rr: rr_ALLQ - rr_arm; dA-dB: arm - ALLQ) on both streams and the shuffled-minus-original change of the gain.
  python3 e57_rule_shuf_report.py --seed S --out J | --selftest"""
import sys,json,argparse,pathlib
import numpy as np
HERE=pathlib.Path(__file__).resolve().parent;ROOT=HERE.parents[1];sys.path.insert(0,str(HERE))
EXP=ROOT/'experiments/exp07_write_ladder'
ARMS=['M3W1N05','M3W8N05','M3T05W16N05','M3TA2W16N05']
def ci(x,B=2000,seed=1):
    x=np.asarray(x,float);x=x[~np.isnan(x)];ix=np.random.default_rng(seed).integers(0,len(x),(B,len(x)));m=x[ix].mean(1);return dict(mean=float(x.mean()),ci95=[float(np.percentile(m,2.5)),float(np.percentile(m,97.5))],n=int(len(x)))
def vec(M,W0,facts):
    rr=np.array([M[f]['rank_branch']/16 for f in facts]);dA=np.array([M[f]['A_first']-W0[f]['A_first'] for f in facts])
    dAB=np.array([(M[f]['A_first']-W0[f]['A_first'])-(M[f]['B_first']-W0[f]['B_first']) if M[f]['B_first'] is not None and W0[f]['B_first'] is not None else np.nan for f in facts]);return dict(rr=rr,dA=dA,dAB=dAB)
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--seed',type=int);ap.add_argument('--out');ap.add_argument('--selftest',action='store_true');a=ap.parse_args()
    if a.selftest:x=ci([1,2,3]);assert abs(x['mean']-2)<1e-9;print('e57_rule_shuf_report selftest ok');return
    import e57_syn_tokread as TR
    sd=a.seed;S_=json.load(open(EXP/f'e57_syn_ruleshuf_s{sd}/e57_syn_tokread_s{sd}.json'));O_=json.load(open(EXP/f'e57_syn_rules_s{sd}/e57_syn_tokread_s{sd}.json'));assert S_['tokids']==O_['tokids']
    rj=json.load(open(next(EXP.glob(f'e57_synq/E55_synqsr_*_s{sd}.read.json'))))['meta'];W0s=TR.metrics(S_['W0'],S_['tokids'],rj);W0o=TR.metrics(O_['W0'],O_['tokids'],rj);facts=sorted(W0s);V={'shuf':{},'orig':{}}
    for ar in ARMS:
        lab=f'{ar}@c127';V['shuf'][ar]=vec(TR.metrics(S_['arms'][lab],S_['tokids'],rj),W0s,facts);V['orig'][ar]=vec(TR.metrics(O_['arms'][lab],O_['tokids'],rj),W0o,facts)
    res={'seed':sd,'arms':{},'gain_over_ALLQ256':{}}
    for ar in ARMS:
        res['arms'][ar]={m:dict(shuffled=ci(V['shuf'][ar][m]),original=ci(V['orig'][ar][m]),original_minus_shuffled=ci(V['orig'][ar][m]-V['shuf'][ar][m])) for m in ('rr','dA','dAB')}
    for ar in ARMS[1:]:
        g={}
        for st in ('shuf','orig'):
            g[st]=dict(rr=V[st]['M3W1N05']['rr']-V[st][ar]['rr'],dAB=V[st][ar]['dAB']-V[st]['M3W1N05']['dAB'],dA=V[st][ar]['dA']-V[st]['M3W1N05']['dA'])
        res['gain_over_ALLQ256'][ar]={m:dict(shuffled=ci(g['shuf'][m]),original=ci(g['orig'][m]),shuffled_minus_original=ci(g['shuf'][m]-g['orig'][m])) for m in ('rr','dAB','dA')}
    pathlib.Path(a.out).write_text(json.dumps(res,indent=1));print('wrote',a.out)
if __name__=='__main__':main()
