"""E57 online ABS2 rule, in-window and out-of-window reads (new file; CPU; job 1153): from the inline reads at a = 128 of the writes (read.json: 'in_g0' = the fact still inside the 8-chunk window, 'out' = outside; W0 reads in the same file): per arm the rank-ratio improvement (W0 - state) and the gold log-probability gain (state - W0) of the 16 facts with 95% CIs, canary rises, and the paired differences online - W0-ABS2 and online - ALLQ256. Arms: ALLQ256 = M3W1N05, M3 w=8, top 5% w=16, ABS2 w=16 with the W0 g (jobs 1071 / 1073, tag synqrl) and the online ABS2 (job 1153, tag synqoi).
  python3 e57_online_inwindow_report.py --seed S --out J | --selftest"""
import sys,json,argparse,pathlib
import numpy as np
HERE=pathlib.Path(__file__).resolve().parent;ROOT=HERE.parents[1]
EXP=ROOT/'experiments/exp07_write_ladder'
ARMS=[('synqrl','M3W1N05'),('synqrl','M3W8N05'),('synqrl','M3T05W16N05'),('synqrl','M3TA2W16N05'),('synqoi','M3OA2W16N05')]
def ci(x,B=2000,seed=1):
    x=np.asarray(x,float);ix=np.random.default_rng(seed).integers(0,len(x),(B,len(x)));m=x[ix].mean(1);return dict(mean=float(x.mean()),ci95=[float(np.percentile(m,2.5)),float(np.percentile(m,97.5))],n=int(len(x)))
def arrays(sc,w0,meta):
    kind=np.asarray(meta['kind']);gi=np.asarray(meta['gi']);rows=[r for r in range(len(kind)) if kind[r]=='fact'];sc=np.asarray(sc,float);w0=np.asarray(w0,float)
    rr=lambda m:np.array([(1+(m[r]>m[r][gi[r]]).sum())/m.shape[1] for r in rows])
    return dict(rr_impr=rr(w0)-rr(sc),gold_lp=np.array([sc[r][gi[r]]-w0[r][gi[r]] for r in rows]),rr=rr(sc))
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--seed',type=int);ap.add_argument('--out');ap.add_argument('--selftest',action='store_true');a=ap.parse_args()
    if a.selftest:x=ci([1,2,3]);assert abs(x['mean']-2)<1e-9;print('e57_online_inwindow_report selftest ok');return
    sd=a.seed;V={};res={'seed':sd,'arms':{},'paired':{}}
    for tag,ar in ARMS:
        x=json.load(open(EXP/f'e57_synq/E55_{tag}_{ar}_s{sd}.read.json'));rd=x['read']['128'];w=x['W0'];s=json.load(open(EXP/f'e57_synq/E55_{tag}_{ar}_s{sd}.saved.json'));cw=s['canary_w0'];k=max(s['reads'],key=int)
        V[ar]=dict(inw=arrays(rd['in_g0'],w['in_g0@128'],x['meta']),out=arrays(rd['out'],w['out'],x['meta']))
        res['arms'][ar]=dict(in_window={m:ci(v) for m,v in V[ar]['inw'].items()},out_of_window={m:ci(v) for m,v in V[ar]['out'].items()},canary=dict(gen_nll_rise=s['reads'][k]['canary']['_gen_nll']-cw['_gen_nll'],tail_nll_rise=s['reads'][k]['canary']['_tail_nll']-cw['_tail_nll']))
        if ar=='M3OA2W16N05':res['arms'][ar]['online_rule']=dict(mean_share_selected=s.get('online_share_mean'),mean_g128_online=s.get('online_g_mean'),end_state_hash=s['reads'][k]['_hash'])
    for ref in ('M3TA2W16N05','M3W1N05'):
        res['paired'][f'online_minus_{ref}']={part:{m:ci(V['M3OA2W16N05'][part][m]-V[ref][part][m]) for m in ('rr_impr','gold_lp')} for part in ('inw','out')}
    pathlib.Path(a.out).write_text(json.dumps(res,indent=1));print('wrote',a.out)
if __name__=='__main__':main()
