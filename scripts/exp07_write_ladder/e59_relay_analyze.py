"""Analysis of job 1359 'relay' (e59_relay.py; CPU; new file; implementation thread 11/26; descriptive: mean over chunks inside a seed, then seeds with t interval and same-direction seeds; no verdict wording). Per weights (W0, P0s, P1s) x condition (A, B, C, D): place-token NLL of the 16 read facts (and differences to D), write-gradient cosines A-D / B-D / C-D / A-B / B-C and gradient norms; weight-independent: prefix-output distance of the fact-sentence rows to D, attention mass of the fact-sentence rows on earlier background (contaminated) keys / background after their own sentence / earlier same- and other-fact sentences / current chunk (layer mean), and of 64 background rows on earlier sentences.
  python e59_relay_analyze.py --dir RELAY_DIR"""
import sys,json,glob,numpy as np
from scipy import stats
D=sys.argv[sys.argv.index('--dir')+1];SEEDS=sorted(int(f.split('_s')[-1].split('.')[0]) for f in glob.glob(D+'/relay_s*.json'));R={s:json.load(open(f'{D}/relay_s{s}.json')) for s in SEEDS}
def tci(x):
    x=np.asarray(x,float);x=x[~np.isnan(x)]
    if len(x)<2:return f'{x.mean() if len(x) else float("nan"):8.4f}'
    m=x.mean();h=stats.t.ppf(.975,len(x)-1)*x.std(ddof=1)/np.sqrt(len(x));return f'{m:8.4f} [{m-h:7.4f},{m+h:7.4f}] {int(max((x>0).sum(),(x<0).sum()))}/{len(x)}'
CO=('A','B','C','D')
def per_seed(fn):return np.asarray([np.nanmean([fn(R[s],k) for k in R[s]['meas']]) for s in SEEDS])
print(f'1359 relay: seeds {SEEDS}; chunks {R[SEEDS[0]]["meas"]}; checks {[R[s]["checks"].get("chunked_natural_record_vs_record") for s in SEEDS][:1]}')
print('== place-token NLL (mean over the 16 read facts and their place tokens); difference to D')
for wn in ('W0','P0s','P1s'):
    nll=lambda r,k,cn:np.mean([np.mean(v) for v in r['weights'][wn]['chunks'][str(k)]['place_nll'][cn].values()]) if wn in r['weights'] else np.nan
    print(f'  {wn}: '+' | '.join(f'{cn} {tci(per_seed(lambda r,k:nll(r,k,cn)))}' for cn in CO))
    print(f'  {wn} minus D: '+' | '.join(f'{cn}-D {tci(per_seed(lambda r,k:nll(r,k,cn)-nll(r,k,"D")))}' for cn in ('A','B','C')))
print('== write-gradient cosine (total) and gradient norm')
for wn in ('W0','P0s','P1s'):
    cs=lambda r,k,p:r['weights'][wn]['chunks'][str(k)]['cos'][p]['total'] if wn in r['weights'] else np.nan
    print(f'  {wn}: '+' | '.join(f'{p} {tci(per_seed(lambda r,k:cs(r,k,p)))}' for p in ('A-D','B-D','C-D','A-B','B-C')))
    print(f'  {wn} |g|: '+' | '.join(f'{cn} {tci(per_seed(lambda r,k:r["weights"][wn]["chunks"][str(k)]["gnorm"][cn] if wn in r["weights"] else np.nan))}' for cn in CO))
print('== prefix-output distance of the fact-sentence rows to D (relative L2, mean over facts) | all rows')
for cn in ('A','B','C'):print(f'  {cn}: {tci(per_seed(lambda r,k:np.mean([v for f,v in r["prefix"][str(k)]["rep_rel_to_D"][cn].items() if f!="all_rows"])))} | {tci(per_seed(lambda r,k:r["prefix"][str(k)]["rep_rel_to_D"][cn]["all_rows"]))}')
print('== attention mass of the fact-sentence rows (mean over prefix layers first / mid / last, over facts)')
for nm in ('earlier_bg','earlier_bg_after_own','earlier_same_fact','earlier_other_fact','current_chunk'):
    print(f'  {nm:22s}: '+' | '.join(f'{cn} {tci(per_seed(lambda r,k:np.mean([np.mean(v[nm]) for f,v in r["prefix"][str(k)]["att"][cn].items() if f!="bg_rows"])))}' for cn in CO))
print('  background rows on earlier sentences: '+' | '.join(f'{cn} {tci(per_seed(lambda r,k:np.mean(r["prefix"][str(k)]["att"][cn]["bg_rows"]["earlier_sentences"])))}' for cn in CO))
