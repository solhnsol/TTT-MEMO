"""Summary of job 1365 'l2diag' (CPU; new file; implementation thread 11/28; descriptive). Per measure: L2 and L2b and L2 - L2b, mean over chunks inside a seed, then seeds (t interval, same-direction seeds); per chunk as well.
  python e59_l2diag_analyze.py --dir L2DIAG_DIR"""
import sys,json,glob,numpy as np
from scipy import stats
D=sys.argv[sys.argv.index('--dir')+1];R={int(f.split('_s')[-1].split('.')[0]):json.load(open(f)) for f in sorted(glob.glob(D+'/l2diag_s*.json'))};SEEDS=sorted(R)
def tci(x):
    x=np.asarray(x,float);x=x[~np.isnan(x)]
    if len(x)<2:return f'{x.mean() if len(x) else float("nan"):8.4f}'
    m=x.mean();h=stats.t.ppf(.975,len(x)-1)*x.std(ddof=1)/np.sqrt(len(x));return f'{m:8.4f} [{m-h:7.4f},{m+h:7.4f}] {int(max((x>0).sum(),(x<0).sum()))}/{len(x)}'
M=[('att other_place','att_prefix_mean_over_layers','other_place'),('att other_name','att_prefix_mean_over_layers','other_name'),('att other_rest','att_prefix_mean_over_layers','other_rest'),('att own_sentence','att_prefix_mean_over_layers','own_sentence'),('att window','att_prefix_mean_over_layers','window'),('att other_current','att_prefix_mean_over_layers','other_current'),
   ('cos place rows between facts','prefix_out_cos_place_rows_between_facts',None),('cos place row to own name','cos_place_row_to_own_name_row',None),('cos place row to other names','cos_place_row_to_other_name_rows',None),('grad cos between facts','grad_cos_between_facts_mean',None)]
get=lambda e,mode,k1,k2:e[mode][k1][k2] if k2 else e[mode][k1]
print(f'1365 l2diag: seeds {SEEDS}; chunks {R[SEEDS[0]]["chunks"]}')
for nm,k1,k2 in M:
    v={mode:np.array([np.mean([get(e,mode,k1,k2) for e in R[s]['per_chunk'].values()]) for s in SEEDS]) for mode in ('L2','L2b')}
    print(f'  {nm:30s}: L2 {tci(v["L2"])} | L2b {tci(v["L2b"])} | L2-L2b {tci(v["L2"]-v["L2b"])}')
print('== per chunk, L2 - L2b (mean over seeds): attention on other place tokens | grad cos between facts | cos place rows between facts')
for k in R[SEEDS[0]]['chunks']:
    f=lambda k1,k2:np.mean([get(R[s]['per_chunk'][str(k)],'L2',k1,k2)-get(R[s]['per_chunk'][str(k)],'L2b',k1,k2) for s in SEEDS])
    print(f'  chunk {k:3d}: {f("att_prefix_mean_over_layers","other_place"):8.4f} | {f("grad_cos_between_facts_mean",None):8.4f} | {f("prefix_out_cos_place_rows_between_facts",None):8.4f}')
if all('cross' in R[s] for s in SEEDS):
    print('== cross influence, one L1 step at chunk 0 from W0, then R1: gold rise | control (same name under other facts) | matched binding | exposed wrong | never exposed')
    for mode in ('L2','L2b'):print(f'  {mode:3s}: '+' | '.join(tci([R[s]['cross'][mode][k] for s in SEEDS]) for k in ('gold_rise','control_rise','matched_binding','exposed_wrong_rise','never_exposed')))
    print('  L2 - L2b matched binding: '+tci([R[s]['cross']['L2']['matched_binding']-R[s]['cross']['L2b']['matched_binding'] for s in SEEDS]))
