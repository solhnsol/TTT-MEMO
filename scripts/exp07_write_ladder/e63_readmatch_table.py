"""Write-regime x read-regime table from stored results (CPU; new file; thread-server 11/25). Source: experiments/exp07_write_ladder/e57_qabs_margin_result.json (job 1299 block stream, 4 seeds; arms NATS = natural write, NATMASK1024 = same-fact hidden write; reads: neutral out-of-window tail at lag 0..5 = bundle written lag read points ago; in0 = in-window read, context of the real last 8 chunks, bundle just written; all as change over W0 of the SAME read regime) and e57_blockstream8_result.json (job 1296-type, 4 seeds). Usage: python e63_readmatch_table.py"""
import json
d=json.load(open('experiments/exp07_write_ladder/e57_qabs_margin_result.json'))['lags']
print('qabs (job 1299): dA = gold first-token gain under own question; dAB = dA minus the gain under the partner question (question-specific part); t95 over 4 seeds')
for lag in ('-1','0','1','2','3','4','5','in0'):
    v=d[lag];s=f'lag {lag:>3s}: '
    for k in ('dA_first','dAB_first','dAB_S'):
        x=v[k];s+=f"{k} NATS {x['NATS']:6.2f} | MASK {x['NATMASK1024']:6.2f} | diff {x['diff']:6.2f} [{x['ci95'][0]:5.2f},{x['ci95'][1]:5.2f}]   "
    print(s)
b=json.load(open('experiments/exp07_write_ladder/e57_blockstream8_result.json'))['result']
print('\nblockstream8 (margin convention max): in_lag0 and out lags 0..2')
for k in ('in_lag0','out_lag0','out_lag1','out_lag2'):
    x=b[k];print(f"{k:9s} NATS {x['NATS']['mean']:6.2f} | NATMASK {x['NATMASK']['mean']:6.2f} | diff {x['diff']['mean']:6.2f} [{x['diff']['ci95'][0]:5.2f},{x['diff']['ci95'][1]:5.2f}]")
