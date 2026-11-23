"""Analysis of job 1345 (block-stream re-read with a real stream context; CPU; new file; EXPLORATORY): margin (gold - best wrong over the 16 candidates, mean over the 16 facts of a block) gain over W0 per lag (lag = read point - block; 0..7), arms NATS and NATMASK1024 (ALLQ1024), seeds 82-85: (a) context = real stream chunks of another block ('bc<j>' reads of the re-run, W0 of the same context), (b) neutral SQuAD tail ('out' reads of the ORIGINAL job-1296 run at the same read points, valid when the final_hash of the re-run equals the stored one; the neutral read of the re-run at the final read point is used as the check). Per lag: NATMASK - NATS in each context with seed-paired t 95% intervals (n = number of seeds, df n - 1), the context effect (stream - neutral) per arm, and the difference (NATMASK - NATS)_stream - (NATMASK - NATS)_neutral; also the linear slope over lag per arm (dilution slope) and the slope difference. python3 e59_bs8_reread_analyze.py --new D --orig OD [--seeds 82,83,84,85]"""
import sys,json,numpy as np
from pathlib import Path
from scipy import stats
a=sys.argv;D=Path(a[a.index('--new')+1]);OD=Path(a[a.index('--orig')+1]);SEEDS=[int(x) for x in a[a.index('--seeds')+1].split(',')] if '--seeds' in a else [82,83,84,85];n=len(SEEDS);t=stats.t.ppf(.975,n-1)
ARMS=('NATS','NATMASK')
def mar(S,gi):
    x=np.asarray(S,float);g=np.asarray(gi);gold=x[np.arange(len(g)),g];w=x.copy();w[np.arange(len(g)),g]=-np.inf;return float((gold-w.max(1)).mean())
def ci(d):d=np.asarray(d,float);m=d.mean();h=t*d.std(ddof=1)/np.sqrt(n);return f'{m:7.3f} [{m-h:7.3f},{m+h:7.3f}]'
R={}
for s in SEEDS:
    perm=json.load(open(D/f'E55_bs8c_perm_s{s}.json'))['block_of_fact']
    for arm in ARMS:
        new=json.load(open(D/f'E55_bs8c_{arm}_s{s}.read.json'));old=json.load(open(OD/f'E55_bs8_{arm}_s{s}.read.json'));gold=np.array(new['meta']['gold_idx']);hs=json.load(open(D/f'E55_bs8c_{arm}_s{s}.saved.json'))['final_hash'];ho=json.load(open(OD/f'E55_bs8_{arm}_s{s}.saved.json'))['final_hash']
        R[(s,arm,'hash')]=(hs==ho)
        for app in sorted(int(k) for k in new['read']):
            jj=app//16-1
            for name,Sx in new['read'][str(app)].items():
                if not name.startswith('bc'):continue
                j=int(name[2:]);ii=new['meta']['item_idx'][str(app)][name];R[(s,arm,'stream',jj-j)]=R.get((s,arm,'stream',jj-j),[])+[mar(Sx,gold[ii])-mar(new['W0'][name],gold[ii])]
            for j in range(jj+1):
                ii=[i for i in range(128) if perm[i]==j];R[(s,arm,'neutral',jj-j)]=R.get((s,arm,'neutral',jj-j),[])+[mar(np.asarray(old['read'][str(app)]['out'])[ii],gold[ii])-mar(np.asarray(old['W0']['out'])[ii],gold[ii])]
print('final_hash of the re-run equals the stored one:',{k[:2]:v for k,v in R.items() if k[2]=='hash'})
M={ctx:{arm:np.array([[np.mean(R[(s,arm,ctx,l)]) for l in range(8)] for s in SEEDS]) for arm in ARMS} for ctx in ('stream','neutral')}
print('gain over W0 per lag (rows = lag 0..7; mean over blocks and seeds)')
for ctx in ('stream','neutral'):
    print(f'  [{ctx}] '+'  '.join(f'lag{l}: S {M[ctx]["NATS"][:,l].mean():6.2f} M {M[ctx]["NATMASK"][:,l].mean():6.2f}' for l in range(8)))
print('NATMASK - NATS per lag, seed-paired [95% t]')
for ctx in ('stream','neutral'):
    print(f'  [{ctx}] '+' | '.join(f'lag{l} {ci(M[ctx]["NATMASK"][:,l]-M[ctx]["NATS"][:,l])}' for l in range(8)))
d_s=M['stream']['NATMASK']-M['stream']['NATS'];d_n=M['neutral']['NATMASK']-M['neutral']['NATS']
print('mean over lags of NATMASK - NATS: stream',ci(d_s.mean(1)),'| neutral',ci(d_n.mean(1)),'| stream - neutral',ci(d_s.mean(1)-d_n.mean(1)))
print('context effect (stream - neutral) per arm, mean over lags:',{arm:ci((M['stream'][arm]-M['neutral'][arm]).mean(1)) for arm in ARMS})
sl=lambda X:np.array([np.polyfit(np.arange(8),x,1)[0] for x in X])
print('dilution slope (gain per lag):',{f'{c}/{arm}':ci(sl(M[c][arm])) for c in ('stream','neutral') for arm in ARMS});print('slope difference NATMASK - NATS: stream',ci(sl(M['stream']['NATMASK'])-sl(M['stream']['NATS'])),'| neutral',ci(sl(M['neutral']['NATMASK'])-sl(M['neutral']['NATS'])))
