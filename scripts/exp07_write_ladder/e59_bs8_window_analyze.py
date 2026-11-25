"""Analysis of job 1345w (block-stream re-read with the KEPT WINDOW; CPU; new file; EXPLORATORY; descriptive statistics, seed-paired t 95% intervals, n = number of seeds). Input: the read.json files of e59_bs8_window.py: at the end of block k ONE context w<k> (the real last 8 chunks = window of block k) asks the facts of the blocks 0..k; lag = k - block; lag 0 = in-window (8 of the 16 exposures of every fact are in the window), lag >= 1 = window kept, facts not in the window. Per run (arm, seed): for lag l the mean over the blocks of the metric gain over W0 of the same context (margin = gold - best wrong over 16 candidates; top-1 = gold is the best of the 16; mean gold rank = 1 + number of wrong candidates above gold; the level (absolute value) is printed too). Runs are given as label=tag:ARM; default NATS=bs8w:NATS,NATMASK1024=bs8w:NATMASK; added band arms later: NATMASK256=bs8w256:NATMASK etc. (the contrasts are every arm vs the first and vs the second run label). Checks: final_hash of the re-run vs job 1296 (--orig OD, arms NATS / NATMASK), context sidecar sha and W0 equality across tags, lag-0 reads vs the original in_g0 reads of 1296 (max abs diff), the neutral out read at the final point vs the stored one (max abs diff), and the neutral-out gain at the final point per arm. python3 e59_bs8_window_analyze.py --new D [--orig OD] [--runs label=tag:ARM,...] [--seeds 82,83,84,85]"""
import sys,json,numpy as np
from pathlib import Path
from scipy import stats
a=sys.argv;D=Path(a[a.index('--new')+1]);OD=Path(a[a.index('--orig')+1]) if '--orig' in a else None
SEEDS=[int(x) for x in a[a.index('--seeds')+1].split(',')] if '--seeds' in a else [82,83,84,85];n=len(SEEDS);T=stats.t.ppf(.975,n-1)
RUNS=[tuple([r.split('=')[0]]+r.split('=')[1].split(':')) for r in (a[a.index('--runs')+1] if '--runs' in a else 'NATS=bs8w:NATS,NATMASK1024=bs8w:NATMASK').split(',')]
def met(S,g):
    x=np.asarray(S,float);g=np.asarray(g);k=np.arange(len(g));gold=x[k,g];w=x.copy();w[k,g]=-np.inf
    return np.array([(gold-w.max(1)).mean(),(gold>w.max(1)).mean(),(1+(w>gold[:,None]).sum(1)).mean()])
def ci(d,fmt='{:7.3f}'):d=np.asarray(d,float);m=d.mean();h=T*d.std(ddof=1)/np.sqrt(len(d)) if len(d)>1 else float('nan');return (fmt+' ['+fmt+','+fmt+'] {}/{}').format(m,m-h,m+h,int((d>0).sum()),len(d))
NAMES=('margin','top-1','mean rank')
def load(tag,arm,s):return json.load(open(D/f'E55_{tag}_{arm}_s{s}.read.json'))
G={};L={};OUT={}
for lab,tag,arm in RUNS:
    for s in SEEDS:
        r=load(tag,arm,s);gold=np.array(r['meta']['gold_idx']);g_=[];l_=[]
        for ap in sorted(int(k) for k in r['read']):
            jj=ap//16-1;name=f'w{jj}';sd=json.load(open(D/f'E55_{tag}_ctx_s{s}.json'))[str(ap)][name];lag=np.array([x if x is not None else -99 for x in sd['d']]);ii=np.array(r['meta']['item_idx'][str(ap)][name]);assert (ii==np.array(sd['items'])).all()
            for l in range(jj+1):
                m=lag==l;gg=gold[ii[m]];g_.append((l,met(np.asarray(r['read'][str(ap)][name])[m],gg)-met(np.asarray(r['W0'][name])[m],gg),met(np.asarray(r['read'][str(ap)][name])[m],gg)))
        G[(lab,s)]={l:np.mean([x[1] for x in g_ if x[0]==l],0) for l in range(8)};L[(lab,s)]={l:np.mean([x[2] for x in g_ if x[0]==l],0) for l in range(8)}
        fa=str(max(int(k) for k in r['read']));OUT[(lab,s)]=met(np.asarray(r['read'][fa]['out'])[:128],gold[:128])-met(np.asarray(r['W0']['out'])[:128],gold[:128])
print('== checks ==')
if OD is not None:
    for lab,tag,arm in RUNS:
        oa={'NATS':'NATS','NATMASK':'NATMASK'}.get(arm)
        if not (OD/f'E55_bs8_{arm}_s{SEEDS[0]}.saved.json').exists() or lab.endswith(('256','128','64','32')):continue
        eq=[json.load(open(D/f'E55_{tag}_{arm}_s{s}.saved.json'))['final_hash']==json.load(open(OD/f'E55_bs8_{arm}_s{s}.saved.json'))['final_hash'] for s in SEEDS];d0=[];dout=[]
        for s in SEEDS:
            r=load(tag,arm,s);o=json.load(open(OD/f'E55_bs8_{arm}_s{s}.read.json'));fa=str(max(int(k) for k in r['read']))
            dout.append(float(np.abs(np.asarray(r['read'][fa]['out'])-np.asarray(o['read'][fa]['out'])).max()))
            for ap in sorted(int(k) for k in r['read']):
                jj=ap//16-1;ii=r['meta']['item_idx'][str(ap)][f'w{jj}'];oi=o['meta']['item_idx'][str(ap)]['in_g0'];x=np.asarray(r['read'][str(ap)][f'w{jj}']);y=np.asarray(o['read'][str(ap)]['in_g0']);pos={v:k for k,v in enumerate(ii)};d0.append(float(max(np.abs(x[pos[v]]-y[k]).max() for k,v in enumerate(oi) if v in pos)))
        print(f'  {lab}: final_hash equal to 1296 {sum(eq)}/{len(eq)} | max|out(final) - stored out| {max(dout):.2e} | max|lag-0 window read - stored in_g0| {max(d0):.2e}')
for lab,tag,arm in RUNS[1:]:
    for s in SEEDS:
        c0=json.load(open(D/f'E55_{RUNS[0][1]}_ctx_s{s}.json'));c1=json.load(open(D/f'E55_{tag}_ctx_s{s}.json'));assert c0==c1,('context sidecar differs',lab,s)
        r0=load(RUNS[0][1],RUNS[0][2],s);r1=load(tag,arm,s);print(f'  {lab} s{s}: contexts equal to {RUNS[0][0]} | max|W0 diff| {max(np.abs(np.asarray(r0["W0"][k])-np.asarray(r1["W0"][k])).max() for k in r0["W0"]):.2e}') if s==SEEDS[0] else None
print('== gain over W0 per lag (mean over blocks and seeds; lag 0 = in-window, lag >= 1 = window kept) ==')
for mi,nm in enumerate(NAMES):
    print(f' [{nm}]')
    for lab,_,_ in RUNS:print(f'  {lab:14s} '+' '.join(f'l{l}:{np.mean([G[(lab,s)][l][mi] for s in SEEDS]):7.3f}' for l in range(8))+f' | level l0 {np.mean([L[(lab,s)][0][mi] for s in SEEDS]):.3f} l1-7 {np.mean([np.mean([L[(lab,s)][l][mi] for l in range(1,8)]) for s in SEEDS]):.3f}')
print('== summaries per seed (mean over lags 1..7 = kept window, facts not in window; lag 0 = in-window), arm contrasts seed-paired [95% t] ==')
S0=lambda lab,mi:np.array([G[(lab,s)][0][mi] for s in SEEDS]);S1=lambda lab,mi:np.array([np.mean([G[(lab,s)][l][mi] for l in range(1,8)]) for s in SEEDS])
for mi,nm in enumerate(NAMES):
    print(f' [{nm}]')
    for lab,_,_ in RUNS:print(f'  {lab:14s} lag0 {ci(S0(lab,mi))} | lag1-7 {ci(S1(lab,mi))} | final neutral out {ci(np.array([OUT[(lab,s)][mi] for s in SEEDS]))}')
    for i,(lab,_,_) in enumerate(RUNS):
        for lab2,_,_ in RUNS[:i]:print(f'  {lab} - {lab2}: lag0 {ci(S0(lab,mi)-S0(lab2,mi))} | lag1-7 {ci(S1(lab,mi)-S1(lab2,mi))} | out {ci(np.array([OUT[(lab,s)][mi]-OUT[(lab2,s)][mi] for s in SEEDS]))}')
print('== per-lag contrasts (margin) ==')
for i,(lab,_,_) in enumerate(RUNS):
    for lab2,_,_ in RUNS[:i]:print(f'  {lab} - {lab2}: '+' | '.join(f'l{l} {ci(np.array([G[(lab,s)][l][0]-G[(lab2,s)][l][0] for s in SEEDS]))}' for l in range(8)))
sl=lambda lab,mi:np.array([np.polyfit(np.arange(8),[G[(lab,s)][l][mi] for l in range(8)],1)[0] for s in SEEDS])
print('== dilution slope of the margin gain per lag ==');[print(f'  {lab}: {ci(sl(lab,0))}') for lab,_,_ in RUNS]
print('== final read point, SAME weights, same items: window read (w7) vs neutral tail (out); gain over W0 per lag, window - neutral, seed-paired [95% t] (margin) ==')
for lab,tag,arm in RUNS:
    d0=[];d1=[]
    for s in SEEDS:
        r=load(tag,arm,s);gold=np.array(r['meta']['gold_idx']);fa=max(int(k) for k in r['read']);jj=fa//16-1;sd=json.load(open(D/f'E55_{tag}_ctx_s{s}.json'))[str(fa)][f'w{jj}'];ii=np.array(sd['items']);lag=np.array([x if x is not None else -99 for x in sd['d']]);Sw=np.asarray(r['read'][str(fa)][f'w{jj}']);Ww=np.asarray(r['W0'][f'w{jj}']);So=np.asarray(r['read'][str(fa)]['out']);Wo=np.asarray(r['W0']['out'])
        dd={l:(met(Sw[lag==l],gold[ii[lag==l]])[0]-met(Ww[lag==l],gold[ii[lag==l]])[0])-(met(So[ii[lag==l]],gold[ii[lag==l]])[0]-met(Wo[ii[lag==l]],gold[ii[lag==l]])[0]) for l in range(jj+1)};d0.append(dd[0]);d1.append(np.mean([dd[l] for l in range(1,jj+1)]))
    print(f'  {lab:14s} lag0 (block 7, in window) {ci(np.array(d0))} | lag1-7 (window kept, facts not in window) {ci(np.array(d1))}')

# ---- familiarity / binding split (job 1345w with UNWRITTEN questions; Hans 11/25 'familiarity control'): the same context w<k> also asks the questions of the blocks k+1..7 (never written, never exposed; their candidates are the places of their own block). Per item dS = candidate scores minus W0 of the SAME context. BINDING = dS[gold] - mean dS[decoys]; DECOY = mean dS[decoys]; FAMILIARITY (exposed - unexposed) = DECOY(written block at lag l) - DECOY(items of unwritten blocks) in the same read.
print('== familiarity / binding split per lag (read points with an unwritten block, k < 7; mean over blocks, read points, seeds; summed log p; bind | decoy | fam = decoy(lag) - decoy(unwritten)) ==')
FB={}
for lab,tag,arm in RUNS:
    for s in SEEDS:
        r=load(tag,arm,s);gold=np.array(r['meta']['gold_idx']);acc={}
        for ap in sorted(int(k) for k in r['read']):
            jj=ap//16-1
            if jj>=7:continue
            name=f'w{jj}';sd=json.load(open(D/f'E55_{tag}_ctx_s{s}.json'))[str(ap)][name];lag=np.array([x if x is not None else -99 for x in sd['d']]);ii=np.array(sd['items'])
            dS=np.asarray(r['read'][str(ap)][name])-np.asarray(r['W0'][name]);ok=lag!=-99;g=gold[ii[ok]];dS=dS[ok];lg=lag[ok]
            bind=np.array([dS[n,g[n]]-np.delete(dS[n],g[n]).mean() for n in range(len(g))]);dec=np.array([np.delete(dS[n],g[n]).mean() for n in range(len(g))]);un=dec[lg<0].mean()
            for l in range(jj+1):
                m=lg==l;acc.setdefault(l,[]).append((bind[m].mean(),dec[m].mean(),dec[m].mean()-un))
            acc.setdefault('un',[]).append((bind[lg<0].mean(),un,0.0))
        FB[(lab,s)]={l:np.mean(v,0) for l,v in acc.items()}
for lab,_,_ in RUNS:
    print(f'  {lab:14s} '+' | '.join(f"l{l} bind {np.mean([FB[(lab,s)][l][0] for s in SEEDS if l in FB[(lab,s)]]):6.3f} dec {np.mean([FB[(lab,s)][l][1] for s in SEEDS if l in FB[(lab,s)]]):6.2f} fam {np.mean([FB[(lab,s)][l][2] for s in SEEDS if l in FB[(lab,s)]]):6.2f}" for l in (0,1,3,6))+f" | unwritten bind {np.mean([FB[(lab,s)]['un'][0] for s in SEEDS]):6.3f} dec {np.mean([FB[(lab,s)]['un'][1] for s in SEEDS]):6.2f}")
for i,(lab,_,_) in enumerate(RUNS):
    for lab2,_,_ in RUNS[:i]:
        print(f'  {lab} - {lab2}: '+' | '.join(f"l{l} bind {ci(np.array([FB[(lab,s)][l][0]-FB[(lab2,s)][l][0] for s in SEEDS]))} fam {ci(np.array([FB[(lab,s)][l][2]-FB[(lab2,s)][l][2] for s in SEEDS]))}" for l in (0,1,3)))
