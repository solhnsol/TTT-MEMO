"""E62 P0a summary (CPU; new file; thread-server 11/23; descriptive only, no threshold): from e62_p0a.json: per fact margin = gold - max(wrong) of the summed answer log-probability (mean over the 3 frames), change over W0, at the read after the 1st / 2nd / 4th mention chunk, the last mention chunk and the book end; arm differences SAME - NAT and RAND - NAT per cell (n, gap kind) and per n, pooled over the 3 books (3 facts per cell; t 95% intervals with df = facts - 1 are wide), s = standard deviation of the paired per-fact differences, NAT - W0 by n, W0 positive control, masking checks. python3 e62_analyze.py --json J"""
import sys,json,numpy as np,collections
from scipy import stats
a=sys.argv;R=json.load(open(a[a.index('--json')+1]))
def mg(S):S=np.asarray(S);return (S[:,:,0]-S[:,:,1:].max(2)).mean(1)       # per fact, mean over frames
rows=[]                                                                       # (book, fact, n, kind, arm, point, value)
for B in R['books']:
    W0=mg(B['W0']);man=collections.defaultdict(list)
    for r in B['manifest']:man[r['cell']].append(r)
    for ci,f in enumerate(B['facts']):
        ms=sorted(man[ci],key=lambda r:r['mention']);pts={'m1':ms[0]['chunk'],'m2':ms[1]['chunk'],'m4':ms[3]['chunk'] if len(ms)>3 else None,'last':ms[-1]['chunk'],'end':B['nch']-1}
        for ar,A in B['arms'].items():
            for p,ch in pts.items():
                if ch is None or str(ch) not in A['reads']:continue
                rows.append((B['book'][:14],ci,f['n'],f['kind'],ar,p,float(mg(A['reads'][str(ch)])[ci]-W0[ci])))
def ci(d):
    d=np.asarray(d,float);m=d.mean()
    if len(d)<2:return f'{m:8.3f}'
    h=stats.t.ppf(.975,len(d)-1)*d.std(ddof=1)/np.sqrt(len(d));return f'{m:8.3f} [{m-h:8.3f},{m+h:8.3f}] {int(max((d>0).sum(),(d<0).sum()))}/{len(d)}'
V={(b,f,n,k,ar,p):v for b,f,n,k,ar,p,v in rows}
arms=sorted({r[4] for r in rows});print('E62 P0a, margin change over W0 (gold - max wrong, summed log p, mean of 3 frames); points m1 m2 m4 last end')
for ar in arms:
    print(f'-- arm {ar}')
    for n in sorted({r[2] for r in rows}):
        print('   n=%2d '%n+' | '.join(f'{p}: {ci([v for (b,f,nn,k,a_,pp),v in V.items() if nn==n and a_==ar and pp==p])}' for p in ('m1','m2','m4','last','end')))
for ar in arms:
    if ar=='NAT':continue
    print(f'-- {ar} - NAT per cell (paired over books), last mention / end')
    keys=sorted({(r[2],r[3]) for r in rows});sd=[]
    for n,k in keys:
        for p in ('last','end'):
            d=[V[(b,f,nn,kk,ar,p)]-V[(b,f,nn,kk,'NAT',p)] for (b,f,nn,kk,a_,pp) in V if a_==ar and pp==p and nn==n and kk==k and (b,f,nn,kk,'NAT',p) in V];sd+=d if p=='last' else []
            if d:print(f'   n={n:2d} {k:6s} {p:4s} {ci(d)}')
    if sd:print(f'   s (sd of paired per-fact differences at the last mention, all cells): {np.std(sd,ddof=1):.3f} over {len(sd)} facts')
    for lo,hi,nm in ((2,8,'n=2..8'),(16,32,'n=16..32')):
        d=[V[(b,f,nn,kk,ar,'last')]-V[(b,f,nn,kk,'NAT','last')] for (b,f,nn,kk,a_,pp) in V if a_==ar and pp=='last' and lo<=nn<=hi and (b,f,nn,kk,'NAT','last') in V];print(f'   {nm} pooled, last mention: {ci(d)}')
    x=[np.log2(nn) for (b,f,nn,kk,a_,pp) in V if a_==ar and pp=='last' and (b,f,nn,kk,'NAT','last') in V];y=[V[(b,f,nn,kk,ar,'last')]-V[(b,f,nn,kk,'NAT','last')] for (b,f,nn,kk,a_,pp) in V if a_==ar and pp=='last' and (b,f,nn,kk,'NAT','last') in V]
    if len(x)>2:print(f'   slope over log2 n of ({ar} - NAT) at the last mention: {np.polyfit(x,y,1)[0]:.3f}')
print('-- W0 positive control (planted sentence in front of the question) vs W0 margin, per fact mean')
for B in R['books']:print('  ',B['book'][:30],'W0 margin mean %.2f, positive control margin mean %.2f'%(mg(B['W0']).mean(),mg(B['W0_positive_control']).mean()))
print('-- masking checks')
for B in R['books']:
    for ar,A in B['arms'].items():print('  ',B['book'][:20],ar,'masked pairs',sum(s['masked_pairs'] for s in A['steps']),'hidden key tokens',sum(s['hidden_key_tokens'] for s in A['steps']),'rows',sum(s['sentence_rows'] for s in A['steps']),'dgnorm before first mask',A.get('pre_first_mask_max_dgnorm_vs_NAT'))
# ---- common axis (request of the design thread): pre-write answer-token NLL change (arm - NAT, natural window) vs the number of later mentions whose previous mention was inside the window
print('-- pre-write answer-token NLL (natural-window forward of the write step), arm - NAT, per mention with the previous mention inside the 8-chunk window; x = running count of such later mentions of the fact')
WINTOK=8*1024
bins=[(1,1),(2,2),(3,4),(5,8),(9,16),(17,99)]
def mean_tok(x):return float(np.mean(x)) if len(x) else None
for B in R['books']:
    pass
for ar in arms:
    if ar=='NAT':continue
    for key in ('nll_nat','nll_arm'):
        acc=collections.defaultdict(list)                                           # bin -> list of per-fact means of (arm - NAT)
        for B in R['books']:
            if ar not in B['arms'] or 'NAT' not in B['arms']:continue
            man={(r['cell'],r['mention']):r for r in B['manifest']};A={(d['cell'],d['mention']):d for d in B['arms'][ar].get('mention_nll',[])};N={(d['cell'],d['mention']):d for d in B['arms']['NAT'].get('mention_nll',[])}
            for cx in range(len(B['facts'])):
                ms=sorted([r for r in B['manifest'] if r['cell']==cx],key=lambda r:r['mention']);cnt=0;per=collections.defaultdict(list)
                for j in range(1,len(ms)):
                    prev=ms[j-1];cur=ms[j]
                    if cur['tok'][0]-prev['tok'][1]<=WINTOK and (cx,j) in A and (cx,j) in N and A[(cx,j)][key] and N[(cx,j)]['nll_nat']:
                        cnt+=1;d=mean_tok(A[(cx,j)][key])-mean_tok(N[(cx,j)]['nll_nat'] if key=='nll_nat' else N[(cx,j)]['nll_arm'])
                        for lo,hi in bins:
                            if lo<=cnt<=hi:per[(lo,hi)].append(d)
                for b_,v in per.items():acc[b_].append(float(np.mean(v)))
        print(f'   {ar} - NAT, {key}: '+' | '.join(f'x={lo}-{hi}: {ci(acc[(lo,hi)]) if acc[(lo,hi)] else "-"}' for lo,hi in bins))
print('-- per-chunk mean NLL of the non-planted tokens, arm vs NAT (natural window), mean over chunks and books')
for ar in arms:
    v=[np.mean([s['nonplanted_ce_nat'] for s in B['arms'][ar]['steps'] if s.get('nonplanted_ce_nat') is not None]) for B in R['books'] if ar in B['arms']]
    print(f'   {ar}: {np.round(v,4).tolist()}')
