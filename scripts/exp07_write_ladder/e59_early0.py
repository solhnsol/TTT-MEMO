"""S2628-EARLY step 0 (CPU, saved values only; new file; implementation thread 11/28; Hans 11/28 15:59 KST "응 0단계부터 해봐"; descriptive). Arms ISO1 (job 1363, redo1/iso1/cbo_s<seed>.json), P1d (job 1362, p1d/), P1s (job 1348, kvmulti/), seeds with ISO1 (231-234).
(1) chunks 0..31: gradient norm before clipping, clip coefficient, loss (P1d / P1s: NTP over the whole chunk; ISO1: over the 16 isolated sentences), update norm (= clip coef x gradient norm x lr; lr x ilr = 1, checked on ISO1 where the update norm is stored), cumulative update norm; fact-token NLL where stored (P1d / P1s: the 32 place-token positions of the 16 read facts every 8 chunks, under C0, weights before the write; ISO1: not stored); R1 matched binding against cumulative update norm at the read points and the P1d binding interpolated at ISO1's cumulative norm.
(2) e8 / e16 / e32: 16 x 16 candidate scores minus W0 split into gold rise, exposed-wrong rise, never-exposed change, familiarity (exposed - never exposed), the matched-binding control term (the same name scored as a wrong candidate of the other facts) and matched binding.
(3) matched binding per sentence order of the chunk just written (place first / name first), per fact and per seed.
  python e59_early0.py --redo REDO1 --p1d P1D --base KVMULTI [--out FILE]"""
import sys,json,glob,random,numpy as np
a=sys.argv;g=lambda k,d=None:a[a.index(k)+1] if k in a else d;RE=g('--redo');PD=g('--p1d');BA=g('--base');OUT=g('--out')
SEEDS=sorted(int(f.split('_s')[-1].split('.')[0]) for f in glob.glob(RE+'/iso1/cbo_s*.json'));L=[]
P=lambda *x:(L.append(' '.join(str(y) for y in x)),print(*x))
def _cands(seed):
    out=[]
    for f in range(16):
        r=random.Random(f"kvpos2|{seed}|{f}");oth=[h for h in range(16) if h!=f];r.shuffle(oth);out.append([f]+oth[:11])
    return out
def parts(dS,seed):
    """per fact: gold rise, exposed-wrong rise, never-exposed change, familiarity, control term, matched binding"""
    dS=np.asarray(dS,float);cf=_cands(seed);out=np.full((16,6),np.nan)
    for f in range(16):
        oth=[dS[q,cf[q].index(f)] for q in range(16) if q!=f and f in cf[q][1:12]];ctl=np.mean(oth) if oth else np.nan
        out[f]=[dS[f,0],dS[f,1:12].mean(),dS[f,12:16].mean(),dS[f,1:12].mean()-dS[f,12:16].mean(),ctl,dS[f,0]-ctl]
    return out
def arm(name,s):
    if name=='ISO1':r=json.load(open(f'{RE}/iso1/cbo_s{s}.json'));x=r['arms']['ISO1'];st=[dict(gnorm=e['steps'][0]['gnorm'],coef=e['steps'][0]['coef'],ce=e['steps'][0]['ce'],un=e['steps'][0]['update_norm']) for e in x['steps']]
    else:
        r=json.load(open(f'{PD if name=="P1d" else BA}/kvmulti_s{s}.json'));x=r['arms'][name];st=[dict(gnorm=e['gnorm'],coef=e['coef'],ce=e['ce'],un=e['coef']*e['gnorm']) for e in x['steps']]
    return x,st
ARMS=('ISO1','P1d','P1s');D={(n,s):arm(n,s) for n in ARMS for s in SEEDS}
lrchk=[abs(e['un']-e['coef']*e['gnorm']) for s in SEEDS for e in D[('ISO1',s)][1]];P(f'S2628-EARLY step 0: seeds {SEEDS}; ISO1 stored update norm vs clip coef x |g|: max abs diff {max(lrchk):.2e} (so lr x ilr = 1 and the P1d / P1s update norm = coef x |g|)')
P('== (1) chunks 0..31, mean over seeds: |g| before clipping / clip coef / loss / update norm / cumulative update norm')
for lo,hi in ((0,4),(4,8),(8,16),(16,24),(24,32)):
    P(f'  chunks {lo:2d}-{hi-1:2d}: '+' | '.join(f'{n} {np.mean([e["gnorm"] for s in SEEDS for e in D[(n,s)][1][lo:hi]]):.3f} / {np.mean([e["coef"] for s in SEEDS for e in D[(n,s)][1][lo:hi]]):.3f} / {np.mean([e["ce"] for s in SEEDS for e in D[(n,s)][1][lo:hi]]):.3f} / {np.mean([e["un"] for s in SEEDS for e in D[(n,s)][1][lo:hi]]):.3f} / {np.mean([sum(e["un"] for e in D[(n,s)][1][:hi]) for s in SEEDS]):.2f}' for n in ARMS))
P('  clip share (coef < 1) over chunks 0..31: '+' | '.join(f'{n} {np.mean([e["coef"]<1 for s in SEEDS for e in D[(n,s)][1][:32]]):.2f}' for n in ARMS))
P('  fact-token NLL (place tokens of the 16 read facts, C0, before the write), mean over seeds, chunks 7/15/23/31: '+' | '.join(f'{n} '+' '.join(f'{np.mean([np.mean(D[(n,s)][0]["diag"][str(e)]["nll_own"]) for s in SEEDS]):.3f}' for e in (8,16,24,32)) for n in ('P1d','P1s'))+' | ISO1 not stored (only its loss over the isolated sentences)')
P('== (1b) R1 matched binding against the cumulative update norm (mean over seeds)')
cum=lambda n,s,e:sum(x['un'] for x in D[(n,s)][1][:e])
mb=lambda n,s,e:float(np.nanmean(parts(D[(n,s)][0]['reads'][str(e)]['dS1'],s)[:,5]))
for e in (8,16,32,64,96,128):P(f'  e{e:3d}: '+' | '.join(f'{n} cum {np.mean([cum(n,s,e) for s in SEEDS]):6.2f} bind {np.mean([mb(n,s,e) for s in SEEDS]):6.3f}' for n in ARMS))
P('  P1d binding interpolated (per seed, over its read points) at the cumulative norm ISO1 has reached, vs ISO1 binding:')
for e in (8,16,32):
    v=[]
    for s in SEEDS:
        xs=[0.0]+[cum('P1d',s,q) for q in (8,16,32,64,96,128)];ys=[0.0]+[mb('P1d',s,q) for q in (8,16,32,64,96,128)];c=cum('ISO1',s,e);v.append((np.interp(c,xs,ys) if c<=xs[-1] else np.nan,mb('ISO1',s,e),c))
    v=np.asarray(v);P(f'  ISO1 e{e}: cum {v[:,2].mean():.2f}, ISO1 bind {v[:,1].mean():.3f}, P1d bind at that cum {np.nanmean(v[:,0]):.3f} (n {int(np.isfinite(v[:,0]).sum())} seeds inside the P1d range)')
P('== (2) e8 / e16 / e32 score parts (mean over facts, then seeds): gold rise | exposed-wrong rise | never-exposed change | familiarity | control term | matched binding')
for e in (8,16,32):
    for n in ARMS:
        m=np.mean([np.nanmean(parts(D[(n,s)][0]['reads'][str(e)]['dS1'],s),0) for s in SEEDS],0);P(f'  e{e:2d} {n:4s}: '+' | '.join(f'{x:7.3f}' for x in m))
P('== (3) matched binding by sentence order of the chunk just written (0 = place first, 1 = name first), mean over seeds of the fact means [n facts]')
for e in (8,16,32):
    for n in ARMS:
        o=[];
        for s in SEEDS:
            pf=parts(D[(n,s)][0]['reads'][str(e)]['dS1'],s)[:,5];od=D[(n,s)][0]['orders'].get(str(e))
            o.append([np.nanmean([pf[f] for f in range(16) if od and od[f]==q]) for q in (0,1)]+[sum(1 for f in range(16) if od and od[f]==q) for q in (0,1)])
        o=np.asarray(o,float);P(f'  e{e:2d} {n:4s}: place first {np.nanmean(o[:,0]):.3f} [{int(o[:,2].sum())}] | name first {np.nanmean(o[:,1]):.3f} [{int(o[:,3].sum())}]')
P('== (3b) per seed matched binding e8 / e16 / e32')
for s in SEEDS:P(f'  seed {s}: '+' | '.join(f'{n} '+' '.join(f'{mb(n,s,e):6.3f}' for e in (8,16,32)) for n in ARMS))
P('== (3c) per fact matched binding at e32 (rows facts 0..15; columns ISO1, P1d, P1s; mean over seeds)')
for f in range(16):P(f'  fact {f:2d}: '+' '.join(f'{np.mean([parts(D[(n,s)][0]["reads"]["32"]["dS1"],s)[f,5] for s in SEEDS]):7.3f}' for n in ARMS))
if OUT:open(OUT,'w').write('\n'.join(L)+'\n')
