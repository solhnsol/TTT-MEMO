"""1344k (key / value decomposition of the write step) re-cut by ENTITY ROLE x SENTENCE ORDER (CPU; new file; thread-server 11/25): e59_kv_analyze.py takes p = the first token of d['second'] (the LATER mentioned entity) as the 'answer' position of fact f. Here p = the first token of the PLACE (role 'place') or of the person NAME (role 'name') of fact f in the sentence of the measured chunk, and the cells are role x sentence order (place first / name first), for the 16 read facts, conditions C0 and SAMEVIS, the 8 measured chunks (16 ... 127), 8 seeds. Same per-position stored arrays (kv_s<seed>.npz: NLL, DN = hidden error norm per layer (printed x 1000), XN = key norm, X2K = x2 . read keys, D16 = unembedding response of the 16 candidate places); layer 8 (index 7). Quantities per cell and condition: NLL at p, error norm DN, input norm XN, cosine to the own read key, to the other 15 keys, centred key dot, m = (u_f . delta)(x2 . k_f) response of the position, and the ratio mean|m|(SAMEVIS)/mean|m|(C0). Per seed the mean over chunks and facts of the cell; mean over seeds [t] ... Usage: python e63_kv1344k_recut.py --dir kv1344k"""
import sys,json,numpy as np
from pathlib import Path
from scipy import stats
a=sys.argv;D=Path(a[a.index('--dir')+1]);SEEDS=[s for s in range(231,239) if (D/f'kv_s{s}.json').exists() and (D/f'sent_order_s{s}.json').exists()];n=len(SEEDS);L7=7;T=stats.t.ppf(.975,max(n-1,1))
def ci(d):d=np.asarray(d,float);m=np.nanmean(d);h=T*np.nanstd(d,ddof=1)/np.sqrt(np.sum(~np.isnan(d))) if np.sum(~np.isnan(d))>1 else float('nan');return f'{m:8.3f} [{m-h:7.3f},{m+h:7.3f}] {int(max((d>0).sum(),(d<0).sum()))}/{len(d)}'
def du(D16):
    n_=D16.shape[1];return np.stack([D16[:,f]-(D16.sum(1)-D16[:,f])/(n_-1) for f in range(n_)],1)
CONDS=('C0','SAMEVIS');NAMES=('nll','dn','xn','cos_self','cos_oth','dot_cent','m')
RES={};CNT=np.zeros((2,2),int)                      # cell = (role, order); role 0 place 1 name; order 0 place-first 1 name-first
for s in SEEDS:
    z=np.load(D/f'kv_s{s}.npz');m=json.load(open(D/f'kv_s{s}.json'));keys=np.load(D/f'kv_s{s}_keys.npy');od=json.load(open(D/f'sent_order_s{s}.json'));kn=np.asarray(m['knorm']);kc=keys[:,0,L7]-keys[:,0,L7].mean(0,keepdims=True);knc=np.linalg.norm(kc,axis=1)
    acc={(cn,r,o):{nm:[] for nm in NAMES} for cn in CONDS for r in (0,1) for o in (0,1)}
    for cn in CONDS:
        for k in m['chunks']:
            X2K,D16,DN,XN,NLL=[z[f'{cn}|{k}|{nm}'] for nm in ('X2K','D16','DN','XN','NLL')];ap_={d['fact']:d for d in m['AP'][str(k)]};DU=du(D16)
            for f in range(16):
                o=od[str(k)][f] if str(k) in od else None
                if o is None:continue
                d=ap_[f]
                for role in (0,1):
                    lst=(d['second'] if o==1 else d['first']) if role==0 else (d['first'] if o==1 else d['second'])      # name first: place = later (second); place first: place = earlier (first)
                    p=lst[0];dot=X2K[p,L7,:];cosv=dot/(XN[p,L7]*kn[:,L7]+1e-30);oth15=[g for g in range(16) if g!=f];A=acc[(cn,role,o)]
                    A['nll'].append(float(NLL[p]));A['dn'].append(float(DN[p,L7])*1000.0);A['xn'].append(float(XN[p,L7]));A['cos_self'].append(float(cosv[f]));A['cos_oth'].append(float(cosv[oth15].mean()));A['dot_cent'].append(float(dot[f]-dot.mean()));A['m'].append(float(DU[p,f]*X2K[p,L7,f]))
                    if cn=='C0' and s==SEEDS[0]:CNT[role,o]+=1
    for key,A in acc.items():
        for nm,v in A.items():
            if v:RES.setdefault((key,nm),[]).append(float(np.mean(v)))
            if nm=='m' and v:RES.setdefault((key,'absm'),[]).append(float(np.mean(np.abs(v))))
print(f'1344k re-cut by role x sentence order: {n} seeds, 8 chunks, 16 read facts (per seed and cell the facts: place-first {CNT[0,0]//1 if False else ""}) ; layer 8; value = mean over seeds [t] same-direction seeds')
RN={0:'place tokens',1:'name tokens'};ON={0:'place-first sentences',1:'name-first sentences'}
for role in (0,1):
    for o in (0,1):
        print(f'== {RN[role]} in {ON[o]} (first token p of the entity; facts of seed 231 only: {CNT[role,o]}; all seeds see the details block)')
        for nm in NAMES+('absm',):
            c0=np.array(RES.get((('C0',role,o),nm),[np.nan]*n));sv=np.array(RES.get((('SAMEVIS',role,o),nm),[np.nan]*n))
            print(f'   {nm:9s} C0 {ci(c0)} | SAMEVIS {ci(sv)} | SAMEVIS - C0 {ci(sv-c0)} | ratio of the seed means SV/C0 {np.nanmean(sv)/np.nanmean(c0) if abs(np.nanmean(c0))>1e-12 else float("nan"):.3f}'+(f' | seed-wise ratio mean {np.nanmean(sv/c0):.3f}' if nm in ('absm','dn') else ''))
print('NOTE: the headline numbers of e59_kv_analyze.py use p = first token of d[second] = the later mentioned entity (place only in name-first sentences).')
# ---- (11/25, design check before paper use) seed counts, per-seed ratios and the seed-paired difference EARLIER-entity cell - LATER-entity cell for each role
print('\n== details for the paper check ==')
print('definitions: layer index 7 = layer 8 of the 8 prime SwiGLU layers (layer 32 of the model); NLL = per-position NLL of the update objective (mean NLL of all valid tokens of the chunk, before clipping) at p = first token of the entity; DN = |delta_z2,p| (L2 norm of the error signal at the output of the layer-8 FFN, i.e. the w2 side); XN = |x2_p|; cosines to the read keys k_f (x2 at the position predicting the first answer token of the gold row of the neutral out-of-window read); m = (u_f . delta_z2,p)(x2_p . k_f) (linear response of the position). Chunks 16 32 48 64 80 96 112 127. C0 = SAME_U1U2 (all other facts and the same fact hidden), SAMEVIS = (ha, OTHERONLY) (same fact visible, other facts hidden): the same CONDS dict as e59_kvpos / 1344p.')
CELLN={}
for s in SEEDS:
    od=json.load(open(D/f'sent_order_s{s}.json'));m=json.load(open(D/f'kv_s{s}.json'))
    for o in (0,1):CELLN[(s,o)]=sum(1 for k in m['chunks'] for f in range(16) if od[str(k)][f]==o)
print(f'facts per cell (16 read facts x 8 chunks; a fact is counted once per entity role): place first '+', '.join(f's{s}:{CELLN[(s,0)]}' for s in SEEDS)+f' (total {sum(CELLN[(s,0)] for s in SEEDS)}) | name first '+', '.join(f's{s}:{CELLN[(s,1)]}' for s in SEEDS)+f' (total {sum(CELLN[(s,1)] for s in SEEDS)})')
def seedvals(role,o,nm,cn):return np.array(RES[((cn,role,o),nm)])
def ratio(role,o,nm):return seedvals(role,o,nm,'SAMEVIS')/seedvals(role,o,nm,'C0')
for role,rn,earlier,later in ((0,'PLACE tokens',0,1),(1,'NAME tokens',1,0)):
    print(f'-- {rn}: EARLIER-entity cell ({ON[earlier]}) minus LATER-entity cell ({ON[later]}), seed-paired; per-seed ratios SAMEVIS/C0')
    for nm,lab in (('nll','NLL C0'),('nll','NLL SAMEVIS'),('dn','|delta_z2| ratio'),('absm','|m| ratio'),('cos_self','cos to own key SAMEVIS-C0')):
        if lab=='NLL C0':v=lambda o:seedvals(role,o,'nll','C0')
        elif lab=='NLL SAMEVIS':v=lambda o:seedvals(role,o,'nll','SAMEVIS')
        elif lab=='cos_self cos to own key SAMEVIS-C0' or nm=='cos_self':v=lambda o:seedvals(role,o,'cos_self','SAMEVIS')-seedvals(role,o,'cos_self','C0')
        else:v=lambda o,nm=nm:ratio(role,o,nm)
        e=v(earlier);l=v(later);print(f'   {lab:28s} earlier {ci(e)} | later {ci(l)} | earlier - later {ci(e-l)}')
