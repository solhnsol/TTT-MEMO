"""Analysis of job 1344k (key-value decomposition of the pre-clip fast-weight gradient; CPU; new file; EXPLORATORY; descriptive; seeds 231-238; per seed the mean over the measured chunks and the 16 read facts, then mean [95% t] over seeds, number of seeds in the sign direction). Input: kv_s<seed>.npz / kv_s<seed>.json / kv_s<seed>_keys.npy of e59_kv1344k.py. Layer index 7 = the last prime layer (layer 32 of the model), w2 = second linear layer (x2 -> z2), first answer token unless stated.
  VALIDATION: relative Frobenius error of sum_p x (x) delta against the autodiff gradient (max over layers, per matrix, median over conditions / chunks / seeds), global gradient norm and clip coefficient (clip 1).
  (A) |delta_p| and the NLL of the answer token (condition, and condition - C0).
  (B) key match of the write-time x2_p with the read keys: dot and cosine to the own key, to the other 15 keys (mean), and the centred value (key minus the mean key of the 16 facts).
  (C) candidate logits of sum_p delta_p (x2_p . k_f) through the final-norm weight and the unembedding (linear approximation): gold - mean(other 15) and gold - max(other 15), first token and all answer tokens (keys of the same token index).
  (D) the same sum for the update direction onto u_f = (gold - mean wrong) unembedding, split by the positions p: own answer tokens / answer tokens of the other facts (other U1 and the U2 facts) / all remaining positions.
  (F) three-way decomposition of log|m_p| (m_p = (u_f . delta_p)(x2_p . k_f), first answer token): condition - C0 change of log|delta_p|, log|cos(delta_p, u_f)| and log|x2_p . k_f| (the last one further into log|x2_p| and log|cos(x2_p, k_f)|); facts whose sign of m_p differs from C0 are excluded and counted; means of the logs and ratio of the means.
  (G-i) key / error swap, A, B in {C0, SAMEVIS} (and {C0, NATVIS}): G(A,B) = sum over ALL positions p of (u_f . delta^B_p)(x2^A_p . k_f), keys from A, errors from B; the four combinations and the two orders of the exact decomposition of SAMEVIS - C0 (key effect, error effect, interaction).
  python3 e59_kv_analyze.py --dir D [--seeds 231-238]"""
import sys,json,numpy as np
from pathlib import Path
from scipy import stats
a=sys.argv;D=Path(a[a.index('--dir')+1]);lo,hi=(231,238)
if '--seeds' in a:lo,hi=[int(x) for x in a[a.index('--seeds')+1].split('-')]
SEEDS=[s for s in range(lo,hi+1) if (D/f'kv_s{s}.json').exists()];n=len(SEEDS);T=stats.t.ppf(.975,max(n-1,1));NL=8;L7=7
def ci(d):d=np.asarray(d,float);m=np.nanmean(d);h=T*np.nanstd(d,ddof=1)/np.sqrt(len(d)) if len(d)>1 else float('nan');return f'{m:9.4f} [{m-h:9.4f},{m+h:9.4f}] {int(max((d>0).sum(),(d<0).sum()))}/{len(d)}'
def load(s):
    z=np.load(D/f'kv_s{s}.npz');m=json.load(open(D/f'kv_s{s}.json'));keys=np.load(D/f'kv_s{s}_keys.npy');return z,m,keys
def du(D16):
    n=D16.shape[1];return D16-(D16.sum(1,keepdims=True)-D16)/(n-1) if False else np.stack([D16[:,f]-(D16.sum(1)-D16[:,f])/(n-1) for f in range(n)],1)
R={}                                              # (cond,name) -> list over seeds of scalar
def put(c,nm,v):R.setdefault((c,nm),[]).append(v)
VAL=[];GN=[];CF=[]
for s in SEEDS:
    z,m,keys=load(s);conds=m['conds'];chunks=m['chunks'];NJ=m['NJ'];kn=np.asarray(m['knorm']);kc=keys[:,0,L7]-keys[:,0,L7].mean(0,keepdims=True);knc=np.linalg.norm(kc,axis=1)
    for r in m['rows']:VAL.append(np.asarray(r['val_relerr']).max(0));GN.append(r['gnorm']);CF.append(r['coef'])
    acc={}                                            # per cond: dict name -> list of per-(chunk,fact) values
    for cn in conds:
        A={}
        for k in chunks:
            X2K,X2KJ,D16,DN,XN,NLL=[z[f'{cn}|{k}|{nm}'] for nm in ('X2K','X2KJ','D16','DN','XN','NLL')];ap_={d['fact']:d for d in m['AP'][str(k)]};DU=du(D16)
            ansall=set(t for d in ap_.values() for t in d['second']);
            for f in range(16):
                d=ap_[f];p=d['second'][0];own=d['second'];oth=sorted(ansall-set(own));rest_mask=np.ones(X2K.shape[0],bool);rest_mask[list(ansall)]=False
                dot=X2K[p,L7,:];cosv=dot/(XN[p,L7]*kn[:,L7]+1e-30);oth15=[g for g in range(16) if g!=f]
                put_=lambda nm,v:A.setdefault(nm,[]).append(v)
                put_('nll_ans',float(NLL[p]));put_('nll_ans_all',float(np.mean([NLL[q] for q in own])));put_('dn',float(DN[p,L7]));put_('xn',float(XN[p,L7]))
                put_('dot_self',float(dot[f]));put_('dot_oth',float(dot[oth15].mean()));put_('cos_self',float(cosv[f]));put_('cos_oth',float(cosv[oth15].mean()));put_('dot_cent',float(dot[f]-dot.mean()))
                put_('cos_cent',float((dot[f]-dot.mean())/(XN[p,L7]*knc[f]+1e-30)))
                lg1=D16[p,:]*dot[f];put_('C1_gold_minus_mean',float(lg1[f]-lg1[oth15].mean()));put_('C1_gold_minus_max',float(lg1[f]-lg1[oth15].max()))
                lga=np.zeros(16)
                for j,q in enumerate(own[:NJ]):
                    dj=X2K[q,L7,f] if j==0 else X2KJ[q,j-1,f];lga+=D16[q,:]*dj
                put_('Call_gold_minus_mean',float(lga[f]-lga[oth15].mean()));put_('Call_gold_minus_max',float(lga[f]-lga[oth15].max()))
                mp=DU[:,f]*X2K[:,L7,f];put_('D_own',float(mp[own].sum()));put_('D_other',float(mp[oth].sum()));put_('D_rest',float(mp[rest_mask].sum()));put_('D_total',float(mp.sum()))
                put_('m',float(mp[p]));put_('logdn',float(np.log(DN[p,L7]+1e-30)));put_('logcosd',float(np.log(abs(DU[p,f])+1e-30)-np.log(DN[p,L7]+1e-30)));put_('logxk',float(np.log(abs(dot[f])+1e-30)));put_('logxn',float(np.log(XN[p,L7]+1e-30)));put_('logcosx',float(np.log(abs(dot[f])+1e-30)-np.log(XN[p,L7]*kn[f,L7]+1e-30)))
        acc[cn]={nm:np.asarray(v) for nm,v in A.items()}
    for cn in conds:
        for nm,v in acc[cn].items():
            if nm not in ('m','logdn','logcosd','logxk','logxn','logcosx'):put(cn,nm,float(v.mean()))
    for cn in conds:                                   # (F) vs C0, facts with the same sign of m
        if cn=='C0' or 'C0' not in acc:continue
        a_,b_=acc[cn],acc['C0'];ok=np.sign(a_['m'])==np.sign(b_['m']);put(cn,'F_nflip',float((~ok).sum()))
        if ok.sum()>0:
            for nm in ('logdn','logcosd','logxk','logxn','logcosx'):put(cn,'F_d'+nm,float((a_[nm]-b_[nm])[ok].mean()))
            put(cn,'F_dlogm',float((np.log(np.abs(a_['m'])+1e-30)-np.log(np.abs(b_['m'])+1e-30))[ok].mean()));put(cn,'F_ratio_mean_absm',float(np.abs(a_['m'][ok]).mean()/np.abs(b_['m'][ok]).mean()))
    for pair in (('C0','SAMEVIS'),('C0','NATVIS')):    # (G-i): all positions, layer 8, key from A, error from B
        if not all(c in conds for c in pair):continue
        Gv={}
        for A_ in pair:
            for B_ in pair:
                tot=[]
                for k in chunks:
                    X2K=z[f'{A_}|{k}|X2K'][:,L7,:];DUb=du(z[f'{B_}|{k}|D16']);tot.append((X2K*DUb).sum(0))
                Gv[(A_,B_)]=np.mean(tot,0).mean()          # mean over chunks and facts
        for (A_,B_),v in Gv.items():put('G',f'{A_}key_{B_}err',float(v))
        c0,x=pair;put('G',f'{x}-{c0}_total',float(Gv[(x,x)]-Gv[(c0,c0)]));put('G',f'{x}-{c0}_key_effect(err=C0)',float(Gv[(x,c0)]-Gv[(c0,c0)]));put('G',f'{x}-{c0}_err_effect(key={x})',float(Gv[(x,x)]-Gv[(x,c0)]))
        put('G',f'{x}-{c0}_err_effect(key=C0)',float(Gv[(c0,x)]-Gv[(c0,c0)]));put('G',f'{x}-{c0}_key_effect(err={x})',float(Gv[(x,x)]-Gv[(c0,x)]))
print(f'KEY-VALUE DECOMPOSITION job 1344k, seeds {SEEDS} (n={n}); layer 8 (index 7), w2, first answer token; per-seed means then mean [95% t] same-direction seeds')
V=np.asarray(VAL);print(f'VALIDATION sum_p x(x)delta vs autodiff: max over layers of the relative Frobenius error, median over (cond, chunk, seed): w1 {np.median(V[:,0]):.2e} w2 {np.median(V[:,1]):.2e} w3 {np.median(V[:,2]):.2e} | max w1 {V[:,0].max():.2e} w2 {V[:,1].max():.2e} w3 {V[:,2].max():.2e} | global gradient norm {np.mean(GN):.3f} [{np.min(GN):.2f},{np.max(GN):.2f}], clip coefficient {np.mean(CF):.3f} (share of chunks with coef<1: {np.mean(np.asarray(CF)<1):.2f})')
conds=sorted({c for c,_ in R if c!='G'},key=lambda c:['C0','SAMEVIS','NATVIS','U1vis','U2vis'].index(c))
names=[nm for nm in ('nll_ans','nll_ans_all','dn','xn','dot_self','dot_oth','dot_cent','cos_self','cos_oth','cos_cent','C1_gold_minus_mean','C1_gold_minus_max','Call_gold_minus_mean','Call_gold_minus_max','D_own','D_other','D_rest','D_total')]
for nm in names:
    print(f'-- {nm}');base=np.asarray(R[('C0',nm)])
    for c in conds:
        v=np.asarray(R[(c,nm)]);print(f'   {c:8s} level {ci(v)}'+('' if c=='C0' else f' | {c} - C0 {ci(v-base)}'))
print('-- (F) change vs C0 of log|m_p| and its parts (seed means over facts with the same sign of m; nflip = facts excluded per seed, mean)')
for c in conds:
    if c=='C0':continue
    print(f'   {c:8s} nflip {np.mean(R[(c,"F_nflip")]):.2f} | dlog|m| {ci(R[(c,"F_dlogm")])} = dlog|delta| {ci(R[(c,"F_dlogdn")])} + dlog|cos(delta,u)| {ci(R[(c,"F_dlogcosd")])} + dlog|x2.k| {ci(R[(c,"F_dlogxk")])}  [dlog|x2.k| = dlog|x2| {ci(R[(c,"F_dlogxn")])} + dlog|cos(x2,k)| {ci(R[(c,"F_dlogcosx")])}] | ratio of mean |m| {ci(R[(c,"F_ratio_mean_absm")])}')
print('-- (G-i) key (A) / error (B) swap, sum over all positions of (u_f . delta^B)(x2^A . k_f), mean over chunks and facts')
for (c,nm),v in sorted(R.items()):
    if c=='G':print(f'   {nm:44s} {ci(v)}')
