"""Analysis of the E62-W pilot (e62w_run.py output e62w_<stream>.json; CPU; new file; thread-server 11/25; design 3/4; descriptive: values, intervals, positive-fact counts; NO verdict wording, no thresholds). All values are changes (Delta) over W0 of the SAME question. Per fact f (type T) at its k-th mention (k = 16, 32, 64; the read after the chunk of that mention): MATCHED binding B_f(k) = Delta[f's question, a_f] - mean over the 5 OTHER questions of the type of Delta[q, a_f]; FAMILIARITY F_f(k) = mean over the 5 other questions of Delta[q, a_f] - mean over the same questions of the mean Delta of the 10 never-exposed candidates; accuracy: top-1 among the 6 exposed candidates, top-1 among all 16, mean rank among 16 (chance 1/6, 1/16, 8.5) of the fact's own question, with the W0 value. Main: D(k) = mean over gap-A facts [B(W4)-B(W8)] - mean over gap-B facts [B(W4)-B(W8)] (same fact in both arms, pairs); also with gap C as the control class; intervals: Welch t interval over facts and a stream-cluster bootstrap (2 pilot streams: reference only).
  python e62w_analyze.py --dir D"""
import sys,json,glob,collections,numpy as np
from scipy import stats
a=sys.argv;D=a[a.index('--dir')+1];ST=[json.load(open(f)) for f in sorted(glob.glob(D+'/e62w_S*.json'))];KS=(16,32,64);CLS='ABC';ARMS=[x for x in ('W8','W4') if all(x in s['arms'] for s in ST)]
def tci(x):
    x=np.asarray(x,float);x=x[~np.isnan(x)]
    if len(x)==0:return '    n/a'
    m=x.mean()
    if len(x)<2:return f'{m:7.2f}'
    h=stats.t.ppf(.975,len(x)-1)*x.std(ddof=1)/np.sqrt(len(x));return f'{m:7.2f} [{m-h:6.2f},{m+h:6.2f}]'
def welch(x,y):
    x=np.asarray(x,float);y=np.asarray(y,float);d=x.mean()-y.mean();vx=x.var(ddof=1)/len(x);vy=y.var(ddof=1)/len(y);se=np.sqrt(vx+vy);df=(vx+vy)**2/(vx**2/(len(x)-1)+vy**2/(len(y)-1));h=stats.t.ppf(.975,df)*se;return d,d-h,d+h
rows=[]                                          # fact-level rows: stream, fact, type, cls, arm, k, B, F, top1_6, top1_16, rank16, W0 values
for si,B in enumerate(ST):
    facts=B['facts'];cand=B['cands'];own=B['cand_owner'];man=collections.defaultdict(dict)
    for r in B['manifest']:man[r['fact']][r['mention']]=r['chunk']
    idx={ty:[i for i,f in enumerate(facts) if f['type']==ty] for ty in B['kept']}
    for arm in ARMS:
        R=B['arms'][arm]['reads']
        for f,fa in enumerate(facts):
            ty=fa['type'];ii=idx[ty];j=ii.index(f)
            for k in KS:
                ch=str(man[f][k-1])
                if ch not in R or ty not in R[ch]['scores']:continue
                S=np.asarray(R[ch]['scores'][ty]);W=np.asarray(B['W0'][ty]);dS=S-W;oth=[q for q in range(6) if q!=j]
                pos=[own[ii[q]].index(f) for q in oth];dec=np.mean([dS[q,p] for q,p in zip(oth,pos)]);unx=np.mean([dS[q,6:].mean() for q in oth])
                Bf=dS[j,0]-dec;Ff=dec-unx
                def acc(X):
                    x6=X[j,:6];x16=X[j];return dict(t6=float(x6[0]>=x6.max()),t16=float(x16[0]>=x16.max()),r16=float(1+(x16[1:]>x16[0]).sum()))
                s_=acc(S);w_=acc(W);rows.append(dict(stream=si,fact=f,type=ty,cls=fa['cls'],arm=arm,k=k,B=float(Bf),F=float(Ff),**{'s_'+x:y for x,y in s_.items()},**{'w_'+x:y for x,y in w_.items()}))
print(f'E62-W pilot: streams {[B["stream"] for B in ST]} (insertion seeds {[B["seed"] for B in ST]}, dropped types {[B["dropped"] for B in ST]}), {len(rows)} fact-arm-read rows; Delta over W0, summed log p; mean [95% t over facts] n/positive')
get=lambda arm,cls,k,key,stream=None,ty=None:np.array([r[key] for r in rows if r['arm']==arm and r['cls']==cls and r['k']==k and (stream is None or r['stream']==stream) and (ty is None or r['type']==ty)])
def pairdiff(cls,k,stream=None):
    d={}
    for r in rows:
        if r['arm']=='W4' and r['cls']==cls and r['k']==k and (stream is None or r['stream']==stream):d[(r['stream'],r['fact'])]=r['B']
    out=[]
    for r in rows:
        if r['arm']=='W8' and r['cls']==cls and r['k']==k and (r['stream'],r['fact']) in d:out.append(d[(r['stream'],r['fact'])]-r['B'])
    return np.array(out)
print('== 1/2. main contrast D(k) = mean_A[B(W4)-B(W8)] - mean_ctrl[B(W4)-B(W8)] (pairs per fact); control class B (main) and C')
rng=np.random.default_rng(0)
for ctrl in ('B','C'):
    for k in (64,32,16):
        dA=pairdiff('A',k);dC=pairdiff(ctrl,k)
        if len(dA)<2 or len(dC)<2:continue
        d,lo,hi=welch(dA,dC);ps=[]
        for _ in range(2000):
            sel=rng.integers(0,len(ST),len(ST));xa=np.concatenate([pairdiff('A',k,s) for s in sel]);xc=np.concatenate([pairdiff(ctrl,k,s) for s in sel]);ps.append(xa.mean()-xc.mean())
        per=[ (pairdiff('A',k,s).mean()-pairdiff(ctrl,k,s).mean()) for s in range(len(ST))]
        print(f'  k={k:2d} control {ctrl}: D {d:7.2f} Welch [{lo:6.2f},{hi:6.2f}] | stream bootstrap [{np.percentile(ps,2.5):6.2f},{np.percentile(ps,97.5):6.2f}] (reference) | per stream {[round(float(x),2) for x in per]} | pairs A {len(dA)} ctrl {len(dC)}')
print('== 3. arm x gap class x k: B (matched binding) | F (familiarity) | top-1 of 6 | top-1 of 16 | mean rank of 16 ; W0 value and chance in the header of the accuracy columns; n facts, positive-B facts')
print('   chance: top-1 of 6 = 0.167, top-1 of 16 = 0.0625, mean rank of 16 = 8.5; W0 per cell on the second line of each block')
for arm in ARMS:
    for cls in CLS:
        for k in KS:
            B_=get(arm,cls,k,'B')
            if len(B_)==0:continue
            print(f'  {arm} {cls} k={k:2d} n={len(B_):2d} pos {int((B_>0).sum()):2d} | B {tci(B_)} | F {tci(get(arm,cls,k,"F"))} | top6 {tci(get(arm,cls,k,"s_t6"))} | top16 {tci(get(arm,cls,k,"s_t16"))} | rank16 {tci(get(arm,cls,k,"s_r16"))}')
            print(f'{"":26s} W0: top6 {get(arm,cls,k,"w_t6").mean():.2f} top16 {get(arm,cls,k,"w_t16").mean():.2f} rank16 {get(arm,cls,k,"w_r16").mean():.2f}')
print('== 4. SD over facts of the paired difference B(W4)-B(W8) by gap class and k (n, mean, SD)')
for cls in CLS:
    for k in KS:
        d=pairdiff(cls,k)
        if len(d)>1:print(f'  class {cls} k={k:2d}: n={len(d):2d} mean {d.mean():6.2f} SD {d.std(ddof=1):6.2f}')
print('== 5. manipulation check: NLL of the 3 answer tokens at the planted mention, weights before the chunk write; arm | the natural-W8 forward on the same weights; by gap class and mention number bin, first-mention NLL per class as the reference')
bins=((1,15,'2-16'),(16,31,'17-32'),(32,63,'33-64'))
for arm in ARMS:
    for cls in CLS:
        s=[];
        for B in ST:
            for m in B['arms'][arm]['mention_nll']:
                if m['cls']==cls and m['ntok']>0 and m['nll_arm']:s.append((m['mention'],np.sum(m['nll_arm']),np.sum(m['nll_nat'])))
        if not s:continue
        s=np.array(s);line=f'  {arm} class {cls}: first mention {s[s[:,0]==0,1].mean():6.2f} | '
        for lo,hi,nm in bins:
            x=s[(s[:,0]>=lo)&(s[:,0]<=hi)];line+=f'{nm}: arm {x[:,1].mean():6.2f} nat {x[:,2].mean():6.2f} | '
        print(line)
print('== 6. per arm, per chunk diagnostics (mean over chunks and streams)')
for arm in ARMS:
    st=[x for B in ST for x in B['arms'][arm]['steps']];npa=[x['nonplanted_ce_arm'] for x in st if x['nonplanted_ce_arm'] is not None];npn=[x['nonplanted_ce_nat'] for x in st if x['nonplanted_ce_nat'] is not None]
    print(f'  {arm}: non-planted CE (arm forward) {np.mean(npa):.3f}, same chunks natural-W8 forward {np.mean(npn):.3f}, gradient norm {np.mean([x["gnorm"] for x in st]):.4f}, clip coefficient {np.mean([x["coef"] for x in st]):.4f}, applied update norm {np.mean([x["upd"] for x in st]):.4f}, chunks with masked pairs {sum(1 for x in st if x["masked_pairs"])}/{len(st)}')
print('== 7. breakdown at k=64: B by type and by stream, arms x gap class')
for key,vals in (('type',sorted({r['type'] for r in rows})),('stream',sorted({r['stream'] for r in rows}))):
    for v in vals:
        for arm in ARMS:
            line=f'  {key} {v} {arm}: '
            for cls in CLS:
                x=np.array([r['B'] for r in rows if r['arm']==arm and r['cls']==cls and r['k']==64 and r[key]==v])
                line+=f'{cls} n={len(x)} {tci(x) if len(x) else "   n/a"} | '
            print(line)

print('== 8. generation reads: greedy continuation (<= 8 tokens) after the question frame, and full-vocabulary ranks of the first answer token (neutral tail + frame; W0, k-th mention reads for ranks, 64th mention and stream end for generation)')
import re as _re
def norm(t):return t.lower().strip()
def first_word(t):
    w=_re.split(r"[^a-z']+",norm(t).lstrip(" ."));w=[x for x in w if x];return w[0] if w else ''
def gen_rows():
    out=[]
    for si,B in enumerate(ST):
        facts=B['facts'];cand=B['cands'];own=B['cand_owner'];man=collections.defaultdict(dict)
        for r in B['manifest']:man[r['fact']][r['mention']]=r['chunk']
        idx={ty:[i for i,f in enumerate(facts) if f['type']==ty] for ty in B['kept']};allans={w.lower() for f_ in facts for w in cand[facts.index(f_)]}
        for arm in ARMS:
            R=B['arms'][arm]['reads']
            for f,fa in enumerate(facts):
                ty=fa['type'];j=idx[ty].index(f);words=cand[f];gold=words[0].lower();exp=[w.lower() for w in words[1:6]];unx=[w.lower() for w in words[6:]]
                for tag,gd in (('W0',B['W0_gen']),)+tuple((f'k{k}',R.get(str(man[f][k-1]),{}).get('gen',{})) for k in KS)+(('end',R.get(str(B['nch']-1),{}).get('gen',{})),):
                    d=gd.get(ty) if gd else None
                    if d is None:continue
                    rk=np.asarray(d['rank'][j],float);row=dict(stream=si,fact=f,type=ty,cls=fa['cls'],arm=arm,tag=tag,top1=float(rk[0]==1),top5=float(rk[0]<=5),mrr=float(1/rk[0]),f_top1=float(np.mean(rk[6:]==1)),f_top5=float(np.mean(rk[6:]<=5)),f_mrr=float(np.mean(1/rk[6:])))
                    if 'text' in d:
                        t=d['text'][j];fw=first_word(t);tk_=d['tok'][j];row.update(text=t,exact=float(norm(t).strip(' .,;:!?"')==gold),starts=float(norm(t).lstrip(' ."').startswith(gold)),fw_gold=float(fw==gold),fw_exp=float(fw in exp),fw_unx=float(fw in unx),fw_unx_per_answer=float(fw in unx)/len(unx),fw_other_type=float(fw in allans and fw!=gold and fw not in exp and fw not in unx),fw_none=float(fw not in allans),tok3=float(list(tk_[:3])==list(B['gold_tok'][f])),tok1=float(tk_[0]==B['gold_tok'][f][0]))
                    out.append(row)
    return out
G=gen_rows()
def gm(arm,cls,tag,key):return np.array([r[key] for r in G if r['arm']==arm and r['cls']==cls and r['tag']==tag and key in r])
for tag in ('W0','k16','k32','k64','end'):
    print(f' -- read {tag}: rank metrics (gold first token; floor = the 10 never-exposed answers, per answer)'+('; generation classes of the first word: gold | other exposed | never-exposed | other; exact / starts-with' if tag in ('W0','k64','end') else ''))
    for arm in ARMS:
        for cls in CLS:
            t1=gm(arm,cls,tag,'top1')
            if len(t1)==0:continue
            line=f'   {arm} {cls} n={len(t1):2d} | top1 {t1.mean():.2f} top5 {gm(arm,cls,tag,"top5").mean():.2f} MRR {gm(arm,cls,tag,"mrr").mean():.3f} | floor top1 {gm(arm,cls,tag,"f_top1").mean():.3f} top5 {gm(arm,cls,tag,"f_top5").mean():.3f} MRR {gm(arm,cls,tag,"f_mrr").mean():.3f}'
            fw=gm(arm,cls,tag,'fw_gold')
            if len(fw):line+=f' | gen: gold {fw.mean():.2f} exposed-other {gm(arm,cls,tag,"fw_exp").mean():.2f} unexposed {gm(arm,cls,tag,"fw_unx").mean():.2f} (per answer {gm(arm,cls,tag,"fw_unx_per_answer").mean():.3f}) other-type {gm(arm,cls,tag,"fw_other_type").mean():.2f} none {gm(arm,cls,tag,"fw_none").mean():.2f} | exact string {gm(arm,cls,tag,"exact").mean():.2f} starts-with {gm(arm,cls,tag,"starts").mean():.2f} | 3-token match {gm(arm,cls,tag,"tok3").mean():.2f} first-token {gm(arm,cls,tag,"tok1").mean():.2f}'
            print(line)
print('== 8b. [gap A (W4-W8)] - [gap B (W4-W8)] for generation (first word = gold) and MRR (pairs per fact; Welch interval)')
def gpair(cls,tag,key):
    d={}
    for r in G:
        if r['arm']=='W4' and r['cls']==cls and r['tag']==tag and key in r:d[(r['stream'],r['fact'])]=r[key]
    return np.array([d[(r['stream'],r['fact'])]-r[key] for r in G if r['arm']=='W8' and r['cls']==cls and r['tag']==tag and key in r and (r['stream'],r['fact']) in d])
for key in ('fw_gold','mrr','top1'):
    for tag in ('k16','k32','k64','end') if key!='fw_gold' else ('k64','end'):
        dA=gpair('A',tag,key);dB=gpair('B',tag,key)
        if len(dA)>1 and len(dB)>1:d,lo,hi=welch(dA,dB);print(f'  {key:8s} {tag:4s}: D {d:7.3f} [{lo:6.3f},{hi:6.3f}] (A pairs {len(dA)}, B pairs {len(dB)})')
print('== 8c. per-fact pairs at the 64th mention (type, class, W8 | W4): first word gold, generated text; gold first-token rank')
for si,B in enumerate(ST):
    for f,fa in enumerate(B['facts']):
        r8=[r for r in G if r['stream']==si and r['fact']==f and r['arm']=='W8' and r['tag']=='k64'];r4=[r for r in G if r['stream']==si and r['fact']==f and r['arm']=='W4' and r['tag']=='k64']
        if r8 and r4 and 'text' in r8[0] and 'text' in r4[0]:print(f'  {B["stream"]} f{f:2d} {fa["type"]:6s} {fa["cls"]} gold {fa["answer"]:12s} | W8 {r8[0]["text"].strip()[:24]!r:28s} rank-top1 {int(r8[0]["top1"])} | W4 {r4[0]["text"].strip()[:24]!r:28s} rank-top1 {int(r4[0]["top1"])}')
