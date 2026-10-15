"""CPU-only analysis of all saved Exp04 artifacts -> summary.md + metrics.json (safe to run at any time)."""
import json,sys,glob,re
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parent))
from stage1 import label_stream,CATS,DIST
from tokenizers import Tokenizer
ROOT=Path(__file__).resolve().parents[2];EXP=ROOT/'experiments/exp04_what_ttt_retains'
TOK=Tokenizer.from_file(str(ROOT/'data/tokenizer/llama3/tokenizer.json'))
CH=1024;HN=24*CH;FN=4*CH
rng=np.random.default_rng(0)
def boot(vals,n=2000):
    v=np.asarray(vals,float);v=v[np.isfinite(v)]
    if len(v)==0:return (float('nan'),)*3,0
    b=[v[rng.integers(0,len(v),len(v))].mean() for _ in range(n)];return (float(v.mean()),float(np.percentile(b,2.5)),float(np.percentile(b,97.5))),len(v)
def fmt(t):
    (m,lo,hi),n=t;return f'{m:+.3f} [{lo:+.3f},{hi:+.3f}] (n={n})'
def load(p):
    z=np.load(p);return z['lp']
# ---------------- Stage 1 ----------------
def stage1():
    docs={}
    for l in open(EXP/'data/stage1_docs.jsonl'):
        d=json.loads(l);docs[d['doc'].replace('.txt','')]=d
    S=EXP/'stage1';res={}
    CONDS=('REL','UNREL','CF','SHUF','NONE')
    tags=[t for t in docs if all((S/f'{t}_{c}_{o}.npz').exists() for c in CONDS for o in ('ON','OFF'))]
    res['docs']=len(tags);lines=[]
    if not tags:return res,['_Stage 1: no complete docs yet._']
    gain_stream={};fcat={};chunk_gain=[];tokgains=[];docgain=[];cntbin={}
    order=['ALL']+[f'{c}|{d}' for c in ('NAME_FIRST','NAME_CONT','LEX') for d in DIST]+['OTHER|NA']
    BINS=[(0,0),(1,2),(3,9),(10,49),(50,10**9)]
    for t in tags:
        d=docs[t];names=set(d['names']);rel=d['conds']['REL']
        cat,dist=label_stream(TOK,rel,names)
        lp={(c,o):load(S/f'{t}_{c}_{o}.npz') for c in CONDS for o in ('ON','OFF')}
        n=len(rel);keysA={'ALL':list(range(9216,n-1))}
        for j in range(9216,n-1):
            ck=cat[j+1]+'|'+dist[j+1] if cat[j+1]!='OTHER' else 'OTHER|NA';keysA.setdefault(ck,[]).append(j)
        for ck,idx in keysA.items():
            gain_stream.setdefault(ck,[]).append((float(lp[('REL','OFF')][idx].mean()-lp[('REL','ON')][idx].mean()),len(idx)))  # lp are LOG-PROBS, so NLL_ON-NLL_OFF = lp_OFF-lp_ON
        g=(lp[('REL','OFF')]-lp[('REL','ON')])[9216:n-1];  # = NLL_ON - NLL_OFF (negative = ON better)
        tokgains.append(g);docgain.append(float(g.mean()))
        chunk_gain.append([float((lp[('REL','OFF')][i*CH:(i+1)*CH]-lp[('REL','ON')][i*CH:(i+1)*CH]).mean()) for i in range(n//CH)])
        # F-region, labels from the REL layout
        Hc=np.bincount(np.asarray(rel[1:1+HN]),minlength=128300)
        for c in CONDS:
            off=0 if c!='NONE' else -HN;f0=1+HN+off
            for o in ('ON','OFF'):
                a_=lp[(c,o)]
                for part,rng_ in (('F1',range(f0,f0+CH-1)),('Fall',range(f0,f0+FN-1))):
                    keys={'ALL':list(rng_)}
                    for jj in rng_:
                        jrel=jj-off;ck=cat[jrel+1]+'|'+dist[jrel+1] if cat[jrel+1]!='OTHER' else 'OTHER|NA';keys.setdefault(ck,[]).append(jj)
                    for ck,idx in keys.items():fcat.setdefault((part,c,o,ck),{})[t]=float(-a_[idx].mean())
                    if part=='Fall':
                        for bi,(lo,hi) in enumerate(BINS):
                            idx=[jj for jj in rng_ if lo<=Hc[rel[jj-off+1]]<=hi]
                            if idx:cntbin.setdefault((c,o,bi),{})[t]=(float(-a_[idx].mean()),len(idx))
    # ---- 1A heterogeneity ----
    allg=np.concatenate(tokgains)
    lines.append('## Stage 1A — is the TTT gain uniform? (REL stream, ON−OFF per-token NLL, targets ≥ 9216, %d docs)\n'%len(tags))
    q=np.percentile(allg,[1,5,25,50,75,95,99]);tot=(-allg).clip(min=0).sum()
    srt=np.sort(-allg)[::-1];top5=srt[:int(0.05*len(srt))].clip(min=0).sum()/max(tot,1e-9)
    res['A_heterogeneity']={'mean_on_minus_off_nll':float(allg.mean()),'frac_tokens_improved':float((allg<0).mean()),'quantiles_1_5_25_50_75_95_99':q.tolist(),'share_of_total_improvement_from_top5pct_tokens':float(top5),'doc_mean_gain_sd':float(np.std(docgain)),'doc_mean_gain_range':[float(min(docgain)),float(max(docgain))]}
    lines.append(f'- mean (ON−OFF) NLL per token = {allg.mean():+.4f} (negative = ON better); tokens improved by ON: {(allg<0).mean()*100:.1f}%')
    lines.append(f'- per-token (ON−OFF) quantiles 1/5/25/50/75/95/99%: '+', '.join(f'{x:+.2f}' for x in q))
    lines.append(f'- top-5% most-improved tokens carry {top5*100:.1f}% of total improvement; doc-level mean gain SD={np.std(docgain):.4f}, range [{min(docgain):+.4f}, {max(docgain):+.4f}]\n')
    lines.append('| category | n docs | ON−OFF nll (neg = TTT helps) [95% CI] | tokens/doc |\n|---|---|---|---|');res['A']={}
    for ck in order:
        if ck not in gain_stream:continue
        v=[g for g,_ in gain_stream[ck]];k=np.mean([n for _,n in gain_stream[ck]]);r=boot(v);res['A'][ck]={'mean':r[0][0],'lo':r[0][1],'hi':r[0][2],'tok_per_doc':float(k)}
        lines.append(f'| {ck} | {len(v)} | {fmt(r)} | {k:.0f} |')
    cg=np.mean(np.asarray(chunk_gain),axis=0);res['chunk_gain_curve']=cg.tolist()
    lines.append('\nPer-chunk mean (ON−OFF) nll by chunk index: '+', '.join(f'{x:+.3f}' for x in cg)+'\n')
    # ---- 1B decomposition ----
    lines.append('## Stage 1B — matched-history decomposition on the SAME held-out tokens F (primary = ON; OFF is auxiliary)\n')
    lines.append('All histories have identical token count (24 chunks) and identical F. Contrast = NLL(first) − NLL(second); positive = second condition predicts F better.\n')
    CONTR={'TOTAL history (NONE→REL)':('NONE','REL'),'generic/frequency adaptation (NONE→UNREL)':('NONE','UNREL'),'HISTORY-SPECIFIC (UNREL→REL)':('UNREL','REL'),
           ' · lexical/topic (UNREL→SHUF)':('UNREL','SHUF'),' · narrative order (SHUF→REL)':('SHUF','REL'),' · name identity binding (CF→REL)':('CF','REL')}
    res['C']={}
    for part,title in (('F1','first F chunk'),('Fall','all 4 F chunks')):
        lines.append(f'### {title}\n');lines.append('| category | contrast | ON [95% CI] | OFF (aux) | ON−OFF (TTT-specific) [95% CI] |\n|---|---|---|---|---|');res['C'][part]={}
        for ck in ['ALL','NAME_FIRST|FAR','NAME_FIRST|LOC','NAME_FIRST|NEW','NAME_CONT|FAR','LEX|FAR','LEX|NEW','OTHER|NA']:
            for name,(a_,b_) in CONTR.items():
                vals={}
                for o in ('ON','OFF'):
                    da=fcat.get((part,a_,o,ck),{});db=fcat.get((part,b_,o,ck),{});vals[o]=[da[t]-db[t] for t in tags if t in da and t in db]
                if not vals['ON']:continue
                dd=[x-y for x,y in zip(vals['ON'],vals['OFF'])]
                res['C'][part].setdefault(ck,{})[name.strip()]={'ON':boot(vals['ON'])[0],'OFF':boot(vals['OFF'])[0][0],'DD':boot(dd)[0]}
                lines.append(f'| {ck} | {name} | {fmt(boot(vals["ON"]))} | {np.mean(vals["OFF"]):+.3f} | {fmt(boot(dd))} |')
        lines.append('')
    lines.append('### Raw mean NLL per condition (all 4 F chunks)\n');lines.append('| category | '+' | '.join(f'{o} {c}' for o in ('ON','OFF') for c in CONDS)+' |\n|'+'---|'*11)
    for ck in order:
        row=[]
        for o in ('ON','OFF'):
            for c in CONDS:
                d=fcat.get(('Fall',c,o,ck),{});row.append(np.mean([d[t] for t in tags if t in d]) if d else float('nan'))
        if not np.all(np.isnan(row)):lines.append(f'| {ck} | '+' | '.join(f'{x:.3f}' for x in row)+' |')
    # ---- 1C frequency vs specific by history-count bin ----
    lines.append('\n### Stage 1C — token-frequency gain vs history-specific gain, binned by how often the target token occurs in the history (all 4 F chunks)\n')
    lines.append('| count of target token in H | tokens/doc | ON: UNREL→REL (history-specific) | OFF: UNREL→REL | ON: NONE→UNREL (generic) |\n|---|---|---|---|---|');res['C_bins']={}
    for bi,(lo,hi) in enumerate(BINS):
        def get(c,o):
            d=cntbin.get((c,o,bi),{});return d
        u=get('UNREL','ON');r_=get('REL','ON');uo=get('UNREL','OFF');ro=get('REL','OFF');nn=get('NONE','ON')
        sp=[u[t][0]-r_[t][0] for t in tags if t in u and t in r_];so=[uo[t][0]-ro[t][0] for t in tags if t in uo and t in ro]
        ge=[nn[t][0]-u[t][0] for t in tags if t in nn and t in u]
        if sp:
            k=np.mean([r_[t][1] for t in tags if t in r_]);res['C_bins'][f'{lo}-{hi}']={'specific_ON':boot(sp)[0],'specific_OFF':float(np.mean(so)),'generic_ON':boot(ge)[0][0]}
            lines.append(f'| {lo}-{hi if hi<10**9 else "inf"} | {k:.0f} | {fmt(boot(sp))} | {np.mean(so):+.3f} | {np.mean(ge):+.3f} |')
    return res,lines
# ---------------- Stage 2/3 ----------------
def stage2():
    D=EXP/'stage2';res={}
    files=sorted(D.glob('seed*_main_A_ON.json'));seeds=sorted({re.match(r'(seed\d+)_',f.name).group(1) for f in files})
    ok=[s for s in seeds if all((D/f'{s}_main_{w}_ON.json').exists() for w in 'AB')]
    res['seeds']=ok;lines=[]
    if not ok:return res,['_Stage 2: no complete seed yet._']
    data={(s,w,o):json.load(open(D/f'{s}_main_{w}_{o}.json')) for s in ok for w in 'AB' for o in ('ON','OFF') if (D/f'{s}_main_{w}_{o}.json').exists()}
    cal={(s,w,o):json.load(open(D/f'{s}_cal_{w}_{o}.json')) for s in ok for w in 'AB' for o in ('ON','OFF') if (D/f'{s}_cal_{w}_{o}.json').exists()}
    def by(rows,k,lv):return {r['tid']:r for r in rows if r['k']==k and r['cond']=='NONE' and r['level']==lv}
    def table(src,ks,o,levels=('L2','L3','L3r'),sub='',seedlist=None,skip=None):
        """items are IDENTICAL held-out inputs in worlds A and B (labels fixed to world A). m = logp(gold_A) - logp(foil_A).
        World A history supports gold_A; the counterfactual history B supports foil_A. sub='1' -> first-subword scoring."""
        out={}
        for lv in levels:
            for k in ks:
                units=[]
                for s in (seedlist if seedlist is not None else ok):
                    if (s,'A',o) not in src or (s,'B',o) not in src:continue
                    A=by(src[(s,'A',o)]['rows'],k,lv);B=by(src[(s,'B',o)]['rows'],k,lv)
                    pairs={}
                    for tid in A:
                        if tid in B:pairs.setdefault(A[tid]['pair'],[]).append(tid)
                    for p,tids in pairs.items():
                        if len(tids)!=2:continue
                        if skip and skip(s,k,p):continue
                        mA={t:A[t]['gold'+sub]-A[t]['foil'+sub] for t in tids};mB={t:B[t]['gold'+sub]-B[t]['foil'+sub] for t in tids}
                        U={'hist':np.mean([mA[t]-mB[t] for t in tids]),'pref':np.mean([(mA[t]+mB[t])/2 for t in tids]),
                           'gnllA':np.mean([-A[t]['gold']/A[t]['ng'] for t in tids]),'gnllB':np.mean([-B[t]['gold']/B[t]['ng'] for t in tids]),
                           'accA':np.mean([mA[t]>0 for t in tids]),'accB':np.mean([mB[t]<0 for t in tids]),
                           'bothA':float(all(mA[t]>0 for t in tids)),'bothB':float(all(mB[t]<0 for t in tids)),
                           'changes':np.mean([(mA[t]>0)!=(mB[t]>0) for t in tids])}
                        U['both2']=U['bothA']*U['bothB'];U['key']=(s,p);units.append(U)
                if units:out[(lv,k)]=units
        return out
    def report(o,title,ks,src=data,sub='',seedlist=None,skip=None):
        T=table(src,ks,o,sub=sub,seedlist=seedlist,skip=skip);lines.append(f'**{title}**\n')
        lines.append('| level | rounds | acc (A hist / B hist) | both roles right in A | in B | in A AND B (bidirectional flip) | answer changes with history | **history-specific margin: m(relevant)−m(counterfactual) nats [95% CI]** | global-preference component | gold NLL/tok (rel / cf hist) | n pairs |\n|---|---|---|---|---|---|---|---|---|---|---|')
        for (lv,k),units in sorted(T.items()):
            h=boot([u['hist'] for u in units]);pf=boot([u['pref'] for u in units])
            r={'accA':float(np.mean([u['accA'] for u in units])),'accB':float(np.mean([u['accB'] for u in units])),'bothA':float(np.mean([u['bothA'] for u in units])),'bothB':float(np.mean([u['bothB'] for u in units])),'both2':float(np.mean([u['both2'] for u in units])),
               'changes':float(np.mean([u['changes'] for u in units])),'hist':h[0],'pref':pf[0][0],'gnllA':float(np.mean([u['gnllA'] for u in units])),'gnllB':float(np.mean([u['gnllB'] for u in units])),'n':len(units)}
            res.setdefault(title,{})[f'{lv}|{ROUND[k]}']=r
            lines.append(f'| {lv} | {ROUND[k]} | {r["accA"]:.2f} / {r["accB"]:.2f} | {r["bothA"]:.2f} | {r["bothB"]:.2f} | {r["both2"]:.2f} | {r["changes"]:.2f} | {fmt((h[0],h[1]))} | {pf[0][0]:+.2f} | {r["gnllA"]:.2f} / {r["gnllB"]:.2f} | {r["n"]} |')
        lines.append('')
    ROUND={14:1,33:4,62:16,6:1}
    lines.append(f'seeds={len(ok)} ({", ".join(ok)}). Same prompt + same gold/foil in both histories (labels fixed to world A); history A supports gold, history B (names of every swap pair exchanged) supports the foil. chance: acc 0.50, both-right-in-one-world 0.25, bidirectional flip 0.0625; a model with only a global name preference gets flip 0 and "changes"=0. Entity-level (full-name) scoring below; first-subword scoring follows.\n')
    import leak_audit
    v1=[x for x in ok if int(x[4:])<100];v2=[x for x in ok if int(x[4:])>=100]
    LEAK={x:leak_audit.leaky_pairs(int(x[4:]),1) for x in v1}
    skipf=lambda sd,k,p:(k in LEAK.get(sd,{}) and p in LEAK[sd][k])
    nleak=sum(len(v) for x in v1 for v in LEAK[x].values())
    lines.append(f'**Integrity note**: in the original worlds (seeds 1-3, "v1") the next exposure round started a few hundred tokens before the 1- and 4-round probe chunk for some pairs ({nleak} (seed,pair,snapshot) units): those entities had a later-round sentence INSIDE the local window and one extra exposure. Seeds >= 100 ("v2") use a corrected layout (assertion: all later rounds start after the probe chunk). 16-round snapshots were never affected.\n')
    GROUPS=[('CLEAN v2 seeds' ,v2,None),('v1 seeds, leak-free pairs only',v1,skipf),('v1 seeds, ALL pairs (1- and 4-round rows CONTAMINATED)',v1,None)]
    for gname,sl,sk in GROUPS:
        if not sl:continue
        report('ON',f'PRIMARY [{gname}]: TTT ON — relevant vs counterfactual history, entity-level',[14,33,62],seedlist=sl,skip=sk)
    for gname,sl,sk in GROUPS[:2]:
        if sl:report('ON',f'[{gname}] (first-subword scoring): TTT ON',[14,33,62],sub='1',seedlist=sl,skip=sk)
    # TTT-specific component: ON minus OFF on the same (seed, pair) units (OFF only exists for some seeds)
    lines.append('**TTT-specific component = history-specific margin (ON) − history-specific margin (OFF), paired by (seed, pair)**. A non-zero margin with OFF means the signal is NOT stored in fast weights (it comes from the context/KV path or from a leak).\n')
    res['onoff']={}
    for gname,sl,sk in GROUPS:
        if not sl:continue
        lines.append(f'*{gname}*\n');lines.append('| level | rounds | ON margin | OFF margin | ON−OFF [95% CI] | n paired pairs |\n|---|---|---|---|---|---|')
        TON=table(data,[14,33,62],'ON',seedlist=sl,skip=sk);TOFF=table(data,[14,33,62],'OFF',seedlist=sl,skip=sk)
        for (lv,k),uon in sorted(TON.items()):
            off={u['key']:u for u in TOFF.get((lv,k),[])};pr=[(u,off[u['key']]) for u in uon if u['key'] in off]
            if pr:
                d=boot([a['hist']-b['hist'] for a,b in pr]);res['onoff'][f'{gname}|{lv}|{ROUND[k]}']=d[0]
                lines.append(f'| {lv} | {ROUND[k]} | {np.mean([a["hist"] for a,b in pr]):+.3f} | {np.mean([b["hist"] for a,b in pr]):+.3f} | {fmt(d)} | {len(pr)} |')
        lines.append('')
    for gname,sl,sk in GROUPS[:2]:
        if sl:report('OFF',f'AUXILIARY diagnostic [{gname}]: TTT OFF (no inner update)',[14,33,62],seedlist=sl,skip=sk)
    if cal:
        report('ON','Source-present readout calibration, TTT ON (exposures inside local window)',[6],cal)
        report('OFF','Source-present readout calibration, TTT OFF',[6],cal)
    # L1 frequency (leaked pairs removed for v1 seeds at the 1- and 4-round snapshots)
    lines.append('**L1 frequency/familiarity (exposed vs matched never-seen control; sum-logp margin; positive = exposed name/place preferred; v1 leaked pairs removed)**\n')
    lines.append('| item | rounds | TTT | CLEAN v2 seeds | v1 seeds, leak-free pairs | v1 seeds, ALL pairs (contaminated at 1,4) |\n|---|---|---|---|---|---|');res['L1']={}
    def l1(sl,sk,lv,k,o):
        m=[]
        for sd in sl:
            for w in 'AB':
                for r in data.get((sd,w,o),{'rows':[]})['rows']:
                    if r['k']==k and r['level']==lv and r['cond']=='NONE' and not (sk and sk(sd,k,r['pair'])):m.append(r['gold']-r['foil'])
        return m
    for lv in ('L1p','L1l'):
        for k in (14,33,62):
            for o in ('ON','OFF'):
                cells=[]
                for gname,sl,sk in GROUPS:
                    m=l1(sl,sk,lv,k,o);cells.append(fmt(boot(m)) if m else 'n/a');res['L1'][f'{lv}|{ROUND[k]}|{o}|{gname}']=boot(m)[0][0] if m else None
                lines.append(f'| {lv} | {ROUND[k]} | {o} | '+' | '.join(cells)+' |')
    # audit + dose
    a0=data[(ok[0],'A','ON')]['audit']
    lines.append('\n**SWA-8K / recency audit (first seed, world A; verified by assertion for all seeds)**\n')
    lines.append('| snapshot chunk k | rounds | last exposure end token | probe chunk start | distance last exposure→probe start | slack beyond 8K window of last probe token | swap-pair last-exposure gap mean/max (tokens) |\n|---|---|---|---|---|---|---|')
    lines.append(f"token length of every person/place name (all worlds identical): {a0.get('token_lengths')}\n")
    for k,v in a0.items():
        if not isinstance(v,dict) or str(k).startswith('_'):continue
        lines.append(f"| {k} | {v['rounds']} | {v['last_exposure_end_token']} | {v['probe_chunk_start']} | {v['distance_last_exposure_to_probe_start']} | {v['min_gap_to_window_start']} | {v['pair_recency_diff_tokens_mean']:.0f}/{v['pair_recency_diff_tokens_max']} |")
    dn=data[(ok[0],'A','ON')]['delta_norms']
    lines.append('\nFast-weight movement ‖W−W0‖₂ (summed over leaves) at each snapshot: '+', '.join(f"k={k}: {sum(v.values()):.2f}" for k,v in dn.items()))
    # ---- Stage 3 ----
    RET={'A':('A','ON'),'OFF':('A','OFF'),'U1':('U1','ON'),'U2':('U2','ON')}
    ret={(s,k):json.load(open(D/f'{s}_ret_{n}_{o}.json')) for s in ok for k,(n,o) in RET.items() if (D/f'{s}_ret_{n}_{o}.json').exists()}
    res['stage3_runs']=len(ret)
    if ret:lines+=stage3(ret,ok)
    return res,lines
def stage3(ret,ok):
    L=['\n## Stage 3 — explicit retention marginal value (identical queries, identical R=48-token slot; oracle = analysis only)\n']
    CANDS=['BIND','BIND_PARTNER','NAME','PLACE','OCC','DISTRACT'];LV=('L2','L3','L3r')
    STATES={'A':'ON, relevant history','OFF':'OFF (aux)','U1':'ON, unrelated history #1','U2':'ON, unrelated history #2'}
    units={}
    for (s,st),d in ret.items():
        rows=d['rows']
        for lv in LV:
            base={r['tid']:r['gold']-r['foil'] for r in rows if r['cond']=='NONE' and r['level']==lv}
            for r in rows:
                if r['level']!=lv or r['cond']=='NONE':continue
                units.setdefault((st,(s,r['pair']),lv,r['cond']),[]).append(r['gold']-r['foil']-base[r['tid']])
    allunits=sorted({u for (st,u,lv,c) in units});sel=[u for u in allunits if int(u[1][1:])%2==0];val=[u for u in allunits if int(u[1][1:])%2==1]
    L.append(f'units = (seed, swap-pair): {len(allunits)}; **selection set** (oracle ranking) = even pairs ({len(sel)}), **final evaluation set** = disjoint odd pairs ({len(val)}). Value = gain in gold−foil margin (nats) over NONE.\n')
    vals={};pol={}
    L.append('Selection-set candidate values:\n');L.append('| level | state | '+' | '.join(CANDS)+' | argmax |\n|---|---|'+'---|'*(len(CANDS)+1))
    for lv in LV:
        for st in STATES:
            v={c:np.mean([np.mean(units[(st,u,lv,c)]) for u in sel if (st,u,lv,c) in units] or [np.nan]) for c in CANDS}
            vals[(lv,st)]=v;pol[(lv,st)]=max(v,key=lambda c:-1e9 if np.isnan(v[c]) else v[c])
            L.append(f'| {lv} | {STATES[st]} | '+' | '.join(f'{v[c]:+.2f}' for c in CANDS)+f' | **{pol[(lv,st)]}** |')
    def rank_stats(a,b):
        ra=np.argsort(np.argsort(-a));rb=np.argsort(np.argsort(-b))
        return (float(np.corrcoef(ra,rb)[0,1]) if ra.std()>0 and rb.std()>0 else float('nan')),len(set(np.argsort(-a)[:3])&set(np.argsort(-b)[:3]))
    L.append('\nRanking change vs noise floor (rank correlation of candidate values; top-3 overlap). The noise floor is the disagreement between two ON models adapted on DIFFERENT unrelated histories (U1 vs U2).\n')
    L.append('| level | A vs OFF | A vs U1 | A vs U2 | **U1 vs U2 (noise floor)** | argmax A / OFF / U1 / U2 |\n|---|---|---|---|---|---|')
    for lv in LV:
        vec={st:np.array([vals[(lv,st)][c] for c in CANDS]) for st in STATES}
        if any(np.isnan(v).any() for v in vec.values()):continue
        r=lambda x,y:'%+.2f / %d/3'%rank_stats(vec[x],vec[y])
        L.append(f'| {lv} | {r("A","OFF")} | {r("A","U1")} | {r("A","U2")} | {r("U1","U2")} | {pol[(lv,"A")]} / {pol[(lv,"OFF")]} / {pol[(lv,"U1")]} / {pol[(lv,"U2")]} |')
    L.append('\nRealized gain on the **final evaluation set** (policy chosen on the selection set; rows = state the policy was selected under; columns = model the retained span is used with):\n')
    L.append('| level | policy selected under | gain: ON relevant | gain: OFF | gain: ON unrelated #1 |\n|---|---|---|---|---|')
    out={}
    for lv in LV:
        for src in ('A','OFF','U1'):
            cells=[]
            for st in ('A','OFF','U1'):
                g=[np.mean(units[(st,u,lv,pol[(lv,src)])]) for u in val if (st,u,lv,pol[(lv,src)]) in units];cells.append(fmt(boot(g)) if g else 'n/a')
            L.append(f'| {lv} | {STATES[src]}: **{pol[(lv,src)]}** | {cells[0]} | {cells[1]} | {cells[2]} |')
        regret=[np.mean(units[('A',u,lv,pol[(lv,'A')])])-np.mean(units[('A',u,lv,pol[(lv,'U1')])]) for u in val if ('A',u,lv,pol[(lv,'A')]) in units and ('A',u,lv,pol[(lv,'U1')]) in units]
        if regret:L.append(f'| {lv} | regret in ON-relevant model of using the unrelated-history-optimal choice | {fmt(boot(regret))} | | |');out[f'regret_{lv}']=boot(regret)[0]
    return L

def stage2r():
    D=EXP/'stage2';files=sorted(D.glob('seed*_dyn_*.json'))
    if not files:return {},['_Stage 2R: no runs yet._']
    L=['\n## Stage 2R — forgetting vs interference after the last exposure (TTT ON; identical held-out inputs, world-A labels)\n',
       'Pair-symmetrised binding margin = mean over the two roles of a swap pair of m=logp(gold)-logp(foil) (global name preference cancels). All conditions share rounds 1-4 (updates on). NOUPD: weights frozen after the exposure phase; UNREL: normal updates on unrelated filler; RELSAME: +3 consistent re-exposure rounds (updates); RELCONF: +3 conflicting (swap-partner) rounds (updates). Probes are >7168 tokens after the last related text.\n']
    seeds=sorted({int(re.match(r'seed(\d+)_',f.name).group(1)) for f in files});conds=['NOUPD','UNREL','RELSAME','RELCONF']
    ok=[s_ for s_ in seeds if all((D/f'seed{s_}_dyn_{c}.json').exists() for c in conds)]
    res={'seeds':ok}
    if not ok:return res,L+['_incomplete_']
    dat={(s_,c):json.load(open(D/f'seed{s_}_dyn_{c}.json')) for s_ in ok for c in conds}
    a=dat[(ok[0],'UNREL')]['audit'];a2=dat[(ok[0],'RELSAME')]['audit']
    L.append('| probe chunk | UNREL/NOUPD distance last exposure→probe | RELSAME/RELCONF distance last related text→probe | updates since last original exposure |\n|---|---|---|---|')
    for k in a:L.append(f"| {k} | {a[k]['distance_last_original_exposure_to_probe']} | {a2[k]['distance_last_related_text_to_probe']} | {a[k]['updates_since_last_original_exposure']} (NOUPD: 0) |")
    L.append('\n| level | scoring | probe chunk | NOUPD | UNREL | RELSAME | RELCONF | UNREL−NOUPD (unrelated-update interference) | RELCONF−UNREL | RELSAME−UNREL |\n|---|---|---|---|---|---|---|---|---|---|')
    for lv in ('L2','L3'):
        for sub,name in (('','entity'),('1','first-subword')):
            for k in (18,28):
                vals={}
                for c in conds:
                    per=[]
                    for s_ in ok:
                        rows=[r for r in dat[(s_,c)]['rows'] if r['k']==k and r['level']==lv]
                        by={}
                        for r in rows:by.setdefault(r['pair'],[]).append(r['gold'+sub]-r['foil'+sub])
                        per.append({p:np.mean(v) for p,v in by.items()})
                    vals[c]={(i,p):x for i,d in enumerate(per) for p,x in d.items()}
                keys=sorted(vals['UNREL'])
                def diff(c1,c2):return boot([vals[c1][q]-vals[c2][q] for q in keys])
                cell=lambda c:fmt(boot([vals[c][q] for q in keys]))
                L.append(f'| {lv} | {name} | {k} | {cell("NOUPD")} | {cell("UNREL")} | {cell("RELSAME")} | {cell("RELCONF")} | {fmt(diff("UNREL","NOUPD"))} | {fmt(diff("RELCONF","UNREL"))} | {fmt(diff("RELSAME","UNREL"))} |')
                res[f'{lv}|{name}|{k}']={c:boot([vals[c][q] for q in keys])[0][0] for c in conds}
    return res,L
def gate(res1,res2):
    g1=False;g2=False;why=[]
    try:
        if res1.get('docs',0)<12:raise ValueError('need >=12 complete Stage-1 docs for a reliable gate, have %s'%res1.get('docs',0))
        d=res1['C']['F1']['ALL']['HISTORY-SPECIFIC (UNREL→REL)']['DD'];g1=d[1]>0;why.append(f'stage1 history-specific TTT-specific gain (F1,ALL) = {d[0]:+.4f} CI[{d[1]:+.4f},{d[2]:+.4f}]')
    except Exception as e:why.append('stage1 gate n/a: '+str(e))
    try:
        if len(res2.get('seeds',[]))<2:raise ValueError('need >=2 complete Stage-2 seeds, have %d'%len(res2.get('seeds',[])))
        for k,v in res2.items():
            if k.startswith('PRIMARY') and isinstance(v,dict):
                for kk,r in v.items():
                    if kk.endswith('|16') and r['hist'][1]>0:g2=True
        why.append('stage2 any level at 16 rounds with history-specific margin CI lower bound >0: '+str(g2))
    except Exception as e:why.append('stage2 gate n/a: '+str(e))
    return {'run_stage3':bool(g1 or g2),'stage1_history_specific_significant':bool(g1),'stage2_binding_significant':bool(g2),'why':why}
def main():
    out={};md=[]
    md.append('# Exp04 — What does TTT actually retain? (auto-generated; numbers only, interpretation in final report)\n')
    v=EXP/'stage0/validation.json'
    if v.exists():
        md.append('## Stage 0 — validation\n```json\n'+v.read_text()+'\n```\n')
    r1,l1=stage1();out['stage1']=r1;md.append('# Stage 1 — NTP decomposition\n');md+=l1
    r2,l2=stage2();out['stage2']=r2;md.append('\n# Stage 2 — repeated entities and entity-role binding\n');md+=l2
    r2r,l2r=stage2r();out['stage2r']=r2r;md+=l2r
    try:
        import analysis_extra;ex,exl=analysis_extra.run();out['extra']=ex;md+=exl
    except Exception as e:md.append(f'_analysis_extra failed: {e!r}_')
    gt=gate(r1,r2);out['gate']=gt;(EXP/'gate.json').write_text(json.dumps(gt,indent=1));md.append('\n## Stage 3 gate\n'+json.dumps(gt))
    (EXP/'summary_auto.md').write_text('\n'.join(md));(EXP/'metrics.json').write_text(json.dumps(out,indent=1,default=str))
    print('\n'.join(md)[:3000])
if __name__=='__main__':main()
