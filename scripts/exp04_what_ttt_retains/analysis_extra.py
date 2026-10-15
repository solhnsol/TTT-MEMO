"""Extra analyses (CPU): Q2 decomposition of retention value, Stage 1H hybrid histories, weights-only (empty-KV) 2x2."""
import json,re,sys
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parent))
from stage1 import label_stream,DIST
ROOT=Path(__file__).resolve().parents[2];EXP=ROOT/'experiments/exp04_what_ttt_retains'
rng=np.random.default_rng(1)
def boot(v,n=2000):
    v=np.asarray(v,float);v=v[np.isfinite(v)]
    if len(v)==0:return (float('nan'),)*3,0
    b=[v[rng.integers(0,len(v),len(v))].mean() for _ in range(n)];return (float(v.mean()),float(np.percentile(b,2.5)),float(np.percentile(b,97.5))),len(v)
def fmt(t):(m,lo,hi),n=t;return f'{m:+.3f} [{lo:+.3f},{hi:+.3f}] (n={n})'
CH=1024;HN=24*CH;FN=4*CH
def q2_decomp():
    D=EXP/'stage2';STATES={'ON, relevant history':'ret_A_ON','OFF (aux)':'ret_A_OFF','ON, unrelated #1':'ret_U1_ON','ON, unrelated #2':'ret_U2_ON'}
    L=['\n## Q2 decomposition — why does the BIND retention value shrink after relevant adaptation?\n',
       'Per state and level (sum-logp over the answer tokens, mean over queries and seeds; pairwise-choice accuracy = P(gold>foil)). gain = [gold(BIND)−foil(BIND)] − [gold(NONE)−foil(NONE)] = Δgold − Δfoil. headroom = −gold(NONE) (gold logp cannot exceed 0).\n',
       '| level | state | seeds | NONE gold | NONE foil | NONE margin | NONE acc | BIND gold | BIND foil | BIND margin | BIND acc | Δgold | Δfoil | gain | headroom | Δgold/headroom |\n|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|']
    out={}
    for lv in ('L2','L3','L3r'):
        for name,suf in STATES.items():
            G={};F={};n=0
            for f in sorted(D.glob(f'seed[0-9]*_{suf}.json')):
                if int(re.match(r'seed(\d+)_',f.name).group(1))>=100:continue   # v1 seeds only (final snapshot, unaffected by the early-round leak)
                rows=json.load(open(f))['rows'];n+=1
                for r in rows:
                    if r['level']==lv:G.setdefault(r['cond'],[]).append(r['gold']);F.setdefault(r['cond'],[]).append(r['foil'])
            if 'BIND' not in G:continue
            gN,fN,gB,fB=[np.mean(x) for x in (G['NONE'],F['NONE'],G['BIND'],F['BIND'])]
            aN=np.mean(np.array(G['NONE'])>np.array(F['NONE']));aB=np.mean(np.array(G['BIND'])>np.array(F['BIND']))
            out[f'{lv}|{name}']={'NONE_gold':gN,'NONE_foil':fN,'BIND_gold':gB,'BIND_foil':fB,'gain':gB-fB-(gN-fN),'headroom':-gN,'dgold_over_headroom':(gB-gN)/(-gN),'acc_NONE':float(aN),'acc_BIND':float(aB)}
            L.append(f'| {lv} | {name} | {n} | {gN:.2f} | {fN:.2f} | {gN-fN:+.2f} | {aN:.2f} | {gB:.2f} | {fB:.2f} | {gB-fB:+.2f} | {aB:.2f} | {gB-gN:+.2f} | {fB-fN:+.2f} | {gB-fB-(gN-fN):+.2f} | {-gN:.2f} | {(gB-gN)/(-gN):.2f} |')
    L.append('\nReading: the no-retention margin is ~0 in every state (no relational binding in weights), but gold AND foil logp are ~10 nats higher under relevant adaptation than under OFF/unrelated adaptation (name familiarity, Stage 2 L1). With BIND the gold logp reaches a similar absolute level (≈−2) in all states, so the gain is bounded by headroom, and the foil prior is also higher under relevant adaptation. The shrinking gain therefore reflects a ceiling/log-odds-scale effect of lexical familiarity, not stored binding.\n')
    return out,L
def _fcat(npzdir,tags,conds,docs):
    res={}
    for t in tags:
        d=docs[t];rel=d['conds']['REL'];cat,dist=label_stream(__import__('tokenizers').Tokenizer.from_file(str(ROOT/'data/tokenizer/llama3/tokenizer.json')),rel,set(d['names']))
        for c,dirx in conds:
            for o in ('ON','OFF'):
                a=np.load(dirx/f'{t}_{c}_{o}.npz')['lp'];f0=1+HN
                for part,rg in (('F1',range(f0,f0+CH-1)),('Fall',range(f0,f0+FN-1))):
                    keys={'ALL':list(rg)}
                    for jj in rg:
                        ck=cat[jj+1]+'|'+dist[jj+1] if cat[jj+1]!='OTHER' else 'OTHER|NA';keys.setdefault(ck,[]).append(jj)
                    for ck,idx in keys.items():res.setdefault((part,c,o,ck),{})[t]=float(-a[idx].mean())
    return res
def stage1h():
    S1=EXP/'stage1';SH=EXP/'stage1h'
    if not SH.exists():return {},['\n## Stage 1H — _not run yet_\n']
    docs={}
    for l in open(EXP/'data/stage1_docs.jsonl'):
        d=json.loads(l);docs[d['doc'].replace('.txt','')]=d
    tags=[t for t in docs if all((SH/f'{t}_{c}_{o}.npz').exists() for c in ('FARREL','NEARREL') for o in ('ON','OFF')) and all((S1/f'{t}_{c}_{o}.npz').exists() for c in ('REL','UNREL') for o in ('ON','OFF'))]
    if len(tags)<3:return {},[f'\n## Stage 1H — _only {len(tags)} docs complete_\n']
    fc=_fcat(None,tags,[('REL',S1),('UNREL',S1),('FARREL',SH),('NEARREL',SH)],docs)
    L=[f'\n## Stage 1H — is the OFF history gain just in-window context? ({len(tags)} docs; same F and token counts as Stage 1)\n',
       'FARREL = relevant first 16 chunks + UNRELATED last 8 chunks (relevant info only BEYOND the local window); NEARREL = unrelated first 16 + relevant last 8 chunks (relevant info only INSIDE the window). Gain = NLL(UNREL) − NLL(condition); positive = better than unrelated history.\n']
    res={}
    for part,title in (('F1','first F chunk'),('Fall','all 4 F chunks')):
        L.append(f'### {title}\n');L.append('| category | TTT | full relevant (REL) | beyond-window only (FARREL) | inside-window only (NEARREL) | beyond/total | inside/total |\n|---|---|---|---|---|---|---|')
        for ck in ('ALL','NAME_FIRST|FAR','NAME_FIRST|LOC','LEX|FAR','OTHER|NA'):
            for o in ('OFF','ON'):
                g={}
                for c in ('REL','FARREL','NEARREL'):
                    u=fc.get((part,'UNREL',o,ck),{});v=fc.get((part,c,o,ck),{});g[c]=[u[t]-v[t] for t in tags if t in u and t in v]
                if not g['REL']:continue
                tot=np.mean(g['REL']);res[f'{part}|{ck}|{o}']={c:boot(g[c])[0] for c in g}
                L.append(f"| {ck} | {o} | {fmt(boot(g['REL']))} | {fmt(boot(g['FARREL']))} | {fmt(boot(g['NEARREL']))} | {np.mean(g['FARREL'])/tot:.2f} | {np.mean(g['NEARREL'])/tot:.2f} |")
        # TTT-specific
        L.append('')
        L.append(f'*TTT-specific (ON−OFF) {title}*: '+'; '.join(f"{ck}: "+', '.join(f"{c} {fmt(boot([(fc[(part,'UNREL','ON',ck)][t]-fc[(part,c,'ON',ck)][t])-(fc[(part,'UNREL','OFF',ck)][t]-fc[(part,c,'OFF',ck)][t]) for t in tags if t in fc.get((part,c,'ON',ck),{}) and t in fc.get((part,c,'OFF',ck),{}) and t in fc.get((part,'UNREL','ON',ck),{}) and t in fc.get((part,'UNREL','OFF',ck),{})]))}" for c in ('REL','FARREL','NEARREL')) for ck in ('ALL',))+'\n')
    return res,L
def kv2x2():
    D=EXP/'stage2';seeds=sorted({int(re.match(r'seed(\d+)_',f.name).group(1)) for f in D.glob('seed1[0-9][0-9]_kvempty_A_ON.json')})
    L=['\n## Weights-only readout (empty cached KV) on clean v2 seeds\n']
    ok=[s for s in seeds if all((D/f).exists() for f in (f'seed{s}_kvempty_B_ON.json',f'seed{s}_main_A_ON.json',f'seed{s}_main_B_ON.json'))]
    if not ok:return {},L+['_not available yet_\n']
    L.append(f'seeds={ok}. history-specific margin m(A)−m(B) on identical held-out inputs. ON carry-KV = normal streaming readout; ON empty-KV = adapted fast weights with ZERO cached keys/values (the exposure text is >8K tokens away and never in the prefix window either); OFF empty-KV has no exposure information by construction (=0).\n')
    L.append('| level | rounds | ON carry-KV | OFF carry-KV (if run) | ON empty-KV (fast weights only) | n pairs |\n|---|---|---|---|---|---|');res={}
    ks={14:1,33:4,62:16}
    def margins(f,lv,k):
        rows=json.load(open(f))['rows'];return {r['tid']:r['gold']-r['foil'] for r in rows if r['k']==k and r['level']==lv and r['cond']=='NONE'},{r['tid']:r['pair'] for r in rows if r['k']==k and r['level']==lv and r['cond']=='NONE'}
    for lv in ('L2','L3','L3r','L1p','L1l'):
        for k,r_ in ks.items():
            cells={'on':[],'off':[],'ke':[]}
            for s in ok:
                try:
                    mA,pA=margins(D/f'seed{s}_main_A_ON.json',lv,k);mB,_=margins(D/f'seed{s}_main_B_ON.json',lv,k)
                    kA,_=margins(D/f'seed{s}_kvempty_A_ON.json',lv,k);kB,_=margins(D/f'seed{s}_kvempty_B_ON.json',lv,k)
                except Exception:continue
                pairs={}
                for t in mA:pairs.setdefault(pA[t],[]).append(t)
                oA=oB=None
                if (D/f'seed{s}_main_A_OFF.json').exists() and (D/f'seed{s}_main_B_OFF.json').exists():oA,_=margins(D/f'seed{s}_main_A_OFF.json',lv,k);oB,_=margins(D/f'seed{s}_main_B_OFF.json',lv,k)
                for p,ts in pairs.items():
                    if lv in ('L1p','L1l'):
                        cells['on'].append(np.mean([mA[t]+mB[t] for t in ts])/2);cells['ke'].append(np.mean([kA[t]+kB[t] for t in ts])/2)
                        if oA:cells['off'].append(np.mean([oA[t]+oB[t] for t in ts])/2)
                    else:
                        cells['on'].append(np.mean([mA[t]-mB[t] for t in ts]));cells['ke'].append(np.mean([kA[t]-kB[t] for t in ts]))
                        if oA:cells['off'].append(np.mean([oA[t]-oB[t] for t in ts]))
            if cells['on']:
                res[f'{lv}|{r_}']={c:boot(v)[0] for c,v in cells.items() if v}
                L.append(f"| {lv}{' (familiarity margin vs unseen control)' if lv.startswith('L1') else ''} | {r_} | {fmt(boot(cells['on']))} | {fmt(boot(cells['off'])) if cells['off'] else 'n/a'} | {fmt(boot(cells['ke']))} | {len(cells['on'])} |")
    return res,L
def stage1_reappearance():
    S=EXP/'stage1';docs=[json.loads(l) for l in open(EXP/'data/stage1_docs.jsonl')];acc={}
    for d in docs:
        t=d['doc'].replace('.txt','')
        if not all((S/f'{t}_{c}_{o}.npz').exists() for c in ('REL','UNREL') for o in ('ON','OFF')):continue
        rel=d['conds']['REL'];H=rel[1:1+HN];near=set(H[-8192:]);far=set(H[:-8192])
        Lp={(c,o):np.load(S/f'{t}_{c}_{o}.npz')['lp'] for c in ('REL','UNREL') for o in ('ON','OFF')}
        for part,rg in (('F1',range(1+HN,1+HN+CH-1)),('Fall',range(1+HN,1+HN+4*CH-1))):
            b={'target token id occurs in the last 8K of H':[],'target token id only in the first 16K of H':[],'target token id not in H':[]}
            for j in rg:
                y=rel[j+1];b['target token id occurs in the last 8K of H' if y in near else ('target token id only in the first 16K of H' if y in far else 'target token id not in H')].append(j)
            for k,idx in b.items():
                if idx:
                    for o in ('ON','OFF'):acc.setdefault((part,k,o),{})[t]=float(-Lp[('UNREL',o)][idx].mean()+Lp[('REL',o)][idx].mean())  # NLL(UNREL)-NLL(REL)
    L=['\n## Stage 1 — does the OFF history gain need the target token to reappear in the history?\n','History-specific gain = NLL(UNREL)−NLL(REL) on the same F tokens, split by whether the target token id occurs in H (positive = relevant history better).\n','| part | bucket | OFF | ON | ON−OFF [95% CI] | docs |\n|---|---|---|---|---|---|'];res={}
    for part in ('F1','Fall'):
        for k in ('target token id occurs in the last 8K of H','target token id only in the first 16K of H','target token id not in H'):
            on=acc.get((part,k,'ON'),{});off=acc.get((part,k,'OFF'),{});ts=[t for t in on if t in off]
            if ts:
                dd=boot([on[t]-off[t] for t in ts]);res[f'{part}|{k}']={'OFF':float(np.mean([off[t] for t in ts])),'ON':float(np.mean([on[t] for t in ts])),'DD':dd[0]}
                L.append(f"| {part} | {k} | {np.mean([off[t] for t in ts]):+.3f} | {np.mean([on[t] for t in ts]):+.3f} | {fmt(dd)} | {len(ts)} |")
    return res,L
def strict_subset():
    ij=EXP/'integrity.json'
    if not ij.exists():return {},['\n## Strict-matching sensitivity — _integrity.json missing_\n']
    I=json.load(open(ij));strict={}
    for rec in I['pair_mention_counts']:
        strict[(rec['seed'],rec['pair'])]=(rec['count_X1']==rec['count_X2'] and rec['token_mismatch_sentences']==0)
    D=EXP/'stage2';L=['\n## Strict-matching sensitivity (pairs with equal total mention counts AND zero A/B token-count mismatch in every exposure sentence)\n',
       'History-specific margin m(A)−m(B), TTT ON, valid snapshots only (v1 seeds: 16-round snapshot; v2 seeds: all). “strict” = fully matched pairs; “other” = pairs with a mention-count or tokenization imbalance.\n',
       '| seed group | level | rounds | strict pairs | other pairs |\n|---|---|---|---|---|'];res={}
    for grp,cond in (('v1 (16 rounds only)',lambda s:s<100),('v2 clean',lambda s:s>=100)):
        for lv in ('L2','L3','L3r'):
            for k,r_ in ((14,1),(33,4),(62,16)):
                if grp.startswith('v1') and k!=62:continue
                st=[];ot=[]
                for f in sorted(D.glob('seed[0-9]*_main_A_ON.json')):
                    sd=int(re.match(r'seed(\d+)_',f.name).group(1))
                    if not cond(sd) or not (D/f'seed{sd}_main_B_ON.json').exists():continue
                    A=json.load(open(f))['rows'];B=json.load(open(D/f'seed{sd}_main_B_ON.json'))['rows']
                    ma={r['tid']:(r['gold']-r['foil'],r['pair']) for r in A if r['k']==k and r['level']==lv and r['cond']=='NONE'};mb={r['tid']:r['gold']-r['foil'] for r in B if r['k']==k and r['level']==lv and r['cond']=='NONE'}
                    by={}
                    for t,(m,p) in ma.items():by.setdefault(p,[]).append(ma[t][0]-mb[t])
                    for p,v in by.items():(st if strict.get((sd,p)) else ot).append(np.mean(v))
                if st or ot:
                    res[f'{grp}|{lv}|{r_}']={'strict':boot(st)[0] if st else None,'other':boot(ot)[0] if ot else None}
                    L.append(f"| {grp} | {lv} | {r_} | {fmt(boot(st)) if st else 'n/a'} | {fmt(boot(ot)) if ot else 'n/a'} |")
    return res,L
def _units(seeds,mode,lv,k,sub=''):
    """per (seed,pair) history-specific margin m(A)-m(B); mode 'carry' (main ON) or 'empty' (kvempty ON)."""
    D=EXP/'stage2';tag='main' if mode=='carry' else 'kvempty';out={}
    for sd in seeds:
        fa,fb=D/f'seed{sd}_{tag}_A_ON.json',D/f'seed{sd}_{tag}_B_ON.json'
        if not (fa.exists() and fb.exists()):continue
        def g(f):
            return {r['tid']:(r['gold'+sub]-r['foil'+sub],r['pair']) for r in json.load(open(f))['rows'] if r['k']==k and r['level']==lv and r['cond']=='NONE'}
        A=g(fa);B=g(fb);by={}
        for t in A:by.setdefault(A[t][1],[]).append(A[t][0]-B[t][0])
        for p,v in by.items():out[(sd,p)]=float(np.mean(v))
    return out
def _perm_p(v,n=20000):
    v=np.asarray(v,float);obs=abs(v.mean());r=np.random.default_rng(7);sg=r.choice([-1,1],size=(n,len(v)));return float((np.mean(np.abs((sg*v).mean(1))>=obs-1e-12)*n+1)/(n+1))
def l3_checks():
    D=EXP/'stage2';seeds=sorted({int(re.match(r'seed(\d+)_',f.name).group(1)) for f in D.glob('seed1[0-9][0-9]_kvempty_A_ON.json')})
    seeds=[s for s in seeds if all((D/f).exists() for f in (f'seed{s}_kvempty_B_ON.json',f'seed{s}_main_A_ON.json',f'seed{s}_main_B_ON.json'))]
    L=['\n## Clean v2: is the weights-only L3 effect a TTT-specific, symmetric association? (no new GPU runs)\n']
    if not seeds:return {},L+['_no clean seeds yet_\n']
    ij=EXP/'integrity.json';strict={}
    if ij.exists():
        for rec in json.load(open(ij))['pair_mention_counts']:strict[(rec['seed'],rec['pair'])]=(rec['count_X1']==rec['count_X2'] and rec['token_mismatch_sentences']==0)
    res={};fam=[]
    L.append(f'seeds={seeds}. Unit = (seed, swap pair). p = two-sided sign-flip permutation test on the per-pair margin difference m(A)−m(B) (20000 flips); Holm correction over the whole family of tests below (3 levels x 3 exposure levels x 2 readouts = 18). Entity-level scoring.\n')
    L.append('| level | rounds | readout | pooled mean [95% CI] (n pairs) | per seed | strict-matched pairs [CI] | p | Holm-adjusted p |\n|---|---|---|---|---|---|---|---|')
    rows=[]
    for lv in ('L2','L3','L3r'):
        for k,r_ in ((14,1),(33,4),(62,16)):
            for mode in ('carry','empty'):
                u=_units(seeds,mode,lv,k)
                if not u:continue
                v=list(u.values());ps=_perm_p(v);b=boot(v)
                perseed={sd:float(np.mean([x for (s2,p),x in u.items() if s2==sd])) for sd in seeds}
                st=[x for q,x in u.items() if strict.get(q)]
                rows.append([lv,r_,mode,b,len(v),perseed,boot(st) if st else None,ps])
    # Holm
    order=sorted(range(len(rows)),key=lambda i:rows[i][7]);m=len(rows);adj=[None]*m;run=0.0
    for rank,i in enumerate(order):
        run=max(run,min(1.0,(m-rank)*rows[i][7]));adj[i]=run
    for i,(lv,r_,mode,b,n,perseed,st,ps) in enumerate(rows):
        res[f'{lv}|{r_}|{mode}']={'mean':b[0][0],'lo':b[0][1],'hi':b[0][2],'n':n,'per_seed':perseed,'p':ps,'p_holm':adj[i]}
        L.append(f"| {lv} | {r_} | {'native (carry KV)' if mode=='carry' else 'weights-only (empty KV)'} | {b[0][0]:+.3f} [{b[0][1]:+.3f},{b[0][2]:+.3f}] ({n}) | "+', '.join(f's{sd}: {x:+.3f}' for sd,x in perseed.items())+f" | {fmt(st) if st else 'n/a'} | {ps:.4f} | {adj[i]:.4f} |")
    # symmetry about the OFF baseline (weights-only readout): OFF-empty exists only for world A
    L.append('\n**Symmetry check (weights-only readout, L3 and L2).** OFF-empty (base weights, zero KV) was run for world A only; its margin is the baseline for BOTH histories because the held-out inputs are identical. A true history-specific association deviates from that baseline in opposite directions: ON,A above it and ON,B below it. A one-sided shift (both worlds above or below) would be a global preference/state artifact.\n')
    L.append('| level | rounds | OFF-empty baseline m | ON-empty m(A) − baseline [CI] | ON-empty m(B) − baseline [CI] | symmetric? |\n|---|---|---|---|---|---|')
    for lv in ('L2','L3','L3r'):
        for k,r_ in ((14,1),(33,4),(62,16)):
            dA=[];dB=[];base=[]
            for sd in seeds:
                fa,fb,fo=D/f'seed{sd}_kvempty_A_ON.json',D/f'seed{sd}_kvempty_B_ON.json',D/f'seed{sd}_kvempty_A_OFF.json'
                if not (fa.exists() and fb.exists() and fo.exists()):continue
                g=lambda f:{r['tid']:(r['gold']-r['foil'],r['pair']) for r in json.load(open(f))['rows'] if r['k']==k and r['level']==lv and r['cond']=='NONE'}
                A,B,O=g(fa),g(fb),g(fo);by={}
                for t in A:by.setdefault(A[t][1],[]).append((A[t][0]-O[t][0],B[t][0]-O[t][0],O[t][0]))
                for p,v in by.items():dA.append(np.mean([x[0] for x in v]));dB.append(np.mean([x[1] for x in v]));base.append(np.mean([x[2] for x in v]))
            if dA:
                a,b_=boot(dA),boot(dB);sym='yes (A up, B down)' if a[0][1]>0 and b_[0][2]<0 else ('no / not resolved')
                res[f'sym|{lv}|{r_}']={'A_minus_base':a[0],'B_minus_base':b_[0],'baseline':float(np.mean(base))}
                L.append(f"| {lv} | {r_} | {np.mean(base):+.3f} | {fmt(a)} | {fmt(b_)} | {sym} |")
    # same-snapshot check: fast-weight movement recorded in the native and weights-only ON streams
    L.append('\n**Are the native and weights-only readouts taken from the same fast-weight snapshot?** Both are separate runs of the identical token stream and identical native inner updates (the readout mode changes only the probe-time suffix cache, never the update path). Recorded ‖W−W0‖₂ at each snapshot:\n')
    L.append('| seed | world | snapshot k | native run | weights-only run | relative diff |\n|---|---|---|---|---|---|')
    for sd in seeds[:2]:
        for w in 'AB':
            try:
                a=json.load(open(D/f'seed{sd}_main_{w}_ON.json'))['delta_norms'];b=json.load(open(D/f'seed{sd}_kvempty_{w}_ON.json'))['delta_norms']
            except Exception:continue
            for k in a:
                x,y=sum(a[k].values()),sum(b[k].values());L.append(f'| {sd} | {w} | {k} | {x:.4f} | {y:.4f} | {abs(x-y)/x:.2e} |')
    L.append('\nDifference between the two readouts (NOT an SWA-suppression test): native = suffix layers read the carried cache of the last 8192 stream positions (7168 filler tokens before the probe chunk, K/V produced by the fast weights as they were at those earlier chunks, RoPE positions 7168+), weights-only = zero cache with the same current fast weights and RoPE positions starting at 0; prefix layers see the same 8192-token window in both. No readout updates the weights. The contrast therefore changes local context, positional offsets and which weight versions produced the keys/values at once; a smaller native effect cannot be attributed to “SWA suppression”, and an empty cache is also out-of-distribution for the suffix layers.\n')
    return res,L
def stage3_validity():
    import leak_audit
    D=EXP/'stage2';L=['\n## Stage 3 validity: dependence on the contaminated v1 layout\n','| seed | layout | snapshots with probes in the ret files | later-round sentences before the k=62 probe chunk | pairs with in-window leak at k=62 | ret states present | valid? |\n|---|---|---|---|---|---|---|'];res={}
    for sd in (1,2,3):
        fs=sorted(D.glob(f'seed{sd}_ret_*.json'))
        if not fs:continue
        ks=sorted({r['k'] for f in fs for r in json.load(open(f))['rows']})
        lk=leak_audit.leaky_pairs(sd,1)[62]
        states=[f.name.replace(f'seed{sd}_ret_','').replace('.json','') for f in fs]
        ok=(ks==[62] and not lk)
        res[sd]={'ks':ks,'leaky_pairs_k62':sorted(lk),'valid':ok}
        L.append(f"| {sd} | v1 | {ks} | {len(lk)} | {len(lk)} | {', '.join(states)} | {'yes' if ok else 'NO'} |")
    L.append('\nStage 3 uses only the final (16-round) snapshot, where the v1 layout has no later-round sentence before the probe chunk (the leak existed only for the 1- and 4-round snapshots). The unrelated-history states (U1/U2) stream other worlds that contain none of the probed entities. Stage 3 results are therefore VALID but come from the v1 layout only; no Stage 3 run exists for clean v2 seeds (not run: no new GPU experiments), so replication on v2 is pending.\n')
    return res,L
def run():
    md=[];out={}
    for name,fn in (('q2',q2_decomp),('stage1h',stage1h),('kv2x2',kv2x2),('strict',strict_subset),('reappear',stage1_reappearance),('l3',l3_checks),('s3valid',stage3_validity)):
        try:r,l=fn();out[name]=r;md+=l
        except Exception as e:md.append(f'\n_{name} failed: {e!r}_\n')
    return out,md
if __name__=='__main__':
    o,m=run();print('\n'.join(m))
