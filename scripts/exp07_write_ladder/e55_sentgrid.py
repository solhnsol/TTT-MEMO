"""E55 32.10 CPU analysis of the sentence dump: the V1-V5 grid, evaluated against the label file (the ONLY place where labels are opened; the selection code e55_sentsel.py never sees them).
  python3 e55_sentgrid.py --dump e55_sentdump_s61.npz --labels e55_sentdump_labels_s61.npz [--out report.md]      |      --selftest
Dump keys '<state>/c<k>/<name>' (state = NAT / SENTPQADAMH), label keys 'c<k>/<name>'. Metrics per (state, chunk) and pooled over the chunks of a state (token-weighted): target = all fact sentences of the window (label fact >= 0); recall = hidden share of the fact-sentence tokens; precision = fact share of the hidden tokens (chance = the fact share of the window); hidden share of the window; background false-mask = hidden share of the background tokens. Pass rules (fixed before any data): STRICT recall >= 0.8 and precision >= 0.8 and hidden <= 0.40; RELAXED recall >= 0.75 and precision >= 0.65 and hidden <= 0.45. Multiple-comparison rule: configurations are SELECTED on the first half of the chunks (16, 32; rank by F1) and the pass is judged on the CONFIRMATION half (64, 127) for the selection-top-1 configuration, state SENTPQADAMH (the arm where a mask would be used); NAT is shown too. Metrics: accuracy, precision, recall, F1 (primary) and hidden share. Groups V1-V5 (sentence-mass A), V6 (sharpest heads + k-times trigger), V7 (head scores sharp / copy / induction, sink and previous-chunk exclusion, attention x V-norm, fact-like query tokens only, sentence aggregation sum / max / top-5 mean, exact n-gram string baseline STR)."""
import sys,json,argparse,itertools
from pathlib import Path
import numpy as np
HERE=Path(__file__).resolve().parent;sys.path.insert(0,str(HERE))
import e55_sentsel as SEL
STRICT=(0.8,0.8,0.40);RELAXED=(0.75,0.65,0.45)
def configs():
    out=[]
    for group,heads in itertools.product(('g2','g3','all'),('sum','head')):
        base=dict(group=group,heads=heads)
        for nv in (1,2,3):
            for q in (30,40,50):out.append(dict(base,name=f'V2 nvote{nv} top{q}%',nvote=nv,rule='rank',rparam=q))
            for z in (1.0,1.5,2.0):out.append(dict(base,name=f'V3 nvote{nv} z{z}',nvote=nv,rule='z',rparam=z))
            for q in (30,40):out.append(dict(base,name=f'V1 gate50 nvote{nv} top{q}%',gate='sharp',ngate=0.5,nvote=nv,rule='rank',rparam=q))
            for ce in (0.25,0.5):out.append(dict(base,name=f'V4 ce{int(ce*100)}% nvote{nv} top30%',ce=ce,nvote=nv,rule='rank',rparam=30))
            for q in (30,40):out.append(dict(base,name=f'V5 rare nvote{nv} top{q}%',rare=True,nvote=nv,rule='rank',rparam=q))
            out.append(dict(base,name=f'A V1+V4 nvote{nv} top30%',gate='sharp',ngate=0.5,ce=0.5,nvote=nv,rule='rank',rparam=30))
            out.append(dict(base,name=f'A V1+V5 nvote{nv} top30%',gate='sharp',ngate=0.5,rare=True,nvote=nv,rule='rank',rparam=30))
    for hs,sel in itertools.product(('all','g3'),('hs','tok')):                 # V6: sharpest heads + k-times trigger
        for hp in (10,25,50):
            for thr in ((0.2,) if sel=='hs' else (0.1,0.2,0.4)):
                for trig,ks in (('q',(1,2,3)),('h',(2,4,8))):
                    for k in ks:out.append(dict(name=f'V6 {sel} heads{hp}% {hs}'+(f' thr{thr}' if sel=='tok' else '')+f' trig{trig}>={k}',v6=True,sel=sel,hgroup=hs,hpct=hp,thr=thr,trig=trig,k=k,group=hs,heads='sharp'))
    for kind,hp,ex,k in itertools.product(('sharp','copy','induct'),(5,10,25),(0,1),(1,2,3)):          # V7: other head scores, sink / recent exclusion, token votes
        out.append(dict(name=f'V7 tok {kind} heads{hp}% thr0.2 '+('excl sink+prev chunk' if ex else 'no excl')+f' q>={k}',v7=True,sel='tok',kind=kind,hpct=hp,thr=0.2,nsink=4*ex,norecent=ex,trig='q',k=k,group='all',heads='sel'))
    for kind,hp,rp in itertools.product(('copy','induct'),(5,10),(30,40)):out.append(dict(name=f'V7 tok {kind} heads{hp}% excl rank top{rp}%',v7=True,sel='tok',kind=kind,hpct=hp,thr=0.2,nsink=4,norecent=1,trig='rank',rparam=rp,group='all',heads='sel'))
    for kind,thr in itertools.product(('sharp','copy'),(1.0,2.0)):out.append(dict(name=f'V7 tok x|v| {kind} heads10% rel-thr{thr} excl q>=2',v7=True,sel='tok',kind=kind,hpct=10,thr=thr*0.1,av=True,nsink=4,norecent=1,trig='q',k=2,group='all',heads='sel'))
    for qf,kind,hp,k in itertools.product(('ce','rare'),('sharp','copy'),(10,25),(1,2)):out.append(dict(name=f'V7 tok fact-like query tokens only ({qf}) {kind} heads{hp}% excl q>={k}',v7=True,sel='tok',kind=kind,hpct=hp,thr=0.2,qf=qf,nsink=4,norecent=1,trig='q',k=k,group='all',heads='sel'))
    for src,agg,kind,hp,k in itertools.product(('AL','AV'),('sum','max','top5'),('sharp','copy'),(5,10,25),(1,2,3)):
        out.append(dict(name=f'V7 sentence {src} agg-{agg} {kind} heads{hp}% excl q>={k}',v7=True,sel='hsagg',src=src,agg=agg,kind=kind,hpct=hp,nsink=4,norecent=1,trig='q',k=k,group='all',heads='sel'))
    for n in (6,10,16):out.append(dict(name=f'STR exact {n}-gram match with the write chunk',v7=True,sel='str',n=n,group='all',heads='string'))
    return out
def counts(hid,d,lab):
    """token counts of one (state, chunk): hidden-fact, fact, hidden, window, hidden-bg, bg"""
    wl=np.asarray(d['w_len'],np.int64);fact=np.asarray(lab['w_fact'])>=0;h=np.asarray(hid,bool)
    return np.array([wl[h&fact].sum(),wl[fact].sum(),wl[h].sum(),wl.sum(),wl[h&~fact].sum(),wl[~fact].sum()],np.float64)
def metrics(c):
    p=c[0]/c[2] if c[2]>0 else 0.0;r=c[0]/max(c[1],1)
    return {'acc':(c[0]+c[5]-c[4])/max(c[3],1),'f1':2*p*r/(p+r) if p+r>0 else 0.0,'recall':c[0]/max(c[1],1),'precision':c[0]/c[2] if c[2]>0 else float('nan'),'hidden':c[2]/max(c[3],1),'bg_false':c[4]/max(c[5],1),'chance':c[1]/max(c[3],1)}
def passes(m,rule):return bool(m['recall']>=rule[0] and (m['precision']>=rule[1]) and m['hidden']<=rule[2])
def load(dump,labels):
    D=np.load(dump);Lb=np.load(labels);states={};chunks={}
    for key in D.files:
        st,ck,nm=key.split('/');chunks.setdefault(st,{}).setdefault(int(ck[1:]),{})[nm]=D[key]
    lab={};
    for key in Lb.files:
        ck,nm=key.split('/');lab.setdefault(int(ck[1:]),{})[nm]=Lb[key]
    return chunks,lab
def run(chunks,lab,cfgs=None):
    cfgs=cfgs or configs();rows=[]
    for cfg in cfgs:
        per={};
        for st,cs in chunks.items():
            tot=np.zeros(6);ksr=sorted(cs);half=len(ksr)//2;ts=np.zeros(6);tc=np.zeros(6)
            for k,d in cs.items():
                c=counts((SEL.select_str if cfg.get('sel')=='str' else SEL.select_v7 if cfg.get('v7') else SEL.select_heads if cfg.get('v6') else SEL.select)(d,cfg),d,lab[k]);per[(st,k)]=metrics(c);tot+=c
                if ksr.index(k)<half:ts+=c
                else:tc+=c
            per[(st,'pooled')]=metrics(tot);per[(st,'sel')]=metrics(ts);per[(st,'conf')]=metrics(tc)
        rows.append((cfg,per))
    return rows
def report(rows,chunks,out=None):
    sts=sorted(chunks);ks=sorted(next(iter(chunks.values())))
    key='SENTPQADAMH' if 'SENTPQADAMH' in sts else sts[0]
    L=['# E55 32.10 sentence selection grid (label-free scores, labels used only for the metrics)','',f'States {sts}, chunks {ks}. Chance precision = fact share of the window. Pass rules fixed before the data: STRICT recall >= {STRICT[0]}, precision >= {STRICT[1]}, hidden <= {STRICT[2]}; RELAXED recall >= {RELAXED[0]}, precision >= {RELAXED[1]}, hidden <= {RELAXED[2]}. Judged on the pooled numbers of state {key}; NAT shown too.','']
    sc=lambda m:m['f1'];ksr=sorted(ks);half=len(ksr)//2
    L+=[f'**Multiple-comparison rule (fixed before the data):** the configuration is SELECTED on the first half of the dump chunks {ksr[:half]} (rank by F1 of state {key}) and CONFIRMED on the second half {ksr[half:]}; the verdict uses ONLY the confirmation numbers of the selection-top-1 configuration (and, as a robustness line, of the selection-top-5). The pooled numbers over all chunks are descriptive only. A different-seed dump would be used the same way (select on one seed, confirm on the other).','']
    ranked=sorted(rows,key=lambda r:-sc(r[1][(key,'sel')]));top1=ranked[0];c1=top1[1][(key,'conf')]
    nS5=sum(passes(r[1][(key,'conf')],STRICT) for r in ranked[:5]);nR5=sum(passes(r[1][(key,'conf')],RELAXED) for r in ranked[:5])
    L+=[f'configurations: {len(rows)}; selection-top-1: {top1[0]["name"]} (selection F1 {top1[1][(key,"sel")]["f1"]:.3f}); its CONFIRMATION recall {c1["recall"]:.3f}, precision {c1["precision"]:.3f}, hidden {c1["hidden"]:.3f}, F1 {c1["f1"]:.3f}; confirmation passes among selection-top-5: STRICT {nS5}, RELAXED {nR5}.','']
    if passes(c1,STRICT):L.append('**Verdict: STRICT criterion met on the confirmation chunks by the selection-top-1 configuration.**')
    elif passes(c1,RELAXED):L.append('**Verdict: RELAXED criterion met on the confirmation chunks (not strict): proceed to the GPU arm test with this selection.**')
    else:L.append('**Verdict: the selection-top-1 configuration does not meet even the RELAXED criterion on the confirmation chunks: label-free sentence masking from this attention is not viable in this form; change direction (lexical prefilter / learned classifier).**')
    L+=['','## top 20 configurations by SELECTION F1 (chunks '+str(ksr[:half])+'), with the confirmation numbers (chunks '+str(ksr[half:])+') in the right columns; metrics over the window tokens, target = fact-sentence tokens, state '+key,'','| config | group | heads | sel F1 | sel recall | sel precision | conf F1 | conf accuracy | conf recall | conf precision | conf hidden | conf bg false | chance | strict | relaxed | NAT conf recall / precision / hidden |','|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|']
    for cfg,per in ranked[:20]:
        m=per[(key,'conf')];sl=per[(key,'sel')];n=per.get(('NAT','conf')) if key!='NAT' else None
        L.append(f"| {cfg['name']} | {cfg['group']} | {cfg['heads']} | {sl['f1']:.3f} | {sl['recall']:.3f} | {sl['precision']:.3f} | {m['f1']:.3f} | {m['acc']:.3f} | {m['recall']:.3f} | {m['precision']:.3f} | {m['hidden']:.3f} | {m['bg_false']:.3f} | {m['chance']:.3f} | {'yes' if passes(m,STRICT) else 'no'} | {'yes' if passes(m,RELAXED) else 'no'} | "+(f"{n['recall']:.3f} / {n['precision']:.3f} / {n['hidden']:.3f}" if n else '-')+' |')
    strs=[r for r in rows if r[0].get('sel')=='str'];hb=[r for r in ranked if r[0].get('sel')!='str'][:1]
    if strs and hb:
        L+=['','## string-match baseline (exact n-gram overlap of window sentences with the write chunk, no attention) and its overlap with the best attention configuration','','| config | F1 | recall | precision | hidden |','|---|---|---|---|---|']
        for cfg_,per_ in strs:
            m=per_[(key,'pooled')];L.append(f"| {cfg_['name']} | {m['f1']:.3f} | {m['recall']:.3f} | {m['precision']:.3f} | {m['hidden']:.3f} |")
        bc=hb[0][0];ov=[];
        for k_,d_ in chunks[key].items():
            a_=(SEL.select_v7 if bc.get('v7') else SEL.select_heads if bc.get('v6') else SEL.select)(d_,bc);b_=SEL.select_str(d_,strs[0][0]);wl_=np.asarray(d_['w_len'])
            ov.append((wl_[a_&b_].sum(),wl_[a_|b_].sum()))
        L.append('');L.append(f"Token overlap (Jaccard) of the best attention configuration ({bc['name']}) with {strs[0][0]['name']}: {sum(o[0] for o in ov)/max(sum(o[1] for o in ov),1):.3f} (state {key}, pooled).")
    cfg,per=ranked[0];L+=['',f'## best configuration ({cfg["name"]}, {cfg["group"]}, {cfg["heads"]}) per state and chunk','','| state | chunk | F1 | accuracy | recall | precision | hidden | bg false | chance |','|---|---|---|---|---|---|---|---|---|']
    for st in sts:
        for k in ks+['pooled']:
            m=per[(st,k)];L.append(f"| {st} | {k} | {m['f1']:.3f} | {m['acc']:.3f} | {m['recall']:.3f} | {m['precision']:.3f} | {m['hidden']:.3f} | {m['bg_false']:.3f} | {m['chance']:.3f} |")
    t='\n'.join(L)+'\n'
    if out:Path(out).write_text(t)
    return t
def selftest():
    import tempfile,re
    src=open(HERE/'e55_sentsel.py').read();assert 'e55_sentdump_labels' not in src.replace('must not import e55_sentdump_labels','') and 'np.load' not in src and 'open(' not in src,'selection code touches labels'
    rng=np.random.default_rng(0);H=4;nQ=12;nW=60;chunks={};lab={}
    for k in (16,32):
        wl=rng.integers(5,20,nW);fact=np.zeros(nW,bool);fact[::3]=True
        A=rng.random((3,H,nQ,nW))*0.002
        for q in range(nQ):A[:,:,q,np.nonzero(fact)[0][q%20]]+=0.2        # planted: every query attends a fact sentence
        nL=6;AL=rng.random((nL,H,nQ,nW))*0.002;Tm=rng.random((nL,H,1024))*0.05;Tw=rng.integers(0,nW,(nL,H,1024));qlen=rng.integers(5,30,nQ);qst=np.cumsum(np.r_[0,qlen[:-1]]);fidx=np.nonzero(fact)[0]
        for q in range(nQ):                        # planted: heads 0,1 of every layer are sharp on the query tokens and point at a fact sentence
            f_=fidx[q%len(fidx)];AL[:,:,q,f_]+=0.2;Tm[:,:2,qst[q]:qst[q]+qlen[q]]=0.6;Tw[:,:2,qst[q]:qst[q]+qlen[q]]=f_
        wst=np.cumsum(np.r_[0,wl[:-1]]);wid_=rng.integers(0,500,int(wl.sum())+10);qid_=rng.integers(0,500,1024);Tk=rng.integers(0,len(wid_),(nL,H,1024))
        for q in range(nQ):
            f_=fidx[q%len(fidx)]
            for t in range(qst[q],qst[q]+qlen[q]):Tk[:,:2,t]=wst[f_]+(t-qst[q])%wl[f_];wid_[wst[f_]+(t-qst[q])%wl[f_]]=qid_[t]
        d={'T_key':Tk.astype(np.int16),'VN':rng.random((nL,2,len(wid_))).astype(np.float16)+0.5,'AV':AL.astype(np.float16),'win_ids':wid_,'q_ids':qid_,'AL':AL.astype(np.float16),'T_max':Tm.astype(np.float16),'T_sum':Tm.astype(np.float16),'T_ent':Tm.astype(np.float16),'T_wsid':Tw.astype(np.int16),'q_start':qst,'q_tce':rng.random(1024).astype(np.float16),'A':A.astype(np.float16),'q_ce':rng.random(nQ).astype(np.float32),'q_len':qlen,'w_len':wl,'w_start':wst,'w_chunk':(np.arange(nW)*8//nW).astype(np.int32)}
        for st in ('NAT','SENTPQADAMH'):chunks.setdefault(st,{})[k]=d
        lab[k]={'w_fact':np.where(fact,np.arange(nW)%16,-1).astype(np.int16)}
    rows=run(chunks,lab);t=report(rows,chunks)
    best=max(rows,key=lambda r:r[1][('SENTPQADAMH','pooled')]['recall']+r[1][('SENTPQADAMH','pooled')]['precision']);m=best[1][('SENTPQADAMH','pooled')]
    assert m['precision']>0.9 and 'Verdict' in t and len(rows)>100,(m,len(rows))
    v6=[r for r in rows if r[0].get('v6')];assert len(v6)>100;b6=max(v6,key=lambda r:r[1][('SENTPQADAMH','pooled')]['f1']);m6=b6[1][('SENTPQADAMH','pooled')];assert m6['f1']>0.6 and m6['precision']>0.9 and 0<=m6['acc']<=1,(b6[0]['name'],m6)
    v7=[r for r in rows if r[0].get('v7')];assert len(v7)>150;b7=max(v7,key=lambda r:r[1][('SENTPQADAMH','pooled')]['f1']);m7=b7[1][('SENTPQADAMH','pooled')];assert m7['f1']>0.4 and m7['precision']>0.8,(b7[0]['name'],m7)
    assert any(r[0]['name'].startswith('STR') for r in rows);print('V7 best',b7[0]['name'],{k:round(float(v),3) for k,v in m7.items()})
    for kd in ('copy','induct','sharp'):
        li,hi=SEL.pick_heads(chunks['NAT'][16],kd,'all',50);assert set(hi.tolist())<={0,1},(kd,hi)
    sel=SEL.head_rank(chunks['NAT'][16],'all',50);assert sel[:,:2].all() and not sel[:,2:].any(),sel
    print('V6 best',b6[0]['name'],{k:round(v,3) for k,v in m6.items()})
    print('e55_sentgrid selftest ok:',len(rows),'configs, planted-signal best',{k:round(v,3) for k,v in m.items()})
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--selftest',action='store_true');ap.add_argument('--dump');ap.add_argument('--labels');ap.add_argument('--out');a=ap.parse_args()
    if a.selftest:selftest();sys.exit(0)
    chunks,lab=load(a.dump,a.labels);print(report(run(chunks,lab),chunks,a.out))
