"""E55 32.10 CPU analysis of the sentence dump: the V1-V5 grid, evaluated against the label file (the ONLY place where labels are opened; the selection code e55_sentsel.py never sees them).
  python3 e55_sentgrid.py --dump e55_sentdump_s61.npz --labels e55_sentdump_labels_s61.npz [--out report.md]      |      --selftest
Dump keys '<state>/c<k>/<name>' (state = NAT / SENTPQADAMH), label keys 'c<k>/<name>'. Metrics per (state, chunk) and pooled over the chunks of a state (token-weighted): target = all fact sentences of the window (label fact >= 0); recall = hidden share of the fact-sentence tokens; precision = fact share of the hidden tokens (chance = the fact share of the window); hidden share of the window; background false-mask = hidden share of the background tokens. Pass rules (fixed before any data): STRICT recall >= 0.8 and precision >= 0.8 and hidden <= 0.40; RELAXED recall >= 0.75 and precision >= 0.65 and hidden <= 0.45. The pass is judged on the pooled numbers of the state SENTPQADAMH (the arm where a mask would be used) and ALSO reported for NAT."""
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
    return out
def counts(hid,d,lab):
    """token counts of one (state, chunk): hidden-fact, fact, hidden, window, hidden-bg, bg"""
    wl=np.asarray(d['w_len'],np.int64);fact=np.asarray(lab['w_fact'])>=0;h=np.asarray(hid,bool)
    return np.array([wl[h&fact].sum(),wl[fact].sum(),wl[h].sum(),wl.sum(),wl[h&~fact].sum(),wl[~fact].sum()],np.float64)
def metrics(c):
    return {'recall':c[0]/max(c[1],1),'precision':c[0]/c[2] if c[2]>0 else float('nan'),'hidden':c[2]/max(c[3],1),'bg_false':c[4]/max(c[5],1),'chance':c[1]/max(c[3],1)}
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
            tot=np.zeros(6)
            for k,d in cs.items():
                c=counts(SEL.select(d,cfg),d,lab[k]);per[(st,k)]=metrics(c);tot+=c
            per[(st,'pooled')]=metrics(tot)
        rows.append((cfg,per))
    return rows
def report(rows,chunks,out=None):
    sts=sorted(chunks);ks=sorted(next(iter(chunks.values())))
    key='SENTPQADAMH' if 'SENTPQADAMH' in sts else sts[0]
    L=['# E55 32.10 sentence selection grid (label-free scores, labels used only for the metrics)','',f'States {sts}, chunks {ks}. Chance precision = fact share of the window. Pass rules fixed before the data: STRICT recall >= {STRICT[0]}, precision >= {STRICT[1]}, hidden <= {STRICT[2]}; RELAXED recall >= {RELAXED[0]}, precision >= {RELAXED[1]}, hidden <= {RELAXED[2]}. Judged on the pooled numbers of state {key}; NAT shown too.','']
    sc=lambda m:(m['recall']+(0 if np.isnan(m['precision']) else m['precision'])) if m['hidden']<=0.5 else -1
    ranked=sorted(rows,key=lambda r:-sc(r[1][(key,'pooled')]))
    nS=sum(passes(r[1][(key,'pooled')],STRICT) for r in rows);nR=sum(passes(r[1][(key,'pooled')],RELAXED) for r in rows)
    L+=[f'configurations: {len(rows)}; STRICT pass: {nS}; RELAXED pass: {nR} (state {key}, pooled over chunks)','']
    if nS:L.append('**Verdict: STRICT criterion met by at least one configuration.**')
    elif nR:L.append('**Verdict: RELAXED criterion met (not strict): proceed to the GPU arm test with this selection.**')
    else:L.append('**Verdict: no configuration meets even the RELAXED criterion: label-free sentence masking from this attention is not viable; change direction (lexical prefilter / learned classifier).**')
    L+=['','## top 20 configurations by recall + precision (hidden <= 0.5), state '+key+', pooled','','| config | group | heads | recall | precision | hidden | bg false | chance | strict | relaxed | NAT recall / precision / hidden |','|---|---|---|---|---|---|---|---|---|---|---|']
    for cfg,per in ranked[:20]:
        m=per[(key,'pooled')];n=per.get(('NAT','pooled')) if key!='NAT' else None
        L.append(f"| {cfg['name']} | {cfg['group']} | {cfg['heads']} | {m['recall']:.3f} | {m['precision']:.3f} | {m['hidden']:.3f} | {m['bg_false']:.3f} | {m['chance']:.3f} | {'yes' if passes(m,STRICT) else 'no'} | {'yes' if passes(m,RELAXED) else 'no'} | "+(f"{n['recall']:.3f} / {n['precision']:.3f} / {n['hidden']:.3f}" if n else '-')+' |')
    cfg,per=ranked[0];L+=['',f'## best configuration ({cfg["name"]}, {cfg["group"]}, {cfg["heads"]}) per state and chunk','','| state | chunk | recall | precision | hidden | bg false | chance |','|---|---|---|---|---|---|---|']
    for st in sts:
        for k in ks+['pooled']:
            m=per[(st,k)];L.append(f"| {st} | {k} | {m['recall']:.3f} | {m['precision']:.3f} | {m['hidden']:.3f} | {m['bg_false']:.3f} | {m['chance']:.3f} |")
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
        d={'A':A.astype(np.float16),'q_ce':rng.random(nQ).astype(np.float32),'q_len':rng.integers(5,30,nQ),'w_len':wl,'w_start':np.cumsum(np.r_[0,wl[:-1]]),'w_chunk':np.zeros(nW,np.int32),'win_ids':rng.integers(0,500,int(wl.sum())+10)}
        for st in ('NAT','SENTPQADAMH'):chunks.setdefault(st,{})[k]=d
        lab[k]={'w_fact':np.where(fact,np.arange(nW)%16,-1).astype(np.int16)}
    rows=run(chunks,lab);t=report(rows,chunks)
    best=max(rows,key=lambda r:r[1][('SENTPQADAMH','pooled')]['recall']+r[1][('SENTPQADAMH','pooled')]['precision']);m=best[1][('SENTPQADAMH','pooled')]
    assert m['precision']>0.9 and 'Verdict' in t and len(rows)>100,(m,len(rows))
    print('e55_sentgrid selftest ok:',len(rows),'configs, planted-signal best',{k:round(v,3) for k,v in m.items()})
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--selftest',action='store_true');ap.add_argument('--dump');ap.add_argument('--labels');ap.add_argument('--out');a=ap.parse_args()
    if a.selftest:selftest();sys.exit(0)
    chunks,lab=load(a.dump,a.labels);print(report(run(chunks,lab),chunks,a.out))
