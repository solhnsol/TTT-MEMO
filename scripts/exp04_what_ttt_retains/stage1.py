"""Stage 1: token-level NTP gain decomposition on held-out narrative streams (labels CPU-only, runner needs Runtime)."""
import json,re,sys,time
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
CATS=['NAME_FIRST','NAME_CONT','LEX','OTHER'];DIST=['LOC','FAR','NEW']
def pieces(tok,ids):return [tok.decode([i]) for i in ids]
def label_stream(tok,ids,names,window=8192):
    """per-token labels (category,dist) for predicting ids[t] given ids[:t] (label describes the TARGET token t)."""
    pcs=pieces(tok,ids);n=len(ids);cat=['OTHER']*n;dist=['NA']*n
    # group tokens into words
    words=[];cur=None
    for t,p in enumerate(pcs):
        starts=p[:1] in (' ','\n') or cur is None or not re.match(r"[A-Za-z'’-]",p[:1] or ' ')
        letters=re.match(r"\s*([A-Za-z][A-Za-z'’-]*)",p)
        if letters and (starts or cur is not None) and (p[:1] in (' ','\n') or cur is None) :
            cur={'w':letters.group(1),'toks':[t]};words.append(cur)
        elif cur is not None and re.fullmatch(r"[A-Za-z'’-]+",p):cur['toks'].append(t);cur['w']+=p
        else:cur=None
    last={}
    for wd in words:
        w=wd['w'];t0=wd['toks'][0];prev=''.join(pcs[max(0,t0-3):t0]).rstrip(' ')
        sent_start=(not prev) or prev[-1] in '.!?\n“"‘'
        isname=w[0].isupper() and w in names and not sent_start
        islex=(w.islower() and len(w)>=7)
        key=w
        if isname or islex:
            if key in last:d='LOC' if t0-last[key]<=window else 'FAR'
            else:d='NEW'
            for j,t in enumerate(wd['toks']):
                cat[t]=('NAME_FIRST' if j==0 else 'NAME_CONT') if isname else 'LEX';dist[t]=d
        last[key]=wd['toks'][-1]
    return cat,dist
def agg(lp,cat,dist,sel):
    """mean NLL by category x distance over positions sel (array of target indices into lp)"""
    out={}
    for c in CATS:
        for d in (DIST if c!='OTHER' else ['NA']):
            idx=[t for t in sel if cat[t]==c and dist[t]==d]
            if idx:out[f'{c}|{d}']=(float(-np.mean(lp[idx])),len(idx))
    return out
def run_stage1(rt,datadir,outdir,doc_slice,deadline=None):
    outdir=Path(outdir);outdir.mkdir(parents=True,exist_ok=True)
    docs=[json.loads(l) for l in open(Path(datadir)/'stage1_docs.jsonl')]
    done=0
    for di in doc_slice:
        d=docs[di];tag=d['doc'].replace('.txt','')
        for cond in ('REL','UNREL','CF','SHUF','NONE'):
            for on in (True,False):
                f=outdir/f'{tag}_{cond}_{"ON" if on else "OFF"}.npz'
                if f.exists():continue
                if deadline and time.time()>deadline:return done
                ids=d['conds'][cond];t0=time.monotonic()
                lp,_=rt.stream(ids,on)
                np.savez_compressed(f,lp=lp.astype(np.float32),ids=np.asarray(ids,dtype=np.int32),sec=time.monotonic()-t0)
                done+=1;print('stage1',tag,cond,'ON' if on else 'OFF',round(time.monotonic()-t0,1),float(-lp[-4096:].mean()),flush=True)
    return done

def run_stage1h(rt,datadir,outdir,doc_slice,deadline=None):
    """Stage 1H (hybrid histories): isolate relevant information that is INSIDE vs OUTSIDE the last 8192 tokens before F.
    FARREL : [relevant first 16 chunks][UNRELATED last 8 chunks] + F   -> relevant info only beyond the local window
    NEARREL: [UNRELATED first 16 chunks][relevant last 8 chunks] + F   -> relevant info only inside the local window
    Same token counts and identical F as REL/UNREL of Stage 1."""
    outdir=Path(outdir);outdir.mkdir(parents=True,exist_ok=True)
    docs=[json.loads(l) for l in open(Path(datadir)/'stage1_docs.jsonl')];done=0
    for di in doc_slice:
        d=docs[di];tag=d['doc'].replace('.txt','');HN=24*1024
        Hrel=d['pre_ids'][-HN:];Hun=d['conds']['UNREL'][1:1+HN];F=d['F']
        assert len(Hrel)==HN and len(Hun)==HN and d['conds']['REL'][1:1+HN]==Hrel and d['conds']['REL'][1+HN:]==F
        conds={'FARREL':[128000]+Hrel[:16384]+Hun[-8192:]+F,'NEARREL':[128000]+Hun[:16384]+Hrel[-8192:]+F}
        for cond,ids in conds.items():
            assert len(ids)==HN+len(F)+1
            for on in (True,False):
                f=outdir/f'{tag}_{cond}_{"ON" if on else "OFF"}.npz'
                if f.exists():continue
                if deadline and time.time()>deadline:return done
                t0=time.monotonic();lp,_=rt.stream(ids,on)
                np.savez_compressed(f,lp=lp.astype(np.float32),ids=np.asarray(ids,dtype=np.int32),sec=time.monotonic()-t0);done+=1
                print('stage1h',tag,cond,'ON' if on else 'OFF',round(time.monotonic()-t0,1),flush=True)
    return done
