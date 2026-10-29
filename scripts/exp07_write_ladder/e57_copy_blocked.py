"""E57 per question: share of the copy paths blocked by the query-band masks vs the question's accuracy / rank ratio (CPU only; new file; no article text in the output). Answer rows = the rows with label A of e57_gain_labels_rows.npz (row predicts the token position pos = chunk*1024 + i + 1 inside an answer-string hit of a bank question), chunks 8..279. COPYABLE = the 4-gram (ids[q-2..q] = previous 3 tokens including the query token, + the target ids[q+1]) occurred before in the window outside the band 512, i.e. a match at j with 512 <= q-(j-1) <= 8191 and NO match inside the band (zone FAR of e57_copy_check.py; also flagged: all FAR matches in articles of the same event). Blocked = the row is a banded row of the mask: GAINQ512 p=.26 (e57_gainq_p0.26), GAINQ512 p=.10 (e57_gainq_p0.10), RANDQ512 p=.10 (e57_randq_p0.10) (no p=.50 mask exists on the server). Per question: A rows, copyable rows, and for copyable / not copyable rows the number and share blocked by each mask; the reads of the written states (cycle 1 end, OUT and IN, the 42-question rows): acc_all, rank_ratio_all of W0, NATADAMH, GAINQ512 p=.26, GAINQ512 p=.10, RANDQ512 p=.10; difference GAINQ - NATADAMH and GAINQ - RANDQ; Spearman correlation across the questions (with at least 1 copyable row) between the blocked share of the copyable rows and the rank-ratio / accuracy difference.
  .venv/bin/python scripts/exp07_write_ladder/e57_copy_blocked.py --bundle B --rows R.npz --exp E --out O.json --npz P.npz"""
import sys,json,pathlib,argparse,collections
import numpy as np
HERE=pathlib.Path(__file__).resolve().parent
for p in (HERE,HERE/'e57',HERE.parents[1]/'scripts/exp04_what_ttt_retains',HERE.parents[1]/'scripts/exp05_write_dynamics'):sys.path.insert(0,str(p))
import e57_closure_masks as CM
CH=1024
def spearman(x,y):
    x=np.asarray(x,float);y=np.asarray(y,float);m=~(np.isnan(x)|np.isnan(y));x=x[m];y=y[m]
    if len(x)<4 or x.std()==0 or y.std()==0:return None,int(m.sum())
    rx=np.argsort(np.argsort(x));ry=np.argsort(np.argsort(y));return float(np.corrcoef(rx,ry)[0,1]),int(m.sum())
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--bundle');ap.add_argument('--rows');ap.add_argument('--exp');ap.add_argument('--out');ap.add_argument('--npz');ap.add_argument('--tokenizer');a=ap.parse_args()
    import gen_world_ctl as G,e57_reads as RD
    E=pathlib.Path(a.exp);enc=G.load_enc(a.tokenizer or str(HERE.parents[1]/'data/tokenizer/llama3/tokenizer.json'))
    items=RD.load_items(pathlib.Path(a.bundle)/'e57_bank'/'bank_v1.json');ids,iv,hits=RD.article_intervals(a.bundle,enc,items);ids=np.asarray(ids);starts=np.array([x[0] for x in iv]);evs=np.array([hash(x[2])%10**9 for x in iv])
    ev_of=lambda p:evs[np.searchsorted(starts,p,side='right')-1]
    R=np.load(a.rows);pos=R['pos'];lab=R['label'];qq=R['question'];sel=np.nonzero(lab==3)[0];D=collections.defaultdict(list)
    for j in range(3,len(ids)):D[tuple(int(x) for x in ids[j-3:j+1])].append(j)
    D={k:np.asarray(v) for k,v in D.items()}
    P=pos[sel];q=P-1;K=q//CH;I=q%CH;nfar=np.zeros(len(sel),int);nband=np.zeros(len(sel),int);same_all=np.zeros(len(sel),bool)
    for r in range(len(sel)):
        qq_=int(q[r]);js=D.get(tuple(int(x) for x in ids[qq_-2:qq_+2]))
        if js is None:continue
        js=js[js<=qq_];band=js[js>=qq_-510];far=js[(js<=qq_-511)&(js>=qq_-8190)];nband[r]=len(band);nfar[r]=len(far)
        if len(far):same_all[r]=bool((ev_of(far-1)==ev_of(qq_)).all())
    copyable=(nfar>0)&(nband==0)
    MK={'GAINQ512_p26':(E/'e57_gainq_p0.26/e57_qband_masks.npz','GAINQ512'),'GAINQ512_p10':(E/'e57_gainq_p0.10/e57_qband_masks.npz','GAINQ512'),'RANDQ512_p10':(E/'e57_randq_p0.10/e57_qband_masks.npz','RANDQ512')}
    blocked={}
    for nm,(f,arm) in MK.items():
        M=np.load(f,allow_pickle=True);b=np.zeros(len(sel),bool);cache={}
        for r in range(len(sel)):
            k=int(K[r])
            if k not in cache:cache[k]=CM.up(M[f'qb_{arm}_{k}'],CH)
            b[r]=bool(cache[k][int(I[r])])
        blocked[nm]=b
    np.savez_compressed(a.npz,pos=P,question=qq[sel],chunk=K,row=I,far_matches=nfar,band_matches=nband,copyable=copyable,far_all_same_event=same_all,**{f'blocked_{k}':v for k,v in blocked.items()})
    reads={}
    SRC={'W0':(E/'e57_main/E57_main_W0.json',1),'NATADAMH':(E/'e57_main/E57_main_NATADAMH.json',1),'GAINQ512_p26':(E/'e57_gainq_p0.26/E57_main_GAINQ512.json',1),'GAINQ512_p10':(E/'e57_gainq_p0.10/E57_main_GAINQ512.json',1),'RANDQ512_p10':(E/'e57_randq_p0.10/E57_main_RANDQ512.json',1)}
    for nm,(f,cyc) in SRC.items():
        rows=json.load(open(f));rows=rows['rows'] if isinstance(rows,dict) and 'rows' in rows else rows
        for r in rows:
            if cyc is not None and r.get('cycle')!=cyc:continue
            if r.get('arm') not in (None,'W0',nm.split('_')[0]) and nm!='W0':continue
            if nm=='W0' and r.get('read_chunk') not in (None,279,559):pass
            reads.setdefault(r['question'],{}).setdefault(nm,{})[r['condition']]=dict(acc=float(r['acc_all']),rank_ratio=float(r['rank_ratio_all']))
    out={'definition':__doc__.split('\n')[0],'A_rows':int(len(sel)),'copyable_rows':int(copyable.sum()),'not_copyable_rows':int((~copyable).sum()),'per_question':{}}
    qids=[it['id'] for it in items];xs={m:[] for m in blocked};dy={}
    for qi,it in enumerate(items):
        m=qq[sel]==qi;d=dict(rows_A=int(m.sum()),copyable=int((m&copyable).sum()),not_copyable=int((m&~copyable).sum()),copyable_same_event_all=int((m&copyable&same_all).sum()))
        for nm,b in blocked.items():
            c=m&copyable;n=m&~copyable;d[nm]=dict(copyable_blocked=int((c&b).sum()),copyable_share=float((c&b).sum()/c.sum()) if c.sum() else None,not_copyable_blocked=int((n&b).sum()),not_copyable_share=float((n&b).sum()/n.sum()) if n.sum() else None)
            xs[nm].append(d[nm]['copyable_share'] if d[nm]['copyable_share'] is not None else np.nan)
        d['reads']=reads.get(it['id'],{});out['per_question'][it['id']]=d
    corr={}
    for cond in ('OUT','IN'):
        for met in ('rank_ratio','acc'):
            for ref in ('NATADAMH','RANDQ512_p10'):
                for tgt in ('GAINQ512_p26','GAINQ512_p10'):
                    y=[ (reads.get(qid,{}).get(tgt,{}).get(cond,{}).get(met,np.nan)-reads.get(qid,{}).get(ref,{}).get(cond,{}).get(met,np.nan)) for qid in qids]
                    for bm in (tgt.replace('512_p26','512_p26').replace('GAINQ512_p26','GAINQ512_p26'),):
                        x=xs['GAINQ512_p26'] if tgt=='GAINQ512_p26' else xs['GAINQ512_p10'];s,nq=spearman(x,y);corr[f'{cond}|{met}|{tgt}-{ref}|vs blocked share of copyable rows by {tgt}']=dict(spearman=s,questions=nq)
    out['spearman']=corr;out['totals']={nm:dict(copyable_blocked=int((copyable&b).sum()),copyable_share=float((copyable&b).sum()/max(copyable.sum(),1)),not_copyable_blocked=int((~copyable&b).sum()),not_copyable_share=float((~copyable&b).sum()/max((~copyable).sum(),1))) for nm,b in blocked.items()}
    json.dump(out,open(a.out,'w'),indent=1,default=float)
    print('A rows',len(sel),'copyable',int(copyable.sum()));print(out['totals']);
    for k,v in corr.items():
        if 'OUT' in k:print(k,v)
if __name__=='__main__':main()
