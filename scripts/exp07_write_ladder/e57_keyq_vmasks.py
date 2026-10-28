"""E57 key-level query masks from the value-corrected dump of job 313 (CPU only; new file, imports the rules of e57_keyq_masks.py). For the names SHV, ALLV (raw value-corrected mass) and SHVn, ALLVn (per-head value-normalised share) it evaluates the rules topN (1..64), tau (.005 .01 .02 .05 .1 .2 as share of the row's far score) and cumq (.1 .25 .5 .75 .9) on the stored TOP-128 far keys of the band (256 or 512) for the GAINQ rows of the even chunks 8..278 (selection set), with the sentence-level recall / precision / F1 of e57_keyq_masks.py (answer sentences = ORCA window sentences with a far key; hidden = >= 2 selected keys inside) and the masking amount (selected / far keys). Odd chunks are the test set (--test). Selection: candidates with recall >= --recall; none passes (limit: ~84 answer sentences per row against at most 128 stored keys) -> per name the highest-recall candidate ('min') and, among the candidates whose masking is closest to 2x of it with recall >= 0.9 x its recall, a second one ('x2'). Mask file as e57_keyq_masks.py (arms KQ<band>V_<name>_<group>_<pass>, KR..., KN...: random / nearest far keys, same count).
  .venv/bin/python scripts/exp07_write_ladder/e57_keyq_vmasks.py --bundle B --dump D --gainq-masks M.npz --pod-masks P.npz --band 512 --out O.npz --report R.json [--arms NAME|GROUP|PASS,...] [--test] | --selftest"""
import sys,json,argparse,pathlib,hashlib
import numpy as np
HERE=pathlib.Path(__file__).resolve().parent
for p in (HERE,HERE/'e57',HERE.parents[1]/'scripts/exp04_what_ttt_retains',HERE.parents[1]/'scripts/exp05_write_dynamics'):sys.path.insert(0,str(p))
import e57_closure_masks as CM
import e57_keyq_masks as KM
CH=1024;WIN=8192;K0=8;K1=280;TOPM=128;NAMES=('SHV','ALLV','SHVn','ALLVn')
GR={'topN':[1,2,4,8,16,32,64],'tau':[.005,.01,.02,.05,.1,.2],'cumq':[.1,.25,.5,.75,.9]}
def shard_file(d,k):return pathlib.Path(d)/f'E57_valuedump_full_s{k//8-1:02d}.npz'
def subset(rows,qb):
    """positions (into the dump rows of a chunk) of the rows that are banded; every banded row must be in the dump"""
    rows=np.asarray(rows).astype(int);qrows=np.nonzero(qb)[0];pos=np.searchsorted(rows,qrows);assert (pos<len(rows)).all() and (rows[pos]==qrows).all(),'dump rows do not contain every GAINQ row';return pos
def get(Z,k,nm,band):
    p=f'c{k}/{nm}_b{band}_';return Z[p+'idx'].astype(np.int64),Z[p+'score'].astype(np.float32),Z[p+'far'].astype(np.float32)
def selftest():
    rows=np.array([1,4,9,20,33]);qb=np.zeros(CH,bool);qb[[4,20,33]]=True;assert list(subset(rows,qb))==[1,3,4]
    try:subset(rows,np.eye(1,CH,2,dtype=bool)[0]);assert False
    except AssertionError as e:assert 'dump rows' in str(e)
    KM.selftest();print('e57_keyq_vmasks selftest ok')
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--bundle');ap.add_argument('--dump');ap.add_argument('--gainq-masks');ap.add_argument('--pod-masks');ap.add_argument('--band',type=int);ap.add_argument('--out');ap.add_argument('--report');ap.add_argument('--recall',type=float,default=.95);ap.add_argument('--tokenizer');ap.add_argument('--selftest',action='store_true');ap.add_argument('--arms',default='');ap.add_argument('--test',action='store_true')
    a=ap.parse_args()
    if a.selftest:selftest();return
    import gen_world_ctl as G,e55_uv as UVm,e57_reads as RD
    ROOT=HERE.parents[1];tfile=a.tokenizer or str(ROOT/'data/tokenizer/llama3/tokenizer.json');enc=G.load_enc(tfile)
    items=RD.load_items(pathlib.Path(a.bundle)/'e57_bank'/'bank_v1.json');ids,iv,hits=RD.article_intervals(a.bundle,enc,items);ids=np.asarray(ids);ends=UVm.sentence_ends(UVm.tokenizer(tfile),ids)
    band=a.band;gq=np.load(a.gainq_masks,allow_pickle=True);pod=np.load(a.pod_masks,allow_pickle=True);name=f'GAINQ{band}'
    cache={}
    def Zof(k):
        s=k//8
        if s not in cache:cache.clear();cache[s]=np.load(shard_file(a.dump,k))
        return cache[s]
    cand=[(nm,g,p) for nm in NAMES for g,ps in GR.items() for p in ps];acc={c:dict(n=0,h=0,j=0,sel=0.0,rows=0,sat=0) for c in cand}
    chunks=[k for k in range(K0,K1) if (k%2==1)==a.test]
    for k in chunks:
        Z=Zof(k);rows=Z[f'rows_{k}'].astype(int);qb=CM.up(gq[f'qb_{name}_{k}'],CH);pos=subset(rows,qb);lim=KM.lim_rows(band)[rows[pos]]
        wsid,wst,wlen=UVm.window_sents(ends,k,CH,8);hw=np.asarray(pod[f'h5_{k}'],bool);ans=np.nonzero(hw)[0].astype(np.int64);st=wst;en=wst+wlen;wsid_a=np.where(wsid>=0,wsid,0);nfar=KM.far_count(band)[rows[pos]].astype(np.float64)
        for nm in NAMES:
            idx,sc,far=get(Z,k,nm,band);idx=idx[pos];sc=sc[pos];far=far[pos];idx=np.where(idx<0,0,idx)
            for g,ps in GR.items():
                for p in ps:
                    sel,ii=KM.select(g,p,idx,sc,far,None,lim);na,ha,jd=KM.sent_score(sel,ii,lim,wsid_a,ans,st,en);c=acc[(nm,g,p)]
                    c['n']+=int(na.sum());c['h']+=int(ha.sum());c['j']+=int(jd.sum());c['sel']+=float((sel.sum(1)/np.maximum(nfar,1)).sum());c['rows']+=len(pos);c['sat']+=int((sel.sum(1)>=KM.TOPM-1).sum())
        if k%32 in (0,1):print('  chunk',k,flush=True)
    rep={}
    for (nm,g,p),c in acc.items():
        rec=c['h']/c['n'] if c['n'] else float('nan');prec=c['h']/c['j'] if c['j'] else float('nan');f1=2*rec*prec/(rec+prec) if rec==rec and prec==prec and rec+prec>0 else float('nan')
        rep[f'{nm}|{g}|{p}']=dict(recall=rec,precision=prec,f1=f1,mask_amount=c['sel']/max(c['rows'],1),saturated_row_share=c['sat']/max(c['rows'],1),answer_sentences=c['n'],rows=c['rows'])
    chosen={}
    for nm in NAMES:
        rows_=[(g,p,rep[f'{nm}|{g}|{p}']) for g,ps in GR.items() for p in ps];ok=[x for x in rows_ if x[2]['recall']>=a.recall]
        if ok:
            g,p,r=min(ok,key=lambda x:x[2]['mask_amount']);chosen[f'{nm}|{g}|min']=dict(par=p,fallback=False,**r)
        else:
            g,p,r=max(rows_,key=lambda x:x[2]['recall'] if x[2]['recall']==x[2]['recall'] else -1);chosen[f'{nm}|{g}|min']=dict(par=p,fallback=True,**r)
            tgt=2*r['mask_amount'];c2=[x for x in rows_ if x[2]['recall']>=.9*r['recall'] and x[2]['mask_amount']>1.2*r['mask_amount']]
            if c2:
                g2,p2,r2=min(c2,key=lambda x:abs(x[2]['mask_amount']-tgt));chosen[f'{nm}|{g2}|x2']=dict(par=p2,fallback=True,**r2)
    json.dump(dict(band=band,set='test (odd chunks)' if a.test else 'selection (even chunks)',recall_min=a.recall,candidates=rep,chosen=chosen),open(a.report,'w'),indent=1,default=float);print('wrote',a.report,'chosen',list(chosen),flush=True)
    wanted=[x for x in a.arms.split(',') if x]
    if not wanted:return
    out={};names=[];rng=np.random.default_rng(291);limr=KM.lim_rows(band)
    for key in wanted:
        nm,g,ps=key.split('|');par=chosen[key]['par'];tag=f'{nm}_{g}_{ps}';arms={'KQ':f'KQ{band}V_{tag}','KR':f'KR{band}V_{tag}','KN':f'KN{band}V_{tag}'}
        for an in arms.values():
            names.append(an);out[f'band_{an}']=np.asarray(band);out[f'all_{an}']=np.asarray(0);out[f'key_{an}']=np.asarray(1);out[f'amt_{an}']=np.zeros(K1)
        for k in range(K0,K1):
            Z=Zof(k);rows=Z[f'rows_{k}'].astype(int);qb=CM.up(gq[f'qb_{name}_{k}'],CH);pos=subset(rows,qb);lim=limr[rows[pos]]
            idx,sc,far=get(Z,k,nm,band);idx=np.where(idx[pos]<0,0,idx[pos]);sel,ii=KM.select(g,par,idx,sc[pos],far[pos],None,lim);rk,nk=KM.matched_controls(sel,ii,lim,rng)
            kq=np.full((len(pos),TOPM),-1,np.int16)
            for r in range(len(pos)):
                v=ii[r][sel[r]][:TOPM];kq[r,:len(v)]=v
            for an,arr in ((arms['KQ'],kq),(arms['KR'],rk.astype(np.int16)),(arms['KN'],nk.astype(np.int16))):
                out[f'qb_{an}_{k}']=CM.pk(qb);out[f'kq_{an}_{k}']=arr;out[f'amt_{an}'][k]=float(((arr>=0).sum(1)/np.maximum(KM.far_count(band)[rows[pos]],1)).mean())
        print('built',tag,flush=True)
    out['names']=np.asarray(names);out['p']=gq['p'];out['chunks']=np.asarray([K0,K1]);np.savez_compressed(a.out,**out)
    d=hashlib.sha256()
    for kk in sorted(out):
        v=np.ascontiguousarray(out[kk]);d.update(f'{kk}|{v.dtype}|{v.shape}|'.encode()+v.tobytes())
    print('mask content sha256',d.hexdigest(),flush=True)
if __name__=='__main__':main()
