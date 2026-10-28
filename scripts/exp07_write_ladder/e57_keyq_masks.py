"""E57 key-level query masks (CPU only; post-hoc, exploratory). Builds the mask file of the arms KQ* (hide only the SELECTED far window keys of the SELECTED query rows, e57_keyq_run.py) from the W0 key-mass dump of job 295 and the query rows of a GAINQ mask file.
Far keys of a row i (band k): window key c with 4 <= c <= lim_i = min(8191, i + 8192 - k) (distance i + 8192 - c >= k). For k = 256 the lists of the band-128 dump are cut at the distance (c <= i + 8192 - 256; fewer than 128 keys remain for some rows); for k = 512 the band-512 lists are used as stored. Every list is the TOP-128 far keys of the head set (SH = sharp heads, ALL = all heads), sorted by mass.
Candidate rules per head set (the stored top-128 is the ceiling: a rule that would need more keys is cut at 128 and counted as saturated):
  topN   N in 1 2 4 8 16 32 64                       the N keys of largest mass
  tau    tau in .005 .01 .02 .05 .1 .2                keys whose share of the row's far mass is >= tau
  cumq   q in .1 .25 .5 .75 .9                        the smallest prefix whose share of the far mass is >= q
  vote   v in 1 2 4 8 16 32                           keys that >= v heads rank in their own top-16 far keys (cnt column of cutoff 16)
  topNV  N in 4 16 64                                 the N keys of largest value-corrected mass (SHV / ALLV lists of the dump, raw V; Vn is not in the dump of job 295)
Sentence-level score (Hans/label-free-mask thread, defined for news by the ORCA window sentences of e55_uv.window_sents): the answer sentences of a row are the ORCA-hidden window sentences with at least one far key; a sentence counts as hidden when >= 2 selected keys lie inside it. recall = hidden answer sentences / answer sentences, precision = hidden answer sentences / all sentences judged hidden, F1; micro sums over the rows; reported for the GAINQ rows and for all eligible rows of the selection chunks (even chunks). Masking amount = selected keys / far keys of the row (mean over the GAINQ rows).
Selection per (head set, group): among the candidates with recall >= --recall on the selection chunks, the one with the least masking ('min') and the one whose masking is closest to 2 x the min ('x2'); if no candidate passes the highest-recall candidate is used and recorded ('fallback'). Test chunks = odd chunks (the report gives the same numbers there).
Output mask file (the keys of e57_qband_masks.npz plus):  names, band_<arm>, all_<arm>=0, key_<arm>=1, qb_<arm>_<k> (packbits rows = the GAINQ rows of chunk k), kq_<arm>_<k> int16 [nsel, 128] (-1 padded: the window keys hidden in that row, rows in ascending order), amt_<arm> [280] mean hidden/far of the selected rows. Controls of every arm: KR... = the same number of RANDOM far keys per row (seed 291), KN... = the same number of NEAREST far keys (the keys just beyond the band).
  .venv/bin/python scripts/exp07_write_ladder/e57_keyq_masks.py --bundle B --dump D --gainq-masks M.npz --pod-masks P.npz --sharp S.npz --band 512 --out O.npz --report R.json  |  --selftest"""
import sys,json,argparse,pathlib,hashlib
import numpy as np
HERE=pathlib.Path(__file__).resolve().parent
for p in (HERE,HERE/'e57',HERE.parents[1]/'scripts/exp04_what_ttt_retains',HERE.parents[1]/'scripts/exp05_write_dynamics'):sys.path.insert(0,str(p))
import e57_closure_masks as CM
CH=1024;WIN=8192;K0=8;K1=280;TOPM=128
GROUPS={'topN':[1,2,4,8,16,32,64],'tau':[.005,.01,.02,.05,.1,.2],'cumq':[.1,.25,.5,.75,.9],'vote':[1,2,4,8,16,32],'topNV':[4,16,64]}
def lim_rows(band,n=CH):return np.minimum(WIN-1,np.arange(n)+WIN-band)
def select(group,par,idx,mass,far,cnt_col,lim,idxv=None):
    """-> sel bool [R, L] over the stored list (idx [R,128] sorted by mass, mass fp32 [R,128], far [R], cnt_col [R,128] votes at the top-16 cutoff, lim [R]); keys beyond lim or < 4 are never selected"""
    R,L=idx.shape;valid=(idx>=4)&(idx<=lim[:,None])&(mass>0)
    if group=='topNV':
        R,L=idxv.shape;ok=(idxv>=4)&(idxv<=lim[:,None]);sel=np.zeros((R,L),bool);rank=np.cumsum(ok,axis=1);sel=ok&(rank<=par);return sel,idxv
    if group=='topN':rank=np.cumsum(valid,axis=1);return valid&(rank<=par),idx
    if group=='tau':return valid&(mass/np.maximum(far[:,None],1e-30)>=par),idx
    if group=='cumq':
        m=np.where(valid,mass,0.0);cs=np.cumsum(m,axis=1)/np.maximum(far[:,None],1e-30);before=cs-m/np.maximum(far[:,None],1e-30);return valid&(before<par),idx      # keys up to and including the one that reaches q
    if group=='vote':return valid&(cnt_col>=par),idx
    raise ValueError(group)
def sent_score(sel,idx,lim,wsid,ans,st,en):
    """-> (answer sentences, hidden answer sentences, sentences judged hidden) as int arrays [R]; ans = indices of the answer sentences of the window, st/en = start / end token of every window sentence"""
    R=idx.shape[0];sid=np.where(sel,wsid[np.clip(idx,0,WIN-1)],-1)
    if len(ans):
        far_ov=(st[ans][None,:]<=lim[:,None])&(en[ans][None,:]>4);n_ans=far_ov.sum(1)
        cnt=((sid[:,:,None]==ans[None,None,:])&sel[:,:,None]).sum(1);hid_ans=((cnt>=2)&far_ov).sum(1)
    else:n_ans=np.zeros(R,int);hid_ans=np.zeros(R,int)
    s=np.sort(np.where(sid>=0,sid,-1),axis=1);same_next=(s[:,:-1]==s[:,1:])&(s[:,:-1]>=0);first=np.ones_like(same_next);first[:,1:]=s[:,:-2]!=s[:,1:-1]
    judged=(same_next&first).sum(1);return n_ans,hid_ans,judged
def brute_sent_score(sel_row,idx_row,lim,wsid,ans,st,en):
    cnt={};
    for j in range(len(idx_row)):
        if sel_row[j]:cnt[int(wsid[idx_row[j]])]=cnt.get(int(wsid[idx_row[j]]),0)+1
    n=h=0
    for a in ans:
        if st[a]<=lim and en[a]>4:
            n+=1;h+=int(cnt.get(int(a),0)>=2)
    return n,h,sum(1 for s_,c in cnt.items() if c>=2 and s_>=0)
def cut_distance(idx,mass,lim):
    ok=(idx>=4)&(idx<=lim[:,None]);return np.where(ok,idx,-1),np.where(ok,mass,0.0)
def load_dump_chunk(Z,k,band_src,names):
    pre=f'W0/c{k}/';return {n:Z[pre+f'{n}'] for n in names}
def chunk_inputs(Z,k,band_file,hs,band):
    """-> dict hs -> (idx int64 [CH,128], mass fp32, far fp32, votes int [CH,128], idxv int64 [CH,64], nheads)"""
    out={}
    for nm in ('SH','ALL'):
        idx=Z[f'W0/c{k}/{nm}{band_file}_idx'].astype(np.int64);mass=Z[f'W0/c{k}/{nm}{band_file}_mass'].astype(np.float32);far=Z[f'W0/c{k}/{nm}{band_file}_far'].astype(np.float32)
        cnt=Z[f'W0/c{k}/{nm}{band_file}_cnt'];idxv=Z[f'W0/c{k}/{nm}V{band_file}_idx'].astype(np.int64)
        lim=lim_rows(band);votes=cnt[:,:,2].astype(np.int32)       # rank cutoff 16 = column 2 (SH: 1/4/16/64, ALL: 1/4/16)
        if band!=band_file:      # cut a band-128 list at the distance of the band; the far mass is the stored far mass minus the mass of the cut keys that were stored (the unstored remainder cannot be removed: far is reported as stored)
            ok=(idx>=4)&(idx<=lim[:,None]);cutm=np.where(ok,0.0,mass).sum(1);far=far-cutm
            idx=np.where(ok,idx,-1);mass=np.where(ok,mass,0.0)
            okv=(idxv>=4)&(idxv<=lim[:,None]);idxv=np.where(okv,idxv,-1)
        out[nm]=(idx,mass,far,votes,idxv)
    return out
def far_count(band):return np.maximum(0,np.minimum(WIN-1,np.arange(CH)+WIN-band)-4+1)
def matched_controls(sel,idx,lim,rng):
    """-> (rand_keys [R,128] int64 -1 padded, near_keys): the same number of selected keys per row, random far keys / the nearest far keys (largest window position <= lim)"""
    R=sel.shape[0];n=sel.sum(1);rk=np.full((R,TOPM),-1,np.int64);nk=np.full((R,TOPM),-1,np.int64)
    for r in range(R):
        m=int(min(n[r],TOPM));hi=int(lim[r])
        if m==0 or hi<4:continue
        pool=hi-4+1;m=min(m,pool);rk[r,:m]=rng.choice(np.arange(4,hi+1),size=m,replace=False);nk[r,:m]=np.arange(hi-m+1,hi+1)[::-1]
    return rk,nk
def selftest():
    rng=np.random.default_rng(0);R=40;L=128;band=512;lim=lim_rows(band,R)
    idx=np.stack([rng.permutation(np.arange(4,WIN-1))[:L] for _ in range(R)]);mass=np.sort(rng.random((R,L)).astype(np.float32)**3,axis=1)[:,::-1].copy();far=mass.sum(1)*1.7
    votes=rng.integers(0,40,(R,L));idx=np.minimum(idx,lim[:,None]+rng.integers(0,200,(R,1)))      # some keys beyond lim
    nw=64;st=np.sort(rng.choice(np.arange(WIN-100),nw,replace=False)).astype(np.int64);en=np.append(st[1:],WIN);wsid=np.zeros(WIN,np.int64)
    for s,(a,b) in enumerate(zip(st,en)):wsid[a:b]=s
    ans=np.array([3,10,20,30,40,50])
    for g,pars in GROUPS.items():
        for p in pars:
            idxv=idx[:,:64]
            sel,ii=select(g,p,idx,mass,far,votes,lim,idxv=idxv)
            assert not (sel&((ii<4)|(ii>lim[:,None]))).any()
            if g=='topN':assert (sel.sum(1)<=p).all()
            na,ha,jd=sent_score(sel,ii,lim,wsid,ans,st,en)
            for r in range(R):
                b=brute_sent_score(sel[r],ii[r],lim[r],wsid,ans,st,en);assert (na[r],ha[r],jd[r])==b,(g,p,r,(na[r],ha[r],jd[r]),b)
    sel,ii=select('cumq',.5,idx,mass,far,votes,lim);v=(idx>=4)&(idx<=lim[:,None])
    for r in range(R):          # brute force: smallest valid prefix whose cumulative share >= q (or all)
        m=np.where(v[r],mass[r],0);c=np.cumsum(m)/far[r];k=np.argmax(c>=.5) if (c>=.5).any() else len(c)-1
        exp=np.zeros(L,bool);exp[:k+1]=v[r][:k+1];assert np.array_equal(sel[r],exp) or not (c>=.5).any(),r
    rk,nk=matched_controls(select('topN',8,idx,mass,far,votes,lim)[0],idx,lim,np.random.default_rng(1));n=select('topN',8,idx,mass,far,votes,lim)[0].sum(1)
    assert ((rk>=0).sum(1)==np.minimum(n,lim-4+1)).all() and ((nk>=0).sum(1)==(rk>=0).sum(1)).all() and ((rk<=lim[:,None]).all()) and (nk[nk>=0]<=np.repeat(lim,TOPM).reshape(R,TOPM)[nk>=0]).all()
    print('e57_keyq_masks selftest ok')
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--bundle');ap.add_argument('--dump');ap.add_argument('--gainq-masks');ap.add_argument('--pod-masks');ap.add_argument('--sharp');ap.add_argument('--band',type=int);ap.add_argument('--out');ap.add_argument('--report');ap.add_argument('--recall',type=float,default=.95);ap.add_argument('--tokenizer');ap.add_argument('--selftest',action='store_true');ap.add_argument('--arms',default='');ap.add_argument('--chunks',default='')
    a=ap.parse_args()
    if a.selftest:selftest();return
    import gen_world_ctl as G,e55_uv as UVm,e57_reads as RD
    ROOT=HERE.parents[1];tfile=a.tokenizer or str(ROOT/'data/tokenizer/llama3/tokenizer.json');enc=G.load_enc(tfile)
    items=RD.load_items(pathlib.Path(a.bundle)/'e57_bank'/'bank_v1.json');ids,iv,hits=RD.article_intervals(a.bundle,enc,items);ids=np.asarray(ids);ends=UVm.sentence_ends(UVm.tokenizer(tfile),ids)
    band=a.band;band_file=128 if band==256 else band;assert band in (256,512)
    gq=np.load(a.gain_q_masks if hasattr(a,'gain_q_masks') else a.gainq_masks,allow_pickle=True);pod=np.load(a.pod_masks,allow_pickle=True)
    import e57_attdump3_run as R3
    shards={}
    def Zof(k):
        s=k//R3.SHARD
        if s not in shards:shards.clear();shards[s]=np.load(R3.shard_file(a.dump,s))
        return shards[s]
    cand=[(nm,g,p) for nm in ('SH','ALL') for g,ps in GROUPS.items() for p in ps]
    chunks=list(range(K0,K1)) if not a.chunks else [int(x) for x in a.chunks.split(',')]
    acc={c:dict(n=0,h=0,j=0,sel=0.0,far=0.0,rows=0,n_all=0,h_all=0,j_all=0,sat=0) for c in cand}
    lim=lim_rows(band);nfar=far_count(band).astype(np.float64);name=f'GAINQ{band}';t0=0
    per_chunk_cache={}
    for k in chunks:
        Z=Zof(k);inp=chunk_inputs(Z,k,band_file,None,band)
        q=UVm.chunk_sents(ends,k,CH);sid_c=np.full(CH,-1)
        for i,(s_,e_) in enumerate(q):sid_c[s_:e_]=i
        wsid,wst,wlen=UVm.window_sents(ends,k,CH,8);hw=np.asarray(pod[f'h5_{k}'],bool);ans=np.nonzero(hw)[0].astype(np.int64);st=wst;en=wst+wlen
        wsid_a=np.where(wsid>=0,wsid,0)           # keys outside any sentence (chunk before the stream start) never reach a counted sentence: ans excludes them
        qb=CM.up(gq[f'qb_{name}_{k}'],CH);selchunk=(k%2==0)
        elig=np.ones(CH,bool);elig[CH-1]=False
        if not selchunk:continue
        for nm in ('SH','ALL'):
            idx,mass,far,votes,idxv=inp[nm]
            for g,ps in GROUPS.items():
                for p in ps:
                    sel,ii=select(g,p,idx,mass,far,votes,lim,idxv=idxv);na,ha,jd=sent_score(sel,ii,lim,wsid_a,ans,st,en);c=acc[(nm,g,p)]
                    c['n']+=int(na[qb].sum());c['h']+=int(ha[qb].sum());c['j']+=int(jd[qb].sum());c['sel']+=float((sel.sum(1)[qb]/np.maximum(nfar[qb],1)).sum());c['rows']+=int(qb.sum());c['sat']+=int(((sel.sum(1)>=TOPM-1)[qb]).sum())
                    c['n_all']+=int(na[elig].sum());c['h_all']+=int(ha[elig].sum());c['j_all']+=int(jd[elig].sum())
        if k%32==0:print(f'  selection chunk {k} done',flush=True)
    rep={}
    for (nm,g,p),c in acc.items():
        rec=c['h']/c['n'] if c['n'] else float('nan');prec=c['h']/c['j'] if c['j'] else float('nan');f1=2*rec*prec/(rec+prec) if rec==rec and prec==prec and rec+prec>0 else float('nan')
        rep[f'{nm}|{g}|{p}']=dict(recall=rec,precision=prec,f1=f1,mask_amount=c['sel']/max(c['rows'],1),saturated_row_share=c['sat']/max(c['rows'],1),answer_sentences=c['n'],recall_all_eligible=c['h_all']/c['n_all'] if c['n_all'] else float('nan'),precision_all_eligible=c['h_all']/c['j_all'] if c['j_all'] else float('nan'))
    chosen={}
    for nm in ('SH','ALL'):
        for g,ps in GROUPS.items():
            rows=[(p,rep[f'{nm}|{g}|{p}']) for p in ps];ok=[(p,r) for p,r in rows if r['recall']>=a.recall]
            if ok:
                pmin,rmin=min(ok,key=lambda x:x[1]['mask_amount']);chosen[f'{nm}|{g}|min']=dict(par=pmin,fallback=False,**rmin)
                tgt=2*rmin['mask_amount'];px2,rx2=min([(p,r) for p,r in rows if r['recall']>=a.recall],key=lambda x:abs(x[1]['mask_amount']-tgt));chosen[f'{nm}|{g}|x2']=dict(par=px2,fallback=False,**rx2)
            else:
                pb,rb=max(rows,key=lambda x:(x[1]['recall'] if x[1]['recall']==x[1]['recall'] else -1));chosen[f'{nm}|{g}|min']=dict(par=pb,fallback=True,**rb)
    json.dump(dict(band=band,recall_min=a.recall,selection='even chunks '+f'{chunks[0]}..{chunks[-1]}',candidates=rep,chosen=chosen),open(a.report,'w'),indent=1,default=float)
    print('wrote',a.report,'candidates',len(rep),'chosen',len(chosen),flush=True)
    wanted=[x for x in a.arms.split(',') if x] if a.arms else None
    if wanted is None:return
    build_masks(a,band,band_file,chosen,wanted,Zof,gq,pod,ends,UVm,lim)
def build_masks(a,band,band_file,chosen,wanted,Zof,gq,pod,ends,UVm,lim):
    """wanted: keys of 'chosen' (e.g. ALL|tau|min); writes arms KQ<band>_<set>_<group>_<pass> and the controls KR... / KN... (same number of keys per row)"""
    import e57_attdump3_run as R3
    out={};names=[];name=f'GAINQ{band}';rng=np.random.default_rng(291)
    for key in wanted:
        nm,g,ps=key.split('|');c=chosen[key];par=c['par'];tag=f'{nm}_{g}_{ps}';arms={'KQ':f'KQ{band}_{tag}','KR':f'KR{band}_{tag}','KN':f'KN{band}_{tag}'}
        for an in arms.values():
            names.append(an);out[f'band_{an}']=np.asarray(band);out[f'all_{an}']=np.asarray(0);out[f'key_{an}']=np.asarray(1);out[f'amt_{an}']=np.zeros(K1)
        for k in range(K0,K1):
            Z=Zof(k);inp=chunk_inputs(Z,k,band_file,None,band);idx,mass,far,votes,idxv=inp[nm];qb=CM.up(gq[f'qb_{name}_{k}'],CH)
            sel,ii=select(g,par,idx,mass,far,votes,lim,idxv=idxv);rk,nk=matched_controls(sel,ii,lim,rng)
            rows=np.nonzero(qb)[0];kq=np.full((len(rows),TOPM),-1,np.int16)
            for rr,row in enumerate(rows):
                v=ii[row][sel[row]][:TOPM];kq[rr,:len(v)]=v
            for an,arr in ((arms['KQ'],kq),(arms['KR'],rk[rows].astype(np.int16)),(arms['KN'],nk[rows].astype(np.int16))):
                out[f'qb_{an}_{k}']=CM.pk(qb);out[f'kq_{an}_{k}']=arr;out[f'amt_{an}'][k]=float(((arr>=0).sum(1)/np.maximum(far_count(band)[rows],1)).mean()) if len(rows) else 0.0
        print('built',tag,flush=True)
    out['names']=np.asarray(names);out['p']=gq['p'];out['chunks']=np.asarray([K0,K1]);np.savez_compressed(a.out,**out)
    d=hashlib.sha256()
    for k in sorted(out):
        v=np.ascontiguousarray(out[k]);d.update(f'{k}|{v.dtype}|{v.shape}|'.encode()+v.tobytes())
    print('mask content sha256',d.hexdigest(),flush=True)
if __name__=='__main__':main()
