"""E57 synthetic-world key-rule candidates, per seed (CPU only; new file). For ONE seed (cell v3_16_1, 128 chunks) and its pass files of e57_syn_pass.py (syn_s<seed>_nll.npz, syn_s<seed>_dump_j*.npz) it evaluates every candidate rule on the stored top-128 far keys of the band (256 or 512):
  sets  ALL (plain mass), ALLV (value-corrected), ALLVn (per-head value-normalised)     rules  topN 1..64, tau .005..2 (share of the row's far score), cumq .1..9
Query rows = the GAINQ rows of the chunk (top round(p x eligible) rows by nll_band_b - nll_win, p = .26, eligible = loss rows whose next token lies in a sentence); counts are also given for all eligible rows. ANSWER SENTENCES of a row (sentence-level spec A for the synthetic world): the window sentences that carry the SAME FACT as the query sentence the row belongs to (e55_sentdump_labels: fact id of a sentence = majority fact of its tokens, share >= .5; rows of background sentences have no answer sentences), restricted to those with a far key of the row. A sentence is hidden when >= 2 selected keys lie inside it. Output: per candidate the SUMS over the chunks 8..127 (answer sentences, hidden answer sentences, sentences judged hidden, selected keys, far keys, rows, saturated rows), so that several seeds are pooled by adding the counts (e57_syn_select.py): <out>/syn_s<seed>_counts_b<band>.json.
  .venv/bin/python scripts/exp07_write_ladder/e57_syn_eval.py --seed 62 --dir D --out O --band 512 [--parity even|odd|all] | --selftest"""
import sys,os,json,argparse,pathlib,types
import numpy as np
HERE=pathlib.Path(__file__).resolve().parent;ROOT=HERE.parents[1]
for p in (HERE,HERE/'e57',ROOT/'scripts/exp04_what_ttt_retains',ROOT/'scripts/exp05_write_dynamics'):sys.path.insert(0,str(p))
import e57_closure_masks as CM
import e57_keyq_masks as KM
CH=1024;WIN=8192;TOPM=128;SHARD=16
NAMES=('ALL','ALLV','ALLVn')
GR={'topN':[1,2,4,8,16,32,64],'tau':[.005,.01,.02,.05,.1,.2],'cumq':[.1,.25,.5,.75,.9]}
def fact_score(sel,idx,lim,wsid,wfact,st,en,rfact):
    """-> (answer sentences n_ans [R], hidden answer sentences h [R], sentences judged hidden j [R]); wfact [nW] fact id of the window sentences (-1 none), rfact [R] fact id of the query sentence of the row (-1 none)"""
    R=idx.shape[0];nW=len(wfact);has=rfact>=0
    A=(wfact[None,:]==rfact[:,None])&has[:,None]&(st[None,:]<=lim[:,None])&(en[None,:]>4);n_ans=A.sum(1)
    sid=np.where(sel,wsid[np.clip(idx,0,WIN-1)],-1);sid=np.where(sid>=0,sid,-1);s=np.sort(sid,axis=1)
    same_next=(s[:,:-1]==s[:,1:])&(s[:,:-1]>=0);first=np.ones_like(same_next);first[:,1:]=s[:,:-2]!=s[:,1:-1];run=same_next&first
    judged=run.sum(1);ss=s[:,:-1];hid=np.zeros(R,int)
    rr,cc=np.nonzero(run);sent=ss[rr,cc];ok=(wfact[sent]==rfact[rr])&has[rr]&A[rr,sent];np.add.at(hid,rr[ok],1)
    return n_ans,hid,judged
def brute(sel_row,idx_row,lim,wsid,wfact,st,en,rf):
    cnt={}
    for j in range(len(idx_row)):
        if sel_row[j] and wsid[idx_row[j]]>=0:cnt[int(wsid[idx_row[j]])]=cnt.get(int(wsid[idx_row[j]]),0)+1
    n=h=0
    for s_,f in enumerate(wfact):
        if rf>=0 and f==rf and st[s_]<=lim and en[s_]>4:
            n+=1;h+=int(cnt.get(s_,0)>=2)
    return n,h,sum(1 for c in cnt.values() if c>=2)
def selftest():
    rng=np.random.default_rng(0);R=30;L=128;nW=40;lim=KM.lim_rows(512,R)+0
    idx=rng.integers(4,WIN-1,(R,L));sel=rng.random((R,L))<.4;st=np.sort(rng.choice(np.arange(WIN-50),nW,replace=False)).astype(np.int64);en=np.append(st[1:],WIN)
    wsid=np.zeros(WIN,np.int64)
    for s,(a,b) in enumerate(zip(st,en)):wsid[a:b]=s
    wfact=rng.integers(-1,4,nW).astype(np.int16);rfact=rng.integers(-1,4,R).astype(np.int16);lim=np.minimum(WIN-1,np.arange(R)+3000)
    n,h,j=fact_score(sel,idx,lim,wsid,wfact,st,en,rfact)
    for r in range(R):
        b=brute(sel[r],idx[r],lim[r],wsid,wfact,st,en,int(rfact[r]));assert (n[r],h[r],j[r])==b,(r,(n[r],h[r],j[r]),b)
    print('e57_syn_eval selftest ok')
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--seed',type=int);ap.add_argument('--dir');ap.add_argument('--out');ap.add_argument('--band',type=int,default=512);ap.add_argument('--p',type=float,default=.26);ap.add_argument('--parity',default='all');ap.add_argument('--selftest',action='store_true');a=ap.parse_args()
    if a.selftest:selftest();return
    import e55_run as R55,e55_uv as UVm,e55_sentdump as SD,e55_sentdump_labels as LB
    import e57_syn_pass as SP
    enc,w,tf=R55.build(SP.synth_args(a.seed),a.seed,arms=['NAT']);S=R55.streams(w,enc);ids,sha=S['A'];ends=UVm.sentence_ends(UVm.tokenizer(tf),ids)
    d=pathlib.Path(a.dir);Z=np.load(d/f'syn_s{a.seed}_nll.npz');bi=list(Z['bands']).index(a.band);nll_w=Z['nll_win'];nll_b=Z['nll_band'][bi];lm=Z['lm']
    cand=[(nm,g,p) for nm in NAMES for g,ps in GR.items() for p in ps];acc={c:dict(n=0,h=0,j=0,sel=0.0,rows=0,sat=0,n_all=0,h_all=0,j_all=0) for c in cand};cache={}
    def shard(k):
        j=k//SHARD
        if j not in cache:cache.clear();cache[j]=np.load(d/f'syn_s{a.seed}_dump_j{j:02d}.npz')
        return cache[j]
    lim=KM.lim_rows(a.band);nfar=KM.far_count(a.band).astype(np.float64)
    for k in range(8,128):
        if a.parity=='even' and k%2:continue
        if a.parity=='odd' and k%2==0:continue
        q,wsid,wst,wlen=SD.layout(ends,k);lb=LB.labels(w['chunks'],ends,k);sid=np.full(CH,-1)
        for i,(s_,e_) in enumerate(q):sid[s_:e_]=i
        qfact=np.asarray(lb['q_fact']);wfact=np.asarray(lb['w_fact']);rfact=np.where(sid>=0,qfact[np.clip(sid,0,len(q)-1)],-1).astype(np.int16)
        el=CM.rows_eligible(lm[k],sid)&lm[k];g=(nll_b[k]-nll_w[k]).astype(np.float64);m=CM.top_rows(g,el,int(round(a.p*el.sum())))
        st=wst;en=wst+wlen;wsid_a=np.where(wsid>=0,wsid,0);Zs=shard(k)
        for nm in NAMES:
            P=f'c{k}/{nm}_b{a.band}_';idx=Zs[P+'idx'].astype(np.int64);sc=Zs[P+'score'].astype(np.float32);far=Zs[P+'far'].astype(np.float32);idx=np.where(idx<0,0,idx)
            for gname,ps in GR.items():
                for p in ps:
                    sel,ii=KM.select(gname,p,idx,sc,far,None,lim);n_ans,h,j=fact_score(sel,ii,lim,wsid_a,wfact,st,en,rfact);c=acc[(nm,gname,p)]
                    c['n']+=int(n_ans[m].sum());c['h']+=int(h[m].sum());c['j']+=int(j[m].sum());c['sel']+=float((sel.sum(1)[m]/np.maximum(nfar[m],1)).sum());c['rows']+=int(m.sum());c['sat']+=int((sel.sum(1)>=TOPM-1)[m].sum())
                    c['n_all']+=int(n_ans[el].sum());c['h_all']+=int(h[el].sum());c['j_all']+=int(j[el].sum())
        if k%32==0:print('  chunk',k,flush=True)
    rows={f'{nm}|{g}|{p}':v for (nm,g,p),v in acc.items()}
    json.dump(dict(seed=a.seed,band=a.band,p=a.p,parity=a.parity,stream_sha=sha,counts=rows),open(a.out,'w'),indent=1)
    best=sorted(rows.items(),key=lambda x:-(x[1]['h']/max(x[1]['n'],1)))[:5]
    for k_,v in best:print(k_,'recall %.3f precision %.3f mask %.4f'%(v['h']/max(v['n'],1),v['h']/max(v['j'],1),v['sel']/max(v['rows'],1)),'answer sentences',v['n'])
if __name__=='__main__':main()
