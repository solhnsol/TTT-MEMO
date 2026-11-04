"""E57 public QA stage 18 (new file; GPU; investigation thread 'binding learned fast at few exposures', Hans card 'P3 결정 행 간격 복습, QA에서 두 팔 실행' 11/04 20:41 KST; design ref/few-exposure-binding-methods-2026-11-04.md section 7): spaced REHEARSAL of the deciding rows on the QA bundle episodes of jobs 790 / 870 (5 bundles, 73 items, 17 versions, rewrites_16_subset.json). Base = ALLQ128 on every query row from chunk 0, update norm 0.5 x the per-chunk NAT norm of job 790 (engine SELC1), one update per 1024-token chunk ('section' = the update chunk). Rehearsal fragment = the 63 tokens before a row plus its next token (64 tokens; the loss is the next token only) of the DECIDING rows of an earlier chunk j (top 30% of the eligible rows by the window gain g = nll_band128 - nll_win of the W0 pass, rule of job 870; rows whose next token is a whitespace-only token are skipped). The fragments of chunk j are rewritten at the steps j+1, j+2, j+4, j+8. At step k the candidates are the fragments of the chunks k-1, k-2, k-4, k-8; at most 16 (1024 / 64) are used, cut by window gain; the 16 fragments form ONE 1024-token sequence with a block-diagonal causal mask (fragments do not see each other, no window keys) in a SECOND suffix forward (structure of jobs 900 / 910: prefix outputs with the same mask); the loss is the sum of the NLL of the main chunk and of the 16 fragment tokens divided by the total number of loss tokens. Arms: REP (deciding-row fragments) and REPRAND (random eligible rows of the same source chunks, same number per source chunk and same tokens). The numbers of fragments per step are in the writes json (rep_log). Reads / analysis as job 870 (e57_pqa15_run functions), n = 1 / 3 / 5 / 9 / 13 / 17, canary at every read chunk.
  python3 e57_pqa18_run.py --stage plan|write|read|analyze|compare --out O ... | --selftest"""
import sys,json,time,gc,hashlib,pathlib
import numpy as np
HERE=pathlib.Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
import e57_closure_masks as CM
import e57_pqa2_run as P2
import e57_pqa2_stream as S
import e57_pqa6_run as P6
import e57_pqa15_run as P15
import e57_pqa16_run as P16
CH=1024;WIN=8*CH;FRAG=64;NFRAG=CH//FRAG;LAGS=(1,2,4,8);ARMS=['NATQ','ALLQ128','REP','REPRAND'];NEW=['REP','REPRAND']
def candidates(ids,dec_rows,gain,skip_ws,j):
    """fragments of the deciding rows of chunk j: [(p, gain)] with p = global position of the input token of the row (fragment = ids[p-62 .. p+1])"""
    out=[]
    for i in dec_rows:
        p=j*CH+int(i)
        if p<FRAG-2 or p+1>=len(ids):continue
        if skip_ws(int(ids[p+1])):continue
        out.append((p,float(gain[i])))
    return out
def plan_bundle(ids,win,band,lm,skip_ws,seed):
    """-> {'REP': {k: [(p, src_chunk)]}, 'REPRAND': {...}, 'counts': ...}"""
    nch=len(ids)//CH;dm=P16.deciding_masks(win,band,lm);cand={};allrows={}
    for j in range(nch):
        rows=np.where(CM.up(dm[j],CH))[0];g=(band[j]-win[j]).astype(np.float64);cand[j]=candidates(ids,rows,g,skip_ws,j)
        el=np.where(CM.rows_eligible(lm[j],np.zeros(CH,np.int32))&lm[j])[0];allrows[j]=candidates(ids,el,g,skip_ws,j)
    rep={};rnd={};cnt={}
    for k in range(1,nch):
        pool=[(p,g,k-l) for l in LAGS if k-l>=0 for p,g in cand[k-l]];pool.sort(key=lambda x:-x[1]);sel=pool[:NFRAG]
        if not sel:continue
        rep[k]=[(p,j) for p,g,j in sel];per={}
        for p,g,j in sel:per[j]=per.get(j,0)+1
        cnt[k]=per;rr=np.random.default_rng(seed+1000*k);rl=[]
        for j,n in per.items():
            pool_j=allrows[j];idx=rr.choice(len(pool_j),size=min(n,len(pool_j)),replace=False);rl+=[(pool_j[t][0],j) for t in idx]
        rnd[k]=rl
    return rep,rnd,cnt
class UVRep:
    """UV-compatible keep object: band 128 for every row of every chunk; prepares the second forward (fragments) of the step in holder['rep']"""
    def __init__(self,plan,ids,holder,rt):self.plan=plan;self.ids=ids;self.holder=holder;self.rt=rt
    def keep(self,k,m,Kc,Vc,ids_k,keep_all):
        import jax.numpy as jnp
        base=np.asarray(keep_all,bool);kp=np.broadcast_to(base,(CH,WIN+CH)).copy();kp[:]=CM.band_keep(base,128);fr=self.plan.get(k,[]);self.holder['rep']=None
        if fr:
            ids2=np.full(CH,self.rt.eos,np.int64);mk=np.zeros(CH,bool);kp2=np.zeros((CH,WIN+CH),bool)
            for f in range(NFRAG):
                for r in range(FRAG):kp2[f*FRAG+r,WIN+f*FRAG:WIN+f*FRAG+r+1]=True
            for f,(p,j) in enumerate(fr):
                ids2[f*FRAG:(f+1)*FRAG]=self.ids[p-FRAG+2:p+2];mk[f*FRAG+FRAG-2]=True        # 63 context tokens (positions 0..62) + the next token (position 63); the loss row is position 62
            L=self.holder['L'];pref2=L.prefix_masked(m,jnp.asarray(ids2,dtype=jnp.int32),Kc,Vc,jnp.int32(k*CH),jnp.asarray(kp2));c2=self.rt.chunk(self.rt.seq(ids2,mask=mk),0);self.holder['rep']=(pref2,jnp.asarray(kp2),c2,len(fr))
        return kp,{'uv_arm':'REP','q_banded':CH,'q_band':128,'n_frag':len(fr),'uv_hidden_tokens':0,'uv_hidden_chunk_tokens':0}
def install_rep_grad(L,s3,holder,rep_w=1.0):
    import jax,jax.numpy as jnp,equinox as eqx
    S3SUF=s3.S3SUF;orig=L.grad_mask
    @eqx.filter_jit
    def grad_r(m,st_suf,c,pref_,kmask,pref2,kmask2,c2):
        def fwd(mm,seq,pf,km):
            S3SUF['kmask']=km
            try:out=mm.language_model.suffix_call(prefix_outputs=pf,state=st_suf,seq=seq)
            finally:S3SUF.clear()
            lp=jax.nn.log_softmax(out.logits.astype(jnp.float32),axis=-1);return -jnp.take_along_axis(lp,seq.target_tokens[:,None],axis=-1)[:,0]
        def total(mm):
            nll=fwd(mm,c,pref_,kmask);nll2=fwd(mm,c2,pref2,kmask2);v1=c.loss_masks.astype(jnp.float32);v2=c2.loss_masks.astype(jnp.float32);n=jnp.maximum(v1.sum()+rep_w*v2.sum(),1e-10)
            return (jnp.sum(v1*nll)+rep_w*jnp.sum(v2*nll2))/n,(nll,nll2)
        (_l,(nll,nll2)),g=eqx.filter_value_and_grad(total,has_aux=True)(m);v1=c.loss_masks.astype(jnp.float32);v2=c2.loss_masks.astype(jnp.float32)
        return g.inner_parameters(),jnp.sum(v1*nll)/jnp.maximum(v1.sum(),1e-10),jnp.sum(v2*nll2)/jnp.maximum(v2.sum(),1e-10),v1.sum(),v2.sum()
    def grad_mask(m,st_suf,c,pref_,kmask):
        rep=holder.get('rep')
        if rep is None:holder['last']=None;return orig(m,st_suf,c,pref_,kmask)
        pref2,kmask2,c2,nf=rep;g,ce,ce2,n1,n2=grad_r(m,st_suf,c,pref_,kmask,pref2,kmask2,c2);holder.setdefault('log',[]).append(dict(n_frag=int(nf),n_loss_main=float(n1),n_loss_rep=float(n2),rep_weight=rep_w,rep_loss_share=float(rep_w*n2/(n1+rep_w*n2)),nll_rep_mean=float(ce2),nll_main_mean=float(ce)));return g,ce
    L.grad_mask=grad_mask
def plan_stage(a):
    out=pathlib.Path(a['out']);out.mkdir(parents=True,exist_ok=True);meta,ids_z=P6.load_groups(out);Z=np.load(a['dwg']);bi=list(Z['bands']).index(P16.BAND);tk=S.tokenizer();cache={}
    def skip_ws(t):
        if t not in cache:cache[t]=tk.decode([t]).strip()==''
        return cache[t]
    plan={};stats={}
    for g in meta['groups']:
        gid=str(g['gid']);ids=np.asarray(ids_z[f'g{gid}']);rep,rnd,cnt=plan_bundle(ids,Z[f'g{gid}_win'],Z[f'g{gid}_band'][bi],Z[f'g{gid}_lm'],skip_ws,seed=20261104+g['gid'])
        plan[gid]=dict(REP={str(k):v for k,v in rep.items()},REPRAND={str(k):v for k,v in rnd.items()});nf=[len(v) for v in rep.values()]
        stats[gid]=dict(n_steps_with_rehearsal=len(rep),fragments_per_step_mean=float(np.mean(nf)),fragments_per_step_min=int(min(nf)),fragments_per_step_max=int(max(nf)),rand_fragments_per_step_mean=float(np.mean([len(v) for v in rnd.values()])),source_chunk_share={str(l):float(np.mean([sum(1 for _,j in v if k-j==l)/max(len(v),1) for k,v in rep.items()])) for l in LAGS})
    (out/'e57_pqa18_plan.json').write_text(json.dumps(plan));(out/'e57_pqa18_plan_stats.json').write_text(json.dumps(stats,indent=1));print(json.dumps(stats)[:1500])
def write_stage(a):
    out=pathlib.Path(a['out']);import e55_run as R,e57_syn_mrun as MR
    LD,s3,C=R.lab();rt=LD.Runtime();can=R.Canary55(LD,rt);lg=lambda *x:print(*x,flush=True);cw0=can(rt.base);lg('canary W0',cw0)
    meta,ids_z=P6.load_groups(out);W640=json.load(open(out/'e57_pqa6_writes.json'));plan=json.load(open(out/'e57_pqa18_plan.json'));fn=out/'e57_pqa18_writes.json';W=json.load(open(fn)) if fn.exists() else {};t00=time.monotonic()
    for g in meta['groups']:
        gid=str(g['gid'])
        if gid in W:continue
        ids=np.asarray(ids_z[f'g{gid}']);nch=len(ids)//CH;assert nch==g['n_chunks'];rd_ch={int(v) for v in g['read_chunks'].values()};ntg=W640[gid]['NAT']['update_norms'];assert len(ntg)==nch;t0=time.monotonic()
        L0=C.Ladder55(rt,save_dir=out/'states'/'rec',canary=None)
        with rt.mesh:Kh0,Vh0,_h=L0.record(ids)
        del _h;rc={(hashlib.sha256(np.asarray(ids,np.int64).tobytes()).hexdigest(),nch):(Kh0,Vh0)};del L0;gc.collect();W[gid]={}
        for arm in NEW:
            pl={int(k):[tuple(x) for x in v] for k,v in plan[gid][arm].items()};holder={};uv=UVRep(pl,ids,holder,rt);L=C.Ladder55(rt,save_dir=out/'states'/f'g{gid}_{arm}',canary=can);L.rec_cache=rc;holder['L']=L;install_rep_grad(L,s3,holder,float(a.get('rep_weight') or 1.0))
            res=L.run_stream('SELC1',1,ids,nch=nch,tag=f'pqa18g{gid}{arm}',log=lambda *x:None,read_chunks=rd_ch,canary_chunks=rd_ch,chunks_meta=[{'sents':[]} for _ in range(nch)],norm_targets=ntg,uv=uv)
            W[gid][arm]=dict(states={str(k):dict(file=v['_file'],hash=v['_hash'],canary=v.get('canary')) for k,v in res['reads'].items()},ce=[e['ce'] for e in res['log']],update_norms=[e['update_norm'] for e in res['log']],rep_log=holder.get('log'),canary_w0=cw0);del L,res;gc.collect()
        del rc,Kh0,Vh0;gc.collect();fn.write_text(json.dumps(W,default=float));lg(f'bundle {gid}: {nch} chunks, 2 arms written in {time.monotonic()-t0:.0f}s (total {time.monotonic()-t00:.0f}s)')
def read_stage(a):
    P15.NREAD=[1,3,5,9,13,17];P15.ARMS=list(NEW)
    out=pathlib.Path(a['out']);import shutil;rows_dir=out/'new_rows';rows_dir.mkdir(exist_ok=True)
    for f in ('e57_pqa6_groups.json','e57_pqa6_group_ids.npz'):
        if not (rows_dir/f).exists():shutil.copy(out/f,rows_dir/f)
    (rows_dir/'e57_pqa6_writes.json').write_text((out/'e57_pqa18_writes.json').read_text());(rows_dir/'e57_pqa7_writes.json').write_text('{}')
    P15.read_stage(dict(a,out=str(rows_dir),skip_repro=True))
def analyze(a):
    P15.NREAD=[1,3,5,9,13,17];P15.ARMS=list(ARMS);P15.analyze(a)
def compare_stage(a):
    """item-level comparison of REP / REPRAND at n = 17 with the ALLQ128 state of job 890 at n = 34 (same total tokens as 2x the stream): first-token dA and d(A-B)"""
    import e57_pqa2_stream as S2
    out=pathlib.Path(a['out']);m=pathlib.Path(a['merged']);r890=pathlib.Path(a['r890']);meta=json.load(open(m/'e57_pqa15_read_meta.json'));ids=meta['item_ids'];tok=json.load(open(m/'e57_pqa15_tokens.json'));tk=S2.tokenizer();dec=lambda t:tk.decode(t)
    ld=lambda d,c,l:json.load(open(d/f'E57_pqa15_rows_{c}_{l}.json'))
    def arr(d,l):
        A=P15.item_arrays(ld(d,'A',l),ids,tok['A'],dec,'q');B=P15.item_arrays(ld(d,'B',l),ids,tok['B'],dec,'q');return A['A_first'],B['A_first']
    w0a,w0b=arr(m,'W0');w0a2,w0b2=arr(r890,'W0');assert np.allclose(w0a,w0a2,atol=1e-3),'W0 rows of job 890 differ'
    def dd(d,l):a_,b_=arr(d,l);return a_-w0a,(a_-w0a)-(b_-w0b)
    res={}
    ref_dA,ref_dAB=dd(r890,'ALLQ128_n34')
    for lab in ('ALLQ128_n17','REP_n17','REPRAND_n17','NATQ_n17'):
        x,y=dd(m,lab);res[lab+'_minus_ALLQ128_n34_of_job890']=dict(first_dA=P15.boot_items(x-ref_dA),first_dAB=P15.boot_items(y-ref_dAB))
    for lab in ('REP_n17','REPRAND_n17'):
        x,y=dd(m,lab);x0,y0=dd(m,'ALLQ128_n17');res[lab+'_minus_ALLQ128_n17']=dict(first_dA=P15.boot_items(x-x0),first_dAB=P15.boot_items(y-y0))
    x,y=dd(m,'REP_n17');x2,y2=dd(m,'REPRAND_n17');res['REP_minus_REPRAND_n17']=dict(first_dA=P15.boot_items(x-x2),first_dAB=P15.boot_items(y-y2))
    pathlib.Path(a['json']).write_text(json.dumps(res,indent=1,default=float));print('wrote',a['json'])
def selftest():
    P16.selftest();rng=np.random.default_rng(0);ids=rng.integers(0,1000,CH*10);win=rng.random((10,CH))*3;band=win+rng.random((10,CH));lm=np.ones((10,CH),bool);lm[:,-1]=False
    rep,rnd,cnt=plan_bundle(ids,win,band,lm,lambda t:t%7==0,seed=1);assert max(len(v) for v in rep.values())<=NFRAG and all(len(rep[k])==len(rnd[k]) for k in rep) and min(rep)==1
    for k,v in rep.items():
        for p,j in v:assert j in [k-l for l in LAGS] and ids[p+1]%7!=0 and p>=FRAG-2
    holder={}
    class RT:
        eos=0;seq=staticmethod(lambda ids_,mask=None:(ids_,mask));chunk=staticmethod(lambda s,j:s)
    class LL:
        @staticmethod
        def prefix_masked(m,ids2,Kc,Vc,q0,kp2):return (np.asarray(ids2),np.asarray(kp2))
    import jax.numpy as jnp
    holder['L']=LL;u=UVRep({3:[(100,2),(900,2)]},ids,holder,RT);kp,info=u.keep(3,None,None,None,None,np.ones(WIN+CH,bool));pref2,kp2,c2,n=holder['rep'];assert n==2 and info['n_frag']==2
    i2,mk=c2;assert list(i2[:FRAG])==list(ids[100-FRAG+2:100+2]) and mk[FRAG-2] and mk[2*FRAG-2] and mk.sum()==2 and not np.asarray(kp2)[:,:WIN].any() and bool(np.asarray(kp2)[FRAG+5,WIN+FRAG]) and not bool(np.asarray(kp2)[FRAG+5,WIN+FRAG-1]) and not bool(np.asarray(kp2)[5,WIN+6])
    print('e57_pqa18_run selftest ok')
def main():
    if '--selftest' in sys.argv:selftest();return
    stage=P2._pop('--stage');g=lambda k:P2._pop(k)
    a=dict(out=g('--out'),src640=g('--src640'),items=g('--items-file'),bank=g('--bank'),assign=g('--assign'),stage1=g('--stage1'),dwg=g('--dwg'),rep_weight=g('--rep-weight'),json=g('--json'),merged=g('--merged'),r890=g('--r890'),max_groups=None)
    {'plan':plan_stage,'write':write_stage,'read':read_stage,'analyze':analyze,'compare':compare_stage}[stage](a)
if __name__=='__main__':main()
