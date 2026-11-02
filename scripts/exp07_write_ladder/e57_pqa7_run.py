"""E57 public QA stage 7 (E57 122-W / 123; new file; GPU): the arms GAINQ512 p.10 (and, as an option, RANDQ512 p.10) on the paragraph-group episodes of job 640 (e57_pqa6_run.py; same 33 groups, same streams, same read points n = 1 / 3 / 5 / 9, same A / B / C reads, same per-group packing). NATQ and ALLQ128 values, the NAT norm targets and the W0 rows are REUSED from job 640 (the W0 FastOUT rows of the first 3 groups are recomputed as a check of the packing).
Query-row mask (as the GAINQ jobs 570 / 575 / 585 / 596, band 512 instead of 128): stage score = forward-only W0 pass over the group's stream (the stream walked with a write step that returns the parameters unchanged, as e57_t1t2v2_gainq_run.score_stage): per row i of every chunk the natural NLL (full window) and the NLL with band-512 attention (row i sees only the previous 512 tokens, both layers, CM.band_keep) -> gain_i = nll_band512 - nll_win; stage mask = per chunk the top p = 10% of the ELIGIBLE rows (CM.rows_eligible: rows with a loss whose predicted token lies inside a sentence; all tokens of the episode streams are inside fragments) by gain (CM.top_rows) -> GAINQ512P10; RANDQ512P10 = a random 10% of the eligible rows of the same chunk (np.random.default_rng(700 + group id)). The mask applies from chunk 0 (as in job 640; the news jobs started at chunk 8). Rows with fewer than 512 preceding tokens (chunk 0, first half of chunk 1 only) have gain exactly 0: their band-512 attention equals the natural one, they can only be picked when fewer than 10% of the rows have positive gain. Write: SELC1 engine with the selected rows banded to 512, Adam direction, update norm 0.5 x the NAT norm of the same chunk (norm targets of job 640).
  python3 e57_pqa7_run.py --stage score|write|read|analyze --out O --src640 D640 [--arms GAINQ512P10,RANDQ512P10] [--max-groups N] [--items-file F --bank B --assign A --stage1 S1] | --selftest"""
import sys,json,time,gc,hashlib,pathlib
import numpy as np
HERE=pathlib.Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
import e57_closure_masks as CM
import e57_pqa2_run as P2
import e57_pqa4_run as P4
import e57_pqa6_run as P6
import e57_squad_w0_reads as Q
CH=1024;WIN_=8*CH;BAND=512;P=0.10;NREAD=P6.NREAD;ALLARMS=['GAINQ512P10','RANDQ512P10'];SEEDR=700
def make_selmask():
    import e57_qband_run as QR
    class SelMaskRows(QR.SelMaskQ):
        """band-B attention for the rows of the mask file, from chunk 0"""
        def keep(self,k,m,Kc,Vc,ids_k,keep_all):
            base=np.asarray(keep_all,bool);qb=CM.up(self.m[f'qb_{self.arm}_{k}'],CH);kp=np.broadcast_to(base,(CH,QR.WIN+CH)).copy();bk=CM.band_keep(base,self.band);kp[qb]=bk[qb]
            return kp,{'uv_arm':self.arm,'q_banded':int(qb.sum()),'q_band':self.band,'uv_hidden_tokens':0,'uv_hidden_chunk_tokens':0}
    return SelMaskRows
def build_group_masks(nll_win,nll_band,lm,gid,p=P):
    """-> dict name -> {chunk: packed bool}, stats; sid is all >= 0 for the episode streams"""
    nch=nll_win.shape[0];sid=np.zeros(CH,np.int32);rng=np.random.default_rng(SEEDR+gid);out={'GAINQ512P10':{},'RANDQ512P10':{}};st=[]
    for k in range(nch):
        el=CM.rows_eligible(lm[k],sid);n=int(round(p*el.sum()));gain=nll_band[k]-nll_win[k]
        sel=CM.top_rows(gain,el,n);out['GAINQ512P10'][k]=CM.pk(sel);out['RANDQ512P10'][k]=CM.pk(CM.rand_rows(el,n,rng))
        st.append(dict(chunk=k,n_selected=int(sel.sum()),n_positive_gain_rows=int(((gain>0)&el).sum()),mean_gain_selected=float(gain[sel].mean()) if sel.any() else 0.0,mean_gain_all=float(gain[el].mean()),n_zero_gain_selected=int(((gain==0)&sel).sum())))
    return out,st
def score_stage(a):
    out=pathlib.Path(a['out']);out.mkdir(parents=True,exist_ok=True);import jax,jax.numpy as jnp,equinox as eqx,e55_run as R
    LD,s3,C=R.lab();rt=LD.Runtime();lg=lambda *x:print(*x,flush=True);meta,ids_z=P6.load_groups(a['src640']);G=meta['groups'][:int(a['max_groups'])] if a.get('max_groups') else meta['groups']
    @eqx.filter_jit
    def logp_mask(m,st,c,pref,kmask):
        s3.S3SUF['kmask']=kmask
        try:res=m.language_model.suffix_call(prefix_outputs=pref,state=st,seq=c)
        finally:s3.S3SUF.clear()
        lp=jax.nn.log_softmax(res.logits.astype(jnp.float32),axis=-1);return jnp.take_along_axis(lp,c.target_tokens[:,None],axis=-1)[:,0]
    fn_out=out/'e57_pqa7_masks.npz';masks={};stats={};t00=time.monotonic()
    for g in G:
        gid=g['gid'];ids=ids_z[f'g{gid}'];nch=len(ids)//CH;rc={};t0=time.monotonic()
        L0=C.Ladder55(rt,save_dir=out/'states'/'rec',canary=None)
        with rt.mesh:Kh0,Vh0,_h=L0.record(ids)
        del _h;rc[(hashlib.sha256(np.asarray(ids,np.int64).tobytes()).hexdigest(),nch)]=(Kh0,Vh0);del L0;gc.collect()
        def nll_of(m,k,ids_k,Kc,Vc,keep,ctx):
            kp=jnp.asarray(keep)
            if keep is ctx['_natural']:return -np.asarray(logp_mask(m,ctx['st_w'],ctx['c'],ctx['pref_w'],kp),np.float64)
            pm=L.prefix_masked(m,jnp.asarray(ids_k,dtype=jnp.int32),Kc,Vc,jnp.int32(k*CH),kp)
            return -np.asarray(logp_mask(m,ctx['st_w'],ctx['c'],pm,kp),np.float64)
        def fn(k,m,Kc,Vc,keep,ids_k,ctx=None):
            keep=np.asarray(keep,bool);assert keep.ndim==1 and keep.shape[0]==WIN_+CH,keep.shape;ctx=dict(ctx);ctx['_natural']=keep;lm=np.asarray(ctx['c'].loss_masks,bool)
            with rt.mesh:
                nw=nll_of(m,k,ids_k,Kc,Vc,keep,ctx).astype(np.float32);nb=nll_of(m,k,ids_k,Kc,Vc,CM.band_keep(keep,BAND),ctx).astype(np.float32)
            return dict(nll=nw,nllb=nb,lm=lm)
        fn.wants_ctx=True
        L=C.Ladder55(rt,save_dir=out/'states'/'score',canary=None);L.rec_cache=rc
        _ws=L.write_step;L.write_step=lambda m,*a_,_ws=_ws,**kw:(m.inner_parameters(),)+tuple(_ws(m,*a_,**kw))[1:]
        res=L.run_stream('NAT',1,ids,nch=nch,tag=f'pqa7score{gid}',log=lambda *x:None,chunks_meta=[{'sents':[]} for _ in range(nch)],diag_fn=fn,diag_chunks=set(range(nch)))
        D=res['diag'];assert sorted(D)==list(range(nch));nw=np.stack([D[k]['nll'] for k in range(nch)]);nb=np.stack([D[k]['nllb'] for k in range(nch)]);lm=np.stack([D[k]['lm'] for k in range(nch)])
        m_,st=build_group_masks(nw,nb,lm,gid)
        for arm in ALLARMS:
            for k,v in m_[arm].items():masks[f'g{gid}_qb_{arm}_{k}']=v
        stats[str(gid)]=st;del L,res,rc,Kh0,Vh0;gc.collect()
        lg(f'group {gid}: {nch} chunks scored in {time.monotonic()-t0:.0f}s (total {time.monotonic()-t00:.0f}s); selected rows with positive gain {np.mean([s["n_positive_gain_rows"] for s in st]):.0f} mean per chunk')
    np.savez_compressed(fn_out,**masks);(out/'e57_pqa7_mask_stats.json').write_text(json.dumps(stats,default=float));lg('score stage done',len(stats),'groups')
def write_stage(a):
    out=pathlib.Path(a['out']);import e55_swap as E,e55_run as R,hashlib as H
    E.ADAM['NATADAMH']=('NAT',0.5,False)
    if 'NATADAMH' not in E.ARMS:E.ARMS.append('NATADAMH')
    LD,s3,C=R.lab();rt=LD.Runtime();lg=lambda *x:print(*x,flush=True);meta,ids_z=P6.load_groups(a['src640']);W640=json.load(open(pathlib.Path(a['src640'])/'e57_pqa6_writes.json'));SM=make_selmask();Z=np.load(out/'e57_pqa7_masks.npz')
    fn=out/'e57_pqa7_writes.json';W=json.load(open(fn)) if fn.exists() else {};G=meta['groups'][:int(a['max_groups'])] if a.get('max_groups') else meta['groups'];t00=time.monotonic()
    for g in G:
        gid=str(g['gid']);todo=[x for x in a['arms'] if x not in W.get(gid,{})]
        if not todo:continue
        ids=ids_z[f'g{gid}'];nch=len(ids)//CH;rd_ch={int(v) for v in g['read_chunks'].values()};ntg=W640[gid]['NAT']['update_norms'];assert len(ntg)==nch;t0=time.monotonic();rc={}
        L0=C.Ladder55(rt,save_dir=out/'states'/'rec',canary=None)
        with rt.mesh:Kh0,Vh0,_h=L0.record(ids)
        del _h;rc[(H.sha256(np.asarray(ids,np.int64).tobytes()).hexdigest(),nch)]=(Kh0,Vh0);del L0;gc.collect();W.setdefault(gid,{})
        for arm in todo:
            masks={'band_'+arm:np.asarray(BAND),'all_'+arm:np.asarray(0)}
            for k in range(nch):masks[f'qb_{arm}_{k}']=Z[f'g{gid}_qb_{arm}_{k}']
            L=C.Ladder55(rt,save_dir=out/'states'/f'g{gid}_{arm}',canary=None);L.rec_cache=rc
            res=L.run_stream('SELC1',1,ids,nch=nch,tag=f'pqa7g{gid}{arm}',log=lambda *x:None,read_chunks=rd_ch,chunks_meta=[{'sents':[]} for _ in range(nch)],norm_targets=ntg,uv=SM(arm,masks,lambda *x:None))
            W[gid][arm]=dict(states={str(k):dict(file=v['_file'],hash=v['_hash']) for k,v in res['reads'].items()},ce=[e['ce'] for e in res['log']],update_norms=[e['update_norm'] for e in res['log']],q_banded=[e.get('q_banded') for e in res['log']]);del L,res;gc.collect()
        del rc,Kh0,Vh0;gc.collect();fn.write_text(json.dumps(W,default=float));lg(f'group {gid}: {nch} chunks, {len(todo)} arm(s) written in {time.monotonic()-t0:.0f}s (total {time.monotonic()-t00:.0f}s)')
    lg('write stage done',len(W),'groups')
def read_stage(a):
    out=pathlib.Path(a['out']);src=pathlib.Path(a['src640']);lg=lambda *x:print(*x,flush=True);meta,_=P6.load_groups(src);W=json.load(open(out/'e57_pqa7_writes.json'));G=meta['groups'][:int(a['max_groups'])] if a.get('max_groups') else meta['groups']
    LD,rt,rt32,rd,items,h,raw0,ntok0,out_tail,tsha=P2.setup_read(a,lg);import e57_reads as RD
    byid={it['id']:it for it in items};asg=json.load(open(a['assign']))['assignment'];q={it['id']:it['prefix'] for it in items};conds=('A','B','C');grows=[];gnt=[];gsizes=[];sel_items=[]
    for g in G:
        its=[byid[i] for i in g['item_ids']];sel_items+=its;rows=[];nts=[];sz=[]
        for c in conds:
            pre=[q[it['id']] if c=='A' else (q[asg[it['id']]] if c=='B' else 'Answer:') for it in its];it2=[dict(it,prefix=p) for it,p in zip(its,pre)];r_,n_,bad=RD.build_rows(rt.tok,it2);assert not bad,bad;rows+=r_;nts.append(n_);sz.append(len(r_))
        grows.append(rows);gnt.append(nts);gsizes.append(sz)
    m32w0=rd.to_fp32(rt.base);fo=P6.FastOUTG(rt32,m32w0,out_tail,grows,log=lg);labels=[f'{arm}@n{n}' for arm in a['arms'] for n in NREAD];R={c:{l:[] for l in labels} for c in conds}
    def put(label,gi,flat):
        its=[byid[i] for i in G[gi]['item_ids']];o=0
        for ci,c in enumerate(conds):
            sub=flat[o:o+gsizes[gi][ci]];o+=gsizes[gi][ci];R[c][label]+=P2.rows_of(its,gnt[gi][ci],sub,f'{c}:{label}')
    chk=[]
    for gi in range(min(3,len(G))):                       # packing check against the W0 rows of job 640
        f0=fo.score_group(m32w0,gi);o=0;ref=[]
        for ci,c in enumerate(conds):
            rr=json.load(open(src/f'E57_pqa6_rows_{c}_W0.json'));its=G[gi]['item_ids'];start=sum(len(x['item_ids']) for x in G[:gi]);ref+=[x for r in rr[start:start+len(its)] for x in r['cand_lp']]
        chk.append(float(np.abs(np.asarray(ref)-f0).max()))
    lg('  W0 packing check vs job 640, max|diff| of the first groups:',chk)
    for gi,g in enumerate(G):
        t0=time.monotonic()
        for arm in a['arms']:
            for n in NREAD:
                s=W[str(g['gid'])][arm]['states'][str(g['read_chunks'][str(n)])];m=rd.load_state(s['file'],expect_hash=s['hash']);put(f'{arm}@n{n}',gi,fo.score_group(rd.to_fp32(m),gi));del m
        lg(f'  group {g["gid"]} read: {time.monotonic()-t0:.0f}s')
    for c in conds:
        for l in labels:(out/f'E57_pqa7_rows_{c}_{l.replace("@","_")}.json').write_text(json.dumps(R[c][l],default=float))
    (out/'e57_pqa7_read_meta.json').write_text(json.dumps(dict(items_sha256=h,out_tail_sha=tsha,item_ids=[it['id'] for it in sel_items],w0_packing_check_max_abs_diff=chk,arms=a['arms'])))
def analyze(a):
    out=pathlib.Path(a['out']);src=pathlib.Path(a['src640']);rm=json.load(open(out/'e57_pqa7_read_meta.json'));ids=rm['item_ids'];assert ids==json.load(open(src/'e57_pqa6_read_meta.json'))['item_ids'];meta,_=P6.load_groups(src);main={i for g in meta['groups'] for i in g['main_ids']}
    ldo=lambda c,l:json.load(open(out/f'E57_pqa7_rows_{c}_{l.replace("@","_")}.json'));ld6=lambda c,l:json.load(open(src/f'E57_pqa6_rows_{c}_{l.replace("@","_")}.json'))
    masks={'items455':np.ones(len(ids),bool),'main373':np.asarray([i in main for i in ids])};A0,B0,C0=ld6('A','W0'),ld6('B','W0'),ld6('C','W0');mets=('lp_sum','lp_mean','rr_sum','rr_mean');res={}
    gv=lambda R_,k:P4.metric_vec(R_,k)
    for arm in rm['arms']:
        for n in NREAD:
            l=f'{arm}@n{n}';A,B,C=ldo('A',l),ldo('B',l),ldo('C',l)
            e={sn:dict(gain=P4.gain_table(A,B,A0,B0,m),A_abs={k:P4.boot(gv(A,k)[m]) for k in mets},B_abs={k:P4.boot(gv(B,k)[m]) for k in mets},C_minus_W0={k:P4.boot((gv(C,k)-gv(C0,k))[m]) for k in ('lp_sum','rr_mean')},acc_mean_A=P4.boot(np.asarray([float(r['acc_mean']) for r in A])[m])) for sn,m in masks.items()}
            for ref in ('NATQ','ALLQ128'):                       # paired differences to the arms of job 640 (A lp, A acc, gain A-B)
                RA,RB=ld6('A',f'{ref}@n{n}'),ld6('B',f'{ref}@n{n}');dA=gv(A,'lp_sum')-gv(RA,'lp_sum');dG=(gv(A,'lp_sum')-gv(B,'lp_sum'))-(gv(RA,'lp_sum')-gv(RB,'lp_sum'));dacc=np.asarray([float(r['acc_mean']) for r in A])-np.asarray([float(r['acc_mean']) for r in RA])
                for sn,m in masks.items():e[sn][f'minus_{ref}']=dict(A_lp_sum=P4.boot(dA[m]),A_acc_mean=P4.boot(dacc[m]),gain_A_minus_B=P4.boot(dG[m]))
            res[l]=e
    ms=json.load(open(out/'e57_pqa7_mask_stats.json'));allst=[s for v in ms.values() for s in v]
    (out/'E57_pqa7_keys.json').write_text(json.dumps(dict(meta=dict(arms=rm['arms'],band=BAND,p=P,w0_packing_check_max_abs_diff=rm['w0_packing_check_max_abs_diff'],mask_rows_per_chunk=float(np.mean([s['n_selected'] for s in allst])),selected_rows_with_zero_gain_share=float(np.sum([s['n_zero_gain_selected'] for s in allst])/max(1,np.sum([s['n_selected'] for s in allst]))),mean_gain_selected=float(np.mean([s['mean_gain_selected'] for s in allst])),mean_gain_all_eligible=float(np.mean([s['mean_gain_all'] for s in allst]))),states=res),indent=1,default=float))
def selftest():
    P6.selftest()
    rng=np.random.default_rng(0);nw=rng.random((4,CH));nb=nw+rng.random((4,CH))-0.3;nb[0,:512]=nw[0,:512];lm=np.ones((4,CH),bool);m,st=build_group_masks(nw,nb,lm,3)
    for arm in ALLARMS:assert all(CM.up(m[arm][k],CH).sum()==int(round(.10*(CH-1))) for k in range(4))
    assert not CM.up(m['GAINQ512P10'][0],CH)[:512].any() and not CM.up(m['GAINQ512P10'][0],CH)[CH-1]
    m2,_=build_group_masks(nw,nb,lm,3);assert all(np.array_equal(m[a][k],m2[a][k]) for a in ALLARMS for k in range(4))
    print('e57_pqa7_run selftest ok')
def main():
    if '--selftest' in sys.argv:selftest();return
    stage=P2._pop('--stage');g=lambda k:P2._pop(k);a=dict(out=g('--out'),src640=g('--src640'),arms=(g('--arms') or 'GAINQ512P10').split(','),max_groups=g('--max-groups'),items=g('--items-file'),bank=g('--bank'),assign=g('--assign'),stage1=g('--stage1'))
    {'score':score_stage,'write':write_stage,'read':read_stage,'analyze':analyze}[stage](a)
if __name__=='__main__':main()
