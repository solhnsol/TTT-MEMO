"""E57 public QA stage 6 (E57 122-V / 123; new file; GPU): paragraph-GROUP episodes with shadowing (Hans: 'write ~10 paragraphs in a row, then the corresponding paragraphs of re-description 1, ..., and score all questions of those 10 paragraphs'). The 332 paragraphs of paragraphs.json are split by the fixed permutation of jobs 610 / 630 (np.random.default_rng(20261101).permutation(332), consecutive pieces; np.array_split into 33 groups of 10 or 11). Episode of a group = ONE stream starting from W0: section 1 = the group's original paragraphs, section 2 = their re-description 1, ..., section 9 = re-description 8 (rewrites_all8.json; paragraph order inside a section = the group order, the same in all sections; layout of e57_pqa2_stream.py: "\n\n"+first sentence, " "+later sentences, every sentence encoded alone). The stream ends at a chunk boundary after a fixed train-filler completion (the same filler text prefix for every group, no filler elsewhere). Read points n = 1, 3, 5, 9 = the end of the chunk that holds the last token of section n (the first tokens of the next section that already sit in that chunk are recorded as spill tokens).
Arms NATQ (engine NATADAMH, Adam direction, 0.5 x NAT norm, no mask) and ALLQ128 (SELC1 + band 128 on all query rows; here the mask is applied from chunk 0 - e57_qband_run.SelMaskQ switches it on only from chunk 8, which would leave ALLQ128 identical to NATQ in the first 8 chunks of a ~13-chunk episode). NAT is written per group only as the norm source (not read). States are saved at the read chunks and read afterwards.
Reads (FastOUT principle, e57_pqa2_run.FastOUT with packing PER GROUP): the group's items (all item ids of its paragraphs, items455 of the paragraph file; main373 flagged) x 16 candidates under A (own question), B (question of another item, E57_pqa4_assignB.json, seed 20261101) and C ("Answer:" only), OUT = the 8192-token stage-1 tail, for W0 and the 8 states of the group. The packing (chunk composition) is fixed inside this job, so comparisons are valid inside the job (scores can differ from jobs 610 / 630 by the packing-offset noise of ~0.05 nat per row).
  python3 e57_pqa6_run.py --stage build --para P --rewrites8 R --train T --dev D --out O | --stage write --out O [--max-groups N] | --stage read --out O --items-file F --bank B --assign A [--max-groups N] | --stage analyze --out O --items-file F --paragraphs P | --selftest"""
import sys,json,time,gc,hashlib,pathlib
import numpy as np
HERE=pathlib.Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
import e57_pqa2_stream as S
import e57_pqa3_stream as S3
import e57_pqa2_run as P2
import e57_pqa4_run as P4
import e57_squad_w0_reads as Q
CH=1024;NREAD=[1,3,5,9];ARMS=['NAT','NATQ','ALLQ128'];RARMS=['NATQ','ALLQ128'];NGROUPS=33;SEED=S3.SEED
def build(para_f,rew_f,train_f,dev_f,out):
    out=pathlib.Path(out);out.mkdir(parents=True,exist_ok=True)
    assert hashlib.sha256(open(para_f,'rb').read()).hexdigest()==S3.PARA_SHA and hashlib.sha256(open(rew_f,'rb').read()).hexdigest()=='eb08d0aacc78a512a19003bbc27cf084357b8ab008b84e037eb82a0f494cdb72','input sha256 differs'
    para=json.load(open(para_f));rew=json.load(open(rew_f));tk=S.tokenizer();n=len(para);perm=np.random.default_rng(SEED).permutation(n)
    fm,fx,ov=S3.filler_paragraphs(json.load(open(train_f)),json.load(open(dev_f)),n,SEED);filler=[]
    for ti,tx in fm[:24]:
        for s,x in S3.enc_par(tk,tx):filler+=x
    groups=[];ids_all={}
    for g,idx in enumerate(np.array_split(perm,NGROUPS)):
        ids=[];ends=[]
        for v in range(9):
            for i in idx:
                text=para[i]['context'] if v==0 else rew[para[i]['pid']][v-1]
                for s,x in S3.enc_par(tk,text):ids+=x
            ends.append(len(ids))
        need=-(-len(ids)//CH)*CH-len(ids);ids+=filler[:need];assert need<len(filler) and len(ids)%CH==0
        rch={str(k):(ends[k-1]-1)//CH for k in NREAD};spill={str(k):int((rch[str(k)]+1)*CH-ends[k-1]) for k in NREAD}
        groups.append(dict(gid=g,pids=[para[i]['pid'] for i in idx],item_ids=[x for i in idx for x in para[i]['item_ids']],main_ids=[x for i in idx for x in para[i]['item_ids'][:para[i]['n_main']]],n_chunks=len(ids)//CH,n_tokens=len(ids),section_end_tokens=ends,read_chunks=rch,spill_tokens=spill,filler_tokens=need,ids_sha=hashlib.sha256(np.asarray(ids,np.int64).tobytes()).hexdigest()[:16]))
        ids_all[f'g{g}']=np.asarray(ids,np.int64)
    np.savez_compressed(out/'e57_pqa6_group_ids.npz',**ids_all);meta=dict(seed=SEED,n_groups=NGROUPS,groups=groups,perm_sha=hashlib.sha256(perm.tobytes()).hexdigest()[:16],filler_dev_title_overlap=ov,read_n=NREAD,para_sha256=S3.PARA_SHA)
    (out/'e57_pqa6_groups.json').write_text(json.dumps(meta));print('groups',len(groups),'chunks per group',sorted({g['n_chunks'] for g in groups}),'items',sum(len(g['item_ids']) for g in groups),'max spill',max(max(g['spill_tokens'].values()) for g in groups),'filler tokens max',max(g['filler_tokens'] for g in groups));return meta
def load_groups(out):
    out=pathlib.Path(out);return json.load(open(out/'e57_pqa6_groups.json')),np.load(out/'e57_pqa6_group_ids.npz')
def make_selmask0():
    import e57_qband_run as QR,e57_closure_masks as CM
    class SelMaskAll0(QR.SelMaskQ):
        """ALLQ128 on all query rows from chunk 0 (SelMaskQ starts at chunk 8)"""
        def keep(self,k,m,Kc,Vc,ids_k,keep_all):
            base=np.asarray(keep_all,bool);kp=np.broadcast_to(base,(CH,QR.WIN+CH)).copy();bk=CM.band_keep(base,self.band);kp[:]=bk
            return kp,{'uv_arm':self.arm,'q_banded':CH,'q_band':self.band,'uv_hidden_tokens':0,'uv_hidden_chunk_tokens':0}
    return SelMaskAll0
def write_stage(a):
    out=pathlib.Path(a['out']);import e55_swap as E,e55_run as R,hashlib as H
    E.ADAM['NATADAMH']=('NAT',0.5,False)
    if 'NATADAMH' not in E.ARMS:E.ARMS.append('NATADAMH')
    LD,s3,C=R.lab();rt=LD.Runtime();lg=lambda *x:print(*x,flush=True);meta,ids_z=load_groups(out);SM=make_selmask0();masks={'band_ALLQ128':np.asarray(128),'all_ALLQ128':np.asarray(1)}
    fn=out/'e57_pqa6_writes.json';W=json.load(open(fn)) if fn.exists() else {};G=meta['groups'][:int(a['max_groups'])] if a.get('max_groups') else meta['groups'];t00=time.monotonic()
    for g in G:
        gid=str(g['gid'])
        if gid in W:continue
        ids=ids_z[f'g{gid}'];nch=len(ids)//CH;assert nch==g['n_chunks'];rd_ch={int(v) for v in g['read_chunks'].values()};t0=time.monotonic();rc={}
        L0=C.Ladder55(rt,save_dir=out/'states'/'rec',canary=None)
        with rt.mesh:Kh0,Vh0,_h=L0.record(ids)
        del _h;rc[(H.sha256(np.asarray(ids,np.int64).tobytes()).hexdigest(),nch)]=(Kh0,Vh0);del L0;gc.collect();ntg=None;res_g={}
        for arm in ARMS:
            L=C.Ladder55(rt,save_dir=out/'states'/f'g{gid}_{arm}',canary=None);L.rec_cache=rc
            uv=SM('ALLQ128',masks,lg) if arm=='ALLQ128' else None
            res=L.run_stream({'NAT':'NAT','NATQ':'NATADAMH','ALLQ128':'SELC1'}[arm],1,ids,nch=nch,tag=f'pqa6g{gid}{arm}',log=lambda *x:None,read_chunks=rd_ch if arm!='NAT' else (),chunks_meta=[{'sents':[]} for _ in range(nch)],norm_targets=ntg,uv=uv)
            if arm=='NAT':ntg=[e['update_norm'] for e in res['log']];res_g['NAT']=dict(update_norms=ntg,ce=[e['ce'] for e in res['log']])
            else:res_g[arm]=dict(states={str(k):dict(file=v['_file'],hash=v['_hash']) for k,v in res['reads'].items()},ce=[e['ce'] for e in res['log']],update_norms=[e['update_norm'] for e in res['log']])
            del L,res;gc.collect()
        del rc,Kh0,Vh0;gc.collect();W[gid]=res_g;fn.write_text(json.dumps(W,default=float));lg(f'group {gid}: {nch} chunks, written in {time.monotonic()-t0:.0f}s (total {time.monotonic()-t00:.0f}s)')
    lg('write stage done',len(W),'groups')
class FastOUTG(P2.FastOUT):
    """FastOUT with the packing done PER GROUP of rows; score_group(m32, g) scores only the chunks of group g"""
    def __init__(self,rt32,m32,tail_ids,groups_rows,GP=2,log=print):
        import e57_kvreuse as KR,packing as PK,jax,jax.numpy as jnp,equinox as eqx
        self.KR,self.PK,self.jax,self.jnp=KR,PK,jax,jnp;self.rt=rt32;self.tail=list(tail_ids);self.TL=TL=len(tail_ids);assert TL==8192;self.GP=GP;W=int(rt32.cfg.model.sliding_window_size)
        collect,resume=KR.make_prefix_fns(rt32);t0=time.monotonic();chunks=[];self.grange=[];self.gwhere=[]
        for rows in groups_rows:
            ch_,wh_=PK.pack_rows(rows,CH);o=len(chunks);chunks+=ch_;self.grange.append((o,o+len(ch_)));self.gwhere.append([(c+o,a,b) for c,a,b in wh_])
        n=len(chunks);ids=np.stack([c[0] for c in chunks]);seg=np.stack([c[1] for c in chunks]);loc=np.stack([c[2] for c in chunks]);self.nchunk=n
        with rt32.mesh,jax.default_matmul_precision('highest'):
            h,K,V_=collect(m32,jnp.asarray(self.tail,jnp.int32),jnp.arange(TL,dtype=jnp.int32),jnp.asarray(KR._attn_mask_np(np.full(TL,-2,np.int32),W)));self.H=h;jax.block_until_ready(V_);pref=[]
            for j in range(n):
                sf=np.concatenate([np.full(TL,-2,np.int32),seg[j]]);mask=KR._attn_mask_np(sf,W)[TL:,:]
                pref.append(np.asarray(resume(m32,jnp.asarray(ids[j]),jnp.asarray(TL+loc[j]),jnp.asarray(mask),K,V_)))
            self.pref1=np.stack(pref);self.tailseq=rt32.seq(self.tail+[rt32.eos]*CH)
            qs=[rt32.seq(list(x)) for x in ids];self.tg=np.stack([np.asarray(q.target_tokens) for q in qs]);self.mk=np.stack([np.asarray(q.loss_masks) for q in qs]);self.ids,self.seg,self.loc=ids,seg,loc
        log(f'  FastOUTG: tail K,V,H and {n} packed chunks ({len(groups_rows)} groups) prepared in {time.monotonic()-t0:.0f}s')
        from ttt.model.data import Batch
        @eqx.filter_jit
        def lp_packed(m,st,ids_,tg_,mk_,pref_,seg_,loc_):
            def one(i,t,k,p,s,l):
                PK.PACK_SUF['seg']=s;PK.PACK_SUF['loc']=l
                try:
                    r=m.language_model.suffix_call(prefix_outputs=p,state=st,seq=Batch(input_ids=i,target_tokens=t,loss_masks=k))
                    lg=r.logits.astype(jnp.float32);return jnp.take_along_axis(lg,t[:,None],axis=-1)[:,0]-jax.nn.logsumexp(lg,axis=-1)
                finally:PK.PACK_SUF.clear()
            return jax.vmap(one)(ids_,tg_,mk_,pref_,seg_,loc_)
        self.lp_packed=lp_packed
    def score_group(self,m32,g):
        rt=self.rt;jax=self.jax;jnp=self.jnp;from core import clone_state
        c0,c1=self.grange[g];idx=list(range(c0,c1));idx+=[idx[-1]]*((-len(idx))%self.GP)
        with rt.mesh,jax.default_matmul_precision('highest'):
            st=clone_state(rt.state_suffix)
            for j in range(self.TL//CH):st,_=rt.chunk_logp(m32,st,rt.chunk(self.tailseq,j),self.H[j*CH:(j+1)*CH])
            out={}
            for s in range(0,len(idx),self.GP):
                sel=idx[s:s+self.GP];r=np.asarray(self.lp_packed(m32,st,jnp.asarray(self.ids[sel]),jnp.asarray(self.tg[sel]),jnp.asarray(self.mk[sel]),jnp.asarray(self.pref1[sel]),jnp.asarray(self.seg[sel]),jnp.asarray(self.loc[sel])))
                for q,c in enumerate(sel):out.setdefault(c,r[q])
        return np.asarray([out[c][a:b].sum() for c,a,b in self.gwhere[g]])
def read_stage(a):
    out=pathlib.Path(a['out']);lg=lambda *x:print(*x,flush=True);meta,_=load_groups(out);W=json.load(open(out/'e57_pqa6_writes.json'));G=meta['groups'][:int(a['max_groups'])] if a.get('max_groups') else meta['groups']
    LD,rt,rt32,rd,items,h,raw0,ntok0,out_tail,tsha=P2.setup_read(a,lg);import e57_reads as RD
    byid={it['id']:it for it in items};asg=json.load(open(a['assign']))['assignment'];q={it['id']:it['prefix'] for it in items}
    conds=('A','B','C');sel_items=[];grows=[];gnt=[];gsizes=[]
    for g in G:
        its=[byid[i] for i in g['item_ids']];sel_items+=its;rows=[];nts=[];sz=[]
        for c in conds:
            pre=[q[it['id']] if c=='A' else (q[asg[it['id']]] if c=='B' else 'Answer:') for it in its];it2=[dict(it,prefix=p) for it,p in zip(its,pre)];r_,n_,bad=RD.build_rows(rt.tok,it2);assert not bad,bad;rows+=r_;nts.append(n_);sz.append(len(r_))
        grows.append(rows);gnt.append(nts);gsizes.append(sz)
    m32w0=rd.to_fp32(rt.base);fo=FastOUTG(rt32,m32w0,out_tail,grows,log=lg);labels=['W0']+[f'{arm}@n{n}' for arm in RARMS for n in NREAD];R={c:{l:[] for l in labels} for c in conds};tim={}
    def put(label,gi,flat):
        its=[byid[i] for i in G[gi]['item_ids']];o=0
        for ci,c in enumerate(conds):
            sub=flat[o:o+gsizes[gi][ci]];o+=gsizes[gi][ci];R[c][label]+=P2.rows_of(its,gnt[gi][ci],sub,f'{c}:{label}')
    t0=time.monotonic()
    for gi,g in enumerate(G):put('W0',gi,fo.score_group(m32w0,gi))
    tim['W0']=time.monotonic()-t0;lg(f'  W0 read: {tim["W0"]:.0f}s')
    for gi,g in enumerate(G):
        t0=time.monotonic()
        for arm in RARMS:
            for n in NREAD:
                s=W[str(g['gid'])][arm]['states'][str(g['read_chunks'][str(n)])];m=rd.load_state(s['file'],expect_hash=s['hash']);put(f'{arm}@n{n}',gi,fo.score_group(rd.to_fp32(m),gi));del m
        lg(f'  group {g["gid"]} read: {time.monotonic()-t0:.0f}s')
    for c in conds:
        for l in labels:(out/f'E57_pqa6_rows_{c}_{l.replace("@","_")}.json').write_text(json.dumps(R[c][l],default=float))
    (out/'e57_pqa6_read_meta.json').write_text(json.dumps(dict(items_sha256=h,out_tail_sha=tsha,n_items=len(sel_items),item_ids=[it['id'] for it in sel_items],timings=tim,assign='E57_pqa4_assignB.json',scorer='FastOUTG (packing per group; A, B, C rows of a group packed together)')))
def analyze(a):
    out=pathlib.Path(a['out']);rm=json.load(open(out/'e57_pqa6_read_meta.json'));ids=rm['item_ids'];meta,_=load_groups(out);main={i for g in meta['groups'] for i in g['main_ids']}
    ld=lambda c,l:json.load(open(out/f'E57_pqa6_rows_{c}_{l.replace("@","_")}.json'));masks={'items455':np.ones(len(ids),bool),'main373':np.asarray([i in main for i in ids])};res={}
    A0,B0,C0=ld('A','W0'),ld('B','W0'),ld('C','W0');mets=('lp_sum','lp_mean','rr_sum','rr_mean')
    for arm in RARMS:
        for n in NREAD:
            l=f'{arm}@n{n}';A,B,C=ld('A',l),ld('B',l),ld('C',l)
            res[l]={sn:dict(gain=P4.gain_table(A,B,A0,B0,m),A_abs={k:P4.boot(P4.metric_vec(A,k)[m]) for k in mets},B_abs={k:P4.boot(P4.metric_vec(B,k)[m]) for k in mets},C_abs={k:P4.boot(P4.metric_vec(C,k)[m]) for k in mets},C_minus_W0={k:P4.boot((P4.metric_vec(C,k)-P4.metric_vec(C0,k))[m]) for k in ('lp_sum','rr_mean')},acc_mean_A=P4.boot(np.asarray([float(r['acc_mean']) for r in A])[m])) for sn,m in masks.items()}
    base={sn:{c:{k:P4.boot(P4.metric_vec(R_,k)[m]) for k in mets} for c,R_ in (('A',A0),('B',B0),('C',C0))} for sn,m in masks.items()}
    (out/'E57_pqa6_keys.json').write_text(json.dumps(dict(meta=dict(n_items=len(ids),subsets={k:int(v.sum()) for k,v in masks.items()},gain_definition='(A(state)-A(W0)) - (B(state)-B(W0)); rank-ratio metrics as improvement (W0 minus state), positive = better',read_points='n = number of versions written (1 = original, 3 = through re-description 2, 5 = through 4, 9 = through 8)'),W0=base,states=res),indent=1,default=float))
def selftest():
    S3.selftest();P4.selftest()
    SM=make_selmask0() if False else None
    print('e57_pqa6_run selftest ok')
def main():
    if '--selftest' in sys.argv:selftest();return
    stage=P2._pop('--stage');g=lambda k:P2._pop(k)
    if stage=='build':build(g('--para'),g('--rewrites8'),g('--train'),g('--dev'),g('--out'));return
    a=dict(out=g('--out'),items=g('--items-file'),bank=g('--bank'),assign=g('--assign'),paragraphs=g('--paragraphs'),max_groups=g('--max-groups'),stage1=g('--stage1'))
    {'write':write_stage,'read':read_stage,'analyze':analyze}[stage](a)
if __name__=='__main__':main()
