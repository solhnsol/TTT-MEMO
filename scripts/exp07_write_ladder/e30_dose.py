"""E30 fact-free context dose-response at W0 (3B TTT-E2E, ONE aggregated step; NOT RUN: written without GPU access, built on s3.py Ladder3 (prefix_masked, record, grad_mask, grad_nat, read_state, set_inner)).
For every cell (e30_core.CELLS) and world A/B: G = sum_k w_k g_k over the c2 write chunks (the 26 fact-intact chunks of the screen, as e29_factors --chunks c2; w_k = loss tokens of chunk k / total; g_k = gradient of the chunk's
mean NTP loss w.r.t. the inner parameters at W0), then W = W0 - eta * coef * s * G (eta = lr x ilr multiplier, coef = min(1, clip/||G||) on the aggregated G over ALL inner parameters), s in --scales; readout = the full R1-alt/R1
ladder evaluation (separate eval process: the write process holds no fp32 model).
Context arms (F, P, R, N): chunk k is written in the mini-stream  [ctx tokens (C = 8192, or ceil(L/1024)*1024 for P)] + [the world's chunk k tokens] + [its next token] + pad to a chunk multiple:
  - prefix layers: K,V of the ctx tokens (record(), positions 0..C-1, left-padded to the 8192 slots) + prefix_masked on the chunk at position q0 = C with keep = nearest L slots visible (e30_core.keep_vector);
  - suffix layers: the carried KV = the suffix state after reading the ctx chunks at W0 (read_state chain), the same keep vector AND-ed into the sliding-window mask (S3SUF kmask);
  - chunk batch = rt.chunk(mini_seq, C/1024), so positions / chunk counter / targets are those of a stream whose chunk C/1024 is this chunk; loss mask = the native loss mask of chunk k (R: first 128 excluded).
  Native A: the natural chain and natural prefix of the world's own stream. M: chunk alone at its stream position (every window key removed, as E29's M / prefix-all-pos).
Checks (--selfcheck, a few chunks, writes nothing): (i) the context pipeline fed with the REAL stream window must approximately reproduce the native gradient of A (cosine >= 0.99, CE within 0.05 nat; not exact because the re-read window lacks the context before it), (ii) F0 vs M gradients (relative difference <= 1e-3).
Modes: --mode save (aggregate, apply, save state files + e30_save json), --mode eval (score the states). Cells in groups (--cells core|all|A,M,F0,...): the accumulators (one inner-parameter pytree per cell and world) must fit the GPU.
Usage: e30_dose.py --seed 101 --ctx-dir DIR --cells core --scales 1,4 --mode save --tag e30core ; then --mode eval ; --selfcheck."""
import os,sys,json,time,argparse,hashlib
from pathlib import Path
HERE=Path(__file__).resolve().parent;sys.path.insert(0,str(HERE))
import e30_core as C
import s3 as S3M
from s3 import *
import ladder as LD
from ladder import *
OUT=LD.OUT
class Ladder30(Ladder3):
    def setup30(self,ctx_dir,seed):
        self.ctx_texts={}
        for src in ('bg','bg2','raw'):
            f=Path(ctx_dir)/f'e30_ctx_{src}_s{seed}.json'
            if f.exists():self.ctx_texts[src]=json.load(open(f))['segments']
        self._ctx_tok={}
    def ctx_ids(self,src,r,world_stream=None,k=None):
        """fact-free tokens (no BOS) of segment r, exactly WIN tokens (aborts if the text is too short); src 'stream' (selfcheck only): the real window before chunk k."""
        if src=='stream':return [int(x) for x in world_stream[k*CH-WIN:k*CH]]
        key=(src,r)
        if key not in self._ctx_tok:
            ids=stage2.enc(self.T.tok,self.ctx_texts[src][r]).ids;assert len(ids)>=WIN,f'context {src}[{r}] has {len(ids)} < {WIN} tokens';self._ctx_tok[key]=ids[:WIN]
        return self._ctx_tok[key]
    def record_ctx(self,ids):
        Kh,Vh,_=self.record(ids);n=len(ids)
        if n<WIN:
            pad=[(0,0),(WIN-n,0)]+[(0,0)]*(Kh.ndim-2);Kh=np.pad(Kh,pad);Vh=np.pad(Vh,pad)
        return jnp.asarray(Kh),jnp.asarray(Vh)
    def prep(self,ctx,chunk_ids,next_tok,lmask,Kc,Vc):
        """mini-stream for one (context, chunk, world): returns (chunk Batch, suffix state after the context, prefix_masked inputs); lmask = native loss mask of the chunk (bool [CH])."""
        rt=self.rt;m=rt.base;Cn=len(ctx)
        full=list(ctx)+[int(x) for x in chunk_ids]+[int(next_tok)];full+=[int(rt.eos)]*((-len(full))%CH)
        mask=np.zeros(len(full),bool);mask[Cn:Cn+CH]=lmask;seq=rt.seq(np.asarray(full,np.int32),mask=mask);pref=rt.prefix(m,seq);st=clone_state(rt.state_suffix)
        for j in range(Cn//CH):st=self.read_state(m,st,rt.chunk(seq,j),pref[j*CH:(j+1)*CH])
        return rt.chunk(seq,Cn//CH),st
    def grad_cell(self,cell,k,nat,ctx_prep,Kz,Vz):
        """gradient (inner parameters) and chunk CE of one cell for one (world, chunk). nat = (chunk Batch, natural suffix state, natural prefix outputs of the chunk)."""
        m=self.rt.base;cd=C.CELLS[cell];c_nat,st_nat,pref_nat_c,ids_chunk=nat
        if cd['kind']=='A':return self.grad_nat(m,st_nat,c_nat,pref_nat_c)
        if cd['kind']=='M':
            keep=jnp.asarray(C.keep_vector('M'));pw=self.prefix_masked(m,jnp.asarray(ids_chunk,dtype=jnp.int32),Kz,Vz,jnp.int32(k*CH),keep);return self.grad_mask(m,st_nat,c_nat,pw,keep)
        c,st,Kc,Vc,q0=ctx_prep;keep=jnp.asarray(C.keep_vector(cell));skip=C.loss_skip(cell)
        if skip:c=eqx.tree_at(lambda b:b.loss_masks,c,c.loss_masks.at[:skip].set(False))
        pw=self.prefix_masked(m,jnp.asarray(ids_chunk,dtype=jnp.int32),Kc,Vc,jnp.int32(q0),keep);return self.grad_mask(m,st,c,pw,keep)
def canon(cells):
    """cells with identical computation (same context, visibility, loss skip) share one accumulator: returns ({cell: representative}, [representatives])."""
    rep={};seen={}
    for n in cells:
        cd=C.CELLS[n];key=(cd['kind'] if cd['kind'] in ('A','M') else 'ctx',C.prep_key(n),C.keep_vector(n).tobytes(),C.loss_skip(n))
        if key in seen:rep[n]=seen[key]
        else:seen[key]=n;rep[n]=n
    return rep,sorted(set(rep.values()),key=cells.index)
def world_setup(L,w,ks=None):
    rt=L.rt;ids=np.asarray(L.T.P[w=='B'].ids)[:NCH*CH];mask=np.zeros(NCH*CH,dtype=bool);mask[:len(ids)-1]=True;seq=rt.seq(ids,mask=mask)
    if ks is None:ks=sorted({x[3]['k'] for x in SP.chunk_rows(L.T.w,w=='B',L.T.tok,rt.bos,rt.eos,'X')[0]})   # c2: the fact-intact chunks (as e29_factors --chunks c2)
    return ids,seq,ks
def check_ctx_files(L,cells,nks):
    for n in cells:
        k=C.prep_key(n)
        if k and k[0]!='stream':
            assert k[0] in L.ctx_texts,f'cell {n}: context file for {k[0]} missing (run e30_ctx.py first)';assert len(L.ctx_texts[k[0]])>=nks,f'{k[0]}: {len(L.ctx_texts[k[0]])} segments < {nks} write chunks'
def sts_ref(L,sq,pf,n):
    st=clone_state(L.rt.state_suffix)
    for j in range(n):st=L.read_state(L.rt.base,st,L.rt.chunk(sq,j),pf[j*CH:(j+1)*CH])
    return st
def selfcheck(L,a):
    """no state files. (i) real-stream window through the context pipeline vs native gradient; (ii) F0 vs M."""
    rt=L.rt;res={'rows':[]};cells=['A','M','F0'];ids,seq,ks=world_setup(L,'A');ks=[8]+[k for k in ks if k>=9][:2]   # k=8: the re-read window is the WHOLE history (chunks 0-7), so the context pipeline must reproduce native exactly; k>=9: the window truncates the receptive field of the 24 prefix layers and the suffix chain, so only approximately
    with rt.mesh:
        pref_nat=rt.prefix(rt.base,seq);st=clone_state(rt.state_suffix);Kz=None
        for k in range(max(ks)+1):
            c=rt.chunk(seq,k);pc=pref_nat[k*CH:(k+1)*CH]
            if k in ks:
                lm=np.asarray(c.loss_masks);ch=ids[k*CH:(k+1)*CH];nt=ids[(k+1)*CH] if (k+1)*CH<len(ids) else rt.eos;nat=(c,st,pc,ch)
                gA,ceA=L.grad_nat(rt.base,st,c,pc)
                ctx=L.ctx_ids('stream',0,ids,k);Kc,Vc=L.record_ctx(ctx);cs,sts=L.prep(ctx,ch,nt,lm,Kc,Vc);gS,ceS=L.grad_mask(rt.base,sts,cs,L.prefix_masked(rt.base,jnp.asarray(ch,dtype=jnp.int32),Kc,Vc,jnp.int32(WIN),jnp.ones(WIN+CH,bool)),jnp.ones(WIN+CH,bool))
                Kz=jnp.zeros_like(Kc);Vz=jnp.zeros_like(Vc);gM,ceM=L.grad_cell('M',k,nat,None,Kz,Vz)
                keepM=jnp.asarray(C.keep_vector('M'));gM8,ceM8=L.grad_mask(rt.base,st,c,L.prefix_masked(rt.base,jnp.asarray(ch,dtype=jnp.int32),Kz,Vz,jnp.int32(WIN),keepM),keepM)   # M with the chunk at position 8192 (same math as M, only the absolute RoPE position differs: bf16 position noise floor)
                c0=L.ctx_texts and ('bg' in L.ctx_texts)
                row={'k':k,'ce_native':float(ceA),'ce_realctx_pipeline':float(ceS),'ce_M':float(ceM)}
                def cmp(x,y):
                    dot=sum(float(jnp.vdot(p,q)) for p,q in zip(jax.tree.leaves(x),jax.tree.leaves(y)));nx=float(optax.global_norm(x));ny=float(optax.global_norm(y))
                    d=float(optax.global_norm(jax.tree.map(lambda p,q:p-q,x,y)));return {'cos':dot/(nx*ny),'rel_diff':d/ny,'norm_x':nx,'norm_y':ny}
                row['realctx_vs_native']=cmp(gS,gA)
                full=list(ctx)+[int(x) for x in ch]+[int(nt)];full+=[int(rt.eos)]*((-len(full))%CH);mk_=np.zeros(len(full),bool);mk_[len(ctx):len(ctx)+CH]=lm;sq=rt.seq(np.asarray(full,np.int32),mask=mk_);pf=rt.prefix(rt.base,sq);gW,ceW=L.grad_nat(rt.base,sts_ref(L,sq,pf,len(ctx)//CH),rt.chunk(sq,len(ctx)//CH),pf[len(ctx):len(ctx)+CH])   # reference: the same 8192+chunk tokens forwarded in one piece at W0 (E30 design 2.x)
                row['realctx_vs_windowref']=cmp(gS,gW);row['ce_windowref']=float(ceW)
                if c0:
                    ff=L.ctx_ids('bg',0);Kf,Vf=L.record_ctx(ff);cf,stf=L.prep(ff,ch,nt,lm,Kf,Vf);gF,ceF=L.grad_mask(rt.base,stf,cf,L.prefix_masked(rt.base,jnp.asarray(ch,dtype=jnp.int32),Kf,Vf,jnp.int32(WIN),jnp.asarray(C.keep_vector('F0'))),jnp.asarray(C.keep_vector('F0')))
                    row['F0_vs_M']=cmp(gF,gM);row['F0_vs_M8192']=cmp(gF,gM8);row['M8192_vs_M']=cmp(gM8,gM);row['ce_F0']=float(ceF)
                res['rows'].append(row);print(json.dumps(row),flush=True)
            st=L.read_state(rt.base,st,c,pc)
    old=os.environ.get('E30_GATES','new')=='old'
    res['gate_realctx_cos_ge_0.99']=all(r['realctx_vs_native']['cos']>=0.99 and abs(r['ce_realctx_pipeline']-r['ce_native'])<0.05 for r in res['rows'])   # original gate (job 59 failed it at k=17-19); now descriptive
    res['gate_realctx_vs_windowref']=all(r['realctx_vs_windowref']['cos']>=0.999 and r['realctx_vs_windowref']['rel_diff']<=0.03 for r in res['rows'])   # E30 design: context path vs the same 8192+chunk tokens forwarded at once at W0
    r8=[r for r in res['rows'] if r['k']==8]
    if r8:res['info_k8_realctx_vs_native_cos']=r8[0]['realctx_vs_native']['cos']   # k=8: window = whole history, expected ~1 (descriptive)
    if all('F0_vs_M' in r for r in res['rows']):
        res['gate_F0_vs_M_rel_diff_le_1e-3']=all(r['F0_vs_M']['rel_diff']<=1e-3 for r in res['rows'])   # original gate, now descriptive (job 59: 1.4e-2)
        res['gate_F0_vs_M_cos_ge_0.999']=all(r['F0_vs_M']['cos']>=0.999 for r in res['rows'])   # E30 design: new gate; rel_diff and F0 vs M8192 (position noise floor) reported only
    return_old=res['gate_realctx_cos_ge_0.99'] and res.get('gate_F0_vs_M_rel_diff_le_1e-3',True)
    json.dump(res,open(OUT/f'e30_selfcheck_s{a.seed}.json','w'),indent=1);print('SELFCHECK',{k:v for k,v in res.items() if k.startswith('gate')},flush=True)
    if old:return 0 if return_old else 4
    return 0 if res['gate_realctx_vs_windowref'] and res.get('gate_F0_vs_M_cos_ge_0.999',True) else 4
def zeros_kv(L):
    """zero K,V with the slot layout [24, WIN, heads, dim] (content irrelevant when every window key is masked, as E29's M)."""
    Kc,Vc=L.record_ctx([int(L.rt.bos)]*CH);return jnp.zeros_like(Kc),jnp.zeros_like(Vc)
def save_mode(L,a,cells):
    rt=L.rt;st_all=clone_state(rt.state_all);inner0=rt.base.inner_parameters();eta=L.lr*float(rt.base.get_ilr_multiplier(st_all.get(rt.base.step_index)));clip=L.clip
    rep,reps=canon(cells);scales=[float(x) for x in a.scales.split(',')]
    man={'seed':a.seed,'cells':cells,'alias':rep,'scales':scales,'eta':eta,'clip':clip,'W0_inner_hash':reuse._hash_pytree(inner0)[0],'code_sha':hashlib.sha256(open(__file__,'rb').read()).hexdigest()}
    nparam=sum(int(np.prod(x.shape)) for x in jax.tree.leaves(inner0) if hasattr(x,'shape'));acc_bytes=len(reps)*2*nparam*4;man['accumulator_bytes']=acc_bytes
    try:lim=jax.devices()[0].memory_stats().get('bytes_limit')
    except Exception:lim=None
    if lim:assert acc_bytes<0.5*lim,f'accumulators {acc_bytes/2**30:.1f} GiB > 50% of the device memory {lim/2**30:.1f} GiB: split --cells into smaller groups'
    print(f'cells {reps}, aliases {({k:v for k,v in rep.items() if k!=v})}; {nparam/1e6:.0f}M inner parameters, accumulators {acc_bytes/2**30:.1f} GiB',flush=True)
    W={w:world_setup(L,w) for w in 'AB'};assert W['A'][2]==W['B'][2],'c2 chunk set differs between worlds';ks=W['A'][2];check_ctx_files(L,reps,len(ks));man['chunks']=ks
    nk={w:{k:np.asarray(rt.chunk(W[w][1],k).loss_masks) for k in ks} for w in 'AB'}
    def count(w,k,n):
        lm=nk[w][k];sk=C.loss_skip(n);return int(lm.sum())-(int(lm[:sk].sum()) if sk else 0)
    Ntot={(w,n):sum(count(w,k,n) for k in ks) for w in 'AB' for n in reps}
    G={(w,n):jax.tree.map(lambda x:jnp.zeros(x.shape,jnp.float32),inner0) for w in 'AB' for n in reps}
    addw=jax.jit(lambda acc,g,wt:jax.tree.map(lambda x,y:x+wt*y.astype(jnp.float32),acc,g))
    per_chunk={(w,n):[] for w in 'AB' for n in reps};t0=time.monotonic();need_zero=any(C.CELLS[n]['kind']=='M' for n in reps);Kz=Vz=None
    with rt.mesh:
        if need_zero:Kz,Vz=zeros_kv(L)
        pref={w:rt.prefix(rt.base,W[w][1]) for w in 'AB'};st={w:clone_state(rt.state_suffix) for w in 'AB'}
        for k in range(max(ks)+1):
            if k in ks:
                r=ks.index(k);tk=time.monotonic();kvs={};preps={};tprep=0.0
                for w in 'AB':
                    ids=W[w][0];c=rt.chunk(W[w][1],k);pc=pref[w][k*CH:(k+1)*CH];ch=ids[k*CH:(k+1)*CH];nt=ids[(k+1)*CH] if (k+1)*CH<len(ids) else rt.eos;nat=(c,st[w],pc,ch);lm=nk[w][k]
                    for n in reps:
                        key=C.prep_key(n)
                        if key is not None and (w,key) not in preps:
                            tp=time.monotonic();ctx=L.ctx_ids(key[0],r)[-key[1]:]
                            if key not in kvs:kvs[key]=L.record_ctx(ctx)
                            cs,sts=L.prep(ctx,ch,nt,lm,kvs[key][0],kvs[key][1]);preps[(w,key)]=(cs,sts,kvs[key][0],kvs[key][1],len(ctx));tprep+=time.monotonic()-tp
                        g,ce=L.grad_cell(n,k,nat,preps.get((w,key)) if key else None,Kz,Vz)
                        G[(w,n)]=addw(G[(w,n)],g,count(w,k,n)/Ntot[(w,n)])
                        per_chunk[(w,n)].append({'k':k,'ce':float(ce),'gnorm':float(optax.global_norm(g)),'n_loss':count(w,k,n)});del g
                del kvs,preps
                print(f'  chunk {k} ({r+1}/{len(ks)}) {time.monotonic()-tk:.0f}s (context prep {tprep:.0f}s) total {time.monotonic()-t0:.0f}s',flush=True)
            for w in 'AB':
                c=rt.chunk(W[w][1],k);st[w]=L.read_state(rt.base,st[w],c,pref[w][k*CH:(k+1)*CH])
    man['sec_write']=time.monotonic()-t0;log={'manifest':man,'cells':{}};done={};sd=OUT/'states'/f'e30_{a.tag}_s{a.seed}';sd.mkdir(parents=True,exist_ok=True)
    for n in cells:
        rn=rep[n];log['cells'][n]={}
        for w in 'AB':
            Gs=G[(w,rn)];gn=float(optax.global_norm(Gs));coef=min(1.0,clip/max(gn,1e-30));ent={'rep':rn,'gnorm':gn,'coef':coef,'chunks':per_chunk[(w,rn)],'scales':{}}
            for s in scales:
                if (w,rn,s) not in done:
                    newp=jax.tree.map(lambda p,x:(p+(-eta*coef*s*x).astype(p.dtype)).astype(p.dtype),inner0,Gs);mm=L.set_inner(rt.base,newp);fn=sd/f'{rn}_{w}_s{s:g}.npz'
                    np.savez(fn,*[np.asarray(x) for x in jax.tree_util.tree_leaves(eqx.partition(mm,L.f)[0])]);done[(w,rn,s)]={'file':str(fn),'hash':reuse._hash_pytree(mm.inner_parameters())[0]};del newp,mm
                ent['scales'][f'{s:g}']=done[(w,rn,s)]
            log['cells'][n][w]=ent
    json.dump(log,open(OUT/f'e30_save_{a.tag}_s{a.seed}.json','w'),default=str);print('saved',OUT/f'e30_save_{a.tag}_s{a.seed}.json',round(man['sec_write']),'s',flush=True)
def eval_mode(L,a):
    sv=json.load(open(OUT/f'e30_save_{a.tag}_s{a.seed}.json'));t0=time.monotonic();res={'manifest':sv['manifest'],'cells':{},'items':L.items,'fam_sizes':{k:len(v) for k,v in L.fam.items()}};cache={}
    ev0=L.evaluate(L.rt.base);res['W0']=ev0
    for n,ws in sv['cells'].items():
        res['cells'][n]={}
        for w,ent in ws.items():
            res['cells'][n][w]={'gnorm':ent['gnorm'],'coef':ent['coef'],'rep':ent['rep'],'scales':{}}
            for s,info in ent['scales'].items():
                fn=info['file']
                if fn not in cache:
                    mm=L.load_state(fn);assert reuse._hash_pytree(mm.inner_parameters())[0]==info['hash'],'loaded weights differ from the saved ones';cache[fn]=L.evaluate(mm)
                    if not a.keep:os.remove(fn)
                res['cells'][n][w]['scales'][s]={'hash':info['hash'],'margins':cache[fn]}
    res['sec_eval']=time.monotonic()-t0;json.dump(res,open(OUT/f'e30_{a.tag}_s{a.seed}.json','w'));print('scored',round(res['sec_eval']),'s',flush=True)
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--seed',type=int,default=101);ap.add_argument('--ctx-dir',default=str(OUT));ap.add_argument('--cells',default='core');ap.add_argument('--scales',default='1,4');ap.add_argument('--mode',choices=['save','eval'],default='save')
    ap.add_argument('--tag',default='e30');ap.add_argument('--keep',action='store_true');ap.add_argument('--selfcheck',action='store_true');a=ap.parse_args();OUT.mkdir(parents=True,exist_ok=True)
    assert 'deterministic_ops=true' in os.environ.get('XLA_FLAGS','')
    vp=ROOT/'experiments/exp05_write_dynamics/run3_readouts/verify_packing_fp32.log';base=[l for l in open(vp) if l.startswith('base ')]
    import ast;b=ast.literal_eval(base[0][5:]) if base else {};assert b and max(b['S_maxdiff_R1'],b['S_maxdiff_R0'],b['L_maxdiff_R1'])<=1e-3,'packed fp32 readout not verified - refusing to run'
    cells=C.parse_cells(a.cells);rt=Runtime();L=Ladder30(rt,a.seed,readout=(a.mode=='eval'),save_dir=None)
    if a.mode=='save':
        L.build3();L.setup30(a.ctx_dir,a.seed)
        if a.selfcheck:sys.exit(selfcheck(L,a))
        save_mode(L,a,cells)
    else:eval_mode(L,a)
if __name__=='__main__':main()
