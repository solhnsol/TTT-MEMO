"""E33 R runner: the exp07 write ladder (ladder.py, server version) on a natural-text world (data/natural-text/worlds/nat_<stream>_<layout>.json).
Loader switch: install() patches gen_world (build_world / realize / probes / frames) and swaps stage2.Probe and readouts.Readouts for natural-text versions, so the
unchanged Ladder code (fp32 packed R1 + R1-alt, canary, native / masked / L1 writes, fs_streams interface) runs on the natural world.
 * stream      = BOS + the realized stream truncated to NCH*CH tokens (the world is built longer; asserts it is long enough and keeps every exposure unit).
 * families    = NW.build_items (L3, XP, PS, FN, FP, FN0, FP0, L3b, L3e) + L3r, L2 appended by Ladder; item order is saved in meta (items) = order of the margin lists.
 * fs_streams  = pass 1 stream; pass p >= 2: the same texts with the background texts replaced IN ORDER by the pass pool of nat_filler.select (other natural streams' background, filtered,
                 no exact paragraph shared between passes, no cycling); padded with further pool paragraphs to NCH*CH tokens; every exposure unit present; the filler audit is re-run here
                 and must pass. The exposure positions shift between passes (different filler lengths): s_cross / s_same / n_out are recounted per pass from real token offsets and saved.
 * L1 rows     = NW.l1_rows_text ('doc': each exposure unit alone, truncated to 1024 tokens).
 * readout     = the margins of every item at W0 (epoch / pass 0) and after each pass are saved per item (person level: item tid / pair in meta['items']).
Levels (cells): L0fs (native, filler swap), L0mwfs (chunk alone, block 1024, scale 1), L1 (offline, epochs). Modes: save (stream writes, state files) then eval (fp32 readout), as ladder.py.
Usage: nat_ladder.py --nat 9105:spread --nat-dir DIR --cells L0fs,L0mwfs,L1 --passes 3 --epochs 8 --worlds A --mode save|eval --tag nat [--keep]
NOT run on the cloud (needs the ttt package, GPU and the 3B checkpoint); the CPU parts are in nat_pass.py / nat_filler.py (selftests)."""
import os,sys,json,time,argparse,random,copy,ast
from pathlib import Path
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
import nat_world as NW,nat_filler as NF,nat_pass as NP
import ladder as LD
from ladder import *
from ttt.model.data import Batch
NAT={}

def install(nat_dir,stream,layout='spread'):
    """loader switch: call BEFORE constructing the Ladder."""
    w=NW.load_world(NF.world_path(nat_dir,stream,layout))
    for k in ('ctrl_persons','ctrl_places','pseudo_persons','pseudo_places'):assert all(w[k]),(k,'empty entry: the ladder families need a full list')
    NAT.update(world=w,stream=stream,layout=layout,nat_dir=str(nat_dir))
    gw.build_world=lambda seed,v=2:w
    gw.realize=NW.realize;gw.probes=NW.probes
    gw.PR_HOME=w['frames']['birth'];gw.PR_FREQ_P=w['frames']['freq_p'];gw.PR_FREQ_L=w['frames']['freq_l']
    stage2.Probe=NatProbe;RD.Readouts=NatReadouts
    orig=LD.squad_tail
    def squad_tail_nat(tok,bos,forbid):
        ids=orig(tok,bos,sorted(set(forbid)|set(NW.forbid_strings(w))));assert len(ids)>=8192-CH,f'SQuAD tail too short after the forbid filter: {len(ids)}';return ids
    LD.squad_tail=squad_tail_nat
    return w

class NatProbe:
    """stand-in for stage2.Probe: .ids (BOS + NCH*CH-1 tokens), .name (canonical world-A names), .items (NW.probes), .spans with token positions."""
    def __init__(self,rt,world,swap,probe_world=None):
        self.rt=rt;self.tok=rt.tok;self.swap=swap;self.world=world
        text,self.stream_name,spans=NW.realize(world,swap);_,self.name,_=NW.realize(world,False)
        e=stage2.enc(self.tok,text);ids=[rt.bos]+e.ids;assert len(ids)>=NCH*CH,('natural stream shorter than NCH*CH tokens',len(ids),NCH*CH)
        self.n_total=len(ids);self.ids=ids[:NCH*CH];self.spans=NP.tok_spans(spans,[b for a,b in e.offsets])
        late=[s for s in self.spans if s['tok_end']>=NCH*CH];assert not late,f'{len(late)} exposure units beyond token {NCH*CH}'
        self.items=NW.probes(world,self.name);self.tri={t['id']:t for t in world['triples']}
        self.strata=NP.strata_real(self.spans);self.audit={'n_total_tokens':self.n_total,'n_expo':len(self.spans)}

class NatReadouts(RD.Readouts):
    """Readouts on the natural world: same probe rows / R1 neutral tail machinery as exp05 readouts.Readouts, but the item families are NW.build_items."""
    def __init__(self,rt,seed):
        RD.Traj.__init__(self,rt,seed)
        w=self.w;P=self.P[False];G=RD.G;TAIL_LEN=RD.TAIL_LEN
        self.tail_ids,tail_text,self.tail_sha=RD.neutral_tail(self.tok,rt.bos);self.tail_report=NP.tail_check(w,tail_text)
        self.fam={k:v for k,v in NW.build_items(w).items() if k not in ('L3r','L2')}   # Ladder appends L3r, L2 from gw.probes
        assert list(self.fam)[:3]==['L3','XP','PS']
        self.n_r0_rows=2*sum(len(self.fam[k]) for k in ('L3','XP','PS'))
        self.items=[it for v in self.fam.values() for it in v]
        ids_l,spans=[],[]
        for it in self.items:
            for c in (it['gold'],it['foil']):
                ids,span=stage2.item_ids(self.tok,stage2.make_slot(self.tok,None),it['prefix'],c);assert len(ids)<=CH
                ids_l.append(ids+[rt.eos]*(CH-len(ids)));spans.append((span[0]-1,span[1]-1))
        self.spans=spans;nrow=len(ids_l);self.nrow=nrow
        pad=(-nrow)%G;ids_l=ids_l+[ids_l[-1]]*pad
        with rt.mesh:
            seqs=[rt.seq(x) for x in ids_l]
            self.b_ids=jnp.stack([q.input_ids for q in seqs]);self.b_tg=jnp.stack([q.target_tokens for q in seqs]);self.b_mk=jnp.stack([q.loss_masks for q in seqs])
            self.pref_short=jnp.stack([rt.prefix(rt.base,q) for q in seqs])
            pn=[];tseq=None
            for x in ids_l:
                q=rt.seq(self.tail_ids+x);pf=rt.prefix(rt.base,q);pn.append(pf[TAIL_LEN:])
                if tseq is None:tseq=q;self._tail_pref=pf[:TAIL_LEN]
            self.pref_neu=jnp.stack(pn);self._tail_seq=tseq;jax.block_until_ready(self.pref_neu)
        @eqx.filter_jit
        def lp_batch(m,st,ids,tg,mk,pref):
            def one(i,t,k,p):
                r=m.language_model.suffix_call(prefix_outputs=p,state=st,seq=Batch(input_ids=i,target_tokens=t,loss_masks=k))
                lg=r.logits.astype(jnp.float32)
                return jnp.take_along_axis(lg,t[:,None],axis=-1)[:,0]-jax.nn.logsumexp(lg,axis=-1)
            return jax.vmap(one)(ids,tg,mk,pref)
        self.lp_batch=lp_batch

class NatLadder(LD.Ladder):
    def l1_rows(self,swap,variant):
        assert variant in ('doc','fact'),variant
        rows=[]
        for t in NW.l1_rows_text(self.T.w,swap,variant):
            x=stage2.enc(self.T.tok,t).ids[:CH];rows.append((x,(0,len(x)-1)))
        assert rows,'no L1 rows';return rows
    def _pools(self,passes):
        if getattr(self,'_pool_cache',(None,))[0]!=passes:
            canary=NF.canary_grams();assert canary,'pg11 / pg1342 not found: the canary overlap check cannot run'
            pools,info=NF.select(NAT['nat_dir'],NAT['stream'],passes,NAT['layout'],canary=canary)
            rep=NF.audit(pools,NF.bg_texts(self.T.w),canary);hits=NF.forbid_hits(pools,self.T.w)
            assert rep['pass'] and all(v==0 for v in hits.values()),('filler audit failed',rep,hits)
            self._pool_cache=(passes,pools,info,rep)
        return self._pool_cache[1:]
    def fs_streams(self,swap,passes):
        """pass 1 = the original stream; pass p >= 2 = same exposure units at the SAME token positions, background = pass pool cut to each slot's token length (nat_pass.build_pass_stream_matched).
        Per pass: the filler audit is re-run on the texts actually used (after the cuts), s_cross / s_same / n_out must equal pass 1 for every person. Saves self.pass_info[swap]."""
        pools,info,rep=self._pools(passes);Pw=self.T.P[swap];w0=self.T.w;tok=self.T.tok;bos=self.rt.bos
        def encode(text):
            e=stage2.enc(tok,text);return e.ids,[b for a,b in e.offsets]
        decode=lambda ids:tok.decode(list(ids))
        streams=[Pw.ids];self._fs_worlds={swap:[w0]};used_all={}
        pi=[{'pass':1,'n_tokens_total':Pw.n_total,'strata':Pw.strata,'expo_start':[s['tok_start'] for s in Pw.spans],'build':None}]
        for ps in range(2,passes+1):
            w,ids,sp,binfo,used=NP.build_pass_stream_matched(w0,pools[ps],swap,encode,decode,bos,NCH*CH)
            assert list(ids[:NCH*CH])!=list(streams[0]),ps
            st=NP.strata_real(sp);same=all((st[t]['s_cross'],st[t]['s_same'],st[t]['n_out'])==(Pw.strata[t]['s_cross'],Pw.strata[t]['s_same'],Pw.strata[t]['n_out']) for t in Pw.strata)
            assert same,f'pass {ps}: s_cross / s_same / n_out differ from pass 1'
            streams.append(ids[:NCH*CH]);self._fs_worlds[swap].append(w);used_all[ps]=used
            pi.append({'pass':ps,'n_tokens_total':len(ids),'strata':st,'expo_start':[s['tok_start'] for s in sp],'build':binfo})
        rep2=NF.audit(used_all,NF.bg_texts(w0),NF.canary_grams());hits=NF.forbid_hits(used_all,w0)
        assert rep2['pass'] and all(v==0 for v in hits.values()),('filler audit on the used (cut) texts failed',rep2,hits)
        self.pass_info=getattr(self,'pass_info',{});self.pass_info[swap]=pi;self.used_audit=rep2
        print('fs_streams check passed:',passes,'passes,',len(Pw.spans),'exposure units each at the same positions (max drift',max([p['build']['max_abs_drift'] for p in pi[1:]] or [0]),'tokens); audit on used texts',rep2['pass'],flush=True)
        return streams
    def run_L1(self,swap,epochs,seed=0,eval_epochs=(1,2,3,8),save_epochs=(),save_dir=None,variant='doc'):
        """ladder.run_L1 with evaluation at the listed epochs (+ epoch 0) instead of every k-th; update code unchanged."""
        rt=self.rt;W=int(rt.cfg.model.sliding_window_size);rows=self.l1_rows(swap,variant);rng=random.Random(seed);log=[]
        m=rt.base;st_all=clone_state(rt.state_all)
        with rt.mesh:
            log.append({'epoch':0,'step':0,'margins':self.evaluate(m)})
            nstep=0;cum=0.0
            for ep in range(1,epochs+1):
                order=list(range(len(rows)));rng.shuffle(order)
                chunks,where=packing.pack_rows([rows[i] for i in order]);mks=[np.zeros(1024,bool) for _ in chunks]
                for c_,a_,b_ in where:mks[c_][a_:b_]=True
                for (ids,seg,loc),mk in zip(chunks,mks):
                    tg=np.concatenate([ids[1:],[rt.eos]]).astype(np.int32)
                    pref=self.prefix_iso(rt.base,jnp.asarray(ids),jnp.asarray(loc),jnp.asarray(packing.prefix_mask(seg,W)))
                    newp,gn,coef,ce=self.step(m,st_all,jnp.asarray(ids),jnp.asarray(tg),jnp.asarray(mk),pref,jnp.asarray(seg),jnp.asarray(loc))
                    old=m;m=self.set_inner(m,newp);nstep+=1
                    d=sum(float(jnp.sum((a.astype(jnp.float32)-b.astype(jnp.float32))**2)) for a,b in zip(jax.tree.leaves(m.inner_parameters()),jax.tree.leaves(old.inner_parameters())));cum+=d**0.5
                    log.append({'epoch':ep,'step':nstep,'grad_norm':float(gn),'clip_coef':float(coef),'ce':float(ce),'update_norm':d**0.5,'cum_update_norm':cum})
                log.append({'epoch':ep,'n_chunks':len(chunks),'n_rows':len(rows)})
                if ep in eval_epochs or ep==epochs:log.append({'epoch':ep,'step':nstep,'margins':self.evaluate(m)});print(f'  L1 {"B" if swap else "A"} epoch {ep} steps {nstep} ({len(chunks)} steps/epoch) ce {log[-3]["ce"]:.3f}',flush=True)
        return log
    def run_L0_masked(self,swap,passes,block=1024,scale=1.0,streams=None):
        """ladder.run_L0_masked with the per-chunk gradient norm and clip coefficient logged ('clip' in each pass entry); update path unchanged."""
        rt=self.rt;W=int(rt.cfg.model.sliding_window_size);Pw=self.T.P[swap];log=[];m=rt.base;st_all=clone_state(rt.state_all);cum=0.0
        if streams is None:streams=[Pw.ids]*passes
        assert len(streams)==passes
        seg=(np.arange(CH)//block).astype(np.int32);loc=(np.arange(CH)%block).astype(np.int32);pm=jnp.asarray(packing.prefix_mask(seg,W))
        samenext=np.concatenate([seg[1:]==seg[:-1],[False]])
        with rt.mesh:
            log.append({'epoch':0,'margins':self.evaluate(m)})
            for ps in range(1,passes+1):
                ids=np.asarray(streams[ps-1]);clips=[]
                for k in range(NCH):
                    c=ids[k*CH:(k+1)*CH];tg=np.concatenate([ids[k*CH+1:(k+1)*CH+1],[rt.eos]])[:CH].astype(np.int32)
                    mk=samenext.copy()
                    pref=self.prefix_iso(rt.base,jnp.asarray(c),jnp.asarray(loc),pm)
                    newp,gn,coef,ce=self.step(m,st_all,jnp.asarray(c),jnp.asarray(tg),jnp.asarray(mk),pref,jnp.asarray(seg),jnp.asarray(loc),jnp.float32(scale))
                    clips.append((float(gn),float(coef)))
                    old=m;m=self.set_inner(m,newp)
                    cum+=sum(float(jnp.sum((a.astype(jnp.float32)-b.astype(jnp.float32))**2)) for a,b in zip(jax.tree.leaves(m.inner_parameters()),jax.tree.leaves(old.inner_parameters())))**0.5
                log.append({'epoch':ps,'cum_update_norm':cum,'clip':clips,'margins':self.evaluate(m)});print(f'  L0mw b{block} x{scale} {"B" if swap else "A"} pass {ps}',flush=True)
        return log

def run_cell(L,cell,w,a):
    swap=w=='B'
    if cell=='L1':return L.run_L1(swap,a.epochs,a.seed,eval_epochs=tuple(int(x) for x in a.l1_eval_epochs.split(',')),variant=a.l1_variant)
    if cell=='L0fs':return L.run_L0_fillerswap(swap,a.passes,None)
    if cell=='L0mwfs':return L.run_L0_masked(swap,a.passes,a.block,a.scale,streams=L.fs_streams(swap,a.passes))
    raise SystemExit(f'unknown cell {cell}')

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--nat',default='9105:spread');ap.add_argument('--nat-dir',default=os.environ.get('NAT_DIR','data/natural-text/worlds'))
    ap.add_argument('--cells',default='L0fs,L0mwfs,L1');ap.add_argument('--passes',type=int,default=3);ap.add_argument('--epochs',type=int,default=8);ap.add_argument('--l1-variant',default='doc',choices=['doc','fact'])
    ap.add_argument('--l1-eval-epochs',default='1,2,3,8');ap.add_argument('--log-clip',action='store_true',help='native cells: per-chunk gradient norm / clip coefficient (one extra forward/backward per chunk)');ap.add_argument('--worlds',default='A');ap.add_argument('--seed',type=int,default=0);ap.add_argument('--tag',default='nat')
    ap.add_argument('--mode',choices=['save','eval'],required=True);ap.add_argument('--keep',action='store_true');ap.add_argument('--block',type=int,default=1024);ap.add_argument('--scale',type=float,default=1.0)
    ap.add_argument('--strata',default=None,help='strata.json for the s_cross comparison (default: nat-dir/../strata.json if present)')
    a=ap.parse_args();OUT.mkdir(parents=True,exist_ok=True)
    assert 'deterministic_ops=true' in os.environ.get('XLA_FLAGS','')
    vp=ROOT/'experiments/exp05_write_dynamics/run3_readouts/verify_packing_fp32.log'
    base=[l for l in open(vp) if l.startswith('base ')];b=ast.literal_eval(base[0][5:]) if base else {}
    assert b and max(b['S_maxdiff_R1'],b['S_maxdiff_R0'],b['L_maxdiff_R1'])<=1e-3,'packed fp32 readout not verified - refusing to run'
    stream,layout=a.nat.split(':');stream=int(stream);install(a.nat_dir,stream,layout)
    rt=Runtime();L=NatLadder(rt,stream,readout=(a.mode!='save'),save_dir=None);L.log_clip=a.log_clip
    mf=OUT/f'meta_nat_{a.tag}_s{stream}.json'
    meta={'items':L.items,'fam_sizes':{k:len(v) for k,v in L.fam.items()},'lr':L.lr,'clip':L.clip,'cells':a.cells,'passes':a.passes,'epochs':a.epochs,'l1_variant':a.l1_variant,'l1_eval_epochs':a.l1_eval_epochs,
          'stream':stream,'layout':layout,'xla_flags':os.environ.get('XLA_FLAGS'),'tail_report':L.T.tail_report,'tail_sha':L.T.tail_sha,'n_stream_tokens':{ 'A':L.T.P[False].n_total,'B':L.T.P[True].n_total}}
    sp=Path(a.strata) if a.strata else Path(a.nat_dir).parent/'strata.json'
    if sp.exists():meta['strata_diff_pass1']=NP.strata_diff(L.T.P[False].strata,json.load(open(sp)),stream,layout)
    json.dump(meta,open(mf,'w'),indent=1,default=str)
    cache={}
    for cell in a.cells.split(','):
        SD=OUT/'states'/f'{cell}_{a.tag}_s{stream}'
        for w in a.worlds:
            f=OUT/f'{cell}_{a.tag}_s{stream}_{w}.json';fs=OUT/f'{cell}_{a.tag}_s{stream}_{w}.saved.json'
            if f.exists():print('exists',f);continue
            if a.mode=='eval':
                d=json.load(open(fs));t0=time.monotonic()
                for e in d['log']:
                    if 'margins' in e and '_file' in e['margins']:
                        fn=e['margins']['_file'];h=e['margins']['_hash'];mm=L.load_state(fn)
                        assert reuse._hash_pytree(mm.inner_parameters())[0]==h,'loaded weights differ from the streamed weights'
                        if h in cache:e['margins']=cache[h]    # identical weights (W0 of every cell / world): the readout is deterministic, evaluate once
                        else:e['margins']=cache[h]=L.evaluate(mm)
                        e['hash']=h
                        if not a.keep:os.remove(fn)
                d['sec_eval']=time.monotonic()-t0;json.dump(d,open(f,'w'));print('scored',f,round(d['sec_eval']),'s',flush=True);continue
            L.save_dir=SD/w;L._nsave=0;(SD/w).mkdir(parents=True,exist_ok=True)
            if fs.exists():print('exists',fs);continue
            t0=time.monotonic();log=run_cell(L,cell,w,a)
            extra={'pass_info':{str(k):v for k,v in getattr(L,'pass_info',{}).items()}} if cell!='L1' else {}
            json.dump({'log':log,'sec':time.monotonic()-t0,**extra},open(fs,'w'),default=str);print('saved',fs,round(time.monotonic()-t0),'s',flush=True)
if __name__=='__main__':main()
