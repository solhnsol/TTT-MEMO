"""Exp05-A run3: readout comparison + null calibration on the run2 (deterministic) seed-101 streams.
Re-streams World A/B with the identical native update (hashes are checked against run2) and, on every state, reads the
L3 relation margin and two null item families with three readouts:
  R0 short   : 1024-token chunk, RoPE from 0, no context, zero suffix KV (= Exp05-A run2 probe)
  R1 neutral : PRIMARY. fixed held-out neutral tail (BOS + 7167 tokens of a book NOT used in any stream) streamed normally
               through prefix AND suffix (no update) with the current weights, then the probe chunk; identical in every
               state and world, so only the weights differ
  R2 exp04   : Exp04 protocol (stream tail 7168 + probe; prefix sees the tail, suffix KV empty / carried), checkpoints only
Item families: L3 (real, 15 pairs), XP (cross-pair null: name of pair p, places of pair p+1), PS (>=15 pseudo-pairs of
never-exposed names and places). Optional --perturb EPS: fixed seeded random perturbation of relative size EPS added to
the chunk-0 update (sign-stability replicate). e2e/, Exp04 and run2 artifacts are not modified."""
import os,sys,json,time,argparse,hashlib,random,re
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'scripts/exp05_write_dynamics'))
from trajectory import *      # Traj, Runtime, CH, NCH, jax, jnp, eqx, np, clone_state, reuse, stage2, gw, V3_K, save
OUT=Path(os.environ.get('EXP05_OUT',ROOT/'experiments/exp05_write_dynamics/run3_readouts'))
RUN2=ROOT/'experiments/exp05_write_dynamics/run2_detops'
TAIL_BOOK='pg1342.txt'           # neutral tail source; must not be one of gen_world.background() books
TAIL_LEN=8192-CH                 # 7168 tokens incl. BOS
N_PSEUDO=16
FAMILIARITY=False                # set by --familiarity
G=16                             # probe rows per batched call (fixed -> single compilation)

def neutral_tail(tok,bos):
    f=ROOT/'data/narrative_gutenberg'/TAIL_BOOK;raw=f.read_bytes();t=raw.decode(errors='ignore')
    t=t[t.find('CHAPTER'):] if 'CHAPTER' in t else t;t=t[len(t)//3:].replace('\r','')
    paras=[p for p in re.split(r'\n\s*\n',t) if 200<len(p)<900 and not re.search(r'CHAPTER|\bVol|\[|\]|\*',p)]
    out=[]
    for p in paras:   # same name scrubbing as gen_world.background()
        p=re.sub(r'(?<=[a-z,;] )(?:[A-Z][a-z]+)(?: [A-Z][a-z]+)*','someone',p)
        p=re.sub(r"\b(?:Mr|Mrs|Miss|Dr|Sir|Lady|Lord)\.? \w+",'the stranger',p)
        out.append(' '.join(p.split()))
    text='\n\n'.join(out);ids=[bos]+stage2.enc(tok,text).ids
    assert len(ids)>=TAIL_LEN
    return ids[:TAIL_LEN],text,hashlib.sha256(raw).hexdigest()

def pseudo_names(world,n,tok,forbid_text):
    """n never-exposed person names and n place names, same token length as the world's, absent from every text."""
    rng=random.Random(world['seed']*1000+7)
    used={t['X'] for t in world['triples']}|{t['P'] for t in world['triples']}|set(world['ctrl_persons'])|set(world['ctrl_places'])
    def draw(gen,L):
        out=[]
        while len(out)<n:
            s=gen()
            if s in used or s in out or gw.ntok(s)!=L or re.search(r'\b'+s+r'\b',forbid_text):continue
            out.append(s)
        return out
    pp=draw(lambda:rng.choice(gw.SYL1)+rng.choice(gw.SYL2),world['name_len'])
    pl=draw(lambda:rng.choice(gw.PL_PRE)+rng.choice(gw.PL_SUF),world['place_len'])
    return pp,pl

class Readouts(Traj):
    def __init__(self,rt,seed):
        super().__init__(rt,seed)
        w=self.w;P=self.P[False];name=P.name   # canonical (world-A) names, as in Exp04/Exp05 probes
        self.tail_ids,tail_text,self.tail_sha=neutral_tail(self.tok,rt.bos)
        streams=gw.realize(w,False)[0]+gw.realize(w,True)[0]
        for t in w['triples']:assert not re.search(r'\b'+t['X']+r'\b|\b'+t['P']+r'\b',tail_text),'world name inside neutral tail'
        tr={t['id']:t for t in w['triples']};npairs=w['pairs']
        xp=[]
        for p in range(npairs):
            q=(p+1)%npairs
            for j in (0,1):
                a=2*p+j
                xp.append({'level':'XP','pair':f'x{p}','tid':a,'prefix':gw.PR_HOME.format(X=name[a]),'gold':' '+tr[2*q+j]['P'],'foil':' '+tr[2*q+1-j]['P']})
        pp,pl=pseudo_names(w,2*N_PSEUDO,self.tok,streams+'\n'+tail_text);self.pseudo={'persons':pp,'places':pl}
        ps=[]
        for p in range(N_PSEUDO):
            for x,y in ((2*p,2*p+1),(2*p+1,2*p)):
                ps.append({'level':'PS','pair':f's{p}','tid':200+x,'prefix':gw.PR_HOME.format(X=pp[x]),'gold':' '+pl[x],'foil':' '+pl[y]})
        self.fam={'L3':self.l3,'XP':xp,'PS':ps}
        self.n_r0_rows=2*sum(len(v) for v in self.fam.values())   # R0 scores only the binding families
        if FAMILIARITY:   # weight-stored familiarity (per world; never-exposed foils) + never-exposed nulls; R1 (and R2) only
            l1=gw.probes(w,name)
            self.fam['FN']=[it for it in l1 if it['level']=='L1p'];self.fam['FP']=[it for it in l1 if it['level']=='L1l']
            cp,cl=w['ctrl_persons'],w['ctrl_places']
            self.fam['FN0']=[{'level':'FN0','pair':f's{x//2}','tid':300+x,'prefix':gw.PR_FREQ_P,'gold':' '+pp[x],'foil':' '+cp[x%len(cp)]} for x in range(2*N_PSEUDO)]
            self.fam['FP0']=[{'level':'FP0','pair':f's{x//2}','tid':400+x,'prefix':gw.PR_FREQ_L,'gold':' '+pl[x],'foil':' '+cl[x%len(cl)]} for x in range(2*N_PSEUDO)]
        self.items=[it for v in self.fam.values() for it in v]
        # Probe rows: one per (item, candidate). The probe chunk tokens are identical for R0 and R1; only the frozen-prefix
        # outputs differ (R1's prefix sees the neutral tail). Prefix blocks are not inner parameters, so prefix outputs are
        # weight-independent: computed ONCE here, never per state.
        ids_l,spans=[],[]
        for it in self.items:
            for c in (it['gold'],it['foil']):
                ids,span=stage2.item_ids(self.tok,stage2.make_slot(self.tok,None),it['prefix'],c);assert len(ids)<=CH
                ids_l.append(ids+[rt.eos]*(CH-len(ids)));spans.append((span[0]-1,span[1]-1))
        self.spans=spans;nrow=len(ids_l);self.nrow=nrow
        pad=(-nrow)%G;ids_l=ids_l+[ids_l[-1]]*pad          # fixed group size G -> one compiled shape
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
    def _score_rows(self,m,st,pref,nrows=None):
        """probe rows [0,nrows) in groups of G through ONE compiled function; returns per-item margins (gold - foil)."""
        n=self.nrow if nrows is None else nrows;lps=[]
        for g in range(0,n,G):
            lps.append(np.asarray(self.lp_batch(m,st,self.b_ids[g:g+G],self.b_tg[g:g+G],self.b_mk[g:g+G],pref[g:g+G])))
        lp=np.concatenate(lps)[:n];sc=np.asarray([lp[i,a:b].sum() for i,(a,b) in enumerate(self.spans[:n])])
        return sc[0::2]-sc[1::2]
    def probe_short_b(self,m):
        """R0, batched: zero suffix KV, no context, RoPE from 0 (same inputs as Traj.probe_short)."""
        with self.rt.mesh:return self._score_rows(m,clone_state(self.rt.state_suffix),self.pref_short,self.n_r0_rows)
    def probe_neutral(self,m):
        """R1: stream the neutral tail through the suffix with weights m (no update), then score every row from that KV."""
        rt=self.rt
        with rt.mesh:
            st=clone_state(rt.state_suffix)
            for j in range(TAIL_LEN//CH):
                st,_=rt.chunk_logp(m,st,rt.chunk(self._tail_seq,j),self._tail_pref[j*CH:(j+1)*CH])
            return self._score_rows(m,st,self.pref_neu)
    def by_family(self,mg):
        res={};i=0
        for f,its in self.fam.items():
            if i>=len(mg):break
            res[f]=self.pair_margin(its,mg[i:i+len(its)]);i+=len(its)
        return res
    def perturbation(self,m,eps,seed=12345):
        """fixed seeded Gaussian direction over inner leaves, scaled to eps * ||Delta_0||; returned as a pytree."""
        f=reuse.get_filter_spec(m,self.rt.cfg.training.spec_inner,'inner parameters')
        inner=eqx.partition(m,f)[0];key=jax.random.PRNGKey(seed);leaves,td=jax.tree_util.tree_flatten(inner)
        keys=jax.random.split(key,len(leaves))
        r=[jax.random.normal(k,x.shape,jnp.float32) if eqx.is_inexact_array(x) else None for k,x in zip(keys,leaves)]
        return f,td,r
    def run(self,swap,probe_states,perturb=0.0,do_short=True):
        rt=self.rt;P=self.P[swap];tag='B' if swap else 'A'
        ids=P.ids;mask=np.zeros(NCH*CH,dtype=bool);mask[:len(ids)-1]=True;seq=rt.seq(ids,mask=mask)
        res={'world':tag,'short':{},'neutral':{},'exp04':{},'hashes':[],'rows':[],'perturb':perturb}
        with rt.mesh:
            pref=rt.prefix(rt.base,seq);jax.block_until_ready(pref)
            m=rt.base;opt=m.inner_optimizer(clone_state(rt.state_all)).init(m.inner_parameters());state_tuple=(clone_state(rt.state_all),clone_state(rt.state_suffix))
            def probes(j,m_,st_):
                t0=time.monotonic()
                if j in probe_states:
                    res['neutral'][j]=self.by_family(self.probe_neutral(m_))
                    if do_short:res['short'][j]=self.by_family(self.probe_short_b(m_))
                k=j-1
                if k in V3_K:
                    res['exp04'][k]={mode:self.by_family(np.asarray([g-fo for g,fo in self.probe_exp04(swap,m_,st_,k,self.items,mode)])) for mode in ('empty','carry')}
                return time.monotonic()-t0
            res['hashes'].append(reuse._hash_pytree(m.inner_parameters())[0]);probes(0,m,state_tuple[1])
            for k in range(NCH):
                c=rt.chunk(seq,k);p=pref[k*CH:(k+1)*CH];t0=time.monotonic()
                r=rt.update(m,opt,state_tuple,c,p);jax.block_until_ready(r.metrics)
                m,opt,state_tuple=r.new_model,r.new_optimizer_state,r.new_state;del r
                row={'chunk':k}
                if k==0 and perturb>0:
                    f,td,rnd=self.perturbation(m,perturb)
                    inner,outer=eqx.partition(m,f);base_inner=eqx.partition(rt.base,f)[0]
                    L,td2=jax.tree_util.tree_flatten(inner);B0=jax.tree_util.tree_leaves(base_inner);assert td2==td
                    dn=float(np.sqrt(sum(float(jnp.sum((a.astype(jnp.float32)-b.astype(jnp.float32))**2)) for a,b in zip(L,B0) if eqx.is_inexact_array(a))))
                    rn=float(np.sqrt(sum(float(jnp.sum(x*x)) for x in rnd if x is not None)))
                    new=[(x.astype(jnp.float32)+perturb*dn/rn*rr).astype(x.dtype) if rr is not None else x for x,rr in zip(L,rnd)]
                    m=eqx.combine(jax.tree_util.tree_unflatten(td,new),outer);row.update(delta0_norm=dn,perturb_norm=perturb*dn)
                res['hashes'].append(reuse._hash_pytree(m.inner_parameters())[0])
                row['sec_update']=time.monotonic()-t0;row['sec_probe']=probes(k+1,m,state_tuple[1]);res['rows'].append(row)
                print(f'  [{tag}] chunk {k} upd {row["sec_update"]:.1f}s probe {row["sec_probe"]:.1f}s',flush=True)
        return res

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--seed',type=int,default=101);ap.add_argument('--worlds',default='AB');ap.add_argument('--tag',default='main')
    ap.add_argument('--states',default='all',help="'all' or comma list of states j (= chunks consumed)");ap.add_argument('--perturb',type=float,default=0.0)
    ap.add_argument('--no-short',action='store_true');ap.add_argument('--familiarity',action='store_true');ap.add_argument('--dry',action='store_true',help='build items/tail, print, no streaming')
    a=ap.parse_args();OUT.mkdir(parents=True,exist_ok=True)
    assert 'deterministic_ops=true' in os.environ.get('XLA_FLAGS',''),'run3 requires deterministic XLA ops (same as run2)'
    global FAMILIARITY;FAMILIARITY=a.familiarity
    rt=Runtime();T=Readouts(rt,a.seed)
    states=set(range(NCH+1)) if a.states=='all' else {int(x) for x in a.states.split(',')}
    meta={'tail_book':TAIL_BOOK,'tail_sha256':T.tail_sha,'tail_len':TAIL_LEN,'pseudo':T.pseudo,'n_items':{k:len(v) for k,v in T.fam.items()},
          'items':T.items,'states':sorted(states),'perturb':a.perturb,'xla_flags':os.environ.get('XLA_FLAGS',''),'jax':jax.__version__}
    json.dump(meta,open(OUT/f'meta_seed{a.seed}_{a.tag}.json','w'),indent=1);print('items',meta['n_items'],'tail sha',T.tail_sha[:12],flush=True)
    if a.dry:   # smoke: batched R0 vs the run2 unbatched path on the base weights, R1 once, compile counts
        t0=time.monotonic();b0=T.probe_short_b(rt.base);t1=time.monotonic();u0=T.probe_short(rt.base,T.items[:12]);t2=time.monotonic()
        r1=T.probe_neutral(rt.base);t3=time.monotonic();r1b=T.probe_neutral(rt.base);t4=time.monotonic()
        print('R0 batched vs unbatched max diff (12 items)',float(np.abs(b0[:12]-u0).max()),'R1 repeat diff',float(np.abs(r1-r1b).max()),flush=True)
        print(f'times: R0 batched {t1-t0:.1f}s (incl. compile), R1 first {t3-t2:.1f}s, R1 second {t4-t3:.1f}s',flush=True)
        try:print('lp_batch compilations',T.lp_batch._cached._cache_size() if hasattr(T.lp_batch,'_cached') else 'n/a')
        except Exception as e:print('cache size n/a',e)
        return
    for w in a.worlds:
        f=OUT/f'readouts_seed{a.seed}_{a.tag}_{w}.json'
        if f.exists():print('exists',f);continue
        t0=time.monotonic();res=T.run(w=='B',states,a.perturb,not a.no_short);res['total_sec']=time.monotonic()-t0
        # gate: identical native stream as run2 (unperturbed runs only)
        r2=RUN2/f'traj_seed{a.seed}_full_{w}.json'
        if a.perturb==0 and r2.exists():
            h2=json.load(open(r2))['hashes'];res['hash_match_run2']=h2==res['hashes']
            if not res['hash_match_run2']:json.dump(res,open(f,'w'),default=lambda o:o.tolist() if hasattr(o,'tolist') else str(o));print('HASH MISMATCH vs run2',flush=True);sys.exit(2)
        json.dump(res,open(f,'w'),default=lambda o:o.tolist() if hasattr(o,'tolist') else str(o));print('saved',f,round(res['total_sec']),'s',flush=True)
if __name__=='__main__':main()
