"""E35 L1 -> native ladder (new file; ladder.py is NOT modified, the server copy has uncommitted edits). NOT RUN: written without GPU access from origin/autosnap/a100.
Levels (--level), all on the world-seed stream, A and B worlds, readout after every epoch/pass (same Ladder.evaluate as L1/L0, so D_alt/R1 come from analyze_ladder.py):
  L1fillfresh   R1: isolated exposure sentences + the epoch's background paragraphs as further isolated rows (each <= 1024 tokens, loss on every token), reshuffled each epoch.
                Epoch 1 uses the world's own background, epoch e >= 2 a FRESH background from a different book than every earlier epoch (same background as native pass e below).
  L1smallbatch  R1b: the plain L1 rows (no background) but ~64 SGD steps per epoch (ceil(n_rows/64) sentences per step), so the step count matches R1.
  L0reset       R2..R5: the native stream of each pass (pass 1 = the original stream, pass p >= 2 = same exposure sentences at the same schedule positions with the fresh background
                of pass p), read with the context reset every --reset-period P chunks: at each block start the suffix KV state is new and the prefix layers see only the block's tokens
                (rt.seq/rt.prefix on the block, positions restart); weights, optimizer and the step counter carry on. P >= 63 = one block = the native read (R5, native-fs-fresh).
Fresh background (E31 correction): books other than the world's own background book and the readout/canary books (pg11, pg1342), one book per pass (order: seeded shuffle of the
candidates), paragraph rule as gen_world.background; audit before any GPU work: pairwise character 8-gram overlap between the passes' background texts (fraction of the smaller set) <= 0.05,
zero exact paragraph matches with the canary books and with the readout tails. Aborts if the audit fails or there are not enough books for the requested passes.
Logging: every SGD step: grad_norm, clip_coef, update_norm, cum_update_norm, n_loss (L1: always; L0reset: with --log-clip, one extra forward/backward per chunk); per epoch: the standard evaluate() margins
(incl. both canaries), background book and paragraph hash, inner weight hash after the pass.
Modes: --mode save|eval|full as ladder.py (the 64K stream + update + fp32 readout can OOM together: save first, then eval). --cpu-checks: no GPU (U1 audit, U2 f(P) table, U3 text-level stream check).
--compare-hash PATH (U4, with --level L0reset --reset-period 63 --passes 1 --mode save): prints whether the pass-1 weight hash equals the entry of an existing L0fs saved.json (PATH, pass 1).
Usage: e35_ladder.py --cpu-checks --world-seed 101
       e35_ladder.py --level L0reset --reset-period 63 --passes 4 --world-seed 101 --tag fresh_P63 --mode save [--log-clip]   (then --mode eval)
       e35_ladder.py --level L1fillfresh --epochs 4 --world-seed 101 --tag fillfresh --mode save"""
import os,sys,json,time,argparse,hashlib,random,re,copy
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
import ladder as LD
from ladder import *
sys.path.insert(0,str(LD.ROOT/'scripts/exp04_what_ttt_retains'))
import s3_sets as SETS
DATA=LD.ROOT/'data/narrative_gutenberg';OUT=LD.OUT;EXCL={'pg11.txt','pg1342.txt'};MAX_OVERLAP=0.05
norm=lambda s:' '.join(s.split())
def book_paras(fname,rng):
    """gen_world.background for ONE named book (same cleaning, name scrubbing and shuffle), no book choice."""
    t=(DATA/fname).read_text(errors='ignore');t=t[t.find('CHAPTER'):] if 'CHAPTER' in t else t
    t=t[len(t)//5:];t=re.sub(r'\r','',t);e=t.find('*** END');t=t[:e] if e>0 else t
    paras=[p for p in re.split(r'\n\s*\n',t) if 200<len(p)<900 and not re.search(r'CHAPTER|\bVol|\[|\]|\*|Gutenberg|indemnif|royalt|copyright|trademark|licen[sc]e|donat',p,re.I)];out=[]
    for p in paras:
        p=re.sub(r'(?<=[a-z,;] )(?:[A-Z][a-z]+)(?: [A-Z][a-z]+)*','someone',p)
        p=re.sub(r"\b(?:Mr|Mrs|Miss|Dr|Sir|Lady|Lord)\.? \w+",'the stranger',p);out.append(' '.join(p.split()))
    rng.shuffle(out);return out
_CB={}
def candidate_books(own_bg,seed):
    key=(hashlib.sha1('|'.join(own_bg).encode()).hexdigest(),seed)
    if key not in _CB:_CB[key]=_candidate_books(own_bg,seed)
    return _CB[key]
def _candidate_books(own_bg,seed):
    """books usable as fresh background: not the canary books, not the world's own background book (detected by exact paragraph overlap), in a seeded order."""
    own=set(own_bg);c=[]
    for f in sorted(p.name for p in DATA.glob('pg*.txt') if p.name not in EXCL):
        paras=book_paras(f,random.Random(0))
        if len(paras)>=100 and len(set(paras)&own)<=MAX_OVERLAP*len(own):c.append(f)
    random.Random(f'e35|{seed}').shuffle(c);return c
def fresh_pool(own_bg,passno,seed,passes,nslots,margin=200):
    """pass p>=2 gets 2-3 books (as many as needed for nslots+margin paragraphs); books are disjoint across passes (candidates in seeded order, taken in sequence). Aborts if the candidates run out."""
    c=candidate_books(own_bg,seed);k=0;seen=set(own_bg)
    for ps in range(2,passno+1):
        got=[];tot=0
        while tot<nslots+margin:
            assert k<len(c),f'ran out of fresh books at pass {ps} ({len(c)} candidates)'
            f=c[k];k+=1;pa=[x for x in book_paras(f,random.Random(f'e35|{seed}|{f}')) if x not in seen];pa=list(dict.fromkeys(pa));seen.update(pa);got.append((f,pa));tot+=len(pa)
    pool=[x for f,pa in got for x in pa];return pool,[f for f,_ in got],c
def baseline_overlap(c):
    """word 8-gram overlap between unrelated book pairs (the 6 last candidates), same formula as audit()."""
    out=[]
    for i in (1,3,5):
        if len(c)<i+1:break
        g=[grams(norm(' '.join(book_paras(f,random.Random(0))))) for f in (c[-i],c[-i-1])];out.append(len(g[0]&g[1])/max(1,min(len(g[0]),len(g[1]))))
    return out
def grams(s,n=8):
    w=s.split();return {tuple(w[i:i+n]) for i in range(len(w)-n+1)}
def canary_texts():
    out={}
    for f in ('pg11.txt','pg1342.txt'):
        t=(DATA/f).read_text(errors='ignore');t=t[t.find('CHAPTER'):] if 'CHAPTER' in t else t;out[f]=norm(t.replace('\r',''))
    return out
def build_fresh_worlds(w0,passes):
    """text-level worlds per pass (pass 1 = w0): the exposure texts stay in place, every 'bg' text is replaced by the pass's fresh pool paragraph (cycling if short), as ladder.fs_streams."""
    seed=w0['seed'];own=[t['text'] for t in w0['texts'] if t['kind']=='bg'];worlds=[w0];meta=[{'pass':1,'book':'world'}]
    for ps in range(2,passes+1):
        pool,book,c=fresh_pool(own,ps,seed,passes,len(own));w=copy.deepcopy(w0);bi=0;new=[]
        for t in w['texts']:
            if t['kind']=='bg':new.append({'kind':'bg','text':pool[bi]});bi+=1
            else:new.append(t)
        w['texts']=new;worlds.append((w,pool));meta.append({'pass':ps,'books':book,'pool_size':len(pool)})
    meta[0]['baseline_overlap']=baseline_overlap(c) if passes>1 else []
    return worlds,meta
def pad_to_tokens(w,pool,swap,enc_ids,bos,nt):
    """extend a world's text with further pool paragraphs until the stream has nt tokens (as fs_streams does)."""
    bi=sum(1 for t in w['texts'] if t['kind']=='bg');ids=[bos]+enc_ids(gw.realize(w,swap)[0])
    while len(ids)<nt:
        assert bi<len(pool),'fresh pool exhausted while padding the stream: abort';w['texts'].append({'kind':'bg','text':pool[bi]});bi+=1;ids=[bos]+enc_ids(gw.realize(w,swap)[0])
    return ids[:nt]
def audit(bg_by_pass,tails,base=None):
    """bg_by_pass: {pass: [bg texts]}; returns the report and a pass flag."""
    G={p:grams(norm(' '.join(v))) for p,v in bg_by_pass.items()};rep={'pair_overlap':{},'dup_within_pass':{},'canary_exact':{},'canary_8gram':{}};ok=True
    ps=sorted(G);thr=(max(base)+0.02) if base else MAX_OVERLAP;rep['threshold']=thr;rep['baseline']=base
    for i in ps:
        for j in ps:
            if i<j:o=len(G[i]&G[j])/max(1,min(len(G[i]),len(G[j])));rep['pair_overlap'][f'{i}-{j}']=o;ok&=o<=thr
    cg={k:grams(v) for k,v in tails.items()}
    for p,v in bg_by_pass.items():
        rep['dup_within_pass'][p]=len(v)-len(set(v));ok&=(rep['dup_within_pass'][p]==0 or p==1);ex=sum(1 for x in set(v) if any(x in t for t in tails.values()));rep['canary_exact'][p]=ex;ok&=(ex==0 or p==1)
        rep['canary_8gram'][p]={k:len(G[p]&g)/max(1,len(G[p])) for k,g in cg.items()}
    return rep,bool(ok)
def f_table(w,swap,tok,Ps=(1,4,8,64)):
    """U2: share of exposure sentences (with an earlier same-triple exposure) whose context contains a same-triple earlier sentence / another triple of the same pair, as a function of the reset period P (chunks).
    context of the place token x in chunk k: positions >= max(block_start, x - 8192) (suffix and prefix layers both look back at most 8192)."""
    import suppress as SP
    tab=SETS.sentence_table(w,swap,tok,gw,SP);out={}
    for P in Ps:
        n=0;same=0;oth=0
        for s in tab:
            x=s['place'][0];b0=(x//CH//P)*P*CH;lo=max(b0,x-8192)
            prior=[u for u in tab if u['t']==s['t'] and u['round']<s['round']]
            if not prior:continue
            n+=1;same+=any(lo<=u['place'][0]<x for u in prior)
            oth+=any(u['pair']==s['pair'] and u['t']!=s['t'] and lo<=u['place'][0]<x for u in tab)
        out[P]={'n_sentences_with_prior':n,'f_same_triple':same/max(n,1),'f_other_triple_same_pair':oth/max(n,1)}
    return out
class L35(LD.Ladder):
    def fresh_streams(self,swap,passes):
        w0=self.T.w;enc=lambda t:stage2.enc(self.T.tok,t).ids;ws,meta=build_fresh_worlds(w0,passes);streams=[self.T.P[swap].ids];worlds=[w0];self._bgmeta=meta
        for (w,pool),m in zip(ws[1:],meta[1:]):
            ids=pad_to_tokens(w,pool,swap,enc,self.rt.bos,NCH*CH);streams.append(ids);worlds.append(w);m['bg_sha']=hashlib.sha1('|'.join(t['text'] for t in w['texts'] if t['kind']=='bg').encode()).hexdigest()[:12]
        meta[0]['bg_sha']=hashlib.sha1('|'.join(t['text'] for t in w0['texts'] if t['kind']=='bg').encode()).hexdigest()[:12]
        _,_,spans=gw.realize(w0,swap)
        for ps,ids in enumerate(streams,1):
            assert len(ids)>=NCH*CH;txt=self.T.tok.decode(list(ids[:NCH*CH]));miss=[s_['text'] for s_ in spans if s_['text'] not in txt];assert not miss,(ps,len(miss),miss[:2])
        rep,ok=audit({i+1:[t['text'] for t in w['texts'] if t['kind']=='bg'] for i,w in enumerate(worlds)},canary_texts(),self._bgmeta[0].get('baseline_overlap'));print('fresh background audit',json.dumps(rep)[:600],'PASS' if ok else 'FAIL',flush=True)
        assert ok,'background audit failed'
        self._fw=worlds;self._audit=rep;return streams
    def run_L1_fresh(self,swap,epochs,seed=0,variant='fillfresh',save_epochs=(),save_dir=None):
        rt=self.rt;W=int(rt.cfg.model.sliding_window_size);plain=self.l1_rows(swap,'plain');rng=random.Random(seed);log=[];m=rt.base;st_all=clone_state(rt.state_all)
        if variant=='fillfresh':self.fresh_streams(swap,epochs);enc=lambda t:stage2.enc(self.T.tok,t).ids
        with rt.mesh:
            log.append({'epoch':0,'step':0,'margins':self.evaluate(m)});nstep=0;cum=0.0
            for ep in range(1,epochs+1):
                rows=list(plain)
                if variant=='fillfresh':
                    for t in self._fw[ep-1]['texts']:
                        if t['kind']=='bg':x=enc('\n\n'+t['text']);rows.append((x[:1024],(0,min(len(x),1024)-1)))
                order=list(range(len(rows)));rng.shuffle(order);rr=[rows[i] for i in order];groups=[]
                if variant=='smallbatch':
                    n_per=-(-len(rr)//64);groups=[rr[i:i+n_per] for i in range(0,len(rr),n_per)]
                else:groups=[rr]
                chunk_sets=[]
                for g_ in groups:
                    chunks,where=packing.pack_rows(g_);mks=[np.zeros(1024,bool) for _ in chunks]
                    for c_,a_,b_ in where:mks[c_][a_:b_]=True
                    chunk_sets+=list(zip(chunks,mks))
                for (ids,seg,loc),mk in chunk_sets:
                    tg=np.concatenate([ids[1:],[rt.eos]]).astype(np.int32)
                    pref=self.prefix_iso(rt.base,jnp.asarray(ids),jnp.asarray(loc),jnp.asarray(packing.prefix_mask(seg,W)))
                    newp,gn,coef,ce=self.step(m,st_all,jnp.asarray(ids),jnp.asarray(tg),jnp.asarray(mk),pref,jnp.asarray(seg),jnp.asarray(loc))
                    old=m;m=self.set_inner(m,newp);nstep+=1
                    d=sum(float(jnp.sum((a.astype(jnp.float32)-b.astype(jnp.float32))**2)) for a,b in zip(jax.tree.leaves(m.inner_parameters()),jax.tree.leaves(old.inner_parameters())))**0.5;cum+=d
                    log.append({'epoch':ep,'step':nstep,'grad_norm':float(gn),'clip_coef':float(coef),'ce':float(ce),'update_norm':d,'cum_update_norm':cum,'n_loss':int(mk.sum())})
                e={'epoch':ep,'step':nstep,'n_steps_epoch':len(chunk_sets),'cum_update_norm':cum,'margins':self.evaluate(m)}
                if variant=='fillfresh':e['bg']=self._bgmeta[ep-1]
                log.append(e);print(f'  L1{variant} {"B" if swap else "A"} epoch {ep} steps {nstep} ({len(chunk_sets)} this epoch) ce {log[-2]["ce"]:.3f}',flush=True)
        return log
    def run_L0_reset(self,swap,passes,P,log_clip=False):
        rt=self.rt;streams=self.fresh_streams(swap,passes);log=[];m=rt.base;opt=m.inner_optimizer(clone_state(rt.state_all)).init(m.inner_parameters());st_all=clone_state(rt.state_all);cum=0.0
        with rt.mesh:
            log.append({'epoch':0,'margins':self.evaluate(m)})
            for ps,ids in enumerate(streams,1):
                ids=list(ids[:NCH*CH]);steps=[];tp=time.monotonic()
                for b0 in range(0,NCH,P):
                    bl=ids[b0*CH:min(NCH,b0+P)*CH];n=len(bl);pad=(-n)%CH;full=bl+[rt.eos]*pad;mask=np.zeros(len(full),dtype=bool);mask[:n-1]=True
                    seq=rt.seq(full,mask=mask);pref=rt.prefix(rt.base,seq);st=(st_all,clone_state(rt.state_suffix))   # suffix KV and prefix context start fresh in every block
                    for k in range(len(full)//CH):
                        c=rt.chunk(seq,k);p_=pref[k*CH:(k+1)*CH];row={'chunk':b0+k}
                        if log_clip:gn_=float(self.gnorm_only(m,st[1],c,p_));row.update(grad_norm=gn_,clip_coef=min(1.0,self.clip/max(gn_,1e-30)))
                        old=m;r=rt.update(m,opt,st,c,p_);m,opt,st=r.new_model,r.new_optimizer_state,r.new_state;del r
                        d=sum(float(jnp.sum((a.astype(jnp.float32)-b.astype(jnp.float32))**2)) for a,b in zip(jax.tree.leaves(m.inner_parameters()),jax.tree.leaves(old.inner_parameters())))**0.5;cum+=d
                        row.update(update_norm=d,cum_update_norm=cum,n_loss=int(np.asarray(c.loss_masks).sum()));steps.append(row)
                    st_all=st[0];del pref
                e={'epoch':ps,'reset_period':P,'sec_pass':time.monotonic()-tp,'sec_per_chunk':(time.monotonic()-tp)/NCH,'cum_update_norm':cum,'steps':steps,'bg':self._bgmeta[ps-1],'inner_hash':reuse._hash_pytree(m.inner_parameters())[0],'margins':self.evaluate(m)}
                log.append(e);print(f'  L0reset P={P} {"B" if swap else "A"} pass {ps} cum {cum:.3f} {e["sec_pass"]:.0f}s ({e["sec_per_chunk"]:.2f}s/chunk, before evaluate)',flush=True)
        return log
def cpu_checks(seed):
    import gen_world as G;w=G.build_world(seed);res={'seed':seed}
    worlds,meta=build_fresh_worlds(w,4);bgs={1:[t['text'] for t in w['texts'] if t['kind']=='bg']}
    for i,(ww,pool) in enumerate(worlds[1:],2):bgs[i]=[t['text'] for t in ww['texts'] if t['kind']=='bg']
    rep,ok=audit(bgs,canary_texts(),meta[0].get('baseline_overlap'));res['U1_audit']=rep;res['U1_pass']=ok;res['books']=meta
    # U3 (text level): the exposure texts and their order are identical in every pass; only the bg texts differ
    base=[(t['t'],t['round'],t['text']) for t in w['texts'] if t['kind']=='expo']
    res['U3_expo_identical']=all([(t['t'],t['round'],t['text']) for t in ww['texts'] if t['kind']=='expo']==base for ww,_ in worlds[1:])
    res['U3_kind_sequence_identical']=all([t['kind'] for t in ww['texts']][:len(w['texts'])]==[t['kind'] for t in w['texts']] for ww,_ in worlds[1:])
    res['U3_bg_differs']=all(bgs[p]!=bgs[1] for p in bgs if p>1)
    res['U2_f_table']={'A':f_table(w,False,G.TOK)};json.dump(res,open(OUT/f'e35_cpu_checks_s{seed}.json','w'),indent=1,default=str)
    md=[f'# E35 CPU checks (seed {seed})\n',f'U1 audit pass: {ok}; books: {meta}\n',f"pair overlap: {rep['pair_overlap']}\ncanary exact: {rep['canary_exact']}  canary 8-gram: {rep['canary_8gram']}\n",f"U3 expo identical: {res['U3_expo_identical']}, bg differs: {res['U3_bg_differs']}\n",'## U2 f(P) (same-triple earlier exposure inside the reset context)\n','| P (chunks) | n | f same triple | f other triple, same pair |','|---|---|---|---|']
    for P,v in res['U2_f_table']['A'].items():md.append(f"| {P} | {v['n_sentences_with_prior']} | {v['f_same_triple']:.3f} | {v['f_other_triple_same_pair']:.3f} |")
    (OUT/f'e35_cpu_checks_s{seed}.md').write_text('\n'.join(md)+'\n');print('\n'.join(md))
def u56(L,world_seed,ks=(1,5,20)):
    """U5: dtypes of the inner parameters after one update by the native path (rt.update, block = one chunk, as L0reset P=1) and by the L1/masked-fs path (L.step);
    U6: the same chunk written both ways from W0 (R2 = L0reset P=1 vs masked-fs block=1024 of ladder.run_L0_masked): update difference. Description, not pass/fail."""
    rt=L.rt;W=int(rt.cfg.model.sliding_window_size);ids=np.asarray(L.T.P[False].ids[:NCH*CH]);m0=rt.base;st_all=clone_state(rt.state_all)
    seg=(np.arange(CH)//CH).astype(np.int32);loc=(np.arange(CH)%CH).astype(np.int32);pm=jnp.asarray(packing.prefix_mask(seg,W));samenext=np.concatenate([seg[1:]==seg[:-1],[False]])
    f32=lambda t:[np.asarray(x,np.float32).ravel() for x in jax.tree.leaves(t)];p0=jax.tree.leaves(m0.inner_parameters());res={'chunks':{},'dtype_W0':sorted({str(np.asarray(x).dtype) for x in p0})}
    with rt.mesh:
        for k in ks:
            c=ids[k*CH:(k+1)*CH];full=[int(t) for t in c];mask=np.zeros(CH,dtype=bool);mask[:CH-1]=True
            seq=rt.seq(full,mask=mask);pref=rt.prefix(rt.base,seq);opt=m0.inner_optimizer(clone_state(rt.state_all)).init(m0.inner_parameters());st=(st_all,clone_state(rt.state_suffix))
            r=rt.update(m0,opt,st,rt.chunk(seq,0),pref[:CH]);A=r.new_model.inner_parameters();del r
            tg=np.concatenate([ids[k*CH+1:(k+1)*CH+1],[rt.eos]])[:CH].astype(np.int32);pr2=L.prefix_iso(rt.base,jnp.asarray(c),jnp.asarray(loc),pm)
            B,gn,coef,ce=L.step(m0,st_all,jnp.asarray(c),jnp.asarray(tg),jnp.asarray(samenext),pr2,jnp.asarray(seg),jnp.asarray(loc),jnp.float32(1.0))
            a_,b_,z=f32(A),f32(B),f32(m0.inner_parameters());dA=np.concatenate([x-y for x,y in zip(a_,z)]);dB=np.concatenate([x-y for x,y in zip(b_,z)])
            na,nb=float(np.linalg.norm(dA)),float(np.linalg.norm(dB));cos=float(dA@dB/max(na*nb,1e-30))
            la,lb=jax.tree.leaves(A),jax.tree.leaves(B)
            res['chunks'][k]={'dtype_rt_update':sorted({str(np.asarray(x).dtype) for x in la}),'dtype_step':sorted({str(np.asarray(x).dtype) for x in lb}),'update_norm_rt_update':na,'update_norm_step':nb,'cosine':cos,
                'norm_of_difference_over_norm_rt_update':float(np.linalg.norm(dA-dB)/max(na,1e-30)),'leaves_bitwise_equal':sum(bool(np.array_equal(np.asarray(x),np.asarray(y))) for x,y in zip(la,lb)),'n_leaves':len(la),'step_grad_norm':float(gn),'step_clip_coef':float(coef)}
            print('U5/U6 chunk',k,res['chunks'][k],flush=True);del A,B
    res['U6_note']=('R2 (L0reset P=1) and masked-fs (run_L0_masked block=1024) both read each chunk alone (fresh suffix KV, prefix recomputed on the chunk, positions restart, last-token loss masked). '
        'Differences: R2 uses rt.update (optimizer path, rt.prefix, rt.seq targets) vs masked-fs L.step (packed isolated-row prefix_iso, explicit clip/scale, params cast back to the parameter dtype); '
        'R2 resets once per block and carries the optimizer state, masked-fs has no optimizer state; masked-fs L0mwfs streams pass-wise fresh filler like R2. The numbers above are the one-step update difference from W0.')
    json.dump(res,open(OUT/f'e35_u56_s{world_seed}.json','w'),indent=1);print('wrote',OUT/f'e35_u56_s{world_seed}.json')
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--level',choices=['L1fillfresh','L1smallbatch','L0reset']);ap.add_argument('--reset-period',type=int,default=64);ap.add_argument('--passes',type=int,default=4);ap.add_argument('--epochs',type=int,default=4)
    ap.add_argument('--worlds',default='AB');ap.add_argument('--world-seed',type=int,default=101);ap.add_argument('--seed',type=int,default=0);ap.add_argument('--tag',default='e35');ap.add_argument('--mode',choices=['full','save','eval'],default='full')
    ap.add_argument('--keep',action='store_true');ap.add_argument('--log-clip',action='store_true');ap.add_argument('--cpu-checks',action='store_true');ap.add_argument('--u56',action='store_true');ap.add_argument('--compare-hash',default=None);a=ap.parse_args();OUT.mkdir(parents=True,exist_ok=True)
    if a.cpu_checks:cpu_checks(a.world_seed);return
    if a.u56:
        assert 'deterministic_ops=true' in os.environ.get('XLA_FLAGS','');rt=Runtime();u56(L35(rt,a.world_seed,readout=False),a.world_seed);return
    assert a.level,'--level required';assert 'deterministic_ops=true' in os.environ.get('XLA_FLAGS','')
    vp=LD.ROOT/'experiments/exp05_write_dynamics/run3_readouts/verify_packing_fp32.log';base=[l for l in open(vp) if l.startswith('base ')]
    import ast;b=ast.literal_eval(base[0][5:]) if base else {};assert b and max(b['S_maxdiff_R1'],b['S_maxdiff_R0'],b['L_maxdiff_R1'])<=1e-3,'packed fp32 readout not verified - refusing to run'
    rt=Runtime();lev=a.level+(f'_P{a.reset_period}' if a.level=='L0reset' else '');SD=OUT/'states'/f'{lev}_{a.tag}_s{a.world_seed}'
    L=L35(rt,a.world_seed,readout=(a.mode!='save'),save_dir=SD);L.log_clip=a.log_clip
    json.dump({'items':L.items,'fam_sizes':{k:len(v) for k,v in L.fam.items()},'lr':L.lr,'clip':L.clip,'level':lev,'epochs':a.epochs,'passes':a.passes,'seed':a.seed,'xla_flags':os.environ.get('XLA_FLAGS')},open(OUT/f'meta_{lev}_{a.tag}_s{a.world_seed}.json','w'),indent=1)
    for w in a.worlds:
        f=OUT/f'{lev}_{a.tag}_s{a.world_seed}_{w}.json';fs=OUT/f'{lev}_{a.tag}_s{a.world_seed}_{w}.saved.json'
        if f.exists():print('exists',f);continue
        if a.mode=='eval':
            d=json.load(open(fs));t0=time.monotonic()
            for e in d['log']:
                if isinstance(e.get('margins'),dict) and '_file' in e['margins']:
                    fn=e['margins']['_file'];h=e['margins']['_hash'];mm=L.load_state(fn);assert reuse._hash_pytree(mm.inner_parameters())[0]==h,'loaded weights differ from the streamed weights'
                    e['margins']=L.evaluate(mm);e['hash']=h
                    if not a.keep:os.remove(fn)
            d['sec_eval']=time.monotonic()-t0;json.dump(d,open(f,'w'));print('scored',f,round(d['sec_eval']),'s',flush=True);continue
        if a.mode=='save':
            L.save_dir=SD/w;L._nsave=0;(SD/w).mkdir(parents=True,exist_ok=True)
            if fs.exists():print('exists',fs);continue
        t0=time.monotonic()
        if a.level=='L0reset':log=L.run_L0_reset(w=='B',a.passes,a.reset_period,a.log_clip)
        else:log=L.run_L1_fresh(w=='B',a.epochs,a.seed,'fillfresh' if a.level=='L1fillfresh' else 'smallbatch',save_dir=SD/w)
        json.dump({'log':log,'sec':time.monotonic()-t0,'level':lev},open(fs if a.mode=='save' else f,'w'));print('saved',fs if a.mode=='save' else f,round(time.monotonic()-t0),'s',flush=True)
        if a.compare_hash:
            ref=json.load(open(a.compare_hash))['log'];rh=[e['margins']['_hash'] for e in ref if isinstance(e.get('margins'),dict) and '_hash' in e['margins']]
            mine=[e['margins']['_hash'] for e in log if isinstance(e.get('margins'),dict) and '_hash' in e['margins']]
            print('U4 pass-1 weight hash equal to the reference L0fs pass 1:',mine[1]==rh[1],mine[1],rh[1] if len(rh)>1 else None,flush=True)
if __name__=='__main__':main()
