"""E54(b): extended-candidate ranking and candidate-free greedy generation on SAVED W1 states (no training). Re-reads the saved states of e49_candidates.py cells.
 (i) R30 / R_ext: the 30 world places plus N_EXTRA distractor places (default 150) that never occur in the stream: unused combinations of the dev place pool (gen_world PL_PRE x PL_SUF) minus the world's own places (triples, ctrl, pseudo), minus the sealed holdout combinations (frame_pools_hold HOLD_PRE x HOLD_SUF), minus any string that occurs in the realized stream text; drawn stratified by token length to the length distribution of the 30 world places (seed fixed); the shortfall per length is reported (the pool may hold fewer than N_EXTRA of the world's length class). Every candidate is scored as the continuation of every L3 item (summed span log-prob, packed fp32 R1 / R1-alt readouts as e49_candidates.py).
 (ii) greedy generation: after the L3 prefix the model decodes greedily up to 8 tokens (argmax of the packed fp32 readout logits, one packed pass per step, no KV cache); the text is cut at the first newline or period; exact match with the bound target place (world A: the named person's place, world B: the partner's place), generated place inside the 30 stream places, a non-bound stream place, the swap partner's place (the unbound place of the pair) separately.
Usage (GPU): e54b.py --seed 101 --epoch 4 --extra 150 --cell w1_native=L0fs_w1main ... [--w0 --w0-name w1_W0] [--no-r1]   |   --analyze (CPU, re-computes the tables from the result files)   |   --selftest (CPU)
Result files OUT/e54b_s<seed>_<name>.json (restart-safe: an existing file is reused). GPU path has NOT been run when this file was written (only the CPU parts are tested)."""
import os,sys,json,argparse,time,random,re
from pathlib import Path
HERE=Path(__file__).resolve().parent;sys.path.insert(0,str(HERE))
import numpy as np,e49_stats as ES
STEPS=8
def pick_distractors(combos,used,sealed,text,world_places,n,seed,ntok):
    """combos: iterable of candidate place strings (no leading space). Returns (chosen list in draw order, report). Stratified by token length to the length distribution of world_places."""
    wl=[ntok(p) for p in world_places];dist={L:wl.count(L) for L in set(wl)}
    pool=sorted({c for c in combos}-set(used)-set(sealed));ok=[c for c in pool if not re.search(r'\b'+re.escape(c)+r'\b',text)]
    byl={}
    for c in ok:byl.setdefault(ntok(c),[]).append(c)
    rng=random.Random(f'e54b|{seed}');chosen=[];rep={'world_length_counts':dist,'available_by_length':{L:len(v) for L,v in sorted(byl.items())},'excluded_in_stream_text':len(pool)-len(ok),'per_length':{}}
    for L,k in sorted(dist.items()):
        want=round(n*k/len(wl));v=sorted(byl.get(L,[]));rng.shuffle(v);take=v[:want];chosen+=take;rep['per_length'][L]={'wanted':want,'taken':len(take),'shortfall':want-len(take)}
    rep['n_chosen']=len(chosen);rep['shortfall_total']=n-len(chosen);return chosen,rep
def cut_gen(text):
    """generated text cut at the first newline or period, stripped"""
    m=re.search(r'[\n.]',text);return (text[:m.start()] if m else text).strip()
def gen_metrics(texts,places30,gi,fi,world):
    """texts: generated (cut) strings per item. places30: ' '+P strings. -> dict of per-item arrays: exact (bound target), in30, other30 (a stream place that is not the bound target), partner (the unbound place of the pair)"""
    P=[p.strip() for p in places30];S=set(P);tg,ot=(gi,fi) if world=='A' else (fi,gi);n=len(texts)
    ex=np.array([texts[i]==P[tg[i]] for i in range(n)],float);in30=np.array([texts[i] in S for i in range(n)],float);o30=np.array([(texts[i] in S) and texts[i]!=P[tg[i]] for i in range(n)],float);pa=np.array([texts[i]==P[ot[i]] for i in range(n)],float)
    return {'exact':ex,'in30':in30,'other30':o30,'partner':pa}
def metrics_sets(S,gi,fi,tl,n_world,world):
    """S (n,C) scores of one world/readout -> {set name: item_metrics}: R30 (first n_world columns), Rext (all), Rext_samelen (columns with the world places' token length), Rext_pertoken (S / token length)"""
    S=np.asarray(S,float);tl=np.asarray(tl);out={'R30':ES.item_metrics(S[:,:n_world],gi,fi,world),'Rext':ES.item_metrics(S,gi,fi,world)}
    L0=tl[:n_world][0];cols=np.where(tl==L0)[0]
    if (tl[:n_world]!=L0).any():cols=np.arange(S.shape[1])   # mixed-length world: same-length set not defined, falls back to all
    out['Rext_samelen']=ES.item_metrics(S[:,cols],gi,fi,world)    # world places are the first columns, so gi / fi are unchanged
    out['Rext_pertoken']=ES.item_metrics(S/tl[None,:],gi,fi,world);return out
def analyze(recs,ro='alt',nboot=4000):
    L=[f'# E54(b) extended candidates and greedy generation (readout {ro}); item = mean over worlds A,B; CI = bootstrap over {len(recs[0]["pairs"])//2 if recs else 0} pairs','']
    sets=['R30','Rext','Rext_samelen','Rext_pertoken'];ms=['top1','top3','rank','p_norm','lognorm']
    L+=['| cell | candidates (stream / total) | '+' | '.join(f'{s} {m}' for s in sets[:2] for m in ('top1','rank'))+' | Rext_samelen top1 | Rext_pertoken top1 |','|'+'---|'*8]
    gen=[]
    for r in recs:
        gi,fi,tl=np.asarray(r['gi']),np.asarray(r['fi']),r['tl'];nw=r['n_world'];pairs=r['pairs'];vals={}
        for w in 'AB':vals[w]=metrics_sets(r['S'][w][ro],gi,fi,tl,nw,w)
        cellv=lambda s,m:ES.ci(ES.pair_means((vals['A'][s][m]+vals['B'][s][m])/2,pairs),nboot)
        f=lambda t:'%.3f [%.3f, %.3f]'%t if True else ''
        L.append(f"| {r['cell']} | {nw} / {len(tl)} | "+' | '.join(f(cellv(s,m)) for s in sets[:2] for m in ('top1','rank'))+f" | {f(cellv('Rext_samelen','top1'))} | {f(cellv('Rext_pertoken','top1'))} |")
        if r.get('gen') and ro in r['gen']['A']:
            gm={w:gen_metrics(r['gen'][w][ro]['text'],r['places'][:nw],gi,fi,w) for w in 'AB'};gen.append((r['cell'],{k:ES.ci(ES.pair_means((gm['A'][k]+gm['B'][k])/2,pairs),nboot) for k in gm['A']}))
    if gen:
        L+=['','## greedy generation (max 8 tokens, cut at newline / period): share of items','| cell | exact match with the bound target | generated place inside the 30 stream places | a stream place that is not the bound target | the swap partner place |','|---|---|---|---|---|']
        for c,d in gen:L.append(f'| {c} | '+' | '.join('%.3f [%.3f, %.3f]'%d[k] for k in ('exact','in30','other30','partner'))+' |')
    return '\n'.join(L)+'\n'
def selftest():
    rng=np.random.default_rng(0);pre=[f'Ab{a}' for a in 'abcdefghij'];suf=['ford','ton','bury','wick','mouth'];combos=[a+b for a in pre for b in suf]
    ntok=lambda s:1+len(s)//6;world=[combos[i] for i in range(30)];used=set(world[:30]);sealed=set(combos[40:45]);text=' '.join(world)+' '+combos[46]
    ch,rep=pick_distractors(combos,used,sealed,text,world,10,101,ntok);assert len(ch)<=10 and not set(ch)&used and not set(ch)&sealed and combos[46] not in ch,rep
    ch2,_=pick_distractors(combos,used,sealed,text,world,10,101,ntok);assert ch==ch2;print('distractors',rep)
    assert cut_gen(' Alderley.\nNext')=='Alderley' and cut_gen(' Bo\n')=='Bo' and cut_gen(' Cleveland of')=='Cleveland of'
    n=30;places=[' '+p for p in world]+[' '+c for c in ch];C=len(places);tl=[ntok(p.strip()) for p in places]
    gi=np.arange(n);fi=np.array([i^1 for i in range(n)]);pairs=[f'p{i//2}' for i in range(n)]
    def rec(boost):
        S={w:{'alt':rng.normal(-12,1,(n,C))} for w in 'AB'}
        for i in range(n):S['A']['alt'][i,gi[i]]+=boost;S['B']['alt'][i,fi[i]]+=boost
        gen={w:{'alt':{'text':[(places[(gi if w=='A' else fi)[i]].strip() if rng.random()<boost/10 else places[fi[i] if w=='A' else gi[i]].strip()) for i in range(n)]}} for w in 'AB'}
        return {'cell':f'c{boost}','places':places,'n_world':n,'tl':tl,'gi':gi.tolist(),'fi':fi.tolist(),'pairs':pairs,'S':{w:{'alt':S[w]['alt'].tolist()} for w in S},'gen':gen}
    recs=[rec(0),rec(8)];txt=analyze(recs);print(txt)
    g=gen_metrics(recs[1]['gen']['A']['alt']['text'],places[:n],gi,fi,'A');assert g['exact'].mean()>0.5 and (g['exact']+g['other30']<=1).all()
    print('e54b SELFTEST OK')
def gpu_main(a):
    assert 'deterministic_ops=true' in os.environ.get('XLA_FLAGS','')
    import ladder as LD
    jax,Runtime,packing,stage2,reuse=LD.jax,LD.Runtime,LD.packing,LD.stage2,LD.reuse
    jnp=packing.jnp;eqx=packing.eqx
    from ttt.model.data import Batch
    OUT=Path(a.out or LD.OUT);fn=lambda n:OUT/f'e54b_s{a.seed}_{n}.json'
    cells=[c.split('=',1) for c in a.cell];w0n=a.w0_name;names=[n for n,_ in cells]+([w0n] if a.w0 else []);cellpre=dict(cells)
    miss=[(n,w) for n,pre in cells for w in 'AB' if not ((OUT/'states'/f'{pre}_s{a.seed}'/w).is_dir() and (OUT/f'{pre}_s{a.seed}_{w}.saved.json').exists())]
    if miss:
        print('MISSING saved states / logs for',miss,flush=True)
        if a.strict:sys.exit(7)
    todo=[n for n,_ in cells if not any(m[0]==n for m in miss) and not fn(n).exists()]+([w0n] if a.w0 and not fn(w0n).exists() else [])
    print('to read:',todo,'| already done:',[n for n in names if fn(n).exists()],flush=True)
    if todo:
        class ArgPacked(packing.Packed):
            """Packed scorer that returns the argmax token at the scored position instead of a log-prob (greedy decoding step); only the R1 frozen prefix (tail + rows) is built. Same masks / segments as Packed."""
            def __init__(s,T,raw_rows,GP=2):
                rt=T.rt;s.T=T;s.rt=rt;s.GP=GP;W=int(rt.cfg.model.sliding_window_size);CH=1024;TL=len(T.tail_ids)
                s.chunks,s.where=packing.pack_rows(raw_rows,CH);n=len(s.chunks);pad=(-n)%GP
                ids=np.stack([c[0] for c in s.chunks]+[s.chunks[-1][0]]*pad);seg=np.stack([c[1] for c in s.chunks]+[s.chunks[-1][1]]*pad);loc=np.stack([c[2] for c in s.chunks]+[s.chunks[-1][2]]*pad)
                s.ids=jnp.asarray(ids);s.seg=jnp.asarray(seg);s.loc=jnp.asarray(loc)
                @eqx.filter_jit
                def prefix_packed(m,ids_,pos_,mask_):
                    packing.PACK_PRE['mask']=mask_
                    try:
                        seq=Batch(input_ids=ids_,target_tokens=ids_,loss_masks=jnp.ones_like(ids_,dtype=bool),position_ids=pos_)
                        return m.language_model.prefix_call(m.language_model.model.h.prefix_blocks,m.language_model.wte_call(ids_),packing._clone(rt.state_prefix),seq).last_hidden_state
                    finally:packing.PACK_PRE.clear()
                @eqx.filter_jit
                def arg_packed(m,st,ids_,pref_,seg_,loc_):
                    def one(i,p,sg,lc):
                        packing.PACK_SUF['seg']=sg;packing.PACK_SUF['loc']=lc
                        try:
                            r=m.language_model.suffix_call(prefix_outputs=p,state=st,seq=Batch(input_ids=i,target_tokens=i,loss_masks=jnp.ones_like(i,dtype=bool)))
                            return jnp.argmax(r.logits.astype(jnp.float32),axis=-1)
                        finally:packing.PACK_SUF.clear()
                    return jax.vmap(one)(ids_,pref_,seg_,loc_)
                s.arg_packed=arg_packed;p1=[]
                with rt.mesh:
                    for j in range(ids.shape[0]):
                        full=np.concatenate([np.asarray(T.tail_ids,np.int32),ids[j]]);pos=np.concatenate([np.arange(TL,dtype=np.int32),TL+loc[j]]);sf=np.concatenate([np.full(TL,-2,np.int32),seg[j]])
                        p1.append(prefix_packed(rt.base,jnp.asarray(full),jnp.asarray(pos),jnp.asarray(packing.prefix_mask(sf,W)))[TL:])
                    s.pref1=jnp.stack(p1);jax.block_until_ready(s.pref1)
            def next_tokens(s,m):
                rt=s.rt;T=s.T
                with rt.mesh:
                    st=packing._clone(rt.state_suffix)
                    for j in range(len(T.tail_ids)//1024):st,_=rt.chunk_logp(m,st,rt.chunk(T._tail_seq,j),T._tail_pref[j*1024:(j+1)*1024])
                    out=[np.asarray(s.arg_packed(m,st,s.ids[g:g+s.GP],s.pref1[g:g+s.GP],s.seg[g:g+s.GP],s.loc[g:g+s.GP])) for g in range(0,s.ids.shape[0],s.GP)]
                am=np.concatenate(out);return np.asarray([am[c,x] for c,x,y in s.where])
        import frame_pools_hold as H
        rt=Runtime();L=LD.Ladder(rt,a.seed,readout=True);T=L.T;w_=T.w;gw=sys.modules['gen_world'] if 'gen_world' in sys.modules else __import__('gen_world')
        places30,gi,fi,rows30,pairs=ES.candidate_items(L.fam['L3'],w_['triples']);n=len(pairs);nw=len(places30)
        ntok=lambda s:len(T.tok.encode(s,add_special_tokens=False).ids)
        used=[t['P'] for t in w_['triples']]+list(w_.get('ctrl_places',[]))+list(w_.get('pseudo_places',[]))
        text=gw.realize(w_,False)[0] if hasattr(gw,'realize') else ''
        dis,drep=pick_distractors([x+y for x in gw.PL_PRE for y in gw.PL_SUF],used,[x+y for x in H.HOLD_PRE for y in H.HOLD_SUF],text,[t['P'] for t in w_['triples']],a.extra,a.seed,lambda s:ntok(' '+s))
        print('distractors:',json.dumps(drep),flush=True)
        if a.min_extra and len(dis)<a.min_extra:raise SystemExit(f'only {len(dis)} distractors available (< --min-extra {a.min_extra}); see the report above')
        places=places30+[' '+d for d in dis];tl=[ntok(p) for p in places];C=len(places)
        slot=stage2.make_slot(T.tok,None);raw=[];
        for it in L.fam['L3']:
            for c in places:
                ids,span=stage2.item_ids(T.tok,slot,it['prefix'],c);raw.append((ids,(span[0]-1,span[1]-1)))
        pre_ids=[]
        for it in L.fam['L3']:
            e=stage2.enc(T.tok,'\n\n'+it['prefix']).ids;ids,span=stage2.item_ids(T.tok,slot,it['prefix'],places30[0]);assert ids[:span[0]]==list(slot)+e,'prefix tokenization differs between the candidate rows and the generation prompt'
            pre_ids.append(list(slot)+e)
        print('items',n,'candidates',C,'(stream',nw,') rows',len(raw),flush=True)
        with L.rt32.mesh,jax.default_matmul_precision('highest'):
            Pa=packing.Packed(L.ctx_alt,raw,GP=2);Pr=None if a.no_r1 else packing.Packed(L.ctx,raw,GP=2)
        def read(m):
            m32=L.to_fp32(m);d={}
            with jax.default_matmul_precision('highest'):
                d['alt']=np.asarray(Pa.r1(m32)).reshape(n,C)
                if Pr is not None:d['R1']=np.asarray(Pr.r1(m32)).reshape(n,C)
            return d
        def greedy(m,ctx):
            m32=L.to_fp32(m);gens=[[] for _ in pre_ids]
            for step in range(STEPS):
                rows=[(ids+g+[0],(len(ids)+len(g)-1,len(ids)+len(g))) for ids,g in zip(pre_ids,gens)]
                with L.rt32.mesh,jax.default_matmul_precision('highest'):
                    P=ArgPacked(ctx,rows,GP=2);nxt=P.next_tokens(m32)
                for g,t in zip(gens,nxt):g.append(int(t))
                if all(re.search(r'[\n.]',T.tok.decode(g)) for g in gens):break
            return {'ids':gens,'text':[cut_gen(T.tok.decode(g)) for g in gens]}
        t0=time.monotonic()
        for name in todo:
            S={};G={}
            if name==w0n:
                m=rt.base;r=read(m);g={'alt':greedy(m,L.ctx_alt)}
                S={'A':r,'B':r};G={'A':g,'B':g}
            else:
                pre=cellpre[name]
                for w in 'AB':
                    log=json.load(open(OUT/f'{pre}_s{a.seed}_{w}.saved.json'))['log'];ent=[e for e in log if isinstance(e.get('margins'),dict) and '_file' in e['margins'] and e.get('epoch')==a.epoch];assert len(ent)==1,f'{name} world {w}: epoch {a.epoch} state not found'
                    m=L.load_state(ent[0]['margins']['_file']);assert reuse._hash_pytree(m.inner_parameters())[0]==ent[0]['margins']['_hash'],'loaded weights differ from the streamed weights'
                    S[w]=read(m);G[w]={'alt':greedy(m,L.ctx_alt)};print(name,w,'read + generated',f'{time.monotonic()-t0:.0f}s',flush=True)
            json.dump({'cell':name,'places':places,'n_world':nw,'tl':tl,'gi':gi.tolist(),'fi':fi.tolist(),'pairs':list(pairs),'S':{w:{ro:np.asarray(v).tolist() for ro,v in d.items()} for w,d in S.items()},'gen':G,'distractor_report':drep,'epoch':a.epoch if name!=w0n else 0,'sec_total':time.monotonic()-t0},open(fn(name),'w'))
    return OUT,fn,names
def count_main(a):
    sys.path.insert(0,str(HERE.parents[0]/'exp04_what_ttt_retains'));import gen_world as gw,frame_pools_hold as H
    w=gw.build_world(a.seed);ntok=lambda s:len(gw.TOK.encode(' '+s,add_special_tokens=False).ids)
    used=[t['P'] for t in w['triples']]+list(w.get('ctrl_places',[]))+list(w.get('pseudo_places',[]))
    dis,rep=pick_distractors([x+y for x in gw.PL_PRE for y in gw.PL_SUF],used,[x+y for x in H.HOLD_PRE for y in H.HOLD_SUF],gw.realize(w,False)[0],[t['P'] for t in w['triples']],a.extra,a.seed,ntok)
    print('E48_FRAMES=',os.environ.get('E48_FRAMES'),'extra wanted',a.extra);print(json.dumps(rep,indent=1))
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--seed',type=int,default=101);ap.add_argument('--epoch',type=int,default=4);ap.add_argument('--extra',type=int,default=120,help='number of distractor places (default 120); the shortfall of the world length class is reported');ap.add_argument('--min-extra',type=int,default=0,help='abort if fewer distractors are available')
    ap.add_argument('--cell',action='append',default=[]);ap.add_argument('--w0',action='store_true');ap.add_argument('--w0-name',default='W0');ap.add_argument('--strict',action='store_true');ap.add_argument('--no-r1',action='store_true',help='skip the R1 (pg1342 tail) candidate readout, alt only (about half the scoring time); generation always uses the alt readout');ap.add_argument('--analyze',action='store_true');ap.add_argument('--selftest',action='store_true');ap.add_argument('--count',action='store_true',help='CPU only (needs tokenizers + gen_world, no model): print the distractor pool report (available places per token length, shortfall) and exit');ap.add_argument('--out',default=None);ap.add_argument('--tag',default='');ap.add_argument('--nboot',type=int,default=4000)
    a=ap.parse_args()
    if a.selftest:selftest();return
    if a.count:count_main(a);return
    if a.analyze:
        OUT=Path(a.out or os.environ.get('EXP07_OUT',HERE.parents[1]/'experiments/exp07_write_ladder'));fn=lambda n:OUT/f'e54b_s{a.seed}_{n}.json';names=[n for n,_ in (c.split('=',1) for c in a.cell)]+([a.w0_name] if a.w0 else [])
    else:OUT,fn,names=gpu_main(a)
    recs=[json.load(open(fn(n))) for n in names if fn(n).exists()]
    if not recs:print('no result files');return
    txt=analyze(recs,'alt',a.nboot)
    if any('R1' in r['S']['A'] for r in recs):txt+='\n'+analyze(recs,'R1',a.nboot)
    print(txt);(OUT/f'e54b_report_s{a.seed}{a.tag}.md').write_text(txt)
if __name__=='__main__':main()
