"""Label-free chunk-selection signal, exact version: window context gain per chunk of the world-A (and optionally B) stream.
For every chunk k of the synthetic stream (world seed s, 63 chunks):
  nll_ctx(k)   mean NLL over the CH-1 targets of the chunk read natively (prefix over the whole stream, suffix state carried through chunks 0..k-1, 8K sliding window as in every cell) under the weights m
  nll_noctx(k) mean NLL over the same CH-1 targets of the chunk alone (fresh suffix state, prefix of the chunk alone; the e37_firstchunk_nll.py read)
  delta(k)     = nll_ctx - nll_noctx  (window context gain as defined by the request: mean NLL with the window minus mean NLL without; negative = the window helps)
Weights m: W0 always; saved states optional (--state LABEL=PATH, or --state-template 'states/L0fs_fskeep_s{seed}/{w}/state_{p:03d}.npz' --state-passes 1,2,3,4). The prefix layers are always W0 (frozen); the carried suffix state is advanced with the same m.
O chunks = chunks whose oracle set (s3_sets.Sets.oracle, name/place tokens of prior exposures of the same triple inside the window) is non-empty; non-O chunks = chunks AFTER the first O chunk with an empty oracle set.
AUC = P(score_O > score_nonO) with score = -delta (larger context gain = more O-like), ties 0.5, plus 2000 chunk bootstrap CI; both are reported per (seed, world, weights). 0.5 = no separation.
Output OUT/ctx_gain_{tag}.json: per seed/world/weights per-chunk values + O labels + n_O; and a summary .md.
Usage (GPU): ctx_gain.py --seeds 101,102,103 --worlds A [--state ...] [--tag ctx]    CPU (server, needs `tokenizers` + gen_world): ctx_gain.py --dry --seeds 101,102,103 [--tokenizer-file F]    |    --selftest"""
import os,sys,json,time,argparse
from pathlib import Path
import numpy as np
HERE=Path(__file__).resolve().parent;sys.path.insert(0,str(HERE))
CH=1024;NCH=63
def auc(score,labels,boot=2000,seed=0):
    """Mann-Whitney AUC of score for label==1 vs 0 (ties 0.5) with a bootstrap CI over chunks; labels in {0,1}, chunks selected by the caller"""
    s=np.asarray(score,float);y=np.asarray(labels,int);p=s[y==1];n=s[y==0]
    if len(p)==0 or len(n)==0:return float('nan'),float('nan'),float('nan')
    f=lambda a,b:float(((a[:,None]>b[None,:]).sum()+0.5*(a[:,None]==b[None,:]).sum())/(len(a)*len(b)))
    pt=f(p,n);rng=np.random.default_rng(seed);bs=[]
    for _ in range(boot):
        a=rng.choice(p,len(p));b=rng.choice(n,len(n));bs.append(f(a,b))
    lo,hi=np.percentile(bs,[2.5,97.5]);return pt,float(lo),float(hi)
def chunk_labels(o_sizes):
    """o_sizes[k] = |oracle(k)|. Returns (list of chunk ids used, labels): O = non-empty, non-O = empty AFTER the first O chunk"""
    first=next((k for k,n in enumerate(o_sizes) if n>0),None)
    if first is None:return [],[]
    ks=[k for k in range(len(o_sizes)) if o_sizes[k]>0 or k>first];return ks,[1 if o_sizes[k]>0 else 0 for k in ks]
def summarize(delta,o_sizes):
    ks,y=chunk_labels(o_sizes);d=np.asarray(delta,float)
    if not ks:return {'n_O':0,'n_nonO':0,'auc':None}
    a=auc(-d[ks],y);return {'n_O':int(sum(y)),'n_nonO':int(len(y)-sum(y)),'auc':a[0],'auc_ci95':[a[1],a[2]],'mean_delta_O':float(d[[k for k,l in zip(ks,y) if l]].mean()),'mean_delta_nonO':float(d[[k for k,l in zip(ks,y) if not l]].mean()) if len(y)-sum(y) else None,'first_O_chunk':ks[0]}
def o_sizes_of(w,swap,tok,seed):
    import gen_world as gw,suppress as SP,s3_sets as SETS
    sets=SETS.Sets(SETS.sentence_table(w,swap,tok,gw,SP),seed,1);return [len(sets.oracle(k)) for k in range(NCH)],sets
def o_sizes_nat(w,swap,stream_tok,nat_world_mod,NS,stage2):
    """natural-text stream: |O| per chunk and 'has fact sentence' per chunk from nat_signal.NatSets on the realized stream (BOS offset handled by nat_signal.build_table)"""
    text=nat_world_mod.realize(w,swap)[0];e=stage2.enc(stream_tok,text);table,_=NS.build_table(w,swap,list(e.offsets));sets=NS.NatSets(table,0)
    return [len(sets.oracle(k)) for k in range(NCH)],sets
def selftest():
    rng=np.random.default_rng(0);osz=[0,0,3,0,0,5,0,2,0,0]+[0]*53
    ks,y=chunk_labels(osz);assert ks[0]==2 and y[:2]==[1,0] and sum(y)==3 and len(ks)==61,(ks[:4],y[:4])   # chunks 0,1 (before the first O chunk) are not used
    d=np.zeros(NCH);d[[2,5,7]]=-1.0;d+=rng.normal(0,0.01,NCH);r=summarize(d,osz);assert r['auc']>0.99 and r['n_O']==3 and r['mean_delta_O']<-0.9,r
    d2=rng.normal(0,1,NCH);r2=summarize(d2,osz);assert r2['auc_ci95'][0]<=r2['auc']<=r2['auc_ci95'][1]
    assert auc([1,2,3],[1,0,0])[0]==0.0 and auc([3,1,1],[1,0,0])[0]==1.0 and auc([1,1],[1,0])[0]==0.5 and np.isnan(auc([1,2],[1,1])[0])
    assert summarize(np.zeros(NCH),[0]*NCH)['auc'] is None
    print('ctx_gain SELFTEST OK',{k:(round(v,3) if isinstance(v,float) else v) for k,v in r.items() if k!='auc_ci95'})
def dry(a):
    if a.nat:return dry_nat(a)
    import nat_dryrun as D
    tk=D.load_tok(a.tokenizer_file);assert tk is not None,'real tokenizer file not found (--tokenizer-file / TOKENIZER_JSON)'   # a tokenizers.Tokenizer (real tokenizer file); gen_world needs the same package
    import gen_world as gw
    for s in [int(x) for x in a.seeds.split(',')]:
        w=gw.build_world(s,v=2)
        for wd in a.worlds:
            osz,_=o_sizes_of(w,wd=='B',tk,s);ks,y=chunk_labels(osz);print(f'seed {s} world {wd}: chunks with O {sum(1 for n in osz if n)}, first O chunk {ks[0] if ks else None}, non-O after it {len(y)-sum(y)}, mean |O| {np.mean([n for n in osz if n] or [0]):.1f}')
            assert sum(y)>=3 and len(y)-sum(y)>=3,'too few O or non-O chunks for an AUC'
    print('DRY OK (label construction only; no model, no GPU)')
def dry_nat(a):
    import nat_dryrun as D,nat_world as NW,nat_signal as NS
    tk=D.load_tok(a.tokenizer_file);assert tk is not None,'real tokenizer file not found';stream,layout=a.nat.split(':');w=NW.load_world(Path(a.nat_dir)/f'nat_{stream}_{layout}.json')
    for wd in a.worlds:
        e=tk.encode(NW.realize(w,wd=='B')[0],add_special_tokens=False);table,_=NS.build_table(w,wd=='B',list(e.offsets));sets=NS.NatSets(table,0);osz=[len(sets.oracle(k)) for k in range(NCH)];ks,y=chunk_labels(osz)
        print(f'nat {a.nat} world {wd}: chunks with O {sum(1 for n in osz if n)}, first O chunk {ks[0] if ks else None}, non-O after it {len(y)-sum(y)}');assert sum(y)>=3 and len(y)-sum(y)>=3
    print('DRY OK (natural-text labels only; no model, no GPU)')
def gpu(a):
    import ladder as LD
    from ladder import clone_state,jnp,reuse
    assert 'deterministic_ops=true' in os.environ.get('XLA_FLAGS','')
    nat=None
    if a.nat:   # E33 natural-text world: same reads, ladder = NatLadder with the natural loader installed
        import nat_ladder as NL,nat_signal as NS,nat_world as NW
        from ladder import stage2
        stream,layout=a.nat.split(':');stream=int(stream);NL.install(a.nat_dir,stream,layout);nat=(NL,NS,NW,stage2)
    rt=LD.Runtime();orig=rt.base;seeds=[int(x) for x in a.seeds.split(',')] if not nat else [stream];res={'seeds':seeds,'worlds':a.worlds,'runs':{},'nat':a.nat};t00=time.monotonic()
    for s in seeds:
        L=nat[0].NatLadder(rt,s,readout=False) if nat else LD.Ladder(rt,s,readout=False)
        for wd in a.worlds:
            swap=wd=='B';ids=np.asarray(L.T.P[swap].ids)[:NCH*CH];mask=np.zeros(NCH*CH,dtype=bool);mask[:len(ids)-1]=True
            osz,_=o_sizes_nat(L.T.w,swap,L.T.tok,nat[2],nat[1],nat[3]) if nat else o_sizes_of(L.T.w,swap,L.T.tok,s);states=[('W0',None)]+[(x.split('=',1)[0],x.split('=',1)[1]) for x in a.state]
            if a.state_template:states+=[(f'p{p}',a.state_template.format(seed=s,w=wd,p=int(p))) for p in a.state_passes.split(',')]
            with rt.mesh:
                seq=rt.seq(ids,mask=mask);pref=rt.prefix(orig,seq);alone=[]
                for k in range(NCH):   # no-context read: the chunk alone, fresh state, prefix of the chunk alone (e37_firstchunk_nll.read)
                    sq=rt.seq([int(t) for t in ids[k*CH:(k+1)*CH]]);pf=rt.prefix(orig,sq);alone.append((sq,pf))
                for lab,p in states:
                    m=orig if p is None else L.load_state(p);hh=reuse._hash_pytree(m.inner_parameters())[0] if p else None;st=clone_state(rt.state_suffix);ctx=[];noc=[]
                    for k in range(NCH):
                        c=rt.chunk(seq,k);st,lp=rt.chunk_logp(m,st,c,pref[k*CH:(k+1)*CH]);ctx.append(float(-np.asarray(lp,np.float64)[:CH-1].mean()))
                        sq,pf=alone[k];_,lp2=rt.chunk_logp(m,clone_state(rt.state_suffix),rt.chunk(sq,0),pf[:CH]);noc.append(float(-np.asarray(lp2,np.float64)[:CH-1].mean()))
                    d=np.asarray(ctx)-np.asarray(noc);r={'nll_ctx':ctx,'nll_noctx':noc,'delta':d.tolist(),'n_O_tokens':osz,'summary':summarize(d,osz),'hash':hh,'path':p}
                    res['runs'][f'{"n" if nat else "s"}{s}_{wd}_{lab}']=r;print(f's{s} {wd} {lab}: AUC {r["summary"]["auc"]} n_O {r["summary"]["n_O"]} mean delta O {r["summary"].get("mean_delta_O")} ({time.monotonic()-t00:.0f}s)',flush=True)
                    if p:del m
                    json.dump(res,open(LD.OUT/f'ctx_gain_{a.tag}.json','w'))
    L2=['| run | n_O | n_nonO | AUC (score = -delta) [95% CI] | mean delta O | mean delta non-O |','|---|---|---|---|---|---|']
    for k,r in res['runs'].items():
        x=r['summary'];L2.append(f"| {k} | {x['n_O']} | {x['n_nonO']} | {x['auc']:.3f} [{x['auc_ci95'][0]:.3f}, {x['auc_ci95'][1]:.3f}] | {x['mean_delta_O']:+.3f} | {x['mean_delta_nonO'] if x['mean_delta_nonO'] is None else format(x['mean_delta_nonO'],'+.3f')} |" if x['auc'] is not None else f'| {k} | no O chunk | | | | |')
    (LD.OUT/f'ctx_gain_{a.tag}.md').write_text('\n'.join(L2)+'\n');print('\n'.join(L2))
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--seeds',default='101,102,103');ap.add_argument('--worlds',default='A');ap.add_argument('--state',action='append',default=[]);ap.add_argument('--state-template',default=None)
    ap.add_argument('--state-passes',default='1,2,3,4');ap.add_argument('--tag',default='ctx');ap.add_argument('--dry',action='store_true');ap.add_argument('--selftest',action='store_true');ap.add_argument('--tokenizer-file',default=None);ap.add_argument('--nat',default=None,help='STREAM:LAYOUT, e.g. 9105:spread: natural-text world (GPU read; --dry then uses nat_signal.dry)');ap.add_argument('--nat-dir',default=os.environ.get('NAT_DIR','data/natural-text/worlds'));a=ap.parse_args()
    if a.selftest:selftest();return
    if a.dry:
        for p in ('../exp05_write_dynamics','../exp04_what_ttt_retains'):sys.path.insert(0,str(HERE/p))
        dry(a);return
    gpu(a)
if __name__=='__main__':main()
