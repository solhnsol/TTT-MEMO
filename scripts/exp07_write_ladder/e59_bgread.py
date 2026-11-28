"""K1 read-background swap (job 1368 'bgread'; GPU; READ ONLY; new file; implementation thread 11/29; design E59 [S2903-CONTRAST] K1, Hans 11/29 04:26 KST "응 둘 다 진행해봐"; descriptive). The saved final weights (128 exposures) of the seven ladder arms (P0s / P1s 1348, P1d 1362, L2 / L2b / L2c 1366, ISO1 1363 / 1366) and W0 are read with the R1 questions (16 x 16 candidates, 12 exposed + 4 never exposed, summed span log-prob) after different out-of-window contexts; the information is the same (no fact in any context), only the read distribution changes:
  SQ8  BOS + SQuAD dev 8192 tokens (= the R1 of every job, e55_run.out_tail_8192; checked against the saved R1 at 128)
  OT8  BOS + another source, 8192 tokens: the narrative tail book pg1342 (readouts.neutral_tail text; not among the 24 stream books)
  SA8  BOS + the SAME source as the write-stream background: paragraphs of the world's own background pool that the stream never uses (pool[len//2:]; the stream cuts its units from pool[:len//2]), 8192 tokens
  SQ1  BOS + the first 1023 tokens of SQ8 (one chunk);  SA1 = BOS + the first 1023 tokens of SA8
  NONE no context (the probe rows alone, Packed.r0)
Every context is checked to contain no name / place / never-exposed place of the world (e55_run.forbid_list; the job stops otherwise). The 8-chunk in-window reads are job 1366r (unchanged).
  python3 e59_bgread.py --seed 231 --out O --srcs 'ARM=PATTERN,...'"""
import sys,os,json,time,pathlib,argparse
import numpy as np
HERE=pathlib.Path(__file__).resolve().parent;ROOT=HERE.parents[1]
for p in (HERE,HERE/'e57',ROOT/'scripts/exp04_what_ttt_retains',ROOT/'scripts/exp05_write_dynamics'):sys.path.insert(0,str(p))
NREAD=16;CTXS=('SQ8','OT8','SA8','SQ1','SA1','NONE')
def contexts(LD,R55,G,RD,tok,bos,w,enc,seed):
    """-> {name: token ids or None (NONE)}, plus the forbid hits per context"""
    fb=R55.forbid_list(w);sq8=list(R55.out_tail_8192(LD,tok,bos,fb))
    _,ot_text,_=RD.neutral_tail(tok,bos);ot8=[bos]+LD.stage2.enc(tok,ot_text).ids
    assert len(ot8)>=8192,len(ot8);ot8=ot8[:8192]
    bg=G.default_bg_pool(seed,enc,G.reference_grams(enc,ROOT),books=G.cell_books(w_cell(w)));half=len(bg)//2;rest=[p for p in bg[half:] if not any(nm in p for nm in fb)]      # gen_facts.cell_world (nth = 2 for U12): the real stream and the 'names' decoy cut their units from bg[:len//2]; bg[len//2:] is only read by the 'decoy' world kind, which U12 does not build
    sa=[bos];i=0
    while len(sa)<8192:sa+=LD.stage2.enc(tok,('\n\n' if i else '')+rest[i]).ids;i+=1
    sa8=sa[:8192];C={'SQ8':sq8,'OT8':ot8,'SA8':sa8,'SQ1':sq8[:1024],'SA1':sa8[:1024],'NONE':None}
    hits={}
    for nm,ids in C.items():
        if ids is None:hits[nm]=0;continue
        t=tok.decode(ids[1:]);hits[nm]=sum(t.count(x) for x in fb)
    return C,hits,dict(sa_paragraphs=i,ot_book=RD.TAIL_BOOK)
def w_cell(w):
    import e59_u12 as U;return U.CELL
def run(a):
    import jax
    import e55_run as R55,e55_read as R,e55_gpu as RG,e57_natmask_run as NM,e57_qband_run as QR,e57_syn_pass as SP,e57_oracle_run as OR,gen_facts as G,e59_u12 as U,readouts as RD
    from e59_kvpos2 import items2,NEXP
    NM.register_engine();QR.K0=0;NM.K0=0;lg=lambda *x:print(*x,flush=True);seed=a.seed;out=pathlib.Path(a.out);out.mkdir(parents=True,exist_ok=True);t_all=time.monotonic()
    U.install();cell=U.CELL;ci=OR.register_cell(cell);LD,s3_,C_=R55.lab();rt=LD.Runtime();rt32=LD.runtime_fp32()
    rd=RG.Reader55(rt,rt32,LD.packing,LD._TailCtx,LD.reuse.get_filter_spec,LD.reuse._hash_pytree,LD.jax,LD.eqx,LD.jnp)
    enc,w,tf=R55.build(SP.synth_args(seed,cell),seed,read_apps=set(ci['read_apps']),arms=['NAT']);UX=G.fresh_places(w,enc,4);G.build_items=lambda ww:items2(ww,UX)
    P=R.plan(w,enc);items=P['items'];raw=R55.rows_for(LD,rt,items);RG.check_rows(raw,len(items),G.NCAND);NC=G.NCAND
    CX,hits,info=contexts(LD,R55,G,RD,rt.tok,rt.bos,w,enc,seed);lg(f'seed {seed}: contexts {[(k,len(v) if v else 0) for k,v in CX.items()]}, forbidden-name hits {hits}, {info}')
    assert all(v==0 for v in hits.values()),('a context contains a world name / place',hits)
    PK={}
    with rt32.mesh,jax.default_matmul_precision('highest'):
        for nm,ids in CX.items():
            T_=LD._TailCtx(rt32,list(ids) if ids else list(CX['SQ1'][:1024]));PK[nm]=LD.packing.Packed(T_,raw,GP=2)      # NONE uses Packed.r0 (rows alone); its tail object is unused
    def score(m):
        m32=rd.to_fp32(m);res={}
        with jax.default_matmul_precision('highest'):
            for nm,pk in PK.items():res[nm]=(np.asarray(pk.r0(m32)) if nm=='NONE' else np.asarray(pk.r1(m32))).reshape(len(items),NC)[:NREAD].tolist()
        del m32;return res
    RES=dict(seed=seed,contexts=list(CTXS),lens={k:(len(v) if v else 0) for k,v in CX.items()},forbid_hits=hits,info=info,cand_exposed=[True]*NEXP+[False]*(NC-NEXP),W0=score(rt.base),arms={},checks={},args=vars(a))
    SRC={x.split('=')[0]:x.split('=')[1] for x in a.srcs.split(',')}
    for arm,pat in SRC.items():
        sj=json.load(open(ROOT/pat.format(seed=seed)));fs=sj['arms'][arm]['reads']['128']['final_state'];m=rd.load_state(ROOT/fs['_file'],fs['_hash']);t0=time.monotonic();RES['arms'][arm]=score(m)
        saved=np.asarray(sj['W0_R1'])+np.asarray(sj['arms'][arm]['reads']['128']['dS1']);RES['checks'][arm]=float(np.abs(np.asarray(RES['arms'][arm]['SQ8'])-saved).max());lg(f'seed {seed} {arm}: scored {time.monotonic()-t0:.0f}s, SQ8 vs saved R1 max abs {RES["checks"][arm]:.4f}');del m
        json.dump(RES,open(out/f'bgread_s{seed}.json','w'))
    json.dump(RES,open(out/f'bgread_s{seed}.json','w'));lg('wrote',out/f'bgread_s{seed}.json',f'{time.monotonic()-t_all:.0f}s')
    bad={k:v for k,v in RES['checks'].items() if v>0.05};assert not bad,('SQ8 read differs from the saved R1',bad)
if __name__=='__main__':
    ap_=argparse.ArgumentParser();ap_.add_argument('--seed',type=int,default=231);ap_.add_argument('--out');ap_.add_argument('--srcs',default='');a=ap_.parse_args();run(a)
