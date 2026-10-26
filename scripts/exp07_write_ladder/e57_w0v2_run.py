"""E57 W0 read of the ORC v2 deleted windows (GPU forward only, no write step; post-hoc, exploratory; E57 section 79): same design as stage w0mask of e57_main_run (job 264) with the ORC v2 rule (e57_orc2 variant 'v2': abbreviation-aware sentence splitting R1 + partial forms R2 of the question inside the articles of its own event) at each question's PRIMARY position of job 264 (E57_w0mask.json 'primary': the position whose FULL window holds the most own-event article tokens). Conditions IN_DELORC_V2 (window = the last 8192 tokens that remain after the v2-hidden sentence fragments are deleted, refilled from earlier chunks) and IN_DELRND_V2 (the same number of deleted tokens taken from random fragments of the same region, seed RS_SEED + 0, one control). OUT and IN_FULL are NOT read again (the values of job 264, same W0, same windows). Only the questions of a position are scored in that position's contexts (rows of those questions only), which reduces the scorer work if its time follows the number of rows.
  python3 e57_w0v2_run.py --bundle B --out O --primary-from E57_w0mask.json --cycles 1 --npos 1 --canary-npos 1 --arms NAT --stage w0mask     (GPU)
  python3 e57_w0v2_run.py --bundle B --primary-from E57_w0mask.json --plan-only                                                                       (CPU plan check)"""
import sys,json,time,pathlib,argparse
import numpy as np
HERE=pathlib.Path(__file__).resolve().parent
for p in (HERE,HERE/'e57',HERE.parents[1]/'scripts/exp04_what_ttt_retains',HERE.parents[1]/'scripts/exp05_write_dynamics'):sys.path.insert(0,str(p))
CH=1024;NW=8
POD={'primary':None}
def _pop(flag,has_value=True):
    if flag in sys.argv:
        i=sys.argv.index(flag)
        if not has_value:del sys.argv[i];return True
        v=sys.argv[i+1];del sys.argv[i:i+2];return v
def plan(items,ids,iv,tk,tfile,primary,UVm,B):
    """-> dict(windows {kk: {cond: (seq, (s,e))}}, positions {kk: [question indices]}, summary)"""
    import e57_orc2 as O2
    bd=O2.build('v2',ids,tk,tfile,items,iv,UVm,B);fr=bd['fr'];hidden=bd['hidden'];fst=np.array([f[1] for f in fr]);fln=np.array([f[2] for f in fr])
    def build(kk,hid):
        e=(kk+1)*CH;hi=int(np.searchsorted(fst+fln,e,side='right'));tot=0;j=hi-1;pieces=[]
        while j>=0 and tot<8192:
            if not hid[j]:pieces.append(j);tot+=int(fln[j])
            j-=1
        assert tot>=8192,('window cannot be filled',kk,tot)
        seq=np.concatenate([ids[fst[q]:fst[q]+fln[q]] for q in reversed(pieces)])[-8192:]
        return [int(x) for x in seq],j+1,hi
    pos={};
    for qi,it in enumerate(items):pos.setdefault(int(primary[it['id']]),[]).append(qi)
    wins={};summ={'hidden_fragments_v2':int(hidden.sum()),'fragments':len(fr),'positions':{}}
    for kk in sorted(pos):
        e=(kk+1)*CH;o,j0,hi=build(kk,hidden);H=int(fln[j0:hi][hidden[j0:hi]].sum());hr=np.zeros(len(fr),bool);hr[j0:hi]=UVm.rands_hidden(fln[j0:hi],H,kk,UVm.RS_SEED+0);r,rj0,_=build(kk,hr)
        wins[kk]={'IN_DELORC_V2':(o,(int(fst[j0]),e)),'IN_DELRND_V2':(r,(int(fst[rj0]),e))};summ['positions'][kk]={'questions':len(pos[kk]),'hidden_tokens_in_region':H,'region_start_chunk':float(fst[j0]/CH),'lens':{k_:len(v[0]) for k_,v in wins[kk].items()}}
        assert all(len(v[0])==8192 for v in wins[kk].values())
    return dict(windows=wins,positions=pos,summary=summ)
def w0v2_stage(a,out,items,bsrc,iv,hits,ids,n,tk,tfile,LD,rt,R,cw0,lg):
    import e55_gpu as RG,e55_uv as UVm,e57_reads as RD,e57_bank as B
    prim=json.load(open(POD['primary']))['primary'];pl=plan(items,ids,iv,tk,tfile,prim,UVm,B);lg('w0v2 plan:',json.dumps(pl['summary']))
    rt32=LD.runtime_fp32();rd=RG.Reader55(rt,rt32,LD.packing,LD._TailCtx,LD.reuse.get_filter_spec,LD.reuse._hash_pytree,LD.jax,LD.eqx,LD.jnp)
    raw,ntok,bad=RD.build_rows(rt.tok,items);assert not bad,bad
    off=np.concatenate([[0],np.cumsum([len(it['cands']) for it in items])])
    def score(P,m,sub):
        m32=rd.to_fp32(m)
        with LD.jax.default_matmul_precision('highest'):return RD.split_scores(np.asarray(P.r1(m32)),[items[i] for i in sub])
    csa={it['id']:it['q'].get('c_SA') for it in items};rows=[];tim={}
    for kk in sorted(pl['windows']):
        sub=pl['positions'][kk];raw_sub=[x for i in sub for x in raw[off[i]:off[i+1]]];nt_sub=[ntok[i] for i in sub];its=[items[i] for i in sub]
        for cond,(seq,(s_,e_)) in pl['windows'][kk].items():
            t0=time.monotonic();P=rd.packed(raw_sub,seq);wp=RD.window_present(its,tk.decode(seq))
            rr=RD.log_rows(its,score(P,rt.base,sub),nt_sub,cycle=0,read_chunk=kk,pos_tokens=(kk+1)*CH,win=(s_,e_),arm='W0',condition=cond,iv=iv,hits=hits,winpresent=wp,csa=csa,shadow=None)
            for r in rr:r['primary_position']=True
            rows+=rr;tim[f'{cond}@{kk}']=dict(sec=time.monotonic()-t0,questions=len(sub),rows=len(raw_sub));lg(f'  W0 {cond} at chunk {kk}: {tim[f"{cond}@{kk}"]["sec"]:.0f}s for {len(sub)} questions ({len(raw_sub)} candidate rows); gold string present in {sum(wp.values())} of {len(sub)} windows')
    (out/'E57_w0v2.json').write_text(json.dumps(dict(arm='W0',bank=bsrc,canary_w0=cw0,primary=prim,summary=pl['summary'],rows=rows,timings=tim),default=float));lg('wrote',out/'E57_w0v2.json')
def main():
    POD['primary']=_pop('--primary-from');plan_only=_pop('--plan-only',False)
    if plan_only:
        a=argparse.ArgumentParser();a.add_argument('--bundle');a.add_argument('--tokenizer');a.add_argument('--bank');a=a.parse_args()
        ROOT=HERE.parents[1];import gen_world_ctl as G,e55_uv as UVm,e57_reads as RD,e57_bank as B
        from tokenizers import Tokenizer
        tf=a.tokenizer or str(ROOT/'data/tokenizer/llama3/tokenizer.json');enc=G.load_enc(tf);tk=Tokenizer.from_file(tf)
        bank=pathlib.Path(a.bank) if a.bank else pathlib.Path(a.bundle)/'e57_bank'/'bank_v1.json';items=RD.load_items(bank);ids,iv,hits=RD.article_intervals(a.bundle,enc,items)
        pl=plan(items,ids,iv,tk,tf,json.load(open(POD['primary']))['primary'],UVm,B);print('w0v2 plan:',json.dumps(pl['summary']));print('w0v2 plan check OK');return
    import e57_main_run as M
    M.stage_w0mask=w0v2_stage;M.main()
if __name__=='__main__':main()
