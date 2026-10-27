"""E57 T1/T2 IN reads of the cycle-2 experiment states (job 283), GPU reads only (post-hoc, exploratory; Hans 09:59Z approved in advance): the frozen 150 T1/T2 frames (e57_t1t2_v1.json, hash e3897ebd...) scored with the IN context = the natural 8192-token window of the end of the cycle (stream tokens n*1024-8192 .. n*1024, the window of the write chunk 280 / of the read at chunk 279; the window at the end of cycle 2 is the same text, checked), with the scorer, candidates and row format of e57_t1t2c2_run (cand_lp, cand_ntok, cand_surface, cand_kinds, ...). States: W0, NAT (job 261 chunks 279 / 559), ORC2 and RAND2 (job 283 chunks 279 / 559). Rows carry window_info = whether the gold string of the frame stands in the window text. Raw scores only.
  python3 e57_t1t2c2_in_run.py --bundle B --out O --cycles 1 --npos 1 --canary-npos 1 --arms NAT --stage reads --frozen F --nat-arms A --c2-dir D     |   --selftest   |   --cpu-check --frozen F --nat-arms A --c2-dir D --bundle B"""
import sys,json,time,hashlib,pathlib
import numpy as np
HERE=pathlib.Path(__file__).resolve().parent
for p in (HERE,HERE/'e57',HERE.parents[1]/'scripts/exp04_what_ttt_retains',HERE.parents[1]/'scripts/exp05_write_dynamics'):sys.path.insert(0,str(p))
POD={'frozen':None,'nat':None,'c2dir':None}
def _pop(flag):
    if flag in sys.argv:
        i=sys.argv.index(flag);v=sys.argv[i+1];del sys.argv[i:i+2];return v
def in_window(ids,n,CH=1024,W=8192):
    """tokens of the natural window at the end of the cycle (and the same text at the end of cycle 2 for a stream of several cycles)"""
    w=np.asarray(ids[n*CH-W:n*CH]);
    if len(ids)>=2*n*CH:assert np.array_equal(w,np.asarray(ids[2*n*CH-W:2*n*CH])),'the window at the end of cycle 2 differs from that of cycle 1'
    return [int(x) for x in w]
def in_reads(a,out,items,bsrc,iv,hits,shtab,ids_all,n,N,kks,arms,tk,LD,rt,R,cw0,lg,can):
    import e55_gpu as RG,e57_t1t2_run as T,e57_reads as RD,e57_t1t2c2_run as C2
    frozen=T.load_frozen(POD['frozen']);new=T.make_items(frozen,out);states=C2.build_states(POD['c2dir'],['ORC2','RAND2'],POD['nat'])
    win=in_window(ids_all,n);wtxt=tk.decode(win);wp=RD.window_present(new,wtxt)
    rt32=LD.runtime_fp32();rd=RG.Reader55(rt,rt32,LD.packing,LD._TailCtx,LD.reuse.get_filter_spec,LD.reuse._hash_pytree,LD.jax,LD.eqx,LD.jnp)
    raw,ntok,bad=RD.build_rows(rt.tok,new);assert not bad,bad
    GROUP=40;groups=[];o=0
    for g0 in range(0,len(new),GROUP):
        sub=new[g0:g0+GROUP];nr=sum(len(it['cands']) for it in sub);groups.append((sub,rd.packed(raw[o:o+nr],win)));o+=nr
    assert o==len(raw);lg(f'IN window: {len(win)} tokens, sha {hashlib.sha256(np.asarray(win,np.int64).tobytes()).hexdigest()[:16]}; gold string inside the window text for {sum(wp.values())} of {len(new)} frames; scorers built')
    def score(m):
        m32=rd.to_fp32(m);res=[]
        with LD.jax.default_matmul_precision('highest'):
            for sub,P in groups:res+=RD.split_scores(np.asarray(P.r1(m32)),sub)
        return res
    t0=time.monotonic();w0=T.rows_for_state(new,score(rt.base),ntok,'W0');rows=list(w0);tim={'W0':time.monotonic()-t0};lg(f'  W0 read: {tim["W0"]:.0f}s')
    part=pathlib.Path(out)/'E57_t1t2c2_in_scores_partial.json';pathlib.Path(out).mkdir(parents=True,exist_ok=True)
    def save_partial():part.write_text(json.dumps(dict(done=sorted({r['state'] for r in rows}),rows=rows),default=float))
    save_partial();live=T.inner_dtypes_(LD,rd,rt.base)
    for lab,f,h in states:
        t0=time.monotonic();m=rd.load_state(str(f),expect_hash=h);dt=T.inner_dtypes_(LD,rd,m);assert dt==live,('precision',lab,dt,live)
        rows+=T.rows_for_state(new,score(m),ntok,lab,w0=w0);del m;tim[lab]=time.monotonic()-t0;save_partial();lg(f'  {lab} read: {tim[lab]:.0f}s')
    for r in rows:r['window_info']='present' if wp[r['id']] else 'absent'
    C2.check_rows(rows,new,['W0']+[s[0] for s in states]);lg('completeness check passed')
    meta=dict(frozen_sha256=frozen['sha256'],context='IN (natural 8192-token window at the end of the cycle)',window_sha=hashlib.sha256(np.asarray(win,np.int64).tobytes()).hexdigest()[:16],gold_in_window=int(sum(wp.values())),states={lab:h for lab,_,h in states},primary_score='cand_mean = cand_lp / cand_ntok (raw score columns only)',timings=tim,n_items=len(new))
    (pathlib.Path(out)/'E57_t1t2c2_in_scores.json').write_text(json.dumps(dict(meta=meta,rows=rows),default=float))
    kn=T.key_numbers(rows,meta);(pathlib.Path(out)/'E57_t1t2c2_in_score_keys.json').write_text(json.dumps(kn,indent=1,default=float));lg('wrote E57_t1t2c2_in_scores.json and E57_t1t2c2_in_score_keys.json')
def selftest():
    n=10;ids2=np.concatenate([np.arange(n*1024)]*2);w=in_window(ids2,n);assert len(w)==8192 and w==list(range(n*1024-8192,n*1024))
    assert len(in_window(np.arange(n*1024),n))==8192            # a one-cycle stream has no second copy to compare
    bad=np.concatenate([np.arange(n*1024),np.arange(n*1024)+1])
    try:in_window(bad,n)
    except AssertionError:pass
    else:raise SystemExit('window mismatch not detected')
    import e57_t1t2c2_run as C2;C2.selftest();print('e57_t1t2c2_in_run selftest ok')
def cpu_check():
    import argparse,e57_t1t2_run as T,e57_t1t2c2_run as C2,gen_world_ctl as G,e57_reads as RD
    ap=argparse.ArgumentParser();ap.add_argument('--frozen');ap.add_argument('--nat-arms');ap.add_argument('--c2-dir');ap.add_argument('--bundle');ap.add_argument('--out');ap.add_argument('--tokenizer');a=ap.parse_args()
    frozen=T.load_frozen(a.frozen);new=T.make_items(frozen,a.out or '/tmp');st=C2.build_states(a.c2_dir,['ORC2','RAND2'],a.nat_arms);print('frozen hash ok; items',len(new),'; states',[(l,pathlib.Path(f).exists()) for l,f,h in st]);assert all(pathlib.Path(f).exists() for l,f,h in st)
    ROOT=HERE.parents[1];tf=a.tokenizer or str(ROOT/'data/tokenizer/llama3/tokenizer.json');enc=G.load_enc(tf);from tokenizers import Tokenizer;tk=Tokenizer.from_file(tf)
    bank=pathlib.Path(a.bundle)/'e57_bank'/'bank_v1.json';b=RD.load_items(bank);ids,iv,hits=RD.article_intervals(a.bundle,enc,b);n=len(ids)//1024;win=in_window(np.tile(ids,2),n);wp=RD.window_present(new,tk.decode(win));print('IN window tokens',len(win),'| gold string inside the window text for',sum(wp.values()),'of',len(new),'frames')
    selftest()
def main():
    if '--selftest' in sys.argv:selftest();return
    if '--cpu-check' in sys.argv:sys.argv.remove('--cpu-check');cpu_check();return
    POD['frozen']=_pop('--frozen');POD['nat']=_pop('--nat-arms');POD['c2dir']=_pop('--c2-dir');assert POD['frozen'] and POD['nat'] and POD['c2dir']
    import e57_main_run as M
    M.stage_reads=in_reads;M.main()
if __name__=='__main__':main()
