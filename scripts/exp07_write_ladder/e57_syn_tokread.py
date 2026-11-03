"""E57 synthetic world, per-token re-read of the saved end states (new file; GPU read only, NO weight update; E57 122-Z (a)+(b)): the states of the end of the stream (appearance 128 = the last chunk) of the seed-64 query-band arms of job 360b (cell v3_16_1, 15 arms: e57_synq/states/E55_synq_<ARM>_s64/state_<ARM>_s64_127.npz, hash of the .saved.json) and of the E55 210s arms NAT, SENTPQ, SENTPQADAMH of seed 61 (cell v3_64_4, tag d64, states/E55_d64s_<ARM>_s61/state_dc<ARM>_511.npz, hash = final_hash of the .saved.json) are read again with the out-of-window reader of the original runs (the neutral 8192-token tail, the same world / items / candidate rows rebuilt by e55_run.build / e55_read.plan / e55_run.rows_for with the same arguments, the reference packed scorer e55_gpu.Reader55.packed -> packing.Packed.r1), but the log-prob of every candidate TOKEN is stored. The reads at appearances 16 / 32 / 64 of those runs were taken inline during the writing and have no saved state, so only the end read (d = 128, OUT) can be repeated. The summed log-probs are compared with the stored read[128]['out'] and W0['out'] matrices of the original .read.json (the maximum difference is reported).
Metrics per fact row (kind 'fact'; gold = the place of the fact, candidates = 16 places): branch-token rank / top-1 (gold against every candidate, both scored from their first token to and including the first token at which the two token sequences differ), first-token rank / top-1, full-sum rank / top-1 (rank = 1 + number of candidates that beat the gold, ratio = rank / 16); the binding: A_f = score of the gold place under the own name frame, B_f = mean score of the same place under the other fact rows that have it as a candidate, for the FIRST-TOKEN score and the full sum; d against W0; per arm mean over facts with the bootstrap 95% CI over facts; masked arms as paired differences to the reference arm (NATQ for seed 64, NAT for seed 61) with the ratio d(d(A-B)) / d(dA) per score.
  python3 e57_syn_tokread.py --stage read --seed 64|61 --out O | --stage analyze --out O | --selftest"""
import sys,os,json,time,gc,types,pathlib,hashlib
import numpy as np
HERE=pathlib.Path(__file__).resolve().parent;ROOT=HERE.parents[1]
for p in (HERE,HERE/'e57',ROOT/'scripts/exp04_what_ttt_retains',ROOT/'scripts/exp05_write_dynamics'):sys.path.insert(0,str(p))
EXP=ROOT/'experiments/exp07_write_ladder'
S64=sorted(p.name.split('_')[2] for p in (EXP/'e57_synq').glob('E55_synq_*_s64.saved.json'))
def arms_s64():
    out={}
    for a in S64:
        d=json.load(open(EXP/f'e57_synq/E55_synq_{a}_s64.saved.json'));r=list(d['reads'].values())[-1];out[a]=dict(file=r['_file'],hash=r['_hash'],read_json=str(EXP/f'e57_synq/E55_synq_{a}_s64.read.json'))
    return out
def arms_s61():
    out={}
    for a in ('NAT','SENTPQ','SENTPQADAMH'):
        d=json.load(open(EXP/f'E55_d64_{a}_s61.saved.json'));out[a]=dict(file=str(EXP/f'states/E55_d64s_{a}_s61/state_dc{a}_511.npz'),hash=d['final_hash'],read_json=str(EXP/f'E55_d64_{a}_s61.read.json'))
    return out
def r1_tokens(P,rt,m):
    """per-row arrays of the token log-probs of the packed scorer (the same computation as packing.Packed.r1)"""
    from core import clone_state
    T=P.T;GP=P.GP
    with rt.mesh:
        st=clone_state(rt.state_suffix)
        for j in range(len(T.tail_ids)//1024):st,_=rt.chunk_logp(m,st,rt.chunk(T._tail_seq,j),T._tail_pref[j*1024:(j+1)*1024])
        out=[]
        for g in range(0,P.ids.shape[0],GP):out.append(np.asarray(P.lp_packed(m,st,P.ids[g:g+GP],P.tg[g:g+GP],P.mk[g:g+GP],P.pref1[g:g+GP],P.seg[g:g+GP],P.loc[g:g+GP])))
    lp=np.concatenate(out);return [lp[c,a:b].astype(np.float64) for c,a,b in P.where]
def read_stage(a):
    out=pathlib.Path(a['out']);out.mkdir(parents=True,exist_ok=True);seed=int(a['seed']);import e55_run as R55,e55_read as R,e55_gpu as RG,gen_facts as G
    cell,arms={64:('v3_16_1',arms_s64()),61:('v3_64_4',arms_s61())}[seed];lg=lambda *x:print(*x,flush=True)
    if a.get('arms'):arms={k:v for k,v in arms.items() if k in a['arms'].split(',')};assert arms,'no arm selected'
    LD,s3,C=R55.lab();rt=LD.Runtime();rt32=LD.runtime_fp32();rd=RG.Reader55(rt,rt32,LD.packing,LD._TailCtx,LD.reuse.get_filter_spec,LD.reuse._hash_pytree,LD.jax,LD.eqx,LD.jnp)
    args=types.SimpleNamespace(cell=cell,seed=seed,tokenizer_file=None,bg='pool',decoy='names',stub=False,arms='NAT')
    enc,w,tf=R55.build(args,seed,read_apps={16,32,64,128},arms=['NAT']);P_=R.plan(w,enc);items=P_['items'];tok=rt.tok;raw=R55.rows_for(LD,rt,items);RG.check_rows(raw,len(items),G.NCAND);out_tail=R55.out_tail_8192(LD,tok,rt.bos,R55.forbid_list(w))
    lg(f'seed {seed}: {len(items)} items, {len(raw)} rows, place_len {w["place_len"]}');P=rd.packed(raw,out_tail);tokids=[[int(x) for x in ids[a_+1:b_+1]] for ids,(a_,b_) in raw]
    def tokens(m):
        with LD.jax.default_matmul_precision('highest'):return [x.tolist() for x in r1_tokens(P,rt32,rd.to_fp32(m))]
    res={'seed':seed,'cell':cell,'n_items':len(items),'tokids':tokids,'W0':tokens(rt.base),'arms':{}};chk={}
    for ar,s in arms.items():
        t0=time.monotonic();m=rd.load_state(s['file'],expect_hash=s['hash']);res['arms'][ar]=tokens(m);del m
        ref=json.load(open(s['read_json']));sm=np.asarray([sum(x) for x in res['arms'][ar]]).reshape(-1,G.NCAND);rm=np.asarray(ref['read']['128']['out'],np.float64)
        w0s=np.asarray([sum(x) for x in res['W0']]).reshape(-1,G.NCAND);w0r=np.asarray(ref['W0']['out'],np.float64);chk[ar]=dict(state=float(np.abs(sm-rm).max()),W0=float(np.abs(w0s-w0r).max()),n_rows=int(rm.shape[0]))
        lg(f'  {ar}: {time.monotonic()-t0:.0f}s, max|diff| to the stored read: state {chk[ar]["state"]:.2e}, W0 {chk[ar]["W0"]:.2e}')
    res['reproduction']=chk;(out/f'e57_syn_tokread_s{seed}.json').write_text(json.dumps(res))
def metrics(tl,tokids,meta,ncand=16):
    """tl: [n_items*16][ntok] token log-probs; -> per fact (kind 'fact') dict of scores"""
    kind=np.asarray(meta['kind']);fact=np.asarray(meta['fact']);cf=np.asarray(meta['cand_fact']);gi=np.asarray(meta['gi']);rows=[r for r in range(len(kind)) if kind[r]=='fact'];out={}
    lp=lambda r,c:np.asarray(tl[r*ncand+c],np.float64);ti=lambda r,c:tokids[r*ncand+c]
    for r in rows:
        g=int(gi[r]);lg_=lp(r,g);nb=nf=nu=0;db=[];df=[];du=[]
        for c in range(ncand):
            if c==g:continue
            tg,tc=ti(r,g),ti(r,c);t=next((q for q in range(min(len(tg),len(tc))) if tg[q]!=tc[q]),min(len(tg),len(tc))-1);lc=lp(r,c);sg=lg_[:t+1].sum();sc=lc[:t+1].sum();nb+=int(sc>sg);db.append(sg-sc);nf+=int(lc[0]>lg_[0]);df.append(lg_[0]-lc[0]);nu+=int(lc.sum()>lg_.sum());du.append(lg_.sum()-lc.sum())
        f=int(fact[r]);Bf=[lp(q,c) for q in rows if q!=r for c in range(ncand) if int(cf[q,c])==f]
        out[f]=dict(rank_branch=1+nb,rank_first=1+nf,rank_full=1+nu,gap_branch=float(np.mean(db)),gap_first=float(np.mean(df)),gap_full=float(np.mean(du)),A_first=float(lg_[0]),A_full=float(lg_.sum()),B_first=float(np.mean([x[0] for x in Bf])) if Bf else None,B_full=float(np.mean([x.sum() for x in Bf])) if Bf else None)
    return out
def boot(x,seed=1,B=2000):
    x=np.asarray(x,np.float64)
    if len(x)==0:return dict(mean=None,ci95=None,n=0)
    ix=np.random.default_rng(seed).integers(0,len(x),(B,len(x)));m=x[ix].mean(1);return dict(mean=float(x.mean()),ci95=[float(np.percentile(m,2.5)),float(np.percentile(m,97.5))],n=int(len(x)))
def ratio(num,den,seed=1,B=2000):
    num=np.asarray(num);den=np.asarray(den);n=len(num);ix=np.random.default_rng(seed).integers(0,n,(B,n));r=num[ix].mean(1)/den[ix].mean(1);return dict(ratio=float(num.mean()/den.mean()),ci95=[float(np.nanpercentile(r,2.5)),float(np.nanpercentile(r,97.5))],n=int(n))
def analyze(a):
    out=pathlib.Path(a['out']);res={}
    for seed,ref in ((64,'NATQ'),(61,'NAT')):
        f=out/f'e57_syn_tokread_s{seed}.json'
        if not f.exists():continue
        d=json.load(open(f));arms={'64':arms_s64,'61':arms_s61}[str(seed)]();meta=json.load(open(next(iter(arms.values()))['read_json']))['meta'];tokids=d['tokids'];W0=metrics(d['W0'],tokids,meta);M={ar:metrics(t,tokids,meta) for ar,t in d['arms'].items()};facts=sorted(W0);g={'reference':ref,'reproduction':d['reproduction'],'W0':{},'by_arm':{},'vs_reference':{}}
        g['W0']={k:boot([W0[f_][k] for f_ in facts]) for k in ('rank_branch','rank_first','rank_full','gap_branch','gap_first','gap_full','A_first','A_full')}
        def dl(m,score):
            ok=[f_ for f_ in facts if m[f_][f'B_{score}'] is not None and W0[f_][f'B_{score}'] is not None];dA=np.asarray([m[f_][f'A_{score}']-W0[f_][f'A_{score}'] for f_ in ok]);dB=np.asarray([m[f_][f'B_{score}']-W0[f_][f'B_{score}'] for f_ in ok]);return ok,dA,dB
        for ar,m in M.items():
            e={'n_facts':len(facts),'mean_rank_ratio':{k:boot([m[f_][f'rank_{k}']/16.0 for f_ in facts]) for k in ('branch','first','full')},'top1':{k:boot([float(m[f_][f'rank_{k}']==1) for f_ in facts]) for k in ('branch','first','full')},'gap':{k:boot([m[f_][f'gap_{k}'] for f_ in facts]) for k in ('branch','first','full')}}
            for sc in ('first','full'):ok,dA,dB=dl(m,sc);e[f'{sc}_dA']=boot(dA);e[f'{sc}_dB']=boot(dB);e[f'{sc}_dA_minus_dB']=boot(dA-dB);e[f'{sc}_n_with_B']=len(ok)
            g['by_arm'][ar]=e
            if ar!=ref:
                c={}
                for sc in ('first','full'):
                    ok,dA,dB=dl(m,sc);okr,dAr,dBr=dl(M[ref],sc);assert ok==okr;ddA=dA-dAr;ddAB=(dA-dB)-(dAr-dBr);c[sc]=dict(d_dA=boot(ddA),d_dAB=boot(ddAB),ratio_dAB_over_dA=ratio(ddAB,ddA))
                c['rank_ratio_diff']={k:boot([(m[f_][f'rank_{k}']-M[ref][f_][f'rank_{k}'])/16.0 for f_ in facts]) for k in ('branch','first','full')};g['vs_reference'][ar]=c
        res[f's{seed}']=g
    (out/'e57_syn_tokread_keys.json').write_text(json.dumps(res,indent=1,default=float))
def selftest():
    meta=dict(kind=['fact','fact','ctrl'],fact=[0,1,2],cand_fact=[[0,1],[0,1],[0,1]],gi=[0,1,0])
    tl=[[-1.,-2.],[-3.,-1.],[-2.,-1.],[-1.,-1.]];tok=[[1,2],[1,3],[4,5],[4,6]]   # row0: cands (gold [1,2], other [1,3]); row1: (other [4,5], gold [4,6])
    m=metrics([[-1.,-2.],[-3.,-1.],[-2.,-1.],[-1.,-1.]],tok,meta,ncand=2)
    assert m[0]['rank_branch']==1 and m[0]['rank_first']==1 and m[1]['rank_full']==1
    assert m[0]['A_full']==-3. and m[1]['A_first']==-1. and m[0]['B_first']==-2. and abs(m[0]['B_full']+3.0)<1e-9
    print('e57_syn_tokread selftest ok')
def main():
    if '--selftest' in sys.argv:selftest();return
    g=lambda k:sys.argv[sys.argv.index(k)+1];st=g('--stage')
    if st=='read':read_stage(dict(out=g('--out'),seed=g('--seed'),arms=g('--arms') if '--arms' in sys.argv else None))
    else:analyze(dict(out=g('--out')))
if __name__=='__main__':main()
