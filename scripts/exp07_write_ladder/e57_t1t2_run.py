"""E57 T1/T2 scoring (E57 section 83; GPU for the reads, CPU for the check). The frozen file e57_t1t2_v1.json (content hash e3897ebd...) holds 150 frames (T1 = bank question with new frames, T2 = reverse questions) with their candidates exactly as frozen: gold (fixed surface) + easy + nominated hard (primary set) + the rest of the same-event hard pool (scored as 'pool', not in the primary rank). The states are W0 and the END OF CYCLE 1 (chunk 279) of the arms NAT, NATADAMH, ORC, RAND of job 261 (saved weights, hash checked). Reading: OUT only (the neutral 8192-token SQuAD tail of job 261, built with the forbid list of the bank v1, so the context equals the OUT context of job 261), read-only, fp32 reader of E55.
Per item and state the summed log-prob of every candidate after the frame prefix (E49 convention, 'cand_lp'), the candidate token counts ('cand_ntok'), the per-token mean ('cand_mean' = cand_lp / cand_ntok; PRIMARY score), and for the arms the per-candidate per-token delta against W0 ('cand_delta_mean'; secondary). Rank / accuracy / margin are given for the per-token mean ('*_mean' fields; primary) and for the raw sums (no suffix).
  python3 e57_t1t2_run.py --cpu-check --frozen F --bank-dir D --manifest M [--tokenizer T] [--out O]       CPU only: frozen hash, items, tokenisation, state manifest (path + hash from the arms.json of job 261), selftest
  python3 e57_t1t2_run.py --bundle B --out O --bank bank_v1.json --cycles 1 --npos 1 --canary-npos 1 --arms NAT,NATADAMH,ORC,RAND --stage reads --frozen F --manifest M [--state-dir DIR]
Outputs O/E57_t1t2_scores.json (rows + settings) and O/E57_t1t2_score_keys.json (small key numbers)."""
import sys,json,time,hashlib,pathlib,os,re,collections
import numpy as np
HERE=pathlib.Path(__file__).resolve().parent
for p in (HERE,HERE/'e57',HERE.parents[1]/'scripts/exp04_what_ttt_retains',HERE.parents[1]/'scripts/exp05_write_dynamics'):sys.path.insert(0,str(p))
import e57_reads as RD
EXPECT_HASH_PREFIX='e3897ebd'
def jhash(o):return hashlib.sha256(json.dumps(o,sort_keys=True,ensure_ascii=False).encode()).hexdigest()
def load_frozen(path):
    b=json.load(open(path));h=b.pop('sha256');assert jhash(b)==h and h.startswith(EXPECT_HASH_PREFIX),('frozen file hash mismatch',h[:12]);b['sha256']=h;return b
def bank_like(frozen):
    """-> bank-format dict (RD.load_items input) with one question per frame; candidates exactly as frozen"""
    qs=[]
    for it in frozen['items']:
        if it['kind']=='T1':
            c=frozen['t1_candidates'][it['source']];gold=c['fixed_surface'];easy=c['easy'];hn=c['hard_nominated'];ha=c['hard_all'];typ=c['type']
        else:
            base=it['id'].replace('-cloze','');c=frozen['t2_candidates'][base];gold=c['fixed_surface'];easy=c['easy'];hn=c['hard_nominated'];ha=c['hard_all'];typ=c['type']
        qs.append(dict(id=it['id'],event=it['event'],type=typ,stratum=f"{it['kind']}-{it['form']}",frame=it['frame'],S='',fixed_surface=gold,canonical=gold,variants=list(it.get('variants',[]))+list(c.get('variants',[]))+[gold],easy=easy,hard_nominated=hn,hard_all=ha,source=it['source']))
    return dict(version='t1t2_bank_like',questions=qs)
def make_items(frozen,tmpdir):
    bl=bank_like(frozen);f=pathlib.Path(tmpdir)/'_t1t2_bank_like.json';f.parent.mkdir(parents=True,exist_ok=True);f.write_text(json.dumps(bl))
    items=RD.load_items(f);src={q['id']:q['source'] for q in bl['questions']}
    for it in items:it['source']=src[it['id']];it['kind_form']=it['stratum']
    return items
def metrics_mean(lp,ntok,it):
    lp=np.asarray(lp,np.float64);nt=np.asarray(ntok,np.float64);mean=lp/nt
    return mean,RD.metrics(mean,it['gold_idx'],it['hard_idx'],it.get('all_idx'),it.get('pool_idx'))
def rows_for_state(items,scores,ntoks,state,w0=None):
    rows=[]
    for k,(it,lp,nt) in enumerate(zip(items,scores,ntoks)):
        mean,mm=metrics_mean(lp,nt,it);ms=RD.metrics(lp,it['gold_idx'],it['hard_idx'],it.get('all_idx'),it.get('pool_idx'))
        r=dict(state=state,id=it['id'],source=it['source'],group=it['kind_form'],event=it['event'],type=it['type'],gold=it['cands'][0],cand_surface=it['cands'],cand_kinds=it['kinds'],cand_ntok=[int(x) for x in nt],cand_lp=[float(x) for x in lp],cand_mean=[float(x) for x in mean],
            cand_delta_mean=None if w0 is None else [float(a-b) for a,b in zip(mean,w0[k]['cand_mean'])],gold_delta_mean=None if w0 is None else float(mean[0]-w0[k]['cand_mean'][0]),
            **{f'{a}_mean':b for a,b in mm.items()},**ms)
        rows.append(r)
    return rows
def key_numbers(rows,meta):
    out={}
    for st in sorted({r['state'] for r in rows}):
        d={}
        for g in sorted({r['group'] for r in rows})+['ALL']:
            R=[r for r in rows if r['state']==st and (g=='ALL' or r['group']==g)]
            if not R:continue
            d[g]=dict(n=len(R),acc_all_mean=float(np.mean([r['acc_all_mean'] for r in R])),acc_hard_mean=float(np.mean([r['acc_hard_mean'] for r in R])),rank_ratio_all_mean=float(np.mean([r['rank_ratio_all_mean'] for r in R])),gold_mean_logprob_per_token=float(np.mean([r['cand_mean'][0] for r in R])),
                acc_all_sum=float(np.mean([r['acc_all'] for r in R])),gold_delta_mean_vs_W0=None if R[0]['gold_delta_mean'] is None else float(np.mean([r['gold_delta_mean'] for r in R])))
        out[st]=d
    return dict(meta,states=out)
def cpu_check(a):
    frozen=load_frozen(a.frozen);items=make_items(frozen,a.out or '/tmp');print('frozen hash ok',frozen['sha256'][:16],'| frames',len(frozen['items']),'| items',len(items),flush=True)
    c=collections.Counter(len(it['cands']) for it in items);print('candidates per item: min',min(c),'max',max(c),'| primary set (gold+easy+hard) per item: min',min(len(it['all_idx']) for it in items),'max',max(len(it['all_idx']) for it in items))
    print('removed duplicates:',{it['id']:it['removed_duplicates'] for it in items if it['removed_duplicates']} or 'none')
    ROOT=HERE.parents[1];import gen_world_ctl as G
    tfile=a.tokenizer or str(ROOT/'data/tokenizer/llama3/tokenizer.json');from tokenizers import Tokenizer;tk=Tokenizer.from_file(tfile)
    raw,ntok,bad=RD.build_rows(tk,items);print('rows',len(raw),'| candidates that failed tokenisation:',bad or 'none',flush=True)
    assert not bad,bad
    nt=[x for l in ntok for x in l];print('candidate tokens: mean %.2f max %d'%(np.mean(nt),max(nt)))
    man=json.load(open(a.manifest)) if a.manifest and pathlib.Path(a.manifest).exists() else None
    if man:
        for arm,s in man.items():
            p=pathlib.Path(s['file']);print(f'state {arm}: {p} exists={p.exists()} size={p.stat().st_size if p.exists() else None} hash={s["hash"][:12]}')
    selftest();return dict(items=len(items),rows=len(raw))
def make_manifest(arms_dir,out,arms=('NAT','NATADAMH','ORC','RAND'),chunk='279'):
    m={}
    for arm in arms:
        A=json.load(open(pathlib.Path(arms_dir)/f'E57_main_{arm}.arms.json'));s=A['states'][chunk];m[arm]=dict(file=s['file'],hash=s['hash'])
    pathlib.Path(out).write_text(json.dumps(m,indent=1));return m
def t1t2_reads(a,out,items,bsrc,iv,hits,shtab,ids_all,n,N,kks,arms,tk,LD,rt,R,cw0,lg,can):
    """replaces e57_main_run.stage_reads: the frozen T1/T2 items, OUT only, W0 and the cycle-1 states"""
    import e55_gpu as RG
    frozen=load_frozen(F['frozen']);new=make_items(frozen,out);man=json.load(open(F['manifest']))
    rt32=LD.runtime_fp32();rd=RG.Reader55(rt,rt32,LD.packing,LD._TailCtx,LD.reuse.get_filter_spec,LD.reuse._hash_pytree,LD.jax,LD.eqx,LD.jnp)
    raw,ntok,bad=RD.build_rows(rt.tok,new);assert not bad,bad
    forbid=sorted({s for it in items for s in it['variants']+[it['cands'][0]] if len(s)>=4 and not any(ch.isdigit() for ch in s)})       # the forbid list of the bank v1 = the OUT tail of job 261
    out_tail=R.out_tail_8192(LD,rt.tok,rt.bos,forbid)
    GROUP=40;groups=[];o=0                                                          # packs of about 40 items (about the 1300 rows of one pack in job 261); 4551 rows in one pack were never run
    for g0 in range(0,len(new),GROUP):
        sub=new[g0:g0+GROUP];nr=sum(len(it['cands']) for it in sub);groups.append((sub,rd.packed(raw[o:o+nr],out_tail)));o+=nr
    assert o==len(raw),(o,len(raw));tail_txt=tk.decode([int(x) for x in out_tail])
    gold_strings=sorted({s for it in new for s in it['variants']+[it['cands'][0]] if len(s)>=4 and not any(ch.isdigit() for ch in s)});in_tail=[s for s in gold_strings if s in tail_txt]
    lg(f'OUT tail: {len(out_tail)} tokens; sha {hashlib.sha256(np.asarray(out_tail,np.int64).tobytes()).hexdigest()[:16]}; frozen gold / variant strings found inside the tail: {in_tail or "none"}')
    def score(m):
        m32=rd.to_fp32(m)
        out_=[]
        with LD.jax.default_matmul_precision('highest'):
            for sub,P in groups:out_+=RD.split_scores(np.asarray(P.r1(m32)),sub)
        return out_
    t0=time.monotonic();w0rows=rows_for_state(new,score(rt.base),ntok,'W0');lg(f'  W0 read: {time.monotonic()-t0:.0f}s');allrows=list(w0rows);tim={'W0':time.monotonic()-t0}
    pathlib.Path(out).mkdir(parents=True,exist_ok=True);part=pathlib.Path(out)/'E57_t1t2_scores_partial.json'
    def save_partial():part.write_text(json.dumps(dict(done=sorted({r['state'] for r in allrows}),rows=allrows),default=float))
    save_partial()
    live_dt=inner_dtypes_(LD,rd,rt.base)
    for arm in arms:
        s=man[arm];f=pathlib.Path(F['state_dir'])/pathlib.Path(s['file']).name if F['state_dir'] else pathlib.Path(s['file'])
        lg(f'  {arm}: loading {f} ...');t0=time.monotonic();m=rd.load_state(str(f),expect_hash=s['hash']);lg(f'  {arm}: state loaded and hash checked in {time.monotonic()-t0:.0f}s');dt=inner_dtypes_(LD,rd,m);assert dt==live_dt,('saved weights have another precision than the training state',dt,live_dt)
        allrows+=rows_for_state(new,score(m),ntok,arm,w0=w0rows);del m;tim[arm]=time.monotonic()-t0;save_partial();lg(f'  {arm} (end of cycle 1) read: {tim[arm]:.0f}s (partial rows saved)')
    meta=dict(frozen_sha256=frozen['sha256'],draft_version=frozen.get('draft_version'),bank_v1_sha256=frozen['bank_v1_sha256'],states={k:v['hash'] for k,v in man.items()},context='OUT (8192-token SQuAD tail of job 261)',out_tail_gold_strings_inside=in_tail,primary_score='cand_mean = cand_lp / cand_ntok (gold against gold+easy+nominated hard)',n_items=len(new),timings=tim,saved_weights_precision=live_dt)
    pathlib.Path(out).mkdir(parents=True,exist_ok=True);(pathlib.Path(out)/'E57_t1t2_scores.json').write_text(json.dumps(dict(meta=meta,rows=allrows),default=float))
    kn=key_numbers(allrows,meta);(pathlib.Path(out)/'E57_t1t2_score_keys.json').write_text(json.dumps(kn,indent=1,default=float));lg('wrote E57_t1t2_scores.json and E57_t1t2_score_keys.json');lg(json.dumps({s:{g:(v['n'],round(v['acc_all_mean'],3)) for g,v in d.items()} for s,d in kn['states'].items()}))
def inner_dtypes_(LD,rd,m):
    import e57_main_run as M;return M.inner_dtypes(LD,rd,m)
def selftest():
    items=[dict(id='a',source='s',kind_form='T1-question',event='e',type='PERSON',stratum='T1-question',cands=['g','x','y','z'],kinds=['gold','easy','hard','pool'],gold_idx=0,hard_idx=[2],all_idx=[0,1,2],pool_idx=[2,3])]
    w0=rows_for_state(items,[np.array([-4.0,-6.0,-5.0,-9.0])],[[2,3,1,3]],'W0');assert abs(w0[0]['cand_mean'][0]+2.0)<1e-9 and w0[0]['acc_all_mean'] and w0[0]['gold_delta_mean'] is None
    st=rows_for_state(items,[np.array([-3.0,-6.0,-5.0,-9.0])],[[2,3,1,3]],'NAT',w0=w0);assert abs(st[0]['gold_delta_mean']-0.5)<1e-9 and st[0]['rank_all_mean']==1 and abs(st[0]['cand_delta_mean'][1])<1e-9
    st2=rows_for_state(items,[np.array([-4.0,-3.0,-5.0,-9.0])],[[2,3,1,3]],'X',w0=w0);assert st2[0]['rank_all_mean']==2 and st2[0]['acc_all']==False
    kn=key_numbers(w0+st,dict(x=1));assert kn['states']['NAT']['ALL']['n']==1 and kn['states']['W0']['ALL']['gold_delta_mean_vs_W0'] is None
    assert jhash({'a':1})==jhash({'a':1})
    print('e57_t1t2_run selftest ok')
F={'frozen':None,'manifest':None,'state_dir':None}
def _pop(flag,default=None):
    if flag in sys.argv:
        i=sys.argv.index(flag);v=sys.argv[i+1];del sys.argv[i:i+2];return v
    return default
def main():
    if '--cpu-check' in sys.argv:
        sys.argv.remove('--cpu-check');import argparse
        ap=argparse.ArgumentParser();ap.add_argument('--frozen',required=True);ap.add_argument('--bank-dir');ap.add_argument('--manifest');ap.add_argument('--tokenizer');ap.add_argument('--out');ap.add_argument('--make-manifest-from');a=ap.parse_args()
        if a.make_manifest_from:make_manifest(a.make_manifest_from,a.manifest);print('manifest written',a.manifest)
        cpu_check(a);return
    if '--selftest' in sys.argv:selftest();return
    F['frozen']=_pop('--frozen');F['manifest']=_pop('--manifest');F['state_dir']=_pop('--state-dir');assert F['frozen'] and F['manifest'],'--frozen and --manifest are required'
    import e57_main_run as M
    M.stage_reads=t1t2_reads;M.main()
if __name__=='__main__':main()
