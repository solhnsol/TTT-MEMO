"""E57 T1/T2 v2 (frozen 98 reverse questions, 30 events; stream A = 37 items on the stream of e57_stream_sentences.jsonl (280 chunks), stream B20 = 61 items on streamB20_stream_sentences.jsonl (286 chunks)): writes and reads of W0 / NAT / NATQ / ALLQ128 (one news cycle per stream, GPU, read only, raw scores, no thresholds; new file).
Streams: the token ids are the fragment texts of the jsonl re-encoded one by one (checked: sha of the id array = ids_sha of the file meta: A 715077dff4f602c6, B20 13089ac9f8c1227f).
Writes (--stage write, per stream): NAT = e55_core arm NAT (plain SGD, saves the per-chunk update norms = norm targets); NATQ = engine NATADAMH (Adam direction, update norm 0.5 x NAT norm, no mask; the unmasked control arm of jobs 560 / 565); ALLQ128 = engine SELC1 with the e57_qband_run.SelMaskQ of band 128 on ALL query rows (chunks >= 8; update norm 0.5 x NAT, Adam direction as ALLQ512 of job 560). States of the last chunk (cycle end) are saved, canary read there. Stream A: NAT / NATQ states already exist (E57_main_NAT 5f235dd9efb9, NATADAMH 23fc03c9f71f), only ALLQ128 is written and the norm targets come from E57_main_NAT.arms.json.
Reads (--stage read): the frozen file t1t2_v2_frozen.json (sha256 f459c1a0...) with the candidate file e57_t1t2v2d3_candidates.json: per item the forms 'question' and 'cloze' as frames, candidates = gold (fixed_surface first) + the 20 easy candidates of the candidate file + the nominated hard candidates of the frozen item (kinds gold / easy / hard; the same bank-like conversion and scoring as e57_t1t2_run.py of jobs 276 / 410: summed log-prob and per-token mean of the candidate after the frame prefix, OUT context = the 8192-token SQuAD tail of the bank-v1 forbid list, fp32, accuracy and rank ratio on the per-token mean (primary) and on the sums); the forward_cloze frames are NOT scored (the frozen file gives no gold for them). W0 on both streams, every state only on the items of its own stream. Outputs: E57_t1t2v2_scores.json (rows) and E57_t1t2v2_score_keys.json (per state and stream and form; and combined over the two streams per event, equal weight per event).
  python3 e57_t1t2v2_run.py --stage write --stream A|B20 --sentences S.jsonl --out O [--nat-ref E57_main_NAT.arms.json] --arms NAT,NATQ,ALLQ128
  python3 e57_t1t2v2_run.py --stage read --frozen F --cands C --bank bank_v1.json --manifest M.json --out O
  python3 e57_t1t2v2_run.py --cpu-check --frozen F --cands C --bank bank_v1.json [--manifest M] [--sentences-a SA --sentences-b SB]  |  --selftest"""
import sys,json,time,gc,hashlib,pathlib,os,collections
import numpy as np
HERE=pathlib.Path(__file__).resolve().parent
for p in (HERE,HERE/'e57',HERE.parents[1]/'scripts/exp04_what_ttt_retains',HERE.parents[1]/'scripts/exp05_write_dynamics'):sys.path.insert(0,str(p))
sys.path.insert(0,str(HERE.parents[1]/'e2e'))
import e57_reads as RD
import e57_t1t2_run as T1
CH=1024;FROZEN_SHA='f459c1a08b60eeb8e1adf01f766deff1e9574328912d30eb64147801b8b04d0f';STREAM_SHA={'A':'715077dff4f602c6','B20':'13089ac9f8c1227f'}
def stream_ids(path,stream):
    from tokenizers import Tokenizer
    tk=Tokenizer.from_file(str(HERE.parents[1]/'data/tokenizer/llama3/tokenizer.json'));ids=[]
    for l in open(path):
        r=json.loads(l)
        if r.get('type')=='frag':
            t=tk.encode(r['text'],add_special_tokens=False).ids;assert len(t)==r['ntok'],(r['fid'],len(t),r['ntok']);ids+=t
    a=np.asarray(ids,np.int64);sha=hashlib.sha256(a.tobytes()).hexdigest()[:16];assert sha==STREAM_SHA[stream],('stream ids sha',stream,sha);assert len(a)%CH==0,len(a);return a
def load_frozen(path):
    raw=open(path,'rb').read();assert hashlib.sha256(raw).hexdigest()==FROZEN_SHA,'frozen file sha256 differs';return json.loads(raw)
def bank_like_v2(frozen,cands):
    cd={c['id']:c for c in cands};qs=[]
    for it in frozen['items']:
        c=cd[it['id']];assert c['gold']==it['gold'] and c['fixed_surface']==it['gold'] and c['event']==it['event']
        hn=[dict(surface=h['surface']) for h in it['hard_nominated']]
        for f in it['forms']:
            qs.append(dict(id=f"{it['id']}-{f['form']}",event=it['event'],type=it['type'],stratum=f"T2-{f['form']}",frame=f['frame'],S='',fixed_surface=it['gold'],canonical=it['gold'],variants=list(it['variants']),easy=[dict(surface=e['surface']) for e in c['easy']],hard_nominated=hn,hard_all=[],source=it['id'],stream=c['stream']))
    return dict(version='t1t2v2_bank_like',questions=qs)
def make_items(frozen,cands,tmpdir):
    bl=bank_like_v2(frozen,cands);f=pathlib.Path(tmpdir)/'_t1t2v2_bank_like.json';f.parent.mkdir(parents=True,exist_ok=True);f.write_text(json.dumps(bl))
    items=RD.load_items(f);meta={q['id']:q for q in bl['questions']}
    for it in items:it['source']=meta[it['id']]['source'];it['kind_form']=it['stratum'];it['stream']=meta[it['id']]['stream']
    return items
def write_stage(args):
    stream=args['stream'];arms=args['arms'];out=pathlib.Path(args['out']);out.mkdir(parents=True,exist_ok=True)
    import e55_swap as E
    E.ADAM['NATADAMH']=('NAT',0.5,False)
    if 'NATADAMH' not in E.ARMS:E.ARMS.append('NATADAMH')
    import e55_run as R
    LD,s3,C=R.lab();rt=LD.Runtime();can=R.Canary55(LD,rt);lg=lambda *x:print(*x,flush=True);cw0=can(rt.base);lg('canary W0',cw0)
    import jax
    ids=stream_ids(args['sentences'],stream);nch=len(ids)//CH;lg(f'stream {stream}: {len(ids)} tokens, {nch} chunks, ids sha ok')
    import e57_qband_run as QR
    masks={'band_ALLQ128':np.asarray(128),'all_ALLQ128':np.asarray(1)}
    dummy_meta=[{'sents':[]} for _ in range(nch)];rc={}
    L0=C.Ladder55(rt,save_dir=out/'states'/'rec',canary=None);t0=time.monotonic()
    with rt.mesh:Kh0,Vh0,_h=L0.record(ids[:nch*CH])
    del _h;rc[(hashlib.sha256(np.asarray(ids[:nch*CH],np.int64).tobytes()).hexdigest(),nch)]=(Kh0,Vh0);lg(f'  natural K,V recorded in {time.monotonic()-t0:.0f}s');del L0;gc.collect();LD.jax.clear_caches()
    ntg=None
    if args.get('nat_ref'):ntg=list(json.load(open(args['nat_ref']))['norm_targets'][:nch]);assert len(ntg)==nch,len(ntg);lg('norm targets from',args['nat_ref'])
    for arm in arms:
        fnA=out/f'E57_v2{stream}_{arm}.arms.json'
        if fnA.exists() and not args.get('rerun'):
            if arm=='NAT':ntg=json.load(open(fnA))['norm_targets']
            print('exists, reused:',fnA);continue
        engine={'NAT':'NAT','NATQ':'NATADAMH','ALLQ128':'SELC1'}[arm]
        if engine!='NAT':assert ntg is not None,'NAT (norm targets) must be available first'
        L=C.Ladder55(rt,save_dir=out/'states'/f'E57_v2{stream}_{arm}',canary=None);L.rec_cache=rc;traj=[]
        uv=QR.SelMaskQ('ALLQ128',masks,lg) if arm=='ALLQ128' else None
        def hook(k,m,arm=arm,traj=traj):
            cn=can(m);traj.append(dict(chunk=int(k),canary=cn));lg(f'  {arm} chunk {k}: canary rise gen {cn["_gen_nll"]-cw0["_gen_nll"]:+.4f} tail {cn["_tail_nll"]-cw0["_tail_nll"]:+.4f}');return {'ok':True}
        t0=time.monotonic()
        res=L.run_stream(engine,1,ids,nch=nch,tag=f'v2{stream}{arm}',log=lg,read_hook=hook,hook_chunks={nch-1},read_chunks={nch-1},chunks_meta=dummy_meta,norm_targets=ntg,uv=uv);log=res['log']
        if arm=='NAT':ntg=[e['update_norm'] for e in log]
        fnA.write_text(json.dumps(dict(arm=arm,engine=engine,stream=stream,n_chunks=nch,ids_sha=STREAM_SHA[stream],canary_w0=cw0,trajectory=traj,states={str(k):{'file':v['_file'],'hash':v['_hash']} for k,v in res['reads'].items()},sec_total=time.monotonic()-t0,
            update_log=[{k_:e[k_] for k_ in ('k','ce','update_norm','coef','sec','n_masked_keys','q_banded','q_band') if k_ in e} for e in log],norm_targets=ntg if arm=='NAT' else None),default=float));print('wrote',fnA,flush=True)
        del L,res;gc.collect();LD.jax.clear_caches()
def aggregate(rows):
    """key numbers per state / stream / form and combined over the streams per event (equal weight per event)"""
    out={'by_stream':{},'combined_by_event':{}}
    def agg(R):return dict(n=len(R),acc_all_mean=float(np.mean([r['acc_all_mean'] for r in R])),rank_ratio_all_mean=float(np.mean([r['rank_ratio_all_mean'] for r in R])),acc_hard_mean=float(np.mean([r['acc_hard_mean'] for r in R])),acc_all_sum=float(np.mean([r['acc_all'] for r in R])),gold_mean_logprob_per_token=float(np.mean([r['cand_mean'][0] for r in R])))
    for st in sorted({r['state_label'] for r in rows}):
        for sm in sorted({r['stream'] for r in rows if r['state_label']==st}):
            d={}
            for g in sorted({r['group'] for r in rows})+['ALL']:
                R=[r for r in rows if r['state_label']==st and r['stream']==sm and (g=='ALL' or r['group']==g)]
                if R:d[g]=agg(R)
            out['by_stream'].setdefault(st,{})[sm]=d
        ev=collections.defaultdict(list)
        for r in rows:
            if r['state_label']==st:ev[r['event']].append(r)
        d={}
        for g in sorted({r['group'] for r in rows})+['ALL']:
            per=[agg([r for r in R if g=='ALL' or r['group']==g]) for e,R in sorted(ev.items()) if any(g=='ALL' or r['group']==g for r in R)]
            if per:d[g]=dict(n_events=len(per),acc_all_mean=float(np.mean([p['acc_all_mean'] for p in per])),rank_ratio_all_mean=float(np.mean([p['rank_ratio_all_mean'] for p in per])),acc_hard_mean=float(np.mean([p['acc_hard_mean'] for p in per])),acc_all_sum=float(np.mean([p['acc_all_sum'] for p in per])),n_rows=int(sum(p['n'] for p in per)))
        out['combined_by_event'][st]=d
    return out
def read_stage(args):
    frozen=load_frozen(args['frozen']);cands=json.load(open(args['cands']));out=pathlib.Path(args['out']);out.mkdir(parents=True,exist_ok=True)
    new=make_items(frozen,cands,out);man=json.load(open(args['manifest']))
    import e55_swap as E
    E.ADAM['NATADAMH']=('NAT',0.5,False)
    if 'NATADAMH' not in E.ARMS:E.ARMS.append('NATADAMH')
    import e55_run as R,e55_gpu as RG,e57_main_run as M
    LD,s3,C=R.lab();rt=LD.Runtime();lg=lambda *x:print(*x,flush=True);can=R.Canary55(LD,rt);lg('canary W0',can(rt.base))
    bank=RD.load_items(args['bank'])
    rt32=LD.runtime_fp32();rd=RG.Reader55(rt,rt32,LD.packing,LD._TailCtx,LD.reuse.get_filter_spec,LD.reuse._hash_pytree,LD.jax,LD.eqx,LD.jnp)
    raw,ntok,bad=RD.build_rows(rt.tok,new);assert not bad,bad
    forbid=sorted({s for it in bank for s in it['variants']+[it['cands'][0]] if len(s)>=4 and not any(ch.isdigit() for ch in s)});out_tail=R.out_tail_8192(LD,rt.tok,rt.bos,forbid)
    tail_txt=rt.tok.decode([int(x) for x in out_tail]) if hasattr(rt.tok,'decode') else ''
    gold_strings=sorted({s for it in new for s in it['variants']+[it['cands'][0]] if len(s)>=4 and not any(ch.isdigit() for ch in s)});in_tail=[s for s in gold_strings if s in tail_txt]
    lg(f'OUT tail: {len(out_tail)} tokens; sha {hashlib.sha256(np.asarray(out_tail,np.int64).tobytes()).hexdigest()[:16]}; frozen gold / variant strings found inside the tail: {in_tail or "none"}')
    # item-major rows; packs per stream
    off=[0]
    for it in new:off.append(off[-1]+len(it['cands']))
    packs={}
    for sm in ('A','B'):
        idx=[k for k,it in enumerate(new) if it['stream']==sm];groups=[]
        for g0 in range(0,len(idx),40):
            sub=idx[g0:g0+40];rr=[r for k in sub for r in raw[off[k]:off[k+1]]];groups.append(([new[k] for k in sub],rd.packed(rr,out_tail)))
        packs[sm]=dict(idx=idx,groups=groups,ntok=[ntok[k] for k in idx],items=[new[k] for k in idx]);lg(f'  stream {sm}: {len(idx)} frames in {len(groups)} packs')
    def score(m,sm):
        m32=rd.to_fp32(m);o=[]
        with LD.jax.default_matmul_precision('highest'):
            for sub,P in packs[sm]['groups']:o+=RD.split_scores(np.asarray(P.r1(m32)),sub)
        return o
    allrows=[];w0r={};tim={};t0=time.monotonic()
    for sm in ('A','B'):
        rows=T1.rows_for_state(packs[sm]['items'],score(rt.base,sm),packs[sm]['ntok'],'W0');w0r[sm]=rows
        for r in rows:r['stream']=sm;r['state_label']='W0'
        allrows+=rows
    tim['W0']=time.monotonic()-t0;lg(f'  W0 read: {tim["W0"]:.0f}s');part=out/'E57_t1t2v2_scores_partial.json'
    live=M.inner_dtypes(LD,rd,rt.base)
    for name,s in man.items():
        sm=s['stream'];label=s['label'];t0=time.monotonic();m=rd.load_state(s['file'],expect_hash=s['hash']);dt=M.inner_dtypes(LD,rd,m);assert dt==live,('saved weights have another precision',dt,live)
        rows=T1.rows_for_state(packs[sm]['items'],score(m,sm),packs[sm]['ntok'],name,w0=w0r[sm])
        for r in rows:r['stream']=sm;r['state_label']=label
        allrows+=rows;del m;tim[name]=time.monotonic()-t0;lg(f'  {name} (stream {sm}) read: {tim[name]:.0f}s');part.write_text(json.dumps(dict(done=list(tim),rows=allrows),default=float))
    meta=dict(frozen_sha256=FROZEN_SHA,cands_sha256=hashlib.sha256(open(args['cands'],'rb').read()).hexdigest(),states={k:dict(v) for k,v in man.items()},context='OUT (8192-token SQuAD tail, forbid list of bank v1)',out_tail_gold_strings_inside=in_tail,primary_score='cand_mean = cand_lp / cand_ntok (gold against gold+easy+nominated hard)',forward_cloze='not scored (no gold in the frozen file)',n_frames=len(new),timings=tim)
    (out/'E57_t1t2v2_scores.json').write_text(json.dumps(dict(meta=meta,rows=allrows),default=float));kn=dict(meta=meta,**aggregate(allrows));(out/'E57_t1t2v2_score_keys.json').write_text(json.dumps(kn,indent=1,default=float));lg('wrote E57_t1t2v2_scores.json and E57_t1t2v2_score_keys.json')
    lg(json.dumps({st:{g:(v['n_events'],round(v['acc_all_mean'],3)) for g,v in d.items()} for st,d in kn['combined_by_event'].items()}))
def cpu_check(a):
    frozen=load_frozen(a['frozen']);cands=json.load(open(a['cands']));items=make_items(frozen,cands,a.get('out') or '/tmp')
    print('frozen sha ok | items',len(frozen['items']),'| frames',len(items),'| by stream',dict(collections.Counter(it['stream'] for it in items)),'| events',len({it['event'] for it in items}))
    c=collections.Counter(len(it['cands']) for it in items);print('candidates per frame: min',min(c),'max',max(c),'| removed duplicates',{it['id']:it['removed_duplicates'] for it in items if it['removed_duplicates']} or 'none')
    from tokenizers import Tokenizer;tk=Tokenizer.from_file(str(HERE.parents[1]/'data/tokenizer/llama3/tokenizer.json'));raw,ntok,bad=RD.build_rows(tk,items);print('rows',len(raw),'failed tokenisation:',bad or 'none');assert not bad
    for sm,key in (('A','sentences_a'),('B20','sentences_b')):
        if a.get(key):ids=stream_ids(a[key],sm);print(f'stream {sm}: {len(ids)} tokens, {len(ids)//CH} chunks, ids sha ok')
    if a.get('manifest'):
        for k,s in json.load(open(a['manifest'])).items():p=pathlib.Path(s['file']);print(f'state {k}: stream {s["stream"]} exists={p.exists()} size={p.stat().st_size if p.exists() else None} hash={s["hash"][:12]}')
def selftest():
    T1.selftest()
    rows=[]
    for sm,ev,acc in (('A','e1',1),('B','e1',0),('B','e2',1)):
        rows.append(dict(state_label='W0',stream=sm,group='T2-question',event=ev,acc_all_mean=acc,rank_ratio_all_mean=0.1*acc,acc_hard_mean=acc,acc_all=acc,cand_mean=[-1.0]))
    ag=aggregate(rows);assert ag['by_stream']['W0']['B']['ALL']['n']==2 and abs(ag['combined_by_event']['W0']['ALL']['acc_all_mean']-0.75)<1e-9 and ag['combined_by_event']['W0']['ALL']['n_events']==2
    print('e57_t1t2v2_run selftest ok')
def _pop(flag,default=None):
    if flag in sys.argv:
        i=sys.argv.index(flag);v=sys.argv[i+1];del sys.argv[i:i+2];return v
    return default
def main():
    if '--selftest' in sys.argv:selftest();return
    cpu='--cpu-check' in sys.argv
    if cpu:sys.argv.remove('--cpu-check')
    rerun='--rerun' in sys.argv
    if rerun:sys.argv.remove('--rerun')
    stage=_pop('--stage');a=dict(stream=_pop('--stream'),sentences=_pop('--sentences'),out=_pop('--out'),nat_ref=_pop('--nat-ref'),arms=(_pop('--arms') or '').split(',') if '--arms' in sys.argv else None,frozen=_pop('--frozen'),cands=_pop('--cands'),bank=_pop('--bank'),manifest=_pop('--manifest'),sentences_a=_pop('--sentences-a'),sentences_b=_pop('--sentences-b'),rerun=rerun)
    if cpu:cpu_check(a);return
    if stage=='write':write_stage(a)
    elif stage=='read':read_stage(a)
    else:raise SystemExit('--stage write|read')
if __name__=='__main__':main()
