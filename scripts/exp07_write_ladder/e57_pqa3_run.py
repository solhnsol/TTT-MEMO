"""E57 public QA stage 3 (E57 122-R / 123; new file; GPU): the accumulating re-description stream (e57_pqa3_stream.py: filler / original / re-description 1..4, 280 chunks, written ONCE) is written by NAT (plain SGD; its per-chunk update norms are the norm targets of NATQ), NATQ (engine NATADAMH, Adam direction, 0.5 x NAT norm, no mask) and ALLQ128 (SELC1 + SelMaskQ on all rows, band 128; the settings of jobs 570 / 600). The state is saved and the canary (generation / tail NLL, measured only) is read at the end of each of the six sections (the chunk that holds the last token of the section). Reads: the 1,000 frozen stage-1 SQuAD items x 16 candidates are scored OUT (8192-token stage-1 tail, FastOUT with the 640-row packing of stage 1 and job 600, e57_pqa2_run.FastOUT) for W0 and for every saved state; the paragraphs' 455 items (373 'main' = the first n_main item ids of each paragraph) are the evaluation subsets, all 1,000 are scored and kept so that the packing (and with it every score) stays the same as in jobs 440 / 600.
  python3 e57_pqa3_run.py --stage write --stream-dir D --out O [--arms NAT,NATQ,ALLQ128] | --stage manifest --out O --manifest M | --stage read --out O --manifest M --paragraphs P.json --items-file F --bank B [--stage1 S1.json] | --selftest"""
import sys,json,time,gc,hashlib,pathlib
import numpy as np
HERE=pathlib.Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
import e57_pqa2_run as P2
import e57_pqa3_stream as S3
CH=1024;ARMS=['NAT','NATQ','ALLQ128'];ENGINE={'NAT':'NAT','NATQ':'NATADAMH','ALLQ128':'SELC1'};SECS=S3.SECTIONS
def write_stage(a):
    out=pathlib.Path(a['out']);import e55_swap as E,e55_run as R,e57_qband_run as QR
    E.ADAM['NATADAMH']=('NAT',0.5,False)
    if 'NATADAMH' not in E.ARMS:E.ARMS.append('NATADAMH')
    LD,s3,C=R.lab();rt=LD.Runtime();can=R.Canary55(LD,rt);lg=lambda *x:print(*x,flush=True);cw0=can(rt.base);lg('canary W0',cw0)
    ids,meta=S3.load(a['stream_dir']);nch=len(ids)//CH;assert nch==meta['n_chunks'];rch=meta['read_chunks'];ends=set(rch.values());lg(f'stream P3: {len(ids)} tokens, {nch} chunks, read chunks {rch}, ids_sha {meta["ids_sha"]}')
    masks={'band_ALLQ128':np.asarray(128),'all_ALLQ128':np.asarray(1)};rc={};L0=C.Ladder55(rt,save_dir=out/'states'/'rec',canary=None);t0=time.monotonic()
    with rt.mesh:Kh0,Vh0,_h=L0.record(ids[:nch*CH])
    del _h;rc[(hashlib.sha256(np.asarray(ids[:nch*CH],np.int64).tobytes()).hexdigest(),nch)]=(Kh0,Vh0);lg(f'  natural K,V recorded in {time.monotonic()-t0:.0f}s');del L0;gc.collect();LD.jax.clear_caches()
    ntg=None
    for arm in a['arms']:
        fnA=out/f'E57_pqa3_{arm}.arms.json'
        if fnA.exists() and not a.get('rerun'):
            if arm=='NAT':ntg=json.load(open(fnA))['norm_targets']
            print('exists, reused:',fnA);continue
        if arm!='NAT' and ntg is None:
            f=out/'E57_pqa3_NAT.arms.json';assert f.exists(),'NAT (norm targets) must be available first';ntg=json.load(open(f))['norm_targets']
        L=C.Ladder55(rt,save_dir=out/'states'/f'E57_pqa3_{arm}',canary=None);L.rec_cache=rc;traj=[]
        uv=QR.SelMaskQ('ALLQ128',masks,lg) if arm=='ALLQ128' else None
        def hook(k,m,arm=arm,traj=traj):
            cn=can(m);traj.append(dict(chunk=int(k),canary=cn));lg(f'  {arm} chunk {k}: canary rise gen {cn["_gen_nll"]-cw0["_gen_nll"]:+.4f} tail {cn["_tail_nll"]-cw0["_tail_nll"]:+.4f}');return {'ok':True}
        t0=time.monotonic()
        res=L.run_stream(ENGINE[arm],1,ids,nch=nch,tag=f'pqa3{arm}',log=lg,read_hook=hook,hook_chunks=ends,read_chunks=ends,chunks_meta=[{'sents':[]} for _ in range(nch)],norm_targets=ntg,uv=uv);log=res['log']
        if arm=='NAT':ntg=[e['update_norm'] for e in log]
        fnA.write_text(json.dumps(dict(arm=arm,engine=ENGINE[arm],stream='P3',n_chunks=nch,read_chunks=rch,ids_sha=meta['ids_sha'],canary_w0=cw0,trajectory=traj,states={str(k):{'file':v['_file'],'hash':v['_hash']} for k,v in res['reads'].items()},sec_total=time.monotonic()-t0,
            update_log=[{k_:e[k_] for k_ in ('k','ce','update_norm','coef','sec','n_masked_keys','q_banded','q_band') if k_ in e} for e in log],norm_targets=ntg if arm=='NAT' else None),default=float));print('wrote',fnA,flush=True)
        del L,res;gc.collect();LD.jax.clear_caches()
def manifest(out,arms):
    """{state name 'ARM@section': {label, section, chunk, file, hash}} in the read order (NATQ, ALLQ128 first, NAT last)"""
    out=pathlib.Path(out);m={}
    for arm in arms:
        d=json.load(open(out/f'E57_pqa3_{arm}.arms.json'))
        for sec in SECS:
            k=str(d['read_chunks'][sec]);s=d['states'][k];m[f'{arm}@{sec}']=dict(label=arm,section=sec,chunk=int(k),file=s['file'],hash=s['hash'])
    return m
def subsets(paragraphs):
    ids=[];main=[]
    for p in paragraphs:
        ids+=p['item_ids'];main+=p['item_ids'][:p['n_main']]
    assert len(set(ids))==len(ids);return dict(items455=set(ids),main373=set(main))
def read_stage(a):
    out=pathlib.Path(a['out']);out.mkdir(parents=True,exist_ok=True);lg=lambda *x:print(*x,flush=True)
    paras=json.load(open(a['paragraphs']));sub=subsets(paras);LD,rt,rt32,rd,items,h,raw,ntok,out_tail,tsha=P2.setup_read(a,lg);man=json.load(open(a['manifest']));m32w0=rd.to_fp32(rt.base)
    assert sub['items455']<={it['id'] for it in items}
    t0=time.monotonic();fo=P2.FastOUT(rt32,m32w0,out_tail,raw,log=lg);tim={'prep':time.monotonic()-t0};rows={}
    t0=time.monotonic();f0=fo.score(m32w0);rows['W0']=P2.rows_of(items,ntok,f0,'W0');tim['W0']=time.monotonic()-t0
    ref1=json.load(open(a.get('stage1') or 'experiments/exp07_write_ladder/e57_squad_w0/E57_squad_w0_reads.json'))['rows'];ref1=np.concatenate([np.asarray(r['cand_lp'],np.float64) for r in ref1 if r['condition']=='OUT']);d1=np.abs(f0-ref1)
    chk=dict(max_abs_diff=float(d1.max()),mean_abs_diff=float(d1.mean()));lg('  FastOUT W0 vs stored stage-1 OUT rows (all 1000 items):',json.dumps(chk))
    def save_rows(lab):(out/f'E57_pqa3_rows_{lab.replace("@","_")}.json').write_text(json.dumps(rows[lab],default=float))
    save_rows('W0')
    for name,s in man.items():
        t0=time.monotonic();m=rd.load_state(s['file'],expect_hash=s['hash']);rows[name]=P2.rows_of(items,ntok,fo.score(rd.to_fp32(m)),name);del m;tim[name]=time.monotonic()-t0;save_rows(name)
        sel=[r for r in rows[name] if r['id'] in sub['items455']];lg(f'  {name} read: {tim[name]:.0f}s acc_mean(455) {np.mean([r["acc_mean"] for r in sel]):.3f}')
    kn=keys(rows,sub,man,items)
    meta=dict(items_sha256=h,out_tail_sha=tsha,scorer='FastOUT (KV reuse, packed in groups of 640 rows as stage 1)',w0_vs_stage1_stored=chk,states=man,timings=tim,subsets={k:len(v) for k,v in sub.items()},primary_score='cand_mean = cand_lp / cand_ntok')
    (out/'E57_pqa3_keys.json').write_text(json.dumps(dict(meta=meta,keys=kn),indent=1,default=float));lg('wrote',out/'E57_pqa3_keys.json')
def keys(rows,sub,man,items):
    """per subset (all1000 / items455 / main373) and per read point (section): the arms' accuracy / rank ratio with bootstrap CI and paired differences to W0 and to NAT; plus the curve acc_mean over the read points"""
    allids=[it['id'] for it in items];out={}
    for sn,idset in [('all1000',set(allids)),('items455',sub['items455']),('main373',sub['main373'])]:
        sel=lambda R:[r for r in R if r['id'] in idset];d={'by_section':{},'curve_acc_mean':{}}
        for sec in SECS:
            rb={'W0':sel(rows['W0'])}
            for arm in ARMS:
                if f'{arm}@{sec}' in rows:rb[arm]=sel(rows[f'{arm}@{sec}'])
            d['by_section'][sec]=P2.keys_paired(rb)
        for arm in ARMS:
            d['curve_acc_mean'][arm]=[float(np.mean([r['acc_mean'] for r in sel(rows[f'{arm}@{sec}'])])) if f'{arm}@{sec}' in rows else None for sec in SECS]
        d['curve_acc_mean']['W0']=float(np.mean([r['acc_mean'] for r in sel(rows['W0'])]));out[sn]=d
    return out
def selftest():
    S3.selftest();P2.selftest()
    paras=[dict(item_ids=['a','b','c'],n_main=2),dict(item_ids=['d'],n_main=1)];s=subsets(paras);assert s['items455']=={'a','b','c','d'} and s['main373']=={'a','b','d'}
    print('e57_pqa3_run selftest ok')
def main():
    if '--selftest' in sys.argv:selftest();return
    stage=P2._pop('--stage');rerun='--rerun' in sys.argv
    if rerun:sys.argv.remove('--rerun')
    a=dict(stream_dir=P2._pop('--stream-dir'),out=P2._pop('--out'),arms=(P2._pop('--arms') or ','.join(ARMS)).split(','),manifest=P2._pop('--manifest'),items=P2._pop('--items-file'),bank=P2._pop('--bank'),stage1=P2._pop('--stage1'),paragraphs=P2._pop('--paragraphs'),rerun=rerun)
    if stage=='manifest':
        m=manifest(a['out'],['NATQ','ALLQ128','NAT']);json.dump(m,open(a['manifest'],'w'),indent=1);print({k:(v['chunk'],v['hash'][:12]) for k,v in m.items()});return
    {'write':write_stage,'read':read_stage}[stage](a)
if __name__=='__main__':main()
