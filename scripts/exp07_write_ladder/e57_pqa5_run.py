"""E57 public QA stage 5 (E57 122-U / 123; new file; GPU): nine-version accumulating stream (e57_pqa5_stream.py: filler / original / re-description 1..8, ten sections of 332 paragraphs, 453 chunks, written ONCE) written by NATQ and ALLQ128 (settings of jobs 570 / 600 / 610: NATQ = engine NATADAMH, Adam direction, 0.5 x NAT norm, no mask; ALLQ128 = SELC1 + SelMaskQ band 128 on all rows). NAT is written too because its per-chunk update norms are the norm targets of both arms, but its states are not read. State + canary (measured only) at the end of each of the ten sections (the chunk that holds the section's last token; the next section's first tokens are already in that chunk, spill tokens are in the stream meta).
Reads (FastOUT, 640-row packing of stage 1, OUT = stage-1 tail, the 1,000 frozen items x 16 candidates): A (own question) at all ten section ends; B (question of another item, assignment of job 620, seed 20261101) and C ("Answer:" only) at the ends of re-description 4 and re-description 8 only. W0 rows are the stored rows of jobs 610 (A) and 620 (B, C), not recomputed. Counts of the gold string and the 16 candidates in the stream (total and per section, 'cnt' substring and 'cnt_wb' between non-alphanumerics) are written into the item rows.
  python3 e57_pqa5_run.py --stage write --stream-dir D --out O | --stage manifest --out O | --stage read --cond A|B|C --out O --manifest M --dir610 D --dir620 D --items-file F --bank B | --stage analyze --out O --dir610 D --dir620 D --stream-dir D --paragraphs P --items-file F | --selftest"""
import sys,json,time,gc,hashlib,pathlib
import numpy as np
HERE=pathlib.Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
import e57_pqa2_run as P2
import e57_pqa3_run as P3
import e57_pqa4_run as P4
import e57_pqa5_stream as S5
import e57_squad_w0_reads as Q
CH=1024;SECS=S5.SECTIONS;ENGINE=P3.ENGINE;WARMS=['NAT','NATQ','ALLQ128'];RARMS=['NATQ','ALLQ128']
def write_stage(a):
    out=pathlib.Path(a['out']);import e55_swap as E,e55_run as R,e57_qband_run as QR
    E.ADAM['NATADAMH']=('NAT',0.5,False)
    if 'NATADAMH' not in E.ARMS:E.ARMS.append('NATADAMH')
    P2.patch_segmented()
    LD,s3,C=R.lab();rt=LD.Runtime();can=R.Canary55(LD,rt);lg=lambda *x:print(*x,flush=True);cw0=can(rt.base);lg('canary W0',cw0)
    ids,meta=S5.load(a['stream_dir']);nch=len(ids)//CH;assert nch==meta['n_chunks'];rch=meta['read_chunks'];ends=set(rch.values());lg(f'stream P5: {len(ids)} tokens, {nch} chunks, read chunks {rch}, ids_sha {meta["ids_sha"]}')
    masks={'band_ALLQ128':np.asarray(128),'all_ALLQ128':np.asarray(1)};rc={};L0=C.Ladder55(rt,save_dir=out/'states'/'rec',canary=None);t0=time.monotonic()
    with rt.mesh:Kh0,Vh0,_h=L0.record(ids[:nch*CH])
    del _h;rc[(hashlib.sha256(np.asarray(ids[:nch*CH],np.int64).tobytes()).hexdigest(),nch)]=(Kh0,Vh0);lg(f'  natural K,V recorded in {time.monotonic()-t0:.0f}s');del L0;gc.collect();LD.jax.clear_caches()
    ntg=None
    for arm in WARMS:
        fnA=out/f'E57_pqa5_{arm}.arms.json'
        if fnA.exists() and not a.get('rerun'):
            if arm=='NAT':ntg=json.load(open(fnA))['norm_targets']
            print('exists, reused:',fnA);continue
        if arm!='NAT' and ntg is None:
            f=out/'E57_pqa5_NAT.arms.json';assert f.exists(),'NAT (norm targets) must be available first';ntg=json.load(open(f))['norm_targets']
        L=C.Ladder55(rt,save_dir=out/'states'/f'E57_pqa5_{arm}',canary=None);L.rec_cache=rc;traj=[]
        uv=QR.SelMaskQ('ALLQ128',masks,lg) if arm=='ALLQ128' else None
        def hook(k,m,arm=arm,traj=traj):
            cn=can(m);traj.append(dict(chunk=int(k),canary=cn));lg(f'  {arm} chunk {k}: canary rise gen {cn["_gen_nll"]-cw0["_gen_nll"]:+.4f} tail {cn["_tail_nll"]-cw0["_tail_nll"]:+.4f}');return {'ok':True}
        t0=time.monotonic()
        res=L.run_stream(ENGINE[arm],1,ids,nch=nch,tag=f'pqa5{arm}',log=lg,read_hook=hook,hook_chunks=ends,read_chunks=ends,chunks_meta=[{'sents':[]} for _ in range(nch)],norm_targets=ntg,uv=uv);log=res['log']
        if arm=='NAT':ntg=[e['update_norm'] for e in log]
        fnA.write_text(json.dumps(dict(arm=arm,engine=ENGINE[arm],stream='P5',n_chunks=nch,read_chunks=rch,ids_sha=meta['ids_sha'],canary_w0=cw0,trajectory=traj,states={str(k):{'file':v['_file'],'hash':v['_hash']} for k,v in res['reads'].items()},sec_total=time.monotonic()-t0,
            update_log=[{k_:e[k_] for k_ in ('k','ce','update_norm','coef','sec','n_masked_keys','q_banded','q_band') if k_ in e} for e in log],norm_targets=ntg if arm=='NAT' else None),default=float));print('wrote',fnA,flush=True)
        del L,res;gc.collect();LD.jax.clear_caches()
def manifest(out,cond):
    """A: both arms, all ten sections (rw8, rw4 first); B / C: rw4 and rw8 only"""
    out=pathlib.Path(out);secs=(['rw8','rw4']+[s for s in SECS if s not in ('rw8','rw4')]) if cond=='A' else ['rw4','rw8'];m={}
    for arm in RARMS:
        d=json.load(open(out/f'E57_pqa5_{arm}.arms.json'))
        for sec in secs:
            k=str(d['read_chunks'][sec]);s=d['states'][k];m[f'{arm}@{sec}']=dict(label=arm,section=sec,chunk=int(k),file=s['file'],hash=s['hash'])
    return m
def read_stage(a):
    out=pathlib.Path(a['out']);out.mkdir(parents=True,exist_ok=True);lg=lambda *x:print(*x,flush=True);cond=a['cond']
    LD,rt,rt32,rd,items,h,raw0,ntok0,out_tail,tsha=P2.setup_read(a,lg);import e57_reads as RD
    asg=json.load(open(pathlib.Path(a['dir620'])/'E57_pqa4_assignB.json'))['assignment'];pre=P4.prefix_of(items,cond,asg);items2=[dict(it,prefix=p) for it,p in zip(items,pre)];raw,ntok,bad=RD.build_rows(rt.tok,items2);assert not bad,bad
    man=json.load(open(a['manifest']));m32w0=rd.to_fp32(rt.base);t0=time.monotonic();fo=P2.FastOUT(rt32,m32w0,out_tail,raw,log=lg);tim={'prep':time.monotonic()-t0};lg(f'cond {cond}: first frames {pre[:2]}')
    for n,s in man.items():
        t0=time.monotonic();m=rd.load_state(s['file'],expect_hash=s['hash']);rows=P2.rows_of(items2,ntok,fo.score(rd.to_fp32(m)),f'{cond}:{n}');del m;tim[n]=time.monotonic()-t0
        (out/f'E57_pqa5_rows_{cond}_{n.replace("@","_")}.json').write_text(json.dumps(rows,default=float));lg(f'  {n} read: {tim[n]:.0f}s gold lp mean {np.mean([r["cand_lp"][0] for r in rows]):.3f}')
    (out/f'E57_pqa5_meta_{cond}.json').write_text(json.dumps(dict(cond=cond,items_sha256=h,out_tail_sha=tsha,timings=tim,states=man,w0_rows='A: job 610 rows (E57_pqa3_rows_W0.json), B / C: job 620 rows (E57_pqa4_rows_{B,C}_W0.json)'),default=float))
def count_by_section(stream_dir):
    txt={s:[] for s in SECS+['tail']}
    for l in open(pathlib.Path(stream_dir)/'streamP5_stream_sentences.jsonl'):
        r=json.loads(l)
        if r['type']=='frag':txt[r['section']].append(r['text'])
    return {k:''.join(v) for k,v in txt.items()}
def counts_of(texts,strings):
    import re
    out={}
    for s in sorted(set(strings)):
        rx=re.compile(r'(?<![A-Za-z0-9])'+re.escape(s)+r'(?![A-Za-z0-9])');out[s]=dict(cnt=[texts[k].count(s) for k in SECS],cnt_wb=[len(rx.findall(texts[k])) for k in SECS])
    return out
def analyze(a):
    out=pathlib.Path(a['out']);d610=pathlib.Path(a['dir610']);d620=pathlib.Path(a['dir620']);items,h=Q.load_items_file(a['items']);sub=P3.subsets(json.load(open(a['paragraphs'])))
    ids=[it['id'] for it in items];masks={'all1000':np.ones(len(ids),bool),'items455':np.asarray([i in sub['items455'] for i in ids]),'main373':np.asarray([i in sub['main373'] for i in ids])}
    ldA=lambda n:json.load(open(out/f'E57_pqa5_rows_A_{n.replace("@","_")}.json'));A0=json.load(open(d610/'E57_pqa3_rows_W0.json'));B0=json.load(open(d620/'E57_pqa4_rows_B_W0.json'));C0=json.load(open(d620/'E57_pqa4_rows_C_W0.json'))
    mets=('lp_sum','lp_mean','rr_sum','rr_mean');res={'A_curve':{},'B_C':{}}
    for arm in RARMS:
        res['A_curve'][arm]={}
        for sec in SECS:
            f=out/f'E57_pqa5_rows_A_{arm}_{sec}.json'
            if not f.exists():continue
            A=json.load(open(f));res['A_curve'][arm][sec]={sn:{k:P4.boot(P4.metric_vec(A,k)[m]) for k in mets}|{'imp_lp_sum':P4.boot((P4.metric_vec(A,'lp_sum')-P4.metric_vec(A0,'lp_sum'))[m]),'imp_lp_mean':P4.boot((P4.metric_vec(A,'lp_mean')-P4.metric_vec(A0,'lp_mean'))[m]),'acc_mean':P4.boot(np.asarray([float(r['acc_mean']) for r in A])[m]),'acc_sum':P4.boot(np.asarray([float(r['acc_sum']) for r in A])[m])} for sn,m in masks.items()}
    for arm in RARMS:
        for sec in ('rw4','rw8'):
            n=f'{arm}@{sec}';fb=out/f'E57_pqa5_rows_B_{arm}_{sec}.json';fc=out/f'E57_pqa5_rows_C_{arm}_{sec}.json'
            if not (fb.exists() and (out/f'E57_pqa5_rows_A_{arm}_{sec}.json').exists()):continue
            A=ldA(n);B=json.load(open(fb));C=json.load(open(fc)) if fc.exists() else None
            res['B_C'][n]={sn:dict(gain=P4.gain_table(A,B,A0,B0,m),C=None if C is None else {k:P4.boot(P4.metric_vec(C,k)[m]) for k in mets},C_minus_W0=None if C is None else {k:P4.boot((P4.metric_vec(C,k)-P4.metric_vec(C0,k))[m]) for k in ('lp_sum','rr_mean')}) for sn,m in masks.items()}
    texts=count_by_section(a['stream_dir']);cnt=counts_of(texts,[s for it in items for s in it['cands']]);asg=json.load(open(d620/'E57_pqa4_assignB.json'))['assignment'];irows=[]
    for k,it in enumerate(items):
        r=dict(id=it['id'],title=it['title'],type=it['type'],assigned_question_id=asg[it['id']],in_items455=bool(masks['items455'][k]),in_main373=bool(masks['main373'][k]),gold=it['cands'][0],cands=it['cands'],sections=SECS,counts_total=[sum(cnt[c]['cnt']) for c in it['cands']],counts_wb_total=[sum(cnt[c]['cnt_wb']) for c in it['cands']],counts_by_section=[cnt[c]['cnt'] for c in it['cands']],counts_wb_by_section=[cnt[c]['cnt_wb'] for c in it['cands']])
        r['W0']=dict(A=A0[k]['cand_lp'][0],B=B0[k]['cand_lp'][0],C=C0[k]['cand_lp'][0])
        for arm in RARMS:
            for sec in SECS:
                f=out/f'E57_pqa5_rows_A_{arm}_{sec}.json'
                if f.exists():r[f'{arm}@{sec}']=dict(A=P5_cache(f)[k]['cand_lp'][0])
        irows.append(r)
    for arm in RARMS:
        for sec in ('rw4','rw8'):
            for c in ('B','C'):
                f=out/f'E57_pqa5_rows_{c}_{arm}_{sec}.json'
                if f.exists():
                    R_=P5_cache(f)
                    for k,r in enumerate(irows):r[f'{arm}@{sec}'][c]=R_[k]['cand_lp'][0]
    (out/'E57_pqa5_keys.json').write_text(json.dumps(dict(meta=dict(items_sha256=h,gain_definition='(A(state)-A(W0)) - (B(state)-B(W0)); rank-ratio metrics as improvement (W0 minus state), positive = better',B_rule='question of another item: same type, different title (assignment of job 620, seed 20261101)',C_rule='"Answer:" only',W0_rows='A: job 610, B / C: job 620',subsets={k:int(v.sum()) for k,v in masks.items()}),**res),indent=1,default=float))
    (out/'E57_pqa5_item_rows.json').write_text(json.dumps(irows,default=float))
_PC={}
def P5_cache(f):
    if f not in _PC:_PC[f]=json.load(open(f))
    return _PC[f]
def selftest():
    S5.selftest();P4.selftest()
    tx={k:'' for k in SECS+['tail']};tx['rw8']='Paris Paris.';c=counts_of(tx,['Paris']);assert c['Paris']['cnt'][-1]==2 and len(c['Paris']['cnt'])==10
    print('e57_pqa5_run selftest ok')
def main():
    if '--selftest' in sys.argv:selftest();return
    stage=P2._pop('--stage');rerun='--rerun' in sys.argv
    if rerun:sys.argv.remove('--rerun')
    a=dict(cond=P2._pop('--cond'),stream_dir=P2._pop('--stream-dir'),out=P2._pop('--out'),manifest=P2._pop('--manifest'),items=P2._pop('--items-file'),bank=P2._pop('--bank'),dir610=P2._pop('--dir610'),dir620=P2._pop('--dir620'),paragraphs=P2._pop('--paragraphs'),stage1=P2._pop('--stage1'),rerun=rerun)
    if stage=='manifest':
        cond=P2._pop('--cond') or a['cond'];m=manifest(a['out'],cond);json.dump(m,open(a['manifest'],'w'),indent=1);print({k:(v['chunk'],v['hash'][:12]) for k,v in m.items()});return
    {'write':write_stage,'read':read_stage,'analyze':analyze}[stage](a)
if __name__=='__main__':main()
