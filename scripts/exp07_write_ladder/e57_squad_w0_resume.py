"""E57 public QA, stage 1 resume (job 440 rerun; GPU, read only, NO weight update, NO state): continues the cancelled run of job 440 (e57_squad_w0_reads.py, commit e7caf2f) from its partial file. The OUT rows (all 1,000 items) and the first IN rows of the partial file are taken over unchanged (checked: same items in the same order, same candidates, items file hash of the run); the OUT scorer is not rebuilt; the IN reads continue with the first item that has no IN row. Everything else is the code of e57_squad_w0_reads.squad_reads (same tail, same scores, same output files and format); only the OUT loop is replaced by the taken-over rows.
  python3 e57_squad_w0_resume.py --resume-from PARTIAL.json <the arguments of e57_squad_w0_reads.py>   |   --selftest"""
import sys,json,time,hashlib,pathlib
import numpy as np
HERE=pathlib.Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
import e57_squad_w0_reads as Q
RD=Q.RD
def take_over(path,items):
    d=json.load(open(path));rows=d['rows'];out=[r for r in rows if r['condition']=='OUT'];inn=[r for r in rows if r['condition']=='IN']
    assert len(out)==len(items),('OUT rows',len(out),len(items))
    for r,it in zip(out,items):assert r['id']==it['id'] and r['cand_surface']==it['cands'],('OUT row mismatch',r['id'],it['id'])
    for r,it in zip(inn,items):assert r['id']==it['id'] and r['cand_surface']==it['cands'],('IN row mismatch',r['id'],it['id'])
    assert rows==out+inn,'partial rows are not OUT-then-IN'
    return out,inn
def squad_reads(a,out,items_bank,bsrc,iv,hits,shtab,ids_all,n,N,kks,arms,tk,LD,rt,R,cw0,lg,can):
    import e55_gpu as RG
    items,h=Q.load_items_file(Q.F['items']);items=items[:int(Q.F['n'])] if Q.F['n'] else items;out=pathlib.Path(out);out.mkdir(parents=True,exist_ok=True)
    rows_out,rows_in=take_over(RESUME['path'],items);lg(f'resume: {len(rows_out)} OUT rows and {len(rows_in)} IN rows taken over from {RESUME["path"]}')
    rt32=LD.runtime_fp32();rd=RG.Reader55(rt,rt32,LD.packing,LD._TailCtx,LD.reuse.get_filter_spec,LD.reuse._hash_pytree,LD.jax,LD.eqx,LD.jnp)
    raw,ntok,bad=RD.build_rows(rt.tok,items);assert not bad,bad
    forbid=sorted({s for it in items_bank for s in it['variants']+[it['cands'][0]] if len(s)>=4 and not any(ch.isdigit() for ch in s)});out_tail=R.out_tail_8192(LD,rt.tok,rt.bos,forbid);assert len(out_tail)==8192
    lg(f'{len(items)} items, {len(raw)} candidate rows, items file sha {h[:12]}; OUT tail sha {hashlib.sha256(np.asarray(out_tail,np.int64).tobytes()).hexdigest()[:16]}')
    off=[0]
    for it in items:off.append(off[-1]+len(it['cands']))
    rows=list(rows_out)+list(rows_in);tim={'OUT':'taken over from the cancelled run'}
    def score(P,m32):
        with LD.jax.default_matmul_precision('highest'):return np.asarray(P.r1(m32))
    m32=rd.to_fp32(rt.base);t0=time.monotonic();tail=list(out_tail);maxp=0
    for it in items:maxp=max(maxp,len(LD.stage2.enc(rt.tok,'\n\n'+' '.join(it['context'].split())+'\n\n').ids))
    for k,it in enumerate(items):
        if k<len(rows_in):continue
        pid=LD.stage2.enc(rt.tok,'\n\n'+' '.join(it['context'].split())+'\n\n').ids;assert len(pid)<7000,(it['id'],len(pid))
        ctx=tail[:8192-len(pid)]+list(pid);assert len(ctx)==8192
        P=rd.packed(raw[off[k]:off[k+1]],ctx);flat=score(P,m32);rows.append(Q.row_of(it,'IN',flat,ntok[k]));del P
        if k%50==0:lg(f'  IN item {k+1}/{len(items)}: {time.monotonic()-t0:.0f}s');(out/'E57_squad_w0_reads_partial.json').write_text(json.dumps(dict(done='IN partial',rows=rows),default=float))
    tim['IN_resumed']=time.monotonic()-t0;kn=Q.keys_of(rows)
    (out/'E57_squad_w0_reads.json').write_text(json.dumps(dict(meta=dict(items_sha256=h,rules='e57_public_qa_build_squad.py',n_items=len(items),context_out='neutral SQuAD dev tail 8192 tokens (news OUT tail)',context_in='tail with the item paragraph at its end (8192 tokens)',max_paragraph_tokens=maxp,timings=tim,resumed_from=RESUME['path'],in_rows_taken_over=len(rows_in)),rows=rows),default=float))
    (out/'E57_squad_w0_keys.json').write_text(json.dumps(dict(items_sha256=h,keys=kn),indent=1,default=float));lg('wrote E57_squad_w0_reads.json and E57_squad_w0_keys.json');lg(json.dumps({c:round(v['ALL']['acc_sum'],3) for c,v in kn.items()}))
RESUME={'path':None}
def selftest():
    Q.selftest()
    import tempfile,os
    it=[dict(id=x,cands=['g','y']) for x in 'ab']
    mk=lambda cond,i:dict(id=i['id'],cand_surface=i['cands'],condition=cond)
    f=os.path.join(tempfile.mkdtemp(),'p.json');json.dump(dict(rows=[mk('OUT',i) for i in it]+[mk('IN',it[0])]),open(f,'w'))
    o,n=take_over(f,it);assert len(o)==2 and len(n)==1
    json.dump(dict(rows=[mk('OUT',i) for i in it[:1]]),open(f,'w'))
    try:take_over(f,it);raise SystemExit('should fail')
    except AssertionError:pass
    print('e57_squad_w0_resume selftest ok')
def main():
    if '--selftest' in sys.argv:selftest();return
    RESUME['path']=Q._pop('--resume-from');assert RESUME['path'],'--resume-from is required'
    Q.F['items']=Q._pop('--items-file');Q.F['n']=Q._pop('--n-items');assert Q.F['items']
    import e57_main_run as M
    M.stage_reads=squad_reads;M.main()
if __name__=='__main__':main()
