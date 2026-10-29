"""E57 public QA W0 reads with the KV-reuse IN scorer (jobs 460 / 480; GPU, read only, NO weight update, NO state): the reads of e57_squad_w0_reads.squad_reads (job 440; same OUT tail, same E49 scores, same output files and format) with two additions, both optional:
  --resume-from PARTIAL.json   take over the OUT rows (all items) and the first IN rows of the partial file of a cancelled run (checked: same items, same candidates, OUT-then-IN) and continue the IN reads with the first item without an IN row; the OUT scorer is not rebuilt;
  the IN scores of an item come from e57_kvreuse.FastIN (shared tail prefix K,V / suffix state cached; checked against the reference path in job 447: 6 items, max |diff| 0, 3.1 s instead of 24.4 s per item) when the file given by --fast-if-exists exists (written only after the check was read) and the paragraph has <= 1024 tokens; otherwise the reference path of job 440 (Reader55.packed) is used.
  --multihop   the items files are the HotpotQA / MuSiQue files of e57_public_qa_build_multihop.py (comma separated, e57_multihop_w0_reads loader and per-dataset keys).
  python3 e57_fastin_reads.py [--resume-from P] [--fast-if-exists F] [--multihop] --bundle B --out O --cycles 1 --npos 1 --canary-npos 1 --arms NAT --items-file F.json [--n-items N] --stage reads   |   --selftest"""
import sys,json,time,hashlib,os,pathlib
import numpy as np
HERE=pathlib.Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
import e57_squad_w0_reads as Q
import e57_squad_w0_resume as RS
RD=Q.RD
OPT={'resume':None,'fast':None}
def squad_reads(a,out,items_bank,bsrc,iv,hits,shtab,ids_all,n,N,kks,arms,tk,LD,rt,R,cw0,lg,can):
    import e55_gpu as RG
    items,h=Q.load_items_file(Q.F['items']);items=items[:int(Q.F['n'])] if Q.F['n'] else items;out=pathlib.Path(out);out.mkdir(parents=True,exist_ok=True)
    rt32=LD.runtime_fp32();rd=RG.Reader55(rt,rt32,LD.packing,LD._TailCtx,LD.reuse.get_filter_spec,LD.reuse._hash_pytree,LD.jax,LD.eqx,LD.jnp)
    raw,ntok,bad=RD.build_rows(rt.tok,items);assert not bad,bad
    forbid=sorted({s for it in items_bank for s in it['variants']+[it['cands'][0]] if len(s)>=4 and not any(ch.isdigit() for ch in s)});out_tail=R.out_tail_8192(LD,rt.tok,rt.bos,forbid);assert len(out_tail)==8192
    lg(f'{len(items)} items, {len(raw)} candidate rows, items file sha {h[:12]}; OUT tail sha {hashlib.sha256(np.asarray(out_tail,np.int64).tobytes()).hexdigest()[:16]}')
    off=[0]
    for it in items:off.append(off[-1]+len(it['cands']))
    def score(P,m32):
        with LD.jax.default_matmul_precision('highest'):return np.asarray(P.r1(m32))
    m32=rd.to_fp32(rt.base);tim={};rows=[];GROUP=40
    if OPT['resume']:
        rows_out,rows_in=RS.take_over(OPT['resume'],items);rows=list(rows_out)+list(rows_in);tim['OUT']='taken over from the cancelled run';lg(f'resume: {len(rows_out)} OUT rows and {len(rows_in)} IN rows taken over from {OPT["resume"]}')
    else:
        t0=time.monotonic()
        for g0 in range(0,len(items),GROUP):
            sub=items[g0:g0+GROUP];P=rd.packed(raw[off[g0]:off[g0+len(sub)]],out_tail);flat=score(P,m32);o=0
            for k,it in enumerate(sub):nc=len(it['cands']);rows.append(Q.row_of(it,'OUT',flat[o:o+nc],ntok[g0+k]));o+=nc
            del P
            if (g0//GROUP)%5==0:lg(f'  OUT pack {g0//GROUP+1}/{(len(items)+GROUP-1)//GROUP}: {time.monotonic()-t0:.0f}s');(out/'E57_squad_w0_reads_partial.json').write_text(json.dumps(dict(done='OUT partial',rows=rows),default=float))
        tim['OUT']=time.monotonic()-t0;lg(f'OUT done in {tim["OUT"]:.0f}s: acc_sum {np.mean([r["acc_sum"] for r in rows]):.3f}')
    start=sum(r['condition']=='IN' for r in rows)
    use_fast=bool(OPT['fast']) and os.path.exists(OPT['fast']);fi=None
    if use_fast:
        import e57_kvreuse as KR;fi=KR.FastIN(rt32,m32,out_tail,R=1024,log=lg);lg(f'IN reads: KV-reuse scorer ({OPT["fast"]} exists)')
    else:lg('IN reads: reference path')
    t0=time.monotonic();tail=list(out_tail);maxp=0;nfast=0
    for k,it in enumerate(items):
        pid=LD.stage2.enc(rt.tok,'\n\n'+' '.join(it['context'].split())+'\n\n').ids;maxp=max(maxp,len(pid));assert len(pid)<7000,(it['id'],len(pid))
        if k<start:continue
        if fi is not None and len(pid)<=1024:flat=fi.score(raw[off[k]:off[k+1]],pid);nfast+=1
        else:
            ctx=tail[:8192-len(pid)]+list(pid);assert len(ctx)==8192;P=rd.packed(raw[off[k]:off[k+1]],ctx);flat=score(P,m32);del P
        rows.append(Q.row_of(it,'IN',flat,ntok[k]))
        if k%50==0:lg(f'  IN item {k+1}/{len(items)}: {time.monotonic()-t0:.0f}s');(out/'E57_squad_w0_reads_partial.json').write_text(json.dumps(dict(done='IN partial',rows=rows),default=float))
    tim['IN']=time.monotonic()-t0;kn=Q.keys_of(rows)
    (out/'E57_squad_w0_reads.json').write_text(json.dumps(dict(meta=dict(items_sha256=h,rules='e57_public_qa_build_*.py',n_items=len(items),context_out='neutral SQuAD dev tail 8192 tokens (news OUT tail)',context_in='tail with the item paragraph(s) at its end (8192 tokens)',max_paragraph_tokens=maxp,timings=tim,resumed_from=OPT['resume'],in_rows_taken_over=start,in_scorer='kvreuse' if use_fast else 'reference',in_items_by_kvreuse=nfast),rows=rows),default=float))
    (out/'E57_squad_w0_keys.json').write_text(json.dumps(dict(items_sha256=h,keys=kn),indent=1,default=float));lg('wrote E57_squad_w0_reads.json and E57_squad_w0_keys.json');lg(json.dumps({c:round(v['ALL']['acc_sum'],3) for c,v in kn.items()}))
def selftest():
    Q.selftest();RS.selftest();print('e57_fastin_reads selftest ok')
def main():
    if '--selftest' in sys.argv:selftest();return
    OPT['resume']=Q._pop('--resume-from');OPT['fast']=Q._pop('--fast-if-exists')
    if '--multihop' in sys.argv:
        sys.argv.remove('--multihop');import e57_multihop_w0_reads as MH;Q.load_items_file=MH.load_multihop;Q.keys_of=MH.keys_ds
    Q.F['items']=Q._pop('--items-file');Q.F['n']=Q._pop('--n-items');assert Q.F['items']
    import e57_main_run as M
    M.stage_reads=squad_reads;M.main()
if __name__=='__main__':main()
