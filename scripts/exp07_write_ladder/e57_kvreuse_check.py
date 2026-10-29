"""E57 check of the KV-reuse IN scorer (e57_kvreuse.FastIN) against the reference path (job 447; GPU, read only, W0 weights, no state written): for a few SQuAD items (shortest, median, longest paragraph <= R tokens and others) the IN scores of the 16 candidates are computed (a) by the reference path of job 440 (Reader55.packed -> packing.Packed.r1, full recomputation) and (b) by FastIN. Output: <out>/E57_kvreuse_check.json with per item the reference and fast scores, max abs difference, rank of the gold under both, the seconds of both paths.
  python3 e57_kvreuse_check.py --bundle B --out O --cycles 1 --npos 1 --canary-npos 1 --arms NAT --items-file F.json [--n-items N] --stage reads   |   --selftest"""
import sys,json,time,pathlib
import numpy as np
HERE=pathlib.Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
import e57_squad_w0_reads as Q
import e57_kvreuse as KR
RD=Q.RD
NCHECK=[6]
def check_reads(a,out,items_bank,bsrc,iv,hits,shtab,ids_all,n,N,kks,arms,tk,LD,rt,R,cw0,lg,can):
    import e55_gpu as RG
    items,h=Q.load_items_file(Q.F['items']);out=pathlib.Path(out);out.mkdir(parents=True,exist_ok=True)
    rt32=LD.runtime_fp32();rd=RG.Reader55(rt,rt32,LD.packing,LD._TailCtx,LD.reuse.get_filter_spec,LD.reuse._hash_pytree,LD.jax,LD.eqx,LD.jnp)
    raw,ntok,bad=RD.build_rows(rt.tok,items);assert not bad,bad
    forbid=sorted({s for it in items_bank for s in it['variants']+[it['cands'][0]] if len(s)>=4 and not any(ch.isdigit() for ch in s)});out_tail=R.out_tail_8192(LD,rt.tok,rt.bos,forbid);assert len(out_tail)==8192
    off=[0]
    for it in items:off.append(off[-1]+len(it['cands']))
    lens=[len(LD.stage2.enc(rt.tok,'\n\n'+' '.join(it['context'].split())+'\n\n').ids) for it in items];order=np.argsort(lens);ok=[i for i in order if lens[i]<=1024]
    pick=[int(ok[0]),int(ok[len(ok)//2]),int(ok[-1]),int(ok[len(ok)//4]),int(ok[3*len(ok)//4]),int(ok[len(ok)//8])][:NCHECK[0]]
    lg(f'paragraph tokens: min {min(lens)} median {int(np.median(lens))} max {max(lens)}; > 1024 tokens: {sum(l>1024 for l in lens)}; checking items {pick} with {[lens[i] for i in pick]} tokens')
    m32=rd.to_fp32(rt.base)
    def ref_score(k,pid):
        ctx=list(out_tail)[:8192-len(pid)]+list(pid);P=rd.packed(raw[off[k]:off[k+1]],ctx)
        with LD.jax.default_matmul_precision('highest'):return np.asarray(P.r1(m32))
    t0=time.monotonic();fi=KR.FastIN(rt32,m32,out_tail,R=1024,log=lg);tbuild=time.monotonic()-t0;res=[]
    for k in pick:
        it=items[k];pid=LD.stage2.enc(rt.tok,'\n\n'+' '.join(it['context'].split())+'\n\n').ids
        t0=time.monotonic();r=ref_score(k,pid);tr=time.monotonic()-t0
        for rep in range(2):
            t0=time.monotonic();f=fi.score(raw[off[k]:off[k+1]],pid);tf=time.monotonic()-t0
        d=np.abs(r-f);rk=lambda s:int((s>s[0]).sum()+1)
        res.append(dict(item=k,id=it['id'],paragraph_tokens=len(pid),ref=r.tolist(),fast=f.tolist(),max_abs_diff=float(d.max()),mean_abs_diff=float(d.mean()),max_rel_diff=float((d/np.maximum(np.abs(r),1e-6)).max()),gold_rank_ref=rk(r),gold_rank_fast=rk(f),sec_ref=tr,sec_fast_second_call=tf))
        lg(f'  item {k} ({len(pid)} tok): max|diff| {d.max():.2e} (scores ~{np.abs(r).mean():.1f}); gold rank ref {rk(r)} fast {rk(f)}; ref {tr:.1f}s, fast {tf:.1f}s')
    (out/'E57_kvreuse_check.json').write_text(json.dumps(dict(items_sha256=h,build_seconds=tbuild,checks=res),default=float))
    lg('wrote E57_kvreuse_check.json')
def selftest():
    Q.selftest();KR.selftest();print('e57_kvreuse_check selftest ok')
def main():
    if '--selftest' in sys.argv:selftest();return
    Q.F['items']=Q._pop('--items-file');Q.F['n']=Q._pop('--n-items');assert Q.F['items']
    import e57_main_run as M
    M.stage_reads=check_reads;M.main()
if __name__=='__main__':main()
