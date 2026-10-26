"""E57 trial run (GPU; NAT single arm; E57 section 39): ONE cycle of the natural news stream (all independent lead articles of the bundle, global date order, real tokenizer, whole 1024-token chunks) through e55_core.Ladder55.run_stream('NAT'), one inner update per chunk, with
  - canary (pg11 generation NLL, pg1342 tail NLL) at W0 and at the chunks --points,
  - memory reads of the frozen bank (E57 section 39) at W0 and at the end of the cycle (and at --read-chunks if given; fixed by stream position, not by question): condition OUT = neutral SQuAD tail of exactly 8192 tokens (no gold string of the bank in it), condition IN = the natural window (the last min(8192, position) stream tokens before the read; IN rows carry window_info = present / absent by gold surface-form match in the decoded window). Reads are read-only: weights pass through an fp32 COPY (Reader55.to_fp32) to the packed scorer; nothing of the fast weights, the optimizer or the carried KV window is touched.
  - W0 is read with the SAME contexts as the later reads (OUT tail and the IN window of that read position), so W0-to-read differences are weight effects.
Result file E57_trial_NAT.json: log rows (e57_reads.log_rows), canary trajectory, per-chunk update log, timings. The candidates, the gold form and the hard/easy split come from the frozen bank bank_v1.json (--bank), nothing is chosen here.
  python3 e57_trial_run.py --bundle OUT_news --out DIR [--bank F] [--points ..] [--read-chunks ..] [--check-only] [--stub]"""
import sys,json,os,time,argparse,hashlib,pathlib
import numpy as np
HERE=pathlib.Path(__file__).resolve().parent
for p in (HERE,HERE/'e57',HERE.parents[1]/'scripts/exp04_what_ttt_retains',HERE.parents[1]/'scripts/exp05_write_dynamics'):sys.path.insert(0,str(p))
import e57_reads as RD
import e57_canary_run as CR
CH=1024
def pseudo_bank(bundle):
    """CPU check only: a bank-like structure from validated.jsonl + distractors_ext.json + nominations_freeze.jsonl (NOT used for a GPU run)"""
    fd=pathlib.Path(bundle)/'e57_bank';V=[json.loads(l) for l in open(fd/'validated.jsonl') if l.strip()];D={d['id']:d for d in json.load(open(fd/'distractors_ext.json'))};N={}
    for l in open(fd/'nominations_freeze.jsonl'):
        x=json.loads(l);N[x['id']]=x
    import e57_bank as B
    qs=[]
    for q in V:
        d=D[q['id']];keys={B.norm(x) for x in N[q['id']]['nominated']};hn=[h for h in d['hard'] if B.norm(h['surface']) in keys]
        qs.append(dict(q,fixed_surface=q['canonical'],easy=d['easy'],hard_nominated=hn))
    return dict(questions=qs)
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--bundle',required=True);ap.add_argument('--out',required=True);ap.add_argument('--bank');ap.add_argument('--points');ap.add_argument('--read-chunks');ap.add_argument('--check-only',action='store_true');ap.add_argument('--stub',action='store_true');ap.add_argument('--tokenizer');ap.add_argument('--rerun',action='store_true');a=ap.parse_args()
    ROOT=HERE.parents[1];sys.path.insert(0,str(ROOT/'scripts/exp04_what_ttt_retains'))
    import gen_world_ctl as G
    tfile=a.tokenizer or str(ROOT/'data/tokenizer/llama3/tokenizer.json');enc=G.StubTok() if a.stub else G.load_enc(tfile)
    bank=pathlib.Path(a.bank) if a.bank else pathlib.Path(a.bundle)/'e57_bank'/'bank_v1.json'
    if bank.exists():items=RD.load_items(bank);bsrc=str(bank)
    else:
        assert a.check_only,f'frozen bank {bank} missing (exit 7)';tmp=pathlib.Path(a.out)/'_pseudo_bank.json';pathlib.Path(a.out).mkdir(parents=True,exist_ok=True);tmp.write_text(json.dumps(pseudo_bank(a.bundle)));items=RD.load_items(tmp);bsrc='PSEUDO (check only)'
    ids,iv,hits=RD.article_intervals(a.bundle,enc,items);n=len(ids)//CH;ctoks=n*CH;shtab=RD.shadow_table(iv,hits,ctoks)
    pts=[int(x) for x in a.points.split(',')] if a.points else CR.points_default(n);rch=sorted({n-1}|({int(x) for x in a.read_chunks.split(',')} if a.read_chunks else set()))
    assert all(0<=p<n for p in pts+rch)
    from tokenizers import Tokenizer;tk=None if a.stub else Tokenizer.from_file(tfile)
    def win_of(k):pos=(k+1)*CH;return (max(0,pos-8192),pos)
    def win_text(k):s,e=win_of(k);return tk.decode([int(x) for x in ids[s:e]])
    print(f'bank {bsrc}: {len(items)} questions, candidates per question min/max {min(len(i["cands"]) for i in items)}/{max(len(i["cands"]) for i in items)}, hard-only sets {min(len(i["hard_idx"])+1 for i in items)}-{max(len(i["hard_idx"])+1 for i in items)}',flush=True)
    allsh=[c1 for v in shtab.values() for a_,e_,c1,cn in v];print(f'n_shadow (same-fact articles inside the 8192-token window before an exposure, cycle 1): mean {np.mean(allsh):.2f}, share with none {np.mean([x==0 for x in allsh]):.2f}, over {len(allsh)} exposures',flush=True)
    print(f'stream: {n} chunks, canary at {pts}, reads at {rch}, ids sha {hashlib.sha256(ids.tobytes()).hexdigest()[:16]}',flush=True)
    if tk is not None:
        raw,ntok,bad=RD.build_rows(tk,items);print(f'rows: {len(raw)}; candidate texts the tokenizer span check rejects: {bad or "none"}',flush=True)
        assert not bad,('candidates failing the span check (exit 7)',bad)
        for k in rch:
            wp=RD.window_present(items,win_text(k));print(f'  read chunk {k}: window {win_of(k)} holds the gold of {sum(wp.values())} of {len(items)} questions',flush=True)
    if a.check_only:print('trial check-only OK');return
    out=pathlib.Path(a.out);fn=out/'E57_trial_NAT.json'
    if fn.exists() and not a.rerun:print('exists, not redone (exit 8):',fn);sys.exit(8)
    import e55_run as R,e55_gpu as RG,stage2
    LD,s3,C=R.lab();rt=LD.Runtime();rt32=LD.runtime_fp32();can=R.Canary55(LD,rt);rd=RG.Reader55(rt,rt32,LD.packing,LD._TailCtx,LD.reuse.get_filter_spec,LD.reuse._hash_pytree,LD.jax,LD.eqx,LD.jnp)
    raw,ntok,bad=RD.build_rows(rt.tok,items);assert not bad,bad
    forbid=sorted({s for it in items for s in it['variants']+[it['cands'][0]] if len(s)>=4 and not any(ch.isdigit() for ch in s)})
    out_tail=R.out_tail_8192(LD,rt.tok,rt.bos,forbid);P_out=rd.packed(raw,out_tail);P_in={}
    def score(P,m):
        m32=rd.to_fp32(m)
        with LD.jax.default_matmul_precision('highest'):return RD.split_scores(np.asarray(P.r1(m32)),items)
    csa={q['id']:q['q'].get('c_SA') for q in items for q in [dict(id=q['id'],q=q['q'])]}
    rows=[];tim={}
    def read(cycle,k,m,tag):
        t0=time.monotonic();pos=(k+1)*CH if cycle>0 else (k+1)*CH
        rows.extend(RD.log_rows(items,score(P_out,m),ntok,cycle=cycle,read_chunk=k,pos_tokens=pos,win=None,arm='NAT',condition='OUT',iv=iv,hits=hits,csa=csa,shadow=shtab))
        if k not in P_in:P_in[k]=rd.packed(raw,[int(x) for x in ids[win_of(k)[0]:win_of(k)[1]]])
        wp=RD.window_present(items,win_text(k));rows.extend(RD.log_rows(items,score(P_in[k],m),ntok,cycle=cycle,read_chunk=k,pos_tokens=pos,win=win_of(k),arm='NAT',condition='IN',iv=iv,hits=hits,winpresent=wp,csa=csa,shadow=shtab))
        tim[f'{tag}']=time.monotonic()-t0;print(f'  reads {tag}: {tim[tag]:.0f}s',flush=True)
    cw0=can(rt.base);print('canary W0',cw0,flush=True)
    for k in rch:read(0,k,rt.base,f'W0@{k}')                       # W0 read with the contexts of every later read position
    traj=[]
    def hook(k,m):
        x={'canary':can(m)} if k in pts else {}
        if x:traj.append({'chunk':int(k),'updates':int(k)+1,'canary':x['canary']})
        if k in rch:read(1,k,m,f'cycle1@{k}')
        return {'ok':True}
    L=C.Ladder55(rt,save_dir=out/'states'/'E57_trial',canary=None);t0=time.monotonic()
    r=L.run_stream('NAT',1,ids,nch=n,tag='trialNAT',log=lambda *x:print(*x,flush=True),read_hook=hook,hook_chunks=set(pts)|set(rch))
    res=dict(arm='NAT',bank=bsrc,shadow_table={q:[dict(article=int(a_),end=int(e_),n_shadow_cycle1=int(c1),n_shadow_later_cycles=int(cn)) for a_,e_,c1,cn in v] for q,v in shtab.items()},n_chunks=n,ids_sha=hashlib.sha256(ids.tobytes()).hexdigest()[:16],canary_w0=cw0,trajectory=traj,rows=rows,log=[{k:e[k] for k in ('k','ce','update_norm','coef','sec') if k in e} for e in r['log']],timings=tim,sec_total=time.monotonic()-t0,
        e55_reference='E55 canary rise at 128 updates: SGD +0.023, 0.5N +0.043, Adam +0.159; limit 0.10 (E57 doc 12.2)',ordering_note='cand_lp = summed span log-prob (E49 convention); cand_ntok lets the analysis form per-token means')
    fn.write_text(json.dumps(res,default=float));print('wrote',fn,len(rows),'rows')
if __name__=='__main__':main()
