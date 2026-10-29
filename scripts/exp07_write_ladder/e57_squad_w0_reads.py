"""E57 public QA, stage 1 (job 440; GPU, read only, NO weight update, NO state): the 1,000 frozen SQuAD dev-v1.1 items of e57_public_qa_build_squad.py (rules hashed into the file, sha256 d80f9987…; 16 candidates = gold + 15 distractors, items of Super_Bowl_50 and Warsaw excluded because the OUT tail holds them) are read with the W0 weights:
  OUT  the neutral 8192-token SQuAD tail of the news jobs (R.out_tail_8192 with the forbid list of bank v1: identical tokens), then the question and the candidate;
  IN   the same tail with its end replaced by the item's own paragraph: ids = BOS + tail[: 8192 - 1 - len(para)] + para tokens (blank line + paragraph + blank line), exactly 8192 tokens, then the question and the candidate (one context, hence one packed scorer, per item).
Score = summed log-prob of the candidate tokens after the question (E49 convention, ' ' + candidate) and the per-token mean (cand_lp / cand_ntok); gold-vs-all-16 rank, rank ratio and accuracy on both (e57_reads.metrics). Output: <out>/E57_squad_w0_reads.json (rows per item and condition) + E57_squad_w0_keys.json (accuracy / rank ratio overall and by answer type), partial rows every 50 items (E57_squad_w0_reads_partial.json).
  python3 e57_squad_w0_reads.py --bundle B --out O --cycles 1 --npos 1 --canary-npos 1 --arms NAT --smoke 0 --items-file F.json [--n-items N] --stage reads   |   --selftest"""
import sys,json,time,hashlib,pathlib,collections
import numpy as np
HERE=pathlib.Path(__file__).resolve().parent
for p in (HERE,HERE/'e57',HERE.parents[1]/'scripts/exp04_what_ttt_retains',HERE.parents[1]/'scripts/exp05_write_dynamics'):sys.path.insert(0,str(p))
import e57_reads as RD
EXPECT='d80f9987'
F={'items':None,'n':None}
def _pop(flag):
    if flag in sys.argv:
        i=sys.argv.index(flag);v=sys.argv[i+1];del sys.argv[i:i+2];return v
def load_items_file(path):
    b=json.load(open(path));h=b.pop('sha256');s=json.dumps(b,ensure_ascii=False,sort_keys=True);assert hashlib.sha256(s.encode()).hexdigest()==h and h.startswith(EXPECT),('items file hash mismatch',h[:12])
    items=[]
    for r in b['items']:
        c=r['candidates'];g=r['gold_index'];cands=[c[g]]+[x for i,x in enumerate(c) if i!=g]
        items.append(dict(id=r['id'],title=r['title'],type=r['type'],wh=r['wh'],run=r['run'],prefix=r['frame'].strip(),context=r['context'],cands=cands,kinds=['gold']+['distractor']*(len(cands)-1),gold_idx=0,hard_idx=[],all_idx=list(range(len(cands))),pool_idx=[],event=r['title']))
    return items,h
def row_of(it,cond,lp,nt):
    lp=np.asarray(lp,np.float64);nt=np.asarray(nt,np.float64);mean=lp/nt;ms=RD.metrics(lp,0,[],list(range(len(lp))),None);mm=RD.metrics(mean,0,[],list(range(len(lp))),None)
    return dict(id=it['id'],title=it['title'],type=it['type'],wh=it['wh'],run=it['run'],condition=cond,cand_surface=it['cands'],cand_ntok=[int(x) for x in nt],cand_lp=[float(x) for x in lp],cand_mean=[float(x) for x in mean],rank_sum=int(ms['rank_all']),n=int(ms['n_all']),acc_sum=bool(ms['acc_all']),rank_ratio_sum=float(ms['rank_ratio_all']),rank_mean=int(mm['rank_all']),acc_mean=bool(mm['acc_all']),rank_ratio_mean=float(mm['rank_ratio_all']))
def keys_of(rows):
    out={}
    for cond in sorted({r['condition'] for r in rows}):
        d={}
        for g in ['ALL']+sorted({r['type'] for r in rows}):
            R=[r for r in rows if r['condition']==cond and (g=='ALL' or r['type']==g)]
            if R:d[g]=dict(n=len(R),acc_sum=float(np.mean([r['acc_sum'] for r in R])),rank_ratio_sum=float(np.mean([r['rank_ratio_sum'] for r in R])),acc_mean=float(np.mean([r['acc_mean'] for r in R])),rank_ratio_mean=float(np.mean([r['rank_ratio_mean'] for r in R])))
        out[cond]=d
    return out
def squad_reads(a,out,items_bank,bsrc,iv,hits,shtab,ids_all,n,N,kks,arms,tk,LD,rt,R,cw0,lg,can):
    import e55_gpu as RG
    items,h=load_items_file(F['items']);items=items[:int(F['n'])] if F['n'] else items;out=pathlib.Path(out);out.mkdir(parents=True,exist_ok=True)
    rt32=LD.runtime_fp32();rd=RG.Reader55(rt,rt32,LD.packing,LD._TailCtx,LD.reuse.get_filter_spec,LD.reuse._hash_pytree,LD.jax,LD.eqx,LD.jnp)
    raw,ntok,bad=RD.build_rows(rt.tok,items);assert not bad,bad
    forbid=sorted({s for it in items_bank for s in it['variants']+[it['cands'][0]] if len(s)>=4 and not any(ch.isdigit() for ch in s)});out_tail=R.out_tail_8192(LD,rt.tok,rt.bos,forbid);assert len(out_tail)==8192
    lg(f'{len(items)} items, {len(raw)} candidate rows, items file sha {h[:12]}; OUT tail sha {hashlib.sha256(np.asarray(out_tail,np.int64).tobytes()).hexdigest()[:16]}')
    off=[0]
    for it in items:off.append(off[-1]+len(it['cands']))
    rows=[];tim={}
    def score(P,m32):
        with LD.jax.default_matmul_precision('highest'):return np.asarray(P.r1(m32))
    m32=rd.to_fp32(rt.base);GROUP=40;t0=time.monotonic()
    for g0 in range(0,len(items),GROUP):
        sub=items[g0:g0+GROUP];P=rd.packed(raw[off[g0]:off[g0+len(sub)]],out_tail);flat=score(P,m32);o=0
        for k,it in enumerate(sub):nc=len(it['cands']);rows.append(row_of(it,'OUT',flat[o:o+nc],ntok[g0+k]));o+=nc
        del P
        if (g0//GROUP)%5==0:lg(f'  OUT pack {g0//GROUP+1}/{(len(items)+GROUP-1)//GROUP}: {time.monotonic()-t0:.0f}s');(out/'E57_squad_w0_reads_partial.json').write_text(json.dumps(dict(done='OUT partial',rows=rows),default=float))
    tim['OUT']=time.monotonic()-t0;lg(f'OUT done in {tim["OUT"]:.0f}s: acc_sum {np.mean([r["acc_sum"] for r in rows]):.3f}');t0=time.monotonic();tail=list(out_tail);maxp=0
    for k,it in enumerate(items):
        pid=LD.stage2.enc(rt.tok,'\n\n'+' '.join(it['context'].split())+'\n\n').ids;maxp=max(maxp,len(pid));assert len(pid)<7000,(it['id'],len(pid))
        ctx=tail[:8192-len(pid)]+list(pid);assert len(ctx)==8192
        P=rd.packed(raw[off[k]:off[k+1]],ctx);flat=score(P,m32);rows.append(row_of(it,'IN',flat,ntok[k]));del P
        if k%50==0:lg(f'  IN item {k+1}/{len(items)}: {time.monotonic()-t0:.0f}s');(out/'E57_squad_w0_reads_partial.json').write_text(json.dumps(dict(done='IN partial',rows=rows),default=float))
    tim['IN']=time.monotonic()-t0;kn=keys_of(rows)
    (out/'E57_squad_w0_reads.json').write_text(json.dumps(dict(meta=dict(items_sha256=h,rules='e57_public_qa_build_squad.py',n_items=len(items),context_out='neutral SQuAD dev tail 8192 tokens (news OUT tail)',context_in='tail with the item paragraph at its end (8192 tokens)',max_paragraph_tokens=maxp,timings=tim),rows=rows),default=float))
    (out/'E57_squad_w0_keys.json').write_text(json.dumps(dict(items_sha256=h,keys=kn),indent=1,default=float));lg('wrote E57_squad_w0_reads.json and E57_squad_w0_keys.json');lg(json.dumps({c:round(v['ALL']['acc_sum'],3) for c,v in kn.items()}))
def selftest():
    it=dict(id='a',title='T',type='LOW',wh='what',run=0,prefix='q?',cands=['g','x','y'],kinds=['gold','distractor','distractor'],gold_idx=0,hard_idx=[],all_idx=[0,1,2],pool_idx=[],event='T')
    r=row_of(it,'OUT',[-4.0,-6.0,-5.0],[2,3,1]);assert r['acc_sum'] and r['rank_sum']==1 and r['n']==3 and abs(r['cand_mean'][0]+2.0)<1e-9
    r2=row_of(it,'IN',[-9.0,-6.0,-5.0],[2,3,1]);assert not r2['acc_sum'] and r2['rank_sum']==3
    kn=keys_of([r,r2]);assert kn['OUT']['ALL']['n']==1 and kn['IN']['LOW']['acc_sum']==0.0
    print('e57_squad_w0_reads selftest ok')
def main():
    if '--selftest' in sys.argv:selftest();return
    F['items']=_pop('--items-file');F['n']=_pop('--n-items');assert F['items'],'--items-file is required'
    import e57_main_run as M
    M.stage_reads=squad_reads;M.main()
if __name__=='__main__':main()
