"""FictionalQA W0 pre-screen (GPU forward only, NO weight update; new file; Hans 11/18 'fictionalQA 사전 점검 승인'; thread-server). For every world fq_201..208_cap3_VAR.json (run seed labels 211..218, SEED_LABEL of fq_world.py) all 24 facts (the 16 world facts + the 8 reserve facts, in that order) are read at W0 (E2E 3B, the exact reader of the NATS runs: question prefix ends 'Answer:', the out-of-window neutral 8192-token tail, summed span log-prob of the candidate).
Candidates of fact i (0..23): its own gold first, then the golds of the facts i+1 .. i+15 (mod 24) -> 16 candidates, all 3 Llama-3 tokens, equal length (asserted). The margin is gold minus best wrong candidate (summed span log-prob). 'screened out' = margin > 0 (Hans: W0 margin > 0 is removed). NOTE: the candidate set of the screen (24-ring) differs from that of the final item set (the 16 golds of the final world): the screen is a property of question + gold against 15 other golds of the same world, the final W0 margins are read again in the NATS run (W0 'out').
Output {out}/fq_w0screen_<label>.json: label, builder_seed, world sha, per fact id / role (world|reserve) / margin / top1 / rank / scores (24x16), tail sha, model hash; the selection (fq_w0_select.py, CPU, new file) turns it into the replacement list.
  python3 fq_screen_run.py --worlds DIR --out OUT [--labels 211,212,...] [--smoke] | --selftest"""
import sys,json,time,hashlib,argparse,pathlib
import numpy as np
HERE=pathlib.Path(__file__).resolve().parent;ROOT=HERE.parents[1]
for p in (HERE,HERE/'e57',ROOT/'scripts/exp04_what_ttt_retains',ROOT/'scripts/exp05_write_dynamics'):sys.path.insert(0,str(p))
import fq_world as FW
NF=16;NRES=8;NALL=NF+NRES;NC=16
def screen_items(world):
    """24 screen items in gen_facts item format (kind 'fact'; gold first)"""
    fs=list(world['facts'])+list(world['reserve'])[:NRES];assert len(fs)==NALL,len(fs);items=[]
    for i,f in enumerate(fs):
        ring=[fs[(i+j)%NALL] for j in range(NC)];assert ring[0] is f
        items.append(dict(kind='fact',fact=i,group=0,name_id=i,place_id=i,name=f['qid'],qframe=0,prefix=f['question'],cands=[' '+g['gold'] for g in ring],gold_idx=0,cand_fact=[(i+j)%NALL for j in range(NC)],cand_group=[0]*NC,cand_same=[True]*NC,cand_exposed=[True]*NC,n_rank=NC,role='world' if i<NF else 'reserve'))
    return items
def margins(S):
    S=np.asarray(S,np.float64);return S[:,0]-S[:,1:].max(1),(S.argmax(1)==0),(S>S[:,:1]).sum(1)+1
def run(a):
    out=pathlib.Path(a.out);out.mkdir(parents=True,exist_ok=True)
    import e55_run as R55,e55_gpu as RG,gen_facts as G
    LD,s3,C=R55.lab();rt=LD.Runtime();rt32=LD.runtime_fp32();rd=RG.Reader55(rt,rt32,LD.packing,LD._TailCtx,LD.reuse.get_filter_spec,LD.reuse._hash_pytree,LD.jax,LD.eqx,LD.jnp);tok=rt.tok
    labels=[int(x) for x in a.labels.split(',')];inv={v:k for k,v in FW.SEED_LABEL.items()};mh=LD.reuse._hash_pytree(rt.base.inner_parameters())[0] if hasattr(LD.reuse,'_hash_pytree') else None
    for lab in labels:
        fn=out/f'fq_w0screen_{lab}.json'
        if fn.exists() and not a.rerun:print('exists, reused',fn,flush=True);continue
        t0=time.monotonic();w=FW.load_world(pathlib.Path(a.worlds)/f'fq_{inv[lab]}_cap3_VAR.json');items=screen_items(w)
        raw=R55.rows_for(LD,rt,items);RG.check_rows(raw,len(items),NC)
        forbid=[f['question'] for f in w['facts']+w['reserve']]+[f['gold'] for f in w['facts']+w['reserve']]
        tail=R55.out_tail_8192(LD,tok,rt.bos,forbid);P=rd.packed(raw,tail);S=np.asarray(rd.score(P,rt.base,len(items),NC),np.float64)
        m,t1,rk=margins(S);rec=dict(label=lab,builder_seed=inv[lab],world_sha=w['sha'],tail_sha=hashlib.sha256(np.asarray(tail,np.int64).tobytes()).hexdigest()[:16],model_fast_hash=str(mh),facts=[dict(idx=i,role=items[i]['role'],fact_id=(w['facts']+w['reserve'])[i]['id'],margin=float(m[i]),top1=bool(t1[i]),rank=int(rk[i]),screened_out=bool(m[i]>0)) for i in range(NALL)],scores=S.tolist(),sec=time.monotonic()-t0,candidates='own gold + golds of facts i+1..i+15 mod 24',margin_def='summed span log-prob gold minus best wrong, out-of-window neutral 8192-token tail, W0')
        json.dump(rec,open(fn,'w'));print(f'world {lab} (builder {inv[lab]}): world facts margin>0: {int((m[:NF]>0).sum())}/16, reserve margin>0: {int((m[NF:]>0).sum())}/8, mean margin {m.mean():+.2f}, {time.monotonic()-t0:.0f}s',flush=True)
def selftest():
    import tempfile,collections
    W=ROOT/'.fq_worlds_unavailable'
    S=np.zeros((2,16));S[0,0]=1.0;S[1,3]=2.0;m,t1,rk=margins(S);assert m[0]>0 and t1[0] and rk[0]==1 and m[1]<0 and not t1[1] and rk[1]==2
    wd=ROOT/"experiments/exp07_write_ladder/fq/worlds"
    if wd.exists():
        w=FW.load_world(wd/'fq_201_cap3_VAR.json');it=screen_items(w);assert len(it)==24 and all(len(x['cands'])==16 and x['cands'][0]==' '+(w['facts']+w['reserve'])[i]['gold'] for i,x in enumerate(it))
        tk=FW.load_tokenizer();assert {len(tk.encode(c,add_special_tokens=False).ids) for x in it for c in x['cands']}=={3},'candidate lengths'
        assert len({c for x in it for c in x['cands']})==24,'24 distinct golds'
    print('fq_screen_run selftest ok')
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--worlds');ap.add_argument('--out');ap.add_argument('--labels',default=','.join(str(x) for x in range(211,219)));ap.add_argument('--rerun',action='store_true');ap.add_argument('--selftest',action='store_true');a=ap.parse_args()
    if a.selftest:selftest();return
    run(a)
if __name__=='__main__':main()
