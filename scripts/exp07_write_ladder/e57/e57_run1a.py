"""E57 stage 1a (design section 11.3 item 5): plain TTT (NAT, no reads of any kind, no masks, no Adam) over R rounds of the frozen fixed-order news round (job 207), canary / DCLM damage measured at the end of the listed rounds, R_max = the largest R whose canary rise stays <= the limit (0.10). 1a-1 = rounds 4,8,12,16; 1a-2 continues the SAME run (resume bundle) with 20,24,28,32.
The stream is a cyclic repetition of one round; the natural K,V of the window and the prefix outputs are recorded for the first P rounds only and reused for later rounds (e55_core.run_stream periodic=...; P = ceil(prefix layers x 8192 / round tokens) + 1 so that the reuse is exact, capped by the RoPE table 2 x seq_len: with the cap the reuse is an approximation and the report says so). Query positions of the reused rounds are those of the last recorded round (relative RoPE); real absolute positions would run out of the RoPE table.
Modes: --check-only (CPU, no jax: frozen files vs manifest, plan, memory estimate), default (GPU, server python)."""
import argparse,json,os,sys,math,time,hashlib,pathlib
import numpy as np
HERE=pathlib.Path(__file__).resolve().parent;sys.path.insert(0,str(HERE.parent))
CH=1024;WIN=8192;LIMIT=0.10
def choose_rec_rounds(rc,seq_len,n_prefix):
    """-> (P, exact, need, maxfit)"""
    need=math.ceil(n_prefix*WIN/(rc*CH))+1;maxfit=int((2*seq_len)//(rc*CH))
    if os.environ.get('ROPE_TABLE'):       # upstream table lookup: recorded positions must stay inside 2*seq_len
        assert maxfit>=2,('one round is longer than half of the RoPE table',rc*CH,2*seq_len);P=max(2,min(need,maxfit))
    else:P=max(2,need)                      # rope_fly (no table): the exact number of rounds always fits
    return P,bool(P>=need),need,maxfit
def load_frozen(fd):
    fd=pathlib.Path(fd);man=json.load(open(fd/'freeze_manifest.json'))
    for name,sha in man['files'].items():
        got=hashlib.sha256((fd/name).read_bytes()).hexdigest();assert got==sha,('frozen file changed',name,got[:12],sha[:12])
    arr=np.load(fd/'round_ids.npy');assert arr.dtype==np.int32 and len(arr)%CH==0 and hashlib.sha256(arr.tobytes()).hexdigest()==man['stats']['sha256_round_ids']
    return arr,man
def save_sts(sts,path):
    import jax
    np.savez(path,*[np.asarray(x) for x in jax.tree.leaves(sts)])
def load_sts(path,template):
    import jax,jax.numpy as jnp
    z=np.load(path);leaves=[jnp.asarray(z[f'arr_{i}']) for i in range(len(z.files))];td=jax.tree.structure(template);assert td.num_leaves==len(leaves);return jax.tree.unflatten(td,leaves)
def n_prefix_layers(rt):
    try:
        import jax;return int(jax.tree_util.tree_leaves(rt.base.language_model.model.h.prefix_blocks)[0].shape[0])
    except Exception:return 24
def run_1a(L,ids_round,rounds,out_dir,tag,seq_len,n_prefix,dcl=None,resume=None,log=print,limit=LIMIT):
    """L: Ladder55 with the canary attached. rounds: ascending list of round counts at whose ends the damage is measured. resume: dict(state=npz file of the weights, sts=npz file, chunk=last finished chunk, canary_w0=..., reads=[earlier read dicts])"""
    import jax
    out=pathlib.Path(out_dir);out.mkdir(parents=True,exist_ok=True);rc=len(ids_round)//CH;Rmax=max(rounds);nch=Rmax*rc;ids=np.tile(np.asarray(ids_round,np.int64),Rmax)
    P,exact,need,maxfit=choose_rec_rounds(rc,seq_len,n_prefix);P=min(P,Rmax);reads={R*rc-1:R for R in rounds}
    start=0;w0=None;weights0=None;sts0=None;prev=[]
    if resume:
        start=int(resume['chunk'])+1;assert start<nch and all(k>=start for k in reads),('resume: every new read must lie after the finished chunk',start,sorted(reads))
        weights0=L.load_state(resume['state']);sts0=load_sts(resume['sts'],L.rt.state_suffix);w0=resume['canary_w0'];prev=list(resume.get('reads',[]))
    t0=time.monotonic()
    r=L.run_stream('NAT',1,ids,nch=nch,tag=tag,log=log,read_chunks=set(reads),canary_chunks=set(reads),periodic={'round_chunks':rc,'rec_chunks':P*rc},start_chunk=start,weights0=weights0,sts0=sts0)
    w0=r.get('canary_w0',w0);assert w0 is not None
    rows=list(prev)
    for k,R in sorted(reads.items(),key=lambda x:x[1]):
        rd=r['reads'][k];c=rd.get('canary') or {}
        row=dict(R=R,chunk=k,state=rd['_file'],hash=rd['_hash'],canary=c,crise_gen=c.get('_gen_nll',float('nan'))-w0['_gen_nll'],crise_tail=c.get('_tail_nll',float('nan'))-w0['_tail_nll'])
        row['crise']=max(row['crise_gen'],row['crise_tail'])
        lg=r['log'][k-start];seg=r['log'][max(0,k-start-rc+1):k-start+1];row['mean_ce_last_round']=float(np.mean([e['ce'] for e in seg]));row['mean_update_norm_last_round']=float(np.mean([e['update_norm'] for e in seg]))
        if dcl is not None:
            m=L.load_state(rd['_file']);row['dclm_nll']=float(dcl(m))
        rows.append(row)
    bad=min([x['R'] for x in rows if x['crise']>limit],default=10**9);cand=[x['R'] for x in rows if x['crise']<=limit and x['R']<bad];Rmax_ok=max(cand) if cand else None
    summ=dict(tag=tag,rounds=list(rounds),round_chunks=rc,nch=nch,start_chunk=start,limit=limit,canary_w0=w0,reads=rows,R_max=Rmax_ok,R_max_rule=f'largest R in the list with canary rise <= {limit} before the first R above it',
        rec_rounds=P,rec_exact=exact,rec_rounds_needed=need,rec_rounds_fit_in_rope_table=maxfit,n_prefix_layers=n_prefix,seq_len=seq_len,sec_run=time.monotonic()-t0,sec_per_chunk_mean=float(np.mean([e['sec'] for e in r['log']])),
        clip_bind_frac=r['clip_bind_frac'],cum_update_norm=r['cum_update_norm'],final_hash=r['final_hash'],stream_hash=r['stream_hash'])
    sp=out/f'E57_{tag}_resume_R{Rmax}.sts.npz';save_sts(r['sts_final'],sp);summ['resume_bundle']=dict(state=rows[-1]['state'],sts=str(sp),chunk=nch-1,canary_w0=w0,reads=rows)
    r.pop('model',None);r.pop('sts_final',None);r['summary']=summ
    json.dump(r,open(out/f'E57_{tag}_R{Rmax}.saved.json','w'),default=float);json.dump(summ,open(out/f'E57_{tag}_R{Rmax}.summary.json','w'),indent=1,default=float)
    return summ
def decision(summ,rounds_all=None):
    """the pre-set branch of design 11.3 (5) / 11.4 in words"""
    rows=summ['reads'];Rm=summ['R_max'];Rl=max(x['R'] for x in rows);last=rows[-1]
    if all(x['crise']<=summ['limit'] for x in rows):return f"canary rise <= {summ['limit']} at every measured R up to {Rl} (last {last['crise']:.3f}): R_max >= {Rl}; continue with 1a-2 (resume bundle) if {Rl} < 32"
    if Rm is None:return f"canary rise > {summ['limit']} already at R = {rows[0]['R']}: R_max < {rows[0]['R']}; Adam-0.5N centred design (11.4)"
    return f"R_max = {Rm} (canary rise > {summ['limit']} at the next measured R); 11.4: exposure c x R_max decides between the 5-arm design and the Adam-0.5N centred design"
def report(summ,man=None):
    L=['# E57 stage 1a report (plain TTT, no reads, periodic K,V sharing)','',f"- tag {summ['tag']}, rounds measured {summ['rounds']}, round = {summ['round_chunks']} chunks, chunks run {summ['start_chunk']}..{summ['nch']-1}, {summ['sec_run']/3600:.2f} h ({summ['sec_per_chunk_mean']:.2f} s per chunk mean)",
       f"- K,V sharing: {summ['rec_rounds']} rounds recorded (need {summ['rec_rounds_needed']} for exactness with {summ['n_prefix_layers']} prefix layers; positions: {'table-free RoPE (rope_fly), no position limit' if not os.environ.get('ROPE_TABLE') else 'upstream table of 2 x seq_len = '+str(2*summ['seq_len'])+' positions, allows '+str(summ['rec_rounds_fit_in_rope_table'])+' rounds'}): **{'exact' if summ['rec_exact'] else 'APPROXIMATE (context beyond the recorded rounds is truncated for the reused rounds)'}**",
       f"- canary at W0: gen {summ['canary_w0']['_gen_nll']:.4f}, tail {summ['canary_w0']['_tail_nll']:.4f}; limit on the rise (max of both) {summ['limit']}; clip bound fraction {summ['clip_bind_frac']:.3f}",'','| R | chunk | canary gen | canary tail | rise gen | rise tail | rise (max) | within limit | DCLM NLL | mean CE last round | mean update norm last round |','|---|---|---|---|---|---|---|---|---|---|---|']
    for x in summ['reads']:
        c=x['canary'];L.append(f"| {x['R']} | {x['chunk']} | {c.get('_gen_nll',float('nan')):.4f} | {c.get('_tail_nll',float('nan')):.4f} | {x['crise_gen']:+.4f} | {x['crise_tail']:+.4f} | {x['crise']:+.4f} | {'yes' if x['crise']<=summ['limit'] else 'NO'} | {x.get('dclm_nll','-') if isinstance(x.get('dclm_nll'),str) or x.get('dclm_nll') is None else format(x['dclm_nll'],'.4f')} | {x['mean_ce_last_round']:.4f} | {x['mean_update_norm_last_round']:.4f} |")
    L+=['','- **R_max = '+str(summ['R_max'])+'** ('+summ['R_max_rule']+')','- decision: '+decision(summ),'- exposure of a target with c lead occurrences = c x R (c = 3-4 gives '+', '.join(f"R={x['R']}: {3*x['R']}-{4*x['R']}" for x in summ['reads'])+').']
    return '\n'.join(L)+'\n'
def check_only(a):
    arr,man=load_frozen(a.frozen);rc=len(arr)//CH;rounds=[int(x) for x in a.rounds.split(',')];Rmax=max(rounds);nch=Rmax*rc
    P,exact,need,maxfit=choose_rec_rounds(rc,a.seq_len,a.n_prefix)
    kv_gib=P*rc*CH*a.n_prefix*a.kv_dim*2*2/2**30            # K and V, bf16, [layers, tokens, kv_dim]
    print(f'1a check-only OK: frozen round {rc} chunks ({len(arr)} tokens, sha {man["stats"]["sha256_round_ids"][:12]}); rounds {rounds} -> {nch} updates, reads at chunks {[R*rc-1 for R in rounds]}; K,V recorded for {P} rounds ({P*rc} chunks, about {kv_gib:.0f} GiB host with {a.n_prefix} prefix layers x kv_dim {a.kv_dim}); exact reuse needs {need} rounds -> {"exact" if exact else "APPROXIMATE"}; largest recorded position {P*rc*CH}; RoPE: {"table-free (rope_fly), no position limit" if not os.environ.get("ROPE_TABLE") else "upstream table 2 x seq_len = "+str(2*a.seq_len)+" (allows "+str(maxfit)+" rounds)"}')
    print(f'  estimate (unmeasured): {nch*a.sec_lo/3600:.1f}-{nch*a.sec_hi/3600:.1f} h for {nch} updates at {a.sec_lo}-{a.sec_hi} s per chunk, plus K,V recording and model load')
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--frozen',required=True);ap.add_argument('--out',default=None);ap.add_argument('--tag',default='1a1');ap.add_argument('--rounds',default='4,8,12,16')
    ap.add_argument('--resume',default=None,help='summary.json of the earlier part (its resume_bundle continues the run)');ap.add_argument('--dclm-ids',default=None);ap.add_argument('--dclm-n',type=int,default=100)
    ap.add_argument('--check-only',action='store_true');ap.add_argument('--seq-len',type=int,default=131072);ap.add_argument('--n-prefix',type=int,default=20);ap.add_argument('--kv-dim',type=int,default=3072)
    ap.add_argument('--sec-lo',type=float,default=3.0);ap.add_argument('--sec-hi',type=float,default=5.0);ap.add_argument('--rerun',action='store_true');a=ap.parse_args()
    if a.check_only:check_only(a);return
    import e55_run as R
    LD,s3,C=R.lab();arr,man=load_frozen(a.frozen);OUT=pathlib.Path(a.out or os.environ.get('EXP07_OUT','.'));rounds=[int(x) for x in a.rounds.split(',')]
    rep=OUT/f'e57_{a.tag}_report.md'
    if rep.exists() and not a.rerun:print('report exists (exit 8; --rerun to redo):',rep);sys.exit(8)
    rt=LD.Runtime();can=R.Canary55(LD,rt);L=C.Ladder55(rt,save_dir=OUT/'states'/f'E57_{a.tag}',canary=can)
    seq_len=int(rt.cfg.model.seq_len);npre=n_prefix_layers(rt);print(f'runtime: seq_len {seq_len} (RoPE table {2*seq_len}), prefix layers {npre}',flush=True)
    dcl=R.DclmNll(LD,rt,a.dclm_ids,a.dclm_n) if a.dclm_ids else None
    resume=None
    if a.resume:resume=json.load(open(a.resume))['resume_bundle']
    summ=run_1a(L,arr,rounds,OUT,a.tag,seq_len,npre,dcl=dcl,resume=resume,log=lambda *x:print(*x,flush=True))
    open(rep,'w').write(report(summ,man));print(open(rep).read())
if __name__=='__main__':main()
