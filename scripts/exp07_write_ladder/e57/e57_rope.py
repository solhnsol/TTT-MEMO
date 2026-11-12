"""E57 G3: RoPE position check on CPU (no model, no weights). Replicates the upstream table arithmetic (ttt/model/attention.py precompute_freqs_cis: freqs = 1/theta^(2i/dim) in float32, outer(t, freqs) in float32, cos / sin in float32) ONLY at the requested positions (an elementwise product, so the values equal the table entries) and compares the relative rotation of positions t and t+d with the exact (float64) angle d*theta_i.
Two numbers per absolute position t and offset d: max phase error over the head dimensions (rad) and the relative error of a random q.k score (median over random vectors). Also reports the RoPE table size 2*seq_len (positions beyond it index out of range) and the largest position of the periodic K,V scheme."""
import numpy as np
def table_at(pos,dim=128,theta=500000.0):
    """float32 values as the upstream table holds them at integer positions `pos` (numpy float32 arithmetic = jax CPU float32 elementwise)"""
    freqs=(1.0/(np.float32(theta)**(np.arange(0,dim,2)[:dim//2].astype(np.float32)/np.float32(dim)))).astype(np.float32)
    ang=np.outer(np.asarray(pos).astype(np.float32),freqs).astype(np.float32)
    return (np.cos(ang)+1j*np.sin(ang)).astype(np.complex64),freqs.astype(np.float64)
def check(positions=(0,131072,262143,524288,1000000,1700000,3400000),offsets=(1,64,1024,8191),dim=128,theta=500000.0,nvec=64,seed=0):
    rng=np.random.default_rng(seed);rows=[]
    q=rng.standard_normal((nvec,dim//2))+1j*rng.standard_normal((nvec,dim//2));k=rng.standard_normal((nvec,dim//2))+1j*rng.standard_normal((nvec,dim//2))
    th=(1.0/(theta**(np.arange(0,dim,2)[:dim//2]/dim)))
    for t in positions:
        for d in offsets:
            ft,_=table_at([t],dim,theta);ftd,_=table_at([t+d],dim,theta);f0,_=table_at([0],dim,theta);fd,_=table_at([d],dim,theta)
            rel=(ftd[0].astype(np.complex128)*np.conj(ft[0].astype(np.complex128)))                 # rotation by d as seen at absolute position t
            exact=np.exp(1j*d*th);perr=np.abs(np.angle(rel*np.conj(exact)))
            s_t=np.real(np.sum((q*ftd[0].astype(np.complex128))*np.conj(k*ft[0].astype(np.complex128)),axis=-1))        # q at position t+d, k at position t
            s_0=np.real(np.sum((q*fd[0].astype(np.complex128))*np.conj(k*f0[0].astype(np.complex128)),axis=-1))        # the same pair at positions d and 0
            rse=np.abs(s_t-s_0)/(np.linalg.norm(q,axis=-1)*np.linalg.norm(k,axis=-1))                                    # relative to the Cauchy-Schwarz bound of the score
            rows.append(dict(t=int(t),d=int(d),max_phase_err=float(perr.max()),median_phase_err=float(np.median(perr)),median_rel_score_err=float(np.median(rse)),p95_rel_score_err=float(np.quantile(rse,0.95))))
    return rows
def run(round_tokens,rec_rounds=2,seq_len=131072,rounds_max=32,dim=128,theta=500000.0):
    """-> dict with the table, per (t,d) errors and a verdict for (a) plain absolute positions up to the end of the longest run and (b) the periodic scheme (positions < rec_rounds rounds)"""
    end_plain=round_tokens*rounds_max;end_per=round_tokens*rec_rounds;table=2*seq_len
    pos=sorted({0,131072,table-1,end_per-1,end_per,1000000,end_plain//2,end_plain-1,3400000})
    rows=check(pos,dim=dim,theta=theta);tol=0.02      # median relative score error; a run counts as position-invariant when the worst row at positions <= its end stays below tol
    def worst(limit):
        r=[x for x in rows if x['t']+x['d']<=limit];return max(x['median_rel_score_err'] for x in r) if r else float('nan'),max(x['max_phase_err'] for x in r) if r else float('nan')
    wp=worst(end_per);wa=worst(end_plain)
    return dict(table_positions=table,seq_len=seq_len,round_tokens=round_tokens,end_periodic=end_per,end_plain_rounds_max=end_plain,rows=rows,tol_median_rel_score_err=tol,
        periodic_inside_table=bool(end_per<=table),periodic_worst_median_rel_score_err=wp[0],periodic_worst_max_phase_err=wp[1],periodic_ok=bool(end_per<=table and wp[0]<=tol),
        plain_inside_table=bool(end_plain<=table),plain_worst_median_rel_score_err=wa[0],plain_worst_max_phase_err=wa[1],plain_ok=bool(end_plain<=table and wa[0]<=tol),plain_numeric_ok=bool(wa[0]<=tol),periodic_numeric_ok=bool(wp[0]<=tol))
def report(R):
    L=['## G3 RoPE position check (float32 table arithmetic, CPU)','',f"- RoPE table length 2 x seq_len = {R['table_positions']} positions (seq_len {R['seq_len']}); one round = {R['round_tokens']} tokens. Positions beyond the table are out of range (the lookup is not an error in JAX): a plain run of {R['end_plain_rounds_max']} tokens ({'inside' if R['plain_inside_table'] else 'OUTSIDE'} the table) is {'possible' if R['plain_inside_table'] else 'not possible'} without remapping.",
       f"- periodic K,V scheme (window and positions taken from the last of the first rec_rounds recorded rounds): largest position {R['end_periodic']} ({'inside' if R['periodic_inside_table'] else 'OUTSIDE'} the table); worst median relative score error {R['periodic_worst_median_rel_score_err']:.4f}, worst max phase error {R['periodic_worst_max_phase_err']:.3f} rad -> **{'OK' if R['periodic_ok'] else 'NOT OK'}** (tolerance {R['tol_median_rel_score_err']} on the median relative score error)",
       f"- plain absolute positions up to {R['end_plain_rounds_max']}: worst median relative score error {R['plain_worst_median_rel_score_err']:.4f}, worst max phase error {R['plain_worst_max_phase_err']:.3f} rad -> numerically {'within' if R['plain_numeric_ok'] else 'OUTSIDE'} tolerance; usable as a plain run: **{'YES' if R['plain_ok'] else 'NO'}**{'' if R['plain_inside_table'] else ' (the lookup would leave the table)'}",'',
       '| absolute position t | offset d | max phase error (rad) | median phase error | median rel. score error | 95th pct rel. score error |','|---|---|---|---|---|---|']
    for x in R['rows']:L.append(f"| {x['t']} | {x['d']} | {x['max_phase_err']:.4f} | {x['median_phase_err']:.5f} | {x['median_rel_score_err']:.4f} | {x['p95_rel_score_err']:.4f} |")
    return '\n'.join(L)+'\n'
def _selftest():
    rows=check(positions=(0,3400000),offsets=(1,1024))
    r0=[x for x in rows if x['t']==0];r3=[x for x in rows if x['t']==3400000]
    assert max(x['max_phase_err'] for x in r0)<1e-3,r0
    assert max(x['max_phase_err'] for x in r3)>10*max(x['max_phase_err'] for x in r0),(r0,r3)       # float32 phase quantisation at 3.4M positions is visible
    print('e57_rope selftest ok')
if __name__=='__main__':
    import sys
    if '--selftest' in sys.argv:_selftest()
    else:print(report(run(int(sys.argv[1]) if len(sys.argv)>1 else 105472)))
