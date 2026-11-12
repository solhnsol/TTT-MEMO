"""E57 job 1280 measurement-runner, CPU PART 1 (new file; core-claims thread 11/12; GPU-free): turns the raw per-branch-chunk token NLL arrays and read margins that the GPU runner will produce into the seed-level record required by e57_1280_analyze.INPUT_SCHEMA, with every definition taken from freeze A (registration items 3, 9, e57_1280_analyze.evaluate). Nothing here runs the model. The GPU runner (not yet written: recompute-oracle / full-window recompute, all-layer all-position mask, branch loop with select_branch_chunks) only has to fill RawSeed.
  RawSeed fields (all numpy arrays over the SAME token set per branch chunk; nat/token):
    chunks              list of branch chunk indices (measured_chunk_min = min, must be >= 8)
    nll[<cond>][i]      per-token NLL array of branch chunk i under condition <cond> in: NC (natural, cached path), A (cached oracle), C (recomputed natural), B (recomputed oracle), A_P / B_P (placebo versions of A / B), BAND (recomputed band mask, mask arm ALLQ256)
    token_ids[i]        the token ids scored in chunk i (identity across conditions is asserted by comparing token_ids per condition: tok[<cond>][i])
    pre_update[i]       True when the scored chunk is scored BEFORE its own update (prequential, no leakage of its own write)
    margins             dict arm -> seed mean per-fact margin (nat/fact) at the read: NATS, NATMASK, RECOMP, RECOMP_PL
    gain_noprior        float nat/fact: NATMASK - NATS margin on the filler-replaced facts (definition from the v3 analysis draft, commit 0f5352d; not in the freeze A text, see the definition-gap note)
    flags               dict t2_ok, t3_ok, p16_ok (runner checks, booleans), half_pl_unmatched (int of 32), determinism_abs (optional float). Definitions found in the earlier drafts (v3 commit 0f5352d, v7 commit ea8ea69), NOT in the freeze A text: t2_ok = the all-keys-visible mask is bit-identical to no mask; t3_ok = the token-substitution leak test passes for the recompute oracle (procedure not specified in any draft); p16_ok = p = 16 visibility check (the 1279 form: ORACLE / RAND / PLACEBO hide 0 keys at p = 16 and equal NATS bit for bit, tolerance |d logit| <= 1e-4 and |d margin| <= .01 G for a non-deterministic kernel; e57_oracle_analyze.py v2 eaadd24, p16_coherence); half_pl_unmatched = number of the 32 facts for which the HALF placebo is not matched to HALF in masked length (exact) and position (within 64 tokens) (v7 flag HALF_PL; matching rule from CONFOUND_TABLE)
Definitions (freeze A): nc, a, c, b, a_P, b_P = token-weighted means over the branch chunks of the per-token NLL of the condition; E_cache = a - nc, E_rec = b - c, X_MASK = E_mask = mean(BAND) - c (the recomputed band NLL minus the recomputed natural NLL); c_minus_nc_abs_max = max over all scored tokens of |C - NC| (the recomputed natural path must equal the cached natural path up to 1e-4); n_tokens = number of scored tokens per quantity, identical across quantities."""
import numpy as np
COND=('NC','A','C','B','A_P','B_P','BAND')
def _cat(raw,cond):return np.concatenate([np.asarray(x,float) for x in raw['nll'][cond]])
def build_record(raw):
    chunks=list(raw['chunks']);assert len(chunks)>0 and all(len(raw['nll'][c])==len(chunks) and len(raw['tok'][c])==len(chunks) for c in COND),'every condition needs one array per branch chunk'
    ref=raw['tok']['NC'];same=all(len(raw['tok'][c][i])==len(ref[i]) and np.array_equal(np.asarray(raw['tok'][c][i]),np.asarray(ref[i])) for c in COND for i in range(len(chunks)));v={c:_cat(raw,c) for c in COND}
    n_tokens={c:int(len(v[c])) for c in COND}
    if not same:n_tokens['NC']=-1                                                   # unequal token sets are made visible to the analyzer, which then labels the result '1 PROTOCOL / INSTRUMENT INVALID'
    fin=bool(all(np.isfinite(v[c]).all() for c in COND) and all(np.isfinite(raw['margins'][k]) for k in ('NATS','NATMASK','RECOMP','RECOMP_PL')) and np.isfinite(raw['gain_noprior']))
    m=lambda c:float(v[c].mean()) if len(v[c]) else float('nan');d=float(np.abs(v['C']-v['NC']).max()) if (len(v['C'])==len(v['NC'])) else float('inf')
    rec=dict(nc=m('NC'),a=m('A'),c=m('C'),b=m('B'),a_P=m('A_P'),b_P=m('B_P'),X_MASK=m('BAND')-m('C'),M_NATS=float(raw['margins']['NATS']),M_MASK=float(raw['margins']['NATMASK']),M_RECOMP=float(raw['margins']['RECOMP']),M_RECOMP_PL=float(raw['margins']['RECOMP_PL']),gain_noprior=float(raw['gain_noprior']),
        n_tokens=n_tokens,measured_chunk_min=int(min(chunks)),timing_pre_update=bool(all(raw['pre_update'])),c_minus_nc_abs_max=d,t2_ok=bool(raw['flags']['t2_ok']),t3_ok=bool(raw['flags']['t3_ok']),p16_ok=bool(raw['flags']['p16_ok']),finite=fin,half_pl_unmatched=int(raw['flags']['half_pl_unmatched']))
    if 'determinism_abs' in raw['flags']:rec['determinism_abs']=float(raw['flags']['determinism_abs'])
    return rec
def synthetic_raw(seed,E_cache=0.07,E_rec=0.30,E_mask=0.35,G=2.0,H=1.6,PLd=0.0,n_chunks=8,ntok=500,noise=0.01):
    """CPU mock of a runner output (no model): per-token NLLs with the requested effects, identical token ids across conditions; used only by the selftest"""
    r=np.random.default_rng(seed);base=2.0;ids=[r.integers(0,1000,ntok) for _ in range(n_chunks)];nc=[base+r.normal(0,noise,ntok) for _ in range(n_chunks)];f=lambda arr,delta:[x+delta for x in arr]
    nll=dict(NC=nc,C=nc,A=f(nc,E_cache),B=f(nc,E_rec),A_P=f(nc,PLd*0.1),B_P=f(nc,PLd*0.1),BAND=f(nc,E_mask));tok={c:ids for c in COND}
    return dict(chunks=list(range(8,8+n_chunks)),nll=nll,tok=tok,pre_update=[True]*n_chunks,margins=dict(NATS=-3.8,NATMASK=-3.8+G,RECOMP=-3.8+H,RECOMP_PL=-3.8+PLd),gain_noprior=1.1,flags=dict(t2_ok=True,t3_ok=True,p16_ok=True,half_pl_unmatched=0,determinism_abs=0.0))
def selftest():
    import e57_1280_analyze as A
    recs=[build_record(synthetic_raw(s)) for s in range(101,131)]
    for r in recs:
        assert all(k in r for k in A.INPUT_SCHEMA if k!='determinism_abs') and 'determinism_abs' in r and set(r)<=set(A.INPUT_SCHEMA),'record fields must equal INPUT_SCHEMA'
    assert A.validate_input(recs[0]) if hasattr(A,'validate_input') else True
    o=A.evaluate(recs,exact=False);assert 'tests' in o and o['instrument_gates']['token_sets_identical'] and o['instrument_gates']['finite'] and o['instrument_gates']['pre_update_timing'] and o['instrument_gates']['t_ge_8'] and o['instrument_gates']['c_nc_ok'],o['instrument_gates']
    assert abs(o['tests']['g2a']['mean']-0.30)<0.01 and abs(o['tests']['g2c']['mean']-2.0)<0.01 and abs(o['tests']['g2b']['mean']-0.35)<0.01,'E_rec, G, E_mask reproduced from per-token arrays'
    bad=synthetic_raw(5);bad['tok']['B']=[x[::-1].copy() for x in bad['tok']['B']];assert A.evaluate([build_record(bad)]*3,exact=False)['label'].startswith('1 PROTOCOL')                  # token sets differ -> invalid
    bad=synthetic_raw(6);bad['pre_update'][2]=False;assert not build_record(bad)['timing_pre_update']
    bad=synthetic_raw(7);bad['chunks'][0]=3;assert build_record(bad)['measured_chunk_min']==3
    bad=synthetic_raw(8);bad['nll']['C']=[x+1e-3 for x in bad['nll']['C']];assert build_record(bad)['c_minus_nc_abs_max']>1e-4
    bad=synthetic_raw(9);bad['nll']['A'][0][0]=float('nan');assert build_record(bad)['finite'] is False
    print('e57_1280_runner_records selftest ok')
if __name__=='__main__':
    import sys
    if '--selftest' in sys.argv:selftest()
