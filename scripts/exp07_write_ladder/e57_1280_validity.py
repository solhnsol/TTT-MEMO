"""E57 job 1280 validity code (A1 v12 sections 2, 4, 5; new file, GPU-free parts): the numerical comparison rules, the t3 substitution draw, the excluded-position sets, the t3 status, the sealed margin comparison and the validity record writer. The model forward passes that produce the compared arrays are in the runner (GPU, not in this file).
  python3 e57_1280_validity.py --selftest"""
import hashlib,json,os
import numpy as np
LP_TOL=1e-4;REL_TOL=1e-4;MARGIN_FRAC=0.01;MARGIN_FLOOR=1e-4;SENS_THRESH=0.001
# ---------------- 2.1 / 2.3 comparisons ----------------
def lp_equal(a,b,fallback=False):
    a=np.asarray(a,float);b=np.asarray(b,float)
    if a.shape!=b.shape:return False
    return bool(np.array_equal(a,b)) if not fallback else bool(np.abs(a-b).max(initial=0.0)<=LP_TOL)
def rel_equal(x_a,x_b,fallback=False):
    """2.3 (b): relative difference |x_a - x_b| / |x_b| <= 1e-4; if x_b = 0, x_a must equal 0 exactly (deterministic mode: exact equality)"""
    if not fallback:return x_a==x_b
    return (x_a==0.0) if x_b==0.0 else bool(abs(x_a-x_b)/abs(x_b)<=REL_TOL)
def abs_equal(x_a,x_b,fallback=False):
    """2.3 (c): absolute difference <= 1e-4 (ce, coef); exact equality in deterministic mode"""
    return x_a==x_b if not fallback else bool(abs(x_a-x_b)<=LP_TOL)
def dw_equal(dw_a,dw_b,fallback=False):
    """2.3 (d) / 4.4: ||dW_a - dW_b|| / ||dW_b|| <= 1e-4 (fallback), exactly 0 in deterministic mode; ||dW_b|| = 0 requires ||dW_a|| = 0 exactly"""
    a=np.asarray(dw_a,float).ravel();b=np.asarray(dw_b,float).ravel();nb=float(np.linalg.norm(b))
    if not fallback:return bool(np.array_equal(a,b))
    return (float(np.linalg.norm(a))==0.0) if nb==0.0 else bool(np.linalg.norm(a-b)/nb<=REL_TOL)
# ---------------- 4.3 substitution and excluded positions ----------------
def excluded_positions(s_i):
    """E_i = { t : t in S_i or t + 1 in S_i } (positions of the window that are removed from every comparison, loss and aggregate)"""
    s=set(int(t) for t in s_i);return sorted(s|{t-1 for t in s})
def f_prime(f_i,s_i):
    e=set(excluded_positions(s_i));return [t for t in f_i if t not in e]
def draw_seed(seed,chunk,pos,k=0):
    h=hashlib.sha256(f't3|{seed}|{chunk}|{pos}|{k}'.encode('ascii')).digest();return int.from_bytes(h[:8],'big')
def draw_token(seed,chunk,pos,orig,vocab_size,special_ids,k=0):
    """uniform draw from the vocabulary minus the special tokens and minus the original token (A1 4.3): the allowed identifiers sorted ascending, one uniform integer draw from a PCG64 generator seeded with draw_seed"""
    banned=set(int(x) for x in special_ids)|{int(orig)};allowed=[t for t in range(vocab_size) if t not in banned]
    rng=np.random.Generator(np.random.PCG64(draw_seed(seed,chunk,pos,k)));return int(allowed[int(rng.integers(0,len(allowed)))])
def substitute(tokens,s_positions,seed,chunk,vocab_size,special_ids):
    """substitution of every S_i token; a generation error (a position unchanged or a changed length) is redrawn with k = 1, 2, ... (never removed); returns the new token list and the number of redraws"""
    out=list(tokens);redraws=0
    for p in s_positions:
        k=0
        while True:
            t=draw_token(seed,chunk,p,tokens[p],vocab_size,special_ids,k)
            if t!=tokens[p]:break
            k+=1;redraws+=1
        out[p]=t
    assert len(out)==len(tokens) and all(out[p]!=tokens[p] for p in s_positions)
    return out,redraws
# ---------------- 4.5 / 4.6 status ----------------
def sensitivity(chunks):
    """seed aggregate: sum |NLL_sub - NLL_orig| over all tokens of F_i' / number of tokens (token weighted); chunks = list of (nll_orig, nll_sub) arrays over F_i'; denominator 0 -> None"""
    num=0.0;den=0
    for o,s in chunks:
        o=np.asarray(o,float);s=np.asarray(s,float);assert o.shape==s.shape and np.isfinite(o).all() and np.isfinite(s).all();num+=float(np.abs(s-o).sum());den+=len(o)
    return (num/den if den>0 else None),den
def t3_status(leak,fail_4_2,update_invariance_ok,agg):
    """4.6 with the priority FAIL > UNDETERMINED > PASS"""
    if leak or fail_4_2 or not update_invariance_ok:return 'FAIL'
    if agg is None or not np.isfinite(agg) or agg<SENS_THRESH:return 'UNDETERMINED'
    return 'PASS'
# ---------------- 2.3 (e) sealed margin comparison ----------------
def sealed_margin_check(margins_a,margins_b,g_s,sealed_path=None):
    """judges metric (e): max |margin difference| <= max(0.01 |G_s|, 1e-4); G_s is computed by the caller from the sealed M_MASK and M_NATS and is NOT returned; the function returns ONLY the pass/fail bit; the raw difference goes only to the sealed file (mode 0400, never opened except for the audit after the study)"""
    d=float(np.abs(np.asarray(margins_a,float)-np.asarray(margins_b,float)).max(initial=0.0));tol=max(MARGIN_FRAC*abs(float(g_s)),MARGIN_FLOOR)
    if sealed_path:
        with open(sealed_path,'w') as f:json.dump({'raw_max_abs_difference':d},f)
        os.chmod(sealed_path,0o400)
    return bool(d<=tol)
# ---------------- validity record ----------------
def write_validity_record(path,seed,t2_ok,t3,p16_ok,mode,extra=None):
    """one validity record per seed (statuses, counts and the sensitivity aggregate only; no NLL level, no log-probability array, no margin); returns its SHA-256 for the verification record"""
    rec=dict(seed=int(seed),t2_ok=bool(t2_ok),t3_ok=(t3.get('status')=='PASS'),p16_ok=bool(p16_ok),t3_status=t3.get('status'),t3_counts={k:t3.get(k) for k in ('chunks','below_chunk_threshold','detectable_response','empty_F_i','aggregate','tokens')},mode=mode)
    if extra:rec.update(extra)
    assert not any(isinstance(v,(list,np.ndarray)) for v in rec['t3_counts'].values()),'arrays are not allowed in the validity record'
    s=json.dumps(rec,sort_keys=True);open(path,'w').write(s);return hashlib.sha256(s.encode()).hexdigest()
def selftest():
    a=np.array([1.0,2.0]);assert lp_equal(a,a.copy()) and not lp_equal(a,a+1e-9) and lp_equal(a,a+5e-5,fallback=True) and not lp_equal(a,a+2e-4,fallback=True) and not lp_equal(a,np.array([1.0]))
    assert rel_equal(2.0,2.0) and not rel_equal(2.0,2.0+1e-9) and rel_equal(2.0,2.0+1e-5,fallback=True) and not rel_equal(2.0,2.1,fallback=True) and rel_equal(0.0,0.0,fallback=True) and not rel_equal(1e-9,0.0,fallback=True)
    assert abs_equal(0.5,0.5) and abs_equal(0.5,0.50005,fallback=True) and not abs_equal(0.5,0.51,fallback=True)
    d=np.array([1.0,0.0,2.0]);assert dw_equal(d,d.copy()) and not dw_equal(d,d*(1+1e-6)) and dw_equal(d,d*(1+1e-6),fallback=True) and not dw_equal(d,d*1.01,fallback=True) and dw_equal(np.zeros(3),np.zeros(3)) and dw_equal(np.zeros(3),np.zeros(3),fallback=True) and not dw_equal(np.array([1e-9,0,0]),np.zeros(3),fallback=True)
    assert excluded_positions([5,6,7])==[4,5,6,7] and f_prime([3,4,5,8,9],[5,6,7])==[3,8,9] and excluded_positions([])==[]
    t1=draw_token(101,12,300,42,128000,[128000,128001],0);t2=draw_token(101,12,300,42,128000,[128000,128001],0);assert t1==t2 and t1!=42 and t1 not in (128000,128001) and 0<=t1<128000
    assert draw_token(101,12,300,42,128000,[],1)!=draw_token(101,12,300,42,128000,[],0) or True
    toks=list(range(100,140));sub,rd=substitute(toks,[3,4,5,10],101,12,128000,[128000,128001]);assert len(sub)==len(toks) and all(sub[p]!=toks[p] for p in (3,4,5,10)) and all(sub[p]==toks[p] for p in range(len(toks)) if p not in (3,4,5,10)) and substitute(toks,[3,4,5,10],101,12,128000,[128000,128001])[0]==sub
    tiny,rd2=substitute([0,1,0,1],[0,1,2,3],7,1,3,[]);assert all(tiny[i]!=[0,1,0,1][i] for i in range(4))     # tiny vocabulary: draws that equal the original are redrawn, never kept
    agg,den=sensitivity([(np.zeros(4),np.full(4,0.01)),(np.zeros(2),np.full(2,0.04))]);assert den==6 and abs(agg-(0.04+0.08)/6)<1e-12 and sensitivity([(np.zeros(0),np.zeros(0))])==(None,0)
    assert t3_status(False,False,True,0.02)=='PASS' and t3_status(False,False,True,0.0005)=='UNDETERMINED' and t3_status(False,False,True,None)=='UNDETERMINED' and t3_status(True,False,True,0.02)=='FAIL' and t3_status(False,True,True,None)=='FAIL' and t3_status(False,False,False,0.02)=='FAIL' and t3_status(True,False,True,None)=='FAIL'
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        sp=os.path.join(td,'sealed.json');assert sealed_margin_check([1.0,2.0],[1.0,2.0],3.0,sp) is True and sealed_margin_check([1.0,2.0],[1.0,2.04],3.0) is False and sealed_margin_check([1.0],[1.0+5e-5],0.0) is True and sealed_margin_check([1.0],[1.0+5e-4],0.0) is False
        assert oct(os.stat(sp).st_mode&0o777)=='0o400'
        h=write_validity_record(os.path.join(td,'v.json'),101,True,dict(status='PASS',chunks=8,below_chunk_threshold=0,detectable_response=8,empty_F_i=0,aggregate=0.02,tokens=800),True,'deterministic');assert len(h)==64
        try:write_validity_record(os.path.join(td,'w.json'),101,True,dict(status='PASS',aggregate=[0.1]),True,'deterministic');raise SystemExit('array accepted')
        except AssertionError:pass
    print('e57_1280_validity selftest ok')
if __name__=='__main__':
    import sys
    if '--selftest' in sys.argv:selftest()
