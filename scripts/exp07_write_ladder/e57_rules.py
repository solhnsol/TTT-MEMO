"""Deciding-row selection rules for the M3 weighted set (job 1070; Hans 11/05 approved candidates ABS, GAINMASS, GATE; g unified to band 128).
All rules are label-free and take per-row arrays of ONE chunk: g = nll_band128 - nll_win, nwin = nll_win, el = eligible bool. They return a float32 weight FRACTION dec in {0,1} (the loss weight is 1 + (w-1) dec, as in install_weighted_grad).
  abs_rule(g, el, tau)            ABS-NAT: dec = 1 where g > tau (nats); share not forced.
  gainmass_rule(g, el, rho)       GAIN-MASS: smallest set S of rows, taken in order of falling g, with sum_S g >= rho * sum_el max(g, 0); share depends on how concentrated g is.
  gate_rule(g, nwin, el, tau, eta) RESOLVED-GATE: dec = 1 where g > tau and nwin < eta (rows that stay hard even with the window are excluded).
A fixed-share rule (top_share) is kept only as the comparison arm. Selftest: python3 e57_rules.py --selftest (CPU, numpy only)."""
import numpy as np,sys
def abs_rule(g,el,tau):
    return (np.asarray(el,bool)&(np.asarray(g,np.float64)>tau)).astype(np.float32)
def gainmass_rule(g,el,rho):
    g=np.asarray(g,np.float64);el=np.asarray(el,bool);d=np.zeros(g.shape,np.float32);idx=np.where(el&(g>0))[0]
    if idx.size==0:return d
    o=idx[np.argsort(-g[idx],kind='stable')];c=np.cumsum(g[o]);tot=c[-1];n=int(np.searchsorted(c,rho*tot-1e-12)+1);d[o[:min(n,o.size)]]=1.0;return d
def gate_rule(g,nwin,el,tau,eta):
    return (np.asarray(el,bool)&(np.asarray(g,np.float64)>tau)&(np.asarray(nwin,np.float64)<eta)).astype(np.float32)
def top_share(g,el,share):
    g=np.asarray(g,np.float64);el=np.asarray(el,bool);d=np.zeros(g.shape,np.float32);idx=np.where(el)[0];n=int(round(share*idx.size))
    if n>0:d[idx[np.argsort(-g[idx],kind='stable')[:n]]]=1.0
    return d
def selftest():
    r=np.random.default_rng(1);n=1024;el=r.random(n)<.9;g=r.standard_normal(n)*2+.8;nw=np.abs(r.standard_normal(n))*2
    a=abs_rule(g,el,3.0);assert a[~el].sum()==0 and (a[el]==(g[el]>3.0)).all()
    assert abs_rule(g,el,1e9).sum()==0 and abs_rule(g,el,-1e9).sum()==el.sum()
    for rho in (.3,.5,.85):
        d=gainmass_rule(g,el,rho);pos=np.where(el&(g>0))[0];sel=d>0;assert d[~el].sum()==0 and (g[sel]>0).all()
        assert g[sel].sum()>=rho*g[pos].sum()-1e-6;m=g[sel].min();assert not ((g>m)&el&~sel).any()      # prefix of the falling-g order
        k=sel.sum()
        if k>1:assert g[sel].sum()-m<rho*g[pos].sum()                                                    # minimal
    assert gainmass_rule(g,el,.5).sum()<=gainmass_rule(g,el,.85).sum()
    gr=gate_rule(g,nw,el,2.0,1.0);assert (gr<=abs_rule(g,el,2.0)).all() and (gr[nw>=1.0]==0).all()
    ts=top_share(g,el,.1);assert ts.sum()==round(.1*el.sum())
    conc=np.zeros(n);conc[:20]=10;conc[20:]=0.01;e2=np.ones(n,bool);dc=gainmass_rule(conc,e2,.5);dd=gainmass_rule(np.ones(n),e2,.5)
    assert dc.sum()==11 and dd.sum()==512                                                                # concentration-adaptive share
    print('e57_rules selftest ok')
if __name__=='__main__':
    selftest() if '--selftest' in sys.argv else print(__doc__)
