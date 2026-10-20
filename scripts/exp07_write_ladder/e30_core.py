"""E30 (fact-free context dose-response) pure helpers: cell table, keep vectors, context lengths, weights. No jax, no tokenizer. Selftest: python3 e30_core.py --selftest
Cells (E30 2.0):
  A      native: the real stream context (natural chain), ntp loss.
  M      chunk alone, positions kept (prefix-all-pos reference of E29/E25: every window key removed, chunk at its own stream position).
  F<L>   fact-free context of 8192 tokens in the carried-KV slots (prefix keys AND suffix keys), only the nearest L slots visible, chunk at position 8192 (slot positions fixed). L in 0,128,512,2048,8192.
  P<L>   position-shift control: the fact-free context is only ceil(L/1024)*1024 real tokens before the chunk (chunk position = that length), the nearest L of them visible. (L multiple of 1024: exactly L tokens.)
  R      F0 with the first 128 chunk tokens excluded from the loss (ramp control).
  Na/Nb  F8192 with another kind of context text: Na = a second independent draw of background-style text, Nb = raw natural book text."""
import sys
import numpy as np
CH=1024;WIN=8192
CELLS={
 'A':dict(kind='A'),'M':dict(kind='M'),
 'F0':dict(kind='F',L=0,src='bg'),'F128':dict(kind='F',L=128,src='bg'),'F512':dict(kind='F',L=512,src='bg'),'F2048':dict(kind='F',L=2048,src='bg'),'F8192':dict(kind='F',L=8192,src='bg'),
 'P512':dict(kind='P',L=512,src='bg'),'P8192':dict(kind='P',L=8192,src='bg'),
 'R':dict(kind='F',L=0,src='bg',skip=128),
 'Na':dict(kind='F',L=8192,src='bg2'),'Nb':dict(kind='F',L=8192,src='raw'),
}
CORE=['A','M','F0','F8192']
ALL=['A','M','F0','F128','F512','F2048','F8192','P512','P8192','R','Na','Nb']
def ctx_len(cell):
    """number of real fact-free tokens placed before the chunk"""
    c=CELLS[cell]
    if c['kind']=='F':return WIN
    if c['kind']=='P':return -(-c['L']//CH)*CH
    return 0
def keep_vector(cell,win=WIN,ch=CH):
    """visibility of the win carried slots + the chunk's own keys (the layout of s3.prefix_masked / S3SUF['kmask']); window slot j is the key at position q0 - win + j."""
    c=CELLS[cell];L={'F':c.get('L',0),'P':c.get('L',0),'M':0}.get(c['kind'],win)
    k=np.zeros(win+ch,bool);k[win-L:]=True;return k
def prep_key(cell):
    """cells with the same key share one context preparation (K,V of the context + suffix state after reading it)"""
    c=CELLS[cell];return None if c['kind'] in ('A','M') else (c['src'],ctx_len(cell))
def loss_skip(cell):return CELLS[cell].get('skip',0)
def weights(counts):
    n=float(sum(counts));return [c/n for c in counts]
def parse_cells(s):
    names=[x for x in s.split(',') if x] if s not in ('core','all') else (CORE if s=='core' else ALL)
    bad=[x for x in names if x not in CELLS];assert not bad,f'unknown cells {bad}';return names
def verdict(rows,old=None):
    """selfcheck verdict from the measured rows (pure Python, CPU testable). Keys starting with 'gate' are the ones that decide (E30 design 10/20, 2nd revision); 'info_*' and 'warn_*' are descriptive.
    gate_realctx_vs_windowref: cos >= 0.999 and |CE(context path) - CE(window reference)| < 0.01 on every row (rel_diff > 0.05 is only a warning); gate_F0_vs_M_cos_ge_0.999 (F0 vs M, when measured).
    E30_GATES=old: the original gates decide (realctx vs native cos >= 0.99 and CE < 0.05, F0 vs M rel <= 1e-3)."""
    import os
    old=(os.environ.get('E30_GATES','new')=='old') if old is None else old;r={}
    r['info_realctx_vs_native_cos_ge_0.99']=all(x['realctx_vs_native']['cos']>=0.99 and abs(x['ce_realctx_pipeline']-x['ce_native'])<0.05 for x in rows)
    r['gate_realctx_vs_windowref']=all(x['realctx_vs_windowref']['cos']>=0.999 and abs(x['ce_realctx_pipeline']-x['ce_windowref'])<0.01 for x in rows)
    r['warn_windowref_rel_gt_0.05']=[x['k'] for x in rows if x['realctx_vs_windowref']['rel_diff']>0.05]
    have=all('F0_vs_M' in x for x in rows)
    if have:
        r['info_F0_vs_M_rel_diff_le_1e-3']=all(x['F0_vs_M']['rel_diff']<=1e-3 for x in rows);r['gate_F0_vs_M_cos_ge_0.999']=all(x['F0_vs_M']['cos']>=0.999 for x in rows)
    if old:ok=r['info_realctx_vs_native_cos_ge_0.99'] and r.get('info_F0_vs_M_rel_diff_le_1e-3',True)
    else:ok=r['gate_realctx_vs_windowref'] and r.get('gate_F0_vs_M_cos_ge_0.999',True)
    r['gates_deciding']=['info_realctx_vs_native_cos_ge_0.99','info_F0_vs_M_rel_diff_le_1e-3'] if old else ['gate_realctx_vs_windowref']+(['gate_F0_vs_M_cos_ge_0.999'] if have else []);r['rc']=0 if ok else 4;return r
def selftest():
    row=lambda k,cw,ce,rel,cf=0.9999,rf=0.014:{'k':k,'ce_native':2.687,'ce_realctx_pipeline':2.6887,'ce_windowref':ce,'realctx_vs_native':{'cos':0.983},'realctx_vs_windowref':{'cos':cw,'rel_diff':rel},'F0_vs_M':{'cos':cf,'rel_diff':rf}}
    v=verdict([row(8,0.99942,2.6871,0.034),row(17,0.99989,2.6871,0.0147)],old=False);assert v['rc']==0 and v['gate_realctx_vs_windowref'] and v['gate_F0_vs_M_cos_ge_0.999'] and v['info_F0_vs_M_rel_diff_le_1e-3'] is False and v['info_realctx_vs_native_cos_ge_0.99'] is False   # job 60 numbers pass
    v=verdict([row(8,0.99942,2.6871,0.06)],old=False);assert v['rc']==0 and v['warn_windowref_rel_gt_0.05']==[8]
    assert verdict([row(8,0.998,2.6871,0.03)],old=False)['rc']==4 and verdict([row(8,0.9995,2.70,0.03)],old=False)['rc']==4 and verdict([row(8,0.9995,2.6871,0.03,cf=0.99)],old=False)['rc']==4
    assert verdict([row(8,0.99942,2.6871,0.034)],old=True)['rc']==4
    print('e30_core verdict SELFTEST OK',{k:x for k,x in verdict([row(8,0.99942,2.6871,0.034)],old=False).items() if k!='rc'})
    for n in ALL:
        k=keep_vector(n);assert k.shape==(WIN+CH,) and k[WIN:].all()
    assert keep_vector('F0')[:WIN].sum()==0 and keep_vector('M')[:WIN].sum()==0
    assert keep_vector('F8192')[:WIN].all() and keep_vector('F512')[:WIN].sum()==512 and keep_vector('F512')[WIN-512:WIN].all() and not keep_vector('F512')[WIN-513]
    assert keep_vector('P512')[:WIN].sum()==512 and ctx_len('P512')==1024 and ctx_len('P8192')==8192 and ctx_len('F128')==8192 and ctx_len('A')==0
    assert prep_key('F0')==prep_key('F8192')==prep_key('F128')==('bg',8192) and prep_key('Na')==('bg2',8192) and prep_key('P512')==('bg',1024) and prep_key('A') is None and prep_key('R')==('bg',8192)
    assert loss_skip('R')==128 and loss_skip('F0')==0 and abs(sum(weights([1,2,3]))-1)<1e-12 and parse_cells('core')==CORE
    print('e30_core SELFTEST OK')
if __name__=='__main__' and '--selftest' in sys.argv:selftest()
