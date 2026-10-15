"""Equivalence of packed probe scoring with one-probe-per-chunk scoring (run BEFORE any packed result is used).
Three comparisons on identical weights (base, and native seed-101 states after chunks 14/33/62 for worlds A and B):
  S  semantic: packed (many probes per chunk) vs packed-code-path with ONE probe per chunk -> same kernels, so any
     difference beyond float noise means leakage between probes or wrong positions.   PASS: max |row logp diff| <= 1e-3
  L  leakage: two packings with different neighbour order.                             PASS: max |row logp diff| <= 1e-3
  K  kernel/path: packed vs the run3 single-row path (cuDNN prefix vs XLA masked prefix) -> numerical only;
     judged at the result level: per-item margin and binding D = mean_p(M_L3 - M_XP).   PASS: |dD| <= 0.005 at every state
All numbers go to verify_packing.json; nothing is interpreted if S or L fails."""
import sys,json,time,random,os
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
from readouts import *
import packing
OUT=Path(os.environ.get('EXP05_OUT',ROOT/'experiments/exp05_write_dynamics/run3_readouts'))
FP32=os.environ.get('VERIFY_FP32')=='1'
KS=[int(x) for x in os.environ.get('VERIFY_KS','14,33,62').split(',') if x]
def fp32_mode():
    """numerics-free semantic check: fp32 compute, full-precision matmuls, cuDNN attention redirected to XLA."""
    jax.config.update('jax_default_matmul_precision','highest')
    orig=reuse.config3b.compose_3b_128k_config
    def comp():
        cfg=orig();cfg.model.compute_dtype='fp32';return cfg
    reuse.config3b.compose_3b_128k_config=comp
    o=jax.nn.dot_product_attention
    def dpa(*a,**k):
        if k.get('implementation')=='cudnn':k=dict(k,implementation='xla')
        return o(*a,**k)
    jax.nn.dot_product_attention=dpa;packing._att.jax.nn.dot_product_attention=dpa
def main():
    assert 'deterministic_ops=true' in os.environ.get('XLA_FLAGS','')
    if FP32:fp32_mode()
    rt=Runtime();T=Readouts(rt,101);raw=[]
    for it in T.items:
        for c in (it['gold'],it['foil']):
            ids,span=stage2.item_ids(T.tok,stage2.make_slot(T.tok,None),it['prefix'],c);raw.append((ids,(span[0]-1,span[1]-1)))
    assert [r[1] for r in raw]==T.spans
    t0=time.monotonic();P=packing.Packed(T,raw);P1=packing.Packed(T,raw,one_per_chunk=True)
    perm=list(range(len(raw)));random.Random(3).shuffle(perm);Ps=packing.Packed(T,[raw[i] for i in perm])
    print('packed chunks',P.nchunk,'single chunks',P1.nchunk,'prefix precompute s',round(time.monotonic()-t0),flush=True)
    res={'n_rows':len(raw),'packed_chunks':P.nchunk,'states':{}}
    def margins(rowlp):return rowlp[0::2]-rowlp[1::2]
    def D(mA,mB):   # binding contrast from item margins of both worlds
        out={}
        for f in ('L3','XP'):
            i0=sum(len(T.fam[g]) for g in list(T.fam)[:list(T.fam).index(f)]);its=T.fam[f]
            pm={}
            for k,it in enumerate(its):pm.setdefault(it['pair'],[]).append((mA[i0+k]-mB[i0+k])/2)
            out[f]=np.asarray([np.mean(v) for _,v in sorted(pm.items(),key=lambda x:int(x[0][1:]))])
        return float(np.mean(out['L3']-out['XP']))
    def one_state(tag,m):
        r={}
        t=time.monotonic();pk0=P.r0(m);pk1=P.r1(m);tp=time.monotonic()-t
        s0=P1.r0(m);s1=P1.r1(m);sh1=Ps.r1(m);sh1=np.asarray([sh1[perm.index(i)] for i in range(len(raw))])
        t=time.monotonic();k0=T.probe_short_b(m);k1=T.probe_neutral(m);tk=time.monotonic()-t
        r['S_maxdiff_R0']=float(np.abs(pk0-s0).max());r['S_maxdiff_R1']=float(np.abs(pk1-s1).max());r['L_maxdiff_R1']=float(np.abs(pk1-sh1).max())
        r['K_item_margin_maxdiff_R0']=float(np.abs(margins(pk0)[:len(k0)]-k0).max());r['K_item_margin_maxdiff_R1']=float(np.abs(margins(pk1)-k1).max())
        r['K_item_margin_meanabs_R1']=float(np.abs(margins(pk1)-k1).mean());r['sec_packed']=tp;r['sec_single']=tk
        r['_m']={'pk1':margins(pk1).tolist(),'k1':k1.tolist(),'pk0':margins(pk0).tolist(),'k0':k0.tolist()}
        print(tag,{k:v for k,v in r.items() if k!='_m'},flush=True);return r
    res['states']['base']=one_state('base',rt.base)
    for w in ('AB' if KS else ''):
        P_=T.P[w=='B'];ids=P_.ids;mask=np.zeros(NCH*CH,dtype=bool);mask[:len(ids)-1]=True;seq=rt.seq(ids,mask=mask)
        with rt.mesh:
            pref=rt.prefix(rt.base,seq);m=rt.base;opt=m.inner_optimizer(clone_state(rt.state_all)).init(m.inner_parameters());st=(clone_state(rt.state_all),clone_state(rt.state_suffix))
            for k in range(max(KS)+1):
                r=rt.update(m,opt,st,rt.chunk(seq,k),pref[k*CH:(k+1)*CH]);m,opt,st=r.new_model,r.new_optimizer_state,r.new_state;del r
                if k in KS:res['states'][f'{w}{k}']=one_state(f'{w}{k}',m)
    dd={}
    for k in KS:
        a=res['states'][f'A{k}']['_m'];b=res['states'][f'B{k}']['_m']
        dd[k]={'D_packed_R1':D(a['pk1'],b['pk1']),'D_single_R1':D(a['k1'],b['k1']),'D_packed_R0':D(a['pk0'],b['pk0']),'D_single_R0':D(a['k0'],b['k0'])}
    res['D']=dd
    S=max(max(v['S_maxdiff_R0'],v['S_maxdiff_R1']) for v in res['states'].values());L=max(v['L_maxdiff_R1'] for v in res['states'].values())
    K=max([max(abs(v['D_packed_R1']-v['D_single_R1']),abs(v['D_packed_R0']-v['D_single_R0'])) for v in dd.values()] or [0.0])
    res['verdict']={'S_max':S,'S_pass':S<=1e-3,'L_max':L,'L_pass':L<=1e-3,'K_dD_max':K,'K_pass':K<=0.005}
    res['verdict']['PASS']=all(res['verdict'][x] for x in ('S_pass','L_pass','K_pass'))
    json.dump(res,open(OUT/('verify_packing_fp32.json' if FP32 else 'verify_packing.json'),'w'),indent=1);print(json.dumps(res['verdict'],indent=1),json.dumps(dd,indent=1))
if __name__=='__main__':main()
