"""E57 band-1024 + loss-weight arms on the CLIPPED-SGD engine (GPU; new file; job 1308 proposal v2, Hans 11/17 'sgd랑 k0=0'; wrapper of the unchanged e57_oracle_run.py, which is imported and NOT edited). Same write run as the band-1024 arm of jobs 1283 / 1293 (engine SGDSEL = clipped SGD: coef = min(1, 1/||g||), lr 1, update-path mask only; mask ALLQ1024 on EVERY row from chunk K0 = 0; reads at exposures 16 / 32 / 64 / 128 out-of-window and in-window at 128; --nat-loss-every 8), with ONE change: the loss gradient of the write step is the gradient of the WEIGHTED loss  sum_i valid_i (1 + (w - 1) dec_i) nll_i / n_valid  (exactly lm_loss for w = 1 up to float rounding), dec = the deciding rows of the chunk (published by the mask object of this file in the module state ST right before the write). The clip rule, the learning rate, the ilr multiplier and the mask are those of the original step; because the weights enlarge the gradient, the clip is active more often and the update is larger than the band-1024 update: gnorm, coef (clip coefficient) and update_norm of every chunk are logged by run_stream as for every arm (summary.clip_frac_coef_lt1, summary.mean_update_norm) and the rows carry n_weighted_rows.
ARMS (names parsed by this wrapper into mask codes; the saved json has mask_arm = the code, band = 1024):
  NATB1024G30W8   code B1024G   weight 8 on the top-30% rows by g1024 = nll_band1024 - nll_win (masks qb_TOP30G1024_<k>, chunks k >= 8; chunks < 8 have no deciding rows: weight 1)
  NATB1024R30W8   code B1024R   weight 8 on the same number of RANDOM eligible rows (masks qb_RAND30_<k>)
  NATB1024G30W8NM code B1024GN  as G, then the update of the chunk is rescaled to the update norm of the band-1024 clipped-SGD run of the same seed and chunk (--ref-dir/E55_bd1024_NATMASK_s<seed>.saved.json, log[k].update_norm): AUXILIARY (norm-matched) arm
  NATB1024R30W8NM code B1024RN  as R, norm matched: AUXILIARY control
  NATB1024W1T     code B1024W1T weights all 1 through the weighted code path (gate arm: must reproduce the band-1024 arm within float rounding)
Any other arm is passed to the unchanged e57_oracle_run.main (e.g. NATMASK with --mask-arm ALLQ1024 = the band-1024 arm itself, used by the bit-identity gate). Implementation: e57_qband_run.SelMaskQ is wrapped by a factory that returns the weighted mask object for the five codes, e57_oracle_run.with_base captures the Ladder object, and e57_oracle_run.swap_step is wrapped so that the write_step of an arm built from the weighted mask object is replaced by the weighted step (the step of every other arm is untouched).
  python3 e57_b1024s_run.py --seed S --cell v3_16_1 --out O --masks MASKS.npz --mask-arm ALLQ1024 --arms NATB1024G30W8,NATB1024R30W8 --save-apps 16,32,64,128 --nat-loss-every 8 --tag b1024s [--ref-dir D] | --selftest"""
import sys,pathlib,json
import numpy as np
HERE=pathlib.Path(__file__).resolve().parent;sys.path.insert(0,str(HERE))
import e57_oracle_run as OR
import e57_qband_run as QR
import e57_closure_masks as CM
CH=1024
ARM2CODE={'NATB1024G30W8':'B1024G','NATB1024R30W8':'B1024R','NATB1024G30W8NM':'B1024GN','NATB1024R30W8NM':'B1024RN','NATB1024W1T':'B1024W1T'}
WEIGHT={'B1024G':8.0,'B1024R':8.0,'B1024GN':8.0,'B1024RN':8.0,'B1024W1T':1.0}
ROWKEY={'B1024G':'qb_TOP30G1024','B1024R':'qb_RAND30','B1024GN':'qb_TOP30G1024','B1024RN':'qb_RAND30','B1024W1T':None}
ST={}
REF={'dir':None,'seed':None}
_orig_parse=OR.parse_arm2;_orig_sel=QR.SelMaskQ;_orig_swap=OR.swap_step;_orig_wb=OR.with_base
def parse_arm3(name,mask_arm,rand_arm='RANDQ256P50'):
    if name in ARM2CODE:return ARM2CODE[name],1.0
    return _orig_parse(name,mask_arm,rand_arm)
def row_weights(valid,dec,weight):
    """per-token loss weights: valid_i (1 + (w - 1) dec_i) / n_valid (numpy / jax arrays)"""
    return valid*(1.0+(weight-1.0)*dec)/max(float(valid.sum()),1e-10)
def match_norm(old,new,target):
    """rescale the update new - old (pytrees of arrays) to the global L2 norm target (fp32), cast back; target <= 0 or a zero update leave the step unchanged"""
    import jax,jax.numpy as jnp
    f32=jnp.float32;d=jax.tree.map(lambda n,o:n.astype(f32)-o.astype(f32),new,old);nrm=float(jnp.sqrt(sum(jnp.sum(x*x) for x in jax.tree.leaves(d))))
    if nrm<=0 or target<=0:return new
    sc=target/nrm;return jax.tree.map(lambda o,x,n:(o.astype(f32)+x*sc).astype(n.dtype),old,d,new)
class UVB1024S(_orig_sel):
    """mask object of the weighted arms: SelMaskQ with the band 1024 on every row from K0 (as ALLQ1024); publishes the deciding rows of the chunk in ST['dec'] for the weighted step"""
    def __init__(self,code,masks,log=print):
        super().__init__(code,masks,log);self.code=code;ST.clear();ST.update(fresh=True,weight=WEIGHT[code],nm=code.endswith('N'),dec=np.zeros(CH,np.float32),k=-1,ref=None)
        if ST['nm']:
            assert REF['dir'] and REF['seed'] is not None,'norm-matched arms need --ref-dir and --seed'
            ST['ref']=[e['update_norm'] for e in json.load(open(pathlib.Path(REF['dir'])/f'E55_bd1024_NATMASK_s{REF["seed"]}.saved.json'))['log']]
    def keep(self,k,m,Kc,Vc,ids_k,keep_all):
        kp,info=super().keep(k,m,Kc,Vc,ids_k,keep_all);key=ROWKEY[self.code];name=f'{key}_{k}' if key else None
        dec=CM.up(self.m[name],CH).astype(np.float32) if name and name in self.m else np.zeros(CH,np.float32);ST['dec']=dec;ST['k']=k
        info=dict(info);info['n_weighted_rows']=int(dec.sum());info['loss_weight']=ST['weight'];return kp,info
def sel_factory(arm,masks,log=print):
    return UVB1024S(arm,masks,log) if arm in WEIGHT else _orig_sel(arm,masks,log)
def build_weighted_step(L):
    """write step of the original rule (clip, lr, ilr multiplier of the Ladder object) with the weighted loss of the module state ST"""
    import jax,jax.numpy as jnp,equinox as eqx,optax,s3
    S3SUF=s3.S3SUF;lr,clip=L.lr,L.clip;lr_scale=float(getattr(L,'lr_scale',1.0));f32=jnp.float32
    @eqx.filter_jit
    def step_w(m,st_all,st_suf,c,pref_,kmask,w):
        def total(mm):
            S3SUF['kmask']=kmask
            try:out=mm.language_model.suffix_call(prefix_outputs=pref_,state=st_suf,seq=c)
            finally:S3SUF.clear()
            lp=jax.nn.log_softmax(out.logits.astype(f32),axis=-1);nll=-jnp.take_along_axis(lp,c.target_tokens[:,None],axis=-1)[:,0];return jnp.sum(w*nll),nll
        (_l,nll),grads=eqx.filter_value_and_grad(total,has_aux=True)(m);valid=c.loss_masks.astype(f32);ce=jnp.sum(valid*nll)/jnp.maximum(jnp.sum(valid),1e-10)
        g=grads.inner_parameters();gn=optax.global_norm(g);coef=jnp.where(gn<clip,1.0,clip/jnp.maximum(gn,1e-30));s=lr*m.get_ilr_multiplier(st_all.get(m.step_index))
        if lr_scale!=1.0:s=s*lr_scale
        newp=jax.tree.map(lambda p,x:(p+(-s*coef*x).astype(p.dtype)).astype(p.dtype),m.inner_parameters(),g);return newp,gn,coef,ce
    return step_w
def with_base_w(uv,hold,L,jnp,*a,**kw):
    ST['L']=L;return _orig_wb(uv,hold,L,jnp,*a,**kw)
def swap_step_w(orig,mult,mode,holder,every=8):
    if not ST.pop('fresh',False):return _orig_swap(orig,mult,mode,holder,every)
    import jax.numpy as jnp
    L=ST['L'];wstep=build_weighted_step(L);weight=ST['weight']
    def orig_w(m,st_all,st_suf,c,pref,kmask):
        valid=jnp.asarray(c.loss_masks,jnp.float32);dec=jnp.asarray(ST['dec'],jnp.float32);w=valid*(1.0+(weight-1.0)*dec)/jnp.maximum(valid.sum(),1e-10)
        newp,gn,coef,ce=wstep(m,st_all,st_suf,c,pref,kmask,w)
        if ST['nm'] and ST.get('ref') is not None and 0<=ST['k']<len(ST['ref']):newp=match_norm(m.inner_parameters(),newp,float(ST['ref'][ST['k']]))
        return newp,gn,coef,ce
    return _orig_swap(orig_w,mult,mode,holder,every)
def install():
    OR.parse_arm2=parse_arm3;QR.SelMaskQ=sel_factory;OR.swap_step=swap_step_w;OR.with_base=with_base_w
def selftest():
    import os
    os.environ.setdefault('JAX_PLATFORMS','cpu')
    import jax.numpy as jnp
    OR.selftest() if False else None
    assert parse_arm3('NATB1024G30W8','ALLQ1024')==('B1024G',1.0) and parse_arm3('NATB1024W1T','ALLQ1024')==('B1024W1T',1.0) and parse_arm3('NATMASK','ALLQ1024')==_orig_parse('NATMASK','ALLQ1024') and parse_arm3('NATS','ALLQ1024')==_orig_parse('NATS','ALLQ1024')
    # weights: weight 1 gives valid / n_valid (the plain mean), weight 8 on the deciding rows
    rng=np.random.default_rng(0);valid=(rng.random(CH)<.8).astype(np.float32);dec=((rng.random(CH)<.3)&(valid>0)).astype(np.float32)
    w1=row_weights(valid,dec,1.0);assert np.allclose(w1,valid/valid.sum()) and abs(w1.sum()-1)<1e-5
    w8=row_weights(valid,dec,8.0);assert np.allclose(w8[dec>0],8*w1[dec>0]) and np.allclose(w8[(dec==0)&(valid>0)],w1[(dec==0)&(valid>0)]) and (w8[valid==0]==0).all()
    # mask object: banded rows = ALLQ1024 (every row), dec rows published, K0 from the module of e57_qband_run
    masks={'band_B1024G':np.asarray(1024),'all_B1024G':np.asarray(1),'band_B1024R':np.asarray(1024),'all_B1024R':np.asarray(1),'band_B1024W1T':np.asarray(1024),'all_B1024W1T':np.asarray(1),'band_ALLQ1024':np.asarray(1024),'all_ALLQ1024':np.asarray(1),'qb_TOP30G1024_9':CM.pk(dec>0),'qb_RAND30_9':CM.pk(np.roll(dec>0,7))}
    nat=np.ones(QR.WIN+CH,bool);QR.K0=0;lg=lambda *x:None
    ref,_=_orig_sel('ALLQ1024',masks,lg).keep(9,None,None,None,None,nat)
    for code,exp in (('B1024G',dec>0),('B1024R',np.roll(dec>0,7)),('B1024W1T',np.zeros(CH,bool))):
        kp,info=UVB1024S(code,masks,lg).keep(9,None,None,None,None,nat);assert np.array_equal(kp,ref) and np.array_equal(ST['dec'],exp.astype(np.float32)) and info['n_weighted_rows']==int(exp.sum()) and info['loss_weight']==WEIGHT[code] and ST['fresh'] and ST['k']==9
    assert CM.band_keep(nat,1024).sum()==ref.sum()
    QR.K0=8;kn,_=UVB1024S('B1024G',masks,lg).keep(3,None,None,None,None,nat);assert kn.ndim==1;QR.K0=0                                     # chunks < K0: natural keep
    kq,_=UVB1024S('B1024G',masks,lg).keep(10,None,None,None,None,nat);assert (ST['dec']==0).all()                                       # chunks without a deciding-row key: weight 1
    ST.clear();assert sel_factory('ALLQ1024',masks,lg).__class__ is _orig_sel and sel_factory('B1024G',masks,lg).__class__ is UVB1024S
    # norm matching: rescales the update to the target norm, direction unchanged, dtype kept
    old={'a':jnp.zeros((3,),jnp.float32),'b':jnp.ones((2,2),jnp.bfloat16)};new={'a':old['a']+jnp.asarray([3.0,0,0]),'b':old['b']+jnp.asarray([[4.,0],[0,0]],jnp.bfloat16)}
    mm=match_norm(old,new,10.0);d=[(mm[k].astype(jnp.float32)-old[k].astype(jnp.float32)) for k in old];n=float(jnp.sqrt(sum(jnp.sum(x*x) for x in d)));assert abs(n-10.0)<0.1 and mm['b'].dtype==jnp.bfloat16 and abs(float(d[0][0])/float(d[1][0,0])-3/4)<0.02
    assert match_norm(old,old,5.0) is old
    # patched names exist in the unchanged runner
    import inspect;src=inspect.getsource(OR.main);assert 'QR.SelMaskQ(ma,masks,lg)' in src and 'swap_step(L.write_step' in src and 'with_base(uv,hold' in src and 'parse_arm2(' in src
    install();assert OR.parse_arm2 is parse_arm3 and QR.SelMaskQ is sel_factory and OR.swap_step is swap_step_w and OR.with_base is with_base_w
    OR.parse_arm2=_orig_parse;QR.SelMaskQ=_orig_sel;OR.swap_step=_orig_swap;OR.with_base=_orig_wb
    print('e57_b1024s_run selftest ok')
def main():
    if '--selftest' in sys.argv:selftest();return
    argv=sys.argv;
    for key in ('--ref-dir',):
        if key in argv:i=argv.index(key);REF['dir']=argv[i+1];del argv[i:i+2]
    seed=None
    for i,x in enumerate(argv):
        if x=='--seed':seed=int(argv[i+1])
    REF['seed']=seed;install();OR.main()
if __name__=='__main__':main()
