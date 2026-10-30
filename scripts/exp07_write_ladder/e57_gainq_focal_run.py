"""E57 GAINQ512 p=.10 write arms with a focal loss on the SELECTED (masked) query rows (job 565; GPU, one news cycle; post-hoc, exploratory; new file, wraps e57_qcurve_run (arms/reads) and e57_qband_run.SelMaskQ). Same stream, SELC1 engine, Adam direction, update norm 0.5 x NAT of the chunk, same selected rows and masked forward as the GAINQ512 arm of job 293 (mask file e57_gainq_p0.10/e57_qband_masks.npz, GAINQ512 rows); the only change is the loss that the write gradient is taken of: for the chunks >= 8, among the selected rows the loss tokens whose masked-context per-token CE (from the same masked forward, weights before the write) is below tau are dropped (weight 0); the loss is the mean over the remaining loss rows (all non-selected rows stay); the update is then rescaled to the same norm as the GAINQ512 arm (the 0.5 x NAT norm_targets, applied by e55_core). Implementation: the per-token CE of the masked forward is computed by an extra forward pass at the same weights with the same per-query keep; the gradient is then taken with loss_masks = loss rows minus the dropped ones (the gradient of the mean CE over the remaining rows).
  GAINQ_FOCAL_NONE   no row dropped (must reproduce GAINQ512 p=.10 of job 293: state hash at chunk 279 2a6836fcba5f...)
  GAINQ_FOCAL_T05    drop selected-row tokens with CE < 0.5 nats
  GAINQ_FOCAL_T10    drop selected-row tokens with CE < 1.0 nats
  GAINQ_FOCAL_RAND   control: per chunk drop the same NUMBER of selected-row tokens as GAINQ_FOCAL_T05 dropped in that chunk (counts read from E57_focal_GAINQ_FOCAL_T05_drop.json of the same output folder), chosen at random (numpy default_rng(1000 + k))
Per chunk the number of selected-row loss tokens and the number / fraction dropped are written to <out>/E57_focal_<ARM>_drop.json.
  python3 e57_gainq_focal_run.py --qband-masks M.npz --ntg-from E57_main_NAT.arms.json <e57_qcurve_run.py arguments, --arms GAINQ_FOCAL_NONE,GAINQ_FOCAL_T05,GAINQ_FOCAL_T10,GAINQ_FOCAL_RAND>   |   --selftest"""
import sys,json,pathlib
import numpy as np
HERE=pathlib.Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
import e57_qband_run as QR
import e57_qcurve_run as QC
import e57_closure_masks as CM
CH=1024;K0=8;BASE='GAINQ512'
FOC={'GAINQ_FOCAL_NONE':('none',None),'GAINQ_FOCAL_T05':('tau',0.5),'GAINQ_FOCAL_T10':('tau',1.0),'GAINQ_FOCAL_RAND':('rand',None)}
FS={'uv':None,'out':None,'recs':[],'arm':None}
_SelMaskQ=QR.SelMaskQ
class FocalSel(_SelMaskQ):
    """SelMaskQ of the GAINQ512 rows of the mask file, plus the selected rows of the current chunk (self.qb) for the loss"""
    def __init__(self,arm,masks,log=print):
        self.focal=arm;self.mode,self.tau=FOC[arm] if arm in FOC else ('plain',None);super().__init__(BASE if arm in FOC else arm,masks,log);self.qb=None;self.k=None
    def keep(self,k,m,Kc,Vc,ids_k,keep_all):
        kp,info=super().keep(k,m,Kc,Vc,ids_k,keep_all);self.k=k
        self.qb=None if (self.none or k<K0 or self.allq) else CM.up(self.m[f'qb_{self.arm}_{k}'],CH)
        return kp,info
def drop_set(mode,tau,nll,sel,k,counts):
    """-> boolean [CH]: the selected-row tokens to drop"""
    if mode=='tau':return sel&(nll<tau)
    if mode=='rand':
        n=min(int(counts.get(k,0)),int(sel.sum()));d=np.zeros_like(sel)
        if n>0:idx=np.flatnonzero(sel);d[np.random.default_rng(1000+k).choice(idx,size=n,replace=False)]=True
        return d
    return np.zeros_like(sel)
def install(C,s3,out,lg):
    import jax,jax.numpy as jnp,equinox as eqx
    from ttt.model.data import Batch
    S3SUF=s3.S3SUF;f32=jnp.float32
    @eqx.filter_jit
    def ce_tok(m,st,c,pref,kmask):
        S3SUF['kmask']=kmask
        try:
            o=m.language_model.suffix_call(prefix_outputs=pref,state=st,seq=c);lp=jax.nn.log_softmax(o.logits.astype(f32),axis=-1);return -jnp.take_along_axis(lp,c.target_tokens[:,None],axis=-1)[:,0]
        finally:S3SUF.clear()
    orig_init=C.Ladder55.__init__;orig_run=C.Ladder55.run_stream;state={'counts':{}}
    def init(self,*aa,**kk):
        orig_init(self,*aa,**kk);og=self.grad_mask
        def gm(m,st,c,pref,kmask):
            uv=FS['uv']
            if uv is None or getattr(uv,'mode','plain') not in ('tau','rand','none') or uv.qb is None:return og(m,st,c,pref,kmask)
            lm=np.asarray(c.loss_masks).astype(bool);sel=uv.qb&lm
            nll=np.asarray(ce_tok(m,st,c,pref,kmask)) if uv.mode=='tau' else np.zeros(CH,np.float32)
            d=drop_set(uv.mode,uv.tau,nll,sel,uv.k,state['counts']);FS['recs'].append(dict(k=int(uv.k),n_loss_tokens=int(lm.sum()),n_selected=int(sel.sum()),n_dropped=int(d.sum()),frac_dropped=float(d.sum()/max(sel.sum(),1)),mean_nll_selected=float(nll[sel].mean()) if uv.mode=='tau' and sel.any() else None))
            if not d.any():return og(m,st,c,pref,kmask)
            c2=Batch(input_ids=c.input_ids,target_tokens=c.target_tokens,loss_masks=jnp.asarray(lm&~d));return og(m,st,c2,pref,kmask)
        self.grad_mask=gm
    def run(self,arm,*aa,**kk):
        uv=kk.get('uv');FS['uv']=uv;FS['recs']=[]
        if isinstance(uv,FocalSel) and uv.mode=='rand':
            fn=pathlib.Path(FS['out'])/'E57_focal_GAINQ_FOCAL_T05_drop.json';assert fn.exists(),'GAINQ_FOCAL_T05 must run before GAINQ_FOCAL_RAND (its drop counts are needed)'
            state['counts']={r['k']:r['n_dropped'] for r in json.load(open(fn))}
        r=orig_run(self,arm,*aa,**kk)
        if isinstance(uv,FocalSel):
            (pathlib.Path(FS['out'])/f'E57_focal_{uv.focal}_drop.json').write_text(json.dumps(FS['recs']));lg(f'  {uv.focal}: mean fraction of selected-row tokens dropped {np.mean([x["frac_dropped"] for x in FS["recs"]]) if FS["recs"] else 0:.3f} over {len(FS["recs"])} chunks')
        FS['uv']=None;return r
    C.Ladder55.__init__=init;C.Ladder55.run_stream=run
    return orig_init,orig_run
def focal_arms(a,out,items,bsrc,ids_all,n,N,nch,kks,rch,cch,arms,tk,tfile,LD,s3,C,rt,can,cw0,lg,SM):
    FS['out']=str(out);orig_init,orig_run=install(C,s3,out,lg)
    try:QC_orig(a,out,items,bsrc,ids_all,n,N,nch,kks,rch,cch,arms,tk,tfile,LD,s3,C,rt,can,cw0,lg,SM)
    finally:C.Ladder55.__init__=orig_init;C.Ladder55.run_stream=orig_run
QC_orig=QC.qcurve_arms
def selftest():
    QR.selftest()
    rng=np.random.default_rng(0);nll=rng.random(CH).astype(np.float32)*2;sel=rng.random(CH)<.1
    d=drop_set('tau',0.5,nll,sel,9,{});assert (d==(sel&(nll<0.5))).all() and not (d&~sel).any()
    r=drop_set('rand',None,nll,sel,9,{9:7});assert r.sum()==7 and not (r&~sel).any() and (r==drop_set('rand',None,nll,sel,9,{9:7})).all()
    assert drop_set('rand',None,nll,sel,9,{9:10**6}).sum()==sel.sum() and not drop_set('none',None,nll,sel,9,{}).any()
    M={'band_GAINQ512':np.asarray(512),'all_GAINQ512':np.asarray(0),'qb_GAINQ512_9':CM.pk(sel)};f=FocalSel('GAINQ_FOCAL_T05',M,log=lambda *x:None);nat=np.ones(8192+CH,bool)
    kp,info=f.keep(9,None,None,None,None,nat);assert kp.shape==(CH,8192+CH) and (f.qb==sel).all() and info['q_banded']==int(sel.sum())
    f.keep(3,None,None,None,None,nat);assert f.qb is None
    print('e57_gainq_focal_run selftest ok')
def main():
    if '--selftest' in sys.argv:selftest();return
    QC['masks']=QC._pop('--qband-masks');QC['ntg']=QC._pop('--ntg-from')
    QR.SelMaskQ=FocalSel
    import e57_main_run as M
    names=[str(x) for x in np.load(QC['masks'])['names']] if QC['masks'] else []
    M.ARMS_ALL=tuple(M.ARMS_ALL)+tuple(names)+tuple(FOC);M.stage_arms=focal_arms;M.stage_reads=QC.qcurve_reads
    if '--stage' in sys.argv and sys.argv[sys.argv.index('--stage')+1]=='arms':assert QC['masks'] and QC['ntg']
    M.main()
if __name__=='__main__':main()
