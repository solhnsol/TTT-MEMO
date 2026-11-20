"""Qwen3-4B-Base runner for the same-fact specificity arms (jobs 1330 / 1331 / 1334; new file; thread-server 11/20; EXPLORATORY; imports e59_qwen_run / e59_qwen_mask / e59_hidearms; nothing existing edited). Arms (all with the fixed-norm normalised-gradient step of the E59 Qwen runs, T = --step-norm; update-path mask in EVERY decoder layer, E2E parity: all 36 layers (the 9 fast layers 27..35 carry the fast weights; asserted: hooked layers == all decoder layers); the carried KV and every read stay natural):
  SAME / BOTH / OTHERONLY / RANDF / RANDC   hide earlier-chunk keys of the sentence rows (definitions in e59_hidearms.py, job 1308z)
  SHOWSAME / SHOWOTHER / ALLHID             every row loses the keys of all earlier chunks; sentence rows keep the earlier same-fact (SHOWSAME) / other-fact (SHOWOTHER) sentences (ALLHID: none = the chunk alone)
The window of the Qwen port is 7 earlier chunks + the current chunk (the E2E window is 8): the key axis of the e59_hidearms masks is [8 earlier chunks | current]; the Qwen mask uses its last 7 chunk slots (the oldest chunk, k-8, is not in the Qwen cache).
  python3 e59_qwen_hide.py [--fq-worlds D] --world-seed 221 --fast fullmlp --lr 1.0 --clip 1.0 --step-norm 0.30426 --arms SAME,BOTH,OTHERONLY,RANDF --reads 16,32,64,128 --out O | --selftest"""
import sys,json,hashlib
from pathlib import Path
HERE=Path(__file__).resolve().parent;sys.path.insert(0,str(HERE));sys.path.insert(0,str(HERE.parent/'exp07_write_ladder'));sys.path.insert(0,str(HERE.parent/'exp07_write_ladder'/'e57'))
import numpy as np,torch
import e59_qwen_run as E,e59_qwen_mask as QM
import e59_hidearms as H
from qwen_runtime import parts
import e59_srv_core as K
HIDEARMS=H.ARMS
class HideWriter(QM.MaskWriter):
    def __init__(self,model,CH,win,device,uv,seed):
        super().__init__(model,CH,win,device,'band',all_layers=True);self.uv=uv;self.seed=seed;self.k=0;self._mk=None;self.win=win
        dec=parts(model)[0];assert self.layer_ids==list(range(len(dec.layers))),('the mask must act in ALL decoder layers',self.layer_ids)
    def set_chunk(self,k):self.k=int(k);self._mk=None
    def allowed(self,L,off,device):
        CH=self.CH;WIN=H.WIN
        if self._mk is not None and self._mk[0]==(L,off):return self._mk[1]
        k=self.k;base=(k*CH-WIN+np.arange(WIN+CH))>=0;kp,info=self.uv.keep(k,None,None,None,None,base);self.last_info=info
        assert 0<=off<=WIN-CH+0*CH or off<=(self.win)*CH,off
        earlier=kp[:L,WIN-off:WIN] if off>0 else np.zeros((L,0),bool);cur=kp[:L,WIN:WIN+L]
        full=np.concatenate([earlier,cur],axis=1)[None,None];tri=np.zeros((L,off+L),bool);ii=np.arange(L)[:,None];jj=np.arange(off+L)[None,:];tri=(jj-off)<=ii
        a=torch.from_numpy(full&tri[None,None]).to(device);self._mk=((L,off),a);return a
    def _hook(self,mod,args,kwargs):
        if not self.active:return None
        hs=kwargs['hidden_states'] if 'hidden_states' in kwargs else args[0];L=hs.shape[1];kwargs=dict(kwargs);kwargs['attention_mask']=self.allowed(L,self._off,hs.device);return args,kwargs
def run_arm_hide(a,arm,w,stream,k_range,model,dev,T,fix,CH,sep,log=print,ST=None):
    uv=H.uv_for(arm,w['chunks'],int(a.world_seed),log);W=HideWriter(model,CH,a.window_chunks,dev,uv,a.world_seed)
    try:
        for k in k_range:
            ch,sp,fpos=E.chunk_data(w,stream,k,CH);c0=k*CH;n=len(stream);e=min((k+1)*CH,n);last=min(e,n-1);ids_c=torch.tensor(stream[c0:e],dtype=torch.long,device=dev);fh=K.ids_hash(K.fact_ids(ch));fp=torch.tensor(fpos,dtype=torch.long,device=dev)
            tgt=torch.tensor(stream[c0+1:last+1],dtype=torch.long,device=dev);box={};W.set_chunk(k)
            with torch.no_grad():_,kv,pos=W.forward(ids_c)
            def lf():
                hs=W.forward_masked(ids_c);nl=E.nll_of(model,None,None,torch.arange(last-c0,device=dev),tgt,hs=hs);box['fact']=float(nl[fp].mean().detach());box['all']=float(nl.mean().detach());box['ntok']=int(nl.numel());return nl.mean()
            rec=fix.step(lf,a.lr,a.scale,a.clip);W.push(kv,pos)
            rec.update(k=k,fact_ce=box['fact'],all_ce=box['all'],n_loss_tokens=box['ntok'],fact_hash=fh,chunk_hash=hashlib.sha256(np.asarray(stream[c0:e],np.int64).tobytes()).hexdigest()[:16],hook_layers=len(W.layer_ids),masked_pairs=int(W.last_info.get('masked_pairs',0)));yield k,rec
    finally:W.close()
def install():
    QM.install();E.ARMS=tuple(E.ARMS)+tuple(x for x in HIDEARMS if x not in E.ARMS);orig=E.run_arm
    def dispatch(a,arm,*r,**kw):return run_arm_hide(a,arm,*r,**kw) if arm in HIDEARMS else orig(a,arm,*r,**kw)
    E.run_arm=dispatch;E.run.__globals__['run_arm']=dispatch
def mask_check():
    """CPU brute force of HideWriter.allowed (key-axis mapping of the 7-chunk Qwen cache) on a synthetic chunk table, every arm, several chunks"""
    CH=1024;rng=np.random.default_rng(0);chunks=[]
    for c in range(24):
        starts=np.sort(rng.choice(np.arange(0,CH-30,30),16,replace=False));f=rng.permutation(16);chunks.append({'sents':[{'fact':int(f[i]),'start':int(st),'end':int(st)+int(rng.integers(18,28))} for i,st in enumerate(starts)]})
    for arm in H.ARMS:
        W=object.__new__(HideWriter);W.CH=CH;W.win=7;W.uv=H.uv_for(arm,chunks,5,lambda *x:None);W._mk=None
        for k in (1,3,7,9,15,23):
            off=min(k,7)*CH;W.k=k;W._mk=None;A=W.allowed(CH,off,torch.device('cpu'))[0,0].numpy();assert A.shape==(CH,off+CH)
            tri=(np.arange(off+CH)[None,:]-off)<=np.arange(CH)[:,None];assert (A[:,off:]==tri[:,off:]).all()                 # current chunk: causal, nothing hidden
            for xq in chunks[k]['sents']:
                a,b,f=xq['start'],xq['end'],xq['fact'];row=A[a,:off];assert (A[a:b,:off]==row[None,:]).all()
                for j in range(off):
                    c=k-off//CH+j//CH;pos=j%CH;snt=[x for x in chunks[c]['sents'] if x['start']<=pos<x['end']]
                    if arm in ('SAME','BOTH','OTHERONLY'):
                        hid=bool(snt) and ((snt[0]['fact']==f) if arm=='SAME' else (True if arm=='BOTH' else snt[0]['fact']!=f));assert row[j]==(not hid),(arm,k,j)
                    elif arm in ('SHOWSAME','SHOWOTHER','ALLHID'):
                        vis=bool(snt) and ((snt[0]['fact']==f) if arm=='SHOWSAME' else (snt[0]['fact']!=f if arm=='SHOWOTHER' else False));assert row[j]==vis,(arm,k,j)
                    elif arm=='RANDF':assert not (row[j]==False and bool(snt)),(arm,k,j)
            if arm in ('SHOWSAME','SHOWOTHER','ALLHID'):                                                         # filler rows: only the own chunk
                fr=np.ones(CH,bool)
                for x in chunks[k]['sents']:fr[x['start']:x['end']]=False
                assert not A[fr][:,:off].any(),arm
    print('e59_qwen_hide mask_check ok: allowed() == brute force for',len(H.ARMS),'arms (key-axis mapping of the 7-chunk cache)')
def selftest():
    mask_check()
    sys.argv=[sys.argv[0]];import tempfile,shutil
    install();td=Path(tempfile.mkdtemp());base=['--tiny','--device','cpu','--max-chunks','4','--chunk','1024','--reads','2,4','--tail-len','256','--readout-batch','16','--threads','4','--lora-layers','99','--world-seed','161','--lr','1e-2','--rerun','--step-norm','0.25']
    arms=('NAT','CHK','SAME','BOTH','OTHERONLY','RANDF','SHOWSAME','SHOWOTHER','ALLHID')
    a=E.parser().parse_args(base+['--out',str(td/'o'),'--arms',','.join(arms)]);torch.manual_seed(0);E.run(a);R={x:json.load(open(td/'o'/f'e59q_{x}_s161.json')) for x in arms}
    nl=R['NAT']['steps'][0].get('n_layers',None)
    for x,r in R.items():s=r['steps'];assert len(s)==4 and all(abs(e['applied_norm']-0.25)<=1e-4*0.25+1e-7 for e in s),x
    assert [e['fact_hash'] for e in R['BOTH']['steps']]==[e['fact_hash'] for e in R['NAT']['steps']]
    for x in ('SAME','BOTH','OTHERONLY','RANDF','SHOWSAME','SHOWOTHER','ALLHID'):
        assert abs(R[x]['steps'][0]['ce']-R['NAT']['steps'][0]['ce'])<1e-5,(x,'chunk 0 has no earlier chunk')
        assert all(e['hook_layers']>0 for e in R[x]['steps']),x
    c=[{x:R[x]['steps'][k]['ce'] for x in arms} for k in (1,2,3)]
    for cc in c:
        assert abs(cc['ALLHID']-cc['CHK'])<=1e-3*abs(cc['CHK'])+1e-4,('tiny model, all layers: ALLHID must equal CHK',cc['ALLHID'],cc['CHK'])
        assert abs(cc['ALLHID']-cc['NAT'])>1e-6
    assert len({round(cc['SHOWSAME'],6) for cc in c})>1
    shutil.rmtree(td,ignore_errors=True);print('e59_qwen_hide selftest ok (tiny model: hooked layers == all decoder layers; ALLHID == CHK; chunk 0 == NAT; fact hashes equal)')
if __name__=='__main__':
    if '--selftest' in sys.argv:selftest()
    else:
        ap=E.parser();ap.add_argument('--fq-worlds');a=ap.parse_args()
        if a.fq_worlds:import fq_qwen_run as F;F.install(a.fq_worlds)          # FictionalQA Qwen worlds (labels 211..218), as fq_qwen_run.py
        install();torch.manual_seed(0);E.run(a)
