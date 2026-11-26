"""Receiving-side / sending-side cut arms of the multi-step real write (job 1344t3; GPU; new file; thread-server 11/26; wrapper around e59_kvmask.py, which stays UNCHANGED: it replaces KM.ARMS and KM.QMaskUV and calls KM.run). Both arms write with the SAMEVIS graph and the ORIGINAL loss (clipped SGD, natural clip, same learning rate) plus an additional attention mask on (query row, key) pairs of the write forward (prefix and suffix layers):
  M2As-r (receiving side cut): every row of the current sentence of a read fact cannot see the keys from the START of the earliest earlier mention K of the same fact in the window up to (excluding) the start of the current sentence: K itself, everything between K and the sentence (earlier-chunk background, and in the current chunk the positions before the sentence). The sentence's own positions and the positions before the earliest K stay visible.
  M2As-w (sending side cut, current chunk only): every row of the current chunk from position 0 up to the end of the current sentence cannot see any earlier mention K of the same fact. (Rows of earlier chunks are not recomputed by the write forward: their carried keys/values are the recorded natural ones, so the cut cannot reach them.)
Chunks without an earlier mention in the window give the SAMEVIS keep. Reads and diagnostics as e59_kvmask.py.
  python3 e59_kvmask2.py --seed 231 --out O [--nch 128] [--arms M2As-r,M2As-w] [--save-final] [--check] | --selftest"""
import sys,argparse
import numpy as np
import e59_kvmask as KM
from e59_kvmask import same_fact_keys,NREAD,CH,WIN
ARMS2={'M2As-x':'x'}      # 'w' (sending-side cut, current chunk only) is implemented in the class but NOT used: replaced by M2As-o (Hans 11/26)
class QMaskUV2(KM.QMaskUV):
    def keep(self,k,m,Kc,Vc,ids_k,keep_all):
        kp,info=self.inner.keep(k,m,Kc,Vc,ids_k,keep_all);kp=np.array(kp,bool,copy=True);pairs=0;nrow=0;nk=0;rows_all={}
        if self.mode=='x':            # M2As-x: only the rows of the sentences of the EXTRA facts (>= NREAD, no read question) cannot see their own earlier mentions K; the read facts keep SAMEVIS
            for xs in self.chunks[k]['sents']:
                f=int(xs['fact'])
                if f<NREAD:continue
                keys=same_fact_keys(self.chunks,k,f)
                if not keys:rows_all[f]=[];continue
                rows=np.arange(int(xs['start']),int(xs['end']));cols=np.concatenate([np.arange(s,e) for s,e in keys]);sub=kp[np.ix_(rows,cols)];pairs+=int(sub.sum());kp[np.ix_(rows,cols)]=False;nrow+=len(rows);nk+=len(cols);rows_all[f]=rows.tolist()
            self.stat[k]=dict(pairs=pairs,rows=nrow,outside=0,keys=nk);self.rows[k]=rows_all;info=dict(info);info['extra_pairs']=pairs;return kp,info
        for d in self.ap[k]:
            f=int(d['fact'])
            if f>=NREAD:continue
            keys=same_fact_keys(self.chunks,k,f);a_,b_=d['span']
            if self.mode=='o':                      # M2As-o = M2As-s + the rows of the current sentence cannot see the other-fact sentences of the CURRENT chunk (also in chunks without an earlier mention)
                rows=np.arange(a_,b_);cl=[np.arange(s,e) for s,e in keys]+[np.arange(WIN+int(x['start']),WIN+int(x['end'])) for x in self.chunks[k]['sents'] if int(x['fact'])!=f];cols=np.concatenate(cl) if cl else np.zeros(0,np.int64)
                if len(cols)==0:rows_all[f]=[];continue
                sub=kp[np.ix_(rows,cols)];pairs+=int(sub.sum());kp[np.ix_(rows,cols)]=False;nrow+=len(rows);nk+=len(cols);rows_all[f]=rows.tolist();continue
            if not keys:rows_all[f]=[];continue
            if self.mode=='r2':rows=np.arange(a_,b_);cols=np.arange(min(s for s,e in keys),WIN+a_)
            else:rows=np.arange(0,b_);cols=np.concatenate([np.arange(s,e) for s,e in keys])
            sub=kp[np.ix_(rows,cols)];pairs+=int(sub.sum());kp[np.ix_(rows,cols)]=False;nrow+=len(rows);nk+=len(cols);rows_all[f]=rows.tolist()
        self.stat[k]=dict(pairs=pairs,rows=nrow,outside=0,keys=nk);self.rows[k]=rows_all;info=dict(info);info['extra_pairs']=pairs;return kp,info
def selftest():
    chunks=[{'sents':[dict(fact=0,start=10,end=20),dict(fact=1,start=30,end=40)]} for _ in range(12)];base=np.ones(WIN+CH,bool)
    class In:
        def keep(self,k,m,Kc,Vc,ids_k,keep_all):return np.broadcast_to(np.asarray(keep_all,bool),(CH,WIN+CH)).copy(),{}
    ap=[[dict(fact=0,span=(10,20),place=[12,13],name=[17,18])] for _ in range(12)]
    u=QMaskUV2(In(),'r2',chunks,ap,1);kp,_=u.keep(3,None,None,None,None,base);K1=(NW_:=8)*CH-3*CH+10
    assert (~kp).sum()==10*(WIN+10-K1) and not kp[10:20,K1:WIN+10].any() and kp[10:20,:K1].all() and kp[10:20,WIN+10:].all() and kp[0:10].all() and kp[20:].all()
    u=QMaskUV2(In(),'w',chunks,ap,1);kp,_=u.keep(3,None,None,None,None,base);cols=[(8-3+j)*CH+np.arange(10,20) for j in range(3)];cols=np.concatenate(cols)
    assert (~kp).sum()==20*30 and not kp[0:20][:,cols].any() and kp[20:].all() and kp[0:20][:,:cols.min()].all()
    u0=QMaskUV2(In(),'w',chunks,ap,1);k0,_=u0.keep(0,None,None,None,None,base);assert k0.all()
    u=QMaskUV2(In(),'o',chunks,ap,1);kp,_=u.keep(3,None,None,None,None,base);cK=np.concatenate([(8-3+j)*CH+np.arange(10,20) for j in range(3)]);cO=WIN+np.arange(30,40)
    assert (~kp).sum()==10*(30+10) and not kp[10:20][:,cK].any() and not kp[10:20][:,cO].any() and kp[10:20][:,WIN+10:WIN+20].all() and kp[20:].all()
    uo0=QMaskUV2(In(),'o',chunks,ap,1);ko0,_=uo0.keep(0,None,None,None,None,base);assert (~ko0).sum()==10*10 and not ko0[10:20][:,cO].any()      # chunk 0: the other-fact part still applies
    chunks2=[{'sents':[dict(fact=0,start=10,end=20),dict(fact=20,start=30,end=40)]} for _ in range(12)];ux=QMaskUV2(In(),'x',chunks2,[[] for _ in range(12)],1);kx,_=ux.keep(3,None,None,None,None,base)
    cK=np.concatenate([(8-3+j)*CH+np.arange(30,40) for j in range(3)]);assert (~kx).sum()==10*30 and not kx[30:40][:,cK].any() and kx[:30].all() and kx[40:].all() and kx[30:40][:,WIN:].all();k0x,_=ux.keep(0,None,None,None,None,base);assert k0x.all()
    print('e59_kvmask2 selftest ok: receiving-side cut (r2), sending-side cut (w, unused), M2As-o matrices, M2As-x (extra-fact rows x own K)')
if __name__=='__main__':
    ap_=argparse.ArgumentParser();ap_.add_argument('--seed',type=int,default=231);ap_.add_argument('--out');ap_.add_argument('--nch',type=int,default=128);ap_.add_argument('--expo',default='8,16,32,48,64,80,96,112,128');ap_.add_argument('--save-final',action='store_true');ap_.add_argument('--selftest',action='store_true');ap_.add_argument('--check',action='store_true');ap_.add_argument('--arms',default='M2As-x');ap_.add_argument('--base-dir',default='experiments/exp07_write_ladder/kvmulti');a=ap_.parse_args()
    if a.selftest:selftest()
    else:
        import inspect,textwrap
        src=textwrap.dedent(inspect.getsource(KM.run));old="assert VAL[f'chunk0_gnorm_rel_diff_{arm_}']<5e-3,VAL";assert src.count(old)==1
        src=src.replace(old,"assert arm_=='M2As-o' or VAL[f'chunk0_gnorm_rel_diff_{arm_}']<5e-3,VAL");ns={};exec(compile(src,'<run patched: chunk-0 check not for M2As-o>','exec'),KM.__dict__,ns)      # M2As-o differs from SAMEVIS already in chunk 0 (other-fact sentences of the current chunk are hidden); the file e59_kvmask.py on disk is untouched
        KM.ARMS=ARMS2;KM.QMaskUV=QMaskUV2;ns['run'](a)
