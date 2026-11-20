"""Hiding / showing arm registry for the same-fact specificity experiments (jobs 1330-1334; new file; thread-server 11/20; Hans '코드 만들기 시작해'; EXPLORATORY). One UV-object registry used by the E2E driver (e59_srv_run.py) and the Qwen driver (e59_qwen_hide.py): every UV has .keep(k, m, Kc, Vc, ids_k, keep_all) -> (bool [CH, WIN+CH] key-visibility of the chunk-k rows, info), key axis = 8 earlier chunks (window, WIN = 8*CH) + the current chunk, like e57_oracle_masks / e59_u12 (current-chunk keys stay visible; the causal mask is applied by the attention).
HIDE arms (earlier chunks of the window only, K0 = 0, sentence rows only; filler rows see everything), the same definitions as job 1308z:
  SAME      hides the earlier sentences of the SAME fact (e59_u12.SetHideUV)            BOTH  hides all earlier fact sentences of every fact (= SAME + all other facts; e59_u12 'SAME_U1' in the 16-fact world; the 1308z ALLPREV)
  OTHERONLY hides the earlier sentences of every OTHER fact, same-fact sentences stay visible (e57_prevctrl_masks.OtherPrevFactUV)
  RANDF / RANDC  number-matched random keys: exactly L_c keys per earlier chunk c (L_c = fact-sentence tokens of chunk c = the BOTH count) drawn without replacement from the FILLER keys (RANDF) or from all visible keys of chunk c (RANDC; for worlds whose filler is too small, e.g. FictionalQA; hit rate on fact keys reported) (e57_prevctrl_masks.RandPrevKeysUV)
SHOW arms (job 1333 / 1334): EVERY row of chunk k loses the keys of all earlier chunks (the chunk alone, as CHK) EXCEPT that the SENTENCE rows keep a set of earlier keys: SHOWSAME = the earlier sentences of the same fact, SHOWOTHER = the earlier sentences of the other facts, ALLHID = nothing (stream-engine equivalent of CHK). Filler rows see only their own chunk in all three arms.
  uv_for(arm, chunks, seed, log) ; ARMS ; python3 e59_hidearms.py --selftest [--seed 231]"""
import sys,pathlib
import numpy as np
HERE=pathlib.Path(__file__).resolve().parent;ROOT=HERE.parents[1]
for p in (HERE,HERE/'e57',ROOT/'scripts/exp04_what_ttt_retains',ROOT/'scripts/exp05_write_dynamics'):sys.path.insert(0,str(p))
CH=1024;NW=8;WIN=NW*CH
HIDE_ARMS=('SAME','BOTH','OTHERONLY','RANDF','RANDC');SHOW_ARMS=('SHOWSAME','SHOWOTHER','ALLHID');ARMS=HIDE_ARMS+SHOW_ARMS
class ShowUV:
    def __init__(self,chunks,mode,K0=0,log=print):
        assert mode in SHOW_ARMS;self.chunks=chunks;self.mode=mode;self.K0=K0;self.log=log
    def allowed_keys(self,k,fq):
        lo=max(0,k-NW);ks=[]
        if self.mode=='ALLHID':return ks
        for c in range(lo,k):
            off=(c-k+NW)*CH
            for x in self.chunks[c]['sents']:
                same=int(x['fact'])==fq
                if (self.mode=='SHOWSAME' and same) or (self.mode=='SHOWOTHER' and not same):ks.append((off+int(x['start']),off+int(x['end'])))
        return ks
    def keep(self,k,m,Kc,Vc,ids_k,keep_all):
        base=np.asarray(keep_all,bool)
        if k<self.K0:return base,{'uv_arm':self.mode,'q_banded':0,'uv_hidden_tokens':0,'uv_hidden_chunk_tokens':0}
        kp=np.broadcast_to(base,(CH,WIN+CH)).copy();kp[:,:WIN]=False;rows=0
        for xq in self.chunks[k]['sents']:
            a,b=int(xq['start']),int(xq['end']);rows+=b-a
            for s,e in self.allowed_keys(k,int(xq['fact'])):kp[a:b,s:e]=base[s:e][None,:]
        n=int((~kp&base[None,:]).sum())
        return kp,{'uv_arm':self.mode,'q_banded':rows,'masked_pairs':n,'sentence_rows':rows,'uv_hidden_tokens':0,'uv_hidden_chunk_tokens':0}
def uv_for(arm,chunks,seed=0,log=print):
    import e59_u12 as U,e57_prevctrl_masks as PM
    if arm=='SAME':return U.SetHideUV(chunks,False,False,K0=0,log=log,name='SAME')
    if arm=='BOTH':return U.SetHideUV(chunks,True,False,K0=0,log=log,name='BOTH')
    if arm=='OTHERONLY':return PM.OtherPrevFactUV(chunks,K0=0,log=log)
    if arm=='RANDF':return PM.RandPrevKeysUV(chunks,seed,K0=0,log=log,population='filler')
    if arm=='RANDC':return PM.RandPrevKeysUV(chunks,seed,K0=0,log=log,population='chunk')
    if arm in SHOW_ARMS:return ShowUV(chunks,arm,K0=0,log=log)
    raise KeyError(arm)
def _hidden(kp,base):return ~kp&base[None,:]
def check_world(ch,seed,ks=(1,5,9,40,127),pop_filler_ok=True,nfact=16):
    """brute-force checks of every arm on the world chunks ch"""
    n=len(ch);out={}
    for k in ks:
        if k>=n:continue
        base=(k*CH-NW*CH+np.arange(WIN+CH))>=0;rows=np.zeros(CH,bool);sent=[(int(x['start']),int(x['end']),int(x['fact'])) for x in ch[k]['sents']]
        for a,b,f in sent:rows[a:b]=True
        for arm in ARMS:
            uv=uv_for(arm,ch,seed,lambda *x:None)
            if arm in ('RANDF',) and not pop_filler_ok:continue
            kp,info=uv.keep(k,None,None,None,None,base);H=_hidden(kp,base)
            assert (kp[:,WIN:]==base[None,WIN:]).all(),(arm,k,'current chunk must stay visible')
            if arm in HIDE_ARMS:assert np.array_equal(kp[~rows],np.broadcast_to(base,(int((~rows).sum()),WIN+CH))),(arm,k,'filler rows must be untouched')
            for a,b,f in sent:
                row=H[a];assert (H[a:b]==row[None,:]).all(),(arm,k,'rows of a sentence share one key set')
                exp_same=np.zeros(WIN+CH,bool);exp_oth=np.zeros(WIN+CH,bool);fill=np.zeros(WIN+CH,bool)
                for c in range(max(0,k-NW),k):
                    off=(c-k+NW)*CH;ins=np.zeros(CH,bool)
                    for x in ch[c]['sents']:
                        s,e,ff=int(x['start']),int(x['end']),int(x['fact']);ins[s:e]=True
                        (exp_same if ff==f else exp_oth)[off+s:off+e]=True
                    fill[off:off+CH]=~ins
                es,eo=exp_same&base,exp_oth&base;fl=fill&base
                if arm=='SAME':assert (row==es).all(),(arm,k)
                elif arm=='BOTH':assert (row==(es|eo)).all(),(arm,k)
                elif arm=='OTHERONLY':assert (row==eo).all(),(arm,k)
                elif arm in ('RANDF','RANDC'):
                    assert not row[WIN:].any();tot=int(((es|eo)&base).sum())
                    # per earlier chunk the number of hidden keys equals L_c (all fact-sentence tokens of that chunk)
                    for c in range(max(0,k-NW),k):
                        off=(c-k+NW)*CH;Lc=sum(int(x['end'])-int(x['start']) for x in ch[c]['sents']);vis=int(base[off:off+CH].sum());assert int(row[off:off+CH].sum())==min(Lc,vis) or (int(row[off:off+CH].sum())<=Lc),(arm,k,c)
                    if arm=='RANDF':assert not (row&(es|eo)).any(),(arm,k,'RANDF must not hide fact-sentence keys')
                elif arm=='SHOWSAME':assert (row[:WIN]==(base[:WIN]&~exp_same[:WIN])).all(),(arm,k)
                elif arm=='SHOWOTHER':assert (row[:WIN]==(base[:WIN]&~exp_oth[:WIN])).all(),(arm,k)
                elif arm=='ALLHID':assert (row[:WIN]==base[:WIN]).all(),(arm,k)
            if arm in SHOW_ARMS:                                    # filler rows: only the own chunk
                assert (H[~rows][:,:WIN]==base[None,:WIN]).all(),(arm,k)
            out[(arm,k)]=int(H.sum())
    assert out[('SAME',ks[-1] if ks[-1]<n else ks[-2])]<=out[('BOTH',ks[-1] if ks[-1]<n else ks[-2])]
    return out
def selftest(seed=231):
    import e55_run as R55,e57_syn_pass as SP
    enc,w,tf=R55.build(SP.synth_args(seed,'v3_16_1'),seed,read_apps={16,32,64,128},arms=['NAT']);ch=w['chunks'];o=check_world(ch,seed)
    print('e59_hidearms selftest ok (E2E synthetic world, brute-force on chunks 1/5/9/40/127); hidden pairs at chunk 40:',{a:o[(a,40)] for a in ARMS})
if __name__=='__main__':
    if '--selftest' in sys.argv:selftest(int(sys.argv[sys.argv.index('--seed')+1]) if '--seed' in sys.argv else 231)
