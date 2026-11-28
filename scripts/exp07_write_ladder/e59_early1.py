"""S2628-EARLY step 1, the ladder from P1d towards ISO1 (job 1364 'early1'; GPU; NEW file; implementation thread 11/28; Hans 11/28 16:03 KST "응 넘어가서 확인해보자"; EXPLORATORY). First 32 chunks only, reads R1 (neutral 8192-token tail) after 8 / 16 / 32 exposures, 16 x 16 candidate scores (12 exposed + 4 never exposed). Every arm is P1d (job 1362: carried 8-chunk window recorded with the D rule, e59_p1d.install) with ONE change on top of the previous rung:
  L1  : the write loss is the NTP loss over the tokens of the 16 READ-fact sentences only (positions predicting a sentence token, i.e. from the token before the sentence to its second-last token; token mean), as ISO1's loss is over its 16 isolated sentences;
  L2  : L1 + in the write forward every row of the current chunk sees, inside the current chunk, only the 16 read-fact sentences, each with the token before it (and itself; positions kept) = the current chunk reduced to the 16 sentences; the D window stays;
  L2b : L2 + each read-fact sentence (with the token before it) sees, inside the current chunk, only itself = the sentences isolated from each other as in ISO1 (the D window still visible; ISO1 has no window).
Everything else is the P1d stream (C0 keep for the sentence rows vs earlier chunks, clipped SGD, natural clip, same lr, the carried window and suffix chain by the D rule, unaffected by the write mask). L3 (ISO1 rescaled to the P1d step norm) is left out: step 0 found the step sizes already equal (cumulative update norm e32 ISO1 32.0 vs P1d 30.4).
Checks (--check): the loss positions are exactly the read-fact sentence positions (count per chunk); the chunk-0 write differs from 1348 P1s for every arm (the loss / mask changed); the keep of L2 / L2b differs from C0 only inside the current chunk.
  python3 e59_early1.py --seed 231 --out O [--nch 32] [--arms L1,L2,L2b] [--check] | --selftest"""
import sys,os,json,pathlib,argparse
import numpy as np
HERE=pathlib.Path(__file__).resolve().parent;ROOT=HERE.parents[1]
for p in (HERE,HERE/'e57',ROOT/'scripts/exp04_what_ttt_retains',ROOT/'scripts/exp05_write_dynamics'):sys.path.insert(0,str(p))
NREAD=16;CH=1024;NW=8;WIN=NW*CH
def read_spans(chunk):return sorted((int(x['start']),int(x['end'])) for x in chunk['sents'] if int(x['fact'])<NREAD)
def loss_positions(chunks,nch):
    """bool [nch*CH]: position i is in the loss when token i+1 lies in a read-fact sentence"""
    m=np.zeros(nch*CH,bool)
    for k in range(nch):
        for a,b in read_spans(chunks[k]):m[k*CH+max(a-1,0):k*CH+b-1]=True
    return m
def keep_L(kp0,chunk,mode):
    """kp0 = the C0 keep [CH, WIN+CH] of the chunk; L2: every row sees, in the current chunk, only read-fact sentence tokens and itself; L2b: rows of a sentence (from the token before it) see only that span"""
    kp=np.array(kp0,bool,copy=True);sp=read_spans(chunk);allow=np.zeros(CH,bool)
    for a,b in sp:allow[max(a-1,0):b]=True      # each sentence with the token before it (ISO1 rows are '\n\n' + sentence)
    cur=kp[:,WIN:];cur&=allow[None,:];np.fill_diagonal(cur,np.diagonal(np.asarray(kp0,bool)[:,WIN:]))
    if mode=='L2b':
        for a,b in sp:
            a0=max(a-1,0);own=np.zeros(CH,bool);own[a0:b]=True;cur[a0:b]&=own[None,:]
    kp[:,WIN:]=cur;return kp
class KeepL:
    def __init__(self,inner,chunks,mode):self.inner=inner;self.chunks=chunks;self.mode=mode;self.name=mode
    def keep(self,k,m,Kc,Vc,ids_k,keep_all):
        kp,info=self.inner.keep(k,m,Kc,Vc,ids_k,keep_all)
        if self.mode=='L1':return kp,info
        return keep_L(kp,self.chunks[k],self.mode),info
def selftest():
    ch=[{'sents':[dict(fact=0,start=10,end=20),dict(fact=20,start=30,end=40),dict(fact=1,start=50,end=60)]} for _ in range(3)]
    m=loss_positions(ch,2);assert m[9:19].all() and not m[19] and not m[29:39].any() and m[49:59].all() and m[CH+9] and m.sum()==2*20
    kp0=np.ones((CH,WIN+CH),bool);kp0[:,WIN:]=np.tril(np.ones((CH,CH),bool))
    K2=keep_L(kp0,ch[0],'L2');assert K2[15,WIN+10:WIN+16].all() and not K2[15,WIN+0:WIN+9].any() and K2[15,WIN+9] and not K2[35,WIN+30:WIN+35].any() and K2[35,WIN+35] and K2[55,WIN+10:WIN+20].all() and K2[:,:WIN].all()
    Kb=keep_L(kp0,ch[0],'L2b');assert Kb[55,WIN+49:WIN+56].all() and not Kb[55,WIN+10:WIN+20].any() and Kb[9,WIN+9] and not Kb[9,WIN+0:WIN+9].any() and Kb[15,WIN+9:WIN+16].all()
    assert not (K2&~kp0).any() and not (Kb&~K2).any()
    print('e59_early1 selftest ok: loss positions, keeps L2 / L2b')
if __name__=='__main__':
    ap_=argparse.ArgumentParser();ap_.add_argument('--seed',type=int,default=231);ap_.add_argument('--out');ap_.add_argument('--nch',type=int,default=32);ap_.add_argument('--expo',default='8,16,32');ap_.add_argument('--arms',default='L1,L2,L2b');ap_.add_argument('--selftest',action='store_true');ap_.add_argument('--check',action='store_true');ap_.add_argument('--base-dir',default='experiments/exp07_write_ladder/kvmulti');a=ap_.parse_args()
    if a.selftest:selftest();sys.exit(0)
    import e59_p1d as PD,e59_kvmulti as KV,e59_shadow as SH,e59_kvpos2 as KP,e55_iso as ISO
    out=pathlib.Path(a.out);out.mkdir(parents=True,exist_ok=True);CHK={};lg=lambda *x:print(*x,flush=True)
    for x in a.arms.split(','):PD.FSETS[x]='all';KP.CONDS[x]=('L',x)
    _mk=SH.mk_uv
    def mk_uv(spec,w,seed,lg_):
        if isinstance(spec,tuple) and spec[0]=='L':return KeepL(_mk(KP.CONDS['C0'],w,seed,lg_),w['chunks'],spec[1])
        return _mk(spec,w,seed,lg_)
    SH.mk_uv=mk_uv;PD.install(lg,CHK)
    orig=ISO.Iso55.__init__
    def init(self,*aa,**kk):
        orig(self,*aa,**kk);oseq=self._stream_seq
        def sseq(ids,nch):
            if getattr(self,'chunks_meta',None) is None:return oseq(ids,nch)
            m=loss_positions(self.chunks_meta,nch);CHK.setdefault('loss_positions',[int(m[k*CH:(k+1)*CH].sum()) for k in range(min(nch,4))]);return self.rt.seq(np.asarray(ids)[:nch*CH],mask=m)
        self._stream_seq=sseq
    ISO.Iso55.__init__=init
    KV.ARMS={x:x for x in a.arms.split(',')}
    KV.run(argparse.Namespace(seed=a.seed,out=str(out),nch=a.nch,expo=a.expo,save_final=False,selftest=False))
    r=json.load(open(out/f'kvmulti_s{a.seed}.json'));b=json.load(open(pathlib.Path(a.base_dir)/f'kvmulti_s{a.seed}.json'));sb=b['arms']['P1s']['steps']
    for arm in a.arms.split(','):s=r['arms'][arm]['steps'];CHK[arm]=dict(chunk0_gnorm=s[0]['gnorm'],chunk0_gnorm_P1s=sb[0]['gnorm'],chunk0_ce=s[0]['ce'],chunk0_ce_P1s=sb[0]['ce'],diverged=r['arms'][arm]['diverged'])
    json.dump(CHK,open(out/f'early1_checks_s{a.seed}.json','w'),default=float);lg('EARLY1 CHECKS',CHK)
    if a.check:
        import e59_u12 as U,e55_run as R55,e57_syn_pass as SP
        enc,w,tf=R55.build(SP.synth_args(a.seed,U.CELL),a.seed,read_apps={16,32,64,128},arms=['NAT']);exp=[sum(b_-1-max(a_-1,0) for a_,b_ in read_spans(w['chunks'][k])) for k in range(4)]
        assert CHK['loss_positions']==exp,('loss positions',CHK['loss_positions'],exp)
        for arm in a.arms.split(','):assert abs(CHK[arm]['chunk0_gnorm']-CHK[arm]['chunk0_gnorm_P1s'])>1e-4 and not CHK[arm]['diverged'],('chunk-0 write equals P1s or diverged',arm,CHK[arm])
        g0={arm:CHK[arm]['chunk0_gnorm'] for arm in a.arms.split(',')};assert len(set(round(v,6) for v in g0.values()))==len(g0),('two rungs give the same chunk-0 write',g0)
