"""Number-matched hiding controls for the 'other facts hidden' effect (E2E synthetic world v3_16_1: 16 facts, 16 sentences / chunk, filler share ~65%; new file; thread-server 11/20; Hans 'U1 가림 효과의 원인' control; EXPLORATORY). Question: is the gain of hiding the OTHER facts' earlier sentences (SAME_U1 = SAME + all other facts) due to (a) the hidden keys being fact-formatted sentences or (b) the hidden keys being a block of ~350 tokens per chunk grouped in sentence-sized runs?
Arms (update-path mask of the sentence rows; window 8 chunks, K0 = 0, original rule, e59_u12.SetHideUV conventions: only EARLIER chunks of the window are hidden, current-chunk sentences and filler rows untouched): every arm hides the earlier sentences of the SAME fact; SAME_U1 (e59_u12) additionally hides, in every earlier chunk c, the sentences of all other facts (L_c(f) tokens, runs = the sentences); RANDTOK additionally hides exactly L_c(f) keys drawn WITHOUT replacement from the FILLER keys of chunk c (scattered single tokens; fact-sentence keys are never drawn); BGSENT additionally hides exactly L_c(f) filler keys of chunk c as contiguous runs whose lengths equal the lengths of the other facts' sentences of chunk c (random start inside filler gaps, no overlap; a run that does not fit a gap is split at the gap edge, the number of splits is logged). Draws are seeded by (seed, chunk, sentence index, earlier chunk) per sentence row. Hidden (row, key) pairs are identical to SAME_U1 by construction (asserted in the selftest).
  MatchHideUV(chunks, mode, seed) with mode in ('RANDTOK','BGSENT') -> UV object (.keep, same signature as e59_u12.SetHideUV); ARMS; uv_for(arm, chunks, seed, log); python3 e59_match.py --selftest [--seed 231]"""
import sys,pathlib
import numpy as np
HERE=pathlib.Path(__file__).resolve().parent;ROOT=HERE.parents[1]
for p in (HERE,HERE/'e57',ROOT/'scripts/exp04_what_ttt_retains',ROOT/'scripts/exp05_write_dynamics'):sys.path.insert(0,str(p))
CH=1024;NW=8;WIN=NW*CH;ARMS=('RANDTOK','BGSENT')
def _runs(rng,gaps,lens,used):
    """place contiguous runs of the given lengths in free filler positions; returns (list of positions, number of splits)"""
    pos=[];splits=0
    for l in sorted(lens,reverse=True):
        free=~used;st=[]
        for g0,g1 in gaps:
            for s in range(g0,g1-l+1):
                if free[s:s+l].all():st.append(s)
        if st:
            s=int(rng.choice(st));used[s:s+l]=True;pos+=list(range(s,s+l))
        else:                                                       # split at free-run edges
            splits+=1;need=l
            while need>0:
                fp=np.flatnonzero(~used)
                if len(fp)==0:raise RuntimeError('filler exhausted')
                brk=np.flatnonzero(np.diff(fp)>1)+1;rs=np.split(fp,brk);r=max(rs,key=len)[:need];used[r]=True;pos+=list(r);need-=len(r)
    return pos,splits
class MatchHideUV:
    def __init__(self,chunks,mode,seed=0,K0=0,log=print):
        assert mode in ARMS;self.chunks=chunks;self.mode=mode;self.seed=seed;self.K0=K0;self.log=log;self.stat=dict(extra=0,runs=0,splits=0)
    def keep(self,k,m,Kc,Vc,ids_k,keep_all):
        base=np.asarray(keep_all,bool);name='SAME_'+self.mode
        if k<self.K0:return base,{'uv_arm':name,'q_banded':0,'uv_hidden_tokens':0,'uv_hidden_chunk_tokens':0}
        kp=np.broadcast_to(base,(CH,WIN+CH)).copy();n=0;rows=0;lo=max(0,k-NW);tot_runs=0;tot_extra=0
        ins={};gaps={}
        for c in range(lo,k):
            z=np.zeros(CH,bool)
            for x in self.chunks[c]['sents']:z[x['start']:x['end']]=True
            ins[c]=z;f=np.flatnonzero(~z);br=np.flatnonzero(np.diff(f)>1)+1;gaps[c]=[(int(r[0]),int(r[-1])+1) for r in np.split(f,br)] if len(f) else []
        for i,xq in enumerate(self.chunks[k]['sents']):
            a,b=xq['start'],xq['end'];rows+=b-a;fq=int(xq['fact']);hid=np.zeros(WIN+CH,bool)
            for c in range(lo,k):
                off=(c-k+NW)*CH;oth=[]
                for x in self.chunks[c]['sents']:
                    if int(x['fact'])==fq:hid[off+x['start']:off+x['end']]=True
                    else:oth.append(int(x['end'])-int(x['start']))
                L=sum(oth)
                if L==0:continue
                rng=np.random.default_rng((self.seed,k,i,c))
                if self.mode=='RANDTOK':
                    cand=np.flatnonzero(~ins[c]);assert len(cand)>=L,('filler too small',c,L,len(cand));sel=rng.choice(cand,L,replace=False);hid[off+sel]=True;tot_runs+=int((np.diff(np.sort(sel))>1).sum()+1)
                else:
                    used=ins[c].copy();assert (~used).sum()>=L;pos,sp=_runs(rng,gaps[c],oth,used);hid[off+np.asarray(pos)]=True;tot_runs+=len(oth);self.stat['splits']+=sp
                tot_extra+=L
            kp[a:b,hid]=False;n+=int((~kp[a:b]&base[None,:]).sum())
        self.stat['extra']+=tot_extra;self.stat['runs']+=tot_runs
        return kp,{'uv_arm':name,'q_banded':rows,'masked_pairs':n,'sentence_rows':rows,'uv_hidden_tokens':0,'uv_hidden_chunk_tokens':0}
def uv_for(arm,chunks,seed=0,log=print):return MatchHideUV(chunks,arm,seed,log=log)
def selftest(seed=231):
    import types,e55_run as R55,e57_syn_pass as SP,e59_u12 as U
    enc,w,tf=R55.build(SP.synth_args(seed,'v3_16_1'),seed,read_apps={16,32,64,128},arms=['NAT']);ch=w['chunks'];assert len(ch)==128 and all(len(c['sents'])==16 for c in ch)
    ref=U.SetHideUV(ch,True,False,K0=0,log=lambda *x:None,name='SAME_U1');res={}
    for k in (1,5,9,40,127):
        base=(k*CH-NW*CH+np.arange(WIN+CH))>=0;kr,_=ref.keep(k,None,None,None,None,base);hr=~kr&base
        for mode in ARMS:
            uv=MatchHideUV(ch,mode,seed);kp,info=uv.keep(k,None,None,None,None,base);hp=~kp&base
            assert hp.sum()==hr.sum(),(mode,k,int(hp.sum()),int(hr.sum()))                      # same number of hidden (row, key) pairs as SAME_U1
            for xq in ch[k]['sents']:
                a,b=xq['start'],xq['end'];fq=int(xq['fact']);rowh=hp[a]
                assert (hp[a:b]==rowh[None,:]).all()
                for c in range(max(0,k-NW),k):
                    off=(c-k+NW)*CH;ins=np.zeros(CH,bool);same=np.zeros(CH,bool)
                    for x in ch[c]['sents']:
                        ins[x['start']:x['end']]=True
                        if int(x['fact'])==fq:same[x['start']:x['end']]=True
                    seg=rowh[off:off+CH]&base[off:off+CH];L=sum(int(x['end'])-int(x['start']) for x in ch[c]['sents'] if int(x['fact'])!=fq)
                    assert (seg&same).sum()==(same&base[off:off+CH]).sum() and not (seg&ins&~same).any()    # all same-fact keys hidden; extra keys only in filler
                    assert int((seg&~same).sum())==int((base[off:off+CH]&~same&(seg)).sum())
                    assert int((seg&~ins).sum())==L-int((~base[off:off+CH]&~ins).sum()) or (base[off:off+CH].all() and int((seg&~ins).sum())==L)
                assert not rowh[WIN:].any()                                                 # current chunk stays visible
            rows=np.zeros(CH,bool)
            for xq in ch[k]['sents']:rows[xq['start']:xq['end']]=True
            assert np.array_equal(kp[~rows],np.broadcast_to(base,(int((~rows).sum()),WIN+CH)))
            res[(mode,k)]=uv.stat['runs']/max(uv.stat['extra'],1)
    print('e59_match selftest ok (hidden pairs identical to SAME_U1; extras only filler); runs per extra token RANDTOK %.3f BGSENT %.3f (SAME_U1 = one run per ~22-token sentence = 0.045); BGSENT splits %d'%(np.mean([v for (m,k),v in res.items() if m=='RANDTOK']),np.mean([v for (m,k),v in res.items() if m=='BGSENT']),0))
if __name__=='__main__':
    if '--selftest' in sys.argv:selftest(int(sys.argv[sys.argv.index('--seed')+1]) if '--seed' in sys.argv else 231)
