"""Partner-cue check (job 1337; new file; thread-server 11/21; Hans '그것도 넣어서 코드 짜 보자'; EXPLORATORY): does writing with ONE fixed co-visible other fact create a connection 'answer <- that fact'? Stream arm PARTNER of the E2E driver (e59_srv_run.py): sentence rows of fact A see the earlier sentences (8-chunk window) of ONE pre-assigned partner fact B(A) only; every other earlier-chunk key is hidden for every row, filler rows see only their own chunk (the ShowUV convention of e59_hidearms.py; ALLHID = nothing visible, SHOWOTHER = all other facts visible are the other two arms of the job). Pairing: order = rng(seed + 7777).permutation(16); partner(f) = order[(pos(f) + 1) % 16], non-partner controls C1(f) = order[(pos(f) + 2) % 16], C2(f) = order[(pos(f) + 3) % 16] (derangements, pairwise disjoint, fixed before any result). Cue reads (--cue of the driver): after the writes the question of fact A is read at the final weights with a context sentence in front of the question: the text of the partner B(A) (condition B), of C1(A), of C2(A), each taken from the last stream chunk (same template as in the stream); the read without context is the standard out-of-window read.
  perms(seed) -> {'B','C1','C2': {fact: other fact}}; PartnerUV(chunks, perm); uv_for(arm, chunks, seed, log); cue_items(items, w, ids, tk, perm, last_chunk) -> read items whose prefix starts with the context sentence; python3 e59_partner.py --selftest [--seed 231]"""
import sys,pathlib
import numpy as np
HERE=pathlib.Path(__file__).resolve().parent;ROOT=HERE.parents[1]
for p in (HERE,HERE/'e57',ROOT/'scripts/exp04_what_ttt_retains',ROOT/'scripts/exp05_write_dynamics'):sys.path.insert(0,str(p))
CH=1024;NW=8;WIN=NW*CH;NF=16
def perms(seed,nf=NF):
    order=np.random.default_rng(int(seed)+7777).permutation(nf).tolist();pos={f:i for i,f in enumerate(order)}
    return {name:{f:order[(pos[f]+d)%nf] for f in range(nf)} for name,d in (('B',1),('C1',2),('C2',3))}
class PartnerUV:
    def __init__(self,chunks,perm,K0=0,log=print):self.chunks=chunks;self.perm=perm;self.K0=K0;self.log=log
    def keep(self,k,m,Kc,Vc,ids_k,keep_all):
        base=np.asarray(keep_all,bool)
        if k<self.K0:return base,{'uv_arm':'PARTNER','q_banded':0,'uv_hidden_tokens':0,'uv_hidden_chunk_tokens':0}
        kp=np.broadcast_to(base,(CH,WIN+CH)).copy();kp[:,:WIN]=False;lo=max(0,k-NW);rows=0
        for xq in self.chunks[k]['sents']:
            a,b=int(xq['start']),int(xq['end']);rows+=b-a;fb=self.perm[int(xq['fact'])]
            for c in range(lo,k):
                off=(c-k+NW)*CH
                for x in self.chunks[c]['sents']:
                    if int(x['fact'])==fb:s,e=off+int(x['start']),off+int(x['end']);kp[a:b,s:e]=base[s:e][None,:]
        n=int((~kp&base[None,:]).sum());return kp,{'uv_arm':'PARTNER','q_banded':rows,'masked_pairs':n,'sentence_rows':rows,'uv_hidden_tokens':0,'uv_hidden_chunk_tokens':0}
def uv_for(arm,chunks,seed,log=print):
    assert arm=='PARTNER';return PartnerUV(chunks,perms(seed)['B'],K0=0,log=log)
def cue_items(items,w,ids,tk,perm,last_chunk):
    """read items (kind 'fact', facts < 16 only) with prefix = <sentence of the context fact from chunk last_chunk> + ' ' + the original question prefix"""
    sent={int(x['fact']):(int(x['start']),int(x['end'])) for x in w['chunks'][last_chunk]['sents']};out=[]
    for it in items:
        if it['kind']!='fact' or it['fact']>=NF:continue
        f=int(it['fact']);a,b=sent[perm[f]];text=tk.decode([int(t) for t in ids[last_chunk*CH+a:last_chunk*CH+b]]).strip();new=dict(it);new['prefix']=text+' '+it['prefix'];out.append(new)
    assert len(out)==NF;return out
def selftest(seed=231):
    import e55_run as R55,e57_syn_pass as SP,e59_hidearms as H
    from tokenizers import Tokenizer
    P=perms(seed);assert all(sorted(P[n])==list(range(NF)) and all(P[n][f]!=f for f in range(NF)) for n in P)
    assert all(len({P['B'][f],P['C1'][f],P['C2'][f],f})==4 for f in range(NF)),'partner and controls must be pairwise different'
    assert perms(seed)==P and perms(seed+1)!=P
    enc,w,tf=R55.build(SP.synth_args(seed,'v3_16_1'),seed,read_apps={16,32,64,128},arms=['NAT']);ch=w['chunks'];tk=Tokenizer.from_file(str(ROOT/'data/tokenizer/llama3/tokenizer.json'))
    for k in (1,5,9,40,127):
        base=(k*CH-NW*CH+np.arange(WIN+CH))>=0;kp,info=uv_for('PARTNER',ch,seed,lambda *x:None).keep(k,None,None,None,None,base);H_=~kp&base[None,:]
        assert (kp[:,WIN:]==base[None,WIN:]).all()
        rows=np.zeros(CH,bool)
        for xq in ch[k]['sents']:rows[xq['start']:xq['end']]=True
        assert (H_[~rows][:,:WIN]==base[None,:WIN]).all(),'filler rows see only their own chunk'
        for xq in ch[k]['sents']:
            a,b=xq['start'],xq['end'];f=int(xq['fact']);row=H_[a];assert (H_[a:b]==row[None,:]).all();exp=np.zeros(WIN+CH,bool)
            for c in range(max(0,k-NW),k):
                off=(c-k+NW)*CH
                for x in ch[c]['sents']:
                    if int(x['fact'])==P['B'][f]:exp[off+x['start']:off+x['end']]=True
            assert (row[:WIN]==(base[:WIN]&~exp[:WIN])).all(),(k,f)
        # visible keys of a sentence row: only sentences of the partner (+ the own chunk)
    import e55_read as R,numpy as np_
    ids=np.asarray(R55.streams(w,enc)['A'][0]);its=R.plan(w,enc)['items']
    for nm in ('B','C1','C2'):
        ci=cue_items(its,w,ids,tk,P[nm],127);assert len(ci)==NF and all(i['prefix'].count('?')>=0 for i in ci)
    print('e59_partner selftest ok; example context:',repr(cue_items(its,w,ids,tk,P['B'],127)[3]['prefix'][:160]))
if __name__=='__main__':
    if '--selftest' in sys.argv:selftest(int(sys.argv[sys.argv.index('--seed')+1]) if '--seed' in sys.argv else 231)
