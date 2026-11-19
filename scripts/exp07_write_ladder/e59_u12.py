"""U1 / U2 hiding experiment (E2E synthetic world with 32 facts, 16 of them OUTSIDE the read candidates; new file; thread-server 11/19; Hans 'U1/U2 가림 실험 진행'; EXPLORATORY): world 'u12_k32' = k = 32 facts, one sentence per fact in every chunk (32 sentences / chunk, chunks stay exactly 1024 tokens), 128 exposures, same frames / bank as v3_16_1. READ facts R = ids 0..15: read items ask the place of a read fact and the 16 candidates are the places of the 16 read facts ONLY (items_subset; no control items); EXTRA facts X = ids 16..31 are exposed identically but their places are never candidates. 'Other facts' of the old ORACLE / ALLPREV arms therefore split into U1 = the other 15 read facts (their answers ARE candidates) and U2 = the 16 extra facts (answers are NOT candidates).
Arms (update-path mask of the sentence rows, window 8 chunks, K0 = 0, original rule; e57_allprev_masks conventions: only sentences of EARLIER chunks are hidden, sentences of the current chunk stay visible; filler rows see everything): NAT (no mask), SAME (the earlier sentences of the SAME fact hidden = ORACLE on this world), SAME_U1 (+ earlier sentences of the other read facts hidden), SAME_U2 (+ earlier sentences of the extra facts hidden), SAME_U1U2 (everything of earlier chunks hidden = ALLPREV on this world).
  SetHideUV(chunks, hide_u1, hide_u2) -> UV object (.keep); items_subset(world); install() (registers the cell and swaps gen_facts.build_items for worlds with k = 32); visible_shares(chunks, exposures) -> per arm the share of (a) same-fact, (b) U1, (c) U2 key tokens visible to the sentence rows.
  python3 e59_u12.py --selftest [--seed 231]"""
import sys,pathlib,json
import numpy as np
HERE=pathlib.Path(__file__).resolve().parent;ROOT=HERE.parents[1]
for p in (HERE,HERE/'e57',ROOT/'scripts/exp04_what_ttt_retains',ROOT/'scripts/exp05_write_dynamics'):sys.path.insert(0,str(p))
CH=1024;NW=8;WIN=NW*CH;NREAD=16;K=32;CELL='u12_k32'
ARMS=('SAME','SAME_U1','SAME_U2','SAME_U1U2')
HIDE={'SAME':(False,False),'SAME_U1':(True,False),'SAME_U2':(False,True),'SAME_U1U2':(True,True)}
def install():
    import gen_facts as G
    G.CELLS[CELL]=dict(k=K,n=1,p=1,bank='300',n_app=128,read_apps=(16,32,64,128));G.CELL_BOOKS[CELL]=G.CELL_BOOKS['v3_64_4']
    if not getattr(G,'_u12_patched',False):
        orig=G.build_items;G.build_items=lambda w:items_subset(w) if w.get('k')==K else orig(w);G._u12_patched=True
    return G
def items_subset(world):
    """16 read items (facts 0..15); candidates = the places of the 16 read facts (own place first at gold_idx 0); no control items"""
    facts=[f for f in world['facts'] if f['id']<NREAD];assert len(facts)==NREAD;items=[]
    QF=__import__('gen_world_ctl').QFRAMES;NQ=len(QF)
    for f in facts:
        cl=[f]+[g for g in facts if g['id']!=f['id']]
        items.append({'kind':'fact','fact':f['id'],'group':f['group'],'name_id':f['id'],'place_id':f['id'],'name':f['X'],'qframe':f['id']%NQ,'prefix':QF[f['id']%NQ].format(X=f['X']),'cands':[' '+x['P'] for x in cl],'gold_idx':0,'cand_fact':[x['id'] for x in cl],'cand_group':[x['group'] for x in cl],'cand_same':[x['group']==f['group'] for x in cl],'cand_exposed':[True]*NREAD,'n_rank':NREAD})
    return items
class SetHideUV:
    def __init__(self,chunks,hide_u1,hide_u2,K0=0,log=print,name='SAME'):self.chunks=chunks;self.u1=hide_u1;self.u2=hide_u2;self.K0=K0;self.log=log;self.name=name
    def hidden_keys(self,k,fq):
        """key slices (window-relative index) hidden for a query sentence of fact fq in chunk k"""
        lo=max(0,k-NW);ks=[]
        for c in range(lo,k):
            off=(c-k+NW)*CH
            for x in self.chunks[c]['sents']:
                f=int(x['fact']);same=(f==fq);u1=(not same) and f<NREAD;u2=f>=NREAD
                if same or (u1 and self.u1) or (u2 and self.u2):ks.append((off+x['start'],off+x['end']))
        return ks
    def keep(self,k,m,Kc,Vc,ids_k,keep_all):
        base=np.asarray(keep_all,bool)
        if k<self.K0:return base,{'uv_arm':self.name,'q_banded':0,'uv_hidden_tokens':0,'uv_hidden_chunk_tokens':0}
        kp=np.broadcast_to(base,(CH,WIN+CH)).copy();n=0;rows=0
        for xq in self.chunks[k]['sents']:
            a,b=xq['start'],xq['end'];rows+=b-a
            for s,e in self.hidden_keys(k,int(xq['fact'])):n+=int(kp[a:b,s:e].sum());kp[a:b,s:e]=False
        return kp,{'uv_arm':self.name,'q_banded':rows,'masked_pairs':n,'sentence_rows':rows,'uv_hidden_tokens':0,'uv_hidden_chunk_tokens':0}
def uv_for(arm,chunks,log=print):
    u1,u2=HIDE[arm];return SetHideUV(chunks,u1,u2,K0=0,log=log,name=arm)
def visible_shares(chunks,exposures,arms=('NAT',)+ARMS):
    """token-pair shares (same fact / U1 / U2) of the earlier-sentence keys inside the 8192-distance window that are visible to the sentence rows of chunk e-1; current-chunk earlier sentences count as keys (visible in every arm)"""
    sp=[(r,int(s['fact']),r*CH+int(s['start']),r*CH+int(s['end'])) for r,c in enumerate(chunks) for s in sorted(c['sents'],key=lambda x:x['start'])];out={a:{} for a in arms}
    for e in exposures:
        r=e-1;acc={a:np.zeros(6,np.int64) for a in arms}
        for (rq,fq,a0,a1) in [s for s in sp if s[0]==r]:
            q=np.arange(a0,a1)
            for (rk,fk,b0,b1) in sp:
                if rk>rq or (rk==rq and b0>=a0) or a0-b1>=8192:continue
                kk=np.arange(b0,b1);d=q[:,None]-kk[None,:];inw=(d>0)&(d<8192)
                if not inw.any():continue
                cls=0 if fk==fq else (1 if fk<NREAD else 2);tot=int(inw.sum())
                for a in arms:
                    if a=='NAT' or rk==rq:vis=tot
                    else:u1,u2=HIDE[a];vis=tot if (cls==1 and not u1) or (cls==2 and not u2) else 0
                    acc[a][2*cls]+=vis;acc[a][2*cls+1]+=tot
        for a in arms:
            v=acc[a];out[a][e]={'same':(v[0]/v[1] if v[1] else None),'U1':(v[2]/v[3] if v[3] else None),'U2':(v[4]/v[5] if v[5] else None)}
    return out
def selftest(seed=231):
    import e55_run as R55,e55_read as R,types
    from tokenizers import Tokenizer
    G=install();a=types.SimpleNamespace(cell=CELL,seed=seed,tokenizer_file=None,bg='pool',decoy='names',stub=False,arms='NAT');enc,w,tf=R55.build(a,seed,read_apps={16,32,64,128},arms=['NAT']);ch=w['chunks']
    assert len(ch)==128 and all(len(c['ids'])==CH for c in ch) and all(len(c['sents'])==32 for c in ch) and w['k']==32
    its=G.build_items(w);assert len(its)==16 and all(len(i['cands'])==16 for i in its) and all(i['cand_fact']==[i['fact']]+[j for j in range(16) if j!=i['fact']] for i in its)
    placeset={' '+f['P'] for f in w['facts'] if f['id']<16};assert all(set(i['cands'])==placeset for i in its) and not any(' '+f['P'] in placeset for f in w['facts'] if f['id']>=16)
    tk=Tokenizer.from_file(str(ROOT/'data/tokenizer/llama3/tokenizer.json'));assert len({len(tk.encode(c,add_special_tokens=False).ids) for i in its for c in i['cands']})==1
    pl=R.plan(w,enc);assert set(pl['read'])=={16,32,64,128} and len(pl['items'])==16
    # brute-force check of every mask arm on several chunks
    rng=np.random.default_rng(0)
    for k in (1,5,9,40,127):
        base=(k*CH-NW*CH+np.arange(WIN+CH))>=0
        for arm in ARMS:
            kp,info=uv_for(arm,ch,lambda *x:None).keep(k,None,None,None,None,base);u1,u2=HIDE[arm];rows=np.zeros(CH,bool)
            for xq in ch[k]['sents']:
                rows[xq['start']:xq['end']]=True;hid=np.zeros(WIN+CH,bool)
                for c in range(max(0,k-NW),k):
                    for x in ch[c]['sents']:
                        f=int(x['fact']);same=(f==int(xq['fact']))
                        if same or ((not same) and f<NREAD and u1) or (f>=NREAD and u2):hid[(c-k+NW)*CH+x['start']:(c-k+NW)*CH+x['end']]=True
                got=~kp[xq['start']:xq['end']]&base[None,:];assert (got==(hid&base)[None,:]).all(),(arm,k,xq)
            assert np.array_equal(kp[~rows],np.broadcast_to(base,(int((~rows).sum()),WIN+CH))) and (kp[rows][:,WIN:]==base[None,WIN:]).all(),(arm,k)
    # nesting: SAME <= SAME_U1, SAME_U2 <= SAME_U1U2 in hidden pairs
    base=(60*CH-NW*CH+np.arange(WIN+CH))>=0;h={a:(~uv_for(a,ch,lambda *x:None).keep(60,None,None,None,None,base)[0]&base).sum() for a in ARMS};assert h['SAME']<h['SAME_U1']<h['SAME_U1U2'] and h['SAME']<h['SAME_U2']<h['SAME_U1U2'],h
    print('e59_u12 selftest ok; hidden pairs of chunk 60:',{k:int(v) for k,v in h.items()},'fact token share %.3f'%np.mean([sum(s['end']-s['start'] for s in c['sents'])/CH for c in ch]))
if __name__=='__main__':
    if '--selftest' in sys.argv:selftest()
