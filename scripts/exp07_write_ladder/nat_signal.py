"""E33 2.13 k-resolved attention-signal check on a natural-text stream (W0, no weight update, world A by default).
For every chunk of the stream, ONE attention capture at W0 (suffix layers, head-max of the layer-mean probability per target, targets = all loss positions of the chunk; candidates = the 8K window before the chunk
minus the first 4 stream positions and the last 128 tokens; ties -> earlier position), then for every k in the grid the per-target top-k union = the masked set S_k (cut from the same scores). Baselines per k: rand-dist
(same count, same 256-token distance-bin histogram, O tokens excluded) and recency (the |S_k| positions right before the chunk).
O (natural text): for every person t with a fact unit (label 'fact') in chunk k: all token positions of the person's three name forms (@X, @XF, @XS realized) and of the birthplace string P inside EARLIER units of the same
person (any label) that lie in the 8K window before the chunk and before the answer. Answer tokens (epsilon): the tokens of the first P occurrence of that fact unit; the chunk is epsilon-eligible only if O is non-empty.
Per chunk and k (all chunks): masked count, candidate count, O tokens (selectable) masked, masked tokens in background text / in expo units of the persons with a fact unit in the chunk / in other persons' expo units.
Per eligible chunk: output distance ||p - e_y|| at the answer tokens under natural, chunk-alone, oracle-O mask and, per k, policy / rand-dist / recency masks (forward outputs at W0, as E25 stage 1 / s3_stage1.eval_chunk).
Usage (GPU): nat_signal.py --nat 9103:clustered --nat-dir DIR --worlds A --ks 4,8,16,32,64,128,256 [--max-chunk N]
CPU:        nat_signal.py --dry --nat 9103:clustered --nat-dir DIR [--tokenizer-file F]  (table, O sizes, answers, window geometry; real tokenizer or the proxy)   |  --selftest"""
import os,sys,re,json,time,bisect,argparse,random
from pathlib import Path
import numpy as np
HERE=Path(__file__).resolve().parent;sys.path.insert(0,str(HERE))
import nat_world as NW
import s3_sets as SETS
CH=1024;WIN=8192;NCH=63;GRID=[4,8,16,32,64,128,256]
def occ_tokens(starts,ends,c0,c1):
    """token positions j (BOS at 0, so token i of the encoding is j=i+1) whose character range overlaps [c0,c1)"""
    i=bisect.bisect_right(ends,c0);out=[]
    while i<len(starts) and starts[i]<c1:
        out.append(i+1);i+=1
    return out
def rx_of(strings):
    ss=sorted({s for s in strings if s},key=len,reverse=True);return re.compile(r'(?<!\w)(?:'+'|'.join(re.escape(s) for s in ss)+r')(?!\w)')
def build_table(w,swap,offs,ntok=NCH*CH):
    """offs = [(start,end)] character offsets of the encoding of the realized text WITHOUT BOS. Returns (table, owner[ntok] = person id of the expo unit a token lies in, -1 = background)."""
    text,_,spans=NW.realize(w,swap);nm=NW.names(w,swap);tri={t['id']:t for t in w['triples']};starts=[a for a,b in offs];ends=[b for a,b in offs]
    owner=np.full(ntok,-1,np.int32);table=[];ordc={}
    for i,sp in enumerate(spans):
        t=sp['t'];base=sp['start'];ut=sp['text'];assert text[base:base+len(ut)]==ut
        ordc[t]=ordc.get(t,0)+1;P=tri[t]['P']
        names=[];pall=[];first=None
        for m in rx_of(nm[t]).finditer(ut):names+=occ_tokens(starts,ends,base+m.start(),base+m.end())
        for m in rx_of([P]).finditer(ut):
            tk=occ_tokens(starts,ends,base+m.start(),base+m.end());pall+=tk
            if first is None:first=tk
        for j in occ_tokens(starts,ends,base,base+len(ut)):
            if j<ntok:owner[j]=t
        lim=lambda xs:sorted({j for j in xs if j<ntok})
        table.append(dict(idx=i,t=t,pair=tri[t]['pair'],ord=ordc[t],label=sp['label'],place=lim(first or []) if sp['label']=='fact' else [],pall=lim(pall),names=lim(names)))
    return table,owner
class NatSets(SETS.Sets):
    def __init__(self,table,seed):
        self.t=table;self.seed=seed;self.passno=1;self.frac=0.25;self.fill_total=None;self.fill_attn_cands=False;self.fill_no_o=False
    def cur_sentences(self,k):return [s for s in self.t if s['place'] and k*CH<=s['place'][0]<(k+1)*CH]
    def prior_units(self,s,lo):
        return [u for u in self.t if u['t']==s['t'] and u['ord']<s['ord'] and any(lo<=p<s['place'][0] for p in u['names']+u['pall'])]
    def oracle(self,k):
        lo=max(0,k*CH-WIN);O=set()
        for s in self.cur_sentences(k):
            for u in self.t:
                if u['t']==s['t'] and u['ord']<s['ord']:O|={p for p in u['names']+u['pall'] if lo<=p<s['place'][0]}
        return sorted(O)
    def answers(self,k):
        out=[];lo=max(0,k*CH-WIN)
        for s in self.cur_sentences(k):
            if self.prior_units(s,lo):out+=[(j,s['pair']) for j in s['place'] if j-1-k*CH>=0]
        return out
def cand_mask(k):
    pos=k*CH-WIN+np.arange(WIN);return (pos>=4)&(pos<k*CH-128)&(pos>=0)
def union_sets(order,candm,ks):
    """order: [CH, WIN] key indices sorted by decreasing head-max score per target (stable, ties -> earlier); returns {k: bool[WIN]} = union over targets of the top-k candidates"""
    out={}
    for kk in ks:
        u=np.zeros(WIN,bool);u[order[:,:kk].ravel()]=True;out[kk]=u&candm
    return out
def chunk_stats(S_idx,k,candm,Oc,owner,cur_t):
    """S_idx: absolute masked positions (< k0 window part). Counts for the record."""
    S=np.asarray(S_idx,int);n=len(S);own=owner[S] if n else np.zeros(0,int)
    Ss=set(int(x) for x in S);tp=len(Ss&Oc)
    return dict(n=int(n),tp=tp,bg=int((own<0).sum()),same=int(np.isin(own,list(cur_t)).sum()) if n and cur_t else 0,other=int(((own>=0)&~np.isin(own,list(cur_t))).sum()) if n else 0)
def dry(w,swap,offs):
    table,owner=build_table(w,swap,offs);sets=NatSets(table,0);rows=[]
    for k in range(NCH):
        O=sets.oracle(k);an=sets.answers(k);cs=sets.cur_sentences(k)
        if cs:rows.append(dict(k=k,n_fact_units=len(cs),n_O=len(O),n_answer_tokens=len(an),n_O_cand=len([p for p in O if k*CH-WIN<=p<k*CH and cand_mask(k)[p-(k*CH-WIN)]]),elig=bool(an)))
    return table,owner,rows
def selftest():
    # synthetic world: two persons, units placed by character offsets; one token per character group for the test
    text_parts=[];w={'triples':[{'id':0,'pair':'p0','X':'Ann Bell','XF':'Ann','XS':'Bell','P':'Oslo'},{'id':1,'pair':'p0','X':'Cy Dunn','XF':'Cy','XS':'Dunn','P':'Rome'}],'texts':[],'pairs':1}
    def bg(n):return {'kind':'bg','text':' '.join(f'bg{i}' for i in range(n))}
    for i in range(14):
        w['texts'].append(bg(900))
        if i in (1,5,12):w['texts'].append({'kind':'expo','t':0,'round':i,'label':'fact' if i!=1 else 'name_only','text':'@X0@ lived. @XF0@ was born in Oslo today. Then Bell left.'})
        if i==7:w['texts'].append({'kind':'expo','t':1,'round':1,'label':'fact','text':'@X1@ was born in Rome.'})
    text,_,spans=NW.realize(w,False);offs=[(m.start(),m.end()) for m in re.finditer(r'\S+',text)]
    table,owner=build_table(w,False,offs,ntok=len(offs)+1)
    assert [u['ord'] for u in table if u['t']==0]==[1,2,3] and table[0]['place']==[] and table[1]['place'] and table[2]['place']
    # token positions: names of unit 0 include Ann, Bell (X form and XF/XS); pall includes Oslo
    u1=table[1];assert len(u1['pall'])==1 and len(u1['names'])>=3,u1
    sets=NatSets(table,0)
    # the fact unit of person 0 (3rd occurrence) has priors; person 1 has none
    ks_=[k for k in range(NCH) if sets.cur_sentences(k)]
    assert owner[u1['pall'][0]]==0 and (owner[:len(offs)]>=0).sum()>0
    anyO=[k for k in range(NCH) if sets.oracle(k)];print('chunks with fact units',ks_,'with O',anyO)
    # union_sets / stats on random scores
    rng=np.random.default_rng(0);M=rng.random((CH,WIN)).astype(np.float32);k=9;cm=cand_mask(k);Mw=M.copy();Mw[:,~cm]=0;order=np.argsort(-Mw,axis=1,kind='stable');U=union_sets(order,cm,[4,16,64])
    assert U[4].sum()<U[16].sum()<U[64].sum() and not (U[64]&~cm).any() and (U[4]<=U[16]).all()
    st=chunk_stats(k*CH-WIN+np.nonzero(U[16])[0],k,cm,set(),np.full(100000,-1),set());assert st['n']==U[16].sum() and st['bg']==st['n']
    print('nat_signal SELFTEST OK',{kk:int(v.sum()) for kk,v in U.items()})
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--nat',default='9103:clustered');ap.add_argument('--nat-dir',default=os.environ.get('NAT_DIR','data/natural-text/worlds'));ap.add_argument('--worlds',default='A')
    ap.add_argument('--ks',default=','.join(map(str,GRID)));ap.add_argument('--max-chunk',type=int,default=None);ap.add_argument('--dry',action='store_true');ap.add_argument('--selftest',action='store_true');ap.add_argument('--tokenizer-file',default=None);ap.add_argument('--tag',default='sig');ap.add_argument('--require-real',action='store_true');ap.add_argument('--save-topk',type=int,default=0,help='also save the per-target top-K candidate indices of every chunk (uint16 [CH,K] per chunk, K <= 256) = the raw W0 scores in a form that reproduces the union for any k <= K offline')
    a=ap.parse_args()
    if a.selftest:selftest();return
    stream,layout=a.nat.split(':');stream=int(stream)
    if a.dry:
        import nat_dryrun as D
        w=NW.load_world(Path(a.nat_dir)/f'nat_{stream}_{layout}.json')
        try:tk=D.load_tok(a.tokenizer_file)
        except Exception as e:print('real tokenizer not usable:',repr(e)[:100]);tk=None
        if tk is not None:
            e=tk.encode(NW.realize(w,False)[0],add_special_tokens=False);offs=e.offsets;kind='REAL tokenizer'
        else:
            px=D.Proxy();_,ends=px.encode(NW.realize(w,False)[0]);starts=[0]+ends[:-1];offs=list(zip(starts,ends));kind='PROXY tokenizer (geometry only)'
        assert tk is not None or not a.require_real,'--require-real: the real tokenizer could not be loaded'
        table,owner,rows=dry(w,False,offs);print('tokenizer:',kind,'; expo units',len(table),'; stream tokens',len(offs)+1)
        for r in rows:print(r)
        print('chunks with fact units',len(rows),'; eligible (answers)',sum(r['elig'] for r in rows),'; answer tokens',sum(r['n_answer_tokens'] for r in rows),'; mean |O| over chunks with O',np.mean([r['n_O'] for r in rows if r['n_O']] or [0]));return
    run_gpu(a,stream,layout)
def run_gpu(a,stream,layout):
    import nat_ladder as NL
    from ladder import Runtime,OUT,clone_state,jnp,eqx,stage2
    import s3_stage1 as ST
    from s3 import SWA  # noqa
    assert 'deterministic_ops=true' in os.environ.get('XLA_FLAGS','')
    ks=[int(x) for x in a.ks.split(',')];NL.install(a.nat_dir,stream,layout);OUT.mkdir(parents=True,exist_ok=True)
    class NatStage1(ST.Stage1):
        def signal(self,swap,ks,seed=0,max_chunk=None,save_topk=0,topk_path=None):
            rt=self.rt;m=rt.base;P=self.T.P[swap];ids=np.asarray(P.ids)[:NCH*CH];mask=np.zeros(NCH*CH,dtype=bool);mask[:len(ids)-1]=True;seq=rt.seq(ids,mask=mask)
            text=NW.realize(self.T.w,swap)[0];e=stage2.enc(self.T.tok,text);offs=list(e.offsets);table,owner=build_table(self.T.w,swap,offs);assert [i for i in ids[1:6]]==list(e.ids[:5]),'stream ids differ from the table encoding'
            sets=NatSets(table,seed);res={'swap':bool(swap),'stream':stream,'layout':layout,'ks':ks,'chunks':{}};W=WIN;topk={}
            seg0=np.zeros(CH,np.int32);loc0=np.arange(CH,dtype=np.int32);pm0=jnp.asarray(ST.packing.prefix_mask(seg0,W))
            with rt.mesh:
                pref_nat=rt.prefix(m,seq);Kh,Vh,_=self.record(ids);st=clone_state(rt.state_suffix);t0=time.monotonic()
                for k in range(NCH if max_chunk is None else max_chunk):
                    k0=k*CH;c=rt.chunk(seq,k);pref_c=pref_nat[k*CH:(k+1)*CH];candm=cand_mask(k)
                    A,Bl,M=self.capture(m,st,c,pref_c);Mw=np.asarray(M)[:,:W].copy();Mw[:,~candm]=0.0;order=np.argsort(-Mw,axis=1,kind='stable');U=union_sets(order,candm,ks)
                    if save_topk:topk[k]=order[:,:save_topk].astype(np.uint16)
                    O=sets.oracle(k);Oset=set(O);Oc=set(p for p in O if k0-W<=p<k0 and candm[p-(k0-W)]);cur=sets.cur_sentences(k);cur_t={s['t'] for s in cur};ans=sets.answers(k)
                    rec={'k':k,'n_cand':int(candm.sum()),'n_O':len(O),'n_Oc':len(Oc),'n_fact_units':len(cur),'n_ans':len(ans),'per_k':{}}
                    rows=jnp.asarray([j-1-k0 for j,_ in ans],dtype=jnp.int32) if ans else None;ys=np.asarray([ids[j] for j,_ in ans]) if ans else None
                    def dist(p):
                        p=np.asarray(p);py=p[np.arange(len(ys)),ys];return np.sqrt(np.maximum((p**2).sum(1)-2*py+1,0)).tolist()
                    if ans:
                        Kc,Vc=self.ctx_kv(Kh,Vh,k)
                        def forward(S):
                            keep=sets.keep_vector(k,list(S));pw=self.prefix_masked(m,jnp.asarray(ids[k0:k0+CH],dtype=jnp.int32),Kc,Vc,jnp.int32(k0),jnp.asarray(keep));return dist(self.logits_masked(m,st,c,pw,jnp.asarray(keep),rows))
                        rec['ans']=[(int(j),p) for j,p in ans];rec['nat']=dist(self.logits_nat(m,st,c,pref_c,rows))
                        tg=np.concatenate([ids[k0+1:k0+CH+1],[rt.eos]])[:CH].astype(np.int32);pfa=self.prefix_iso(m,jnp.asarray(ids[k0:k0+CH]),jnp.asarray(loc0),pm0)
                        rec['alone']=dist(self.logits_alone(m,jnp.asarray(ids[k0:k0+CH]),jnp.asarray(tg),jnp.asarray(np.concatenate([np.ones(CH-1,bool),[False]])),pfa,jnp.asarray(seg0),jnp.asarray(loc0),rows))
                        rec['oracle']=forward(O)
                    rng=np.random.default_rng([seed,k,int(swap)])
                    for kk in ks:
                        S=k0-W+np.nonzero(U[kk])[0];d=chunk_stats(S,k,candm,Oc,owner,cur_t);n=len(S)
                        if ans and n:
                            d['pol']=forward(S);d['rand']=forward(self.rand_like(sets,k,S,Oset,rng));d['rec']=forward(np.arange(max(0,k0-n),k0))
                        rec['per_k'][kk]=d
                    res['chunks'][k]=rec
                    if k%8==0:print(f'  signal {"B" if swap else "A"} chunk {k} {time.monotonic()-t0:.0f}s answers {len(ans)}',flush=True)
                    st=self.read_state(m,st,c,pref_c)
            if save_topk:np.savez_compressed(topk_path,**{f'k{k_:02d}':v for k_,v in topk.items()})
            return res
    rt=Runtime();L=NatStage1(rt,stream,readout=False);L.build3();L.build1()
    for wd in a.worlds:
        f=OUT/f'nat_signal_{a.tag}_s{stream}_{layout}_{wd}.json'
        if f.exists():print('exists',f);continue
        res=L.signal(wd=='B',ks,max_chunk=a.max_chunk,save_topk=a.save_topk,topk_path=OUT/f'nat_signal_{a.tag}_s{stream}_{layout}_{wd}_topk.npz');json.dump(res,open(f,'w'));print('saved',f,flush=True)
if __name__=='__main__':main()
