"""E57 window gain of the ANSWER tokens by appearance order of the fact (CPU only; new file; numbers only). Facts = the 42 questions of the frozen bank (bank_v1.json); an appearance of a fact = a regex hit (word-boundary, case-insensitive) of any surface form of its answer (canonical + variants + fixed_surface, as e57_occurrence_count.py) inside an article of the question's own event, in stream order. [The 178-fact list e57_freeze/bank.json belongs to another article set: none of its 858 article ids is in the 871 articles of the stream, so it cannot be located in the stream.] The answer tokens of an appearance are the stream tokens overlapping the hit; the loss row of token p is (chunk p // 1024, row p % 1024 - 1). Per state (E57_windowless_<LABEL>.npz: nll_win, nll_chunk) and per class the mean of  gain = baseline NLL - window NLL  over the answer tokens in chunks 8..279 with a loss, with baseline = no-window NLL (--kind nowin) or the band-b NLL (--kind band --band b with E57_band_<LABEL>.npz of job 345 / 348). Classes: FIRST (first appearance of the fact in the stream), SECOND, THIRD+ ; for appearances >= 2 also by the distance to the previous appearance of the same fact: P512 (<= 512 tokens), P8192 (<= 8192, > 512), POUT (> 8192). Reported per class: appearances, tokens, mean gain, W0 mean gain on the same tokens, ratio, excess over the FIRST class of the same state, 95 % chunk-bootstrap intervals of the excess and of the ratio; plus the distribution of the number of appearances per fact and the number of facts without a hit.
  .venv/bin/python scripts/exp07_write_ladder/e57_fact_gain.py --bundle B --states-dir D --labels W0,NAT,... --kind nowin|band [--band 512] --out O.json | --selftest"""
import sys,json,re,pathlib,argparse,collections
import numpy as np
HERE=pathlib.Path(__file__).resolve().parent
for p in (HERE,HERE/'e57',HERE.parents[1]/'scripts/exp04_what_ttt_retains',HERE.parents[1]/'scripts/exp05_write_dynamics'):sys.path.insert(0,str(p))
CH=1024
def forms(q):
    vs=[]
    for v in [q['canonical']]+list(q.get('variants',[]))+[q.get('fixed_surface','')]:
        if v and v.lower() not in {x.lower() for x in vs}:vs.append(v)
    return vs
def char_to_tokens(strs,a,b):
    """token indices (relative to the first token whose cumulative text is in strs) overlapping the character span [a,b)"""
    return None
def appearances(text,offs,rx):
    """text of an article, offs[i] = char offset of token i (len n+1) -> list of (first token, last token) of each hit"""
    out=[]
    for m in rx.finditer(text):
        a,b=m.span();i0=int(np.searchsorted(offs,a,side='right')-1);i1=int(np.searchsorted(offs,b,side='left')-1);out.append((i0,max(i0,i1)))
    return out
def selftest():
    strs=['The',' cat',' sat',' on',' Rajya',' Sabha','.'];offs=np.cumsum([0]+[len(s) for s in strs]);text=''.join(strs)
    rx=re.compile(r'(?<![\w])Rajya Sabha(?![\w])',re.I);a=appearances(text,offs,rx);assert a==[(4,5)],a
    rx2=re.compile(r'(?<![\w])cat(?![\w])',re.I);assert appearances(text,offs,rx2)==[(1,1)]
    print('e57_fact_gain selftest ok')
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--bundle');ap.add_argument('--states-dir');ap.add_argument('--labels');ap.add_argument('--kind',default='nowin');ap.add_argument('--band',type=int,default=512);ap.add_argument('--out');ap.add_argument('--tokenizer');ap.add_argument('--selftest',action='store_true');a=ap.parse_args()
    if a.selftest:selftest();return
    import gen_world_ctl as G,e55_uv as UVm,e57_reads as RD
    tfile=a.tokenizer or str(HERE.parents[1]/'data/tokenizer/llama3/tokenizer.json');enc=G.load_enc(tfile);tk=UVm.tokenizer(tfile)
    items=RD.load_items(pathlib.Path(a.bundle)/'e57_bank'/'bank_v1.json');ids,iv,hits=RD.article_intervals(a.bundle,enc,items);ids=np.asarray(ids)
    strs=[tk.decode([int(t)]) for t in ids];offs=np.cumsum([0]+[len(s) for s in strs]);full=''.join(strs)
    facts=[]
    for it in items:
        q=it['q'];vs=forms(q);rx=re.compile('|'.join(r'(?<![\w])'+re.escape(v)+r'(?![\w])' for v in vs),re.I);ap_=[]
        for (s0,e0,ev,rid) in iv:
            e0=min(e0,len(ids))
            if s0>=len(ids):continue
            if ev!=it['event']:continue
            for (i0,i1) in appearances(full[offs[s0]:offs[e0]],offs[s0:e0+1]-offs[s0],rx):ap_.append((s0+i0,s0+i1))
        ap_.sort();facts.append(dict(id=it['id'],event=it['event'],app=ap_))
    cnt=collections.Counter(min(len(f['app']),12) for f in facts);nofound=sum(1 for f in facts if not f['app'])
    # observations: token position, fact, appearance order, class of previous distance
    obs=[]
    for fi,f in enumerate(facts):
        prev=None
        for j,(i0,i1) in enumerate(f['app']):
            order='FIRST' if j==0 else ('SECOND' if j==1 else 'THIRD+');d=None if prev is None else i0-prev
            pc=None if j==0 else ('P512' if d<=512 else ('P8192' if d<=8192 else 'POUT'))
            for p in range(i0,i1+1):obs.append((p,fi,j,order,pc))
            prev=i0
    labels=a.labels.split(',');P=np.array([o[0] for o in obs]);K=P//CH;R=(P%CH)-1;K=np.where(R<0,K-1,K);R=np.where(R<0,CH-1,R)
    ok=(K>=8)&(K<=279)
    base={}
    def gain(label):
        z=np.load(pathlib.Path(a.states_dir)/(f'E57_windowless_{label}.npz' if a.kind=='nowin' else f'E57_band_{label}.npz'))
        if a.kind=='nowin':b=z['nll_chunk']
        else:b=z['nll_band'][list(z['bands']).index(a.band)]
        g=(b-z['nll_win'])[K.clip(0,279),R];return g,z['lm'][K.clip(0,279),R]
    G_={l:gain(l) for l in labels};m_ok=ok&G_[labels[0]][1];orders=['FIRST','SECOND','THIRD+'];rng=np.random.default_rng(0);res={}
    classes={'FIRST':lambda o:o[3]=='FIRST','SECOND':lambda o:o[3]=='SECOND','THIRD+':lambda o:o[3]=='THIRD+','SECOND+P512':lambda o:o[3]=='SECOND' and o[4]=='P512','SECOND+P8192':lambda o:o[3]=='SECOND' and o[4]=='P8192','SECOND+POUT':lambda o:o[3]=='SECOND' and o[4]=='POUT','THIRD+P512':lambda o:o[3]=='THIRD+' and o[4]=='P512','THIRD+P8192':lambda o:o[3]=='THIRD+' and o[4]=='P8192','THIRD+POUT':lambda o:o[3]=='THIRD+' and o[4]=='POUT'}
    masks={c:np.array([f(o) for o in obs])&m_ok for c,f in classes.items()};chunks=np.unique(K[m_ok])
    def stat(label,cm,ch):
        s=np.isin(K,ch)&cm;g=G_[label][0][s];return (g.sum(),int(s.sum()))
    for l in labels:
        res[l]={}
        for c,cm in masks.items():
            s,n=stat(l,cm,chunks);s0,_=stat(labels[0],cm,chunks);res[l][c]=dict(appearances=int(len({(o[1],o[2]) for o,mm in zip(obs,cm) if mm})),tokens=n,mean_gain=s/n if n else None,mean_gain_W0=s0/n if n else None,ratio=(s/s0 if s0 else None))
        for c in ('SECOND','THIRD+'):
            b=[]
            for _ in range(500):
                ch=rng.choice(chunks,len(chunks));sf,nf=stat(l,masks['FIRST'],ch);sc,nc=stat(l,masks[c],ch);s0f,_=stat(labels[0],masks['FIRST'],ch);s0c,_=stat(labels[0],masks[c],ch)
                if nf and nc:b.append(((sc/nc-sf/nf),(sc/nc-sf/nf)/((s0c/nc-s0f/nf) if (s0c/nc-s0f/nf)!=0 else np.nan)))
            b=np.array(b);ex=res[l][c]['mean_gain']-res[l]['FIRST']['mean_gain'] if res[l][c]['tokens'] and res[l]['FIRST']['tokens'] else None
            ex0=res[l][c]['mean_gain_W0']-res[l]['FIRST']['mean_gain_W0'] if ex is not None else None
            res[l][c].update(excess_over_first=ex,excess_W0=ex0,excess_ratio_to_W0=(ex/ex0 if ex0 else None),excess_ci95=[float(np.nanpercentile(b[:,0],2.5)),float(np.nanpercentile(b[:,0],97.5))] if len(b) else None,excess_ratio_ci95=[float(np.nanpercentile(b[:,1],2.5)),float(np.nanpercentile(b[:,1],97.5))] if len(b) else None)
    out=dict(kind=a.kind,band=a.band if a.kind=='band' else None,facts=len(facts),facts_without_hit=nofound,appearances_per_fact_distribution={str(k):v for k,v in sorted(cnt.items())},answer_tokens_total=len(obs),answer_tokens_used=int(m_ok.sum()),states=res)
    json.dump(out,open(a.out,'w'),indent=1,default=float)
    print('facts',len(facts),'without hit',nofound,'appearances/fact',dict(sorted(cnt.items())),'answer tokens',len(obs),'used',int(m_ok.sum()))
    for l in labels:
        print('==',l)
        for c in ('FIRST','SECOND','THIRD+','SECOND+P512','SECOND+P8192','SECOND+POUT','THIRD+P512','THIRD+P8192','THIRD+POUT'):
            v=res[l][c];print(f"  {c:13s} app {v['appearances']:4d} tok {v['tokens']:5d} gain {v['mean_gain'] if v['mean_gain'] is None else round(v['mean_gain'],3)} W0 {v['mean_gain_W0'] if v['mean_gain_W0'] is None else round(v['mean_gain_W0'],3)}"+(f" excess {v['excess_over_first']:+.3f} (W0 {v['excess_W0']:+.3f}) ratio {v['excess_ratio_to_W0'] if v['excess_ratio_to_W0'] is None else round(v['excess_ratio_to_W0'],3)}" if 'excess_over_first' in v and v['excess_over_first'] is not None else ''))
if __name__=='__main__':main()
