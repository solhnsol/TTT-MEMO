"""E57 signal bottleneck: how much of the answer-bearing rows does the window-gain signal catch, and what do other signals / a combination add (new file; CPU only; existing files only; GPM thread, Hans 11/08 '1번 2번 둘 다 병렬로'). Sources: news stream A (labels of e57_gain_labels_rows.npz: A = target token inside an answer string of the 42 bank questions, SENT/S = sentence or subject of a question of the same event; here 'answer sentence' = rows of sentences that hold an A hit = label A or SENT; chunks 8..279) and the QA 65-version streams (bundles 5/8/9/10/12; A = target token inside a hit of the gold string of a bundle question (token-subsequence search of ' '+gold and gold in the stream ids), answer sentence = all rows of the sentences holding an A hit; chunks >= 8). Row i of chunk k predicts the token k*1024+i+1; eligible rows as in the other analyses. Signals per row: g128 (W0), g' (suffix band, jobs 1211), nll_win, token type of the target, and for news the attention far-window mass / far-mass entropy of the prefix layers (attdump3: ALL128_far, ALL512_far, ALL1024_far, ALL128_ent, SH128_far, win_ALL; output-logit entropy is not stored on the server). (1) recall / precision / lift of each signal at the same selection share (5 / 10 / 15 / 30 % rows per chunk and the ABS2 count of the chunk), (2) which A rows the top-30 % by g misses (token type, position inside the answer), (3) logistic regression on the signals, separate training / evaluation sources (QA tune -> QA hold, news tune events -> news hold, QA -> news, news -> QA; news with attention features), recall at the same shares versus g128 alone. Chunk-level bootstrap (2000 draws), 95% CI.
  CUDA_VISIBLE_DEVICES= JAX_PLATFORMS=cpu python3 e57_signal_bottleneck_cpu.py --out F.json | --selftest"""
import sys,json,re,argparse,pathlib
import numpy as np
from scipy.optimize import minimize
HERE=pathlib.Path(__file__).resolve().parent;ROOT=HERE.parents[1];sys.path.insert(0,str(HERE));sys.path.insert(0,str(ROOT/'scripts/exp04_what_ttt_retains'))
EXP=ROOT/'experiments/exp07_write_ladder';CH=1024;K0=8;NB=2000;SHARES=(5,10,15,30)
import e57_closure_masks as CM
TYPES=['digit','cap_word','cap_piece','lower_word','lower_piece','punct','other']
def tok_type(s):
    if s is None:return 'other'
    sp=s.startswith('Ġ');b=s[1:] if sp else s
    if not b:return 'other'
    if any(c.isdigit() for c in b):return 'digit'
    if b[0].isalpha():return ('cap_word' if sp else 'cap_piece') if b[0].isupper() else ('lower_word' if sp else 'lower_piece')
    return 'punct' if not b[0].isspace() and not b.startswith('Ċ') else 'other'
def type_arrays(tokenizer,tokids):
    cache={};out=np.zeros(len(tokids),np.int8)
    for i,t in enumerate(tokids):
        t=int(t)
        if t not in cache:cache[t]=TYPES.index(tok_type(tokenizer.id_to_token(t)))
        out[i]=cache[t]
    return out
def sentence_ids(UVm,ids,tokf):
    ends=np.asarray(UVm.sentence_ends(UVm.tokenizer(tokf),ids),bool);return (np.cumsum(ends)-ends).astype(np.int64)
def sent_label(a_tok,sid):
    """rows whose own target token is in a sentence that holds an A token"""
    bad=np.unique(sid[a_tok&(sid>=0)]);return np.isin(sid,bad)&(sid>=0)
# ------------------------------------------------ sources
def news_source():
    import e55_uv as UVm,e57_t1t2v2_run as V
    path=EXP/'e57_t1t2v2_in/e57_stream_sentences.jsonl';tokf=str(ROOT/'data/tokenizer/llama3/tokenizer.json');ids=np.asarray(V.stream_ids(str(path),'A'));Z=np.load(EXP/'e57_band/e57_band_W0.npz');Y=np.load(EXP/'e57_suffixband/news/E57_suffixband_W0.npz');b=list(Z['bands']).index(128)
    L=np.load(EXP/'e57_gain_labels_rows.npz');lab_tok=np.full(len(ids),-1,np.int8);lab_tok[L['pos']]=L['label']
    sp=json.load(open(EXP/'e57_news_split.json'));evs=sorted(set(sp['events_tune'])|set(sp['events_hold']));ev=np.full(len(ids),-1,int)
    for l in open(path):
        x=json.loads(l)
        if x.get('type')=='frag':ev[x['tok0']:x['tok0']+x['ntok']]=evs.index(x['event'])
    ends=UVm.sentence_ends(UVm.tokenizer(tokf),ids)
    from tokenizers import Tokenizer
    tk=Tokenizer.from_file(tokf);man=json.load(open(EXP/'e57_attdump3/e57_attdump3_manifest.json'));shards=man['chunk_to_shard'];files={}
    def att(k,name):
        s=shards[str(k)]
        if s not in files:files[s]=np.load(EXP/'e57_attdump3'/man['shards'][str(s)]['file'])
        return files[s][f'W0/c{k}/{name}']
    cols={n:[] for n in ('ck','i','g','gp','nllwin','tt','A','SENTlab','ev','far128','far512','far1024','ent128','sh128','win')};
    for k in range(K0,len(ids)//CH):
        q=UVm.chunk_sents(ends,k,CH);sid=np.full(CH,-1)
        for i,(s_,e_) in enumerate(q):sid[s_:e_]=i
        lm=Z['lm'][k];ix=np.nonzero(CM.rows_eligible(lm,sid)&lm)[0];p=k*CH+ix+1
        cols['ck'].append(np.full(len(ix),k));cols['i'].append(ix);cols['g'].append((Z['nll_band'][b][k]-Z['nll_win'][k])[ix]);cols['gp'].append((Y['nll_suffixband'][0][k]-Z['nll_win'][k])[ix]);cols['nllwin'].append(Z['nll_win'][k][ix]);cols['tt'].append(type_arrays(tk,ids[p]))
        lt=lab_tok[p];cols['A'].append(lt==3);cols['SENTlab'].append((lt==3)|(lt==1));cols['ev'].append(ev[k*CH+ix])
        for nm,key in (('far128','ALL128_far'),('far512','ALL512_far'),('far1024','ALL1024_far'),('ent128','ALL128_ent'),('sh128','SH128_far'),('win','win_ALL')):cols[nm].append(att(k,key)[ix])
    r={k:np.concatenate(v) for k,v in cols.items()};r['part']=np.where(np.isin(r['ev'],[evs.index(x) for x in sp['events_tune']]),0,1);r['first_in_answer']=None;return r
def qa_source():
    import e55_uv as UVm,e57_pqa2_stream as S2
    tk=S2.tokenizer();tokf=str(ROOT/'data/tokenizer/llama3/tokenizer.json');q=json.load(open(EXP/'e57_public_qa/e57_public_qa_squad_items.json'));byid={x['id']:x for x in q['items']};m=json.load(open(EXP/'e57_public_qa_n65/e57_pqa6_groups.json'));zz=np.load(EXP/'e57_public_qa_n65/e57_pqa6_group_ids.npz')
    sp={r['item_id']:r['split'] for r in json.load(open(EXP/'e57_qa_split.json'))['items']};N=np.load(EXP/'e57_public_qa_n65/e57_dwg_qa_nll.npz');bi=list(N['bands']).index(128);Y=np.load(EXP/'e57_suffixband/qa/e57_dwg_qa_nllsuf.npz')
    from tokenizers import Tokenizer
    tkz=Tokenizer.from_file(tokf);cols={n:[] for n in ('ck','i','g','gp','nllwin','tt','A','SENTlab','bundle','part')};cov={}
    for gr in m['groups']:
        gid=gr['gid']
        if gid not in (5,8,9,10,12):continue
        ids=np.asarray(zz[f'g{gid}']);A=np.zeros(len(ids)+1,bool);hits=0;nit=0
        for iid in gr['main_ids']:
            it=byid.get(iid)
            if it is None:continue
            nit+=1;cands=[tk.encode(' '+it['gold'],add_special_tokens=False).ids,tk.encode(it['gold'],add_special_tokens=False).ids]
            for c in cands:
                n=len(c)
                if n==0:continue
                a0=np.flatnonzero(ids[:len(ids)-n+1]==c[0])
                for s in a0:
                    if np.array_equal(ids[s:s+n],c):A[s:s+n]=True;hits+=1
        sid=sentence_ids(UVm,ids,tokf);Asent=np.zeros(len(ids)+1,bool);Asent[:len(ids)]=sent_label(A[:len(ids)],sid);cov[gid]=dict(items=nit,hits=hits,frac_tokens_A=float(A.mean()),frac_tokens_sentlab=float(Asent.mean()))
        win=N[f'g{gid}_win'];
        for k in range(K0,win.shape[0]):
            lm=N[f'g{gid}_lm'][k];ix=np.nonzero(CM.rows_eligible(lm,np.zeros(CH,np.int32))&lm)[0]
            if len(ix)<50:continue
            p=k*CH+ix+1;cols['ck'].append(np.full(len(ix),k+1000*gid));cols['i'].append(ix);cols['g'].append((N[f'g{gid}_band'][bi][k]-win[k])[ix]);cols['gp'].append((Y[f'g{gid}_suf'][0][k]-win[k])[ix]);cols['nllwin'].append(win[k][ix]);cols['tt'].append(type_arrays(tkz,ids[np.minimum(p,len(ids)-1)]));cols['A'].append(A[p]);cols['SENTlab'].append(Asent[p]);cols['bundle'].append(np.full(len(ix),gid));cols['part'].append(np.full(len(ix),0 if gid in (8,10,12) else 1))
    r={k:np.concatenate(v) for k,v in cols.items()};r['coverage']=cov;return r
# ------------------------------------------------ analysis
def groups(ck):
    o=np.argsort(ck,kind='stable');s=ck[o];b=np.flatnonzero(np.diff(s))+1;return [o[i:j] for i,j in zip(np.r_[0,b],np.r_[b,len(s)])]
def ci(x):x=np.asarray(x,float);return [float(np.nanmean(x)),float(np.nanpercentile(x,2.5)),float(np.nanpercentile(x,97.5))]
def select_stats(score,y,g,grp,rng,mask=None,shares=SHARES):
    """per chunk selection: share p % of the evaluated rows and the ABS2 count (g > 2); micro recall / precision / lift with a chunk bootstrap"""
    ch=[ix[mask[ix]] if mask is not None else ix for ix in grp];ch=[ix for ix in ch if len(ix)>=50]
    res={}
    H={};S={};Y={}
    for nm in [f'p{p}' for p in shares]+['abs2']:H[nm]=[];S[nm]=[]
    Y=[]
    for ix in ch:
        n=len(ix);Y.append(int(y[ix].sum()));o=np.argsort(-score[ix],kind='stable')
        for p in shares:
            k=int(np.ceil(p/100*n));H[f'p{p}'].append(int(y[ix][o[:k]].sum()));S[f'p{p}'].append(k)
        k=int((g[ix]>2).sum());H['abs2'].append(int(y[ix][o[:k]].sum()));S['abs2'].append(k)
    Y=np.array(Y,float);nrow=np.array([len(ix) for ix in ch],float);idx=rng.integers(0,len(ch),(NB,len(ch)));base=Y/nrow
    for nm in H:
        h=np.array(H[nm],float);s=np.array(S[nm],float);rec=h[idx].sum(1)/np.maximum(Y[idx].sum(1),1);prec=h[idx].sum(1)/np.maximum(s[idx].sum(1),1);b=Y[idx].sum(1)/nrow[idx].sum(1)
        res[nm]=dict(recall=ci(rec),precision=ci(prec),lift=ci(prec/np.maximum(b,1e-9)),mean_selected_share=float(s.sum()/nrow.sum()))
    res['label_rate']=float(Y.sum()/nrow.sum());res['n_chunks']=len(ch);res['n_label_rows']=int(Y.sum());return res
def fit_logreg(X,y,lam=1e-3,maxiter=300):
    n,d=X.shape;y=y.astype(np.float64)
    def f(w):
        z=X@w[:-1]+w[-1];l=np.mean(np.logaddexp(0,z)-y*z)+0.5*lam*np.sum(w[:-1]**2);p=1/(1+np.exp(-z));return l,np.concatenate([X.T@(p-y)/n+lam*w[:-1],[np.mean(p-y)]])
    return minimize(f,np.zeros(d+1),jac=True,method='L-BFGS-B',options=dict(maxiter=maxiter)).x
def features(S,att=False,types=True):
    cols=[S['g'],S['gp'],S['nllwin']]
    if types:cols+=[(S['tt']==i).astype(np.float64) for i in range(len(TYPES)-1)]
    if att:cols+=[S[k] for k in ('far128','far512','far1024','ent128','sh128','win')]
    return np.stack(cols,1)
def sub(S,m):return {k:(v[m] if isinstance(v,np.ndarray) and len(v)==len(m) else v) for k,v in S.items()}
def analyze_source(name,S,rng,lg):
    out={}
    for lab,key in (('A_token','A'),('answer_sentence','SENTlab')):
        y=S[key];grp=groups(S['ck']);sigs={'g128':S['g'],'gprime':S['gp'],'nll_win':S['nllwin'],'random':rng.random(len(y))}
        if 'far128' in S:sigs.update({'att_far128':S['far128'],'att_far512':S['far512'],'att_far1024':S['far1024'],'att_ent128':S['ent128']})
        out[lab]={'signals':{nm:select_stats(sc,y,S['g'],grp,rng) for nm,sc in sigs.items()}}
        lg(f'  {name} {lab}: base rate {y.mean():.4f}')
    # misses of g: A tokens outside the top-30% by g
    grp=groups(S['ck']);miss=np.zeros(len(S['g']),bool)
    for ix in grp:
        n=len(ix);k=int(np.ceil(.30*n));o=np.argsort(-S['g'][ix],kind='stable');inn=np.zeros(n,bool);inn[o[:k]]=True;miss[ix]=~inn
    A=S['A'];tt=S['tt']
    def dist(m):
        c=np.bincount(tt[m],minlength=len(TYPES));return {TYPES[i]:float(c[i]/max(c.sum(),1)) for i in range(len(TYPES))}
    out['misses_of_g_top30']={'n_A_rows':int(A.sum()),'recall_g_top30':float((A&~miss).sum()/max(A.sum(),1)),'token_types_missed_A':dist(A&miss),'token_types_caught_A':dist(A&~miss),'token_types_all_rows':dist(np.ones(len(A),bool)),'mean_g_missed_A':float(S['g'][A&miss].mean()) if (A&miss).any() else None,'mean_g_caught_A':float(S['g'][A&~miss].mean()) if (A&~miss).any() else None,'mean_nllwin_missed_A':float(S['nllwin'][A&miss].mean()) if (A&miss).any() else None,'mean_nllwin_caught_A':float(S['nllwin'][A&~miss].mean()) if (A&~miss).any() else None}
    return out
def combo(train,test,key,att,rng,lg,name):
    Xt=features(train,att);mu=Xt.mean(0);sd=Xt.std(0)+1e-9;w=fit_logreg((Xt-mu)/sd,train[key]);Xe=(features(test,att)-mu)/sd;sc=Xe@w[:-1]+w[-1];grp=groups(test['ck'])
    r={'combined':select_stats(sc,test[key],test['g'],grp,rng),'g128_alone':select_stats(test['g'],test[key],test['g'],grp,rng),'weights_standardized':{n:float(x) for n,x in zip(['g','gp','nllwin']+TYPES[:-1]+(['far128','far512','far1024','ent128','sh128','win'] if att else []),w[:-1])}};lg(f'  combo {name} {key}');return r
def selftest():
    assert tok_type('ĠHello')=='cap_word' and tok_type('1999')=='digit' and tok_type('ing')=='lower_piece' and tok_type(',')=='punct'
    r=np.random.default_rng(0);g=r.standard_normal(2000);y=g>1.5;ck=np.repeat(np.arange(20),100);s=select_stats(g,y,g,groups(ck),r);assert s['p10']['recall'][0]>0.5;print('e57_signal_bottleneck_cpu selftest ok')
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out');ap.add_argument('--selftest',action='store_true');a=ap.parse_args()
    if a.selftest:selftest();return
    lg=lambda *x:print(*x,flush=True);rng=np.random.default_rng(0);res={'definitions':__doc__.split('\n')[0]}
    Nw=news_source();lg('news rows',len(Nw['g']));Q=qa_source();lg('qa rows',len(Q['g']),Q['coverage']);res['qa_label_coverage']=Q['coverage']
    res['news_A']=analyze_source('news',Nw,rng,lg);res['qa_all']=analyze_source('qa',Q,rng,lg);res['qa_hold']=analyze_source('qa_hold',sub(Q,Q['part']==1),rng,lg);res['news_hold']=analyze_source('news_hold',sub(Nw,Nw['part']==1),rng,lg)
    res['combination']={}
    for lab,key in (('A_token','A'),('answer_sentence','SENTlab')):
        c={}
        c['qa_tune_to_qa_hold']=combo(sub(Q,Q['part']==0),sub(Q,Q['part']==1),key,False,rng,lg,'qa_tune->qa_hold')
        c['news_tune_to_news_hold']=combo(sub(Nw,Nw['part']==0),sub(Nw,Nw['part']==1),key,False,rng,lg,'news_tune->news_hold')
        c['news_tune_to_news_hold_with_attention']=combo(sub(Nw,Nw['part']==0),sub(Nw,Nw['part']==1),key,True,rng,lg,'news_tune->news_hold+att')
        c['qa_to_news_hold']=combo(Q,sub(Nw,Nw['part']==1),key,False,rng,lg,'qa->news_hold');c['news_to_qa_hold']=combo(Nw,sub(Q,Q['part']==1),key,False,rng,lg,'news->qa_hold')
        res['combination'][lab]=c
    pathlib.Path(a.out).write_text(json.dumps(res,indent=1,default=float));lg('wrote',a.out)
if __name__=='__main__':main()
