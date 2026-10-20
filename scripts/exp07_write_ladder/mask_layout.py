"""E41 spatial layout of the masked sets (CPU only, no GPU; server: needs gen_world, suppress, the tokenizer and the saved S3_<tag>_s<seed>_<world>.saved.json files of the cells). Extends e42_k2_tokens.py (same S reading, same world / sentence reconstruction, same token classes).
Input per cell: log[epoch].chunks[i] = {k, S (absolute stream positions, BOS = 0), n_masked, ...}. S may contain positions inside the chunk (O tokens of earlier sentences of the same chunk): they count in the totals and the sentence fractions (and are reported as S_tokens_in_chunk), runs and distances use the window part only. oraclefill cells (U35) store no S: give them as LABEL=TAG@oraclefill0.35 and S is rebuilt from the sentence table (sets.oraclefill, seeded random.Random(seed|pass|k|oraclefill)), checked against the stored n_masked and n_O (checks.rebuild_mismatch must be 0). Cells e.g. U35 (oraclefill 0.35), A35 (attnfill 0.35), attn-sel.
For every chunk, S is compared with a UNIFORM NULL that keeps the O part and replaces the rest: S_null = (S and O) + n_fill positions drawn uniformly without replacement from the candidates (positions >= 4, < chunk start - 128, inside the window) that are not O, n_fill = |S| - |S and O|; R repetitions per chunk (seeded). For a cell that masks no O (rand), the null is a plain uniform draw of |S| candidates.
(a) per EARLIER SAME-TRIPLE SENTENCE of the chunk's fact sentences (sentence_table triple t, round < the fact sentence's round; every earlier exposure of that fact, not only inside the window pieces that are candidates): masked fraction of its tokens in the window. Two versions: all tokens, and non-O tokens only (O = name / place tokens of those sentences, always masked by the O-including cells, so the non-O version is the fill's coverage).
    reported: n sentences, mean fraction, variance of the fraction, share of sentences with fraction > 0.8; observed vs null (mean over R repetitions).
(b) mean length of runs of consecutive masked positions: of S and of S minus O (the fill only); observed vs null.
(c) distance to the chunk start (k*CH - position): mean distance of S minus O and the share of S minus O per 1024-token distance bin (8 bins over the 8K window); observed vs null (the candidate-uniform expectation).
(d) K2 token classes (e42_k2_tokens.CLASSES) of S: observed share, and the share expected if the same number of positions were drawn uniformly from the candidates (e42_k2_tokens.tally, not O-preserving; the O class is therefore expected small).
Groups: chunks with an oracle set (withO) and without (noO; only cells that mask there, e.g. attn-sel, have anything in it).
Unit checks (per chunk, counted): sum of the K2 classes == n_masked; sum of the distance bins == |S|; sum of run lengths == |S|; |S| == n_masked; S inside [window start, chunk start). They are reported in the json ('checks') and must all be 0 mismatches.
Usage (server): mask_layout.py --seeds 101 --cells U35=<TAG>@oraclefill0.35 A35=<TAG> attnsel=attnsel_p4 --in-dir $OUT --out-dir /tmp/mask_layout [--reps 20]   (--cells LABEL=TAG reads S3_<TAG>_s<seed>_<A|B>.saved.json exactly as e42_k2_tokens.py; TAG is the --tag of the cell's job)
CPU: mask_layout.py --selftest"""
import os,sys,json,argparse
from pathlib import Path
import numpy as np
HERE=Path(__file__).resolve().parent;sys.path.insert(0,str(HERE))
import e42_k2_tokens as K2
import s3_sets as SETS
CH=K2.CH;W=K2.W;NTOK=K2.NTOK;BIN=1024;NB=W//BIN
def runs_of(S):
    """lengths of maximal runs of consecutive positions"""
    S=np.sort(np.asarray(S,int))
    if len(S)==0:return []
    cut=np.nonzero(np.diff(S)>1)[0];b=np.concatenate([[0],cut+1,[len(S)]]);return np.diff(b).tolist()
def sent_fracs(S,tok_lists,Oset):
    """masked fraction of each sentence's window tokens: all tokens, and non-O tokens (skipped when a sentence has none)"""
    Ss=set(int(x) for x in S);fa=[];fn=[]
    for toks in tok_lists:
        if toks:fa.append(sum(1 for p in toks if p in Ss)/len(toks))
        nt=[p for p in toks if p not in Oset]
        if nt:fn.append(sum(1 for p in nt if p in Ss)/len(nt))
    return fa,fn
def measure(S,k,Oset,tok_lists):
    """S may contain positions inside the chunk itself (O tokens of earlier sentences of the same chunk, >= chunk start): they count in n, n_in_chunk and the sentence fractions; runs and distances use the window part (< chunk start) only"""
    k0=k*CH;Sl=sorted(set(int(x) for x in S));win=np.asarray([p for p in Sl if p<k0],int);fill=np.asarray([p for p in win if p not in Oset],int);fa,fn=sent_fracs(Sl,tok_lists,Oset)
    d=lambda a:np.bincount(np.minimum((k0-a)//BIN,NB-1),minlength=NB) if len(a) else np.zeros(NB,int)
    return {'n':len(Sl),'n_win':len(win),'n_in_chunk':len(Sl)-len(win),'n_fill':len(fill),'runs':runs_of(win),'runs_fill':runs_of(fill),'frac_all':fa,'frac_nonO':fn,'dist_fill':d(fill),'dist_all':d(win),'dist_sum_fill':float((k0-fill).sum())}
def null_set(S,k,Oset,cand_pos,rng):
    S=set(int(x) for x in S);fixed=S&Oset;nf=len(S)-len(fixed);pool=np.asarray([p for p in cand_pos if p not in Oset],int);nf=min(nf,len(pool))
    pick=rng.choice(pool,nf,replace=False) if nf>0 else np.zeros(0,int);return sorted(fixed)+pick.tolist()
def prior_token_lists(sents,sets,k):
    """for chunk k: tokens of every earlier same-triple sentence of the chunk's fact sentences (positions in [window start, chunk end): an earlier sentence can sit inside the chunk itself)"""
    k0=k*CH;lo=max(0,k0-W);cur=[i for i,s in enumerate(sents) if k*CH<=s['place'][0]<(k+1)*CH];seen=set();out=[]
    for c in cur:
        for i,u in enumerate(sents):
            if u['t']==sents[c]['t'] and u['round']<sents[c]['round'] and i not in seen:
                seen.add(i);toks=sorted(p for p in u['tokens'] if lo<=p<k0+CH)
                if toks:out.append(toks)
    return out
class Agg:
    def __init__(self):self.o={'frac_all':[],'frac_nonO':[],'runs':[],'runs_fill':[]};self.n=0;self.dist=np.zeros(NB);self.dist_all=np.zeros(NB);self.nfill=0;self.dsum=0.0;self.nS=0;self.nin=0;self.k2=np.zeros(len(K2.CLASSES));self.k2e=np.zeros(len(K2.CLASSES));self.out_cand=0
    def add(self,m,w=1.0):
        for k_ in self.o:self.o[k_]+=[(x,w) for x in m[k_]]
        self.dist+=w*m['dist_fill'];self.dist_all+=w*m['dist_all'];self.nfill+=w*m['n_fill'];self.dsum+=w*m['dist_sum_fill'];self.nS+=w*m['n'];self.nin+=w*m['n_in_chunk']
    def summary(self):
        def st(key,thr=None):
            v=self.o[key]
            if not v:return None
            x=np.asarray([a for a,_ in v]);w=np.asarray([b for _,b in v]);mu=float((x*w).sum()/w.sum());var=float((w*(x-mu)**2).sum()/w.sum());r={'n':float(w.sum()),'mean':mu,'var':var}
            if thr is not None:r['share_gt_0.8']=float((w*(x>thr)).sum()/w.sum())
            return r
        return {'frac_all':st('frac_all',0.8),'frac_nonO':st('frac_nonO',0.8),'run_len_S':st('runs'),'run_len_fill':st('runs_fill'),'mean_dist_fill':self.dsum/self.nfill if self.nfill else None,'dist_share_fill':(self.dist/self.dist.sum()).tolist() if self.dist.sum() else None,'tokens_S':self.nS,'tokens_S_in_chunk':self.nin,'tokens_fill':self.nfill}
def process(S,k,sents,sid,role,sets,n_text,reps,rng,obs,nul,chk,k2obs,k2exp):
    k0=k*CH;Sa=np.asarray(S,int);O=sets.oracle(k);Oset=set(int(p) for p in O);cand=K2.candidates(k,NTOK);cand_pos=np.nonzero(cand)[0].tolist();tl=prior_token_lists(sents,sets,k)
    m=measure(Sa,k,Oset,tl);obs.add(m)
    chk['chunks']+=1;chk['n_masked_vs_S']+=int(len(set(Sa.tolist()))!=len(Sa));chk['dist_sum']+=int(m['dist_all'].sum()!=m['n_win']);chk['run_sum']+=int(sum(m['runs'])!=m['n_win']);chk['outside_window']+=int(len(Sa)>0 and (Sa.min()<max(0,k0-W) or Sa.max()>=k0+CH));chk['S_tokens_in_chunk']+=m['n_in_chunk']
    chk['S_outside_cand_nonO']+=int(sum(1 for p in Sa if p not in Oset and not (0<=p<len(cand) and cand[p])))
    cls=K2.classify_chunk(sents,sid,role,NTOK,sets,k,n_text);cnt,ex=K2.tally(cls,Sa,k,NTOK);chk['k2_sum']+=int(cnt.sum()!=len(Sa));k2obs+=cnt;k2exp+=ex*(cnt.sum()/max(ex.sum(),1))
    for _ in range(reps):nul.add(measure(null_set(Sa,k,Oset,cand_pos,rng),k,Oset,tl),1.0/reps)
def run(a):
    sys.path.insert(0,str(K2.ROOT/'scripts/exp04_what_ttt_retains'));import gen_world as gw;import suppress as SP
    inD=Path(a.in_dir);res={'cells':{},'args':vars(a)};cache={};chk={k:0 for k in ('chunks','n_masked_vs_S','dist_sum','run_sum','outside_window','S_outside_cand_nonO','S_tokens_in_chunk','k2_sum','n_masked_field','rebuild_mismatch','rebuilt_chunks')};rng=np.random.default_rng(41)
    for lab,tag in (c.split('=') for c in a.cells):
        tag,_,rb=tag.partition('@');rfrac=float(rb[len('oraclefill'):]) if rb.startswith('oraclefill') else None
        ag={g:(Agg(),Agg(),np.zeros(len(K2.CLASSES)),np.zeros(len(K2.CLASSES)),[0]) for g in ('withO','noO')}
        for seed in [int(x) for x in a.seeds.split(',')]:
            for wld in a.worlds:
                f=inD/f'{tag}_s{seed}_{wld}.saved.json'
                if not f.exists():f=inD/f'S3_{tag}_s{seed}_{wld}.saved.json'
                if not f.exists():res.setdefault('missing',[]).append(str(f));continue
                for e in json.load(open(f))['log']:
                    if 'chunks' not in e:continue
                    key=(seed,e['epoch'],wld)
                    if key not in cache:
                        table,sents,n_text=K2.real_tables(gw,SP,seed,e['epoch'],wld=='B');sid,role=K2.pos_table(sents,NTOK);cache[key]=(sents,sid,role,SETS.Sets(table,seed,e['epoch']),n_text)
                    sents,sid,role,sets,n_text=cache[key]
                    for c in e['chunks']:
                        if 'S' not in c:   # oraclefill cells do not store S: rebuilt from sets, seed, pass and the fill fraction (pure python, random.Random(seed|pass|k|oraclefill)); checked against the stored n_masked / n_O
                            assert rfrac is not None,f'{f}: no S stored; give the cell as LABEL=TAG@oraclefill<frac> to rebuild it'
                            sets.frac=rfrac;Sr,_=sets.oraclefill(c['k']);chk['rebuilt_chunks']+=1;chk['rebuild_mismatch']+=int(len(Sr)!=c['n_masked'] or len(sets.oracle(c['k']))!=c.get('n_O',len(sets.oracle(c['k']))));c=dict(c,S=Sr)
                        if c['n_masked']==0 and not sets.oracle(c['k']):continue
                        g='withO' if sets.oracle(c['k']) else 'noO';o,n,k2o,k2e,cnt=ag[g];chk['n_masked_field']+=int(len(set(c['S']))!=c['n_masked'])
                        process(c['S'],c['k'],sents,sid,role,sets,n_text,a.reps,rng,o,n,chk,k2o,k2e);cnt[0]+=1
        res['cells'][lab]={g:{'chunks':ag[g][4][0],'observed':ag[g][0].summary(),'uniform_null':ag[g][1].summary(),'k2_share_observed':(ag[g][2]/max(ag[g][2].sum(),1)).tolist(),'k2_share_expected_uniform':(ag[g][3]/max(ag[g][3].sum(),1)).tolist(),'classes':K2.CLASSES} for g in ag if ag[g][4][0]}
    res['checks']=chk;return res
def md_of(res):
    f=lambda v,k:'-' if v is None else format(v[k],'.3f')
    L=['# E41 spatial layout of the masked sets (CPU)\n',f"checks (all mismatch counts must be 0, S_outside_cand_nonO is informative): {json.dumps(res['checks'])}\n",'Null = O part kept, remaining positions uniform among the non-O candidates (same count), R repetitions per chunk.\n']
    for lab,gs in res['cells'].items():
        for g,v in gs.items():
            o,u=v['observed'],v['uniform_null'];L+=[f"## {lab}, chunks {'with' if g=='withO' else 'without'} an oracle set (n = {v['chunks']} chunks)\n",'| quantity | observed | uniform null |','|---|---|---|']
            for nm,key in (('(a) masked fraction per earlier same-triple sentence, non-O tokens','frac_nonO'),('(a) same, all tokens (O included)','frac_all')):
                for st in ('mean','var','share_gt_0.8'):L.append(f"| {nm}: {st} | {f(o[key],st)} | {f(u[key],st)} |")
            L.append(f"| (a) number of sentence observations (non-O) | {f(o['frac_nonO'],'n')} | {f(u['frac_nonO'],'n')} |")
            L+=[f"| (b) mean run length, S | {f(o['run_len_S'],'mean')} | {f(u['run_len_S'],'mean')} |",f"| (b) mean run length, S minus O (fill) | {f(o['run_len_fill'],'mean')} | {f(u['run_len_fill'],'mean')} |"]
            L.append(f"| (c) mean distance to the chunk start, fill | {o['mean_dist_fill'] and round(o['mean_dist_fill'],1)} | {u['mean_dist_fill'] and round(u['mean_dist_fill'],1)} |")
            for i in range(NB):L.append(f"| (c) share of the fill at distance {i*BIN}-{(i+1)*BIN-1} | {'-' if not o['dist_share_fill'] else format(o['dist_share_fill'][i],'.3f')} | {'-' if not u['dist_share_fill'] else format(u['dist_share_fill'][i],'.3f')} |")
            L+=['','(d) K2 token classes of S: observed share vs share expected for the same number of positions drawn uniformly from the candidates (not O-preserving)\n','| class | observed | expected uniform |','|---|---|---|']
            L+=[f"| {c} | {v['k2_share_observed'][i]:.4f} | {v['k2_share_expected_uniform'][i]:.4f} |" for i,c in enumerate(v['classes'])];L.append('')
    return '\n'.join(L)+'\n'
def selftest():
    # runs / fractions / null on hand-made sets
    assert runs_of([5,6,7,20,22,23])==[3,1,2] and runs_of([])==[]
    toks=[list(range(100,110)),list(range(200,205))];fa,fn=sent_fracs(list(range(100,109))+[200],toks,{100,101});assert fa==[0.9,0.2] and abs(fn[0]-7/8)<1e-9 and fn[1]==0.2,(fa,fn)
    k=12;k0=k*CH;Oset={k0-3000,k0-3001};S=sorted(Oset|{k0-500,k0-499,k0-498,k0-2000});m=measure(S,k,Oset,[[k0-3001,k0-3000,k0-499,k0-1500]])
    assert m['n']==6 and m['n_fill']==4 and m['dist_all'].sum()==6 and m['dist_fill'].sum()==4 and sum(m['runs'])==6 and sum(m['runs_fill'])==4
    assert sorted(m['runs'])==[1,2,3] and sorted(m['runs_fill'])==[1,3]
    m2=measure(S+[k0+5,k0+6],k,Oset,[[k0-3001,k0+5,k0+6]]);assert m2['n']==8 and m2['n_win']==6 and m2['n_in_chunk']==2 and m2['dist_all'].sum()==6 and sum(m2['runs'])==6 and m2['frac_all']==[1.0]
    rng=np.random.default_rng(0);cand=list(range(max(4,k0-W),k0-128));ns=null_set(S,k,Oset,cand,rng);assert len(ns)==len(S) and Oset<=set(ns) and len(set(ns))==len(ns) and all(p in cand for p in ns if p not in Oset)
    # null mean fraction == n_fill / |non-O candidates| (hypergeometric mean), checked by simulation on one sentence of 400 tokens
    toks2=[list(range(k0-3500,k0-3100))];pool=[p for p in cand if p not in Oset];nf=2000;fr=[]
    for _ in range(200):
        S2=rng.choice(pool,nf,replace=False);fr.append(sent_fracs(S2,toks2,Oset)[1][0])
    assert abs(np.mean(fr)-nf/len(pool))<0.01,(np.mean(fr),nf/len(pool))
    # the process() unit checks on the synthetic world of e42_k2_tokens: sums equal n_masked
    t=[]
    for tr in range(30):
        for r in range(1,5):
            pos=(r-1)*6144+tr*150+15;t.append(dict(idx=len(t),t=f't{tr}',pair=f'p{tr//2}',round=r,place=[pos],names=[pos-3,pos-2],spans=[[pos],[pos-3,pos-2]],text=''))
    t=sorted(t,key=lambda s:s['place'][0]);roles=['other','other','name','name','rel','place','punct','punct'];sents=[dict(s,tokens={s['place'][0]-5+j:r for j,r in enumerate(roles)}) for s in t]
    sid,role=K2.pos_table(sents,NTOK);sets=SETS.Sets(t,101,2);kk=[x for x in range(8,63) if sets.oracle(x)][0];O=sets.oracle(kk);k0=kk*CH
    cn=K2.candidates(kk,NTOK);pool2=[p for p in np.nonzero(cn)[0].tolist() if p not in set(O)];Sx=sorted(set(O)|set(rng.choice(pool2,1500,replace=False).tolist()))
    obs,nul=Agg(),Agg();chk={x:0 for x in ('chunks','n_masked_vs_S','dist_sum','run_sum','outside_window','S_outside_cand_nonO','S_tokens_in_chunk','k2_sum','n_masked_field')};k2o=np.zeros(len(K2.CLASSES));k2e=np.zeros(len(K2.CLASSES))
    process(Sx,kk,sents,sid,role,sets,NTOK,5,rng,obs,nul,chk,k2o,k2e)
    assert chk['k2_sum']==0 and chk['dist_sum']==0 and chk['run_sum']==0 and chk['outside_window']==0 and chk['n_masked_vs_S']==0 and k2o.sum()==len(Sx),chk
    sets.frac=0.35;Sr1,_=sets.oraclefill(kk);Sr2,_=sets.oraclefill(kk);assert Sr1==Sr2 and len(Sr1)==int(round(0.35*len(sets.window_before(kk)))) and set(O)<=set(Sr1)   # the U35 rebuild is deterministic and has the registered size
    obs3,nul3=Agg(),Agg();k2a=np.zeros(len(K2.CLASSES));k2b=np.zeros(len(K2.CLASSES));process(Sr1,kk,sents,sid,role,sets,NTOK,2,rng,obs3,nul3,chk,k2a,k2b);assert chk['k2_sum']==0 and k2a.sum()==len(Sr1)
    so,sn=obs.summary(),nul.summary();assert so['tokens_S']==len(Sx) and abs(sn['tokens_S']-len(Sx))<1e-6,(so['tokens_S'],sn['tokens_S'])
    print('mask_layout SELFTEST OK',{'chunk':kk,'|S|':len(Sx),'obs mean frac nonO':round(so['frac_nonO']['mean'],3),'null':round(sn['frac_nonO']['mean'],3)})
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--seeds',default='101');ap.add_argument('--worlds',default='AB');ap.add_argument('--cells',nargs='+',default=[]);ap.add_argument('--in-dir',default=os.environ.get('EXP07_OUT',str(K2.ROOT/'experiments/exp07_write_ladder')))
    ap.add_argument('--out-dir',default='/tmp/mask_layout');ap.add_argument('--reps',type=int,default=20);ap.add_argument('--selftest',action='store_true');a=ap.parse_args()
    if a.selftest:selftest();sys.exit(0)
    assert a.cells,'--cells LABEL=TAG ...';res=run(a);od=Path(a.out_dir);od.mkdir(parents=True,exist_ok=True);json.dump(res,open(od/'mask_layout.json','w'),indent=1,default=float);md=md_of(res);(od/'mask_layout.md').write_text(md);print(md)
