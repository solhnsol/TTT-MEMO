"""E57 post-write window gain by token class (CPU only; new file; numbers only). Over the chunks 8..279 of the saved window / window-less per-token NLL files (E57_windowless_<LABEL>.npz of job 298) and the exact p=.26 GAINQ512 row masks: per row group (SEL = banded GAINQ512 rows, NOSEL = other eligible rows, ALL), per token class: rows, mean W0 gain (nll_chunk - nll_win of W0 on the same tokens), mean post-write gain of the state, their ratio, and the share of the state's total remaining gain that the class holds (sum of post-write gain in the class / sum over the group).
Classes: (a) recency of the TARGET token in the stream before it: R512 = occurred within the previous 512 tokens, R8192 = occurred within the previous 8192 but not 512, NEW = not within 8192 (token id identity; positions p-8192..p-1 of the stream);
 (b) type of the target token: SENT1 (first token of a sentence), DIGIT (token text has a digit), PUNCT (no letter / digit), CAP (token text, stripped, starts with an upper-case letter), FUNC (stripped lower-cased text in the function-word list), OTHER;
 (c) size of the W0 gain: TOP1 (the 1 % of the group's tokens with the largest W0 gain), TOP10 (the next 9 %), REST; and the same by the state's own post-write gain (P_TOP1 / P_TOP10 / P_REST).
  .venv/bin/python scripts/exp07_write_ladder/e57_gain_tokens.py --bundle B --states-dir D --masks M.npz --labels W0,NAT,... --out O.json"""
import sys,json,argparse,pathlib
import numpy as np
HERE=pathlib.Path(__file__).resolve().parent
for p in (HERE,HERE/'e57',HERE.parents[1]/'scripts/exp04_what_ttt_retains',HERE.parents[1]/'scripts/exp05_write_dynamics'):sys.path.insert(0,str(p))
import e57_closure_masks as CM
CH=1024
FUNC=set('the of and to in a an is are was were be been being it its this that these those he she they we you i his her their our your my as at by for from with on or but not no if then so than also has have had do does did will would can could may might shall should which who whom whose what when where while about into over after before between under more most other some such only own same too very just up out'.split())
def token_class(s,first):
    if first:return 'SENT1'
    t=s.strip()
    if any(ch.isdigit() for ch in t):return 'DIGIT'
    if not any(ch.isalpha() for ch in t):return 'PUNCT'
    if t[:1].isupper():return 'CAP'
    if t.lower() in FUNC:return 'FUNC'
    return 'OTHER'
def recency(ids):
    last={};d=np.full(len(ids),10**9,np.int64)
    for p,t in enumerate(ids):
        q=last.get(int(t))
        if q is not None:d[p]=p-q
        last[int(t)]=p
    return d
def selftest():
    ids=np.array([5,6,5,7,8,9,6]);d=recency(ids);assert list(d)==[10**9,10**9,2,10**9,10**9,10**9,5]
    assert token_class(' The',False)=='CAP' and token_class(' the',False)=='FUNC' and token_class(' 1999',False)=='DIGIT' and token_class(',',False)=='PUNCT' and token_class(' dog',True)=='SENT1' and token_class(' dog',False)=='OTHER'
    print('e57_gain_tokens selftest ok')
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--bundle');ap.add_argument('--states-dir');ap.add_argument('--masks');ap.add_argument('--labels');ap.add_argument('--out');ap.add_argument('--band',type=int,default=512);ap.add_argument('--tokenizer');ap.add_argument('--selftest',action='store_true');a=ap.parse_args()
    if a.selftest:selftest();return
    import gen_world_ctl as G,e55_uv as UVm,e57_reads as RD
    tfile=a.tokenizer or str(HERE.parents[1]/'data/tokenizer/llama3/tokenizer.json');enc=G.load_enc(tfile);tk=UVm.tokenizer(tfile)
    items=RD.load_items(pathlib.Path(a.bundle)/'e57_bank'/'bank_v1.json');ids,iv,hits=RD.article_intervals(a.bundle,enc,items);ids=np.asarray(ids);ends=UVm.sentence_ends(tk,ids)
    M=np.load(a.masks,allow_pickle=True);labels=a.labels.split(',');Z={l:np.load(pathlib.Path(a.states_dir)/f'E57_windowless_{l}.npz') for l in labels}
    rec=recency(ids);strs={}
    def txt(t):
        if t not in strs:strs[t]=tk.decode([int(t)])
        return strs[t]
    # per-token arrays over chunks 8..279: group, classes, gains
    cols={g:[] for g in ('sel','el','tgt','pos','first')};gain={l:[] for l in labels}
    for k in range(8,280):
        q=UVm.chunk_sents(ends,k,CH);sid=np.full(CH,-1);first=np.zeros(CH,bool)
        for i,(s_,e_) in enumerate(q):sid[s_:e_]=i;first[s_]=True
        z0=Z[labels[0]];lm=z0['lm'][k];qb=CM.up(M[f'qb_GAINQ{a.band}_{k}'],CH);el=CM.rows_eligible(lm,sid)&lm;tgt=z0['tgt'][k]
        rows=np.nonzero(el)[0];pos=k*CH+rows+1
        assert (ids[pos]==tgt[rows]).all(),'target tokens do not match the stream'
        cols['sel'].append(qb[rows]);cols['tgt'].append(tgt[rows]);cols['pos'].append(pos);cols['first'].append(first[np.minimum(rows+1,CH-1)]&(rows+1<CH))
        for l in labels:gain[l].append((Z[l]['nll_chunk'][k]-Z[l]['nll_win'][k])[rows].astype(np.float64))
    sel=np.concatenate(cols['sel']);tgt=np.concatenate(cols['tgt']);pos=np.concatenate(cols['pos']);first=np.concatenate(cols['first'])
    G_={l:np.concatenate(v) for l,v in gain.items()};d=rec[pos]
    cls_a=np.where(d<=512,'R512',np.where(d<=8192,'R8192','NEW'))
    cls_b=np.array([token_class(txt(int(t)),bool(f)) for t,f in zip(tgt,first)])
    w0=G_[labels[0]];out={'groups':{}}
    for gname,mask in (('SEL',sel),('NOSEL',~sel),('ALL',np.ones(len(sel),bool))):
        idx=np.nonzero(mask)[0];w0g=w0[idx];q99=np.quantile(w0g,.99);q90=np.quantile(w0g,.90)
        cls_c0=np.where(w0g>=q99,'TOP1',np.where(w0g>=q90,'TOP10','REST'))
        res={}
        for l in labels[1:]:
            pg=G_[l][idx];p99=np.quantile(pg,.99);p90=np.quantile(pg,.90);cls_c1=np.where(pg>=p99,'P_TOP1',np.where(pg>=p90,'P_TOP10','P_REST'));tot=pg.sum();r={}
            for fam,cl in (('a',cls_a[idx]),('b',cls_b[idx]),('c0',cls_c0),('c1',cls_c1)):
                for c in sorted(set(cl)):
                    s=cl==c;n=int(s.sum());g0=float(w0g[s].mean());g1=float(pg[s].mean());r[f'{fam}:{c}']=dict(rows=n,row_share=n/len(idx),gain_W0=g0,gain_post=g1,ratio=g1/g0 if g0!=0 else None,share_of_remaining_gain=float(pg[s].sum()/tot))
            res[l]=r
        out['groups'][gname]=dict(rows=int(len(idx)),gain_W0_mean=float(w0g.mean()),thr_W0_q99=float(q99),thr_W0_q90=float(q90),states=res)
    out['note']='chunks 8..279; eligible rows; classes as in the module doc; share_of_remaining_gain = sum of the state post-write gain in the class / sum over the group'
    json.dump(out,open(a.out,'w'),indent=1)
    for gname in ('ALL','SEL','NOSEL'):
        for l in labels[1:]:
            print(f'== {gname} {l} (rows {out["groups"][gname]["rows"]})')
            for k_,v in out['groups'][gname]['states'][l].items():print(f"  {k_:14s} rows {v['rows']:7d} W0 {v['gain_W0']:+.3f} post {v['gain_post']:+.3f} ratio {('%.3f'%v['ratio']) if v['ratio'] is not None else 'nan'} share {v['share_of_remaining_gain']:+.3f}")
if __name__=='__main__':main()
