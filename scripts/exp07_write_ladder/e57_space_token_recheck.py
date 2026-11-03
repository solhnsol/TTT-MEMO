"""E57 space-token recheck (new file, CPU only; request of the thread 'QA에서 가림이 안 되는 이유 규명', Hans approved): the candidate strings of the QA reads are scored as ' ' + string; the Llama-3 pre-tokenizer splits numbers without a leading space (\\p{N}{1,3}), so a numeric answer (or a candidate starting with a digit) starts with a SPACE-ONLY token (e.g. ' ' then '198' '7'), and every first-token quantity then measures the space, not the content. Inputs: the per-token rows of job 674 (e57_public_qa_stage13: items455, original 16 candidates, conditions A / B / C, states W0, NATQ, ALLQ128, GAINQ512 p.10 at n = 9) and of job 672 (e57_public_qa_stage12: bundle matrix with the gold strings of the other items as candidates, A and B_bundle).
 (1) counts: items whose gold starts with a space-only token, and candidates (all candidates, the distractors only) that start with one, in stage 13 and stage 12;
 (2) the first-token columns recomputed from the FIRST CONTENT TOKEN (leading whitespace-only tokens skipped, each string by its own count): rank ratio and top-1 over the candidates, first-token dA, dB (dB_bundle in stage 12), d(A-B) and the first-token gold-minus-candidate difference, side by side with the original first-token values, for W0 / NATQ / ALLQ128 / GAINQ512 p.10, for all items and for the items whose gold starts with a space token;
 (3) branch columns: the number of (gold, candidate) pairs whose branch position (first differing token) is a whitespace-only token in either string, and the branch rank / top-1 recomputed after skipping the leading whitespace of both strings (branch measured from the first content token to the first content token at which they differ).
  python3 e57_space_token_recheck.py --stage13 D13 --stage12 D12 --out OUT.json | --selftest"""
import sys,json,pathlib
import numpy as np
HERE=pathlib.Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
import e57_pqa2_stream as S
STATES=(('NATQ','NATQ_n9'),('ALLQ128','ALLQ128_n9'),('GAINQ512P10','GAINQ512P10_n9'))
def boot(x,seed=1,B=2000):
    x=np.asarray(x,np.float64);x=x[~np.isnan(x)]
    if len(x)==0:return dict(mean=None,ci95=None,n=0)
    ix=np.random.default_rng(seed).integers(0,len(x),(B,len(x)));m=x[ix].mean(1);return dict(mean=float(x.mean()),ci95=[float(np.percentile(m,2.5)),float(np.percentile(m,97.5))],n=int(len(x)))
def ws_token_ids(tk,all_ids):
    """ids whose decoded string is non-empty and only whitespace"""
    out=set()
    for t in all_ids:
        s=tk.decode([int(t)])
        if s!='' and s.strip()=='':out.add(int(t))
    return out
def lead(tokens,ws):
    """number of leading whitespace-only tokens (at least one token stays)"""
    n=0
    while n<len(tokens)-1 and int(tokens[n]) in ws:n+=1
    return n
def row_metrics(lps,toks,ws):
    """lps: list over candidates (gold first) of token log-prob lists; toks: token ids -> dict of orig / skipped first-token and branch metrics, number of pairs with a whitespace branch"""
    n=len(lps);g=np.asarray(lps[0],np.float64);o={}
    off=[lead(t,ws) for t in toks];fo=[lps[c][0] for c in range(n)];fs=[lps[c][off[c]] for c in range(n)]
    for tag,f in (('orig',fo),('skip',fs)):
        beat=sum(1 for c in range(1,n) if f[c]>f[0]);o[f'rr_first_{tag}']=(1+beat)/n;o[f'top1_first_{tag}']=float(beat==0);o[f'gap_first_{tag}']=float(f[0]-np.mean(f[1:])) if n>1 else np.nan;o[f'gold_first_{tag}']=float(f[0])
    nws=0;nb_o=0;nb_s=0
    for c in range(1,n):
        tg,tc=toks[0],toks[c];lc=np.asarray(lps[c],np.float64);L=min(len(tg),len(tc));t=next((q for q in range(L) if tg[q]!=tc[q]),L-1)
        if int(tg[t]) in ws or int(tc[t]) in ws:nws+=1
        nb_o+=int(lc[:t+1].sum()>g[:t+1].sum())
        a,b=off[0],off[c];tg2,tc2=tg[a:],tc[b:];L2=min(len(tg2),len(tc2));t2=next((q for q in range(L2) if tg2[q]!=tc2[q]),L2-1);nb_s+=int(lc[b:b+t2+1].sum()>g[a:a+t2+1].sum())
    o['rr_branch_orig']=(1+nb_o)/n;o['top1_branch_orig']=float(nb_o==0);o['rr_branch_skip']=(1+nb_s)/n;o['top1_branch_skip']=float(nb_s==0);o['n_pairs_ws_branch']=nws;o['n_pairs']=n-1;o['gold_space']=float(off[0]>0);o['n_cands_space']=sum(1 for c in range(1,n) if off[c]>0);return o
def stage13(d,tk):
    d=pathlib.Path(d);rm=json.load(open(d/'e57_pqa13_read_meta.json'));ids=rm['item_ids'];tok=json.load(open(d/'e57_pqa13_tokens.json'));ld=lambda c,l:json.load(open(d/f'E57_pqa13_rows_{c}_{l}.json'))
    ws=ws_token_ids(tk,{t for c in tok.values() for i in c.values() for cand in i for t in cand});labels=('W0',)+tuple(s for _,s in STATES);M={}
    for c in 'ABC':
        for l in labels:
            R=ld(c,l);assert [r['id'] for r in R]==ids;M[(c,l)]=[row_metrics(r['cand_tok_lp'],tok[c][i],ws) for r,i in zip(R,ids)]
    typ=[r['type'] for r in ld('A','W0')];return ids,ws,M,np.asarray(typ)
def table13(ids,M,typ):
    arr=lambda c,l,k:np.asarray([x[k] for x in M[(c,l)]]);gs=arr('A','W0','gold_space')>0;res={'counts':{},'tables':{}}
    cs=arr('A','W0','n_cands_space');res['counts']=dict(n_items=len(ids),n_gold_starts_with_space_token=int(gs.sum()),n_distractor_candidates=int(len(ids)*15),n_distractor_candidates_starting_with_space_token=int(cs.sum()),items_with_at_least_one_space_candidate=int((cs>0).sum()),gold_space_by_type={t:int((gs&(typ==t)).sum()) for t in sorted(set(typ.tolist()))},type_counts={t:int((typ==t).sum()) for t in sorted(set(typ.tolist()))},pairs_with_whitespace_branch=int(arr('A','W0','n_pairs_ws_branch').sum()),n_pairs=int(arr('A','W0','n_pairs').sum()),items_with_a_whitespace_branch_pair=int((arr('A','W0','n_pairs_ws_branch')>0).sum()))
    for sn,mk in (('ALL',np.ones(len(ids),bool)),('gold_space',gs),('gold_no_space',~gs)):
        t={}
        for c in 'ABC':
            for l in ('W0',)+tuple(s for _,s in STATES):
                t[f'{c}|{l}']={k:boot(arr(c,l,k)[mk]) for k in ('rr_first_orig','rr_first_skip','top1_first_orig','top1_first_skip','rr_branch_orig','rr_branch_skip','top1_branch_orig','top1_branch_skip','gap_first_orig','gap_first_skip')}
        for name,st in STATES:
            for tag in ('orig','skip'):
                dA=arr('A',st,f'gold_first_{tag}')-arr('A','W0',f'gold_first_{tag}');dB=arr('B',st,f'gold_first_{tag}')-arr('B','W0',f'gold_first_{tag}')
                t[f'{name}|d_first_{tag}']=dict(dA=boot(dA[mk]),dB=boot(dB[mk]),dA_minus_dB=boot((dA-dB)[mk]))
            for k in ('rr_first','rr_branch'):
                for tag in ('orig','skip'):
                    imp=lambda c:arr(c,'W0',f'{k}_{tag}')-arr(c,st,f'{k}_{tag}');t[f'{name}|gain_{k}_{tag}_improvement']=boot((imp('A')-imp('B'))[mk])
        res['tables'][sn]={'n':int(mk.sum()),**t}
    return res
def stage12(d,tk,ws):
    d=pathlib.Path(d);rm=json.load(open(d/'e57_pqa12_read_meta.json'));ids=rm['item_ids'];lay=json.load(open(d/'e57_pqa12_layout.json'));tok=json.load(open(d/'e57_pqa12_tokens.json'));ld=lambda l:json.load(open(d/f'E57_pqa12_rows_{l}.json'));labels=('W0',)+tuple(s for _,s in STATES);M={}
    for l in labels:
        R=ld(l);assert [r['id'] for r in R]==ids;M[l]=[row_metrics(r['cand_tok_lp'],tok[i],ws) for r,i in zip(R,ids)]
    # A / B_bundle first-token (orig / skip) per item
    ft={}
    for l in labels:
        R=ld(l);F={'orig':{},'skip':{}}
        for r,i in zip(R,ids):
            offs=[lead(t,ws) for t in tok[i]];F['orig'][i]=dict(zip(lay[i],[x[0] for x in r['cand_tok_lp']]));F['skip'][i]=dict(zip(lay[i],[r['cand_tok_lp'][c][offs[c]] for c in range(len(offs))]))
        ft[l]=F
    arr=lambda l,k:np.asarray([x[k] for x in M[l]]);gs=arr('W0','gold_space')>0;res={'counts':dict(n_questions=len(ids),n_gold_starts_with_space_token=int(gs.sum()),n_candidates_total=int(sum(len(v) for v in lay.values())),n_candidates_starting_with_space_token=int(sum(x['n_cands_space'] for x in M['W0'])+gs.sum()),pairs_with_whitespace_branch=int(arr('W0','n_pairs_ws_branch').sum()),n_pairs=int(arr('W0','n_pairs').sum())),'tables':{}}
    for sn,mk in (('ALL',np.ones(len(ids),bool)),('gold_space',gs)):
        t={}
        for l in labels:t[l]={k:boot(arr(l,k)[mk]) for k in ('rr_first_orig','rr_first_skip','top1_first_orig','top1_first_skip','rr_branch_orig','rr_branch_skip','top1_branch_orig','top1_branch_skip')}
        for name,st in STATES:
            for tag in ('orig','skip'):
                dA=[];dB=[]
                for i in ids:
                    a0=ft['W0'][tag][i][i];a1=ft[st][tag][i][i];vs0=[ft['W0'][tag][q][i] for q in ids if q!=i and i in ft['W0'][tag][q]];vs1=[ft[st][tag][q][i] for q in ids if q!=i and i in ft[st][tag][q]];dA.append(a1-a0);dB.append(np.mean(vs1)-np.mean(vs0) if vs0 else np.nan)
                dA=np.asarray(dA);dB=np.asarray(dB);t[f'{name}|d_first_{tag}']=dict(dA=boot(dA[mk]),dB_bundle=boot(dB[mk]),dA_minus_dB_bundle=boot((dA-dB)[mk]))
        res['tables'][sn]={'n':int(mk.sum()),**t}
    return res
def selftest():
    ws={220}
    assert lead([220,1,2],ws)==1 and lead([220],ws)==0 and lead([5,220],ws)==0
    lps=[[-3.,-1.,-1.],[-1.,-1.]];toks=[[220,198,7],[5,6]];o=row_metrics(lps,toks,ws)
    assert o['rr_first_orig']==1.0 and o['top1_first_orig']==0.0
    assert o['n_pairs_ws_branch']==1 and o['gold_space']==1.0 and o['gold_first_skip']==-1.0 and o['gold_first_orig']==-3.0
    assert o['rr_first_skip']==0.5 and o['top1_first_skip']==1.0
    print('e57_space_token_recheck selftest ok')
def main():
    if '--selftest' in sys.argv:selftest();return
    g=lambda k:sys.argv[sys.argv.index(k)+1];tk=S.tokenizer();ids,ws,M,typ=stage13(g('--stage13'),tk);res={'whitespace_only_token_ids':sorted(ws),'stage13_job674':table13(ids,M,typ),'stage12_job672':stage12(g('--stage12'),tk,ws),'definition':__doc__.split('Inputs:')[0]}
    pathlib.Path(g('--out')).parent.mkdir(parents=True,exist_ok=True);pathlib.Path(g('--out')).write_text(json.dumps(res,indent=1,default=float))
if __name__=='__main__':main()
