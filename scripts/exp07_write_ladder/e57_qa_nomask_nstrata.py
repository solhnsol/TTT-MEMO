"""E57 QA no-mask analysis by n, W0 tercile and answer type on the FIRST CONTENT token (new file, CPU only; thread 'QA에서 가림이 안 되는 이유 규명', Hans approved: 'stratify and analyse every other cause'). Inputs: the per-token rows of job 672 (e57_public_qa_stage12: bundle-matrix reads of the states NATQ, ALLQ128, GAINQ512 p.10 at n = 1, 3, 5, 9 and W0) and of job 674 (e57_public_qa_stage13, n = 9, original 16 candidates, conditions A and B).
Token positions of the gold string (leading whitespace-only tokens skipped, the first remaining token = 'first content token'): first = first content token, second = the next one, later = the mean of all following tokens (null when the gold is shorter), full = summed log-prob of all tokens.
Per item and state: dA_x = log-prob(state) - log-prob(W0) of the gold token x under its own question (A); dB_first = the same for the first content token of the gold string under the questions of the OTHER items that have it as a candidate (stage 12: B_bundle) or under the other-item question (stage 13: condition B); d(A-B) = dA_first - dB_first; rank ratio of the first content token among the candidates (each candidate by its own first content token; rank / number of candidates). Masking add-on = the same quantity for ALLQ128 (GAINQ512 p.10) minus NATQ, per item.
Tables per source (stage12 and stage13), per n, for all items, per W0 first-content-token log-prob tercile of the gold (low / mid / high, tercile cuts from all items) and per answer type (LOW / CAP / NUM / DATE): mean over items with the bootstrap 95% CI over items for dA first / second / later / full, d(A-B) first, rank ratio first, and the add-ons of all of them; plus the share first-token add-on / full-sum add-on per n.
  python3 e57_qa_nomask_nstrata.py --stage12 D --stage13 D --out OUT.json | --selftest"""
import sys,json,pathlib
import numpy as np
HERE=pathlib.Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
import e57_pqa2_stream as S
import e57_space_token_recheck as ST
NS=(1,3,5,9);ARMS=('NATQ','ALLQ128','GAINQ512P10')
def boot(x,seed=1,B=2000):
    x=np.asarray(x,np.float64);x=x[~np.isnan(x)]
    if len(x)==0:return dict(mean=None,ci95=None,n=0)
    ix=np.random.default_rng(seed).integers(0,len(x),(B,len(x)));m=x[ix].mean(1);return dict(mean=float(x.mean()),ci95=[float(np.percentile(m,2.5)),float(np.percentile(m,97.5))],n=int(len(x)))
def gold_parts(lp,off):
    lp=np.asarray(lp,np.float64);first=lp[off];second=lp[off+1] if len(lp)>off+1 else np.nan;later=lp[off+2:].mean() if len(lp)>off+2 else np.nan;return first,second,later,lp.sum()
def item_vectors(rows_by_label,ids,tok,cand_layout,ws):
    """-> {label: dict of arrays over ids}: first, second, later, full of the gold under its own question, B_first (mean over other questions), rr_first; tok[i] = token ids of the candidates of item i's row (candidate 0 = gold); cand_layout[i] = item ids of the candidates (stage 12) or None (stage 13: B from the B rows)"""
    out={}
    for lab,rows in rows_by_label.items():
        n=len(ids);v={k:np.full(n,np.nan) for k in ('first','second','later','full','B_first','rr_first')};S_={}
        for k,(i,r) in enumerate(zip(ids,rows)):
            toks=tok[i];offs=[ST.lead(t,ws) for t in toks];lps=r['cand_tok_lp'];f,s,l,fu=gold_parts(lps[0],offs[0]);v['first'][k]=f;v['second'][k]=s;v['later'][k]=l;v['full'][k]=fu
            fc=[lps[c][offs[c]] for c in range(len(lps))];v['rr_first'][k]=(1+sum(1 for c in range(1,len(fc)) if fc[c]>fc[0]))/len(fc)
            if cand_layout is not None:S_[i]={cid:lps[c][offs[c]] for c,cid in enumerate(cand_layout[i])}
        if cand_layout is not None:
            for k,i in enumerate(ids):
                vals=[S_[q][i] for q in ids if q!=i and i in S_[q]]
                if vals:v['B_first'][k]=np.mean(vals)
        out[lab]=v
    return out
def summarize(V,mask,B_override=None):
    """V: {label: vectors}; -> metrics for the states NATQ / ALLQ128 / GAINQ512P10 and the add-ons, for the items of `mask`"""
    W0=V['W0'];d={}
    for a in ARMS:
        m=V[a];x={}
        for k in ('first','second','later','full'):x[f'dA_{k}']=m[k]-W0[k]
        x['dB_first']=m['B_first']-W0['B_first'];x['dAB_first']=x['dA_first']-x['dB_first'];x['rr_first']=m['rr_first'];x['rr_first_improvement']=W0['rr_first']-m['rr_first'];d[a]=x
    d['addon_ALLQ128']={k:d['ALLQ128'][k]-d['NATQ'][k] for k in d['NATQ']};d['addon_GAINQ512P10']={k:d['GAINQ512P10'][k]-d['NATQ'][k] for k in d['NATQ']}
    res={'n':int(mask.sum())}
    for a in ARMS+('addon_ALLQ128','addon_GAINQ512P10'):res[a]={k:boot(v[mask]) for k,v in d[a].items()}
    for a in ('ALLQ128','GAINQ512P10'):
        f=d[f'addon_{a}']['dA_first'][mask];fu=d[f'addon_{a}']['dA_full'][mask];res[f'share_first_over_full_addon_{a}']=float(np.nanmean(f)/np.nanmean(fu)) if abs(np.nanmean(fu))>1e-9 else None
    return res
def build(V_by_n,typ,w0_first):
    res={}
    q=np.percentile(w0_first,[100/3,200/3]);ter=np.digitize(w0_first,q);masks={'ALL':np.ones(len(typ),bool),'tercile_low':ter==0,'tercile_mid':ter==1,'tercile_high':ter==2}
    for t in sorted(set(typ.tolist())):masks[f'type_{t}']=typ==t
    for n,V in V_by_n.items():res[f'n{n}']={name:summarize(V,m) for name,m in masks.items()}
    res['tercile_cuts_W0_first_content_token_logprob']=[float(x) for x in q];res['tercile_ranges']={k:[float(w0_first[m].min()),float(w0_first[m].max())] for k,m in masks.items() if k.startswith('tercile')};return res
def run(d12,d13,out):
    tk=S.tokenizer();d12=pathlib.Path(d12);d13=pathlib.Path(d13);rm=json.load(open(d12/'e57_pqa12_read_meta.json'));ids=rm['item_ids'];lay=json.load(open(d12/'e57_pqa12_layout.json'));tok12=json.load(open(d12/'e57_pqa12_tokens.json'))
    ws=ST.ws_token_ids(tk,{t for i in tok12.values() for c in i for t in c});ld12=lambda l:json.load(open(d12/f'E57_pqa12_rows_{l}.json'))
    rows0=ld12('W0');typ=np.asarray([r['type'] for r in rows0]);res={'whitespace_token_ids':sorted(ws),'stage12_job672':{},'stage13_job674':{}}
    V_by_n={}
    for n in NS:
        R={'W0':rows0};R.update({a:ld12(f'{a}@n{n}'.replace('@','_') if False else f'{a}_n{n}') for a in ARMS});V_by_n[n]=item_vectors(R,ids,tok12,lay,ws)
    w0f=V_by_n[9]['W0']['first'];res['stage12_job672']=build(V_by_n,typ,w0f)
    # stage 13 (n = 9, original candidates; B from the B condition rows)
    ids13=json.load(open(d13/'e57_pqa13_read_meta.json'))['item_ids'];tok13=json.load(open(d13/'e57_pqa13_tokens.json'));ld13=lambda c,l:json.load(open(d13/f'E57_pqa13_rows_{c}_{l}.json'))
    VA=item_vectors({l:ld13('A',l) for l in ('W0','NATQ_n9','ALLQ128_n9','GAINQ512P10_n9')},ids13,tok13['A'],None,ws);VB=item_vectors({l:ld13('B',l) for l in ('W0','NATQ_n9','ALLQ128_n9','GAINQ512P10_n9')},ids13,tok13['B'],None,ws)
    V9={'W0':dict(VA['W0']),**{a:dict(VA[f'{a}_n9']) for a in ARMS}}
    for lab,src in (('W0','W0'),)+tuple((a,f'{a}_n9') for a in ARMS):V9[lab]['B_first']=VB[src]['first']
    typ13=np.asarray([r['type'] for r in ld13('A','W0')]);res['stage13_job674']=build({9:V9},typ13,V9['W0']['first'])
    pathlib.Path(out).parent.mkdir(parents=True,exist_ok=True);pathlib.Path(out).write_text(json.dumps(dict(definition=__doc__.split('Inputs:')[0],**res),indent=1,default=float));return res
def selftest():
    ws={220};lp=[-3.,-1.,-2.,-4.];f,s,l,fu=gold_parts(lp,1);assert f==-1. and s==-2. and l==-4. and fu==-10.
    f,s,l,fu=gold_parts([-1.,-2.],1);assert f==-2. and np.isnan(s) and np.isnan(l)
    print('e57_qa_nomask_nstrata selftest ok')
def main():
    if '--selftest' in sys.argv:selftest();return
    g=lambda k:sys.argv[sys.argv.index(k)+1];run(g('--stage12'),g('--stage13'),g('--out'))
if __name__=='__main__':main()
