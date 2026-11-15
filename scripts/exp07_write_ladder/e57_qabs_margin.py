"""E57 QA block-stream (job 1299), EXPLORATORY secondary analysis on CPU from the stored per-token log-probs of e57_qabs/qabs_s<seed>_reads.json (new file; definitions fixed BEFORE the result was looked at, core-claims thread 11/16 05:27 KST). No GPU, no new reads.
Per item (bundle of 8 candidates: own gold + the golds of the other bundle paragraphs, related strings dropped), per condition A (own question) and B (question of the partner item):
  lp_S(c) = sum of the log-probs of ALL tokens of the candidate span;  lp_N(c) = lp_S(c) / number of tokens (length-normalised mean)
  margin_X = lp_X(gold) - max over the wrong candidates of lp_X          (1283 convention, X = S or N)
Per item and read: state minus W0 (Delta) of margin_A  -> dA_S, dA_N;  question-specific part  dAB_X = Delta margin_A - Delta margin_B  -> dAB_S, dAB_N. Together with the first-content-token measures of the main analysis (dA_first = Delta gold lp under A, dAB_first = Delta(A) - Delta(B), e57_qabs_run.analyze). Read tables: lag -1 (bundle not yet written), 0..5 (bundle j - b at read point j, out-of-window neutral tail), in0 (in-window, tail = the real last 8 chunks, bundle just written).
Statistic: per seed the mean over the items of a lag, contrast NATMASK1024 - NATS, t 95% interval over seeds (df = n_seeds - 1). All measures and all lags are reported (no selection).
  python3 e57_qabs_margin.py --dir O --seeds 101,102,103,104 --nb 6 --json J | --selftest"""
import sys,json,pathlib,argparse
import numpy as np
HERE=pathlib.Path(__file__).resolve().parent;sys.path.insert(0,str(HERE))
NV=16;ARMS=('NATS','NATMASK');MEAS=('dA_first','dAB_first','dA_S','dAB_S','dA_N','dAB_N')
def item_scores(rows,meta,bi,dec):
    """rows of one bundle (A rows then B rows, item-major, tokens per candidate) -> per item {'A': dict, 'B': dict} of gold first-content lp, margin_S, margin_N"""
    by={(m['bundle'],m['cond']):m for m in meta};na=by[(bi,'A')]['n_cands'];off=[0]+list(np.cumsum(na));out={}
    from e57_qabs_run import first_content
    for cond,base in (('A',0),('B',sum(na))):
        tk=by[(bi,cond)]['tok']
        for k,n in enumerate(na):
            tl=rows[base+int(off[k]):base+int(off[k])+n];ti=tk[k];assert len(tl)==n and all(len(a)==len(b) for a,b in zip(tl,ti)),'lp / token mismatch'
            S=np.asarray([float(np.sum(t)) for t in tl]);N=np.asarray([float(np.mean(t)) for t in tl]);w=np.arange(n)!=0
            out.setdefault(k,{})[cond]=dict(first=float(first_content(tl,ti,dec)[0]),S=float(S[0]-S[w].max()),N=float(N[0]-N[w].max()))
    return out
def deltas(st,w0):
    """state vs W0 -> {(bundle,item): tuple of MEAS}"""
    o={}
    for key,a in st.items():
        b=w0[key];dA={m:a['A'][m]-b['A'][m] for m in ('first','S','N')};dB={m:a['B'][m]-b['B'][m] for m in ('first','S','N')}
        o[key]=(dA['first'],dA['first']-dB['first'],dA['S'],dA['S']-dB['S'],dA['N'],dA['N']-dB['N'])
    return o
def analyze(a):
    import e57_pqa2_stream as S
    from scipy import stats
    out=pathlib.Path(a['dir']);tk=S.tokenizer();dec=lambda t:tk.decode(t);NB=a['nb'];seeds=a['seeds'];D={}
    for s in seeds:
        r=json.load(open(out/f'qabs_s{s}_reads.json'));meta=r['meta']
        w0={bi:item_scores(r['W0'][str(bi)],meta,bi,dec) for bi in range(NB)};w0={(bi,k):v for bi in w0 for k,v in w0[bi].items()}
        for arm in ARMS:
            for j in range(NB):
                st={};
                for bi in range(NB):
                    for k,v in item_scores(r[arm][str(NV*j+NV-1)][str(bi)],meta,bi,dec).items():st[(bi,k)]=v
                for key,d in deltas(st,w0).items():
                    lag=j-key[0];lag=lag if lag>=0 else -1;D.setdefault((arm,lag,s),[]).append(d)
            w0i={};
            for j in range(NB):
                w0j=item_scores(r['IN'][str(j)]['W0'],meta,j,dec);stj=item_scores(r['IN'][str(j)][arm],meta,j,dec)
                for key,d in deltas({(j,k):v for k,v in stj.items()},{(j,k):v for k,v in w0j.items()}).items():D.setdefault((arm,'in0',s),[]).append(d)
    res={'definition':__doc__.split('Statistic')[0].strip(),'lags':{}}
    for lag in sorted({k[1] for k in D},key=lambda x:(isinstance(x,str),str(x))):
        row={}
        for mi,m in enumerate(MEAS):
            v={arm:np.asarray([np.mean([x[mi] for x in D[(arm,lag,s)]]) for s in seeds]) for arm in ARMS};d=v['NATMASK']-v['NATS'];n=len(d);se=float(d.std(ddof=1)/np.sqrt(n));tq=float(stats.t.ppf(0.975,n-1))
            row[m]=dict(NATS=float(v['NATS'].mean()),NATMASK1024=float(v['NATMASK'].mean()),diff=float(d.mean()),ci95=[float(d.mean()-tq*se),float(d.mean()+tq*se)],per_seed=[float(x) for x in d],n_seeds=n)
        res['lags'][str(lag)]=row
    pathlib.Path(a['json']).write_text(json.dumps(res,indent=1))
    print('lag | '+' | '.join(MEAS))
    for lag,row in res['lags'].items():print(lag,'| '+' | '.join(f"{row[m]['diff']:+.2f} [{row[m]['ci95'][0]:+.2f},{row[m]['ci95'][1]:+.2f}]" for m in MEAS))
def selftest():
    dec=lambda t:{1:' ',2:' x',3:'y'}[t[0]]
    meta=[dict(bundle=0,cond='A',n_cands=[2],tok=[[[2,3],[2]]]),dict(bundle=0,cond='B',n_cands=[2],tok=[[[2,3],[2]]])]
    rows=[[-1.,-3.],[-2.]]*2;sc=item_scores(rows,meta,0,dec)[0]
    assert abs(sc['A']['S']-(-4.0-(-2.0)))<1e-9 and abs(sc['A']['N']-(-2.0-(-2.0)))<1e-9 and sc['A']['first']==-1.
    w=deltas({(0,0):sc},{(0,0):sc});assert all(abs(x)<1e-12 for x in w[(0,0)])
    print('e57_qabs_margin selftest ok')
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--dir');ap.add_argument('--seeds',default='101,102,103,104');ap.add_argument('--nb',type=int,default=6);ap.add_argument('--json');ap.add_argument('--selftest',action='store_true');x=ap.parse_args()
    if x.selftest:selftest();return
    analyze(dict(dir=x.dir,seeds=[int(s) for s in x.seeds.split(',')],nb=x.nb,json=x.json))
if __name__=='__main__':main()
