"""E57 synthetic exposure curve (new file; CPU; QA thread, Hans '1번만: 합성 n별 재읽기'): from the per-token reads of the states NATQ-like (M3NMW1N05 = no mask, natural window, loss weight 1, Adam direction, update norm 0.5 x NAT) and ALLQ256 (M3W1N05) saved at the exposures a = 9 / 17 / 33 / 65 / 128 per fact: per a and arm the first content token dA (state minus W0), d(A-B) (B = the same place string under the questions of the other facts), the masking extra ALLQ - NATQ, the first-token and branch-token rank ratio and its improvement (W0 minus state), top-1, the second content token and the mean of the remaining tokens; 95% bootstrap CIs over the facts; reproduction of the a = 128 values against the stored reads of job 676 (and the end-state hashes, written by the job).
  python3 e57_syn_ncurve_report.py --seed 64|61 --tag synqn --tokread F --ref676 F --out J | --selftest"""
import sys,json,pathlib,argparse
import numpy as np
HERE=pathlib.Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
import e57_syn_tokread as TR
import e57_pqa2_stream as S
def content(tl,ti,dec):
    ci=[j for j,t in enumerate(ti) if dec([t]).strip()!='']
    return np.asarray(tl,np.float64)[ci]
def second_rest(tlist,tokids,meta,dec,ncand=16):
    """per fact: A second / rest log-prob of the gold place and B (the same string under the other fact questions)"""
    kind=np.asarray(meta['kind']);fact=np.asarray(meta['fact']);cf=np.asarray(meta['cand_fact']);gi=np.asarray(meta['gi']);rows=[r for r in range(len(kind)) if kind[r]=='fact'];out={}
    for r in rows:
        f=int(fact[r]);g=content(tlist[r*ncand+int(gi[r])],tokids[r*ncand+int(gi[r])],dec);Bs=[];Br=[]
        for q in rows:
            if q==r:continue
            for c in range(ncand):
                if int(cf[q,c])==f:
                    x=content(tlist[q*ncand+c],tokids[q*ncand+c],dec)
                    if len(x)>1:Bs.append(x[1])
                    if len(x)>2:Br.append(x[2:].mean())
        out[f]=dict(A_second=float(g[1]) if len(g)>1 else None,A_rest=float(g[2:].mean()) if len(g)>2 else None,B_second=float(np.mean(Bs)) if Bs else None,B_rest=float(np.mean(Br)) if Br else None)
    return out
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--seed',type=int);ap.add_argument('--tag',default='synqn');ap.add_argument('--tokread');ap.add_argument('--ref676');ap.add_argument('--out');ap.add_argument('--selftest',action='store_true');a=ap.parse_args()
    if a.selftest:
        dec=lambda t:{1:' ',2:'x',3:'y'}[t[0]];assert list(content([-1,-2,-3],[1,2,3],dec))==[-2,-3];print('e57_syn_ncurve_report selftest ok');return
    tk=S.tokenizer();dec=lambda t:tk.decode(t);d=json.load(open(a.tokread));meta=json.load(open(next(iter(TR.arms_curve(a.seed,a.tag).values()))['read_json']))['meta'];tokids=d['tokids'];W0=TR.metrics(d['W0'],tokids,meta);facts=sorted(W0);W0s=second_rest(d['W0'],tokids,meta,dec)
    M={l:TR.metrics(t,tokids,meta) for l,t in d['arms'].items()};SR={l:second_rest(t,tokids,meta,dec) for l,t in d['arms'].items()};labels=sorted(M,key=lambda l:(l.split('@c')[0],int(l.split('@c')[1])));res={'seed':a.seed,'reproduction':d['reproduction'],'curve':{},'extra':{},'arms':sorted({l.split('@c')[0] for l in M})}
    def dd(m,w0,key,bkey):
        ok=[f for f in facts if m[f][bkey] is not None and w0[f][bkey] is not None];return ok,np.asarray([m[f][key]-w0[f][key] for f in ok]),np.asarray([m[f][bkey]-w0[f][bkey] for f in ok])
    def ddsr(m,w0,key,bkey):
        ok=[f for f in facts if None not in (m[f][key],w0[f][key],m[f][bkey],w0[f][bkey])];return ok,np.asarray([m[f][key]-w0[f][key] for f in ok]),np.asarray([m[f][bkey]-w0[f][bkey] for f in ok])
    vals={}
    for l in labels:
        m=M[l];e={'rr_first':TR.boot([m[f]['rank_first']/16.0 for f in facts]),'rr_branch':TR.boot([m[f]['rank_branch']/16.0 for f in facts]),'top1_first':TR.boot([float(m[f]['rank_first']==1) for f in facts]),'top1_branch':TR.boot([float(m[f]['rank_branch']==1) for f in facts])}
        e['rr_first_improvement']=TR.boot([(W0[f]['rank_first']-m[f]['rank_first'])/16.0 for f in facts]);e['rr_branch_improvement']=TR.boot([(W0[f]['rank_branch']-m[f]['rank_branch'])/16.0 for f in facts])
        ok,dA,dB=dd(m,W0,'A_first','B_first');e['first_dA']=TR.boot(dA);e['first_dA_minus_dB']=TR.boot(dA-dB);vals[(l,'first_dA')]=(ok,dA);vals[(l,'first_dAB')]=(ok,dA-dB)
        for nm in ('second','rest'):
            ok2,dA2,dB2=ddsr(SR[l],W0s,f'A_{nm}',f'B_{nm}');e[f'{nm}_dA']=TR.boot(dA2);e[f'{nm}_dA_minus_dB']=TR.boot(dA2-dB2);vals[(l,f'{nm}_dA')]=(ok2,dA2)
        vals[(l,'rr_first_improvement')]=(facts,np.asarray([(W0[f]['rank_first']-m[f]['rank_first'])/16.0 for f in facts]));vals[(l,'rr_branch_improvement')]=(facts,np.asarray([(W0[f]['rank_branch']-m[f]['rank_branch'])/16.0 for f in facts]))
        res['curve'].setdefault(l.split('@c')[0],{})[l.split('@c')[1]]=e
    nat='M3NMW1N05';allq='M3W1N05'
    for ch in sorted({int(l.split('@c')[1]) for l in labels}):
        lx,ly=f'{nat}@c{ch}',f'{allq}@c{ch}'
        if lx in M and ly in M:
            x={}
            for k in ('first_dA','first_dAB','second_dA','rest_dA','rr_first_improvement','rr_branch_improvement'):
                (okx,vx),(oky,vy)=vals[(lx,k)],vals[(ly,k)];assert okx==oky;x[k]=TR.boot(vy-vx)
            res['extra'][str(ch)]=x
    if a.ref676:
        o=json.load(open(a.ref676));rep={}
        for new,old in (('M3NMW1N05','NATQ'),('M3W1N05','ALLQ256')):
            for l in labels:
                if l.startswith(new+'@c') and old in o['arms'] and int(l.split('@c')[1]) in (127,511):rep[f'{new}@{l.split("@c")[1]} vs {old}']=float(max(abs(np.asarray(x)-np.asarray(y)).max() for x,y in zip(d['arms'][l],o['arms'][old]) if len(x)==len(y)))
        res['reproduction_vs_676_max_abs_token_logprob_diff']=rep
    pathlib.Path(a.out).write_text(json.dumps(res,indent=1,default=float));print('wrote',a.out)
if __name__=='__main__':main()
