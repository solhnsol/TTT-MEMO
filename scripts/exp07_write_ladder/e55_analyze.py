"""E55 report. Thin wrapper over e55_metrics (design 10.3 / 4c) for the k >= 16 cells; plus the existence-check report (stage 0', k = 8 < 16, p = 1: candidates = the k exposed places, ranks normalised u = (K - rank) / (K - 1), chance 0.5; familiarity = mean score of the never-exposed places minus mean score of the exposed non-gold places). Records are the JSON files of `e55_run.py read` (format: e55_read.py docstring); several seeds are given as a comma list and pooled over facts.
  python3 e55_analyze.py --nat <nat.read.json[,...]> [--swd <swd.read.json[,...]>] [--a 16] [--out report.md]
  python3 e55_analyze.py --selftest"""
import sys,argparse,json
from pathlib import Path
import numpy as np
HERE=Path(__file__).resolve().parent;sys.path.insert(0,str(HERE))
import e55_metrics as MT
def _items(rec,a,ctx):
    """per-item arrays for the existence check: u (W), u (W0), familiarity (W, W0), kind mask, valid mask"""
    m=rec['meta'];n=len(m['kind']);ex=np.asarray(m['cand_exposed'],bool);gold=np.asarray(m['gold_idx']);K=int(m['n_rank'][0])
    def full(S):
        S=np.asarray(S,float)
        if len(S)==n:return S
        F=np.full((n,16),np.nan);F[m['item_idx'][str(a)][ctx]]=S;return F
    S=full(rec['read'][str(a)][ctx]);W0=rec['W0'];S0=full(W0[f'{ctx}@{a}'] if f'{ctx}@{a}' in W0 else W0[ctx]);r=np.arange(n)
    def stats(X):
        sg=X[r,gold];rank=1+((X>sg[:,None])&ex).sum(1);u=(K-rank)/(K-1)
        exn=ex.copy();exn[r,gold]=False;fam=np.where((~ex).any(1),(np.where(~ex,X,0).sum(1)/np.maximum((~ex).sum(1),1)),np.nan)-(np.where(exn,X,0).sum(1)/np.maximum(exn.sum(1),1));return u,fam
    u,f=stats(S);u0,f0=stats(S0);valid=~np.isnan(S).any(1)
    if ctx!='out':
        d=m['d'][str(a)][ctx];valid&=np.array([x is not None and 1<=x<=8 for x in d])
    return {'u':u,'u0':u0,'fam':f,'fam0':f0,'fact':(np.asarray(m['kind'])=='fact')&valid,'ctrl':(np.asarray(m['kind'])!='fact')&valid}
def existence(nat_files,swd_files=None,a_list=(4,16),out=None):
    load=lambda fs:[json.load(open(f)) for f in fs.split(',')];nat=load(nat_files);swd=load(swd_files) if swd_files else None
    L=['# E55 existence check (stage 0-prime): K = %d exposed candidates, u = (K - rank)/(K - 1) (chance 0.5; gain = u(W) - u(W0)); %d seeds'%(int(nat[0]['meta']['n_rank'][0]),len(nat)),'']
    def cat(recs,a,ctx,key,kind='fact'):return np.concatenate([_items(r,a,ctx)[key][_items(r,a,ctx)[kind]] for r in recs])
    cells={'NAT':nat,**({'SWD':swd} if swd else {})};ctxs=['out']+sorted(c for c in nat[0]['read'].get(str(max(a_list)),{}) if c!='out')
    L+=['| cell | read | context | gain in u [95% CI] | u(W) | u(W0) | familiarity change (unexposed - exposed non-gold, nats) [95% CI] | control items u (fake gold; expect ~0.5) | per-seed gain in u |','|'+'---|'*9]
    for nm,recs in cells.items():
        for a in a_list:
            for ctx in (['out'] if a!=max(a_list) else ctxs):
                if str(a) not in recs[0]['read'] or ctx not in recs[0]['read'][str(a)]:continue
                g=cat(recs,a,ctx,'u')-cat(recs,a,ctx,'u0');fam=cat(recs,a,ctx,'fam')-cat(recs,a,ctx,'fam0');cu=cat(recs,a,ctx,'u','ctrl')
                ps=[float((_items(r,a,ctx)['u']-_items(r,a,ctx)['u0'])[_items(r,a,ctx)['fact']].mean()) for r in recs]
                L.append(f'| {nm} | appearance {a} | {ctx} | {MT.fmt(MT.ci(g))} | {cat(recs,a,ctx,"u").mean():.3f} | {cat(recs,a,ctx,"u0").mean():.3f} | {MT.fmt(MT.ci(fam))} | {cu.mean():.3f} (n={len(cu)}) | {", ".join("%+.3f"%x for x in ps)} |')
    if swd:
        a=max(a_list);gN=cat(nat,a,'out','u')-cat(nat,a,'out','u0');gS=cat(swd,a,'out','u')-cat(swd,a,'out','u0')
        L+=['',f'shadowing contrast SWD - NAT at appearance {a}, out-of-window, gain in u (paired by fact): {MT.fmt(MT.ci(gS-gN))}']
    L+=['','Existence: gain in u > 0 with the CI excluding 0 at the last appearance (and the control items near 0.5) = the binding exists in this setting; the familiarity column separates "knows these places" from "knows which place goes with the name".']
    t='\n'.join(L)+'\n'
    if out:Path(out).write_text(t)
    return t
def is_existence(files):return json.load(open(files.split(',')[0]))['meta'].get('n_rank',[16])[0]<16
def curve(nat,alts,out_apps=None):
    """binding curve (seeds with the full per-chunk read): per appearance G_rank / PM2 / PM3 of every arm (out-of-window), contrasts alt - NAT (paired by fact), and the in-window readout at the appearances that have one. nat: records of NAT, alts: {name: records}; only records whose 'read' holds more than 4 appearances."""
    full=lambda rs:[r for r in rs if len(r['read'])>4];nf=full(nat)
    if not nf:return ''
    apps=sorted(int(x) for x in nf[0]['read']);L=['','## binding curve (seed(s) with the full read: %s), out-of-window read after every appearance'%(','.join(str(r['meta']['seed']) for r in nf)),'','| appearance | arm | G_rank [95% CI] | PM2 | PM3 | contrast vs NAT: G_rank |','|---|---|---|---|---|---|']
    for a in apps:
        if 'out' not in nf[0]['read'][str(a)]:continue
        t=MT.cell_table(nf,a,'out');L.append(f'| {a} | NAT | {MT.fmt(t["G_rank"])} | {MT.fmt(t["PM2"])} | {MT.fmt(t["PM3"])} | |')
        for nm,rs in alts.items():
            rf=[r for r in full(rs) if r['meta']['seed'] in [x['meta']['seed'] for x in nf]]
            if rf:t=MT.cell_table(rf,a,'out');c=MT.contrast(rf,[x for x in nf if x['meta']['seed'] in [y['meta']['seed'] for y in rf]],a,'out');L.append(f'| {a} | {nm} | {MT.fmt(t["G_rank"])} | {MT.fmt(t["PM2"])} | {MT.fmt(t["PM3"])} | {MT.fmt(c["G_rank"])} |')
    L+=['','in-window readout (contexts shorter than 8 chunks while a < 8; decision column = exit drop rank(out) - rank(in)):']
    for nm,rs in (('NAT',nf),)+tuple((k,full(v)) for k,v in alts.items()):
        for a in apps:
            ins=sorted(c for c in rs[0]['read'][str(a)] if c!='out') if rs else []
            if ins:
                w=MT.window_effect(rs,a,ins);L.append(f'  {nm} a={a}: exit drop {MT.fmt(w["exit_rank"])}, G_in-G_out {MT.fmt(w["G_in_minus_G_out"])}, W0 in-window rank {MT.fmt(w["W0_in_rank"])}'+(' [metric mismatch: no call]' if w['metric_mismatch'] else ''))
    return '\n'.join(L)+'\n'
def w0_report(recs,a=16,ctx='out'):
    """W0 (before training) rank of the gold place per question frame: histogram over ranks 1..16, mean, and the ceiling rate (W0 rank 1: no rank gain possible) / floor rate (W0 rank 16: no rank loss possible), pooled over the records; facts only"""
    rk=[];qf=[]
    for r in recs:
        it=MT.rec_items(r,a,ctx);m=MT._sel(r,a,ctx);rk+=list(it['rank_W0'][m]);qf+=list(np.asarray(r['meta']['qframe'])[m])
    rk=np.asarray(rk);qf=np.asarray(qf);L=['','## W0 rank of the gold place per question frame (before any update; chance mean 8.5)','','| question frame | n | mean rank W0 | ceiling (rank 1) | floor (rank 16) | histogram rank 1..16 |','|---|---|---|---|---|---|']
    for q in [None]+sorted(set(qf.tolist())):
        x=rk if q is None else rk[qf==q];h=np.bincount(x.astype(int),minlength=17)[1:17]
        L.append(f"| {'all' if q is None else q} | {len(x)} | {x.mean():.2f} | {np.mean(x==1):.3f} | {np.mean(x==16):.3f} | {' '.join(map(str,h))} |")
    return '\n'.join(L)+'\n'
def train_summary(saved_files):
    """per arm and seed from the saved.json logs: mean pre-clip gradient norm, mean clip coefficient c (1 = not clipped), share of clipped chunks, mean / cumulative update norm, mean CE (the per-chunk values are in the log of the file)"""
    L=['','## training statistics per arm and seed (from the per-chunk logs)','','| arm | seed | chunks | mean pre-clip gnorm | mean clip coefficient c | clipped chunks | mean update norm | cumulative update norm | mean CE |','|---|---|---|---|---|---|---|---|---|']
    for f in saved_files.split(','):
        r=json.load(open(f));lg=r['log'];g=lambda k:np.asarray([e[k] for e in lg],float)
        L.append(f"| {Path(f).name.split('_')[2]} | {r.get('world_seed')} | {len(lg)} | {g('gnorm').mean():.4g} | {g('coef').mean():.3f} | {np.mean(g('coef')<1):.2f} | {g('update_norm').mean():.4g} | {g('update_norm').sum():.4g} | {g('ce').mean():.3f} |")
    return '\n'.join(L)+'\n'
def stage0(nat_files,swd_files=None,a=16,out=None,swb_files=None,saved=None):
    if is_existence(nat_files):return existence(nat_files,swd_files,(4,16),out)
    load=lambda fs:[json.load(open(f)) for f in fs.split(',')]
    t=MT.stage0(nat_files,swd_files,a,None)
    if swb_files:
        nat=load(nat_files);swb=load(swb_files);sh=[r for r in nat if r['meta']['seed'] in [x['meta']['seed'] for x in swb]];sb=[x for x in swb if x['meta']['seed'] in [r['meta']['seed'] for r in sh]]
        s_=MT.shadowing(sh,sb,a);t+=f"\nSWD-B (fact sentences removed, background fill; seeds {[r['meta']['seed'] for r in sb]}) - NAT: G_rank {MT.fmt(s_['S']['G_rank'])}, PM2 {MT.fmt(s_['S']['PM2'])}, PM3 {MT.fmt(s_['S']['PM3'])} -> {s_['shadowing']}"+(' (METRIC MISMATCH)' if s_['metric_mismatch'] else '')+'\n'
    t+=w0_report(load(nat_files),a)
    t+=curve(load(nat_files),{**({'SWD-A':load(swd_files)} if swd_files else {}),**({'SWD-B':load(swb_files)} if swb_files else {})})
    if saved:t+=train_summary(saved)
    if out:Path(out).write_text(t)
    return t
def selftest():
    rng=np.random.default_rng(1);K=8;nf=8
    def rec(sig,fam=0.0):
        n=nf+4;gold=rng.integers(0,K,n);ex=np.array([[True]*K+[False]*8]*n);meta={'kind':['fact']*nf+['ctrl']*4,'gold_idx':gold.tolist(),'cand_exposed':ex.tolist(),'n_rank':[K]*n,'d':{'16':{'in_g0':[1]*nf+[None]*4}},'item_idx':{'16':{'in_g0':list(range(n))}}}
        base=rng.normal(size=(n,16));W=base+0.2*rng.normal(size=(n,16));W[np.arange(nf),gold[:nf]]+=sig;W[:,8:]+=fam
        return {'meta':meta,'read':{'4':{'out':W},'16':{'out':W,'in_g0':W}},'W0':{'out':base,'in_g0@16':base}}
    import tempfile,os
    d=tempfile.mkdtemp();fs=[]
    for i,sig in enumerate((3.0,3.0,3.0,3.0)):
        r=rec(sig);r['read']={k:{c:np.asarray(v).tolist() for c,v in cs.items()} for k,cs in r['read'].items()};r['W0']={c:np.asarray(v).tolist() for c,v in r['W0'].items()};fn=f'{d}/s{i}.json';json.dump(r,open(fn,'w'));fs.append(fn)
    t=existence(','.join(fs),','.join(fs));assert 'shadowing contrast' in t and '| NAT | appearance 16 | out |' in t
    print(t.splitlines()[3]);print('e55_analyze SELFTEST OK (existence report on synthetic records)')
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--selftest',action='store_true');ap.add_argument('--nat');ap.add_argument('--swd',default=None);ap.add_argument('--swb',default=None);ap.add_argument('--saved',default=None,help='comma list of *.saved.json (training statistics table)');ap.add_argument('--a',type=int,default=16);ap.add_argument('--out',default=None);a=ap.parse_args()
    if a.selftest:selftest()
    else:print(stage0(a.nat,a.swd,a.a,a.out,a.swb,a.saved))
