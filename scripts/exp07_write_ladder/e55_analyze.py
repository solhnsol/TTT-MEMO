"""E55 analysis (CPU, numpy only): pre-registered metrics and rules of E55-controlled-world-binding-design.md sections 3, 4 and 2.5 from the raw 16-candidate score matrices of e55_read records.
Unit = pair (the two names of a pair are averaged, worlds A and B averaged); several seeds of the same cell are concatenated (pairs of different worlds); CI = 95% pair bootstrap (4000 draws, seed 55).
Metrics (change vs W0, same items, same context): P1 = [log p(bound target) - mean log p over the 16 candidates] (nats), P2 = top-1 rate (chance 1/16 = 0.0625). Sub-metrics: rank, normalized probability, mass on the partner place, absolute target log-prob.
Records: see e55_read.py. Only EXPOSED items are used for P1 / P2; the 8 unexposed control names are reported separately (their gold = a never-shown place).
Rules implemented: 1 (readable?), 2 (binding present per cell), 3 (shadowing S(p) = metric(SWD) - metric(NAT) and the p = 9 negative control), 4 (ordering of S over p, signs only), 5 (MSK vs SWD), 6 (same-group share), 9 (window readout: W0 check, exit drop, write share). Rules 7 (information variables) and 8 (invalid-run conditions: unit test failures, identity-test mismatch, name overlap, clip fraction, canary) are read from the generic `contrast` table and from the run logs respectively and are NOT automated here. Thresholds are the ones fixed in the design (P2 +3 / +5 / +10 percentage points, 1/3 and 0.3 / 0.7 fractions, 20 pp for rule 9(i))."""
import json,sys,argparse
from pathlib import Path
import numpy as np
HERE=Path(__file__).resolve().parent;sys.path.insert(0,str(HERE))
import e49_stats as ES
CHANCE=1/16;NB=4000
def _ci(v,nb=NB):
    v=np.asarray(v,float);rng=np.random.default_rng(55);idx=rng.integers(0,len(v),(nb,len(v)));b=v[idx].mean(1);return float(v.mean()),*np.percentile(b,[2.5,97.5]).tolist()
def item_vals(SA,SB,gi,fi):
    """per item, mean over worlds A and B: P1, top1, rank, p_norm, mass_partner, abs_lp"""
    out={}
    for w,S in (('A',SA),('B',SB)):
        S=np.asarray(S,float);m=ES.item_metrics(S,gi,fi,w);tg=gi if w=='A' else fi;r=np.arange(len(S))
        out[w]={'P1':S[r,tg]-S.mean(1),'P2':m['top1'],'rank':m['rank'],'p_norm':m['p_norm'],'mass_partner':m['mass_partner'],'abs_lp':m['abs_lp'],'PM3':S[r,tg]-(S.sum(1)-S[r,tg])/(S.shape[1]-1)}
    return {k:(out['A'][k]+out['B'][k])/2 for k in out['A']}
def pair_vec(vals,meta,metric,mask=None):
    kind=np.asarray(meta['kind']);sel=(kind=='exposed') if mask is None else ((kind=='exposed')&np.asarray(mask))
    pairs=[p for p,s_ in zip(meta['pair'],sel) if s_];return ES.pair_means(vals[metric][sel],pairs)
def read_vals(rec,a,ctx):
    meta=rec['meta'];gi=np.asarray(meta['gi']);fi=np.asarray(meta['fi']);S=rec['read'][str(a)][ctx];return item_vals(S['A'],S['B'],gi,fi)
def w0_vals(rec,ctx,a=None):
    """W0 scores of the same items in the same context. 'out' (neutral tail) is one matrix shared by A and B; an in-window context depends on the world (names in the tail) and on the read point, so the GPU reader stores it as W0['<ctx>@<a>'] = {'A': S, 'B': S}"""
    meta=rec['meta'];gi=np.asarray(meta['gi']);fi=np.asarray(meta['fi']);W=rec['W0'];S=W[f'{ctx}@{a}'] if a is not None and f'{ctx}@{a}' in W else W[ctx]
    return item_vals(S['A'],S['B'],gi,fi) if isinstance(S,dict) else item_vals(S,S,gi,fi)
def change_vec(rec,a,ctx,metric,mask=None):
    """pair-level change vs W0 (same items, same context)"""
    v=read_vals(rec,a,ctx);v0=w0_vals(rec,ctx,a);d={k:v[k]-v0[k] for k in v};return pair_vec(d,rec['meta'],metric,mask)
def cat(recs,fn):return np.concatenate([fn(r) for r in recs])
def cell_row(recs,a=16,ctx='out'):
    return {m:_ci(cat(recs,lambda r:change_vec(r,a,ctx,m))) for m in ('P1','P2','PM3','rank','p_norm','mass_partner','abs_lp')}
def contrast(recs_x,recs_y,a=16,ctx='out',metric='P2',ctx_y=None,a_y=None):
    """pair-level paired difference X - Y of the change-vs-W0 metric (same worlds: records of the same seeds in the same order)"""
    dx=cat(recs_x,lambda r:change_vec(r,a,ctx,metric));dy=cat(recs_y,lambda r:change_vec(r,a_y or a,ctx_y or ctx,metric));assert dx.shape==dy.shape;return _ci(dx-dy)
def excl0(ci):return ci[1]>0 or ci[2]<0
def rule1(nat,swd,a=16):
    """readable? unreadable if in BOTH cells P2 < chance + 0.03 and the P1 CI contains 0"""
    bad=[]
    for r in (nat,swd):
        row=cell_row(r,a);bad.append(row['P2'][0]<CHANCE+0.03 and not excl0(row['P1']))
    return {'unreadable':all(bad),'per_cell_unreadable':bad}
def rule2(recs,a=16):
    row=cell_row(recs,a);return {'binding':bool(excl0(row['P1']) and row['P1'][0]>0 and row['P2'][0]>=CHANCE+0.05),'row':row}
def rule3(nat,swd,a=16):
    c2=contrast(swd,nat,a,metric='P2');c1=contrast(swd,nat,a,metric='P1')
    return {'S_P2':c2,'S_P1':c1,'shadowing':bool(excl0(c2) and c2[0]>=0.05 and excl0(c1)),'negctrl_pass':bool(abs(c2[0])<=0.02 and not excl0(c2))}
def rule4(S_by_p):
    """S_by_p: {p: S_P2 point estimate} for p in (1,2,4,9): only the sign of adjacent differences"""
    ps=sorted(S_by_p);return {'monotone_nonincreasing':all(S_by_p[ps[i]]>=S_by_p[ps[i+1]] for i in range(len(ps)-1)),'differences':[S_by_p[ps[i]]-S_by_p[ps[i+1]] for i in range(len(ps)-1)]}
def rule5(nat,swd,msk,a=16):
    s=contrast(swd,nat,a,metric='P2');m=contrast(msk,swd,a,metric='P2');third=abs(s[0])/3
    cls='consistent' if abs(m[0])<=third else ('mask adds a window-length effect' if m[0]>third else 'mask removes less than the swap')
    return {'S':s,'MSK_minus_SWD':m,'class':cls}
def rule6(nat,swd,swds,a=16):
    tot=contrast(swd,nat,a,metric='P2')[0];part=contrast(swds,nat,a,metric='P2')[0]
    if abs(tot)<1e-9:return {'share':None,'class':'no total effect'}
    sh=part/tot;return {'share':sh,'class':'same-fact chunks carry most' if sh>=0.7 else ('other-pair context carries most' if sh<=0.3 else 'not classified (value only)')}
def rule9(w,a=16):
    """window readout of one cell (corrected design 14:40 KST): the in-window read of every exposed item is taken in its own group's context, so d = p for the whole cell (rec['meta']['d'][a]['in'], None = control). Per cell: P2 / P1 in the window (W), out of the window (W, neutral tail), in the window at W0; exit drop = in - out, write share = W in - W0 in. (i) W0 check: only meaningful where d = 1 (the p = 1 cell): W0 in-window top-1 >= chance + 0.20. Distance curve = the cells' values across p (d = p)."""
    acc={'P2':{'W_in':[],'W0_in':[],'W_out':[]},'P1':{'W_in':[],'W0_in':[],'W_out':[]}};dset=set()
    for rec in w:
        key=str(a)
        if 'in' not in rec['read'].get(key,{}):continue
        d=rec['meta']['d'][key]['in'];m_=np.asarray([x is not None for x in d]);dset|={x for x in d if x is not None}
        vw=read_vals(rec,a,'in');vout=read_vals(rec,a,'out');v0=w0_vals(rec,'in',a)
        for met in acc:
            acc[met]['W_in'].append(pair_vec(vw,rec['meta'],met,m_));acc[met]['W0_in'].append(pair_vec(v0,rec['meta'],met,m_));acc[met]['W_out'].append(pair_vec(vout,rec['meta'],met,m_))
    if not dset:return {'d':None}
    assert len(dset)==1,('mixed distances in one cell',dset);res={'d':dset.pop()}
    for met,e in acc.items():
        wi,w0,wo=(np.concatenate(e[x]) for x in ('W_in','W0_in','W_out'))
        res[met]={'W_in':_ci(wi),'W0_in':_ci(w0),'W_out':_ci(wo),'exit_drop_W_in_minus_out':_ci(wi-wo),'write_share_W_in_minus_W0_in':_ci(wi-w0)}
    res['check_i_W0_top1']=res['P2']['W0_in'][0] if res['d']==1 else None
    res['interpretable']=(res['check_i_W0_top1']>=CHANCE+0.20) if res['d']==1 else None
    return res
def fmt_row(name,row):
    return f'| {name} | '+' | '.join('%+.3f [%+.3f, %+.3f]'%row[m] for m in ('P1','P2','rank','p_norm','mass_partner','abs_lp'))+' |'
def report(cells,a=16):
    L=['| cell | P1 (nats) | P2 (top-1 rate change; chance 0.0625) | rank change | p_norm change | mass on partner change | target log-prob change |','|'+'---|'*7]
    for n,recs in cells.items():L.append(fmt_row(n,cell_row(recs,a)))
    return '\n'.join(L)
# ------------------------------------------------------------------ synthetic records for the self test
def _rec(N,rng,signal=0.0,seed=0,with_d=False,inwin=0.0):
    """exposed names 2N + 8 controls, 16 candidates; the score of the bound target gets +signal (in A the gold, in B the partner); W0 has noise only"""
    n=2*N+8;kind=['exposed']*(2*N)+['ctrl']*8;pair=[f'p{i//2}' for i in range(2*N)]+[f'c{i//2}' for i in range(8)]
    gi=np.zeros(n,int);fi=np.ones(n,int);meta={'kind':kind,'pair':pair,'gi':gi.tolist(),'fi':fi.tolist(),'qframe':[i%3 for i in range(n)],'name_id':list(range(n))}
    def S(sig,d=None):
        A=rng.normal(size=(n,16));B=rng.normal(size=(n,16))
        for i in range(2*N):
            s_=sig if d is None else sig*(d[i] is not None and d[i]<=8)
            A[i,gi[i]]+=s_;B[i,fi[i]]+=s_
        return A,B
    W0={'out':rng.normal(size=(n,16))};rec={'meta':meta,'W0':W0,'read':{}}
    A,B=S(signal);rec['read']['16']={'out':{'A':A,'B':B}}
    if with_d:
        dd=with_d if isinstance(with_d,int) and with_d>1 else 1
        d=[dd]*(2*N)+[None]*8;meta['d']={'16':{'in':d}}
        W0['in@16']={'A':rng.normal(size=(n,16)),'B':rng.normal(size=(n,16))}
        for i in range(2*N):W0['in@16']['A'][i,gi[i]]+=inwin if dd<=2 else 0;W0['in@16']['B'][i,fi[i]]+=inwin if dd<=2 else 0
        A2=rng.normal(size=(n,16));B2=rng.normal(size=(n,16))
        for i in range(2*N):A2[i,gi[i]]+=(inwin+signal) if dd<=2 else 0;B2[i,fi[i]]+=(inwin+signal) if dd<=2 else 0
        rec['read']['16']['in']={'A':A2,'B':B2}
    return rec
def selftest():
    rng=np.random.default_rng(1);N=16
    nat=[_rec(N,rng,0.0,s) for s in range(3)];swd=[_rec(N,rng,3.0,s) for s in range(3)]
    r1=rule1(nat,swd);assert not r1['unreadable']   # SWD is clearly readable
    r1b=rule1(nat,[_rec(N,rng,0.0,s) for s in range(3)]);assert r1b['unreadable']
    assert rule2(swd)['binding'] and not rule2(nat)['binding']
    r3=rule3(nat,swd);assert r3['shadowing'] and r3['S_P2'][0]>0.2
    nat9=[_rec(72,rng,0.0,s) for s in range(3)];swd9=[_rec(72,rng,0.0,s) for s in range(3)];r39=rule3(nat9,swd9);assert r39['negctrl_pass'] and not r39['shadowing']
    assert rule4({1:.4,2:.3,4:.2,9:.0})['monotone_nonincreasing'] and not rule4({1:.1,2:.3})['monotone_nonincreasing']
    msk=[_rec(N,rng,3.0,s) for s in range(3)];assert rule5(nat,swd,msk)['class']=='consistent'
    half=[_rec(N,rng,1.5,s) for s in range(3)];r6=rule6(nat,swd,half);assert r6['class'] in ('not classified (value only)','same-fact chunks carry most','other-pair context carries most') and 0.2<r6['share']<0.8
    w=[_rec(N,rng,1.0,s,with_d=True,inwin=4.0) for s in range(3)];r9=rule9(w);assert r9['d']==1 and r9['interpretable'] and r9['P2']['exit_drop_W_in_minus_out'][0]>0.1 and r9['P2']['write_share_W_in_minus_W0_in'][1]<r9['P2']['write_share_W_in_minus_W0_in'][0]<r9['P2']['write_share_W_in_minus_W0_in'][2]
    w9=[_rec(N,rng,0.0,s,with_d=9,inwin=4.0) for s in range(3)];r99=rule9(w9);assert r99['d']==9 and r99['interpretable'] is None and abs(r99['P2']['exit_drop_W_in_minus_out'][0])<0.15
    print(report({'NAT':nat,'SWD':swd,'SWD@p9':swd9}))
    print('e55_analyze SELFTEST OK (synthetic scores)')
def stage0(nat_files,swd_files,a=16,out=None):
    """records of the stage-0 cells (e55_gpu.make_records, one file per cell and seed) -> text report: table of the changes vs W0, rules 1-3 and, when the records carry in-window reads, rule 9. No verdict wording beyond the pre-registered rule outputs."""
    load=lambda fs:[json.load(open(f)) for f in fs.split(',')];nat=load(nat_files);swd=load(swd_files) if swd_files else None;L=['# E55 stage 0 (%s) -- pre-registered rule outputs'%('NAT vs SWD' if swd else 'NAT only'),'',report({'NAT':nat,**({'SWD':swd} if swd else {})},a),'']
    if swd:r1=rule1(nat,swd,a);L.append(f'rule 1 (unreadable if BOTH cells have P2 < chance+0.03 and P1 CI contains 0): unreadable={r1["unreadable"]} per cell {r1["per_cell_unreadable"]}')
    for nm,r in (('NAT',nat),)+((('SWD',swd),) if swd else ()):x=rule2(r,a);L.append(f'rule 2 {nm}: binding present={x["binding"]}')
    if swd:x=rule3(nat,swd,a);L.append(f'rule 3 S = SWD - NAT: P2 {x["S_P2"][0]:+.3f} [{x["S_P2"][1]:+.3f}, {x["S_P2"][2]:+.3f}], P1 {x["S_P1"][0]:+.3f} [{x["S_P1"][1]:+.3f}, {x["S_P1"][2]:+.3f}] -> shadowing flag={x["shadowing"]}; negative control (|S|<=0.02 and CI contains 0)={x["negctrl_pass"]} (only meaningful for p=9)')
    if all('in' in r['read'].get(str(a),{}) for r in nat):
        x=rule9(nat,a);L.append('');L.append(f'rule 9 (window readout, NAT; every pair read in its own group context just before its next appearance, d = {x["d"]} chunks; d >= 9 = outside the 8-chunk window)')
        for met in ('P2','P1'):
            v=x[met];L.append(f'  {met}: in-window {v["W_in"][0]:+.3f} [{v["W_in"][1]:+.3f}, {v["W_in"][2]:+.3f}], out-of-window {v["W_out"][0]:+.3f}, W0 in-window {v["W0_in"][0]:+.3f}; exit drop (in - out) {v["exit_drop_W_in_minus_out"][0]:+.3f} [{v["exit_drop_W_in_minus_out"][1]:+.3f}, {v["exit_drop_W_in_minus_out"][2]:+.3f}]; write share (W in - W0 in) {v["write_share_W_in_minus_W0_in"][0]:+.3f} [{v["write_share_W_in_minus_W0_in"][1]:+.3f}, {v["write_share_W_in_minus_W0_in"][2]:+.3f}]')
        L.append(f'  W0 check (only at d = 1): top-1 {x["check_i_W0_top1"]} -> interpretable={x["interpretable"]}')
    t='\n'.join(L)+'\n'
    if out:Path(out).write_text(t)
    return t
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--selftest',action='store_true');ap.add_argument('--nat',default=None,help='comma list of NAT read records');ap.add_argument('--swd',default=None);ap.add_argument('--a',type=int,default=16);ap.add_argument('--out',default=None);a=ap.parse_args()
    if a.selftest:selftest()
    elif a.nat:print(stage0(a.nat,a.swd,a.a,a.out))
    else:print(__doc__)
