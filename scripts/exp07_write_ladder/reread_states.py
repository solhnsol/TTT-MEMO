"""E34 step 1: re-read stored states with the standard fp32 packed readouts (R1 neutral tail, R1-alt SQuAD) AND a new same-frame readout family. Reads only; no training; ladder.py untouched.
Usage: reread_states.py --seed 101 --state LABEL=PATH [...] [--saved LABEL=saved.json[:INDEX]] [--k-frames 4]
  PATH = an npz written by Ladder.save_state (leaves arr_0..), e.g. states/L0fs_fskeep_s101/A/state_001.npz (native 1-pass final, state_000 = W0) or states/L1_ck_s101/A/epoch_008.npz.
  Label pairs NAME_A / NAME_B are combined into D (world A - world B, pair bootstrap). 'W0' (rt.base) is always evaluated and is the baseline for familiarity.
Families (all read with R1 and R1-alt): the standard ones from Ladder.evaluate (L3, XP, PS, L3r, L2, FN, FP, FN0, FP0) plus
  L3same_seen / L3same_new / XPsame_seen: exposure-frame prefixes cut right before the place (or the person name if it comes first), gold = the triple's own place/name,
  foil = its swap-pair partner's (L3same), XPsame = same prefixes with gold/foil from the next pair (cross-pair null). seen = frames this triple really read, new = frames it did not read.
Output: OUT/e34_reread_s{seed}.json (per-state margins, D per family with CI)."""
import os,sys,json,argparse,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
import ladder as LD
from ladder import *
ap=argparse.ArgumentParser();ap.add_argument('--seed',type=int,default=101);ap.add_argument('--state',action='append',default=[]);ap.add_argument('--saved',action='append',default=[]);ap.add_argument('--k-frames',type=int,default=4)
ap.add_argument('--nboot',type=int,default=4000);ap.add_argument('--tag',default='');a=ap.parse_args()
OUT=LD.OUT
def main():
    assert 'deterministic_ops=true' in os.environ.get('XLA_FLAGS','')
    rt=Runtime();L=LD.Ladder(rt,a.seed,readout=True);T=L.T;w=T.w;name=T.P[False].name;tok=T.tok;tr={t['id']:t for t in w['triples']};npairs=w['pairs']
    # ---- same-frame items ----
    seen={};first_round={}
    for t_ in w['texts']:
        if t_['kind']=='expo':seen.setdefault(t_['t'],{}).setdefault(t_['text'],t_['round'])
    def cut(text,t):
        ph='@X%d@'%t['id'];xp=text.find(ph);pp=text.find(t['P'])
        if xp<0 or pp<0:return None
        if xp<pp:return ('P',text[:pp].replace(ph,name[t['id']]).rstrip())   # target = the place
        return ('X',text[:xp].rstrip())                                      # target = the person (place occurs before X in this frame)
    byp={}
    for t in w['triples']:byp.setdefault(t['pair'],[]).append(t)
    partner={t['id']:next(u for u in byp[t['pair']] if u['id']!=t['id']) for t in w['triples']}
    def mk(frames_of,fam_l3,fam_xp):
        l3=[];xp=[]
        for t in w['triples']:
            for text in frames_of(t):
                c=cut(text,t)
                if c is None:continue
                kind,prefix=c;b=partner[t['id']];j=t['id']%2;p=t['id']//2;q=(p+1)%npairs;ga=tr[2*q+j];gb=tr[2*q+1-j]
                val=(lambda u:' '+u['P']) if kind=='P' else (lambda u:' '+name[u['id']])
                l3.append({'level':fam_l3,'pair':t['pair'],'tid':t['id'],'prefix':prefix,'gold':val(t),'foil':val(b)})
                if fam_xp:xp.append({'level':fam_xp,'pair':f'x{p}','tid':t['id'],'prefix':prefix,'gold':val(ga),'foil':val(gb)})
        return l3,xp
    K=a.k_frames
    def seen_frames(t):
        d=seen.get(t['id'],{});return [x for x,_ in sorted(d.items(),key=lambda kv:(kv[1],kv[0]))][:K]
    def new_frames(t):
        allf=[f.format(X='@X%d@'%t['id'],P=t['P'],o=t['o'],a_o=gw.A(t['o'])) for f in gw.FR];s=set(seen.get(t['id'],{}));return [x for x in allf if x not in s][:K]
    l3s,xps=mk(seen_frames,'L3same_seen','XPsame_seen');l3n,_=mk(new_frames,'L3same_new',None)
    fams={'L3same_seen':l3s,'XPsame_seen':xps,'L3same_new':l3n}
    for f,its in fams.items():
        for it in its:
            assert it['gold']!=it['foil'] and it['gold'].strip() not in it['prefix'],(f,it)
            gi=len(tok.encode(it['gold'],add_special_tokens=False).ids);fi=len(tok.encode(it['foil'],add_special_tokens=False).ids);assert gi==fi,(f,it,gi,fi)
    print({f:len(v) for f,v in fams.items()},flush=True)
    def raw_rows(items):
        rows=[]
        for it in items:
            for c in (it['gold'],it['foil']):
                ids,span=stage2.item_ids(tok,stage2.make_slot(tok,None),it['prefix'],c);rows.append((ids,(span[0]-1,span[1]-1)))
        return rows
    allitems=[it for v in fams.values() for it in v];rows=raw_rows(allitems)
    with L.rt32.mesh,jax.default_matmul_precision('highest'):
        Psame=packing.Packed(L.ctx,rows,GP=2);Psame_alt=packing.Packed(L.ctx_alt,rows,GP=2)
    def read(m):
        m32=L.to_fp32(m);ev=L.evaluate(m)
        with jax.default_matmul_precision('highest'):r1=Psame.r1(m32);ra=Psame_alt.r1(m32)
        out={'ev':ev,'same':{'R1':{},'alt':{}}};i=0
        for f,its in fams.items():
            n=len(its);out['same']['R1'][f]=(r1[0::2]-r1[1::2])[i:i+n].tolist();out['same']['alt'][f]=(ra[0::2]-ra[1::2])[i:i+n].tolist();i+=n
        return out
    states={'W0':rt.base}
    for s in a.state:lab,p=s.split('=',1);states[lab]=p
    for s in a.saved:
        lab,rest=s.split('=',1);fn,_,ix=rest.partition(':');log=json.load(open(fn))['log'];fl=[e['margins']['_file'] for e in log if isinstance(e.get('margins'),dict) and '_file' in e['margins']];states[lab]=fl[int(ix) if ix else -1]
    R={};t0=time.monotonic()
    for lab,p in states.items():
        m=p if lab=='W0' else L.load_state(p);R[lab]=read(m);print(lab,'read',f'{time.monotonic()-t0:.0f}s',flush=True)
    # ---- D (world A - world B) per pair, bootstrap over pairs ----
    rng=np.random.default_rng(34);items_of=dict(L.fam);items_of.update(fams)
    def ci(v):
        v=np.asarray(v);b=[v[rng.integers(0,len(v),len(v))].mean() for _ in range(a.nboot)];return float(v.mean()),*np.percentile(b,[2.5,97.5]).tolist()
    def margins(r,ro,f):return np.asarray(r['same'][ro][f]) if f in r['same'][ro] else np.asarray(r['ev']['_alt'][f] if ro=='alt' else r['ev'][f])
    def pairmean(f,x):
        d={}
        for it,v in zip(items_of[f],x):d.setdefault(it['pair'],[]).append(v)
        ks=sorted(d,key=lambda s:int(s[1:]));return np.asarray([np.mean(d[k]) for k in ks])
    D={};names=sorted({l[:-2] for l in R if l.endswith('_A') and l[:-2]+'_B' in R})
    fl=['L3','XP','PS','L3r','L2']+list(fams)
    for nm in names:
        D[nm]={}
        for ro in ('R1','alt'):
            Dd={}
            for f in fl:
                if f not in items_of or (f not in fams and f not in R[nm+'_A']['ev'] ):continue
                Dd[f]=pairmean(f,(margins(R[nm+'_A'],ro,f)-margins(R[nm+'_B'],ro,f))/2)
            out={f:ci(v) for f,v in Dd.items()}
            if 'L3' in Dd and 'XP' in Dd and len(Dd['L3'])==len(Dd['XP']):out['L3-XP']=ci(Dd['L3']-Dd['XP'])
            if 'L3same_seen' in Dd and 'XPsame_seen' in Dd and len(Dd['L3same_seen'])==len(Dd['XPsame_seen']):out['L3same_seen-XPsame_seen']=ci(Dd['L3same_seen']-Dd['XPsame_seen'])
            D[nm][ro]=out
        print(nm,json.dumps({ro:{k:[round(x,3) for x in v] for k,v in o.items() if k in ('L3','XP','L3-XP','L3same_seen','L3same_new','L3same_seen-XPsame_seen')} for ro,o in D[nm].items()}),flush=True)
    json.dump({'seed':a.seed,'states':{k:(v if isinstance(v,str) else 'W0') for k,v in states.items()},'k_frames':K,'n_items':{f:len(v) for f,v in fams.items()},'D_mean_ci95':D,'raw':R},open(OUT/f'e34_reread_s{a.seed}{a.tag}.json','w'))
if __name__=='__main__':main()
