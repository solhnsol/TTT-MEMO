"""Stage 2R: forgetting vs interference after the last related exposure.
Exposure phase (rounds 1-4, chunks 0-6, ON updates) is identical in every condition. Then a 21-chunk gap phase:
  NOUPD   : unrelated filler, KV carried but NO inner updates (pure token distance, weights frozen)
  UNREL   : unrelated filler with normal inner updates (unrelated-update interference)
  RELSAME : filler + 3 extra rounds of CONSISTENT re-exposures at chunks 8-10 with updates (related reinforcement)
  RELCONF : filler + 3 rounds of CONFLICTING (swap-partner) exposures at chunks 8-10 with updates (related interference)
Probes (identical held-out inputs, world-A labels, L2 and L3) after chunk 18 and 28, always >7168 tokens after the last related
text. Compare conditions at the same token distance; the number of updates since the last exposure differs by design."""
import json,re,random,time,sys
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parent))
import gen_world as gw,stage2
from stage2 import Probe,enc
from core import CH,jax
NCH_R=29
PROBE_K=(18,28)
def build_text(seed,cond):
    w=gw.build_world(seed);rng=random.Random(seed*7919+13);bg=gw.background(rng);np_=w['pairs']
    sched=[]
    for r in range(1,5):                      # rounds 1-4 over chunks 0-6
        lo=(r-1)*(7*CH/4)
        for p in range(np_):sched.append((lo+rng.random()*(7*CH/4)*0.9,('base',r),p))
    if cond in ('RELSAME','RELCONF'):
        for r in range(5,8):                  # three related rounds over chunks 8-10
            lo=8*CH+(r-5)*CH
            for p in range(np_):sched.append((lo+rng.random()*CH*0.9,('rel',r),p))
    sched.sort();fr={t['id']:rng.sample(gw.FR,len(gw.FR)) for t in w['triples']}
    texts=[];si=0;ntoks=0;bi=0
    while ntoks<NCH_R*CH+2500:
        if si<len(sched) and ntoks>=sched[si][0]:
            _,(kind,r),p=sched[si];si+=1;ts=[w['triples'][2*p],w['triples'][2*p+1]]
            if rng.random()<0.5:ts=ts[::-1]
            for t in ts:
                partner=w['triples'][2*p+1] if t['id']==2*p else w['triples'][2*p]
                name_id=partner['id'] if (cond=='RELCONF' and kind=='rel') else t['id']
                f=fr[t['id']][(r-1)%len(gw.FR)]
                txt=f.format(X='@X%d@'%name_id,P=t['P'],o=t['o'],a_o=gw.A(t['o']))
                texts.append({'kind':'expo','t':t['id'],'round':r,'rel':kind=='rel','text':txt})
                ntoks+=len(gw.TOK.encode(txt.replace('@X','Xx').replace('@',''),add_special_tokens=False).ids)+1
        else:
            txt=bg[bi%len(bg)];bi+=1;texts.append({'kind':'bg','text':txt});ntoks+=len(gw.TOK.encode(txt,add_special_tokens=False).ids)+1
    w2=dict(w);w2['texts']=texts
    return w,w2
class DynProbe(Probe):
    def __init__(self,rt,seed,cond):
        w,w2=build_text(seed,cond)
        super().__init__(rt,w,False)       # reuse items/labels/keep texts from the canonical world
        # replace the stream by the dynamics text
        text,_,spans=gw.realize(w2,False);e=enc(self.tok,text);ids=[rt.bos]+e.ids
        assert len(ids)>=NCH_R*CH+1;self.ids=ids[:NCH_R*CH]
        ends=[b for a,b in e.offsets]
        ctok=lambda c:1+next(i for i,b in enumerate(ends) if b>c)
        sp=[dict(s,tok_end=ctok(s['end']-1)) for s in spans]
        self.dyn_audit={}
        for k in PROBE_K:
            limit=(k+1)*CH-(8192-CH)
            orig=max(s['tok_end'] for s in sp if s['round']<=4);rel=[s['tok_end'] for s in sp if s['round']>4]
            last_any=max([orig]+rel)
            assert last_any<limit,(k,last_any,limit)
            self.dyn_audit[k]={'last_original_exposure_end':orig,'last_related_text_end':last_any,'probe_chunk_start':(k+1)*CH,'swa_slack':limit-last_any,
                               'distance_last_original_exposure_to_probe':(k+1)*CH-orig,'distance_last_related_text_to_probe':(k+1)*CH-last_any,'updates_since_last_original_exposure':k-int(orig//CH)}
        self.dyn_items=[it for it in self.items if it['level'] in ('L2','L3')]
def run_dyn(rt,seed,cond,outdir):
    outdir=Path(outdir);outdir.mkdir(parents=True,exist_ok=True);f=outdir/f'seed{seed}_dyn_{cond}.json'
    if f.exists():return False
    P=DynProbe(rt,seed,cond);rows=[];t0=time.monotonic()
    mask=[True]*NCH_R
    if cond=='NOUPD':mask=[k<7 for k in range(NCH_R)]   # exposure phase is chunks 0-6 (+ chunk 7 partly filler) -> freeze from chunk 7
    def hook(k):
        def h(rt_,m,st):
            for it in P.dyn_items:
                (g,fo),(ng,nf)=P.score(m,st,k,it,None);rows.append({'k':k,'cond':cond,'level':it['level'],'pair':it['pair'],'tid':it['tid'],'gold':g[0],'foil':fo[0],'gold1':g[1],'foil1':fo[1],'ng':ng,'nf':nf})
            print('  dyn probes',seed,cond,k,flush=True)
        return h
    lp,_=rt.stream(P.ids,True,hooks={k:hook(k) for k in PROBE_K},update_mask=mask)
    json.dump({'seed':seed,'cond':cond,'rows':rows,'audit':{str(k):v for k,v in P.dyn_audit.items()},'chunk_nll':[float(-lp[i*CH:(i+1)*CH].mean()) for i in range(NCH_R)],'sec':time.monotonic()-t0},open(f,'w'));return True
