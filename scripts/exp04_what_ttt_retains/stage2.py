"""Stage 2 (dose-response entity memory with matched counterfactual worlds) and Stage 3 (retention value). Needs Runtime."""
import json,re,time,sys
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parent))
import gen_world as gw
from core import CH,clone_state,jax,jnp,eqx
R=48     # retained-slot token budget (identical in every condition)
NCH=gw.NCH
STAGES=gw.ROUNDS   # probe chunk -> cumulative exposure rounds per entity
FILLER_SENT="The road wound on through the fields and the long afternoon passed without any news of consequence to anyone."
def enc(tok,s):return tok.encode(s,add_special_tokens=False)
def item_ids(tok,slot_ids,prefix,cand):
    text="\n\n"+prefix+cand;e=enc(tok,text);a=len("\n\n"+prefix)
    pos=[i for i,(s,t) in enumerate(e.offsets) if t>a and s<len(text)]
    assert pos and tok.decode([e.ids[i] for i in pos]).strip()==cand.strip(),(prefix,cand,pos)
    ids=list(slot_ids)+e.ids;off=len(slot_ids)
    return ids,(off+pos[0],off+pos[-1]+1)
def make_slot(tok,keep_text):
    fill=enc(tok,(FILLER_SENT+' ')*6).ids
    if keep_text is None:return fill[:R]
    k=enc(tok,' '+keep_text).ids[:R];return fill[:R-len(k)]+k
class Probe:
    def __init__(self,rt,world,swap,probe_world=None):
        self.rt=rt;self.tok=rt.tok;self.swap=swap;pw=probe_world or world;self.world=pw
        text,self.stream_name,spans=gw.realize(world,swap)
        _,self.name,pspans=gw.realize(pw,False)   # canonical (world-A) names/labels for the probes: IDENTICAL held-out inputs in every history
        e=enc(self.tok,text);ids=[rt.bos]+e.ids
        assert len(ids)>=NCH*CH+1,len(ids)
        self.ids=ids[:NCH*CH]
        ends=[b for a,b in e.offsets]
        def ctok(c):return 1+next(i for i,b in enumerate(ends) if b>c)  # token holding char c (BOS offset +1)
        self.spans=[dict(s,tok_start=ctok(s['start']),tok_end=ctok(s['end']-1)) for s in spans]
        self.items=gw.probes(pw,self.name);self.tri={t['id']:t for t in pw['triples']}
        # --- verification: last exposure of EVERY entity is outside the SWA window of every probe position ---
        self.audit={}
        for k,rounds in STAGES.items():
            sp=[s for s in self.spans if s['round']<=rounds]
            last_end=max(s['tok_end'] for s in sp);limit=(k+1)*CH-(8192-CH)   # first position visible to the last probe token
            assert last_end<limit,(k,last_end,limit)
            # per-entity recency inside swap pairs
            lastpt={};cnt={}
            for s in sp:lastpt[s['t']]=max(lastpt.get(s['t'],0),s['tok_end']);cnt[s['t']]=cnt.get(s['t'],0)+1
            assert set(cnt.values())=={rounds},cnt
            if world.get('v',2)>=2:   # later rounds must lie entirely AFTER the probe chunk start (exposure counts are exact, no in-window leak)
                assert all(x['tok_start']>=(k+1)*CH for x in self.spans if x['round']>rounds),'later-round exposure before probe chunk'
            self.audit.setdefault('_early_later_round_sentences',{})[k]=sum(1 for x in self.spans if x['round']>rounds and x['tok_start']<(k+1)*CH)
            diffs=[abs(lastpt[2*p]-lastpt[2*p+1]) for p in range(world['pairs'])]
            self.audit[k]={'rounds':rounds,'last_exposure_end_token':last_end,'probe_chunk_start':(k+1)*CH,'min_gap_to_window_start':limit-last_end,
                           'distance_last_exposure_to_probe_start':(k+1)*CH-last_end,'pair_recency_diff_tokens_mean':float(np.mean(diffs)),'pair_recency_diff_tokens_max':int(max(diffs)),'exposures_per_entity':rounds}
        # token-length control: all person names / places have equal token length in both worlds
        lens={len(enc(self.tok,' '+n).ids) for n in self.name.values()}|{len(enc(self.tok,' '+t['P']).ids) for t in pw['triples']}
        self.audit['token_lengths']=sorted(lens);assert len(lens)==1,lens
        by={}
        for s in pspans:by.setdefault(s['t'],[]).append(s['text'])
        self.expo=by   # retention candidates are always the TRUE (world-A) sentences of the probed world
    def keep_texts(self,item):
        """candidate retention spans for the item's query (all padded/cut to R tokens by make_slot)."""
        t=self.tri[item['tid']];pr=[x for x in self.world['triples'] if x['pair']==t['pair']];a,b=pr
        other=[x for x in self.world['triples'] if x['pair']!=t['pair']][(t['id']*7)%(len(self.world['triples'])-2)]
        c={'NONE':None,
           'BIND':self.expo[t['id']][0],                                   # exposure sentence of the queried entity (name+place+occupation)
           'BIND_PARTNER':self.expo[b['id'] if a['id']==t['id'] else a['id']][0],  # binding sentence of the swap partner (same pair, wrong binding)
           'NAME':f"{self.name[a['id']]}, {self.name[b['id']]}.",            # names only
           'PLACE':f"{a['P']}, {b['P']}.",                                   # places only
           'OCC':f"the {a['o']}, the {b['o']}.",                             # occupations only
           'DISTRACT':self.expo[other['id']][0]}                             # binding sentence of an unrelated entity
        return c
    def score(self,m,st,k,item,keep_text):
        tail=self.ids[(k+1)*CH-(8192-CH):(k+1)*CH];slot=make_slot(self.tok,keep_text);out=[];n=[]
        for cand in (item['gold'],item['foil']):
            probe_ids,span=item_ids(self.tok,slot,item['prefix'],cand);assert len(probe_ids)<=CH
            full=tail+probe_ids+[self.rt.eos]*(CH-len(probe_ids))
            b=len(tail)   # span indices are relative to probe_ids; score_probe_full wants absolute indices inside the 8192-token window
            ent,first=self.rt.score_probe_full(m,st,full,[(span[0]+b,span[1]+b),(span[0]+b,span[0]+b+1)]);out.append((ent,first));n.append(span[1]-span[0])
        return out,n
def delta_norms(rt,m):
    out={};flat_m=jax.tree_util.tree_flatten_with_path(m.inner_parameters())[0];flat_b=jax.tree_util.tree_flatten_with_path(rt.base.inner_parameters())[0]
    for (p,a),(_,b) in zip(flat_m,flat_b):
        if eqx.is_inexact_array(a):out[jax.tree_util.keystr(p)]=float(jnp.linalg.norm((a-b).astype(jnp.float32)))
    return out
def save_snapshot(rt,m,path):
    flat_m=jax.tree_util.tree_flatten_with_path(m.inner_parameters())[0];flat_b=jax.tree_util.tree_flatten_with_path(rt.base.inner_parameters())[0]
    arrs={}
    for (p,a),(_,b) in zip(flat_m,flat_b):
        if eqx.is_inexact_array(a):arrs[jax.tree_util.keystr(p).replace('.','_').replace('[','_').replace(']','')]=np.asarray((a-b).astype(jnp.float16))
    np.savez(path,**arrs)
def run_world(rt,world,swap,on,tag,outdir,retention=False,snapshot=False,probe_world=None,name=None,kv_mode='carry'):
    """one stream through world A/B; probes at the three dose snapshots. retention=True -> Stage 3 pass (final snapshot only)."""
    outdir=Path(outdir);outdir.mkdir(parents=True,exist_ok=True)
    kind='ret' if retention else ('main' if kv_mode=='carry' else 'kvempty')
    f=outdir/f'{tag}_{kind}_{name or ("B" if swap else "A")}_{"ON" if on else "OFF"}.json'
    if f.exists():return False
    P=Probe(rt,world,swap,probe_world);rows=[];norms={};t0=time.monotonic()
    def hook(k):
        def h(rt_,m,st):
            t1=time.monotonic()
            if on:norms[k]=delta_norms(rt,m)
            if on and snapshot and not retention:save_snapshot(rt,m,outdir/f'{tag}_{"B" if swap else "A"}_snapshot_k{k}_rounds{STAGES[k]}.npz')
            if kv_mode=='empty':st=clone_state(rt.state_suffix)   # weights-only readout: no cached keys/values at all
            if not retention:
                for it in P.items:
                    (g,fo),(ng,nf)=P.score(m,st,k,it,None);rows.append({'k':k,'rounds':STAGES[k],'cond':'NONE','level':it['level'],'pair':it['pair'],'tid':it['tid'],'gold':g[0],'foil':fo[0],'gold1':g[1],'foil1':fo[1],'ng':ng,'nf':nf})
            else:
                for it in P.items:
                    if it['level'] not in ('L2','L3','L3r'):continue
                    for cname,ktxt in P.keep_texts(it).items():
                        (g,fo),(ng,nf)=P.score(m,st,k,it,ktxt);rows.append({'k':k,'rounds':STAGES[k],'cond':cname,'level':it['level'],'pair':it['pair'],'tid':it['tid'],'gold':g[0],'foil':fo[0],'gold1':g[1],'foil1':fo[1],'ng':ng,'nf':nf})
            print('  probes',tag,kind,'B' if swap else 'A','ON' if on else 'OFF','k',k,round(time.monotonic()-t1,1),flush=True)
        return h
    ks=[max(STAGES)] if retention else sorted(STAGES)
    lp,_=rt.stream(P.ids,on,hooks={k:hook(k) for k in ks})
    json.dump({'tag':tag,'swap':swap,'on':on,'kind':kind,'stream_seed':world['seed'],'probe_seed':(probe_world or world)['seed'],'rows':rows,'chunk_nll':[float(-lp[i*CH:(i+1)*CH].mean()) for i in range(NCH)],'audit':{str(k):v for k,v in P.audit.items()},
               'delta_norms':{str(k):v for k,v in norms.items()},'sec':time.monotonic()-t0,'name_map':P.stream_name},open(f,'w'))
    return True
def run_cal(rt,world,swap,on,tag,outdir):
    """source-present readout calibration: all rounds' exposures (480 sentences) are NOT used; one round of every entity + probe right
    after, inside the local window. Tests that the readout/probe path can bind at all when the source is visible."""
    outdir=Path(outdir);outdir.mkdir(parents=True,exist_ok=True)
    f=outdir/f'{tag}_cal_{"B" if swap else "A"}_{"ON" if on else "OFF"}.json'
    if f.exists():return False
    P=Probe(rt,world,swap);tok=rt.tok
    sents=[s['text'] for s in P.spans if s['round']==1]   # 30 sentences, one per entity
    body=enc(tok,'\n\n'.join(sents)).ids;assert len(body)<=7167,len(body)
    fill=enc(tok,(FILLER_SENT+' ')*400).ids
    ids=[rt.bos]+fill[:7167-len(body)]+body;assert len(ids)==7168
    P.ids=ids;rows=[];t0=time.monotonic()
    def h(rt_,m,st):
        for it in P.items:
            (g,fo),(ng,nf)=P.score(m,st,6,it,None);rows.append({'k':6,'rounds':1,'cond':'NONE','level':it['level'],'pair':it['pair'],'tid':it['tid'],'gold':g[0],'foil':fo[0],'gold1':g[1],'foil1':fo[1],'ng':ng,'nf':nf})
    lp,_=rt.stream(ids,on,hooks={6:h})
    json.dump({'tag':tag,'swap':swap,'on':on,'rows':rows,'sec':time.monotonic()-t0},open(f,'w'));return True
