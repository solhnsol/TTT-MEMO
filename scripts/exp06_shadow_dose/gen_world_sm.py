"""Exp06 spaced/massed factorial worlds (CPU only). Reuses the Exp04 clean-v2 vocabulary, frames and background prose.
32 swap-pairs (64 triples), every triple exposed R times (one sentence per exposure; a pair's two sentences adjacent).
Condition per pair: SPACED (consecutive exposures of the pair > GAP_S tokens apart, i.e. outside the 8192 SWA window)
or MASSED (all R exposures inside one burst; consecutive exposures < GAP_M tokens apart, i.e. inside the window).
Variant v0: pairs 0..15 spaced, 16..31 massed; variant v1: roles swapped (same entities, same frames, same background).
Bursts are placed so that the mean stream position of massed and spaced exposures match (checked).
All exposures end > PROBE_SLACK tokens before the probe chunk (no in-window leak at the probe).
World B (swap=True) swaps the two person names of every pair, as in Exp04."""
import random,re,sys,json
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'exp04_what_ttt_retains'))
import gen_world as gw
TOK=gw.TOK;CH=1024
R=8                 # exposures per triple
NPAIRS=32
GAP_S=10240         # spaced: start-to-start distance between consecutive exposures of a pair (> 8192 + sentence length)
GAP_M=256           # massed: target distance between consecutive exposures inside a burst (burst ~ R*GAP_M = 2K)
PROBE_SLACK=9216    # last exposure ends at least this many tokens before the probe chunk start
def ntoks(s):return len(TOK.encode(s,add_special_tokens=False).ids)+1
def build(seed,variant):
    rng=random.Random(seed);nt=NPAIRS*2
    pn=gw.person_names(rng,nt*4);pl=gw.place_names(rng,nt*4)
    byl={};byp={}
    for n in pn:byl.setdefault(gw.ntok(n),[]).append(n)
    for n in pl:byp.setdefault(gw.ntok(n),[]).append(n)
    L=max(byl,key=lambda k:len(byl[k]));Lp=max(byp,key=lambda k:len(byp[k]));persons=byl[L];places=byp[Lp]
    assert len(persons)>=nt+8 and len(places)>=nt+8,(len(persons),len(places))
    rng.shuffle(persons);rng.shuffle(places)
    occs=(gw.OCC*3)[:nt]    # occupations repeat across pairs here (L2 not used as a primary readout)
    triples=[{'id':i,'pair':f'p{i//2}','X':persons[i],'P':places[i],'o':occs[i]} for i in range(nt)]
    cond={p:('spaced' if (p<NPAIRS//2)!=(variant==1) else 'massed') for p in range(NPAIRS)}
    fr={t['id']:rng.sample(gw.FR,len(gw.FR)) for t in triples}
    # ---- target token positions of every pair exposure ----
    span_s=R*GAP_S                                   # extent of one spaced schedule
    sched=[]
    sp=[p for p in range(NPAIRS) if cond[p]=='spaced'];ms=[p for p in range(NPAIRS) if cond[p]=='massed']
    off=rng.sample(range(len(sp)),len(sp))
    for i,p in enumerate(sp):
        s0=2048+off[i]*(GAP_S/len(sp))               # interleaved starts, one pair every GAP_S/16 tokens
        for r in range(R):sched.append((s0+r*GAP_S,r+1,p))
    # massed bursts spread evenly over the same extent -> matched mean position
    bstarts=[2048+(j+0.5)*span_s/len(ms)-R*GAP_M/2 for j in range(len(ms))];rng.shuffle(bstarts)
    for p,b in zip(ms,bstarts):
        for r in range(R):sched.append((b+r*GAP_M,r+1,p))
    sched.sort();bg=gw.background(rng);texts=[];ntok_=0;si=0;bi=0
    end_expo=None
    while True:
        if si<len(sched) and ntok_>=sched[si][0]:
            _,r,p=sched[si];si+=1;ts=[triples[2*p],triples[2*p+1]]
            if rng.random()<0.5:ts=ts[::-1]
            for t in ts:
                f=fr[t['id']][(r-1)%len(gw.FR)]
                texts.append({'kind':'expo','t':t['id'],'round':r,'cond':cond[p],'text':f.format(X='@X%d@'%t['id'],P=t['P'],o=t['o'],a_o=gw.A(t['o']))})
                ntok_+=ntoks(texts[-1]['text'].replace('@X','Xx').replace('@',''))
            if si==len(sched):end_expo=ntok_
        else:
            txt=bg[bi%len(bg)];bi+=1;texts.append({'kind':'bg','text':txt});ntok_+=ntoks(txt)
        if end_expo is not None and ntok_>=end_expo+PROBE_SLACK+2*CH:break
    w={'seed':seed,'variant':variant,'v':3,'triples':triples,'cond':cond,'ctrl_persons':persons[nt:nt+8],'ctrl_places':places[nt:nt+8],
       'texts':texts,'name_len':L,'place_len':Lp,'pairs':NPAIRS,'nch':None,'R':R}
    # probe chunk from the ACTUAL tokenization (both name orders): first chunk starting >= PROBE_SLACK after the last exposure
    last=0
    for sw in (False,True):
        text,_,spans=gw.realize(w,sw);e=TOK.encode(text,add_special_tokens=False);ends=[b for a,b in e.offsets]
        cend=max(sp['end'] for sp in spans)-1;last=max(last,1+next(i for i,b in enumerate(ends) if b>cend))
    w['nch']=-(-(last+PROBE_SLACK)//CH)
    while min(len(TOK.encode(gw.realize(w,sw)[0],add_special_tokens=False).ids)+1 for sw in (False,True))<(w['nch']+1)*CH:
        texts.append({'kind':'bg','text':bg[bi%len(bg)]});bi+=1
    return w
def audit(world,swap=False):
    """token-level checks: gaps per condition, mean positions, probe slack. Returns a dict; asserts the design."""
    text,name,spans=gw.realize(world,swap);e=TOK.encode(text,add_special_tokens=False);ends=[b for a,b in e.offsets]
    def ctok(c):return 1+next(i for i,b in enumerate(ends) if b>c)
    tri={t['id']:t for t in world['triples']};ev={}
    for s in spans:
        p=int(tri[s['t']]['pair'][1:]);a,b=ctok(s['start']),ctok(s['end']-1)
        ev.setdefault(p,{}).setdefault(s['round'],[]).append((a,b))
    gaps={'spaced':[],'massed':[]};pos={'spaced':[],'massed':[]};last=0
    for p,rr in ev.items():
        c=world['cond'][p];prev_end=None
        for r in sorted(rr):
            st=min(a for a,b in rr[r]);en=max(b for a,b in rr[r]);pos[c].append(st);last=max(last,en)
            if prev_end is not None:gaps[c].append(st-prev_end)
            prev_end=en
        assert len(rr)==world['R'],(p,len(rr))
    probe=world['nch']*CH
    out={'min_gap_spaced':min(gaps['spaced']),'max_gap_massed':max(gaps['massed']),'mean_pos_spaced':sum(pos['spaced'])/len(pos['spaced']),
         'mean_pos_massed':sum(pos['massed'])/len(pos['massed']),'last_exposure_end':last,'probe_chunk_start':probe,'probe_slack':probe-last,'n_tokens':len(e.ids)+1}
    assert out['min_gap_spaced']>8192,out;assert out['max_gap_massed']<2048,out;assert out['probe_slack']>8192,out
    assert abs(out['mean_pos_spaced']-out['mean_pos_massed'])<0.05*out['mean_pos_spaced'],out
    assert out['n_tokens']>=probe+CH,out
    return out
if __name__=='__main__':
    seed=int(sys.argv[1]) if len(sys.argv)>1 else 201
    for v in (0,1):
        w=build(seed,v)
        for sw in (False,True):print(f'seed {seed} variant {v} swap {sw}',json.dumps(audit(w,sw)))
