"""E55 readout: candidate scoring of the controlled world (CPU plan + record format; the GPU path at the bottom is NOT run anywhere yet).
Items (gen_world_ctl.build_items): one per name (2N exposed names + 8 unexposed control names), 3 question frames, 16 candidate places (gold, partner, 14 other world places), scores = summed span log-probs (e49 convention, packed fp32 readouts).
Readouts per read point (after appearance a = 2, 4, 8, 16 of every pair): 'out' = the neutral SQuAD tail (information outside the window, headline readout) and, only where asked (stage 0, stage 1, base world NAT; E55 2.5, rule 9), 'in_e<j>' = the context is the 8 chunks of the REAL stream of the state's own world that end at chunk e = r, r-1, r-3, r-7 (r = the chunk after which the state was saved), the question is appended after it; the same contexts are read at W0.
Information distance d of a pair in a context = how many chunks before the question the pair's latest appearance is (1 = the last context chunk); pairs whose latest appearance is more than 8 chunks back (or that has not appeared yet) are 'out' (d = inf); bins 1 / 2 / 3-4 / 5-8 / inf. d = 8 is a boundary case (the sentence sits at the window edge once the question tokens are added).
Record (json, one per cell and seed): {'cell','arm','p','M','s','bank','seed','items': meta, 'read': {'<a>': {'<ctx>': {'A': S, 'B': S}}}, 'W0': {'<ctx>': S}}; S = (n_items, 16) scores. 'W0' is read once per world and shared by all arms."""
import os,sys,json,argparse
from pathlib import Path
import numpy as np
HERE=Path(__file__).resolve().parent;sys.path.insert(0,str(HERE));sys.path.insert(0,str(HERE.parents[0]/'exp04_what_ttt_retains'))
import gen_world_ctl as G
BINS=['1','2','3-4','5-8','inf']
def bin_of(d):
    if d is None or d>8:return 'inf'
    return '1' if d==1 else '2' if d==2 else '3-4' if d<=4 else '5-8'
def contexts(world,r):
    """-> list of {'name','end','tail_chunks':[e-7..e]} for the read chunk r (contexts that would start before chunk 0 are skipped and listed)"""
    out=[];skipped=[]
    for lag in (0,1,3,7):
        e=r-lag
        if e-7<0:skipped.append(lag);continue
        out.append({'name':f'in_e{lag}','end':e,'tail_chunks':list(range(e-7,e+1))})
    return out,skipped
def info_distance(world,items,end):
    """d per item for a context ending at chunk `end` (exposed items only; control items get None)"""
    p=world['p'];d=[]
    for it in items:
        if it['kind']!='exposed':d.append(None);continue
        g=int(it['pair'][1:])%p
        if end<g:d.append(None);continue
        j=end-((end-g)%p);d.append(end-j+1)
    return d
def tail_ids(real_chunks,ctx):
    t=[];[t.extend(real_chunks[c]['ids']) for c in ctx['tail_chunks']];assert len(t)==8*G.CH;return t
def item_meta(items):
    return {'kind':[i['kind'] for i in items],'pair':[i['pair'] for i in items],'gi':[i['gi'] for i in items],'fi':[i['fi'] for i in items],'qframe':[i['qframe'] for i in items],'name_id':[i['name_id'] for i in items]}
def plan(world,enc,which=('out','in')):
    """CPU: everything a reader needs for one world: items, rows, the contexts per read point with their tails and distances, bin counts"""
    items=G.build_items(world);A_=G.realize(world,False,enc);B_=G.realize(world,True,enc);pl={'items':items,'rows':G.rows(items),'read':{}}
    for a,r in zip(G.READ_APPS,world['read_chunks']):
        cx,skipped=contexts(world,r);ent={'read_chunk':r,'skipped_lags':skipped,'ctx':{}}
        for c in cx:
            d=info_distance(world,items,c['end']);cnt={b:0 for b in BINS}
            for x in d:
                if x is not None:cnt[bin_of(x)]+=1
            ent['ctx'][c['name']]={'end':c['end'],'d':d,'bin_counts':cnt,'tail_A':tail_ids(A_,c),'tail_B':tail_ids(B_,c)}
        pl['read'][a]=ent
    return pl
# ----------------------------------------------------------------------- GPU path (UNRUN)
def gpu_read(args):  # pragma: no cover
    """NOT RUN. Reads saved states at read points with the out-of-window SQuAD tail and with the in-window real-stream tails, plus W0. Modelled on e49_candidates.py: ladder.Ladder gives the runtime, to_fp32 and load_state; ladder._TailCtx(rt32, tail_ids) takes any tail whose length is a multiple of 1024 (here 8192 = 8 chunks, the question rows follow in the packed chunk); packing.Packed(ctx, raw_rows, GP=2).r1(m32) scores the rows.
    Open points that need the server: (1) states of the E55 runner at the read chunks (the runner e55_run does not exist yet); (2) Ladder(rt, seed) builds the OLD world's Readouts, only its runtime and loaders are used here; (3) the SQuAD tail must contain none of the world's names (squad_tail forbid list = persons + places of the real, decoy and control sets)."""
    raise NotImplementedError('E55 GPU reader is not written yet (needs the runner states); see the docstring')
def selftest():
    enc=G.StubTok();bg=G._synth_bg(6000);w=G.build_ctl_world(55,8,1,2,16,'50',enc,bg);pl=plan(w,enc)
    assert len(pl['rows'])==16*(2*w['N']+8) and set(pl['read'])==set(G.READ_APPS)
    ent=pl['read'][16];assert ent['read_chunk']==31 and set(ent['ctx'])=={'in_e0','in_e1','in_e3','in_e7'}
    # p = 2: the latest appearance of a pair is at lag 0 (group 31%2) or lag 1 (the other group): d = 1 or 2 in the context ending at 31; N=16 pairs, 8 per group
    c=ent['ctx']['in_e0'];assert c['bin_counts']['1']==16 and c['bin_counts']['2']==16 and c['bin_counts']['inf']==0 and sum(1 for x in c['d'] if x is None)==8
    c1=ent['ctx']['in_e1'];assert c1['bin_counts']['1']==16 and c1['bin_counts']['2']==16   # one chunk earlier: groups swap roles, same counts
    assert len(c['tail_A'])==8192 and c['tail_A']!=c['tail_B']
    # tail = the real stream chunks e-7..e, in order
    A_=G.realize(w,False,enc);assert c['tail_A'][:1024]==A_[24]['ids'] and c['tail_A'][-1024:]==A_[31]['ids']
    # read after appearance 2 (chunk 3) has fewer than 8 chunks: every in-window context is skipped
    assert pl['read'][2]['ctx']=={} and pl['read'][2]['skipped_lags']==[0,1,3,7]
    # p = 1: contexts end at e, e-1, e-3, e-7 and give d = 1, 2, 4, 8 for every pair? (all pairs appear in every chunk -> d = 1 in all)
    w1=G.build_ctl_world(55,8,1,1,16,'50',enc,bg);p1=plan(w1,enc);assert all(p1['read'][16]['ctx'][k]['bin_counts']['1']==16 for k in p1['read'][16]['ctx'])
    # p = 9: pair groups appear every 9 chunks: for the context ending at the read chunk 143 (group 143%9=8) the other groups are 1..8 chunks back or absent
    w9=G.build_ctl_world(55,8,1,9,16,'50',enc,bg);p9=plan(w9,enc);c9=p9['read'][16]['ctx']['in_e0'];assert c9['bin_counts']=={'1':16,'2':16,'3-4':32,'5-8':64,'inf':16},c9['bin_counts']   # items = 2 names per pair; group 8 d=1, 7 d=2, 6-5 d=3-4, 4-1 d=5-8, group 0 d=9 (outside)
    print('e55_read SELFTEST OK (CPU plan only; GPU reader not written)')
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--selftest',action='store_true');a=ap.parse_args()
    if a.selftest:selftest()
