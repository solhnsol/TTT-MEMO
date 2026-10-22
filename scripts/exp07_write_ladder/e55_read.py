"""E55 readout: candidate scoring of the controlled world (CPU plan + record format; the GPU path at the bottom is NOT run anywhere yet).
Items (gen_world_ctl.build_items): one per name (2N exposed names + 8 unexposed control names), 3 question frames, 16 candidate places (gold, partner, 14 other world places), scores = summed span log-probs (e49 convention, packed fp32 readouts).
Readouts per read point (after appearance a = 2, 4, 8, 16 of every pair): 'out' = the neutral SQuAD tail (information outside the window, headline readout) and, only where asked (stage 0, stage 1, base world NAT; E55 2.5, rule 9), 'in_e<j>' = the context is the 8 chunks of the REAL stream of the state's own world that end at chunk e = r, r-1, r-3, r-7 (r = the chunk after which the state was saved), the question is appended after it; the same contexts are read at W0.
Information distance d of a pair in a context = how many chunks before the question the pair's latest appearance is (1 = the last context chunk); pairs whose latest appearance is more than 8 chunks back (or that has not appeared yet) are 'out' (d = inf); bins 1 / 2 / 3-4 / 5-8 / inf. d = 8 is a boundary case (the sentence sits at the window edge once the question tokens are added).
Record: see e55_gpu.py (ctx 'out' = neutral tail, ctx 'in' = every exposed item scored in ITS OWN group's context, d = p for all of them)."""
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
    """in-window contexts of the corrected design (E55 2.5, 14:40 KST): 'read just before the next appearance'. Group g (pairs i with i % p == g) gets ONE context of the 8 REAL stream chunks that end at e_g = the last chunk <= r that is the chunk just before a chunk of group g (e_g % p == (g-1) % p). The latest appearance of every pair of group g inside it is then exactly p chunks before the question (d = p; p = 1: d = 1, 2: 2, 4: 4, 9: 9 = outside the 8-chunk window). Each exposed pair is read only in its own group's context. -> (list of {'name':'in_g<g>','group':g,'end':e_g,'tail_chunks':[e-7..e]}, skipped groups whose context would start before chunk 0)"""
    p=world['p'];out=[];skipped=[]
    for g in range(p):
        e=r-((r-(g-1))%p)
        if e-7<0:skipped.append(g);continue
        out.append({'name':f'in_g{g}','group':g,'end':e,'tail_chunks':list(range(e-7,e+1))})
    return out,skipped
def latest_distance(world,g,end):
    """chunks between the latest appearance of group g's pairs at or before chunk `end` and the question (1 = the last context chunk); None if the group has not appeared yet"""
    p=world['p']
    if end<g:return None
    j=end-((end-g)%p);return end-j+1
def item_group(world,it):
    return None if it['kind']!='exposed' else int(it['pair'][1:])%world['p']
def tail_ids(real_chunks,ctx):
    t=[];[t.extend(real_chunks[c]['ids']) for c in ctx['tail_chunks']];assert len(t)==8*G.CH;return t
def item_meta(items):
    return {'kind':[i['kind'] for i in items],'pair':[i['pair'] for i in items],'gi':[i['gi'] for i in items],'fi':[i['fi'] for i in items],'qframe':[i['qframe'] for i in items],'name_id':[i['name_id'] for i in items]}
def plan(world,enc,which=('out','in')):
    """CPU: everything a reader needs for one world: items, rows, the in-window context per group at each read point with its tails, the items read in it (its own group's exposed items; the unexposed control items ride along with group 0) and d per item (= p for the exposed items)"""
    items=G.build_items(world);A_=G.realize(world,False,enc);B_=G.realize(world,True,enc);pl={'items':items,'rows':G.rows(items),'read':{}}
    for a,r in zip(G.READ_APPS,world['read_chunks']):
        cx,skipped=contexts(world,r);ent={'read_chunk':r,'skipped_groups':skipped,'ctx':{}}
        for c in cx:
            g=c['group'];idx=[i for i,it in enumerate(items) if item_group(world,it)==g or (g==0 and it['kind']!='exposed')]
            d=[latest_distance(world,g,c['end']) if (item_group(world,it)==g) else None for it in items]
            ent['ctx'][c['name']]={'group':g,'end':c['end'],'items':idx,'d':d,'bin_counts':{'p':sum(1 for i in idx if items[i]['kind']=='exposed')},'tail_A':tail_ids(A_,c),'tail_B':tail_ids(B_,c)}
        pl['read'][a]=ent
    return pl
# ----------------------------------------------------------------------- GPU path (UNRUN)
def gpu_read(args):  # pragma: no cover
    """NOT RUN. Reads saved states at read points with the out-of-window SQuAD tail and with the in-window real-stream tails, plus W0. Modelled on e49_candidates.py: ladder.Ladder gives the runtime, to_fp32 and load_state; ladder._TailCtx(rt32, tail_ids) takes any tail whose length is a multiple of 1024 (here 8192 = 8 chunks, the question rows follow in the packed chunk); packing.Packed(ctx, raw_rows, GP=2).r1(m32) scores the rows.
    Open points that need the server: (1) states of the E55 runner at the read chunks (the runner e55_run does not exist yet); (2) Ladder(rt, seed) builds the OLD world's Readouts, only its runtime and loaders are used here; (3) the SQuAD tail must contain none of the world's names (squad_tail forbid list = persons + places of the real, decoy and control sets)."""
    raise NotImplementedError('E55 GPU reader is not written yet (needs the runner states); see the docstring')
def selftest():
    enc=G.StubTok();bg=G._synth_bg(6000)
    for p in (1,2,4,9):
        w=G.build_ctl_world(55,8,1,p,16,'50',enc,bg);pl=plan(w,enc);assert len(pl['rows'])==16*(2*w['N']+8) and set(pl['read'])==set(G.READ_APPS)
        ent=pl['read'][16];r=ent['read_chunk'];assert r==16*p-1 and len(ent['ctx'])==p and not ent['skipped_groups']
        items=pl['items'];seen=set()
        for name,c in ent['ctx'].items():
            g=c['group'];e=c['end'];assert e<=r and e%p==(g-1)%p and len(c['tail_A'])==8192 and c['tail_A']!=c['tail_B']
            ex=[i for i in c['items'] if items[i]['kind']=='exposed'];assert len(ex)==2*8 and all(int(items[i]['pair'][1:])%p==g for i in ex)      # M = 8 pairs x 2 names per group
            assert all(c['d'][i]==p for i in ex) and all(c['d'][i] is None for i in range(len(items)) if i not in ex)
            seen|=set(ex)
            A_=G.realize(w,False,enc);assert c['tail_A'][:1024]==A_[e-7]['ids'] and c['tail_A'][-1024:]==A_[e]['ids']
            # the pair's latest appearance in the context is p chunks before the question; for p <= 8 it is inside the 8-chunk context, for p = 9 it is not
            ch=[x for x in range(e-7,e+1) if x%p==g];assert (len(ch)>0)==(p<=8) and (not ch or e-max(ch)+1==p)
        assert seen==set(i for i,it in enumerate(items) if it['kind']=='exposed')           # every exposed item is read exactly once
        ctl=[i for g in ent['ctx'].values() for i in g['items'] if items[i]['kind']!='exposed'];assert len(ctl)==8 and len(set(ctl))==8
    # a=2 (read chunk 2p-1): contexts that would start before chunk 0 are skipped
    w=G.build_ctl_world(55,8,1,2,16,'50',enc,bg);pl=plan(w,enc);assert pl['read'][2]['ctx']=={} and pl['read'][2]['skipped_groups']==[0,1]
    print('e55_read SELFTEST OK (CPU plan only; GPU reader in e55_gpu.py)')
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--selftest',action='store_true');a=ap.parse_args()
    if a.selftest:selftest()
