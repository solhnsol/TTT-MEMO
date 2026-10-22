"""E55 readout plan (pairless fact world, design section 10.2; CPU plan + record format; the GPU path is e55_gpu.py / e55_run.py read and is NOT run anywhere yet).
Items (gen_facts.build_items): one per fact (kind 'fact', question frame = fact id mod 3) + 4 control items (kind 'ctrl'); 16 candidate places (gold + 15 places of other facts, scored as summed span log-probs, e49 convention, packed fp32 readouts).
Readouts per read point (after appearance a = 2, 4, 8, 16 of every fact, chunk a p - 1): 'out' = the neutral SQuAD tail padded to 8192 tokens (information outside the window; all items), and, only where asked (stage 0: a = 16), one in-window context per group g: 'in_g<g>' = the 8 REAL stream chunks that end at e_g = the last chunk <= r that is the chunk just before a chunk of group g ("read just before the next appearance"), the question is appended after it. The latest appearance of every fact of group g inside it is then exactly p chunks before the question: d = p (p = 1: 1, 2: 2, 4: 4, 9: 9 = outside the 8-chunk window = negative control). Each fact is read only in its own group's context (and in 'out'); the control items ride along with group 0. The same contexts are read at W0.
Record (e55_gpu.make_records): {'meta': item_meta + {k,n,p,bank,seed,arm,d,item_idx}, 'read': {str(a): {ctx: S}}, 'W0': {'out': S, 'in_g<g>@<a>': S}} with S (n_rows, 16); rows of an in-window context are the items listed in meta['item_idx'][str(a)][ctx] (e55_metrics fills the others with NaN). meta['d'][str(a)][ctx] = d per item (None = not read in that context)."""
import sys,json,argparse
from pathlib import Path
import numpy as np
HERE=Path(__file__).resolve().parent;sys.path.insert(0,str(HERE));sys.path.insert(0,str(HERE.parents[0]/'exp04_what_ttt_retains'))
import gen_facts as G
def contexts(world,r):
    """in-window contexts: group g (facts with id % p == g) gets ONE context of the 8 REAL stream chunks that end at e_g = the last chunk <= r with e % p == (g-1) % p, i.e. the chunk just before a chunk of group g (d = p). -> (list of {'name':'in_g<g>','group':g,'end':e_g,'tail_chunks':[e-7..e]}, skipped groups whose context would start before chunk 0)"""
    p=world['p'];out=[];skipped=[]
    for g in range(p):
        e=r-((r-(g-1))%p)
        if e<0:skipped.append(g);continue
        out.append({'name':f'in_g{g}','group':g,'end':e,'tail_chunks':list(range(max(0,e-7),e+1))})     # early in the stream the window is shorter than 8 chunks (the context starts at chunk 0)
    return out,skipped
def latest_distance(world,g,end):
    """chunks between the latest appearance of group g at or before chunk `end` and the question (1 = the last context chunk); None if the group has not appeared yet"""
    p=world['p']
    if end<g:return None
    j=end-((end-g)%p);return end-j+1
def tail_ids(real_chunks,ctx):
    t=[];[t.extend(real_chunks[c]['ids']) for c in ctx['tail_chunks']];assert len(t)==len(ctx['tail_chunks'])*G.CH and len(t)<=8*G.CH;return t
def plan(world,enc=None):
    """CPU: items, rows, and per read point the in-window context of every group (tail ids, item indices, d per item)"""
    items=G.build_items(world);A_=G.realize(world);pl={'items':items,'rows':G.rows(items),'read':{}}
    for a,r in zip(world['read_apps'],world['read_chunks']):
        cx,skipped=contexts(world,r);ent={'read_chunk':r,'skipped_groups':skipped,'ctx':{}}
        for c in cx:
            g=c['group'];idx=[i for i,it in enumerate(items) if it['group']==g or (g==0 and it['kind']=='ctrl')]
            d=[latest_distance(world,g,c['end']) if it['group']==g else None for it in items]
            ent['ctx'][c['name']]={'group':g,'end':c['end'],'items':idx,'d':d,'tail':tail_ids(A_,c),'window_chunks':len(c['tail_chunks'])}
        pl['read'][a]=ent
    return pl
def selftest():
    enc=G.StubTok();bg=G._synth_bg(9000)
    for cell in ('p1','base','p4','p9'):
        w=G.build_fact_world(55,enc=enc,bg=bg,**G.CELLS[cell]);p=w['p'];pl=plan(w);assert len(pl['rows'])==16*(w['k']+G.NCTRL) and set(pl['read'])==set(G.READ_APPS)
        ent=pl['read'][16];r=ent['read_chunk'];assert r==16*p-1 and len(ent['ctx'])==p and not ent['skipped_groups']
        items=pl['items'];seen=set();A_=G.realize(w)
        for name,c in ent['ctx'].items():
            g=c['group'];e=c['end'];assert e<=r and e%p==(g-1)%p and len(c['tail'])==8192 and c['window_chunks']==8
            ex=[i for i in c['items'] if items[i]['kind']=='fact'];assert len(ex)==w['k']//p and all(items[i]['group']==g for i in ex)
            assert all(c['d'][i]==p for i in ex) and all(c['d'][i] is None for i in range(len(items)) if i not in ex)
            assert c['tail'][:1024]==A_[e-7]['ids'] and c['tail'][-1024:]==A_[e]['ids']
            ch=[x for x in range(e-7,e+1) if x%p==g];assert (len(ch)>0)==(p<=8) and (not ch or e-max(ch)+1==p)         # latest appearance p chunks before the question; p = 9: none inside the context
            seen|=set(ex)
        assert seen==set(i for i,it in enumerate(items) if it['kind']=='fact')                  # every fact is read exactly once in-window
        ctl=[i for g in ent['ctx'].values() for i in g['items'] if items[i]['kind']=='ctrl'];assert len(ctl)==4 and len(set(ctl))==4
    w=G.build_fact_world(55,enc=enc,bg=bg,**G.CELLS['base']);pl=plan(w);A_=G.realize(w);c2=pl['read'][2]['ctx']
    assert set(c2)=={'in_g0','in_g1'} and c2['in_g0']['window_chunks']==4 and c2['in_g1']['window_chunks']==3 and len(c2['in_g0']['tail'])==4*1024 and c2['in_g0']['tail']==A_[0]['ids']+A_[1]['ids']+A_[2]['ids']+A_[3]['ids']      # a = 2: read chunk 3, the window is shorter than 8 chunks
    w=G.build_fact_world(55,enc=enc,bg=bg,read_apps=tuple(range(1,17)),**G.CELLS['p1']);pl=plan(w);assert sorted(pl['read'])==list(range(1,17)) and [pl['read'][a]['ctx']['in_g0']['window_chunks'] for a in range(1,17)]==[min(a,8) for a in range(1,17)] and all(pl['read'][a]['ctx']['in_g0']['d'][0]==1 for a in pl['read'])
    print('e55_read SELFTEST OK (CPU plan only; GPU reader in e55_gpu.py)')
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--selftest',action='store_true');a=ap.parse_args()
    if a.selftest:selftest()
