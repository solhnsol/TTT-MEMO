"""E57 block-stream dilution experiment (new file; wrapper of the unchanged e57_oracle_run.py, like e57_dclm_band_run.py). Fact blocks are streamed one after the other instead of interleaved: cell 'blk<F>x<NB>_c<LC>' = NB blocks of F facts, block j occupies the chunks LC*j .. LC*j+LC-1 (one sentence per fact of the block in EVERY chunk of the block, so a fact is exposed LC times, in LC different frames; facts of other blocks do not appear in these chunks); k = F*NB facts with ids 0..k-1 (block of fact f = f // F), n_app = LC*NB chunks, p = 1, read points after every block (appearances LC, 2 LC, ..., NB LC = chunks LC*(j+1)-1). The reads of the unchanged runner (all items, 'out' = neutral 8192-token tail, 'in_g0' = the last 8 real chunks) are made at every read point (the wrapper sets the in-window reads to every read point, the runner's default is the last one only); the lag of block i at read point j is j - i blocks (lag 0 = read right after the block, the last LC chunks are in the window if LC >= 8; lag >= 1: the last exposure is LC chunks or more before the read point, outside the window of 8 chunks for LC >= 8 at an end-of-block read). Candidates of an item = the F facts of its own block (gold + F - 1 same-block places; F = 16 = NCAND), control items take the candidates of a random block. The 'names' / 'nofact' decoy streams use the same layout. Everything else (arms NATS / NATMASK, mask arm ALLQ<W>, K0, clip, token diag, logs) is the option of e57_oracle_run.py.
  python3 e57_blockstream_run.py --cell blk16x4_c16 --seed S --out O --masks none --mask-arm ALLQ256 --arms NATS,NATMASK --save-apps 16,32,48,64 --nat-loss-every 8 --tag bs | --selftest"""
import sys,re,random,pathlib
import numpy as np
HERE=pathlib.Path(__file__).resolve().parent;ROOT=HERE.parents[1]
for p in (HERE,HERE/'e57',ROOT/'scripts/exp04_what_ttt_retains',ROOT/'scripts/exp05_write_dynamics'):sys.path.insert(0,str(p))
import gen_facts as G
PAT=re.compile(r'blk(\d+)x(\d+)_c(\d+)')
def parse(cell):
    m=PAT.fullmatch(cell);assert m,('cell must be blk<F>x<NB>_c<LC>',cell);F,NB,LC=map(int,m.groups());assert F>=2 and NB>=1 and LC>=1;return F,NB,LC
def register(cell):
    F,NB,LC=parse(cell)
    if cell not in G.CELLS:G.CELLS[cell]=dict(k=F*NB,n=1,p=1,bank='300',n_app=LC*NB,read_apps=tuple(LC*(j+1) for j in range(NB)));G.CELL_BOOKS[cell]=G.CELL_BOOKS['v3_64_4']
    return G.CELLS[cell]
_orig_stream=G._stream;_orig_items=G.build_items
def block_stream(facts,k,n,p,n_app,bank,enc,units,seed,tag,sep,F,LC):
    """copy of gen_facts._stream with the block layout: chunk c carries only the facts of block c // LC, frame index c (distinct frame at every exposure), 'app' = exposure number within the block (c % LC)"""
    CH=G.CH;nch=n_app;ui=0;chunks=[];stats=[]
    for c in range(nch):
        b=c//LC;app=c%LC;rng=random.Random(f'E55f|{seed}|chunk|{tag}|{c}');mine=[f for f in facts if f['id']//F==b];assert len(mine)==F
        sents=[]
        for f in mine:
            fi=f['frames'][c];t=bank[fi].format(X=f['X'],P=f['P'],o=f['o'],a_o=G.A(f['o']));ids=enc(t);sents.append((f['id'],fi,ids+sep,t))
        rng.shuffle(sents);cost=sum(len(s[2]) for s in sents);assert cost<CH-100,('sentences fill the chunk',cost,len(sents))
        rem=CH-cost;whole=[];tail=None
        while True:
            assert ui<len(units),('background units exhausted',tag,c)
            ids=units[ui][0]
            if len(ids)<=rem:whole.append(ids);rem-=len(ids);ui+=1
            else:
                if rem>0:tail=ids[:rem]
                ui+=1;break
        nb=len(whole);assert nb>=2,('too few background units per chunk',nb);stats.append(nb)
        slots=[rng.randint(1,nb-1) for _ in sents];order=sorted(range(len(sents)),key=lambda i:(slots[i],i))
        ids_=[];sp=[];oi=0
        for pos in range(nb+1):
            while oi<len(order) and slots[order[oi]]==pos:
                f_,fi,ids,t_=sents[order[oi]];sp.append({'fact':f_,'frame':fi,'app':app,'start':len(ids_),'end':len(ids_)+len(ids)-len(sep),'text':t_,'block':b});ids_+=ids;oi+=1
            if pos<nb:ids_+=whole[pos]
        if tail:ids_+=tail
        assert len(ids_)==CH,(c,len(ids_));chunks.append({'k':c,'group':0,'app':app,'block':b,'ids':ids_,'sents':sp})
    return chunks,ui,min(stats)
def block_items(world,F):
    """gen_facts.build_items with the candidates of an item = the facts of its own block (gold + F - 1; control items: the block of a random fact); every item keeps group 0 so that the unchanged in-window read (group 0 + controls) reads all items"""
    facts=world['facts'];seed=world['seed'];k=world['k'];items=[];NC=G.NCAND;assert F==NC,'candidates of a block: F must equal NCAND = 16'
    def cands(r,f):
        same=[g for g in facts if g['id']//F==f['id']//F and g['id']!=f['id']];r.shuffle(same);assert len(same)==F-1;return same
    ux=world.get('unexposed',[])
    def tail(d):
        d['cands']+=[' '+u for u in ux];d['cand_fact']+=[-1]*len(ux);d['cand_group']+=[-1]*len(ux);d['cand_same']+=[False]*len(ux);d['cand_exposed']=[True]*(NC-len(ux))+[False]*len(ux);d['n_rank']=NC-len(ux);return d
    for f in facts:
        r=random.Random(f"E55f|{seed}|q|{f['id']}");cl=[f]+cands(r,f);r.shuffle(cl);gi=[x['id'] for x in cl].index(f['id'])
        items.append(tail({'kind':'fact','fact':f['id'],'group':0,'block':f['id']//F,'name_id':f['id'],'place_id':f['id'],'name':f['X'],'qframe':f['id']%G.NQ,'prefix':G.QFRAMES[f['id']%G.NQ].format(X=f['X']),'cands':[' '+x['P'] for x in cl],'gold_idx':gi,'cand_fact':[x['id'] for x in cl],'cand_group':[x['group'] for x in cl],'cand_same':[True]*len(cl)}))
    for i,c in enumerate(world['ctrl']):
        r=random.Random(f"E55f|{seed}|qc|{i}");fg=facts[r.randrange(k)];cl=[fg]+cands(r,fg);r.shuffle(cl);gi=[x['id'] for x in cl].index(fg['id'])
        items.append(tail({'kind':'ctrl','fact':-1,'group':-1,'block':-1,'name_id':c['id'],'place_id':fg['id'],'name':c['X'],'qframe':c['id']%G.NQ,'prefix':G.QFRAMES[c['id']%G.NQ].format(X=c['X']),'cands':[' '+x['P'] for x in cl],'gold_idx':gi,'cand_fact':[x['id'] for x in cl],'cand_group':[x['group'] for x in cl],'cand_same':[False]*len(cl)}))
    return items
def patch(cell):
    F,NB,LC=parse(cell);register(cell)
    G._stream=lambda facts,k,n,p,n_app,bank,enc,units,seed,tag,sep:block_stream(facts,k,n,p,n_app,bank,enc,units,seed,tag,sep,F,LC)
    G.build_items=lambda world:block_items(world,F)
def patch_reads():
    import e55_gpu as RG;orig=RG.Reads
    def mk(*a,**k):
        a=list(a);a[8]=list(a[7]);return orig(*a,**k)           # in-window reads at every read point (read_apps -> in_apps)
    RG.Reads=mk
def selftest():
    cell='blk16x4_c16';patch(cell);enc=G.StubTok();bg=G._synth_bg(9000);cfg=G.CELLS[cell]
    w=G.cell_world(7,cell,enc,bg,decoys=('names','nofact'));F,NB,LC=parse(cell)
    assert w['k']==64 and w['p']==1 and w['n_chunks']==64 and w['read_chunks']==[15,31,47,63] and w['read_apps']==[16,32,48,64]
    for c,ch in enumerate(w['chunks']):
        assert len(ch['ids'])==G.CH and len(ch['sents'])==16 and {s['fact'] for s in ch['sents']}==set(range(16*(c//16),16*(c//16)+16)) and {s['app'] for s in ch['sents']}=={c%16}
    for name in ('names','nofact'):assert len(w[name]['chunks'])==64 and all(len(x['ids'])==G.CH for x in w[name]['chunks'])
    for f in range(64):assert sum(1 for ch in w['chunks'] for s in ch['sents'] if s['fact']==f)==16 and len({s['frame'] for ch in w['chunks'] for s in ch['sents'] if s['fact']==f})==16     # 16 exposures, 16 different frames
    import e55_read as R;P=R.plan(w,enc);items=P['items'];assert len(items)==64+G.NCTRL
    for it in items[:64]:
        assert it['cands'][it['gold_idx']]==' '+w['facts'][it['place_id']]['P'] and len(set(it['cand_fact']))==16 and {x//16 for x in it['cand_fact']}=={it['fact']//16}
    for it in items[64:]:assert len({x//16 for x in it['cand_fact']})==1
    assert sorted(P['read'])==[16,32,48,64] and all(set(P['read'][a]['ctx'])=={'in_g0'} and len(P['read'][a]['ctx']['in_g0']['items'])==68 and P['read'][a]['ctx']['in_g0']['window_chunks']==8 for a in P['read'])
    for a,r in zip(w['read_apps'],w['read_chunks']):assert P['read'][a]['ctx']['in_g0']['tail'][-1024:]==w['chunks'][r]['ids']
    G.rows(items);G.item_meta(items)
    print('e57_blockstream_run selftest ok')
if __name__=='__main__':
    if '--selftest' in sys.argv:selftest()
    else:
        i=sys.argv.index('--cell');patch(sys.argv[i+1]);patch_reads()
        import e57_oracle_run as O;O.main()
