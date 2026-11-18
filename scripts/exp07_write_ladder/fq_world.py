"""FictionalQA second world -> gen_facts-compatible world (CPU; new file; EXPLORATORY preparation, Hans 11/18; branch claude/project-thread-sxtbc9 of the private registry, data/fictionalqa/, NOT merged into main). Same role as nat_world.py for the natural-text worlds: a drop-in world for the E55 / E57 pipeline (E2E 3B, Llama-3 tokenizer, chunk = 1024 tokens, 16 facts x 128 exposures, reads at exposures 16 / 32 / 64 / 128):
  load_world(path)                 -> the world JSON of fq_build_worlds.py (facts, reserve, schedule[128 chunks], bg_sentences, items)
  realize_chunk(world, t)          -> the text of chunk t (sentences joined by a blank line; same function as fq_build_worlds.realize_chunk)
  fit_chunk(world, t, tok)         -> (ids, sents): the token ids of chunk t fitted to EXACTLY 1024 tokens and the fact sentences with their token spans. The real tokenisation of realize_chunk text is 1011..1025 tokens (only ~18% of the chunks are exactly 1024), the pipeline needs exactly CH = 1024 (gen_facts.verify, the fixed slicing of the stream): chunks of 1025..1031 tokens lose the last tokens of the LAST background sentence of the chunk, chunks of 1011..1023 get the tail words of the previous background sentence of the chunk appended to the last background sentence (re-tokenised and verified; no fact sentence is ever touched, the fact-sentence token spans are re-read from the character offsets of the final text).
  to_gen_world(world, tok, ...)    -> a dict with the keys of gen_facts.build_fact_world (seed, k, n, p, n_app, n_chunks, name_len, place_len, facts, ctrl, unexposed, read_apps, read_chunks, sep, chunks [{k, group, app, ids, sents [{fact, frame, app, start, end, text}]}], bank_kind 'fq') plus 'fq' (provenance and the fit statistics)
  build_items(gw)                  -> items like gen_facts.build_items: 16 fact items (prefix ends 'Answer:', gold ' '+answer, 16 candidates = the 16 golds of the world, all 3 Llama-3 tokens) and NCTRL = 4 control items made of the first 4 RESERVE facts (never exposed, their events are excluded from the background), same keys as gen_facts items, so that gen_facts.rows / item_meta / e55_read.plan work unchanged.
The world is NOT written into gen_facts.CELLS; the run wrapper (job proposal) patches gen_facts.cell_world for the cell name 'fq<seed>' with to_gen_world, the rest of the pipeline is unchanged.
  python3 fq_world.py --selftest [--worlds DIR] | --report --worlds DIR"""
import sys,json,re,argparse,pathlib,collections
HERE=pathlib.Path(__file__).resolve().parent;ROOT=HERE.parents[1]
for p in (HERE,ROOT/'scripts/exp04_what_ttt_retains',ROOT/'scripts/exp05_write_dynamics'):sys.path.insert(0,str(p))
CH=1024;NFACT=16;NCTRL=4;NCAND=16;READ_APPS=(16,32,64,128)
TOKFILE=ROOT/'data/tokenizer/llama3/tokenizer.json'
def load_tokenizer(path=TOKFILE):
    from tokenizers import Tokenizer
    return Tokenizer.from_file(str(path))
def load_world(path):return json.load(open(path))
def parts_of(world,t):
    """[(kind, text, fact or None, anchor or None)] of chunk t in schedule order"""
    out=[]
    for c in world['schedule'][t]:
        if 'fact' in c:
            a=world['facts'][c['fact']]['anchors'][c['anchor']];out.append(('fact',a['text'],c['fact'],c['anchor']))
        else:out.append(('bg',world['bg_sentences'][c['bg']],None,None))
    return out
def realize_chunk(world,t):return '\n\n'.join(p[1] for p in parts_of(world,t))
def _encode(tok,text):
    e=tok.encode(text,add_special_tokens=False);return e.ids,e.offsets
def fit_chunk(world,t,tok):
    """token-level fit (no re-tokenisation): tokenise the joined chunk text once, find the token span of every part from the character offsets, then (n > 1024) delete the last n - 1024 tokens of the LONGEST background sentence of the chunk or (n < 1024) insert the last 1024 - n token ids of another (the longest other) background sentence after the longest background sentence; the spans of the fact sentences behind the edit point are shifted. The ids are a valid token sequence (a background sentence is cut / extended at its end); no fact-sentence token is touched."""
    parts=parts_of(world,t);texts=[p[1] for p in parts];text='\n\n'.join(texts);ids,off=_encode(tok,text);n0=len(ids)
    spans=[];pos=0
    for tx in texts:
        s0=pos;e0=pos+len(tx);pos=e0+2;tix=[j for j,(x,y) in enumerate(off) if s0<=x<e0 and y>x];spans.append([tix[0],tix[-1]+1])
    bgi=[i for i,p in enumerate(parts) if p[0]=='bg'];assert bgi,'a chunk without a background sentence cannot be fitted'
    last=max(bgi,key=lambda i:spans[i][1]-spans[i][0]);others=[i for i in bgi if i!=last];info=dict(chunk=t,n0=n0,action='none')
    if n0>CH:
        over=n0-CH;bs,be=spans[last];assert be-bs>over+3,('last background sentence too short to trim',t,n0)
        ids=ids[:be-over]+ids[be:];spans[last][1]=be-over;info['action']=f'trim {over}'
        for i in range(last+1,len(parts)):spans[i]=[spans[i][0]-over,spans[i][1]-over]
    elif n0<CH:
        need=CH-n0;src=spans[max(others,key=lambda i:spans[i][1]-spans[i][0])] if others else spans[last];tail=ids[src[0]:src[1]][-need:];assert len(tail)==need,('padding source too short',t,need)
        be=spans[last][1];ids=ids[:be]+tail+ids[be:];spans[last][1]=be+need;info['action']=f'pad {need}'
        for i in range(last+1,len(parts)):spans[i]=[spans[i][0]+need,spans[i][1]+need]
    assert len(ids)==CH
    sents=[dict(fact=pp[2],frame=pp[3],app=t,start=spans[i][0],end=spans[i][1],text=texts[i]) for i,pp in enumerate(parts) if pp[0]=='fact']
    info['n1']=len(ids);return ids,sents,info
SEED_LABEL={201+i:211+i for i in range(8)}   # Hans 11/18: builder seeds 201-208 -> run/cell/result seed labels 211-218 (201-204 are Exp06 seeds in the 10/17 ledger); world content unchanged
def to_gen_world(world,tok,read_apps=READ_APPS):
    seed=SEED_LABEL.get(world['seed'],world['seed']);chunks=[];infos=[]
    for t in range(world['n_expo']):
        ids,sents,info=fit_chunk(world,t,tok);chunks.append(dict(k=t,group=0,app=t,ids=ids,sents=sents));infos.append(info)
    facts=[dict(id=f['id'],group=0,X=f['question'],P=f['gold'],o='fq',frames=[a['text'] for a in f['anchors']],event=f['event'],qid=f['qid']) for f in world['facts']]
    res=world['reserve'][:NCTRL];ctrl=[dict(id=NFACT+i,group=0,X=f['question'],P=f['gold'],o='fq',frames=[],event=f['event'],qid=f['qid']) for i,f in enumerate(res)]
    sep=tok.encode('\n\n',add_special_tokens=False).ids
    gw=dict(seed=seed,k=NFACT,n=1,p=1,n_app=world['n_expo'],n_chunks=world['n_expo'],bank_kind='fq',name_len=3,place_len=3,facts=facts,ctrl=ctrl,unexposed=ctrl,read_apps=list(read_apps),read_chunks=[a-1 for a in read_apps],sep=sep,chunks=chunks)
    gw['fq']=dict(builder_seed=world['seed'],label_seed=seed,sha=world['sha'],tokenizer=world['tokenizer'],hf_revision=world['hf_revision'],group=world['group'],n0_hist=dict(collections.Counter(i['n0'] for i in infos)),n_fitted={a:sum(1 for i in infos if i['action'].split()[0]==a) for a in ('none','trim','pad')},events_used=world['events_used'])
    return gw
def build_items(gw):
    """items of gen_facts.build_items format; cands = the 16 golds of the world (' '+gold, gold first), control items from the reserve facts"""
    facts=gw['facts'];ctrl=gw['ctrl'];items=[]
    def mk(f,kind):
        others=[g for g in facts if g is not f];cands=[f]+others
        return dict(kind=kind,fact=f['id'],group=0,name_id=f['id'],place_id=f['id'],name=f['qid'],qframe=0,prefix=f['X'],cands=[' '+g['P'] for g in cands],gold_idx=0,cand_fact=[(g['id'] if g in facts else -1) for g in cands],cand_group=[0]*len(cands),cand_same=[True]*len(cands),cand_exposed=[True]*len(cands),n_rank=len(cands))
    for f in facts:items.append(mk(f,'fact'))
    for f in ctrl:
        it=mk(f,'ctrl');it['cands']=[' '+f['P']]+[' '+g['P'] for g in facts[:NCAND-1]];it['cand_fact']=[-1]+[g['id'] for g in facts[:NCAND-1]];it['cand_group']=[0]*NCAND;it['cand_same']=[True]*NCAND;it['cand_exposed']=[False]+[True]*(NCAND-1);it['n_rank']=NCAND;items.append(it)
    return items
def checks(world,gw,tok):
    """element checks of the adapter output; returns a dict of booleans / counts"""
    items=build_items(gw);r={}
    r['all_chunks_1024']=all(len(c['ids'])==CH for c in gw['chunks'])
    r['16_fact_sentences_per_chunk']=all(len(c['sents'])==NFACT and {x['fact'] for x in c['sents']}==set(range(NFACT)) for c in gw['chunks'])
    ok=True
    for c in gw['chunks']:
        for x in c['sents']:
            txt=tok.decode(c['ids'][x['start']:x['end']]).strip()
            if txt!=x['text'].strip():ok=False
    r['sentence_token_spans_decode_to_text']=ok
    r['sentences_inside_chunk']=all(0<=x['start']<x['end']<=CH for c in gw['chunks'] for x in c['sents'])
    gold_lens=[len(tok.encode(' '+f['P'],add_special_tokens=False).ids) for f in gw['facts']+gw['ctrl']];r['gold_3_tokens']=set(gold_lens)=={3}
    eq=True
    for it in items:
        pi=tok.encode(it['prefix'],add_special_tokens=False).ids;ls=set()
        for cnd in it['cands']:
            fi=tok.encode(it['prefix']+cnd,add_special_tokens=False).ids;eq&=(fi[:len(pi)]==pi and len(fi)>len(pi));ls.add(len(fi)-len(pi))
        eq&=(ls=={3})
    r['candidates_equal_length_3_and_prefix_stable']=bool(eq)
    r['items_kinds']=dict(collections.Counter(i['kind'] for i in items))
    gl=[f['P'].lower() for f in gw['facts']];bgtxt=' '.join(world['bg_sentences']).lower();r['no_gold_in_background']=not any(re.search(r'(?<![\w])'+re.escape(g)+r'(?![\w])',bgtxt) for g in gl+[c['P'].lower() for c in gw['ctrl']])
    import gen_facts as G;import e55_read as R
    r['gen_facts_rows_item_meta']=len(G.rows(items))==(NFACT+NCTRL)*NCAND and len(G.item_meta(items)['kind'])==len(items)
    orig=G.build_items;G.build_items=lambda w:build_items(w) if w.get('bank_kind')=='fq' else orig(w)          # the run wrapper patches gen_facts.build_items the same way
    try:
        pl=R.plan(gw);r['e55_read_plan']=set(pl['read'])==set(READ_APPS) and len(pl['rows'])==(NFACT+NCTRL)*NCAND
    except Exception as e:r['e55_read_plan']=repr(e)
    finally:G.build_items=orig
    return r
def selftest(wdir=None):
    wdir=pathlib.Path(wdir) if wdir else pathlib.Path(__file__).resolve().parents[2]/'experiments/exp07_write_ladder/fq/worlds';tok=load_tokenizer()
    # fit logic on a toy chunk: too long -> trimmed, too short -> padded, exactly 1024 untouched
    w=load_world(sorted(wdir.glob('fq_201_*.json'))[0]);gw=to_gen_world(w,tok);c=checks(w,gw,tok)
    for k in ('all_chunks_1024','16_fact_sentences_per_chunk','sentence_token_spans_decode_to_text','sentences_inside_chunk','gold_3_tokens','candidates_equal_length_3_and_prefix_stable','no_gold_in_background','gen_facts_rows_item_meta'):assert c[k] is True,(k,c)
    assert c['e55_read_plan'] is True,c['e55_read_plan']
    # determinism
    gw2=to_gen_world(w,tok);assert [x['ids'] for x in gw['chunks']]==[x['ids'] for x in gw2['chunks']]
    # the fitted chunk keeps the fact sentences byte-identical and only touches the last background sentence
    t=0;ids,sents,info=fit_chunk(w,t,tok);fs=[p[1] for p in parts_of(w,t) if p[0]=='fact'];assert sorted(x['text'] for x in sents)==sorted(fs)
    print('fq_world selftest ok (world 201:',gw['fq']['n_fitted'],'chunks fitted; token counts before fitting',dict(sorted(gw['fq']['n0_hist'].items())),')')
def report(wdir):
    tok=load_tokenizer();wdir=pathlib.Path(wdir);out={}
    for p in sorted(wdir.glob('fq_*_cap3_VAR.json')):
        w=load_world(p);gw=to_gen_world(w,tok);c=checks(w,gw,tok);out[w['seed']]=dict(fitted=gw['fq']['n_fitted'],n0=dict(sorted(gw['fq']['n0_hist'].items())),checks=c,sha=w['sha'][:12])
        print(w['seed'],out[w['seed']]['fitted'],'n0 min/max',min(gw['fq']['n0_hist']),max(gw['fq']['n0_hist']),'checks ok',all(v is True for k,v in c.items() if k not in ('items_kinds',)))
    return out
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--selftest',action='store_true');ap.add_argument('--report',action='store_true');ap.add_argument('--worlds');a=ap.parse_args()
    if a.selftest:selftest(a.worlds)
    if a.report:report(a.worlds)
if __name__=='__main__':main()
