"""E35 5.11 step 1 (CPU only, no GPU): composition of the background of passes 1..P for (a) the fresh background of R5 (job 51; e35_ladder.build_fresh_worlds) and (b) the existing background of job 37 (L0fs: ladder.fs_streams, pass p>=2 = gw.background(Random(7000+p)) cycled).
Per pass: source books (title/author from the Gutenberg header), paragraph count, paragraph length (chars/words, mean/median/min/max), boilerplate/licence paragraph ratio, exact duplicate paragraphs within the pass, share of paragraphs that were also in the previous pass, word 8-gram overlap with the previous pass (both normalisations) and with pass 1.
Text level, like e35_ladder.cpu_checks: the paragraphs that replace the world's `bg` texts (the stream is later padded with further paragraphs of the same pool; padding is token level and not counted here, the pool size is given).
Usage (same environment as e35_ladder.py --cpu-checks): e35_bg_composition.py [--world-seed 101] [--passes 4]; writes OUT/e35_bg_composition_s{seed}.json / .md. `--selftest` runs the pure functions on synthetic data."""
import os,sys,json,re,random,argparse,statistics,collections
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
BOIL=re.compile(r'Gutenberg|indemnif|royalt|copyright|trademark|licen[sc]e|donat|electronic work|\*\*\*',re.I)
GW_FILES=('pg98.txt','pg1260.txt','pg766.txt','pg768.txt')   # gen_world.background draws from these four (sorted by name)
norm=lambda s:' '.join(s.split())
def grams(s,n=8):
    w=s.split();return {tuple(w[i:i+n]) for i in range(len(w)-n+1)}
def header(path):
    try:t=Path(path).read_text(errors='ignore')[:4000]
    except OSError:return {}
    g=lambda k:(re.search(rf'^{k}:\s*(.+)$',t,re.M) or [None,''])[1].strip()
    return {'title':g('Title'),'author':g('Author'),'language':g('Language')}
def compose(texts_by_pass,source_of):
    """texts_by_pass {pass: [paragraph texts in stream order]}; source_of(text) -> book name or None. Returns {pass: record}."""
    rec={};G={p:grams(norm(' '.join(v))) for p,v in texts_by_pass.items()};ps=sorted(texts_by_pass)
    for p in ps:
        v=texts_by_pass[p];ch=[len(x) for x in v];wd=[len(x.split()) for x in v];src=collections.Counter(source_of(x) or '(unknown)' for x in v)
        r={'n_paragraphs':len(v),'n_distinct':len(set(v)),'duplicates_within_pass':len(v)-len(set(v)),'sources':dict(src),
           'chars':{'mean':statistics.mean(ch),'median':statistics.median(ch),'min':min(ch),'max':max(ch)},'words':{'mean':statistics.mean(wd),'median':statistics.median(wd),'total':sum(wd)},
           'boilerplate_paragraphs':sum(1 for x in v if BOIL.search(x)),'n_8grams':len(G[p])}
        r['boilerplate_ratio']=r['boilerplate_paragraphs']/max(1,len(v))
        if p>ps[0]:
            q=ps[ps.index(p)-1];pv=set(texts_by_pass[q]);inter=len(G[p]&G[q])
            r['prev_pass']=q;r['paragraphs_shared_with_prev']=sum(1 for x in set(v) if x in pv)/max(1,len(set(v)))
            r['overlap_8gram_vs_prev_min']=inter/max(1,min(len(G[p]),len(G[q])));r['overlap_8gram_vs_prev_of_this']=inter/max(1,len(G[p]))
            i1=len(G[p]&G[ps[0]]);r['overlap_8gram_vs_pass1_min']=i1/max(1,min(len(G[p]),len(G[ps[0]])))
        rec[p]=r
    return rec
def md_of(res):
    f=lambda x:'-' if x is None else (f'{x:.3f}' if isinstance(x,float) else str(x))
    L=[f"# E35 background composition (seed {res['seed']}, text level, passes 1..{res['passes']}; CPU only)\n",res['note'],'']
    for key,title in (('R5_fresh','R5 / job 51: fresh background (e35_ladder.build_fresh_worlds)'),('job37_existing','job 37 / L0fs: existing background (ladder.fs_streams, gw.background(Random(7000+p)) cycled)')):
        c=res[key];L+=[f'## {title}\n',f"pool: {c.get('pool_info')}\n",'| pass | books (paragraphs) | title / author | n paragraphs (distinct) | dup. within pass | chars mean / median / max | words total | boilerplate ratio | shared paragraphs with prev. | 8-gram overlap vs prev. (min / of this) | vs pass 1 |','|---|---|---|---|---|---|---|---|---|---|---|']
        for p,r in sorted(c['passes'].items(),key=lambda kv:int(kv[0])):
            bk='; '.join(f'{b} ({n})' for b,n in r['sources'].items());hd='; '.join(f"{c['headers'].get(b,{}).get('title','?')} / {c['headers'].get(b,{}).get('author','?')}" for b in r['sources'])
            L.append(f"| {p} | {bk} | {hd} | {r['n_paragraphs']} ({r['n_distinct']}) | {r['duplicates_within_pass']} | {r['chars']['mean']:.0f} / {r['chars']['median']:.0f} / {r['chars']['max']} | {r['words']['total']} | {r['boilerplate_ratio']:.3f} ({r['boilerplate_paragraphs']}) | {f(r.get('paragraphs_shared_with_prev'))} | {f(r.get('overlap_8gram_vs_prev_min'))} / {f(r.get('overlap_8gram_vs_prev_of_this'))} | {f(r.get('overlap_8gram_vs_pass1_min'))} |")
        L.append('')
    return '\n'.join(L)+'\n'
def run(seed,passes):
    import e35_ladder as E,gen_world as G
    w=G.build_world(seed);DATA=E.DATA;own=[t['text'] for t in w['texts'] if t['kind']=='bg'];res={'seed':seed,'passes':passes,'note':f'n_bg slots in the world = {len(own)}. Padding paragraphs appended at token level (to reach {E.NCH*E.CH} tokens) come from the same pool and are not in this table.'}
    # R5 fresh
    worlds,meta=E.build_fresh_worlds(w,passes);tb={1:own};book_of={}
    for ps,(ww,pool) in enumerate(worlds[1:],2):tb[ps]=[t['text'] for t in ww['texts'] if t['kind']=='bg']
    books=sorted({b for m in meta[1:] for b in m.get('books',[])})
    for b in books:
        for x in E.book_paras(b,random.Random(f'e35|{seed}|{b}')):book_of.setdefault(x,b)
    own_book=None
    for b in GW_FILES:                      # the world's own background book, identified by paragraph overlap with the gen_world filter
        if (DATA/b).exists():
            ps_=set(gw_paras(DATA/b))
            if sum(1 for x in own if x in ps_)>0.5*len(own):own_book=b;break
    for x in own:book_of.setdefault(x,own_book or '(world own book)')
    res['R5_fresh']={'pool_info':{str(m['pass']):m.get('pool_size') for m in meta[1:]},'passes':compose(tb,lambda x:book_of.get(x)),'headers':{b:header(DATA/b) for b in set(book_of.values()) if (DATA/b).exists()},'meta':meta}
    # job 37 existing background: pass p>=2 = gw.background(Random(7000+p)), cycled over the n_bg slots
    tb2={1:own};src2={};pool_info={}
    files=[f for f in sorted(DATA.glob('pg*.txt')) if f.name in GW_FILES]
    for ps in range(2,passes+1):
        pool=G.background(random.Random(7000+ps));pool_info[str(ps)]=len(pool);bk=random.Random(7000+ps).choice(files).name
        tb2[ps]=[pool[i%len(pool)] for i in range(len(own))]
        for x in pool:src2.setdefault(x,bk)
    for x in own:src2.setdefault(x,own_book or '(world own book)')
    res['job37_existing']={'pool_info':pool_info,'passes':compose(tb2,lambda x:src2.get(x)),'headers':{b:header(DATA/b) for b in set(src2.values()) if (DATA/b).exists()}}
    return res
def gw_paras(path):
    """the paragraphs gen_world.background keeps from one file (same filter and scrubbing, no shuffle)."""
    t=Path(path).read_text(errors='ignore');t=t[t.find('CHAPTER'):] if 'CHAPTER' in t else t;t=t[len(t)//5:];t=re.sub(r'\r','',t);out=[]
    for p in [p for p in re.split(r'\n\s*\n',t) if 200<len(p)<900 and not re.search(r'CHAPTER|\bVol|\[|\]|\*',p)]:
        p=re.sub(r'(?<=[a-z,;] )(?:[A-Z][a-z]+)(?: [A-Z][a-z]+)*','someone',p);p=re.sub(r"\b(?:Mr|Mrs|Miss|Dr|Sir|Lady|Lord)\.? \w+",'the stranger',p);out.append(' '.join(p.split()))
    return out
def selftest():
    rng=random.Random(0);words=[f'w{i}' for i in range(400)]
    mk=lambda k:' '.join(rng.choice(words) for _ in range(60))+f' tag{k}'
    a=[mk(i) for i in range(30)];b=[mk(i) for i in range(30)];b[0]=a[0];b[1]='Project Gutenberg licence text '+mk(99);c=(b[:10]*3)[:30]   # c = cycled pool of 10
    src=lambda x:'bookA' if x in a else 'bookB'
    r=compose({1:a,2:b,3:c},src)
    assert r[1]['duplicates_within_pass']==0 and r[2]['boilerplate_paragraphs']==1 and r[3]['duplicates_within_pass']==20 and r[2]['paragraphs_shared_with_prev']>0 and 0<=r[3]['overlap_8gram_vs_prev_min']<=1
    print(md_of({'seed':0,'passes':3,'note':'selftest','R5_fresh':{'pool_info':{},'passes':r,'headers':{}},'job37_existing':{'pool_info':{},'passes':r,'headers':{}}})[:900]);print('SELFTEST OK')
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--world-seed',type=int,default=101);ap.add_argument('--passes',type=int,default=4);ap.add_argument('--selftest',action='store_true');a=ap.parse_args()
    if a.selftest:selftest();sys.exit()
    import e35_ladder as E;res=run(a.world_seed,a.passes);E.OUT.mkdir(parents=True,exist_ok=True)
    json.dump(res,open(E.OUT/f'e35_bg_composition_s{a.world_seed}.json','w'),indent=1,default=str);md=md_of(res);(E.OUT/f'e35_bg_composition_s{a.world_seed}.md').write_text(md);print(md)
