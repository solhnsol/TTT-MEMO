"""E30 fact-free context files (CPU only; no tokenizer, no GPU). One json per kind with one text segment per context slot (slot r = the r-th write chunk of the c2 set, chunk-specific, no segment reuse):
  bg   main context: background-style paragraphs (gen_world.background cleaning: 200-900 chars, capitalised non-initial words -> 'someone', titles -> 'the stranger') from Gutenberg books that are not canary books
       (pg11, pg1342) and not the world's own background book (exact-paragraph overlap <= 5%)
  bg2  a second independent draw of the same style, from books disjoint from bg (N-a: content control)
  raw  natural book text (N-b): paragraphs of 100-3000 chars without the name scrubbing, from books disjoint from bg and bg2, Gutenberg header/footer removed
Every paragraph is dropped if it contains a world string (all triples' names/places, control persons/places; whole-word match), equals a canary or own-background paragraph, or repeats a paragraph used earlier
(inside a kind and across kinds). Segments are filled to >= TARGET_CHARS characters (8192 tokens need about 36K characters; the GPU script re-checks the token count and aborts if short).
Audit (written next to the files and into each json): world-string hits 0, canary exact 0, word 8-gram overlap between kinds, with the canary books and with the world's own background (<= MAX_OVERLAP, books are disjoint), duplicates 0.
Usage: e30_ctx.py --world-seed 101 --out-dir DIR [--n-segments 26] | --selftest. Exit 3 if the audit fails."""
import os,sys,json,re,random,argparse,hashlib
from pathlib import Path
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
MAX_OVERLAP=0.05;TARGET_CHARS=44000;EXCL={'pg11.txt','pg1342.txt'}
norm=lambda s:' '.join(s.split())
def grams(s,n=8):
    w=s.split();return {tuple(w[i:i+n]) for i in range(len(w)-n+1)}
def strip_pg(t):
    a=t.find('CHAPTER');t=t[a:] if a>=0 else t;e=t.find('*** END');t=t[:e] if e>0 else t;return t.replace('\r','')
def paras_bg(t):
    """gen_world.background / e35_ladder.book_paras cleaning"""
    t=strip_pg(t);t=t[len(t)//5:]
    ps=[p for p in re.split(r'\n\s*\n',t) if 200<len(p)<900 and not re.search(r'CHAPTER|\bVol|\[|\]|\*|Gutenberg|indemnif|royalt|copyright|trademark|licen[sc]e|donat',p,re.I)];out=[]
    for p in ps:
        p=re.sub(r'(?<=[a-z,;] )(?:[A-Z][a-z]+)(?: [A-Z][a-z]+)*','someone',p);p=re.sub(r"\b(?:Mr|Mrs|Miss|Dr|Sir|Lady|Lord)\.? \w+",'the stranger',p);out.append(norm(p))
    return out
def paras_raw(t):
    t=strip_pg(t);ps=[norm(p) for p in re.split(r'\n\s*\n',t)];return [p for p in ps if 100<len(p)<3000 and not re.search(r'\[|\]|\*|Gutenberg|indemnif|royalt|copyright|trademark|licen[sc]e|donat',p,re.I)]
def world_strings(w):
    s=set()
    for t in w['triples']:s|={t['X'],t['P']}
    s|=set(w['ctrl_persons'])|set(w['ctrl_places']);return sorted(x for x in s if x)
def build(data_dir,w,nseg,seed=0):
    """returns {kind: [segment text]}, audit dict"""
    data=Path(data_dir);rx=re.compile(r'(?<!\w)(?:'+'|'.join(re.escape(x) for x in sorted(world_strings(w),key=len,reverse=True))+r')(?!\w)')
    own=[t['text'] for t in w['texts'] if t['kind']=='bg'];ownset=set(own)
    canary={f:norm(strip_pg((data/f).read_text(errors='ignore'))) for f in EXCL if (data/f).exists()};cg={k:grams(v) for k,v in canary.items()};ctext=' '.join(canary.values())
    books=sorted(p.name for p in data.glob('pg*.txt') if p.name not in EXCL);rng=random.Random(f'e30|{seed}');rng.shuffle(books)
    # books that carry the world's own background are excluded from bg
    def usable(f):
        ps=paras_bg((data/f).read_text(errors='ignore'));return len(ps)>=100 and len(set(ps)&ownset)<=MAX_OVERLAP*len(ownset)
    free=[f for f in books if usable(f)]
    kinds=['bg','bg2','raw'];used_books={k:[] for k in kinds};seen=set(ownset);seg={};drop={k:{'world':0,'canary':0,'dup':0} for k in kinds};bi=0
    for kind in kinds:
        segs=[];cur=[];n=0;pool=[]
        def refill():
            nonlocal bi
            assert bi<len(free),f'ran out of books for kind {kind} ({len(free)} usable, {nseg} segments of {TARGET_CHARS} chars needed per kind)'
            f=free[bi];bi+=1;used_books[kind].append(f);t=(data/f).read_text(errors='ignore')
            ps=paras_raw(t) if kind=='raw' else paras_bg(t);random.Random(f'e30|{seed}|{kind}|{f}').shuffle(ps)
            for p in ps:
                if rx.search(p):drop[kind]['world']+=1;continue
                if p in seen:drop[kind]['dup']+=1;continue
                if p in ctext or any(len(grams(p)&g)>0 for g in cg.values()):drop[kind]['canary']+=1;continue
                seen.add(p);pool.append(p)
        while len(segs)<nseg:
            while n<TARGET_CHARS:
                if not pool:refill()
                p=pool.pop();cur.append(p);n+=len(p)+2
            segs.append('\n\n'.join(cur));cur=[];n=0
        seg[kind]=segs
    audit={'books':used_books,'dropped':drop,'n_segments':nseg,'chars_per_segment_min':{k:min(len(s) for s in v) for k,v in seg.items()},'pass':True}
    G={k:grams(norm(' '.join(v))) for k,v in seg.items()};G['own_bg']=grams(norm(' '.join(own)))
    for k,g in cg.items():G['canary_'+k]=g
    names=sorted(G);audit['overlap_8gram']={}
    for i,a in enumerate(names):
        for b in names[i+1:]:
            if a.startswith('canary_') and b.startswith('canary_'):continue
            o=len(G[a]&G[b])/max(1,min(len(G[a]),len(G[b])));audit['overlap_8gram'][f'{a}|{b}']=o;audit['pass']&=o<=MAX_OVERLAP
    allp=[p for v in seg.values() for s in v for p in s.split('\n\n')]
    audit['duplicate_paragraphs']=len(allp)-len(set(allp));audit['world_string_hits']=sum(1 for p in allp if rx.search(p));audit['canary_exact']=sum(1 for p in allp if p in ctext)
    audit['pass']&=audit['duplicate_paragraphs']==0 and audit['world_string_hits']==0 and audit['canary_exact']==0
    return seg,audit
def selftest():
    import tempfile
    d=Path(tempfile.mkdtemp());rng=random.Random(1);V=[f'w{i}' for i in range(4000)]
    def para(n,cap=False):return ' '.join((rng.choice(V).capitalize() if cap and i==0 else rng.choice(V)) for i in range(n))+'.'
    for i in range(14):
        body='CHAPTER I\n\n'+'\n\n'.join(para(rng.randint(40,120)) for _ in range(500))+'\n\n*** END OF THE PROJECT'
        (d/f'pg{100+i}.txt').write_text(body)
    (d/'pg11.txt').write_text('CHAPTER I\n\n'+'\n\n'.join(para(80) for _ in range(300)));(d/'pg1342.txt').write_text('CHAPTER I\n\n'+'\n\n'.join(para(80) for _ in range(300)))
    own=[para(80) for _ in range(50)];(d/'pg200.txt').write_text('CHAPTER I\n\n'+'\n\n'.join(own+[para(80) for _ in range(300)]))   # the world's own background book: excluded
    w={'triples':[{'X':'Zork Qua','P':'Bamton'}],'ctrl_persons':['Ctl P'],'ctrl_places':['Ctlville'],'texts':[{'kind':'bg','text':p} for p in own]}
    bad=[p for p in paras_bg((d/'pg100.txt').read_text())][:3];assert bad
    # plant a world string into a book: must be dropped
    for i in range(14):
        f=d/f'pg{100+i}.txt';f.write_text(f.read_text().replace('CHAPTER I','CHAPTER I\n\n'+para(60)+' Zork Qua walked to Bamton '+para(60)+'\n\n'+para(150)))
    seg,aud=build(d,w,3);assert aud['pass'],aud
    assert all(len(s)>=TARGET_CHARS for v in seg.values() for s in v) and all(len(v)==3 for v in seg.values())
    assert aud['world_string_hits']==0 and 'pg200.txt' not in sum(aud['books'].values(),[]) and sum(aud['dropped'][k]['world'] for k in aud['dropped'])>=1
    assert len({s for v in seg.values() for s in v})==9
    try:build(d,w,400);raise SystemExit('book shortage not detected')
    except AssertionError:pass
    print('e30_ctx SELFTEST OK',aud['dropped'],aud['books'])
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--world-seed',type=int,default=101);ap.add_argument('--out-dir',default=os.environ.get('EXP07_OUT','/tmp/e30'));ap.add_argument('--n-segments',type=int,default=26);ap.add_argument('--data-dir',default=str(ROOT/'data/narrative_gutenberg'))
    ap.add_argument('--selftest',action='store_true');a=ap.parse_args()
    if a.selftest:selftest();return
    sys.path.insert(0,str(ROOT/'scripts/exp04_what_ttt_retains'));import gen_world as gw
    w=gw.build_world(a.world_seed,v=2);seg,aud=build(a.data_dir,w,a.n_segments,a.world_seed);od=Path(a.out_dir);od.mkdir(parents=True,exist_ok=True)
    for k,v in seg.items():json.dump({'kind':k,'world_seed':a.world_seed,'segments':v,'audit':aud},open(od/f'e30_ctx_{k}_s{a.world_seed}.json','w'))
    (od/f'e30_ctx_audit_s{a.world_seed}.json').write_text(json.dumps(aud,indent=1));print(json.dumps(aud,indent=1));sys.exit(0 if aud['pass'] else 3)
if __name__=='__main__':main()
