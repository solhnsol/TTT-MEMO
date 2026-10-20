"""E33 R filler reuse (CPU only, no tokenizer, no GPU): pass 2..P background paragraphs for a natural-text stream, taken from the background paragraphs of the OTHER natural streams (E33 2.9/2.10).
Filter (applied before any assignment): drop a paragraph that contains a target string of the stream's world (NW.forbid_strings: fake names, birthplaces, in-bundle foils, schools, control and pseudo names/places; whole-word match), equals one of the stream's own pass-1 background paragraphs,
repeats another candidate exactly, or shares word 8-grams with the canary books (pg11, pg1342; optionally the SQuAD tail paragraphs). The origin streams are split over the passes (whole streams per pass, largest first onto the lightest pass), so consecutive passes never
share an origin stream and paragraphs of two passes never match exactly; inside a pass the order is a seeded shuffle. Aborts if a pass has fewer than (n_bg slots + margin) paragraphs.
Audit (printed, json and md): per pass counts / origin streams / words, exact duplicates inside and across passes and with the own pass-1 background, word 8-gram overlap between passes and with pass 1 (fraction of the smaller set), canary 8-gram share, forbid hits (must be 0).
Pass rule: exact overlaps 0, forbid hits 0, canary exact 0, every pass-pair 8-gram overlap <= MAX_OVERLAP. Exit code 3 if the audit fails.
Usage: nat_filler.py --nat-dir DIR_WITH_nat_91xx_spread.json --stream 9105 --passes 3 [--out-dir /tmp/e33] [--selftest]"""
import os,sys,json,re,glob,random,argparse,hashlib
from pathlib import Path
import numpy as np
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
sys.path.insert(0,str(HERE))
import nat_world as NW
MAX_OVERLAP=0.05;MARGIN=150
norm=lambda s:' '.join(s.split())
def grams(s,n=8):
    w=s.split();return {tuple(w[i:i+n]) for i in range(len(w)-n+1)}
def forbid_regex(world):
    fs=sorted(NW.forbid_strings(world),key=len,reverse=True);return re.compile(r'(?<!\w)(?:'+'|'.join(re.escape(s) for s in fs if s)+r')(?!\w)')
def world_path(nat_dir,stream,layout='spread'):return Path(nat_dir)/f'nat_{stream}_{layout}.json'
def bg_texts(world):return [t['text'] for t in world['texts'] if t['kind']=='bg']
_CANARY={}
def canary_grams():
    """8-gram sets of the canary / readout contexts: pg11 (gen canary), pg1342 (neutral tail R1) and, when data/squad/dev-v1.1.json exists, the first 400 SQuAD dev paragraphs (R1-alt tail)."""
    if _CANARY:return _CANARY
    for f in ('pg11.txt','pg1342.txt'):
        p=ROOT/'data/narrative_gutenberg'/f
        if p.exists():
            t=p.read_text(errors='ignore');t=t[t.find('CHAPTER'):] if 'CHAPTER' in t else t;_CANARY[f]=grams(norm(t.replace('\r','')))
    sq=ROOT/'data/squad/dev-v1.1.json'
    if sq.exists():
        d=json.load(open(sq));paras=[' '.join(q['context'].split()) for a in d['data'] for q in a['paragraphs']][:400];_CANARY['squad_dev_first400']=grams(norm(' '.join(paras)))
    return _CANARY
def select(nat_dir,stream,passes,layout='spread',seed=0,canary=None):
    """returns (pools {pass: [paragraphs]}, info). pass 1 is the stream's own background (not in pools)."""
    w=NW.load_world(world_path(nat_dir,stream,layout));own=bg_texts(w);rx=forbid_regex(w);ownset=set(own)
    others=sorted(Path(nat_dir).glob(f'nat_91*_{layout}.json'));others=[p for p in others if p.name!=world_path(nat_dir,stream,layout).name]
    cand=[];seen=set(ownset);drop={'forbid':0,'own_bg':0,'duplicate':0,'canary':0}
    for p in others:
        x=NW.load_world(p);o=x['seed']
        for t in bg_texts(x):
            if rx.search(t):drop['forbid']+=1;continue
            if t in ownset:drop['own_bg']+=1;continue
            if t in seen:drop['duplicate']+=1;continue
            if canary and any(len(grams(norm(t))&g)>0 for g in canary.values()):drop['canary']+=1;continue
            seen.add(t);cand.append((o,t))
    by={}
    for o,t in cand:by.setdefault(o,[]).append(t)
    need=len(own)+MARGIN;npass=passes-1;assert npass>=1 and len(by)>=npass,(len(by),npass)
    bins=[[] for _ in range(npass)];tot=[0]*npass
    for o in sorted(by,key=lambda o:-len(by[o])):
        i=min(range(npass),key=lambda j:tot[j]);bins[i].append(o);tot[i]+=len(by[o])
    pools={};origin={}
    for i,os_ in enumerate(bins):
        ps=2+i;pool=[t for o in sorted(os_) for t in by[o]];random.Random(f'nat|{stream}|{seed}|{ps}').shuffle(pool)
        assert len(pool)>=need,f'pass {ps}: {len(pool)} paragraphs < {need} (slots {len(own)} + margin {MARGIN}); origin streams {os_}'
        pools[ps]=pool;origin[ps]=os_
    return pools,{'own_bg':len(own),'candidates':len(cand),'dropped':drop,'origin_streams':origin,'n_slots':len(own)}
def audit(pools,own,canary=None):
    own_t=list(own);G={1:grams(norm(' '.join(own_t)))};texts={1:own_t};texts.update(pools)
    for p,v in pools.items():G[p]=grams(norm(' '.join(v)))
    ps=sorted(texts);rep={'passes':{},'pair_overlap_8gram':{},'exact_overlap':{},'canary_8gram':{},'pass':True}
    for p in ps:
        v=texts[p];words=[len(x.split()) for x in v]
        rep['passes'][p]={'n_paragraphs':len(v),'duplicates_inside':len(v)-len(set(v)),'words_mean':float(np.mean(words)),'words_total':int(sum(words))}
        rep['pass']&=rep['passes'][p]['duplicates_inside']==0
    for i in ps:
        for j in ps:
            if i<j:
                o=len(G[i]&G[j])/max(1,min(len(G[i]),len(G[j])));e=len(set(texts[i])&set(texts[j]))
                rep['pair_overlap_8gram'][f'{i}-{j}']=o;rep['exact_overlap'][f'{i}-{j}']=e;rep['pass']&=(o<=MAX_OVERLAP and e==0)
    if canary:
        for p in ps:
            rep['canary_8gram'][p]={k:len(G[p]&g)/max(1,len(G[p])) for k,g in canary.items()};rep['pass']&=all(v==0 for v in rep['canary_8gram'][p].values()) if p>1 else True
    return rep
def forbid_hits(pools,world):
    rx=forbid_regex(world);return {p:sum(1 for t in v if rx.search(t)) for p,v in pools.items()}
def md_of(stream,info,rep,hits):
    L=[f'# E33 R filler audit (stream {stream}, CPU)\n',f"slots {info['n_slots']}, candidates after the filter {info['candidates']}, dropped {info['dropped']}, origin streams per pass {info['origin_streams']}\n",f"forbid hits per pass (must be 0): {hits}\n",'| pass | paragraphs | dup. inside | words mean | words total |','|---|---|---|---|---|']
    for p,v in rep['passes'].items():L.append(f"| {p} | {v['n_paragraphs']} | {v['duplicates_inside']} | {v['words_mean']:.1f} | {v['words_total']} |")
    L+=['','| pass pair | word 8-gram overlap (min-normalised) | exact paragraph matches |','|---|---|---|']+[f"| {k} | {v:.4f} | {rep['exact_overlap'][k]} |" for k,v in rep['pair_overlap_8gram'].items()]
    L+=['',f"canary 8-gram share per pass: {rep['canary_8gram']}",f"AUDIT {'PASS' if rep['pass'] else 'FAIL'} (threshold {MAX_OVERLAP})"]
    return '\n'.join(L)+'\n'
def selftest():
    import tempfile
    tmp=Path(tempfile.mkdtemp());rng=random.Random(0);V=[f'w{i}' for i in range(3000)]
    mk=lambda:' '.join(rng.choice(V) for _ in range(60))
    def world(seed,names):
        tri=[{'id':i,'pair':f'p{i//2}','X':f'{n} Zork','XF':n,'XS':'Zork','P':f'Town{i}','Q':None,'E':None} for i,n in enumerate(names)]
        texts=[{'kind':'bg','text':mk()} for _ in range(60)];return {'seed':seed,'triples':tri,'texts':texts,'ctrl_persons':['Ctrl A'],'ctrl_places':['Ctrlville'],'pseudo_persons':['Ps B'],'pseudo_places':['Psville'],'pairs':1,'frames':{}}
    ws={s:world(s,['Al','Bo']) for s in (9101,9102,9103,9104,9106)}
    ws[9102]['texts'][0]['text']=ws[9101]['texts'][1]['text']          # exact duplicate across streams
    ws[9103]['texts'][2]['text']='he visited Town0 once '+mk()           # contains a target string
    ws[9104]['texts'][3]['text']=ws[9101]['texts'][5]['text']            # another duplicate of own bg
    ws[9106]['texts'][4]['text']=ws[9102]['texts'][7]['text']            # duplicate between two other streams
    for s,w in ws.items():json.dump(w,open(tmp/f'nat_{s}_spread.json','w'))
    global MARGIN;m0=MARGIN;MARGIN=0
    try:
        pools,info=select(tmp,9101,3,canary=None);allp=[t for v in pools.values() for t in v]
        rep=audit(pools,bg_texts(ws[9101]))
    finally:MARGIN=m0
    assert not (set(pools[2])&set(pools[3])),'passes share a paragraph'
    assert not (set(allp)&set(bg_texts(ws[9101]))),'own pass-1 background reused'
    assert len(allp)==len(set(allp)),'duplicate paragraph across candidates'
    assert set(info['origin_streams'][2]).isdisjoint(info['origin_streams'][3])
    assert all('Town0' not in t and 'Town1' not in t for t in allp),'target string leaked'
    assert info['dropped']['forbid']>=1 and info['dropped']['duplicate']>=1 and info['dropped']['own_bg']>=1,info['dropped']
    assert rep['pass'],rep
    print('SELFTEST OK',info['dropped'])
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--nat-dir',default=os.environ.get('NAT_DIR','/mnt/project-files/data/natural-text/worlds'));ap.add_argument('--stream',type=int,default=9105);ap.add_argument('--passes',type=int,default=3);ap.add_argument('--layout',default='spread')
    ap.add_argument('--out-dir',default=os.environ.get('EXP07_OUT','/tmp/e33'));ap.add_argument('--selftest',action='store_true');ap.add_argument('--budget-only',action='store_true');a=ap.parse_args()
    if a.selftest:selftest();return
    w=NW.load_world(world_path(a.nat_dir,a.stream,a.layout));canary=canary_grams()
    if not canary:print('WARNING: pg11 / pg1342 not found under data/narrative_gutenberg: canary check skipped',flush=True)
    pools,info=select(a.nat_dir,a.stream,a.passes,a.layout,canary=canary or None)
    own_c=sum(len(t) for t in bg_texts(w));bud={p:sum(len(t) for t in v)/own_c for p,v in pools.items()};print('pool budget (pool characters / pass-1 background characters; character proxy for tokens):',{p:round(r,2) for p,r in bud.items()},flush=True)
    if a.budget_only:sys.exit(0 if all(r>=1.2 for r in bud.values()) else 3)
    rep=audit(pools,bg_texts(w),canary or None);hits=forbid_hits(pools,w);rep['pass']&=all(v==0 for v in hits.values())
    od=Path(a.out_dir);od.mkdir(parents=True,exist_ok=True);md=md_of(a.stream,info,rep,hits);(od/f'nat_filler_audit_{a.stream}.md').write_text(md)
    json.dump({'info':info,'audit':rep,'forbid_hits':hits,'sha_pool':{p:hashlib.sha1('|'.join(v).encode()).hexdigest()[:12] for p,v in pools.items()}},open(od/f'nat_filler_audit_{a.stream}.json','w'),indent=1,default=str);print(md)
    sys.exit(0 if rep['pass'] else 3)
if __name__=='__main__':main()
