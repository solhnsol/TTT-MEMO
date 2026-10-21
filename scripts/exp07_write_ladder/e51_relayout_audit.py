"""E51 one-command filler audit (E33 2.6 rules) for relayout worlds: for every target stream and layout, pass-1 = the relayout world's own background, passes 2..P = pools drawn from OTHER streams' background
(nat_filler.select + audit + forbid_hits, canary 8-grams when pg11 / pg1342 exist). Default pool mode = paragraph-level pools (--pool-mode para, 1.2x character rule per filler pass; stream = old whole-stream bins). Pool streams are explicit (--pool-streams, default = the 1st-tier streams 9101,9103,9104,9105; the sealed 9102 / 9106 are NOT used unless
named; add --pool-sealed to include them). The background paragraphs are identical in every layout of a stream, so the pool of a stream is read from the file given for it (relayout file if present in --relayout-dir, else the base file in --nat-dir).
Reports per (stream, layout): pool budget per pass (paragraphs vs slots + margin), duplicate paragraphs inside the own pass-1 background (relayout reuse), 8-gram overlap between passes, forbidden-string hits, AUDIT PASS/FAIL; exit 3 on any FAIL or an impossible pool.
Usage: e51_relayout_audit.py --relayout-dir DIR --nat-dir DIR --streams 9103,9101,9104,9105 --layouts relA,relB --passes 3 [--pool-streams ...] [--pool-sealed] [--out-dir D]   |   --selftest"""
import os,sys,json,argparse,tempfile,shutil
from pathlib import Path
HERE=Path(__file__).resolve().parent;sys.path.insert(0,str(HERE))
import nat_world as NW,nat_filler as NF,e51_nat_table as T
SEALED=(9102,9106)
CHAR_RULE=1.2   # nat_filler --budget-only: pool characters / own pass-1 characters >= 1.2 per filler pass
def select_para(nat_dir,stream,passes,layout,canary=None,seed=0):
    """paragraph-level pools (decision E51 11): candidates = background paragraphs of the OTHER streams present in nat_dir (the caller stages only non-sealed ones) after the nat_filler filters (forbidden strings of the target, own background, duplicates, canary 8-grams);
    shuffled deterministically and cut into passes-1 bins of equal character share; every bin must reach CHAR_RULE x the own pass-1 characters. Returns (pools {pass: [paragraphs]}, info)."""
    import random
    w=NW.load_world(NF.world_path(nat_dir,stream,layout));own=NF.bg_texts(w);rx=NF.forbid_regex(w);ownset=set(own)
    others=[p for p in sorted(Path(nat_dir).glob(f'nat_91*_{layout}.json')) if p.name!=NF.world_path(nat_dir,stream,layout).name]
    cand=[];seen=set(ownset);drop={'forbid':0,'own_bg':0,'duplicate':0,'canary':0};origin={}
    for p in others:
        x=NW.load_world(p)
        for t in NF.bg_texts(x):
            if rx.search(t):drop['forbid']+=1;continue
            if t in ownset:drop['own_bg']+=1;continue
            if t in seen:drop['duplicate']+=1;continue
            if canary and any(len(NF.grams(NF.norm(t))&g)>0 for g in canary.values()):drop['canary']+=1;continue
            seen.add(t);cand.append(t);origin[t]=x['seed']
    random.Random(f'nat-para|{stream}|{seed}').shuffle(cand);npass=passes-1;oc=sum(len(t) for t in own);tot=sum(len(t) for t in cand);bins=[[] for _ in range(npass)];acc=0
    for t in cand:
        b=min(npass-1,int(acc/(tot/npass)));bins[b].append(t);acc+=len(t)
    ratios=[sum(len(t) for t in b)/oc for b in bins]
    if min(ratios)<CHAR_RULE:raise AssertionError(f'paragraph pool too small for {passes} passes: char ratios per filler pass {[round(r,2) for r in ratios]} < {CHAR_RULE} (candidates {len(cand)}, own paragraphs {len(own)})')
    pools={2+i:b for i,b in enumerate(bins)}
    return pools,{'own_bg':len(own),'candidates':len(cand),'dropped':drop,'origin_streams':{ps:sorted({origin[t] for t in v}) for ps,v in pools.items()},'n_slots':len(own),'char_ratio_per_pass':[round(r,2) for r in ratios]}
def pool_use(pool,slot_tok,enc_len,slack=8):
    """replicates the consumption of nat_pass.build_pass_stream_matched: slots in stream order take consecutive pool paragraphs until n + slack tokens are covered (the last one is cut, the rest of it is lost); returns (paragraphs used or None when exhausted, token share of the pool used)"""
    pl=[enc_len(t)+1 for t in pool];cur=0;used=0
    for n in slot_tok:
        acc=0
        while True:
            if cur>=len(pl):return None,None
            acc+=pl[cur];used+=pl[cur];cur+=1
            if acc>=n+slack:break
    return cur,used/sum(pl)
def audit_one(wdir,stream,layout,passes,margin=None,mode='para',enc_len=None,max_share=0.95):
    canary=NF.canary_grams() or None
    w=NW.load_world(NF.world_path(wdir,stream,layout))
    if margin is not None:NF.MARGIN=margin
    try:pools,info=(select_para(wdir,stream,passes,layout,canary) if mode=='para' else NF.select(wdir,stream,passes,layout,canary=canary))
    except AssertionError as e:return dict(stream=stream,layout=layout,ok=False,error=str(e)[:300])
    own=NF.bg_texts(w);rep=NF.audit(pools,own,canary);hits=NF.forbid_hits(pools,w);rep['pass']&=all(v==0 for v in hits.values())
    cons={}
    if enc_len is not None:   # matched-builder consumption (nat_pass.build_pass_stream_matched): every filler pass must fit in its pool with room to spare
        slot_tok=[enc_len(t) for t in own]
        for ps,v in pools.items():
            k,sh=pool_use(v,slot_tok,enc_len);cons[ps]=dict(paragraphs_used=k,pool_paragraphs=len(v),share=sh);rep['pass']&=bool(k) and sh<=max_share
    return dict(consumption=cons,stream=stream,layout=layout,ok=bool(rep['pass']),own_duplicates=rep['passes'][1]['duplicates_inside'],n_slots=info['n_slots'],pool_sizes={p:len(v) for p,v in pools.items()},origin=info['origin_streams'],dropped=info['dropped'],
                overlap=rep['pair_overlap_8gram'],exact=rep['exact_overlap'],forbid_hits=hits,canary=bool(canary),char_ratio_per_pass=info.get('char_ratio_per_pass'))
def stage(tmp,nat_dir,rel_dir,streams,pool_streams,layouts):
    """one temp dir per layout holding exactly the target streams' relayout files and the pool streams' files (same layout name), via symlinks"""
    for lay in layouts:
        for st in sorted(set(streams)|set(pool_streams)):
            src=NF.world_path(rel_dir,st,lay)
            if not src.exists():src=NF.world_path(nat_dir,st,'spread')
            if not src.exists():raise SystemExit(f'e51_relayout_audit: no file for stream {st} layout {lay} ({src})')
            dst=NF.world_path(tmp,st,lay)
            os.symlink(src.resolve(),dst)
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--relayout-dir');ap.add_argument('--nat-dir',default=None);ap.add_argument('--streams',default='9103,9101,9104,9105');ap.add_argument('--layouts',default='relA,relB');ap.add_argument('--passes',type=int,default=2);ap.add_argument('--pool-mode',default='para',choices=['para','stream'],help='para = paragraph-level pools with the 1.2x character rule (E51 decision); stream = nat_filler.select (whole streams per pass, +150 paragraphs margin)')
    ap.add_argument('--pool-streams',default='9101,9103,9104,9105');ap.add_argument('--pool-sealed',action='store_true');ap.add_argument('--out-dir',default=None);ap.add_argument('--selftest',action='store_true');ap.add_argument('--consumption',action='store_true',help='also simulate the pass-2.. pool consumption of the matched builder with the tokenizer (needs --tokenizer-file or the server tokenizer; --require-real to refuse the proxy)');ap.add_argument('--tokenizer-file',default=None);ap.add_argument('--require-real',action='store_true');a=ap.parse_args()
    if a.selftest:selftest();return
    nd=T.find_nat_dir(a.nat_dir);streams=[int(x) for x in a.streams.split(',')];pool=[int(x) for x in a.pool_streams.split(',')]
    if a.pool_sealed:pool=sorted(set(pool)|set(SEALED))
    enc_len=None
    if a.consumption:
        import nat_dryrun as D,nat_relayout as RL
        try:tk=D.load_tok(a.tokenizer_file)
        except Exception as e:print('real tokenizer not usable:',repr(e)[:100]);tk=None
        assert tk is not None or not a.require_real,'--require-real: no real tokenizer'
        enc_len=RL.ntok_fn(tk);print('consumption simulation tokenizer:','REAL' if tk is not None else 'PROXY (logic only)')
    layouts=a.layouts.split(',');allres=[];tmp=tempfile.mkdtemp()
    try:
        stage(tmp,nd,a.relayout_dir,streams,pool,layouts)
        for lay in layouts:
            for st in streams:
                r=audit_one(tmp,st,lay,a.passes,mode=a.pool_mode,enc_len=enc_len);allres.append(r);print(json.dumps(r,default=str));print(f"{st} {lay}: {'PASS' if r['ok'] else 'FAIL'}"+(f" ({r.get('error')})" if r.get('error') else ''))
    finally:shutil.rmtree(tmp,ignore_errors=True)
    if a.out_dir:Path(a.out_dir).mkdir(parents=True,exist_ok=True);(Path(a.out_dir)/'e51_relayout_audit.json').write_text(json.dumps(allres,indent=1,default=str))
    print('pool streams used:',pool,'(sealed 9102/9106 excluded)' if not a.pool_sealed else '(sealed streams INCLUDED)')
    sys.exit(0 if all(r['ok'] for r in allres) else 3)
def selftest():
    import random
    tmp=Path(tempfile.mkdtemp());rng=random.Random(0);V=[f'w{i}' for i in range(5000)];mk=lambda:' '.join(rng.choice(V) for _ in range(60))
    def world(seed):
        tri=[{'id':i,'pair':f'p{i//2}','X':f'N{seed}{i} Zork{seed}{i}','XF':f'N{seed}{i}','XS':f'Zork{seed}{i}','P':f'Town{seed}{i}','Q':None,'E':None} for i in range(2)]
        return {'seed':seed,'layout':'relA','triples':tri,'texts':[{'kind':'bg','text':mk()} for _ in range(40)],'ctrl_persons':[f'C{seed}'],'ctrl_places':[f'Cv{seed}'],'pseudo_persons':[f'Ps{seed}'],'pseudo_places':[f'Pv{seed}'],'pairs':1,'frames':{}}
    for s in (9101,9103,9104,9105,9102):json.dump(world(s),open(tmp/f'nat_{s}_relA.json','w'))
    json.dump(world(9101),open(tmp/'nat_9101_spread.json','w'))
    t2=Path(tempfile.mkdtemp());NF.MARGIN=0
    stage(t2,tmp,tmp,[9103],[9101,9104,9105],['relA']);r=audit_one(t2,9103,'relA',3,margin=0);assert r['ok'] and sorted(r['origin'].keys())==[2,3],r
    assert not any(9102 in v for v in r['origin'].values()),'sealed stream used'
    # a pool that is too small is reported, not silently passed
    t3=Path(tempfile.mkdtemp());stage(t3,tmp,tmp,[9103],[9101],['relA']);r=audit_one(t3,9103,'relA',3,margin=0);assert not r['ok'] and 'error' in r
    t4=Path(tempfile.mkdtemp());stage(t4,tmp,tmp,[9103],[9101,9104,9105],['relA']);r=audit_one(t4,9103,'relA',3,mode='para');assert r['ok'] and r['char_ratio_per_pass'] and min(r['char_ratio_per_pass'])>=1.2,r
    t5=Path(tempfile.mkdtemp());stage(t5,tmp,tmp,[9103],[9101],['relA']);r=audit_one(t5,9103,'relA',3,mode='para');assert not r['ok'] and 'too small' in r['error'],r
    print('e51_relayout_audit SELFTEST OK')
if __name__=='__main__':main()
