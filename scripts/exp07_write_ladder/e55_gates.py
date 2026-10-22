"""E55 v3 gate checks of the background (design 13.3), CPU, before any training starts. For one cell and seed (decoys names + nofact, so the paragraph pool is cut in halves: real / names stream = first half, B-arm filler = second half):
 (a) no background unit is used twice in the stream (token-sequence duplicates among the used units = 0);
 (b) no 13-gram of tokens occurs in two different background units of the stream (fixed-format paragraphs would show up here);
 (c) 13-gram overlap with reference texts is 0: frame bank sentences (slots removed), question frames, and every --ref NAME=PATH text file (SQuAD readout context, canary books pg11 / pg1342, probe text ...). A missing file is reported NOT CHECKED, never as passed;
 (d) the B-arm filler (spare half) is not used twice, has no 13-gram in common with the stream background or the references;
 (e) need / available tokens per slice, ratio >= 1.5 (pool total vs. both needs reported too).
  python3 e55_gates.py --cell v3_16_1 --seed 61 [--ref squad=PATH ...] [--out gates.md]    (exit 1 if a checked gate fails)
  python3 e55_gates.py --selftest                                                             (synthetic data, injected violations are caught)"""
import sys,argparse,re
from pathlib import Path
HERE=Path(__file__).resolve().parent;sys.path.insert(0,str(HERE));sys.path.insert(0,str(HERE.parents[0]/'exp04_what_ttt_retains'))
import gen_facts as G
NG=13
def grams(ids,n=NG):
    ids=list(ids);return {tuple(ids[i:i+n]) for i in range(len(ids)-n+1)}
def _owner_map(seqs):
    """13-gram -> index of the first sequence that has it; -> (map, number of (gram, later sequence) collisions, first examples)"""
    own={};coll=0;ex=[]
    for i,u in enumerate(seqs):
        for g_ in grams(u):
            if g_ in own and own[g_]!=i:
                coll+=1
                if len(ex)<3:ex.append((own[g_],i))
            own.setdefault(g_,i)
    return own,coll,ex
def check(world,enc,bg,refs=None,ratio_min=1.5):
    """-> (dict of results, markdown text, ok). refs: {name: text or None (file missing)}"""
    sep=world['sep'];ls=len(sep);half=len(bg)//2;R={}
    units=G.make_units(bg[:half],enc,sep);spare=G.make_units(bg[half:],enc,sep);used=world['units_used']
    U=[tuple(u[0][:-ls]) for u in units[:used]];R['units_used']=used;R['units_available_real_half']=len(units)
    R['a_dup_units_used']=len(U)-len(set(U))
    own,coll,ex=_owner_map(U);R['b_13gram_shared_between_units']=coll;R['b_examples']=ex
    bg_tok=sum(len(u) for u in U);avail_real=sum(len(u[0])-ls for u in units)
    refg={}
    refg['frame bank (slots removed)']=set().union(*[grams(enc(x)) for t in world['bank'] for x in re.split(r'\{\w+\}',t) if len(x.split())>=3])
    refg['question frames']=set().union(*[grams(enc(q.replace('{X}','X'))) for q in G.QFRAMES])
    status={}
    for nm,txt in (refs or {}).items():
        if txt is None:status[nm]='NOT CHECKED (file missing)';continue
        refg[nm]=grams(enc(txt));status[nm]='checked'
    R['c_overlap_stream_vs_ref']={nm:len(allg&set(own)) for nm,allg in refg.items()};R['c_ref_status']=status
    # (d) filler: exactly the tokens the nofact stream takes from the spare half
    need_fill=sum(x['end']-x['start'] for c in world['chunks'] for x in c['sents']);fill=[t for u in spare for t in u[0][:-ls]]
    R['d_filler_tokens_needed']=need_fill;R['d_filler_tokens_available']=len(fill)
    fused=fill[:need_fill];fg=grams(fused);R['d_overlap_filler_vs_stream']=len(fg&set(own));R['d_overlap_filler_vs_refs']={nm:len(fg&g_) for nm,g_ in refg.items()}
    fu=[tuple(u[0][:-ls]) for u in spare];R['d_dup_units_in_spare']=len(fu)-len(set(fu))
    if 'nofact' in world:
        ids=[t for c in world['nofact']['chunks'] for t in c['ids']];R['d_nofact_stream_is_real_with_filler']=len(ids)==len(world['chunks'])*G.CH
    R['e_ratio_real_half']=avail_real/max(bg_tok,1);R['e_ratio_filler_half']=len(fill)/max(need_fill,1);R['e_tokens']={'stream_background_used':bg_tok,'real_half_available':avail_real,'filler_needed':need_fill,'spare_half_available':len(fill),'pool_total':avail_real+len(fill)}
    ok=(R['a_dup_units_used']==0 and R['b_13gram_shared_between_units']==0 and all(v==0 for v in R['c_overlap_stream_vs_ref'].values()) and R['d_overlap_filler_vs_stream']==0 and all(v==0 for v in R['d_overlap_filler_vs_refs'].values())
        and R['d_dup_units_in_spare']==0 and R['e_ratio_real_half']>=ratio_min and R['e_ratio_filler_half']>=ratio_min)
    P=lambda b:'PASS' if b else 'FAIL'
    L=[f"# E55 v3 background gates: cell {world['k']}x{world['n']}, {world['n_chunks']} chunks, seed {world['seed']}",'','| gate | value | result |','|---|---|---|',
       f"| (a) background units used twice in the stream | {R['a_dup_units_used']} of {used} | {P(R['a_dup_units_used']==0)} |",f"| (b) 13-grams shared by two different background units | {coll} {ex} | {P(coll==0)} |"]
    for nm,v in R['c_overlap_stream_vs_ref'].items():L.append(f"| (c) 13-gram overlap stream vs {nm} | {v} ({status.get(nm,'checked')}) | {P(v==0)} |")
    for nm,s_ in status.items():
        if s_!='checked':L.append(f"| (c) {nm} | {s_} | NOT CHECKED |")
    L+=[f"| (d) filler vs stream background 13-gram overlap | {R['d_overlap_filler_vs_stream']} | {P(R['d_overlap_filler_vs_stream']==0)} |",f"| (d) filler vs references | {R['d_overlap_filler_vs_refs']} | {P(all(v==0 for v in R['d_overlap_filler_vs_refs'].values()))} |",f"| (d) duplicate units in the spare half | {R['d_dup_units_in_spare']} | {P(R['d_dup_units_in_spare']==0)} |",
        f"| (e) available / needed, stream half | {R['e_ratio_real_half']:.2f} ({avail_real} / {bg_tok} tokens) | {P(R['e_ratio_real_half']>=ratio_min)} |",f"| (e) available / needed, filler half | {R['e_ratio_filler_half']:.2f} ({len(fill)} / {need_fill} tokens) | {P(R['e_ratio_filler_half']>=ratio_min)} |",'',
        f"checked references: {', '.join(k for k in refg)}; every reference given with a missing file stays NOT CHECKED and the run must not start on a fully clean table without it (the caller decides).",f"overall (of the checked gates): **{P(ok)}**"]
    return R,'\n'.join(L)+'\n',ok
def selftest():
    import random
    enc=G.StubTok();r=random.Random(3);vocab=[f'w{i}' for i in range(6000)]
    para=lambda:' '.join(r.choice(vocab) for _ in range(60))+'.'
    bg=[para() for _ in range(3000)]
    w=G.build_fact_world(61,enc=enc,bg=bg,decoys=('names','nofact'),**G.CELLS['v3_4_8']);R,t,ok=check(w,enc,bg,{'canary':'w1 w2 w3 w4 w5 w6 w7 w8 w9 w10 w11 w12 w13 w14','missing':None})
    assert ok and R['b_13gram_shared_between_units']==0 and R['e_ratio_real_half']>=1.5,t
    # injected violations: a duplicated paragraph inside the used region, and a reference that equals a stream passage
    bg2=bg[:];bg2[1]=bg2[0];w2=G.build_fact_world(61,enc=enc,bg=bg2,decoys=('names','nofact'),**G.CELLS['v3_4_8']);R2,t2,ok2=check(w2,enc,bg2)
    assert not ok2 and (R2['a_dup_units_used']>0 or R2['b_13gram_shared_between_units']>0),t2
    ref=' '.join(bg[0].split()[:40]);R3,t3,ok3=check(w,enc,bg,{'leak':ref});assert not ok3 and R3['c_overlap_stream_vs_ref']['leak']>0
    sp=' '.join(bg[len(bg)//2+1].split()[:40]);R4,t4,ok4=check(w,enc,bg,{'leak2':sp});assert not ok4 and R4['d_overlap_filler_vs_refs']['leak2']>0
    print(t.splitlines()[2:12]);print('e55_gates SELFTEST OK (synthetic pool; clean case passes, 4 injected violations are caught)')
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--selftest',action='store_true');ap.add_argument('--cell');ap.add_argument('--seed',type=int);ap.add_argument('--ref',action='append',default=[]);ap.add_argument('--require-refs',action='store_true',help='a reference whose file is missing = failure (exit 2)');ap.add_argument('--default-refs',action='store_true',help='server: SQuAD dev paragraphs [:400] (the neutral readout context), canary books pg11 / pg1342, E35 probe text');ap.add_argument('--tokenizer-file',default=None);ap.add_argument('--out',default=None);a=ap.parse_args()
    if a.selftest:selftest();sys.exit(0)
    ROOT=HERE.parents[1];enc=G.load_enc(a.tokenizer_file or str(ROOT/'data/tokenizer/llama3/tokenizer.json'));bg=G.default_bg_pool(a.seed)
    refs={}
    if a.default_refs:
        import json
        sq=ROOT/'data/squad/dev-v1.1.json';refs['SQuAD readout context']=' '.join(' '.join(p['context'].split()) for p in [p for x in json.load(open(sq))['data'] for p in x['paragraphs']][:400]) if sq.exists() else None
        for b in ('pg11','pg1342'):
            f=ROOT/f'data/narrative_gutenberg/{b}.txt';refs['canary '+b]=f.read_text(errors='ignore') if f.exists() else None
    for x in a.ref:
        nm,_,pth=x.partition('=');refs[nm]=Path(pth).read_text(errors='ignore') if Path(pth).exists() else None
    miss=[k for k,v in refs.items() if v is None]
    if a.require_refs and miss:print('reference files missing:',miss);sys.exit(2)
    w=G.cell_world(a.seed,a.cell,enc,bg,decoys=('names','nofact'));R,t,ok=check(w,enc,bg,refs);print(t)
    if a.out:Path(a.out).write_text(t)
    sys.exit(0 if ok else 1)
