"""E47 OTHER decomposition, CPU analysis (no GPU; server: gen_world, suppress and the tokenizer): does the X set (sentences of other pairs inside the window, outside K3, chunks with an oracle set) share vocabulary with the chunk's own pair, and are the cells kx_s / kx_o (same / another sentence frame) sized as e47_counts says. Seed(s), worlds A / B, passes 1..P, all 63 chunks.
(a) World vocabulary (per world and pass, from the sentence table and the realized text): every triple has its own person name, place and occupation (gen_world.build_world: persons[i], places[i], occs[i], all distinct per triple) -> word-level multiplicity 1 by construction (asserted: no name / place / occupation string is used by two triples); but the strings are built from small syllable sets (30 x 20 person names, 20 x 15 place names) and tokenized into pieces, so the TOKEN ids repeat across triples: per category (name / place / occupation) the number of distinct token ids, the number of ids used by >= 2 triples, the mean number of triples per id; the 30 sentence frames (gen_world.FR) are shared by all triples (each triple uses its own random order of the frames; uses per frame are counted).
(b) For every chunk with O and every X sentence (a sentence of another pair with tokens in the window [k*1024-8192, k*1024), as Sets.pxsplit's X): shares a WORD with the chunk's pair(s) = the sentence text contains (whole-word) the name, place or occupation string of a person of a pair that has a fact sentence in the chunk; shares a TOKEN = some token id of its name / place / occupation tokens occurs among the ids of the same category of the chunk pairs' triples. Counted as sentences and as the window tokens of those sentences, split by frame (same frame id as a fact sentence of the chunk / other frame, Sets.xframe_split) and in total; with the shares of the group.
(c) Sizes: n tokens of kx_s (K3 + X same frame), kx_o (K3 + X other frame), kx (K3 + X), their ratios to K3 (sums over the O chunks), and the equality with e47_counts.counts (n_kx, n_X_same_frame, n_X_other_frame; asserted per chunk).
Output: <out-dir>/e47_other.{json,md}.   Usage: e47_other.py --seeds 101 --passes 4 --worlds AB [--out-dir D]   |   --selftest (synthetic table, ids and words)"""
import sys,json,argparse,re
from pathlib import Path
import numpy as np
HERE=Path(__file__).resolve().parent;sys.path.insert(0,str(HERE))
import s3_sets as SETS
import e47_counts as C47
CH=SETS.CH;W=SETS.W;NCH=63
def vocab(table,ids,words,frames_n=30):
    """(a): table with names / place / occ token positions, ids (BOS at 0), words {triple: {'X','P','o'}}"""
    res={'n_triples':len(words)}
    for cat,key in (('name','X'),('place','P'),('occ','o')):
        strs=[w[key] for w in words.values()];assert len(set(strs))==len(strs),(cat,'a string is used by two triples');res[f'{cat}_strings']=len(strs);res[f'{cat}_multiplicity_max']=1
    for cat,fld in (('name','names'),('place','place'),('occ','occ')):
        use={}
        for u in table:
            for j in u[fld]:use.setdefault(ids[j],set()).add(u['t'])
        n=[len(v) for v in use.values()];res[f'{cat}_token_ids']=len(use);res[f'{cat}_ids_in_ge2_triples']=sum(1 for x in n if x>=2);res[f'{cat}_mean_triples_per_id']=float(np.mean(n)) if n else None;res[f'{cat}_max_triples_per_id']=max(n) if n else 0
    fr={}
    for u in table:fr[u.get('frame',-1)]=fr.get(u.get('frame',-1),0)+1
    res['frames_used']=len([f for f in fr if f>=0]);res['frame_uses_min']=min(fr.values()) if fr else 0;res['frame_uses_max']=max(fr.values()) if fr else 0;res['frame_unknown_sentences']=fr.get(-1,0);return res
def analyse(table,ids,words,seed,ps):
    sets=SETS.Sets(table,seed,ps);byt={}
    for u in table:byt.setdefault(u['t'],[]).append(u)
    cat_ids={t:{'name':{ids[j] for u in byt[t] for j in u['names']},'place':{ids[j] for u in byt[t] for j in u['place']},'occ':{ids[j] for u in byt[t] for j in u['occ']}} for t in byt}
    pair_ts={}
    for u in table:pair_ts.setdefault(u['pair'],set()).add(u['t'])
    rows=[];agg={}
    def add(g,key,val):agg.setdefault(g,{}).setdefault(key,0);agg[g][key]+=val
    for k in range(NCH):
        O=sets.oracle(k)
        if not O:continue
        k0=k*CH;lo=max(0,k0-W);cur=sets.cur_sentences(k);cpairs={s['pair'] for s in cur};ts={s['t'] for s in cur}
        pw=set();pids={'name':set(),'place':set(),'occ':set()}
        for pr in cpairs:
            for t in pair_ts[pr]:
                pw|={words[t]['X'],words[t]['P'],words[t]['o']}
                for c in pids:pids[c]|=cat_ids[t][c]
        fr={s.get('frame',-1) for s in cur}-{-1};P,X=sets.pxsplit(k);Xs,Xo=sets.xframe_split(k);Xset=set(X)
        for u in table:
            if u['pair'] in cpairs:continue
            tk=[p for p in u['toks'] if lo<=p<k0]
            if not tk:continue
            g='same_frame' if u.get('frame',-1) in fr else 'other_frame'
            w_share=any(re.search(r'\b'+re.escape(x)+r'\b',u['text']) for x in pw) if u.get('text') else False
            tshare={c:any(ids[j] in pids[c] for j in u[f]) for c,f in (('name','names'),('place','place'),('occ','occ'))}
            for gg in (g,'all'):
                add(gg,'sentences',1);add(gg,'tokens',len(tk))
                if w_share:add(gg,'word_sent',1);add(gg,'word_tok',len(tk))
                for c,v in tshare.items():
                    if v:add(gg,f'tok_{c}_sent',1);add(gg,f'tok_{c}_tok',len(tk))
                if any(tshare.values()):add(gg,'tok_any_sent',1);add(gg,'tok_any_tok',len(tk))
        K3=set(sets.sentencecover(k));kxs,kxo,kx=sets.kxsmask(k),sets.kxomask(k),sets.kxmask(k)
        assert set(Xs)|set(Xo)==Xset and len(kx)==len(K3|Xset)
        rows.append({'k':k,'n_K3':len(K3),'n_X':len(X),'n_X_same':len(Xs),'n_X_other':len(Xo),'n_kx_s':len(kxs),'n_kx_o':len(kxo),'n_kx':len(kx)})
    ref=C47.counts(sets);assert [(r['n_kx'],r['n_X_same_frame'],r['n_X_other_frame']) for r in ref]==[(r['n_kx'],r['n_X_same'],r['n_X_other']) for r in rows],'disagrees with e47_counts'
    S={'n_O_chunks':len(rows)}
    for f in ('n_K3','n_X','n_X_same','n_X_other','n_kx_s','n_kx_o','n_kx'):S['sum_'+f]=int(sum(r[f] for r in rows))
    S['kx_s_over_K3']=S['sum_n_kx_s']/max(1,S['sum_n_K3']);S['kx_o_over_K3']=S['sum_n_kx_o']/max(1,S['sum_n_K3']);S['kx_over_K3']=S['sum_n_kx']/max(1,S['sum_n_K3']);S['X_same_share_tokens']=S['sum_n_X_same']/max(1,S['sum_n_X'])
    return rows,S,agg
def md_of(res):
    L=['# E47 OTHER decomposition: X vocabulary sharing and kx_s / kx_o sizes (CPU)\n','## (a) world vocabulary (per run: token ids per category: distinct / used by >= 2 triples / mean triples per id; strings are unique per triple)\n','| run | triples | name ids | place ids | occupation ids | frames used (uses min-max) |','|---|---|---|---|---|---|']
    for k,v in res['runs'].items():
        a=v['vocab'];L.append(f"| {k} | {a['n_triples']} | {a['name_token_ids']} / {a['name_ids_in_ge2_triples']} / {a['name_mean_triples_per_id']:.2f} | {a['place_token_ids']} / {a['place_ids_in_ge2_triples']} / {a['place_mean_triples_per_id']:.2f} | {a['occ_token_ids']} / {a['occ_ids_in_ge2_triples']} / {a['occ_mean_triples_per_id']:.2f} | {a['frames_used']} ({a['frame_uses_min']}-{a['frame_uses_max']}) |")
    L+=['\n## (b) X sentences (window, other pairs, chunks with O) sharing a word / a token id with the chunk pair (sentences / window tokens, share of the group)\n','| run | group | X sentences | X tokens | word shared (sent) | token shared any (sent / tok) | name ids (sent) | place ids (sent) | occ ids (sent) |','|---|---|---|---|---|---|---|---|---|']
    for k,v in res['runs'].items():
        for g in ('all','same_frame','other_frame'):
            a=v['agg'].get(g,{});n=max(1,a.get('sentences',0));t=max(1,a.get('tokens',0));L.append(f"| {k} | {g} | {a.get('sentences',0)} | {a.get('tokens',0)} | {a.get('word_sent',0)} ({a.get('word_sent',0)/n:.3f}) | {a.get('tok_any_sent',0)} ({a.get('tok_any_sent',0)/n:.3f}) / {a.get('tok_any_tok',0)} ({a.get('tok_any_tok',0)/t:.3f}) | {a.get('tok_name_sent',0)} ({a.get('tok_name_sent',0)/n:.3f}) | {a.get('tok_place_sent',0)} ({a.get('tok_place_sent',0)/n:.3f}) | {a.get('tok_occ_sent',0)} ({a.get('tok_occ_sent',0)/n:.3f}) |")
    L+=['\n## (c) cell sizes (sums over the O chunks; equal to e47_counts, asserted)\n','| run | O chunks | sum K3 | sum X | X same frame | X other frame | kx_s | kx_o | kx | kx_s / K3 | kx_o / K3 | kx / K3 |','|---|---|---|---|---|---|---|---|---|---|---|---|']
    for k,v in res['runs'].items():
        s=v['summary'];L.append(f"| {k} | {s['n_O_chunks']} | {s['sum_n_K3']} | {s['sum_n_X']} | {s['sum_n_X_same']} | {s['sum_n_X_other']} | {s['sum_n_kx_s']} | {s['sum_n_kx_o']} | {s['sum_n_kx']} | {s['kx_s_over_K3']:.3f} | {s['kx_o_over_K3']:.3f} | {s['kx_over_K3']:.3f} |")
    return '\n'.join(L)+'\n'
def run(a):
    import e42_k2_tokens as K2
    sys.path.insert(0,str(K2.ROOT/'scripts/exp04_what_ttt_retains'));import gen_world as gw;import suppress as SP
    res={'runs':{},'args':vars(a)}
    for seed in [int(x) for x in a.seeds.split(',')]:
        for wd in a.worlds:
            for ps in range(1,a.passes+1):
                w=K2.build_world_pass(gw,seed,ps);swap=wd=='B';table=SETS.sentence_table(w,swap,gw.TOK,gw,SP);text,name,_=gw.realize(w,swap);ids=[None]+list(gw.TOK.encode(text,add_special_tokens=False).ids)
                words={t['id']:{'X':name[t['id']],'P':t['P'],'o':t['o']} for t in w['triples']};rows,S,agg=analyse(table,ids,words,seed,ps);res['runs'][f's{seed}_{wd}_p{ps}']={'vocab':vocab(table,ids,words),'summary':S,'agg':agg,'chunks':rows}
    return res
def selftest():
    import e47_sets_test as T
    tb=T.table();rng=np.random.default_rng(3);N=max(p for u in tb for p in u['toks'])+2;ids=[None]+[int(x) for x in rng.integers(0,40,N)]
    for u in tb:u['text']=f"Person{u['t']} lived in Place{u['t']} as smith{u['t']} ."
    words={u['t']:{'X':f"Person{u['t']}",'P':f"Place{u['t']}",'o':f"smith{u['t']}"} for u in tb}
    for u in tb:u['occ']=[u['place'][0]+1]
    rows,S,agg=analyse(tb,ids,words,101,2);v=vocab(tb,ids,words);assert rows and S['sum_n_X']>0 and agg['all']['sentences']>0 and agg['all'].get('word_sent',0)==0 and v['name_token_ids']>0
    assert agg['all']['sentences']==agg.get('same_frame',{}).get('sentences',0)+agg.get('other_frame',{}).get('sentences',0)
    print('e47_other SELFTEST OK (synthetic):',{k:S[k] for k in ('n_O_chunks','sum_n_K3','sum_n_X','kx_s_over_K3','kx_o_over_K3')},agg['all']);print(v)
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--seeds',default='101');ap.add_argument('--passes',type=int,default=4);ap.add_argument('--worlds',default='AB');ap.add_argument('--out-dir',default='/tmp/e47_other');ap.add_argument('--selftest',action='store_true');a=ap.parse_args()
    if a.selftest:selftest();sys.exit(0)
    res=run(a);od=Path(a.out_dir);od.mkdir(parents=True,exist_ok=True);json.dump(res,open(od/'e47_other.json','w'),indent=1);md=md_of(res);(od/'e47_other.md').write_text(md);print(md)
