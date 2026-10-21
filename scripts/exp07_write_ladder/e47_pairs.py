"""E47 CPU analysis (no GPU; server: needs gen_world, suppress and the tokenizer, like e47_counts.py): how many exposure sentences of the SAME PAIR, OTHER PERSON (other triple) lie in the write window of each chunk, and how they relate to the chunk's fact sentences. All 63 chunks x worlds A / B x seeds x passes 1..P, on the real sentence table (s3_sets.sentence_table). Definitions (Sets.cur_sentences(k) = the exposure sentences whose place token lies in chunk k = the chunk's fact sentences; ts / ps = their triples / pairs; window = positions [k*1024-8192, k*1024)):
  has_O      Sets.oracle(k) non-empty (as in all S3 cells).
  pair_all   sentences of a pair in ps whose triple is another triple of that pair than the fact sentence's, window sentences = those with the place token in the window. np = name + place tokens of those sentences in the window, full = all their tokens in the window. Includes the partner when the partner is ITSELF a fact triple of the chunk.
  P          pair_all minus the sentences of triples that are fact triples of the chunk (= Sets.pxsplit(k)'s P: the partner counts as P only when it has no sentence in the chunk).
  inchunk    the same sentences inside the chunk before the fact sentence's place token (tokens in [k*1024, place token of the fact sentence of that pair)): np / full (union over the chunk's fact sentences).
  per fact sentence: its pair partner (other triple of the pair); partner_prev_round_in_window (a partner sentence of an EARLIER round with the place token in [window start, this sentence's place token)); partner_same_round_before (the partner's sentence of the SAME round before this sentence); partner_in_chunk (a partner sentence with place token in the same chunk); own_prev_in_window (an earlier same-triple sentence in the window: this sentence contributes O).
Usage: e47_pairs.py --seeds 101,102,103 --passes 4 --worlds AB [--out-dir D]   |   --selftest.   Output: <out-dir>/e47_pairs_chunks.csv (one row per chunk), e47_pairs_facts.csv (one row per fact sentence), e47_pairs_summary.md / .json (per seed, world, pass)."""
import sys,json,csv,argparse
from pathlib import Path
import numpy as np
HERE=Path(__file__).resolve().parent;sys.path.insert(0,str(HERE))
import s3_sets as SETS
CH=SETS.CH;W=SETS.W;NCH=63
def analyse(table,seed,world,ps):
    sets=SETS.Sets(table,seed,ps);byp={};byidx={u['idx']:u for u in table}
    for u in table:byp.setdefault(u['pair'],[]).append(u)
    chunks=[];facts=[]
    for k in range(NCH):
        k0=k*CH;lo=max(0,k0-W);cur=sets.cur_sentences(k);O=sets.oracle(k);ts={s['t'] for s in cur};ps_={s['pair'] for s in cur}
        allw=set();Pw=set();inch_np=set();inch_full=set();n_partner_cur=0
        for s in cur:
            part=[u for u in byp[s['pair']] if u['t']!=s['t']]
            if any(u['t'] in ts for u in part if u['t']!=s['t']) and any(k0<=u['place'][0]<k0+CH for u in part):n_partner_cur+=1
            for u in part:
                if lo<=u['place'][0]<k0:allw.add(u['idx'])
                if u['t'] not in ts and lo<=u['place'][0]<k0:Pw.add(u['idx'])
                inch_np|={p for p in u['names']+u['place'] if k0<=p<s['place'][0]};inch_full|={p for p in u['toks'] if k0<=p<s['place'][0]}
            own=[u for u in byp[s['pair']] if u['t']==s['t'] and u['round']<s['round'] and any(lo<=p<s['place'][0] for p in u['names']+u['place'])]
            facts.append({'seed':seed,'world':world,'pass':ps,'k':k,'sent_idx':s['idx'],'triple':s['t'],'pair':s['pair'],'round':s['round'],'has_O_chunk':bool(O),
                'own_prev_in_window':bool(own),'partner_prev_round_in_window':any(u['round']<s['round'] and lo<=u['place'][0]<s['place'][0] for u in part),'partner_same_round_before':any(u['round']==s['round'] and u['place'][0]<s['place'][0] and u['place'][0]>=lo for u in part),
                'partner_in_chunk':any(k0<=u['place'][0]<k0+CH for u in part),'partner_in_window_any_round':any(lo<=u['place'][0]<s['place'][0] for u in part)})
        def tok(idxs,kind):
            S=set()
            for i in idxs:u=byidx[i];S|={p for p in (u['names']+u['place'] if kind=='np' else u['toks']) if lo<=p<k0}
            return len(S)
        P,X=sets.pxsplit(k)
        row={'seed':seed,'world':world,'pass':ps,'k':k,'has_O':bool(O),'n_fact_sent':len(cur),'n_O':len(O),'n_pairs':len(ps_),'n_partner_also_fact':n_partner_cur,
             'win_sent_pair_all':len(allw),'win_np_pair_all':tok(allw,'np'),'win_full_pair_all':tok(allw,'full'),'win_sent_P':len(Pw),'win_np_P':tok(Pw,'np'),'win_full_P':tok(Pw,'full'),'pxsplit_P':len(P),'inchunk_np':len(inch_np),'inchunk_full':len(inch_full)}
        assert (not O) or row['win_full_P']==row['pxsplit_P'],(k,row['win_full_P'],row['pxsplit_P'])   # for the chunks with O, P here == Sets.pxsplit's P (pxsplit is empty for chunks without O, the column is 0 there)
        chunks.append(row)
    return chunks,facts
def summ(chunks,facts):
    s={};O=[r for r in chunks if r['has_O']];N=[r for r in chunks if not r['has_O']];Nc=[r for r in N if r['n_fact_sent']>0]
    s['chunks']=len(chunks);s['chunks_O']=len(O);s['chunks_noO_with_fact_sentence']=len(Nc);s['chunks_noO_no_exposure']=len(N)-len(Nc)
    for nm,rows in (('O',O),('noO_fact',Nc)):
        for f in ('win_sent_pair_all','win_np_pair_all','win_full_pair_all','win_sent_P','win_np_P','win_full_P','inchunk_np','inchunk_full'):s[f'{nm}_sum_{f}']=int(sum(r[f] for r in rows))
        s[f'{nm}_chunks_with_pair_all_in_window']=sum(1 for r in rows if r['win_sent_pair_all']>0);s[f'{nm}_chunks_with_P_in_window']=sum(1 for r in rows if r['win_sent_P']>0)
    s['facts']=len(facts)
    for f in ('own_prev_in_window','partner_prev_round_in_window','partner_same_round_before','partner_in_chunk','partner_in_window_any_round'):s['facts_'+f]=sum(1 for x in facts if x[f])
    s['O_chunks_with_a_partner_also_fact']=sum(1 for r in O if r['n_partner_also_fact']>0);return s
def md_of(res):
    L=['# E47 same-pair other-person sentences in the write window (CPU)\n','Per run (seed, world, pass): chunks with / without O; for the O chunks the summed window tokens of the same-pair other-triple sentences (np = name + place tokens, full = all tokens), for all of them (pair_all) and for those whose triple is not a fact triple of the chunk (P); the number of fact sentences whose partner has an earlier exposure in the window, a same-round sentence before it, a sentence in the same chunk.\n',
       '| run | O chunks | no-O chunks with a fact sentence | O: pair_all np / full | O: P np / full | O chunks with P>0 | no-O: pair_all np / full | no-O chunks with pair_all>0 | facts | own prev in window | partner prev round in window | partner same round before | partner in chunk |','|---|---|---|---|---|---|---|---|---|---|---|---|---|']
    for k,s in res['runs'].items():L.append(f"| {k} | {s['chunks_O']} | {s['chunks_noO_with_fact_sentence']} | {s['O_sum_win_np_pair_all']} / {s['O_sum_win_full_pair_all']} | {s['O_sum_win_np_P']} / {s['O_sum_win_full_P']} | {s['O_chunks_with_P_in_window']} | {s['noO_fact_sum_win_np_pair_all']} / {s['noO_fact_sum_win_full_pair_all']} | {s['noO_fact_chunks_with_pair_all_in_window']} | {s['facts']} | {s['facts_own_prev_in_window']} | {s['facts_partner_prev_round_in_window']} | {s['facts_partner_same_round_before']} | {s['facts_partner_in_chunk']} |")
    return '\n'.join(L)+'\n'
def run(a):
    import e42_k2_tokens as K2
    sys.path.insert(0,str(K2.ROOT/'scripts/exp04_what_ttt_retains'));import gen_world as gw;import suppress as SP
    C=[];F=[];res={'runs':{},'args':vars(a)}
    for seed in [int(x) for x in a.seeds.split(',')]:
        for wd in a.worlds:
            for ps in range(1,a.passes+1):
                w=K2.build_world_pass(gw,seed,ps);table=SETS.sentence_table(w,wd=='B',gw.TOK,gw,SP);c,f=analyse(table,seed,wd,ps);C+=c;F+=f;res['runs'][f's{seed}_{wd}_p{ps}']=summ(c,f)
    return res,C,F
def write(od,res,C,F):
    od=Path(od);od.mkdir(parents=True,exist_ok=True)
    for nm,rows in (('chunks',C),('facts',F)):
        with open(od/f'e47_pairs_{nm}.csv','w',newline='') as fh:
            w=csv.DictWriter(fh,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    json.dump(res,open(od/'e47_pairs_summary.json','w'),indent=1);md=md_of(res);(od/'e47_pairs_summary.md').write_text(md);return md
def selftest():
    import e43_sets_test as T
    tb=T.table();c,f=analyse(tb,101,'A',2);s=summ(c,f);assert len(c)==63 and s['chunks_O']>0 and s['facts']>0 and sum(1 for r in c if r['n_fact_sent'])>0
    for r in c:assert r['win_full_P']<=r['win_full_pair_all']
    print('e47_pairs SELFTEST OK (synthetic table)',{k:s[k] for k in ('chunks_O','chunks_noO_with_fact_sentence','O_sum_win_np_pair_all','O_sum_win_np_P','facts','facts_partner_in_chunk')})
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--seeds',default='101,102,103');ap.add_argument('--passes',type=int,default=4);ap.add_argument('--worlds',default='AB');ap.add_argument('--out-dir',default='/tmp/e47_pairs');ap.add_argument('--selftest',action='store_true');a=ap.parse_args()
    if a.selftest:selftest();sys.exit(0)
    res,C,F=run(a);print(write(a.out_dir,res,C,F))
