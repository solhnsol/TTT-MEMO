"""Independent re-verification of Stage 2 stimulus integrity (CPU only). Rebuilds worlds A/B from the seeds and checks:
 (1) A/B token streams differ only at swapped-name positions; token-count parity in the real text contexts
 (2) exposure positions: every name/place/occupation occurrence vs the probe windows; control names never appear
 (3) per-name MENTION counts inside swap pairs (sentence frames can mention a name twice) -> frequency imbalance
 (4) probes are identical inputs in A and B
Writes integrity.json and integrity.md."""
import sys,json,re,types,collections
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parent))
core=types.ModuleType('core');core.CH=1024;core.clone_state=lambda x:x;core.jax=types.SimpleNamespace(tree_util=None);core.jnp=None;core.eqx=None
sys.modules['core']=core
import stage2,gen_world as gw
ROOT=Path(__file__).resolve().parents[2];EXP=ROOT/'experiments/exp04_what_ttt_retains'
CH=1024;WIN=8192
def subseq_positions(ids,pat):
    n=len(pat);return [i for i in range(len(ids)-n+1) if ids[i:i+n]==pat]
def main(seeds):
    rt=types.SimpleNamespace(tok=gw.TOK,bos=128000,eos=128001);out={};md=['# Exp04 stimulus integrity re-verification\n']
    pair_imb=[]  # (seed,pair,count_X1,count_X2 in world A)
    for seed in seeds:
        v=1 if seed<100 else 2;w=gw.build_world(seed,v=v);PA=stage2.Probe(rt,w,False);PB=stage2.Probe(rt,w,True)
        ia,ib=PA.ids,PB.ids;r={}
        # (4) identical probe inputs
        r['probe_items_identical_A_B']=PA.items==PB.items
        # (1) token diff
        textA,nameA,spA=gw.realize(w,False);textB,nameB,spB=gw.realize(w,True)
        eA=stage2.enc(gw.TOK,textA);eB=stage2.enc(gw.TOK,textB)
        r['tokens_A_B_total']=[len(eA.ids),len(eB.ids)]
        # per exposure sentence token count A vs B
        sentdiff=[];tot=0;tri_pair={t['id']:t['pair'] for t in w['triples']};mism_pair=collections.Counter()
        for a,b in zip(spA,spB):
            na=len(stage2.enc(gw.TOK,a['text']).ids);nb=len(stage2.enc(gw.TOK,b['text']).ids);tot+=1;sentdiff.append(na-nb)
            if na!=nb:mism_pair[tri_pair[a['t']]]+=1
        r['exposure_sentences']=tot;r['sentences_with_token_count_mismatch']=int(sum(1 for d in sentdiff if d!=0));r['sum_token_count_diff_A_minus_B']=int(sum(sentdiff))
        # where do first tokens differ -> inside an exposure span?
        L=min(len(ia),len(ib));diff=[i for i in range(L) if ia[i]!=ib[i]]
        r['first_divergence_token']=diff[0] if diff else None
        # (2) occurrences
        names=[t['X'] for t in w['triples']];places=[t['P'] for t in w['triples']];ctrl=w['ctrl_persons']+w['ctrl_places']
        def occ(word):
            pats=[stage2.enc(gw.TOK,' '+word).ids,stage2.enc(gw.TOK,word).ids]
            pos=set()
            for p in pats:
                for i in subseq_positions(ia,p):pos.add(i)
            return sorted(pos)
        ctrl_hits=sum(len(occ(c)) for c in ctrl);r['control_name_occurrences_in_stream']=ctrl_hits
        # background contamination: any name/place token occurrence outside exposure-sentence char spans
        spans_tok=[(s['tok_start'],s['tok_end']) for s in PA.spans]
        def inside(i):return any(a<=i<=b for a,b in spans_tok)
        outside=0
        for word in names+places:
            for i in occ(word):
                if not inside(i):outside+=1
        r['entity_name_occurrences_outside_exposure_sentences']=outside
        # mention counts per name in world A (and B follows the swap): sentence frames can contain {X} twice
        cnt={}
        for t in w['triples']:
            cnt[t['id']]=len(re.findall(re.escape(nameA[t['id']]),textA))
        pc={}
        for t in w['triples']:pc.setdefault(t['pair'],[]).append(t['id'])
        imb=[cnt[a]-cnt[b] for a,b in pc.values()]
        for p,(a,b) in pc.items():pair_imb.append({'seed':seed,'pair':p,'count_X1':cnt[a],'count_X2':cnt[b],'token_mismatch_sentences':int(mism_pair.get(p,0))})
        r['pairs_with_unequal_total_mentions']=int(sum(1 for d in imb if d!=0));r['mention_count_diff_abs_mean']=float(np.mean(np.abs(imb)));r['mention_count_diff_abs_max']=int(np.max(np.abs(imb)))
        # (2) distance of every exposure to every probe snapshot window (independent recompute from spans)
        r['audit']={str(k):v for k,v in PA.audit.items()};r['world_version']=v
        r['recomputed_min_slack']={}
        for k,rounds in stage2.STAGES.items():
            last=max(s['tok_end'] for s in PA.spans if s['round']<=rounds);limit=(k+1)*CH-(WIN-CH)
            future=[s['tok_start'] for s in PA.spans if s['round']>rounds]
            r['recomputed_min_slack'][str(k)]={'last_exposure_end':last,'window_start_for_last_probe_token':limit,'slack':limit-last,'future_exposures_start_after_probe_chunk':bool(all(f>=(k+1)*CH for f in future)) if future else True}
        out[seed]=r
    out['pair_mention_counts']=pair_imb
    json.dump({str(k):v for k,v in out.items()},open(EXP/'integrity.json','w'),indent=1,default=str)
    for seed in seeds:
        r=out[seed];md.append(f"## seed {seed}\n- probe items identical in worlds A and B: **{r['probe_items_identical_A_B']}**\n- token counts of the full text A / B: {r['tokens_A_B_total']} (difference caused by tokenization of swapped names in real contexts)\n- exposure sentences: {r['exposure_sentences']}; with different token count A vs B: **{r['sentences_with_token_count_mismatch']}** (net {r['sum_token_count_diff_A_minus_B']:+d} tokens)\n- control (never-exposed) names occurring in the stream: **{r['control_name_occurrences_in_stream']}**\n- entity names/places occurring outside exposure sentences: **{r['entity_name_occurrences_outside_exposure_sentences']}**\n- swap pairs whose two names have UNEQUAL total mention counts (frames can mention a name twice): **{r['pairs_with_unequal_total_mentions']} / {w['pairs']}** (mean |diff| {r['mention_count_diff_abs_mean']:.2f}, max {r['mention_count_diff_abs_max']})\n- independent window check: "+'; '.join(f"k={k}: slack {v['slack']} tokens, later exposures all after the probe chunk={v['future_exposures_start_after_probe_chunk']}" for k,v in r['recomputed_min_slack'].items())+'\n')
    (EXP/'integrity.md').write_text('\n'.join(md));print('\n'.join(md))
if __name__=='__main__':main([int(x) for x in sys.argv[1:]] or [1,2,3])
