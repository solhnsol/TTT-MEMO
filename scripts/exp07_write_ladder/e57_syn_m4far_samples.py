"""E57 synthetic world: what is the far 4-gram match behind a fact row? (CPU only; new file; the output contains synthetic-world text). Rows = fact rows (target token inside a fact sentence) of e57_syn_findability_rows.npz (seeds 62 / 63 / 64, chunks 8..127) with m4_far = 1: the 4-gram (previous 3 tokens incl. the query token + target) occurred earlier at distance q-(j-1) in [512, 8191] and not in the band. Sample: the 20 rows with the largest band-512 gain among the TOP 1 % (by band-512 gain) of those rows... [if the top 1 % holds fewer than 20 rows per seed mix, the 20 largest are taken]; per sample: the sentence of the target token with its previous / next sentence, the same for the NEAREST earlier occurrence (inside the window, outside the band), whether the two sentences carry the same fact id (repetition of the same fact sentence) or a different one / background (shared sentence frame), the matched 4-gram text, the number of far matches and how many of them are in sentences of the same fact. Aggregation over ALL fact-row m4_far rows: the share whose nearest far match is in a sentence of the same fact, the most frequent 4-grams (text, rows), and the same for the sample rows.
  .venv/bin/python scripts/exp07_write_ladder/e57_syn_m4far_samples.py --findability R.npz --out O.json"""
import sys,json,argparse,pathlib,collections
import numpy as np
HERE=pathlib.Path(__file__).resolve().parent;ROOT=HERE.parents[1]
for p in (HERE,HERE/'e57',ROOT/'scripts/exp04_what_ttt_retains',ROOT/'scripts/exp05_write_dynamics'):sys.path.insert(0,str(p))
CH=1024
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--findability');ap.add_argument('--out');ap.add_argument('--n',type=int,default=20);a=ap.parse_args()
    import e55_run as R55,e55_uv as UVm,e55_sentdump_labels as LB
    import e57_syn_pass as SP
    F=np.load(a.findability);seed=F['seed'];sel=(F['is_fact']>0)&(F['m4_far']>0);out={'definition':__doc__.split('\n')[0]};agg_all=collections.Counter();same_all=0;tot_all=0;samples=[];allsel=np.nonzero(sel)[0]
    # top 1 % by band-512 gain among the selected rows
    g=F['g512'][allsel];order=allsel[np.argsort(-g,kind='stable')];top=order[:max(a.n,int(round(.01*len(order))))]
    pick=[];seen=set()
    for r in top:
        pick.append(int(r))
        if len(pick)>=a.n:break
    per_seed={}
    for sd in (62,63,64):
        enc,w,tf=R55.build(SP.synth_args(sd),sd,arms=['NAT']);S=R55.streams(w,enc);ids,sha=S['A'];ends=UVm.sentence_ends(UVm.tokenizer(tf),ids);N=len(ids);tok=UVm.tokenizer(tf)
        spans=[];tf_=np.full(N,-1,np.int16)
        for k in range(N//CH):
            tf_[k*CH:(k+1)*CH]=LB.token_fact(w['chunks'],k,CH)
            for (s_,e_) in UVm.chunk_sents(ends,k,CH):spans.append((k*CH+s_,k*CH+e_))
        starts=np.array([s for s,e in spans]);dec=lambda i0,i1:tok.decode([int(t) for t in ids[i0:i1]])
        def sent_idx(p):return int(np.searchsorted(starts,p,side='right')-1)
        def fact_of(si):
            s,e=spans[si];f=tf_[s:e];f=f[f>=0]
            return (int(np.bincount(f).argmax()) if len(f)>=0.5*(e-s) else -1)
        def show(p):
            si=sent_idx(p);t=[dec(*spans[x]) for x in (si-1,si,si+1) if 0<=x<len(spans)];return dict(previous=t[0] if si>0 else '',sentence=dec(*spans[si]),next=t[2] if si+1<len(spans) and len(t)>2 else '',fact_id=fact_of(si))
        D=collections.defaultdict(list)
        for j in range(3,N):D[tuple(int(x) for x in ids[j-3:j+1])].append(j)
        rows=np.nonzero(sel&(seed==sd))[0];ssame=0;stot=0;cnt=collections.Counter()
        for r in rows:
            q=int(F['chunk'][r])*CH+int(F['row'][r]);key=tuple(int(x) for x in ids[q-2:q+2]);js=np.asarray(D[key]);far=js[(js<=q-511)&(js>=q-8190)]
            if len(far)==0:continue
            near=int(far.max());fsame=[fact_of(sent_idx(int(j)))==fact_of(sent_idx(q+1)) for j in far];same=bool(fsame[int(np.argmax(far))]);ssame+=same;stot+=1;cnt[dec(q-2,q+2)]+=1
            if int(r) in pick:
                samples.append(dict(seed=sd,chunk=int(F['chunk'][r]),row=int(F['row'][r]),g512=float(F['g512'][r]),g_win=float(F['g_win'][r]),matched_4gram=dec(q-2,q+2),target_token=dec(q+1,q+2),n_far_matches=int(len(far)),far_matches_in_same_fact=int(sum(fsame)),nearest_far_distance=int(q-(near-1)),target_context=show(q+1),earlier_context=show(near),same_fact_sentence=same,nearest_match_same_fact=same))
        per_seed[sd]=dict(rows=int(stot),nearest_match_same_fact_share=float(ssame/max(stot,1)),top_4grams=[(k,v) for k,v in cnt.most_common(15)]);agg_all.update(cnt);same_all+=ssame;tot_all+=stot
    out['rows_fact_m4far']=int(tot_all);out['nearest_match_same_fact_share_all']=float(same_all/max(tot_all,1));out['per_seed']=per_seed;out['top_4grams_all']=agg_all.most_common(25);out['samples']=samples
    for s in samples:s['text']=f"seed {s['seed']} chunk {s['chunk']} row {s['row']} gain {s['g512']:+.2f}: 4-gram {s['matched_4gram']!r} -> target {s['target_token']!r}; far matches {s['n_far_matches']} (same fact {s['far_matches_in_same_fact']}), nearest distance {s['nearest_far_distance']}, {'same fact id (a different sentence of the same fact)' if s['same_fact_sentence'] else 'different fact or background (shared sentence frame)'}\n  TARGET: {s['target_context']['previous']!r} || {s['target_context']['sentence']!r} || {s['target_context']['next']!r}\n  EARLIER: {s['earlier_context']['previous']!r} || {s['earlier_context']['sentence']!r} || {s['earlier_context']['next']!r}"
    json.dump(out,open(a.out,'w'),indent=1,ensure_ascii=False)
    print('rows',tot_all,'nearest same fact share %.3f'%(same_all/max(tot_all,1)),'samples',len(samples));print(out['top_4grams_all'][:10]);print(samples[0]['text'][:900])
if __name__=='__main__':main()
