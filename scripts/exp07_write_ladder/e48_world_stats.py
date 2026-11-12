"""E48 CPU table (server: needs tokenizers, data/, gen_world, suppress like e47_counts.py; no GPU): how the frame-disjoint world variants differ from the current world (frames='shared'), world seed 101 (and others), passes 1-4, world A (the sizes are equal in world B: the names have equal token length).
Per world mode (shared = current, pair = W1, sentence = W2) and pass:
  sentences / tokens per exposure sentence (mean, sd, min, max), the stream position range of the exposure sentences,
  O chunks (chunks with a non-empty oracle set), fact sentences per O chunk (mean), masked tokens summed over the O chunks: oracle O, K3 (sentencecover), K3-all (k3allmask), SENT (sentmask),
  frame sharing: share of the window sentences of OTHER pairs (place token in the write window before an O chunk) that use the same frame as one of the chunk's fact sentences; probability that two random exposure sentences of different pairs / of the same pair / of the same triple use the same frame.
Usage: e48_world_stats.py --seeds 101 --passes 4 [--out-dir D]   |   --selftest (synthetic table).   Output <out-dir>/e48_world_stats.{md,json}."""
import os,sys,json,argparse
from pathlib import Path
import numpy as np
HERE=Path(__file__).resolve().parent;sys.path.insert(0,str(HERE))
import s3_sets as SETS
CH=SETS.CH;W=SETS.W;NCH=63
def stats_of(table,seed,ps):
    sets=SETS.Sets(table,seed,ps,0.35);L=[len(u['toks']) for u in table];pos=[u['place'][0] for u in table]
    R={'sentences':len(table),'tok_mean':float(np.mean(L)),'tok_sd':float(np.std(L,ddof=1)),'tok_min':int(min(L)),'tok_max':int(max(L)),'pos_min':int(min(pos)),'pos_max':int(max(pos))}
    nO=0;nf=[];sO=sK=sA=sS=0;xs=xsame=0;byt={u['idx']:u for u in table}
    for k in range(NCH):
        O=sets.oracle(k)
        if not O:continue
        nO+=1;cur=sets.cur_sentences(k);nf.append(len(cur));sO+=len(O);sK+=len(sets.sentencecover(k));sA+=len(sets.k3allmask(k));sS+=len(sets.sentmask(k))
        fr={s['frame'] for s in cur};prs={s['pair'] for s in cur};lo=max(0,k*CH-W)
        for u in table:
            if u['pair'] in prs or not (lo<=u['place'][0]<k*CH):continue
            xs+=1;xsame+=u['frame'] in fr
    R.update({'O_chunks':nO,'fact_sent_per_O_chunk':float(np.mean(nf)) if nf else 0.0,'sum_O':sO,'sum_K3':sK,'sum_K3all':sA,'sum_SENT':sS,'X_window_sentences':xs,'X_same_frame_share':xsame/xs if xs else float('nan')})
    fr=[(u['pair'],u['t'],u['frame']) for u in table];n=len(fr);cnt={'diff_pair':[0,0],'same_pair_other_triple':[0,0],'same_triple':[0,0]}
    for i in range(n):
        for j in range(i+1,n):
            a,b=fr[i],fr[j];key='same_triple' if a[1]==b[1] else ('same_pair_other_triple' if a[0]==b[0] else 'diff_pair');cnt[key][0]+=a[2]==b[2];cnt[key][1]+=1
    R['p_same_frame']={k:(v[0]/v[1] if v[1] else float('nan')) for k,v in cnt.items()};R['frames_unidentified']=sum(1 for u in table if u['frame']<0);return R
def md_of(res):
    keys=[('sentences','sentences'),('tok_mean','tokens/sentence mean'),('tok_sd','sd'),('tok_min','min'),('tok_max','max'),('pos_max','last exposure place token position'),('O_chunks','O chunks'),('fact_sent_per_O_chunk','fact sentences per O chunk'),('sum_O','sum O (masked tokens, all O chunks)'),('sum_K3','sum K3'),('sum_K3all','sum K3-all'),('sum_SENT','sum SENT'),('X_same_frame_share','share of window X sentences with the same frame as a fact sentence of the chunk'),('frames_unidentified','sentences with frame not identified (-1)')]
    L=['# E48 frame-disjoint worlds: size and structure table (CPU)\n','shared = the current world (30 shared frames), pair = W1 (every pair owns R = 16 frames, no frame shared between pairs), sentence = W2 (every sentence its own frame). World A; ratios are mode / shared.\n']
    for run,modes in res.items():
        L+=[f'\n## {run}\n','| quantity | shared | pair (W1) | pair / shared | sentence (W2) | sentence / shared |','|---|---|---|---|---|---|']
        for k,lab in keys:
            v=[modes[m][k] for m in ('shared','pair','sentence')];f=lambda x:f'{x:.3f}' if isinstance(x,float) else str(x);r=lambda x:f'{x/v[0]:.3f}' if v[0] else ''
            L.append(f'| {lab} | {f(v[0])} | {f(v[1])} | {r(v[1])} | {f(v[2])} | {r(v[2])} |')
        for k in ('diff_pair','same_pair_other_triple','same_triple'):
            v=[modes[m]['p_same_frame'][k] for m in ('shared','pair','sentence')];L.append(f'| P(two sentences use the same frame): {k} | {v[0]:.4f} | {v[1]:.4f} | | {v[2]:.4f} | |')
    return '\n'.join(L)+'\n'
def run(a):
    import e42_k2_tokens as K2
    sys.path.insert(0,str(K2.ROOT/'scripts/exp04_what_ttt_retains'));import gen_world as gw;import suppress as SP
    res={}
    for seed in [int(x) for x in a.seeds.split(',')]:
        for ps in range(1,a.passes+1):
            out={}
            for mode in ('shared','pair','sentence'):
                os.environ['E48_FRAMES']=mode;w=K2.build_world_pass(gw,seed,ps);out[mode]=stats_of(SETS.sentence_table(w,False,gw.TOK,gw,SP),seed,ps)
            res[f's{seed}_pass{ps}']=out
    os.environ.pop('E48_FRAMES',None);return res
def selftest():
    import e43_sets_test as T,random
    tb=T.table()
    def with_frames(tb,mode):
        out=[]
        for u in tb:
            u=dict(u);i=int(u['t'][1:]);p=i//2
            u['frame']={'shared':(i*7+u['round']*3)%30,'pair':p*16+(u['round']-1),'sentence':i*16+u['round']-1}[mode];out.append(u)
        return out
    res={'synthetic':{m:stats_of(with_frames(tb,m),101,2) for m in ('shared','pair','sentence')}}
    assert res['synthetic']['pair']['p_same_frame']['diff_pair']==0.0 and res['synthetic']['sentence']['p_same_frame']['same_triple']==0.0 and res['synthetic']['shared']['p_same_frame']['diff_pair']>0
    assert res['synthetic']['pair']['X_same_frame_share']==0.0 and res['synthetic']['shared']['O_chunks']==res['synthetic']['pair']['O_chunks']>0
    print('e48_world_stats SELFTEST OK (synthetic table)');print(md_of(res)[:900])
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--seeds',default='101');ap.add_argument('--passes',type=int,default=4);ap.add_argument('--out-dir',default='/tmp/e48_world_stats');ap.add_argument('--selftest',action='store_true');a=ap.parse_args()
    if a.selftest:selftest();sys.exit(0)
    res=run(a);od=Path(a.out_dir);od.mkdir(parents=True,exist_ok=True);md=md_of(res);(od/'e48_world_stats.md').write_text(md);json.dump(res,open(od/'e48_world_stats.json','w'),indent=1);print(md)
