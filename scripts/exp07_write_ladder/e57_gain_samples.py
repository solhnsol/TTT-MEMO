"""E57 samples of high-window-gain query rows and what they attend to (CPU only; new file; the output contains article text: keep it in the private project folder only). Rows = the eligible rows of the chunks 8..279 (e57_gain_labels_rows.npz: stream position of the TARGET token, label, question, gains), ranked by the W0 band-512 gain (band NLL - window NLL). Seeded stratified sample: 10 rows of the top 0.1 %, 10 of 0.1-1 %, 10 of 1-10 %, 10 A-labelled rows of the top 3 % (different questions when possible). Per sample: chunk, row, stream position, article number and event; the ~40 tokens up to and including the query token (input token) and the target (next) token; window NLL, band-512 NLL, gain, label; the attention list from the key-mass dump of job 295 (ALL heads, band 512: top-5 far keys by mass: window key index, absolute stream position, distance, mass and its share of the row's far mass, +-10 tokens of text with the key token in [[ ]], article / event of the key and whether it is the same article / the same event as the query).
  .venv/bin/python scripts/exp07_write_ladder/e57_gain_samples.py --bundle B --rows R.npz --band-npz e57_band_W0.npz --dump D --out O.json [--seed 1]"""
import sys,json,pathlib,argparse
import numpy as np
HERE=pathlib.Path(__file__).resolve().parent
for p in (HERE,HERE/'e57',HERE.parents[1]/'scripts/exp04_what_ttt_retains',HERE.parents[1]/'scripts/exp05_write_dynamics'):sys.path.insert(0,str(p))
CH=1024;NAMES=['OTHER','SENT','S','A']
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--bundle');ap.add_argument('--rows');ap.add_argument('--band-npz');ap.add_argument('--dump');ap.add_argument('--out');ap.add_argument('--seed',type=int,default=1);ap.add_argument('--tokenizer');a=ap.parse_args()
    import gen_world_ctl as G,e55_uv as UVm,e57_reads as RD,e57_attdump3_run as R3
    tfile=a.tokenizer or str(HERE.parents[1]/'data/tokenizer/llama3/tokenizer.json');enc=G.load_enc(tfile);tk=UVm.tokenizer(tfile)
    items=RD.load_items(pathlib.Path(a.bundle)/'e57_bank'/'bank_v1.json');ids,iv,hits=RD.article_intervals(a.bundle,enc,items);ids=np.asarray(ids);N=len(ids)
    starts=np.array([x[0] for x in iv]);evs=[x[2] for x in iv]
    def txt(i0,i1):return tk.decode([int(t) for t in ids[max(0,i0):min(N,i1)]])
    def art(p):
        j=int(np.searchsorted(starts,p,side='right')-1);return j,evs[j]
    Rz=np.load(a.rows);pos=Rz['pos'];lab=Rz['label'];qq=Rz['question'];g=Rz['gain512'];B=np.load(a.band_npz);bi=list(B['bands']).index(512)
    n=len(pos);order=np.argsort(-g,kind='stable');rank=np.empty(n,int);rank[order]=np.arange(n);rng=np.random.default_rng(a.seed)
    def pick(lo,hi,k,mask=None,distinct=False):
        cand=order[int(round(lo*n)):int(round(hi*n))]
        if mask is not None:cand=cand[mask[cand]]
        if distinct:
            by={}
            for c in cand:by.setdefault(int(qq[c]),[]).append(c)
            qs=list(by);rng.shuffle(qs);ch=[by[q][int(rng.integers(len(by[q])))] for q in qs[:k]]
            if len(ch)<k:rest=[c for c in cand if c not in set(ch)];ch+=list(rng.choice(rest,k-len(ch),replace=False))
            return ch
        return list(rng.choice(cand,k,replace=False))
    strata=[('top0.1%',pick(0,.001,10)),('0.1-1%',pick(.001,.01,10)),('1-10%',pick(.01,.10,10)),('A_in_top3%',pick(0,.03,10,mask=(lab==3),distinct=True))]
    cache={}
    def shard(k):
        s=k//R3.SHARD
        if s not in cache:cache.clear();cache[s]=np.load(R3.shard_file(a.dump,s))
        return cache[s]
    out=[]
    for sname,idxs in strata:
        for r in idxs:
            p=int(pos[r]);q=p-1;k=q//CH;i=q%CH;aj,ev=art(q);win=float(B['nll_win'][k][i]);bnd=float(B['nll_band'][bi][k][i])
            e=dict(stratum=sname,chunk=int(k),row=int(i),stream_position_of_query_token=int(q),target_position=p,article=aj,event=ev,label=NAMES[int(lab[r])],question=(items[int(qq[r])]['id'] if qq[r]>=0 else None),gain_rank_of_278256=int(rank[r])+1,window_nll=win,band512_nll=bnd,gain=bnd-win,
                query_context=txt(q-39,q+1),target_token=tk.decode([int(ids[p])]) if p<N else '')
            Z=shard(k);P=f'W0/c{k}/ALL512_';kidx=Z[P+'idx'][i].astype(int);kmass=Z[P+'mass'][i].astype(float);kfar=float(Z[P+'far'][i]);att=[]
            for j in range(5):
                w=int(kidx[j]);ap_=k*CH-8192+w;kj,kev=art(ap_)
                att.append(dict(rank=j+1,window_key_index=w,absolute_position=ap_,distance=int(q-ap_),mass=float(kmass[j]),mass_share_of_far=float(kmass[j]/kfar),article=kj,event=kev,same_article=bool(kj==aj),same_event=bool(kev==ev),key_token=tk.decode([int(ids[ap_])]),text=txt(ap_-10,ap_)+'[['+tk.decode([int(ids[ap_])])+']]'+txt(ap_+1,ap_+11)))
            e['attention_top5_all_heads_band512']=att
            e['text']=f"[{sname}] chunk {k} row {i} article #{aj} ({ev}) label {e['label']}{(' / '+e['question']) if e['question'] else ''}: window NLL {win:.2f}, band512 NLL {bnd:.2f}, gain {bnd-win:+.2f}\n  query context: ...{e['query_context']!r} -> target {e['target_token']!r}\n"+'\n'.join(f"  attends #{x['rank']} dist {x['distance']} ({'same article' if x['same_article'] else ('same event' if x['same_event'] else 'other event')}, share {x['mass_share_of_far']:.3f}): ...{x['text']!r}" for x in att)
            out.append(e)
    json.dump(dict(note='private: contains article text; sample seed %d; strata by W0 band-512 gain rank over 278,256 eligible rows; attention = key-mass dump job 295, ALL heads, band 512 (far keys only: distance >= 512)'%a.seed,samples=out),open(a.out,'w'),indent=1,ensure_ascii=False)
    print(len(out),'samples');print('\n'.join(x['text'] for x in out[:3]))
if __name__=='__main__':main()
