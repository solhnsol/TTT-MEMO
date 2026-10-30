"""E57 stream B (new events) extraction (CPU only, no model, no GPU, no download; private internal research material). Input: the 20 clusters drawn in data/news-lite/newevents_streamB_draw.json (seed 20261030). Per event: members of the job-215 cluster (clusters_big.pkl, rank -> ci via e57_biglist.json), restricted to core members (title holds >= 2 of the 3 most frequent title tokens), article texts from the parquet shards, near-duplicate removal (exact 5-gram word shingles, Jaccard >= 0.5, keep the longest; e57_common.minhash_dup), one article per hostname (the longest), lead = first sentences up to 350 tokens (e57_news.variant rule), events with < 8 independent domains are flagged.
Writes <out>/e57_streamB/streamB_store.pkl (SERVER ONLY) and streamB_counts.json (counts only).
  python3 e57_streamb_new.py --out experiments/exp07_write_ladder/e57_news --draw data/news-lite/newevents_streamB_draw.json"""
import sys,json,pickle,pathlib,argparse,collections,time
HERE=pathlib.Path(__file__).resolve().parent;sys.path.insert(0,str(HERE))
import e57_common as C
import e57_news as N
LEAD=350;MIN_DOM=8
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out');ap.add_argument('--draw');a=ap.parse_args();out=pathlib.Path(a.out);t0=time.time()
    draw=json.load(open(a.draw))['draw'];rows={r['rank']:r for r in json.load(open(out/'e57_bigclusters/e57_biglist.json'))['rows']}
    D=pickle.load(open(out/'titles.pkl','rb'));CL=pickle.load(open(out/'e57_bigclusters/clusters_big.pkl','rb'))['clusters'];fd=out/'e57_streamB';fd.mkdir(exist_ok=True)
    ev={};need=set()
    for x in draw:
        r=rows[x['rank']];mem=CL[r['ci']]['members'];top=[w for w,_ in r['top_tokens']]
        core=[i for i in mem if len(set(N.ttoks(D['title'][i]))&set(top[:3]))>=2]
        ev[x['rank']]=dict(ci=r['ci'],all=mem,core=core,top=top);need|=set(core)
    print('members',sum(len(v['all']) for v in ev.values()),'core',len(need),f'{time.time()-t0:.0f}s',flush=True)
    got=N.read_texts(D,sorted(need));print('texts read',len(got),f'{time.time()-t0:.0f}s',flush=True)
    store={};counts={};e=N.enc()
    for x in draw:
        k=x['rank'];E=ev[k];mem=[i for i in E['core'] if D['rid'][i] in got];texts=[got[D['rid'][i]] for i in mem]
        keep=C.minhash_dup(texts,0.5);byh={}
        for j in keep:
            h=D['host'][mem[j]]
            if h not in byh or len(texts[j])>len(texts[byh[h]]):byh[h]=j
        sel=sorted(byh.values());arts=[]
        for j in sel:
            i=mem[j];ss=N.split_article(texts[j])[:200]
            if not ss:continue
            tl=[len(z.ids) for z in e.encode_batch(ss,add_special_tokens=False)];cum=[0]
            for t in tl:cum.append(cum[-1]+t)
            kp=0
            while kp<len(ss) and (cum[kp+1]<=LEAD or kp==0):kp+=1
            arts.append(dict(rid=D['rid'][i],host=D['host'][i],date=int(D['date'][i]),title=D['title'][i],sents=ss,cum=cum,kept=kp))
        store[k]=dict(rank=k,title1=x['title1'],category=x['category'],arts=arts)
        counts[k]=dict(title1=x['title1'],category=x['category'],cluster_members=len(E['all']),core_members=len(E['core']),texts_found=len(mem),after_5gram_dedup=len(keep),independent_domains=len(arts),lead_tokens=sum(a['cum'][a['kept']]+2 for a in arts),meets_min8=len(arts)>=MIN_DOM)
        print(k,counts[k],flush=True)
    pickle.dump(dict(seed=json.load(open(a.draw))['seed'],lead_tok=LEAD,events=store),open(fd/'streamB_store.pkl','wb'));json.dump(counts,open(fd/'streamB_counts.json','w'),indent=1,ensure_ascii=False)
    print('done',f'{time.time()-t0:.0f}s')
if __name__=='__main__':main()
