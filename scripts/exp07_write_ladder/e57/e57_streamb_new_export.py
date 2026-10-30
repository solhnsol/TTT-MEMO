"""E57 stream B (new events): sentence file in the layout of e57_stream_sentences.jsonl and per-event excerpts (CPU only, no model, no GPU; private internal research only). Stream = all independent lead articles of the 20 events in global date order (date, event rank, rid), each lead text + blank line, llama3 tokens, cut to whole 1024-token chunks (same rule as e57_reads.article_intervals); fragments = ORC2 v2 sentence ends, chunk-local (same as e57_export_stream). Question-dependent fields (packet_a, packet_s, strict_gold, orc2_hidden) are null / false until questions exist.
Excerpts: <out>/news-lite/streamB_<rank>.json, per event up to --per-event articles chosen by a seeded shuffle (seed 57241 + rank), whole lead sentences up to --excerpt-chars characters.
  python3 e57_streamb_new_export.py --store S.pkl --out DIR"""
import sys,json,pickle,pathlib,argparse,random,hashlib,bisect
import numpy as np
HERE=pathlib.Path(__file__).resolve().parent
for p in (HERE,HERE.parent,HERE.parents[2]/'scripts/exp04_what_ttt_retains',HERE.parents[2]/'scripts/exp05_write_dynamics'):sys.path.insert(0,str(p))
CH=1024
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--store');ap.add_argument('--out');ap.add_argument('--per-event',type=int,default=40);ap.add_argument('--excerpt-chars',type=int,default=700);a=ap.parse_args()
    ROOT=HERE.parents[2];import gen_world_ctl as G,e55_uv as UVm,e57_orc2 as O2
    from tokenizers import Tokenizer
    tf=str(ROOT/'data/tokenizer/llama3/tokenizer.json');enc=G.load_enc(tf);tk=Tokenizer.from_file(tf)
    S=pickle.load(open(a.store,'rb'));arts=[]
    for k,E in S['events'].items():
        for i,ar in enumerate(E['arts']):arts.append((ar['date'],k,ar['rid'],i,' '.join(ar['sents'][:ar['kept']])))
    arts.sort(key=lambda x:(x[0],x[1],x[2]));ids=[];iv=[];t=0
    for d,k,rid,i,txt in arts:
        e=list(enc(txt+'\n\n'));iv.append((t,t+len(e),k,rid,i));ids+=e;t+=len(e)
    n=len(ids)//CH;ids=np.asarray(ids[:n*CH],np.int64);out=pathlib.Path(a.out);out.mkdir(parents=True,exist_ok=True)
    ends=O2.sentence_ends_v2(UVm.tokenizer(tf),ids);fr=[]
    for c in range(n):
        for s_,e_ in UVm.chunk_sents(ends,c,CH):fr.append((c,c*CH+s_,e_-s_))
    starts=[x[0] for x in iv];cic={}
    meta=dict(type='meta',note='private, internal research only; no publication or redistribution',stream='B_new_events',seed=S['seed'],n_chunks=n,chunk_tokens=CH,n_fragments=len(fr),n_articles_in_stream=len(iv),stream_tokens=n*CH,stream_tokens_before_cut=t,tokenizer='llama3',layout='ORC2 v2 sentence ends',ids_sha=hashlib.sha256(ids.tobytes()).hexdigest()[:16])
    with open(out/'streamB_stream_sentences.jsonl','w') as f:
        f.write(json.dumps(meta,ensure_ascii=False)+'\n')
        for j,(c,s,l) in enumerate(fr):
            art=max(0,bisect.bisect_right(starts,s)-1);t0,t1,k,rid,ai=iv[art];ci=cic.get(c,0);cic[c]=ci+1;ar=S['events'][k]['arts'][ai]
            f.write(json.dumps(dict(type='frag',fid=j,chunk=c,ci=ci,tok0=s,tok_in_chunk=s-c*CH,ntok=l,event=f"B{k}",art=art,rid=rid,packet_a=None,art_sent=None,packet_s=None,art_sents_overlap=None,packet_s_overlap=None,orc2_hidden=False,strict_gold=False,date=ar['date'],host=ar['host'],text=tk.decode([int(x) for x in ids[s:s+l]])),ensure_ascii=False)+'\n')
    nl=out/'news-lite';nl.mkdir(exist_ok=True);cnt={}
    for k,E in S['events'].items():
        sel=list(range(len(E['arts'])));random.Random(57241+k).shuffle(sel);sel=sorted(sel[:a.per_event],key=lambda i:(E['arts'][i]['date'],E['arts'][i]['rid']));A=[]
        for i in sel:
            ar=E['arts'][i];o=[];m=0
            for si in range(max(1,ar['kept'])):
                if si>=len(ar['sents']):break
                if o and m+len(ar['sents'][si])>a.excerpt_chars:break
                o.append(ar['sents'][si]);m+=len(ar['sents'][si])+1
            A.append(dict(event=f"B{k}",rid=ar['rid'],host=ar['host'],date=str(ar['date']),title=ar['title'],excerpt=' '.join(o)))
        (nl/f'streamB_{k:03d}.json').write_text(json.dumps(dict(note='private, internal research only; no publication or redistribution',event=f"B{k}",title1=E['title1'],category=E['category'],independent_articles=len(E['arts']),articles=A),indent=1,ensure_ascii=False));cnt[k]=len(A)
    per={}
    for t0,t1,k,rid,ai in iv:per[k]=per.get(k,0)+1
    print(json.dumps(dict(chunks=n,tokens=n*CH,tokens_before_cut=t,fragments=len(fr),articles=len(iv),events=len(per),articles_in_stream_per_event=per,excerpts=cnt,ids_sha=meta['ids_sha']),ensure_ascii=False))
if __name__=='__main__':main()
