"""E57 stream sentence export (CPU only; private use). One JSONL: line 1 = meta, then one line per fragment of the 1-cycle news stream in the ORC2 sentence layout (abbreviation-aware splitting, e57_orc2.sentence_ends_v2; fragments are chunk-local, a sentence crossing a chunk border is two fragments).
Fields: fid (global fragment index = e57_orc2.build 'fr' order, also the index into bd['hidden']), chunk, ci (index inside the chunk = index i of ch7_<chunk>/ch8_<chunk> in e57_orca2_masks.npz), tok0 (global token offset), tok_in_chunk, ntok, event, art (stream article index), rid, packet_a (a## if the article is in the event's 12-article packet, else null), art_sent (index into the stored lead sentences of the article when the fragment starts exactly at one, else null), packet_s (s### of the packet for that sentence), art_sents_overlap / packet_s_overlap (all stored sentences / packet s### the fragment overlaps by characters; use these when art_sent is null: the two splittings differ), orc2_hidden, strict_gold, text.
Window rule: h7_<k>/h8_<k> index i is the window sentence at token base+wst[i], base=(k-8)*1024, i.e. the fragment with tok0 = base+wst[i]; tok0 is the join key.
  python3 e57_export_stream.py --bundle B --out O.jsonl"""
import sys,json,argparse,pathlib,bisect
import numpy as np
HERE=pathlib.Path(__file__).resolve().parent
for p in (HERE,HERE/'e57',HERE.parents[1]/'scripts/exp04_what_ttt_retains',HERE.parents[1]/'scripts/exp05_write_dynamics'):sys.path.insert(0,str(p))
CH=1024
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--bundle');ap.add_argument('--out');ap.add_argument('--tokenizer');a=ap.parse_args()
    ROOT=HERE.parents[1];import gen_world_ctl as G,e55_uv as UVm,e57_reads as RD,e57_bank as B,e57_orc2 as O2
    from tokenizers import Tokenizer
    tf=a.tokenizer or str(ROOT/'data/tokenizer/llama3/tokenizer.json');enc=G.load_enc(tf);tk=Tokenizer.from_file(tf)
    bank=pathlib.Path(a.bundle)/'e57_bank'/'bank_v1.json';items=RD.load_items(bank);ids,iv,hits=RD.article_intervals(a.bundle,enc,items);n=len(ids)//CH
    bd=O2.build('v2',ids,tk,tf,items,iv,UVm,B)
    st,_=B.load(a.bundle);rid2=None
    pk={}
    for f in (pathlib.Path(a.bundle)/'e57_bank'/'packets').glob('*.packet.json'):
        p=json.load(open(f));pk[p['event']]={x['rid']:(x['a'],[s['s'] for s in x['sentences']],[s['text'] for s in x['sentences']]) for x in p['articles']}
    starts=[x[0] for x in iv];lead={}
    for ev,E in st.items():
        for ai,rid in enumerate(E['rids']):lead[(ev,rid)]=E['sents'][ai][:max(1,E['kept'][ai])]
    out=open(a.out,'w');ev_cnt={};pk_in={}
    meta=dict(type='meta',note='private, internal research only; no publication or redistribution',n_chunks=n,chunk_tokens=CH,n_fragments=len(bd['fr']),n_articles_in_stream=len(iv),stream_tokens=n*CH,tokenizer='llama3',layout='ORC2 v2 sentence ends',ids_sha=__import__('hashlib').sha256(ids.tobytes()).hexdigest()[:16])
    out.write(json.dumps(meta,ensure_ascii=False)+'\n');cic={}
    for j,(c,s,l) in enumerate(bd['fr']):
        art=bd['art'][j];t0,t1,ev,rid=iv[art];ci=cic.get(c,0);cic[c]=ci+1
        sents=lead.get((ev,rid),[]);full=' '.join(sents)+'\n\n'
        # char offset of this fragment inside the article text, via decoded token prefix lengths
        pre=tk.decode([int(x) for x in ids[t0:s]]) if s>t0 else '';frag=bd['txt'][j].strip();off=len(pre)
        asi=None;cum=0;spans=[]
        for k,x in enumerate(sents):spans.append((cum,cum+len(x)));cum+=len(x)+1
        for k,(u,v) in enumerate(spans):
            if abs(u-off)<=1 or (u<=off<v and frag and full[u:v].strip().startswith(frag[:20])):asi=k;break
        pa=ps=None
        if ev in pk and rid in pk[ev]:
            pa=pk[ev][rid][0]
            if asi is not None and asi<len(pk[ev][rid][1]) and pk[ev][rid][2][asi].strip()[:30]==sents[asi].strip()[:30]:ps=pk[ev][rid][1][asi]
        fe=off+len(bd['txt'][j].rstrip());fs0=off+(len(bd['txt'][j])-len(bd['txt'][j].lstrip()))
        ov=[k for k,(u,v) in enumerate(spans) if u<fe and v>fs0]
        pso=[pk[ev][rid][1][k] for k in ov if k<len(pk[ev][rid][1])] if ev in pk and rid in pk[ev] else []
        out.write(json.dumps(dict(type='frag',fid=j,chunk=c,ci=ci,tok0=s,tok_in_chunk=s-c*CH,ntok=l,event=ev,art=art,rid=rid,packet_a=pa,art_sent=asi,packet_s=ps,art_sents_overlap=ov,packet_s_overlap=pso,orc2_hidden=bool(bd['hidden'][j]),strict_gold=bool(bd['strict'][j]),text=bd['txt'][j]),ensure_ascii=False)+'\n')
    out.close()
    print('fragments',len(bd['fr']),'chunks',n,'articles',len(iv),'events',len({x[2] for x in iv}))
if __name__=='__main__':main()
