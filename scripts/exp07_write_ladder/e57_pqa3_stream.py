"""E57 public QA stage 3 (E57 122-R / 123), accumulating re-description stream (new file, CPU): ONE stream = filler / original 332 paragraphs / re-description 1 / 2 / 3 / 4 (six sections, written once; each section is read at its end, six read points).
Inputs: paragraphs.json (332 SQuAD dev paragraphs with their stage-1 item ids; sha256 18aaec5b...), rewrites_all.json (pid -> 4 re-descriptions; sha256 5ae0c686...). Filler = SQuAD TRAIN-v1.1 paragraphs of documents whose title (normalised, sha256) is not a dev title (overlap 0, asserted), documents in a seeded shuffle, paragraphs of a document in the original order, exactly 332 paragraphs. The five content sections use ONE fixed permutation of the 332 paragraphs (np.random.default_rng(SEED).permutation), the same order in every section; the filler keeps document order (coherent unrelated text). Layout as stream S (e57_pqa2_stream.py): paragraph = "\n\n" + first sentence, later sentences " " + sentence, every sentence encoded alone, ids concatenated.
The stream is a whole number of chunks of 1024 tokens: after the last re-description the stream is completed by further unrelated TRAIN filler paragraphs (the continuation of the filler documents) up to the chunk boundary. A read point is the END OF THE CHUNK that holds the last token of a section; the first tokens of the next section (< 1024) already sit in that chunk (reported as spill tokens).
Outputs in <out>: streamP3_stream_sentences.jsonl (meta + one frag line per sentence with section / pid / par), streamP3_ids.npy, streamP3_meta.json (section token ends, read chunks, spill tokens, ids_sha).
  python3 e57_pqa3_stream.py --para P.json --rewrites R.json --train train-v1.1.json --dev dev-v1.1.json --out DIR  |  --check --out DIR  |  --selftest"""
import sys,json,re,hashlib,pathlib
import numpy as np
HERE=pathlib.Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
import e57_pqa2_stream as S
CH=1024;SEED=20261101;SECTIONS=['filler','orig','rw1','rw2','rw3','rw4']
PARA_SHA='18aaec5b1fbf8135f61791d2d4fc93725525aa6b31c114fd01c2ccd720a3b281';REW_SHA='5ae0c6861cd77ef171c131fc77d668dfa301abe6ffff0ab6a54f97f1d5fb9220'
def tsha(t):return hashlib.sha256(' '.join(t.lower().replace('_',' ').split()).encode()).hexdigest()
def filler_paragraphs(train,dev,n,seed=SEED,extra=64):
    """-> (paragraphs [(title, text)] of the main filler (n) and of the continuation (extra), dev-title overlap count)"""
    dt={tsha(a['title']) for a in dev['data']};docs=[a for a in train['data'] if tsha(a['title']) not in dt]
    ov=sum(tsha(a['title']) in dt for a in docs);assert ov==0
    order=np.random.default_rng(seed).permutation(len(docs));ps=[]
    for i in order:
        for p in docs[i]['paragraphs']:ps.append((docs[i]['title'],p['context']))
        if len(ps)>=n+extra:break
    return ps[:n],ps[n:n+extra],ov
def enc_par(tk,text):
    """-> [(sentence text with prefix, ids)]"""
    out=[]
    for si,s in enumerate(S.sentences(text)):
        t=('\n\n' if si==0 else ' ')+s;out.append((t,tk.encode(t,add_special_tokens=False).ids))
    return out
def build(para,rew,train,dev,seed=SEED):
    tk=S.tokenizer();n=len(para);perm=np.random.default_rng(seed).permutation(n)
    fm,fx,ov=filler_paragraphs(train,dev,n,seed)
    secs={'filler':[(t,'filler',p) for t,p in fm],'orig':[(para[i]['pid'],'orig',para[i]['context']) for i in perm]}
    for k in range(4):secs[f'rw{k+1}']=[(para[i]['pid'],f'rw{k+1}',rew[para[i]['pid']][k]) for i in perm]
    frags=[];ids=[];ends={};t=0
    def add(sec,key,tag,text):
        nonlocal t
        for si,(s,x) in enumerate(enc_par(tk,text)):frags.append(dict(section=sec,key=key,tag=tag,sent=si,text=s,ntok=len(x),tok0=t));ids.extend(x);t+=len(x)
    for sec in SECTIONS:
        for key,tag,text in secs[sec]:add(sec,key,tag,text)
        ends[sec]=t
    target=-(-ends['rw4']//CH)*CH
    for ti,tx in fx:
        if t>=target:break
        add('tail',ti,'tail',tx)
    assert t>=target,'continuation filler too short'
    keep=target;a=np.asarray(ids[:keep],np.int64);cut=t-keep
    last={s:(ends[s]-1)//CH for s in SECTIONS};spill={s:int((last[s]+1)*CH-ends[s]) for s in SECTIONS}
    return frags,a,dict(seed=seed,n_paragraphs=n,section_end_tokens=ends,read_chunks=last,spill_tokens=spill,n_chunks=keep//CH,cut_tokens=cut,filler_dev_title_overlap=ov,n_tail_filler_paragraphs=sum(f['sent']==0 and f['section']=='tail' for f in frags),
        section_tokens={s:ends[s]-(ends[SECTIONS[i-1]] if i else 0) for i,s in enumerate(SECTIONS)},ids_sha=hashlib.sha256(a.tobytes()).hexdigest()[:16],para_sha256=PARA_SHA,rewrites_sha256=REW_SHA)
def write(para_f,rew_f,train_f,dev_f,out):
    out=pathlib.Path(out);out.mkdir(parents=True,exist_ok=True)
    assert hashlib.sha256(open(para_f,'rb').read()).hexdigest()==PARA_SHA and hashlib.sha256(open(rew_f,'rb').read()).hexdigest()==REW_SHA,'input sha256 differs'
    para=json.load(open(para_f));rew=json.load(open(rew_f));assert len(para)==332 and all(len(rew[p['pid']])==4 for p in para)
    frags,a,meta=build(para,rew,json.load(open(train_f)),json.load(open(dev_f)))
    np.save(out/'streamP3_ids.npy',a);(out/'streamP3_meta.json').write_text(json.dumps(meta,indent=1))
    with open(out/'streamP3_stream_sentences.jsonl','w') as f:
        f.write(json.dumps(dict(type='meta',note='private, internal research only; no publication or redistribution',**meta))+'\n')
        for i,fr in enumerate(frags):f.write(json.dumps(dict(type='frag',fid=i,**fr))+'\n')
    print(json.dumps(meta));return meta
def load(out):
    out=pathlib.Path(out);return np.load(out/'streamP3_ids.npy'),json.load(open(out/'streamP3_meta.json'))
def check(out):
    out=pathlib.Path(out);a,meta=load(out);tk=S.tokenizer();ids=[]
    for l in open(out/'streamP3_stream_sentences.jsonl'):
        r=json.loads(l)
        if r['type']=='frag':t=tk.encode(r['text'],add_special_tokens=False).ids;assert len(t)==r['ntok'];ids+=t
    ids=np.asarray(ids[:len(a)],np.int64);assert np.array_equal(ids,a) and hashlib.sha256(a.tobytes()).hexdigest()[:16]==meta['ids_sha'] and len(a)%CH==0
    assert all(meta['read_chunks'][s]<meta['n_chunks'] for s in SECTIONS);print(f'stream P3 ok: {len(a)} tokens, {meta["n_chunks"]} chunks, read chunks {meta["read_chunks"]}, ids_sha {meta["ids_sha"]}');return a
def selftest():
    train={'data':[{'title':'T%d'%i,'paragraphs':[{'context':'One. Two three.'}]*3} for i in range(60)]};dev={'data':[{'title':'T1','paragraphs':[]}]}
    fm,fx,ov=filler_paragraphs(train,dev,10,extra=4);assert len(fm)==10 and len(fx)==4 and ov==0 and all(tsha(t)!=tsha('T1') for t,_ in fm+fx)
    print('e57_pqa3_stream selftest ok')
def main():
    if '--selftest' in sys.argv:selftest();return
    g=lambda k:sys.argv[sys.argv.index(k)+1]
    if '--check' in sys.argv:check(g('--out'));return
    write(g('--para'),g('--rewrites'),g('--train'),g('--dev'),g('--out'));check(g('--out'))
if __name__=='__main__':main()
