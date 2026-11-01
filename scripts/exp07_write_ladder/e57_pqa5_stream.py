"""E57 public QA stage 5 (E57 122-U / 123), nine-version accumulating stream (new file, CPU): filler / original 332 paragraphs / re-description 1..8 = ten sections of 332 paragraphs, written once. Same construction as e57_pqa3_stream.py (job 610): same filler (SQuAD train, seed 20261101), same fixed permutation of the 332 paragraphs for every content section, same sentence layout, the stream completed to a chunk boundary by train filler paragraphs; only the number of re-descriptions differs (rewrites_all8.json, sha256 eb08d0aa...; its first four re-descriptions equal rewrites_all.json of job 610, checked). The first six sections therefore have the same tokens as the stream of job 610.
Outputs in <out>: streamP5_ids.npy, streamP5_meta.json (section token ends, read chunks, spill tokens, ids_sha), streamP5_stream_sentences.jsonl.
  python3 e57_pqa5_stream.py --para P.json --rewrites8 R8.json --train train-v1.1.json --dev dev-v1.1.json --out DIR | --check --out DIR | --selftest"""
import sys,json,hashlib,pathlib
import numpy as np
HERE=pathlib.Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
import e57_pqa2_stream as S
import e57_pqa3_stream as S3
CH=1024;SEED=S3.SEED;SECTIONS=['filler','orig']+[f'rw{k}' for k in range(1,9)]
REW8_SHA='eb08d0aacc78a512a19003bbc27cf084357b8ab008b84e037eb82a0f494cdb72'
def build(para,rew,train,dev,seed=SEED):
    tk=S.tokenizer();n=len(para);perm=np.random.default_rng(seed).permutation(n);fm,fx,ov=S3.filler_paragraphs(train,dev,n,seed)
    secs={'filler':[(t,'filler',p) for t,p in fm],'orig':[(para[i]['pid'],'orig',para[i]['context']) for i in perm]}
    for k in range(8):secs[f'rw{k+1}']=[(para[i]['pid'],f'rw{k+1}',rew[para[i]['pid']][k]) for i in perm]
    frags=[];ids=[];ends={};t=0
    def add(sec,key,tag,text):
        nonlocal t
        for si,(s,x) in enumerate(S3.enc_par(tk,text)):frags.append(dict(section=sec,key=key,tag=tag,sent=si,text=s,ntok=len(x),tok0=t));ids.extend(x);t+=len(x)
    for sec in SECTIONS:
        for key,tag,text in secs[sec]:add(sec,key,tag,text)
        ends[sec]=t
    target=-(-ends['rw8']//CH)*CH
    for ti,tx in fx:
        if t>=target:break
        add('tail',ti,'tail',tx)
    assert t>=target,'continuation filler too short'
    a=np.asarray(ids[:target],np.int64);last={s:(ends[s]-1)//CH for s in SECTIONS};spill={s:int((last[s]+1)*CH-ends[s]) for s in SECTIONS}
    return frags,a,dict(seed=seed,n_paragraphs=n,sections=SECTIONS,section_end_tokens=ends,read_chunks=last,spill_tokens=spill,n_chunks=target//CH,cut_tokens=t-target,filler_dev_title_overlap=ov,
        section_tokens={s:ends[s]-(ends[SECTIONS[i-1]] if i else 0) for i,s in enumerate(SECTIONS)},ids_sha=hashlib.sha256(a.tobytes()).hexdigest()[:16],para_sha256=S3.PARA_SHA,rewrites8_sha256=REW8_SHA)
def write(para_f,rew_f,train_f,dev_f,out):
    out=pathlib.Path(out);out.mkdir(parents=True,exist_ok=True)
    assert hashlib.sha256(open(para_f,'rb').read()).hexdigest()==S3.PARA_SHA and hashlib.sha256(open(rew_f,'rb').read()).hexdigest()==REW8_SHA,'input sha256 differs'
    para=json.load(open(para_f));rew=json.load(open(rew_f));assert len(para)==332 and all(len(rew[p['pid']])==8 for p in para)
    frags,a,meta=build(para,rew,json.load(open(train_f)),json.load(open(dev_f)))
    np.save(out/'streamP5_ids.npy',a);(out/'streamP5_meta.json').write_text(json.dumps(meta,indent=1))
    with open(out/'streamP5_stream_sentences.jsonl','w') as f:
        f.write(json.dumps(dict(type='meta',note='private, internal research only; no publication or redistribution',**meta))+'\n')
        for i,fr in enumerate(frags):f.write(json.dumps(dict(type='frag',fid=i,**fr))+'\n')
    print(json.dumps(meta));return meta
def load(out):
    out=pathlib.Path(out);return np.load(out/'streamP5_ids.npy'),json.load(open(out/'streamP5_meta.json'))
def check(out,ref610=None):
    out=pathlib.Path(out);a,meta=load(out);tk=S.tokenizer();ids=[]
    for l in open(out/'streamP5_stream_sentences.jsonl'):
        r=json.loads(l)
        if r['type']=='frag':t=tk.encode(r['text'],add_special_tokens=False).ids;assert len(t)==r['ntok'];ids+=t
    ids=np.asarray(ids[:len(a)],np.int64);assert np.array_equal(ids,a) and hashlib.sha256(a.tobytes()).hexdigest()[:16]==meta['ids_sha'] and len(a)%CH==0
    if ref610:
        b=np.load(pathlib.Path(ref610)/'streamP3_ids.npy');m=meta['section_end_tokens']['rw4'];assert np.array_equal(a[:m],b[:m]),'first six sections differ from the stream of job 610'
    print(f'stream P5 ok: {len(a)} tokens, {meta["n_chunks"]} chunks, read chunks {meta["read_chunks"]}, ids_sha {meta["ids_sha"]}'+(' (first six sections identical to job 610)' if ref610 else ''));return a
def selftest():
    S3.selftest();assert SECTIONS[-1]=='rw8' and len(SECTIONS)==10;print('e57_pqa5_stream selftest ok')
def main():
    if '--selftest' in sys.argv:selftest();return
    g=lambda k:sys.argv[sys.argv.index(k)+1]
    ref=g('--ref610') if '--ref610' in sys.argv else None
    if '--check' in sys.argv:check(g('--out'),ref);return
    write(g('--para'),g('--rewrites8'),g('--train'),g('--dev'),g('--out'));check(g('--out'),ref)
if __name__=='__main__':main()
