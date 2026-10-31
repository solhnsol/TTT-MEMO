"""E57 public QA stage 2 (E57 122-Q), stream S builder (new file, CPU): SQuAD dev-v1.1 without Super_Bowl_50 and Warsaw (46 documents, 1,964 paragraphs, ~242k words). One cycle = all documents once; the paragraphs of a document keep the original order, the document order is shuffled per cycle (np.random.default_rng(SEED + cycle).permutation). Text layout as the sentence jsonls of the news streams: a paragraph is whitespace-normalised, split into sentences (regex: end mark . ! ? followed by a space and a capital letter / digit / quote / bracket), the first sentence of a paragraph is written with the prefix "\n\n", later ones with the prefix " "; every fragment (= sentence) is encoded on its own and the ids are concatenated (as e57_t1t2v2_run.stream_ids). Each cycle is cut to a multiple of 1024 tokens (cut tokens are reported), the two cycles are concatenated (cycle 2 continues the stream of cycle 1).
Outputs: <out>/streamS_stream_sentences.jsonl (meta line + one 'frag' line per sentence with cycle / doc / par / chunk / tok0 / tok_in_chunk / ntok / text; the meta line holds ids_sha, n_chunks, chunks per cycle), <out>/streamS_ids.npy (int64), <out>/streamS_sid.npy (int32 [n_chunks, 1024]: sentence index per token, -1 = none; no token is uncovered here, so every row with a loss is eligible).
  python3 e57_pqa2_stream.py --squad dev-v1.1.json --out DIR [--seed 20261031] [--cycles 2]  |  --check --out DIR (re-encode the texts of the jsonl and compare the ids_sha)  |  --selftest"""
import sys,json,re,hashlib,pathlib
import numpy as np
HERE=pathlib.Path(__file__).resolve().parent
CH=1024;SEED=20261031;EXCL=('Super_Bowl_50','Warsaw')
SPLIT=re.compile(r'(?<=[.!?])\s+(?=[A-Z0-9"\'(\[])')
def sentences(par):
    t=' '.join(par.split());return [s for s in SPLIT.split(t) if s]
def tokenizer():
    from tokenizers import Tokenizer
    return Tokenizer.from_file(str(HERE.parents[1]/'data/tokenizer/llama3/tokenizer.json'))
def docs_of(squad):
    return [(a['title'],[p['context'] for p in a['paragraphs']]) for a in squad['data'] if a['title'] not in EXCL]
def build(squad,cycles=2,seed=SEED,tk=None):
    """-> frags (each with 'keep' = number of its tokens that are inside the stream; the last sentence of a cycle may be cut), ids int64, tokens cut per cycle, chunks per cycle, n docs"""
    tk=tk or tokenizer();docs=docs_of(squad);frags=[];ids=[];ncut=[];ncyc=[]
    for cyc in range(cycles):
        order=np.random.default_rng(seed+cyc).permutation(len(docs));cf=[];ci=[]
        for di in order:
            title,pars=docs[di]
            for pi,par in enumerate(pars):
                for si,s in enumerate(sentences(par)):
                    txt=('\n\n' if si==0 else ' ')+s;t=tk.encode(txt,add_special_tokens=False).ids
                    cf.append(dict(cycle=cyc,doc=title,par=pi,sent=si,text=txt,ntok=len(t)));ci+=t
        keep=len(ci)//CH*CH;ncut.append(len(ci)-keep);ncyc.append(keep//CH);ids+=ci[:keep];t=0
        for f in cf:
            if t>=keep:break
            f['keep']=min(f['ntok'],keep-t);t+=f['ntok'];frags.append(f)
    return frags,np.asarray(ids,np.int64),ncut,ncyc,len(docs)
def sid_of(frags,nch):
    """int32 [nch, CH]: index of the sentence (fragment) of every token of the stream"""
    sid=np.concatenate([np.full(f['keep'],i,np.int32) for i,f in enumerate(frags)]);assert len(sid)==nch*CH;return sid.reshape(nch,CH)
def write(squad,out,cycles=2,seed=SEED):
    out=pathlib.Path(out);out.mkdir(parents=True,exist_ok=True)
    frags,a,ncut,ncyc,nd=build(squad,cycles,seed);nch=len(a)//CH;assert len(a)%CH==0 and sum(ncyc)==nch;sid=sid_of(frags,nch);sha=hashlib.sha256(a.tobytes()).hexdigest()[:16]
    np.save(out/'streamS_ids.npy',a);np.save(out/'streamS_sid.npy',sid)
    meta=dict(type='meta',note='private, internal research only; no publication or redistribution',stream='S_squad_dev_minus_SB50_Warsaw',seed=seed,cycles=cycles,n_docs=nd,n_paragraphs=sum(len(p) for _,p in docs_of(squad)),n_chunks=nch,chunks_per_cycle=ncyc,chunk_tokens=CH,cut_tokens_per_cycle=ncut,n_fragments=len(frags),stream_tokens=len(a),tokenizer='llama3',layout='paragraph = "\\n\\n"+sentence, later sentences " "+sentence; the last sentence of a cycle may be cut at the chunk boundary (keep < ntok)',ids_sha=sha)
    with open(out/'streamS_stream_sentences.jsonl','w') as f:
        f.write(json.dumps(meta)+'\n');t=0
        for i,fr_ in enumerate(frags):f.write(json.dumps(dict(type='frag',fid=i,chunk=t//CH,tok0=t,tok_in_chunk=t%CH,**fr_))+'\n');t+=fr_['keep']
    print(json.dumps({k:v for k,v in meta.items() if k not in ('note','layout')}));return meta
def load(out):
    out=pathlib.Path(out);return np.load(out/'streamS_ids.npy'),np.load(out/'streamS_sid.npy'),json.loads(open(out/'streamS_stream_sentences.jsonl').readline())
def check(out):
    out=pathlib.Path(out);a,sid,meta=load(out);tk=tokenizer();ids=[]
    for l in open(out/'streamS_stream_sentences.jsonl'):
        r=json.loads(l)
        if r['type']=='frag':t=tk.encode(r['text'],add_special_tokens=False).ids;assert len(t)==r['ntok'];ids+=t[:r['keep']]
    ids=np.asarray(ids,np.int64);n=len(a);assert np.array_equal(ids,a),'jsonl texts do not give streamS_ids.npy';assert hashlib.sha256(a.tobytes()).hexdigest()[:16]==meta['ids_sha']
    assert sid.shape==(n//CH,CH) and (sid>=0).all();print(f'stream S ok: {n} tokens, {n//CH} chunks, ids_sha {meta["ids_sha"]}');return a
def selftest():
    s=sentences('Hello world.  This is Mr. Smith. It is 5 p.m. Then "Quoted." (Paren) Done!');assert len(s)>=4 and s[0]=='Hello world.'
    sq={'data':[{'title':'A','paragraphs':[{'context':'One. Two three. Four.'},{'context':'Five. Six.'}]},{'title':'Warsaw','paragraphs':[{'context':'X.'}]},{'title':'B','paragraphs':[{'context':'Seven. Eight.'}]}]}
    d=docs_of(sq);assert [t for t,_ in d]==['A','B'];o1=np.random.default_rng(SEED).permutation(2);o2=np.random.default_rng(SEED+1).permutation(2);assert sorted(o1)==[0,1]
    print('e57_pqa2_stream selftest ok')
def main():
    if '--selftest' in sys.argv:selftest();return
    out=sys.argv[sys.argv.index('--out')+1]
    if '--check' in sys.argv:check(out);return
    sq=json.load(open(sys.argv[sys.argv.index('--squad')+1]));seed=int(sys.argv[sys.argv.index('--seed')+1]) if '--seed' in sys.argv else SEED;cy=int(sys.argv[sys.argv.index('--cycles')+1]) if '--cycles' in sys.argv else 2
    write(sq,out,cy,seed);check(out)
if __name__=='__main__':main()
