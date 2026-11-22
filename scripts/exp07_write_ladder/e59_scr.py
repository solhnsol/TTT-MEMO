"""Scrambled-U2 world for the 'why does hiding OTHER facts help in the synthetic world' check (job 1339; new file; thread-server 11/22; EXPLORATORY): the U12 world (e59_u12.py: 32 facts, 16 read facts U1 + 16 extra facts U2, one sentence per fact in every chunk) with the tokens of every U2 sentence PERMUTED inside the sentence span (np.random.default_rng((seed + 9999, chunk, fact)).permutation): bag of words, names and places preserved, the sentence structure (template, syntax, order of name / place / occupation / verb / object) destroyed. U1 sentences, filler, separators and all chunk boundaries are unchanged; the read items and the world seeds are those of job 1328. Question: does the extra gain of hiding the other facts survive when the hidden extra facts no longer share the structure of the read facts (structure / template similarity) or does it come from their lexical content?
  scramble_u2(ids, chunks, seed, nread=16) -> new ids (np.int64) ; python3 e59_scr.py --selftest [--seed 231]"""
import sys,pathlib
import numpy as np
HERE=pathlib.Path(__file__).resolve().parent;ROOT=HERE.parents[1]
for p in (HERE,HERE/'e57',ROOT/'scripts/exp04_what_ttt_retains',ROOT/'scripts/exp05_write_dynamics'):sys.path.insert(0,str(p))
CH=1024
def scramble_u2(ids,chunks,seed,nread=16):
    out=np.array(ids,dtype=np.int64,copy=True)
    for c,ch in enumerate(chunks):
        for x in ch['sents']:
            f=int(x['fact'])
            if f<nread:continue
            a,b=c*CH+int(x['start']),c*CH+int(x['end'])
            if b>len(out):continue
            rng=np.random.default_rng((int(seed)+9999,c,f));out[a:b]=out[a:b][rng.permutation(b-a)]
    return out
def selftest(seed=231):
    import e55_run as R55,e57_syn_pass as SP,e59_u12 as U
    U.install();import types
    a=types.SimpleNamespace(cell=U.CELL,seed=seed,tokenizer_file=None,bg='pool',decoy='names',stub=False,arms='NAT');enc,w,tf=R55.build(a,seed,read_apps={16,32,64,128},arms=['NAT']);ids=np.asarray(R55.streams(w,enc)['A'][0]);ch=w['chunks']
    new=scramble_u2(ids,ch,seed);assert new.shape==ids.shape and (new!=ids).any()
    mask=np.zeros(len(ids),bool);n_u2=0;changed=0
    for c,chk in enumerate(ch):
        for x in chk['sents']:
            a_,b_=c*CH+int(x['start']),c*CH+int(x['end'])
            if int(x['fact'])>=16:
                mask[a_:b_]=True;n_u2+=1;assert sorted(ids[a_:b_].tolist())==sorted(new[a_:b_].tolist()),'token multiset of a U2 sentence must be preserved';changed+=int((ids[a_:b_]!=new[a_:b_]).any())
    assert (ids[~mask]==new[~mask]).all(),'only U2 sentence spans change'
    assert changed>0.95*n_u2,(changed,n_u2)
    assert (scramble_u2(ids,ch,seed)==new).all() and (scramble_u2(ids,ch,seed+1)!=new).any()
    print('e59_scr selftest ok: %d U2 sentences scrambled (%.3f of the stream tokens), U1 / filler unchanged, multiset preserved, deterministic'%(n_u2,mask.mean()))
if __name__=='__main__':
    if '--selftest' in sys.argv:selftest(int(sys.argv[sys.argv.index('--seed')+1]) if '--seed' in sys.argv else 231)
