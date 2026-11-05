"""E57 synthetic world, pair-shuffled stream (new file; import for its side effect, CPU): with the environment variable E57_SYN_SHUF set (value = sha-tag of the scheme, e.g. 'perm1'), R55.build returns the world with the place tokens permuted INSIDE every chunk: the 16 sentences of a chunk (p = 1: one sentence per fact) exchange their place strings by a random derangement of that chunk (rng seeded by seed and chunk index). Every other token (names, frames, filler) keeps its position; a place string is re-encoded in the form of the sentence it lands in (' Place' after a space, 'Place' at a sentence start) and only assignments that keep the token length of every span are accepted, so each chunk keeps its length and its multiset of place strings (total exposures per place string = original). Questions, candidates and facts of the world are not touched (scored with the original pairs). Without the variable nothing changes."""
import os,random
import e55_run as R55
SHUF=os.environ.get('E57_SYN_SHUF','')
def shuffle_chunks(w,enc,seed):
    F={f['id']:f for f in w['facts']};stats=dict(chunks=0,spans=0,fixed_points=0,tries=0)
    form={}
    for f in F.values():form[f['id']]={pre:list(enc(pre+f['P'])) for pre in (' ','')}
    for c in w['chunks']:
        ids=list(c['ids']);spans=[]
        for s in c['sents']:
            seg=ids[s['start']:s['end']];hit=[(pre,i) for pre in (' ','') for i in range(len(seg)-len(form[s['fact']][pre])+1) if seg[i:i+len(form[s['fact']][pre])]==form[s['fact']][pre]]
            assert len(hit)==1,(c['k'],s['fact'],hit);pre,i=hit[0];spans.append((s['fact'],pre,s['start']+i,len(form[s['fact']][pre])))
        r=random.Random(f'E57shuf|{seed}|{c["k"]}');fs=[x[0] for x in spans];n=len(fs)
        while True:
            stats['tries']+=1;perm=fs[:];r.shuffle(perm)
            if any(perm[j]==fs[j] for j in range(n)):continue
            if all(len(form[perm[j]][spans[j][1]])==spans[j][3] for j in range(n)):break
        new=ids[:]
        for j,(f,pre,pos,L) in enumerate(spans):new[pos:pos+L]=form[perm[j]][pre]
        assert len(new)==len(ids);c['ids']=new;c['shuf']=dict(zip(fs,perm));stats['chunks']+=1;stats['spans']+=n
        for s in c['sents']:s['text_orig']=s.get('text')
    return stats
if SHUF:
    _b=R55.build
    def build(a,seed=None,read_apps=None,arms=None):
        enc,w,tf=_b(a,seed,read_apps=read_apps,arms=arms)
        if not w.get('_shuf'):w['_shuf']=shuffle_chunks(w,enc,w['seed'])
        return enc,w,tf
    R55.build=build
def check(out):
    """CPU invariants (selftest of the job): same length, only place tokens changed, per-chunk derangement with the same place multiset, name -> place pair counts, stream hashes"""
    import json,collections,e57_syn_pass as SP,gen_facts as G
    enc,w,tf=R55.build(SP.synth_args(64,'v3_16_1'),64,arms=['NAT']);_,w0,_=_b(SP.synth_args(64,'v3_16_1'),64,arms=['NAT'])
    F={f['id']:f for f in w['facts']};pairs=collections.Counter();changed=tot=0
    for c,c0 in zip(w['chunks'],w0['chunks']):
        assert len(c['ids'])==len(c0['ids'])==1024
        pos=set()
        for s in c0['sents']:
            f=F[s['fact']];seg0=c0['ids'][s['start']:s['end']]
            for pre in (' ',''):
                t=list(enc(pre+f['P']))
                for i in range(len(seg0)-len(t)+1):
                    if seg0[i:i+len(t)]==t:pos|=set(range(s['start']+i,s['start']+i+len(t)))
        d={i for i in range(1024) if c['ids'][i]!=c0['ids'][i]};assert d<=pos,'a non-place token changed';changed+=len(d);tot+=len(pos)
        perm=c['shuf'];assert sorted(perm.values())==sorted(perm) and all(perm[k]!=k for k in perm) and len(perm)==16
        for k,v in perm.items():pairs[(k,v)]+=1
    ex=collections.Counter();ex0=collections.Counter()
    for c in w['chunks']:ex.update(c['shuf'].values())
    for c in w0['chunks']:ex0.update(s['fact'] for s in c['sents'])
    assert ex==ex0,'exposures per place string differ'
    res=dict(stats=w['_shuf'],place_tokens=tot,changed_tokens=changed,distinct_name_place_pairs=len(pairs),original_pairs_kept=sum(1 for (k,v) in pairs if k==v),max_chunks_per_pair=max(pairs.values()),min_chunks_per_pair=min(pairs.values()),exposures_per_place_equal=True,stream_sha_shuffled=G.stream_sha(w['chunks']),stream_sha_original=G.stream_sha(w0['chunks']))
    json.dump(res,open(out,'w'),indent=1);print('shuffled-stream check ok',res)
if __name__=='__main__':
    import sys
    if '--check' in sys.argv:check(sys.argv[sys.argv.index('--out')+1])
