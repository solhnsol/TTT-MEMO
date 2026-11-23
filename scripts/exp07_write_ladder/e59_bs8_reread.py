"""Block-stream re-read with a REAL STREAM CONTEXT (job 1345; GPU; new file; thread-server 11/23; Hans via coordinator 'is the gain of the masked write an artefact of the neutral read context?'; EXPLORATORY): the block-stream experiment of job 1296 (blk16x8_c16: 8 blocks of 16 facts streamed one after the other, 128 chunks, shuffled block assignment per seed, arms NATS and NATMASK with band ALLQ1024) is RE-RUN with exactly the same writes (the unchanged runner e57_oracle_run.main through the patches of e57_blockstream8_run.py; the reads only score frozen weights, so the trajectory is bit-identical and the final_hash can be compared with the stored one) and the reads are changed: at every block end (read points = appearances 16 .. 128) the facts of every block j already written (lag = j_now - j) are read with the context = the first 8191 tokens of 8 real stream chunks of ANOTHER block i = (j + 4) % 8 (BOS + stream tokens, 8192 in total, none of block j's facts occurs in it: asserted from the chunk sentence tables) instead of the neutral SQuAD tail; the question rows, candidates and the margin (gold - best wrong over 16 candidates) are those of the original reads. The neutral 'out' read of all items is kept at the final read point only (the original run has it at every point; if the final_hash of the re-run equals the stored one the original neutral reads of every point are valid for the pairing). W0 reads per block context are cached by context name. Optional: states are saved at every block end (--save-apps).
  python3 e59_bs8_reread.py --cell blk16x8_c16 --seed 82 --out O --masks none --mask-arm ALLQ1024 --arms NATS,NATMASK --save-apps 128 --nat-loss-every 8 --tag bs8c --shuffle-blocks [--smoke N]   |   --selftest"""
import sys,pathlib,json
import numpy as np
HERE=pathlib.Path(__file__).resolve().parent;ROOT=HERE.parents[1]
for p in (HERE,HERE/'e57',ROOT/'scripts/exp04_what_ttt_retains',ROOT/'scripts/exp05_write_dynamics'):sys.path.insert(0,str(p))
import e57_blockstream8_run as B8
CH=1024;STASH={}
def build_ctx(plan,w,ids,bos,nblk=8,F=16,LC=16,offset=4,nitems=128):
    """-> new plan: for every read point a (appearance) the contexts bc<j> (j = blocks written so far): items of block j, tail = BOS + 8191 stream tokens of the first 8 chunks of block (j + offset) % nblk, 'd' = lag"""
    new={}
    for a,ent in plan.items():
        jj=a//LC-1;ctx={}
        for j in range(jj+1):
            items=[i for i in range(nitems) if B8.BLK(i)==j];ib=(j+offset)%nblk;c0=ib*LC
            for c in range(c0,c0+8):
                for x in w['chunks'][c]['sents']:assert B8.BLK(x['fact'])==ib and B8.BLK(x['fact'])!=j,('context chunk carries facts of the read block',c,j)
            tail=[int(bos)]+[int(t) for t in ids[c0*CH:c0*CH+8191]];assert len(tail)==8192
            ctx[f'bc{j}']={'items':items,'tail':tail,'window_chunks':8,'d':jj-j}
        ent['ctx']=ctx;new[a]=ent                                   # IN PLACE: make_records of the runner reads the contexts (names, 'd') from the same plan object
    return new
def patch_reads_new():
    import e55_gpu as RG,e55_run as R55;orig=RG.Reads
    b0=R55.build;s0=R55.streams
    def build2(*a,**k):
        r=b0(*a,**k);STASH['w']=r[1];return r
    def streams2(w,enc):
        S=s0(w,enc);STASH['ids']=np.asarray(S['A'][0]);return S
    R55.build=build2;R55.streams=streams2
    class MR(orig):
        def at(self,a,m):
            out={}
            if a in self.apps:out['out']=self.rd.score(self.P_out,m,self.n,self.nc)
            if a in self.in_apps:
                for name,c in sorted(self.plan[a]['ctx'].items()):
                    ii=np.asarray(c['items']);rows=[self.raw[i*self.nc+k] for i in ii for k in range(self.nc)];self.idx.setdefault(a,{})[name]=ii.tolist();P=self.rd.packed(rows,c['tail'])
                    if name not in self.W0:self.W0[name]=self.rd.score(P,self.m0,len(ii),self.nc)       # W0 cached by context name (one W0 read per block context, not per read point)
                    out[name]=self.rd.score(P,m,len(ii),self.nc);del P
            return out
    def mk(*a,**k):
        a=list(a);plan=a[4];out_tail=a[5];last=max(plan);a[4]=build_ctx(plan,STASH['w'],STASH['ids'],out_tail[0]);a[7]=[last];a[8]=sorted(plan)
        return MR(*a,**k)
    RG.Reads=mk
def selftest():
    import gen_facts as G,e55_read as R
    cell='blk16x8_c16';B8.patch(cell,seed=7);enc=G.StubTok();bg=G._synth_bg(9000);w=G.cell_world(7,cell,enc,bg,decoys=('names','nofact'));ids=np.asarray(G.stream_ids(w['chunks']));P=R.plan(w,enc)
    new=build_ctx(P['read'],w,ids,1)
    assert sorted(new)==[16*(j+1) for j in range(8)]
    for a,ent in new.items():
        jj=a//16-1;assert sorted(ent['ctx'])==[f'bc{j}' for j in range(jj+1)] and all(len(c['tail'])==8192 and len(c['items'])==16 and c['d']==jj-int(n[2:]) for n,c in ent['ctx'].items())
        for n,c in ent['ctx'].items():
            j=int(n[2:]);assert all(B8.BLK(i)==j for i in c['items']);assert c['tail'][1:]==[int(t) for t in ids[((j+4)%8)*16*CH:((j+4)%8)*16*CH+8191]]
    n_item_reads=sum(len(c['items']) for ent in new.values() for c in ent['ctx'].values())
    print(f'e59_bs8_reread selftest ok: 8 read points, contexts bc<j> for the blocks written so far, tails = BOS + 8191 stream tokens of another block ((j+4) mod 8), {n_item_reads} item-reads per run (+ W0 128, out at the final point {128+G.NCTRL})')
if __name__=='__main__':
    if '--selftest' in sys.argv:selftest()
    else:
        A=sys.argv;cell=A[A.index('--cell')+1];seed=int(A[A.index('--seed')+1]);shuf='--shuffle-blocks' in A
        sys.argv=[x for x in A if x!='--shuffle-blocks'];F,NB,LC=B8.parse(cell);B8.patch(cell,seed if shuf else None);patch_reads_new()
        if shuf:
            o=pathlib.Path(A[A.index('--out')+1]);o.mkdir(parents=True,exist_ok=True);tag=A[A.index('--tag')+1] if '--tag' in A else 'bs8c'
            json.dump(dict(world_seed=seed,F=F,NB=NB,block_of_fact=[B8.BLK(f) for f in range(F*NB)]),open(o/f'E55_{tag}_perm_s{seed}.json','w'))
        import e57_oracle_run as O;O.main()
