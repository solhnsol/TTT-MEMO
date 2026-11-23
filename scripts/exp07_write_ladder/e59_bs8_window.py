"""Block-stream re-read with the KEPT WINDOW (job 1345w; GPU; new file; thread-server 11/23; design: Hans 11/23 'W0 group 0 question; write group 0, window holds group 0, ask group 0; write group 1, window holds group 1, ask groups 0 and 1; ...'; EXPLORATORY): the block-stream experiment of job 1296 (blk16x8_c16: 8 blocks of 16 facts, 128 chunks, shuffled block assignment per seed, band arm ALLQ<W>) is RE-RUN with exactly the same writes (the unchanged runner e57_oracle_run.main through the patches of e57_blockstream8_run.py; the reads only score frozen weights, so the trajectory is bit-identical and final_hash can be compared with the stored one of job 1296) and the reads are changed: at the end of block k (read point = appearance 16 (k+1)) ONE context 'w<k>' = the REAL current window (the last 8 stream chunks = chunks 16k+8 .. 16k+15 = exactly the 'in_g0' tail of the original run; it holds 8 of the 16 exposures of EVERY fact of block k, because every chunk of a block carries one sentence of each of its 16 facts) is used to ask ALL facts of the blocks 0..k (+ the control items): block k = in-window read (lag 0), blocks j < k = reads of facts that are NOT in the window (lag k - j; their facts occur nowhere in the window: asserted from the chunk sentence tables), with the SAME window kept. The window is never changed for a question; only the question set differs. W0 (before the first update) reads the same contexts w<k> (one W0 read per context, cached by name; W0 of the block-0 question under w0 = Hans' first step). Questions, candidates (gold + 15 same-block places) and margin are those of the original reads. The neutral 'out' read of all items is kept at the FINAL read point only (link to jobs 1293 / 1296; with equal final_hash the original neutral reads of every point are valid for the pairing). Added arms later: the same call with --arms NATMASK --mask-arm ALLQ256 (or ALLQ128, ALLQ64) and another --tag (e.g. bs8w256): nothing about the write path, the seeds, the block permutation, the contexts or the reads changes; the sidecar E55_<tag>_ctx_s<seed>.json records sha256 of every context tail and the item lists so that the analyzer can assert equality across tags. NATS / NATMASK(1024) do not have to be run again.
  python3 e59_bs8_window.py --cell blk16x8_c16 --seed 82 --out O --masks none --mask-arm ALLQ1024 --arms NATS,NATMASK --save-apps 128 --nat-loss-every 8 --tag bs8w --shuffle-blocks [--smoke N]   |   --selftest"""
import sys,pathlib,json,hashlib
import numpy as np
HERE=pathlib.Path(__file__).resolve().parent;ROOT=HERE.parents[1]
for p in (HERE,HERE/'e57',ROOT/'scripts/exp04_what_ttt_retains',ROOT/'scripts/exp05_write_dynamics'):sys.path.insert(0,str(p))
import e57_blockstream8_run as B8
CH=1024;LC=16;NB=8;NF=128;STASH={'base':{}}
def build_win(plan,w,ids,nitems=NF):
    """plan = e55_read.plan(world)['read'] (original in_g0 contexts) -> same dict with, per read point a (block jj = a / LC - 1), ONE context 'w<jj>': items = facts of the blocks 0..jj (sorted by block, then id) + control items, tail = the original in_g0 tail (the real last 8 chunks), 'd' per item = lag jj - block (None for controls). Modified IN PLACE (make_records reads the contexts from the same plan object); the original tails are stashed so that a second call (second arm) works."""
    new={}
    for a,ent in plan.items():
        jj=a//LC-1
        if a not in STASH['base']:
            base=ent['ctx']['in_g0'];STASH['base'][a]=(list(base['tail']),[i for i in base['items'] if i>=nitems])
        tail,ctrl=STASH['base'][a];assert len(tail)==8*CH and [int(t) for t in tail[-CH:]]==[int(t) for t in ids[(a-1)*CH:a*CH]],('window tail is not the last 8 stream chunks',a)
        for c in range(a-8,a):
            for x in w['chunks'][c]['sents']:assert B8.BLK(x['fact'])==jj,('window chunk carries facts of another block',c,jj)
        items=sorted([i for i in range(nitems) if B8.BLK(i)<=jj],key=lambda i:(B8.BLK(i),i))+list(ctrl)
        ent['ctx']={f'w{jj}':{'items':items,'tail':tail,'window_chunks':8,'d':[jj-B8.BLK(i) if i<nitems else None for i in items]}};new[a]=ent
    return new
def sidecar(plan,path):
    """sha256 of every context tail, item lists and per-item lag -> json (the same for every arm / tag of one seed)"""
    json.dump({str(a):{n:{'tail_sha':hashlib.sha256(np.asarray(c['tail'],dtype=np.int64).tobytes()).hexdigest(),'items':c['items'],'d':c['d']} for n,c in ent['ctx'].items()} for a,ent in plan.items()},open(path,'w'))
def patch_reads_new(out=None,tag='bs8w',seed=0):
    import e55_gpu as RG,e55_run as R55;orig=RG.Reads
    b0=R55.build;s0=R55.streams
    def build2(*a,**k):
        r=b0(*a,**k);STASH['w']=r[1];return r
    def streams2(w,enc):
        S=s0(w,enc);STASH['ids']=np.asarray(S['A'][0]);return S
    R55.build=build2;R55.streams=streams2
    class MR(orig):
        def at(self,a,m):
            o={}
            if a in self.apps:o['out']=self.rd.score(self.P_out,m,self.n,self.nc)
            if a in self.in_apps:
                for name,c in sorted(self.plan[a]['ctx'].items()):
                    ii=np.asarray(c['items']);rows=[self.raw[i*self.nc+k] for i in ii for k in range(self.nc)];self.idx.setdefault(a,{})[name]=ii.tolist();P=self.rd.packed(rows,c['tail'])
                    if name not in self.W0:self.W0[name]=self.rd.score(P,self.m0,len(ii),self.nc)       # W0 of the SAME context, cached by name (one per block end)
                    o[name]=self.rd.score(P,m,len(ii),self.nc);del P
            return o
    def mk(*a,**k):
        a=list(a);plan=a[4];out_tail=a[5];last=max(plan);a[4]=build_win(plan,STASH['w'],STASH['ids']);a[7]=[last];a[8]=sorted(plan)
        if out is not None:sidecar(a[4],pathlib.Path(out)/f'E55_{tag}_ctx_s{seed}.json')
        return MR(*a,**k)
    RG.Reads=mk
def selftest():
    import gen_facts as G,e55_read as R
    cell='blk16x8_c16';B8.patch(cell,seed=7);enc=G.StubTok();bg=G._synth_bg(9000);w=G.cell_world(7,cell,enc,bg,decoys=('names','nofact'));ids=np.asarray(G.stream_ids(w['chunks']));P=R.plan(w,enc)
    for rep in range(2):                                           # second call = second arm: must give the same plan
        new=build_win(P['read'],w,ids)
        assert sorted(new)==[LC*(j+1) for j in range(NB)]
        for a,ent in new.items():
            jj=a//LC-1;(name,c),=ent['ctx'].items();assert name==f'w{jj}' and len(c['tail'])==8*CH and len(c['items'])==16*(jj+1)+G.NCTRL and len(c['d'])==len(c['items'])
            assert sorted(set(x for x in c['d'] if x is not None))==list(range(jj+1)) and all(c['d'][n]==jj-B8.BLK(i) for n,i in enumerate(c['items']) if i<NF)
            assert c['items'][:-G.NCTRL]==sorted(c['items'][:-G.NCTRL],key=lambda i:(B8.BLK(i),i))
            for bj in range(jj):assert all(x['fact']!=i for ch in range(a-8,a) for x in w['chunks'][ch]['sents'] for i in range(NF) if B8.BLK(i)==bj)
    import tempfile,os;td=tempfile.mkdtemp();sidecar(new,td+'/c.json');assert len(json.load(open(td+'/c.json')))==NB
    n_item_reads=sum(len(c['items']) for ent in new.values() for c in ent['ctx'].values())
    print(f'e59_bs8_window selftest ok: 8 read points, one context w<k> = real last 8 chunks (tail of the original in_g0), items = blocks 0..k + {G.NCTRL} controls, lag per item, {n_item_reads} item-reads per run (+ W0 same, out at the final point {NF+G.NCTRL})')
if __name__=='__main__':
    if '--selftest' in sys.argv:selftest()
    else:
        A=sys.argv;cell=A[A.index('--cell')+1];seed=int(A[A.index('--seed')+1]);shuf='--shuffle-blocks' in A;o=pathlib.Path(A[A.index('--out')+1]);tag=A[A.index('--tag')+1] if '--tag' in A else 'bs8w'
        sys.argv=[x for x in A if x!='--shuffle-blocks'];F,NB_,LC_=B8.parse(cell);assert (F,NB_,LC_)==(16,8,16),'the wrapper is written for blk16x8_c16';B8.patch(cell,seed if shuf else None);o.mkdir(parents=True,exist_ok=True);patch_reads_new(o,tag,seed)
        if shuf:json.dump(dict(world_seed=seed,F=F,NB=NB_,block_of_fact=[B8.BLK(f) for f in range(F*NB_)]),open(o/f'E55_{tag}_perm_s{seed}.json','w'))
        import e57_oracle_run as O;O.main()
