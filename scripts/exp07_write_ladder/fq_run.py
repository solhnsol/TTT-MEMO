"""FictionalQA second world run wrapper (GPU; new file; Hans 11/18; EXPLORATORY; seeds = run labels 211..218 = builder seeds 201..208, SEED_LABEL of fq_world.py). Wrapper of the UNCHANGED e57_oracle_run.main (E2E 3B v3_16_1 weights, clipped SGD engine, K0 = 0 via --k0 0 default, same reads at exposures 16/32/64/128 (out-of-window neutral 8192 tail + in-window of group 0), canary, logs, norm statistics as the NATS / band-1024 arms of jobs 1307 / 1308). What the wrapper changes (in this process only; nothing existing is edited): cell 'fq<label>' (a) e57_oracle_run.register_cell returns the read points (16,32,64,128), (b) e55_run.build returns (tokenizer, world, file) with the world of fq_world.to_gen_world(<--fq-worlds>/fq_<builder seed>_cap3_VAR_w0.json) (exactly 1024-token chunks by the token-level fit of fq_world, 16 fact sentences per chunk, 20 items = 16 facts + 4 controls), (c) gen_facts.build_items for bank_kind 'fq' = fq_world.build_items (question frame ends 'Answer:', candidates = the 16 golds ' '+gold, gold first).
Arms: NATS (no mask) and NATMASK with --mask-arm ALLQ1024 (band 1024, every row) as in 1308 gate / 1307.
  python3 fq_run.py --seed 211 --fq-worlds DIR --out O --arms NATS,NATMASK --mask-arm ALLQ1024 --save-apps 16,32,64,128 --nat-loss-every 8 --tag fq [--smoke N] [--rerun] | --selftest"""
import sys,pathlib,json,types
HERE=pathlib.Path(__file__).resolve().parent;ROOT=HERE.parents[1]
for p in (HERE,HERE/'e57',ROOT/'scripts/exp04_what_ttt_retains',ROOT/'scripts/exp05_write_dynamics'):sys.path.insert(0,str(p))
import fq_world as FW
INV={v:k for k,v in FW.SEED_LABEL.items()}
STATE={}
def world_path(dirp,label):return pathlib.Path(dirp)/f'fq_{INV[label]}_cap3_VAR_w0.json'
def install(dirp):
    import e57_oracle_run as OR,e55_run as R55,gen_facts as G
    tok=FW.load_tokenizer();orig_reg=OR.register_cell;orig_build=R55.build;orig_items=G.build_items
    def register_cell(cell):
        if cell.startswith('fq'):return dict(read_apps=FW.READ_APPS)
        return orig_reg(cell)
    def build(a,seed=None,read_apps=None,arms=None):
        if str(getattr(a,'cell','')).startswith('fq'):
            seed=a.seed if seed is None else seed;w=FW.load_world(world_path(dirp,int(seed)));gw=FW.to_gen_world(w,tok);STATE['gw']=gw;return tok,gw,__file__
        return orig_build(a,seed,read_apps,arms)
    OR.register_cell=register_cell;R55.build=build;G.build_items=lambda w:FW.build_items(w) if w.get('bank_kind')=='fq' else orig_items(w)
    return OR
def main():
    if '--selftest' in sys.argv:selftest();return
    a=sys.argv[1:];d=a.index('--fq-worlds');dirp=a[d+1];rest=a[:d]+a[d+2:]
    OR=install(dirp);seed=int(rest[rest.index('--seed')+1]);assert seed in INV,seed
    sys.argv=[sys.argv[0]]+rest+['--cell',f'fq{seed}'];OR.main()
def selftest():
    wd=ROOT/'experiments/exp07_write_ladder/fq/worlds_w0c';assert wd.exists(),wd
    import e55_run as R55,gen_facts as G,e55_read as R,types
    OR=install(wd);assert OR.register_cell('fq211')['read_apps']==(16,32,64,128)
    a=types.SimpleNamespace(cell='fq211',seed=211);enc,w,_=R55.build(a,211,arms=['NAT'])
    assert w['seed']==211 and w['fq']['builder_seed']==201 and len(w['chunks'])==128 and all(len(c['ids'])==1024 for c in w['chunks'])
    S=R55.streams(w,enc);ids,sha=S['A'];assert len(ids)==128*1024 and sha==G.stream_sha(G.realize(w))
    P=R.plan(w,enc);assert len(P['items'])==20 and set(P['read'])=={16,32,64,128},len(P['items'])
    fl=R55.forbid_list(w);assert all(isinstance(x,str) for x in fl) and len(fl)>=16
    print('fq_run selftest ok (world 211: 128 chunks x 1024, 20 items, stream sha',sha,')')
if __name__=='__main__':main()
