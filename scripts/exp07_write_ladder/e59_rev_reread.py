"""E59 reverse-question RE-READ of saved post-write states (GPU, read only, NO training; new file; Hans 11/19): the weights saved by the E59 E2E stream arms (NAT, NATF, CHK, CHKF) of seeds 71..74 at exposures 16 / 32 / 64 / 128 (experiments/exp07_write_ladder/e59/states/E59_e59_<arm>_s<seed>/state_*.npz; file names and hashes taken from the arm json 'reads') are scored with the reverse items of e59_rev.py (place -> person, 16 name candidates, same summed span margin, out-of-window neutral tail at every exposure, in-window context at exposure 128) and W0. Output E59_<tag>_<arm>_s<seed>.rev.read.json in the format of the inline reverse readout (RG.make_records). ISO / FACT arms save no states: their reverse readout comes from a re-run with --reverse (e59_srv_run.py).
  python3 e59_rev_reread.py --seed 71 --arms NAT,CHK --e59-dir DIR --tag e59 --out O | --selftest"""
import sys,os,json,time,hashlib,argparse,pathlib
import numpy as np
HERE=pathlib.Path(__file__).resolve().parent;ROOT=HERE.parents[1]
for p in (HERE,HERE/'e57',ROOT/'scripts/exp04_what_ttt_retains',ROOT/'scripts/exp05_write_dynamics'):sys.path.insert(0,str(p))
def run(a):
    import e55_run as R55,e55_read as R,e55_gpu as RG,gen_facts as G,e57_syn_pass as SP,e57_oracle_run as OR,e59_rev as RV
    out=pathlib.Path(a.out);out.mkdir(parents=True,exist_ok=True);seed=a.seed;cell='v3_16_1';lg=lambda *x:print(*x,flush=True)
    ci=OR.register_cell(cell);ro=list(ci['read_apps']);assert ro==[16,32,64,128]
    enc,w,tf=R55.build(SP.synth_args(seed,cell),seed,read_apps=set(ro),arms=['NAT'])
    LD,s3,C=R55.lab();rt=LD.Runtime();rt32=LD.runtime_fp32();rd=RG.Reader55(rt,rt32,LD.packing,LD._TailCtx,LD.reuse.get_filter_spec,LD.reuse._hash_pytree,LD.jax,LD.eqx,LD.jnp);tok=rt.tok
    P=R.plan(w,enc);items_r=RV.rev_items(w);raw_r=R55.rows_for(LD,rt,items_r);RG.check_rows(raw_r,len(items_r),G.NCAND);P_r=RV.rev_plan(P)['read'];out_tail=R55.out_tail_8192(LD,tok,rt.bos,R55.forbid_list(w))
    todo=[ar for ar in a.arms.split(',') if a.rerun or not (out/f'E59_{a.tag}_{ar}_s{seed}.rev.read.json').exists()]
    if not todo:lg('all records exist');return
    sf={}
    for ar in todo:
        r=json.load(open(pathlib.Path(a.e59_dir)/f'E59_{a.tag}_{ar}_s{seed}.json'));sf[ar]={}
        for k,e in r['reads'].items():
            f=e.get('_file')
            if f is None:continue
            assert os.path.exists(ROOT/f) or os.path.exists(f),('state file missing',f);sf[ar][int(k)+1]=(str(ROOT/f) if not os.path.isabs(f) and (ROOT/f).exists() else f,e['_hash'])
        assert sorted(sf[ar])==ro,(ar,sorted(sf[ar]))
    t0=time.monotonic();read_,W0,idx=RG.read_world(rd,raw_r,len(items_r),G.NCAND,P_r,out_tail,sf,log=lg,only_apps=None,which_ctx=('out','in'),w0_model=rt.base,in_apps=[128])
    info={'cell':cell,'seed':seed,'k':w['k'],'n':w['n'],'p':1,'bank':w['bank_kind'],'tag':a.tag,'complete':True,'direction':'reverse (place -> person)','source':'saved post-write states (re-read, no training)','sec_read':time.monotonic()-t0,'place_len':w['place_len']}
    for ar,rec in RG.make_records(read_,W0,G.item_meta(items_r),P_r,idx,info).items():
        json.dump(rec,open(out/f'E59_{a.tag}_{ar}_s{seed}.rev.read.json','w'));lg('wrote',out/f'E59_{a.tag}_{ar}_s{seed}.rev.read.json')
def selftest():
    import ast;ast.parse(open(__file__).read());print('e59_rev_reread selftest ok (syntax)')
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--seed',type=int);ap.add_argument('--arms',default='NAT');ap.add_argument('--e59-dir',default='experiments/exp07_write_ladder/e59');ap.add_argument('--tag',default='e59');ap.add_argument('--out');ap.add_argument('--rerun',action='store_true');ap.add_argument('--selftest',action='store_true');a=ap.parse_args()
    if a.selftest:selftest();return
    run(a)
if __name__=='__main__':main()
