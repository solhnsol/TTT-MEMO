"""FictionalQA second world on Qwen3-4B-Base (GPU; new file; thread-server 11/18; EXPLORATORY; wrapper of e59_qwen_run.run / e59_qwen_mask arms; nothing existing edited): world seeds = labels 211..218 (builder seeds 201..208), worlds from --fq-worlds (Qwen3 worlds after the Qwen W0 screen / replacement / control completion, fq_<builder>_cap3_VAR_w0.json; every chunk exactly 1024 Qwen3 tokens, 16 fact sentences per chunk), items = fq_world.build_items (question frame ends 'Answer:', candidates = the 16 golds of the world, 3 Qwen3 tokens each). Fast weights = full gate / up / down MLP matrices of the last 9 layers (fullmlp, fp32), the fixed per-chunk update norm T = --step-norm (0.30426 = 6% path, as 1311 / E59), same reads (exposures 16 / 32 / 64 / 128, out-of-window neutral tail + in-window of the last 7 chunks), canary and per-step records as e59_qwen_run.py. Arms: NAT (= NATS: carried detached rolling KV of 7 chunks + the current chunk, loss over all targets), BAND1024 (band mask in the 9 fast layers, as 1311), BANDALL1024 (band 1024 in ALL 36 layers = the E2E mask, see e59_qwen_mask.py). Output e59q_<arm>_s<label>.json (+ sketch npy).
  python3 fq_qwen_run.py --world-seed 211 --fq-worlds DIR --fast fullmlp --lr 1.0 --clip 1.0 --step-norm 0.30426 --arms NAT,BAND1024,BANDALL1024 --reads 16,32,64,128 --out O [--max-chunks N] | --selftest"""
import sys,json
from pathlib import Path
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
sys.path.insert(0,str(HERE));sys.path.insert(0,str(ROOT/'scripts/exp07_write_ladder'))
import torch
import cell_band_ladder as CB
import e59_qwen_run as E,e59_qwen_mask as M
import fq_world as FW
INV={v:k for k,v in FW.SEED_LABEL.items()}
def parser():
    ap=E.parser();ap.add_argument('--fq-worlds');return ap
def install(dirp):
    import gen_facts as G
    def build_world(tokfile,seed,cell,read_apps):
        from tokenizers import Tokenizer
        tk=Tokenizer.from_file(str(tokfile));w=FW.load_world(Path(dirp)/f'fq_{INV[int(seed)]}_cap3_VAR_w0.json');gw=FW.to_gen_world(w,tk,read_apps=tuple(read_apps))
        assert all(len(c['ids'])==1024 for c in gw['chunks']),'chunks are not exactly 1024 tokens of this tokenizer'
        ch=gw['chunks'];ids=[int(x) for x in G.stream_ids(ch)];return tk,gw,ids,G.stream_sha(ch),FW.build_items(gw)
    CB.build_world=build_world;M.install()
def selftest():
    sys.argv=[sys.argv[0]];import tempfile,shutil
    wd=ROOT/'experiments/exp07_write_ladder/fq/worlds_qwen3_check';assert wd.exists(),'selftest needs experiments/exp07_write_ladder/fq/worlds_qwen3_check (Qwen worlds named *_w0.json): run fq_qwen_prep.sh first or copy worlds_qwen3 with the _w0 suffix'
    td=Path(tempfile.mkdtemp());install(wd);a=parser().parse_args(['--tiny','--device','cpu','--max-chunks','3','--chunk','1024','--reads','1,3','--tail-len','256','--readout-batch','16','--threads','4','--lora-layers','99','--world-seed','211','--lr','1e-2','--rerun','--step-norm','0.25','--fq-worlds',str(wd),'--out',str(td),'--arms','NAT,BAND1024,BANDALL1024']);torch.manual_seed(0);E.run(a)
    R={x:json.load(open(td/f'e59q_{x}_s211.json')) for x in ('NAT','BAND1024','BANDALL1024')};assert all(len(r['steps'])==3 and abs(r['steps'][2]['applied_norm']-0.25)<=1e-4*0.25+1e-7 for r in R.values()) and R['NAT']['meta']['n_chunks']==3
    shutil.rmtree(td,ignore_errors=True);print('fq_qwen_run selftest ok')
if __name__=='__main__':
    if '--selftest' in sys.argv:selftest()
    else:
        a=parser().parse_args();install(a.fq_worlds);torch.manual_seed(0);E.run(a)
