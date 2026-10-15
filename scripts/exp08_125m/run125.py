"""Exp08: the Exp07 binding pipeline on the official 125M TTT-E2E checkpoint (pretrain DCLM 8K), to check whether 125M
shows a measurable binding signal (L1 positive control) before any 125M outer-loop (meta-learning) experiment.
The 3B code is reused unchanged; only the config the runtime loads is swapped (125M pretrain config, checkpoint
125m_ttt_e2e_pretrain_dclm_8k_1x_cc). training.seq_length is raised so RoPE tables cover the 64K stream (RoPE has no
learned parameters; positions inside the 8K window are unchanged).
Usage: run125.py verify | ladder <ladder.py args...>   (outputs: experiments/exp08_125m/)"""
import os,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
os.environ.setdefault('EXP07_OUT',str(ROOT/'experiments/exp08_125m'));os.environ.setdefault('EXP05_OUT',str(ROOT/'experiments/exp08_125m'))
sys.path.insert(0,str(ROOT/'scripts/exp05_write_dynamics'));sys.path.insert(0,str(ROOT/'scripts/exp07_write_ladder'))
sys.path.insert(0,str(ROOT/'scripts/exp04_what_ttt_retains'))
import core
reuse=core.reuse
import importlib.util
spec=importlib.util.spec_from_file_location('baseline_125m_cfg',ROOT/'scripts/baseline_125m/common/baseline_config.py');c125=importlib.util.module_from_spec(spec);spec.loader.exec_module(c125)
def compose_125m():
    cfg=c125.compose_baseline_config();cfg.training.seq_length=131072;return cfg
reuse.config3b.compose_3b_128k_config=compose_125m
reuse.config3b.CHECKPOINT_DIR=c125.CHECKPOINT_DIR
if __name__=='__main__':
    mode=sys.argv[1];sys.argv=[sys.argv[0]]+sys.argv[2:]
    Path(os.environ['EXP07_OUT']).mkdir(parents=True,exist_ok=True)
    if mode=='verify':
        os.environ['VERIFY_FP32']='1';os.environ['VERIFY_KS']=os.environ.get('VERIFY_KS','')
        import verify_packing as V
        V.KS=[];V.main()
    elif mode=='ladder':
        import ladder as LD
        vp=ROOT/'experiments/exp05_write_dynamics/run3_readouts/verify_packing_fp32.log'   # 3B check; the 125M check is verify_packing_fp32.json in OUT
        LD.main()
    elif mode=='info':
        rt=core.Runtime();import jax
        print('suffix_len',int(rt.cfg.model.suffix_len),'hidden',int(rt.cfg.model.hidden_size),'layers',int(rt.cfg.model.num_hidden_layers),'window',int(rt.cfg.model.sliding_window_size),'mb',int(rt.cfg.model.mini_batch_size))
        print('inner params',sum(x.size for x in jax.tree.leaves(rt.base.inner_parameters())))
