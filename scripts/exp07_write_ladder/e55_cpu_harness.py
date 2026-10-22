"""CPU test harness for the E55 GPU code: a tiny randomly initialised TTT-E2E built from the REAL e2e classes (SWA with window 8192 and chunk 1024 as in the 3B model; only layers / width / vocabulary are small), plus stand-in modules so that the lab's s3.py (Ladder3: record / prefix_masked / write_step / read_state / native_step) imports unchanged on a machine without the 3B checkpoint, the GPU or the lab data. Used by e55_units.py. NOT a model of the 3B behaviour; it only shows that the swap composition, the plans and the step equivalences hold on the real code paths.
Needs: the public e2e source (E2E_SRC, default <repo>/e2e if it holds configs, else /tmp/e2e_src/e2e; commit a4fc4788 = the lab server's), jax 0.5.3 (the e2e lock), equinox 0.12.1, optax, hydra-core, omegaconf, orbax-checkpoint, tqdm."""
import os,sys,types
from pathlib import Path
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
os.environ.setdefault('JAX_PLATFORMS','cpu');os.environ['XLA_PYTHON_CLIENT_PREALLOCATE']='false'
def _e2e():
    for c in (os.environ.get('E2E_SRC'),str(ROOT/'e2e'),'/tmp/e2e_src/e2e'):
        if c and (Path(c)/'configs').is_dir():return c
    raise SystemExit('e2e source not found (set E2E_SRC)')
E2E=_e2e();sys.path.insert(0,E2E)
import jax,jax.numpy as jnp,equinox as eqx,numpy as np,optax
from hydra import compose,initialize_config_dir
from ttt.config import register_configs
from ttt.model.sharding import ModelSharding
from ttt.model.transformer import MetaModel,BlockCollectionSplit
from ttt.model.data import Batch
from ttt.utils.jax_utils import set_random_seed
from ttt.utils.filter_utils import get_filter_spec
CH=1024
_dpa=jax.nn.dot_product_attention
def _dpa_xla(*a,**k):
    if k.get('implementation')=='cudnn':k=dict(k,implementation='xla')
    return _dpa(*a,**k)
jax.nn.dot_product_attention=_dpa_xla          # no cuDNN on CPU
def clone_state(s):return jax.tree.map(lambda x:x,s)
def hash_pytree(tree):
    import hashlib
    d=hashlib.sha256()
    for _,leaf in jax.tree.flatten_with_path(tree)[0]:
        if eqx.is_inexact_array(leaf):d.update(np.asarray(leaf).tobytes())
    return d.hexdigest()[:16],{}
class TinyRuntime:
    """the attributes of core.Runtime that s3.Ladder3 / e55_core use"""
    def __init__(self,layers=4,suffix=2,hidden=64,heads=2,inter=128,vocab=512,seed=0):
        register_configs()
        ov=["+deploy=interactive","+experiment=3b/extension/ext-3b-e2e-128K","training.dummy_dataset=true","deploy_paths.data.dclm_filter_8k=unused","deploy_paths.data.books3=unused","checkpoint.resume_checkpoint_dir=/tmp/none","training.load_part=params",
            f"model.num_hidden_layers={layers}",f"model.suffix_len={suffix}",f"model.hidden_size={hidden}",f"model.num_attention_heads={heads}",f"model.intermediate_size={inter}",f"model.vocab_size={vocab}","model.bos_token_id=1","model.eos_token_id=2","model.compute_dtype=fp32","model.state_dtype=fp32","backend.num_devices=1","model.seq_len=16384","training.seq_length=16384",f"training.model_seed={seed}"]
        with initialize_config_dir(version_base=None,config_dir=str(Path(E2E)/'configs')):cfg=compose(config_name='config',overrides=ov)
        cfg.model.seq_len=cfg.training.seq_length;key=set_random_seed(cfg.training.model_seed);self.mesh=ModelSharding(cfg).mesh
        m,s=eqx.nn.make_with_state(MetaModel)(cfg,key=key);s=s.set(m.step_index,jnp.asarray(1000,dtype=jnp.int32))
        ps,ss=BlockCollectionSplit.split_state(s.substate(m.language_model.model.h.blocks),cfg.model.suffix_len)
        coll=BlockCollectionSplit(cfg.model,block_collection=m.language_model.model.h.blocks,prime_storage=m.language_model.model.h.prime_storage,key=jax.random.PRNGKey(0))
        m=eqx.tree_at(lambda x:x.language_model.model.h,m,coll)
        self.cfg=cfg;self.base=m;self.state_all=s;self.state_prefix=ps;self.state_suffix=ss;self.eos=int(cfg.model.eos_token_id);self.bos=int(cfg.model.bos_token_id)
        assert int(cfg.model.sliding_window_size)==8192 and int(cfg.model.mini_batch_size)==CH
        state_prefix=self.state_prefix
        @eqx.filter_jit
        def prefix(m,seq):return m.language_model.prefix_call(m.language_model.model.h.prefix_blocks,m.language_model.wte_call(seq.input_ids),clone_state(state_prefix),seq).last_hidden_state
        @eqx.filter_jit
        def chunk_logp(m,st,seq,pref):
            result=m.language_model.suffix_call(prefix_outputs=pref,state=st,seq=seq);lp=jax.nn.log_softmax(result.logits.astype(jnp.float32),axis=-1)
            return result.new_state,jnp.take_along_axis(lp,seq.target_tokens[:,None],axis=-1)[:,0]
        self.prefix=prefix;self.chunk_logp=chunk_logp
    def seq(self,ids,mask=None):
        ids=jnp.asarray(ids,dtype=jnp.int32);targets=jnp.concatenate([ids[1:],jnp.asarray([self.eos],dtype=jnp.int32)])
        if mask is None:mask=jnp.ones(len(ids),dtype=bool)
        return Batch(input_ids=ids,target_tokens=targets,loss_masks=jnp.asarray(mask,dtype=bool))
    def chunk(self,seq,k):
        s=slice(k*CH,(k+1)*CH);return Batch(input_ids=seq.input_ids[s],target_tokens=seq.target_tokens[s],loss_masks=seq.loss_masks[s])
def install_stubs(rt):
    """register stand-ins for ladder / suppress / margins_split dependencies so that `import s3` works; returns the s3 module"""
    sys.path.insert(0,str(HERE))
    reuse=types.SimpleNamespace(_hash_pytree=hash_pytree,get_filter_spec=get_filter_spec)
    ladder=types.ModuleType('ladder')
    class Ladder:pass
    ladder.__dict__.update(dict(jax=jax,jnp=jnp,eqx=eqx,np=np,optax=optax,MetaModel=MetaModel,CH=CH,clone_state=clone_state,reuse=reuse,Ladder=Ladder,ROOT=ROOT,OUT=Path('/tmp/e55_cpu_out'),gw=types.SimpleNamespace(),Batch=Batch))
    sys.modules['ladder']=ladder;sys.modules['suppress']=types.ModuleType('suppress')
    sys.path.insert(0,str(ROOT/'scripts/exp04_what_ttt_retains'))
    import s3;return s3,reuse
