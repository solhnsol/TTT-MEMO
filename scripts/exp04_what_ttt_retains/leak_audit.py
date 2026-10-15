"""CPU-only: which (seed,pair,snapshot) units of the ORIGINAL (v1) Stage-2 worlds had later-round exposure sentences
located before the probe chunk (i.e. inside the 8K window / extra exposures)."""
import sys,types
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
if 'core' not in sys.modules:
    core=types.ModuleType('core');core.CH=1024;core.clone_state=lambda x:x;core.jax=types.SimpleNamespace(tree_util=None);core.jnp=None;core.eqx=None;sys.modules['core']=core
import stage2,gen_world as gw
CH=1024
def leaky_pairs(seed,v):
    rt=types.SimpleNamespace(tok=gw.TOK,bos=128000,eos=128001);w=gw.build_world(seed,v=v);P=stage2.Probe(rt,w,False)
    tri={t['id']:t['pair'] for t in w['triples']};out={}
    for k,rounds in stage2.STAGES.items():
        start=(k+1)*CH;out[k]={tri[s['t']] for s in P.spans if s['round']>rounds and s['tok_start']<start}
    return out
