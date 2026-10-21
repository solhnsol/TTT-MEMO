"""Smoke / timing launcher of the OFFICIAL outer-loop training (ttt.train) on the 125M checkpoint, optionally with a random loss mask in the data path (no change to e2e/).
Usage: E2E_ROOT=<repo>/e2e MASK_FRAC=0.25 python3 outer_smoke.py <hydra overrides as for `train`>   (cwd is switched to E2E_ROOT: hydra reads configs/ relative to it).
MASK_FRAC > 0: lm_dataset._to_batch is replaced by outer_mask.make_patched_to_batch (loss_masks = default AND a deterministic random mask of that fraction): the removed positions leave the inner write loss and the outer loss of every chunk (outer_mask.py header). MASK_SEED (default 0).
With the dummy dataset (training.dummy_dataset=true, random tokens 0..19) only timing / mechanics are measured, not learning."""
import os,sys
from pathlib import Path
HERE=Path(__file__).resolve().parent;E2E=Path(os.environ.get('E2E_ROOT',HERE.parents[1]/'e2e')).resolve()
sys.path.insert(0,str(E2E));sys.path.insert(0,str(HERE));os.chdir(E2E)
import outer_mask as OM
import ttt.dataloader.lm_dataset as LDS
frac=float(os.environ.get('MASK_FRAC','0'))
if frac>0:
    LDS._to_batch=OM.make_patched_to_batch(LDS._to_batch,frac,int(os.environ.get('MASK_SEED','0')));print(f'outer_smoke: loss mask fraction {frac} active (random, seed {os.environ.get("MASK_SEED","0")})',flush=True)
else:print('outer_smoke: no extra loss mask (official data path)',flush=True)
import ttt.train as T
if __name__=='__main__':T.main()
