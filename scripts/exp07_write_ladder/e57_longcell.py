"""E57 synthetic world, long streams (new file; import for its side effect): registers the cell v3_16_1_L1024 in gen_facts.CELLS / CELL_BOOKS at run time (gen_facts.py itself is not edited): 16 facts, one sentence per fact in every chunk (p = 1), n_app = 1024 exposures per fact = 1024 chunks, the bank of 300 frames (cycled about 3.4 times per fact), read apps (16, 32, 64, 128), the 24-book background pool of the cell v3_64_4 (the 4-book pool of v3_16_1 is exhausted at chunk 370)."""
import sys,pathlib
HERE=pathlib.Path(__file__).resolve().parent;ROOT=HERE.parents[1]
for p in (HERE,ROOT/'scripts/exp04_what_ttt_retains'):sys.path.insert(0,str(p))
import gen_facts as G
CELL='v3_16_1_L1024'
G.CELLS[CELL]=dict(k=16,n=1,p=1,bank='300',n_app=1024,read_apps=(16,32,64,128));G.CELL_BOOKS[CELL]=G.CELL_BOOKS['v3_64_4']
