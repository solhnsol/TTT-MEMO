"""E57 public QA stage 23 (new file; GPU; QA part of job 1143, pair-shuffle control of the rule arms, Hans 11/06 '1번 2번 먼저 하자'): the PAIR-SHUFFLED 65-version QA episodes of the tune bundles 8 / 10 / 12 (data and stream of job 940, e57_public_qa_n65_pairshuffle) with the rule arms ABS2 w=16 (g128 > 2 nats) and the fixed top 5% rows w=16 (ALLQ128 on every row, update norm = the plain NAT write of this run), the masks computed from the g128 of the SHUFFLED stream (e57_dwg_qa_nll.npz of job 940 through e57_rule_masks_real.py --qa-nll). ALLQ128 and M3 w=8 on the shuffled stream are NOT rerun (job 940). Read point n = 65, scored with the original pairs.
  python3 e57_pqa23_run.py --stage write|read --out O ... | --selftest"""
import sys,pathlib
HERE=pathlib.Path(__file__).resolve().parent;sys.path.insert(0,str(HERE))
import e57_pqa2_run as P2
import e57_pqa15_run as P15
import e57_pqa19_run as P19
NR=[65];ARMS=['M3qtA2W16Q128','M3qt05W16Q128'];ROOT=HERE.parents[1]
def write_stage(a):
    P19.ARMS_W=['NAT']+list(ARMS);P19.MASK_NPZ=str(ROOT/'experiments/exp07_write_ladder/e57_public_qa_n65_pairshuffle/qa_rules_masks_shuf.npz');P19.write_stage(a)
def read_stage(a):
    P15.NREAD=list(NR);P15.ARMS=list(ARMS);P15.read_stage(dict(a,skip_repro=True))
def selftest():
    import re
    for arm in ARMS:assert re.match(r'^M3(q[A-Za-z0-9_]*?)W(\d+)Q128$',arm)
    P19.selftest();print('e57_pqa23_run selftest ok')
def main():
    if '--selftest' in sys.argv:selftest();return
    stage=P2._pop('--stage');g=lambda k:P2._pop(k)
    a=dict(out=g('--out'),src640=g('--src640'),items=g('--items-file'),bank=g('--bank'),assign=g('--assign'),stage1=g('--stage1'),json=g('--json'),max_groups=None)
    {'write':write_stage,'read':read_stage}[stage](a)
if __name__=='__main__':main()
