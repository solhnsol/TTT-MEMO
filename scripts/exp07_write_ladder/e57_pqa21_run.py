"""E57 public QA stage 21 (new file; GPU; QA part of job 1080, Hans 11/05 'a b c 승인 / 진행시켜'): the QA 65-version episodes of the five bundles with the RULE arms of job 1070 (ABS-NAT tau = 2 / 4 / 6 nats on g128, weight 16, and the fixed top 5% rows, weight 16; ALLQ128 on every row, update norm = the plain NAT write of this run) from the external mask file e57_public_qa_n65/qa_rules_masks.npz (e57_rule_masks_real.py). Uses the stream, group files and band-128 window-gain file of job 930 (copied into the new directory); ALLQ128 and M3 w=8 are NOT rerun (jobs 930 values are reused by the scoring script e57_split_score.py). Read points n = 9 / 33 / 65.
  python3 e57_pqa21_run.py --stage write|read|analyze --out O ... | --selftest"""
import sys,json,pathlib
HERE=pathlib.Path(__file__).resolve().parent;sys.path.insert(0,str(HERE))
import e57_pqa2_run as P2
import e57_pqa15_run as P15
import e57_pqa19_run as P19
NR=[9,33,65];ARMS=['M3qtA2W16Q128','M3qtA4W16Q128','M3qtA6W16Q128','M3qt05W16Q128'];ROOT=HERE.parents[1]
def write_stage(a):
    P19.ARMS_W=['NAT']+list(ARMS);P19.MASK_NPZ=str(ROOT/'experiments/exp07_write_ladder/e57_public_qa_n65/qa_rules_masks.npz');P19.write_stage(a)
def read_stage(a):
    P15.NREAD=list(NR);P15.ARMS=list(ARMS);P15.read_stage(dict(a,skip_repro=True))
def analyze(a):
    P15.NREAD=list(NR);P15.ARMS=list(ARMS);P15.analyze(a)
def selftest():
    import re
    for arm in ARMS:assert re.match(r'^M3(q[A-Za-z0-9_]*?)W(\d+)Q128$',arm)
    P19.selftest();print('e57_pqa21_run selftest ok')
def main():
    if '--selftest' in sys.argv:selftest();return
    stage=P2._pop('--stage');g=lambda k:P2._pop(k)
    a=dict(out=g('--out'),src640=g('--src640'),items=g('--items-file'),bank=g('--bank'),assign=g('--assign'),stage1=g('--stage1'),json=g('--json'),max_groups=None)
    {'write':write_stage,'read':read_stage,'analyze':analyze}[stage](a)
if __name__=='__main__':main()
