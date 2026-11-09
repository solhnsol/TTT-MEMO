"""E57 public QA stage 22 (new file; GPU; QA part of job 1250/1251, Hans 11/08 '응 돌려보자' (copy of e57_pqa26_run.py): combined-score arms): the QA 65-version episodes of the five bundles with the MASS-MATCHED top-share arms (top 10% rows by the W0 g128 with weight 8 and top 30% with weight 3.5, extra loss mass 0.7 / 0.75; ALLQ128 on every row, update norm = the plain NAT write of this run) from the external mask file e57_public_qa_n65/qa_combined_masks.npz (e57_rule_masks_real.py). Uses the stream, group files and band-128 window-gain file of job 930 (copied into the new directory); ALLQ128 and M3 w=8 are NOT rerun (jobs 930 values are reused by the scoring script e57_split_score.py). Read points n = 9 / 33 / 65.
  python3 e57_pqa21_run.py --stage write|read|analyze --out O ... | --selftest"""
import sys,os,json,pathlib
HERE=pathlib.Path(__file__).resolve().parent;sys.path.insert(0,str(HERE))
import e57_pqa2_run as P2
import e57_pqa15_run as P15
import e57_pqa19_run as P19
NR=[65];ARMS=['M3qcmA05W16Q128','M3qcmA10W8Q128','M3qcmB05W16Q128','M3qcmB10W8Q128'];ROOT=HERE.parents[1]
def write_stage(a):
    P19.ARMS_W=['NAT']+list(ARMS);P19.MASK_NPZ=os.environ.get('QA_COMB_MASKS') or str(ROOT/'experiments/exp07_write_ladder/e57_public_qa_n65/qa_combined_masks.npz');P19.write_stage(a)
def read_stage(a):
    P15.NREAD=list(NR);P15.ARMS=list(ARMS);P15.read_stage(dict(a,skip_repro=True))
def analyze(a):
    P15.NREAD=list(NR);P15.ARMS=list(ARMS);P15.analyze(a)
def selftest():
    import re
    for arm in ARMS:assert re.match(r'^M3(q[A-Za-z0-9_]*?)W(\d+(?:p\d+)?)Q128$',arm)
    P19.selftest();print('e57_pqa27_run selftest ok')
def main():
    if '--selftest' in sys.argv:selftest();return
    stage=P2._pop('--stage');g=lambda k:P2._pop(k)
    a=dict(out=g('--out'),src640=g('--src640'),items=g('--items-file'),bank=g('--bank'),assign=g('--assign'),stage1=g('--stage1'),json=g('--json'),max_groups=None)
    {'write':write_stage,'read':read_stage,'analyze':analyze}[stage](a)
if __name__=='__main__':main()
