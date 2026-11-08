"""E57 public QA stage 24 (new file; GPU; QA part of job 1230, Hans 11/08 'g′ + 뉴스 온라인 g 팔' (copy of e57_pqa24_run.py)): the QA 65-version episodes of the five bundles (stream, groups, band-128 file of job 930) with the ONLINE g' ABS2 rule arm M3OS2W16Q128 (g' = nll_suffixband128 - nll_win from the current fast weights at write time, g > 2 nats among the eligible rows, loss weight 16, ALLQ128 on every row; e57_online_uv.py); plain NAT write for the update norms in the same run; read point n = 65 (A, B, BUN rows of jobs 930 / 1080). ALLQ128 / M3 w=8 (job 930) and W0-ABS2 / top 5% (job 1080) are not rerun.
  python3 e57_pqa24_run.py --stage write|read --out O ... | --selftest"""
import sys,pathlib
HERE=pathlib.Path(__file__).resolve().parent;sys.path.insert(0,str(HERE))
import e57_pqa2_run as P2
import e57_pqa15_run as P15
import e57_pqa19_run as P19
NR=[65];ARMS=['M3OS2W16Q128']
def write_stage(a):
    P19.ARMS_W=['NAT']+list(ARMS);P19.write_stage(a)
def read_stage(a):
    P15.NREAD=list(NR);P15.ARMS=list(ARMS);P15.read_stage(dict(a,skip_repro=True))
def selftest():
    import e57_online_uv;P19.selftest();print('e57_pqa25_run selftest ok')
def main():
    if '--selftest' in sys.argv:selftest();return
    stage=P2._pop('--stage');g=lambda k:P2._pop(k)
    a=dict(out=g('--out'),src640=g('--src640'),items=g('--items-file'),bank=g('--bank'),assign=g('--assign'),stage1=g('--stage1'),json=g('--json'),max_groups=None)
    {'write':write_stage,'read':read_stage}[stage](a)
if __name__=='__main__':main()
