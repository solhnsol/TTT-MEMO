"""E57 T1/T2 scoring (e57_t1t2_run.py, job 276 / 277 / 410 method) for further written query-band states (job 450): the arms GAINQ128P26 (GAINQ128 p=.26), GAINQ128P10 (p=.10), GAINQ256P26 (GAINQ256 p=.26) and RANDQ512P10 (re-read, as in job 410) are registered as arm names of e57_main_run and the unchanged t1t2 reader is run (manifest with state file + hash). Same wrapper as e57_t1t2_qstates_run.py (job 410), only the arm names differ.
  python3 e57_t1t2_qstates2_run.py <the arguments of e57_t1t2_run.py with --arms GAINQ128P26,GAINQ128P10,GAINQ256P26,RANDQ512P10>   |   --selftest"""
import sys,pathlib
HERE=pathlib.Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
NAMES=('GAINQ128P26','GAINQ128P10','GAINQ256P26','RANDQ512P10')
def main():
    import e57_main_run as M
    M.ARMS_ALL=tuple(M.ARMS_ALL)+tuple(n for n in NAMES if n not in M.ARMS_ALL)
    import e57_t1t2_run as T
    T.main()
if __name__=='__main__':
    if '--selftest' in sys.argv:
        import e57_main_run as M;M.ARMS_ALL=tuple(M.ARMS_ALL)+tuple(n for n in NAMES if n not in M.ARMS_ALL);assert all(n in M.ARMS_ALL for n in NAMES);import e57_t1t2_run as T;T.selftest();print('e57_t1t2_qstates2_run selftest ok')
    else:main()
