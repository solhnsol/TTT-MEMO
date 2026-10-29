"""E57 T1/T2 scoring (e57_t1t2_run.py, job 276 / 277 method) for the written states of the query-band arms (job 410): the arms GAINQ512P26 (GAINQ512 p=.26, job 292b), GAINQ512P10 (p=.10, job 293), RANDQ512P10 (RANDQ512 p=.10, job 297) are registered as arm names of e57_main_run and the unchanged t1t2 reader is run (OUT reads of the frozen t1t2_v1.json at W0 and at the given states, manifest with state file + hash). No new code path: only the arm names are added before e57_t1t2_run.main().
  python3 e57_t1t2_qstates_run.py <the arguments of e57_t1t2_run.py with --arms GAINQ512P26,GAINQ512P10,RANDQ512P10>   |   --selftest"""
import sys,pathlib
HERE=pathlib.Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
NAMES=('GAINQ512P26','GAINQ512P10','RANDQ512P10')
def main():
    import e57_main_run as M
    M.ARMS_ALL=tuple(M.ARMS_ALL)+tuple(n for n in NAMES if n not in M.ARMS_ALL)
    import e57_t1t2_run as T
    T.main()
if __name__=='__main__':
    if '--selftest' in sys.argv:
        import e57_main_run as M;M.ARMS_ALL=tuple(M.ARMS_ALL)+NAMES;assert all(n in M.ARMS_ALL for n in NAMES);import e57_t1t2_run as T;T.selftest();print('e57_t1t2_qstates_run selftest ok')
    else:main()
