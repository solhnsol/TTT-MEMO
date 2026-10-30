"""E57 T1/T2 scoring (e57_t1t2_run.py, job 410 / 450 / 540 / 550 method) for the chunk-279 states of the GAINQ512 p=.10 focal-loss arms (job 565): arm names registered as arm names of e57_main_run, the unchanged t1t2 reader is run (manifest with state file + hash).
  python3 e57_t1t2_focal_run.py <the arguments of e57_t1t2_run.py with --arms GAINQ_FOCAL_NONE,...>   |   --selftest"""
import sys,pathlib
HERE=pathlib.Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
NAMES=('GAINQ_FOCAL_NONE','GAINQ_FOCAL_T05','GAINQ_FOCAL_T10','GAINQ_FOCAL_RAND')
def main():
    import e57_main_run as M
    M.ARMS_ALL=tuple(M.ARMS_ALL)+tuple(n for n in NAMES if n not in M.ARMS_ALL)
    import e57_t1t2_run as T
    T.main()
if __name__=='__main__':
    if '--selftest' in sys.argv:
        import e57_main_run as M;M.ARMS_ALL=tuple(M.ARMS_ALL)+tuple(n for n in NAMES if n not in M.ARMS_ALL);assert all(n in M.ARMS_ALL for n in NAMES);import e57_t1t2_run as T;T.selftest();print('e57_t1t2_focal_run selftest ok')
    else:main()
