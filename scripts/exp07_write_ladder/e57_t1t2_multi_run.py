"""E57 T1/T2 scoring (e57_t1t2_run.py, job 276 / 277 / 410 / 450 / 540 method) for the chunk-279 states of the momentum-variant, slot-momentum and control arms (job 550): the arm names are registered as arm names of e57_main_run and the unchanged t1t2 reader is run (manifest with state file + hash). Same wrapper as e57_t1t2_qstates2_run.py, only the arm names differ.
  python3 e57_t1t2_multi_run.py <the arguments of e57_t1t2_run.py with --arms W_MULTI,...>   |   --selftest"""
import sys,pathlib
HERE=pathlib.Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
NAMES=('W_MULTI','W_CAUTIOUS','W_MUON','W_WHITEN_A','W_WHITEN_B','S_ORACLE','S_KMEANS','S_RANDOM','R2B09','NATADAMH','NAT')
def main():
    import e57_main_run as M
    M.ARMS_ALL=tuple(M.ARMS_ALL)+tuple(n for n in NAMES if n not in M.ARMS_ALL)
    import e57_t1t2_run as T
    T.main()
if __name__=='__main__':
    if '--selftest' in sys.argv:
        import e57_main_run as M;M.ARMS_ALL=tuple(M.ARMS_ALL)+tuple(n for n in NAMES if n not in M.ARMS_ALL);assert all(n in M.ARMS_ALL for n in NAMES);import e57_t1t2_run as T;T.selftest();print('e57_t1t2_multi_run selftest ok')
    else:main()
