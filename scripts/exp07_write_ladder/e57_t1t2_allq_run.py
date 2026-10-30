"""E57 T1/T2 scoring (e57_t1t2_run.py, job 410 method) of the chunk-279 states of the ALLQ512 / NATQ write run (job 560): the two arm names are registered in e57_main_run and the unchanged t1t2 reader is run. No new code path.
  python3 e57_t1t2_allq_run.py <the arguments of e57_t1t2_run.py with --arms ALLQ512,NATQ>   |   --selftest"""
import sys,pathlib
HERE=pathlib.Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
NAMES=('ALLQ512','NATQ')
def main():
    import e57_main_run as M
    M.ARMS_ALL=tuple(M.ARMS_ALL)+tuple(n for n in NAMES if n not in M.ARMS_ALL)
    import e57_t1t2_run as T
    T.main()
if __name__=='__main__':
    if '--selftest' in sys.argv:
        import e57_main_run as M;M.ARMS_ALL=tuple(M.ARMS_ALL)+NAMES;assert all(n in M.ARMS_ALL for n in NAMES);import e57_t1t2_run as T;T.selftest();print('e57_t1t2_allq_run selftest ok')
    else:main()
