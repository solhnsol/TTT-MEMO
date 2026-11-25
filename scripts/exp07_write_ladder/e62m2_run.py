"""E62 MAIN, SECOND BATCH of 12 new novels (job 1346b; wrapper of the UNCHANGED e62m_run.py; new file; thread-server 11/25; Hans 'E62 책 추가': the same design, arms, reads and storage as the main run on 12 novels that do not overlap the first 12). The 12 new units are PG-19 TRAIN novels (the test split has no further novel of >= 220 chunks): In Spite of All (Edna Lyall, 55324), The Clever Woman of the Family (Yonge, 3292), Mrs. Halliburton's Troubles (Mrs Henry Wood, 34587), Lord Montagu's Page (G. P. R. James, 40295), Madam (Oliphant, 55125), Anne (Woolson, 32707), All Sorts and Conditions of Men (Besant and Rice, 47098), The Master (Zangwill, 49795), The Cryptogram (De Mille, 28435), Sir Jasper Carew (Lever, 33081), Horse-Shoe Robinson (Kennedy, 33478), Magnum Bonum (Yonge, 5080); first 180 chunks each; possibly seen in the book fine-tuning of the checkpoint (as the 5 train novels of the first batch). The only change besides the units is the default seed 1 (invented names / answers and plant schedules differ from the first batch, which used seed 0); the leak check of the invented words runs over the 100 test books + the 17 train books.
  python3 e62m2_run.py --out O [--books N] [--max-chunks N] [--arms NAT,SAME] [--seed 1] | --selftest"""
import sys,pathlib,argparse
HERE=pathlib.Path(__file__).resolve().parent;sys.path.insert(0,str(HERE))
import e62m_run as M
NEW=('55324','3292','34587','40295','55125','32707','47098','49795','28435','33081','33478','5080')
M.UNITS=[(f'pg19tr_{i}',0) for i in NEW];M.TRAIN_IDS=tuple(M.TRAIN_IDS)+NEW
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--out');ap.add_argument('--books',type=int,default=12);ap.add_argument('--arms',default='NAT,SAME');ap.add_argument('--max-chunks',type=int,default=0);ap.add_argument('--seed',type=int,default=1);ap.add_argument('--selftest',action='store_true');a=ap.parse_args()
    M.selftest(a.seed) if a.selftest else M.run(a)
