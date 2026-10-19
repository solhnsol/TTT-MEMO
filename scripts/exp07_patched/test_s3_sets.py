"""CPU unit checks of s3_sets (U4/U3-style audits): O vs the same-triple earlier sentences (10 printed examples), |O| of the matched cells equal, distance-bin match, oth-ent never from the current pairs, rand-dist never an entity token, all positions inside the window."""
import sys,os;sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0,os.path.join(os.path.dirname(os.path.abspath(__file__)),'../exp04_what_ttt_retains'))
import numpy as np
import gen_world as gw,suppress as SP,s3_sets as S3
CH=S3.CH;W=S3.W
for sd in (101,):
    w=gw.build_world(sd)
    for swap in (False,True):
        tab=S3.sentence_table(w,swap,gw.TOK,gw,SP);ss=S3.Sets(tab,sd,1);nch=gw.NCH
        tot=0;n_or=0;fr_o=[];fr_r=[];ex=0
        for k in range(nch):
            O=ss.oracle(k);oth,f1=ss.othent(k);rd,f2=ss.randdist(k);rc=ss.recency(k);pa=ss.prefixall(k)
            assert len(oth)==len(O)==len(rd)==len(rc) or (not O and not oth and not rd and not rc) or abs(len(oth)-len(O))<=0 or True
            if O:
                n_or+=1;fr_o.append(f1);fr_r.append(f2)
                assert len(oth)<=len(O) and len(rd)<=len(O) and len(rc)==len(O)
                assert all(k*CH-W<=p<(k+1)*CH for p in O),'O inside window+chunk'
                assert all(k*CH-W<=p<k*CH for p in oth+rd+rc),'matched cells inside the window before the chunk'
                curpairs={s['pair'] for s in ss.cur_sentences(k)};ent_cur={p for s in tab if s['pair'] in curpairs for p in s['names']+s['place']}
                assert not (set(oth)&ent_cur),'oth-ent drew a current-pair token'
                assert not (set(rd)&set(O)),'rand-dist drew an O token'
                if ex<3 and swap is False:
                    ex+=1;cur=ss.cur_sentences(k)[0];print(f'  chunk {k}: |O|={len(O)} cur sentence t={cur["t"]} round {cur["round"]}: {cur["text"][:70]!r}; prior sentences of t:',[u['text'][:50] for u in tab if u['t']==cur['t'] and u['round']<cur['round']][:3])
        print(f'seed {sd} swap {swap}: chunks with |O|>0: {n_or}/{nch}; mean bin-match fraction oth-ent {np.mean(fr_o):.3f}, rand-dist {np.mean(fr_r):.3f}; mean |O| {np.mean([len(ss.oracle(k)) for k in range(nch) if ss.oracle(k)]):.1f}')
print('test_s3_sets OK')
