"""Position of each exposure's most recent previous exposure of the same fact (same triple = same name-place binding; the same pair also reported) in the fs streams (native fs and masked fs read the SAME streams):
(a) same chunk, (b) a different chunk but inside the 8192-token window before the current chunk, (c) outside the window or no previous exposure (shown separately). Position = the place token of the sentence. CPU, no D.
Passes: pass 1 = the original stream; pass p >= 2 = same sentences, new background (ladder.fs_streams: background pool random.Random(7000+p)). Usage: count_prior_exposure.py"""
import sys,os,copy,random,json;sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)));sys.path.insert(0,os.path.join(os.path.dirname(os.path.abspath(__file__)),'../exp04_what_ttt_retains'))
import numpy as np
import gen_world as gw,suppress as SP,s3_sets as S3
CH=S3.CH;W=S3.W;res={}
def world_pass(w0,ps):
    if ps==1:return w0
    w=copy.deepcopy(w0);pool=gw.background(random.Random(7000+ps));bi=0;new=[]
    for t in w['texts']:
        if t['kind']=='bg':new.append({'kind':'bg','text':pool[bi%len(pool)]});bi+=1
        else:new.append(t)
    w['texts']=new;return w
for sd in (101,102,103):
    w0=gw.build_world(sd)
    for swap in (False,True):
        for ps in (1,2,8):
            tab=S3.sentence_table(world_pass(w0,ps),swap,gw.TOK,gw,SP);cnt={'triple':[0,0,0,0],'pair':[0,0,0,0]}   # same chunk, window, outside window, none
            for key in ('triple','pair'):
                for s in tab:
                    k=s['place'][0]//CH;k0=k*CH
                    prev=[u for u in tab if (u['t']==s['t'] if key=='triple' else u['pair']==s['pair']) and u['place'][0]<s['place'][0]]
                    if not prev:cnt[key][3]+=1;continue
                    p=max(u['place'][0] for u in prev)
                    if p//CH==k:cnt[key][0]+=1
                    elif p>=k0-W:cnt[key][1]+=1
                    else:cnt[key][2]+=1
            res[f'{sd}{"B" if swap else "A"}|pass{ps}']=cnt
            n=len(tab);t=cnt['triple'];q=cnt['pair']
            print(f"seed {sd} world {'B' if swap else 'A'} pass {ps}: {n} exposures; same fact (triple): same chunk {t[0]} ({t[0]/n:.1%}), other chunk inside 8K {t[1]} ({t[1]/n:.1%}), outside 8K {t[2]} ({t[2]/n:.1%}), no previous {t[3]} ({t[3]/n:.1%}) | same pair: {q[0]}/{q[1]}/{q[2]}/{q[3]}")
json.dump(res,open(os.path.join(os.path.dirname(os.path.abspath(__file__)),'../../experiments/exp07_write_ladder/prior_exposure_counts.json'),'w'),indent=1)
