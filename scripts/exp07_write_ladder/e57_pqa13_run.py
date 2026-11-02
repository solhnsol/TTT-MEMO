"""E57 public QA stage 13 (E57 122-AC / 123; new file; read only, NO weight update): per-token re-read of the end states of the paragraph-group episodes. The n = 9 states (NATQ and ALLQ128 of job 640, GAINQ512 p.10 of job 650) and W0 are read again on the items455 with the ORIGINAL 16 candidates and the conditions A (own question), B (question of another item, E57_pqa4_assignB.json) and C ("Answer:" only), exactly as jobs 640 / 650 (e57_pqa6_run.FastOUTG packing per group; the summed log-probs reproduce the stored rows, the maximum difference is reported), but the log-prob of every candidate TOKEN is stored (cand_tok_lp) with the token ids (e57_pqa13_tokens.json).
Metrics (branch-token ranks, Hans 11/02: once a token of the answer is revealed the following tokens depend on it): per item and condition (A / B / C) (a) branch rank: the gold against each of the 15 distractors, both scored from their first token up to and including the first token at which the two token sequences differ (if one is a prefix of the other: up to the end of the shorter one); rank = 1 + number of distractors that beat the gold at their own branch point, rank ratio = rank / 16, top-1 = rank 1, mean gold-minus-distractor difference at the branch; (b) first-token rank (the same with the first token only); (c) full-sum rank (as stored in jobs 640 / 650); and for the binding: gain = (X_A(state) - X_A(W0)) - (X_B(state) - X_B(W0)) for X = gold first-token log-prob, gold full-sum log-prob, and the rank-ratio improvement of the branch / first-token / full ranks (W0 minus state). Items455 and main373, bootstrap 95% CI over items, arm differences as paired differences per item (ALLQ128 - NATQ, GAINQ512 - NATQ, GAINQ512 - ALLQ128).
  python3 e57_pqa13_run.py --stage read --out O --src640 D --src650 D --items-file F --bank B --assign A --stage1 S1 [--max-groups N] | --stage analyze --out O --src640 D | --selftest"""
import sys,json,time,hashlib,pathlib
import numpy as np
HERE=pathlib.Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
import e57_pqa2_run as P2
import e57_pqa4_run as P4
import e57_pqa6_run as P6
import e57_pqa12_run as P12
ARMS=[('NATQ','640'),('ALLQ128','640'),('GAINQ512P10','650')];CONDS=('A','B','C')
def read_stage(a):
    out=pathlib.Path(a['out']);out.mkdir(parents=True,exist_ok=True);lg=lambda *x:print(*x,flush=True);meta,_=P6.load_groups(a['src640']);W={'640':json.load(open(pathlib.Path(a['src640'])/'e57_pqa6_writes.json')),'650':json.load(open(pathlib.Path(a['src650'])/'e57_pqa7_writes.json'))}
    G=meta['groups'][:int(a['max_groups'])] if a.get('max_groups') else meta['groups'];LD,rt,rt32,rd,items,h,raw0,ntok0,out_tail,tsha=P2.setup_read(a,lg);import e57_reads as RD
    byid={it['id']:it for it in items};asg=json.load(open(a['assign']))['assignment'];q={it['id']:it['prefix'] for it in items};grows=[];gnt=[];gsizes=[];tokids={c:{} for c in CONDS}
    for g in G:
        its=[byid[i] for i in g['item_ids']];rows=[];nts=[];sz=[]
        for c in CONDS:
            pre=[q[it['id']] if c=='A' else (q[asg[it['id']]] if c=='B' else 'Answer:') for it in its];it2=[dict(it,prefix=p) for it,p in zip(its,pre)];r_,n_,bad=RD.build_rows(rt.tok,it2);assert not bad,bad
            for k,it in enumerate(its):tokids[c][it['id']]=[[int(x) for x in r_[k*16+j][0][r_[k*16+j][1][0]+1:r_[k*16+j][1][1]+1]] for j in range(16)]
            rows+=r_;nts.append(n_);sz.append(len(r_))
        grows.append(rows);gnt.append(nts);gsizes.append(sz)
    (out/'e57_pqa13_tokens.json').write_text(json.dumps(tokids))
    m32w0=rd.to_fp32(rt.base);fo=P12.FastOUTG3(rt32,m32w0,out_tail,grows,log=lg);labels=['W0']+[f'{arm}@n9' for arm,_ in ARMS];R={c:{l:[] for l in labels} for c in CONDS}
    def put(label,gi,m32):
        toks=fo.score_group_tokens(m32,gi);its=[byid[i] for i in G[gi]['item_ids']];o=0
        for ci,c in enumerate(CONDS):
            tc=toks[o:o+gsizes[gi][ci]];o+=gsizes[gi][ci];rows=P2.rows_of(its,gnt[gi][ci],np.asarray([t.sum() for t in tc]),f'{c}:{label}')
            for k,r in enumerate(rows):r['cand_tok_lp']=[[float(x) for x in t] for t in tc[k*16:(k+1)*16]]
            R[c][label]+=rows
    t00=time.monotonic()
    for gi,g in enumerate(G):
        t0=time.monotonic();put('W0',gi,m32w0)
        for arm,job in ARMS:
            s=W[job][str(g['gid'])][arm]['states'][str(g['read_chunks']['9'])];m=rd.load_state(s['file'],expect_hash=s['hash']);put(f'{arm}@n9',gi,rd.to_fp32(m));del m
        lg(f'  group {g["gid"]} read: {time.monotonic()-t0:.0f}s (total {time.monotonic()-t00:.0f}s)')
    src=pathlib.Path(a['src640']);chk={}
    for c in CONDS:                                                       # reproduction of the stored sums
        for l,f in (('W0',f'E57_pqa6_rows_{c}_W0.json'),('NATQ@n9',f'E57_pqa6_rows_{c}_NATQ_n9.json'),('ALLQ128@n9',f'E57_pqa6_rows_{c}_ALLQ128_n9.json')):
            ref=json.load(open(src/f))[:len(R[c][l])];chk[f'{c}:{l}']=float(max(np.abs(np.asarray(x['cand_lp'])-np.asarray(y['cand_lp'])).max() for x,y in zip(R[c][l],ref)))
    lg('reproduction of the stored sums, max|diff|:',max(chk.values()))
    for c in CONDS:
        for l in labels:(out/f'E57_pqa13_rows_{c}_{l.replace("@","_")}.json').write_text(json.dumps(R[c][l],default=float))
    (out/'e57_pqa13_read_meta.json').write_text(json.dumps(dict(items_sha256=h,out_tail_sha=tsha,item_ids=[i for g in G for i in g['item_ids']],reproduction_max_abs_diff=chk,labels=labels)))
def branch_metrics(rows,ids,tok):
    """per item arrays for one condition: rank / rank ratio / top-1 / mean gold-minus-distractor at the branch point, at the first token, and for the full sum; plus the gold first-token and full log-prob"""
    n=len(ids);o={k:np.zeros(n) for k in ('rank_branch','top1_branch','gap_branch','rank_first','top1_first','gap_first','rank_full','top1_full','gap_full','gold_first','gold_full')}
    for k,(i,r) in enumerate(zip(ids,rows)):
        tl=[np.asarray(x,np.float64) for x in r['cand_tok_lp']];ti=tok[i];g=tl[0];nb=nf=nu=0;db=[];df=[];du=[]
        for c in range(1,len(tl)):
            tg,tc=ti[0],ti[c];t=next((q for q in range(min(len(tg),len(tc))) if tg[q]!=tc[q]),min(len(tg),len(tc))-1);sg=g[:t+1].sum();sc=tl[c][:t+1].sum();nb+=int(sc>sg);db.append(sg-sc)
            nf+=int(tl[c][0]>g[0]);df.append(g[0]-tl[c][0]);nu+=int(tl[c].sum()>g.sum());du.append(g.sum()-tl[c].sum())
        o['rank_branch'][k]=1+nb;o['top1_branch'][k]=float(nb==0);o['gap_branch'][k]=np.mean(db);o['rank_first'][k]=1+nf;o['top1_first'][k]=float(nf==0);o['gap_first'][k]=np.mean(df);o['rank_full'][k]=1+nu;o['top1_full'][k]=float(nu==0);o['gap_full'][k]=np.mean(du);o['gold_first'][k]=g[0];o['gold_full'][k]=g.sum()
    for w in ('branch','first','full'):o[f'rr_{w}']=o[f'rank_{w}']/16.0
    return o
def analyze(a):
    out=pathlib.Path(a['out']);rm=json.load(open(out/'e57_pqa13_read_meta.json'));ids=rm['item_ids'];meta,_=P6.load_groups(a['src640']);main={i for g in meta['groups'] for i in g['main_ids']};tokall=json.load(open(out/'e57_pqa13_tokens.json'))
    masks={'items455':np.ones(len(ids),bool),'main373':np.asarray([i in main for i in ids])};ld=lambda c,l:json.load(open(out/f'E57_pqa13_rows_{c}_{l.replace("@","_")}.json'));labels=rm['labels']
    M={c:{l:branch_metrics(ld(c,l),ids,tokall[c]) for l in labels} for c in CONDS};res={'states':{},'arm_differences':{},'reproduction_max_abs_diff':rm['reproduction_max_abs_diff']}
    def gains(m_,w):
        g={}
        for k in ('gold_first','gold_full'):g[f'gain_{k}']=(m_['A'][k]-w['A'][k])-(m_['B'][k]-w['B'][k])
        for k in ('rr_branch','rr_first','rr_full'):g[f'gain_{k}_improvement']=(w['A'][k]-m_['A'][k])-(w['B'][k]-m_['B'][k])
        for k in ('top1_branch','top1_first','top1_full'):g[f'gain_{k}']=(m_['A'][k]-w['A'][k])-(m_['B'][k]-w['B'][k])
        return g
    for arm,_ in ARMS:
        l=f'{arm}@n9';w={c:M[c]['W0'] for c in CONDS};s={c:M[c][l] for c in CONDS};g=gains(s,w);e={}
        for sn,mk in masks.items():
            e[sn]=dict(gains={k:P4.boot(v[mk]) for k,v in g.items()},**{c:{k:P4.boot(s[c][k][mk]) for k in ('rr_branch','rr_first','rr_full','top1_branch','top1_first','top1_full','gap_branch','gap_first','gap_full','gold_first','gold_full')} for c in CONDS},
                W0={c:{k:P4.boot(w[c][k][mk]) for k in ('rr_branch','rr_first','rr_full','top1_branch','top1_first','top1_full','gap_branch','gap_first','gap_full','gold_first','gold_full')} for c in CONDS})
        res['states'][l]=e
    for a1,a2 in (('ALLQ128','NATQ'),('GAINQ512P10','NATQ'),('GAINQ512P10','ALLQ128')):
        w={c:M[c]['W0'] for c in CONDS};g1=gains({c:M[c][f'{a1}@n9'] for c in CONDS},w);g2=gains({c:M[c][f'{a2}@n9'] for c in CONDS},w);e={}
        for sn,mk in masks.items():e[sn]={k:P4.boot((g1[k]-g2[k])[mk]) for k in g1}
        res['arm_differences'][f'{a1}-{a2}@n9']=e
    (out/'E57_pqa13_keys.json').write_text(json.dumps(dict(meta=dict(definition=__doc__.split('Metrics')[1][:1800],n_items=len(ids),subsets={k:int(v.sum()) for k,v in masks.items()}),**res),indent=1,default=float))
def selftest():
    P12.selftest()
    rows=[dict(cand_tok_lp=[[-1.,-2.]]+[[-2.,-2.]]*15),dict(cand_tok_lp=[[-1.,-1.]]+[[-3.,-1.]]*15)];tok={'a':[[1,2]]+[[4,5]]*15,'b':[[7,8]]+[[7,9]]*15}
    m=branch_metrics(rows,['a','b'],tok);assert m['rank_branch'].tolist()==[1,1] and m['rank_first'].tolist()==[1,1] and m['rank_full'].tolist()==[1,1] and abs(m['gap_branch'][1]-2.0)<1e-9
    rows[1]['cand_tok_lp'][1]=[-1.,-9.];m=branch_metrics(rows,['a','b'],tok);assert m['rank_first'][1]==1+0 and m['rank_branch'][1]==1+0
    print('e57_pqa13_run selftest ok')
def main():
    if '--selftest' in sys.argv:selftest();return
    stage=P2._pop('--stage');g=lambda k:P2._pop(k);a=dict(out=g('--out'),src640=g('--src640'),src650=g('--src650'),items=g('--items-file'),bank=g('--bank'),assign=g('--assign'),stage1=g('--stage1'),max_groups=g('--max-groups'))
    {'read':read_stage,'analyze':analyze}[stage](a)
if __name__=='__main__':main()
