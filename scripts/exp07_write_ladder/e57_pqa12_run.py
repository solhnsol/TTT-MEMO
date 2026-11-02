"""E57 public QA stage 12 (E57 122-AB / 123; new file; read only, NO weight update): bundle-matrix reads. The stored states of jobs 640 (NATQ, ALLQ128) and 650 (GAINQ512 p.10) at n = 1 / 3 / 5 / 9 versions (33 paragraph-group episodes) and W0 are read on the items455 with the candidates replaced: the candidates of the question of item i are the GOLD STRINGS of the other items of the same group (the answers of the other questions about the same written paragraphs), plus its own gold; a gold string of another item that equals the own gold or contains / is contained in it (case-insensitive substring) is left out, duplicate strings are listed once (so a question has up to group size - 1 distractors, not 15). Reader, prompt frame (own question), OUT context (stage-1 tail) and packing per group as jobs 640 / 650 (e57_pqa6_run.FastOUTG).
Per question i and state: rank of the own gold among its candidates (rank ratio = rank / number of candidates, top-1 = rank 1), mean gold-minus-candidate log-prob difference (summed log-prob; per-token mean as a second score), and the matrix version of the binding: A_i = score of the gold string of item i under its OWN question, B_bundle_i = mean score of the SAME string under the questions of the OTHER items of the group that have it as a candidate; dA, dB, d(A-B_bundle) against W0 per item; mean over items with the bootstrap 95% CI over items (items455 and main373), arm differences as paired differences per item. The number of occurrences of every candidate string in the written text of its group (first n versions, case-insensitive, between non-alphanumeric characters; n = 1, 3, 5, 9) is stored in e57_pqa12_counts.json, and the share with 9 occurrences at n = 9 (once per version) is reported.
  python3 e57_pqa12_run.py --stage read --out O --src640 D --src650 D --items-file F --bank B --stage1 S1 --para P --rewrites8 R8 [--max-groups N] | --stage analyze --out O --src640 D --src650 D | --selftest"""
import sys,json,time,hashlib,pathlib
import numpy as np
HERE=pathlib.Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
import e57_pqa2_run as P2
import e57_pqa4_run as P4
import e57_pqa6_run as P6
import e57_pqa8_run as P8
NREAD=P6.NREAD;ARMS=[('NATQ','640'),('ALLQ128','640'),('GAINQ512P10','650')]
def related(a,b):
    a=a.strip().lower();b=b.strip().lower();return a==b or a in b or b in a
def layout(item_ids,golds):
    """-> {item id i: [item ids j of the candidates, own first]} for the items of one group; golds: id -> gold string"""
    out={}
    for i in item_ids:
        cl=[i];seen={golds[i].strip().lower()}
        for j in item_ids:
            if j==i:continue
            g=golds[j].strip().lower()
            if g in seen or related(golds[i],golds[j]):continue
            seen.add(g);cl.append(j)
        out[i]=cl
    return out
def read_stage(a):
    out=pathlib.Path(a['out']);out.mkdir(parents=True,exist_ok=True);lg=lambda *x:print(*x,flush=True);meta,_=P6.load_groups(a['src640']);W={'640':json.load(open(pathlib.Path(a['src640'])/'e57_pqa6_writes.json')),'650':json.load(open(pathlib.Path(a['src650'])/'e57_pqa7_writes.json'))}
    G=meta['groups'][:int(a['max_groups'])] if a.get('max_groups') else meta['groups'];LD,rt,rt32,rd,items,h,raw0,ntok0,out_tail,tsha=P2.setup_read(a,lg);import e57_reads as RD
    byid={it['id']:it for it in items};golds={i:it['cands'][0] for i,it in byid.items()};para=json.load(open(a['para']));rew=json.load(open(a['rewrites8']));gt=P8.group_texts(para,rew,G);grows=[];gnt=[];gits=[];lay={};counts={}
    for g in G:
        lo=layout(g['item_ids'],golds);lay.update(lo);its=[];txt={n:'\n'.join(gt[g['gid']][:n]).lower() for n in NREAD}
        for i in g['item_ids']:
            its.append(dict(byid[i],cands=[golds[j] for j in lo[i]],kinds=['gold']+['distractor']*(len(lo[i])-1),all_idx=list(range(len(lo[i])))))
            counts[i]={f'n{n}':P8.count_in(txt[n],golds[i]) for n in NREAD}
        r_,n_,bad=RD.build_rows(rt.tok,its);assert not bad,bad;grows.append(r_);gnt.append(n_);gits.append(its)
    (out/'e57_pqa12_layout.json').write_text(json.dumps(lay));(out/'e57_pqa12_counts.json').write_text(json.dumps(counts))
    m32w0=rd.to_fp32(rt.base);fo=P6.FastOUTG(rt32,m32w0,out_tail,grows,log=lg);labels=['W0']+[f'{arm}@n{n}' for arm,_ in ARMS for n in NREAD];R={l:[] for l in labels}
    def put(label,gi,m32):R[label]+=P2.rows_of(gits[gi],gnt[gi],fo.score_group(m32,gi),f'M:{label}')
    t00=time.monotonic()
    for gi,g in enumerate(G):
        t0=time.monotonic();put('W0',gi,m32w0)
        for arm,job in ARMS:
            for n in NREAD:
                s=W[job][str(g['gid'])][arm]['states'][str(g['read_chunks'][str(n)])];m=rd.load_state(s['file'],expect_hash=s['hash']);put(f'{arm}@n{n}',gi,rd.to_fp32(m));del m
        lg(f'  group {g["gid"]} ({len(g["item_ids"])} questions) read: {time.monotonic()-t0:.0f}s (total {time.monotonic()-t00:.0f}s)')
    for l in labels:(out/f'E57_pqa12_rows_{l.replace("@","_")}.json').write_text(json.dumps(R[l],default=float))
    (out/'e57_pqa12_read_meta.json').write_text(json.dumps(dict(items_sha256=h,out_tail_sha=tsha,item_ids=[i for g in G for i in g['item_ids']],arms=[x[0] for x in ARMS],n_candidates={i:len(v) for i,v in lay.items()})))
def item_metrics(rows,ids,lay,score='cand_lp'):
    """rows: one row per question in the order of ids -> dict metric -> array over ids; B is nan where the string is no candidate of another question"""
    S={i:dict(zip(lay[i],r[score])) for i,r in zip(ids,rows)};n=len(ids);A=np.zeros(n);B=np.full(n,np.nan);rank=np.zeros(n);rr=np.zeros(n);top1=np.zeros(n);gap=np.zeros(n)
    for k,i in enumerate(ids):
        sc=np.asarray(rows[k][score],np.float64);A[k]=sc[0];rank[k]=1+(sc[1:]>sc[0]).sum();nc=len(sc);rr[k]=rank[k]/nc;top1[k]=float(rank[k]==1);gap[k]=sc[0]-sc[1:].mean() if nc>1 else np.nan
        vals=[S[q][i] for q in ids if q!=i and i in S[q]]
        if vals:B[k]=np.mean(vals)
    return dict(A=A,B=B,rank=rank,rank_ratio=rr,top1=top1,gap=gap)
def analyze(a):
    out=pathlib.Path(a['out']);rm=json.load(open(out/'e57_pqa12_read_meta.json'));ids=rm['item_ids'];lay=json.load(open(out/'e57_pqa12_layout.json'));meta,_=P6.load_groups(a['src640']);main={i for g in meta['groups'] for i in g['main_ids']}
    masks={'items455':np.ones(len(ids),bool),'main373':np.asarray([i in main for i in ids])};ld=lambda l:json.load(open(out/f'E57_pqa12_rows_{l.replace("@","_")}.json'));res={'states':{},'arm_differences':{}}
    for score in ('cand_lp','cand_mean'):
        M={l:item_metrics(ld(l),ids,lay,score) for l in ['W0']+[f'{arm}@n{n}' for arm,_ in ARMS for n in NREAD]};W0=M['W0'];sk=score
        for arm,_ in ARMS:
            for n in NREAD:
                l=f'{arm}@n{n}';m_=M[l];dA=m_['A']-W0['A'];dB=m_['B']-W0['B'];dAB=dA-dB;ok=~np.isnan(dB)
                e={}
                for sn,mk in masks.items():
                    k=mk&ok;e[sn]=dict(n=int(mk.sum()),n_with_B=int(k.sum()),dA=P4.boot(dA[mk]),dA_with_B=P4.boot(dA[k]),dB=P4.boot(dB[k]),dA_minus_dB=P4.boot(dAB[k]),
                        rank_ratio=P4.boot(m_['rank_ratio'][mk]),rank_ratio_W0=P4.boot(W0['rank_ratio'][mk]),top1=P4.boot(m_['top1'][mk]),top1_W0=P4.boot(W0['top1'][mk]),gap=P4.boot(m_['gap'][mk]),gap_W0=P4.boot(W0['gap'][mk]),gap_rise=P4.boot((m_['gap']-W0['gap'])[mk]))
                res['states'].setdefault(sk,{})[l]=e
        for a1,a2 in (('ALLQ128','NATQ'),('GAINQ512P10','NATQ'),('GAINQ512P10','ALLQ128')):
            for n in NREAD:
                x=M[f'{a1}@n{n}'];y=M[f'{a2}@n{n}'];dxA=x['A']-y['A'];dxB=x['B']-y['B'];e={}
                for sn,mk in masks.items():
                    k=mk&~np.isnan(dxB);e[sn]=dict(dA=P4.boot(dxA[mk]),dB=P4.boot(dxB[k]),dA_minus_dB=P4.boot((dxA-dxB)[k]),rank_ratio=P4.boot((x['rank_ratio']-y['rank_ratio'])[mk]),top1=P4.boot((x['top1']-y['top1'])[mk]),gap=P4.boot((x['gap']-y['gap'])[mk]))
                res['arm_differences'].setdefault(sk,{})[f'{a1}-{a2}@n{n}']=e
    cn=json.load(open(out/'e57_pqa12_counts.json'));c9=np.asarray([cn[i]['n9'] for i in ids]);res['counts']={f'n{n}':dict(share_exact_n=float((np.asarray([cn[i][f'n{n}'] for i in ids])==n).mean()),median=float(np.median([cn[i][f'n{n}'] for i in ids]))) for n in NREAD};res['counts']['n9_distribution']={str(k):int((c9==k).sum()) for k in sorted(set(c9.tolist()))[:30]}
    res['layout']=dict(candidates_per_question=dict(min=int(min(len(v) for v in lay.values())),median=float(np.median([len(v) for v in lay.values()])),max=int(max(len(v) for v in lay.values()))),n_questions=len(ids))
    (out/'E57_pqa12_keys.json').write_text(json.dumps(dict(meta=dict(arms=rm['arms'],definition='candidates = gold strings of the other items of the same group (related / duplicate strings left out); A = own question, B_bundle = the same string under the other questions of the group that have it as a candidate; scores = summed log-prob (cand_lp) and per-token mean (cand_mean)'),**res),indent=1,default=float))
def selftest():
    P6.selftest()
    golds={'a':'Paris','b':'paris france','c':'Rome','d':'Rome','e':'Berlin'};lo=layout(['a','b','c','d','e'],golds);assert lo['a']==['a','c','e'] and lo['c']==['c','a','b','e'] and lo['b']==['b','c','e']
    rows=[dict(cand_lp=[-1.,-3.,-5.]),dict(cand_lp=[-2.,-1.,-4.,-6.]),dict(cand_lp=[-1.5,-2.,-3.])];ids=['a','c','e'];lay={'a':['a','c','e'],'c':['c','a','e'],'e':['e','a','c']};m=item_metrics(rows,ids,lay)
    assert m['A'].tolist()==[-1.,-2.,-1.5] and m['rank'].tolist()==[1,2,1] and abs(m['B'][0]-(-1.5))<1e-9 and abs(m['B'][1]-(-3.0))<1e-9
    print('e57_pqa12_run selftest ok')
def main():
    if '--selftest' in sys.argv:selftest();return
    stage=P2._pop('--stage');g=lambda k:P2._pop(k);a=dict(out=g('--out'),src640=g('--src640'),src650=g('--src650'),items=g('--items-file'),bank=g('--bank'),stage1=g('--stage1'),para=g('--para'),rewrites8=g('--rewrites8'),max_groups=g('--max-groups'))
    {'read':read_stage,'analyze':analyze}[stage](a)
if __name__=='__main__':main()
