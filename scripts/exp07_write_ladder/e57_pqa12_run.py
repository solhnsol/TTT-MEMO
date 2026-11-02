"""E57 public QA stage 12 (E57 122-AB / 123; new file; read only, NO weight update): bundle-matrix reads. The stored states of jobs 640 (NATQ, ALLQ128) and 650 (GAINQ512 p.10) at n = 1 / 3 / 5 / 9 versions (33 paragraph-group episodes) and W0 are read on the items455 with the candidates replaced: the candidates of the question of item i are the GOLD STRINGS of the other items of the same group (the answers of the other questions about the same written paragraphs), plus its own gold; a gold string of another item that equals the own gold or contains / is contained in it (case-insensitive substring) is left out, duplicate strings are listed once (so a question has up to group size - 1 distractors, not 15). Reader, prompt frame (own question), OUT context (stage-1 tail) and packing per group as jobs 640 / 650 (e57_pqa6_run.FastOUTG).
Per question i and state: rank of the own gold among its candidates (rank ratio = rank / number of candidates, top-1 = rank 1), mean gold-minus-candidate log-prob difference (summed log-prob; per-token mean as a second score), and the matrix version of the binding: A_i = score of the gold string of item i under its OWN question, B_bundle_i = mean score of the SAME string under the questions of the OTHER items of the group that have it as a candidate; dA, dB, d(A-B_bundle) against W0 per item; mean over items with the bootstrap 95% CI over items (items455 and main373), arm differences as paired differences per item. The number of occurrences of every candidate string in the written text of its group (first n versions, case-insensitive, between non-alphanumeric characters; n = 1, 3, 5, 9) is stored in e57_pqa12_counts.json, and the share with 9 occurrences at n = 9 (once per version) is reported.
Branch-token metrics (Hans: once a token of the answer is revealed the following tokens depend on it): the log-prob of every scored candidate TOKEN is stored (cand_tok_lp, cand_tok_ids per row). (a) branch rank: gold g against every candidate c, both scored from their first token up to and including the first token at which the two token sequences differ (if one is a prefix of the other, up to the end of the shorter one); rank = 1 + number of candidates that beat the gold at their own branch point, top-1 = rank 1, mean gold-minus-candidate difference at the branch; (b) first-token rank: the same with the first token only; (c) A / B_bundle / A-B in both the first-token score and the full summed score (the branch point is defined only for pairs, so for a single string the first-token and the full score are both reported).
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
class FastOUTG3(P6.FastOUTG):
    def score_group_tokens(self,m32,g):
        """per-row arrays of the log-prob of every scored token (list in row order)"""
        rt=self.rt;jax=self.jax;jnp=self.jnp;from core import clone_state
        c0,c1=self.grange[g];idx=list(range(c0,c1));idx+=[idx[-1]]*((-len(idx))%self.GP)
        with rt.mesh,jax.default_matmul_precision('highest'):
            st=clone_state(rt.state_suffix)
            for j in range(self.TL//P6.CH):st,_=rt.chunk_logp(m32,st,rt.chunk(self.tailseq,j),self.H[j*P6.CH:(j+1)*P6.CH])
            out={}
            for s in range(0,len(idx),self.GP):
                sel=idx[s:s+self.GP];r=np.asarray(self.lp_packed(m32,st,jnp.asarray(self.ids[sel]),jnp.asarray(self.tg[sel]),jnp.asarray(self.mk[sel]),jnp.asarray(self.pref1[sel]),jnp.asarray(self.seg[sel]),jnp.asarray(self.loc[sel])))
                for q,c in enumerate(sel):out.setdefault(c,r[q])
        return [np.asarray(out[c][a:b],np.float64) for c,a,b in self.gwhere[g]]
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
    m32w0=rd.to_fp32(rt.base);fo=FastOUTG3(rt32,m32w0,out_tail,grows,log=lg);labels=['W0']+[f'{arm}@n{n}' for arm,_ in ARMS for n in NREAD];R={l:[] for l in labels}
    tokids={}
    for gi,g in enumerate(G):
        o=0
        for it in gits[gi]:
            tokids[it['id']]=[[int(x) for x in grows[gi][o+q][0][grows[gi][o+q][1][0]+1:grows[gi][o+q][1][1]+1]] for q in range(len(it['cands']))];o+=len(it['cands'])
    (out/'e57_pqa12_tokens.json').write_text(json.dumps(tokids))
    def put(label,gi,m32):
        toks=fo.score_group_tokens(m32,gi);rows=P2.rows_of(gits[gi],gnt[gi],np.asarray([t.sum() for t in toks]),f'M:{label}');o=0
        for it,r in zip(gits[gi],rows):
            r['cand_tok_lp']=[[float(x) for x in t] for t in toks[o:o+len(it['cands'])]];o+=len(it['cands'])
        R[label]+=rows
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
def branch_item_metrics(rows,ids,lay,tokids):
    """per-question branch-token and first-token metrics (see the module docstring); rows carry cand_tok_lp"""
    n=len(ids);out={k:np.zeros(n) for k in ('rank_branch','rr_branch','top1_branch','gap_branch','rank_first','rr_first','top1_first','gap_first','A_first')};out['B_first']=np.full(n,np.nan);F={}
    for k,(i,r) in enumerate(zip(ids,rows)):
        tl=[np.asarray(x,np.float64) for x in r['cand_tok_lp']];ti=tokids[i];F[i]=dict(zip(lay[i],[x[0] for x in tl]));g=tl[0];nb=0;nf=0;db=[];df=[]
        for c in range(1,len(tl)):
            tg,tc=ti[0],ti[c];t=next((q for q in range(min(len(tg),len(tc))) if tg[q]!=tc[q]),min(len(tg),len(tc))-1);sg=g[:t+1].sum();sc=tl[c][:t+1].sum();nb+=int(sc>sg);db.append(sg-sc)
            nf+=int(tl[c][0]>g[0]);df.append(g[0]-tl[c][0])
        nc=len(tl);out['rank_branch'][k]=1+nb;out['rr_branch'][k]=(1+nb)/nc;out['top1_branch'][k]=float(nb==0);out['gap_branch'][k]=np.mean(db) if db else np.nan
        out['rank_first'][k]=1+nf;out['rr_first'][k]=(1+nf)/nc;out['top1_first'][k]=float(nf==0);out['gap_first'][k]=np.mean(df) if df else np.nan;out['A_first'][k]=g[0]
    for k,i in enumerate(ids):
        v=[F[q][i] for q in ids if q!=i and i in F[q]]
        if v:out['B_first'][k]=np.mean(v)
    return out
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
    tokids=json.load(open(out/'e57_pqa12_tokens.json'));BM={l:branch_item_metrics(ld(l),ids,lay,tokids) for l in ['W0']+[f'{arm}@n{n}' for arm,_ in ARMS for n in NREAD]};res['branch']={'states':{},'arm_differences':{}};W0b=BM['W0']
    for arm,_ in ARMS:
        for n in NREAD:
            l=f'{arm}@n{n}';m_=BM[l];dA=m_['A_first']-W0b['A_first'];dB=m_['B_first']-W0b['B_first'];ok=~np.isnan(dB);e={}
            for sn,mk in masks.items():
                k=mk&ok;e[sn]=dict(n=int(mk.sum()),n_with_B=int(k.sum()),first_dA=P4.boot(dA[mk]),first_dA_with_B=P4.boot(dA[k]),first_dB=P4.boot(dB[k]),first_dA_minus_dB=P4.boot((dA-dB)[k]),
                    **{f'{m}_{w}':P4.boot(m_[f'{m}_{w}'][mk]) for m in ('rr','top1','gap') for w in ('branch','first')},**{f'{m}_{w}_W0':P4.boot(W0b[f'{m}_{w}'][mk]) for m in ('rr','top1','gap') for w in ('branch','first')})
            res['branch']['states'][l]=e
    for a1,a2 in (('ALLQ128','NATQ'),('GAINQ512P10','NATQ'),('GAINQ512P10','ALLQ128')):
        for n in NREAD:
            x=BM[f'{a1}@n{n}'];y=BM[f'{a2}@n{n}'];dxA=x['A_first']-y['A_first'];dxB=x['B_first']-y['B_first'];e={}
            for sn,mk in masks.items():
                k=mk&~np.isnan(dxB);e[sn]=dict(first_dA=P4.boot(dxA[mk]),first_dB=P4.boot(dxB[k]),first_dA_minus_dB=P4.boot((dxA-dxB)[k]),**{f'{m}_{w}':P4.boot((x[f'{m}_{w}']-y[f'{m}_{w}'])[mk]) for m in ('rr','top1','gap') for w in ('branch','first')})
            res['branch']['arm_differences'][f'{a1}-{a2}@n{n}']=e
    cn=json.load(open(out/'e57_pqa12_counts.json'));c9=np.asarray([cn[i]['n9'] for i in ids]);res['counts']={f'n{n}':dict(share_exact_n=float((np.asarray([cn[i][f'n{n}'] for i in ids])==n).mean()),median=float(np.median([cn[i][f'n{n}'] for i in ids]))) for n in NREAD};res['counts']['n9_distribution']={str(k):int((c9==k).sum()) for k in sorted(set(c9.tolist()))[:30]}
    res['layout']=dict(candidates_per_question=dict(min=int(min(len(v) for v in lay.values())),median=float(np.median([len(v) for v in lay.values()])),max=int(max(len(v) for v in lay.values()))),n_questions=len(ids))
    (out/'E57_pqa12_keys.json').write_text(json.dumps(dict(meta=dict(arms=rm['arms'],definition='candidates = gold strings of the other items of the same group (related / duplicate strings left out); A = own question, B_bundle = the same string under the other questions of the group that have it as a candidate; scores = summed log-prob (cand_lp) and per-token mean (cand_mean)'),**res),indent=1,default=float))
def selftest():
    P6.selftest()
    golds={'a':'Paris','b':'paris france','c':'Rome','d':'Rome','e':'Berlin'};lo=layout(['a','b','c','d','e'],golds);assert lo['a']==['a','c','e'] and lo['c']==['c','a','b','e'] and lo['b']==['b','c','e']
    rows=[dict(cand_lp=[-1.,-3.,-5.]),dict(cand_lp=[-2.,-1.,-4.,-6.]),dict(cand_lp=[-1.5,-2.,-3.])];ids=['a','c','e'];lay={'a':['a','c','e'],'c':['c','a','e'],'e':['e','a','c']};m=item_metrics(rows,ids,lay)
    assert m['A'].tolist()==[-1.,-2.,-1.5] and m['rank'].tolist()==[1,2,1] and abs(m['B'][0]-(-1.5))<1e-9 and abs(m['B'][1]-(-3.0))<1e-9
    rows=[dict(cand_tok_lp=[[-1.,-2.],[-3.,-1.],[-0.5,-4.]]),dict(cand_tok_lp=[[-1.,-1.],[-2.,-2.]])];tok={'a':[[1,2],[1,3],[4,5]],'b':[[7,8],[9,8]]};lay2={'a':['a','b','c'],'b':['b','a']}
    r=branch_item_metrics(rows,['a','b'],lay2,tok)
    assert r['rank_branch'].tolist()==[2,1] and r['top1_first'].tolist()==[0,1] and r['rank_first'][0]==2 and r['A_first'].tolist()==[-1.,-1.]
    assert abs(r['gap_branch'][0]-np.mean([-3-(-4),-1-(-0.5)]))<1e-9
    print('e57_pqa12_run selftest ok')
def main():
    if '--selftest' in sys.argv:selftest();return
    stage=P2._pop('--stage');g=lambda k:P2._pop(k);a=dict(out=g('--out'),src640=g('--src640'),src650=g('--src650'),items=g('--items-file'),bank=g('--bank'),stage1=g('--stage1'),para=g('--para'),rewrites8=g('--rewrites8'),max_groups=g('--max-groups'))
    {'read':read_stage,'analyze':analyze}[stage](a)
if __name__=='__main__':main()
