"""E57 public QA stage 20 (new file; GPU; QA cause thread, Hans card 'QA 65버전 + 합성 1시드 짝 섞기' 11/05 09:15 KST, QA part): the PAIR-SHUFFLED 65-version QA episodes. Data: rewrites_64_pairshuffle.json (sha256 2a4dee68784df7ca...): inside each of the bundles 5 / 8 / 9 / 10 / 12 the answer strings are cyclically substituted in the original paragraph and in all 64 re-descriptions (the same substitution; field new_golds = original gold string -> string now written in its place). The stream (orig + 64 re-descriptions, sections of the 10 paragraphs, filler completion) is written exactly as in job 930 with the arms NATQ, ALLQ128 and M3 w=8 (NAT norm source; ALLQ128 on every row from chunk 0; M3 weight 8 on the deciding rows of the W0 pass over the SHUFFLED streams; canary at every read chunk), read points n = 1 / 3 / 5 / 9 / 17 / 25 / 33 / 49 / 65, and SCORED with the ORIGINAL pairs (original question, original gold, the original 16 candidates; reads of job 930 unchanged). Stage compare: per n and arm the item-level and bundle-level differences (shuffled run minus job 930) of the first-content-token dA, d(A-B), second token and rank-ratio improvement, and the same metrics for the NEW pair (the candidate that equals the string now written in the paragraph, when it is among the 16 candidates).
  python3 e57_pqa20_run.py --stage build|write|read|analyze|compare --out O ... | --selftest"""
import sys,json,time,gc,hashlib,pathlib
import numpy as np
HERE=pathlib.Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
import e57_pqa2_run as P2
import e57_pqa2_stream as S
import e57_pqa3_stream as S3
import e57_pqa6_run as P6
import e57_pqa15_run as P15
import e57_pqa19_run as P19
import e57_squad_w0_reads as Q
CH=1024;NR=P19.NR_ALL;ARMS=['NATQ','ALLQ128','M3W8Q128'];SHA12='2a4dee68784d'
def build(a):
    out=pathlib.Path(a['out']);out.mkdir(parents=True,exist_ok=True);raw=open(a['shuf'],'rb').read();assert hashlib.sha256(raw).hexdigest().startswith(SHA12),'pairshuffle json sha256 differs'
    dat=json.loads(raw);src=pathlib.Path(a['src790']);meta,ids_z=P6.load_groups(src);tk=S.tokenizer();train=json.load(open(a['train']));dev=json.load(open(a['dev']));para=json.load(open(a['para']));fm,fx,ov=S3.filler_paragraphs(train,dev,len(para),P6.SEED);filler=[]
    for ti,tx in fm[:24]:
        for s,x in S3.enc_par(tk,tx):filler+=x
    groups=[];ids_all={};ref=pathlib.Path(a['ref930']);m930,i930=P6.load_groups(ref);g930={g['gid']:g for g in m930['groups']}
    for g in meta['groups']:
        ids=[];ends=[]
        for v in range(65):
            for pid in g['pids']:
                assert len(dat[pid]['rewrites'])==64
                text=dat[pid]['context'] if v==0 else dat[pid]['rewrites'][v-1]
                for s,x in S3.enc_par(tk,text):ids+=x
            ends.append(len(ids))
        need=-(-len(ids)//CH)*CH-len(ids);ids+=filler[:need];assert need<len(filler) and len(ids)%CH==0
        rch={str(n):(ends[n-1]-1)//CH for n in NR};spill={str(n):int((rch[str(n)]+1)*CH-ends[n-1]) for n in NR}
        groups.append(dict(gid=g['gid'],pids=g['pids'],item_ids=g['item_ids'],main_ids=g['main_ids'],n_chunks=len(ids)//CH,n_tokens=len(ids),section_end_tokens=ends,read_chunks=rch,spill_tokens=spill,filler_tokens=need,ids_sha=hashlib.sha256(np.asarray(ids,np.int64).tobytes()).hexdigest()[:16],versions=65,n_chunks_job930=g930[g['gid']]['n_chunks'],section_end_tokens_job930=g930[g['gid']]['section_end_tokens']))
        ids_all[f"g{g['gid']}"]=np.asarray(ids,np.int64)
    np.savez_compressed(out/'e57_pqa6_group_ids.npz',**ids_all);(out/'e57_pqa6_groups.json').write_text(json.dumps(dict(seed=P6.SEED,n_groups=len(groups),groups=groups,read_n=NR,versions=65,pairshuffle_sha12=SHA12)))
    print('groups',[(g['gid'],g['n_chunks'],g['n_chunks_job930'],g['read_chunks']['65'],g930[g['gid']]['read_chunks']['65']) for g in groups])
def write_stage(a):
    P19.ARMS_W=['NAT','NATQ','ALLQ128','M3W8Q128'];P19.write_stage(a)
def read_stage(a):
    P15.NREAD=list(NR);P15.ARMS=list(ARMS);P15.read_stage(dict(a,skip_repro=True))
def analyze(a):
    P15.NREAD=list(NR);P15.ARMS=list(ARMS);P15.analyze(a)
def compare_stage(a):
    import e57_pqa2_stream as S2
    out=pathlib.Path(a['out']);shuf=pathlib.Path(a['merged']);orig=pathlib.Path(a['r930']);meta=json.load(open(shuf/'e57_pqa15_read_meta.json'));ids=meta['item_ids'];tok=json.load(open(shuf/'e57_pqa15_tokens.json'));tk=S2.tokenizer();dec=lambda t:tk.decode(t)
    items,_=Q.load_items_file(a['items']);byid={it['id']:it for it in items};dat=json.load(open(a['shuf_json']));bun=[next(g['gid'] for g in P6.load_groups(shuf)[0]['groups'] if i in g['item_ids']) for i in ids]
    pid_of={}
    for pid,v in dat.items():
        pass
    para=json.load(open(a['para']));owner={i:p['pid'] for p in para for i in p['item_ids']};newg={}
    for i in ids:
        ng=dat[owner[i]]['new_golds'];g0=byid[i]['cands'][0];newg[i]=ng.get(g0) if g0 in ng else next((v for k,v in ng.items() if k.lower()==g0.lower()),None)
    nidx={}
    for i in ids:
        c=[x.strip().lower() for x in byid[i]['cands']];n=(newg[i] or '').strip().lower();nidx[i]=c.index(n) if n in c else None
    ld=lambda d,c,l:json.load(open(d/f'E57_pqa15_rows_{c}_{l}.json'))
    def firsts(rows,tokd,idx_of):
        o=np.full(len(ids),np.nan);byrow={r['id']:r for r in rows}
        for q,i in enumerate(ids):
            j=idx_of(i)
            if j is None:continue
            cl,_=P15.content_lps(byrow[i]['cand_tok_lp'],tokd[i],dec)
            if len(cl[j]):o[q]=cl[j][0]
        return o
    def metrics(d,l):
        A=P15.item_arrays(ld(d,'A',l),ids,tok['A'],dec,'q');B=P15.item_arrays(ld(d,'B',l),ids,tok['B'],dec,'q')
        nA=firsts(ld(d,'A',l),tok['A'],lambda i:nidx[i]);nB=firsts(ld(d,'B',l),tok['B'],lambda i:nidx[i])
        return dict(A=A['A_first'],B=B['A_first'],A2=A['A_second'],rr=A['rr_first'],nA=nA,nB=nB)
    res={'n_items_with_new_gold_among_candidates':int(sum(v is not None for v in nidx.values())),'by_n':{}}
    W0s=metrics(shuf,'W0');W0o=metrics(orig,'W0')
    def st(x):return dict(item=P15.boot_items(x),bundle=P15.boot_bundles(x,bun))
    for n in NR:
        res['by_n'][str(n)]={}
        for arm in ARMS:
            l=f'{arm}_n{n}';ms=metrics(shuf,l);mo=metrics(orig,l)
            def d(m,w0):return dict(dA=m['A']-w0['A'],dAB=(m['A']-w0['A'])-(m['B']-w0['B']),dA2=m['A2']-w0['A2'],rr_impr=w0['rr']-m['rr'],new_dA=m['nA']-w0['nA'],new_dAB=(m['nA']-w0['nA'])-(m['nB']-w0['nB']))
            ds=d(ms,W0s);do=d(mo,W0o)
            res['by_n'][str(n)][arm]={'shuffled':{k:st(v) for k,v in ds.items()},'job930':{k:st(v) for k,v in do.items()},'shuffled_minus_job930':{k:st(ds[k]-do[k]) for k in ds}}
    pathlib.Path(a['json']).write_text(json.dumps(res,indent=1,default=float));print('wrote',a['json'])
def selftest():
    P19.selftest();print('e57_pqa20_run selftest ok')
def main():
    if '--selftest' in sys.argv:selftest();return
    stage=P2._pop('--stage');g=lambda k:P2._pop(k)
    a=dict(out=g('--out'),src640=g('--src640'),src790=g('--src790'),items=g('--items-file'),bank=g('--bank'),assign=g('--assign'),stage1=g('--stage1'),para=g('--para'),train=g('--train'),dev=g('--dev'),shuf=g('--shuf'),shuf_json=g('--shuf'),ref930=g('--ref930'),r930=g('--r930'),merged=g('--merged'),json=g('--json'),max_groups=None)
    {'build':build,'write':write_stage,'read':read_stage,'analyze':analyze,'compare':compare_stage}[stage](a)
if __name__=='__main__':main()
