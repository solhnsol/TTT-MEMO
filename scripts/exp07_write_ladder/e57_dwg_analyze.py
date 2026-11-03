"""E57 decision-token window gain (new file; thread 'QA에서 가림이 안 되는 이유 규명', one run approved by Hans; W0 model, read only, no writing): at every position where the answer string occurs in the write stream the log-prob of (a) the first content token (a space-only token is skipped), (b) the second token, (c) the mean of the later tokens, under (i) the natural window and (ii) the window with the band erased (each row sees only the previous b tokens, b = 128 for QA / news, 256 for synthetic = the ALLQ mask range); window gain = lp(i) - lp(ii) = nll_band - nll_win.
Settings: QA = the 33 paragraph-group episode streams of job 640 (versions 1..9; answer = gold string of the items of the group, word-boundary match, case-insensitive; own = inside the paragraph of the item); synthetic s64 (the place of the fact in every sentence about it; exposure = chunk index + 1; chunks 8..127 only, the stored pass starts at chunk 8); news stream A (100 overlap-free T1/T2 frames, gold string in the fragments of the frame's event, exposure = ordinal in stream order; chunks 8..279 only).
  python3 e57_dwg_analyze.py --stage locate --out O | --stage aggregate --out O --qa-nll F [--json J] | --selftest
Positions: a token at stream position p (p >= 1) is predicted by row (p-1) % 1024 of chunk (p-1) // 1024. Raw per-position file: <out>/e57_dwg_positions.json (locate) and <out>/e57_dwg_raw.npz (aggregate)."""
import sys,os,re,json,types,pathlib,hashlib
import numpy as np
HERE=pathlib.Path(__file__).resolve().parent;ROOT=HERE.parents[1]
for p in (HERE,HERE/'e57',ROOT/'scripts/exp04_what_ttt_retains',ROOT/'scripts/exp05_write_dynamics'):sys.path.insert(0,str(p))
EXP=ROOT/'experiments/exp07_write_ladder';CH=1024
import e57_pqa2_stream as S
import e57_pqa3_stream as S3
def pat(s):return re.compile(r'(?<![a-z0-9])'+re.escape(s)+r'(?![a-z0-9])',re.I)
def content_positions(tk,ids,offs,a,b,p0):
    """positions (global, = p0 + index) of the tokens overlapping the char span [a,b), space-only tokens dropped"""
    out=[]
    for j,(i_,(s,e)) in enumerate(zip(ids,offs)):
        if e>a and s<b and tk.decode([i_]).strip()!='':out.append(p0+j)
    return out
def find_occ(tk,pieces,gold):
    """pieces: [(text, ids, offsets, p0)] consecutive; -> [positions per occurrence of gold in the concatenated text]"""
    text=''.join(t for t,_,_,_ in pieces);starts=np.cumsum([0]+[len(t) for t,_,_,_ in pieces]);res=[]
    for m in pat(gold).finditer(text):
        a,b=m.span();pos=[]
        for (t,ids,offs,p0),c0 in zip(pieces,starts):
            if c0+len(t)<=a or c0>=b:continue
            pos+=content_positions(tk,ids,offs,a-c0,b-c0,p0)
        res.append(pos)
    return res
def encode_piece(tk,t):
    e=tk.encode(t,add_special_tokens=False);return list(e.ids),list(e.offsets)
def locate_qa(tk,meta,ids_z,items,para,rew):
    byid={it['id']:it for it in items};byp={p['pid']:p for p in para};recs=[];checks=0
    for g in meta['groups']:
        ids=ids_z[f"g{g['gid']}"];pos=0;ps=g['pids']
        for v in range(9):
            pieces_by_par=[]
            for pid in ps:
                text=byp[pid]['context'] if v==0 else rew[pid][v-1];pcs=[]
                for t,x in S3.enc_par(tk,text):
                    ii,oo=encode_piece(tk,t);assert ii==list(x),'encoding differs from the stream';assert list(ids[pos:pos+len(ii)])==ii,'stream differs';pcs.append((t,ii,oo,pos));pos+=len(ii)
                pieces_by_par.append(pcs)
            assert pos==g['section_end_tokens'][v];checks+=1
            for it_id in g['item_ids']:
                it=byid[it_id]
                for pid,pcs in zip(ps,pieces_by_par):
                    for oc in find_occ(tk,pcs,it['gold']):
                        if oc:recs.append(dict(setting='QA',unit=it_id,gid=g['gid'],version=v+1,type=it['type'],own=bool(it_id in byp[pid]['item_ids']),main=bool(it_id in g['main_ids']),pos=oc))
    return recs,checks
def locate_syn(tk,w,tgt):
    recs=[]
    for c in w['chunks']:
        k=c['k'];ids=list(c['ids'])
        for s in c['sents']:
            P=w['facts'][s['fact']]['P'];seg=ids[s['start']:s['end']];found=None
            for pre in (' ',''):
                p=tk.encode(pre+P,add_special_tokens=False).ids
                for i in range(len(seg)-len(p)+1):
                    if seg[i:i+len(p)]==p:found=(i,len(p));break
                if found:break
            assert found,'place not found'
            pos=[k*CH+s['start']+found[0]+j for j in range(found[1])]
            for q in pos:
                if q>=1 and tgt is not None and q-1<tgt.size:assert int(tgt.reshape(-1)[q-1])==int(ids[(q-k*CH)] if True else 0),'token mismatch'
            recs.append(dict(setting='SYN',unit=int(s['fact']),version=int(s['app'])+1,type='PLACE',own=True,main=True,pos=pos))
    return recs
def locate_news(tk,frames,frags,tgt):
    byev={}
    for f in frags:
        if f.get('type')=='frag':byev.setdefault(f['event'],[]).append(f)
    recs=[]
    for fr in frames:
        n=0
        for f in byev.get(fr['event'],[]):
            ii,oo=encode_piece(tk,f['text']);assert len(ii)==f['ntok'],('fragment length',f['fid'],len(ii),f['ntok'])
            for oc in find_occ(tk,[(f['text'],ii,oo,f['tok0'])],fr['gold']):
                if not oc:continue
                n+=1
                for q in oc:
                    if tgt is not None and (q-1)//CH>=8:assert int(tgt[(q-1)//CH,(q-1)%CH])==ii[q-f['tok0']],('token mismatch',f['fid'],q)
                recs.append(dict(setting='NEWS',unit=fr['id'],version=n,type=fr['type'],own=True,main=True,pos=oc))
    return recs
def locate_stage(a):
    out=pathlib.Path(a['out']);out.mkdir(parents=True,exist_ok=True);tk=S.tokenizer();R={}
    meta=json.load(open(EXP/'e57_public_qa_stage6/e57_pqa6_groups.json'));ids_z=np.load(EXP/'e57_public_qa_stage6/e57_pqa6_group_ids.npz');items=json.load(open(EXP/'e57_public_qa/e57_public_qa_squad_items.json'))['items']
    para=json.load(open(EXP/'e57_public_qa_stage3/data/paragraphs.json'));rew=json.load(open(EXP/'e57_public_qa_stage5/data/rewrites_all8.json'))
    qa,chk=locate_qa(tk,meta,ids_z,items,para,rew);print('QA',len(qa),'occurrences; section checks',chk)
    import e55_run as R55
    args=types.SimpleNamespace(cell='v3_16_1',seed=64,tokenizer_file=None,bg='pool',decoy='names',stub=False,arms='NAT');enc,w,tf=R55.build(args,64,read_apps={16,32,64,128},arms=['NAT'])
    z=np.load(EXP/'e57_syn_s64/syn_s64_nll.npz');syn=locate_syn(tk,w,None);print('SYN',len(syn),'occurrences')
    for r in syn:
        for q in r['pos']:
            k=(q-1)//CH
            if k>=8 and k<z['tgt'].shape[0]:assert int(z['tgt'][k,(q-1)%CH])==int(w['chunks'][q//CH]['ids'][q%CH]),'syn token mismatch'
    nw=json.load(open(EXP/'e57_news/e57_t1t2/t1t2_v1.json'))['items'];ov=json.load(open(EXP/'e57_news/e57_t1t2/e57_t1t2_overlap.json'))['items'];ovd={x['id']:x for x in ov}
    frames=[x for x in nw if ovd[x['id']]['stream']['1']==0];print('news frames with zero stream overlap at k=1:',len(frames))
    frags=[json.loads(x) for x in open(EXP/'e57_t1t2v2_in/e57_stream_sentences.jsonl')];tgt=np.load(EXP/'e57_band/e57_band_W0.npz')['tgt'];news=locate_news(tk,frames,frags,tgt);print('NEWS',len(news),'occurrences')
    (out/'e57_dwg_positions.json').write_text(json.dumps(dict(qa=qa,syn=syn,news=news,n_news_frames=len(frames))))
    for n,r in (('QA',qa),('SYN',syn),('NEWS',news)):print(n,'units',len({x['unit'] for x in r}),'tokens per occurrence',np.bincount([len(x['pos']) for x in r]).tolist())
def boot(x,seed=1,B=2000):
    x=np.asarray(x,np.float64);x=x[~np.isnan(x)]
    if len(x)==0:return None
    r=np.random.default_rng(seed);m=[x[r.integers(0,len(x),len(x))].mean() for _ in range(B)];return dict(mean=float(x.mean()),lo=float(np.percentile(m,2.5)),hi=float(np.percentile(m,97.5)),n_units=int(len(x)))
def token_parts(lpw,lpb,pos):
    """-> dict first / second / later-mean of lp_win, lp_band, gain for one occurrence (positions with data only)"""
    a=[(float(lpw[i]),float(lpb[i])) for i in range(len(pos))]
    def col(idx):
        if isinstance(idx,int):
            if idx>=len(a):return None
            w,b=a[idx]
        else:
            s=a[idx]
            if not s:return None
            w=float(np.mean([x for x,_ in s]));b=float(np.mean([y for _,y in s]))
        return dict(lp_win=w,lp_band=b,gain=w-b)
    return dict(first=col(0),second=col(1),later=col(slice(2,None)))
def gather(positions,nll_win,nll_band,valid_chunks):
    """per occurrence: lp (=-nll) of every position under the window and the band, only if all its chunks are valid"""
    res=[]
    for r in positions:
        ks=[(q-1)//CH for q in r['pos']]
        if any(k not in valid_chunks for k in ks):res.append(None);continue
        lw=[-float(nll_win[(q-1)//CH,(q-1)%CH]) for q in r['pos']];lb=[-float(nll_band[(q-1)//CH,(q-1)%CH]) for q in r['pos']];res.append(token_parts(lw,lb,r['pos']))
    return res
def cells(recs,parts,keyf):
    """group occurrences by cell key; per unit mean first -> bootstrap over units"""
    byc={}
    for r,p in zip(recs,parts):
        if p is None:continue
        byc.setdefault(keyf(r),[]).append((r['unit'],p))
    out={}
    for key,L in sorted(byc.items(),key=lambda kv:str(kv[0])):
        d=dict(n_occurrences=len(L),n_units=len({u for u,_ in L}))
        for c in ('first','second','later'):
            for m in ('lp_win','lp_band','gain'):
                perunit={}
                for u,p in L:
                    if p[c] is not None:perunit.setdefault(u,[]).append(p[c][m])
                d[f'{c}_{m}']=boot([np.mean(v) for v in perunit.values()])
        out[str(key)]=d
    return out
def aggregate_stage(a):
    out=pathlib.Path(a['out']);P=json.load(open(out/'e57_dwg_positions.json'));res={}
    qz=np.load(a['qa_nll']);bands=list(qz['bands']);bi=bands.index(128)
    # QA: group streams
    qa=P['qa'];parts=[]
    for r in qa:
        g=r['gid'];parts.append(gather([r],qz[f'g{g}_win'],qz[f'g{g}_band'][bi],set(range(qz[f'g{g}_win'].shape[0])))[0])
    res['QA']=dict(band=128,by_version_type=cells(qa,parts,lambda r:('v%d'%r['version'],r['type'])),by_version=cells(qa,parts,lambda r:('v%d'%r['version'],'all')),by_version_own=cells(qa,parts,lambda r:('v%d'%r['version'],'own' if r['own'] else 'other')),by_type=cells(qa,parts,lambda r:('all',r['type'])))
    # synthetic
    z=np.load(EXP/'e57_syn_s64/syn_s64_nll.npz');bs=list(z['bands']);valid={k for k in range(z['nll_win'].shape[0]) if k>=8}
    syn=P['syn'];sp=gather(syn,z['nll_win'],z['nll_band'][bs.index(256)],valid)
    def sbin(v):return '1-8' if v<=8 else '9-16' if v<=16 else '17-32' if v<=32 else '33-64' if v<=64 else '65-128'
    res['SYN']=dict(band=256,by_exposure=cells(syn,sp,lambda r:(sbin(r['version']),'PLACE')),n_excluded_before_chunk8=int(sum(p is None for p in sp)))
    # news
    z=np.load(EXP/'e57_band/e57_band_W0.npz');bs=list(z['bands']);nw=P['news']
    npart=gather(nw,z['nll_win'],z['nll_band'][bs.index(128)],set(range(8,280)))
    def nbin(v):return str(v) if v<=4 else '5-8' if v<=8 else '9+'
    res['NEWS']=dict(band=128,by_exposure=cells(nw,npart,lambda r:(nbin(r['version']),'all')),by_exposure_type=cells(nw,npart,lambda r:(nbin(r['version']),r['type'])),n_excluded_before_chunk8=int(sum(p is None for p in npart)),n_frames=P['n_news_frames'])
    # raw per-position file
    raw=[]
    for name,recs,pp in (('QA',qa,parts),('SYN',syn,sp),('NEWS',nw,npart)):
        for r,p in zip(recs,pp):raw.append(dict(setting=name,unit=r['unit'],version=r['version'],type=r['type'],own=r['own'],pos=r['pos'],parts=p))
    (out/'e57_dwg_raw.json').write_text(json.dumps(raw))
    res['definitions']='lp = log-prob of the token (= -nll); gain = lp(natural window) - lp(band-erased window); first = first content token (space-only skipped), second = next token, later = mean of the remaining tokens of the occurrence; CIs: 95% bootstrap over units (items / facts / frames; mean over a unit first), 2000 draws'
    fn=pathlib.Path(a.get('json') or out/'e57_decision_window_gain.json');fn.write_text(json.dumps(res,indent=1));print('wrote',fn)
def selftest():
    tk=S.tokenizer();ids,offs=encode_piece(tk,'\n\nIt happened in 1990 near Ash Tree Road.')
    pcs=[('\n\nIt happened in 1990 near Ash Tree Road.',ids,offs,100)]
    o=find_occ(tk,pcs,'1990');assert len(o)==1 and o[0] and all(tk.decode([ids[p-100]]).strip() for p in o[0]),o
    o=find_occ(tk,pcs,'ash tree');assert len(o)==1 and len(o[0])==2,o
    assert find_occ(tk,pcs,'tree road')[0][0]==find_occ(tk,pcs,'tree')[0][0]
    assert find_occ(tk,pcs,'sh')==[],'word boundary'
    lw=[-1.0,-2.0,-3.0,-5.0];lb=[-4.0,-2.0,-6.0,-5.0];p=token_parts(lw,lb,[1,2,3,4]);assert p['first']['gain']==3.0 and p['second']['gain']==0.0 and abs(p['later']['gain']-1.5)<1e-9
    assert token_parts([-1.0],[-2.0],[1])['second'] is None
    nl=np.zeros((2,CH),np.float32);nb=np.zeros((2,CH),np.float32);nl[0,5]=1;nb[0,5]=4;g=gather([dict(pos=[6]),dict(pos=[CH+3])],nl,nb,{0})
    assert g[0]['first']['gain']==3.0 and g[1] is None
    b=boot([1,1,1]);assert b['mean']==1 and b['lo']==1
    print('e57_dwg_analyze selftest ok')
def main():
    if '--selftest' in sys.argv:selftest();return
    g=lambda k,d=None:sys.argv[sys.argv.index(k)+1] if k in sys.argv else d
    st=g('--stage');a=dict(out=g('--out'),qa_nll=g('--qa-nll'),json=g('--json'))
    {'locate':locate_stage,'aggregate':aggregate_stage}[st](a)
if __name__=='__main__':main()
