"""E57 public QA, stage 2 (job 460; GPU, read only, NO weight update, NO state): the W0 OUT / IN reads of job 440 (e57_squad_w0_reads.squad_reads, unchanged) for the multi-hop item files of e57_public_qa_build_multihop.py: HotpotQA-distractor-dev (1,000 items, sha256 c993393b…) and MuSiQue-Ans-dev (448 items, sha256 291351ba…), 16 candidates each. Schema: SQuAD items carry context (one string); multi-hop items carry paragraphs = [{title,text}] of the supporting paragraphs only. The IN context is the supporting paragraphs joined by a blank line (squad_reads then collapses whitespace and puts the text with blank lines at the END of the 8192-token tail, as in job 440). Ids are prefixed hotpot: / musique:, the answer type is prefixed likewise; keys are also given per dataset.
  python3 e57_multihop_w0_reads.py --items-file HOT.json,MUS.json  (same other arguments as e57_squad_w0_reads.py)   |   --selftest"""
import sys,json,hashlib,pathlib
HERE=pathlib.Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
import e57_squad_w0_reads as Q
EXPECT={'hotpot':'c993393b','musique':'291351ba'}
def load_multihop(path):
    items=[];hs=[]
    for p in path.split(','):
        b=json.load(open(p));h=b.pop('sha256');s=json.dumps(b,ensure_ascii=False,sort_keys=True);k=b['kind']
        assert hashlib.sha256(s.encode()).hexdigest()==h and h.startswith(EXPECT[k]),('items file hash mismatch',p,h[:12]);hs.append(k+':'+h)
        for r in b['items']:
            c=r['candidates'];g=r['gold_index'];cands=[c[g]]+[x for i,x in enumerate(c) if i!=g];ty=k+'-'+r['type']
            items.append(dict(id=k+':'+r['id'],title=k+':'+'|'.join(q['title'] for q in r['paragraphs']),type=ty,wh=r['wh'],run=r['run'],prefix=r['frame'].strip(),context='\n\n'.join(q['text'] for q in r['paragraphs']),cands=cands,kinds=['gold']+['distractor']*(len(cands)-1),gold_idx=0,hard_idx=[],all_idx=list(range(len(cands))),pool_idx=[],event=r['id']))
    return items,hashlib.sha256('+'.join(hs).encode()).hexdigest()
_keys=Q.keys_of
def keys_ds(rows):
    out=_keys(rows)
    for ds in EXPECT:
        R=[r for r in rows if r['id'].startswith(ds+':')]
        for cond,d in _keys(R).items():out.setdefault(cond,{})[ds+'-ALL']=d['ALL']
    return out
def selftest():
    Q.selftest()
    import tempfile,os
    d=tempfile.mkdtemp();f=[]
    for k in EXPECT:
        b=dict(kind=k,rules={},n_items=1,items=[dict(id='a',frame='q?',paragraphs=[dict(title='T1',text='x y'),dict(title='T2',text='z')],gold='g',variants=['g'],candidates=['x','g'],gold_index=1,type='CAP',wh='who',run=0,hops=2)])
        b['sha256']=hashlib.sha256(json.dumps(b,ensure_ascii=False,sort_keys=True).encode()).hexdigest();EXPECT_old=EXPECT[k];EXPECT[k]=b['sha256'][:8];f.append(os.path.join(d,k+'.json'));json.dump(b,open(f[-1],'w'))
    it,h=load_multihop(','.join(f));assert len(it)==2 and it[0]['cands'][0]=='g' and it[0]['context']=='x y\n\nz' and it[1]['id'].startswith('musique:') or it[1]['id'].startswith('musique:')
    print('e57_multihop_w0_reads selftest ok')
def main():
    if '--selftest' in sys.argv:selftest();return
    Q.load_items_file=load_multihop;Q.keys_of=keys_ds;Q.main()
if __name__=='__main__':main()
