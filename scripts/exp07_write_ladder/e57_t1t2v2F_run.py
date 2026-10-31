"""E57 T1/T2 v2 FORWARD-pair reads (prepared for a job after the freeze; CPU check now): the reader of e57_t1t2v2_run.py (job 570: same OUT context, same scoring: summed log-prob and per-token mean of the candidate after the frame prefix, per-token mean primary, accuracy / rank ratio, the states of the job-570 manifest W0 / NAT / NATQ / ALLQ128 on stream A and stream B20) for the FORWARD items: one frame per item ('question' form of the forward draft t1t2_v2F_*.json: items with id, event, type, gold, variants, forms[0].frame, hard_nominated [surfaces], stream), candidates = gold (first variant) + the 20 easy candidates of the forward candidate file (e57_t1t2v2F_candidates.json of the server check) + the hard candidates of the item (--hard-source nominated: the nominated ones; nominated+auto: the nominated ones, and when an item has none the hard_auto surfaces of the candidate file, at most 4). Group names 'T1F-question'. Only the conversion of the frozen forward file to bank-like items differs from job 570; everything else (packs, scoring, aggregation, outputs) is the code of e57_t1t2v2_run.py.
  python3 e57_t1t2v2F_run.py --cpu-check --frozen F --cands C --bank B [--allow-unfrozen]
  python3 e57_t1t2v2F_run.py --stage read --frozen F --frozen-sha SHA --cands C --cands-sha SHA --bank B --manifest M --out O [--hard-source nominated|nominated+auto]  |  --selftest"""
import sys,json,hashlib,pathlib
HERE=pathlib.Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
import e57_t1t2v2_run as V
FS={'sha':None,'hard_source':'nominated','allow_unfrozen':False}
def load_frozen(path):
    raw=open(path,'rb').read();h=hashlib.sha256(raw).hexdigest()
    if not FS['allow_unfrozen']:assert FS['sha'] and h==FS['sha'],('forward frozen file sha256 differs from the given one',h[:16])
    return json.loads(raw)
def bank_like_F(frozen,cands):
    cd={c['id']:c for c in cands};qs=[]
    for it in frozen['items']:
        c=cd[it['id']];assert c['gold']==it['gold'] and c['event']==it['event'],it['id']
        hn=[dict(surface=h if isinstance(h,str) else h['surface']) for h in it['hard_nominated']]
        if FS['hard_source']=='nominated+auto' and not hn:hn=[dict(surface=h['surface']) for h in c['hard_auto'][:4]]
        for f in it['forms']:
            qs.append(dict(id=f"{it['id']}-{f['form']}",event=it['event'],type=it['type'],stratum=f"T1F-{f['form']}",frame=f['frame'],S='',fixed_surface=it['variants'][0],canonical=it['variants'][0],variants=list(it['variants']),easy=[dict(surface=e['surface']) for e in c['easy']],hard_nominated=hn,hard_all=[],source=it['source'],stream=c['stream']))
    return dict(version='t1t2v2F_bank_like',questions=qs)
def make_items(frozen,cands,tmpdir):
    bl=bank_like_F(frozen,cands);f=pathlib.Path(tmpdir)/'_t1t2v2F_bank_like.json';f.parent.mkdir(parents=True,exist_ok=True);f.write_text(json.dumps(bl))
    items=V.RD.load_items(f);meta={q['id']:q for q in bl['questions']}
    for it in items:it['source']=meta[it['id']]['source'];it['kind_form']=it['stratum'];it['stream']=meta[it['id']]['stream']
    return items
def selftest():
    V.selftest()
    FS['allow_unfrozen']=True
    fr=dict(items=[dict(id='x-fwd',source='x',event='E',type='PERSON',gold='G',variants=['G','Gg'],hard_nominated=[],stream='A',forms=[dict(form='question',frame='Who? ___')])])
    cd=[dict(id='x-fwd',stream='A',event='E',gold='G',easy=[dict(surface='e%d'%i) for i in range(20)],hard_auto=[dict(surface='h%d'%i) for i in range(6)])]
    q=bank_like_F(fr,cd)['questions'][0];assert q['stratum']=='T1F-question' and len(q['easy'])==20 and q['hard_nominated']==[]
    FS['hard_source']='nominated+auto';q=bank_like_F(fr,cd)['questions'][0];assert [h['surface'] for h in q['hard_nominated']]==['h0','h1','h2','h3'];FS['hard_source']='nominated'
    print('e57_t1t2v2F_run selftest ok')
def main():
    if '--selftest' in sys.argv:selftest();return
    FS['sha']=V._pop('--frozen-sha');cs=V._pop('--cands-sha');FS['hard_source']=V._pop('--hard-source','nominated')
    if '--allow-unfrozen' in sys.argv:sys.argv.remove('--allow-unfrozen');FS['allow_unfrozen']=True
    V.load_frozen=load_frozen;V.make_items=make_items
    if FS['sha']:V.FROZEN_SHA=FS['sha']
    if cs:
        orig_read=V.read_stage
        def read_stage(args):
            assert hashlib.sha256(open(args['cands'],'rb').read()).hexdigest()==cs,'candidate file sha256 differs from the given one';return orig_read(args)
        V.read_stage=read_stage
    V.main()
if __name__=='__main__':main()
