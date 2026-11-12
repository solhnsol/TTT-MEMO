"""E57 job 1285b: CPU pre-check of the read items and candidate token spans (new file; GPU-free). For each seed the world and the read items are regenerated with the pinned generator (gen_facts via e55_run.build, the call of e57_oracle_run.main) and the pinned tokenizer, and for every item (20 items = 16 fact items and 4 controls) and every one of its 16 candidates (the gold candidate and the 15 wrong candidates, in the order of the item) the token identifiers, the span (start, end), the span length and the gold index are collected into a manifest whose SHA-256 (canonical JSON: sort_keys, separators (',', ':'), ensure_ascii False, UTF-8) is reported. The read rows do not depend on the arm or on K0 (the same rows are scored for every arm), so one manifest per seed covers NATS, NATMASK (K0 = 0) and NATMASK_K08 alike.
  python3 e57_1285_token_manifest.py --seeds 71,...,78 --out manifest.json | --check manifest.json | --selftest
--check recomputes the manifests and compares every hash with the stored file (abort code 14 on a difference). The stored hashes are computed from the pinned code, not from rows stored by job 1283 (job 1283 stored no token identifiers); the link to the 1283 run is the pinned code and tokenizer hashes (versions.json of 1283) plus the bit identical W0 scores of all items in the out and in_g0 reads checked by the job."""
import sys,os,json,hashlib,argparse,pathlib
HERE=pathlib.Path(__file__).resolve().parent;ROOT=HERE.parents[1]
for p in (HERE,HERE/'e57',ROOT/'scripts/exp04_what_ttt_retains',ROOT/'scripts/exp05_write_dynamics'):sys.path.insert(0,str(p))
def sha(o):return hashlib.sha256(json.dumps(o,sort_keys=True,separators=(',',':'),ensure_ascii=False).encode('utf-8')).hexdigest()
def manifest(seed,cell='v3_16_1',read_apps=(16,32,64,128)):
    import e55_run as R55,e57_syn_pass as SP,e55_read as R,stage2
    from tokenizers import Tokenizer
    a=SP.synth_args(seed,cell);enc,w,tf=R55.build(a,seed,read_apps=set(read_apps),arms=['NAT']);P=R.plan(w,enc);items=P['items'];tok=Tokenizer.from_file(tf);slot=stage2.make_slot(tok,None);rows=[]
    for k,it in enumerate(items):
        cands=[]
        for c in it['cands']:
            ids,span=stage2.item_ids(tok,slot,it['prefix'],c);cands.append(dict(text=c,ids=[int(x) for x in ids],span=[int(span[0]-1),int(span[1]-1)],length=int(span[1]-span[0])))
        rows.append(dict(item=k,kind=it['kind'],fact=it['fact'],group=it['group'],qframe=it['qframe'],gold_idx=int(it['gold_idx']),prefix=it['prefix'],cands=cands))
    return dict(seed=seed,cell=cell,n_items=len(rows),n_cands=sorted({len(r['cands']) for r in rows}),span_lengths=sorted({c['length'] for r in rows for c in r['cands']}),tokenizer_sha256=hashlib.sha256(open(tf,'rb').read()).hexdigest(),manifest_sha256=sha(rows))
def run(seeds):return [manifest(s) for s in seeds]
def selftest():
    a=sha([{'x':1,'y':[2,3]}]);b=sha([{'y':[2,3],'x':1}]);assert a==b and a!=sha([{'x':1,'y':[3,2]}])
    m=manifest(71);assert m['n_items']==20 and m['n_cands']==[16] and len(m['span_lengths'])==1 and len(m['manifest_sha256'])==64
    assert manifest(71)['manifest_sha256']==m['manifest_sha256'] and manifest(72)['manifest_sha256']!=m['manifest_sha256']
    print('e57_1285_token_manifest selftest ok')
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--seeds');ap.add_argument('--out');ap.add_argument('--check');ap.add_argument('--selftest',action='store_true');a=ap.parse_args()
    if a.selftest:selftest()
    elif a.check:
        ref=json.load(open(a.check));new={m['seed']:m for m in run([m['seed'] for m in ref])};bad=[m['seed'] for m in ref if new[m['seed']]['manifest_sha256']!=m['manifest_sha256'] or new[m['seed']]['tokenizer_sha256']!=m['tokenizer_sha256']]
        print('token manifest check:','OK' if not bad else f'DIFFERENT for seeds {bad}');sys.exit(0 if not bad else 14)
    else:
        out=run([int(x) for x in a.seeds.split(',')]);print(json.dumps(out,indent=1))
        if a.out:json.dump(out,open(a.out,'w'),indent=1)
