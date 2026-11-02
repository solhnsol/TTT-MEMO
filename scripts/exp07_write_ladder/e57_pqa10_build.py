"""E57 public QA stage 10 (E57 122-Z / 123), unrelated-text control stream builder (new file, CPU): the 33 paragraph-group episodes of job 640 with the SAME structure - same group split, same section boundaries (token ends of the 9 sections), same chunk count, same fixed filler completion of the last chunk, same read chunks / spill tokens, same questions - but the written text of every section is UNRELATED SQuAD train-v1.1 text: paragraphs of documents that are not dev documents, whose text contains none of the strings that matter for the items455 (the gold strings, their variants, and every candidate of length >= 4 that contains a letter; case-insensitive, between non-alphanumeric characters). The paragraphs of a group are drawn in a group-seeded random order from the allowed pool, without reuse inside the group (groups are independent episodes from W0, so they may share paragraphs; the allowed pool is only ~1,400 paragraphs / ~175k tokens); sentences are encoded as in e57_pqa2_stream.py ("\n\n" + first sentence of a paragraph, " " + later ones) and the last sentence of a section is cut at the token level so that every section end equals the section end token of job 640's group stream (hence the same amount of text and the same read chunks).
Outputs in <out>: e57_pqa6_group_ids.npz, e57_pqa6_groups.json (the file names e57_pqa6_run.py reads: the write / read / analyze stages of e57_pqa6_run.py run on this directory unchanged), e57_pqa10_build_meta.json (pool size, excluded fraction, text hash, paragraphs used).
  python3 e57_pqa10_build.py --src640 D --items-file F --paragraphs P --train T --dev DEV --out O | --selftest"""
import sys,json,re,hashlib,pathlib
import numpy as np
HERE=pathlib.Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
import e57_pqa2_stream as S
import e57_pqa3_stream as S3
import e57_pqa6_run as P6
CH=1024;SEEDB=20261103
def bad_strings(items_raw,keep_ids):
    out=set()
    for it in items_raw:
        if it['id'] not in keep_ids:continue
        for s in [it['gold']]+list(it.get('variants',[]))+[c for c in it['candidates'] if len(c)>=4 and re.search('[A-Za-z]',c)]:
            if s.strip():out.add(s.strip().lower())
    return out
def allowed_pool(train,dev,bad):
    dt={S3.tsha(a['title']) for a in dev['data']};pool=[]
    rx=re.compile(r'(?<![a-z0-9])('+'|'.join(re.escape(s) for s in sorted(bad,key=len,reverse=True))+r')(?![a-z0-9])')
    n_all=0
    for a in train['data']:
        if S3.tsha(a['title']) in dt:continue
        for p in a['paragraphs']:
            n_all+=1;t=' '.join(p['context'].split())
            if len(t.split())>=40 and not rx.search(t.lower()):pool.append(t)
    return pool,n_all
def build(src640,items_f,para_f,train_f,dev_f,out):
    out=pathlib.Path(out);out.mkdir(parents=True,exist_ok=True);meta,ids_z=P6.load_groups(src640);raw=json.load(open(items_f))['items'];keep={i for g in meta['groups'] for i in g['item_ids']}
    train=json.load(open(train_f));dev=json.load(open(dev_f));bad=bad_strings(raw,keep);pool,n_all=allowed_pool(train,dev,bad);tk=S.tokenizer()
    fm,_,_=S3.filler_paragraphs(train,dev,332,S3.SEED);fil_texts={t for _,t in fm[:24]};pool=[p for p in pool if p not in fil_texts];ids_all={};groups=[];used=0;tot_tokens=0;used_ids=[]
    for g in meta['groups']:
        ids_g=ids_z[f'g{g["gid"]}'];ends=g['section_end_tokens'];new=[];start=0;order=np.random.default_rng(SEEDB+g['gid']).permutation(len(pool));pos=0
        for v in range(9):
            budget=ends[v]-start;sec=[]
            while len(sec)<budget:
                text=pool[order[pos]];pos+=1;used+=1;used_ids.append(int(order[pos-1]))
                for s,x in S3.enc_par(tk,text):sec+=x
            new+=sec[:budget];start=ends[v]
        assert len(new)==ends[8];tail=list(ids_g[ends[8]:]);full=np.asarray(new+tail,np.int64);assert len(full)==len(ids_g)
        ids_all[f'g{g["gid"]}']=full;gg=dict(g);gg['ids_sha']=hashlib.sha256(full.tobytes()).hexdigest()[:16];groups.append(gg);tot_tokens+=len(full)
    np.savez_compressed(out/'e57_pqa6_group_ids.npz',**ids_all);m=dict(meta);m['groups']=groups;m['control']='unrelated SQuAD train text, same section ends as job 640';(out/'e57_pqa6_groups.json').write_text(json.dumps(m))
    h=hashlib.sha256(json.dumps([pool[i] for i in used_ids]).encode()).hexdigest()
    (out/'e57_pqa10_build_meta.json').write_text(json.dumps(dict(seed=SEEDB,n_train_paragraphs_non_dev=n_all,n_pool_allowed=len(pool),excluded_share=1-len(pool)/max(1,n_all),n_bad_strings=len(bad),paragraphs_used=used,text_sha256=h,tokens=tot_tokens)));print(json.dumps(dict(n_all=n_all,pool=len(pool),excluded=round(1-len(pool)/n_all,3),used=used,bad=len(bad),tokens=tot_tokens)));return m
def selftest():
    raw=[{'id':'a','gold':'Paris','variants':['paris'],'candidates':['Paris','1','London bridge']},{'id':'b','gold':'zzz','variants':[],'candidates':[]}]
    b=bad_strings(raw,{'a'});assert 'paris' in b and 'london bridge' in b and '1' not in b and 'zzz' not in b
    train={'data':[{'title':'T0','paragraphs':[{'context':'word '*50},{'context':'Paris '+'x '*50}]}]};dev={'data':[{'title':'D','paragraphs':[]}]}
    pool,n=allowed_pool(train,dev,b);assert n==2 and len(pool)==1
    print('e57_pqa10_build selftest ok')
def main():
    if '--selftest' in sys.argv:selftest();return
    g=lambda k:sys.argv[sys.argv.index(k)+1]
    build(g('--src640'),g('--items-file'),g('--paragraphs'),g('--train'),g('--dev'),g('--out'))
if __name__=='__main__':main()
