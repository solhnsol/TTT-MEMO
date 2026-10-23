"""Extra public-domain background books for the 512-chunk E55 world (cell v3_64_4, E57 section 12 recommendation D). The background of an E55 stream is never repeated (design 13.3), so a 4x longer stream needs about 4x more text than the 4-book pool of v3_16_1. The books below are public-domain novels from Project Gutenberg (English prose, none of the canary books pg11 / pg1342, none of the four v3 pool books); the existing four-book worlds are NOT changed (gen_facts.CELL_BOOKS only extends the pool of the new cell).
  fetch  --dest DIR [--manifest FILE]   download every book (3 URL patterns), cut the Project Gutenberg header / licence footer at the START / END markers, check the title, write DIR/pg<ID>.txt (existing files are kept when their text is identical) and a manifest {id, title, url, bytes, sha256, tokens-free char count}
  verify --dest DIR [--pins FILE]       every book of the list present, titled as expected and (when --pins gives e55_bgbooks_sha.json) hashed as pinned; exit 6 otherwise
  --selftest                           synthetic header / footer trimming, title check, hash check (no network)
Tested offline only (no network access in the code sandbox): the first real run is the fetch on the lab server."""
import sys,json,hashlib,argparse,re,time,urllib.request
from pathlib import Path
BOOKS=[(84,'Frankenstein'),(74,'Tom Sawyer'),(76,'Huckleberry Finn'),(1400,'Great Expectations'),(730,'Oliver Twist'),(158,'Emma'),(161,'Sense and Sensibility'),(121,'Northanger Abbey'),(105,'Persuasion'),(141,'Mansfield Park'),(145,'Middlemarch'),(550,'Silas Marner'),(345,'Dracula'),(174,'Dorian Gray'),(120,'Treasure Island'),(829,"Gulliver"),(2701,'Moby'),(1257,'Three Musketeers'),(36,'War of the Worlds'),(2852,'Hound of the Baskervilles')]
URLS=('https://www.gutenberg.org/cache/epub/{n}/pg{n}.txt','https://www.gutenberg.org/files/{n}/{n}-0.txt','https://www.gutenberg.org/ebooks/{n}.txt.utf-8')
def trim(text):
    t=text.replace('\r\n','\n').replace('\r','\n');m=re.search(r'^\*{3}\s*START OF (?:THE|THIS) PROJECT GUTENBERG EBOOK[^\n]*\n',t,re.M);e=re.search(r'^\*{3}\s*END OF (?:THE|THIS) PROJECT GUTENBERG EBOOK',t,re.M)
    if not m or not e or e.start()<=m.end():return None
    return t[m.end():e.start()].strip()+'\n'
def head_title_ok(text,key):return key.lower() in text[:3000].lower()
def sha(b):return hashlib.sha256(b).hexdigest()
def get(n,log):
    for u in URLS:
        url=u.format(n=n)
        for att in range(3):
            try:
                with urllib.request.urlopen(urllib.request.Request(url,headers={'User-Agent':'Mozilla/5.0 (research fetch; contact: repo owner)'}),timeout=60) as r:b=r.read()
                return url,b.decode('utf-8',errors='replace')
            except Exception as ex:log(f'  {url} attempt {att+1}: {repr(ex)[:90]}');time.sleep(2*(att+1))
    return None,None
def fetch(dest,manifest,books=BOOKS,log=print):
    dest=Path(dest);dest.mkdir(parents=True,exist_ok=True);rows=[];bad=[]
    for n,key in books:
        f=dest/f'pg{n}.txt'
        if f.exists():
            b=f.read_bytes();rows.append(dict(id=n,key=key,url='existing file',bytes=len(b),sha256=sha(b)));log(f'pg{n}: kept existing file ({len(b)} bytes)');continue
        url,raw=get(n,log)
        if raw is None:bad.append((n,'download failed'));continue
        if not head_title_ok(raw,key):bad.append((n,f'title "{key}" not in the first 3000 characters'));continue
        t=trim(raw)
        if t is None or len(t)<150_000:bad.append((n,f'START/END markers missing or text too short ({0 if t is None else len(t)} chars)'));continue
        b=t.encode('utf-8');f.write_bytes(b);rows.append(dict(id=n,key=key,url=url,bytes=len(b),sha256=sha(b)));log(f'pg{n}: {key}: {len(b)} bytes from {url}')
    if bad:log('FAILED: '+'; '.join(f'pg{n}: {w}' for n,w in bad))
    Path(manifest).write_text(json.dumps({'books':rows,'failed':bad,'note':'public-domain Project Gutenberg texts, header / licence cut at the START / END markers; pins go to scripts/exp07_write_ladder/e55_bgbooks_sha.json'},indent=1))
    return rows,bad
def verify(dest,pins=None,books=BOOKS,log=print):
    dest=Path(dest);pin=json.load(open(pins))['sha256'] if pins and Path(pins).exists() else None;bad=[]
    for n,key in books:
        f=dest/f'pg{n}.txt'
        if not f.exists():bad.append((n,'missing'));continue
        b=f.read_bytes()
        if pin is not None and sha(b)!=pin.get(str(n)):bad.append((n,f'sha256 {sha(b)[:12]} != pinned {str(pin.get(str(n)))[:12]}'))
    if pin is None:log('  (no pin file: book hashes not verified, first run)')
    return bad
def selftest():
    body='Chapter I\n\n'+('A long paragraph of plain prose about nothing in particular. '*40+'\n\n')*20
    raw='The Project Gutenberg eBook of Test Novel\n\n*** START OF THE PROJECT GUTENBERG EBOOK TEST NOVEL ***\n\n'+body+'*** END OF THE PROJECT GUTENBERG EBOOK TEST NOVEL ***\n\nLicence text\n'
    t=trim(raw);assert t is not None and 'Licence' not in t and 'START OF' not in t and t.startswith('Chapter I') and head_title_ok(raw,'test novel') and not head_title_ok(raw,'other title')
    assert trim('no markers') is None and trim(raw.replace('*** END','*** NOT'))is None
    import tempfile;d=Path(tempfile.mkdtemp());(d/'pg1.txt').write_bytes(t.encode());bk=[(1,'test novel')];man=d/'m.json'
    rows,bad=fetch(d,man,bk,log=lambda *x:None);assert not bad and rows[0]['sha256']==sha(t.encode())
    pins=d/'p.json';pins.write_text(json.dumps({'sha256':{'1':sha(t.encode())}}));assert verify(d,pins,bk,log=lambda *x:None)==[]
    pins.write_text(json.dumps({'sha256':{'1':'0'*64}}));assert verify(d,pins,bk,log=lambda *x:None)[0][0]==1 and verify(d/'x',pins,bk,log=lambda *x:None)[0][1]=='missing'
    assert len({n for n,_ in BOOKS})==len(BOOKS)==20 and not ({11,1342,98,1260,768,766}&{n for n,_ in BOOKS})
    print('e55_bgbooks selftest ok')
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('cmd',nargs='?',choices=('fetch','verify'));ap.add_argument('--dest',default='data/narrative_gutenberg');ap.add_argument('--manifest',default='e55_bgbooks_manifest.json');ap.add_argument('--pins',default=None);ap.add_argument('--selftest',action='store_true');a=ap.parse_args()
    if a.selftest:selftest();sys.exit(0)
    if a.cmd=='fetch':
        rows,bad=fetch(a.dest,a.manifest,log=lambda *x:print(*x,flush=True));sys.exit(6 if bad else 0)
    if a.cmd=='verify':
        bad=verify(a.dest,a.pins,log=lambda *x:print(*x,flush=True));[print('BAD',n,w) for n,w in bad];sys.exit(6 if bad else 0)
