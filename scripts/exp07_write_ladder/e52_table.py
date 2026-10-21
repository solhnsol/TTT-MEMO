"""E52 per-person table (CPU, real tokenizer on the server): for each of the 60 persons of the Level-2 stream (default 9201, layouts L2a / L2b): pair, condition in each layout, number of exposures, consecutive gaps (tokens), whether every earlier exposure lies in an earlier 1024-token chunk, last-exposure position in each layout and their difference, s_cross / s_same / n_out strata. Every row carries PASS / FAIL against the E52 rules: C0 1 exposure; C1 4 exposures, every gap <= 4096 and in different chunks; C2 4 exposures, every gap >= 12288; last exposure difference <= 150 tokens; conditions flipped between layouts (C0 equal).
Usage: e52_table.py --nat-dir DIR [--stream 9201] [--tokenizer-file F] [--require-real] [--out FILE.tsv] | --selftest.  Exit 1 on any FAIL. No GPU."""
import os,sys,json,argparse
from pathlib import Path
HERE=Path(__file__).resolve().parent;sys.path.insert(0,str(HERE))
import nat_world as NW,nat_pass as NP,nat_filler as NF,e51_nat_table as T
CH=1024;C1_MAX=4096;C2_MIN=12288;DL_MAX=150
def person_rows(w,enc_offs):
    text,_,spans=NW.realize(w,False);offs=enc_offs(text);sp=NP.tok_spans(spans,[b for a,b in offs]);by={}
    for s in sp:
        if s['label']=='fact':by.setdefault(s['t'],[]).append(s['tok_start'])
    st=NP.strata_real(sp);return {t:sorted(v) for t,v in by.items()},st,len(offs)+1
def table(wa,wb,enc_offs):
    pa,sa,na=person_rows(wa,enc_offs);pb,sb,nb=person_rows(wb,enc_offs);ca=wa['insert']['cond'];cb=wb['insert']['cond'];rows=[];nfail=0
    for t in sorted(pa):
        p=t//2;a,b=pa[t],pb[t];res=[]
        def chk(pos,c,tag):
            g=[y-x for x,y in zip(pos,pos[1:])];cross=all(x//CH!=y//CH for x,y in zip(pos,pos[1:]))
            if c=='C0':ok=len(pos)==1
            elif c=='C1':ok=len(pos)==4 and all(x<=C1_MAX for x in g) and cross
            else:ok=len(pos)==4 and all(x>=C2_MIN for x in g)
            if not ok:res.append(f'{tag} rule')
            return g,cross
        ga,xa=chk(a,ca[str(p)],'L2a');gb,xb=chk(b,cb[str(p)],'L2b');dl=abs(a[-1]-b[-1])
        if dl>DL_MAX:res.append('last-exposure diff')
        if (ca[str(p)]=='C0')!=(cb[str(p)]=='C0') or (ca[str(p)]!='C0' and ca[str(p)]==cb[str(p)]):res.append('condition not flipped')
        nfail+=bool(res)
        rows.append(dict(person=t,pair=p,cond_L2a=ca[str(p)],cond_L2b=cb[str(p)],n_expo=len(a),gaps_L2a=ga,gaps_L2b=gb,prior_in_earlier_chunk_L2a=xa,prior_in_earlier_chunk_L2b=xb,last_L2a=a[-1],last_L2b=b[-1],last_diff=dl,strata_L2a=(sa[t]['s_cross'],sa[t]['s_same'],sa[t]['n_out']),strata_L2b=(sb[t]['s_cross'],sb[t]['s_same'],sb[t]['n_out']),status='PASS' if not res else 'FAIL: '+', '.join(res)))
    return rows,nfail,(na,nb)
def fmt(rows):
    h=['person','pair','L2a','L2b','n','gaps L2a','gaps L2b','earlier-chunk a/b','last a','last b','diff','strata a (cross,same,out)','strata b','status'];L=['\t'.join(h)]
    for r in rows:L.append('\t'.join(str(x) for x in (r['person'],r['pair'],r['cond_L2a'],r['cond_L2b'],r['n_expo'],r['gaps_L2a'],r['gaps_L2b'],f"{r['prior_in_earlier_chunk_L2a']}/{r['prior_in_earlier_chunk_L2b']}",r['last_L2a'],r['last_L2b'],r['last_diff'],r['strata_L2a'],r['strata_L2b'],r['status'])))
    return '\n'.join(L)
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--nat-dir',default=os.environ.get('NAT_DIR'));ap.add_argument('--stream',type=int,default=9201);ap.add_argument('--tokenizer-file',default=None);ap.add_argument('--require-real',action='store_true');ap.add_argument('--out',default=None);ap.add_argument('--selftest',action='store_true');a=ap.parse_args()
    import nat_dryrun as D
    try:tk=D.load_tok(a.tokenizer_file)
    except Exception as e:print('real tokenizer not usable:',repr(e)[:100]);tk=None
    assert tk is not None or not a.require_real,'--require-real: no real tokenizer'
    enc=lambda text:T.encode_offs(tk,text)
    if a.selftest:
        import e52_insert as E;E.selftest();return
    nd=T.find_nat_dir(a.nat_dir);wa=NW.load_world(NF.world_path(nd,a.stream,'L2a'));wb=NW.load_world(NF.world_path(nd,a.stream,'L2b'))
    rows,nfail,(na,nb)=table(wa,wb,enc);out=fmt(rows);print(out);print(f'tokenizer: {"REAL" if tk is not None else "PROXY"}; stream tokens L2a {na}, L2b {nb}; persons {len(rows)}; conditions L2a: '+str({c:sum(r["cond_L2a"]==c for r in rows) for c in ("C0","C1","C2")})+f'; FAIL rows {nfail}')
    if a.out:Path(a.out).write_text(out+'\n')
    sys.exit(1 if nfail else 0)
if __name__=='__main__':main()
