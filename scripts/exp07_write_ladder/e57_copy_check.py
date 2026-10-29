"""E57 can the target be copied from the window? (CPU only; new file; numbers only). For a row with query token position q (input token ids[q], target ids[q+1]; label and band-512 gain from e57_gain_labels_rows.npz) the n-gram (ids[q-2], ids[q-1], ids[q], ids[q+1]) = the previous 3-gram INCLUDING the query token plus the target is searched in the stream before q: a match at position j (the target-token position of the earlier occurrence, j <= q) is visible through its key j-1 (the token before the copy source), at distance d = q - (j-1). Zones (SWA window 8192, band 512): BAND = 1 <= d <= 511; FAR = 512 <= d <= 8191 (inside the window, outside the band); NONE otherwise. A row is BAND if it has a BAND match, else FAR if it has a FAR match, else NONE. Also with shorter patterns: M2 = (query token, target), M1 = target token alone. Groups: all eligible rows, top 1 % and top 10 % of the band-512 gain, the 40 samples of e57_gain_samples.json; for the 4-gram also per label (A / S / SENT / OTHER); for rows with FAR matches the number of FAR matches (distribution) and whether the matches are in articles of the same event (fraction of FAR matches in the same event, share of rows whose FAR matches are all / any in the same event).
  .venv/bin/python scripts/exp07_write_ladder/e57_copy_check.py --bundle B --rows R.npz --samples S.json --out O.json"""
import sys,json,pathlib,argparse,collections
import numpy as np
HERE=pathlib.Path(__file__).resolve().parent
for p in (HERE,HERE/'e57',HERE.parents[1]/'scripts/exp04_what_ttt_retains',HERE.parents[1]/'scripts/exp05_write_dynamics'):sys.path.insert(0,str(p))
NAMES=['OTHER','SENT','S','A']
def build(ids,n):
    d=collections.defaultdict(list)
    for j in range(n-1,len(ids)):d[tuple(int(x) for x in ids[j-n+1:j+1])].append(j)
    return {k:np.asarray(v) for k,v in d.items()}
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--bundle');ap.add_argument('--rows');ap.add_argument('--samples');ap.add_argument('--out');ap.add_argument('--tokenizer');a=ap.parse_args()
    import gen_world_ctl as G,e57_reads as RD
    enc=G.load_enc(a.tokenizer or str(HERE.parents[1]/'data/tokenizer/llama3/tokenizer.json'))
    items=RD.load_items(pathlib.Path(a.bundle)/'e57_bank'/'bank_v1.json');ids,iv,hits=RD.article_intervals(a.bundle,enc,items);ids=np.asarray(ids);N=len(ids)
    starts=np.array([x[0] for x in iv]);evs=np.array([hash(x[2])%10**9 for x in iv])
    ev_of=lambda p:evs[np.searchsorted(starts,p,side='right')-1]
    R=np.load(a.rows);pos=R['pos'];lab=R['label'];g=R['gain512'];n=len(pos);q=pos-1
    order=np.argsort(-g,kind='stable');top1=np.zeros(n,bool);top1[order[:int(round(.01*n))]]=True;top10=np.zeros(n,bool);top10[order[:int(round(.10*n))]]=True
    S=json.load(open(a.samples))['samples'];sp={s['target_position'] for s in S};samp=np.isin(pos,list(sp))
    res={};zone={}
    for nn,name in ((4,'M4'),(2,'M2'),(1,'M1')):
        D=build(ids,nn);nb=np.zeros(n,int);nf=np.zeros(n,int);same_all=np.zeros(n,bool);same_any=np.zeros(n,bool);nfar_same=np.zeros(n,int)
        for r in range(n):
            qq_=int(q[r]);key=tuple(int(x) for x in ids[qq_+1-nn+1:qq_+2]) if nn>1 else (int(ids[qq_+1]),)
            js=D.get(key)
            if js is None:continue
            # exclude the row's own occurrence (j = q+1) and anything at or after q+1
            js=js[js<=qq_]
            if len(js)==0:continue
            band=js[(js>=qq_-510)];far=js[(js<=qq_-511)&(js>=qq_-8190)]
            nb[r]=len(band);nf[r]=len(far)
            if len(far):
                e=ev_of(far-1)==ev_of(qq_);same_all[r]=bool(e.all());same_any[r]=bool(e.any());nfar_same[r]=int(e.sum())
        z=np.where(nb>0,0,np.where(nf>0,1,2));zone[name]=dict(z=z,nf=nf,nb=nb,same_all=same_all,same_any=same_any,nfar_same=nfar_same)
    def frac(m,z):
        t=int(m.sum());return {k:(int(((z==i)&m).sum()),float(((z==i)&m).sum()/t) if t else None) for i,k in enumerate(('BAND','FAR','NONE'))}|{'rows':t}
    groups={'ALL':np.ones(n,bool),'TOP1pct':top1,'TOP10pct':top10,'SAMPLES40':samp}
    out={'definition':__doc__.split('\n')[0],'rows':n}
    for name in ('M4','M2','M1'):
        z=zone[name]['z'];out[name]={gn:frac(m,z) for gn,m in groups.items()}
    out['M4_by_label']={gn:{NAMES[i]:frac(m&(lab==i),zone['M4']['z']) for i in range(4)} for gn,m in groups.items() if gn in ('ALL','TOP1pct')}
    # FAR details for M4
    Z=zone['M4'];det={}
    for gn,m in groups.items():
        f=m&(Z['z']==1);cnt=Z['nf'][f]
        if f.sum()==0:det[gn]=None;continue
        b=lambda lo,hi:int(((cnt>=lo)&(cnt<=hi)).sum())
        det[gn]=dict(rows_far=int(f.sum()),count_distribution={'1':b(1,1),'2':b(2,2),'3-5':b(3,5),'6-10':b(6,10),'>10':b(11,10**9)},median_count=float(np.median(cnt)),share_far_matches_in_same_event=float(Z['nfar_same'][f].sum()/cnt.sum()),rows_all_matches_same_event=float(Z['same_all'][f].mean()),rows_any_match_same_event=float(Z['same_any'][f].mean()))
    out['M4_far_details']=det
    sam=[]
    for s in S:
        r=int(np.nonzero(pos==s['target_position'])[0][0]);sam.append(dict(stratum=s['stratum'],chunk=s['chunk'],row=s['row'],label=s['label'],gain=s['gain'],zone4=('BAND','FAR','NONE')[int(Z['z'][r])],zone2=('BAND','FAR','NONE')[int(zone['M2']['z'][r])],zone1=('BAND','FAR','NONE')[int(zone['M1']['z'][r])],far_matches4=int(Z['nf'][r]),band_matches4=int(Z['nb'][r])))
    out['samples']=sam;json.dump(out,open(a.out,'w'),indent=1)
    for name in ('M4','M2','M1'):
        print(name,{gn:(v['rows'],{k:round(v[k][1],3) if v[k][1] is not None else None for k in ('BAND','FAR','NONE')}) for gn,v in out[name].items()})
    print('M4 by label top1%:',{k:(v['rows'],{z:round(v[z][1],3) if v[z][1] is not None else None for z in ('BAND','FAR','NONE')}) for k,v in out['M4_by_label']['TOP1pct'].items()})
    print('M4 by label all:',{k:(v['rows'],{z:round(v[z][1],3) if v[z][1] is not None else None for z in ('BAND','FAR','NONE')}) for k,v in out['M4_by_label']['ALL'].items()})
    print(json.dumps(det)[:1200])
if __name__=='__main__':main()
