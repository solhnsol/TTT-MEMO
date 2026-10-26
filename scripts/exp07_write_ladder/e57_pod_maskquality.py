"""E57 pod masks: mask quality against the ORACLE positions (CPU only; reads e57_pod_masks.npz of the pod run, no GPU). Oracle = the ORCA masks (rank 5: sentence fragments holding the gold of ANY bank question, window and current chunk). For the masks C1 (rank 1), LRN (rank 2) and the controls RC1 (3), RLRN (4), RORCA (6) over the write chunks k = 8 .. 279 (full window): token-level recall = hidden oracle tokens / oracle tokens, precision = hidden oracle tokens / hidden tokens, masked fraction, F1 (recall, precision), pooled and per event (the event of the article the fragment lies in), reported for the WINDOW part and the CURRENT CHUNK part separately. Chance recall of a mask = its masked fraction.
  python3 e57_pod_maskquality.py --bundle B --masks e57_pod_masks.npz --out OUT.json"""
import sys,json,argparse,bisect,pathlib,collections
import numpy as np
HERE=pathlib.Path(__file__).resolve().parent
for p in (HERE,HERE/'e57',HERE.parents[1]/'scripts/exp04_what_ttt_retains',HERE.parents[1]/'scripts/exp05_write_dynamics'):sys.path.insert(0,str(p))
CH=1024;NW=8
RANKS={'C1':1,'LRN':2,'RC1':3,'RLRN':4,'RORCA':6}
def f1(r,p):return None if (r is None or p is None or r+p==0) else 2*r*p/(r+p)
def stats(c):
    r=c['tp']/c['o'] if c['o'] else None;p=c['tp']/c['m'] if c['m'] else None
    return dict(oracle_tokens=c['o'],masked_tokens=c['m'],hit_tokens=c['tp'],tokens=c['t'],recall=r,precision=p,masked_fraction=c['m']/c['t'] if c['t'] else None,f1=f1(r,p),oracle_fraction=c['o']/c['t'] if c['t'] else None)
def quality(masks,ends,iv_starts,iv_events,UVm,ks,ranks=RANKS,oracle_rank=5):
    """-> {mask: {part: {'pooled': stats, 'events': {event: stats}}}}"""
    acc={m:{part:collections.defaultdict(lambda:collections.Counter()) for part in('window','chunk')} for m in ranks}
    for k in ks:
        wsid,wst,wlen=UVm.window_sents(ends,k,CH,NW);q=UVm.chunk_sents(ends,k,CH);base=(k-NW)*CH
        parts={'window':(np.asarray(wlen),[base+int(s) for s in wst],masks[f'h{oracle_rank}_{k}']),'chunk':(np.asarray([b-a for a,b in q]),[k*CH+a for a,b in q],masks[f'ch{oracle_rank}_{k}'])}
        for m,r in ranks.items():
            for part,(ln,starts,orc) in parts.items():
                mk=np.asarray(masks[f'h{r}_{k}' if part=='window' else f'ch{r}_{k}'],bool);orc=np.asarray(orc,bool)
                for i in range(len(ln)):
                    ev=iv_events[max(0,bisect.bisect_right(iv_starts,starts[i]%TOTAL[0])-1)]
                    for key in ('pooled',ev):
                        c=acc[m][part][key];c['t']+=int(ln[i]);c['o']+=int(ln[i]) if orc[i] else 0;c['m']+=int(ln[i]) if mk[i] else 0;c['tp']+=int(ln[i]) if (orc[i] and mk[i]) else 0
    out={}
    for m in ranks:
        out[m]={}
        for part in('window','chunk'):
            d=acc[m][part];out[m][part]=dict(pooled=stats(d['pooled']),events={e:stats(c) for e,c in sorted(d.items()) if e!='pooled'})
    return out
TOTAL=[None]
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--bundle');ap.add_argument('--masks');ap.add_argument('--out');ap.add_argument('--bank');ap.add_argument('--tokenizer');a=ap.parse_args()
    ROOT=HERE.parents[1];import gen_world_ctl as G,e55_uv as UVm,e57_reads as RD
    tf=a.tokenizer or str(ROOT/'data/tokenizer/llama3/tokenizer.json');enc=G.load_enc(tf)
    bank=pathlib.Path(a.bank) if a.bank else pathlib.Path(a.bundle)/'e57_bank'/'bank_v1.json';items=RD.load_items(bank);ids,iv,hits=RD.article_intervals(a.bundle,enc,items)
    ends=UVm.sentence_ends(UVm.tokenizer(tf),ids);masks=dict(np.load(a.masks));TOTAL[0]=len(ids);ks=[k for k in range(NW,len(ids)//CH) if f'h5_{k}' in masks]
    res=quality(masks,ends,[x[0] for x in iv],[x[2] for x in iv],UVm,ks)
    ref={'synthetic_world_seed61 (labels = fact sentences, not answer sentences; different truth, window only)':{'cand1':{'recall':0.90,'precision':0.53,'masked_fraction':0.61},'learned':{'recall':0.90,'masked_fraction':0.324}}}
    json.dump(dict(chunks=f'{ks[0]}..{ks[-1]} ({len(ks)} write chunks)',oracle='ORCA masks (rank 5: fragment holds the gold of any bank question)',synthetic_reference=ref,quality=res),open(a.out,'w'),indent=1,default=float)
    for m in res:
        for part in('window','chunk'):
            s=res[m][part]['pooled'];print(f"{m:6s} {part:6s} recall {s['recall']:.3f} precision {s['precision']:.3f} masked {s['masked_fraction']:.3f} f1 {s['f1']:.3f} (oracle fraction {s['oracle_fraction']:.3f})")
if __name__=='__main__':main()
