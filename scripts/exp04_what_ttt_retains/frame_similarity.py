"""E55 v3 (design 13.1): semantic pairwise similarity of the frame bank. CPU: small sentence-embedding model (all-MiniLM-L6-v2 through fastembed / onnxruntime; model id and version printed), cosine of every pair of frames.
Variants (reported separately): 'filled' = slots filled with one fixed neutral name / place / occupation (the primary, preregistered one: the share of pairs with cosine > 0.9 must be < 1%), 'slotless' = the slots deleted (only the carrier wording is compared).
  python3 frame_similarity.py [--frames frames300.txt] [--out report.md] [--selftest]"""
import sys,argparse,json,hashlib
from pathlib import Path
import numpy as np
HERE=Path(__file__).resolve().parent
MODEL='sentence-transformers/all-MiniLM-L6-v2';THR=0.9;PASS_SHARE=0.01
def fill(t,mode):
    if mode=='filled':return t.replace('{X}','Marta Velkov').replace('{P}','Orlenhaven').replace('{a_o}','a cooper').replace('{o}','cooper')
    import re;return re.sub(r'\s+',' ',re.sub(r"\{\w+\}('s)?",'',t)).strip()
def embed(texts):
    from fastembed import TextEmbedding
    m=TextEmbedding(MODEL);e=np.array(list(m.embed(texts)),dtype=np.float64);return e/np.linalg.norm(e,axis=1,keepdims=True)
def pairwise(frames,mode):
    E=embed([fill(t,mode) for t in frames]);C=E@E.T;iu=np.triu_indices(len(frames),1);v=C[iu];order=np.argsort(-v)[:10]
    return {'mode':mode,'n_frames':len(frames),'n_pairs':int(len(v)),'mean':float(v.mean()),'median':float(np.median(v)),'p95':float(np.percentile(v,95)),'p99':float(np.percentile(v,99)),'max':float(v.max()),'share_gt_0.9':float((v>THR).mean()),'n_gt_0.9':int((v>THR).sum()),
            'share_gt_0.95':float((v>0.95).mean()),'top_pairs':[(int(iu[0][i])+1,int(iu[1][i])+1,float(v[i])) for i in order],'pass':bool((v>THR).mean()<PASS_SHARE)}
def report(frames):
    L=[f'# E55 frame bank: semantic pairwise cosine ({len(frames)} frames, model {MODEL}, fastembed; criterion fixed before the result: share of pairs with cosine > {THR} below {PASS_SHARE:.0%})','']
    res=[pairwise(frames,m) for m in ('filled','slotless')]
    L+=['| variant | pairs | mean | median | p95 | p99 | max | pairs > 0.9 | pairs > 0.95 | criterion |','|---|---|---|---|---|---|---|---|---|---|']
    for r in res:L.append(f"| {r['mode']} | {r['n_pairs']} | {r['mean']:.3f} | {r['median']:.3f} | {r['p95']:.3f} | {r['p99']:.3f} | {r['max']:.3f} | {r['n_gt_0.9']} ({r['share_gt_0.9']:.2%}) | {r['share_gt_0.95']:.2%} | {'PASS' if r['pass'] else 'FAIL'} |")
    for r in res:
        L+=['',f"top 10 pairs ({r['mode']}): "+', '.join(f'({a},{b}) {c:.3f}' for a,b,c in r['top_pairs'])]
    L+=['','frames sha12 '+hashlib.sha256('\n'.join(frames).encode()).hexdigest()[:12]]
    return '\n'.join(L)+'\n',res
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--frames',default=str(HERE/'frames300.txt'));ap.add_argument('--out',default=None);ap.add_argument('--selftest',action='store_true');a=ap.parse_args()
    fr=[l for l in Path(a.frames).read_text().split('\n') if l]
    if a.selftest:fr=fr[:20]
    t,res=report(fr);print(t)
    if a.out:Path(a.out).write_text(t)
