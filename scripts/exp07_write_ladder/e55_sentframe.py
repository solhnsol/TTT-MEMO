"""E55 32.12 frame labels of the sentence dump (CPU only, no GPU, no model): regenerates the world of --seed (cell v3_16_1, as the sentdump job's --check-only path does) and writes, per dump chunk, the sentence-frame id of every window sentence (majority frame of its fact tokens, -1 = background) next to the fact id, so that the learned head weighting can be evaluated leave-one-frame-out. Asserts that the regenerated fact ids equal the saved label file.
  python3 e55_sentframe.py --seed 61 --out DIR [--chunks 16,32,64,127]   ->  DIR/e55_sentdump_frames_s<seed>.npz  (keys c<k>/w_frame, c<k>/q_frame)"""
import sys,argparse
from pathlib import Path
import numpy as np
HERE=Path(__file__).resolve().parent;sys.path.insert(0,str(HERE))
def token_attr(meta,c,key,CH=1024):
    f=np.full(CH,-1,np.int16)
    for x in meta[c]['sents']:f[x['start']:x['end']]=x[key]
    return f
def sent_attr(tf,tfact,a,b):
    """majority value of `tf` over the fact tokens of [a,b), same >= 0.5 fact-token share rule as the label file (-1 = background)"""
    t=tfact[a:b];fact=t[t>=0]
    if len(fact)==0 or len(fact)/max(b-a,1)<0.5:return -1
    v=tf[a:b][t>=0];vals,cnt=np.unique(v,return_counts=True);return int(vals[np.argmax(cnt)])
def frames(meta,ends,k,SD,CH=1024,nw=8):
    q,wsid,wst,wlen=SD.layout(ends,k,CH,nw);tq=token_attr(meta,k,'fact');fq=token_attr(meta,k,'frame');out={'q_frame':np.asarray([sent_attr(fq,tq,a,b) for a,b in q],np.int16)}
    wf=[];wfact=[]
    for s,n in zip(wst,wlen):
        c=k-nw+int(s)//CH;o=int(s)-(c-k+nw)*CH;tf_=token_attr(meta,c,'frame');tt=token_attr(meta,c,'fact');wf.append(sent_attr(tf_,tt,o,o+int(n)));wfact.append(sent_attr(tt,tt,o,o+int(n)))
    out['w_frame']=np.asarray(wf,np.int16);out['w_fact_check']=np.asarray(wfact,np.int16);return out
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--seed',type=int,required=True);ap.add_argument('--out',required=True);ap.add_argument('--chunks',default='16,32,64,127');ap.add_argument('--stub',action='store_true');a=ap.parse_args()
    import e55_run as R,e55_uv as UVm,e55_sentdump as SD
    ns=argparse.Namespace(seed=a.seed,cell='v3_16_1',bg='pool',tokenizer_file=None,arms='NAT',decoy='names',stub=a.stub)
    enc,w,tf=R.build(ns,a.seed,arms=['NAT']);S=R.streams(w,enc);ids,_=S['A']
    ends=UVm.sentence_ends(UVm.tokenizer(tf),ids) if not a.stub else UVm.sentence_ends(type('T',(),{'id_to_token':staticmethod(lambda i:'.' if i%11==0 else 'w')})(),ids)
    out={};lbf=Path(a.out)/f'e55_sentdump_labels_s{a.seed}.npz';ref=np.load(lbf) if lbf.exists() else None
    for k in [int(x) for x in a.chunks.split(',')]:
        fr=frames(w['chunks'],ends,k,SD)
        if ref is not None:assert np.array_equal(fr['w_fact_check'],ref[f'c{k}/w_fact']),('regenerated fact ids differ from the saved labels',k)
        out[f'c{k}/w_frame']=fr['w_frame'];out[f'c{k}/q_frame']=fr['q_frame'];nf=(fr['w_frame']>=0).sum();print(f'chunk {k}: {len(fr["w_frame"])} window sentences, {nf} fact sentences, {len(set(fr["w_frame"][fr["w_frame"]>=0].tolist()))} distinct frames')
    np.savez(Path(a.out)/f'e55_sentdump_frames_s{a.seed}.npz',**out);print('wrote frames, fact ids verified against the saved labels' if ref is not None else 'wrote frames (no label file to verify)')
