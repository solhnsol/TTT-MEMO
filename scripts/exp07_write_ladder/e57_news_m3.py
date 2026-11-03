"""E57 news stream A: M3 write arm with the ALLQ128 mask and a loss weight on the deciding rows (new file; GPU; thread 'ALLQ보다 나은 가림 방법 찾기' / method thread, Hans '실행' 11/03 23:33 KST; one cycle of stream A = 280 chunks, 1 update per chunk): the write of E57_v2A_ALLQ128 (e57_t1t2v2_run.write_stage: engine SELC1, Adam direction, update norm 0.5 x the per-chunk norm of the plain NAT write E57_main_NAT, ALLQ128 from chunk 8 = every query row sees only the previous 128 tokens, window AND chunk keys) with the loss of the deciding rows weighted by w (e57_syn_mrun.install_weighted_grad); deciding rows = the top 30% of the eligible rows (loss rows whose next token is inside a sentence) of every chunk by the window gain g = nll_band128 - nll_win of the stored W0 pass (e57_band/e57_band_W0.npz, chunks 8..279). The state of the last chunk is saved and the canary (generation NLL, tail NLL) is stored at chunk 279.
  python3 e57_news_m3.py --stage masks --out O | --stage write --out O --sentences S.jsonl --nat-ref E57_main_NAT.arms.json [--weight 8] | --selftest"""
import sys,json,time,gc,hashlib,pathlib,argparse
import numpy as np
HERE=pathlib.Path(__file__).resolve().parent;ROOT=HERE.parents[1]
for p in (HERE,HERE/'e57',ROOT/'scripts/exp04_what_ttt_retains',ROOT/'scripts/exp05_write_dynamics',ROOT/'e2e'):sys.path.insert(0,str(p))
EXP=ROOT/'experiments/exp07_write_ladder'
import e57_closure_masks as CM
CH=1024;WIN=8*CH;K0=8;P=0.30;BAND=128
def deciding(gain,elig,p=P):
    n=int(round(p*elig.sum()));return CM.top_rows(gain,elig,n)
class UVNewsM3:
    """UV-compatible keep object: ALLQ128 for every row from chunk K0; publishes the deciding-row vector of the chunk in holder['dec'] for the weighted gradient"""
    def __init__(self,masks,holder,log=print):self.m=masks;self.holder=holder;self.log=log
    def keep(self,k,m,Kc,Vc,ids_k,keep_all):
        base=np.asarray(keep_all,bool);self.holder['dec']=np.zeros(CH,np.float32)
        if k<K0:return base,{'uv_arm':'M3Q128','q_banded':0,'uv_hidden_tokens':0,'uv_hidden_chunk_tokens':0}
        dec=CM.up(self.m[f'qb_TOP30_{k}'],CH);self.holder['dec']=dec.astype(np.float32);kp=np.broadcast_to(base,(CH,WIN+CH)).copy();kp[:]=CM.band_keep(base,BAND)
        if k%16==0:self.log(f'  M3Q128 chunk {k}: deciding rows {int(dec.sum())}')
        return kp,{'uv_arm':'M3Q128','q_banded':CH,'q_band':BAND,'n_deciding':int(dec.sum()),'uv_hidden_tokens':0,'uv_hidden_chunk_tokens':0}
def masks_stage(a):
    out=pathlib.Path(a['out']);out.mkdir(parents=True,exist_ok=True);import e55_uv as UVm,e57_t1t2v2_run as V
    ids=V.stream_ids(a['sentences'],'A');Z=np.load(EXP/'e57_band/e57_band_W0.npz');b=list(Z['bands']).index(BAND);ends=UVm.sentence_ends(UVm.tokenizer(str(ROOT/'data/tokenizer/llama3/tokenizer.json')),ids);res={};st=[]
    for k in range(K0,len(ids)//CH):
        q=UVm.chunk_sents(ends,k,CH);sid=np.full(CH,-1)
        for i,(s_,e_) in enumerate(q):sid[s_:e_]=i
        lm=Z['lm'][k];el=CM.rows_eligible(lm,sid)&lm;g=(Z['nll_band'][b][k]-Z['nll_win'][k]).astype(np.float64);m=deciding(g,el);res[f'qb_TOP30_{k}']=CM.pk(m)
        st.append(dict(chunk=k,n_eligible=int(el.sum()),n_deciding=int(m.sum()),mean_gain_deciding=float(g[m].mean()),share_of_positive_gain=float(g[m].clip(0).sum()/max(g[el].clip(0).sum(),1e-9))))
    res['chunks']=np.asarray([K0,len(ids)//CH]);res['p']=np.asarray(P);res['band']=np.asarray(BAND);fn=out/'news_m30_masks.npz';np.savez_compressed(fn,**res)
    s={k:float(np.mean([x[k] for x in st])) for k in st[0] if k!='chunk'};fn.with_suffix('.json').write_text(json.dumps(dict(summary=s,chunks=st,out_sha=hashlib.sha256(open(fn,'rb').read()).hexdigest()[:16])));print(json.dumps(s))
def write_stage(a):
    out=pathlib.Path(a['out']);out.mkdir(parents=True,exist_ok=True);w=float(a['weight']);arm=a.get('arm') or f'M3W{int(w)}Q128';saves=[int(x) for x in a['saves'].split(',')] if a.get('saves') else None;tag=arm+('C' if saves else '')
    import e55_run as R,e57_t1t2v2_run as V,e57_syn_mrun as MR,e57_qband_run as QR
    LD,s3,C=R.lab();rt=LD.Runtime();can=R.Canary55(LD,rt);lg=lambda *x:print(*x,flush=True);cw0=can(rt.base);lg('canary W0',cw0)
    ids=V.stream_ids(a['sentences'],'A');nch=len(ids)//CH;lg(f'stream A: {len(ids)} tokens, {nch} chunks, ids sha ok');saves=saves or [nch-1]
    masks={k:v for k,v in np.load(out/'news_m30_masks.npz',allow_pickle=True).items()};ntg=list(json.load(open(a['nat_ref']))['norm_targets'][:nch]);assert len(ntg)==nch
    L0=C.Ladder55(rt,save_dir=out/'states'/'rec',canary=None);t0=time.monotonic()
    with rt.mesh:Kh0,Vh0,_h=L0.record(ids[:nch*CH])
    del _h;rc={(hashlib.sha256(np.asarray(ids[:nch*CH],np.int64).tobytes()).hexdigest(),nch):(Kh0,Vh0)};lg(f'  natural K,V recorded in {time.monotonic()-t0:.0f}s');del L0;gc.collect();LD.jax.clear_caches()
    holder={'dec':np.zeros(CH,np.float32)};L=C.Ladder55(rt,save_dir=out/'states'/f'E57_news_{tag}',canary=None);L.rec_cache=rc;traj=[]
    if arm=='ALLQ128':uv=QR.SelMaskQ('ALLQ128',{'band_ALLQ128':np.asarray(128),'all_ALLQ128':np.asarray(1)},lg)    # the original ALLQ128 write of job 570 (ordinary loss)
    else:uv=UVNewsM3(masks,holder,lg);MR.install_weighted_grad(L,s3,holder,w)
    def hook(k,m):
        cn=can(m);traj.append(dict(chunk=int(k),canary=cn));lg(f'  {tag} chunk {k}: canary rise gen {cn["_gen_nll"]-cw0["_gen_nll"]:+.4f} tail {cn["_tail_nll"]-cw0["_tail_nll"]:+.4f}');return {'ok':True}
    t0=time.monotonic()
    res=L.run_stream('SELC1',1,ids,nch=nch,tag=f'news{tag}',log=lg,read_hook=hook,hook_chunks=set(saves),read_chunks=set(saves),chunks_meta=[{'sents':[]} for _ in range(nch)],norm_targets=ntg,uv=uv);log=res['log']
    ref_f=out/'E57_news_M3W8Q128.arms.json' if arm!='ALLQ128' else EXP/'e57_t1t2v2/streamA/E57_v2A_ALLQ128.arms.json';rh=json.load(open(ref_f))['states'][str(nch-1)]['hash'];nh=next((v['_hash'] for k,v in res['reads'].items() if int(k)==nch-1),None)
    (out/f'E57_news_{tag}.arms.json').write_text(json.dumps(dict(arm=tag,engine='SELC1',stream='A',n_chunks=nch,ids_sha=V.STREAM_SHA['A'],weight=w if arm!='ALLQ128' else 1.0,band=BAND,p_deciding=P,canary_w0=cw0,trajectory=traj,states={str(k):{'file':v['_file'],'hash':v['_hash']} for k,v in res['reads'].items()},sec_total=time.monotonic()-t0,end_state_hash=nh,original_end_state_hash=rh,end_state_hash_equal=bool(nh==rh),
        update_log=[{k_:e[k_] for k_ in ('k','ce','update_norm','coef','sec','n_masked_keys','q_banded','q_band','n_deciding') if k_ in e} for e in log],norm_source='E57_main_NAT first cycle x 0.5 (engine SELC1)'),default=float));lg('wrote',out/f'E57_news_{tag}.arms.json','end-state hash equal to the original:',nh==rh)
def merge_stage(a):
    """job 677 v1 tokens (all stored states) + the new state read in this job -> <out>/merged/e57_news_tokread.json (analyze of e57_news_tokread.py then compares every state, paired per frame)"""
    out=pathlib.Path(a['out']);old=json.load(open(a['old']));new=json.load(open(out/'read'/'e57_news_tokread.json'));mo=old['meta']['v1'];mn=new['meta']['v1'];assert [m['id'] for m in mo]==[m['id'] for m in mn] and all(x['tokids']==y['tokids'] for x,y in zip(mo,mn)),'frames or token ids differ'
    mx=max(float(np.max(np.abs(np.asarray([sum(t) for t in x])-np.asarray([sum(t) for t in y])))) for x,y in zip(old['tokens']['v1']['W0'],new['tokens']['v1']['W0']));assert mx<1e-3,('W0 of the two reads differs',mx)
    tok={**old['tokens']['v1'],**{k:v for k,v in new['tokens']['v1'].items() if k!='W0'}};d=dict(meta={'v1':mo},tokens={'v1':tok},reproduction=dict(old['reproduction'],**{k:v for k,v in new['reproduction'].items() if not k.endswith(':W0')},W0_old_vs_new_max_abs_diff_sum=mx),registry=dict(old['registry'],new=new['registry']),out_tail_sha=new['out_tail_sha'])
    (out/'merged').mkdir(exist_ok=True);(out/'merged'/'e57_news_tokread.json').write_text(json.dumps(d));print('merged labels',list(tok),'W0 old vs new max|diff|',mx)
def selftest():
    import e57_syn_mrun as MR;MR.selftest()
    rng=np.random.default_rng(0);dec=rng.random(CH)<.3;holder={};u=UVNewsM3({'qb_TOP30_9':CM.pk(dec)},holder,log=lambda *x:None);nat=np.ones(WIN+CH,bool);kp,info=u.keep(9,None,None,None,None,nat)
    assert np.array_equal(kp,CM.band_keep(nat,128)) and info['n_deciding']==int(dec.sum()) and np.array_equal(holder['dec']>0,dec);kp0,_=u.keep(3,None,None,None,None,nat);assert kp0.ndim==1 and not holder['dec'].any()
    print('e57_news_m3 selftest ok')
def main():
    if '--selftest' in sys.argv:selftest();return
    ap=argparse.ArgumentParser();ap.add_argument('--stage');ap.add_argument('--out');ap.add_argument('--sentences',default=str(EXP/'e57_t1t2v2_in/e57_stream_sentences.jsonl'));ap.add_argument('--nat-ref',default=str(EXP/'e57_main/E57_main_NAT.arms.json'));ap.add_argument('--weight',default='8');ap.add_argument('--arm');ap.add_argument('--saves');ap.add_argument('--old',default=str(EXP/'e57_news_tokread/e57_news_tokread.json'));a=ap.parse_args()
    {'masks':masks_stage,'write':write_stage,'merge':merge_stage}[a.stage](dict(out=a.out,sentences=a.sentences,nat_ref=a.nat_ref,weight=a.weight,old=a.old,arm=a.arm,saves=a.saves))
if __name__=='__main__':main()
