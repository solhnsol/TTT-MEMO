"""E57 public QA stage 16 (new file; GPU; method thread, Hans '같은 규칙 그대로 실행' 11/04 11:33 KST): M3 w=8 and M4 on the QA bundle episodes with 17 exposures of job 790 (bundles 5 / 8 / 9 / 10 / 12, 50 paragraphs, 73 items, rewrites_16_subset.json). Same rules as the synthetic and news runs: ALLQ128 mask on every query row (from chunk 0, as the QA ALLQ128 of job 640), loss weight 8 on the deciding rows = the top 30% of the eligible rows of every chunk by the window gain g = nll_band128 - nll_win of the W0 pass over the episode (e57_dwg_qa_pass.py, band 128; eligible = loss rows), update norm 0.5 x the per-chunk norm of the NAT write of job 790 (engine SELC1); M4: the deciding rows see only the previous 32 tokens (the ratio 1/4 of the synthetic 64 / 256 and news 32 / 128 applied to the QA band 128). States at the read chunks of n = 1 / 3 / 5 / 9 / 13 / 17 with the canary; reads like job 790 (question read A / B, closed same-bundle read) with the stage functions of e57_pqa15_run.py; the W0, NATQ and ALLQ128 rows of job 790 are reused (the W0 rows are recomputed as a packing check). Analyze = e57_pqa15_run.analyze with the four arms.
  python3 e57_pqa16_run.py --stage write|read|analyze --out O ... | --selftest"""
import sys,json,time,gc,hashlib,pathlib
import numpy as np
HERE=pathlib.Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
import e57_closure_masks as CM
import e57_pqa2_run as P2
import e57_pqa6_run as P6
import e57_pqa12_run as P12
import e57_pqa15_run as P15
CH=1024;WIN=8*CH;BAND=128;DEC_BAND_M4=32;P=0.30;NEW_ARMS=['M3W8Q128','M4W8Q128']
class UVQA:
    """UV-compatible keep object: band 128 for every row of every chunk; M4: the deciding rows see only the previous dec_band tokens; publishes the deciding rows of the chunk in holder['dec'] for the weighted gradient"""
    def __init__(self,masks,holder,dec_band=None):self.m=masks;self.holder=holder;self.dec_band=dec_band
    def keep(self,k,m,Kc,Vc,ids_k,keep_all):
        mk=self.m[k]
        if isinstance(mk,np.ndarray) and mk.dtype.kind=='f' and mk.shape==(CH,):w=mk.astype(np.float32);dec=w>0;self.holder['dec']=w        # continuous per-row weight in [0, 1] (loss weight 1 + (W - 1) f); the rows with f > 0 count as deciding rows for the band
        else:dec=CM.up(mk,CH);self.holder['dec']=dec.astype(np.float32)
        base=np.asarray(keep_all,bool);kp=np.broadcast_to(base,(CH,WIN+CH)).copy();kp[:]=CM.band_keep(base,BAND)
        if self.dec_band:kp[dec]=CM.band_keep(base,self.dec_band)[dec]
        return kp,{'uv_arm':'M4' if self.dec_band else 'M3','q_banded':CH,'q_band':BAND,'n_deciding':int(dec.sum()),'uv_hidden_tokens':0,'uv_hidden_chunk_tokens':0}
def masks_from_npz(path,gid,key='qb_TOP30'):
    """external mask source (job 1060 validation work): npz with the keys g<gid>_<key>_<chunk> (CM.pk packed bool rows, or a float32 [CH] weight vector in [0, 1]) -> {chunk: mask} for UVQA"""
    Z=np.load(path,allow_pickle=True);pre=f'g{gid}_{key}_';return {int(k[len(pre):]):Z[k] for k in Z.files if k.startswith(pre)}
def write_masks_npz(nll_npz,out,frac=P,band=BAND):
    """top-frac masks of every bundle of a QA W0 npz (e57_dwg_qa_nll.npz) in the format of masks_from_npz"""
    Z=np.load(nll_npz);bi=list(Z['bands']).index(band);res={}
    for gid in sorted({int(k[1:].split('_')[0]) for k in Z.files if k.startswith('g') and k.endswith('_win')}):
        win=Z[f'g{gid}_win'];bd=Z[f'g{gid}_band'][bi];lm=Z[f'g{gid}_lm']
        for k in range(win.shape[0]):
            el=CM.rows_eligible(lm[k],np.zeros(CH,np.int32))&lm[k];g=(bd[k]-win[k]).astype(np.float64);res[f'g{gid}_qb_TOP{int(round(frac*100))}_{k}']=CM.pk(CM.top_rows(g,el,int(round(frac*el.sum()))))
    np.savez_compressed(out,**res);return res
def deciding_masks(win,band,lm):
    out={}
    for k in range(win.shape[0]):
        el=CM.rows_eligible(lm[k],np.zeros(CH,np.int32))&lm[k];g=(band[k]-win[k]).astype(np.float64);n=int(round(P*el.sum()));out[k]=CM.pk(CM.top_rows(g,el,n))
    return out
def write_stage(a):
    out=pathlib.Path(a['out']);import e55_run as R,e57_syn_mrun as MR
    LD,s3,C=R.lab();rt=LD.Runtime();can=R.Canary55(LD,rt);lg=lambda *x:print(*x,flush=True);cw0=can(rt.base);lg('canary W0',cw0)
    meta,ids_z=P6.load_groups(out);W640=json.load(open(out/'e57_pqa6_writes.json'));Z=np.load(out/'e57_dwg_qa_nll.npz');bi=list(Z['bands']).index(BAND);fn=out/'e57_pqa16_writes.json';W=json.load(open(fn)) if fn.exists() else {};t00=time.monotonic()
    for g in meta['groups']:
        gid=str(g['gid'])
        if gid in W:continue
        ids=ids_z[f'g{gid}'];nch=len(ids)//CH;assert nch==g['n_chunks'];rd_ch={int(v) for v in g['read_chunks'].values()};ntg=W640[gid]['NAT']['update_norms'];assert len(ntg)==nch
        dm=deciding_masks(Z[f'g{gid}_win'],Z[f'g{gid}_band'][bi],Z[f'g{gid}_lm']);t0=time.monotonic();L0=C.Ladder55(rt,save_dir=out/'states'/'rec',canary=None)
        with rt.mesh:Kh0,Vh0,_h=L0.record(ids)
        del _h;rc={(hashlib.sha256(np.asarray(ids,np.int64).tobytes()).hexdigest(),nch):(Kh0,Vh0)};del L0;gc.collect();W[gid]={}
        for arm in NEW_ARMS:
            holder={'dec':np.zeros(CH,np.float32)};uv=UVQA(dm,holder,dec_band=DEC_BAND_M4 if arm.startswith('M4') else None);L=C.Ladder55(rt,save_dir=out/'states'/f'g{gid}_{arm}',canary=can);L.rec_cache=rc;MR.install_weighted_grad(L,s3,holder,8.0)
            res=L.run_stream('SELC1',1,ids,nch=nch,tag=f'pqa16g{gid}{arm}',log=lambda *x:None,read_chunks=rd_ch,canary_chunks=rd_ch,chunks_meta=[{'sents':[]} for _ in range(nch)],norm_targets=ntg,uv=uv)
            W[gid][arm]=dict(states={str(k):dict(file=v['_file'],hash=v['_hash'],canary=v.get('canary')) for k,v in res['reads'].items()},ce=[e['ce'] for e in res['log']],update_norms=[e['update_norm'] for e in res['log']],n_deciding=[e.get('n_deciding') for e in res['log']],canary_w0=cw0);del L,res;gc.collect()
        del rc,Kh0,Vh0;gc.collect();fn.write_text(json.dumps(W,default=float));lg(f'bundle {gid}: {nch} chunks, {len(NEW_ARMS)} arms written in {time.monotonic()-t0:.0f}s (total {time.monotonic()-t00:.0f}s)')
def read_stage(a):
    out=pathlib.Path(a['out']);lg=lambda *x:print(*x,flush=True);P15.ARMS=list(NEW_ARMS);P15.P6.SEED=P6.SEED
    W={'640':json.load(open(out/'e57_pqa16_writes.json')),'650':{}};src=pathlib.Path(a['src790'])
    # run the read stage of job 790 for the new arms only: it writes E57_pqa15_rows_* for W0 (not used: file names of the W0 rows are redirected) and the new labels
    import shutil;rows_dir=out/'new_rows';rows_dir.mkdir(exist_ok=True)
    a2=dict(a,out=str(rows_dir));
    for f in ('e57_pqa6_groups.json','e57_pqa6_group_ids.npz'):
        if not (rows_dir/f).exists():shutil.copy(out/f,rows_dir/f)
    (rows_dir/'e57_pqa6_writes.json').write_text(json.dumps(W['640']));(rows_dir/'e57_pqa7_writes.json').write_text('{}')
    # the reproduction part of P15.read_stage needs the NAT / NATQ / ALLQ128 update norms of job 640: give it empty-safe copies
    P15.read_stage(dict(a2,src640=a['src640'],skip_repro=True))
def selftest():
    P15.selftest();rng=np.random.default_rng(0);win=rng.random((3,CH))*3;band=win+rng.random((3,CH));lm=np.ones((3,CH),bool);lm[:,-1]=False
    dm=deciding_masks(win,band,lm);d=CM.up(dm[1],CH);assert d.sum()==int(round(.3*lm[1].sum())) and not d[-1]
    holder={};u=UVQA(dm,holder,dec_band=32);nat=np.ones(WIN+CH,bool);kp,_=u.keep(1,None,None,None,None,nat);assert np.array_equal(kp[d],CM.band_keep(nat,32)[d]) and np.array_equal(kp[~d],CM.band_keep(nat,128)[~d])
    print('e57_pqa16_run selftest ok')
def main():
    if '--selftest' in sys.argv:selftest();return
    stage=P2._pop('--stage');g=lambda k:P2._pop(k)
    a=dict(out=g('--out'),src640=g('--src640'),src790=g('--src790'),items=g('--items-file'),bank=g('--bank'),assign=g('--assign'),stage1=g('--stage1'),json=g('--json'),max_groups=None)
    if stage=='analyze':P15.ARMS=['NATQ','ALLQ128']+NEW_ARMS
    {'write':write_stage,'read':read_stage,'analyze':P15.analyze}[stage](a)
if __name__=='__main__':main()
