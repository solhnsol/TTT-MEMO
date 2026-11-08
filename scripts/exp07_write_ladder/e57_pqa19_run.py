"""E57 public QA stage 19 (new file; GPU; QA cause thread, Hans '수리부터 하고 한번에 돌려보자' 11/05 00:39 KST): the QA bundle episodes with up to 65 versions: original + 64 re-descriptions per paragraph (rewrites_64_subset.json, sha256 5f99556baa5d251d...; the first 16 re-descriptions equal rewrites_16_subset.json, so the first 17 sections equal the streams of jobs 790 / 870 up to the filler), 5 bundles (5 / 8 / 9 / 10 / 12; 50 paragraphs, 73 items), sections of the 10 paragraphs joined as in job 640, SQuAD filler completion to whole chunks. Arms NATQ, ALLQ128, M3 w=8 and M4 (deciding rows see only the previous 32 tokens) as in jobs 870 / 840: ALLQ128 on every row from chunk 0, update norm 0.5 x the per-chunk NAT norm (NAT written here as the norm source), M3 / M4 weight 8 on the deciding rows (top 30% of the eligible rows by g = nll_band128 - nll_win of the W0 pass over the streams); canary at every read chunk. Read points n = 1 / 3 / 5 / 9 / 17 / 25 / 33 / 49 / 65 sections; reads and metrics = the functions of e57_pqa15_run.py (question read A / B, closed same-bundle read; first content token, second token, rest; masking extras; item and bundle CIs). The reads of n <= 33 are done and analysed first (intermediate file), then n = 49 / 65.
  python3 e57_pqa19_run.py --stage build|write|read|analyze --out O ... [--n-set A|B|all] | --selftest"""
import re,sys,json,time,gc,hashlib,pathlib
import numpy as np
HERE=pathlib.Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
import e57_closure_masks as CM
import e57_pqa2_run as P2
import e57_pqa2_stream as S
import e57_pqa3_stream as S3
import e57_pqa6_run as P6
import e57_pqa15_run as P15
import e57_pqa16_run as P16
CH=1024;NR_ALL=[1,3,5,9,17,25,33,49,65];NR_A=[1,3,5,9,17,25,33];NR_B=[49,65];ARMS=['NATQ','ALLQ128','M3W8Q128','M4W8Q128'];ARMS_W=['NAT','NATQ','ALLQ128','M3W8Q128','M4W8Q128'];MASK_NPZ=None;MASK_KEY='qb_TOP30'   # MASK_NPZ: optional external mask file (e57_pqa16_run.masks_from_npz format) replacing the top-30% rule of the M3 / M4 arms;REW64_SHA12='5f99556baa5d'
def build(a):
    out=pathlib.Path(a['out']);out.mkdir(parents=True,exist_ok=True);raw=open(a['rew64'],'rb').read();assert hashlib.sha256(raw).hexdigest().startswith(REW64_SHA12),'rewrites_64_subset.json sha256 differs'
    r64=json.loads(raw);src=pathlib.Path(a['src790']);meta,ids_z=P6.load_groups(src);para=json.load(open(a['para']));byp={p['pid']:p for p in para};tk=S.tokenizer()
    train=json.load(open(a['train']));dev=json.load(open(a['dev']));fm,fx,ov=S3.filler_paragraphs(train,dev,len(para),P6.SEED);filler=[]
    for ti,tx in fm[:24]:
        for s,x in S3.enc_par(tk,tx):filler+=x
    groups=[];ids_all={};same17={}
    for g in meta['groups']:
        ids=[];ends=[]
        for v in range(65):
            for pid in g['pids']:
                assert len(r64[pid])==64
                text=byp[pid]['context'] if v==0 else r64[pid][v-1]
                for s,x in S3.enc_par(tk,text):ids+=x
            ends.append(len(ids))
        e17=g['section_end_tokens'];same17[g['gid']]=bool(ends[16]==e17[16] and list(ids[:e17[16]])==list(np.asarray(ids_z[f"g{g['gid']}"])[:e17[16]]))
        need=-(-len(ids)//CH)*CH-len(ids);ids+=filler[:need];assert need<len(filler) and len(ids)%CH==0
        rch={str(n):(ends[n-1]-1)//CH for n in NR_ALL};spill={str(n):int((rch[str(n)]+1)*CH-ends[n-1]) for n in NR_ALL}
        groups.append(dict(gid=g['gid'],pids=g['pids'],item_ids=g['item_ids'],main_ids=g['main_ids'],n_chunks=len(ids)//CH,n_tokens=len(ids),section_end_tokens=ends,read_chunks=rch,spill_tokens=spill,filler_tokens=need,ids_sha=hashlib.sha256(np.asarray(ids,np.int64).tobytes()).hexdigest()[:16],versions=65))
        ids_all[f"g{g['gid']}"]=np.asarray(ids,np.int64)
    np.savez_compressed(out/'e57_pqa6_group_ids.npz',**ids_all);(out/'e57_pqa6_groups.json').write_text(json.dumps(dict(seed=P6.SEED,n_groups=len(groups),groups=groups,read_n=NR_ALL,versions=65,first17_sections_equal_job790=same17)))
    print('groups',[(g['gid'],g['n_chunks'],g['read_chunks']) for g in groups],'first 17 sections equal to job 790:',same17)
def write_stage(a):
    out=pathlib.Path(a['out']);import e55_swap as E,e55_run as R,e57_syn_mrun as MR
    E.ADAM['NATADAMH']=('NAT',0.5,False)
    if 'NATADAMH' not in E.ARMS:E.ARMS.append('NATADAMH')
    LD,s3,C=R.lab();rt=LD.Runtime();can=R.Canary55(LD,rt);lg=lambda *x:print(*x,flush=True);cw0=can(rt.base);lg('canary W0',cw0)
    meta,ids_z=P6.load_groups(out);Z=np.load(out/'e57_dwg_qa_nll.npz');bi=list(Z['bands']).index(P16.BAND);SM=P6.make_selmask0();masks={'band_ALLQ128':np.asarray(128),'all_ALLQ128':np.asarray(1)}
    fn=out/'e57_pqa6_writes.json';W=json.load(open(fn)) if fn.exists() else {};t00=time.monotonic()
    for g in meta['groups']:
        gid=str(g['gid'])
        if gid in W and ARMS_W[-1] in W[gid]:continue
        ids=ids_z[f'g{gid}'];nch=len(ids)//CH;assert nch==g['n_chunks'];rd_ch={int(v) for v in g['read_chunks'].values()};t0=time.monotonic()
        dm=P16.masks_from_npz(MASK_NPZ,gid,MASK_KEY) if MASK_NPZ else P16.deciding_masks(Z[f'g{gid}_win'],Z[f'g{gid}_band'][bi],Z[f'g{gid}_lm']);L0=C.Ladder55(rt,save_dir=out/'states'/'rec',canary=None)
        with rt.mesh:Kh0,Vh0,_h=L0.record(ids)
        del _h;rc={(hashlib.sha256(np.asarray(ids,np.int64).tobytes()).hexdigest(),nch):(Kh0,Vh0)};del L0;gc.collect();ntg=None;res_g={}
        for arm in ARMS_W:
            L=C.Ladder55(rt,save_dir=out/'states'/f'g{gid}_{arm}',canary=can if arm!='NAT' else None);L.rec_cache=rc;uv=None;eng={'NAT':'NAT','NATQ':'NATADAMH'}.get(arm,'SELC1')
            if arm=='ALLQ128':uv=SM('ALLQ128',masks,lg)
            if arm in ('M3OA2W16Q128','M3OS2W16Q128'):             # job 1160 / 1230 (OS2: g' with the band only in the suffix layers): ONLINE ABS2 rule (g128 from the current fast weights, tau 2, weight 16, ALLQ128 on every row)
                import e57_online_uv as OU
                holder={'dec':np.zeros(CH,np.float32),'L':L};uv=OU.UVOnlineBand(holder,lg);MR.install_weighted_grad(L,s3,holder,16.0);OU.install_online_dec_band(L,s3,holder,2.0,lg,suf=(arm=='M3OS2W16Q128'))
            mm=re.match(r'^M3(q[A-Za-z0-9_]*?)W(\d+)Q128$',arm)       # rule arms (job 1070): M3<mask key>W<weight>Q128, masks from MASK_NPZ (e57_rule_masks_real.py), e.g. M3qtA4W16Q128
            if mm:
                holder={'dec':np.zeros(CH,np.float32)};uv=P16.UVQA(P16.masks_from_npz(MASK_NPZ,gid,mm.group(1)),holder);MR.install_weighted_grad(L,s3,holder,float(mm.group(2)))
            if arm in ('M3W8Q128','M4W8Q128'):
                holder={'dec':np.zeros(CH,np.float32)};uv=P16.UVQA(dm,holder,dec_band=P16.DEC_BAND_M4 if arm.startswith('M4') else None);MR.install_weighted_grad(L,s3,holder,8.0)
            res=L.run_stream(eng,1,ids,nch=nch,tag=f'pqa19g{gid}{arm}',log=lambda *x:None,read_chunks=rd_ch if arm!='NAT' else (),canary_chunks=rd_ch if arm!='NAT' else None,chunks_meta=[{'sents':[]} for _ in range(nch)],norm_targets=ntg,uv=uv)
            if arm=='NAT':ntg=[e['update_norm'] for e in res['log']];res_g['NAT']=dict(update_norms=ntg,ce=[e['ce'] for e in res['log']])
            else:res_g[arm]=dict(states={str(k):dict(file=v['_file'],hash=v['_hash'],canary=v.get('canary')) for k,v in res['reads'].items()},ce=[e['ce'] for e in res['log']],update_norms=[e['update_norm'] for e in res['log']],canary_w0=cw0)
            if arm in ('M3OA2W16Q128','M3OS2W16Q128'):res_g[arm]['online_share_mean']=float(np.mean(holder['shares'])) if holder.get('shares') else None;res_g[arm]['online_g_mean']=float(np.mean(holder['g_mean'])) if holder.get('g_mean') else None
            del L,res;gc.collect();lg(f'  bundle {gid} {arm} done ({time.monotonic()-t0:.0f}s)')
        del rc,Kh0,Vh0;gc.collect();W[gid]=res_g;fn.write_text(json.dumps(W,default=float));lg(f'bundle {gid}: {nch} chunks, {len(ARMS_W)} arms written in {time.monotonic()-t0:.0f}s (total {time.monotonic()-t00:.0f}s)')
def read_stage(a):
    nset={'A':NR_A,'B':NR_B,'all':NR_ALL}[a.get('n_set') or 'all'];P15.NREAD=list(nset);P15.ARMS=list(ARMS);P15.read_stage(dict(a,skip_repro=True))
def analyze(a):
    nset={'A':NR_A,'B':NR_B,'all':NR_ALL}[a.get('n_set') or 'all'];P15.NREAD=list(nset);P15.ARMS=list(ARMS);P15.analyze(a)
def selftest():
    P16.selftest();print('e57_pqa19_run selftest ok')
def main():
    if '--selftest' in sys.argv:selftest();return
    stage=P2._pop('--stage');g=lambda k:P2._pop(k)
    a=dict(out=g('--out'),src640=g('--src640'),src790=g('--src790'),items=g('--items-file'),bank=g('--bank'),assign=g('--assign'),stage1=g('--stage1'),para=g('--para'),train=g('--train'),dev=g('--dev'),rew64=g('--rew64'),json=g('--json'),n_set=g('--n-set'),max_groups=None)
    {'build':build,'write':write_stage,'read':read_stage,'analyze':analyze}[stage](a)
if __name__=='__main__':main()
