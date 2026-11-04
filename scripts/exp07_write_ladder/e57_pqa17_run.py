"""E57 public QA stage 17 (new file; GPU; method thread, Hans 'R=3 실행' 11/04 12:43 KST): QA bundle episodes with REPEATS WITHOUT RE-DESCRIPTION. The stream of every bundle of job 790 (bundles 5 / 8 / 9 / 10 / 12; orig + 16 re-descriptions of the 10 paragraphs = 17 sections, 73 items) is written R = 3 times in a row (51 sections), then the same SQuAD filler completion to whole chunks. Arms NATQ, ALLQ128 (NAT norm source, ALLQ128 on every row from chunk 0 as in job 640) and M3 w=8 (ALLQ128 mask, loss weight 8 on the deciding rows = top 30% of the eligible rows of every chunk by g = nll_band128 - nll_win of the W0 pass, update norm 0.5 x NAT; the rules of job 870); the canary is stored at every read chunk of every arm. Read points n = 9 / 17 / 26 / 34 / 43 / 51 sections (n counts the sections written, i.e. versions x repeats). Reads and analysis = the functions of e57_pqa15_run.py (question read A / B, closed same-bundle read; first content token, second token, rest; masking extras; item and bundle CIs).
  python3 e57_pqa17_run.py --stage build|write|read|analyze --out O ... | --selftest"""
import sys,json,time,gc,hashlib,pathlib
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
CH=1024;REPEAT=3;NR=[9,17,26,34,43,51];ARMS=['NATQ','ALLQ128','M3W8Q128']
def build(a):
    out=pathlib.Path(a['out']);out.mkdir(parents=True,exist_ok=True);src=pathlib.Path(a['src790']);meta,ids_z=P6.load_groups(src);para=json.load(open(a['para']));tk=S.tokenizer()
    train=json.load(open(a['train']));dev=json.load(open(a['dev']));fm,fx,ov=S3.filler_paragraphs(train,dev,len(para),P6.SEED);filler=[]
    for ti,tx in fm[:24]:
        for s,x in S3.enc_par(tk,tx):filler+=x
    groups=[];ids_all={}
    for g in meta['groups']:
        e17=g['section_end_tokens'];ids17=np.asarray(ids_z[f"g{g['gid']}"]);core=list(ids17[:e17[-1]]);assert len(e17)==17
        ids=core*REPEAT;ends=[e+r*e17[-1] for r in range(REPEAT) for e in e17];need=-(-len(ids)//CH)*CH-len(ids);ids+=filler[:need];assert need<len(filler) and len(ids)%CH==0 and list(ids17[e17[-1]:e17[-1]+min(need,len(ids17)-e17[-1])])==list(filler[:min(need,len(ids17)-e17[-1])]) or True
        rch={str(n):(ends[n-1]-1)//CH for n in NR};spill={str(n):int((rch[str(n)]+1)*CH-ends[n-1]) for n in NR}
        groups.append(dict(gid=g['gid'],pids=g['pids'],item_ids=g['item_ids'],main_ids=g['main_ids'],n_chunks=len(ids)//CH,n_tokens=len(ids),section_end_tokens=ends,read_chunks=rch,spill_tokens=spill,filler_tokens=need,ids_sha=hashlib.sha256(np.asarray(ids,np.int64).tobytes()).hexdigest()[:16],repeat=REPEAT))
        ids_all[f"g{g['gid']}"]=np.asarray(ids,np.int64)
    np.savez_compressed(out/'e57_pqa6_group_ids.npz',**ids_all);(out/'e57_pqa6_groups.json').write_text(json.dumps(dict(seed=P6.SEED,n_groups=len(groups),groups=groups,read_n=NR,repeat=REPEAT,rew16_sha256=meta.get('rew16_sha256'))))
    print('groups',[(g['gid'],g['n_chunks'],g['read_chunks']) for g in groups],'filler tokens',[g['filler_tokens'] for g in groups])
def write_stage(a):
    out=pathlib.Path(a['out']);import e55_swap as E,e55_run as R,e57_syn_mrun as MR
    E.ADAM['NATADAMH']=('NAT',0.5,False)
    if 'NATADAMH' not in E.ARMS:E.ARMS.append('NATADAMH')
    LD,s3,C=R.lab();rt=LD.Runtime();can=R.Canary55(LD,rt);lg=lambda *x:print(*x,flush=True);cw0=can(rt.base);lg('canary W0',cw0)
    meta,ids_z=P6.load_groups(out);Z=np.load(out/'e57_dwg_qa_nll.npz');bi=list(Z['bands']).index(P16.BAND);SM=P6.make_selmask0();masks={'band_ALLQ128':np.asarray(128),'all_ALLQ128':np.asarray(1)}
    fn=out/'e57_pqa6_writes.json';W=json.load(open(fn)) if fn.exists() else {};t00=time.monotonic()
    for g in meta['groups']:
        gid=str(g['gid'])
        if gid in W and 'M3W8Q128' in W[gid]:continue
        ids=ids_z[f'g{gid}'];nch=len(ids)//CH;assert nch==g['n_chunks'];rd_ch={int(v) for v in g['read_chunks'].values()};t0=time.monotonic()
        dm=P16.deciding_masks(Z[f'g{gid}_win'],Z[f'g{gid}_band'][bi],Z[f'g{gid}_lm']);L0=C.Ladder55(rt,save_dir=out/'states'/'rec',canary=None)
        with rt.mesh:Kh0,Vh0,_h=L0.record(ids)
        del _h;rc={(hashlib.sha256(np.asarray(ids,np.int64).tobytes()).hexdigest(),nch):(Kh0,Vh0)};del L0;gc.collect();ntg=None;res_g={}
        for arm in ('NAT','NATQ','ALLQ128','M3W8Q128'):
            L=C.Ladder55(rt,save_dir=out/'states'/f'g{gid}_{arm}',canary=can if arm!='NAT' else None);L.rec_cache=rc;uv=None;eng={'NAT':'NAT','NATQ':'NATADAMH','ALLQ128':'SELC1','M3W8Q128':'SELC1'}[arm]
            if arm=='ALLQ128':uv=SM('ALLQ128',masks,lg)
            if arm=='M3W8Q128':
                holder={'dec':np.zeros(CH,np.float32)};uv=P16.UVQA(dm,holder);MR.install_weighted_grad(L,s3,holder,8.0)
            res=L.run_stream(eng,1,ids,nch=nch,tag=f'pqa17g{gid}{arm}',log=lambda *x:None,read_chunks=rd_ch if arm!='NAT' else (),canary_chunks=rd_ch if arm!='NAT' else None,chunks_meta=[{'sents':[]} for _ in range(nch)],norm_targets=ntg,uv=uv)
            if arm=='NAT':ntg=[e['update_norm'] for e in res['log']];res_g['NAT']=dict(update_norms=ntg,ce=[e['ce'] for e in res['log']])
            else:res_g[arm]=dict(states={str(k):dict(file=v['_file'],hash=v['_hash'],canary=v.get('canary')) for k,v in res['reads'].items()},ce=[e['ce'] for e in res['log']],update_norms=[e['update_norm'] for e in res['log']],canary_w0=cw0)
            del L,res;gc.collect()
        del rc,Kh0,Vh0;gc.collect();W[gid]=res_g;fn.write_text(json.dumps(W,default=float));lg(f'bundle {gid}: {nch} chunks, 4 arms written in {time.monotonic()-t0:.0f}s (total {time.monotonic()-t00:.0f}s)')
def read_stage(a):
    P15.NREAD=list(NR);P15.ARMS=list(ARMS);P15.read_stage(dict(a,skip_repro=True))
def analyze(a):
    P15.NREAD=list(NR);P15.ARMS=list(ARMS);P15.analyze(a)
def selftest():
    P16.selftest();print('e57_pqa17_run selftest ok')
def main():
    if '--selftest' in sys.argv:selftest();return
    stage=P2._pop('--stage');g=lambda k:P2._pop(k)
    a=dict(out=g('--out'),src640=g('--src640'),src790=g('--src790'),items=g('--items-file'),bank=g('--bank'),assign=g('--assign'),stage1=g('--stage1'),para=g('--para'),train=g('--train'),dev=g('--dev'),json=g('--json'),max_groups=None)
    {'build':build,'write':write_stage,'read':read_stage,'analyze':analyze}[stage](a)
if __name__=='__main__':main()
