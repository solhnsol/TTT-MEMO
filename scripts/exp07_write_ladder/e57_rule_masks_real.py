"""E57 rule masks for the QA and news streams (new file; CPU; job 1070 preparation, Hans 11/05): from the band-128 window gain g = nll_band128 - nll_win over the eligible rows, per chunk: ABS-NAT rows (e57_rules.abs_rule, tau = 2 / 4 / 6 nats; key names qtA2 / qtA4 / qtA6), fixed top 5% (qt05) and the M3 top 30% (qb_TOP30), all packed with CM.pk in the formats of e57_pqa16_run.masks_from_npz (QA: keys g<bundle>_<name>_<chunk>) and e57_news_m3.UVNewsM3 (news: keys <name>_<chunk>, chunks >= K0). Also writes the mean / min / max share of rows per rule (overall and for the tune / hold-out parts).
  python3 e57_rule_masks_real.py --qa-out F.npz --news-out F.npz [--taus 2,4,6] | --selftest"""
import sys,json,argparse,pathlib
import numpy as np
HERE=pathlib.Path(__file__).resolve().parent;ROOT=HERE.parents[1];sys.path.insert(0,str(HERE))
EXP=ROOT/'experiments/exp07_write_ladder';CH=1024
import e57_closure_masks as CM,e57_rules as RU
def fmt(x):return ('%g'%x).replace('.','p')
def rule_dict(g,el,taus,rhos=()):
    r={f'qtA{fmt(t)}':RU.abs_rule(g,el,t) for t in taus};r.update({f'qg{fmt(x)}':RU.gainmass_rule(g,el,x/100) for x in rhos});r['qt05']=RU.top_share(g,el,.05);r['qb_TOP30']=RU.top_share(g,el,.30);return r
def qa(out,taus,rhos=(),tune=(8,10,12),hold=(5,9)):
    Z=np.load(EXP/'e57_public_qa_n65/e57_dwg_qa_nll.npz');bi=list(Z['bands']).index(128);res={};sh={}
    for gid in (5,8,9,10,12):
        win=Z[f'g{gid}_win'];bd=Z[f'g{gid}_band'][bi];lm=Z[f'g{gid}_lm']
        for k in range(win.shape[0]):
            el=CM.rows_eligible(lm[k],np.zeros(CH,np.int32))&lm[k];g=(bd[k]-win[k]).astype(np.float64)
            for nm,d in rule_dict(g,el,taus,rhos).items():res[f'g{gid}_{nm}_{k}']=CM.pk(d>0);sh.setdefault((gid,nm),[]).append(float((d>0).sum()/max(el.sum(),1)))
    np.savez_compressed(out,**res)
    part=lambda ids:{nm:float(np.mean([x for gid in ids for x in sh[(gid,nm)]])) for nm in {n for _,n in sh}}
    return dict(mean_share_tune=part(tune),mean_share_hold=part(hold),mean_share_all=part((5,8,9,10,12)))
def news(out,taus,rhos=()):
    import e55_uv as UVm,e57_t1t2v2_run as V,e57_news_m3 as NM
    ids=V.stream_ids(str(EXP/'e57_t1t2v2_in/e57_stream_sentences.jsonl'),'A');Z=np.load(EXP/'e57_band/e57_band_W0.npz');b=list(Z['bands']).index(128);ends=UVm.sentence_ends(UVm.tokenizer(str(ROOT/'data/tokenizer/llama3/tokenizer.json')),ids);res={};sh={}
    for k in range(NM.K0,len(ids)//CH):
        q=UVm.chunk_sents(ends,k,CH);sid=np.full(CH,-1)
        for i,(s_,e_) in enumerate(q):sid[s_:e_]=i
        lm=Z['lm'][k];el=CM.rows_eligible(lm,sid)&lm;g=(Z['nll_band'][b][k]-Z['nll_win'][k]).astype(np.float64)
        for nm,d in rule_dict(g,el,taus,rhos).items():res[f'{nm}_{k}']=CM.pk(d>0);sh.setdefault(nm,[]).append(float((d>0).sum()/max(el.sum(),1)))
    res['chunks']=np.asarray([NM.K0,len(ids)//CH]);np.savez_compressed(out,**res);return {nm:float(np.mean(v)) for nm,v in sh.items()}
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--qa-out');ap.add_argument('--news-out');ap.add_argument('--taus',default='2,4,6');ap.add_argument('--rhos',default='');ap.add_argument('--selftest',action='store_true');a=ap.parse_args()
    if a.selftest:
        r=np.random.default_rng(0);g=r.standard_normal(CH)*2+.8;el=r.random(CH)<.9;d=rule_dict(g,el,[2,4]);assert set(d)=={'qtA2','qtA4','qt05','qb_TOP30'};d2=rule_dict(g,el,[2],[50]);assert 'qg50' in d2 and 0<d2['qg50'].sum()<d2['qb_TOP30'].sum()*3 and d['qtA2'].sum()>=d['qtA4'].sum() and d['qb_TOP30'].sum()==round(.3*el.sum());print('e57_rule_masks_real selftest ok');return
    taus=[float(x) for x in a.taus.split(',') if x];rhos=[float(x) for x in a.rhos.split(',') if x];res={}
    if a.qa_out:res['qa']=qa(a.qa_out,taus,rhos);pathlib.Path(a.qa_out).with_suffix('.json').write_text(json.dumps(res['qa'],indent=1))
    if a.news_out:res['news']=news(a.news_out,taus,rhos);pathlib.Path(a.news_out).with_suffix('.json').write_text(json.dumps(res['news'],indent=1))
    print(json.dumps(res))
if __name__=='__main__':main()
