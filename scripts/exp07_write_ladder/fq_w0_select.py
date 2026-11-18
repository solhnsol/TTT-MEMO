"""CPU, new file: turns the W0 screen (fq_screen_run.py output fq_w0screen_<label>.json) into the final worlds. For each world: dropped = world facts with margin > 0; reserve facts with margin > 0 are removed from the reserve list (they are never used as replacement); the remaining reserve facts replace the dropped facts in the order of the reserve (positions kept, as fq_w0_apply.py does: same group, event not yet used). The filtered world is written with the reserve reduced and passed to the unchanged fq_w0_apply.py (finish() + checks). If the non-positive reserve runs out: stop and report.
  python3 fq_w0_select.py --worlds DIR --screen DIR --pool INDIR --out OUT   (INDIR = the fq pool dir argument of fq_w0_apply.py) | --selftest"""
import sys,json,subprocess,argparse,pathlib
FWD=pathlib.Path('/home/solhnsol/workspace/TTT-MEMO/experiments/exp07_write_ladder/fq/code/fq_w0_apply.py')
def plan(scr,NF=16):
    f=scr['facts'];drop=[x['fact_id'] for x in f if x['role']=='world' and x['screened_out']];bad_res={x['idx']-16 for x in f if x['role']=='reserve' and x['screened_out']};return drop,bad_res
def filtered_world(w,bad_res,drop=()):
    w=json.loads(json.dumps(w));w['ctrl_extra']=[f for f in w['facts'] if f['id'] in set(drop)];w['reserve']=[r for i,r in enumerate(w['reserve']) if i not in bad_res];w['w0_screen_removed_reserve_ids']=sorted(bad_res);return w
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--worlds');ap.add_argument('--screen');ap.add_argument('--pool');ap.add_argument('--out');ap.add_argument('--selftest',action='store_true');a=ap.parse_args()
    if a.selftest:
        scr=dict(facts=[dict(role='world',fact_id=i,screened_out=i in (2,5)) for i in range(16)]+[dict(role='reserve',fact_id=i,idx=16+i,screened_out=i==0) for i in range(8)]);d,b=plan(scr);assert d==[2,5] and b=={0}
        w=filtered_world(dict(facts=[],reserve=[dict(id=100+i) for i in range(8)]),b);assert len(w['reserve'])==7 and w['reserve'][0]['id']==101;print('fq_w0_select selftest ok');return
    out=pathlib.Path(a.out);out.mkdir(parents=True,exist_ok=True);rep={}
    for sc in sorted(pathlib.Path(a.screen).glob('fq_w0screen_*.json')):
        s=json.load(open(sc));bs=s['builder_seed'];wf=pathlib.Path(a.worlds)/f'fq_{bs}_cap3_VAR.json';w=json.load(open(wf));drop,bad=plan(s);usable=len(w['reserve'])-len(bad)
        rep[s['label']]=dict(builder_seed=bs,dropped=drop,reserve_screened_out=sorted(bad),reserve_left=usable-len(drop))
        if usable<len(drop):rep[s['label']]['STOP']='reserve exhausted';continue
        tmp=out/f'_tmp_{bs}.json';json.dump(filtered_world(w,bad,drop),open(tmp,'w'),ensure_ascii=False)
        r=subprocess.run([sys.executable,'-I',str(FWD),str(tmp),','.join(map(str,drop)),a.pool,str(out/f'fq_{bs}_cap3_VAR_w0.json')],capture_output=True,text=True);rep[s['label']]['apply']=(r.stdout+r.stderr).strip()[-300:];tmp.unlink()
    json.dump(rep,open(out/'fq_w0_select_report.json','w'),indent=1);print(json.dumps(rep,indent=1))
if __name__=='__main__':main()
