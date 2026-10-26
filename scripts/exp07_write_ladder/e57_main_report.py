"""E57 main run report (CPU only; reads E57_main_<ARM>.json and E57_main_W0.json; writes e57_main_report.md / .json). For every arm, cycle and read position: the metrics of section 39 by condition (OUT; IN combined, IN present, IN absent): accuracy over all candidates, mean rank ratio over all candidates, accuracy and mean rank ratio over the hard-only set, mean rank ratio over the full hard pool, mean cumulative exposure n_k and mean shadow level. Every table is given twice: all 42 questions and WITHOUT the two questions whose wording overlaps an article sentence (may-person-01, may-person-03; leakcheck). Differences to W0 and to the NAT arm at the same read position (paired by question) are listed for the end-of-cycle reads. No inference is drawn here (42 questions, 10 events: first observation); canary trajectories per arm are listed with the first chunk over the limit.
  python3 e57_main_report.py --dir DIR"""
import sys,json,argparse,pathlib,collections,statistics
EXCL=('may-person-01','may-person-03')
def mean(x):x=[v for v in x if v is not None];return statistics.mean(x) if x else None
def agg(rows):
    if not rows:return None
    return dict(n=len(rows),acc_all=mean([r['acc_all'] for r in rows]),rr_all=mean([r['rank_ratio_all'] for r in rows]),acc_hard=mean([r['acc_hard'] for r in rows]),rr_hard=mean([r['rank_ratio_hard'] for r in rows]),rr_pool=mean([r['rank_ratio_pool'] for r in rows]),n_k=mean([r['n_k'] for r in rows]),shadow=mean([r.get('shadow_mean') for r in rows]))
def f(x):return '-' if x is None else (f'{x:.3f}' if isinstance(x,float) else str(x))
def groups(rows):
    out={}
    for r in rows:
        key=(r['arm'],r['cycle'],r['read_chunk'] if r['cycle']==0 else r['read_chunk']%1000000,r['condition'])
        out.setdefault(key,[]).append(r)
    return out
def run(d):
    d=pathlib.Path(d);files={a:json.load(open(d/f'E57_main_{a}.json')) for a in ('W0','NAT','NATADAMH','ORC','RAND') if (d/f'E57_main_{a}.json').exists()}
    rows=[r for v in files.values() for r in v['rows']];first_over={a:v.get('first_chunk_over_canary_limit') for a,v in files.items() if a!='W0'}
    for r in rows:r['explore']=bool(r['cycle']>0 and first_over.get(r['arm']) is not None and r['read_chunk']>=first_over[r['arm']])      # E57 section 47: the canary is measured at the end of every cycle; a cycle whose end canary rise is above the limit (and every later cycle) is exploratory, kept apart
    L=['# E57 main run report (counts and means only; first observation, 42 questions, 10 events, no inference)','']
    for a,v in files.items():
        if a=='W0':L.append(f"- W0: canary {v['canary_w0']}")
        else:L.append(f"- {a}: run chunks {v['run_chunks']} of {v['n_chunks']}, first chunk over the canary limit: {v['first_chunk_over_canary_limit']}, update_norm mean {mean([e['update_norm'] for e in v['update_log']]):.3g}, sec {v['sec_total']:.0f}")
    L+=['','## canary trajectories (rise over W0)','','| arm | chunk | cycle | gen NLL rise | tail NLL rise |','|---|---|---|---|---|']
    w0=files['W0']['canary_w0'] if 'W0' in files else None
    for a,v in files.items():
        if a=='W0':continue
        for t in v['trajectory']:L.append(f"| {a} | {t['chunk']} | {t['cycle']} | {t['canary']['_gen_nll']-w0['_gen_nll']:+.4f} | {t['canary']['_tail_nll']-w0['_tail_nll']:+.4f} |")
    res={}
    for excl in (False,True):
        sel=[r for r in rows if not (excl and r['question'] in EXCL)];tag='without may-person-01 / may-person-03' if excl else 'all 42 questions'
        L+=['',f'## metrics per arm / cycle / read position ({tag})','','| arm | cycle | read chunk in cycle | cycle end canary over +0.10 (exploratory) | condition | n | acc all | rank ratio all | acc hard | rank ratio hard | rank ratio pool | mean n_k | mean shadow |','|---|---|---|---|---|---|---|---|---|---|---|---|---|']
        g=groups(sel);res[tag]={}
        for key in sorted(g,key=lambda k:(k[0]!='W0',k[0],k[1],k[2],k[3])):
            conds=[(key[3],g[key])]
            if key[3]=='IN':conds+=[('IN present',[r for r in g[key] if r['window_info']=='present']),('IN absent',[r for r in g[key] if r['window_info']=='absent'])]
            for cname,rr in conds:
                m=agg(rr)
                if m is None:continue
                L.append(f"| {key[0]} | {key[1]} | {key[2]} | {'yes' if g[key][0].get('explore') else 'no'} | {cname} | {m['n']} | {f(m['acc_all'])} | {f(m['rr_all'])} | {f(m['acc_hard'])} | {f(m['rr_hard'])} | {f(m['rr_pool'])} | {f(m['n_k'])} | {f(m['shadow'])} |")
                res[tag][f'{key[0]}|{key[1]}|{key[2]}|{cname}']=m
        # end-of-cycle paired differences to NAT and to W0 (by question)
        L+=['',f'### reads at the end of each cycle: paired differences ({tag}); positive acc / negative rank ratio = better than the reference','','| arm | cycle | condition | vs | mean diff acc all | mean diff rank ratio all | mean diff acc hard | mean diff rank ratio hard |','|---|---|---|---|---|---|---|---|']
        last={r['read_chunk']%1000000 for r in sel if r['cycle']>0};endpos=max(last) if last else None
        def by(arm,cyc,cond):return {r['question']:r for r in sel if r['arm']==arm and r['cycle']==cyc and r['read_chunk']%1000000==endpos and r['condition']==cond}
        w0={cond:{r['question']:r for r in sel if r['arm']=='W0' and r['read_chunk']==endpos and r['condition']==cond} for cond in ('OUT','IN')}
        for arm in ('NAT','NATADAMH','ORC','RAND'):
            for cyc in sorted({r['cycle'] for r in sel if r['arm']==arm and r['cycle']>0}):
                for cond in ('OUT','IN'):
                    A=by(arm,cyc,cond)
                    for ref,B in (('W0',w0[cond]),('NAT',by('NAT',cyc,cond) if arm!='NAT' else None)):
                        if not B or not A:continue
                        qs=[q for q in A if q in B];
                        if not qs:continue
                        dif=lambda k:statistics.mean(float(A[q][k])-float(B[q][k]) for q in qs)
                        L.append(f"| {arm} | {cyc} | {cond} | {ref} | {dif('acc_all'):+.3f} | {dif('rank_ratio_all'):+.3f} | {dif('acc_hard'):+.3f} | {dif('rank_ratio_hard'):+.3f} |")
    (d/'e57_main_report.md').write_text('\n'.join(L)+'\n');(d/'e57_main_report.json').write_text(json.dumps(res,indent=1,default=float));print('\n'.join(L[:12]));return res
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--dir',required=True);a=ap.parse_args();run(a.dir)
