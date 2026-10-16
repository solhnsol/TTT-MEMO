import json,glob,sys,os,statistics as st,random
from collections import Counter
sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)));import nat_world as nw
OUT=sys.argv[1];priv=json.load(open(f'{OUT}/private/real_name_map.json'))
S=[];tot=Counter();natg=Counter();gen=Counter();fu_all=[]
for f in sorted(glob.glob(f'{OUT}/nat_*_spread.json')):
    w=nw.load_world(f);s=w['stats'];fam=nw.build_items(w);c=json.load(open(f.replace('_spread','_clustered')))['stats']
    fu=[s['fact_units'].get(str(t['id']),0) for t in w['triples']];fu_all+=fu
    natg.update(t['natg'] for t in w['triples']);gen.update(t['gender'] for t in w['triples']);tot.update('+'.join(priv[f"{w['seed']}:{t['id']}"]['src']) for t in w['triples'])
    S.append(dict(stream=w['seed'],pairs=w['pairs'],fact_units_mean=round(st.mean(fu),2),persons_fact_ge2=sum(x>=2 for x in fu),in8k_spread=round(s['in_window_ratio'] or 0,2),in8k_clustered=round(c['in_window_ratio'] or 0,2),L3=len(fam['L3']),L3b=len(fam['L3b']),L3e=len(fam['L3e']),expo_tokens_llama=s['expo_tokens_llama']))
summ={'streams':S,'pairs':sum(x['pairs'] for x in S),'persons':len(fu_all),'fact_units_mean':round(st.mean(fu_all),2),'fact_units_hist':dict(sorted(Counter(fu_all).items())),'natg':dict(natg.most_common()),'gender':dict(gen),'src':dict(tot)}
json.dump(summ,open(f'{OUT}/summary.json','w'),indent=1);print(json.dumps(summ,indent=0)[:1500])
w=nw.load_world(f'{OUT}/nat_9001_spread.json');A,n,sp=nw.realize(w,False);random.seed(11)
L=['# 자연 텍스트 stream 샘플 (stream 9001, 분산 배치, A 세계)','','이름은 모두 새로 만든 이름입니다. 각 단위 앞 표지는 [사람 id · 라운드 · 노출 표지]입니다. 배경 문단은 뺐습니다.','']
for s_ in sorted(random.sample(sp,20),key=lambda x:x['start']):L+=[f"**[t{s_['t']} · r{s_['round']} · {s_['label']}]** {s_['text']}",'']
L+=['## 판독 문항 예시 (family마다 하나)','']
for k,v in nw.build_items(w).items():
    if v:it=v[0];L.append(f"- {k}: {it['prefix']} → gold `{it['gold'].strip()}` / foil `{it['foil'].strip()}`")
open(f'{OUT}/samples.md','w').write('\n'.join(L)+'\n')
