import pickle,re,json
import sys,os;sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)))
from collections import Counter,defaultdict
from common import W as OUT
M={'targets':{},'pleads':{},'back':defaultdict(list),'bcount':Counter(),'given':Counter(),'surn':Counter()}
for i in range(41):
    d=pickle.load(open(f'{OUT}/scan1_{i}.pkl','rb'))
    M['targets'].update(d['targets']);M['pleads'].update(d['pleads'])
    for k,v in d['back'].items():M['back'][k]+=v
    for k in ('bcount','given','surn'):M[k].update(d[k])
M['back']=dict(M['back'])
pickle.dump(M,open(f'{OUT}/scan1_all.pkl','wb'))
print('targets',len(M['targets']),'pleads',len(M['pleads']),'given keys',len(M['given']),'surn keys',len(M['surn']))
