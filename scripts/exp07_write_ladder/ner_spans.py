"""NER PERSON spans for the label-free suppression screen (PROTOCOL_stage01.md "Label-free suppression screen", 2026-10-16
22:15 KST). Runs in the torch env (.venv-exp02a), CPU. For each world seed and world (A, B) the realized world text is tagged
by dslim/bert-base-NER (pinned revision), aggregation "simple", entity group PER, score >= 0.50, windows of <= 400 words with
50-word overlap; char spans are snapped to word boundaries and merged. Writes OUT/ner_spans_s{seed}_{A,B}.json with the
spans and the quality report (precision / recall of PER spans vs the gold person-name occurrences, target-name detection
rate, non-target detections). Usage: CUDA_VISIBLE_DEVICES= .venv-exp02a/bin/python ner_spans.py 101 102 103"""
import sys,os,re,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'scripts/exp04_what_ttt_retains'))
import gen_world as gw
MODEL='dslim/bert-base-NER';REV='d1a3e8f13f8c3566299d95fcfc9a8d2382a9affc';THR=0.50;WIN=400;OV=50
OUT=Path(os.environ.get('EXP07_OUT',ROOT/'experiments/exp07_write_ladder'))
def words(text):return [(m.start(),m.end()) for m in re.finditer(r'\S+',text)]
def tag(nlp,text):
    W=words(text);spans=[];i=0
    while i<len(W):
        j=min(len(W),i+WIN);a=W[i][0];b=W[j-1][1];seg=text[a:b]
        for e in nlp(seg):
            if e['entity_group']=='PER' and e['score']>=THR:spans.append((a+e['start'],a+e['end'],float(e['score'])))
        if j==len(W):break
        i=j-OV
    # snap to word boundaries (strip punctuation), merge overlaps
    out=[]
    for s,e,sc in sorted(spans):
        while s>0 and text[s-1].isalnum():s-=1
        while e<len(text) and text[e].isalnum():e+=1
        while s<e and not text[s].isalnum():s+=1
        while e>s and not text[e-1].isalnum():e-=1
        if e<=s:continue
        if out and s<=out[-1][1]:out[-1]=(out[-1][0],max(out[-1][1],e),max(out[-1][2],sc))
        else:out.append((s,e,sc))
    return out
def gold(world,swap):
    text,name,spans=gw.realize(world,swap);g=[]
    for n in set(name.values()):
        g+=[(m.start(),m.end(),n) for m in re.finditer(r'\b'+re.escape(n)+r'\b',text)]
    return text,sorted(g),spans
def main():
    from transformers import pipeline
    nlp=pipeline('token-classification',model=MODEL,revision=REV,aggregation_strategy='simple',device=-1)
    for seed in map(int,sys.argv[1:]):
        w=gw.build_world(seed)
        for wn,swap in (('A',False),('B',True)):
            text,g,_=gold(w,swap);d=tag(nlp,text)
            gs={(a,b) for a,b,_ in g};ds={(a,b) for a,b,_ in d}
            tp=len(gs&ds);prec=tp/max(len(ds),1);rec=tp/max(len(gs),1)
            # target-name detection: gold occurrences overlapped by any detected span
            hit=sum(1 for a,b,_ in g if any(s<b and e>a for s,e,_ in d))
            nontarget=[text[s:e] for s,e,_ in d if not any(s<b and e>a for a,b,_ in g)]
            rep={'seed':seed,'world':wn,'model':MODEL,'revision':REV,'threshold':THR,'n_gold':len(g),'n_detected':len(d),'exact_tp':tp,
                 'precision_exact':prec,'recall_exact':rec,'target_detection_rate':hit/max(len(g),1),'n_nontarget':len(nontarget),
                 'nontarget_examples':sorted(set(nontarget))[:40]}
            json.dump({'report':rep,'spans':[[s,e,sc] for s,e,sc in d]},open(OUT/f'ner_spans_s{seed}_{wn}.json','w'),indent=1)
            print(json.dumps(rep),flush=True)
if __name__=='__main__':main()
