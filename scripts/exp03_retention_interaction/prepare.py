"""Frozen 8-sample S-NIAH pilot; selectors receive source only, never query/gold."""
import hashlib,json,random,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'scripts/baseline_125m/common'))
from ruler_niah_labels import NEEDLE_RE,QUERY_SINGLE,overlapping_token_indices
from tokenizers import Tokenizer
SOURCE=ROOT/'data/ruler_niah/S-NIAH/niah_single_2/16k/niah_single_2_16k_n200.jsonl'
TOKENIZER=ROOT/'data/tokenizer/llama3/tokenizer.json'
BUDGET=8192;MAX_NEW=32;HEADER=128 # historical prototype; actual scaffold length is source-template-only

def select_source_spans(source_text,source_ids,source_offsets,seed):
    needles=list(NEEDLE_RE.finditer(source_text))
    assert len(needles)==1
    needle=needles[0]
    positions=overlapping_token_indices(source_offsets,needle.start(),needle.end())
    a,b=positions[0],positions[-1]+1
    if a<256 or b>len(source_ids)-BUDGET:return None
    n=b-a
    rng=random.Random(seed)
    starts=[i for i in range(256,len(source_ids)-BUDGET-n+1) if i+n<=a or i>=b]
    assert starts
    r=rng.choice(starts)
    return {'ORACLE':{'start':a,'end':b,'ids':source_ids[a:b],'text':needle.group(0)},'RANDOM':{'start':r,'end':r+n,'ids':source_ids[r:r+n]},'budget':n}

def prepare(out,n=8):
    tok=Tokenizer.from_file(str(TOKENIZER));delimiter=tok.encode('\n\n',add_special_tokens=False).ids
    records=[]
    for rownum,line in enumerate(SOURCE.read_text().splitlines()):
        raw=json.loads(line);text=raw['input'];q=list(QUERY_SINGLE.finditer(text));assert len(q)==1
        enc=tok.encode(text,add_special_tokens=False)
        qstart=overlapping_token_indices(enc.offsets,q[0].start(),q[0].end())[0]
        source_ids=enc.ids[:qstart];source_text=text[:enc.offsets[qstart][0]]
        selected=select_source_spans(source_text,source_ids,enc.offsets[:qstart],20261012+rownum)
        if selected is None:continue
        # Future query and answer are accessed only after selector has returned.
        needle=list(NEEDLE_RE.finditer(source_text))[0]
        assert needle.group(3)==raw['outputs'][0] and len(raw['outputs'])==1
        query_text=text[enc.offsets[qstart][0]:]+raw['answer_prefix']+' '
        query_enc=tok.encode(query_text,add_special_tokens=False)
        gold_enc=tok.encode(query_text+raw['outputs'][0],add_special_tokens=False)
        gold_start=len(query_text)
        gold_positions=overlapping_token_indices(gold_enc.offsets,gold_start,len(query_text+raw['outputs'][0]))
        assert gold_enc.ids[:gold_positions[0]]==query_enc.ids, 'unstable answer boundary'
        assert tok.decode([gold_enc.ids[i] for i in gold_positions]).strip()==raw['outputs'][0]
        instruction='I will quiz you about the number afterwards.'
        instruction_end=source_text.index(instruction)+len(instruction)
        header_n=overlapping_token_indices(enc.offsets[:qstart],0,instruction_end)[-1]+1
        recall={}
        prompt_len=BUDGET-MAX_NEW
        for retention in ('NONE','ORACLE','RANDOM'):
            retained=[] if retention=='NONE' else selected[retention]['ids']
            tail_n=prompt_len-header_n-len(query_enc.ids)-2*len(delimiter)-len(retained)
            assert tail_n>0 and selected['ORACLE']['end']<=len(source_ids)-tail_n
            tail_start=len(source_ids)-tail_n
            header=source_ids[:header_n];tail=source_ids[tail_start:]
            ids=header+tail+delimiter+retained+delimiter+query_enc.ids
            assert len(ids)==prompt_len and ids[0]==128000
            value=raw['outputs'][0]
            assert value not in tok.decode(header+tail+delimiter+delimiter)
            if retention=='RANDOM':assert value not in tok.decode(retained)
            recall[retention]={'prompt_ids':ids,'prompt_len':len(ids),'recent_tail_start':tail_start,'recent_tail_tokens':tail_n,'retained_tokens':len(retained),'query_tokens':len(query_enc.ids),'fixed_header_tokens':header_n,'delimiter_tokens':2*len(delimiter),'gold_ids':[gold_enc.ids[i] for i in gold_positions],'prompt_sha256':hashlib.sha256(json.dumps(ids).encode()).hexdigest()}
        rec={'sample_id':raw['sample_id'],'source_row':rownum,'task':raw['task'],'source_ids':source_ids,'source_text':source_text,'source_query_boundary_j':qstart,'needle_to_query_distance':qstart-selected['ORACLE']['end']+1,'selection':selected,'query_text':query_text,'gold':raw['outputs'][0],'recall':recall,'source_sha256':hashlib.sha256(source_text.encode()).hexdigest()}
        records.append(rec)
        if len(records)==n:break
    assert len(records)==n
    out.mkdir(parents=True,exist_ok=False)
    (out/'dataset.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in records))
    metadata={'n':n,'source_path':str(SOURCE),'source_file_sha256':hashlib.sha256(SOURCE.read_bytes()).hexdigest(),'tokenizer_sha256':hashlib.sha256(TOKENIZER.read_bytes()).hexdigest(),'local_budget':BUDGET,'max_new_tokens':MAX_NEW,'prompt_tokens':BUDGET-MAX_NEW,'selection':'first eligible records, sole source needle found without query; random historical background span same tokens, source-only seed','oracle_upper_bound':True,'min_needle_to_query_distance':min(r['needle_to_query_distance'] for r in records)}
    (out/'dataset_metadata.json').write_text(json.dumps(metadata,indent=2));return records
if __name__=='__main__':
    records=prepare(Path(sys.argv[1]));print([(r['sample_id'],r['needle_to_query_distance'],r['selection']['budget']) for r in records])
