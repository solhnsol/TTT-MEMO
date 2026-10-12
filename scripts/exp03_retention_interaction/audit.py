"""CPU-only paired interaction audit from frozen token layouts and saved outputs."""
import hashlib,json,math,statistics,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'experiments/exp03_retention_interaction/pilot8_20261012'
sys.path.insert(0,str(Path(__file__).resolve().parent))
from prepare import select_source_spans
from tokenizers import Tokenizer

def main():
    dataset=list(map(json.loads,(OUT/'dataset.jsonl').read_text().splitlines()));samples=list(map(json.loads,(OUT/'samples.jsonl').read_text().splitlines()))
    assert len(dataset)==len(samples)==8
    tok=Tokenizer.from_file(str(ROOT/'data/tokenizer/llama3/tokenizer.json'))
    for d,s in zip(dataset,samples):
        assert d['sample_id']==s['sample_id'] and d['needle_to_query_distance']>=8192
        oracle=d['selection']['ORACLE'];random=d['selection']['RANDOM'];assert len(oracle['ids'])==len(random['ids'])==d['selection']['budget']
        assert oracle['end']<=len(d['source_ids'])-8192
        assert random['end']<=len(d['source_ids'])-8192
        assert random['end']<=oracle['start'] or random['start']>=oracle['end']
        source_enc=tok.encode(d['source_text'],add_special_tokens=False)
        # source text/token boundary may retain a newline subpiece; selection must
        # agree from archived source offsets without query/gold parameters.
        assert source_enc.ids==d['source_ids']
        selected=select_source_spans(d['source_text'],d['source_ids'],source_enc.offsets,20261012+d['source_row'])
        assert selected==d['selection']
        assert s['OFF_inner_updates']==0 and s['ON_inner_updates']==s['source_native_chunks']
        assert s['same_ON_fast_state_all_retention_conditions']
        assert sum(w['loss_tokens'] for w in s['write_ON'])==s['source_loss_tokens']
        for w in s['write_ON']:assert w['fast_step_max_abs_delta']>0 and w['frozen_max_abs_delta']==0
        for retention in ('NONE','ORACLE','RANDOM'):
            r=d['recall'][retention];n=r['retained_tokens']
            assert n==(0 if retention=='NONE' else len(oracle['ids']))
            assert r['fixed_header_tokens']+r['recent_tail_tokens']+r['delimiter_tokens']+r['query_tokens']+n==8160
            assert len(r['prompt_ids'])==8160 and r['prompt_len']+32==8192
            expected=d['source_ids'][r['recent_tail_start']:]
            assert len(expected)==r['recent_tail_tokens']
            assert d['gold'] not in tok.decode(expected)
            for ttt in ('ON','OFF'):
                c=s['cells'][ttt+'_'+retention]
                assert c['recall_inner_updates']==0 and c['source_kv_discarded'] and not c['retained_span_retrained']
                assert c['accuracy']==int(d['gold'].lower() in c['prediction'].lower())
                assert c['prompt_sha256']==hashlib.sha256(json.dumps(r['prompt_ids']).encode()).hexdigest()
                assert len(c['gold_token_logp'])==len(c['gold_ids'])
                assert math.isfinite(c['gold_nll_mean'])
                assert abs(-statistics.mean(c['gold_token_logp'])-c['gold_nll_mean'])<1e-4
                assert abs(-sum(c['gold_token_logp'])-c['gold_nll_sum'])<1e-4
                assert c['prompt_tokens']+len(c['generated_ids'])<=8192
    cells={key:{'n_correct':sum(s['cells'][key]['accuracy'] for s in samples),'accuracy':statistics.mean(s['cells'][key]['accuracy'] for s in samples),'gold_nll_mean':statistics.mean(s['cells'][key]['gold_nll_mean'] for s in samples)} for key in samples[0]['cells']}
    interactions={}
    for ret in ('ORACLE','RANDOM'):
        rows=[]
        for s in samples:
            c=s['cells'];rows.append({'sample_id':s['sample_id'],'accuracy':(c['ON_'+ret]['accuracy']-c['ON_NONE']['accuracy'])-(c['OFF_'+ret]['accuracy']-c['OFF_NONE']['accuracy']),'nll_gain':(c['ON_NONE']['gold_nll_mean']-c['ON_'+ret]['gold_nll_mean'])-(c['OFF_NONE']['gold_nll_mean']-c['OFF_'+ret]['gold_nll_mean'])})
        interactions[ret]={'accuracy':statistics.mean(r['accuracy'] for r in rows),'nll_gain':statistics.mean(r['nll_gain'] for r in rows),'paired':rows}
    saved=json.loads((OUT/'summary.json').read_text())
    for k,c in cells.items():
        for f,v in c.items():assert abs(saved['conditions'][k][f]-v)<1e-8
    for ret,ints in interactions.items():
        for f in ('accuracy','nll_gain'):assert abs(saved['interaction'][ret][f]-ints[f])<1e-8
    report={'invariants_passed':True,'conditions':cells,'interaction':interactions,'total_native_ON_updates':sum(s['ON_inner_updates'] for s in samples),'total_ON_loss_tokens':sum(s['source_loss_tokens'] for s in samples),'min_needle_to_query_distance':min(d['needle_to_query_distance'] for d in dataset)}
    (OUT/'independent_audit.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
if __name__=='__main__':main()
