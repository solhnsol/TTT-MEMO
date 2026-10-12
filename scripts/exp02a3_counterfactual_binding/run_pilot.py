"""Counterfactual entity binding: fixed K=32, independent A/B writes.
D(e)=[S_A(v1|e)-S_A(v2|e)]-[S_B(v1|e)-S_B(v2|e)].
B_entity=D(E)-D(E_control). No generation or hyperparameter search.
"""
from __future__ import annotations
import os
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import random
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / 'scripts/exp02a_source_free_lora'
sys.path.insert(0, str(BASE))
import torch
from model_runtime import (load_runtime, reset_example_state, set_adapter_enabled,
                          lora_state_cpu, lora_max_abs_diff, frozen_watch_max_abs_diff)
from prompts import encode_write, encode_recall_prompt, encode_candidate
from evaluate import score_candidate_logprobs
from ttt_write import write_fact
from protocol import SPEC
from metrics import paired_bootstrap_ci


def metric(scores, field):
    preferences = {}
    for condition in ('A', 'B'):
        preferences[condition] = {
            entity: scores[condition][entity]['V1'][field] - scores[condition][entity]['V2'][field]
            for entity in ('E', 'E_control')
        }
    d = {e: preferences['A'][e] - preferences['B'][e] for e in ('E', 'E_control')}
    return {'preferences': preferences, 'D_E': d['E'], 'D_control': d['E_control'],
            'B_entity': d['E'] - d['E_control']}


def validate_metric():
    def pack(a, b):
        return {e: {'V1': {'logp_mean': x}, 'V2': {'logp_mean': 0}}
                for e, x in zip(('E', 'E_control'), (a, b))}
    assert metric({'A': pack(3, 3), 'B': pack(-2, -2)}, 'logp_mean')['B_entity'] == 0
    assert metric({'A': pack(3, 0), 'B': pack(-2, 0)}, 'logp_mean')['B_entity'] == 5


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--n', type=int, default=8, choices=(2, 4, 8))
    args = ap.parse_args()
    validate_metric()
    args.out.mkdir(parents=True, exist_ok=False)
    random.seed(20261012)
    torch.manual_seed(20261012)
    torch.cuda.manual_seed_all(20261012)
    started = time.monotonic()
    source = ROOT / 'scripts/exp02a1_binding_specificity/frozen/dataset_pairs.jsonl'
    pairs = [json.loads(l) for l in source.read_text().splitlines()]
    # Deterministic first exact pairs, selected before model evaluation; no outcome filtering.
    pairs = [p for p in pairs if p['info_type'] == 'exact'][:args.n]
    assert len(pairs) == args.n
    runtime = load_runtime()
    torch.cuda.reset_peak_memory_stats()
    trips = []
    for p in pairs:
        trip = {'triplet_id': p['pair_id'], 'E': p['e1'], 'E_control': p['e2'],
                'V1': p['v1'], 'V2': p['v2'],
                'template': p['write_template'],
                'question_template': 'What is the passphrase assigned to {entity}?'}
        lengths = [len(encode_candidate(runtime.tokenizer,
                    '<|im_start|>assistant\n', trip[v])['candidate_token_indices']) for v in ('V1', 'V2')]
        assert lengths[0] == lengths[1], (trip, lengths)
        trip['candidate_n_tokens'] = lengths
        trips.append(trip)
    dataset_text = ''.join(json.dumps(t, sort_keys=True) + '\n' for t in trips)
    (args.out / 'dataset.jsonl').write_text(dataset_text)
    config = {'experiment': 'exp02a3_counterfactual_binding', 'seed': 20261012,
              'n': args.n, 'k_steps': 32, 'base_spec': SPEC,
              'source_dataset': str(source), 'source_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
              'dataset_sha256': hashlib.sha256(dataset_text.encode()).hexdigest(),
              'D': '(A_V1-A_V2)-(B_V1-B_V2)', 'B_entity': 'D_E-D_control',
              'loss_mask': 'payload only; one textual fact, reused for 32 optimizer steps',
              'primary_score': 'mean token logp; sum logp diagnostic; EOS excluded',
              'selection': 'first exact pairs, before evaluation; no outcome filtering',
              'packages': {p: importlib.metadata.version(p) for p in ('torch','transformers','peft','accelerate')},
              'gpu': runtime.gpu_name, 'torch_cuda': torch.version.cuda}
    (args.out / 'config.json').write_text(json.dumps(config, indent=2))
    failures = []
    records = []
    original_forward = runtime.model.forward
    # Check actual model arguments and returned cache, including reused scorer/write helpers.
    def guarded_forward(*a, **kw):
        assert kw.get('past_key_values') is None
        assert kw.get('use_cache') is False
        out = original_forward(*a, **kw)
        assert getattr(out, 'past_key_values', None) is None
        return out
    runtime.model.forward = guarded_forward
    for trip in trips:
        scores = {}
        write_logs = {}
        prompts = {}
        for condition, value_key in (('A', 'V1'), ('B', 'V2')):
            optimizer = reset_example_state(runtime)
            assert lora_max_abs_diff(runtime.model, runtime.lora_init_cpu) == 0
            assert not optimizer.state
            fact = trip['template'].format(entity=trip['E'], payload=trip[value_key])
            question = trip['question_template'].format(entity=trip['E'])
            example = {'example_id': trip['triplet_id'], 'info_type': 'exact',
                       'fact_sentence': fact, 'write_payload': trip[value_key],
                       'answer': trip[value_key], 'question': question}
            encoded = encode_write(runtime.tokenizer, example)
            assert trip['E_control'] not in encoded.text
            write_logs[condition] = {'write_text': encoded.text,
                                    'decoded_oracle_tokens': runtime.tokenizer.decode(
                                        [encoded.input_ids[i] for i in encoded.oracle_token_indices]),
                                    **write_fact(runtime, encoded, optimizer, k_steps=32)}
            if not write_logs[condition]['loss_decreased']:
                failures.append(f"{trip['triplet_id']}/{condition}: write loss not decreased")
            optimizer.zero_grad(set_to_none=True)
            del encoded, optimizer, example
            assert all(p.grad is None for p in runtime.model.parameters())
            post_write = lora_state_cpu(runtime.model)
            scores[condition] = {}
            prompts[condition] = {}
            for entity_key in ('E', 'E_control'):
                probe = {'example_id': trip['triplet_id'], 'info_type': 'exact',
                         'fact_sentence': fact, 'write_payload': trip[value_key],
                         'answer': trip[value_key],
                         'question': trip['question_template'].format(entity=trip[entity_key])}
                prompt = encode_recall_prompt(runtime.tokenizer, probe, source_present=False)
                assert all(trip[v] not in prompt['text'] for v in ('V1', 'V2'))
                prompts[condition][entity_key] = prompt['text']
                scores[condition][entity_key] = {}
                for v in ('V1', 'V2'):
                    candidate = encode_candidate(runtime.tokenizer, prompt['text'], trip[v])
                    result = score_candidate_logprobs(runtime, candidate)
                    result['scoring_text'] = candidate['text']
                    result.pop('token_ids', None)
                    scores[condition][entity_key][v] = result
                del probe, prompt
            assert lora_max_abs_diff(runtime.model, post_write) == 0
            assert frozen_watch_max_abs_diff(runtime.model, runtime.frozen_watch_cpu) == 0
            del post_write
            reset_example_state(runtime)
        assert prompts['A'] == prompts['B']
        assert write_logs['A']['write_text'].replace(trip['V1'], '{VALUE}') == write_logs['B']['write_text'].replace(trip['V2'], '{VALUE}')
        record = {'triplet': trip, 'write': write_logs, 'recall_prompts': prompts, 'scores': scores,
                  'mean_logp': metric(scores, 'logp_mean'), 'sum_logp': metric(scores, 'logp_sum')}
        records.append(record)
        with (args.out / 'samples.jsonl').open('a') as f:
            f.write(json.dumps(record) + '\n')
        print(json.dumps({'triplet': trip['triplet_id'], **record['mean_logp']}), flush=True)
    assert lora_max_abs_diff(runtime.model, runtime.lora_init_cpu) == 0
    summary = {'n': len(records), 'sanity_passed': not failures, 'failures': failures,
               'elapsed_seconds': time.monotonic() - started,
               'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
               'peak_reserved_bytes': torch.cuda.max_memory_reserved(),
               'invariants': ['same initial LoRA, fresh empty optimizer per condition',
                              'same write except value; same A/B recall prompts',
                              'source/value absent from recall; actual cache return is None',
                              'no gradients/optimizer at recall; LoRA unchanged during recall',
                              'all base parameters frozen; watched base tensors unchanged',
                              'final adapter reset; matched candidate token lengths']}
    for field in ('mean_logp', 'sum_logp'):
        b = [r[field]['B_entity'] for r in records]
        summary[field] = {'mean_D_E': sum(r[field]['D_E'] for r in records)/len(records),
                         'mean_D_control': sum(r[field]['D_control'] for r in records)/len(records),
                         'mean_B_entity': sum(b)/len(b), 'n_positive': sum(x > 0 for x in b),
                         'descriptive_bootstrap': paired_bootstrap_ci(b, [0]*len(b), seed=20261012)}
    (args.out / 'summary.json').write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2), flush=True)
    if failures:
        raise RuntimeError(failures)

if __name__ == '__main__':
    main()
