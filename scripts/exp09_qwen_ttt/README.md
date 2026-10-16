# Exp09: the Exp07 write ladder on Qwen3-4B with LoRA test-time training

This directory ports the Exp07 write ladder (`scripts/exp07_write_ladder/ladder.py`, TTT-E2E 3B in JAX) to a second
model family. The new model is Qwen3-4B-Instruct-2507, and its fast weights are LoRA adapters trained by plain NTP
SGD at test time. The worlds, exposure sentences, probe items, readout contexts, canaries, log format and analysis
are the same as Exp07. The model, the fast weights and the inner rule are what change.
Nothing under `e2e/`, `scripts/exp04*`, `scripts/exp05*`, `scripts/exp07*` or `experiments/exp07*` is modified.

## Files

| file | role |
|---|---|
| `world_text.py` | Builds the texts without JAX: worlds (imports `gen_world.py` directly), probe items, the pg1342 tail, the SQuAD tail, the pg11 canary, L1 rows, filler-swap streams. It works with any `tokenizers.Tokenizer`. |
| `qwen_runtime.py` | Loads Qwen3 (bf16) or builds a tiny random Qwen3 for tests. Wraps PEFT LoRA in fp32, provides the frozen-weight watch and LoRA state save and restore, and casts the base in place for an fp32 readout. |
| `inner.py` | Inner rule: native (sliding-window) batches, masked batches, L1 packing, and the clipped SGD step. |
| `readout.py` | Encodes the context once and scores probes in batches from its KV cache. Also computes the canary NLLs and the `Evaluator` that produces the exp07 `margins` dict. |
| `qwen_ladder.py` | Driver: `--level L0/L0mp/L0mw/L0mwfs/L1`. |
| `analyze_qwen.py` | Analysis. It reuses `analyze_ladder.py` and `analyze_stage01.py` from exp07, unmodified, through `EXP07_OUT`. |
| `ref_dump_jax.py` | Runs in the JAX env. It dumps the JAX pipeline's own texts and llama3 token lists, which test 7 compares against. |
| `test_*.py`, `_testutil.py` | CPU unit tests on a tiny random Qwen3. |

## Defaults (every one is a CLI flag)

| setting | default | flag |
|---|---|---|
| model | cached `Qwen/Qwen3-4B-Instruct-2507`, bf16 base, sdpa attention | `--model`, `--base-dtype`, `--attn` |
| fast weights | LoRA r=16, alpha=32 (scaling 2), dropout 0, fp32, on `gate_proj,up_proj,down_proj` of the last 9 of 36 layers (27-35). PEFT default init: A is Kaiming, B=0, so W0 equals the base model. 5.31M params. | `--lora-r --lora-alpha --lora-targets --lora-layers` |
| inner rule | Plain SGD with one step per chunk. The loss is the mean token NTP loss over the chunk's own tokens. Global-norm clip 1.0 (coef = 1 if norm < clip, else clip/norm, as in optax). Step = lr x scale x coef x g. | `--lr 1e-3 --scale 1.0 --clip 1.0` |
| chunk / window | 1024-token chunks. Native context is the previous 7 chunks (an 8K window), recomputed every step with no KV cache. | `--chunk --window-chunks` |
| stream | BOS + the whole realised world text, cut into ceil(n/1024) chunks; the last chunk is partial. | `--stream-chunks cover` (or e.g. `63` to truncate like JAX), `--max-chunks` (smoke) |
| BOS | `<|endoftext|>` (151643) before the stream, the tails and the canary. Probes get no BOS, same as JAX, where the probes start with the 48-token filler slot. | `--bos endoftext|none` |
| L1 | Isolated `'\n\n'+sentence` rows. They are packed first-fit into steps of at most 1024 tokens; positions restart per row, which makes Qwen3's mask block-diagonal. The order is shuffled every epoch with `random.Random(--seed)`, as in ladder.py. 8 epochs. | `--variant plain|ctx|irr|fill --epochs --seed` |
| L0mp / L0mw / L0mwfs | 8 passes. Weights carry over between passes and every pass restarts at the stream start. L0mwfs gets new background each pass (draw `7000+p`), cut to the pass-1 length. | `--passes` |
| L0 | One pass, evaluated every 16 chunks and at the end (`state` key). | `--eval-every` |
| readout | R1 is the pg1342 tail, 7168 tokens including BOS. R1-alt is the SQuAD tail, 7168 tokens. All 276 items (552 rows), in batches of 8, with gold-minus-foil summed candidate log-prob. Logits are fp32. The base computes in bf16 unless `--readout-dtype fp32`, which casts the frozen base to fp32 in place for the readout and back afterwards; that round trip is exact. | `--tail-len --readout-batch --readout-dtype` |
| canaries | `_gen_nll`: mean NLL over the first 8192 tokens of pg11. `_tail_nll`: the same over the R1 tail. | `--canary-len` |

## Output

The output format is identical to exp07, so the exp07 analysis code reads it:
- `experiments/exp09_qwen_ttt/{LEVEL}_{TAG}_s{SEED}_{A,B}.json` contains `{'log':[...], 'sec':...}`. Evaluation entries have `epoch` (or `state` for L0) and `margins` = `{family: [per-item margins], '_gen_nll', '_tail_nll', '_alt': {family: [...]}}`, plus `lora_hash`, `cum_update_norm`, and per-step `ce/grad_norm/clip_coef` lists for the L0 levels. L1 also logs one entry per step (`epoch, step, ce, grad_norm, clip_coef, update_norm, cum_update_norm`).
- `meta_{LEVEL}_{TAG}_s{SEED}.json` contains `items` and `fam_sizes` (family order L3, XP, PS, FN, FP, FN0, FP0, L3r, L2), all args, the LoRA config, stream token and chunk counts, Qwen token lengths of the names, versions, and the frozen hash.
- `analyze_qwen.py --seed 101 L1:main L0mp:main L0mw:main` prints D (R1, R1-alt), acc/flip, familiarity gain, M_L3r, M_L2 and the canary rise per evaluation point. It needs `meta_L1_main_s{SEED}.json`, as `analyze_ladder.py` does.

## Differences from TTT-E2E (Exp07)

1. **No meta-learning.** TTT-E2E's initial weights and per-step inner-lr multipliers were trained by the outer loop for this inner rule. Here W0 is an off-the-shelf instruct model and the LoRA rule is plain NTP SGD. The lr has to be tuned; the default 1e-3 is a placeholder. The JAX config used lr 1.0 times a learned multiplier, which is not transferable.
2. **Fast weights.** TTT-E2E updates the full suffix-MLP weights. Here a rank-16 LoRA delta on the MLPs of the last 9 of 36 layers is updated. Because B starts at 0, the first step moves only B (dL/dA = 0), which is standard LoRA behaviour. A full-rank variant is not implemented.
3. **Context during the write.** TTT-E2E carries the suffix KV of earlier chunks. Those KVs were computed with older weights and receive no gradient. Here the 7-chunk context is recomputed with the current weights at every step, and gradient flows through the context positions in the LoRA layers too. Native steps therefore cost about 8 chunk-forwards.
4. **Attention.** TTT-E2E uses an 8K sliding window everywhere. Qwen uses full attention, but every sequence here is at most 8K tokens (window + chunk, or tail + probe), so the effective context matches.
5. **Tokenizer and stream length.** The world text is identical, but Qwen tokenizes it to 66.0-66.2K tokens, i.e. 65 chunks of 1024 under `cover`. JAX truncated to 63 x 1024 llama tokens. The extra 1-2 chunks are trailing background only: exposures end by JAX chunk 52. Use `--stream-chunks 63` to truncate instead. Names were length-matched under llama3. Under Qwen, seed 103 has two name pairs whose names differ in token length (2 vs 3 tokens: Nevek/Ashane, Rasume/Yarish); every place and every name in seeds 101 and 102 is 2 tokens. L3, the binding family, scores places, so it is unaffected. L2, L3r and FN score names.
6. **BOS.** Qwen has no BOS by default. `<|endoftext|>`, the Qwen document separator, takes llama's `<|begin_of_text|>` position.
7. **Readout implementation.** JAX packed several probes per 1024-token chunk under a custom mask, with a fully fp32 model. Here each probe row runs against a copy of the context KV cache. That is mathematically the same conditioning (test 6). The default compute is bf16 with fp32 logits. The JAX notes found about 0.05 nat per candidate of bf16 rounding noise; `--readout-dtype fp32` removes it.
8. **Instruct model on raw text.** No chat template is used anywhere. Streams, tails and probes are plain text, as in Exp07.

## Tests (CPU only, tiny random Qwen3: 4 layers, hidden 128, real tokenizer)

Run every test with the GPU hidden:
```
cd scripts/exp09_qwen_ttt
for t in test_lora_step test_window_packing test_readout test_texts_vs_jax test_smoke_e2e; do
  CUDA_VISIBLE_DEVICES= ../../.venv-exp02a/bin/python $t.py || echo FAILED $t; done
```
1-3 `test_lora_step.py`: W0 identity; one step equals the hand-computed clipped SGD step, both clipped and unclipped; only LoRA changes (frozen hash).
4-5 `test_window_packing.py`: the sliding-window step equals the explicit concatenation; labels cover chunk k only; a token before the window has no effect; the masked write ignores earlier chunks; the packed L1 step equals the rows run alone (plus a negative control).
6 `test_readout.py`: cached equals full-concatenation scoring at a 7168-token context; the gold-minus-foil sign is checked in both directions; the canary is checked.
7 `test_texts_vs_jax.py`: texts, sentences, items, L1 rows (4 variants), fs streams, tails and the canary are identical to the JAX pipeline for seeds 101-103. This test runs `ref_dump_jax.py` in `.venv`.
8 `test_smoke_e2e.py`: a driver smoke run (3x128-token chunks, 2 passes or epochs, both worlds; L1, L1 ctx, L0mp, L0mw, L0mwfs, L0). Exp07 analysis runs on the output.

## Running on the GPU (only when the A100 is free)

```
cd scripts/exp09_qwen_ttt; PY=../../.venv-exp02a/bin/python
$PY qwen_ladder.py --level L1   --world-seed 101 --epochs 8 --lr 1e-3 --tag main
$PY qwen_ladder.py --level L0mp --world-seed 101 --passes 8 --lr 1e-3 --tag main
$PY qwen_ladder.py --level L0mw --world-seed 101 --passes 8 --lr 1e-3 --tag main
$PY analyze_qwen.py --seed 101 L1:main L0mp:main L0mw:main
```
Each run does worlds A then B and skips an output file that already exists. Tune `--lr` first. A reasonable procedure is a short L1 and L0mp sweep (for example 1e-4, 3e-4, 1e-3, 3e-3) that watches the canary rise (`_gen_nll`, `_tail_nll`; the exp07 limit was 0.10 nat) and D. Add `--deterministic` for bit-reproducible reruns.

### Rough cost on one A100 80GB (estimates, not measured)

Qwen3-4B forward costs about 7.5 GFLOP/token in matmuls, plus about 2.4 GFLOP/token of attention at a 4K average context. Backward only goes through the last 9 layers. Assumed throughput is 35-50% MFU in bf16.
- Native step (8K window): about 100 TFLOP, so about 0.7-1.0 s. One pass of 65 chunks takes about 1 min.
- Masked step (1K tokens): about 10 TFLOP, so about 0.1-0.2 s, or 10-15 s per pass.
- L1 epoch: about 9.4K exposure tokens in about 10 packed steps, so about 2-3 s.
- Evaluation (bf16 readout): 2 x (7168-token context + 552 probe rows of 70 tokens or fewer, batch 8) plus the two canaries comes to about 15-30 s. With `--readout-dtype fp32` it is about 1-2 min, because fp32 matmuls run without TF32 by default.
- Model load plus world build: about 1-2 min per process.

| run (per world; double for A+B, per seed) | bf16 readout | fp32 readout |
|---|---|---|
| L1, 8 epochs (9 evaluations) | ~0.05-0.1 h | ~0.2-0.3 h |
| L0mp, 8 passes | ~0.15-0.3 h | ~0.3-0.5 h |
| L0mw, 8 passes | ~0.05-0.1 h | ~0.2-0.3 h |

All three levels for one seed with both worlds come to about 0.5-1 GPU-hour with the bf16 readout, or about 1.5-2 GPU-hours with fp32.

### Memory

- bf16 weights take about 8 GB (tied embeddings). The LoRA params are 21 MB; their grads are the same size.
- Native step: autograd keeps activations only for the 9 LoRA layers. The first 27 layers run without grad history because no tensor there requires grad. That is about 1 GB per layer at 8K tokens, about 9-12 GB in total. Logits are formed only at the 1024 loss positions: 1024 x 151936 fp32 is 0.6 GB, and its grad is the same size. The peak is about 25 GB.
- Readout: the context KV is 1.06 GB in bf16 (36 layers x 8 KV heads x 128 x 7168 x K,V). A batch of B rows copies it B times, so batch 8 needs about 9 GB. With `--readout-dtype fp32` the base doubles to about 16 GB during the readout and the KV is fp32, so use `--readout-batch 4`. The peak is about 40 GB.
- Do not start anything while the JAX job holds the GPU. Every test here is CPU-only and asserts `CUDA_VISIBLE_DEVICES=`.
