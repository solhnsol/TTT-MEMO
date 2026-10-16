# Exp09: the Exp07 write ladder on Qwen3-4B-Base with LoRA test-time training

**Design approved 2026-10-16.** This directory ports the Exp07 write ladder (`scripts/exp07_write_ladder/ladder.py`,
TTT-E2E 3B in JAX) to a second model family. The new model is Qwen3-4B-Base by default; Instruct-2507 can be selected
with a flag. Its fast weights are LoRA adapters trained by plain NTP SGD at test time. The worlds, exposure sentences,
probe items, readout contexts, canaries, log format and analysis are the same as Exp07. The model, the fast weights
and the inner rule are what change.
Nothing under `e2e/`, `scripts/exp04*`, `scripts/exp05*`, `scripts/exp07*` or `experiments/exp07*` is modified.

## Files

| file | role |
|---|---|
| `world_text.py` | Builds the texts without JAX: worlds (imports `gen_world.py` directly), probe items, the pg1342 tail, the SQuAD tail, the pg11 canary, L1 rows, filler-swap streams. It works with any `tokenizers.Tokenizer`. |
| `qwen_runtime.py` | Resolves the model id: a cached snapshot, otherwise the hub id, which downloads on the pod. Loads the base in bf16, or builds a tiny random Qwen3 for tests. Wraps PEFT LoRA in fp32 and provides the frozen-weight watch, LoRA state save/restore, and the in-place base cast for the fp32 readout. |
| `inner.py` | Inner rule: `DetachedWriter` (native write with a rolling, detached, re-rotated KV), recompute-mode batches, masked batches, L1 packing, and the clipped SGD step. |
| `readout.py` | Encodes the context once and scores probes in batches from its KV cache. Also computes the canary NLLs and the `Evaluator` that produces the exp07 `margins` dict. |
| `qwen_ladder.py` | Driver: `--level L0/L0mp/L0mw/L0mwfs/L1`. |
| `run_lr_sweep.py` | lr sweep: {1e-4, 3e-4, 1e-3, 3e-3} x {L0mp, L0mw}, s101, 8 passes. Writes a selection report. |
| `preflight.py` | Run on the pod once the model is available. Prints and asserts the config values below, and checks that the model's tokenizer gives the same token ids on every experiment text as the tokenizer the tests used. |
| `analyze_qwen.py` | Analysis. It reuses `analyze_ladder.py` and `analyze_stage01.py` from exp07, unmodified, through `EXP07_OUT`. |
| `ref_dump_jax.py` | Runs in the JAX env. It dumps the JAX pipeline's own texts and token lists, which test 7 compares against. |
| `test_*.py`, `_testutil.py` | CPU unit tests on a tiny random Qwen3. |

## Model: Qwen3-4B-Base (default), config to verify on the pod

The Base weights and config are **not cached on this machine**; they will be downloaded on the GPU pod. The values
below are ASSUMED from the public model card. **They have to be verified on the pod with `preflight.py`**, which
asserts them:

| key | assumed for Qwen/Qwen3-4B-Base (to verify on the pod) | Instruct-2507 (cached, verified) |
|---|---|---|
| num_hidden_layers | 36 | 36 |
| hidden_size / intermediate_size | 2560 / 9728 | 2560 / 9728 |
| num_attention_heads / num_key_value_heads / head_dim | 32 / 8 / 128 | 32 / 8 / 128 |
| max_position_embeddings | 32768 | 262144 |
| rope_theta / rope_scaling | 1,000,000 / none | 5,000,000 / none |
| tie_word_embeddings / vocab_size | true / 151936 | true / 151936 |
| `<|endoftext|>` id | 151643 | 151643 |

The design needs positions up to 8191 only (see the native write below), which is within either model's trained
range. Tokenizer: the Qwen3 family shares one tokenizer, but `preflight.py` checks that assumption on every
experiment text before any run. `--model-id instruct` selects Instruct-2507; `--model DIR` points at a local copy.
The tiny test model takes its architecture and tokenizer from the cached Instruct-2507; its weights are random.

```
CUDA_VISIBLE_DEVICES= python preflight.py                    # Base; downloads config + tokenizer if needed, no weights
CUDA_VISIBLE_DEVICES= python preflight.py --model-id instruct
```

## Design decisions (approved 2026-10-16; every one is a CLI flag)

| setting | default | flag |
|---|---|---|
| model | `Qwen/Qwen3-4B-Base`, bf16 base, sdpa attention | `--model-id base|instruct|<hub id>`, `--model DIR`, `--base-dtype`, `--attn` |
| fast weights | LoRA r=16, alpha=32 (scaling 2), dropout 0, fp32, on `gate_proj,up_proj,down_proj` of the last 9 of 36 layers (27-35). PEFT default init: A is Kaiming, B=0, so W0 equals the base model. 5.31M params. | `--lora-r --lora-alpha --lora-targets --lora-layers` |
| inner rule | Plain SGD with one step per chunk. The loss is the mean token NTP loss over the chunk's own tokens. Global-norm clip 1.0 (coef = 1 if norm < clip, else clip/norm, as in optax). Step = lr x scale x coef x g. The lr comes from `run_lr_sweep.py`. | `--lr 1e-3 --scale 1.0 --clip 1.0` |
| **native write** | **`detached`, the TTT-E2E structure (see below)** | `--native-ctx detached|recompute`, `--chunk 1024 --window-chunks 7` |
| stream | BOS + realised world text, **63 chunks of 1024 (64512 tokens), the same length as the JAX stream**. | `--stream-chunks 63` (or `cover`: 65 chunks covering the text), `--max-chunks` (smoke) |
| BOS | `<|endoftext|>` (151643) before the stream, the tails and the canary. Probes get no BOS, same as JAX. | `--bos endoftext|none` |
| L1 | Isolated `'\n\n'+sentence` rows. They are packed first-fit into steps of at most 1024 tokens; positions restart per row, which makes Qwen3's mask block-diagonal. The order is shuffled every epoch with `random.Random(--seed)`. 8 epochs. | `--variant plain|ctx|irr|fill --epochs --seed` |
| L0mp / L0mw / L0mwfs | 8 passes. Weights carry over between passes; every pass restarts at the stream start with an empty KV. L0mwfs gets new background each pass (draw `7000+p`), cut to the pass-1 length. | `--passes` |
| L0 | One pass, evaluated every 16 chunks and at the end (`state` key). | `--eval-every` |
| **readout** | R1 is the pg1342 tail, 7168 tokens including BOS. R1-alt is the SQuAD tail, 7168 tokens. All 276 items (552 rows) are scored in batches of 4 from the context's KV cache, with gold-minus-foil summed candidate log-prob. **fp32 by default**: the frozen base is cast in place to fp32 for the readout and back afterwards; the round trip is exact. Logits are fp32 in either mode. | `--readout-dtype fp32|bf16`, `--readout-batch 4`, `--tail-len` |
| canaries | `_gen_nll`: mean NLL over the first 8192 tokens of pg11. `_tail_nll`: the same over the R1 tail. | `--canary-len` |

### Native write, `--native-ctx detached` (default)

This follows TTT-E2E's structure (`inner.DetachedWriter`):
- **Step.** Chunk k's step is ONE forward of chunk k's 1024 tokens, with the KV of the previous 7 or fewer chunks as `past_key_values`. That KV is **detached**: no gradient flows into it. The loss is on chunk k's own tokens; the last token predicts the next chunk's first token, as in the JAX stream. Then backward and the clipped SGD step.
- **Cache update.** The KV of chunk k from that same forward, computed with the **pre-update** weights, is appended to the cache. Entries older than 7 chunks (7168 tokens) are dropped. Each chunk's KV is therefore computed with the weights in force at its own step, as with TTT-E2E's carried suffix KV.
- **Positions stay in an 8K window.** A 64K stream would otherwise exceed what Qwen was trained on; the 3B's sliding window also stays at 8K. Window positions are [0, 8192): the oldest cached chunk starts at 0 and chunk k sits at the end. When the window slides by 1024, the cached post-RoPE keys are re-rotated by -1024 positions (a RoPE angle shift). Values carry no position and are not touched. The shift is always applied in fp32 to the key **as originally computed**, by the total shift. So there is one rotation per use, rounding does not accumulate across slides, and the result is cast back to the cache dtype.
- **Why not HF's own cache handling.** HF's DynamicCache has no re-rotation, so the cache is rebuilt per step from the stored chunks. This means one concatenation and rotation of at most 7 chunks per layer (about 1 GB bf16 on the 4B), which costs little compared with the forward. An alternative was to cache pre-RoPE keys and rotate them at use. HF's Qwen3 attention caches post-RoPE keys, so that would need a patched attention module; not implemented.
- **Depth receptive field.** The carried KV of chunk k-7 was computed while chunk k-7 attended to the chunks before it. With more than one layer, chunk k therefore depends indirectly on chunks older than the window. This holds even with frozen weights. It is the same in TTT-E2E's carried SWA KV, so a recomputed 8K window is *not* numerically equal to the carried one (tests 9a and 9c measure this).

`--native-ctx recompute` keeps the earlier mode: the window of 7 previous chunks plus chunk k is recomputed with the current weights at every step (positions 0-8191, no cache), and gradient flows through the whole window. It costs about 8x more per step.

## Output

The output format is identical to exp07, so the exp07 analysis code reads it:
- `experiments/exp09_qwen_ttt/{LEVEL}_{TAG}_s{SEED}_{A,B}.json` contains `{'log':[...], 'sec':...}`. Evaluation entries have `epoch` (or `state` for L0) and `margins` = `{family: [...], '_gen_nll', '_tail_nll', '_alt': {family: [...]}}`, plus `lora_hash`, `cum_update_norm`, and per-step `ce/grad_norm/clip_coef` lists. L1 also logs one entry per step.
- `meta_{LEVEL}_{TAG}_s{SEED}.json` contains `items` and `fam_sizes` (order L3, XP, PS, FN, FP, FN0, FP0, L3r, L2), `model_id`, all args, the LoRA config, stream token and chunk counts, Qwen token lengths of the names, versions, and the frozen hash.
- `analyze_qwen.py --seed 101 L1:main L0mp:main L0mw:main` prints D (R1, R1-alt), acc/flip, familiarity gain, M_L3r, M_L2 and the canary rise. It needs `meta_L1_main_s{SEED}.json`; the lr sweep makes its own copy.

## lr sweep (`run_lr_sweep.py`)

The sweep runs lr in {1e-4, 3e-4, 1e-3, 3e-3} x {L0mp, L0mw}, world seed 101, worlds A and B, 8 passes, with every other setting at the defaults. Extra `qwen_ladder.py` flags go after `--`. Output goes to `experiments/exp09_qwen_ttt/lr_sweep/`, and the sweep resumes from existing files.

**Selection rule:** per level, the largest lr whose **canary rise at pass 1** is ≤ 0.10 nat. The canary rise is the maximum, over worlds A and B and over `_gen_nll` and `_tail_nll`, of the pass-1 value minus the pass-0 value; this is the `analyze_stage01` definition. Every lr is reported: pass-1 rise, maximum rise over all passes, D under R1 and R1-alt at pass 1 and at the last pass, and acc/flip. The output files are `lr_sweep_report_s101.md` and `lr_sweep_s101.json`. The report flags it when the largest grid lr is eligible (grid too low) or when none is (grid too high).

```
cd scripts/exp09_qwen_ttt; ../../.venv-exp02a/bin/python run_lr_sweep.py            # 16 world-runs
../../.venv-exp02a/bin/python run_lr_sweep.py --report-only                          # re-make the report
```

## Differences from TTT-E2E (Exp07) that remain

1. **No meta-learning.** TTT-E2E's W0 and inner-lr multipliers were trained for its inner rule; Qwen's were not. That is why the lr is swept.
2. **Fast weights.** TTT-E2E updates the full suffix-MLP weights. Here a rank-16 LoRA delta on the last 9 layers' MLPs is updated. Because B starts at 0, the first step moves only B.
3. **Attention.** Qwen uses full attention with every sequence capped at 8K positions: the native window, the readout context plus probe, and the canary. TTT-E2E uses SWA 8K. The detached cache reproduces TTT-E2E's carried-KV semantics, including the depth receptive field.
4. **Tokenizer.** The texts are identical (test 7) but are tokenized by Qwen. The stream uses the first 63x1024 Qwen tokens; the text is 66.0-66.2K Qwen tokens, so about 1.7K tokens of trailing background are unused, as in JAX. Names were length-matched under llama3. Under Qwen, seed 103 has two name pairs that differ in token length (Nevek/Ashane and Rasume/Yarish, 3 vs 2 tokens). L3 scores places, so binding is unaffected; L2, L3r and FN score names and are. Places and seeds 101/102 are fully matched.
5. **BOS.** `<|endoftext|>` takes llama's `<|begin_of_text|>` position.
6. **Readout implementation.** JAX packed probes into chunks; here each probe row runs against a copy of the context KV. Test 6 shows the two are the same conditioning. Both readouts are fp32.

## Tests (CPU only, tiny random Qwen3: 4 layers, hidden 128, real tokenizer)

```
cd scripts/exp09_qwen_ttt
for t in test_lora_step test_window_packing test_detached_write test_readout test_texts_vs_jax test_smoke_e2e test_lr_sweep; do
  CUDA_VISIBLE_DEVICES= ../../.venv-exp02a/bin/python $t.py || echo FAILED $t; done
```
1-3 `test_lora_step.py`: W0 identity; one step equals the hand-computed clipped SGD step; only LoRA changes.
4-5 `test_window_packing.py`: recompute-mode window equals the explicit concatenation; masked write isolation; L1 packing equals the rows alone.
9a-d `test_detached_write.py`:
- (a) A 1-layer model with frozen weights: the rolling detached KV plus re-rotation equals the explicit concatenation from position 0, after 2 and 4 slides. For 1 and 4 layers, the re-rotated window equals the same cache kept at absolute positions.
- (b) No graph in the cache; grads equal those from explicitly detached copies.
- (c) 1-layer, frozen weights: a chunk older than the window has no effect. In the 4-layer model the indirect effect through the deep carried KV is measured and reported.
- (d) The cached KV equals the KV computed with the weights in force at that chunk's step, not the current weights.

6 `test_readout.py`: cached equals full-concatenation scoring at a 7K context; margin sign; canary.
7 `test_texts_vs_jax.py`: texts and token lists are identical to the JAX pipeline for seeds 101-103.
8 `test_smoke_e2e.py`: driver smoke for every level and both native modes. Also checks the defaults and that exp07 analysis runs on the output.
10 `test_lr_sweep.py`: sweep end to end; the selection matches the rule recomputed from the raw logs, including an over-threshold lr.

## Running on the GPU (only when the A100 is free)

```
cd scripts/exp09_qwen_ttt; PY=../../.venv-exp02a/bin/python
CUDA_VISIBLE_DEVICES= $PY preflight.py                     # config + tokenizer checks (no weights)
$PY run_lr_sweep.py                                          # pick lr per level
$PY qwen_ladder.py --level L1   --world-seed 101 --epochs 8 --lr <L1 lr> --tag main
$PY qwen_ladder.py --level L0mp --world-seed 101 --passes 8 --lr <sel L0mp> --tag main
$PY qwen_ladder.py --level L0mw --world-seed 101 --passes 8 --lr <sel L0mw> --tag main
$PY analyze_qwen.py --seed 101 L1:main L0mp:main L0mw:main
```
Add `--deterministic` for bit-reproducible reruns.

### Rough cost on one A100 80GB (estimates, not measured; detached mode, fp32 readout)

Qwen3-4B costs about 7.5 GFLOP/token in matmuls, plus attention. Backward only goes through the last 9 layers.
- **Detached native step:** one forward of 1024 tokens against at most 7168 cached tokens, about 12 TFLOP, plus backward through 9 layers, about 6 TFLOP. That is about 0.15-0.25 s per step, or about 10-15 s per 63-chunk pass. Recompute mode is about 0.7-1 s per step.
- **Masked step:** about 0.1-0.2 s, or about 10 s per pass.
- **L1 epoch:** about 10 packed steps, a few seconds.
- **Evaluation, fp32 readout:** 2 x (7168-token context + 552 probe rows of 70 tokens or fewer) plus the canaries is about 0.9 PFLOP in fp32. Without TF32 that is about 1.5-2 min, and with 9 evaluations per run **readout dominates**. With `--readout-dtype bf16` it is about 15-30 s.
- **Setup:** model load plus world build is about 1-2 min per process, more on the first download.

| run (per world; double for A+B, per seed) | fp32 readout (default) | bf16 readout |
|---|---|---|
| L1, 8 epochs (9 evaluations) | ~0.25-0.35 h | ~0.05-0.1 h |
| L0mp, 8 passes, detached | ~0.3-0.4 h | ~0.07-0.12 h |
| L0mw, 8 passes | ~0.25-0.35 h | ~0.05-0.1 h |
| lr sweep (16 world-runs of L0mp/L0mw) | ~4.5-6 h | ~1-1.7 h |

All three levels for one seed with both worlds come to about 1.6-2.2 GPU-hours with the fp32 readout. A cheaper sweep is possible with `run_lr_sweep.py -- --readout-dtype bf16`: the rule uses the canaries only, which are 8K-token NLLs whose bf16 noise is far below 0.10. This is an option, not the default.

### Memory

- The bf16 weights take about 8 GB. The LoRA params are 21 MB.
- Detached native step: activations of the 9 LoRA layers for 1024 tokens (a few GB), plus the rebuilt 7-chunk cache (1.06 GB bf16) and 1024 x 151936 fp32 logits (0.6 GB). The peak is about 15 GB. Recompute mode peaks at about 25 GB.
- fp32 readout: the base is cast in place to fp32 (16 GB, about 24 GB transiently while casting). The context KV is fp32 at 2.1 GB and is copied per row, so batch 4 needs about 8.5 GB. The peak is about 30-35 GB.
- Do not start anything while the JAX job holds the GPU. Every test here is CPU-only and asserts `CUDA_VISIBLE_DEVICES=`.
