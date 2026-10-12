# Exp 02A4: native TTT-E2E 3B source-free binding

Checkpoint: `3b_ttt_e2e_finetune_books_128k_3x_cc`, official extension
configuration, 24 prefix + 8 suffix layers, 8192-token SWA, 1024-token chunks.
Native inner SGD lr=1, gradient clipping=1, one update for one observed chunk.
Only the eight suffix `feed_forward_prime` weights update. No LoRA or K=32.
The existing Exp01 3B restoration procedure is reused with full state retained.
The existing Exp02A1 frozen exact smoke pairs and metrics are reused unchanged.

Write contains `Fact:\n` and the original mapping's two sentences, each once.
Real next-token targets after BOS are labeled; future EOS padding is masked.
This differs from Qwen's oracle value-only NTP mask and 32 repeated updates.
Update counts, real/loss tokens, native loss, parameter delta and time are saved.

Source eviction is explicit: all source tokens, prefix hidden states, optimizer
and write KV state are discarded before readout. Fresh zero KV/chunk counter
are used for every candidate, with only adapted prime FFN retained. This makes
source entirely unavailable to local attention; it is NOT a measured 8192+
filler stream or a retention/interference experiment. Readout positions restart
in a fresh context. No source-conditioned cached states are carried.

Base-LM readout uses the existing Exp01 T3 Context/Question/Answer template.
Source-free context is empty; source-present uses the two original facts with
no update. Both query styles from Exp02A1 are evaluated. A single leading space
precedes the candidate; full text is tokenized once. Candidate overlaps are
identified by offsets, token i scored at logits i-1, EOS excluded. Mean-token
logp is primary; sum logp and individual token logp are saved and independently
checked. No readout update or free generation occurs.

A/B always start from the same immutable initial model and zero suffix state.
All frozen parameter differences are checked, fast weights must change,
original initialization hashes must remain unchanged, and returned write state
never enters readout. Native JAX/Equinox primitives run outside `e2e/`.

The 4-pair pilot matches the exact pairs of the prior Qwen Exp02A1 smoke,
including alternating entity order. Cross-model magnitudes are descriptive:
model training, tokenizer, prompt format, inner parameterization, objective
mask, number of updates and compute differ. Source-present readout calibration
must be considered before attributing a negative result to memory storage.

Run from repository root with a NEW output directory:

    LD_LIBRARY_PATH="$PWD/.venv/lib/python3.12/site-packages/nvidia/cuda_nvrtc/lib" \
      .venv/bin/python -B scripts/exp02a4_ttt_e2e_binding/run_pilot.py \
      --n 4 --out experiments/exp02a4_ttt_e2e_binding/pilot_<run_id>

The runner sets NVRTC library lookup and disables JAX preallocation. Existing
output directories are rejected. Dataset/config, complete per-sample score
matrices and candidate token scores, write diagnostics and aggregate metrics
are stored in the output directory; redirect stdout/stderr to a separate log.

NVRTC lookup must be set before launching Python: changing LD_LIBRARY_PATH
inside Python may not affect the dynamic loader. Failed smoke logs are kept.
