# Exp07 multi-pass / intensity controls: pre-registered reading rules

Written 2026-10-14 ~15:20 UTC, before any L0mp / L0x result exists (the runs start after queues 1-2). Copy: `experiments/exp07_write_ladder/PROTOCOL_multipass.md`. Follows the reviewer verdict `reviews/2026-10-14-L1-multipass-verdict.md` (CONDITIONAL GO).

## Status of L1 under the ORIGINAL rules (PROTOCOL.md), stated as is
L1 s101 (32 epochs, two shuffles) is in the **gray zone**: accuracy 0.73 / flip 0.47 is below the success rule (0.80 / 0.50) and above the failure rule (<= 0.60). The matched-familiarity primary has little power because familiarity saturates after epoch 1. Correct description: "no verdict under the pre-registered rules; in an exploratory exposure-count comparison, binding appears from epoch 2 and reaches D +2.35 [+1.60, +3.28] at epoch 32". It is not reported as "L1 success".

## Runs (world seed 101, A and B, fp32 packed R1 readout, bf16 native updates)
| run | what | evaluation points |
|---|---|---|
| L0mp | native TTT re-reading the stream 8 times; weights carry over, suffix KV restarts each pass | end of each pass (k = 1..8) |
| L0x1 | re-implemented native step, step size x1 (sanity) | every 16 chunks and the end |
| L0x2, L0x4, L0x8 | one pass, every native step size x2 / x4 / x8 (after clipping) | every 16 chunks and the end |
Every evaluation also records the held-out general-text NLL (8 x 1024 tokens of `pg11.txt`, a book used in no stream or tail; bf16, fresh KV) and the cumulative update norm (sum of ||Delta|| per step).

## Rules
1. **Sanity gate (L0x1).** The final-state item margins of L0x1 must match the native L0 s101 final state (same fp32 readout) within max |difference| <= 0.01 nat. Otherwise L0mp / L0x results are not interpreted.
2. **Primary axis = exposures.** L0mp pass k and L1 epoch k are placed on the same axis (16k exposures per triple). D, accuracy and flip are reported per k; step count and cumulative update norm are the secondary axes.
3. **Primary ratio** R = D(L0mp, pass 8) / D(L1, epoch 8), pair-bootstrap CI (pairs resampled jointly for both runs).
   - R >= 0.70 and CI lower bound >= 0.50: the missing binding is explained by **exposure count**; TTT-specific conditions cost little (general NTP sample efficiency).
   - R <= 0.30 and CI upper bound <= 0.50: **TTT conditions block binding** at matched exposures; only then is the one-factor TTT decomposition (shadowing, chunk averaging, filler interference) run.
   - otherwise: mixed; factors are decomposed one at a time.
4. **Overshoot rule (L0x2/4/8).** If the general-text NLL at any evaluated state of a scaled run is more than **0.10 nat/token** above its W0 value, a lack of binding in that run is read as overshoot, not as evidence for sequential learning (H-seq). The scale-binding curve (x1, x2, x4, x8) is reported together with familiarity gain and general NLL.
5. **H-rep vs H-mag.** If L0x8 (no overshoot) reaches the binding of L0mp at a similar cumulative update norm, magnitude (H-mag) is sufficient; if only L0mp writes binding, repetition at different times is needed (H-rep). Neither result is read as H-seq; H-seq is decided only by the gradient measurement below.
6. **Not claimable from these runs:** "TTT writes binding when repeated" before rule 1 and 3 are met; "binding is learned sequentially"; anything beyond seed 101.

## Reported alongside (no decision)
- Layer-wise update norms of L1 epoch 1 vs L0 exposure chunks (dilution check, M2a), from the saved logs where available.
- The earlier counter-example: TTT-E2E K=1->4 inner steps per chunk did not improve mapping recall (Exp02a4, bf16 readout). Any H-rep claim must state how it is compatible with this (e.g. same-context repetition vs re-exposure at a different time).

## Next, independent of these results (reviewer priority 1)
Exp05-B-style gradient decomposition at W0 vs at the L1 epoch-1 weights: per-world a_t, binding component (a^A - a^B)/2 vs familiarity component, on the exposure sentences. This is the only test of H-seq. Literature check before any H-seq framing: Zucchet et al. 2025 (arXiv 2503.21676), Chang et al. 2024.

## Amendment (2026-10-14 17:25 UTC = 02:25 KST, before any L0mp / L0x data)
- The general-text canary `pg11.txt` turned out to be memorized by the model (NLL 0.191 nat/token at W0; native L0 moves it to 0.198 over 63 chunks). A memorized text can lose its memorization under any update and may flag "overshoot" too easily. A second canary is added: the name-scrubbed neutral tail (pg1342, 7 x 1024 tokens).
- Rule 4 now reads: overshoot iff the NLL of EITHER canary, in either world, rises more than 0.10 nat/token above its W0 value at any evaluated state. Both canaries are reported.
- Equivalence of the split save/eval pipeline (checked on L0 s101): loaded-weight hashes = saved hashes (enforced per state); all 64 saved-state hashes = run2 hashes for A and B; L0 state-0 readout = L1 epoch-0 readout, max |diff| 0.0 for every item.
