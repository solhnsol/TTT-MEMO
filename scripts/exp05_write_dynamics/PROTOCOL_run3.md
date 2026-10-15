# Exp05-A run3 protocol: readout comparison and null calibration (pre-registered)

Written 2026-10-14, before any run3 data exists. Copy of `experiments/exp05_write_dynamics/run3_readouts/protocol.md`.

## Why
Exp05-A run2 (deterministic) gave verdict N, but three problems block any reading of it:
1. **Readouts disagree.** At chunk 62 the run2 short probe gives L3 binding M = +0.028 = (m_A - m_B)/2, the Exp04 readout gives m_A - m_B = +0.169 (= +0.085 in M units), on identical weights.
   Code check: the Exp04 "weights-only" readout (`core._score_probe_full` with an empty state) runs the frozen prefix over stream tail + probe (8192 tokens), but the suffix only over the probe chunk with an empty KV and RoPE from 0. Prefix and suffix see different contexts, a state the model never sees in native streaming. The run2 short probe has no context at all.
   At intermediate chunks the Exp04 tail would also contain exposure sentences, so it cannot be used as a per-chunk readout.
2. **The null is invalid.** The 4 C4 pseudo-pairs end at M_63 = -0.027, -0.094, +0.242, -0.022 (sd 0.149), the real pairs have sd 0.044. Four items cannot calibrate 15 pairs.
3. **Numerical sensitivity.** M_63 = +0.017 (run1, default XLA) vs +0.028 (run2, deterministic ops); a ~3e-3 relative change of the chunk-0 update moved item margins by 0.09-0.23 nat.

## What is run (seed 101, worlds A and B, `XLA_FLAGS=--xla_gpu_deterministic_ops=true`)
Native stream identical to run2 (no update change). On every state j = 0..63 (j = chunks consumed):

| readout | definition | where |
|---|---|---|
| **R1 neutral (primary)** | fixed held-out tail: BOS + 7167 tokens of `pg1342.txt` (not one of the stream background books; same name scrubbing as `gen_world.background`), streamed through prefix **and** suffix with the current weights and no update, then the probe chunk. Identical tail in every state and world, so only the weights differ. RoPE positions continue naturally (probe at 7168+). | all states |
| R0 short | run2 probe: one 1024-token chunk, no context, empty suffix KV, RoPE from 0 | all states |
| R2 Exp04 | Exp04 protocol, empty KV and carried KV | chunks 14, 33, 62 only (states 15, 34, 63) |

Item families (all `PR_HOME` template, gold/foil span = whole place name, log-prob sum; pair margin = mean of the pair's two items):
- **L3** (real): 15 pairs, 30 items, as in Exp04/Exp05.
- **XP** (cross-pair null): for pair p, the two names of pair p probed with the two places of pair p+1 (mod 15). Same names, same exposure history, same template, exposed places, but no binding between them. 15 pairs.
- **PS** (pseudo-pair null): 16 pairs of never-exposed names and never-exposed places (same token lengths as the world's; checked absent from both streams and from the tail; seeded `random.Random(seed*1000+7)`), both gold/foil orders. 16 pairs.

**Perturbation replicate:** the same stream with a fixed seeded Gaussian direction of relative size 3e-3 x ||Delta_0|| added to the chunk-0 update (matches the observed default-vs-deterministic difference); R1 and R2 at states 15/34/63 only.

## Gates (any failure: stop, no interpretation)
- G1: the 64 weight hashes of the main run equal run2 (`traj_seed101_full_{A,B}.json`) for both worlds.
- G2: R2 L3 pair margins equal run2 V3 probes bitwise (max diff <= 1e-6; R2 uses the unchanged run2 code path).
- R0 vs run2 `margins_state` is reported, not gated (amendment below).

## Execution (no repeated compilation)
Prefix blocks are frozen, so prefix outputs of all probe rows (short and neutral-tail versions) and of the tail are computed once at start. Per state only the suffix runs: 7 tail chunks (`chunk_logp`, one compiled shape) and 12 batched calls of 16 rows (`lp_batch`, one compiled shape; the last group is padded by repeating a row). The native update is one compiled function as in run2. A `--dry` smoke compares batched and unbatched R0 on the base weights and times R1 twice to confirm no recompilation.

## Metrics and decisions
- M_{p,j} = (m_A - m_B)/2 per pair. **Primary quantity: D_62 = mean_p (M^L3_{p,63} - M^XP_{p,63}) under R1**, 95% pair bootstrap CI (pairs matched by p).
- **Binding detected under R1** iff the D_62 CI lower bound > 0 **and** the perturbation replicate keeps the sign of pooled D_62 with |change| <= 50%.
- **Pair-level M is interpretable** (usable for per-pair or per-chunk analysis in Exp05-B) iff the pair sign agreement of L3 M at k=62 between main and perturbed run is >= 0.80. Otherwise only pooled quantities are used.
- **Readout comparison (descriptive, no decision):** at k = 14/33/62 the mean L3 M, XP, PS and D under R0, R1, R2-empty, R2-carry; pair-level correlation and ratio of means between readouts. The working hypothesis "the Exp04 readout inflates the signal" is supported if D(R2-empty)/D(R1) >= 2 with R1's D CI excluding 0; it is not tested if R1's D includes 0.
- **Trajectory under R1 (descriptive):** own-exposure DeltaM for L3 and for XP of the same pair at the same event, their difference (binding-specific write per exposure), and per-round means. This is the input for the per-round binding efficiency question; no decision is taken here.

## What follows
- Gates pass and binding detected under R1: Exp05-B pilot with R1 as the readout for the margin gradients (per-world a_t, binding/common split, exact path decomposition).
- Gates pass, binding not detected under R1: Exp05-B pilot still runs (the question there is whether NTP gradients point along the binding direction at all), with R1 as readout and only pooled metrics; the "Lazy" framing stays on hold.
- Seeds 102/103 remain on hold until after the Exp05-B pilot.

## Cost estimate (to be checked on the first chunks)
Per state about 92 items x 2 candidates under R0 and R1 plus 7 tail chunks: roughly 2-2.5 min per state, about 2.5 h per world, about 5 h for A+B, plus about 30 min for the perturbation replicate. One A100 80GB.

## Amendment (2026-10-14, after the GPU smoke, before any run3 data)
The `--dry` smoke and `diag_batch.py` (base weights, 24 candidate rows) showed:
- vmap batching is numerically neutral: batched vs unbatched with the same prefix outputs differ by <= 1e-6 nat, and group composition does not matter (0 difference).
- The run2 short probe (`Traj.short_logp`) differs by up to 0.068 nat per candidate from the same computation done as separate prefix and suffix calls, although the materialized prefix outputs are bitwise identical. The cause is XLA fusing prefix and suffix inside one jit. The native stream (run2 and run3 alike) uses the separate path, so run3's R0/R1 follow the stream's numerics and run2's short probe is the odd one out.
- Consequence: R0-vs-run2 equality (old G2 first half) cannot hold and is dropped as a gate; the difference is reported as compilation-path numerical noise. Gates are G1 (hashes) and R2 bitwise. Everything else is unchanged.
- This per-candidate compilation noise (~0.07 nat) is the same order as the effects in question; it is one more reason the pre-registered decisions use the A-B contrast, the cross-pair null and the perturbation replicate rather than raw margins.
- Measured cost: R0 ~26 s and R1 ~26 s per state (lp_batch compiled once), so about 75 min per world with the R2 checkpoints.
