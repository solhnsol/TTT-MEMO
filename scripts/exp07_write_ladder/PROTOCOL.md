# Exp07 write ladder L0 vs L1: pre-registration draft

Written 2026-10-14, before any Exp07 data. Status: DRAFT, waits for (1) Exp05-A run3 to finish, (2) `verify_packing.py` PASS, (3) the user's go for GPU.

## Question
Is the missing binding a property of TTT-E2E's online write (context, chunk aggregation, streaming order), or does next-token learning at this write location (the suffix "prime" MLPs) fail to store binding even under favourable offline conditions?

## Levels (same tokens, same W0, same inner parameters, same step rule)
| level | what is written | context at write time | update | order |
|---|---|---|---|---|
| **L0** native | the seed-101 clean-v2 stream (63 chunks of 1024, exposures + filler) | 8K SWA window, carried KV | one clipped SGD step per 1024-token chunk (inner rule: config lr x ilr multiplier, clip 1.0) | stream order, one pass |
| **L1** offline isolated | the 480 exposure sentences of the same world, exactly as realized in the stream (16 per triple) | none: each sentence alone (packed with a block-diagonal mask, positions restart at 0, empty suffix KV) | same rule, one step per 1024-token chunk of packed sentences; loss on tokens whose target is in the same sentence | reshuffled every epoch, up to 32 epochs |

L1 removes three things at once relative to L0: in-window shadowing, filler tokens in the update average, and single-pass streaming. If L1 writes binding, the follow-up varies ONE factor at a time between L0 and L1 (masked write for context; sentence-only vs chunk-with-filler for aggregation; shuffled vs stream order). If L1 does not, the follow-up is L2 (same offline write, all-layer MLPs) and L3 (NTP-LoRA-TTT on a plain LM, no meta-learned W0).

## Readout and items
- R1 neutral-tail readout (Exp05 run3) through the packed scorer; used only after `verify_packing.py` PASS.
- Families: L3 binding (15 pairs), XP cross-pair null, PS pseudo null (16 pairs), FN/FP familiarity (exposed vs never-exposed foil), FN0/FP0 familiarity nulls, L3r and L2 (secondary directions).
- World A and World B (names swapped) are trained separately at every level; binding is always the A-B contrast at the same point (epoch / state).

## Metrics
- Binding D = mean over pairs of (M_L3 - M_XP), M = (m_A - m_B)/2, pair bootstrap CI.
- Familiarity gain F = mean over FN items and both worlds of the margin change from the untrained state.
- Accuracy = mean of [m_A > 0] and [m_B < 0] over L3 items; counterfactual flip = fraction of items with m_A > 0 and m_B < 0.
- Budget is compared by the binding-vs-familiarity curve, not by step counts; cumulative update norm is reported.

## Decisions
- **Primary**: D of L1 interpolated at L0's final familiarity gain, minus L0's final D (pair bootstrap).
- **L1 success** (binding is writable here; TTT-specific factor): the primary difference CI > 0, or L1 reaches accuracy >= 0.80 and flip >= 0.50 at any epoch <= 32.
- **L1 failure** (positive control fails; NTP at this location does not store binding): the best-epoch D CI includes 0 or accuracy <= 0.60, and the primary difference CI includes 0. Next: L2/L3.
- Otherwise inconclusive: reported, no branching.

## Cost (estimate, to be measured)
L1: about 12 steps per epoch, 32 epochs, one packed R1 evaluation per epoch, two worlds. L0: one native stream per world with 10 packed evaluations. Both are expected to take well under an hour each with the packed scorer; to be confirmed on the first epoch.
