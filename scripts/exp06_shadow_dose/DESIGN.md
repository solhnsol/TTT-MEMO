# Exp06 design draft: spaced/massed factorial world, continuous shadow dose, test-time masked write

Status: DRAFT for review (2026-10-14). Not pre-registered yet; no GPU run. Follows the reviewer's plan of 2026-10-14 06:20Z:
finish Exp05-A run3 unchanged, then (1) factorial world + continuous shadow dose, (2) masked write + random-mask control,
(3) only if (2) is null: NTP-LoRA-TTT control (no outer loop) or 125M window ablation with the suppression measurement.

## 1. World (`gen_world_sm.py`, CPU, verified for seed 201)
- 32 swap-pairs (64 triples), every triple exposed R = 8 times; a pair's two sentences are adjacent (as in Exp04).
- Per pair: SPACED (consecutive exposures > 8192 tokens apart, outside the SWA window) or MASSED (all 8 exposures in one ~2K burst, inside the window).
- Two variants with the roles swapped (v0: pairs 0-15 spaced; v1: pairs 0-15 massed); same entities, frames and background. x World A/B name swap (for the binding contrast) = 4 streams per seed.
- Audit (seed 201, all 4 streams): min spaced gap 9.8K tokens, max massed gap 369 tokens, mean exposure position spaced 41.9K vs massed 42.1K (0.4% apart), last exposure ends 9.4K tokens before the probe chunk, stream 89 chunks + probe chunk.
- Only one probe point (end of stream) is needed for the primary outcome, so a stream costs about the update time (89 x ~5 s) plus one readout; far cheaper than Exp05-A.

## 2. Outcomes
- Primary: weight-stored familiarity of each entity's name and place under the R1 readout (neutral tail, Exp05 run3), against length-matched never-exposed control names/places. Contrast: spaced minus massed, within entity across the two variants (each entity is its own control), pair-cluster bootstrap.
- Secondary: binding M = (m_A - m_B)/2 with the cross-pair null (run3 design), relation types L3 / L3r / L2.
- Continuous shadow dose per exposure: d = NLL(target tokens | prior same-entity span masked) - NLL(target | native), computed at the time of the write with the pre-update weights = loss explained by attention. Target tokens: the name and place tokens of the exposure sentence.
- Suppression law (descriptive, then modelled): per-exposure weight write (DeltaF from per-exposure readouts on a subset of entities, or the first-order a_t = -<u, g_t>) as a function of d, by token type.

## 3. Masked write (M1 vs M1' / M3)
- On massed entities only: for every exposure after the first, compute the update with the prior same-entity span(s) hidden from all later queries in all prefix and suffix layers (write computation only; reading/probing unchanged).
- Control: random mask of the same length at a matched distance (unrelated background span).
- Prediction M1: massed familiarity (and binding, if measurable) recovers toward the spaced level; random mask does not. Null result is ambiguous (masking is off-distribution) -> step 3 of the plan.

## 4. Mask implementation feasibility (code read, no run)
- All attention goes through `jax.nn.dot_product_attention` (`e2e/ttt/model/attention.py`):
  - suffix (`SlidingWindowAttention.__call__`): explicit mask from `sw_causal_mask(chunk_id)` with ABSOLUTE key indices `ki` -> AND with `~blocked[ki]` by patching `sw_causal_mask` outside e2e.
  - prefix (`full_sw_attention`): `is_causal + local_window_size`, cuDNN. For the masked computation, replace with an explicit mask (causal & window & ~blocked) and the XLA implementation, on an 8K-16K slice only (the chunk being written plus enough look-back that intermediate tokens are recomputed under the mask; information otherwise leaks through intermediate hidden states, as Exp04 Stage 1H showed).
- No recompilation per exposure: the blocked-position vector is a traced argument of one jitted function (set as a module-level tracer during tracing), fixed shape (slice length). One compile for the masked write, one for the dose forward.
- Cost risk: XLA attention over an 8K slice materialises heads x 8K x 8K scores per prefix layer; must be measured (smoke) before committing to 16K look-back.

## 5. Open choices (for review)
1. R = 8 exposures (vs 4 or 16). 8 keeps massed bursts ~2K and spaced schedules within ~82K tokens.
2. Number of seeds: start with 1 seed (4 streams), then 2 more if the primary contrast is in the expected range.
3. Whether the per-exposure readout (needed for the suppression law) uses a subset of entities to keep the cost low.
