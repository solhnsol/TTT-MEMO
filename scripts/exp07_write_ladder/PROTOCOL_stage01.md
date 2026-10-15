# Exp07 stage 0 (readout re-checks) and stage 1 (damage-free repetition): pre-registered rules

Written 2026-10-15 ~10:40 KST (01:40 UTC), before any stage-0/1 run. Follows the reviewer verdict `reviews/2026-10-15-overnight-ladder-multipass-verdict.md` (CONDITIONAL GO) and Hans's go (2026-10-15 10:04 KST). World seed 101 first; 102/103 only if an effect is seen.

## Post-hoc amendment to PROTOCOL_multipass.md, stated as such
The overshoot rule (either canary rising > 0.10 nat/token over W0) was registered for the scaled runs only. After seeing the results it is also applied to L0mp: passes 3-8 exceed it (tail canary +0.52 ... +1.93), so the multi-pass R = 0.06 is NOT interpreted. Only pass 2 (canary +0.05) is clean; its ratio 0.43 is in the mixed band and is not used as a conclusion.

## Measurements added to every evaluation
- Two damage canaries (as before): pg11 (memorized, insensitive) and the scrubbed pg1342 neutral tail (= R1 readout context). Damage at a point = either canary, in either world, more than 0.10 nat/token above W0.
- A second readout context, R1-alt: BOS + 7167 tokens of SQuAD-dev Wikipedia paragraphs (expository, unlike the fiction stream and pg1342; paragraphs containing any world name removed), same packed fp32 scorer and items. Agreement R1 vs R1-alt is reported at W0 and at the L1 states.

## Stage 0 (re-checks)
- **0a. L1 damage trajectory.** L1 s101 (32 epochs, shuffle seed 0) is re-run with both canaries and R1-alt at every epoch; weights saved at epochs 1, 2, 4, 8, 32 (for the gradient measurement). **Verdict "L1 reached binding without damage"** iff at the first epoch whose R1 binding D has a CI lower bound > 0.5 nat, no canary exceeds the 0.10 damage threshold; otherwise "L1 binding came with damage" and the L1 vs native contrast is restated accordingly.
- **0b. L0mp read in an undamaged context.** L0mp s101 (8 passes) is re-streamed (deterministic, same weights) and every pass is scored with R1 and R1-alt. Rule on R1-alt at pass 8, Ralt = D_alt(L0mp, 8) / D_alt(L1, 8) (pair bootstrap):
  - Ralt < 0.30 (CI upper < 0.50): "native repetition does not write binding even when read in an undamaged context" (robust to the readout damage).
  - Ralt >= 0.70 (CI lower >= 0.50): binding was written but unreadable in the damaged R1 context; the previous reading reverses.
  - otherwise: mixed.
  Caveat recorded in advance: R1-alt's own context could also be damaged by the stream; this is checked by reporting R1-alt binding at W0 and L1 (agreement with R1) and the NLL of the SQuAD context at pass 8 if it is anomalous.

## Stage 1 (damage-free repetition), 8 passes each, pass 1 = the full native stream
- **(a) L0or, oracle exposure-only re-reads:** passes 2-8 re-read only the 480 exposure sentences of the world, in stream order, as one continuous ~8.9K-token stream (native TTT: window context, chunk mean, one step per chunk). Removes filler; keeps shadowing and chunk averaging. Oracle: a real document does not mark exposures.
- **(b) L0fs, filler swap:** passes 2-8 read the same text list with every background paragraph replaced by a fresh draw (overlap with the original filler 2-3%).
- **Rule:** passes with damage (> 0.10) are excluded. Let k* = the last undamaged pass. If k* < 2: "damage-limited" (no verdict). Otherwise R* = D(variant, k*) / D(L1, k*) on the R1 readout (R1-alt reported alongside), pair bootstrap: R* >= 0.70 with CI lower >= 0.50 -> the variant reproduces offline binding at matched exposures; R* <= 0.30 with CI upper <= 0.50 -> it does not; otherwise mixed.
- **Reading:** if (a) succeeds and native multi-pass does not, filler (budget taken by filler / stream overfitting) is the main factor, and selective re-writing of relation spans becomes a method candidate (closest prior work: GDWM, to be positioned first). If (a) also fails at matched exposures, the cause sits in the windowed write itself (shadowing, chunk averaging).

## Not claimable
Anything beyond seed 101 until replicated; "TTT conditions block binding" from damaged passes; H-seq (decided only by the stage-2 gradient measurement).

## Stage 2 (gradient components; written before the run)
`grad_components.py`: at W0 and the L1-ck weights after epochs 1, 2, 4, 8 (per world), the first-order effect of the L1 update direction on each pair's binding margin (binding component b = (a_A - a_B)/2, common c = (a_A + a_B)/2) and on name familiarity (r). Readout for the margin gradients: empty context (R0), a stated approximation. **H-seq is supported** iff b at W0 has a CI including 0 or |b|/|r| at W0 is below 1/10 of its value at epoch 2, AND b after epoch 1 or 2 has a CI lower bound > 0. **H-seq is rejected** iff b at W0 has a CI lower bound > 0 and is at least half its epoch-2 value. Otherwise inconclusive.

## Stage 3 (K sweep; written before the run)
L0k: one native pass with K = 2, 4, 8 inner steps per chunk (K = 1 equals native L0). Same canary rule: a run whose canary rises > 0.10 nat/token at any evaluated state is "overshoot" and its binding is not read as evidence. Reported: final D (R1 and R1-alt), familiarity gain, canary rise, cumulative update norm.

## Stage 1 prediction table (registered 2026-10-15 12:45 KST: L0or had finished running at 12:30 KST but its result files were not opened or analysed before this table was written; L0fs had not run)
Three explanations for "native repetition writes ~2-5% of offline binding at matched exposures" (stage 0b) and what each predicts. "Recovery" = R* (registered rule above) >= 0.70; "unchanged" = R* <= 0.30 like L0mp; "worse" = D at the last undamaged pass below native L0mp at the same pass.
| explanation | L0or (exposure-only re-reads) | L0fs (fresh filler each pass) |
|---|---|---|
| E1 stream overfitting / filler takes the write budget | recovery | recovery (no repeated filler to overfit) |
| E2 junk interference (Physics of LMs 3.3: irrelevant data in the same updates harms knowledge storage) | recovery (no junk) | worse or unchanged (new junk every pass) |
| E3 windowed write conditions (in-window shadowing, chunk averaging, one step per chunk) | unchanged (exposures even more shadowed, still chunk-averaged) | unchanged |
Readings: L0or recovery + L0fs recovery -> E1; L0or recovery + L0fs not -> E2; both unchanged -> E3. Other patterns are reported as such. Damage-limited conditions count as no verdict.

## Write-signal screening (registered 2026-10-15 13:05 KST, before the run; attached to stage 2)
At W0 and at the L1 epoch-1 weights, the same first-order components (binding b, common c, familiarity r) are computed for two training-free alternative write signals on the exposure sentences of rounds 2-16: ctxKL (soft-target cross-entropy to the same model's next-token distribution when the previous round's pair sentences are in context) and logitMSE (squared difference of centred logits, an approximation of TTCD's window-difference distillation). Normalised per unit gradient norm. **A signal passes the screen** iff at W0 its b/|g| has a CI lower bound > 0 and its b/r is at least 2x that of NTP. Only passing signals go to a streaming pilot (which will be pre-registered separately).

## Amendment before opening L0or (2026-10-15 13:00 KST; reviewer CONDITIONAL GO, Hans's go)
Record of the unopened L0or result files (sha256, mtime UTC). Reported in the paper as "registered after completion, opened after registration".
| file | sha256 | mtime (UTC) |
|---|---|---|
| L0or_main_s101_A.saved.json | a6abe981af560cf146729dd6e19636919e1ef4a3c3f68e660d28ccb0e733471d | 2026-10-15 03:01:50 |
| L0or_main_s101_B.saved.json | 2e612c38b32289ece30fe66d05e99518409e612a000ec0956cda982f8b15581f | 2026-10-15 03:04:26 |
| L0or_main_s101_A.json | cb6eb5024f253318be95029136de1ba04d588195f83cb390cfa42dd11e7e1cd0 | 2026-10-15 03:25:35 |
| L0or_main_s101_B.json | 6b810fdd5e030d6ae46c8c656b35eef3d024b6f47d9c3bd993e0cceb44943519 | 2026-10-15 03:30:28 |

Numeric classes for the prediction table (they replace the 0.70/0.30 R* bands for this table only; the R* rule above is still reported): with R* = D(variant, k*)/D(L1, k*) on R1,
- **recovery**: R* >= 0.30;
- **unchanged**: R* <= 0.10;
- **partial**: in between;
- **worse**: D(variant, k*) below D(L0mp, k*) by more than the L0mp CI half-width.
The analysis script (`analyze_stage01.py`, this commit) applies these classes; it is committed before the L0or files are opened.

## Screening definitions (tightened 2026-10-15 13:10 KST, before the run; replaces the short rule above where they differ)
All quantities in fp32, inner (suffix "prime" MLP) parameters only, Euclidean inner product over all inner leaves.
- Update signal of world w at weights W_w: g_w = (1/N_c) * sum over packed chunks c of grad_W L_c(W_w), where L_c is the chunk-mean loss of the signal (NTP: token cross-entropy on the isolated exposure sentences, all 480; ctxKL: soft-target cross-entropy to the teacher distribution on rounds 2-16; logitMSE: mean squared difference of centred logits teacher - student on rounds 2-16). |g| = sqrt(<g_A,g_A>)/2 + sqrt(<g_B,g_B>)/2.
- Teacher (fixed): the W0 model (fp32) reading "previous round's two pair sentences + current sentence"; the same teacher for every state and world.
- Margin gradients at W_w with an empty-context readout: u_w(p) = grad m_p^L3, f_w(p) = grad m_p^FN, n_w(p) = grad m_p^XP (cross-pair null, pair p's names with pair p+1's places).
- First-order effects of a step along -g: a_w(p) = -<u_w(p), g_w>, likewise with f (familiarity) and n (null).
- binding b(p) = (a_A(p) - a_B(p))/2; null b0(p) = (a_A^XP(p) - a_B^XP(p))/2; common c(p) = (a_A(p) + a_B(p))/2; familiarity r(p) = (a_A^FN(p) + a_B^FN(p))/2.
- **Screen (necessary, not sufficient):** at W0, (b - b0)/|g| has a pair-bootstrap CI lower bound > 0, and mean(b)/mean(r) >= 2 x the NTP value. The actual criterion for any write signal is the binding D measured after a real one-pass TTT stream with that signal (to be pre-registered before its pilot).
- Analysis-script record (git commit was blocked by the session's permission settings, so the script is fixed by hash instead; committed later when allowed): `analyze_stage01.py` sha256 1930eef5d77fd9248c035aa99a7769f243626becfc8a17e85762c1b2d468f86d, `analyze_ladder.py` sha256 0ce26cbbcb404b5926635daa7b559a62064f3564e09317f94194e3d1e77202a4, recorded 2026-10-15 03:53 UTC, before the L0or files are opened (they are first read by this script when the stage-1 queue finishes).

## Decision rules after the 3B queue (registered 2026-10-15 13:35 KST, before the seed 102/103 replication and before the screening result; reviewer priority GO, Hans's go)
1. **Core observation.** Ralt (stage-0b rule, L0mp read with R1-alt, pass 8 vs L1 epoch 8) on seeds 102 AND 103 both < 0.30 -> the core observation (native repetition writes binding at a small fraction of offline at matched exposures) is confirmed on 3 seeds. Either seed >= 0.30 -> conclusions are rewritten and all further steps are held until re-planned with the reviewer.
2. **L0fs replication.** L0fs runs on seeds 102 and 103 with the same stage-1 rule and classes; reported with s101.
3. **Write signal.** If a signal passes the screen, its real one-pass TTT D is measured (design registered separately before that run). Then: a passing signal with one-pass R* >= 0.30 (vs L1 at matched exposures) -> method development on 3B is the main line; no signal passes the screen -> the 125M outer-loop experiment becomes the main line; a passing signal with one-pass R* in 0.10-0.30 -> both lines in parallel until the early-November gate.
4. The shadowing ordering (L0or < L0fs) stays exploratory; it is not used for any decision above.
5. K sweep reduced to K = 1 and 4 and moved to the end of the queue.

## L1 variants (registered 2026-10-15 14:00 KST, before any run; Hans's go)
Two offline rungs between L1 and native, seed 101, 8 epochs each, same step rule, readouts and canaries as L1 ck:
- **L1-ctx:** each exposure sentence is trained with the previous round's two pair sentences in front of it as context (loss on the current sentence only). Adds in-context shadowing to the offline write; no filler, shuffled.
- **L1-fill:** the plain L1 rows plus all background paragraphs of the world's own stream as further isolated rows (loss on all tokens), i.e. the native filler share inside the offline write; no context, shuffled.
- **Rule:** R = D(variant, epoch 8) / D(plain L1 ck, epoch 8) on R1, pair bootstrap. R >= 0.70 (CI lower >= 0.50): no effect; R <= 0.30 (CI upper <= 0.50): strong reduction; otherwise partial. Prediction: if in-context shadowing is the main factor, L1-ctx shows a strong reduction and L1-fill no effect.
- Check amendment (2026-10-15 14:10 KST, after the first check run, before any screening output): C2a failed (rel 0.76 / 0.42) because with eps = 1e-3 along a random unit direction the loss moves by ~2e-7, below fp32 resolution at loss ~4 (the finite differences equal one or two ulps). C2a now uses (i) the normalised gradient direction with eps = 1e-3 (pass <= 2%) and (ii) a random unit direction with eps = 1.0 (pass <= 5%). C1 (0.0), C2b (ratio 8e-8 / 0), C2c (450 rows, context = the two previous pair sentences) and C3 (binding +7.06 [+5.34, +9.17], null/b 0.04) passed in the first run.

## Amended decision rules (registered 2026-10-15 15:10 KST, before the seed 102/103 results, the N-sweep and any Qwen run; reviewer verdict + Hans's go)
Reason for the change: the earlier rule "no passing signal -> 125M outer loop is the main line" assumed W0 is the cause. That assumption can be tested more cheaply with a TTT family that has no outer loop (NTP-LoRA-TTT on Qwen), which is also the required second family. This is a change of order, not a re-reading of results.
- Order: finish the 3B queue -> **N-sweep** (below) -> port the pipeline to Qwen NTP-LoRA-TTT (L0 and L1 with the same worlds, readouts and nulls) -> decision on 2026-10-18.
- Decision on 2026-10-18:
  1. Seed replication fails (Ralt >= 0.30 on seed 102 or 103) -> rewrite the core observation first.
  2. Qwen one-pass R* >= 0.30 (vs Qwen L1 at matched exposures) -> W0 (meta-learned initialization) is a cause candidate; the 125M outer-loop experiment becomes the main line after 2026-10-26.
  3. Qwen also fails (R* < 0.30) and the N-sweep slope is near -1 (95% CI includes -1 and excludes 0) -> the cause is the one-step structure shared by both families; methods are sought in the inner update; 125M becomes optional.
  4. Other combinations: reported and re-planned with the reviewer.
- 125M read-objective outer-loop code may be written 2026-10-19..25 without GPU use.

## N-sweep (registered 2026-10-15 15:10 KST, before the run)
Gradient-only, at W0, seed 101, with the same checked functions (loss_grad, margin_grad; checks passed at sha256 5c2cf579...): the NTP update direction g_S is computed on the exposure sentences of a subset S of N pairs only, and b(p), r(p) are measured for p in S. N in {1, 2, 4, 8, 15}; 6 random subsets for N = 1, 2, 4, 8 (seeded), the full set for N = 15. Quantity: q(N) = mean over subsets of mean_p b(p) / mean_p r(p). Fit: log q(N) = a + s log N (least squares over the subset-level values; pair-bootstrap within subsets for the CI of s). **Supports the statistical-dominance explanation** iff the 95% CI of s includes -1 and excludes 0; **no N dependence** iff the CI of s includes 0. Otherwise reported as is.

## Greedy EM (registered 2026-10-15 17:30 KST, before any EM output exists; reviewer EM verdict, Hans's go)
- Method: greedy exact match of the place name continuing the L3 probe, computed as "every gold place token is the full-vocabulary arg-max at its position" (identical to greedy decoding while all earlier tokens match). Same packed fp32 readouts.
- **Primary readout: R1-alt. Primary number: flip-EM** = fraction of the 30 L3 items whose greedy continuation is the world-correct place in world A AND whose probe gives the partner place in world B.
- Reported with: per-world EM; the W0 value; candidate-set baselines (uniform over the 30 world places = 1/30 per world; uniform over the probed pair's two places = 1/2, i.e. the 2-way level); and, for wrong greedy first tokens, the share that is the first token of (i) the partner's place, (ii) another pair's place, (iii) anything else.
- Runs: L1 s101 ck (saved epochs 1, 2, 4, 8, 32), native L0 s101 (re-streamed with the verified L0x scale-1 path: states 16, 32, 48, 63) and L0fs s101 (re-streamed, passes 1-8); weights are re-saved because the earlier scored states were deleted after scoring. s102 L1 EM is deferred until after 2026-10-18. EM is part of the Qwen readout from the start.
- Not a decision input for 2026-10-18.

### W0 place bias (descriptive, computed before EM from existing logs)
At W0 both worlds have identical weights, so the per-item L3 margin there is a pure name-independent place preference: mean |m| = 3.75 nat under R1 vs 2.46 under R1-alt (s101); 4.80 vs 3.29 (s102); W0 2-way accuracy 0.50 (R1) and 0.53 / 0.50 (R1-alt). The common-mode magnitude |(m_A + m_B)/2| is still 3.67 (R1) vs 2.19 (R1-alt) at L1 epoch 8 (s101). The lower R1 accuracy of L1 is therefore explained by a place bias that exists before any update and is larger in the fiction-tail context.
