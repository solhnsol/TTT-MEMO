# Exp08 125M port: pre-registration (2026-10-15 13:40 KST, before any 125M ladder run)
Purpose: decide whether the 125M TTT-E2E checkpoint (pretrain DCLM 8K) is usable for the planned outer-loop (meta-learning) experiment, i.e. whether binding is measurable at 125M.
- Pipeline: the 3B Exp07 code unchanged (run125.py swaps the config/checkpoint); fp32 packed readout verified on 125M base weights (packed vs one-per-chunk max diff 9.5e-6, leakage test 1.1e-5, vs single path 1.6e-5; `experiments/exp08_125m/verify125.log`).
- Run 1: L1 s101, 32 epochs, shuffle seed 0 (same as 3B L1 ck: canaries, R1 and R1-alt every epoch).
- **"Binding measurable at 125M"** iff L1 reaches a binding D (R1) with CI lower bound > 0.5 nat at some epoch <= 32 (same threshold as 3B stage 0a). If not, 125M is not used for the outer-loop experiment (the effect to be changed would not be measurable), and this is reported.
- If measurable: Run 2: native L0 s101 (save/eval), and the 125M outer-loop design (two continued meta-training conditions: read/QA outer vs NTP outer, same data and steps) is written and registered before any training, after the 3B write-signal screening.
