# Exp 02A-3 counterfactual entity–value binding

This runner reuses the original Exp 02A model runtime, masked NTP write,
chat prompt encoding, and candidate scoring without modifying those files.
Backbone/revision, q_proj/o_proj rank-16 LoRA, AdamW lr=1e-4 and K=32
match the existing recipe. Only the value changes between A and B writes.

The dataset takes the first eight exact pairs from the frozen Exp 02A-1
binding dataset before inspecting any new scores. For each triplet, e1 is
written and e2 is never written. The control entity changes only the entity
in the same question template. Candidate lengths are checked to match.

For candidate mean-token logp S:

    preference_c(e) = S_c(V1 | q_e) - S_c(V2 | q_e)
    D(e) = preference_A(e) - preference_B(e)
    B_entity = D(E) - D(E_control)

Positive B_entity means the counterfactual preference change is greater for
the written entity than the unwritten control. A query-independent value
boost cancels. Sum-token logp uses the same calculation as a diagnostic.
A positive average does not establish successful free generation, complete
binding, or generalization beyond these exact entities and question forms.

Each condition restores the exact initial adapter snapshot and starts a
fresh empty AdamW. Write tensors, optimizer, and gradients are removed before
recall. A forward guard checks use_cache=False, past_key_values=None and the
actual returned cache=None on every model call. No source text or values
appear in recall prompts. The adapter is checked unchanged during recall;
base weights are frozen and selected base projections are checked unchanged.

Run from the repository root, choosing a new output directory:

    .venv-exp02a/bin/python -B scripts/exp02a3_counterfactual_binding/run_pilot.py \
      --n 8 --out experiments/exp02a3_counterfactual_binding/pilot_<new_run_id>

Offline loading is enforced. Existing output directories are rejected.
`dataset.jsonl` and `config.json` preserve the selection, hashes and settings;
`samples.jsonl` contains actual write/recall/scoring texts, loss trajectories,
and all sample scores. `summary.json` records metrics, descriptive bootstrap
intervals, checks, elapsed time and peak VRAM. Small-N bootstrap intervals
are descriptive only. There is no hyperparameter search or long context.
