"""Exp 02A-1 frozen protocol: source-free binding specificity.

User-specified settings are in SPEC. Remaining design choices needed to make
the experiment executable are in DESIGN_CHOICES. Do not silently treat the
latter as results of the paper or of TTT-E2E.

Backbone / LoRA / optimizer match the validated Exp 02A K=32 recipe exactly.
"""

from __future__ import annotations

SPEC = {
    "experiment_id": "exp02a1_binding_specificity",
    "research_question": (
        "Does K=32 NTP-LoRA merely increase the probability of value tokens "
        "seen during write, or does it preserve which value was bound to which "
        "entity after the source context and KV cache are completely removed?"
    ),
    "not_this_experiment": [
        "retention / interference under filler",
        "semantic_inferential (canonical answer absent from write)",
        "K=8",
        "new LoRA hyperparameters",
        "TTT-E2E comparison",
        "routing or explicit memory",
        "full 32+32 before smoke",
    ],
    "backbone": {
        "model_id": "Qwen/Qwen3-4B-Instruct-2507",
        "precision": "bfloat16",
        "forbid_fp8": True,
        "thinking_mode": False,
        "use_official_chat_template": True,
        "revision_sha": "cdbee75f17c01a7cc42f958dc650907174af0554",
    },
    "lora": {
        "target_modules": ["q_proj", "o_proj"],
        "r": 16,
        "alpha": 32,
        "dropout": 0.0,
        "bias": "none",
        "freeze_base": True,
        "optimizer": "AdamW",
        "lr": 1e-4,
        "weight_decay": 0.0,
        "k_steps": 32,
        "fresh_adapter_and_optimizer_per_mapping": True,
        "one_pair_exposure": True,
    },
    "write_loss": {
        "future_question_never_shown": True,
        "objective": "causal_next_token_ce",
        "mask_wrappers_out_of_loss": True,
        "both_value_spans_labeled": True,
        "record_loss_before_and_after_every_step": True,
    },
    "conditions": [
        "static_source_free",
        "ttt_mapping_A_source_free",
        "ttt_mapping_B_source_free",
        "static_mapping_A_source_present",
        "static_mapping_B_source_present",
    ],
    "query_styles": ["lexical_near", "paraphrase"],
    "info_types": ["exact", "semantic_direct"],
    "smoke": {"n_exact_pairs": 4, "n_semantic_direct_pairs": 4, "k_steps": 32},
    "frozen_pool": {"n_exact_pairs": 32, "n_semantic_direct_pairs": 32},
    "generation": {
        "do_sample": False,
        "max_new_tokens": 32,
        "stop": "eos",
    },
    "primary_candidate_score": "mean_token_log_likelihood",
    "primary_binding_metric": "B_swap = (B_A + B_B) / 2",
    "source_present_calibrated_min_two_way_acc": 0.75,
}

DESIGN_CHOICES = {
    "write_format": (
        "Single-turn chat user message: wrapper 'Fact:\\n' plus two fact sentences "
        "joined by a newline. No assistant turn at write time. The pair is one "
        "sequence and one exposure; K=32 is the optimizer budget, not two replays."
    ),
    "article_free_semantic_template": (
        "User illustration used 'works as an electrician' / 'works as a librarian'. "
        "That a/an alternation would change the token inventory when values swap. "
        "The identical-inventory constraint takes precedence, so semantic write is "
        "'{entity}'s occupation is {payload}.' with no article. Exact write matches "
        "the specified frame: 'The passphrase assigned to {entity} is {payload}.'"
    ),
    "oracle_fact_span": (
        "Loss is on both value spans. Wrappers, chat specials, and scaffolding "
        "('The passphrase assigned to {entity} is ' / '{entity}'s occupation is ') "
        "are masked to -100. Character-offset overlap, same as Exp 02A."
    ),
    "sentence_order": (
        "Entity order is identical within A/B for a pair. Half of pairs write e1 "
        "first (even pair index), half write e2 first (odd pair index), so first/"
        "second position is not confounded with mapping."
    ),
    "recall_source_free": (
        "Fresh conversation. User message = question + 'Reply with only the answer.' "
        "No fact sentences, no write tokens, no past_key_values. Only updated LoRA. "
        "The same adapter is scored on both entities and both query styles; recall "
        "does no gradient step."
    ),
    "recall_source_present": (
        "Fresh conversation. User message = same 'Fact:\\n'+two-sentence block, "
        "blank line, question + reply instruction. Adapters disabled. In-context "
        "upper bound, not KV reuse. Evaluated for both Mapping A and Mapping B."
    ),
    "static_source_free": (
        "Same queries once with adapters disabled and no write. Diagnostic only. "
        "Causal evidence is the A/B swap interaction, not TTT vs static. Static "
        "B_swap is not defined from a single score matrix (the two mappings would "
        "cancel). Report the static 2x2 and static B_A (base-model preference for "
        "the A pairing) as the static score."
    ),
    "candidate_scoring": (
        "Teacher-force both pair values as the assistant message. PRIMARY score is "
        "mean token log-likelihood. Sum logp is stored as a diagnostic. Two-way "
        "accuracy = gold has rank 1 under mean logp (string tie-break, same as Exp 02A). "
        "Flip diagnostics use a strict mean-logp inequality with no string tie-break."
    ),
    "occupation_reuse": (
        "32 validated occupation labels from Exp 02A SEMANTIC_ITEMS are paired "
        "preferring equal tokenizer length. 14 of 16 unique pairs match exactly; "
        "the two leftover odd-count length buckets are paired across adjacent "
        "lengths and flagged values_token_length_matched=false. Each value-pair "
        "is used twice with different entities, yielding 32 semantic_direct pairs."
    ),
    "passphrases": (
        "64 unique three-word passphrases from common English nouns, paired so that "
        "tokenizer length of v1 equals v2 within each pair. Recorded at freeze time "
        "with the official Qwen tokenizer."
    ),
    "entities": (
        "128 unique Stem-NN nonce IDs, disjoint from Exp 02A v1 two-word names and "
        "v2 Norvik-style IDs. e1/e2 are unique within and across pairs."
    ),
    "adamw_defaults": {
        "betas": [0.9, 0.999],
        "eps": 1e-8,
        "foreach": "pytorch_default",
        "scheduler": None,
        "grad_clip": None,
    },
    "lora_trainable_dtype": (
        "LoRA A/B kept in float32; frozen base stays bfloat16. PEFT default init "
        "(kaiming A, zeros B) so a fresh adapter is ~identity."
    ),
    "static_uses_disabled_adapter": (
        "static_* conditions call disable_adapter_layers() so they are the exact "
        "base model, not a near-zero LoRA residual."
    ),
    "em_normalization": "squad_v1.1 (lowercase, strip punct/articles, collapse space)",
    "dataset_seed": 20260918,
    "interpretation": (
        "A positive B_swap supports source-free NTP-LoRA preserving entity-value "
        "relational binding beyond a global value-token probability imprint. Do not "
        "call this semantic memory. Semantic abstraction requires a later "
        "semantic_inferential experiment where the canonical answer token is absent "
        "from the write payload. Do not interpret a query style whose source-present "
        "two-way accuracy is below 0.75."
    ),
    "smoke_bootstrap": (
        "Paired bootstrap 95% CI on B_swap is written as a descriptive interval at "
        "n=4. Do not make a statistical claim from the smoke subset."
    ),
}

CONDITIONS = tuple(SPEC["conditions"])
TTT_CONDITIONS = ("ttt_mapping_A_source_free", "ttt_mapping_B_source_free")
SOURCE_FREE_CONDITIONS = (
    "static_source_free",
    "ttt_mapping_A_source_free",
    "ttt_mapping_B_source_free",
)
SOURCE_PRESENT_CONDITIONS = (
    "static_mapping_A_source_present",
    "static_mapping_B_source_present",
)
QUERY_STYLES = tuple(SPEC["query_styles"])
