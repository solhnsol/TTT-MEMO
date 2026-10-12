"""Exp 02A-2 source-free semantic binding protocol (memory stage).

Uses frozen dataset_v3. Does not alter descriptions, pairs, entities, or queries.
LoRA / optimizer match the validated Exp 02A K=32 recipe exactly.
"""

from __future__ import annotations

EXPECTED_DATASET_V3_SHA256 = "07d5749cf7710bf6106c24e558e580d11ea11649af2aa4a519b5c12bb1f32b40"

SPEC = {
    "experiment_id": "exp02a2_semantic_v3_source_free",
    "dataset_id": "semantic_inferential_v3",
    "dataset_sha256": EXPECTED_DATASET_V3_SHA256,
    "research_question": (
        "After NTP-LoRA write of entity-bound descriptions with occupation labels "
        "absent, does source-free recall (1) bind the correct description to the "
        "entity, and (2) make the unseen occupation label accessible?"
    ),
    "not_this_experiment": [
        "retention / interference",
        "query paraphrase robustness",
        "K=8",
        "new LoRA hyperparameters",
        "TTT-E2E",
        "source-present recalibration",
        "occupation search",
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
        "mask_entity_and_scaffold": True,
        "targets": "description_tokens_plus_final_period",
        "independent_sequences": True,
        "batch_the_two_sequences": True,
        "aggregation": "mean_of_per_sequence_mean_ce",
        "occupation_label_absent_from_write": True,
        "record_loss_before_and_after_every_step": True,
    },
    "conditions": [
        "static_source_free",
        "ttt_mapping_A_source_free",
        "ttt_mapping_B_source_free",
    ],
    "probes": ["description_binding", "semantic_label"],
    "generation": {
        "do_sample": False,
        "max_new_tokens": 32,
        "stop": "eos",
    },
    "primary_candidate_score": "mean_token_log_likelihood",
    "primary_binding_metric": "B_swap = (B_A + B_B) / 2",
}

DESIGN_CHOICES = {
    "independent_sequence_batch": (
        "Each mapping is two independent chat-rendered write sequences, never "
        "concatenated. They are stacked as batch size 2. Inner-loop loss is the "
        "unweighted mean of the two sequence-level mean CEs (equal sequence "
        "weight, not token-weighted across the batch). Token-weighted batch CE "
        "is recorded as a diagnostic only."
    ),
    "write_surface": (
        "Use the frozen chat-rendered write ids and ce_labels. User turn is "
        "'Fact:\\n' + write sentence. Labels cover description + final period only."
    ),
    "description_binding_probe": (
        "Source-free user message is exactly '{entity} spends most workdays' "
        "(the write prefix). Candidates are the two pair descriptions, scored "
        "as the assistant continuation. No extra reply instruction."
    ),
    "semantic_label_probe": (
        "Source-free user message is the frozen occupation_query, which already "
        "contains 'Reply with only the occupation title.' Candidates are the two "
        "occupation labels. Labels never appear in the write sequences."
    ),
    "static_source_free": (
        "Same probes with adapters disabled and no write. Diagnostic base-model "
        "preference. Static B_swap is not defined from one matrix; report the "
        "static 2x2 and static B_A (preference for the A pairing)."
    ),
    "no_source_present_this_stage": (
        "Source-present calibration already ran on frozen v3. This stage is "
        "source-free only: write tensors and KV are discarded before recall."
    ),
    "interpretation": (
        "Positive description B_swap supports source-free relational description "
        "binding. Positive semantic-label B_swap, with labels absent from write, "
        "supports semantic accessibility beyond direct lexical imprint. Do not "
        "call this semantic memory from candidate top-1 alone. n=4 is descriptive."
    ),
    "adamw_defaults": {
        "betas": [0.9, 0.999],
        "eps": 1e-8,
        "scheduler": None,
        "grad_clip": None,
    },
    "lora_trainable_dtype": (
        "LoRA A/B kept in float32; frozen base stays bfloat16. PEFT default init "
        "(kaiming A, zeros B) so a fresh adapter is ~identity."
    ),
}

CONDITIONS = tuple(SPEC["conditions"])
TTT_CONDITIONS = ("ttt_mapping_A_source_free", "ttt_mapping_B_source_free")
PROBES = tuple(SPEC["probes"])
