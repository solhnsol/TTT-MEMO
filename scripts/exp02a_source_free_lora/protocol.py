"""Exp 02A frozen protocol.

User-specified settings are in SPEC. Remaining design choices needed to make
the experiment executable are in DESIGN_CHOICES. Do not silently treat the
latter as results of the paper or of TTT-E2E.
"""

from __future__ import annotations

SPEC = {
    "experiment_id": "exp02a_source_free_lora",
    "research_question": (
        "Can standard NTP-based LoRA test-time adaptation on a strong "
        "pretrained instruction model carry a fact in the adapted weights "
        "after the source tokens and KV cache have been completely removed?"
    ),
    "not_this_experiment": [
        "long-context retrieval",
        "saliency / teacher attention",
        "learned routing",
        "explicit memory",
        "8K/16K/32K retention under filler",
        "continual unrelated-context TTT",
        "TTT-E2E comparison",
    ],
    "backbone": {
        "model_id": "Qwen/Qwen3-4B-Instruct-2507",
        "precision": "bfloat16",
        "forbid_fp8": True,
        "thinking_mode": False,
        "use_official_chat_template": True,
        # Resolved from huggingface_hub model_info on 2026-09-18.
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
        "k_steps_primary": 8,
        "k_steps_preregistered_escalation": 32,
        "fresh_adapter_and_optimizer_per_example": True,
        "one_textual_fact_exposure": True,
    },
    "write_loss": {
        "future_question_never_shown": True,
        "objective": "causal_next_token_ce",
        "mask_wrappers_out_of_loss": True,
        "record_loss_before_and_after_every_step": True,
    },
    "conditions": [
        "static_source_free",
        "ttt_correct_source_free",
        "ttt_shuffled_source_free",
        "static_source_present",
    ],
    "smoke": {"n_exact": 8, "n_semantic": 8, "k_steps": 8},
    "pilot_v2_k8": {"n_exact": 32, "n_semantic_direct": 32, "k_steps": 8},
    "pilot_v2_k32": {
        "n_exact": 32,
        "n_semantic_direct": 32,
        "k_steps": 32,
        "conditions": ["ttt_correct_source_free", "ttt_shuffled_source_free"],
        "reuse_from_k8": ["static_source_free", "static_source_present"],
        "only_changed_variable": "k_steps",
    },
    "generation": {
        "do_sample": False,
        "max_new_tokens": 32,
        "stop": "eos",
    },
    "primary_candidate_score": "mean_token_log_likelihood",
}

# Choices that the prompt did not fully pin down. Frozen before smoke.
DESIGN_CHOICES = {
    "write_format": (
        "Single-turn chat user message: wrapper 'Fact:\\n' + one fact sentence. "
        "No assistant turn at write time. This is NTP on the observed sequence "
        "(TTT-style), not SFT restatement of the fact. The fact string occurs once."
    ),
    "oracle_fact_span": (
        "Loss is on the payload tokens only. Exact payload = three-word passphrase. "
        "v1 semantic / v2 semantic_inferential payload = occupational paraphrase "
        "(canonical occupation word not in the write sentence). v2 semantic_direct "
        "payload = the canonical occupation label, which also appears in the write "
        "sentence. Chat specials, 'Fact:\\n', and fact-sentence scaffolding are "
        "masked to -100."
    ),
    "recall_source_free": (
        "Fresh conversation. User message = question + 'Reply with only the answer.' "
        "No fact sentence, no write tokens, no past_key_values. Only updated LoRA."
    ),
    "recall_source_present": (
        "Fresh conversation. User message = same 'Fact:\\n'+fact_sentence, then a "
        "blank line, then the question + reply instruction. Adapters disabled. "
        "This is the in-context upper bound, not a KV-reuse path."
    ),
    "recall_question": (
        "One held-out question form per information type, independent of the "
        "write template, so surface-form copying is not required for success."
    ),
    "candidate_scoring": (
        "Teacher-force each of 8 candidates as the assistant message. "
        "Smoke (dataset_v1) ranked by sum log-likelihood. From dataset_v2 "
        "onward the PRIMARY score is mean token log-likelihood: "
        "sum_j log p(a_j | q, a_<j) / n_answer_tokens. Sum logp is still "
        "stored as a diagnostic. Accuracy = gold has rank 1 under mean logp."
    ),
    "dataset_v2": (
        "Exact rows copied unchanged from frozen v1. v1 paraphrased semantic "
        "rows are stored as semantic_inferential and are not in the K=8 pilot. "
        "semantic_direct writes the canonical occupation label in the fact "
        "sentence; that label is the NTP payload and the recall gold. Recall "
        "question is a held-out paraphrase ('What kind of work does {entity} "
        "do?'), not a cloze of the write templates. 8 occupation labels, "
        "4 examples each, matched 8-way candidate set."
    ),
    "adamw_defaults": {
        "betas": [0.9, 0.999],
        "eps": 1e-8,
        "foreach": "pytorch_default",
        "scheduler": None,
        "grad_clip": None,
    },
    "lora_trainable_dtype": (
        "LoRA A/B kept in float32; frozen base stays bfloat16. PEFT default "
        "init (kaiming A, zeros B) so a fresh adapter is ~identity."
    ),
    "static_uses_disabled_adapter": (
        "static_* conditions call disable_adapter_layers() so they are the exact "
        "base model, not a near-zero LoRA residual."
    ),
    "shuffle_mapping": (
        "Derangement within information type over the full frozen 32+32 pool. "
        "Smoke may therefore adapt on a partner that is not itself a smoke target. "
        "Partner shares type and has no entity/value overlap."
    ),
    "candidates": (
        "8-way closed set drawn from the same-type frozen pool: gold plus 7 "
        "other canonical answers, order shuffled with a dataset seed. Matched "
        "across all four conditions."
    ),
    "generation": {
        "do_sample": False,
        "max_new_tokens": 32,
        "stop": "eos",
    },
    "em_normalization": "squad_v1.1 (lowercase, strip punct/articles, collapse space)",
    "dataset_seed": 20260918,
    "preregistered_escalation": (
        "If K=8 later shows no item-specific memory despite successful write "
        "optimization and a strong source-present upper bound, escalate only to "
        "K=32. Do not search other LoRA ranks/LRs/targets yet."
    ),
}

CONDITIONS = tuple(SPEC["conditions"])
TTT_CONDITIONS = ("ttt_correct_source_free", "ttt_shuffled_source_free")
SOURCE_FREE_CONDITIONS = (
    "static_source_free",
    "ttt_correct_source_free",
    "ttt_shuffled_source_free",
)
SOURCE_PRESENT_CONDITIONS = ("static_source_present",)
