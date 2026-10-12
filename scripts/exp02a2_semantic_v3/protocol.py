"""Exp 02A-2 dataset-preparation protocol: semantic_inferential_v3.

This stage constructs and validates the dataset only. It does not run LoRA TTT,
source-free recall, or any memory experiment.
"""

from __future__ import annotations

SPEC = {
    "experiment_id": "exp02a2_semantic_v3",
    "dataset_id": "semantic_inferential_v3",
    "stage": "dataset_preparation",
    "research_requirement": (
        "entity + descriptive behavior -> source removed later -> unseen semantic "
        "occupation label. The gold occupation label must never appear in the write "
        "sentence or write-loss target."
    ),
    "backbone": {
        "model_id": "Qwen/Qwen3-4B-Instruct-2507",
        "revision_sha": "cdbee75f17c01a7cc42f958dc650907174af0554",
        "precision": "bfloat16",
        "thinking_mode": False,
        "use_official_chat_template": True,
    },
    "write_scaffold": "{entity} spends most workdays {description}.",
    "query_templates": {
        "occupation_query": (
            "What is {entity}'s occupation?\nReply with only the occupation title."
        ),
    },
    "calibration_gates": {
        "occupation_query_required_correct": 16,
        "occupation_query_n_cells": 16,
        "primary_score": "mean_token_log_likelihood",
        "greedy_em_is_not_the_gate": True,
    },
    "n_pairs": 4,
    "n_concepts": 8,
}

# User-fixed content. Do not rewrite unless calibration fails AND that failure
# is reported first. Silent regeneration is forbidden.
CONCEPTS = {
    "S1": {
        "label": "electrician",
        "description": (
            "installing wiring, replacing circuit breakers, and fixing faulty "
            "outlets in buildings"
        ),
    },
    "S2": {
        "label": "librarian",
        "description": (
            "cataloging books, organizing collections, and helping visitors find "
            "reading materials"
        ),
    },
    "S3": {
        "label": "plumber",
        "description": (
            "repairing leaking pipes, clearing blocked drains, and installing "
            "sinks and faucets"
        ),
    },
    "S4": {
        "label": "cashier",
        "description": (
            "scanning purchased items, accepting payments, and giving receipts "
            "and change to customers at a checkout counter"
        ),
    },
    "S5": {
        "label": "pharmacist",
        "description": (
            "dispensing prescribed medicines, checking doses and drug interactions, "
            "and advising people on safe use"
        ),
    },
    "S6": {
        "label": "mechanic",
        "description": (
            "diagnosing engine problems, replacing worn brakes, and repairing "
            "vehicle transmissions"
        ),
    },
    "S7": {
        "label": "pilot",
        "description": (
            "operating passenger aircraft, following flight plans, and managing "
            "takeoff and landing"
        ),
    },
    "S8": {
        "label": "veterinarian",
        "description": (
            "examining sick animals, prescribing treatments, and performing medical "
            "procedures on pets"
        ),
    },
}

PAIRS = [
    {
        "pair_id": "P1",
        "e1": "Norvik-17",
        "e2": "Talven-03",
        "c1": "S1",
        "c2": "S2",
    },
    {
        "pair_id": "P2",
        "e1": "Keldar-29",
        "e2": "Vossin-41",
        "c1": "S3",
        "c2": "S5",
    },
    {
        "pair_id": "P3",
        "e1": "Merak-52",
        "e2": "Drelon-08",
        "c1": "S4",
        "c2": "S8",
    },
    {
        "pair_id": "P4",
        "e1": "Sovik-36",
        "e2": "Pelran-64",
        "c1": "S6",
        "c2": "S7",
    },
]

LEXICAL_FORBIDDEN = {
    "S1": ["electrician", "electrical", "electricians"],
    "S2": ["librarian", "library", "librarians", "libraries"],
    "S3": ["plumber", "plumbing", "plumbers"],
    "S4": ["cashier", "cashiers"],
    "S5": ["pharmacist", "pharmacy", "pharmacists", "pharmacies"],
    "S6": ["mechanic", "mechanical", "mechanics"],
    "S7": ["pilot", "pilots", "co-pilot", "copilot"],
    "S8": ["veterinarian", "veterinary", "veterinarians"],
}

# Whole-word only: "vet" must not match inside other tokens accidentally,
# but must be caught as a standalone occupation synonym.
LEXICAL_FORBIDDEN_WORD_BOUNDARY = {
    "S8": ["vet", "vets"],
}

SCAFFOLD_FORBIDDEN = [
    "occupation",
    "occupations",
    "works as",
    "job",
    "jobs",
    "profession",
    "career",
    "job title",
]

SYNONYM_FORBIDDEN = {
    "S8": ["animal doctor", "pet doctor"],
}

DESIGN_CHOICES = {
    "independent_sequences": (
        "Each mapping is two independent write sequences, never concatenated. "
        "Future LoRA will batch the two sequences and average their CE. Concatenation "
        "would introduce sentence-order / cross-sentence causal conditioning."
    ),
    "chat_rendered_write_metadata": (
        "Raw sentences are the dataset source of truth. Chat-rendered token spans "
        "are also stored using the Exp 02A convention (official chat template, "
        "user turn = 'Fact:\\n' + sentence, no assistant turn) so future NTP masking "
        "does not have to re-derive offsets. This wrapper is metadata only in this "
        "stage; no optimization is run."
    ),
    "prefix_target_split": (
        "Prefix/context = '{entity} spends most workdays' (no trailing space). "
        "Target = '{description}.' including the final period. The space between "
        "workdays and the description is a boundary token and is not part of the "
        "description substring; future CE labels use character-offset overlap with "
        "the description span plus the sentence-final period."
    ),
    "source_present_prompt": (
        "Calibration puts BOTH mapping sentences in one user message (Fact:\\n + "
        "sentence1 + newline + sentence2), then the query. This is in-context "
        "semantic inference, not the write format, and not KV reuse."
    ),
    "static_source_free_diagnostic": (
        "Same occupation-label scoring with descriptions absent and no LoRA. "
        "Diagnostic only; it does not select or rewrite the dataset."
    ),
    "canonical_query_only": (
        "Core evaluation uses a single occupation_query: "
        "\"What is {entity}'s occupation?\\nReply with only the occupation title.\" "
        "The previous paraphrase Q2 is not part of dataset_v3. The query already "
        "contains the reply instruction, so the recall renderer does not append "
        "'Reply with only the answer.'"
    ),
    "carpenter_replaced_by_cashier": (
        "S4 carpenter failed one source-present cell under the occupation query. "
        "It was replaced by cashier rather than tuning the carpenter description "
        "after observing that failure. Carpenter is not in the active v3 core."
    ),
}
