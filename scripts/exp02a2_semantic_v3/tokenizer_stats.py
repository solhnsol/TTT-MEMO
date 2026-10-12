"""Tokenizer statistics for semantic_inferential_v3. No dataset rewriting."""

from __future__ import annotations

from typing import Any


def collect_tokenizer_stats(pairs: list[dict[str, Any]]) -> dict[str, Any]:
    entities = {}
    labels = {}
    descriptions = {}
    sentences = []
    pair_rows = []
    for pair in pairs:
        c1 = pair["concepts"]["c1"]
        c2 = pair["concepts"]["c2"]
        # labels/descriptions from either mapping; same concepts.
        for key, concept in (("c1", c1), ("c2", c2)):
            seq = next(
                s
                for s in pair["sequences"]["mapping_A"]
                if s["concept_key"] == key
            )
            t = seq["tokenization"]
            labels[concept["label"]] = {
                "n_tokens": t["label_n_tokens"],
                "token_ids": t["label_token_ids"],
            }
            descriptions[concept["label"]] = {
                "n_tokens": t["description_n_tokens"],
                "token_ids": t["description_token_ids"],
                "description": concept["description"],
            }
            entities[seq["entity"]] = {
                "n_tokens": t["entity_n_tokens"],
                "token_ids": t["entity_token_ids"],
            }
        e2_seq = pair["sequences"]["mapping_A"][1]
        entities[e2_seq["entity"]] = {
            "n_tokens": e2_seq["tokenization"]["entity_n_tokens"],
            "token_ids": e2_seq["tokenization"]["entity_token_ids"],
        }
        for mapping in ("mapping_A", "mapping_B"):
            for seq in pair["sequences"][mapping]:
                raw = seq["tokenization"]["raw_sentence"]
                sentences.append(
                    {
                        "sequence_id": seq["sequence_id"],
                        "n_tokens": raw["n_tokens"],
                        "n_labeled_target_tokens": raw["n_labeled_target_tokens"],
                        "target_token_indices": raw["target_token_indices"],
                        "prefix_token_indices": raw["prefix_token_indices"],
                    }
                )
        n1 = labels[c1["label"]]["n_tokens"]
        n2 = labels[c2["label"]]["n_tokens"]
        d1 = descriptions[c1["label"]]["n_tokens"]
        d2 = descriptions[c2["label"]]["n_tokens"]
        pair_rows.append(
            {
                "pair_id": pair["pair_id"],
                "label_c1": c1["label"],
                "label_c2": c2["label"],
                "label_n_tokens_c1": n1,
                "label_n_tokens_c2": n2,
                "label_lengths_match": n1 == n2,
                "description_n_tokens_c1": d1,
                "description_n_tokens_c2": d2,
                "description_length_abs_diff": abs(d1 - d2),
                "description_highly_imbalanced": abs(d1 - d2) >= 6,
            }
        )
    return {
        "note": (
            "Token lengths are recorded, not edited. Do not rewrite the dataset "
            "to force equality at this stage."
        ),
        "entities": entities,
        "labels": labels,
        "descriptions": descriptions,
        "write_sentences": sentences,
        "per_pair": pair_rows,
    }
