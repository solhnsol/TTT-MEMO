"""CPU tests for semantic_inferential_v3 construction. No GPU, no LoRA."""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dataset import build_pair_records, expand_pair_records, sha256_bytes, write_sentence
from lexical_audit import run_lexical_audit
from protocol import CONCEPTS, PAIRS, SPEC


def test_prior_datasets_untouched() -> None:
    root = Path(__file__).resolve().parents[2]
    v1 = root / "exp02a_source_free_lora" / "frozen" / "dataset_v1.jsonl"
    v2 = root / "exp02a_source_free_lora" / "frozen" / "dataset_v2.jsonl"
    assert sha256_bytes(v1.read_bytes()) == "431d14327bef0a48fa86da5d7148c982368d730ba8265730bf057561c96a3a32"
    assert sha256_bytes(v2.read_bytes()) == "5a9c483e31e1856479369b1b5853972dc822b2a44e622e8156abdb445b776a01"


def test_fixed_pairs_and_scaffold() -> None:
    pairs = build_pair_records()
    assert len(pairs) == 4
    assert [p["pair_id"] for p in pairs] == ["P1", "P2", "P3", "P4"]
    assert pairs[0]["entities"] == {"e1": "Norvik-17", "e2": "Talven-03"}
    assert pairs[0]["concepts"]["c1"]["label"] == "electrician"
    assert pairs[0]["concepts"]["c2"]["label"] == "librarian"
    seq = pairs[0]["sequences"]["mapping_A"][0]
    assert seq["write_sentence"] == write_sentence("Norvik-17", CONCEPTS["S1"]["description"])
    assert seq["write_sentence"].startswith("Norvik-17 spends most workdays ")
    assert "electrician" not in seq["write_sentence"]
    expanded = expand_pair_records(pairs)
    assert len(expanded) == 16
    assert pairs[0]["info_type"] == "semantic_inferential_v3"


def test_independent_sequences_not_concatenated() -> None:
    pairs = build_pair_records()
    for p in pairs:
        for mapping in ("mapping_A", "mapping_B"):
            seqs = p["sequences"][mapping]
            assert len(seqs) == 2
            assert seqs[0]["write_sentence"] not in seqs[1]["write_sentence"]
            joined = seqs[0]["write_sentence"] + "\n" + seqs[1]["write_sentence"]
            assert seqs[0]["write_sentence"] != joined


def test_lexical_audit_passes_on_fixed_copy() -> None:
    pairs = build_pair_records()
    audit = run_lexical_audit(pairs)
    assert audit["passed"], audit["failures"][:3]
    assert audit["n_sequences_audited"] == 16


def test_pair_spec_entities_unique() -> None:
    ents = [e for p in PAIRS for e in (p["e1"], p["e2"])]
    assert len(ents) == len(set(ents)) == 8


def test_cashier_replaces_carpenter() -> None:
    labels = {c["label"] for c in CONCEPTS.values()}
    assert "cashier" in labels
    assert "carpenter" not in labels
    assert CONCEPTS["S4"]["label"] == "cashier"
    assert CONCEPTS["S4"]["description"] == (
        "scanning purchased items, accepting payments, and giving receipts "
        "and change to customers at a checkout counter"
    )
    pairs = build_pair_records()
    p3 = next(p for p in pairs if p["pair_id"] == "P3")
    assert p3["entities"] == {"e1": "Merak-52", "e2": "Drelon-08"}
    assert p3["candidates"] == ["cashier", "veterinarian"]
    blob = json.dumps(pairs)
    assert "carpenter" not in blob
    merak_a = p3["sequences"]["mapping_A"][0]
    assert merak_a["entity"] == "Merak-52"
    assert merak_a["write_sentence"] == (
        "Merak-52 spends most workdays scanning purchased items, accepting payments, "
        "and giving receipts and change to customers at a checkout counter."
    )
    assert "cashier" not in merak_a["write_sentence"]
    assert "cashier" not in merak_a["target"]


def test_canonical_occupation_query_only() -> None:
    templates = SPEC["query_templates"]
    assert list(templates.keys()) == ["occupation_query"]
    assert templates["occupation_query"] == (
        "What is {entity}'s occupation?\nReply with only the occupation title."
    )
    pairs = build_pair_records()
    for p in pairs:
        assert list(p["query_templates"].keys()) == ["occupation_query"]
        assert "What kind of work does {entity} do?" not in p["query_templates"].values()


if __name__ == "__main__":
    test_prior_datasets_untouched()
    test_fixed_pairs_and_scaffold()
    test_independent_sequences_not_concatenated()
    test_lexical_audit_passes_on_fixed_copy()
    test_pair_spec_entities_unique()
    test_cashier_replaces_carpenter()
    test_canonical_occupation_query_only()
    print("test_dataset_v3: PASS")
