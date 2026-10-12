"""CPU tests for Exp 02A-2 source-free setup. No GPU, no LoRA."""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dataset import DATASET_PATH, sha256_bytes
from evaluate_memory import gold_value, probe_candidates, probe_question, write_sequences
from memory_protocol import EXPECTED_DATASET_V3_SHA256, SPEC
from metrics import b_swap, binding_score_from_matrix
from ttt_write import collate_write_batch


def _load_pairs() -> list[dict]:
    return [
        json.loads(line)
        for line in DATASET_PATH.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def test_frozen_hash_and_core() -> None:
    digest = sha256_bytes(DATASET_PATH.read_bytes())
    assert digest == EXPECTED_DATASET_V3_SHA256
    pairs = _load_pairs()
    assert [p["pair_id"] for p in pairs] == ["P1", "P2", "P3", "P4"]
    labels = [p["concepts"][k]["label"] for p in pairs for k in ("c1", "c2")]
    assert "cashier" in labels
    assert "carpenter" not in labels
    blob = DATASET_PATH.read_text(encoding="utf-8")
    assert "What kind of work does" not in blob
    assert "occupation_query" in blob


def test_write_has_no_occupation_labels() -> None:
    pairs = _load_pairs()
    for p in pairs:
        for mapping in ("mapping_A", "mapping_B"):
            seqs = write_sequences(p, mapping)
            assert len(seqs) == 2
            for seq in seqs:
                assert seq["label"] not in seq["write_sentence"]
                assert seq["label"] not in seq["text"]
                labeled = [i for i, y in enumerate(seq["ce_labels"]) if y != -100]
                assert labeled, seq["sequence_id"]
                decoded = seq["text"]
                assert "Fact:\n" in decoded


def test_probe_strings() -> None:
    pairs = _load_pairs()
    p3 = next(p for p in pairs if p["pair_id"] == "P3")
    q = probe_question(p3, "e1", "description_binding")
    assert q == "Merak-52 spends most workdays"
    q2 = probe_question(p3, "e1", "semantic_label")
    assert q2 == "What is Merak-52's occupation?\nReply with only the occupation title."
    assert probe_candidates(p3, "semantic_label") == ["cashier", "veterinarian"]
    gold_desc = gold_value(p3, mapping="mapping_A", entity_key="e1", probe="description_binding")
    assert gold_desc.startswith("scanning purchased items")
    gold_lab = gold_value(p3, mapping="mapping_B", entity_key="e1", probe="semantic_label")
    assert gold_lab == "veterinarian"


def test_bswap_formula() -> None:
    mat_a = {"e1": {"v1": 1.0, "v2": 0.0}, "e2": {"v1": 0.0, "v2": 1.0}}
    mat_b = {"e1": {"v1": 0.0, "v2": 1.0}, "e2": {"v1": 1.0, "v2": 0.0}}
    out = b_swap(mat_a, mat_b)
    assert abs(out["B_A"] - 1.0) < 1e-9
    assert abs(out["B_B"] - 1.0) < 1e-9
    assert abs(out["B_swap"] - 1.0) < 1e-9
    static = binding_score_from_matrix(mat_a, "A")
    assert abs(static - 1.0) < 1e-9


def test_collate_batch_padding() -> None:
    import torch

    seqs = [
        {"input_ids": [1, 2, 3], "ce_labels": [-100, 2, 3]},
        {"input_ids": [4, 5], "ce_labels": [-100, 5]},
    ]
    batch = collate_write_batch(seqs, pad_token_id=0, device=torch.device("cpu"))
    assert batch["input_ids"].shape == (2, 3)
    assert batch["input_ids"][1].tolist() == [4, 5, 0]
    assert batch["labels"][1].tolist() == [-100, 5, -100]
    assert batch["attention_mask"][1].tolist() == [1, 1, 0]


def test_k32_recipe() -> None:
    assert SPEC["lora"]["k_steps"] == 32
    assert SPEC["lora"]["target_modules"] == ["q_proj", "o_proj"]
    assert SPEC["lora"]["r"] == 16
    assert SPEC["lora"]["alpha"] == 32
    assert SPEC["lora"]["lr"] == 1e-4
    assert SPEC["lora"]["weight_decay"] == 0.0
    assert SPEC["backbone"]["revision_sha"] == "cdbee75f17c01a7cc42f958dc650907174af0554"


if __name__ == "__main__":
    test_frozen_hash_and_core()
    test_write_has_no_occupation_labels()
    test_probe_strings()
    test_bswap_formula()
    test_collate_batch_padding()
    test_k32_recipe()
    print("test_memory_setup: PASS")
