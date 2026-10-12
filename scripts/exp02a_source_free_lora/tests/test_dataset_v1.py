"""CPU tests for frozen dataset_v1. No GPU, no model scores."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dataset_v1 import (
    DATASET_SEED,
    build_dataset_v1,
    freeze_dataset_v1,
    load_dataset_v1,
    sha256_bytes,
    dataset_bytes,
    validate_dataset_v1,
)


def test_build_is_deterministic() -> None:
    a = build_dataset_v1(seed=DATASET_SEED)
    b = build_dataset_v1(seed=DATASET_SEED)
    assert dataset_bytes(a) == dataset_bytes(b)


def test_validate_and_balance() -> None:
    examples = build_dataset_v1()
    validate_dataset_v1(examples)
    exact = [ex for ex in examples if ex["info_type"] == "exact"]
    semantic = [ex for ex in examples if ex["info_type"] == "semantic"]
    assert len(exact) == 32 and len(semantic) == 32
    assert sum(ex["smoke"] for ex in exact) == 8
    assert sum(ex["smoke"] for ex in semantic) == 8
    assert all(len(ex["candidates"]) == 8 for ex in examples)
    assert all("?" not in ex["fact_sentence"] for ex in examples)


def test_semantic_answer_not_in_write() -> None:
    for ex in build_dataset_v1():
        if ex["info_type"] != "semantic":
            continue
        assert ex["answer"].lower() not in ex["fact_sentence"].lower()
        assert ex["answer"] != ex["write_payload"]


def test_shuffle_is_derangement_same_type() -> None:
    examples = build_dataset_v1()
    by_id = {ex["example_id"]: ex for ex in examples}
    for ex in examples:
        partner = by_id[ex["shuffle_partner_id"]]
        assert partner["example_id"] != ex["example_id"]
        assert partner["info_type"] == ex["info_type"]
        assert partner["entity"] != ex["entity"]
        assert partner["answer"] != ex["answer"]


def test_one_fact_exposure_fields() -> None:
    for ex in build_dataset_v1():
        assert ex["fact_sentence"].count(ex["write_payload"]) == 1
        assert ex["question"] not in ex["fact_sentence"]
        assert ex["entity"] in ex["question"]


def test_freeze_roundtrip() -> None:
    meta = freeze_dataset_v1(force=False)
    loaded = load_dataset_v1(smoke=False)
    smoke = load_dataset_v1(smoke=True)
    assert len(loaded) == 64
    assert len(smoke) == 16
    assert meta["sha256"] == sha256_bytes(dataset_bytes(loaded))
    assert {ex["example_id"] for ex in smoke} == {
        *[f"exact-{i:02d}" for i in range(8)],
        *[f"semantic-{i:02d}" for i in range(8)],
    }


if __name__ == "__main__":
    test_build_is_deterministic()
    test_validate_and_balance()
    test_semantic_answer_not_in_write()
    test_shuffle_is_derangement_same_type()
    test_one_fact_exposure_fields()
    test_freeze_roundtrip()
    print("test_dataset_v1: PASS")
