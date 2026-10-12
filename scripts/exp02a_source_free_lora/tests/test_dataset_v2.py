"""CPU tests for frozen dataset_v2. Does not rewrite dataset_v1."""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dataset_v1 import load_dataset_v1, sha256_bytes
from dataset_v1 import DATASET_PATH as V1_PATH
from dataset_v2 import (
    EXPECTED_V1_SHA256,
    SEMANTIC_DIRECT_OCCUPATIONS,
    build_dataset_v2,
    freeze_dataset_v2,
    load_dataset_v2,
)


def test_v1_hash_untouched() -> None:
    assert sha256_bytes(V1_PATH.read_bytes()) == EXPECTED_V1_SHA256
    meta = json.loads(
        (Path(__file__).resolve().parents[1] / "frozen" / "dataset_v1.meta.json").read_text()
    )
    assert meta["sha256"] == EXPECTED_V1_SHA256


def test_exact_copied_from_v1() -> None:
    v1 = [ex for ex in load_dataset_v1(smoke=False) if ex["info_type"] == "exact"]
    v2 = [ex for ex in build_dataset_v2() if ex["info_type"] == "exact"]
    assert v1 == v2


def test_inferential_is_relabeled_v1_semantic() -> None:
    v1 = [ex for ex in load_dataset_v1(smoke=False) if ex["info_type"] == "semantic"]
    inf = [ex for ex in build_dataset_v2() if ex["info_type"] == "semantic_inferential"]
    assert len(inf) == 32
    for old, new in zip(v1, inf):
        assert new["example_id"] == old["example_id"]
        assert new["fact_sentence"] == old["fact_sentence"]
        assert new["write_payload"] == old["write_payload"]
        assert new["answer"] == old["answer"]
        assert new["answer"] not in new["fact_sentence"]


def test_semantic_direct_gold_in_write() -> None:
    direct = [ex for ex in build_dataset_v2() if ex["info_type"] == "semantic_direct"]
    assert len(direct) == 32
    assert {ex["answer"] for ex in direct} == set(SEMANTIC_DIRECT_OCCUPATIONS)
    for ex in direct:
        assert ex["write_payload"] == ex["answer"]
        assert ex["fact_sentence"].count(ex["answer"]) == 1
        assert "?" not in ex["fact_sentence"]
        assert ex["question"] not in ex["fact_sentence"]
        assert "What kind of work" in ex["question"]
        assert len(ex["candidates"]) == 8
        assert {c["text"] for c in ex["candidates"]} == set(SEMANTIC_DIRECT_OCCUPATIONS)


def test_semantic_direct_shuffle_no_value_overlap() -> None:
    examples = build_dataset_v2()
    by_id = {ex["example_id"]: ex for ex in examples}
    for ex in examples:
        if ex["info_type"] != "semantic_direct":
            continue
        partner = by_id[ex["shuffle_partner_id"]]
        assert partner["info_type"] == "semantic_direct"
        assert partner["answer"] != ex["answer"]
        assert partner["entity"] != ex["entity"]


def test_pilot_subset_excludes_inferential() -> None:
    meta = freeze_dataset_v2(force=False)
    assert meta["source_v1_sha256"] == EXPECTED_V1_SHA256
    full = load_dataset_v2(pilot=False)
    pilot = load_dataset_v2(pilot=True)
    assert len(full) == 96
    assert len(pilot) == 64
    assert {ex["info_type"] for ex in pilot} == {"exact", "semantic_direct"}
    assert sha256_bytes(V1_PATH.read_bytes()) == EXPECTED_V1_SHA256


if __name__ == "__main__":
    test_v1_hash_untouched()
    test_exact_copied_from_v1()
    test_inferential_is_relabeled_v1_semantic()
    test_semantic_direct_gold_in_write()
    test_semantic_direct_shuffle_no_value_overlap()
    test_pilot_subset_excludes_inferential()
    print("test_dataset_v2: PASS")
