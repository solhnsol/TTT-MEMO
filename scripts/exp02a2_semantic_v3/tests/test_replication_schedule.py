"""CPU tests for Exp 02A-2R entity schedule. Tokenizer only, no GPU."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dataset import DATASET_PATH, sha256_bytes
from encode_write import load_tokenizer
from memory_protocol import EXPECTED_DATASET_V3_SHA256
from protocol import SPEC
from replication_schedule import (
    _V3_ENTITIES,
    N_ENTITIES,
    build_replicate_pairs,
    build_schedule,
    schedule_bytes,
)


def test_schedule_constraints() -> None:
    assert sha256_bytes(DATASET_PATH.read_bytes()) == EXPECTED_DATASET_V3_SHA256
    tokenizer = load_tokenizer()
    rows = build_schedule(tokenizer)
    assert len(rows) == 32
    ents = [e for r in rows for e in (r["e1"], r["e2"])]
    assert len(ents) == len(set(ents)) == N_ENTITIES
    assert not (set(ents) & _V3_ENTITIES)
    for r in rows:
        assert r["write_scaffold"] == SPEC["write_scaffold"]
        assert r["occupation_query"] == SPEC["query_templates"]["occupation_query"]
        assert r["dataset_v3_sha256"] == EXPECTED_DATASET_V3_SHA256
        assert r["e1"].count("-") == 1 and r["e2"].count("-") == 1
        assert r["label_c1"] not in r["e1"] and r["label_c1"] not in r["e2"]
        assert r["label_c2"] not in r["e1"] and r["label_c2"] not in r["e2"]
    by_src = {}
    for r in rows:
        by_src.setdefault(r["source_pair_id"], 0)
        by_src[r["source_pair_id"]] += 1
    assert by_src == {"P1": 8, "P2": 8, "P3": 8, "P4": 8}
    matched = sum(int(r["token_lengths_match"]) for r in rows)
    assert matched >= 24, matched
    payload = schedule_bytes(rows)
    again = schedule_bytes(build_schedule(tokenizer))
    assert payload == again
    pairs = build_replicate_pairs(rows)
    assert len(pairs) == 32
    assert pairs[0]["concepts"]["c1"]["label"] == "electrician"
    assert "cashier" in {p["concepts"]["c1"]["label"] for p in pairs if p["source_pair_id"] == "P3"}
    for p in pairs:
        for mapping in ("mapping_A", "mapping_B"):
            for seq in p["sequences"][mapping]:
                assert seq["label"] not in seq["write_sentence"]
                assert seq["write_sentence"].startswith(seq["entity"] + " spends most workdays ")


if __name__ == "__main__":
    test_schedule_constraints()
    print("test_replication_schedule: PASS")
