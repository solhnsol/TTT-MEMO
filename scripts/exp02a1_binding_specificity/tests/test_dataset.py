"""CPU+tokenizer tests for frozen binding pairs. Does not touch Exp 02A datasets."""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dataset import (
    DATASET_PATH,
    EXTRA_PASSPHRASE_WORDS,
    OCCUPATIONS,
    V1_PASSPHRASE_WORDS,
    freeze_dataset_pairs,
    load_dataset_pairs,
    sha256_bytes,
    token_id_multiset,
)
from protocol import SPEC


def test_exp02a_datasets_untouched() -> None:
    v1 = Path(__file__).resolve().parents[2] / "exp02a_source_free_lora" / "frozen" / "dataset_v1.jsonl"
    v2 = Path(__file__).resolve().parents[2] / "exp02a_source_free_lora" / "frozen" / "dataset_v2.jsonl"
    assert sha256_bytes(v1.read_bytes()) == "431d14327bef0a48fa86da5d7148c982368d730ba8265730bf057561c96a3a32"
    assert sha256_bytes(v2.read_bytes()) == "5a9c483e31e1856479369b1b5853972dc822b2a44e622e8156abdb445b776a01"


def test_freeze_and_invariants() -> None:
    meta = freeze_dataset_pairs(force=False)
    assert meta["n_pairs"] == 64
    assert meta["smoke_n_per_type"] == 4
    assert meta["backbone_revision_sha"] == SPEC["backbone"]["revision_sha"]
    pairs = load_dataset_pairs(smoke=False)
    smoke = load_dataset_pairs(smoke=True)
    assert len(pairs) == 64
    assert len(smoke) == 8
    assert {p["info_type"] for p in smoke} == {"exact", "semantic_direct"}
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(
        SPEC["backbone"]["model_id"],
        revision=SPEC["backbone"]["revision_sha"],
        use_fast=True,
    )
    n_e1_first = 0
    for p in pairs:
        assert token_id_multiset(tok, p["write_block_A"]) == token_id_multiset(tok, p["write_block_B"])
        if p["info_type"] == "exact":
            assert p["tokenizer"]["v1_n_tokens"] == p["tokenizer"]["v2_n_tokens"]
        assert p["tokenizer"]["values_token_length_matched"] == (
            p["tokenizer"]["v1_n_tokens"] == p["tokenizer"]["v2_n_tokens"]
        )
        assert p["write_block_A"].count(p["v1"]) == 1
        assert p["write_block_B"].count(p["v1"]) == 1
        assert p["candidates"] == [p["v1"], p["v2"]]
        if p["write_order"] == ["e1", "e2"]:
            n_e1_first += 1
        for ent_key in ("e1", "e2"):
            for q in p["questions"][ent_key].values():
                assert q not in p["write_block_A"]
                assert q not in p["write_block_B"]
        if p["info_type"] == "semantic_direct":
            assert p["v1"] in OCCUPATIONS and p["v2"] in OCCUPATIONS
            assert "works as a " not in p["write_block_A"]
    assert n_e1_first == 32  # 16 exact + 16 semantic
    assert set(V1_PASSPHRASE_WORDS).isdisjoint(EXTRA_PASSPHRASE_WORDS)
    assert DATASET_PATH.exists()
    manifest = json.loads(
        (Path(__file__).resolve().parents[1] / "frozen" / "dataset_manifest.json").read_text()
    )
    assert manifest["sha256"] == sha256_bytes(DATASET_PATH.read_bytes())


if __name__ == "__main__":
    test_exp02a_datasets_untouched()
    test_freeze_and_invariants()
    print("test_dataset: PASS")
