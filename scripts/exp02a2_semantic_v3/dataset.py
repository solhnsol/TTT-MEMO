"""Build and freeze semantic_inferential_v3. Does not run LoRA."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from protocol import CONCEPTS, DESIGN_CHOICES, PAIRS, SPEC

FROZEN_DIR = Path(__file__).resolve().parent / "frozen"
DATASET_PATH = FROZEN_DIR / "dataset_v3.jsonl"
MANIFEST_PATH = FROZEN_DIR / "dataset_v3_manifest.json"

WRITE_SCAFFOLD = SPEC["write_scaffold"]
QUERY_TEMPLATES = SPEC["query_templates"]


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def dataset_bytes(rows: list[dict[str, Any]]) -> bytes:
    lines = [json.dumps(row, ensure_ascii=False, sort_keys=True) for row in rows]
    return ("\n".join(lines) + "\n").encode("utf-8")


def write_sentence(entity: str, description: str) -> str:
    return WRITE_SCAFFOLD.format(entity=entity, description=description)


def prefix_text(entity: str) -> str:
    return f"{entity} spends most workdays"


def target_text(description: str) -> str:
    return f"{description}."


def build_pair_records() -> list[dict[str, Any]]:
    rows = []
    used_entities: list[str] = []
    for spec in PAIRS:
        c1 = CONCEPTS[spec["c1"]]
        c2 = CONCEPTS[spec["c2"]]
        e1, e2 = spec["e1"], spec["e2"]
        used_entities.extend([e1, e2])
        sent_e1_c1 = write_sentence(e1, c1["description"])
        sent_e2_c2 = write_sentence(e2, c2["description"])
        sent_e1_c2 = write_sentence(e1, c2["description"])
        sent_e2_c1 = write_sentence(e2, c1["description"])
        row = {
            "pair_id": spec["pair_id"],
            "info_type": "semantic_inferential_v3",
            "entities": {"e1": e1, "e2": e2},
            "concepts": {
                "c1": {
                    "concept_id": spec["c1"],
                    "label": c1["label"],
                    "description": c1["description"],
                },
                "c2": {
                    "concept_id": spec["c2"],
                    "label": c2["label"],
                    "description": c2["description"],
                },
            },
            "mapping_A": {"e1": "c1", "e2": "c2"},
            "mapping_B": {"e1": "c2", "e2": "c1"},
            "query_templates": dict(QUERY_TEMPLATES),
            "sequences": {
                "mapping_A": [
                    {
                        "sequence_id": f"{spec['pair_id']}_A_e1",
                        "entity_key": "e1",
                        "entity": e1,
                        "concept_key": "c1",
                        "label": c1["label"],
                        "description": c1["description"],
                        "write_sentence": sent_e1_c1,
                        "prefix": prefix_text(e1),
                        "target": target_text(c1["description"]),
                    },
                    {
                        "sequence_id": f"{spec['pair_id']}_A_e2",
                        "entity_key": "e2",
                        "entity": e2,
                        "concept_key": "c2",
                        "label": c2["label"],
                        "description": c2["description"],
                        "write_sentence": sent_e2_c2,
                        "prefix": prefix_text(e2),
                        "target": target_text(c2["description"]),
                    },
                ],
                "mapping_B": [
                    {
                        "sequence_id": f"{spec['pair_id']}_B_e1",
                        "entity_key": "e1",
                        "entity": e1,
                        "concept_key": "c2",
                        "label": c2["label"],
                        "description": c2["description"],
                        "write_sentence": sent_e1_c2,
                        "prefix": prefix_text(e1),
                        "target": target_text(c2["description"]),
                    },
                    {
                        "sequence_id": f"{spec['pair_id']}_B_e2",
                        "entity_key": "e2",
                        "entity": e2,
                        "concept_key": "c1",
                        "label": c1["label"],
                        "description": c1["description"],
                        "write_sentence": sent_e2_c1,
                        "prefix": prefix_text(e2),
                        "target": target_text(c1["description"]),
                    },
                ],
            },
            "candidates": [c1["label"], c2["label"]],
        }
        rows.append(row)
    if len(set(used_entities)) != 8:
        raise RuntimeError(f"expected 8 unique entities, got {used_entities}")
    validate_pair_records(rows)
    return rows


def expand_pair_records(pairs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """One row per independent write sequence."""
    out = []
    for pair in pairs:
        for mapping in ("mapping_A", "mapping_B"):
            for seq in pair["sequences"][mapping]:
                out.append(
                    {
                        "pair_id": pair["pair_id"],
                        "info_type": "semantic_inferential_v3",
                        "mapping": mapping,
                        **seq,
                        "candidates": list(pair["candidates"]),
                        "query_templates": dict(pair["query_templates"]),
                        "entities": dict(pair["entities"]),
                        "concepts": json.loads(json.dumps(pair["concepts"])),
                    }
                )
    if len(out) != 16:
        raise RuntimeError(f"expected 16 independent sequences, got {len(out)}")
    return out


def validate_pair_records(pairs: list[dict[str, Any]]) -> None:
    if len(pairs) != 4:
        raise ValueError(f"expected 4 pairs, got {len(pairs)}")
    entities = []
    labels = []
    for p in pairs:
        if p["info_type"] != "semantic_inferential_v3":
            raise ValueError("info_type must be semantic_inferential_v3")
        e1, e2 = p["entities"]["e1"], p["entities"]["e2"]
        entities.extend([e1, e2])
        c1, c2 = p["concepts"]["c1"], p["concepts"]["c2"]
        labels.extend([c1["label"], c2["label"]])
        if e1 == e2 or c1["label"] == c2["label"]:
            raise ValueError(f"{p['pair_id']}: entities or labels not distinct")
        if p["mapping_A"] != {"e1": "c1", "e2": "c2"}:
            raise ValueError(f"{p['pair_id']}: mapping A must be e1->c1, e2->c2")
        if p["mapping_B"] != {"e1": "c2", "e2": "c1"}:
            raise ValueError(f"{p['pair_id']}: mapping B must be swapped")
        if p["candidates"] != [c1["label"], c2["label"]]:
            raise ValueError(f"{p['pair_id']}: candidates must be [c1, c2] labels")
        for mapping in ("mapping_A", "mapping_B"):
            seqs = p["sequences"][mapping]
            if len(seqs) != 2:
                raise ValueError(f"{p['pair_id']} {mapping}: expected 2 independent sequences")
            sents = [s["write_sentence"] for s in seqs]
            if sents[0] == sents[1]:
                raise ValueError(f"{p['pair_id']} {mapping}: sequences identical")
            for s in seqs:
                expected = write_sentence(s["entity"], s["description"])
                if s["write_sentence"] != expected:
                    raise ValueError(f"{s['sequence_id']}: write sentence mismatch")
                if not s["write_sentence"].startswith(s["prefix"]):
                    raise ValueError(f"{s['sequence_id']}: prefix is not a prefix of the sentence")
                if not s["write_sentence"].endswith(s["target"]):
                    raise ValueError(f"{s['sequence_id']}: target is not a suffix of the sentence")
                if s["label"] in s["write_sentence"]:
                    raise ValueError(f"{s['sequence_id']}: gold label leaked into write")
                if "occupation" in s["write_sentence"].lower():
                    raise ValueError(f"{s['sequence_id']}: occupation in write")
        labels_a = {s["label"] for s in p["sequences"]["mapping_A"]}
        labels_b = {s["label"] for s in p["sequences"]["mapping_B"]}
        if labels_a != labels_b:
            raise ValueError(f"{p['pair_id']}: A/B label sets differ")
        desc_a = sorted(s["description"] for s in p["sequences"]["mapping_A"])
        desc_b = sorted(s["description"] for s in p["sequences"]["mapping_B"])
        if desc_a != desc_b:
            raise ValueError(f"{p['pair_id']}: A/B description inventories differ")
        ent_a = sorted(s["entity"] for s in p["sequences"]["mapping_A"])
        ent_b = sorted(s["entity"] for s in p["sequences"]["mapping_B"])
        if ent_a != ent_b:
            raise ValueError(f"{p['pair_id']}: A/B entity inventories differ")
        # Full sentence strings MUST differ: that is the swapped association.
        if {s["write_sentence"] for s in p["sequences"]["mapping_A"]} == {
            s["write_sentence"] for s in p["sequences"]["mapping_B"]
        }:
            raise ValueError(f"{p['pair_id']}: A and B sentence sets are identical; swap failed")
    if len(set(entities)) != 8:
        raise ValueError(f"entity reuse: {entities}")
    if len(set(labels)) != 8:
        raise ValueError(f"label reuse: {labels}")


def _pair_token_inventory_equal(pair: dict[str, Any]) -> bool:
    def bag(mapping: str) -> Counter:
        c: Counter = Counter()
        for seq in pair["sequences"][mapping]:
            ids = seq["tokenization"]["raw_sentence"]["input_ids"]
            c.update(ids)
        return c

    return bag("mapping_A") == bag("mapping_B")


def attach_token_metadata(pairs: list[dict[str, Any]], tokenizer) -> list[dict[str, Any]]:
    from encode_write import encode_independent_sequence

    out = []
    for pair in json.loads(json.dumps(pairs)):
        for mapping in ("mapping_A", "mapping_B"):
            encoded_seqs = []
            for seq in pair["sequences"][mapping]:
                meta = encode_independent_sequence(tokenizer, seq)
                seq["tokenization"] = meta
                encoded_seqs.append(seq)
            pair["sequences"][mapping] = encoded_seqs
        pair["token_inventory_A_equals_B"] = _pair_token_inventory_equal(pair)
        if not pair["token_inventory_A_equals_B"]:
            raise RuntimeError(f"{pair['pair_id']}: A/B combined token inventories differ")
        out.append(pair)
    return out


def freeze_dataset_v3(
    pairs: list[dict[str, Any]],
    *,
    manifest_extra: dict[str, Any],
    force: bool = False,
) -> dict[str, Any]:
    validate_pair_records(pairs)
    payload = dataset_bytes(pairs)
    digest = sha256_bytes(payload)
    repo_root = Path(__file__).resolve().parents[2]
    manifest = {
        "dataset_id": SPEC["dataset_id"],
        "spec_experiment_id": SPEC["experiment_id"],
        "sha256": digest,
        "path": str(DATASET_PATH.relative_to(repo_root)),
        "n_pairs": 4,
        "n_independent_write_sequences": 16,
        "backbone": SPEC["backbone"],
        "write_scaffold": WRITE_SCAFFOLD,
        "query_templates": QUERY_TEMPLATES,
        "labels": {cid: CONCEPTS[cid]["label"] for cid in CONCEPTS},
        "descriptions": {cid: CONCEPTS[cid]["description"] for cid in CONCEPTS},
        "pairs": [
            {
                "pair_id": p["pair_id"],
                "e1": p["entities"]["e1"],
                "e2": p["entities"]["e2"],
                "c1": p["concepts"]["c1"]["concept_id"],
                "c2": p["concepts"]["c2"]["concept_id"],
                "label_c1": p["concepts"]["c1"]["label"],
                "label_c2": p["concepts"]["c2"]["label"],
            }
            for p in pairs
        ],
        "entities": [ent for p in PAIRS for ent in (p["e1"], p["e2"])],
        "mapping_definitions": {
            "A": {"e1": "c1", "e2": "c2"},
            "B": {"e1": "c2", "e2": "c1"},
        },
        "design_choices": DESIGN_CHOICES,
        "created_utc": datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"),
        "note": (
            "Frozen only after lexical audit and source-present calibration. "
            "Do not alter in place. Future changes require dataset_v4."
        ),
        **manifest_extra,
    }
    FROZEN_DIR.mkdir(parents=True, exist_ok=True)
    if DATASET_PATH.exists() and not force:
        existing = sha256_bytes(DATASET_PATH.read_bytes())
        if existing != digest:
            raise RuntimeError(
                f"refusing to overwrite frozen dataset_v3 (existing={existing}, new={digest})"
            )
        saved = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
        if saved.get("sha256") != digest:
            raise RuntimeError("frozen jsonl and manifest sha256 diverged")
        return saved
    DATASET_PATH.write_bytes(payload)
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest
