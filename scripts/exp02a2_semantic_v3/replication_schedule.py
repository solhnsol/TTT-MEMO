"""Freeze the Exp 02A-2R entity-replication schedule before any model eval.

64 unique nonce entities, 32 replicates = 4 frozen concept pairs × 8 entity pairs.
Does not modify dataset_v3 occupations, descriptions, scaffold, or query.
"""

from __future__ import annotations

import hashlib
import json
import random
import re
from collections import Counter
from pathlib import Path
from typing import Any

from dataset import (
    DATASET_PATH,
    QUERY_TEMPLATES,
    prefix_text,
    sha256_bytes,
    target_text,
    write_sentence,
)
from memory_protocol import EXPECTED_DATASET_V3_SHA256
from protocol import CONCEPTS, SPEC

SCHEDULE_SEED = 202609182
N_REPLICATES_PER_PAIR = 8
N_PAIRS = 4
N_ENTITIES = 64

# Same Stem-NN surface as dataset_v3 (e.g. Norvik-17). New stems/numbers only.
_ONSETS = [
    "Bex", "Cyl", "Dov", "Frel", "Gan", "Hix", "Jol", "Kru",
    "Laz", "Miv", "Nox", "Pliq", "Quz", "Ruv", "Tix", "Weln",
    "Ysk", "Zor", "Glex", "Hurn", "Jiv", "Kred", "Lunx", "Morb",
    "Nesk", "Plix", "Ulx", "Coth", "Vyn", "Qel",
]
_CODAS = ["uq", "ez", "yn", "ib", "og", "ax"]

_V3_ENTITIES = {
    "Norvik-17", "Talven-03", "Keldar-29", "Vossin-41",
    "Merak-52", "Drelon-08", "Sovik-36", "Pelran-64",
}
_V3_NUMBERS = {17, 3, 29, 41, 52, 8, 36, 64}

_V2_FORBIDDEN_STEMS = {
    "Norvik", "Helzun", "Qorlan", "Zembrex", "Bravok", "Nythex", "Tormel", "Sablun",
    "Keldor", "Orvix", "Peldan", "Weskor", "Jorvex", "Cindok", "Mavlen", "Yeldor",
    "Thorvik", "Lumex", "Dravok", "Rilnex", "Haskor", "Orrel", "Talvex", "Greshun",
    "Mirvok", "Solnex", "Bexlan", "Kivrel", "Dorvik", "Fenlor", "Qorlen", "Nalvik",
    "Talven", "Keldar", "Vossin", "Merak", "Drelon", "Sovik", "Pelran",
}

# Common English / prior-experiment tokens that must not appear as stems.
_ENGLISH_DENY = {
    "the", "and", "for", "are", "but", "not", "you", "all", "can", "her", "was",
    "one", "our", "out", "day", "get", "has", "him", "his", "how", "its", "may",
    "new", "now", "old", "see", "two", "way", "who", "boy", "did", "let", "put",
    "say", "she", "too", "use", "job", "work", "name", "user", "data", "test",
    "true", "false", "none", "self", "this", "that", "with", "from", "have",
    "been", "will", "hotel", "solar", "panel", "human", "motor", "super", "extra",
    "ultra", "open", "close", "start", "stop", "model", "pilot", "cash", "vet",
}


def _n_tokens(tokenizer, text: str) -> int:
    ids = tokenizer(text, add_special_tokens=False)["input_ids"]
    return len(ids)


def _is_englishish(stem: str) -> bool:
    s = stem.lower()
    if s in _ENGLISH_DENY:
        return True
    for word in _ENGLISH_DENY:
        if len(word) >= 4 and word in s:
            return True
    for concept in CONCEPTS.values():
        lab = concept["label"].lower()
        if lab in s or lab.rstrip("s") in s:
            return True
    return False


def _candidate_stems() -> list[str]:
    stems = []
    for coda in _CODAS:
        for onset in _ONSETS:
            stem = onset + coda
            if stem in _V2_FORBIDDEN_STEMS:
                continue
            if _is_englishish(stem):
                continue
            if not re.fullmatch(r"[A-Z][a-z]{2,7}", stem):
                # onset+coda should stay in the Norvik-like class: one capital, rest lower.
                # Some onsets already mix case; normalize.
                stem = stem[0].upper() + stem[1:].lower()
            if _is_englishish(stem) or stem in _V2_FORBIDDEN_STEMS:
                continue
            stems.append(stem)
    if len(set(stems)) != len(stems):
        raise RuntimeError("duplicate stems in generator")
    if len(stems) < N_ENTITIES:
        raise RuntimeError(f"stem pool too small: {len(stems)}")
    return stems


def generate_entity_pool(tokenizer) -> list[dict[str, Any]]:
    rng = random.Random(SCHEDULE_SEED)
    stems = _candidate_stems()
    rng.shuffle(stems)
    numbers = [i for i in range(10, 100) if i not in _V3_NUMBERS]
    rng.shuffle(numbers)
    pool = []
    for stem, num in zip(stems, numbers):
        entity = f"{stem}-{num:02d}"
        if entity in _V3_ENTITIES:
            continue
        if any(ent.lower() in entity.lower() or entity.lower() in ent.lower() for ent in _V3_ENTITIES):
            continue
        n_tok = _n_tokens(tokenizer, entity)
        pool.append({"entity": entity, "stem": stem, "number": num, "n_tokens": n_tok})
        if len(pool) >= 96:
            break
    if len(pool) < N_ENTITIES:
        raise RuntimeError(f"entity pool too small: {len(pool)}")
    ents = [p["entity"] for p in pool]
    if len(set(ents)) != len(ents):
        raise RuntimeError("duplicate entities in pool")
    return pool


def pair_entities(pool: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rng = random.Random(SCHEDULE_SEED + 1)
    by_len: dict[int, list[dict[str, Any]]] = {}
    for row in pool:
        by_len.setdefault(row["n_tokens"], []).append(row)
    matched: list[dict[str, Any]] = []
    leftovers: list[dict[str, Any]] = []
    for n in sorted(by_len):
        items = list(by_len[n])
        rng.shuffle(items)
        while len(items) >= 2:
            a = items.pop()
            b = items.pop()
            matched.append(
                {
                    "e1": a["entity"],
                    "e2": b["entity"],
                    "e1_n_tokens": a["n_tokens"],
                    "e2_n_tokens": b["n_tokens"],
                    "token_lengths_match": True,
                }
            )
        leftovers.extend(items)
    if len(matched) < N_PAIRS * N_REPLICATES_PER_PAIR:
        rng.shuffle(leftovers)
        while len(matched) < N_PAIRS * N_REPLICATES_PER_PAIR and len(leftovers) >= 2:
            a = leftovers.pop()
            b = leftovers.pop()
            matched.append(
                {
                    "e1": a["entity"],
                    "e2": b["entity"],
                    "e1_n_tokens": a["n_tokens"],
                    "e2_n_tokens": b["n_tokens"],
                    "token_lengths_match": a["n_tokens"] == b["n_tokens"],
                }
            )
    if len(matched) < N_PAIRS * N_REPLICATES_PER_PAIR:
        raise RuntimeError(f"could not form 32 entity pairs, got {len(matched)}")
    chosen = matched[: N_PAIRS * N_REPLICATES_PER_PAIR]
    used = [x for p in chosen for x in (p["e1"], p["e2"])]
    if len(set(used)) != N_ENTITIES:
        raise RuntimeError(f"expected 64 unique entities, got {len(set(used))}")
    return chosen


def load_frozen_v3_pairs() -> list[dict[str, Any]]:
    digest = sha256_bytes(DATASET_PATH.read_bytes())
    if digest != EXPECTED_DATASET_V3_SHA256:
        raise RuntimeError(f"dataset_v3 sha256 mismatch: {digest}")
    return [
        json.loads(line)
        for line in DATASET_PATH.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def build_schedule(tokenizer) -> list[dict[str, Any]]:
    v3 = load_frozen_v3_pairs()
    if [p["pair_id"] for p in v3] != ["P1", "P2", "P3", "P4"]:
        raise RuntimeError("frozen v3 pair order drifted")
    pool = generate_entity_pool(tokenizer)
    entity_pairs = pair_entities(pool)
    rows = []
    k = 0
    for src in v3:
        for r in range(1, N_REPLICATES_PER_PAIR + 1):
            ent = entity_pairs[k]
            k += 1
            row = {
                "replicate_id": f"{src['pair_id']}_R{r:02d}",
                "source_pair_id": src["pair_id"],
                "replicate_index": r,
                "e1": ent["e1"],
                "e2": ent["e2"],
                "e1_n_tokens": ent["e1_n_tokens"],
                "e2_n_tokens": ent["e2_n_tokens"],
                "token_lengths_match": ent["token_lengths_match"],
                "c1": src["concepts"]["c1"]["concept_id"],
                "c2": src["concepts"]["c2"]["concept_id"],
                "label_c1": src["concepts"]["c1"]["label"],
                "label_c2": src["concepts"]["c2"]["label"],
                "description_c1": src["concepts"]["c1"]["description"],
                "description_c2": src["concepts"]["c2"]["description"],
                "write_scaffold": SPEC["write_scaffold"],
                "occupation_query": QUERY_TEMPLATES["occupation_query"],
                "dataset_v3_sha256": EXPECTED_DATASET_V3_SHA256,
            }
            rows.append(row)
    _validate_schedule(rows, v3)
    return rows


def _validate_schedule(rows: list[dict[str, Any]], v3: list[dict[str, Any]]) -> None:
    if len(rows) != 32:
        raise RuntimeError(f"expected 32 replicates, got {len(rows)}")
    ents = [e for r in rows for e in (r["e1"], r["e2"])]
    if len(set(ents)) != 64:
        raise RuntimeError(f"entity reuse: {len(set(ents))}")
    overlap = set(ents) & _V3_ENTITIES
    if overlap:
        raise RuntimeError(f"reused v3 entities: {overlap}")
    by_src = Counter(r["source_pair_id"] for r in rows)
    if dict(by_src) != {"P1": 8, "P2": 8, "P3": 8, "P4": 8}:
        raise RuntimeError(f"replicate allocation {dict(by_src)}")
    v3_by_id = {p["pair_id"]: p for p in v3}
    for r in rows:
        src = v3_by_id[r["source_pair_id"]]
        if r["label_c1"] != src["concepts"]["c1"]["label"]:
            raise RuntimeError("label_c1 drifted from frozen v3")
        if r["label_c2"] != src["concepts"]["c2"]["label"]:
            raise RuntimeError("label_c2 drifted from frozen v3")
        if r["description_c1"] != src["concepts"]["c1"]["description"]:
            raise RuntimeError("description_c1 drifted from frozen v3")
        if r["description_c2"] != src["concepts"]["c2"]["description"]:
            raise RuntimeError("description_c2 drifted from frozen v3")
        for lab in (r["label_c1"], r["label_c2"]):
            if lab.lower() in r["e1"].lower() or lab.lower() in r["e2"].lower():
                raise RuntimeError(f"occupation label in entity: {lab}")
        if r["occupation_query"] != QUERY_TEMPLATES["occupation_query"]:
            raise RuntimeError("query drifted")
        if r["write_scaffold"] != SPEC["write_scaffold"]:
            raise RuntimeError("scaffold drifted")


def schedule_bytes(rows: list[dict[str, Any]]) -> bytes:
    lines = [json.dumps(r, ensure_ascii=False, sort_keys=True) for r in rows]
    return ("\n".join(lines) + "\n").encode("utf-8")


def freeze_schedule(path: Path, tokenizer, *, force: bool = False) -> dict[str, Any]:
    rows = build_schedule(tokenizer)
    payload = schedule_bytes(rows)
    digest = hashlib.sha256(payload).hexdigest()
    path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path = path.with_name("entity_schedule_manifest.json")
    manifest = {
        "experiment_id": "exp02a2r_entity_replication",
        "n_replicates": 32,
        "n_entities": 64,
        "n_replicates_per_concept_pair": 8,
        "seed": SCHEDULE_SEED,
        "dataset_v3_sha256": EXPECTED_DATASET_V3_SHA256,
        "sha256": digest,
        "n_token_length_matched_pairs": sum(int(r["token_lengths_match"]) for r in rows),
        "note": (
            "Frozen before any model evaluation. 32 entity replicates are not "
            "32 independent semantic concepts."
        ),
    }
    if path.exists() and not force:
        existing = hashlib.sha256(path.read_bytes()).hexdigest()
        if existing != digest:
            raise RuntimeError(
                f"refusing to overwrite entity schedule (existing={existing}, new={digest})"
            )
        return json.loads(manifest_path.read_text(encoding="utf-8"))
    path.write_bytes(payload)
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def load_schedule(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def build_replicate_pairs(schedule: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Instantiate frozen v3 concepts onto the new entity schedule."""
    rows = []
    for spec in schedule:
        c1 = {"concept_id": spec["c1"], "label": spec["label_c1"], "description": spec["description_c1"]}
        c2 = {"concept_id": spec["c2"], "label": spec["label_c2"], "description": spec["description_c2"]}
        e1, e2 = spec["e1"], spec["e2"]
        pid = spec["replicate_id"]
        row = {
            "pair_id": pid,
            "source_pair_id": spec["source_pair_id"],
            "replicate_index": spec["replicate_index"],
            "info_type": "semantic_inferential_v3_entity_replication",
            "entities": {"e1": e1, "e2": e2},
            "concepts": {"c1": c1, "c2": c2},
            "mapping_A": {"e1": "c1", "e2": "c2"},
            "mapping_B": {"e1": "c2", "e2": "c1"},
            "query_templates": dict(QUERY_TEMPLATES),
            "token_lengths_match": spec["token_lengths_match"],
            "sequences": {
                "mapping_A": [
                    {
                        "sequence_id": f"{pid}_A_e1",
                        "entity_key": "e1",
                        "entity": e1,
                        "concept_key": "c1",
                        "label": c1["label"],
                        "description": c1["description"],
                        "write_sentence": write_sentence(e1, c1["description"]),
                        "prefix": prefix_text(e1),
                        "target": target_text(c1["description"]),
                    },
                    {
                        "sequence_id": f"{pid}_A_e2",
                        "entity_key": "e2",
                        "entity": e2,
                        "concept_key": "c2",
                        "label": c2["label"],
                        "description": c2["description"],
                        "write_sentence": write_sentence(e2, c2["description"]),
                        "prefix": prefix_text(e2),
                        "target": target_text(c2["description"]),
                    },
                ],
                "mapping_B": [
                    {
                        "sequence_id": f"{pid}_B_e1",
                        "entity_key": "e1",
                        "entity": e1,
                        "concept_key": "c2",
                        "label": c2["label"],
                        "description": c2["description"],
                        "write_sentence": write_sentence(e1, c2["description"]),
                        "prefix": prefix_text(e1),
                        "target": target_text(c2["description"]),
                    },
                    {
                        "sequence_id": f"{pid}_B_e2",
                        "entity_key": "e2",
                        "entity": e2,
                        "concept_key": "c1",
                        "label": c1["label"],
                        "description": c1["description"],
                        "write_sentence": write_sentence(e2, c1["description"]),
                        "prefix": prefix_text(e2),
                        "target": target_text(c1["description"]),
                    },
                ],
            },
            "candidates": [c1["label"], c2["label"]],
        }
        for mapping in ("mapping_A", "mapping_B"):
            for seq in row["sequences"][mapping]:
                if seq["label"] in seq["write_sentence"]:
                    raise RuntimeError(f"{seq['sequence_id']}: label leaked into write")
        rows.append(row)
    return rows
