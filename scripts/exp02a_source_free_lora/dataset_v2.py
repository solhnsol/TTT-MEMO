"""Deterministic synthetic dataset_v2 for the Exp 02A K=8 formal pilot.

Does not modify frozen dataset_v1. Exact rows are copied from the frozen v1
file. v1 paraphrased semantic rows are kept as `semantic_inferential` and are
not in the main pilot. The new primary semantic condition is `semantic_direct`.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from random import Random
from typing import Any

from dataset_v1 import (
    DATASET_PATH as V1_PATH,
    META_PATH as V1_META_PATH,
    dataset_bytes,
    load_dataset_v1,
    sha256_bytes,
)
from protocol import SPEC

FROZEN_DIR = Path(__file__).resolve().parent / "frozen"
DATASET_PATH = FROZEN_DIR / "dataset_v2.jsonl"
META_PATH = FROZEN_DIR / "dataset_v2.meta.json"

DATASET_SEED = 20260918
N_PER_TYPE = 32
N_TEMPLATES = 4
N_CANDIDATES = 8
EXPECTED_V1_SHA256 = "431d14327bef0a48fa86da5d7148c982368d730ba8265730bf057561c96a3a32"

# Same 8 occupation labels as v1 smoke semantic golds, reused 4 times each.
SEMANTIC_DIRECT_OCCUPATIONS = [
    "electrician",
    "botanist",
    "cartographer",
    "veterinarian",
    "plumber",
    "astronomer",
    "baker",
    "translator",
]

# New nonce IDs; disjoint from v1 two-word names. Pattern matches the
# protocol example "Norvik-17".
SEMANTIC_DIRECT_ENTITIES = [
    "Norvik-17",
    "Helzun-04",
    "Qorlan-23",
    "Zembrex-11",
    "Bravok-08",
    "Nythex-31",
    "Tormel-19",
    "Sablun-02",
    "Keldor-27",
    "Orvix-14",
    "Peldan-06",
    "Weskor-22",
    "Jorvex-09",
    "Cindok-28",
    "Mavlen-03",
    "Yeldor-16",
    "Thorvik-21",
    "Lumex-07",
    "Dravok-30",
    "Rilnex-12",
    "Haskor-25",
    "Orrel-01",
    "Talvex-18",
    "Greshun-29",
    "Mirvok-10",
    "Solnex-26",
    "Bexlan-05",
    "Kivrel-24",
    "Dorvik-13",
    "Fenlor-20",
    "Qorlen-15",
    "Nalvik-32",
]

# Four write templates. None uses a/an, so the occupation string is unique and
# is the NTP payload. The recall question is a held-out paraphrase, not a
# cloze of these frames.
SEMANTIC_DIRECT_FACT_TEMPLATES = [
    "{entity}'s occupation is {payload}.",
    "{entity} holds the job title {payload}.",
    "The profession of {entity} is {payload}.",
    "{entity} is employed under the title {payload}.",
]

SEMANTIC_DIRECT_QUESTION = "What kind of work does {entity} do?"


def _require_unique(xs: list[str], *, kind: str) -> None:
    counts = Counter(xs)
    dups = [k for k, n in counts.items() if n > 1]
    if dups:
        raise ValueError(f"duplicate {kind}: {dups[:8]}")


def _derangement_distinct_answers(examples: list[dict[str, Any]], rng: Random) -> dict[str, str]:
    ids = [ex["example_id"] for ex in examples]
    answers = {ex["example_id"]: ex["answer"] for ex in examples}
    if len(ids) < 2:
        raise ValueError("need at least 2 ids")
    for _ in range(100000):
        partner = ids[:]
        rng.shuffle(partner)
        ok = all(a != b and answers[a] != answers[b] for a, b in zip(ids, partner))
        if ok:
            return dict(zip(ids, partner))
    raise RuntimeError("failed to sample a derangement with distinct answers")


def _closed_eight_way(examples: list[dict[str, Any]], labels: list[str], rng: Random) -> None:
    if len(labels) != N_CANDIDATES:
        raise ValueError(f"expected {N_CANDIDATES} labels, got {len(labels)}")
    first_src: dict[str, str] = {}
    for ex in examples:
        first_src.setdefault(ex["answer"], ex["example_id"])
    missing = [lab for lab in labels if lab not in first_src]
    if missing:
        raise ValueError(f"label missing from examples: {missing}")
    for i, ex in enumerate(examples):
        local = Random(rng.randint(0, 2**31 - 1) + i)
        pack = [
            {
                "text": lab,
                "source_example_id": first_src[lab],
                "is_gold": lab == ex["answer"],
            }
            for lab in labels
        ]
        order = list(range(len(pack)))
        local.shuffle(order)
        ex["candidates"] = [pack[j] for j in order]


def _copy_v1_exact(v1: list[dict[str, Any]]) -> list[dict[str, Any]]:
    exact = [json.loads(json.dumps(ex)) for ex in v1 if ex["info_type"] == "exact"]
    if len(exact) != 32:
        raise RuntimeError(f"v1 exact count {len(exact)}")
    return exact


def _copy_v1_semantic_inferential(v1: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for ex in v1:
        if ex["info_type"] != "semantic":
            continue
        row = json.loads(json.dumps(ex))
        row["info_type"] = "semantic_inferential"
        out.append(row)
    if len(out) != 32:
        raise RuntimeError(f"v1 semantic count {len(out)}")
    return out


def build_semantic_direct(*, seed: int = DATASET_SEED) -> list[dict[str, Any]]:
    if len(SEMANTIC_DIRECT_ENTITIES) != 32:
        raise ValueError(f"expected 32 entities, got {len(SEMANTIC_DIRECT_ENTITIES)}")
    _require_unique(SEMANTIC_DIRECT_ENTITIES, kind="semantic_direct entity")
    _require_unique(SEMANTIC_DIRECT_OCCUPATIONS, kind="occupation")
    examples: list[dict[str, Any]] = []
    for i in range(N_PER_TYPE):
        entity = SEMANTIC_DIRECT_ENTITIES[i]
        payload = SEMANTIC_DIRECT_OCCUPATIONS[i % 8]
        tmpl_id = i // 8
        fact = SEMANTIC_DIRECT_FACT_TEMPLATES[tmpl_id].format(entity=entity, payload=payload)
        examples.append(
            {
                "example_id": f"semantic_direct-{i:02d}",
                "info_type": "semantic_direct",
                "entity": entity,
                "answer": payload,
                "write_payload": payload,
                "write_template_id": tmpl_id,
                "fact_sentence": fact,
                "question": SEMANTIC_DIRECT_QUESTION.format(entity=entity),
                "smoke": False,
            }
        )
    partner = _derangement_distinct_answers(examples, Random(seed + 11))
    for ex in examples:
        ex["shuffle_partner_id"] = partner[ex["example_id"]]
    _closed_eight_way(examples, SEMANTIC_DIRECT_OCCUPATIONS, Random(seed + 12))
    return examples


def build_dataset_v2(*, seed: int = DATASET_SEED) -> list[dict[str, Any]]:
    v1 = load_dataset_v1(smoke=False)
    v1_raw = V1_PATH.read_bytes()
    v1_sha = sha256_bytes(v1_raw)
    if v1_sha != EXPECTED_V1_SHA256:
        raise RuntimeError(
            f"refusing to build v2 from a moved v1: file={v1_sha} expected={EXPECTED_V1_SHA256}"
        )
    exact = _copy_v1_exact(v1)
    inferential = _copy_v1_semantic_inferential(v1)
    direct = build_semantic_direct(seed=seed)
    examples = exact + direct + inferential
    validate_dataset_v2(examples, v1=v1)
    return examples


def validate_dataset_v2(examples: list[dict[str, Any]], *, v1: list[dict[str, Any]]) -> None:
    if len(examples) != 96:
        raise ValueError(f"expected 96 examples, got {len(examples)}")
    by_id = {ex["example_id"]: ex for ex in examples}
    if len(by_id) != 96:
        raise ValueError("example_id collision")

    exact = [ex for ex in examples if ex["info_type"] == "exact"]
    direct = [ex for ex in examples if ex["info_type"] == "semantic_direct"]
    infer = [ex for ex in examples if ex["info_type"] == "semantic_inferential"]
    if len(exact) != 32 or len(direct) != 32 or len(infer) != 32:
        raise ValueError(f"type counts exact={len(exact)} direct={len(direct)} infer={len(infer)}")

    v1_exact = [ex for ex in v1 if ex["info_type"] == "exact"]
    if exact != v1_exact:
        raise ValueError("v2 exact rows drifted from frozen v1")

    v1_sem = [ex for ex in v1 if ex["info_type"] == "semantic"]
    for old, new in zip(v1_sem, infer):
        expected = json.loads(json.dumps(old))
        expected["info_type"] = "semantic_inferential"
        if new != expected:
            raise ValueError(f"{new['example_id']}: inferential row drifted from v1 semantic")

    _require_unique([ex["entity"] for ex in examples], kind="entity")
    v1_entities = {ex["entity"] for ex in v1}
    overlap = v1_entities.intersection(ex["entity"] for ex in direct)
    if overlap:
        raise ValueError(f"semantic_direct entity overlaps v1: {sorted(overlap)[:4]}")

    tmpl_counts = Counter(ex["write_template_id"] for ex in direct)
    if set(tmpl_counts) != set(range(N_TEMPLATES)) or set(tmpl_counts.values()) != {8}:
        raise ValueError(f"unbalanced semantic_direct templates: {tmpl_counts}")
    occ_counts = Counter(ex["answer"] for ex in direct)
    if set(occ_counts) != set(SEMANTIC_DIRECT_OCCUPATIONS) or set(occ_counts.values()) != {4}:
        raise ValueError(f"unbalanced semantic_direct occupations: {occ_counts}")

    for ex in direct:
        if ex["answer"] != ex["write_payload"]:
            raise ValueError(f"{ex['example_id']}: gold must equal write payload")
        if ex["fact_sentence"].count(ex["write_payload"]) != 1:
            raise ValueError(f"{ex['example_id']}: payload not unique in fact")
        if ex["answer"] not in ex["fact_sentence"]:
            raise ValueError(f"{ex['example_id']}: gold missing from write")
        if "?" in ex["fact_sentence"] or ex["question"] in ex["fact_sentence"]:
            raise ValueError(f"{ex['example_id']}: question leaked into write")
        if ex["entity"] not in ex["fact_sentence"] or ex["entity"] not in ex["question"]:
            raise ValueError(f"{ex['example_id']}: entity missing")
        if "occupation is" in ex["question"] or "job title" in ex["question"]:
            raise ValueError(f"{ex['example_id']}: recall question copies a write frame")
        partner = by_id[ex["shuffle_partner_id"]]
        if partner["info_type"] != "semantic_direct":
            raise ValueError(f"{ex['example_id']}: shuffle type mismatch")
        if partner["example_id"] == ex["example_id"]:
            raise ValueError(f"{ex['example_id']}: shuffle fixed point")
        if partner["entity"] == ex["entity"] or partner["answer"] == ex["answer"]:
            raise ValueError(f"{ex['example_id']}: shuffle entity/value overlap")
        golds = [c for c in ex["candidates"] if c["is_gold"]]
        if len(golds) != 1 or golds[0]["text"] != ex["answer"]:
            raise ValueError(f"{ex['example_id']}: candidate gold mismatch")
        if len(ex["candidates"]) != N_CANDIDATES:
            raise ValueError(f"{ex['example_id']}: expected {N_CANDIDATES} candidates")
        cand_texts = [c["text"] for c in ex["candidates"]]
        _require_unique(cand_texts, kind=f"{ex['example_id']} candidate")
        if set(cand_texts) != set(SEMANTIC_DIRECT_OCCUPATIONS):
            raise ValueError(f"{ex['example_id']}: candidate set is not the frozen 8 occupations")


def freeze_dataset_v2(*, force: bool = False) -> dict[str, Any]:
    examples = build_dataset_v2()
    payload = dataset_bytes(examples)
    digest = sha256_bytes(payload)
    repo_root = Path(__file__).resolve().parents[2]
    v1_sha = json.loads(V1_META_PATH.read_text(encoding="utf-8"))["sha256"]
    meta = {
        "dataset_id": "exp02a_dataset_v2",
        "seed": DATASET_SEED,
        "n_examples": len(examples),
        "n_exact": 32,
        "n_semantic_direct": 32,
        "n_semantic_inferential": 32,
        "pilot_types": ["exact", "semantic_direct"],
        "held_out_from_pilot": ["semantic_inferential"],
        "source_v1_sha256": v1_sha,
        "expected_v1_sha256": EXPECTED_V1_SHA256,
        "sha256": digest,
        "path": str(DATASET_PATH.relative_to(repo_root)),
        "spec_experiment_id": SPEC["experiment_id"],
        "note": (
            "Frozen before v2 pilot scores. v1 jsonl was not rewritten. "
            "semantic_inferential is stored but not run in the K=8 pilot."
        ),
    }
    FROZEN_DIR.mkdir(parents=True, exist_ok=True)
    if DATASET_PATH.exists() and not force:
        existing = DATASET_PATH.read_bytes()
        existing_digest = sha256_bytes(existing)
        if existing_digest != digest:
            raise RuntimeError(
                f"refusing to overwrite frozen dataset_v2 (existing sha256={existing_digest}, "
                f"new sha256={digest}). Pass force=True only if you intend to invalidate v2."
            )
        saved_meta = json.loads(META_PATH.read_text(encoding="utf-8"))
        if saved_meta.get("sha256") != digest:
            raise RuntimeError("frozen v2 jsonl and meta sha256 diverged")
        return saved_meta
    DATASET_PATH.write_bytes(payload)
    META_PATH.write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return meta


def load_dataset_v2(*, pilot: bool = False) -> list[dict[str, Any]]:
    if not DATASET_PATH.exists():
        raise FileNotFoundError(f"frozen dataset missing: {DATASET_PATH}")
    raw = DATASET_PATH.read_bytes()
    meta = json.loads(META_PATH.read_text(encoding="utf-8"))
    digest = sha256_bytes(raw)
    if digest != meta["sha256"]:
        raise RuntimeError(f"dataset_v2 sha256 mismatch: file={digest} meta={meta['sha256']}")
    if meta.get("source_v1_sha256") != EXPECTED_V1_SHA256:
        raise RuntimeError("dataset_v2 meta does not pin the frozen v1 hash")
    examples = [json.loads(line) for line in raw.decode("utf-8").splitlines() if line.strip()]
    v1 = load_dataset_v1(smoke=False)
    validate_dataset_v2(examples, v1=v1)
    if pilot:
        examples = [ex for ex in examples if ex["info_type"] in ("exact", "semantic_direct")]
        if len(examples) != 64:
            raise RuntimeError(f"pilot subset size {len(examples)}")
    return examples


if __name__ == "__main__":
    meta = freeze_dataset_v2()
    print(json.dumps(meta, indent=2))
