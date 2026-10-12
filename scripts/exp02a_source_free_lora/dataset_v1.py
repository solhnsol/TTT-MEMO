"""Deterministic synthetic dataset_v1 for Exp 02A.

Frozen before any model scores are inspected. The JSONL in `frozen/` is the
source of truth. Regenerating requires an explicit --force flag.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from pathlib import Path
from random import Random
from typing import Any

from protocol import SPEC

FROZEN_DIR = Path(__file__).resolve().parent / "frozen"
DATASET_PATH = FROZEN_DIR / "dataset_v1.jsonl"
META_PATH = FROZEN_DIR / "dataset_v1.meta.json"

DATASET_SEED = 20260918
N_PER_TYPE = 32
N_TEMPLATES = 4
N_CANDIDATES = 8
SMOKE_N = 8

# 64 unique two-word nonce entities. Given names and surnames are unique.
ENTITIES = [
    "Varnox Quindle",
    "Helvar Nossim",
    "Quorin Belthas",
    "Zelkora Dinel",
    "Brammel Vossar",
    "Nyxara Pellun",
    "Torven Quillix",
    "Sablex Ornett",
    "Keldra Munnis",
    "Orvath Skelun",
    "Pellion Draeve",
    "Westrin Calorn",
    "Jorvika Thalen",
    "Cindrel Voleth",
    "Mavrik Soneth",
    "Yeldrin Paxor",
    "Thorsen Qadrel",
    "Lumera Skiffin",
    "Draven Olmeth",
    "Rilven Torask",
    "Haskel Briune",
    "Orridy Nalvex",
    "Talxen Hrowek",
    "Greshik Vandor",
    "Mireth Qolune",
    "Solvane Kirrex",
    "Bexlar Ondrel",
    "Kivora Thessun",
    "Dorlen Yavrik",
    "Fenqor Ulmeth",
    "Qorvyn Iskara",
    "Nalvem Pravel",
    "Veldrix Praxon",
    "Praxel Xandrel",
    "Xandora Ulmera",
    "Shorvek Tannix",
    "Elvorn Rhodan",
    "Calveth Nymora",
    "Jastel Ophren",
    "Wydrel Kestra",
    "Lomvik Arveth",
    "Sindrel Yuvora",
    "Thessik Quenlar",
    "Vaskira Brelun",
    "Haldrix Sorvane",
    "Ithrel Noxara",
    "Pelvyn Zandrik",
    "Orlune Fendrel",
    "Qivara Heldrin",
    "Norveth Caldix",
    "Bralune Veskar",
    "Tevrin Olqar",
    "Sarnel Quivor",
    "Uldrex Panvir",
    "Moxara Drelun",
    "Yendrel Kavoss",
    "Rovika Telsun",
    "Caldren Uxora",
    "Pelnor Vaskel",
    "Ixora Darnel",
    "Vorthen Liskar",
    "Zelune Marvik",
    "Osveth Quinlar",
    "Relvyn Tashor",
]

# 96 unique common English tokens, grouped into 32 three-word passphrases.
PASSPHRASE_WORDS = [
    "cedar", "violet", "harbor",
    "maple", "amber", "quartz",
    "willow", "ember", "granite",
    "pebble", "lantern", "meadow",
    "copper", "ivy", "canyon",
    "frost", "clover", "marble",
    "pine", "coral", "barley",
    "silver", "moss", "dune",
    "orchard", "flint", "raven",
    "sage", "thistle", "pearl",
    "birch", "ochre", "fennel",
    "cobble", "lagoon", "hazel",
    "walnut", "saffron", "inlet",
    "poplar", "topaz", "ridge",
    "alder", "crimson", "brook",
    "hickory", "indigo", "bluff",
    "spruce", "linen", "marsh",
    "laurel", "scarlet", "cove",
    "aspen", "bronze", "glen",
    "olive", "cobalt", "reef",
    "beech", "ivory", "knoll",
    "cypress", "azure", "fjord",
    "holly", "russet", "delta",
    "juniper", "teal", "islet",
    "magnolia", "maroon", "atoll",
    "sycamore", "khaki", "sound",
    "hemlock", "umber", "spit",
    "redwood", "mauve", "firth",
    "linden", "tawny", "bight",
    "sequoia", "golden", "loch",
    "yew", "onyx", "fir",
    "oak", "ruby", "bay",
]

SEMANTIC_ITEMS = [
    {"answer": "electrician", "paraphrase": "repairing and installing electrical wiring in buildings"},
    {"answer": "botanist", "paraphrase": "classifying wild plants and recording how they grow outdoors"},
    {"answer": "cartographer", "paraphrase": "drawing detailed maps of coasts and mountain ranges"},
    {"answer": "veterinarian", "paraphrase": "treating injuries and illnesses in domestic animals"},
    {"answer": "plumber", "paraphrase": "fixing pipes and clearing blocked water drains"},
    {"answer": "astronomer", "paraphrase": "observing stars and measuring the motion of planets"},
    {"answer": "baker", "paraphrase": "preparing bread and pastries in a commercial oven"},
    {"answer": "translator", "paraphrase": "converting legal documents from one language into another"},
    {"answer": "architect", "paraphrase": "designing the layout and structure of new buildings"},
    {"answer": "geologist", "paraphrase": "collecting rock samples to determine the age of landforms"},
    {"answer": "pharmacist", "paraphrase": "preparing prescribed medicines and explaining dosages"},
    {"answer": "carpenter", "paraphrase": "cutting lumber and assembling wooden building frames"},
    {"answer": "journalist", "paraphrase": "interviewing witnesses and writing reports for a newspaper"},
    {"answer": "dentist", "paraphrase": "examining teeth and filling cavities in a clinic"},
    {"answer": "tailor", "paraphrase": "cutting fabric and sewing custom-fitted clothing"},
    {"answer": "archaeologist", "paraphrase": "excavating ruins and cataloguing ancient artifacts"},
    {"answer": "welder", "paraphrase": "fusing metal parts with a high-temperature torch"},
    {"answer": "midwife", "paraphrase": "assisting women during childbirth in a clinic"},
    {"answer": "locksmith", "paraphrase": "cutting keys and opening locked doors without damage"},
    {"answer": "optometrist", "paraphrase": "measuring eyesight and fitting corrective lenses"},
    {"answer": "mason", "paraphrase": "laying bricks and stone to form walls"},
    {"answer": "blacksmith", "paraphrase": "heating iron and hammering it into tools"},
    {"answer": "librarian", "paraphrase": "organizing bound volumes and helping visitors find reference materials"},
    {"answer": "chef", "paraphrase": "planning menus and cooking meals in a restaurant kitchen"},
    {"answer": "firefighter", "paraphrase": "extinguishing burning buildings and rescuing trapped occupants"},
    {"answer": "pilot", "paraphrase": "flying passenger aircraft between distant cities"},
    {"answer": "bartender", "paraphrase": "mixing drinks and serving them at a counter"},
    {"answer": "florist", "paraphrase": "arranging cut flowers into decorative bouquets"},
    {"answer": "surveyor", "paraphrase": "measuring land boundaries with a tripod-mounted instrument"},
    {"answer": "composer", "paraphrase": "writing original music for orchestras and films"},
    {"answer": "sculptor", "paraphrase": "shaping stone and clay into statues"},
    {"answer": "glazier", "paraphrase": "cutting glass panes and fitting them into window frames"},
]

EXACT_FACT_TEMPLATES = [
    "The access passphrase for {entity} is {payload}.",
    "{entity} is assigned the secret phrase {payload}.",
    "Record that {entity}'s passphrase equals {payload}.",
    "The three-word code belonging to {entity} is {payload}.",
]

SEMANTIC_FACT_TEMPLATES = [
    "{entity} earns a living by {payload}.",
    "Day after day, {entity} works at {payload}.",
    "The daily work of {entity} consists of {payload}.",
    "{entity}'s occupation involves {payload}.",
]

EXACT_QUESTION = "What is the passphrase for {entity}?"
SEMANTIC_QUESTION = "What is {entity}'s occupation?"


def _require_unique(xs: list[str], *, kind: str) -> None:
    counts = Counter(xs)
    dups = [k for k, n in counts.items() if n > 1]
    if dups:
        raise ValueError(f"duplicate {kind}: {dups[:8]}")


def _derangement(ids: list[str], rng: Random) -> dict[str, str]:
    if len(ids) < 2:
        raise ValueError("need at least 2 ids for a derangement")
    for _ in range(10000):
        partner = ids[:]
        rng.shuffle(partner)
        if all(a != b for a, b in zip(ids, partner)):
            return dict(zip(ids, partner))
    raise RuntimeError("failed to sample a derangement")


def _candidate_pack(examples: list[dict[str, Any]], rng: Random) -> None:
    values = [(ex["example_id"], ex["answer"]) for ex in examples]
    for i, ex in enumerate(examples):
        others = [pair for pair in values if pair[0] != ex["example_id"]]
        local = Random(rng.randint(0, 2**31 - 1) + i)
        decoys = local.sample(others, N_CANDIDATES - 1)
        pack = [{"text": ex["answer"], "source_example_id": ex["example_id"], "is_gold": True}]
        pack.extend(
            {"text": text, "source_example_id": src, "is_gold": False} for src, text in decoys
        )
        order = list(range(len(pack)))
        local.shuffle(order)
        ex["candidates"] = [pack[j] for j in order]


def _whole_word_in(haystack: str, needle: str) -> bool:
    return re.search(rf"\b{re.escape(needle.lower())}\b", haystack.lower()) is not None


def build_dataset_v1(*, seed: int = DATASET_SEED) -> list[dict[str, Any]]:
    if len(ENTITIES) != 64:
        raise ValueError(f"expected 64 entities, got {len(ENTITIES)}")
    if len(PASSPHRASE_WORDS) != 96:
        raise ValueError(f"expected 96 passphrase words, got {len(PASSPHRASE_WORDS)}")
    if len(SEMANTIC_ITEMS) != 32:
        raise ValueError(f"expected 32 semantic items, got {len(SEMANTIC_ITEMS)}")

    _require_unique(ENTITIES, kind="entity")
    _require_unique([n.split()[0] for n in ENTITIES], kind="given name")
    _require_unique([n.split()[1] for n in ENTITIES], kind="surname")
    _require_unique(PASSPHRASE_WORDS, kind="passphrase word")
    _require_unique([x["answer"] for x in SEMANTIC_ITEMS], kind="occupation")
    _require_unique([x["paraphrase"] for x in SEMANTIC_ITEMS], kind="paraphrase")

    exact_entities = ENTITIES[:32]
    semantic_entities = ENTITIES[32:]
    passphrases = [" ".join(PASSPHRASE_WORDS[i * 3 : (i + 1) * 3]) for i in range(32)]
    _require_unique(passphrases, kind="passphrase")

    examples: list[dict[str, Any]] = []
    for i in range(N_PER_TYPE):
        payload = passphrases[i]
        tmpl_id = i % N_TEMPLATES
        entity = exact_entities[i]
        fact = EXACT_FACT_TEMPLATES[tmpl_id].format(entity=entity, payload=payload)
        examples.append(
            {
                "example_id": f"exact-{i:02d}",
                "info_type": "exact",
                "entity": entity,
                "answer": payload,
                "write_payload": payload,
                "write_template_id": tmpl_id,
                "fact_sentence": fact,
                "question": EXACT_QUESTION.format(entity=entity),
                "smoke": i < SMOKE_N,
            }
        )
    for i in range(N_PER_TYPE):
        item = SEMANTIC_ITEMS[i]
        tmpl_id = i % N_TEMPLATES
        entity = semantic_entities[i]
        payload = item["paraphrase"]
        fact = SEMANTIC_FACT_TEMPLATES[tmpl_id].format(entity=entity, payload=payload)
        examples.append(
            {
                "example_id": f"semantic-{i:02d}",
                "info_type": "semantic",
                "entity": entity,
                "answer": item["answer"],
                "write_payload": payload,
                "write_template_id": tmpl_id,
                "fact_sentence": fact,
                "question": SEMANTIC_QUESTION.format(entity=entity),
                "smoke": i < SMOKE_N,
            }
        )

    exact = [ex for ex in examples if ex["info_type"] == "exact"]
    semantic = [ex for ex in examples if ex["info_type"] == "semantic"]
    exact_map = _derangement([ex["example_id"] for ex in exact], Random(seed + 1))
    semantic_map = _derangement([ex["example_id"] for ex in semantic], Random(seed + 2))
    partner = {**exact_map, **semantic_map}
    for ex in examples:
        ex["shuffle_partner_id"] = partner[ex["example_id"]]

    _candidate_pack(exact, Random(seed + 3))
    _candidate_pack(semantic, Random(seed + 4))
    validate_dataset_v1(examples)
    return examples


def validate_dataset_v1(examples: list[dict[str, Any]]) -> None:
    if len(examples) != 64:
        raise ValueError(f"expected 64 examples, got {len(examples)}")
    by_id = {ex["example_id"]: ex for ex in examples}
    if len(by_id) != 64:
        raise ValueError("example_id collision")

    _require_unique([ex["entity"] for ex in examples], kind="entity")
    for info_type in ("exact", "semantic"):
        subset = [ex for ex in examples if ex["info_type"] == info_type]
        if len(subset) != N_PER_TYPE:
            raise ValueError(f"{info_type} count {len(subset)}")
        _require_unique([ex["answer"] for ex in subset], kind=f"{info_type} answer")
        _require_unique([ex["write_payload"] for ex in subset], kind=f"{info_type} payload")
        tmpl_counts = Counter(ex["write_template_id"] for ex in subset)
        if set(tmpl_counts) != set(range(N_TEMPLATES)) or set(tmpl_counts.values()) != {8}:
            raise ValueError(f"unbalanced templates for {info_type}: {tmpl_counts}")
        smoke = [ex for ex in subset if ex["smoke"]]
        if len(smoke) != SMOKE_N:
            raise ValueError(f"smoke {info_type} count {len(smoke)}")

    occupations = [ex["answer"] for ex in examples if ex["info_type"] == "semantic"]
    for ex in examples:
        if "?" in ex["fact_sentence"]:
            raise ValueError(f"{ex['example_id']}: fact sentence contains '?'")
        if ex["question"] in ex["fact_sentence"]:
            raise ValueError(f"{ex['example_id']}: question leaked into fact")
        if ex["fact_sentence"].count(ex["write_payload"]) != 1:
            raise ValueError(f"{ex['example_id']}: payload is not unique in fact")
        if ex["entity"] not in ex["fact_sentence"] or ex["entity"] not in ex["question"]:
            raise ValueError(f"{ex['example_id']}: entity missing")
        if ex["info_type"] == "exact" and ex["answer"] != ex["write_payload"]:
            raise ValueError(f"{ex['example_id']}: exact answer must equal payload")
        if ex["info_type"] == "semantic":
            if _whole_word_in(ex["fact_sentence"], ex["answer"]):
                raise ValueError(
                    f"{ex['example_id']}: occupation '{ex['answer']}' occurs in write sentence"
                )
            for occ in occupations:
                if occ != ex["answer"] and _whole_word_in(ex["fact_sentence"], occ):
                    raise ValueError(
                        f"{ex['example_id']}: other occupation '{occ}' occurs in write sentence"
                    )
        partner = by_id[ex["shuffle_partner_id"]]
        if partner["example_id"] == ex["example_id"]:
            raise ValueError(f"{ex['example_id']}: shuffle fixed point")
        if partner["info_type"] != ex["info_type"]:
            raise ValueError(f"{ex['example_id']}: shuffle type mismatch")
        if partner["entity"] == ex["entity"] or partner["answer"] == ex["answer"]:
            raise ValueError(f"{ex['example_id']}: shuffle entity/value overlap")
        if partner["write_payload"] == ex["write_payload"]:
            raise ValueError(f"{ex['example_id']}: shuffle payload overlap")
        golds = [c for c in ex["candidates"] if c["is_gold"]]
        if len(golds) != 1 or golds[0]["text"] != ex["answer"]:
            raise ValueError(f"{ex['example_id']}: candidate gold mismatch")
        if len(ex["candidates"]) != N_CANDIDATES:
            raise ValueError(f"{ex['example_id']}: expected {N_CANDIDATES} candidates")
        cand_texts = [c["text"] for c in ex["candidates"]]
        _require_unique(cand_texts, kind=f"{ex['example_id']} candidate")


def dataset_bytes(examples: list[dict[str, Any]]) -> bytes:
    lines = [json.dumps(ex, ensure_ascii=False, sort_keys=True) for ex in examples]
    return ("\n".join(lines) + "\n").encode("utf-8")


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def freeze_dataset_v1(*, force: bool = False) -> dict[str, Any]:
    examples = build_dataset_v1()
    payload = dataset_bytes(examples)
    digest = sha256_bytes(payload)
    repo_root = Path(__file__).resolve().parents[2]
    meta = {
        "dataset_id": "exp02a_dataset_v1",
        "seed": DATASET_SEED,
        "n_examples": len(examples),
        "n_exact": 32,
        "n_semantic": 32,
        "smoke_n_per_type": SMOKE_N,
        "sha256": digest,
        "path": str(DATASET_PATH.relative_to(repo_root)),
        "spec_experiment_id": SPEC["experiment_id"],
        "note": "Frozen before model scores. Do not regenerate after looking at results.",
    }
    FROZEN_DIR.mkdir(parents=True, exist_ok=True)
    if DATASET_PATH.exists() and not force:
        existing = DATASET_PATH.read_bytes()
        existing_digest = sha256_bytes(existing)
        if existing_digest != digest:
            raise RuntimeError(
                f"refusing to overwrite frozen dataset (existing sha256={existing_digest}, "
                f"new sha256={digest}). Pass force=True only if you intend to invalidate v1."
            )
        saved_meta = json.loads(META_PATH.read_text(encoding="utf-8"))
        if saved_meta.get("sha256") != digest:
            raise RuntimeError("frozen jsonl and meta sha256 diverged")
        return saved_meta
    DATASET_PATH.write_bytes(payload)
    META_PATH.write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return meta


def load_dataset_v1(*, smoke: bool = False) -> list[dict[str, Any]]:
    if not DATASET_PATH.exists():
        raise FileNotFoundError(f"frozen dataset missing: {DATASET_PATH}")
    raw = DATASET_PATH.read_bytes()
    meta = json.loads(META_PATH.read_text(encoding="utf-8"))
    digest = sha256_bytes(raw)
    if digest != meta["sha256"]:
        raise RuntimeError(f"dataset sha256 mismatch: file={digest} meta={meta['sha256']}")
    examples = [json.loads(line) for line in raw.decode("utf-8").splitlines() if line.strip()]
    validate_dataset_v1(examples)
    if smoke:
        examples = [ex for ex in examples if ex["smoke"]]
        if len(examples) != 16:
            raise RuntimeError(f"smoke subset size {len(examples)}")
    return examples


if __name__ == "__main__":
    meta = freeze_dataset_v1()
    print(json.dumps(meta, indent=2))
