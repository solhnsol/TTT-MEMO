"""Deterministic binding-pair dataset for Exp 02A-1.

Frozen before any model scores are inspected. Mapping A and Mapping B share
the same entity, value, and relation token inventory; only the entity-value
association (and therefore the slot occupied by each value) differs.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

from protocol import DESIGN_CHOICES, SPEC

FROZEN_DIR = Path(__file__).resolve().parent / "frozen"
DATASET_PATH = FROZEN_DIR / "dataset_pairs.jsonl"
MANIFEST_PATH = FROZEN_DIR / "dataset_manifest.json"

DATASET_SEED = int(DESIGN_CHOICES["dataset_seed"])
N_PER_TYPE = 32
SMOKE_N = 4

# Exp 02A v2 semantic_direct stems. New IDs must not reuse these strings.
_V2_FORBIDDEN_STEMS = {
    "Norvik", "Helzun", "Qorlan", "Zembrex", "Bravok", "Nythex", "Tormel", "Sablun",
    "Keldor", "Orvix", "Peldan", "Weskor", "Jorvex", "Cindok", "Mavlen", "Yeldor",
    "Thorvik", "Lumex", "Dravok", "Rilnex", "Haskor", "Orrel", "Talvex", "Greshun",
    "Mirvok", "Solnex", "Bexlan", "Kivrel", "Dorvik", "Fenlor", "Qorlen", "Nalvik",
    "Talven",
}

_ONSETS = [
    "Ar", "Bel", "Cal", "Dre", "Elv", "Fen", "Gor", "Hal",
    "Isk", "Jas", "Kav", "Lom", "Mox", "Nym", "Oph", "Pan",
    "Que", "Rho", "Sin", "Tel", "Uxo", "Vas", "Wyd", "Xan",
    "Yuv", "Zan", "Aes", "Bri", "Cor", "Dal", "Emv", "Fae",
]
_CODAS = ["eth", "mar", "dor", "vik"]

# 96 words from Exp 02A v1 passphrases, reused as a common-vocab pool.
V1_PASSPHRASE_WORDS = [
    "cedar", "violet", "harbor", "maple", "amber", "quartz",
    "willow", "ember", "granite", "pebble", "lantern", "meadow",
    "copper", "ivy", "canyon", "frost", "clover", "marble",
    "pine", "coral", "barley", "silver", "moss", "dune",
    "orchard", "flint", "raven", "sage", "thistle", "pearl",
    "birch", "ochre", "fennel", "cobble", "lagoon", "hazel",
    "walnut", "saffron", "inlet", "poplar", "topaz", "ridge",
    "alder", "crimson", "brook", "hickory", "indigo", "bluff",
    "spruce", "linen", "marsh", "laurel", "scarlet", "cove",
    "aspen", "bronze", "glen", "olive", "cobalt", "reef",
    "beech", "ivory", "knoll", "cypress", "azure", "fjord",
    "holly", "russet", "delta", "juniper", "teal", "islet",
    "magnolia", "maroon", "atoll", "sycamore", "khaki", "sound",
    "hemlock", "umber", "spit", "redwood", "mauve", "firth",
    "linden", "tawny", "bight", "sequoia", "golden", "loch",
    "yew", "onyx", "fir", "oak", "ruby", "bay",
]

# 96 additional common nouns, disjoint from v1. Prefer short words so
# within-pair tokenizer lengths can be matched.
EXTRA_PASSPHRASE_WORDS = [
    "apple", "river", "stone", "cloud", "forest", "needle",
    "garden", "pepper", "lemon", "ginger", "nutmeg", "spice",
    "vanilla", "orchid", "daisy", "tulip", "rose", "lily",
    "poppy", "iris", "fern", "honey", "butter", "sugar",
    "anvil", "barrel", "candle", "dagger", "eagle", "falcon",
    "garlic", "helmet", "ink", "jumper", "kettle", "ladder",
    "mirror", "nectar", "orange", "pillow", "quiver", "ribbon",
    "saddle", "table", "velvet", "window", "yellow", "anchor",
    "bridge", "castle", "desert", "engine", "fountain", "glacier",
    "hammer", "island", "jacket", "kernel", "lizard", "magnet",
    "nickel", "planet", "rocket", "tunnel", "valley", "xenon",
    "yogurt", "zephyr", "apricot", "blossom", "cactus", "feather",
    "gravel", "harvest", "iceberg", "jasmine", "kindling", "prairie",
    "quince", "raindrop", "starfish", "vineyard", "pecan", "spark",
    "circle", "dragon", "echo", "flame", "grape", "horizon",
    "cream", "jewel", "kite", "lotus", "pasture", "night",
]

# 32 validated occupation labels from Exp 02A dataset_v1 SEMANTIC_ITEMS.
OCCUPATIONS = [
    "electrician", "botanist", "cartographer", "veterinarian",
    "plumber", "astronomer", "baker", "translator",
    "architect", "geologist", "pharmacist", "carpenter",
    "journalist", "dentist", "tailor", "archaeologist",
    "welder", "midwife", "locksmith", "optometrist",
    "mason", "blacksmith", "librarian", "chef",
    "firefighter", "pilot", "bartender", "florist",
    "surveyor", "composer", "sculptor", "glazier",
]

EXACT_TEMPLATE = "The passphrase assigned to {entity} is {payload}."
SEMANTIC_TEMPLATE = "{entity}'s occupation is {payload}."

EXACT_QUESTIONS = {
    "lexical_near": "What is the passphrase assigned to {entity}?",
    "paraphrase": "Which secret passphrase belongs to {entity}?",
}
SEMANTIC_QUESTIONS = {
    "lexical_near": "What is {entity}'s occupation?",
    "paraphrase": "What kind of work does {entity} do?",
}


def _require_unique(xs: list[str], *, kind: str) -> None:
    counts = Counter(xs)
    dups = [k for k, n in counts.items() if n > 1]
    if dups:
        raise ValueError(f"duplicate {kind}: {dups[:8]}")


def dataset_bytes(examples: list[dict[str, Any]]) -> bytes:
    lines = [json.dumps(ex, ensure_ascii=False, sort_keys=True) for ex in examples]
    return ("\n".join(lines) + "\n").encode("utf-8")


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def load_tokenizer():
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(
        SPEC["backbone"]["model_id"],
        revision=SPEC["backbone"]["revision_sha"],
        use_fast=True,
    )
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    return tok


def n_tokens(tokenizer, text: str) -> int:
    ids = tokenizer(text, add_special_tokens=False)["input_ids"]
    if not ids:
        raise RuntimeError(f"empty tokenization: {text!r}")
    return len(ids)


def token_id_multiset(tokenizer, text: str) -> Counter:
    ids = tokenizer(text, add_special_tokens=False)["input_ids"]
    return Counter(int(i) for i in ids)


def _make_entities() -> list[str]:
    stems: list[str] = []
    for coda in _CODAS:
        for onset in _ONSETS:
            stem = onset + coda
            stems.append(stem)
    _require_unique(stems, kind="entity stem")
    overlap = _V2_FORBIDDEN_STEMS.intersection(stems)
    if overlap:
        raise ValueError(f"entity stem overlaps Exp 02A v2: {sorted(overlap)}")
    if len(stems) != 128:
        raise RuntimeError(f"expected 128 stems, got {len(stems)}")
    return [f"{stem}-{i:03d}" for i, stem in enumerate(stems, start=1)]


# Extra common nouns used if the v1+extra pool cannot form equal-length pairs.
BACKUP_PASSPHRASE_WORDS = [
    "cat", "dog", "bird", "fish", "horse", "sheep",
    "goat", "wolf", "bear", "lion", "tiger", "mouse",
    "whale", "shark", "trout", "crab", "seal", "deer",
    "fox", "hare", "duck", "goose", "swan", "crow",
    "owl", "bee", "ant", "frog", "toad", "snake",
    "rabbit", "squirrel", "otter", "badger", "camel", "zebra",
    "bread", "cheese", "honey", "sugar", "water", "flour",
    "salt", "rice", "bean", "corn", "wheat", "oats",
    "chair", "desk", "door", "floor", "roof", "wall",
    "road", "path", "hill", "lake", "pond", "stream",
    "wind", "rain", "snow", "storm", "sun", "moon",
    "star", "sky", "field", "farm", "barn", "gate",
    "book", "page", "pen", "ink", "paper", "note",
    "ship", "boat", "sail", "port", "dock", "pier",
    "drum", "harp", "bell", "horn", "pipe", "flute",
    "ring", "coin", "gold", "iron", "steel", "clay",
    "sand", "dust", "smoke", "flame", "spark", "ash",
    "leaf", "root", "seed", "branch", "trunk", "bark",
    "silk", "wool", "cotton", "linen", "cloth", "thread",
]


def _make_matched_phrase_pairs(tokenizer) -> list[tuple[str, str]]:
    from itertools import combinations

    overlap = set(V1_PASSPHRASE_WORDS).intersection(EXTRA_PASSPHRASE_WORDS)
    if overlap:
        raise ValueError(f"passphrase word overlap: {sorted(overlap)[:8]}")
    pool = []
    seen = set()
    for w in V1_PASSPHRASE_WORDS + EXTRA_PASSPHRASE_WORDS + BACKUP_PASSPHRASE_WORDS:
        if w in seen:
            continue
        seen.add(w)
        pool.append(w)
    pairs: list[tuple[str, str]] = []
    used_phrases: set[str] = set()
    i = 0
    while len(pairs) < 32:
        if i + 6 > len(pool):
            raise RuntimeError(
                f"exhausted word pool while building passphrases; have {len(pairs)} pairs"
            )
        six = pool[i : i + 6]
        found = None
        for comb in combinations(range(6), 3):
            a_idx = list(comb)
            b_idx = [j for j in range(6) if j not in a_idx]
            a = " ".join(six[j] for j in a_idx)
            b = " ".join(six[j] for j in b_idx)
            if a == b or a in used_phrases or b in used_phrases:
                continue
            if n_tokens(tokenizer, a) == n_tokens(tokenizer, b):
                found = (a, b)
                break
        if found is None:
            i += 1
            continue
        a, b = found
        used_phrases.add(a)
        used_phrases.add(b)
        pairs.append((a, b))
        i += 6
    return pairs


def _pair_equal_length(
    items: list[str],
    lengths: list[int],
    *,
    kind: str,
    allow_unmatched: bool = False,
) -> list[tuple[str, str]]:
    if len(items) != len(lengths) or len(items) % 2 != 0:
        raise ValueError(f"{kind}: need even nonempty lists")
    by_len: dict[int, list[str]] = {}
    length_of = dict(zip(items, lengths))
    for item, n in zip(items, lengths):
        by_len.setdefault(n, []).append(item)
    pairs: list[tuple[str, str]] = []
    stranded = []
    for n in sorted(by_len):
        group = by_len[n]
        while len(group) >= 2:
            pairs.append((group.pop(0), group.pop(0)))
        if group:
            stranded.extend(group)
    if stranded:
        if not allow_unmatched or len(stranded) % 2 != 0:
            raise RuntimeError(
                f"cannot pair {kind} with equal tokenizer length; "
                f"stranded={[(w, length_of[w]) for w in stranded]}"
            )
        while stranded:
            pairs.append((stranded.pop(0), stranded.pop(0)))
    if 2 * len(pairs) != len(items):
        raise RuntimeError(f"{kind}: pairing lost items")
    return pairs


def _questions(info_type: str, entity: str) -> dict[str, str]:
    table = EXACT_QUESTIONS if info_type == "exact" else SEMANTIC_QUESTIONS
    return {style: tmpl.format(entity=entity) for style, tmpl in table.items()}


def _sentences(template: str, order: list[tuple[str, str]]) -> list[str]:
    return [template.format(entity=ent, payload=val) for ent, val in order]


def _build_pair(
    *,
    pair_id: str,
    info_type: str,
    e1: str,
    e2: str,
    v1: str,
    v2: str,
    smoke: bool,
    write_e1_first: bool,
    tokenizer,
) -> dict[str, Any]:
    template = EXACT_TEMPLATE if info_type == "exact" else SEMANTIC_TEMPLATE
    if write_e1_first:
        order_keys = ["e1", "e2"]
        order_A = [(e1, v1), (e2, v2)]
        order_B = [(e1, v2), (e2, v1)]
    else:
        order_keys = ["e2", "e1"]
        order_A = [(e2, v2), (e1, v1)]
        order_B = [(e2, v1), (e1, v2)]
    sentences_A = _sentences(template, order_A)
    sentences_B = _sentences(template, order_B)
    block_A = "\n".join(sentences_A)
    block_B = "\n".join(sentences_B)
    ms_A = token_id_multiset(tokenizer, block_A)
    ms_B = token_id_multiset(tokenizer, block_B)
    if ms_A != ms_B:
        raise RuntimeError(
            f"{pair_id}: Mapping A/B token multisets differ. "
            f"A_only={ms_A - ms_B} B_only={ms_B - ms_A}"
        )
    questions = {"e1": _questions(info_type, e1), "e2": _questions(info_type, e2)}
    pair = {
        "pair_id": pair_id,
        "info_type": info_type,
        "smoke": smoke,
        "e1": e1,
        "e2": e2,
        "v1": v1,
        "v2": v2,
        "write_order": order_keys,
        "write_template": template,
        "sentences_A": sentences_A,
        "sentences_B": sentences_B,
        "write_block_A": block_A,
        "write_block_B": block_B,
        "payloads_in_write_order_A": [val for _ent, val in order_A],
        "payloads_in_write_order_B": [val for _ent, val in order_B],
        "entities_in_write_order": [ent for ent, _val in order_A],
        "questions": questions,
        "candidates": [v1, v2],
        "mapping_A": {"e1": v1, "e2": v2},
        "mapping_B": {"e1": v2, "e2": v1},
        "tokenizer": {
            "v1_n_tokens": n_tokens(tokenizer, v1),
            "v2_n_tokens": n_tokens(tokenizer, v2),
            "e1_n_tokens": n_tokens(tokenizer, e1),
            "e2_n_tokens": n_tokens(tokenizer, e2),
            "write_A_n_tokens": n_tokens(tokenizer, block_A),
            "write_B_n_tokens": n_tokens(tokenizer, block_B),
            "write_token_multiset_sha256": sha256_bytes(
                json.dumps({str(k): ms_A[k] for k in sorted(ms_A)}, sort_keys=True).encode()
            ),
            "values_token_length_matched": n_tokens(tokenizer, v1) == n_tokens(tokenizer, v2),
        },
    }
    return pair


def build_dataset_pairs(*, tokenizer=None) -> list[dict[str, Any]]:
    tok = tokenizer or load_tokenizer()
    _require_unique(OCCUPATIONS, kind="occupation")
    entities = _make_entities()
    exact_value_pairs = _make_matched_phrase_pairs(tok)
    occ_len = [n_tokens(tok, o) for o in OCCUPATIONS]
    occ_pairs_once = _pair_equal_length(
        OCCUPATIONS, occ_len, kind="occupation", allow_unmatched=True
    )
    semantic_value_pairs = occ_pairs_once + occ_pairs_once  # 16 unique pairs, used twice
    if len(exact_value_pairs) != 32 or len(semantic_value_pairs) != 32:
        raise RuntimeError("expected 32 value pairs per type")

    exact_entities = entities[:64]
    semantic_entities = entities[64:]
    pairs: list[dict[str, Any]] = []
    for i in range(N_PER_TYPE):
        v1, v2 = exact_value_pairs[i]
        e1 = exact_entities[2 * i]
        e2 = exact_entities[2 * i + 1]
        pairs.append(
            _build_pair(
                pair_id=f"exact-{i:02d}",
                info_type="exact",
                e1=e1,
                e2=e2,
                v1=v1,
                v2=v2,
                smoke=i < SMOKE_N,
                write_e1_first=(i % 2 == 0),
                tokenizer=tok,
            )
        )
    for i in range(N_PER_TYPE):
        v1, v2 = semantic_value_pairs[i]
        e1 = semantic_entities[2 * i]
        e2 = semantic_entities[2 * i + 1]
        pairs.append(
            _build_pair(
                pair_id=f"semantic_direct-{i:02d}",
                info_type="semantic_direct",
                e1=e1,
                e2=e2,
                v1=v1,
                v2=v2,
                smoke=i < SMOKE_N,
                write_e1_first=(i % 2 == 0),
                tokenizer=tok,
            )
        )
    validate_dataset_pairs(pairs, tokenizer=tok)
    return pairs


def validate_dataset_pairs(pairs: list[dict[str, Any]], *, tokenizer) -> None:
    if len(pairs) != 64:
        raise ValueError(f"expected 64 pairs, got {len(pairs)}")
    by_id = {p["pair_id"]: p for p in pairs}
    if len(by_id) != 64:
        raise ValueError("pair_id collision")
    exact = [p for p in pairs if p["info_type"] == "exact"]
    semantic = [p for p in pairs if p["info_type"] == "semantic_direct"]
    if len(exact) != 32 or len(semantic) != 32:
        raise ValueError(f"type counts exact={len(exact)} semantic={len(semantic)}")
    if sum(1 for p in exact if p["smoke"]) != SMOKE_N:
        raise ValueError("exact smoke count")
    if sum(1 for p in semantic if p["smoke"]) != SMOKE_N:
        raise ValueError("semantic smoke count")
    if sum(1 for p in exact if p["write_order"] == ["e1", "e2"]) != 16:
        raise ValueError("exact write-order not balanced 16/16")
    if sum(1 for p in semantic if p["write_order"] == ["e1", "e2"]) != 16:
        raise ValueError("semantic write-order not balanced 16/16")

    _require_unique([p["e1"] for p in pairs] + [p["e2"] for p in pairs], kind="entity")
    _require_unique([p["v1"] for p in exact] + [p["v2"] for p in exact], kind="exact value")

    all_questions: list[str] = []
    for p in pairs:
        if p["e1"] == p["e2"] or p["v1"] == p["v2"]:
            raise ValueError(f"{p['pair_id']}: entity or value not distinct")
        if p["tokenizer"]["v1_n_tokens"] != p["tokenizer"]["v2_n_tokens"]:
            if p["info_type"] == "exact":
                raise ValueError(f"{p['pair_id']}: unmatched exact value token length")
            if p["tokenizer"].get("values_token_length_matched", True):
                raise ValueError(f"{p['pair_id']}: matched flag disagrees with token counts")
        if p["entities_in_write_order"][0] not in (p["e1"], p["e2"]):
            raise ValueError(f"{p['pair_id']}: write order entity mismatch")
        if p["entities_in_write_order"] != [
            p[k] for k in p["write_order"]
        ]:
            raise ValueError(f"{p['pair_id']}: entities_in_write_order drifted")
        if p["sentences_A"] == p["sentences_B"]:
            raise ValueError(f"{p['pair_id']}: A and B sentences identical")
        if p["write_block_A"].count(p["v1"]) != 1 or p["write_block_A"].count(p["v2"]) != 1:
            raise ValueError(f"{p['pair_id']}: each value must appear once in A")
        if p["write_block_B"].count(p["v1"]) != 1 or p["write_block_B"].count(p["v2"]) != 1:
            raise ValueError(f"{p['pair_id']}: each value must appear once in B")
        if p["write_block_A"].count(p["e1"]) != 1 or p["write_block_A"].count(p["e2"]) != 1:
            raise ValueError(f"{p['pair_id']}: each entity must appear once in A")
        if p["entities_in_write_order"] != [
            s.split("'s")[0] if p["info_type"] == "semantic_direct" else _entity_from_exact(s)
            for s in p["sentences_A"]
        ]:
            raise ValueError(f"{p['pair_id']}: A entity order mismatch")
        if [
            s.split("'s")[0] if p["info_type"] == "semantic_direct" else _entity_from_exact(s)
            for s in p["sentences_B"]
        ] != p["entities_in_write_order"]:
            raise ValueError(f"{p['pair_id']}: B entity order differs from A")
        ms_A = token_id_multiset(tokenizer, p["write_block_A"])
        ms_B = token_id_multiset(tokenizer, p["write_block_B"])
        if ms_A != ms_B:
            raise ValueError(f"{p['pair_id']}: A/B token multiset mismatch at validate")
        if "?" in p["write_block_A"] or "?" in p["write_block_B"]:
            raise ValueError(f"{p['pair_id']}: question mark in write")
        for ent_key in ("e1", "e2"):
            for style, q in p["questions"][ent_key].items():
                all_questions.append(q)
                if q in p["write_block_A"] or q in p["write_block_B"]:
                    raise ValueError(f"{p['pair_id']}: {style} question leaked into write")
        if p["candidates"] != [p["v1"], p["v2"]]:
            raise ValueError(f"{p['pair_id']}: candidates must be [v1, v2] in that order")
        if p["mapping_A"]["e1"] != p["v1"] or p["mapping_A"]["e2"] != p["v2"]:
            raise ValueError(f"{p['pair_id']}: mapping A is not e1->v1, e2->v2")
        if p["mapping_B"]["e1"] != p["v2"] or p["mapping_B"]["e2"] != p["v1"]:
            raise ValueError(f"{p['pair_id']}: mapping B is not swapped")
        if p["info_type"] == "semantic_direct":
            if " occupation is " not in p["write_block_A"]:
                raise ValueError(f"{p['pair_id']}: unexpected semantic template")
            if " works as a " in p["write_block_A"] or " works as an " in p["write_block_A"]:
                raise ValueError(f"{p['pair_id']}: article present in semantic write")
        if p["info_type"] == "exact":
            if "passphrase assigned to" not in p["write_block_A"]:
                raise ValueError(f"{p['pair_id']}: unexpected exact template")
    _require_unique(all_questions, kind="question string")


def _entity_from_exact(sentence: str) -> str:
    prefix = "The passphrase assigned to "
    suffix = " is "
    if not sentence.startswith(prefix):
        raise ValueError(f"exact sentence missing prefix: {sentence!r}")
    mid = sentence[len(prefix) :]
    idx = mid.find(suffix)
    if idx < 0:
        raise ValueError(f"exact sentence missing ' is ': {sentence!r}")
    return mid[:idx]


def freeze_dataset_pairs(*, force: bool = False, tokenizer=None) -> dict[str, Any]:
    pairs = build_dataset_pairs(tokenizer=tokenizer)
    payload = dataset_bytes(pairs)
    digest = sha256_bytes(payload)
    repo_root = Path(__file__).resolve().parents[2]
    manifest = {
        "dataset_id": "exp02a1_binding_pairs_v1",
        "seed": DATASET_SEED,
        "n_pairs": len(pairs),
        "n_exact_pairs": 32,
        "n_semantic_direct_pairs": 32,
        "smoke_n_per_type": SMOKE_N,
        "sha256": digest,
        "path": str(DATASET_PATH.relative_to(repo_root)),
        "spec_experiment_id": SPEC["experiment_id"],
        "backbone_revision_sha": SPEC["backbone"]["revision_sha"],
        "note": (
            "Frozen before smoke scores. Mapping A/B are token-multiset matched. "
            "Do not regenerate after looking at results. Does not modify Exp 02A datasets."
        ),
    }
    FROZEN_DIR.mkdir(parents=True, exist_ok=True)
    if DATASET_PATH.exists() and not force:
        existing = DATASET_PATH.read_bytes()
        existing_digest = sha256_bytes(existing)
        if existing_digest != digest:
            raise RuntimeError(
                f"refusing to overwrite frozen pairs (existing sha256={existing_digest}, "
                f"new sha256={digest}). Pass force=True only if you intend to invalidate."
            )
        saved = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
        if saved.get("sha256") != digest:
            raise RuntimeError("frozen jsonl and manifest sha256 diverged")
        return saved
    DATASET_PATH.write_bytes(payload)
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def load_dataset_pairs(*, smoke: bool = False) -> list[dict[str, Any]]:
    if not DATASET_PATH.exists():
        raise FileNotFoundError(f"frozen dataset missing: {DATASET_PATH}")
    raw = DATASET_PATH.read_bytes()
    meta = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    digest = sha256_bytes(raw)
    if digest != meta["sha256"]:
        raise RuntimeError(f"dataset sha256 mismatch: file={digest} meta={meta['sha256']}")
    pairs = [json.loads(line) for line in raw.decode("utf-8").splitlines() if line.strip()]
    tokenizer = load_tokenizer()
    validate_dataset_pairs(pairs, tokenizer=tokenizer)
    if smoke:
        pairs = [p for p in pairs if p["smoke"]]
        n_exact = sum(1 for p in pairs if p["info_type"] == "exact")
        n_sem = sum(1 for p in pairs if p["info_type"] == "semantic_direct")
        if n_exact != SMOKE_N or n_sem != SMOKE_N:
            raise RuntimeError(f"smoke subset exact={n_exact} semantic={n_sem}")
    return pairs


if __name__ == "__main__":
    meta = freeze_dataset_pairs()
    print(json.dumps(meta, indent=2))
