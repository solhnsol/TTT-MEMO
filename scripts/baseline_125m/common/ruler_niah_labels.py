"""RULER NIAH token-level relevance labels. No model, no TTT, no JAX.

Needle reconstruction uses the official generator template, not answer-string
search:

    One of the special magic {numbers|uuids} for {key} is: {value}.

Indexing (must not be collapsed into a single `input_ids[t]` label):

    raw token position j  --  input_ids[j]
    loss/gradient position t = j - 1

    l_t = -log p(input_ids[j] | prefix) with j = t + 1

So G_t is attached to the *predicted target token* at raw position t+1.
A token at chunk boundary j=1024 is predicted by loss t=1023 (previous chunk).

Char→token: a token is in a span iff its offset interval overlaps the char
span. Leading space / trailing newline glued onto the same BPE piece are
included on purpose (STEP 6A).
"""

from __future__ import annotations

import hashlib
import math
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from baseline_config import REPO_ROOT, TOKENIZER_PATH

# Keep in sync with upstream/llama3-tokenizer.toml and verify_tokenizer.py.
TOKENIZER_SHA256 = "e134af98b985517b4f068e3755ae90d4e9cd2d45d328325dc503f1c6b2d06cc7"

RULER_ROOT = REPO_ROOT / "data" / "ruler_niah"
LABEL_RESULT_DIR = REPO_ROOT / "results" / "raw" / "baseline_125m" / "ruler_niah_labels"

BOS_ID = 128000
PAD_ID = BOS_ID  # right-pad only; pad tokens as *targets* are loss-masked
MINI_BATCH = 1024
SWA = 8192
OUT_OF_SWA_MIN_DISTANCE = SWA - 1  # out_of_swa iff distance_to_query > 8191

TASK_FAMILY = {
    "niah_single_1": "S-NIAH",
    "niah_single_2": "S-NIAH",
    "niah_single_3": "S-NIAH",
    "niah_multikey_1": "MK-NIAH",
    "niah_multikey_2": "MK-NIAH",
    "niah_multikey_3": "MK-NIAH",
    "niah_multiquery": "MQ-NIAH",
    "niah_multivalue": "MV-NIAH",
}

# Needle-sentence plural type. Query text uses the singular. From synthetic.yaml
# + STEP 6A reconstruction (not a stored JSONL field).
TASK_VTYPE = {
    "niah_single_1": "numbers",
    "niah_single_2": "numbers",
    "niah_single_3": "uuids",
    "niah_multikey_1": "numbers",
    "niah_multikey_2": "numbers",
    "niah_multikey_3": "uuids",
    "niah_multiquery": "numbers",
    "niah_multivalue": "numbers",
}

# Observed in STEP 6A; compared after labeling, never used to rewrite spans.
EXPECTED_N_RELEVANT = {
    "niah_single_1": 1,
    "niah_single_2": 1,
    "niah_single_3": 1,
    "niah_multikey_1": 1,
    "niah_multikey_2": 1,
    "niah_multikey_3": 1,
    "niah_multiquery": 4,
    "niah_multivalue": 4,
}

QUERY_SINGLE = re.compile(
    r"What is the special magic (number|uuid) for (.+?) mentioned in the provided text\?"
)
QUERY_MULTI = re.compile(
    r"What are all the special magic (numbers|uuids) for (.+?) mentioned in the provided text\?"
)
NEEDLE_RE = re.compile(
    r"One of the special magic (numbers|uuids) for (.+?) is: "
    r"([0-9]{7}|[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12})\."
)

PADDING_MASK_RULE = (
    "Right-pad unpadded ids to T_pad = ceil(T/1024)*1024 with PAD_ID=BOS=128000. "
    "NTP has n_loss = T_pad-1 positions: target at raw j=t+1. "
    "mask[t]=1 iff t < T-1 (target is a real token). "
    "Valid loss count = T-1. Original tokens are never cropped."
)


class LabelError(Exception):
    """Unrecoverable labeling failure. Do not invent a span to paper over this."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def needle_sentence(vtype: str, key: str, value: str) -> str:
    return f"One of the special magic {vtype} for {key} is: {value}."


def raw_j_to_loss_t(raw_js: list[int], T: int) -> list[int]:
    """Map raw token indices j to loss positions t = j-1.

    Valid t satisfy 0 <= t <= T-2, i.e. the target j = t+1 is a real token
    in the *unpadded* sequence and is not position 0 (nothing predicts BOS).
    """
    if T < 2:
        raise LabelError(f"T={T} is too short for any NTP loss")
    out: list[int] = []
    for j in raw_js:
        t = j - 1
        if 0 <= t <= T - 2:
            out.append(t)
    return out


def padded_length(T: int, mini_batch: int = MINI_BATCH) -> int:
    if T <= 0:
        raise LabelError(f"T={T}")
    return math.ceil(T / mini_batch) * mini_batch


def right_pad_ids(ids: list[int], pad_id: int = PAD_ID, mini_batch: int = MINI_BATCH) -> tuple[list[int], int, int]:
    T = len(ids)
    T_pad = padded_length(T, mini_batch=mini_batch)
    if T_pad < T:
        raise LabelError("padding must not shorten the sequence")
    return ids + [pad_id] * (T_pad - T), T, T_pad


def ntp_loss_mask(T: int, T_pad: int) -> list[int]:
    """mask[t] for t = 0 .. T_pad-2. 1 iff the target raw token t+1 is real."""
    n_loss = T_pad - 1
    mask = [1 if t < T - 1 else 0 for t in range(n_loss)]
    if sum(mask) != T - 1:
        raise LabelError(f"valid loss count {sum(mask)} != T-1={T - 1}")
    return mask


def overlapping_token_indices(offsets: list[tuple[int, int]], start: int, end: int) -> list[int]:
    """Inclusive token indices whose char interval overlaps [start, end).

    Offsets are monotonically nondecreasing in start. Binary search is O(log T)
    so MK2/MK3 (thousands of needles × 128k tokens) stays CPU-feasible.
    A token [a, b) overlaps [start, end) iff a < end and b > start.
    """
    if start >= end:
        raise LabelError(f"empty char span [{start}, {end})")
    n = len(offsets)
    lo, hi = 0, n
    while lo < hi:
        mid = (lo + hi) // 2
        if offsets[mid][1] > start:
            hi = mid
        else:
            lo = mid + 1
    i0 = lo
    lo, hi = 0, n
    while lo < hi:
        mid = (lo + hi) // 2
        if offsets[mid][0] < end:
            lo = mid + 1
        else:
            hi = mid
    i1 = lo - 1
    if i0 > i1 or i0 >= n or i1 < 0:
        raise LabelError(f"no token overlaps char span [{start}, {end})")
    a0, b0 = offsets[i0]
    a1, b1 = offsets[i1]
    if not (a0 < end and b0 > start and a1 < end and b1 > start):
        raise LabelError(f"binary-search overlap invariant failed on [{start}, {end})")
    return list(range(i0, i1 + 1))


def _unique_find(haystack: str, needle: str, *, within: tuple[int, int] | None = None) -> int:
    if within is None:
        region, offset = haystack, 0
    else:
        a, b = within
        region, offset = haystack[a:b], a
    n = region.count(needle)
    if n == 0:
        raise LabelError(f"zero match for {needle!r}")
    if n != 1:
        raise LabelError(f"multiple unexpected matches ({n}) for {needle!r}")
    return offset + region.find(needle)


def parse_query(text: str) -> tuple[str, list[str], tuple[int, int]]:
    singles = list(QUERY_SINGLE.finditer(text))
    multis = list(QUERY_MULTI.finditer(text))
    if len(singles) + len(multis) != 1:
        raise LabelError(
            f"expected exactly 1 query sentence, got single={len(singles)} multi={len(multis)}"
        )
    if singles:
        m = singles[0]
        return "single", [m.group(2)], m.span()
    m = multis[0]
    blob = m.group(2)
    # MQ: "k1, k2, k3, and k4". MV: a single key with "What are all ... numbers".
    if ", and " in blob:
        head, last = blob.rsplit(", and ", 1)
        keys = [p.strip() for p in head.split(",")] + [last.strip()]
    else:
        keys = [blob.strip()]
    if any(k == "" for k in keys):
        raise LabelError(f"empty key in {blob!r}")
    return "multi", keys, m.span()


def relevant_pairs(task: str, keys: list[str], outputs: list[str]) -> list[tuple[str, str, str]]:
    if task not in TASK_VTYPE:
        raise LabelError(f"unknown task {task!r}")
    if not outputs:
        raise LabelError("empty outputs")
    vtype = TASK_VTYPE[task]
    if task == "niah_multivalue":
        if len(keys) != 1:
            raise LabelError(f"MV expected 1 query key, got {keys!r}")
        return [(vtype, keys[0], value) for value in outputs]
    if task == "niah_multiquery":
        if len(keys) != len(outputs):
            raise LabelError(f"MQ key/value length mismatch: {len(keys)} keys, {len(outputs)} outputs")
        return [(vtype, key, value) for key, value in zip(keys, outputs, strict=True)]
    if len(keys) != 1:
        raise LabelError(f"{task} expected 1 query key, got {keys!r}")
    return [(vtype, keys[0], value) for value in outputs]


def find_all_template_needles(text: str) -> list[tuple[str, str, str, int, int]]:
    """All generator-template needles as (vtype, key, value, char_start, char_end)."""
    out: list[tuple[str, str, str, int, int]] = []
    for m in NEEDLE_RE.finditer(text):
        sent = m.group(0)
        reconstructed = needle_sentence(m.group(1), m.group(2), m.group(3))
        if sent != reconstructed:
            raise LabelError("NEEDLE_RE match does not equal reconstructed template sentence")
        out.append((m.group(1), m.group(2), m.group(3), m.start(), m.end()))
    return out


@dataclass
class SpanLabel:
    role: str  # relevant | distractor
    span_index: int
    vtype: str
    key: str
    value: str
    sentence_char_start: int
    sentence_char_end: int
    key_char_start: int
    key_char_end: int
    value_char_start: int
    value_char_end: int
    sentence_raw_j_start: int
    sentence_raw_j_end: int
    sentence_loss_t_start: int
    sentence_loss_t_end: int
    key_raw_j_start: int
    key_raw_j_end: int
    key_loss_t_start: int
    key_loss_t_end: int
    value_raw_j_start: int
    value_raw_j_end: int
    value_loss_t_start: int
    value_loss_t_end: int
    distance_to_query: int
    out_of_swa: bool
    sentence_raw_chunk_start: int
    sentence_raw_chunk_end: int
    sentence_loss_chunk_start: int
    sentence_loss_chunk_end: int
    query_after_needle: bool


@dataclass
class ExampleLabels:
    task: str
    task_family: str
    seq_length: int
    seq_length_label: str
    sample_id: int
    source_jsonl: str
    ruler_index_char: int
    T: int
    T_pad: int
    pad_id: int
    bos_id: int
    tokenizer_sha256: str
    padding_mask_rule: str
    query_char_start: int
    query_char_end: int
    query_raw_j_start: int
    query_raw_j_end: int
    query_loss_t_start: int
    query_loss_t_end: int
    query_chunk_raw: int
    query_chunk_loss: int
    n_relevant_spans: int
    n_distractor_spans: int
    n_out_of_swa_spans: int
    any_relevant_out_of_swa: bool
    all_relevant_out_of_swa: bool
    valid_loss_count: int
    input_ids: list[int]
    spans: list[SpanLabel] = field(default_factory=list)


def _span_from_char(
    *,
    role: str,
    span_index: int,
    vtype: str,
    key: str,
    value: str,
    sent_start: int,
    sent_end: int,
    text: str,
    offsets: list[tuple[int, int]],
    T: int,
    query_start_j: int,
) -> SpanLabel:
    sent = text[sent_start:sent_end]
    expected = needle_sentence(vtype, key, value)
    if sent != expected:
        raise LabelError(f"sentence mismatch: {sent!r} vs {expected!r}")
    key_rel = _unique_find(sent, key)
    val_rel = _unique_find(sent, value)
    key_start, key_end = sent_start + key_rel, sent_start + key_rel + len(key)
    val_start, val_end = sent_start + val_rel, sent_start + val_rel + len(value)
    if not (sent_start <= key_start < key_end <= sent_end):
        raise LabelError("key char span not inside sentence")
    if not (sent_start <= val_start < val_end <= sent_end):
        raise LabelError("value char span not inside sentence")
    if key_end > val_start:
        raise LabelError("key/value char spans overlap or are out of template order")

    def mapped(cs: int, ce: int) -> tuple[tuple[int, int], tuple[int, int]]:
        raw = overlapping_token_indices(offsets, cs, ce)
        loss = raw_j_to_loss_t(raw, T)
        if not loss:
            raise LabelError(f"empty loss_t for char [{cs}, {ce}) raw={raw}")
        if loss != list(range(loss[0], loss[-1] + 1)):
            raise LabelError(f"non-contiguous loss_t {loss}")
        return (raw[0], raw[-1]), (loss[0], loss[-1])

    sent_raw, sent_loss = mapped(sent_start, sent_end)
    key_raw, key_loss = mapped(key_start, key_end)
    val_raw, val_loss = mapped(val_start, val_end)
    if not (sent_raw[0] <= key_raw[0] <= key_raw[1] <= sent_raw[1]):
        raise LabelError("key tokens not inside sentence tokens")
    if not (sent_raw[0] <= val_raw[0] <= val_raw[1] <= sent_raw[1]):
        raise LabelError("value tokens not inside sentence tokens")

    needle_last_j = sent_raw[1]
    distance = query_start_j - needle_last_j
    query_after = query_start_j > needle_last_j
    return SpanLabel(
        role=role,
        span_index=span_index,
        vtype=vtype,
        key=key,
        value=value,
        sentence_char_start=sent_start,
        sentence_char_end=sent_end,
        key_char_start=key_start,
        key_char_end=key_end,
        value_char_start=val_start,
        value_char_end=val_end,
        sentence_raw_j_start=sent_raw[0],
        sentence_raw_j_end=sent_raw[1],
        sentence_loss_t_start=sent_loss[0],
        sentence_loss_t_end=sent_loss[1],
        key_raw_j_start=key_raw[0],
        key_raw_j_end=key_raw[1],
        key_loss_t_start=key_loss[0],
        key_loss_t_end=key_loss[1],
        value_raw_j_start=val_raw[0],
        value_raw_j_end=val_raw[1],
        value_loss_t_start=val_loss[0],
        value_loss_t_end=val_loss[1],
        distance_to_query=distance,
        out_of_swa=distance > OUT_OF_SWA_MIN_DISTANCE,
        sentence_raw_chunk_start=sent_raw[0] // MINI_BATCH,
        sentence_raw_chunk_end=sent_raw[1] // MINI_BATCH,
        sentence_loss_chunk_start=sent_loss[0] // MINI_BATCH,
        sentence_loss_chunk_end=sent_loss[1] // MINI_BATCH,
        query_after_needle=query_after,
    )


def label_example(
    rec: dict[str, Any],
    tokenizer,
    *,
    source_jsonl: str,
    tokenizer_sha256: str,
) -> ExampleLabels:
    task = rec["task"]
    text = rec["input"]
    outputs = list(rec["outputs"])
    enc = tokenizer.encode(text, add_special_tokens=False)
    ids = list(enc.ids)
    offsets = list(enc.offsets)
    T = len(ids)
    if T < 2:
        raise LabelError(f"T={T}")
    if ids[0] != BOS_ID:
        raise LabelError(f"first token is {ids[0]}, expected BOS {BOS_ID}")

    _, T_check, T_pad = right_pad_ids(ids)
    if T_check != T:
        raise LabelError("pad helper changed T")
    if T_pad % MINI_BATCH != 0:
        raise LabelError(f"T_pad={T_pad} not divisible by {MINI_BATCH}")
    mask = ntp_loss_mask(T, T_pad)
    if sum(mask) != T - 1:
        raise LabelError("valid loss count")

    _qkind, keys, qspan = parse_query(text)
    q_raw = overlapping_token_indices(offsets, qspan[0], qspan[1])
    q_loss = raw_j_to_loss_t(q_raw, T)
    if not q_loss:
        raise LabelError("empty query loss_t")
    query_start_j = q_raw[0]

    pairs = relevant_pairs(task, keys, outputs)
    relevant_sents = [needle_sentence(v, k, val) for v, k, val in pairs]
    if len(set(relevant_sents)) != len(relevant_sents):
        raise LabelError("duplicate reconstructed relevant sentences")

    relevant_spans: list[SpanLabel] = []
    for i, (vtype, key, value) in enumerate(pairs):
        sent = relevant_sents[i]
        start = _unique_find(text, sent)
        end = start + len(sent)
        relevant_spans.append(
            _span_from_char(
                role="relevant",
                span_index=i,
                vtype=vtype,
                key=key,
                value=value,
                sent_start=start,
                sent_end=end,
                text=text,
                offsets=offsets,
                T=T,
                query_start_j=query_start_j,
            )
        )

    template_hits = find_all_template_needles(text)
    relevant_sent_set = set(relevant_sents)
    distractor_spans: list[SpanLabel] = []
    d_i = 0
    n_template_relevant = 0
    for vtype, key, value, start, end in template_hits:
        sent = needle_sentence(vtype, key, value)
        if sent in relevant_sent_set:
            n_template_relevant += 1
            continue
        distractor_spans.append(
            _span_from_char(
                role="distractor",
                span_index=d_i,
                vtype=vtype,
                key=key,
                value=value,
                sent_start=start,
                sent_end=end,
                text=text,
                offsets=offsets,
                T=T,
                query_start_j=query_start_j,
            )
        )
        d_i += 1
    if n_template_relevant != len(relevant_sents):
        raise LabelError(
            f"template regex relevant hits {n_template_relevant} != reconstructed {len(relevant_sents)}"
        )

    if not relevant_spans:
        raise LabelError("relevant mask empty")
    for sp in relevant_spans:
        if not sp.query_after_needle:
            raise LabelError(
                f"query_start_j={query_start_j} is not after needle last raw j={sp.sentence_raw_j_end}"
            )
        if sp.sentence_loss_t_start < 0 or sp.sentence_loss_t_end > T - 2:
            raise LabelError("relevant loss_t out of range")

    n_out = sum(1 for sp in relevant_spans if sp.out_of_swa)
    return ExampleLabels(
        task=task,
        task_family=TASK_FAMILY[task],
        seq_length=int(rec["seq_length"]),
        seq_length_label=str(rec["seq_length_label"]),
        sample_id=int(rec["sample_id"]),
        source_jsonl=source_jsonl,
        ruler_index_char=int(rec["index"]),
        T=T,
        T_pad=T_pad,
        pad_id=PAD_ID,
        bos_id=BOS_ID,
        tokenizer_sha256=tokenizer_sha256,
        padding_mask_rule=PADDING_MASK_RULE,
        query_char_start=qspan[0],
        query_char_end=qspan[1],
        query_raw_j_start=q_raw[0],
        query_raw_j_end=q_raw[-1],
        query_loss_t_start=q_loss[0],
        query_loss_t_end=q_loss[-1],
        query_chunk_raw=q_raw[0] // MINI_BATCH,
        query_chunk_loss=q_loss[0] // MINI_BATCH,
        n_relevant_spans=len(relevant_spans),
        n_distractor_spans=len(distractor_spans),
        n_out_of_swa_spans=n_out,
        any_relevant_out_of_swa=n_out > 0,
        all_relevant_out_of_swa=n_out == len(relevant_spans),
        valid_loss_count=T - 1,
        input_ids=ids,
        spans=relevant_spans + distractor_spans,
    )


def processed_jsonl(task: str, seq_length_label: str) -> Path:
    family = TASK_FAMILY[task]
    return (
        RULER_ROOT
        / family
        / task
        / seq_length_label
        / f"{task}_{seq_length_label}_n200.jsonl"
    )


def example_row(ex: ExampleLabels) -> dict[str, Any]:
    row = {k: v for k, v in asdict(ex).items() if k != "spans"}
    return row


def span_rows(ex: ExampleLabels) -> list[dict[str, Any]]:
    rows = []
    for sp in ex.spans:
        row = asdict(sp)
        row.update(
            {
                "task": ex.task,
                "task_family": ex.task_family,
                "seq_length": ex.seq_length,
                "seq_length_label": ex.seq_length_label,
                "sample_id": ex.sample_id,
            }
        )
        rows.append(row)
    return rows
