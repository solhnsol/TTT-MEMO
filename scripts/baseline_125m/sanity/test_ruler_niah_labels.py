"""CPU unit tests for RULER NIAH labeling. No checkpoint, no TTT, no GPU.

Run from repo root:

    PYTHONPATH=scripts/baseline_125m/common .venv/bin/python \
        scripts/baseline_125m/sanity/test_ruler_niah_labels.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

_BASELINE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_BASELINE_ROOT / "common"))
sys.path.insert(0, str(_BASELINE_ROOT / "sanity"))

from ruler_niah_labels import (  # noqa: E402
    BOS_ID,
    MINI_BATCH,
    PAD_ID,
    TOKENIZER_PATH,
    TOKENIZER_SHA256,
    LabelError,
    label_example,
    needle_sentence,
    ntp_loss_mask,
    overlapping_token_indices,
    padded_length,
    processed_jsonl,
    raw_j_to_loss_t,
    right_pad_ids,
    sha256_file,
)
from tokenizers import Tokenizer  # noqa: E402
from verify_tokenizer import sha256_of  # noqa: E402


def assert_eq(got, expected, msg: str) -> None:
    if got != expected:
        raise SystemExit(f"FAIL {msg}: got {got!r}, expected {expected!r}")


def test_raw_to_loss_indexing() -> None:
    # Gradient position t is attached to the predicted target at raw j = t+1.
    assert_eq(raw_j_to_loss_t([1024], T=2048), [1023], "chunk boundary j=1024 -> t=1023")
    assert_eq(1024 // MINI_BATCH, 1, "raw j=1024 is chunk 1")
    assert_eq(1023 // MINI_BATCH, 0, "loss t=1023 is chunk 0")
    assert_eq(raw_j_to_loss_t([0], T=8), [], "BOS j=0 has no loss")
    assert_eq(raw_j_to_loss_t([1], T=8), [0], "first real target")
    assert_eq(raw_j_to_loss_t([7], T=8), [6], "last real token j=T-1 -> t=T-2")
    assert_eq(raw_j_to_loss_t([8], T=8), [], "j=T is not a real token")
    assert_eq(raw_j_to_loss_t([0, 1, 2], T=8), [0, 1], "drop j=0 keep the rest")


def test_padding_and_valid_loss_count() -> None:
    ids = [BOS_ID] + list(range(1, 8040))  # T=8040, not a multiple of 1024
    padded, T, T_pad = right_pad_ids(ids)
    assert_eq(T, 8040, "T")
    assert_eq(T_pad, 8192, "ceil(8040/1024)*1024")
    assert_eq(T_pad % MINI_BATCH, 0, "T_pad % 1024")
    assert_eq(len(padded), T_pad, "padded length")
    assert_eq(padded[:T], ids, "unpadded prefix preserved")
    assert_eq(padded[T:], [PAD_ID] * (T_pad - T), "right pad with BOS")
    mask = ntp_loss_mask(T, T_pad)
    assert_eq(len(mask), T_pad - 1, "n_loss = T_pad-1")
    assert_eq(sum(mask), T - 1, "valid loss count == T-1")
    assert_eq(mask[T - 2], 1, "last real target t=T-2 unmasked")
    assert_eq(mask[T - 1], 0, "first pad as target is masked")
    assert_eq(mask[-1], 0, "final pad target masked")
    assert_eq(padded_length(1024), 1024, "already aligned")
    assert_eq(padded_length(1025), 2048, "just over one chunk")


def test_overlap_matches_linear_scan() -> None:
    tok = Tokenizer.from_file(str(TOKENIZER_PATH))
    path = processed_jsonl("niah_single_1", "8k")
    rec = json.loads(path.open().readline())
    enc = tok.encode(rec["input"], add_special_tokens=False)
    offsets = list(enc.offsets)
    sent = needle_sentence("numbers", "roomy-devil", rec["outputs"][0])
    cs = rec["input"].find(sent)
    ce = cs + len(sent)
    fast = overlapping_token_indices(offsets, cs, ce)
    slow = [i for i, (a, b) in enumerate(offsets) if a < ce and b > cs]
    assert_eq(fast, slow, "binary search vs linear overlap")

    tok = Tokenizer.from_file(str(TOKENIZER_PATH))
    sent = needle_sentence("numbers", "roomy-devil", "3094235")
    text = f"<|begin_of_text|>preamble\n{sent}\nWhat is the special magic number for roomy-devil mentioned in the provided text?"
    enc = tok.encode(text, add_special_tokens=False)
    cs = text.find(sent)
    ce = cs + len(sent)
    idx = overlapping_token_indices(list(enc.offsets), cs, ce)
    last_j = idx[-1]
    a, b = enc.offsets[last_j]
    # Period is in the sentence; the following newline is often the same BPE piece.
    assert_eq(text[cs:ce] == sent, True, "char slice is the sentence")
    if b > ce:
        assert_eq("\n" in text[ce:b] or text[ce:b] != "", True, "last token may extend past sentence")
    decoded = tok.decode([enc.ids[j] for j in idx], skip_special_tokens=False)
    if decoded != sent:
        # Allowed: extra leading space or trailing newline from overlap.
        if not (decoded.strip() == sent.strip() or decoded.startswith(sent) or sent in decoded):
            raise SystemExit(f"FAIL overlap decode unrelated to sentence: {decoded!r} vs {sent!r}")
    assert_eq(enc.ids[0], BOS_ID, "BOS from in-text special token")
    enc_true = tok.encode(text, add_special_tokens=True)
    assert_eq(enc_true.ids[0] == BOS_ID and enc_true.ids[1] == BOS_ID, True, "True would double BOS")


def test_real_s1_sample0() -> None:
    path = processed_jsonl("niah_single_1", "8k")
    if not path.exists():
        print(f"skip real-file test, missing {path}")
        return
    rec = json.loads(path.open().readline())
    tok = Tokenizer.from_file(str(TOKENIZER_PATH))
    digest = sha256_file(TOKENIZER_PATH)
    assert_eq(digest, TOKENIZER_SHA256, "tokenizer pin")
    ex = label_example(
        rec,
        tok,
        source_jsonl="S-NIAH/niah_single_1/8k/niah_single_1_8k_n200.jsonl",
        tokenizer_sha256=digest,
    )
    assert_eq(ex.n_relevant_spans, 1, "S1 relevant count")
    assert_eq(ex.n_distractor_spans, 0, "S1 distractor count")
    assert_eq(ex.input_ids[0], BOS_ID, "real example BOS")
    assert_eq(ex.valid_loss_count, ex.T - 1, "valid loss")
    assert_eq(ex.T_pad % 1024, 0, "T_pad")
    sp = ex.spans[0]
    assert_eq(sp.role, "relevant", "role")
    assert_eq(sp.value, rec["outputs"][0], "value")
    assert_eq(sp.key, "roomy-devil", "key")
    assert_eq(
        rec["input"][sp.sentence_char_start : sp.sentence_char_end],
        needle_sentence("numbers", "roomy-devil", rec["outputs"][0]),
        "sentence slice",
    )
    assert_eq(
        rec["input"][sp.value_char_start : sp.value_char_end],
        rec["outputs"][0],
        "value slice",
    )
    # Stored RULER token_position_answer is the *value* start, not the sentence.
    assert_eq(sp.value_raw_j_start, rec["token_position_answer"], "value raw j vs stored tpa")
    assert_eq(sp.sentence_raw_j_start < sp.value_raw_j_start, True, "sentence starts before value")
    assert_eq(sp.sentence_loss_t_start, sp.sentence_raw_j_start - 1, "t = j-1 on sentence start")
    assert_eq(sp.query_after_needle, True, "query after needle")
    # Trailing . + newline overlap on noise haystack (STEP 6A).
    last_off = tok.encode(rec["input"], add_special_tokens=False).offsets[sp.sentence_raw_j_end]
    assert_eq(last_off[1] > sp.sentence_char_end, True, "last BPE token extends past sentence")


def test_real_mv_sample0() -> None:
    path = processed_jsonl("niah_multivalue", "8k")
    if not path.exists():
        print(f"skip MV real-file test, missing {path}")
        return
    rec = json.loads(path.open().readline())
    tok = Tokenizer.from_file(str(TOKENIZER_PATH))
    ex = label_example(
        rec,
        tok,
        source_jsonl="MV-NIAH/niah_multivalue/8k/niah_multivalue_8k_n200.jsonl",
        tokenizer_sha256=sha256_file(TOKENIZER_PATH),
    )
    assert_eq(ex.n_relevant_spans, 4, "MV relevant")
    assert_eq(ex.n_distractor_spans, 0, "MV distractor")
    assert_eq(len(rec["outputs"]), 4, "MV outputs")
    keys = {sp.key for sp in ex.spans if sp.role == "relevant"}
    assert_eq(len(keys), 1, "MV single key")
    values = [sp.value for sp in ex.spans if sp.role == "relevant"]
    assert_eq(values, rec["outputs"], "MV values in output order")
    for sp in ex.spans:
        assert_eq(sp.query_after_needle, True, "each MV needle before query")


def test_zero_match_is_error() -> None:
    tok = Tokenizer.from_file(str(TOKENIZER_PATH))
    rec = {
        "task": "niah_single_1",
        "task_family": "S-NIAH",
        "seq_length": 8192,
        "seq_length_label": "8k",
        "sample_id": 0,
        "index": 0,
        "input": (
            "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n"
            "No needle here.\n"
            "What is the special magic number for missing-key mentioned in the provided text?"
            "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
        ),
        "outputs": ["3094235"],
    }
    try:
        label_example(rec, tok, source_jsonl="synthetic", tokenizer_sha256="x")
    except LabelError as exc:
        if "zero match" not in str(exc):
            raise SystemExit(f"FAIL expected zero-match LabelError, got {exc}")
        return
    raise SystemExit("FAIL expected LabelError for missing needle")


def main() -> None:
    if not TOKENIZER_PATH.exists():
        raise SystemExit(f"tokenizer missing: {TOKENIZER_PATH}")
    if sha256_of(TOKENIZER_PATH) != TOKENIZER_SHA256:
        raise SystemExit("tokenizer sha256 mismatch")
    test_raw_to_loss_indexing()
    test_padding_and_valid_loss_count()
    test_overlap_matches_linear_scan()
    test_zero_match_is_error()
    test_real_s1_sample0()
    test_real_mv_sample0()
    print("all ruler_niah labeling unit tests passed")


if __name__ == "__main__":
    main()
