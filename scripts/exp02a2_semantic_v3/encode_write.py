"""Tokenize independent write sequences and record prefix/target mask metadata.

No optimization is performed. Character-offset overlap is the same convention
as Exp 02A so future NTP CE labels stay inspectable.
"""

from __future__ import annotations

from typing import Any

from protocol import SPEC

WRITE_WRAPPER_PREFIX = "Fact:\n"


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


def render_chat(tokenizer, messages: list[dict[str, str]], *, add_generation_prompt: bool) -> str:
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=add_generation_prompt,
    )
    if not isinstance(text, str):
        raise TypeError(f"apply_chat_template returned {type(text)}")
    if "<think>" in text:
        raise RuntimeError("chat template inserted a thinking block")
    return text


def _unique_span(haystack: str, needle: str, *, kind: str) -> tuple[int, int]:
    start = haystack.find(needle)
    if start < 0:
        raise ValueError(f"{kind} not found: {needle!r}")
    if haystack.find(needle, start + 1) >= 0:
        raise ValueError(f"{kind} is not unique: {needle!r}")
    return start, start + len(needle)


def _tokenize_with_offsets(tokenizer, text: str) -> tuple[list[int], list[str], list[tuple[int, int]]]:
    enc = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
    ids = [int(i) for i in enc["input_ids"]]
    offsets = [(int(s), int(e)) for s, e in enc["offset_mapping"]]
    pieces = tokenizer.convert_ids_to_tokens(ids)
    if len(ids) != len(offsets):
        raise RuntimeError("offset mapping length mismatch")
    return ids, pieces, offsets


def _indices_overlapping(offsets: list[tuple[int, int]], span: tuple[int, int]) -> list[int]:
    out = []
    for i, (s, e) in enumerate(offsets):
        if e <= s:
            continue
        if s < span[1] and e > span[0]:
            out.append(i)
    return out


def encode_plain_sentence(tokenizer, seq: dict[str, Any]) -> dict[str, Any]:
    sentence = seq["write_sentence"]
    ids, pieces, offsets = _tokenize_with_offsets(tokenizer, sentence)
    prefix_span = _unique_span(sentence, seq["prefix"], kind="prefix")
    desc_span = _unique_span(sentence, seq["description"], kind="description")
    target_span = _unique_span(sentence, seq["target"], kind="target")
    prefix_idx = _indices_overlapping(offsets, prefix_span)
    target_idx = _indices_overlapping(offsets, target_span)
    desc_idx = _indices_overlapping(offsets, desc_span)
    if not target_idx:
        raise RuntimeError(f"{seq['sequence_id']}: target overlapped no tokens")
    labels = [-100] * len(ids)
    for i in target_idx:
        labels[i] = ids[i]
    return {
        "surface": "raw_sentence",
        "text": sentence,
        "input_ids": ids,
        "tokens": pieces,
        "n_tokens": len(ids),
        "prefix_char_span": list(prefix_span),
        "description_char_span": list(desc_span),
        "target_char_span": list(target_span),
        "prefix_token_indices": prefix_idx,
        "description_token_indices": desc_idx,
        "target_token_indices": target_idx,
        "ce_labels": labels,
        "n_labeled_target_tokens": len(target_idx),
    }


def encode_chat_write(tokenizer, seq: dict[str, Any]) -> dict[str, Any]:
    content = WRITE_WRAPPER_PREFIX + seq["write_sentence"]
    text = render_chat(
        tokenizer,
        [{"role": "user", "content": content}],
        add_generation_prompt=False,
    )
    if seq["label"] in text:
        raise RuntimeError(f"{seq['sequence_id']}: gold label present in chat-rendered write")
    ids, pieces, offsets = _tokenize_with_offsets(tokenizer, text)
    desc_span = _unique_span(text, seq["description"], kind="chat_description")
    target_span = _unique_span(text, seq["target"], kind="chat_target")
    target_idx = _indices_overlapping(offsets, target_span)
    desc_idx = _indices_overlapping(offsets, desc_span)
    if not target_idx:
        raise RuntimeError(f"{seq['sequence_id']}: chat target overlapped no tokens")
    labels = [-100] * len(ids)
    for i in target_idx:
        labels[i] = ids[i]
    return {
        "surface": "chat_rendered_fact_user_turn",
        "text": text,
        "input_ids": ids,
        "tokens": pieces,
        "n_tokens": len(ids),
        "description_char_span": list(desc_span),
        "target_char_span": list(target_span),
        "description_token_indices": desc_idx,
        "target_token_indices": target_idx,
        "ce_labels": labels,
        "n_labeled_target_tokens": len(target_idx),
        "note": (
            "Future NTP CE should use ce_labels (description+period only). "
            "Entity, scaffold, Fact:\\n wrapper, and chat specials are -100."
        ),
    }


def encode_independent_sequence(tokenizer, seq: dict[str, Any]) -> dict[str, Any]:
    raw = encode_plain_sentence(tokenizer, seq)
    chat = encode_chat_write(tokenizer, seq)
    for blob in (raw, chat):
        decoded_target = tokenizer.decode(
            [blob["input_ids"][i] for i in blob["target_token_indices"]],
            skip_special_tokens=False,
        )
        if seq["label"].lower() in decoded_target.lower():
            raise RuntimeError(
                f"{seq['sequence_id']} {blob['surface']}: gold label in decoded target {decoded_target!r}"
            )
    return {
        "raw_sentence": raw,
        "chat_rendered_write": chat,
        "label_n_tokens": len(tokenizer(seq["label"], add_special_tokens=False)["input_ids"]),
        "label_token_ids": [
            int(i) for i in tokenizer(seq["label"], add_special_tokens=False)["input_ids"]
        ],
        "entity_n_tokens": len(tokenizer(seq["entity"], add_special_tokens=False)["input_ids"]),
        "entity_token_ids": [
            int(i) for i in tokenizer(seq["entity"], add_special_tokens=False)["input_ids"]
        ],
        "description_n_tokens": len(
            tokenizer(seq["description"], add_special_tokens=False)["input_ids"]
        ),
        "description_token_ids": [
            int(i) for i in tokenizer(seq["description"], add_special_tokens=False)["input_ids"]
        ],
    }


def encode_candidate(tokenizer, prompt_text: str, candidate: str) -> dict[str, Any]:
    if not prompt_text.endswith("<|im_start|>assistant\n"):
        raise RuntimeError(
            "unexpected generation prompt tail; refuse to guess candidate tokenization: "
            f"{prompt_text[-40]!r}"
        )
    full_text = prompt_text + candidate + "<|im_end|>\n"
    ids, _pieces, offsets = _tokenize_with_offsets(tokenizer, full_text)
    suffix_start = len(prompt_text)
    if full_text[suffix_start : suffix_start + len(candidate)] != candidate:
        raise RuntimeError("candidate is not a prefix of the assistant suffix")
    cand_start = suffix_start
    cand_end = suffix_start + len(candidate)
    labels = [-100] * len(ids)
    cand_idx: list[int] = []
    for i, (s, e) in enumerate(offsets):
        if e <= s:
            continue
        if s < cand_end and e > cand_start:
            labels[i] = ids[i]
            cand_idx.append(i)
    if not cand_idx or cand_idx[0] == 0:
        raise RuntimeError("candidate token span invalid")
    return {
        "text": full_text,
        "input_ids": ids,
        "labels": labels,
        "candidate_token_indices": cand_idx,
        "n_tokens": len(cand_idx),
        "candidate": candidate,
    }


def recall_user_content(question: str, *, fact_block: str | None) -> str:
    """Canonical occupation_query already includes the reply instruction."""
    if fact_block is None:
        return question
    return f"{WRITE_WRAPPER_PREFIX}{fact_block}\n\n{question}"


def encode_recall_prompt(
    tokenizer,
    *,
    question: str,
    fact_block: str | None,
) -> dict[str, Any]:
    content = recall_user_content(question, fact_block=fact_block)
    text = render_chat(
        tokenizer,
        [{"role": "user", "content": content}],
        add_generation_prompt=True,
    )
    ids, _pieces, _offsets = _tokenize_with_offsets(tokenizer, text)
    return {"text": text, "input_ids": ids, "source_present": fact_block is not None}
