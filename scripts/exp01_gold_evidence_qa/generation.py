"""Greedy generation on the official 125M TTT-E2E backbone.

Conflicts with the official repository (documented, not worked around silently):

1. Official `loss_for_sequence` never returns logits. Exp 01 calls the same
   `prefix_call` + `suffix_call` that `lm_loss` uses and keeps the logits.
2. Suffix SWA asserts `xq.shape[0] == mini_batch_size` (1024). Sequences are
   right-padded with EOS to one chunk. Causal attention: padding sits in the
   future of the last real token, so it does not affect next-token logits.
3. Official inner-loop CE/logits for a chunk are computed *before* that
   chunk's fast-weight update. All Exp 01 prompts fit in one 1024-token chunk,
   so generation uses the meta-learned initial `feed_forward_prime`, not an
   intra-prompt TTT update. We do not run `inner_loop_step` and we do not add
   a second forward after the unused inner update.

Postprocessing (frozen before the main run):
  raw_prediction  = tokenizer.decode(all 32 generated ids, specials kept)
  scored_prediction = decode ids until EOS, take text before the first newline, strip
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from prompts import PromptTemplate, render_prompt

SWA_SAFETY_MARGIN = 32
MAX_NEW_TOKENS = 32
PAD_TO_CHUNKS = 1  # one TTT mini-batch


@dataclass
class EncodedPrompt:
    input_ids_unpadded: list[int]
    padded_ids: list[int]
    prompt_len: int
    evidence_token_length: int
    evidence_start: int
    evidence_end: int
    question_start: int
    question_end: int
    evidence_question_distance: int
    swa_window: int
    text: str


def encode_prompt(
    tokenizer,
    template: PromptTemplate,
    *,
    context: str,
    question: str,
    bos_id: int,
    eos_id: int,
    chunk_size: int,
    swa_window: int,
    max_new_tokens: int = MAX_NEW_TOKENS,
    safety_margin: int = SWA_SAFETY_MARGIN,
) -> EncodedPrompt:
    text = render_prompt(template, context=context, question=question)
    enc = tokenizer.encode(text, add_special_tokens=False)
    ids = [bos_id] + list(enc.ids)
    offsets = list(enc.offsets)  # character offsets into `text`, not including BOS

    def overlapping_token_range(char_start: int, char_end: int) -> tuple[int, int]:
        hits = []
        for i, (s, e) in enumerate(offsets):
            if e > char_start and s < char_end:
                hits.append(i + 1)  # +1 for BOS
        if not hits:
            raise ValueError(f"no tokens overlap char span [{char_start}, {char_end})")
        return hits[0], hits[-1] + 1

    ctx_start = text.find(context)
    q_start = text.find(question)
    if ctx_start < 0 or q_start < 0:
        raise ValueError("context or question not found in rendered prompt")
    ev_start, ev_end = overlapping_token_range(ctx_start, ctx_start + len(context))
    q_tok_start, q_tok_end = overlapping_token_range(q_start, q_start + len(question))

    prompt_len = len(ids)
    if prompt_len + max_new_tokens > chunk_size:
        raise ValueError(
            f"prompt_len {prompt_len} + max_new_tokens {max_new_tokens} exceeds chunk_size {chunk_size}"
        )
    padded = ids + [eos_id] * (chunk_size - prompt_len)
    last_prompt_index = prompt_len - 1
    # Look-back from the last prompt token to the first evidence token.
    evidence_question_distance = last_prompt_index - ev_start
    if evidence_question_distance + safety_margin >= swa_window:
        raise AssertionError(
            f"evidence_question_distance={evidence_question_distance} + margin={safety_margin} "
            f">= swa_window={swa_window}"
        )
    if evidence_question_distance >= swa_window:
        raise AssertionError(
            f"evidence_question_distance={evidence_question_distance} >= swa_window={swa_window}"
        )
    return EncodedPrompt(
        input_ids_unpadded=ids,
        padded_ids=padded,
        prompt_len=prompt_len,
        evidence_token_length=ev_end - ev_start,
        evidence_start=ev_start,
        evidence_end=ev_end,
        question_start=q_tok_start,
        question_end=q_tok_end,
        evidence_question_distance=int(evidence_question_distance),
        swa_window=int(swa_window),
        text=text,
    )


def postprocess_generation(token_ids, tokenizer, eos_id: int) -> tuple[str, str]:
    """Frozen scoring rule: first generated line before newline, cut at EOS."""
    ids = [int(t) for t in list(token_ids)]
    raw = tokenizer.decode(ids, skip_special_tokens=False)
    cut: list[int] = []
    for t in ids:
        if t == eos_id:
            break
        cut.append(t)
    text = tokenizer.decode(cut, skip_special_tokens=False)
    scored = text.split("\n", 1)[0].strip()
    return raw, scored


def make_greedy_generate_fn(loaded, max_new_tokens: int = MAX_NEW_TOKENS):
    """Compile one greedy loop: 32 argmax steps on a length-1024 padded prefix.

    Fast weights and suffix KV state are closed over from `loaded` and never
    written back, so each call starts from the same initialization.
    """
    import equinox as eqx
    import jax
    import jax.numpy as jnp

    from ttt.model.data import Batch

    model = loaded.model
    state_prefix = loaded.state_prefix
    state_suffix = loaded.state_suffix
    chunk = int(loaded.mini_batch_size)

    mesh = loaded.mesh

    @eqx.filter_jit
    def greedy_jit(padded_ids, prompt_len):
        ids = padded_ids.astype(jnp.int32)
        prompt_len = prompt_len.astype(jnp.int32)

        def body(i, ids):
            last_pos = prompt_len + i - 1
            seq = Batch(
                input_ids=ids,
                target_tokens=ids,
                loss_masks=jnp.ones((chunk,), dtype=bool),
            )
            xt = model.language_model.wte_call(ids)
            prefix_out = model.language_model.prefix_call(
                model.language_model.model.h.prefix_blocks,
                xt,
                state_prefix,
                seq,
            )
            lm_out = model.language_model.suffix_call(
                prefix_outputs=prefix_out.last_hidden_state,
                state=state_suffix,
                seq=seq,
            )
            logits = lm_out.logits[last_pos].astype(jnp.float32)
            tok = jnp.argmax(logits).astype(jnp.int32)
            ids = ids.at[prompt_len + i].set(tok)
            return ids

        ids = jax.lax.fori_loop(0, max_new_tokens, body, ids)
        return jax.lax.dynamic_slice(ids, (prompt_len,), (max_new_tokens,))

    def greedy(padded_ids, prompt_len):
        # Official prefix SWA uses with_sharding_constraint(..., P(None, "state", None)).
        with mesh:
            return greedy_jit(padded_ids, prompt_len)

    return greedy


def generate_one(greedy_fn, encoded: EncodedPrompt, tokenizer, eos_id: int) -> dict[str, Any]:
    import jax.numpy as jnp

    padded = jnp.asarray(encoded.padded_ids, dtype=jnp.int32)
    prompt_len = jnp.asarray(encoded.prompt_len, dtype=jnp.int32)
    gen_ids = greedy_fn(padded, prompt_len)
    gen_ids.block_until_ready()
    token_ids = [int(x) for x in list(gen_ids)]
    raw, scored = postprocess_generation(token_ids, tokenizer, eos_id)
    return {
        "generated_token_ids": token_ids,
        "raw_prediction": raw,
        "scored_prediction": scored,
        "input_token_length": encoded.prompt_len,
        "evidence_token_length": encoded.evidence_token_length,
        "evidence_question_distance": encoded.evidence_question_distance,
        "swa_window": encoded.swa_window,
        "prompt_text": encoded.text,
    }
