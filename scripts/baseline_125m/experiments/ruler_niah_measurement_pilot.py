"""STEP 6D: first RULER NIAH × TTT measurement pipeline pilot.

One 8K S-NIAH example. Not H1/H2, not future utility, not 16K, not 8000 examples.
Does not edit e2e/. Does not overwrite STEP 3 / 6C.1 files.

Confirms indexing, right-BOS padding, official 1024-chunk TTT trajectory, and
scalable Ahat/Ghat/Ihat join against precomputed labels.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path

_BASELINE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_BASELINE_ROOT / "common"))

from baseline_config import (  # noqa: E402
    CHECKPOINT_DIR,
    EXPERIMENT_RESULT_DIR,
    TOKENIZER_PATH,
    add_e2e_to_path,
    compose_baseline_config,
)
from measurement_utils import (  # noqa: E402
    error_metrics,
    leading_axis_size,
    official_masked_reduction,
    pytree_l2_norm,
    pytree_max_abs_diff,
    pytree_vdot,
    rademacher_like,
    select_even_valid_indices,
    summarize_values,
    token_ttt_ce_from_logits,
)
from ruler_niah_labels import (  # noqa: E402
    BOS_ID,
    LABEL_RESULT_DIR,
    MINI_BATCH,
    PAD_ID,
    ntp_loss_mask,
    right_pad_ids,
)
from runtime_helpers import AttentionOverride, describe_gpu, tokens_to_batch  # noqa: E402

TASK = "niah_single_1"
SEQ_LABEL = "8k"
SAMPLE_ID = 0
JVP_SEED = 0
K_G = 32
N_EXACT = 32
NEIGHBOR = 8
REL_EPS = 1e-12
ROLE_CODES = {
    "padding": 0,
    "background": 1,
    "relevant_sentence": 2,
    "relevant_key": 3,
    "relevant_value": 4,
    "query": 5,
}
NEEDLE_ROLES = ("relevant_value", "relevant_key", "relevant_sentence")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--attention-cudnn-override",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Default off: official cuDNN prefix SWA on A100.",
    )
    return parser.parse_args()


def block_until_ready(tree):
    import jax

    jax.tree.map(lambda x: x.block_until_ready() if hasattr(x, "block_until_ready") else x, tree)
    return tree


def timed(label: str, fn, *, warmup: bool = False, quiet: bool = False):
    t0 = time.perf_counter()
    result = fn()
    block_until_ready(result)
    elapsed = time.perf_counter() - t0
    if not quiet:
        kind = "compile+run" if warmup else "run"
        print(f"  time[{kind}] {label}: {elapsed:.3f}s", flush=True)
    return result, elapsed


def to_jsonable(obj):
    import numpy as np

    if isinstance(obj, dict):
        return {str(k): to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(v) for v in obj]
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, np.integer, np.bool_)):
        return obj.item()
    return obj


def try_memory_stats():
    try:
        import jax

        stats = jax.devices()[0].memory_stats()
        if not stats:
            return None
        return {str(k): (int(v) if isinstance(v, (int,)) else v) for k, v in stats.items()}
    except Exception:
        return None


def finite_flags(name: str, arr) -> dict:
    import numpy as np

    x = np.asarray(arr)
    finite = np.isfinite(x)
    return {
        "name": name,
        "any_nan": bool(np.any(np.isnan(x))),
        "any_inf": bool(np.any(np.isinf(x))),
        "all_finite": bool(np.all(finite)),
    }


def clip_scale(g_norm: float, max_norm: float) -> float:
    if g_norm < max_norm:
        return 1.0
    return float(max_norm / g_norm)


def inspect_inexact_paths(tree) -> list[dict]:
    import equinox as eqx
    import jax

    rows = []
    for path, value in jax.tree.flatten_with_path(tree)[0]:
        if eqx.is_inexact_array(value):
            rows.append(
                {
                    "path": jax.tree_util.keystr(path),
                    "shape": list(value.shape),
                    "dtype": str(value.dtype),
                    "size": int(value.size),
                }
            )
    return rows


def sign_agreement(exact, estimated) -> dict:
    import numpy as np

    e = np.asarray(exact, dtype=np.float64)
    h = np.asarray(estimated, dtype=np.float64)
    se = np.sign(e)
    sh = np.sign(h)
    neg = e < 0
    pos = e > 0
    return {
        "n": int(e.size),
        "agree": float(np.mean(se == sh)) if e.size else None,
        "n_exact_neg": int(np.sum(neg)),
        "n_exact_pos": int(np.sum(pos)),
        "neg_recall": float(np.sum(neg & (h < 0)) / np.sum(neg)) if np.any(neg) else None,
        "pos_recall": float(np.sum(pos & (h > 0)) / np.sum(pos)) if np.any(pos) else None,
    }


def stop_gradient_tree(tree):
    import equinox as eqx
    import jax

    return jax.tree.map(lambda x: jax.lax.stop_gradient(x) if eqx.is_inexact_array(x) else x, tree)


def inclusive_range(start: int, end: int) -> range:
    return range(int(start), int(end) + 1)


def assign_roles(*, n_loss: int, valid, query_t0: int, query_t1: int, span) -> list[str]:
    roles = ["padding"] * n_loss
    for t in range(n_loss):
        if not valid[t]:
            continue
        if int(span.value_loss_t_start) <= t <= int(span.value_loss_t_end):
            roles[t] = "relevant_value"
        elif int(span.key_loss_t_start) <= t <= int(span.key_loss_t_end):
            roles[t] = "relevant_key"
        elif int(span.sentence_loss_t_start) <= t <= int(span.sentence_loss_t_end):
            roles[t] = "relevant_sentence"
        elif int(query_t0) <= t <= int(query_t1):
            roles[t] = "query"
        else:
            roles[t] = "background"
    return roles


def select_needle_chunk_exact(roles_local, valid_local, *, n: int, neighbor: int):
    import numpy as np

    t_loc = int(len(valid_local))
    valid_local = np.asarray(valid_local).astype(bool)
    chosen: list[int] = []

    def add(idxs):
        for i in idxs:
            if len(chosen) >= n:
                return
            i = int(i)
            if 0 <= i < t_loc and valid_local[i] and i not in chosen:
                chosen.append(i)

    for role in NEEDLE_ROLES:
        add([i for i in range(t_loc) if roles_local[i] == role])
    needle = {i for i in range(t_loc) if roles_local[i] in NEEDLE_ROLES}
    nearby: list[int] = []
    for i in sorted(needle):
        for j in range(max(0, i - neighbor), min(t_loc, i + neighbor + 1)):
            if valid_local[j] and j not in needle and roles_local[j] == "background":
                nearby.append(j)
    seen: set[int] = set()
    nearby_u = []
    for j in nearby:
        if j not in seen:
            seen.add(j)
            nearby_u.append(j)
    add(nearby_u)
    spread, spread_meta = select_even_valid_indices(valid_local, n)
    add(spread)
    counts = {role: int(sum(roles_local[i] == role for i in chosen)) for role in ROLE_CODES}
    meta = {
        "n_requested": n,
        "n_selected": len(chosen),
        "neighbor": neighbor,
        "role_counts": counts,
        "spread_rule": spread_meta["rule"],
        "selected_local": chosen,
    }
    return chosen, meta


def load_pilot_example():
    import pandas as pd

    examples = pd.read_parquet(LABEL_RESULT_DIR / "examples.parquet")
    spans = pd.read_parquet(LABEL_RESULT_DIR / "spans.parquet")
    hit = examples[
        (examples["task"] == TASK)
        & (examples["seq_length_label"] == SEQ_LABEL)
        & (examples["sample_id"] == SAMPLE_ID)
    ]
    if len(hit) != 1:
        raise SystemExit(f"expected one example {TASK}/{SEQ_LABEL}/sid={SAMPLE_ID}, got {len(hit)}")
    row = hit.iloc[0]
    sp = spans[
        (spans["task"] == TASK)
        & (spans["seq_length_label"] == SEQ_LABEL)
        & (spans["sample_id"] == SAMPLE_ID)
        & (spans["role"] == "relevant")
    ]
    if len(sp) != 1:
        raise SystemExit(f"expected one relevant span, got {len(sp)}")
    return row, sp.iloc[0]


def build_batch(ids: list[int], T: int, T_pad: int):
    import jax.numpy as jnp
    import numpy as np

    if len(ids) != T:
        raise SystemExit(f"stored input_ids len {len(ids)} != T={T}")
    if ids[0] != BOS_ID:
        raise SystemExit("unpadded sequence does not start with BOS")
    if any(i == BOS_ID for i in ids[1:]):
        raise SystemExit("real tokens after BOS include PAD/BOS id; NTP mask would drop them")
    padded, T2, T_pad2 = right_pad_ids(ids, pad_id=PAD_ID, mini_batch=MINI_BATCH)
    if T2 != T or T_pad2 != T_pad:
        raise SystemExit(f"recomputed pad mismatch T {T2} vs {T}, T_pad {T_pad2} vs {T_pad}")
    if padded[:T] != ids:
        raise SystemExit("right-pad cropped or mutated real tokens")
    if any(x != PAD_ID for x in padded[T:]):
        raise SystemExit("padding region is not PAD_ID")
    tokens = padded + [PAD_ID]
    seq = tokens_to_batch(jnp.asarray(tokens, dtype=jnp.int32), bos_token_id=BOS_ID)
    mask = np.asarray(seq.loss_masks).astype(bool)
    label_mask = np.asarray(ntp_loss_mask(T, T_pad), dtype=bool)
    if seq.input_ids.shape[0] != T_pad:
        raise SystemExit(f"Batch length {seq.input_ids.shape[0]} != T_pad={T_pad}")
    if mask.shape[0] != T_pad:
        raise SystemExit("official Batch loss_masks is length T_pad, not T_pad-1")
    if not np.array_equal(mask[: T_pad - 1], label_mask):
        raise SystemExit("tokens_to_batch mask[:T_pad-1] != ntp_loss_mask")
    if mask[-1]:
        raise SystemExit("final extra-PAD target should be invalid")
    if int(mask.sum()) != T - 1:
        raise SystemExit(f"valid loss count {int(mask.sum())} != T-1={T - 1}")
    if not mask[: T - 1].all() or mask[T - 1 :].any():
        raise SystemExit("valid region is not exactly t < T-1")
    return seq, padded, mask


def decode_id(tokenizer, token_id: int) -> str | None:
    if tokenizer is None:
        return None
    return tokenizer.id_to_token(int(token_id))


def main(*, attention_cudnn_override: bool) -> None:
    wall0 = time.perf_counter()
    row, span = load_pilot_example()
    T = int(row["T"])
    T_pad = int(row["T_pad"])
    ids = [int(x) for x in row["input_ids"]]
    n_chunks = T_pad // MINI_BATCH
    if T_pad % MINI_BATCH != 0:
        raise SystemExit(f"T_pad={T_pad} not divisible by {MINI_BATCH}")

    seq, padded, valid_global = build_batch(ids, T, T_pad)
    roles = assign_roles(
        n_loss=T_pad,
        valid=valid_global,
        query_t0=int(row["query_loss_t_start"]),
        query_t1=int(row["query_loss_t_end"]),
        span=span,
    )
    needle_chunk = int(span["sentence_loss_chunk_start"])
    if int(span["sentence_loss_chunk_end"]) != needle_chunk:
        raise SystemExit("needle sentence spans two chunks; refuse a silent split in this pilot")
    for t in inclusive_range(int(span["value_loss_t_start"]), int(span["value_loss_t_end"])):
        if not valid_global[t] or roles[t] != "relevant_value":
            raise SystemExit(f"relevant value loss t={t} is not valid/value")
    for t in inclusive_range(int(row["query_loss_t_start"]), int(row["query_loss_t_end"])):
        if not valid_global[t] or roles[t] != "query":
            raise SystemExit(f"query loss t={t} failed join")

    print("--- sample / batch ---", flush=True)
    print(
        f"  {TASK}/{SEQ_LABEL}/sample_id={SAMPLE_ID}  T={T} T_pad={T_pad} n_chunks={n_chunks}",
        flush=True,
    )
    print(
        f"  key={span['key']!r} value={span['value']!r}  "
        f"needle loss chunk={needle_chunk}  query chunk={int(row['query_chunk_loss'])}  "
        f"distance_to_query={int(span['distance_to_query'])} out_of_swa={bool(span['out_of_swa'])}",
        flush=True,
    )
    print(
        f"  valid_loss={int(valid_global.sum())} (expect T-1={T - 1})  "
        f"padding loss={int((~valid_global).sum())}",
        flush=True,
    )

    tokenizer = None
    if TOKENIZER_PATH.exists():
        from tokenizers import Tokenizer

        tokenizer = Tokenizer.from_file(str(TOKENIZER_PATH))
    print("\n--- label join (human-readable relevant tokens) ---", flush=True)
    join_rows = []
    for role, j0, j1, t0, t1 in (
        (
            "relevant_sentence",
            int(span["sentence_raw_j_start"]),
            int(span["sentence_raw_j_end"]),
            int(span["sentence_loss_t_start"]),
            int(span["sentence_loss_t_end"]),
        ),
        (
            "relevant_key",
            int(span["key_raw_j_start"]),
            int(span["key_raw_j_end"]),
            int(span["key_loss_t_start"]),
            int(span["key_loss_t_end"]),
        ),
        (
            "relevant_value",
            int(span["value_raw_j_start"]),
            int(span["value_raw_j_end"]),
            int(span["value_loss_t_start"]),
            int(span["value_loss_t_end"]),
        ),
    ):
        for j, t in zip(inclusive_range(j0, j1), inclusive_range(t0, t1), strict=True):
            tok = decode_id(tokenizer, ids[j])
            rec = {
                "role": role,
                "raw_j": j,
                "loss_t": t,
                "chunk": t // MINI_BATCH,
                "token_id": ids[j],
                "decoded": tok,
                "assigned_role": roles[t],
            }
            join_rows.append(rec)
            print(
                f"  {role:20s} j={j:5d} t={t:5d} id={ids[j]:6d} {tok!r}  assigned={roles[t]}",
                flush=True,
            )
            if t != j - 1:
                raise SystemExit(f"loss t={t} is not j-1 for j={j}")
            if j >= T:
                raise SystemExit("relevant raw j is in padding")

    cfg = compose_baseline_config()
    add_e2e_to_path()

    import equinox as eqx
    import jax
    import jax.numpy as jnp
    import jax.random as jrandom
    import numpy as np
    from jax.sharding import PartitionSpec as P

    from ttt.infra.checkpoint import Checkpointer, unify_dict_with_eqx_module
    from ttt.model.data import tree_slice
    from ttt.model.sharding import ModelSharding
    from ttt.model.transformer import BlockCollectionSplit, MetaModel
    from ttt.utils.filter_utils import get_filter_spec
    from ttt.utils.jax_utils import clone_pytree, eval_shape_and_sharding, initialize_distibuted, set_random_seed, tree_rearrange

    cfg.model.seq_len = cfg.training.seq_length
    mini_batch = int(cfg.model.mini_batch_size)
    if mini_batch != MINI_BATCH:
        raise SystemExit(f"mini_batch {mini_batch} != {MINI_BATCH}")
    if T_pad > int(cfg.model.seq_len):
        raise SystemExit(
            f"T_pad={T_pad} > model.seq_len={cfg.model.seq_len}; "
            "this 8K pilot should not need a seq_len override"
        )

    initialize_distibuted(cfg.backend)
    gpu = describe_gpu()
    override = AttentionOverride(enabled=attention_cudnn_override)
    override.install()
    clip_norm = float(cfg.training.optimizer_inner.clip_gradient)
    inner_lr = float(cfg.training.optimizer_inner.lr)
    eval_step = int(jnp.iinfo(jnp.int32).max - 100)

    key = set_random_seed(cfg.training.model_seed)
    model_sharding = ModelSharding(cfg)
    mesh = model_sharding.mesh

    @eqx.filter_jit
    def create_sharded_model_and_state():
        model, state = eqx.nn.make_with_state(MetaModel)(cfg, key=key)
        state = jax.device_put(state, jax.NamedSharding(mesh, P()))
        model = model_sharding.shard_params(model)
        return model, state

    abstract_model_weights = eval_shape_and_sharding(lambda: create_sharded_model_and_state()[0].weights())
    loader = Checkpointer(config=cfg, for_saving=False)
    if not loader.checkpoint_exists():
        raise FileNotFoundError(f"No Orbax checkpoint under {CHECKPOINT_DIR}")
    out_state = loader.load_checkpoint(
        step=cfg.training.resume_step,
        targets={"model_weights": abstract_model_weights},
        restore=cfg.training.load_part,
    )
    model_fresh, state = create_sharded_model_and_state()
    model, missing = unify_dict_with_eqx_module(out_state["model_weights"], model_fresh)
    if missing:
        raise SystemExit(f"checkpoint did not cover {len(missing)} model paths: {missing[:8]}")
    del out_state, model_fresh
    state = state.set(model.step_index, jnp.array(eval_step, dtype=jnp.int32))

    attn_setup = override.report_for_config(
        force_flash=bool(cfg.model.force_flash),
        seq_modeling_block=str(cfg.model.seq_modeling_block),
    )
    print("\n--- setup ---", flush=True)
    print(f"GPU: {gpu.name}  jax={gpu.jax_device}", flush=True)
    print(
        f"attention override_enabled={attn_setup['override_enabled']} "
        f"prefix_effective={attn_setup['prefix_effective_backend']}",
        flush=True,
    )
    print(f"model.seq_len={cfg.model.seq_len} (unchanged)  T_pad={T_pad}", flush=True)

    memory = {"before_compute": try_memory_stats()}
    runtime: dict = {"ghat_per_chunk_s": [], "inner_update_per_chunk_s": []}
    sanity: dict = {"per_chunk": []}

    token_ce = np.full((n_chunks, MINI_BATCH), np.nan, dtype=np.float32)
    ghat = np.full((n_chunks, MINI_BATCH), np.nan, dtype=np.float32)
    qhat = np.full((n_chunks, MINI_BATCH), np.nan, dtype=np.float32)
    ahat = np.full((n_chunks, MINI_BATCH), np.nan, dtype=np.float32)
    ihat = np.full((n_chunks, MINI_BATCH), np.nan, dtype=np.float32)
    valid_chunks = np.zeros((n_chunks, MINI_BATCH), dtype=np.uint8)
    g_bar_norms = np.full((n_chunks,), np.nan, dtype=np.float32)
    clip_cs = np.full((n_chunks,), np.nan, dtype=np.float32)
    etas = np.full((n_chunks,), np.nan, dtype=np.float32)
    n_valid_chunk = np.zeros((n_chunks,), dtype=np.int32)

    with mesh:
        block_collection = model.language_model.model.h.blocks
        prime_storage = model.language_model.model.h.prime_storage if cfg.model.prime else None
        if prime_storage is None:
            raise SystemExit("checkpoint has prime=False")
        state_prefix_suffix = state.substate(model.language_model.model.h.blocks)
        state_prefix, state_suffix = BlockCollectionSplit.split_state(state_prefix_suffix, cfg.model.suffix_len)
        state_all = clone_pytree(state)
        new_collection = BlockCollectionSplit(
            cfg.model, block_collection=block_collection, prime_storage=prime_storage, key=jrandom.PRNGKey(0)
        )
        model = eqx.tree_at(lambda m: m.language_model.model.h, model, new_collection)
        model_outer = jax.tree.map(lambda p: p.astype(model.state_dtype), model)
        model_inner = model_outer
        inner_opt_state = model_outer.inner_optimizer(state_all).init(model_outer.inner_parameters())

        prefix_len = leading_axis_size(model_outer.language_model.model.h.prefix_blocks)
        suffix_len = leading_axis_size(model_outer.language_model.model.h.suffix_blocks)
        spec_inner = get_filter_spec(model_outer, cfg.training.spec_inner, "inner parameters")
        inspect_inner0, _ = eqx.partition(model_outer, spec_inner)
        inner_paths = {r["path"] for r in inspect_inexact_paths(inspect_inner0)}
        if not any("suffix_blocks.feed_forward_prime" in p for p in inner_paths):
            raise SystemExit("spec_inner did not select suffix_blocks.feed_forward_prime")
        print(f"  P={prefix_len} S={suffix_len}", flush=True)

        def recombine_inner(model_i, model_o):
            spec = get_filter_spec(model_i, model_o.config.training.spec_inner, "inner parameters")
            inner_params, _ = eqx.partition(model_i, spec)
            _, outer_params = eqx.partition(model_o, spec)
            return eqx.combine(inner_params, outer_params)

        @eqx.filter_jit
        def official_chunk_step(model_i, opt_state, state_tuple, suffix_chunk, prefix_chunk, model_o):
            model_i = recombine_inner(model_i, model_o)
            return MetaModel.inner_loop_step(model_i, opt_state, state_tuple, suffix_chunk, prefix_chunk)

        xt_embed = model_outer.language_model.wte_call(seq.input_ids)
        prefix_output, t_prefix = timed(
            f"prefix_call T={T_pad}",
            lambda: eqx.filter_checkpoint(model_outer.language_model.prefix_call)(
                model_outer.language_model.model.h.prefix_blocks, xt_embed, state_prefix, seq
            ).last_hidden_state,
            warmup=True,
        )
        runtime["prefix_compile_s"] = t_prefix
        seq_chunks = tree_rearrange(seq, "(chunk token) ... -> chunk token ...", token=mini_batch)
        prefix_chunks = tree_rearrange(prefix_output, "(chunk token) ... -> chunk token ...", token=mini_batch)

        _, outer_params = eqx.partition(model_outer, spec_inner)
        suffix_state = state_suffix
        eval_step_arr = jnp.array(eval_step, dtype=jnp.int32)
        ilr_mult = float(np.asarray(model_outer.get_ilr_multiplier(eval_step_arr)))
        eta_eval = inner_lr * ilr_mult

        def token_ttt_ce_only(inner, state, prefix_chunk, seq_chunk):
            model_i = eqx.combine(inner, outer_params)
            out = model_i.language_model.suffix_call(
                prefix_outputs=prefix_chunk, state=state, seq=seq_chunk
            )
            return token_ttt_ce_from_logits(out.logits, seq_chunk.target_tokens)

        def token_ce_at_index(inner, idx, state, prefix_chunk, seq_chunk):
            return token_ttt_ce_only(inner, state, prefix_chunk, seq_chunk)[idx]

        def L_chunk(inner, state, prefix_chunk, seq_chunk):
            return official_masked_reduction(
                token_ttt_ce_only(inner, state, prefix_chunk, seq_chunk), seq_chunk.loss_masks
            )

        def jvp_token_ce(inner, tangent, state, prefix_chunk, seq_chunk):
            def f(w):
                return token_ttt_ce_only(w, state, prefix_chunk, seq_chunk)

            return eqx.filter_jvp(f, (inner,), (tangent,))

        f_jit = eqx.filter_jit(token_ttt_ce_only)
        grad_L_jit = eqx.filter_jit(eqx.filter_value_and_grad(L_chunk))
        grad_token_jit = eqx.filter_jit(eqx.filter_grad(token_ce_at_index))
        jvp_jit = eqx.filter_jit(jvp_token_ce)

        compiled = {"f": False, "gbar": False, "jvp": False, "exact": False, "update": False}
        exact_pack = None

        for c in range(n_chunks):
            print(f"\n========== chunk {c}/{n_chunks - 1} ==========", flush=True)
            seq_chunk = tree_slice(seq_chunks, c)
            prefix_chunk = prefix_chunks[c]
            model_meas = recombine_inner(model_inner, model_outer)
            inner_w, _ = eqx.partition(model_meas, spec_inner)
            s_canonical = clone_pytree(suffix_state)

            def fresh_state(s=s_canonical):
                return clone_pytree(s)

            inner_before = clone_pytree(inner_w)
            state_before = clone_pytree(s_canonical)
            mask_c = np.asarray(seq_chunk.loss_masks).astype(bool)
            n_valid = int(mask_c.sum())
            n_valid_chunk[c] = n_valid
            valid_chunks[c] = mask_c.astype(np.uint8)
            g0 = c * MINI_BATCH
            if not np.array_equal(mask_c, valid_global[g0 : g0 + MINI_BATCH]):
                raise SystemExit(f"chunk {c} mask does not match global mask slice")

            ce, t_f = timed(
                f"chunk{c} f(Wi)",
                lambda: f_jit(inner_w, fresh_state(), prefix_chunk, seq_chunk),
                warmup=not compiled["f"],
            )
            compiled["f"] = True
            ce_np = np.asarray(ce)
            token_ce[c] = ce_np.astype(np.float32)

            (L_val, g_bar), t_gbar = timed(
                f"chunk{c} g_bar",
                lambda: grad_L_jit(inner_w, fresh_state(), prefix_chunk, seq_chunk),
                warmup=not compiled["gbar"],
            )
            if not compiled["gbar"]:
                runtime["g_bar_compile_s"] = t_gbar
            compiled["gbar"] = True
            gbn = float(pytree_l2_norm(g_bar))
            c_clip = clip_scale(gbn, clip_norm)
            g_bar_norms[c] = gbn
            clip_cs[c] = c_clip
            etas[c] = eta_eval
            print(
                f"  n_valid={n_valid}  L={float(np.asarray(L_val)):.6f}  "
                f"||g_bar||={gbn:.6f}  c={c_clip:.4f}  clip={not (gbn < clip_norm)}  eta={eta_eval:.4f}",
                flush=True,
            )

            g_bar_tangent = stop_gradient_tree(g_bar)
            (_, ahat_c), t_ahat = timed(
                f"chunk{c} Ahat JVP",
                lambda: jvp_jit(inner_w, g_bar_tangent, fresh_state(), prefix_chunk, seq_chunk),
                warmup=not compiled["jvp"],
            )
            compiled["jvp"] = True
            ahat_c = np.asarray(ahat_c, dtype=np.float64)
            ahat[c] = ahat_c.astype(np.float32)

            jvp_keys = jrandom.split(jrandom.PRNGKey(JVP_SEED), K_G)
            t_ghat0 = time.perf_counter()
            sumsq = None
            for k_i in range(K_G):
                v_k = rademacher_like(inner_w, jvp_keys[k_i])
                (_, jv_k), _dt = timed(
                    f"chunk{c} ghat {k_i}",
                    lambda v=v_k: jvp_jit(inner_w, v, fresh_state(), prefix_chunk, seq_chunk),
                    quiet=True,
                )
                proj = np.asarray(jv_k, dtype=np.float64) ** 2
                sumsq = proj if sumsq is None else sumsq + proj
            runtime["ghat_per_chunk_s"].append(time.perf_counter() - t_ghat0)
            qhat_c = sumsq / float(K_G)
            ghat_c = np.sqrt(qhat_c)
            qhat[c] = qhat_c.astype(np.float32)
            ghat[c] = ghat_c.astype(np.float32)
            if n_valid > 0:
                ihat_c = ahat_c - qhat_c / float(n_valid)
            else:
                ihat_c = np.full_like(ahat_c, np.nan)
            ihat_c = np.where(mask_c, ihat_c, np.nan)
            ihat[c] = ihat_c.astype(np.float32)
            ahat[c] = np.where(mask_c, ahat[c], np.nan)
            ghat[c] = np.where(mask_c, ghat[c], np.nan)
            qhat[c] = np.where(mask_c, qhat[c], np.nan)
            token_ce[c] = np.where(mask_c, token_ce[c], np.nan)

            if c == needle_chunk:
                roles_local = roles[g0 : g0 + MINI_BATCH]
                selected, sel_meta = select_needle_chunk_exact(
                    roles_local, mask_c, n=N_EXACT, neighbor=NEIGHBOR
                )
                print(f"  exact subset n={len(selected)} roles={sel_meta['role_counts']}", flush=True)
                a_exact = np.full((len(selected),), np.nan, dtype=np.float64)
                g_exact = np.full((len(selected),), np.nan, dtype=np.float64)
                g0t, t_ex0 = timed(
                    f"exact t_local={selected[0]}",
                    lambda: grad_token_jit(
                        inner_w, jnp.int32(selected[0]), fresh_state(), prefix_chunk, seq_chunk
                    ),
                    warmup=not compiled["exact"],
                )
                compiled["exact"] = True
                runtime["exact_compile_s"] = t_ex0
                for i, t_idx in enumerate(selected):
                    if i == 0:
                        g_t = g0t
                    else:
                        g_t, _dt = timed(
                            f"exact t={t_idx}",
                            lambda t=int(t_idx): grad_token_jit(
                                inner_w, jnp.int32(t), fresh_state(), prefix_chunk, seq_chunk
                            ),
                            quiet=True,
                        )
                    g_exact[i] = float(pytree_l2_norm(g_t))
                    a_exact[i] = float(pytree_vdot(g_t, g_bar))
                    if i == 0 or i + 1 == len(selected) or (i + 1) % 8 == 0:
                        print(
                            f"    exact {i + 1}/{len(selected)} local={t_idx}  "
                            f"G={g_exact[i]:.4f} A={a_exact[i]:.6e} role={roles_local[t_idx]}",
                            flush=True,
                        )
                ahat_sub = np.asarray([ahat_c[t] for t in selected], dtype=np.float64)
                ghat_sub = np.asarray([ghat_c[t] for t in selected], dtype=np.float64)
                i_exact = a_exact - (g_exact ** 2) / float(n_valid)
                ihat_sub = np.asarray([ihat_c[t] for t in selected], dtype=np.float64)
                a_abs = np.abs(ahat_sub - a_exact)
                a_rel = a_abs / (np.abs(a_exact) + REL_EPS)
                a_metrics = error_metrics(a_exact, ahat_sub, np.ones(len(selected), dtype=bool), eps=REL_EPS)
                a_metrics["relative_error_vs_abs_exact_median"] = float(np.median(a_rel))
                exact_pack = {
                    "chunk": c,
                    "selection": sel_meta,
                    "local_indices": selected,
                    "global_indices": [g0 + t for t in selected],
                    "roles": [roles_local[t] for t in selected],
                    "a_exact": a_exact,
                    "g_exact": g_exact,
                    "i_exact": i_exact,
                    "ahat": ahat_sub,
                    "ghat": ghat_sub,
                    "ihat": ihat_sub,
                    "g_metrics": error_metrics(g_exact, ghat_sub, np.ones(len(selected), dtype=bool), eps=REL_EPS),
                    "a_metrics": a_metrics,
                    "a_sign": sign_agreement(a_exact, ahat_sub),
                    "i_metrics": error_metrics(i_exact, ihat_sub, np.ones(len(selected), dtype=bool), eps=REL_EPS),
                    "i_sign": sign_agreement(i_exact, ihat_sub),
                }
                print(
                    f"  exact G Spearman={exact_pack['g_metrics'].get('spearman')}  "
                    f"A Spearman={exact_pack['a_metrics'].get('spearman')}  "
                    f"A sign={exact_pack['a_sign']['agree']}  "
                    f"I Spearman={exact_pack['i_metrics'].get('spearman')}  "
                    f"I sign={exact_pack['i_sign']['agree']}",
                    flush=True,
                )

            w_unmoved = pytree_max_abs_diff(inner_before, inner_w)
            s_unmoved = pytree_max_abs_diff(state_before, suffix_state)
            out_step, t_upd = timed(
                f"inner_loop_step chunk{c}",
                lambda: official_chunk_step(
                    model_inner, inner_opt_state, (state_all, suffix_state), seq_chunk, prefix_chunk, model_outer
                ),
                warmup=not compiled["update"],
            )
            compiled["update"] = True
            runtime["inner_update_per_chunk_s"].append(t_upd)
            new_inner, _ = eqx.partition(recombine_inner(out_step.new_model, model_outer), spec_inner)
            dw = pytree_max_abs_diff(inner_before, new_inner)
            ds = pytree_max_abs_diff(s_canonical, out_step.new_state[1])
            sanity["per_chunk"].append(
                {
                    "chunk": c,
                    "n_valid": n_valid,
                    "measure_left_W_unchanged": w_unmoved,
                    "measure_left_S_unchanged": s_unmoved,
                    "update_dW_max_abs": dw,
                    "update_dS_max_abs": ds,
                }
            )
            print(
                f"  purity measure dW={w_unmoved:.3e} dS={s_unmoved:.3e}  "
                f"after update dW={dw:.3e} dS={ds:.3e}",
                flush=True,
            )
            if w_unmoved != 0.0 or s_unmoved != 0.0:
                raise SystemExit("measurement mutated W or suffix state before inner_loop_step")
            model_inner = out_step.new_model
            inner_opt_state = out_step.new_optimizer_state
            state_all, suffix_state = out_step.new_state

    roles_arr = np.array([ROLE_CODES[r] for r in roles], dtype=np.int8)
    roles_2d = roles_arr.reshape(n_chunks, MINI_BATCH)
    valid_flat = valid_global
    ghat_flat = ghat.reshape(-1)
    ahat_flat = ahat.reshape(-1)
    ihat_flat = ihat.reshape(-1)
    ce_flat = token_ce.reshape(-1)

    def role_mask(name: str):
        return (np.array(roles) == name) & valid_flat

    descriptive = {}
    for name in ("relevant_value", "relevant_key", "relevant_sentence", "background", "query"):
        m = role_mask(name)
        descriptive[name] = {
            "n": int(m.sum()),
            "ghat": summarize_values(ghat_flat[m]),
            "nll": summarize_values(ce_flat[m]),
            "ahat_n_neg": int(np.sum(ahat_flat[m] < 0)) if m.any() else 0,
            "ahat_n_pos": int(np.sum(ahat_flat[m] > 0)) if m.any() else 0,
            "ihat_n_neg": int(np.sum(ihat_flat[m] < 0)) if m.any() else 0,
            "ihat_n_pos": int(np.sum(ihat_flat[m] > 0)) if m.any() else 0,
            "ahat": summarize_values(ahat_flat[m]),
            "ihat": summarize_values(ihat_flat[m]),
        }
        print(
            f"  {name:20s} n={descriptive[name]['n']:5d}  "
            f"Ghat med={descriptive[name]['ghat'].get('median')}  "
            f"NLL med={descriptive[name]['nll'].get('median')}  "
            f"Ahat<0={descriptive[name]['ahat_n_neg']}  Ihat<0={descriptive[name]['ihat_n_neg']}",
            flush=True,
        )
    rel_m = role_mask("relevant_value") | role_mask("relevant_key") | role_mask("relevant_sentence")
    print(
        f"  relevant_any Ahat<0 = {int(np.sum(ahat_flat[rel_m] < 0))}/{int(rel_m.sum())}",
        flush=True,
    )

    memory["after_compute"] = try_memory_stats()
    runtime["wall_s"] = time.perf_counter() - wall0
    EXPERIMENT_RESULT_DIR.mkdir(parents=True, exist_ok=True)
    stem = EXPERIMENT_RESULT_DIR / "ruler_niah_measurement_pilot_niah_single_1_8k_sid0_a100_cudnn"
    npz = {
        "token_ttt_ce": token_ce,
        "ghat_k32": ghat,
        "qhat_k32": qhat,
        "ahat": ahat,
        "ihat": ihat,
        "valid_masks": valid_chunks,
        "roles": roles_2d,
        "g_bar_norm": g_bar_norms,
        "clip_scale_c": clip_cs,
        "eta": etas,
        "n_valid_chunk": n_valid_chunk,
        "input_ids_padded": np.asarray(padded, dtype=np.int32),
        "target_ids": np.asarray(seq.target_tokens),
    }
    if exact_pack is not None:
        npz["exact_local_indices"] = np.asarray(exact_pack["local_indices"], dtype=np.int32)
        npz["exact_global_indices"] = np.asarray(exact_pack["global_indices"], dtype=np.int32)
        npz["a_exact"] = exact_pack["a_exact"].astype(np.float32)
        npz["g_exact"] = exact_pack["g_exact"].astype(np.float32)
        npz["i_exact"] = exact_pack["i_exact"].astype(np.float32)
        npz["ahat_on_exact"] = exact_pack["ahat"].astype(np.float32)
        npz["ghat_on_exact"] = exact_pack["ghat"].astype(np.float32)
        npz["ihat_on_exact"] = exact_pack["ihat"].astype(np.float32)
    np.savez(stem.with_suffix(".npz"), **npz)

    exact_json = None
    if exact_pack is not None:
        exact_json = {
            "chunk": exact_pack["chunk"],
            "selection": exact_pack["selection"],
            "global_indices": exact_pack["global_indices"],
            "roles": exact_pack["roles"],
            "ghat_vs_exact": exact_pack["g_metrics"],
            "ahat_vs_exact": exact_pack["a_metrics"],
            "ahat_sign": exact_pack["a_sign"],
            "ihat_vs_exact": exact_pack["i_metrics"],
            "ihat_sign": exact_pack["i_sign"],
        }
    meta = {
        "name": "ruler_niah_measurement_pilot",
        "step": "6D",
        "not": "H1/H2 / future utility / 16K / 8000-example stats / router",
        "sample": {
            "task": TASK,
            "task_family": "S-NIAH",
            "seq_length_label": SEQ_LABEL,
            "sample_id": SAMPLE_ID,
            "source_jsonl": str(row["source_jsonl"]),
            "T": T,
            "T_pad": T_pad,
            "n_chunks": n_chunks,
            "valid_loss_count": T - 1,
            "key": str(span["key"]),
            "value": str(span["value"]),
            "needle_loss_chunk": needle_chunk,
            "query_loss_chunk": int(row["query_chunk_loss"]),
            "distance_to_query": int(span["distance_to_query"]),
            "out_of_swa": bool(span["out_of_swa"]),
            "in_window": True,
        },
        "label_definitions": {
            "indexing": "loss_t = raw_j - 1; G_t attached to predicted target input_ids[t+1]",
            "roles": list(ROLE_CODES.keys()),
            "role_priority": "padding > relevant_value > relevant_key > relevant_sentence > query > background",
            "padding": "right BOS/PAD; official mask is target != BOS; valid iff t < T-1",
            "N_valid_chunk": "per-chunk valid loss count used in Ihat = Ahat - Qhat / N",
        },
        "measurement": {
            "K": K_G,
            "jvp_seed": JVP_SEED,
            "backend": "cudnn" if not attention_cudnn_override else "xla_override",
            "checkpoint_dir": str(CHECKPOINT_DIR),
            "model_seq_len": int(cfg.model.seq_len),
            "seq_len_overridden": False,
        },
        "label_join_preview": join_rows,
        "descriptive": descriptive,
        "exact_validation_needle_chunk": exact_json,
        "gpu": asdict(gpu),
        "software": {
            "python": sys.version.split()[0],
            "jax": jax.__version__,
            "jaxlib": getattr(__import__("jaxlib"), "__version__", "unknown"),
        },
        "attention": attn_setup,
        "sanity": sanity,
        "runtime_s": runtime,
        "memory": memory,
        "g_bar": {
            "norm_per_chunk": g_bar_norms.tolist(),
            "clip_scale_c": clip_cs.tolist(),
            "eta": etas.tolist(),
            "clip_norm": clip_norm,
        },
    }
    stem.with_suffix(".json").write_text(json.dumps(to_jsonable(meta), indent=2) + "\n")
    print(f"\nraw:  {stem.with_suffix('.npz')}", flush=True)
    print(f"meta: {stem.with_suffix('.json')}", flush=True)
    print(f"wall: {runtime['wall_s']:.1f}s", flush=True)


if __name__ == "__main__":
    args = parse_args()
    main(attention_cudnn_override=args.attention_cudnn_override)
