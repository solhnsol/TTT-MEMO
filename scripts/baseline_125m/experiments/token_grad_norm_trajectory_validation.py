"""Multi-chunk / multi-seed JVP estimator robustness on one 8K TTT trajectory.

Not a saliency study. Does not edit e2e/. Does not overwrite STEP 2/3 results.

Question:

    On one 8K DCLM sequence, across 8 sequential official TTT evaluation
    points (W_i, S_i) and 4 Rademacher projection seeds, how stable is the
    JVP estimator of ||∇_{W_fast} l_t^TTT||_2?

These 8 chunks are sequential TTT states on one sequence, not independent
documents. G_t is a gradient-signal magnitude, not importance.

Run (canonical path; no extra XLA memory flags):

    docker compose run --rm -e PYTHONUNBUFFERED=1 ttt \\
      python scripts/baseline_125m/experiments/token_grad_norm_trajectory_validation.py

Expects, from repo root:

    checkpoints/125m_ttt_e2e_pretrain_dclm_8k_1x_cc
    data/_probe_val_head.bin
    data/tokenizer/llama3/tokenizer.json   # optional, for decoded top-20

Writes:

    results/raw/baseline_125m/experiments/token_grad_norm_trajectory_validation_t8192.{json,npz}
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
    PROBE_PATH,
    TOKENIZER_PATH,
    add_e2e_to_path,
    compose_baseline_config,
)
from measurement_utils import (  # noqa: E402
    TIE_RULE,
    cutoff_margin,
    error_metrics,
    leading_axis_size,
    pytree_l2_norm,
    pytree_max_abs_diff,
    rademacher_like,
    ranking_order,
    score_capture,
    slice_prime_layer,
    summarize_values,
    token_ttt_ce_from_logits,
    topk_overlap,
)
from runtime_helpers import AttentionOverride, describe_gpu, tokens_to_batch  # noqa: E402

STEP3_STEM = EXPERIMENT_RESULT_DIR / "token_grad_norm_full_validation_t1024"
SEQ_LEN = 8192
MINI_BATCH = 1024
N_CHUNKS = 8
PROJ_SEEDS = (0, 1, 2, 3)
K_LIST = (4, 8, 16, 32, 64)
K_MAX = 64
REL_EPS = 1e-12
TOP_FRACTIONS = (0.01, 0.05, 0.10)
PREV_JVP_REGRESSION_ABS = 1e-3
EXACT_PROGRESS_EVERY = 256


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--attention-cudnn-override",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Same RTX 3060 compatibility flag as forward_sanity.py.",
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


def try_load_tokenizer():
    if not TOKENIZER_PATH.exists():
        return None
    from tokenizers import Tokenizer

    return Tokenizer.from_file(str(TOKENIZER_PATH))


def decode_token_id(tokenizer, token_id: int) -> str | None:
    if tokenizer is None:
        return None
    return tokenizer.id_to_token(int(token_id))


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


def exact_distribution(g, valid) -> dict:
    import numpy as np

    x = np.asarray(g, dtype=np.float64)[np.asarray(valid).astype(bool)]
    return {
        "n": int(x.size),
        "min": float(np.min(x)),
        "p25": float(np.percentile(x, 25)),
        "median": float(np.median(x)),
        "p75": float(np.percentile(x, 75)),
        "p90": float(np.percentile(x, 90)),
        "p95": float(np.percentile(x, 95)),
        "max": float(np.max(x)),
        "mean": float(np.mean(x)),
    }


def make_ntp_batch(probe, *, bos_token_id: int, seq_len: int):
    """Build a length-`seq_len` NTP Batch from the local probe.

    Official Dataset fetches seq_len+1 tokens (`lm_dataset.Dataset.__getitem__`).
    `_probe_val_head.bin` is 8192 uint32 IDs, one short of 8193.

    If the probe has seq_len+1 tokens, this is the official `_to_batch` path.
    If it has exactly seq_len tokens, input_ids = probe, targets are the
    shifted stream, and the final position is masked invalid (no fabricated ID
    enters the loss). Chunk 0 then matches STEP 3: input_ids[t]=probe[t],
    target[t]=probe[t+1] for t=0..1023.
    """
    import jax.numpy as jnp
    import numpy as np
    from ttt.model.data import Batch

    probe = np.asarray(probe, dtype=np.int32)
    if probe.size >= seq_len + 1:
        seq = tokens_to_batch(
            jnp.asarray(probe[: seq_len + 1], dtype=jnp.int32),
            bos_token_id=int(bos_token_id),
        )
        return seq, {
            "probe_n_tokens": int(probe.size),
            "batch_T": seq_len,
            "construction": "official_to_batch_T_plus_1",
            "n_masked_because_probe_short": 0,
        }
    if probe.size != seq_len:
        raise SystemExit(f"probe has {probe.size} tokens; need {seq_len} or {seq_len + 1}")

    input_ids = probe.astype(np.int32)
    target_tokens = np.empty((seq_len,), dtype=np.int32)
    target_tokens[:-1] = probe[1:]
    target_tokens[-1] = 0
    loss_masks = np.ones((seq_len,), dtype=bool)
    loss_masks[:-1] = probe[1:] != bos_token_id
    loss_masks[-1] = False
    seq = Batch(
        input_ids=jnp.asarray(input_ids),
        target_tokens=jnp.asarray(target_tokens),
        loss_masks=jnp.asarray(loss_masks),
    )
    return seq, {
        "probe_n_tokens": int(probe.size),
        "batch_T": seq_len,
        "construction": "probe_len_eq_T_last_target_masked",
        "n_masked_because_probe_short": 1,
        "masked_position": seq_len - 1,
        "note": (
            "Official 8K sample is seq_len+1=8193 tokens. Local probe is 8192. "
            "Last NTP position is excluded from exact/JVP metrics. "
            "Chunks 0-6 are full 1024 valid if no BOS targets. Chunk 0 matches STEP 3."
        ),
    }


def direction_key(seed: int, direction_index: int, step3_seed0_keys):
    """PRNG for Rademacher direction (seed, k). Independent of K_max.

    Seed 0, k<32 uses STEP 3's `split(PRNGKey(0), 32)[k]` so chunk-0 K<=32
    can regress against the previous full-validation JVP. Every other
    (seed, k) uses fold_in(PRNGKey(seed), k).
    """
    import jax.random as jrandom

    if seed == 0 and direction_index < step3_seed0_keys.shape[0]:
        return step3_seed0_keys[direction_index]
    return jrandom.fold_in(jrandom.PRNGKey(int(seed)), int(direction_index))


def compare_vectors(a, b, valid, *, eps: float) -> dict:
    import numpy as np

    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    mask = np.asarray(valid).astype(bool)
    d = np.abs(a - b)
    rel = d[mask] / (np.abs(b[mask]) + eps)
    row = {
        "max_abs": float(np.max(d[mask])),
        "mean_abs": float(np.mean(d[mask])),
        "median_abs": float(np.median(d[mask])),
        "median_relative": float(np.median(rel)),
        "p95_relative": float(np.percentile(rel, 95)),
        "max_relative": float(np.max(rel)),
        "n": int(mask.sum()),
    }
    m = error_metrics(b, a, mask, eps=eps)
    row["spearman"] = m["spearman"]
    overlap = {}
    capture = {}
    for frac in TOP_FRACTIONS:
        ov = topk_overlap(b, a, mask, frac)
        cap = score_capture(b, a, mask, frac, eps=eps)
        overlap[str(frac)] = ov["topk_overlap"]
        capture[str(frac)] = cap["score_capture"]
    row["topk_overlap"] = overlap
    row["score_capture"] = capture
    return row


def main(attention_cudnn_override: bool = True) -> None:
    wall0 = time.perf_counter()
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
    from ttt.model.loss import token_log_probs
    from ttt.model.sharding import ModelSharding
    from ttt.model.transformer import BlockCollectionSplit, MetaModel
    from ttt.utils.filter_utils import get_filter_spec
    from ttt.utils.jax_utils import clone_pytree, eval_shape_and_sharding, initialize_distibuted, set_random_seed, tree_rearrange

    step3_npz = STEP3_STEM.with_suffix(".npz")
    if not step3_npz.exists():
        raise SystemExit(f"STEP 3 result missing: {step3_npz}")
    step3 = dict(np.load(step3_npz, allow_pickle=False))

    cfg.model.seq_len = cfg.training.seq_length
    mini_batch = int(cfg.model.mini_batch_size)
    if mini_batch != MINI_BATCH:
        raise SystemExit(f"expected mini_batch={MINI_BATCH}, got {mini_batch}")
    if SEQ_LEN % mini_batch != 0 or SEQ_LEN // mini_batch != N_CHUNKS:
        raise SystemExit(f"SEQ_LEN={SEQ_LEN} is not {N_CHUNKS} chunks of {mini_batch}")

    probe = np.fromfile(PROBE_PATH, dtype="<u4")
    print("--- probe / sequence ---", flush=True)
    print(f"  {PROBE_PATH}  n_uint32={probe.size}", flush=True)

    initialize_distibuted(cfg.backend)
    gpu = describe_gpu()
    override = AttentionOverride(enabled=attention_cudnn_override)
    override.install()

    seq, batch_meta = make_ntp_batch(probe, bos_token_id=int(cfg.model.bos_token_id), seq_len=SEQ_LEN)
    print(f"  batch construction: {batch_meta['construction']}", flush=True)
    print(f"  T={int(seq.input_ids.shape[0])}  masked_short={batch_meta['n_masked_because_probe_short']}", flush=True)

    key = set_random_seed(cfg.training.model_seed)
    model_sharding = ModelSharding(cfg)
    mesh = model_sharding.mesh
    tokenizer = try_load_tokenizer()
    step3_seed0_keys = jrandom.split(jrandom.PRNGKey(0), 32)

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

    # Official eval inner-LR: skip warmup. Measurement (exact/JVP) does not
    # read step_index; trajectory advancement via inner_loop_step does.
    state = state.set(model.step_index, jnp.array(jnp.iinfo(jnp.int32).max - 100, dtype=jnp.int32))

    print("--- setup ---", flush=True)
    print(f"GPU: {gpu.name}  jax={gpu.jax_device}", flush=True)
    print(f"T={SEQ_LEN} chunks={N_CHUNKS}  compute={cfg.model.compute_dtype}", flush=True)
    print(f"proj seeds={list(PROJ_SEEDS)}  K={list(K_LIST)}", flush=True)
    print("step_index: eval (int32.max-100) for inner_loop_step only", flush=True)

    memory = {"before_compute": try_memory_stats()}
    runtime: dict = {
        "exact_per_chunk_s": [],
        "exact_per_token_times_s": [],
        "jvp_per_direction_times_s": [],
        "jvp_per_chunk_seed_s": {},
        "inner_update_per_chunk_s": [],
    }
    sanity: dict = {"per_chunk": []}
    trajectory: list[dict] = []

    with mesh:
        block_collection = model.language_model.model.h.blocks
        prime_storage = model.language_model.model.h.prime_storage if cfg.model.prime else None
        if prime_storage is None:
            raise SystemExit("checkpoint has prime=False; this measurement needs feed_forward_prime")
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
        global_layer_map = [int(prefix_len + i) for i in range(suffix_len)]
        spec_inner = get_filter_spec(model_outer, cfg.training.spec_inner, "inner parameters")
        inspect_inner0, _ = eqx.partition(model_outer, spec_inner)
        inspect_inner = inspect_inexact_paths(inspect_inner0)
        n_fast = sum(r["size"] for r in inspect_inner)
        inner_paths = {r["path"] for r in inspect_inner}
        if not any("suffix_blocks.feed_forward_prime" in p for p in inner_paths):
            raise SystemExit("spec_inner did not select suffix_blocks.feed_forward_prime")
        if any("ffn_prime_norm" in p for p in inner_paths):
            raise SystemExit("spec_inner unexpectedly selected prime norms")
        print(f"  P={prefix_len} S={suffix_len} layers={global_layer_map}  #W_fast={n_fast:,}", flush=True)

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
            "prefix_call T=8192",
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

        def token_ttt_ce_only(inner, state, prefix_chunk, seq_chunk):
            model_i = eqx.combine(inner, outer_params)
            out = model_i.language_model.suffix_call(
                prefix_outputs=prefix_chunk, state=state, seq=seq_chunk
            )
            return token_ttt_ce_from_logits(out.logits, seq_chunk.target_tokens), out.logits

        def token_ce_at_index(inner, idx, state, prefix_chunk, seq_chunk):
            ce, _logits = token_ttt_ce_only(inner, state, prefix_chunk, seq_chunk)
            return ce[idx]

        def jvp_token_ce(inner, tangent, state, prefix_chunk, seq_chunk):
            def f(w):
                ce, _logits = token_ttt_ce_only(w, state, prefix_chunk, seq_chunk)
                return ce

            return eqx.filter_jvp(f, (inner,), (tangent,))

        suffix_jit = eqx.filter_jit(token_ttt_ce_only)
        grad_token_jit = eqx.filter_jit(eqx.filter_grad(token_ce_at_index))
        jvp_jit = eqx.filter_jit(jvp_token_ce)

        input_ids = np.zeros((N_CHUNKS, MINI_BATCH), dtype=np.int32)
        target_ids = np.zeros((N_CHUNKS, MINI_BATCH), dtype=np.int32)
        valid_masks = np.zeros((N_CHUNKS, MINI_BATCH), dtype=np.int8)
        token_ttt_ce = np.zeros((N_CHUNKS, MINI_BATCH), dtype=np.float32)
        legacy_nll = np.zeros((N_CHUNKS, MINI_BATCH), dtype=np.float32)
        exact_norm = np.full((N_CHUNKS, MINI_BATCH), np.nan, dtype=np.float32)
        exact_norm_sq = np.full((N_CHUNKS, MINI_BATCH), np.nan, dtype=np.float32)
        exact_layer = np.full((N_CHUNKS, MINI_BATCH, suffix_len), np.nan, dtype=np.float32)
        jv_projections = np.zeros((N_CHUNKS, len(PROJ_SEEDS), K_MAX, MINI_BATCH), dtype=np.float32)
        estimated_norm = {k: np.zeros((N_CHUNKS, len(PROJ_SEEDS), MINI_BATCH), dtype=np.float32) for k in K_LIST}
        estimated_sq = {k: np.zeros((N_CHUNKS, len(PROJ_SEEDS), MINI_BATCH), dtype=np.float32) for k in K_LIST}

        exact_compiled = False
        jvp_compiled = False
        update_compiled = False
        metrics_table: dict = {}
        ranking_diag: dict = {}
        cutoff_table: dict = {}

        for c in range(N_CHUNKS):
            print(f"\n========== chunk {c} ==========", flush=True)
            seq_chunk = tree_slice(seq_chunks, c)
            prefix_chunk = prefix_chunks[c]
            model_meas = recombine_inner(model_inner, model_outer)
            inner_w, _ = eqx.partition(model_meas, spec_inner)
            s_canonical = clone_pytree(suffix_state)

            def fresh_state(s=s_canonical):
                return clone_pytree(s)

            inner_before = clone_pytree(inner_w)
            state_before = clone_pytree(s_canonical)

            (ce, logits), t_f = timed(
                f"chunk{c} f(Wi)",
                lambda: suffix_jit(inner_w, fresh_state(), prefix_chunk, seq_chunk),
                warmup=(c == 0),
            )
            (y2, _logits2), _ = timed(
                f"chunk{c} f(Wi) repeat",
                lambda: suffix_jit(inner_w, fresh_state(), prefix_chunk, seq_chunk),
                quiet=True,
            )
            ce_np = np.asarray(ce)
            y2_np = np.asarray(y2)
            nll_np = np.asarray(-token_log_probs(logits, seq_chunk.target_tokens))
            valid = np.asarray(seq_chunk.loss_masks).astype(bool)
            n_valid = int(valid.sum())
            input_ids[c] = np.asarray(seq_chunk.input_ids)
            target_ids[c] = np.asarray(seq_chunk.target_tokens)
            valid_masks[c] = valid.astype(np.int8)
            token_ttt_ce[c] = ce_np.astype(np.float32)
            legacy_nll[c] = nll_np
            print(f"  n_valid={n_valid}/{MINI_BATCH}  f_repeat_max_abs={float(np.max(np.abs(ce_np - y2_np))):.3e}", flush=True)

            if c == 0:
                ce_diff = float(np.max(np.abs(ce_np.astype(np.float64) - np.asarray(step3["token_ttt_ce"]).astype(np.float64))))
                ids_diff = float(np.max(np.abs(input_ids[c] - np.asarray(step3["input_ids"]))))
                print(f"  vs STEP3 chunk0: token_ttt_ce max_abs={ce_diff:.3e}  input_ids max_abs={ids_diff:.3e}", flush=True)
                if ids_diff != 0.0:
                    raise SystemExit("chunk 0 input_ids do not match STEP 3; refuse to continue")
                chunk0_vs_step3 = {
                    "input_ids_match": True,
                    "token_ttt_ce_max_abs": ce_diff,
                    "prefix_note": (
                        "This run uses official prefix_call on T=8192 then slices chunk 0. "
                        "STEP 3 compiled prefix at T=1024. Causal SWA should agree in exact "
                        "arithmetic; bf16 XLA kernels need not be bit-identical. "
                        "Not used as a refuse-to-run criterion."
                    ),
                }

            valid_positions = np.flatnonzero(valid).astype(np.int32)
            print(f"--- exact G_t chunk {c} ({n_valid} valid) ---", flush=True)
            t0 = int(valid_positions[0])
            g0, dt0 = timed(
                f"exact chunk{c} t={t0}",
                lambda: grad_token_jit(inner_w, jnp.int32(t0), fresh_state(), prefix_chunk, seq_chunk),
                warmup=not exact_compiled,
            )
            if not exact_compiled:
                runtime["exact_compile_s"] = dt0
                exact_compiled = True
                print(f"  exact compile+first={dt0:.3f}s", flush=True)
                remaining_s = 0.0
            else:
                remaining_s = dt0
                runtime["exact_per_token_times_s"].append(dt0)

            g_t = g0
            chunk_exact_s = dt0
            for i, t_idx in enumerate(valid_positions):
                if i == 0:
                    g_t = g0
                else:
                    g_t, dt = timed(
                        f"exact t={int(t_idx)}",
                        lambda t=int(t_idx): grad_token_jit(
                            inner_w, jnp.int32(t), fresh_state(), prefix_chunk, seq_chunk
                        ),
                        quiet=True,
                    )
                    remaining_s += dt
                    chunk_exact_s += dt
                    runtime["exact_per_token_times_s"].append(dt)
                total = float(pytree_l2_norm(g_t))
                exact_norm[c, t_idx] = total
                exact_norm_sq[c, t_idx] = total * total
                prime_g = g_t.language_model.model.h.suffix_blocks.feed_forward_prime
                for layer_i in range(suffix_len):
                    exact_layer[c, t_idx, layer_i] = float(pytree_l2_norm(slice_prime_layer(prime_g, layer_i)))
                if i == 0 or (i + 1) % EXACT_PROGRESS_EVERY == 0 or i + 1 == n_valid:
                    print(f"  exact chunk{c} {i + 1}/{n_valid}  last G={total:.4f}", flush=True)

            runtime["exact_per_chunk_s"].append(chunk_exact_s)
            print(f"  exact chunk{c} total={chunk_exact_s:.3f}s", flush=True)

            print(f"--- JVP chunk {c} seeds={list(PROJ_SEEDS)} K_max={K_MAX} ---", flush=True)
            for s_i, seed in enumerate(PROJ_SEEDS):
                seed_t0 = time.perf_counter()
                for k_i in range(K_MAX):
                    v_k = rademacher_like(inner_w, direction_key(seed, k_i, step3_seed0_keys))
                    warmup = (not jvp_compiled) and s_i == 0 and k_i == 0
                    (_, jv_k), dt = timed(
                        f"jvp c{c} seed{seed} dir{k_i}",
                        lambda v=v_k: jvp_jit(inner_w, v, fresh_state(), prefix_chunk, seq_chunk),
                        warmup=warmup,
                        quiet=not warmup,
                    )
                    if warmup:
                        runtime["jvp_compile_s"] = dt
                        jvp_compiled = True
                        print(f"  jvp compile+first={dt:.3f}s", flush=True)
                    else:
                        runtime["jvp_per_direction_times_s"].append(dt)
                    jv_projections[c, s_i, k_i] = np.asarray(jv_k, dtype=np.float32)
                sumsq = np.zeros((MINI_BATCH,), dtype=np.float64)
                for k_i in range(K_MAX):
                    jv_f = jv_projections[c, s_i, k_i].astype(np.float64)
                    sumsq = sumsq + jv_f * jv_f
                    k_now = k_i + 1
                    if k_now in K_LIST:
                        ghat_sq = sumsq / float(k_now)
                        estimated_sq[k_now][c, s_i] = ghat_sq.astype(np.float32)
                        estimated_norm[k_now][c, s_i] = np.sqrt(ghat_sq).astype(np.float32)
                seed_elapsed = time.perf_counter() - seed_t0
                runtime["jvp_per_chunk_seed_s"].setdefault(str(c), {})[str(seed)] = seed_elapsed
                print(f"  jvp chunk{c} seed={seed} wall={seed_elapsed:.3f}s", flush=True)

            w_after_meas = pytree_max_abs_diff(inner_before, inner_w)
            s_after_meas = pytree_max_abs_diff(state_before, s_canonical)
            chunk_sanity = {
                "chunk": c,
                "n_valid": n_valid,
                "f_repeat_max_abs": float(np.max(np.abs(ce_np - y2_np))),
                "token_ttt_ce_dtype": str(ce_np.dtype),
                "legacy_nll_dtype": str(nll_np.dtype),
                "finite_ce": finite_flags("token_ttt_ce", ce_np),
                "finite_exact": finite_flags("exact_valid", exact_norm[c, valid]),
                "finite_jvp": finite_flags("jvp", jv_projections[c]),
                "exact_nonnegative": bool(np.all(exact_norm[c, valid] >= 0)),
                "jvp_estimates_nonnegative": bool(np.all(estimated_norm[K_MAX][c] >= 0)),
                "W_unchanged_during_measurement": w_after_meas,
                "S_unchanged_during_measurement": s_after_meas,
            }
            sanity["per_chunk"].append(chunk_sanity)
            if w_after_meas != 0.0 or s_after_meas != 0.0:
                print(f"  PURITY: W diff={w_after_meas:.3e} S diff={s_after_meas:.3e}", flush=True)

            dist = exact_distribution(exact_norm[c], valid)
            print(
                f"  exact G: min={dist['min']:.4f} med={dist['median']:.4f} "
                f"p90={dist['p90']:.4f} max={dist['max']:.4f}",
                flush=True,
            )
            cutoff_table[str(c)] = {str(frac): cutoff_margin(exact_norm[c], valid, frac, eps=REL_EPS) for frac in TOP_FRACTIONS}
            metrics_table[str(c)] = {}
            for s_i, seed in enumerate(PROJ_SEEDS):
                metrics_table[str(c)][str(seed)] = {}
                for k in K_LIST:
                    m = error_metrics(exact_norm[c], estimated_norm[k][c, s_i], valid, eps=REL_EPS)
                    ov = {str(frac): topk_overlap(exact_norm[c], estimated_norm[k][c, s_i], valid, frac) for frac in TOP_FRACTIONS}
                    cap = {str(frac): score_capture(exact_norm[c], estimated_norm[k][c, s_i], valid, frac, eps=REL_EPS) for frac in TOP_FRACTIONS}
                    metrics_table[str(c)][str(seed)][str(k)] = {
                        "error": m,
                        "topk_overlap": ov,
                        "score_capture": cap,
                    }
                if seed == 0:
                    k32 = metrics_table[str(c)][str(seed)]["32"]
                    print(
                        f"  seed0 K=32 spearman={k32['error']['spearman']:.4f} "
                        f"rel_med={k32['error']['relative_error_median']:.4f} "
                        f"top1%={k32['topk_overlap']['0.01']['topk_overlap']:.3f} "
                        f"C1%={k32['score_capture']['0.01']['score_capture']:.3f}",
                        flush=True,
                    )

            order_exact = valid_positions[ranking_order(exact_norm[c, valid_positions])]

            def top_rows(order, n=20):
                rows = []
                for t in order[:n]:
                    t = int(t)
                    tgt = int(target_ids[c, t])
                    rows.append(
                        {
                            "position": t,
                            "target_token_id": tgt,
                            "decoded_target": decode_token_id(tokenizer, tgt),
                            "token_ttt_ce": float(token_ttt_ce[c, t]),
                            "legacy_nll": float(legacy_nll[c, t]),
                            "exact_grad_norm": float(exact_norm[c, t]),
                        }
                    )
                return rows

            ranking_diag[str(c)] = {
                "exact_top20": top_rows(order_exact),
                "k32_seed0_top20_positions": valid_positions[
                    ranking_order(estimated_norm[32][c, 0][valid_positions])
                ][:20].tolist(),
            }

            w_before_upd = clone_pytree(inner_w)
            s_before_upd = clone_pytree(suffix_state)
            out_step, dt_upd = timed(
                f"inner_loop_step chunk{c}",
                lambda: official_chunk_step(
                    model_inner, inner_opt_state, (state_all, suffix_state), seq_chunk, prefix_chunk, model_outer
                ),
                warmup=not update_compiled,
            )
            if not update_compiled:
                runtime["inner_update_compile_s"] = dt_upd
                update_compiled = True
            runtime["inner_update_per_chunk_s"].append(dt_upd)
            new_inner, _ = eqx.partition(recombine_inner(out_step.new_model, model_outer), spec_inner)
            w_changed = pytree_max_abs_diff(w_before_upd, new_inner)
            s_changed = pytree_max_abs_diff(s_before_upd, out_step.new_state[1])
            w_l2 = float(pytree_l2_norm(w_before_upd))
            dW = jax.tree.map(
                lambda a, b: (a - b) if eqx.is_inexact_array(a) else a,
                w_before_upd,
                new_inner,
            )
            dw_l2 = float(pytree_l2_norm(dW))
            traj = {
                "chunk": c,
                "W_max_abs_change": w_changed,
                "W_l2_before": w_l2,
                "dW_l2": dw_l2,
                "S_max_abs_change": s_changed,
                "inner_update_s": dt_upd,
                "official_chunk_loss": float(out_step.metrics[MetaModel.MetricType.loss]),
            }
            trajectory.append(traj)
            print(
                f"  trajectory update: dW_max={w_changed:.3e} dW_l2={dw_l2:.4f} "
                f"dS_max={s_changed:.3e}  (sanity only)",
                flush=True,
            )
            if w_changed == 0.0:
                print("  WARNING: W_{i+1} equals W_i after official inner_loop_step", flush=True)

            model_inner = out_step.new_model
            inner_opt_state = out_step.new_optimizer_state
            state_all, suffix_state = out_step.new_state

        # ---- aggregates ----
        def collect(k, getter):
            vals = []
            for c in range(N_CHUNKS):
                for seed in PROJ_SEEDS:
                    vals.append(getter(metrics_table[str(c)][str(seed)][str(k)]))
            return vals

        aggregate = {}
        per_chunk_seed_summary = {}
        seed_stability = {}
        for k in K_LIST:
            spear = collect(k, lambda r: r["error"]["spearman"])
            relmed = collect(k, lambda r: r["error"]["relative_error_median"])
            aggregate[str(k)] = {
                "spearman": summarize_values(spear),
                "pearson": summarize_values(collect(k, lambda r: r["error"]["pearson"])),
                "relative_error_median": summarize_values(relmed),
                "relative_error_mean": summarize_values(collect(k, lambda r: r["error"]["relative_error_mean"])),
                "topk_overlap": {
                    str(frac): summarize_values(
                        collect(k, lambda r, f=frac: r["topk_overlap"][str(f)]["topk_overlap"])
                    )
                    for frac in TOP_FRACTIONS
                },
                "score_capture": {
                    str(frac): summarize_values(
                        collect(k, lambda r, f=frac: r["score_capture"][str(f)]["score_capture"])
                    )
                    for frac in TOP_FRACTIONS
                },
            }
            per_chunk_seed_summary[str(k)] = {}
            seed_stability[str(k)] = {}
            for c in range(N_CHUNKS):
                spears_c = [metrics_table[str(c)][str(seed)][str(k)]["error"]["spearman"] for seed in PROJ_SEEDS]
                ov_c = {
                    str(frac): [metrics_table[str(c)][str(seed)][str(k)]["topk_overlap"][str(frac)]["topk_overlap"] for seed in PROJ_SEEDS]
                    for frac in TOP_FRACTIONS
                }
                cap_c = {
                    str(frac): [metrics_table[str(c)][str(seed)][str(k)]["score_capture"][str(frac)]["score_capture"] for seed in PROJ_SEEDS]
                    for frac in TOP_FRACTIONS
                }
                per_chunk_seed_summary[str(k)][str(c)] = {
                    "spearman_mean": float(np.mean(spears_c)),
                    "spearman_std": float(np.std(spears_c, ddof=0)),
                    "topk_overlap_mean": {str(frac): float(np.mean(ov_c[str(frac)])) for frac in TOP_FRACTIONS},
                    "topk_overlap_std": {str(frac): float(np.std(ov_c[str(frac)], ddof=0)) for frac in TOP_FRACTIONS},
                    "score_capture_mean": {str(frac): float(np.mean(cap_c[str(frac)])) for frac in TOP_FRACTIONS},
                    "score_capture_std": {str(frac): float(np.std(cap_c[str(frac)], ddof=0)) for frac in TOP_FRACTIONS},
                }
                seed_stability[str(k)][str(c)] = {
                    "spearman_std_across_seeds": float(np.std(spears_c, ddof=0)),
                    "topk_overlap_std_across_seeds": {str(frac): float(np.std(ov_c[str(frac)], ddof=0)) for frac in TOP_FRACTIONS},
                    "score_capture_std_across_seeds": {str(frac): float(np.std(cap_c[str(frac)], ddof=0)) for frac in TOP_FRACTIONS},
                }

        exact_token_times = np.asarray(runtime["exact_per_token_times_s"], dtype=np.float64)
        jvp_dir_times = np.asarray(runtime["jvp_per_direction_times_s"], dtype=np.float64)
        runtime["exact_steady_mean_s"] = float(np.mean(exact_token_times)) if exact_token_times.size else None
        runtime["exact_steady_median_s"] = float(np.median(exact_token_times)) if exact_token_times.size else None
        runtime["exact_all_chunks_s"] = float(np.sum(runtime["exact_per_chunk_s"]))
        runtime["jvp_steady_mean_s"] = float(np.mean(jvp_dir_times)) if jvp_dir_times.size else None
        runtime["jvp_steady_median_s"] = float(np.median(jvp_dir_times)) if jvp_dir_times.size else None
        # Compile-excluded JVP_K time: K * steady per-direction. Exact full-chunk: mean of 8 chunk totals.
        exact_chunk_mean = float(np.mean(runtime["exact_per_chunk_s"]))
        runtime["exact_full_chunk_mean_s"] = exact_chunk_mean
        runtime["jvp_cumulative_K_s"] = {}
        runtime["speedup_K_compile_excluded"] = {}
        jvp_mean = runtime["jvp_steady_mean_s"] or 0.0
        for k in K_LIST:
            jvp_k = jvp_mean * k
            runtime["jvp_cumulative_K_s"][str(k)] = jvp_k
            runtime["speedup_K_compile_excluded"][str(k)] = (exact_chunk_mean / jvp_k) if jvp_k > 0 else None
        runtime["jvp_total_direction_calls"] = int(N_CHUNKS * len(PROJ_SEEDS) * K_MAX)
        runtime["hardware_note"] = "Timing is machine-specific; not a performance claim"

        noise_ceiling = None
        chunk0_valid = valid_masks[0].astype(bool)
        noise_ceiling = compare_vectors(exact_norm[0], step3["exact_grad_norm"], chunk0_valid, eps=REL_EPS)
        noise_ceiling["note"] = (
            "chunk-0 exact vs STEP 3 exact. bf16 reverse-mode run-to-run reference, "
            "not a JVP accuracy number."
        )
        print("\n--- chunk-0 exact vs STEP 3 noise ceiling ---", flush=True)
        print(
            f"  max_abs={noise_ceiling['max_abs']:.3e} med_rel={noise_ceiling['median_relative']:.3e} "
            f"spearman={noise_ceiling['spearman']}",
            flush=True,
        )

        jvp_regression = {"per_k": {}}
        print("\n--- chunk-0 seed-0 JVP vs STEP 3 ---", flush=True)
        worst = 0.0
        for k in (4, 8, 16, 32):
            prev = np.asarray(step3[f"estimated_grad_norm_k{k}"], dtype=np.float64)
            cur = estimated_norm[k][0, 0].astype(np.float64)
            max_abs = float(np.max(np.abs(prev - cur)))
            mean_abs = float(np.mean(np.abs(prev - cur)))
            jvp_regression["per_k"][str(k)] = {"max_abs": max_abs, "mean_abs": mean_abs}
            worst = max(worst, max_abs)
            print(f"  K={k:2d} max_abs={max_abs:.3e} mean_abs={mean_abs:.3e}", flush=True)
        jvp_regression["max_abs_any_k"] = worst
        jvp_regression["flagged"] = worst > PREV_JVP_REGRESSION_ABS
        jvp_regression["prng_note"] = (
            "seed 0 directions 0..31 use STEP 3 split(PRNGKey(0), 32); "
            "directions 32..63 and seeds 1-3 use fold_in(PRNGKey(seed), k)"
        )
        if jvp_regression["flagged"]:
            print(
                f"REGRESSION: chunk-0 seed-0 JVP differs from STEP 3 by max abs {worst:.3e}. "
                "Do not treat trajectory robustness as confirmed until explained.",
                flush=True,
            )

        memory["after_compute"] = try_memory_stats()
        attention_cfg = override.report_for_config(
            force_flash=bool(cfg.model.force_flash),
            seq_modeling_block=str(cfg.model.seq_modeling_block),
        )
        runtime["wall_s"] = time.perf_counter() - wall0

        EXPERIMENT_RESULT_DIR.mkdir(parents=True, exist_ok=True)
        stem = EXPERIMENT_RESULT_DIR / f"token_grad_norm_trajectory_validation_t{SEQ_LEN}"
        npz_kwargs = {
            "chunk_indices": np.arange(N_CHUNKS, dtype=np.int32),
            "input_ids": input_ids,
            "target_ids": target_ids,
            "valid_masks": valid_masks,
            "token_ttt_ce": token_ttt_ce,
            "legacy_token_nll": legacy_nll,
            "exact_grad_norm": exact_norm,
            "exact_grad_norm_sq": exact_norm_sq,
            "exact_layerwise_norm": exact_layer,
            "jv_projections": jv_projections,
            "projection_seeds": np.asarray(PROJ_SEEDS, dtype=np.int32),
            "k_values": np.asarray(K_LIST, dtype=np.int32),
            "global_layer_index": np.asarray(global_layer_map, dtype=np.int32),
        }
        for k in K_LIST:
            npz_kwargs[f"estimated_norm_k{k}"] = estimated_norm[k]
            npz_kwargs[f"estimated_norm_sq_k{k}"] = estimated_sq[k]
        np.savez(stem.with_suffix(".npz"), **npz_kwargs)

        meta = {
            "hypothesis": (
                "JVP Rademacher estimates of token-wise ||∇_{W_fast} l_t^TTT||_2 remain "
                "stable across 8 sequential TTT evaluation points on one 8K DCLM sequence "
                "and 4 projection seeds. Not cross-document generalization. G_t is not importance."
            ),
            "seq_len": SEQ_LEN,
            "n_chunks": N_CHUNKS,
            "mini_batch_size": mini_batch,
            "unit_of_analysis": "one_8k_sequence_eight_sequential_ttt_states",
            "checkpoint_dir": str(CHECKPOINT_DIR),
            "probe_path": str(PROBE_PATH),
            "batch_construction": batch_meta,
            "spec_inner": list(cfg.training.spec_inner),
            "n_fastweight_scalars": n_fast,
            "wfast_runtime_spec_inner": inspect_inner,
            "prefix_len": prefix_len,
            "suffix_len": suffix_len,
            "global_layer_map": global_layer_map,
            "projection_seeds": list(PROJ_SEEDS),
            "k_list": list(K_LIST),
            "rademacher": {
                "distribution": "independent ±1 per inner parameter element",
                "normalized": False,
                "dtype_rule": "match inner leaf dtype",
                "prng": jvp_regression["prng_note"],
            },
            "loss_definitions": {
                "primary": "l_t^TTT = -log_softmax(logits.astype(fp32))[target_t], unmasked",
                "legacy_diagnostic": "official -token_log_probs(logits, target)",
            },
            "topk": {
                "fractions": list(TOP_FRACTIONS),
                "metric": "topk_overlap = |TopExact ∩ TopEstimated| / k",
                "score_capture": "sum_{t in Shat} exact G_t / sum_{t in S*} exact G_t",
                "tie_rule": TIE_RULE,
            },
            "relative_error": "|Ghat-G| / (G + eps)",
            "rel_eps": REL_EPS,
            "dtypes": {
                "param_dtype_config": str(cfg.model.param_dtype),
                "compute_dtype_config": str(cfg.model.compute_dtype),
                "state_dtype_config": str(cfg.model.state_dtype),
                "inner_eval_cast": str(model_outer.state_dtype),
                "norm_accumulation_dtype": "float32",
            },
            "step_index": {
                "set_to_eval_warmup_value": True,
                "value": "int32.max - 100",
                "reason": "official eval inner LR for trajectory inner_loop_step; not used by exact/JVP measurement",
            },
            "trajectory_advance": "official MetaModel.inner_loop_step after each chunk's measurements",
            "gpu": asdict(gpu),
            "attention": attention_cfg,
            "tokenizer_path": str(TOKENIZER_PATH) if tokenizer is not None else None,
            "exact_g_distribution_by_chunk": {
                str(c): exact_distribution(exact_norm[c], valid_masks[c].astype(bool)) for c in range(N_CHUNKS)
            },
            "cutoff_margin_by_chunk": cutoff_table,
            "metrics_by_chunk_seed_k": metrics_table,
            "aggregate_over_chunk_and_seed": aggregate,
            "per_chunk_across_seeds": per_chunk_seed_summary,
            "seed_stability_by_chunk": seed_stability,
            "ranking_diagnostic_by_chunk": ranking_diag,
            "chunk0_vs_step3_prefix": chunk0_vs_step3,
            "chunk0_exact_noise_ceiling_vs_step3": noise_ceiling,
            "chunk0_seed0_jvp_regression_vs_step3": jvp_regression,
            "trajectory_sanity": trajectory,
            "sanity": sanity,
            "runtime_s": runtime,
            "memory": memory,
        }
        stem.with_suffix(".json").write_text(json.dumps(to_jsonable(meta), indent=2) + "\n")
        print(f"\nraw:  {stem.with_suffix('.npz')}", flush=True)
        print(f"meta: {stem.with_suffix('.json')}", flush=True)
        print(f"wall: {runtime['wall_s']:.1f}s", flush=True)
        print("\n--- aggregate Spearman ---", flush=True)
        for k in K_LIST:
            s = aggregate[str(k)]["spearman"]
            print(
                f"  K={k:2d} med={s['median']:.4f} mean={s['mean']:.4f} "
                f"std={s['std']:.4f} min={s['min']:.4f} max={s['max']:.4f}",
                flush=True,
            )


if __name__ == "__main__":
    args = parse_args()
    main(attention_cudnn_override=args.attention_cudnn_override)
