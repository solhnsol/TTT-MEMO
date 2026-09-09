"""STEP 6C.1: validate chunk-gradient alignment metrics on DCLM T=1024 chunk 0.

Not a saliency / future-utility / long-range-memory result. Does not edit e2e/.
Does not overwrite STEP 2/3 files. Does not run RULER.

Question:

    At the official bf16 frozen T=1024 chunk-0 evaluation point, does one
    JVP along stop-gradient g_bar = ∇_{W_fast} L_chunk recover reverse-mode
    A_t = <g_t, g_bar> well enough to use as a scalable alignment signal?

Also records H2 interference I_t = A_t - G_t^2 / N on the same 32-token
exact subset, and an optional first-order CE-change diagnostic after one
official inner SGD step with original chunk KV held fixed.
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
    token_ttt_ce_from_logits,
)
from runtime_helpers import AttentionOverride, describe_gpu, tokens_to_batch  # noqa: E402

STEP3_CUDNN_STEM = EXPERIMENT_RESULT_DIR / "token_grad_norm_full_validation_t1024_a100_cudnn"
STEP3_CANONICAL_STEM = EXPERIMENT_RESULT_DIR / "token_grad_norm_full_validation_t1024"
SEQ_LEN = 1024
CHUNK_INDEX = 0
JVP_SEED = 0
K_G = 32
N_EXACT = 32
REL_EPS = 1e-12
R_EPS = 1e-12
EVAL_POINT_CE_ABS = 1e-3
GHAT_VS_STEP3_ABS = 1e-3


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--attention-cudnn-override",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Default off: official cuDNN prefix SWA on A100. Pass --attention-cudnn-override for XLA.",
    )
    parser.add_argument(
        "--skip-first-order",
        action="store_true",
        help="Skip the optional inner_loop_step ΔCE diagnostic.",
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
    """Same trigger as optax.clip_by_global_norm: unclipped iff ||g|| < max_norm."""
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
        "n_exact_zero": int(np.sum(e == 0)),
        "neg_recovered": int(np.sum(neg & (h < 0))),
        "neg_recall": float(np.sum(neg & (h < 0)) / np.sum(neg)) if np.any(neg) else None,
        "pos_recovered": int(np.sum(pos & (h > 0))),
        "pos_recall": float(np.sum(pos & (h > 0)) / np.sum(pos)) if np.any(pos) else None,
    }


def classify_interference(i_vals):
    import numpy as np

    x = np.asarray(i_vals, dtype=np.float64)
    out = np.empty(x.shape, dtype=object)
    out[x > 0] = "reinforcement"
    out[x < 0] = "suppression"
    out[x == 0] = "zero"
    return out


def class_agreement(exact_i, estimated_i) -> dict:
    import numpy as np

    ce = classify_interference(exact_i)
    ch = classify_interference(estimated_i)
    agree = ce == ch
    return {
        "n": int(ce.size),
        "agree": float(np.mean(agree)) if ce.size else None,
        "n_exact_suppression": int(np.sum(ce == "suppression")),
        "n_exact_reinforcement": int(np.sum(ce == "reinforcement")),
        "n_exact_zero": int(np.sum(ce == "zero")),
        "suppression_recovered": int(np.sum((ce == "suppression") & (ch == "suppression"))),
        "reinforcement_recovered": int(np.sum((ce == "reinforcement") & (ch == "reinforcement"))),
    }


def stop_gradient_tree(tree):
    import equinox as eqx
    import jax

    return jax.tree.map(lambda x: jax.lax.stop_gradient(x) if eqx.is_inexact_array(x) else x, tree)


def main(*, attention_cudnn_override: bool, skip_first_order: bool) -> None:
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
    from ttt.model.sharding import ModelSharding
    from ttt.model.transformer import BlockCollectionSplit, MetaModel
    from ttt.utils.filter_utils import get_filter_spec
    from ttt.utils.jax_utils import clone_pytree, eval_shape_and_sharding, initialize_distibuted, set_random_seed, tree_rearrange

    cfg.model.seq_len = cfg.training.seq_length
    mini_batch = int(cfg.model.mini_batch_size)
    if SEQ_LEN != mini_batch:
        raise SystemExit(f"this validation is T=chunk=mini_batch ({mini_batch}), got T={SEQ_LEN}")

    initialize_distibuted(cfg.backend)
    gpu = describe_gpu()
    override = AttentionOverride(enabled=attention_cudnn_override)
    override.install()

    probe = np.fromfile(PROBE_PATH, dtype="<u4")
    if probe.size < SEQ_LEN + 1:
        raise SystemExit(f"{PROBE_PATH} has {probe.size} tokens, need {SEQ_LEN + 1}")
    tokens = jnp.asarray(probe[: SEQ_LEN + 1], dtype=jnp.int32)
    seq = tokens_to_batch(tokens, bos_token_id=int(cfg.model.bos_token_id))

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

    eval_step = int(jnp.iinfo(jnp.int32).max - 100)
    clip_norm = float(cfg.training.optimizer_inner.clip_gradient)
    inner_lr = float(cfg.training.optimizer_inner.lr)

    print("--- setup ---", flush=True)
    print(f"GPU: {gpu.name}  jax={gpu.jax_device}", flush=True)
    print(f"jax={jax.__version__}  jaxlib={getattr(__import__('jaxlib'), '__version__', 'unknown')}", flush=True)
    print(f"T/chunk={SEQ_LEN}/{CHUNK_INDEX}  compute={cfg.model.compute_dtype}", flush=True)
    attn_setup = override.report_for_config(
        force_flash=bool(cfg.model.force_flash),
        seq_modeling_block=str(cfg.model.seq_modeling_block),
    )
    print(
        f"attention override_enabled={attn_setup['override_enabled']} "
        f"prefix_effective={attn_setup['prefix_effective_backend']}",
        flush=True,
    )
    print(f"step_index: eval {eval_step} (inner LR warmup complete)", flush=True)

    memory = {"before_compute": try_memory_stats()}
    runtime: dict = {}
    sanity: dict = {}

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

        prefix_len = leading_axis_size(model_outer.language_model.model.h.prefix_blocks)
        suffix_len = leading_axis_size(model_outer.language_model.model.h.suffix_blocks)
        spec_inner = get_filter_spec(model_outer, cfg.training.spec_inner, "inner parameters")
        inner_w0, _ = eqx.partition(model_outer, spec_inner)
        _, outer_params = eqx.partition(model_outer, spec_inner)
        inspect_inner = inspect_inexact_paths(inner_w0)
        inner_paths = {r["path"] for r in inspect_inner}
        if not any("suffix_blocks.feed_forward_prime" in p for p in inner_paths):
            raise SystemExit("spec_inner did not select suffix_blocks.feed_forward_prime")
        if any("ffn_prime_norm" in p for p in inner_paths):
            raise SystemExit("spec_inner unexpectedly selected prime norms")
        print(f"  P={prefix_len} S={suffix_len}  #W_fast={sum(r['size'] for r in inspect_inner):,}", flush=True)

        xt_embed = model_outer.language_model.wte_call(seq.input_ids)
        prefix_output = eqx.filter_checkpoint(model_outer.language_model.prefix_call)(
            model_outer.language_model.model.h.prefix_blocks, xt_embed, state_prefix, seq
        ).last_hidden_state
        prefix_output.block_until_ready()

        seq_chunks = tree_rearrange(seq, "(chunk token) ... -> chunk token ...", token=mini_batch)
        prefix_chunks = tree_rearrange(prefix_output, "(chunk token) ... -> chunk token ...", token=mini_batch)
        seq_chunk = tree_slice(seq_chunks, CHUNK_INDEX)
        prefix_chunk = prefix_chunks[CHUNK_INDEX]
        suffix_state_canonical = clone_pytree(state_suffix)
        mask = seq_chunk.loss_masks

        def fresh_state():
            return clone_pytree(suffix_state_canonical)

        def token_ttt_ce_only(inner, state):
            model_i = eqx.combine(inner, outer_params)
            out = model_i.language_model.suffix_call(
                prefix_outputs=prefix_chunk, state=state, seq=seq_chunk
            )
            return token_ttt_ce_from_logits(out.logits, seq_chunk.target_tokens)

        def token_ttt_ce_at_index(inner, idx, state):
            return token_ttt_ce_only(inner, state)[idx]

        def L_chunk(inner, state):
            return official_masked_reduction(token_ttt_ce_only(inner, state), mask)

        def jvp_token_ttt_ce(inner, tangent, state):
            def f(w):
                return token_ttt_ce_only(w, state)

            return eqx.filter_jvp(f, (inner,), (tangent,))

        f_jit = eqx.filter_jit(token_ttt_ce_only)
        L_jit = eqx.filter_jit(L_chunk)
        grad_L_jit = eqx.filter_jit(eqx.filter_value_and_grad(L_chunk))
        grad_token_jit = eqx.filter_jit(eqx.filter_grad(token_ttt_ce_at_index))
        jvp_jit = eqx.filter_jit(jvp_token_ttt_ce)

        inner_before = clone_pytree(inner_w0)
        state_before = clone_pytree(suffix_state_canonical)

        token_ttt_ce, t_f = timed("f(W0)", lambda: f_jit(inner_w0, fresh_state()), warmup=True)
        token_ttt_ce = np.asarray(token_ttt_ce)
        runtime["f_compile_s"] = t_f
        valid = np.asarray(mask).astype(bool)
        n_valid = int(valid.sum())
        if n_valid < N_EXACT:
            raise SystemExit(f"need {N_EXACT} valid tokens, got {n_valid}")

        L_recon = float(np.asarray(L_jit(inner_w0, fresh_state())))
        print(f"  N_valid={n_valid}  L_chunk={L_recon:.6f}  CE mean valid={float(token_ttt_ce[valid].mean()):.6f}", flush=True)

        print("\n--- g_bar = reverse-mode ∇ L_chunk (pre-clip) ---", flush=True)
        (L_val, g_bar), t_gbar = timed(
            "value_and_grad L_chunk",
            lambda: grad_L_jit(inner_w0, fresh_state()),
            warmup=True,
        )
        runtime["g_bar_compile_s"] = t_gbar
        L_val = float(np.asarray(L_val))
        g_bar_norm = float(pytree_l2_norm(g_bar))
        c = clip_scale(g_bar_norm, clip_norm)
        clip_fired = not (g_bar_norm < clip_norm)
        eval_step_arr = jnp.array(eval_step, dtype=jnp.int32)
        ilr_mult = float(np.asarray(model_outer.get_ilr_multiplier(eval_step_arr)))
        eta = inner_lr * ilr_mult
        print(
            f"  L_chunk={L_val:.8f}  ||g_bar||={g_bar_norm:.6f}  c={c:.6f}  clip_fired={clip_fired}  eta={eta:.6f}",
            flush=True,
        )
        if abs(L_val - L_recon) > 1e-5:
            print(f"  WARN L_chunk value_and_grad vs reconstructed {abs(L_val - L_recon):.3e}", flush=True)

        print("\n--- Ahat = JVP(f, stopgrad g_bar) ---", flush=True)
        g_bar_tangent = stop_gradient_tree(g_bar)
        (y_jvp_a, ahat), t_ahat = timed(
            "jvp along g_bar",
            lambda: jvp_jit(inner_w0, g_bar_tangent, fresh_state()),
            warmup=True,
        )
        runtime["ahat_jvp_s"] = t_ahat
        ahat = np.asarray(ahat, dtype=np.float64)
        y_jvp_a = np.asarray(y_jvp_a)
        sanity["jvp_A_primal_vs_f_max_abs"] = float(np.max(np.abs(y_jvp_a - token_ttt_ce)))
        print(
            f"  Ahat valid mean={float(ahat[valid].mean()):.6e}  "
            f"min={float(ahat[valid].min()):.6e}  max={float(ahat[valid].max()):.6e}  "
            f"n_neg={int(np.sum(ahat[valid] < 0))}/{n_valid}",
            flush=True,
        )

        print(f"\n--- Ghat Hutchinson K={K_G} (same recipe as STEP 3) ---", flush=True)
        jvp_keys = jrandom.split(jrandom.PRNGKey(JVP_SEED), K_G)
        v0 = rademacher_like(inner_w0, jvp_keys[0])
        (y_jvp_g, jv0), t_jvp0 = timed(
            "jvp rademacher 0",
            lambda: jvp_jit(inner_w0, v0, fresh_state()),
            warmup=True,
        )
        runtime["ghat_jvp_compile_s"] = t_jvp0
        sumsq = np.asarray(jv0, dtype=np.float64) ** 2
        for k_i in range(1, K_G):
            v_k = rademacher_like(inner_w0, jvp_keys[k_i])
            (_, jv_k), dt = timed(
                f"jvp rademacher {k_i}",
                lambda v=v_k: jvp_jit(inner_w0, v, fresh_state()),
                quiet=True,
            )
            sumsq = sumsq + np.asarray(jv_k, dtype=np.float64) ** 2
            if (k_i + 1) % 8 == 0 or k_i + 1 == K_G:
                print(f"  ghat progress {k_i + 1}/{K_G}  last_s={dt:.3f}", flush=True)
        ghat = np.sqrt(sumsq / float(K_G))
        sanity["jvp_G_primal_vs_f_max_abs"] = float(np.max(np.abs(np.asarray(y_jvp_g) - token_ttt_ce)))

        ihat = ahat - (ghat ** 2) / float(n_valid)
        rhat = (float(n_valid) * ahat) / (ghat ** 2 + R_EPS)
        ihat_cls = classify_interference(ihat[valid])
        print(
            f"  Ihat valid n_suppression={int(np.sum(ihat_cls == 'suppression'))} "
            f"n_reinforcement={int(np.sum(ihat_cls == 'reinforcement'))}",
            flush=True,
        )

        selected, selection_meta = select_even_valid_indices(valid, N_EXACT)
        print(f"\n--- exact reverse-mode on {N_EXACT} spread valid tokens ---", flush=True)
        print(f"  indices: {selected}", flush=True)
        a_exact = np.full((N_EXACT,), np.nan, dtype=np.float64)
        g_exact = np.full((N_EXACT,), np.nan, dtype=np.float64)
        ahat_sub = np.asarray([ahat[t] for t in selected], dtype=np.float64)
        ghat_sub = np.asarray([ghat[t] for t in selected], dtype=np.float64)

        g0, t_ex0 = timed(
            f"exact t={selected[0]}",
            lambda: grad_token_jit(inner_w0, jnp.int32(selected[0]), fresh_state()),
            warmup=True,
        )
        runtime["exact_compile_s"] = t_ex0
        for i, t_idx in enumerate(selected):
            if i == 0:
                g_t = g0
            else:
                g_t, dt = timed(
                    f"exact t={t_idx}",
                    lambda t=int(t_idx): grad_token_jit(inner_w0, jnp.int32(t), fresh_state()),
                    quiet=True,
                )
            g_exact[i] = float(pytree_l2_norm(g_t))
            a_exact[i] = float(pytree_vdot(g_t, g_bar))
            if i == 0 or i + 1 == N_EXACT or (i + 1) % 8 == 0:
                print(
                    f"  exact {i + 1}/{N_EXACT} t={t_idx}  G={g_exact[i]:.4f}  A={a_exact[i]:.6e}",
                    flush=True,
                )

        i_exact = a_exact - (g_exact ** 2) / float(n_valid)
        ihat_sub = np.asarray([ihat[t] for t in selected], dtype=np.float64)
        ihat_a_only = ahat_sub - (g_exact ** 2) / float(n_valid)

        a_metrics = error_metrics(a_exact, ahat_sub, np.ones(N_EXACT, dtype=bool), eps=REL_EPS)
        a_metrics["relative_error_note"] = (
            "error_metrics relative_* divides by signed exact; use relative_error_vs_abs_exact_*"
        )
        a_abs = np.abs(ahat_sub - a_exact)
        a_rel = a_abs / (np.abs(a_exact) + REL_EPS)
        a_metrics["relative_error_vs_abs_exact_mean"] = float(np.mean(a_rel))
        a_metrics["relative_error_vs_abs_exact_median"] = float(np.median(a_rel))
        a_metrics["relative_error_vs_abs_exact_max"] = float(np.max(a_rel))
        a_sign = sign_agreement(a_exact, ahat_sub)

        i_metrics = error_metrics(i_exact, ihat_sub, np.ones(N_EXACT, dtype=bool), eps=REL_EPS)
        i_sign = sign_agreement(i_exact, ihat_sub)
        i_class = class_agreement(i_exact, ihat_sub)
        i_a_only_metrics = error_metrics(i_exact, ihat_a_only, np.ones(N_EXACT, dtype=bool), eps=REL_EPS)
        i_a_only_class = class_agreement(i_exact, ihat_a_only)
        g_metrics = error_metrics(g_exact, ghat_sub, np.ones(N_EXACT, dtype=bool), eps=REL_EPS)

        print("\n--- Ahat vs A_exact (32 tokens) ---", flush=True)
        print(
            f"  Pearson={a_metrics.get('pearson')}  Spearman={a_metrics.get('spearman')}  "
            f"abs_mean={a_metrics['absolute_error_mean']:.3e}  "
            f"rel_vs_|A|_median={a_metrics['relative_error_vs_abs_exact_median']:.3e}",
            flush=True,
        )
        print(
            f"  sign agree={a_sign['agree']:.3f}  exact_neg={a_sign['n_exact_neg']}  "
            f"neg_recall={a_sign['neg_recall']}",
            flush=True,
        )
        print("\n--- Ihat vs I_exact (32 tokens; Ghat vs G_exact in each) ---", flush=True)
        print(
            f"  Spearman={i_metrics.get('spearman')}  sign agree={i_sign['agree']:.3f}  "
            f"class agree={i_class['agree']:.3f}  "
            f"exact suppression/reinforcement="
            f"{i_class['n_exact_suppression']}/{i_class['n_exact_reinforcement']}",
            flush=True,
        )
        print(
            f"  Ihat_A-only (Ahat, G_exact) Spearman={i_a_only_metrics.get('spearman')}  "
            f"class agree={i_a_only_class['agree']:.3f}",
            flush=True,
        )

        first_order = None
        delta_full = None
        predicted_full = None
        if not skip_first_order:
            print("\n--- optional first-order sanity: one official inner_loop_step, original KV ---", flush=True)
            try:
                state_all_eval = state_all.set(model.step_index, eval_step_arr)
                inner_opt_state = model_outer.inner_optimizer(state_all_eval).init(
                    model_outer.inner_parameters()
                )
                model_inner0 = eqx.combine(inner_w0, outer_params)

                def official_chunk_step():
                    return MetaModel.inner_loop_step(
                        model_inner0,
                        inner_opt_state,
                        (state_all_eval, clone_pytree(suffix_state_canonical)),
                        seq_chunk,
                        prefix_chunk,
                    )

                out_step, t_step = timed("inner_loop_step chunk0", official_chunk_step, warmup=True)
                runtime["inner_loop_step_s"] = t_step
                new_inner, _ = eqx.partition(out_step.new_model, spec_inner)
                ce_after, t_after = timed("f(W1) original KV", lambda: f_jit(new_inner, fresh_state()))
                runtime["f_after_s"] = t_after
                ce_after = np.asarray(ce_after)
                delta = ce_after - token_ttt_ce
                predicted = (-eta * c) * ahat
                d_valid = delta[valid]
                p_valid = predicted[valid]
                fo_metrics = error_metrics(d_valid, p_valid, np.ones(n_valid, dtype=bool), eps=REL_EPS)
                fo_sign = sign_agreement(d_valid, p_valid)
                delta_full = delta.astype(np.float32)
                predicted_full = predicted.astype(np.float32)
                first_order = {
                    "implemented": True,
                    "kv_held": "canonical pre-update suffix state (isolates ΔW from KV)",
                    "step_index_set_only_for_this_step": eval_step,
                    "eta": eta,
                    "clip_scale_c": c,
                    "delta_valid_mean": float(np.mean(d_valid)),
                    "predicted_valid_mean": float(np.mean(p_valid)),
                    "spearman": fo_metrics.get("spearman"),
                    "pearson": fo_metrics.get("pearson"),
                    "sign_agreement": fo_sign,
                    "note": (
                        "diagnostic for the A_t first-order identity on this chunk only; "
                        "not long-range memory or future utility"
                    ),
                }
                print(
                    f"  mean Δl_t={first_order['delta_valid_mean']:.6e}  "
                    f"mean predicted={first_order['predicted_valid_mean']:.6e}  "
                    f"Spearman={first_order['spearman']}  sign={fo_sign['agree']:.3f}",
                    flush=True,
                )
            except Exception as exc:
                import traceback

                traceback.print_exc()
                first_order = {"implemented": False, "error": repr(exc)}
                print(f"  first-order sanity skipped: {exc!r}", flush=True)

        sanity["inner_w_after_max_abs"] = pytree_max_abs_diff(inner_before, inner_w0)
        sanity["canonical_s0_after_max_abs"] = pytree_max_abs_diff(state_before, suffix_state_canonical)
        sanity["finite_ahat"] = finite_flags("ahat_valid", ahat[valid])
        sanity["finite_ghat"] = finite_flags("ghat_valid", ghat[valid])
        sanity["finite_ihat"] = finite_flags("ihat_valid", ihat[valid])
        sanity["finite_a_exact"] = finite_flags("a_exact", a_exact)
        sanity["L_chunk_matches_recon"] = abs(L_val - L_recon) < 1e-5
        sanity["n_valid"] = n_valid

        step3_check = None
        for stem_prev in (STEP3_CUDNN_STEM, STEP3_CANONICAL_STEM):
            npz_prev = stem_prev.with_suffix(".npz")
            if not npz_prev.exists():
                continue
            prev = dict(np.load(npz_prev, allow_pickle=False))
            if "token_ttt_ce" not in prev:
                continue
            prev_ce = np.asarray(prev["token_ttt_ce"])
            if prev_ce.shape != token_ttt_ce.shape:
                continue
            ce_abs = float(np.max(np.abs(prev_ce.astype(np.float64) - token_ttt_ce.astype(np.float64))))
            step3_check = {
                "source": str(npz_prev),
                "token_ttt_ce_max_abs": ce_abs,
                "token_ttt_ce_matches": ce_abs < EVAL_POINT_CE_ABS,
            }
            if "estimated_grad_norm_k32" in prev:
                prev_ghat = np.asarray(prev["estimated_grad_norm_k32"], dtype=np.float64)
                ghat_abs = float(np.max(np.abs(prev_ghat - ghat)))
                step3_check["ghat_k32_max_abs"] = ghat_abs
                step3_check["ghat_k32_matches"] = ghat_abs < GHAT_VS_STEP3_ABS
            if "exact_grad_norm" in prev:
                prev_g = np.asarray(prev["exact_grad_norm"], dtype=np.float64)[np.asarray(selected)]
                gex_abs = float(np.max(np.abs(prev_g - g_exact)))
                step3_check["exact_G_on_32_max_abs"] = gex_abs
                step3_check["exact_G_on_32_matches"] = gex_abs < GHAT_VS_STEP3_ABS
            print(
                f"  vs STEP 3 {npz_prev.name}: CE max_abs={ce_abs:.3e} "
                f"Ghat32={step3_check.get('ghat_k32_max_abs')} "
                f"exactG32={step3_check.get('exact_G_on_32_max_abs')}",
                flush=True,
            )
            break

        memory["after_compute"] = try_memory_stats()
        runtime["wall_s"] = time.perf_counter() - wall0

        EXPERIMENT_RESULT_DIR.mkdir(parents=True, exist_ok=True)
        stem = EXPERIMENT_RESULT_DIR / "token_grad_align_validation_t1024_a100_cudnn"
        npz_kwargs = {
            "token_ttt_ce": token_ttt_ce.astype(np.float32),
            "valid_masks": valid.astype(np.uint8),
            "ahat": ahat.astype(np.float32),
            "ghat_k32": ghat.astype(np.float32),
            "ihat": ihat.astype(np.float32),
            "rhat": rhat.astype(np.float32),
            "exact_token_indices": np.asarray(selected, dtype=np.int32),
            "a_exact": a_exact.astype(np.float32),
            "g_exact": g_exact.astype(np.float32),
            "i_exact": i_exact.astype(np.float32),
            "ahat_on_exact": ahat_sub.astype(np.float32),
            "ghat_on_exact": ghat_sub.astype(np.float32),
            "ihat_on_exact": ihat_sub.astype(np.float32),
            "ihat_a_only_on_exact": ihat_a_only.astype(np.float32),
            "g_bar_norm": np.asarray(g_bar_norm, dtype=np.float32),
            "clip_scale_c": np.asarray(c, dtype=np.float32),
            "eta": np.asarray(eta, dtype=np.float32),
            "n_valid": np.asarray(n_valid, dtype=np.int32),
        }
        if delta_full is not None:
            npz_kwargs["delta_ce"] = delta_full
            npz_kwargs["predicted_delta_ce"] = predicted_full
        np.savez(stem.with_suffix(".npz"), **npz_kwargs)
        meta = {
            "name": "token_grad_align_validation",
            "step": "6C.1",
            "not": "saliency / future utility / long-range TTT memory / RULER",
            "seq_len": SEQ_LEN,
            "chunk_index": CHUNK_INDEX,
            "mini_batch_size": mini_batch,
            "checkpoint_dir": str(CHECKPOINT_DIR),
            "probe_path": str(PROBE_PATH),
            "definitions": {
                "l_t": "unmasked fp32 NTP CE (token_ttt_ce_from_logits)",
                "L_chunk": "official masked mean: sum(m l)/sum(m)",
                "g_t": "∇_{W_fast} l_t at pre-update W",
                "g_bar": "∇_{W_fast} L_chunk, pre-clip, inner spec only",
                "A_t": "<g_t, g_bar>",
                "Ahat_t": "JVP(f, stopgrad g_bar); not called exact",
                "Ghat_t": f"Hutchinson K={K_G} unnormalized Rademacher JVP, same as STEP 3",
                "I_t": "A_t - G_t^2 / N_valid",
                "R_t": "N * A_t / (G_t^2 + eps); auxiliary, unstable at small G",
                "N": n_valid,
                "clip": (
                    f"optax.clip_by_global_norm({clip_norm}); "
                    "c=1 if ||g_bar||<clip_norm else clip_norm/||g_bar||"
                ),
                "eta": "inner SGD lr * ilr_multiplier at eval step_index (not used by A/G/JVP)",
            },
            "g_bar": {
                "norm": g_bar_norm,
                "clip_norm": clip_norm,
                "clip_scale_c": c,
                "clip_fired": clip_fired,
                "eta": eta,
                "ilr_multiplier": ilr_mult,
                "inner_lr": inner_lr,
                "L_chunk": L_val,
            },
            "n_valid": n_valid,
            "selection": selection_meta,
            "ahat_vs_exact": {
                "error_metrics": a_metrics,
                "sign_agreement": a_sign,
            },
            "ihat_vs_exact": {
                "error_metrics": i_metrics,
                "sign_agreement": i_sign,
                "class_agreement": i_class,
                "note": "Ihat uses Ghat K=32; I_exact uses reverse-mode G_t on the same 32 tokens",
            },
            "ihat_a_only_vs_exact": {
                "error_metrics": i_a_only_metrics,
                "class_agreement": i_a_only_class,
                "note": "Ahat - G_exact^2/N; isolates Ahat error from Hutchinson Ghat error",
            },
            "ghat_vs_exact_on_32": g_metrics,
            "first_order_delta_ce": first_order,
            "step3_eval_point_check": step3_check,
            "gpu": asdict(gpu),
            "software": {
                "python": sys.version.split()[0],
                "jax": jax.__version__,
                "jaxlib": getattr(__import__("jaxlib"), "__version__", "unknown"),
            },
            "attention": attn_setup,
            "step_index": {
                "set_on_A_G_JVP_path": False,
                "set_only_for_optional_inner_loop_step": True,
                "eval_warmup_value": eval_step,
                "reason": "A/G/JVP match STEP 3 frozen eval point; eta uses eval warmup only for first-order",
            },
            "sanity": sanity,
            "runtime_s": runtime,
            "memory": memory,
        }
        stem.with_suffix(".json").write_text(json.dumps(to_jsonable(meta), indent=2) + "\n")
        print(f"\nraw:  {stem.with_suffix('.npz')}", flush=True)
        print(f"meta: {stem.with_suffix('.json')}", flush=True)
        print(f"wall: {runtime['wall_s']:.1f}s", flush=True)


if __name__ == "__main__":
    args = parse_args()
    main(
        attention_cudnn_override=args.attention_cudnn_override,
        skip_first_order=args.skip_first_order,
    )
