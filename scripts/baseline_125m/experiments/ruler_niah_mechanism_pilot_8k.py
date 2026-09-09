"""STEP 7A: RULER 8K mechanism pilot (20 single_2 + 20 MK1).

Not a long-range memory experiment. All selected 8K examples are in-SWA.
Does not edit e2e/. Does not overwrite STEP 6D / 6C.1 files.

Each sample starts from the same checkpoint W0 and a fresh suffix/optimizer
state. Fast weights and KV are not carried across samples.

Labels (relevant / distractor / query) are used only after the official
causal forward, for offline matching and analysis.
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
    add_e2e_to_path,
    compose_baseline_config,
)
from measurement_utils import (  # noqa: E402
    error_metrics,
    official_masked_reduction,
    pytree_l2_norm,
    pytree_max_abs_diff,
    pytree_vdot,
    rademacher_like,
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

SEQ_LABEL = "8k"
N_TAKE = 20
SAMPLE_SELECT_SEED = 20260905
JVP_SEED = 0
K_G = 32
REL_EPS = 1e-12
RESULT_DIR = EXPERIMENT_RESULT_DIR / "ruler_niah_mechanism_pilot_8k"

TASKS = (
    {"name": "niah_single_2", "family": "S-NIAH", "salt": 0},
    {"name": "niah_multikey_1", "family": "MK-NIAH", "salt": 1},
)

ROLE_CODES = {
    "padding": 0,
    "background": 1,
    "query": 2,
    "relevant_sentence": 3,
    "relevant_key": 4,
    "relevant_value": 5,
    "distractor_sentence": 6,
    "distractor_key": 7,
    "distractor_value": 8,
}
NEEDLE_PREFIXES = ("relevant_", "distractor_")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--attention-cudnn-override",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Default off: official cuDNN prefix SWA.",
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


def median_iqr(values) -> dict | None:
    import numpy as np

    x = np.asarray(values, dtype=np.float64)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return None
    return {
        "n": int(x.size),
        "median": float(np.median(x)),
        "q25": float(np.percentile(x, 25)),
        "q75": float(np.percentile(x, 75)),
        "min": float(np.min(x)),
        "max": float(np.max(x)),
        "mean": float(np.mean(x)),
    }


def direction_counts(values) -> dict:
    import numpy as np

    x = np.asarray(values, dtype=np.float64)
    x = x[np.isfinite(x)]
    return {
        "n": int(x.size),
        "n_pos": int(np.sum(x > 0)),
        "n_neg": int(np.sum(x < 0)),
        "n_zero": int(np.sum(x == 0)),
    }


def spearman_safe(a, b):
    import numpy as np
    from scipy.stats import spearmanr

    x = np.asarray(a, dtype=np.float64)
    y = np.asarray(b, dtype=np.float64)
    m = np.isfinite(x) & np.isfinite(y)
    x, y = x[m], y[m]
    if x.size < 3 or np.std(x) == 0 or np.std(y) == 0:
        return None
    return float(spearmanr(x, y).statistic)


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
            rows.append({"path": jax.tree_util.keystr(path), "shape": list(value.shape)})
    return rows


def sign_agreement(exact, estimated) -> dict:
    import numpy as np

    e = np.asarray(exact, dtype=np.float64)
    h = np.asarray(estimated, dtype=np.float64)
    neg = e < 0
    pos = e > 0
    return {
        "n": int(e.size),
        "agree": float(np.mean(np.sign(e) == np.sign(h))) if e.size else None,
        "n_exact_neg": int(np.sum(neg)),
        "neg_recall": float(np.sum(neg & (h < 0)) / np.sum(neg)) if np.any(neg) else None,
        "n_exact_pos": int(np.sum(pos)),
        "pos_recall": float(np.sum(pos & (h > 0)) / np.sum(pos)) if np.any(pos) else None,
    }


def stop_gradient_tree(tree):
    import equinox as eqx
    import jax

    return jax.tree.map(lambda x: jax.lax.stop_gradient(x) if eqx.is_inexact_array(x) else x, tree)


def inclusive_range(start: int, end: int) -> range:
    return range(int(start), int(end) + 1)


def pick_sample_ids(n_available: int, n_take: int, salt: int) -> list[int]:
    import numpy as np

    rng = np.random.default_rng(SAMPLE_SELECT_SEED + salt)
    chosen = rng.choice(n_available, size=n_take, replace=False)
    return sorted(int(x) for x in chosen)


def build_batch(ids: list[int], T: int, T_pad: int):
    import jax.numpy as jnp
    import numpy as np

    if len(ids) != T:
        raise SystemExit(f"input_ids len {len(ids)} != T={T}")
    if ids[0] != BOS_ID:
        raise SystemExit("sequence does not start with BOS")
    if any(i == BOS_ID for i in ids[1:]):
        raise SystemExit("real tokens include PAD/BOS id")
    padded, T2, T_pad2 = right_pad_ids(ids, pad_id=PAD_ID, mini_batch=MINI_BATCH)
    if T2 != T or T_pad2 != T_pad:
        raise SystemExit("padding length mismatch")
    if padded[:T] != ids:
        raise SystemExit("right-pad mutated real tokens")
    tokens = padded + [PAD_ID]
    seq = tokens_to_batch(jnp.asarray(tokens, dtype=jnp.int32), bos_token_id=BOS_ID)
    mask = np.asarray(seq.loss_masks).astype(bool)
    label_mask = np.asarray(ntp_loss_mask(T, T_pad), dtype=bool)
    if not np.array_equal(mask[: T_pad - 1], label_mask) or mask[-1] or int(mask.sum()) != T - 1:
        raise SystemExit("NTP mask does not match t < T-1")
    return seq, mask


def assign_roles(*, n_loss: int, valid, query_t0: int, query_t1: int, spans) -> tuple[list[str], list[int]]:
    roles = ["padding"] * n_loss
    span_uid = [-1] * n_loss
    for t in range(n_loss):
        if not valid[t]:
            continue
        if int(query_t0) <= t <= int(query_t1):
            roles[t] = "query"
        else:
            roles[t] = "background"
    for sp in spans:
        kind = str(sp["role"])
        if kind not in ("relevant", "distractor"):
            raise SystemExit(f"unexpected span role {kind}")
        uid = (0 if kind == "relevant" else 100) + int(sp["span_index"])
        prefix = kind
        for t in inclusive_range(sp["sentence_loss_t_start"], sp["sentence_loss_t_end"]):
            if not valid[t]:
                raise SystemExit(f"span token t={t} is not a valid loss position")
            if roles[t] == "query":
                raise SystemExit(f"span overlaps query at t={t}")
            if span_uid[t] not in (-1, uid):
                raise SystemExit(f"overlapping needles at t={t}")
            if int(sp["value_loss_t_start"]) <= t <= int(sp["value_loss_t_end"]):
                roles[t] = f"{prefix}_value"
            elif int(sp["key_loss_t_start"]) <= t <= int(sp["key_loss_t_end"]):
                roles[t] = f"{prefix}_key"
            else:
                roles[t] = f"{prefix}_sentence"
            span_uid[t] = uid
    return roles, span_uid


def is_needle_role(role: str) -> bool:
    return role.startswith(NEEDLE_PREFIXES)


def match_nll_same_chunk(*, targets: list[int], nll, valid, roles) -> tuple[list[dict], list[int]]:
    import numpy as np

    used: set[int] = set()
    pairs: list[dict] = []
    unmatched: list[int] = []
    nll = np.asarray(nll, dtype=np.float64)
    for t in sorted(int(x) for x in targets):
        c = t // MINI_BATCH
        lo, hi = c * MINI_BATCH, (c + 1) * MINI_BATCH
        pool = [
            p
            for p in range(lo, hi)
            if valid[p] and roles[p] == "background" and p not in used and np.isfinite(nll[p])
        ]
        if not pool or not np.isfinite(nll[t]):
            unmatched.append(t)
            continue
        gaps = np.abs(nll[t] - nll[np.asarray(pool)])
        j = int(np.argmin(gaps))
        m = int(pool[j])
        used.add(m)
        pairs.append(
            {
                "t": t,
                "match": m,
                "chunk": c,
                "gap": float(gaps[j]),
                "nll_t": float(nll[t]),
                "nll_m": float(nll[m]),
                "role_t": roles[t],
            }
        )
    return pairs, unmatched


def chunk_percentile(values, valid, t: int) -> float:
    import numpy as np

    c = t // MINI_BATCH
    sl = slice(c * MINI_BATCH, (c + 1) * MINI_BATCH)
    x = np.asarray(values[sl], dtype=np.float64)[np.asarray(valid[sl], dtype=bool)]
    v = float(values[t])
    if x.size == 0 or not np.isfinite(v):
        return float("nan")
    return float((np.sum(x < v) + 0.5 * np.sum(x == v)) / x.size)


def role_subset(roles, prefix: str, *, remainder: bool = False) -> list[int]:
    out = []
    for t, r in enumerate(roles):
        if remainder:
            if r == f"{prefix}_sentence":
                out.append(t)
        elif r.startswith(prefix):
            if prefix.endswith("_"):
                out.append(t)
            elif r == prefix or r.startswith(prefix + "_"):
                out.append(t)
    return out


def load_task_tables():
    import pandas as pd

    examples = pd.read_parquet(LABEL_RESULT_DIR / "examples.parquet")
    spans = pd.read_parquet(LABEL_RESULT_DIR / "spans.parquet")
    return examples, spans


def sample_rows(examples, spans, task: str, sample_id: int):
    hit = examples[
        (examples["task"] == task)
        & (examples["seq_length_label"] == SEQ_LABEL)
        & (examples["sample_id"] == sample_id)
    ]
    if len(hit) != 1:
        raise SystemExit(f"{task} sample_id={sample_id} expected 1 row, got {len(hit)}")
    sp = spans[
        (spans["task"] == task)
        & (spans["seq_length_label"] == SEQ_LABEL)
        & (spans["sample_id"] == sample_id)
    ]
    return hit.iloc[0], [sp.iloc[i] for i in range(len(sp))]


def summarize_pairs(pairs, ghat, nll, ahat, valid, roles, *, span_uid=None, want_uid=None):
    import numpy as np

    rows = []
    for p in pairs:
        if want_uid is not None and span_uid[p["t"]] != want_uid:
            continue
        t, m = p["t"], p["match"]
        rows.append(
            {
                "g_t": float(ghat[t]),
                "g_m": float(ghat[m]),
                "d_g": float(ghat[t] - ghat[m]),
                "nll_t": float(nll[t]),
                "nll_m": float(nll[m]),
                "d_nll": float(nll[t] - nll[m]),
                "ahat_t": float(ahat[t]),
                "pct": chunk_percentile(ghat, valid, t),
                "gap": p["gap"],
                "role": roles[t],
            }
        )
    if not rows:
        return None
    d_g = [r["d_g"] for r in rows]
    ahat_neg = [r["ahat_t"] < 0 for r in rows]
    return {
        "n_tokens": len(rows),
        "mean_g_t": float(np.mean([r["g_t"] for r in rows])),
        "mean_g_m": float(np.mean([r["g_m"] for r in rows])),
        "mean_d_g": float(np.mean(d_g)),
        "mean_d_nll": float(np.mean([r["d_nll"] for r in rows])),
        "mean_nll_t": float(np.mean([r["nll_t"] for r in rows])),
        "mean_percentile": float(np.nanmean([r["pct"] for r in rows])),
        "ahat_neg_frac": float(np.mean(ahat_neg)),
        "mean_match_gap": float(np.mean([r["gap"] for r in rows])),
    }


def analyze_single2(sample_summaries, match_gaps) -> dict:
    d_g = [s["needle_all"]["mean_d_g"] for s in sample_summaries if s.get("needle_all")]
    out = {"n_samples": len(sample_summaries), "paired_d_ghat_all_needle": median_iqr(d_g)}
    for key in ("needle_all", "relevant_value", "relevant_key", "relevant_sentence"):
        vals = [s[key]["mean_d_g"] for s in sample_summaries if s.get(key)]
        pct = [s[key]["mean_percentile"] for s in sample_summaries if s.get(key)]
        nll = [s[key]["mean_d_nll"] for s in sample_summaries if s.get(key)]
        ahat = [s[key]["ahat_neg_frac"] for s in sample_summaries if s.get(key)]
        out[key] = {
            "paired_d_ghat": median_iqr(vals),
            "direction": direction_counts(vals),
            "percentile": median_iqr(pct),
            "paired_d_nll": median_iqr(nll),
            "ahat_neg_frac": median_iqr(ahat),
        }
    out["match_gap"] = median_iqr(match_gaps)
    out["note"] = "sample-level means; 20 aggregates, not token-iid tests"
    return out


def analyze_mk1(sample_summaries, match_gaps) -> dict:
    rel = [s["relevant"]["mean_d_g"] for s in sample_summaries if s.get("relevant")]
    dist = [s["distractor_mean"]["mean_d_g"] for s in sample_summaries if s.get("distractor_mean")]
    paired = []
    for s in sample_summaries:
        if s.get("relevant") and s.get("distractor_mean"):
            paired.append(s["relevant"]["mean_d_g"] - s["distractor_mean"]["mean_d_g"])
    return {
        "n_samples": len(sample_summaries),
        "relevant_excess_vs_matched_bg": median_iqr(rel),
        "relevant_direction": direction_counts(rel),
        "distractor_mean_excess_vs_matched_bg": median_iqr(dist),
        "relevant_minus_distractor_excess": median_iqr(paired),
        "relevant_minus_distractor_direction": direction_counts(paired),
        "relevant_percentile": median_iqr(
            [s["relevant"]["mean_percentile"] for s in sample_summaries if s.get("relevant")]
        ),
        "distractor_percentile": median_iqr(
            [s["distractor_mean"]["mean_percentile"] for s in sample_summaries if s.get("distractor_mean")]
        ),
        "relevant_ahat_neg": median_iqr(
            [s["relevant"]["ahat_neg_frac"] for s in sample_summaries if s.get("relevant")]
        ),
        "distractor_ahat_neg": median_iqr(
            [s["distractor_mean"]["ahat_neg_frac"] for s in sample_summaries if s.get("distractor_mean")]
        ),
        "match_gap": median_iqr(match_gaps),
        "note": (
            "excess = span mean(Ghat - matched Ghat). relevant > distractor is NOT an "
            "expected value; query is still in the future at needle time."
        ),
    }


def main(*, attention_cudnn_override: bool) -> None:
    wall0 = time.perf_counter()
    import numpy as np
    import pandas as pd

    examples, spans_df = load_task_tables()
    selected = {}
    for spec in TASKS:
        pool = examples[(examples["task"] == spec["name"]) & (examples["seq_length_label"] == SEQ_LABEL)]
        ids_avail = sorted(int(x) for x in pool["sample_id"].tolist())
        if ids_avail != list(range(200)):
            raise SystemExit(f"{spec['name']} sample_id set is not 0..199")
        chosen = pick_sample_ids(200, N_TAKE, spec["salt"])
        selected[spec["name"]] = chosen
        print(f"selected {spec['name']}: {chosen}", flush=True)

    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    (RESULT_DIR / "selected_ids.json").write_text(
        json.dumps(
            {
                "sample_select_seed": SAMPLE_SELECT_SEED,
                "rule": "numpy default_rng(seed+salt).choice(200, 20, replace=False); sorted",
                "independent_of_model_output": True,
                "tasks": selected,
                "n_take": N_TAKE,
            },
            indent=2,
        )
        + "\n"
    )

    cfg = compose_baseline_config()
    add_e2e_to_path()

    import equinox as eqx
    import jax
    import jax.numpy as jnp
    import jax.random as jrandom
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
        raise SystemExit("mini_batch mismatch")

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
        raise FileNotFoundError(str(CHECKPOINT_DIR))
    out_state = loader.load_checkpoint(
        step=cfg.training.resume_step,
        targets={"model_weights": abstract_model_weights},
        restore=cfg.training.load_part,
    )
    model_fresh, state = create_sharded_model_and_state()
    model, missing = unify_dict_with_eqx_module(out_state["model_weights"], model_fresh)
    if missing:
        raise SystemExit(f"checkpoint missing paths: {missing[:8]}")
    del out_state, model_fresh
    state = state.set(model.step_index, jnp.array(eval_step, dtype=jnp.int32))
    attn_setup = override.report_for_config(
        force_flash=bool(cfg.model.force_flash),
        seq_modeling_block=str(cfg.model.seq_modeling_block),
    )
    print("--- setup ---", flush=True)
    print(f"GPU: {gpu.name}  prefix_effective={attn_setup['prefix_effective_backend']}", flush=True)
    print(f"model.seq_len={cfg.model.seq_len} (unchanged)", flush=True)

    match_rows: list[dict] = []
    exact_rows: list[dict] = []
    task_outputs: dict[str, dict] = {}
    sanity_all: list[dict] = []
    compiled = {"f": False, "gbar": False, "jvp": False, "exact": False, "update": False, "prefix": False}
    runtime: dict = {"per_sample_s": []}

    with mesh:
        block_collection = model.language_model.model.h.blocks
        prime_storage = model.language_model.model.h.prime_storage if cfg.model.prime else None
        if prime_storage is None:
            raise SystemExit("prime=False")
        state_prefix_suffix = state.substate(model.language_model.model.h.blocks)
        state_prefix, state_suffix_0 = BlockCollectionSplit.split_state(state_prefix_suffix, cfg.model.suffix_len)
        state_all_0 = clone_pytree(state)
        new_collection = BlockCollectionSplit(
            cfg.model, block_collection=block_collection, prime_storage=prime_storage, key=jrandom.PRNGKey(0)
        )
        model = eqx.tree_at(lambda m: m.language_model.model.h, model, new_collection)
        model_outer = jax.tree.map(lambda p: p.astype(model.state_dtype), model)
        spec_inner = get_filter_spec(model_outer, cfg.training.spec_inner, "inner parameters")
        inner_paths = {r["path"] for r in inspect_inexact_paths(eqx.partition(model_outer, spec_inner)[0])}
        if not any("suffix_blocks.feed_forward_prime" in p for p in inner_paths):
            raise SystemExit("spec_inner mismatch")
        inner_w0_ref, _ = eqx.partition(model_outer, spec_inner)
        inner_w0_ref = clone_pytree(inner_w0_ref)
        _, outer_params = eqx.partition(model_outer, spec_inner)
        eval_step_arr = jnp.array(eval_step, dtype=jnp.int32)
        eta_eval = inner_lr * float(np.asarray(model_outer.get_ilr_multiplier(eval_step_arr)))

        def recombine_inner(model_i, model_o):
            spec = get_filter_spec(model_i, model_o.config.training.spec_inner, "inner parameters")
            inner_params, _ = eqx.partition(model_i, spec)
            _, outer_p = eqx.partition(model_o, spec)
            return eqx.combine(inner_params, outer_p)

        @eqx.filter_jit
        def official_chunk_step(model_i, opt_state, state_tuple, suffix_chunk, prefix_chunk, model_o):
            model_i = recombine_inner(model_i, model_o)
            return MetaModel.inner_loop_step(model_i, opt_state, state_tuple, suffix_chunk, prefix_chunk)

        def token_ttt_ce_only(inner, st, prefix_chunk, seq_chunk):
            model_i = eqx.combine(inner, outer_params)
            out = model_i.language_model.suffix_call(prefix_outputs=prefix_chunk, state=st, seq=seq_chunk)
            return token_ttt_ce_from_logits(out.logits, seq_chunk.target_tokens)

        def token_ce_at_index(inner, idx, st, prefix_chunk, seq_chunk):
            return token_ttt_ce_only(inner, st, prefix_chunk, seq_chunk)[idx]

        def L_chunk(inner, st, prefix_chunk, seq_chunk):
            return official_masked_reduction(
                token_ttt_ce_only(inner, st, prefix_chunk, seq_chunk), seq_chunk.loss_masks
            )

        def jvp_token_ce(inner, tangent, st, prefix_chunk, seq_chunk):
            def f(w):
                return token_ttt_ce_only(w, st, prefix_chunk, seq_chunk)

            return eqx.filter_jvp(f, (inner,), (tangent,))

        f_jit = eqx.filter_jit(token_ttt_ce_only)
        grad_L_jit = eqx.filter_jit(eqx.filter_value_and_grad(L_chunk))
        grad_token_jit = eqx.filter_jit(eqx.filter_grad(token_ce_at_index))
        jvp_jit = eqx.filter_jit(jvp_token_ce)
        prefix_fn = eqx.filter_checkpoint(model_outer.language_model.prefix_call)

        for spec in TASKS:
            task = spec["name"]
            sids = selected[task]
            n_samp = len(sids)
            n_chunks = 8
            token_ce = np.full((n_samp, n_chunks, MINI_BATCH), np.nan, dtype=np.float32)
            ghat = np.full_like(token_ce, np.nan)
            qhat = np.full_like(token_ce, np.nan)
            ahat = np.full_like(token_ce, np.nan)
            ihat = np.full_like(token_ce, np.nan)
            valid_arr = np.zeros((n_samp, n_chunks, MINI_BATCH), dtype=np.uint8)
            role_arr = np.zeros((n_samp, n_chunks, MINI_BATCH), dtype=np.int8)
            span_arr = np.full((n_samp, n_chunks, MINI_BATCH), -1, dtype=np.int16)
            g_bar_norms = np.full((n_samp, n_chunks), np.nan, dtype=np.float32)
            clip_cs = np.full((n_samp, n_chunks), np.nan, dtype=np.float32)
            etas = np.full((n_samp, n_chunks), np.nan, dtype=np.float32)
            n_valid_chunk = np.zeros((n_samp, n_chunks), dtype=np.int32)
            sample_T = np.zeros((n_samp,), dtype=np.int32)
            spearman_rows = []
            sample_level = []

            for si, sid in enumerate(sids):
                t_sample = time.perf_counter()
                row, span_list = sample_rows(examples, spans_df, task, sid)
                T = int(row["T"])
                T_pad = int(row["T_pad"])
                if T_pad != 8192:
                    raise SystemExit(f"{task} sid={sid} T_pad={T_pad} != 8192")
                if T_pad > int(cfg.model.seq_len):
                    raise SystemExit("T_pad exceeds model.seq_len")
                ids = [int(x) for x in row["input_ids"]]
                seq, valid = build_batch(ids, T, T_pad)
                roles, span_uid = assign_roles(
                    n_loss=T_pad,
                    valid=valid,
                    query_t0=int(row["query_loss_t_start"]),
                    query_t1=int(row["query_loss_t_end"]),
                    spans=span_list,
                )
                sample_T[si] = T
                print(
                    f"\n===== {task} {si + 1}/{n_samp} sample_id={sid} T={T} "
                    f"spans={len(span_list)} =====",
                    flush=True,
                )

                model_inner = clone_pytree(model_outer)
                inner_opt_state = model_outer.inner_optimizer(state_all_0).init(model_outer.inner_parameters())
                suffix_state = clone_pytree(state_suffix_0)
                state_all = clone_pytree(state_all_0)

                xt_embed = model_outer.language_model.wte_call(seq.input_ids)
                prefix_output, t_prefix = timed(
                    "prefix",
                    lambda: prefix_fn(
                        model_outer.language_model.model.h.prefix_blocks, xt_embed, state_prefix, seq
                    ).last_hidden_state,
                    warmup=not compiled["prefix"],
                    quiet=compiled["prefix"],
                )
                compiled["prefix"] = True
                seq_chunks = tree_rearrange(seq, "(chunk token) ... -> chunk token ...", token=mini_batch)
                prefix_chunks = tree_rearrange(prefix_output, "(chunk token) ... -> chunk token ...", token=mini_batch)

                nll_flat = np.full((T_pad,), np.nan, dtype=np.float64)
                ghat_flat = np.full((T_pad,), np.nan, dtype=np.float64)
                ahat_flat = np.full((T_pad,), np.nan, dtype=np.float64)
                sample_pairs: list[dict] = []
                purity_ok = True

                for c in range(n_chunks):
                    seq_chunk = tree_slice(seq_chunks, c)
                    prefix_chunk = prefix_chunks[c]
                    model_meas = recombine_inner(model_inner, model_outer)
                    inner_w, _ = eqx.partition(model_meas, spec_inner)
                    s_canonical = clone_pytree(suffix_state)

                    def fresh_state(s=s_canonical):
                        return clone_pytree(s)

                    inner_before = clone_pytree(inner_w)
                    mask_c = np.asarray(seq_chunk.loss_masks).astype(bool)
                    n_valid = int(mask_c.sum())
                    g0 = c * MINI_BATCH
                    if not np.array_equal(mask_c, valid[g0 : g0 + MINI_BATCH]):
                        raise SystemExit("chunk mask mismatch")

                    ce, _ = timed(
                        "f",
                        lambda: f_jit(inner_w, fresh_state(), prefix_chunk, seq_chunk),
                        warmup=not compiled["f"],
                        quiet=True,
                    )
                    compiled["f"] = True
                    ce_np = np.asarray(ce, dtype=np.float64)
                    (L_val, g_bar), _ = timed(
                        "gbar",
                        lambda: grad_L_jit(inner_w, fresh_state(), prefix_chunk, seq_chunk),
                        warmup=not compiled["gbar"],
                        quiet=True,
                    )
                    compiled["gbar"] = True
                    gbn = float(pytree_l2_norm(g_bar))
                    c_clip = clip_scale(gbn, clip_norm)
                    g_bar_tangent = stop_gradient_tree(g_bar)
                    (_, ahat_c), _ = timed(
                        "ahat",
                        lambda: jvp_jit(inner_w, g_bar_tangent, fresh_state(), prefix_chunk, seq_chunk),
                        warmup=not compiled["jvp"],
                        quiet=True,
                    )
                    compiled["jvp"] = True
                    ahat_c = np.asarray(ahat_c, dtype=np.float64)
                    jvp_keys = jrandom.split(jrandom.PRNGKey(JVP_SEED), K_G)
                    sumsq = None
                    for k_i in range(K_G):
                        v_k = rademacher_like(inner_w, jvp_keys[k_i])
                        (_, jv_k), _ = timed(
                            "ghat",
                            lambda v=v_k: jvp_jit(inner_w, v, fresh_state(), prefix_chunk, seq_chunk),
                            quiet=True,
                        )
                        proj = np.asarray(jv_k, dtype=np.float64) ** 2
                        sumsq = proj if sumsq is None else sumsq + proj
                    qhat_c = sumsq / float(K_G)
                    ghat_c = np.sqrt(qhat_c)
                    ihat_c = ahat_c - qhat_c / float(n_valid) if n_valid else np.full_like(ahat_c, np.nan)

                    def store_masked(src, dst):
                        out = np.where(mask_c, src, np.nan)
                        dst[si, c] = out.astype(np.float32)
                        return out

                    store_masked(ce_np, token_ce)
                    store_masked(ghat_c, ghat)
                    store_masked(qhat_c, qhat)
                    store_masked(ahat_c, ahat)
                    store_masked(ihat_c, ihat)
                    valid_arr[si, c] = mask_c.astype(np.uint8)
                    role_arr[si, c] = np.asarray([ROLE_CODES[r] for r in roles[g0 : g0 + MINI_BATCH]], dtype=np.int8)
                    span_arr[si, c] = np.asarray(span_uid[g0 : g0 + MINI_BATCH], dtype=np.int16)
                    g_bar_norms[si, c] = gbn
                    clip_cs[si, c] = c_clip
                    etas[si, c] = eta_eval
                    n_valid_chunk[si, c] = n_valid
                    nll_flat[g0 : g0 + MINI_BATCH] = np.where(mask_c, ce_np, np.nan)
                    ghat_flat[g0 : g0 + MINI_BATCH] = np.where(mask_c, ghat_c, np.nan)
                    ahat_flat[g0 : g0 + MINI_BATCH] = np.where(mask_c, ahat_c, np.nan)

                    targets_c = [g0 + i for i in range(MINI_BATCH) if mask_c[i] and is_needle_role(roles[g0 + i])]
                    pairs_c, unmatched_c = match_nll_same_chunk(
                        targets=targets_c, nll=nll_flat, valid=valid, roles=roles
                    )
                    for p in pairs_c:
                        rec = {
                            "task": task,
                            "sample_id": sid,
                            "span_uid": int(span_uid[p["t"]]),
                            **p,
                        }
                        match_rows.append(rec)
                        sample_pairs.append(rec)
                    if unmatched_c:
                        print(f"  chunk {c}: unmatched needle tokens {unmatched_c}", flush=True)

                    exact_locals = sorted({p["t"] % MINI_BATCH for p in pairs_c} | {p["match"] % MINI_BATCH for p in pairs_c})
                    for t_idx in exact_locals:
                        g_t, dt = timed(
                            "exact",
                            lambda t=int(t_idx): grad_token_jit(
                                inner_w, jnp.int32(t), fresh_state(), prefix_chunk, seq_chunk
                            ),
                            warmup=not compiled["exact"],
                            quiet=True,
                        )
                        compiled["exact"] = True
                        g_ex = float(pytree_l2_norm(g_t))
                        a_ex = float(pytree_vdot(g_t, g_bar))
                        i_ex = a_ex - (g_ex ** 2) / float(n_valid)
                        gt = g0 + t_idx
                        exact_rows.append(
                            {
                                "task": task,
                                "sample_id": sid,
                                "chunk": c,
                                "local_t": int(t_idx),
                                "global_t": int(gt),
                                "role": roles[gt],
                                "span_uid": int(span_uid[gt]),
                                "is_match": roles[gt] == "background",
                                "g_exact": g_ex,
                                "a_exact": a_ex,
                                "i_exact": i_ex,
                                "ghat": float(ghat_c[t_idx]),
                                "ahat": float(ahat_c[t_idx]),
                                "ihat": float(ihat_c[t_idx]),
                                "nll": float(ce_np[t_idx]),
                                "n_valid_chunk": n_valid,
                                "exact_s": dt,
                            }
                        )

                    w_unmoved = pytree_max_abs_diff(inner_before, inner_w)
                    out_step, _ = timed(
                        "update",
                        lambda: official_chunk_step(
                            model_inner,
                            inner_opt_state,
                            (state_all, suffix_state),
                            seq_chunk,
                            prefix_chunk,
                            model_outer,
                        ),
                        warmup=not compiled["update"],
                        quiet=True,
                    )
                    compiled["update"] = True
                    if w_unmoved != 0.0:
                        purity_ok = False
                        raise SystemExit("measurement mutated W")
                    model_inner = out_step.new_model
                    inner_opt_state = out_step.new_optimizer_state
                    state_all, suffix_state = out_step.new_state

                inner_now, _ = eqx.partition(model_outer, spec_inner)
                w0_drift = pytree_max_abs_diff(inner_w0_ref, inner_now)
                if w0_drift != 0.0:
                    raise SystemExit(f"checkpoint W0 mutated after {task} sid={sid}: {w0_drift}")
                sanity_all.append(
                    {
                        "task": task,
                        "sample_id": sid,
                        "purity_ok": purity_ok,
                        "w0_leaf_unchanged": True,
                        "any_out_of_swa": bool(row["any_relevant_out_of_swa"]),
                    }
                )
                valid_f = np.asarray(valid)
                spearman_all = spearman_safe(ghat_flat[valid_f], nll_flat[valid_f])
                spearman_chunk = []
                for c in range(n_chunks):
                    sl = slice(c * MINI_BATCH, (c + 1) * MINI_BATCH)
                    spearman_chunk.append(spearman_safe(ghat_flat[sl][valid_f[sl]], nll_flat[sl][valid_f[sl]]))
                spearman_rows.append(
                    {"task": task, "sample_id": sid, "all_valid": spearman_all, "per_chunk": spearman_chunk}
                )

                sl_s = {
                    "sample_id": sid,
                    "needle_all": summarize_pairs(
                        [p for p in sample_pairs if roles[p["t"]].startswith("relevant_")],
                        ghat_flat,
                        nll_flat,
                        ahat_flat,
                        valid,
                        roles,
                    ),
                    "relevant_value": summarize_pairs(
                        [p for p in sample_pairs if roles[p["t"]] == "relevant_value"],
                        ghat_flat,
                        nll_flat,
                        ahat_flat,
                        valid,
                        roles,
                    ),
                    "relevant_key": summarize_pairs(
                        [p for p in sample_pairs if roles[p["t"]] == "relevant_key"],
                        ghat_flat,
                        nll_flat,
                        ahat_flat,
                        valid,
                        roles,
                    ),
                    "relevant_sentence": summarize_pairs(
                        [p for p in sample_pairs if roles[p["t"]] == "relevant_sentence"],
                        ghat_flat,
                        nll_flat,
                        ahat_flat,
                        valid,
                        roles,
                    ),
                }
                if task == "niah_multikey_1":
                    uids = sorted({int(sp["span_index"]) + (0 if sp["role"] == "relevant" else 100) for sp in span_list})
                    rel_uid = [u for u in uids if u < 100]
                    dist_uid = [u for u in uids if u >= 100]
                    rel_stats = []
                    dist_stats = []
                    for u in rel_uid:
                        stt = summarize_pairs(
                            [p for p in sample_pairs if p["span_uid"] == u],
                            ghat_flat,
                            nll_flat,
                            ahat_flat,
                            valid,
                            roles,
                        )
                        if stt:
                            rel_stats.append(stt)
                    for u in dist_uid:
                        stt = summarize_pairs(
                            [p for p in sample_pairs if p["span_uid"] == u],
                            ghat_flat,
                            nll_flat,
                            ahat_flat,
                            valid,
                            roles,
                        )
                        if stt:
                            dist_stats.append(stt)
                    sl_s["relevant"] = rel_stats[0] if rel_stats else None
                    sl_s["distractors"] = dist_stats
                    if dist_stats:
                        sl_s["distractor_mean"] = {
                            "mean_d_g": float(np.mean([d["mean_d_g"] for d in dist_stats])),
                            "mean_percentile": float(np.mean([d["mean_percentile"] for d in dist_stats])),
                            "mean_nll_t": float(np.mean([d["mean_nll_t"] for d in dist_stats])),
                            "ahat_neg_frac": float(np.mean([d["ahat_neg_frac"] for d in dist_stats])),
                            "n_spans": len(dist_stats),
                        }
                    else:
                        sl_s["distractor_mean"] = None
                sample_level.append(sl_s)
                elapsed = time.perf_counter() - t_sample
                runtime["per_sample_s"].append(elapsed)
                print(
                    f"  spearman Ghat-NLL={spearman_all}  pairs={len(sample_pairs)}  "
                    f"wall_sample={elapsed:.1f}s",
                    flush=True,
                )

            np.savez(
                RESULT_DIR / f"{task}.npz",
                sample_ids=np.asarray(sids, dtype=np.int32),
                T=sample_T,
                token_ttt_ce=token_ce,
                ghat_k32=ghat,
                qhat_k32=qhat,
                ahat=ahat,
                ihat=ihat,
                valid_masks=valid_arr,
                roles=role_arr,
                span_uid=span_arr,
                g_bar_norm=g_bar_norms,
                clip_scale_c=clip_cs,
                eta=etas,
                n_valid_chunk=n_valid_chunk,
            )
            gaps = [p["gap"] for p in match_rows if p["task"] == task]
            if task == "niah_single_2":
                analysis = analyze_single2(sample_level, gaps)
            else:
                analysis = analyze_mk1(sample_level, gaps)
            task_outputs[task] = {
                "sample_ids": sids,
                "spearman_ghat_nll": spearman_rows,
                "sample_level": sample_level,
                "analysis": analysis,
            }

    if match_rows:
        pd.DataFrame(match_rows).to_parquet(RESULT_DIR / "matches.parquet", index=False)
    if exact_rows:
        pd.DataFrame(exact_rows).to_parquet(RESULT_DIR / "exact.parquet", index=False)

    exact_by_task = {}
    for task_name in selected:
        sub = [r for r in exact_rows if r["task"] == task_name]
        if not sub:
            exact_by_task[task_name] = None
            continue
        g_e = np.asarray([r["g_exact"] for r in sub])
        g_h = np.asarray([r["ghat"] for r in sub])
        a_e = np.asarray([r["a_exact"] for r in sub])
        a_h = np.asarray([r["ahat"] for r in sub])
        i_e = np.asarray([r["i_exact"] for r in sub])
        i_h = np.asarray([r["ihat"] for r in sub])
        exact_by_task[task_name] = {
            "n": len(sub),
            "ghat_vs_exact": error_metrics(g_e, g_h, np.ones(len(sub), dtype=bool), eps=REL_EPS),
            "ahat_vs_exact": error_metrics(a_e, a_h, np.ones(len(sub), dtype=bool), eps=REL_EPS),
            "ahat_sign": sign_agreement(a_e, a_h),
            "ihat_vs_exact_diagnostic_only": error_metrics(i_e, i_h, np.ones(len(sub), dtype=bool), eps=REL_EPS),
            "i_exact": median_iqr(i_e),
            "i_exact_n_neg": int(np.sum(i_e < 0)),
            "note": "Ihat is not treated as a substitute for I_exact",
        }

    spearman_summary = {}
    for task_name, blob in task_outputs.items():
        alls = [r["all_valid"] for r in blob["spearman_ghat_nll"] if r["all_valid"] is not None]
        chunk_vals = [
            v
            for r in blob["spearman_ghat_nll"]
            for v in r["per_chunk"]
            if v is not None
        ]
        spearman_summary[task_name] = {
            "per_sample_all_valid": median_iqr(alls),
            "per_chunk": median_iqr(chunk_vals),
        }

    runtime["wall_s"] = time.perf_counter() - wall0
    meta = {
        "name": "ruler_niah_mechanism_pilot_8k",
        "step": "7A",
        "not": "long-range memory / 16K / router / teacher saliency / formal significance / H1-H2 conclusion",
        "in_swa": True,
        "sample_select_seed": SAMPLE_SELECT_SEED,
        "selected_ids": selected,
        "K": K_G,
        "checkpoint_dir": str(CHECKPOINT_DIR),
        "backend": "cudnn" if not attention_cudnn_override else "xla_override",
        "fresh_w0_per_sample": True,
        "label_use": "offline analysis only; not an input to the model",
        "gpu": asdict(gpu),
        "attention": attn_setup,
        "software": {"jax": jax.__version__, "python": sys.version.split()[0]},
        "runtime_s": runtime,
        "sanity": sanity_all,
        "spearman_ghat_nll": spearman_summary,
        "exact_spotcheck": exact_by_task,
        "tasks": {k: {"analysis": v["analysis"], "sample_ids": v["sample_ids"]} for k, v in task_outputs.items()},
        "n_match_rows": len(match_rows),
        "n_exact_rows": len(exact_rows),
    }
    (RESULT_DIR / "summary.json").write_text(json.dumps(to_jsonable(meta), indent=2) + "\n")
    (RESULT_DIR / "sample_level.json").write_text(
        json.dumps(to_jsonable({k: v["sample_level"] for k, v in task_outputs.items()}), indent=2) + "\n"
    )
    print(f"\nresults: {RESULT_DIR}", flush=True)
    print(f"wall: {runtime['wall_s']:.1f}s  exact_rows={len(exact_rows)} matches={len(match_rows)}", flush=True)


if __name__ == "__main__":
    args = parse_args()
    main(attention_cudnn_override=args.attention_cudnn_override)
