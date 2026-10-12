"""STEP 8A: RULER 8K within-sample ranking (NLL vs log Ghat).

Not classification. Not long-range memory (8K is in-SWA). Not a router.
Does not edit e2e/. Does not overwrite STEP 7A / label parquets.

M0: mean NLL
M1: mean log(Ghat)
M2: mean NLL + mean log(Ghat)
Scorer: regularized linear pairwise logistic, one scorer per dataset.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import asdict
from pathlib import Path

_BASELINE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_BASELINE_ROOT / "common"))

from baseline_config import (  # noqa: E402
    CHECKPOINT_DIR,
    EXPERIMENT_RESULT_DIR,
    REPO_ROOT,
    add_e2e_to_path,
    compose_baseline_config,
)
from ranking_windows import (  # noqa: E402
    BOOTSTRAP_SEED,
    L_PRIMARY,
    LAMBDA_GRID,
    LOG_G_EPS,
    N_BOOTSTRAP,
    SHUFFLE_SEED,
    SPLIT_SEED,
    WindowRow,
    bootstrap_delta,
    build_mk2_windows,
    build_single2_windows,
    fit_linear_ranker,
    mean_metrics_from_scores,
    pairwise_diffs,
    per_sample_rank_table,
    select_lambda,
    shuffle_logg_within_sample,
    split_sample_ids,
    standardize_apply,
    standardize_fit,
    window_features_from_tokens,
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
N_SAMPLES = 200
JVP_SEED = 0
K_G = 32
CHECKPOINT_NAME = "125m_ttt_e2e_pretrain_dclm_8k_1x_cc"
RESULT_DIR = EXPERIMENT_RESULT_DIR / "ruler_predictive_ranking_8k"
NVRTC_LIB = REPO_ROOT / ".venv/lib/python3.12/site-packages/nvidia/cuda_nvrtc/lib"

TASKS = (
    {"name": "niah_single_2", "kind": "single2", "salt": 0},
    {"name": "niah_multikey_2", "kind": "mk2", "salt": 1},
)

FEATURE_KEYS = {
    "M0": ["mean_nll"],
    "M1": ["mean_log_ghat"],
    "M2": ["mean_nll", "mean_log_ghat"],
    "M2_shuffled": ["mean_nll", "mean_log_ghat"],
}

DELTA_KEYS = ("mrr", "recall_at_1", "recall_at_5", "normalized_rank")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--phase", choices=("windows", "features", "rank", "all"), default="all")
    p.add_argument("--limit", type=int, default=0, help="0 = all 200 samples per task")
    p.add_argument(
        "--attention-cudnn-override",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Default off: official cuDNN prefix SWA.",
    )
    p.add_argument("--force", action="store_true", help="Allow replacing files in the STEP 8A directory")
    return p.parse_args()


def refuse_overwrite(path: Path, *, force: bool) -> None:
    if path.exists() and not force:
        raise SystemExit(f"refusing to overwrite {path} (pass --force)")


def dump_json(path: Path, obj, *, force: bool) -> None:
    refuse_overwrite(path, force=force)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, default=str) + "\n")


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


def ensure_nvrtc() -> None:
    if not NVRTC_LIB.is_dir():
        return
    cur = os.environ.get("LD_LIBRARY_PATH", "")
    prefix = str(NVRTC_LIB)
    if prefix not in cur.split(":"):
        os.environ["LD_LIBRARY_PATH"] = prefix + (":" + cur if cur else "")


def summarize_int(values) -> dict:
    import numpy as np

    x = np.asarray(list(values), dtype=np.int64)
    return {
        "n": int(x.size),
        "min": int(x.min()) if x.size else None,
        "median": float(np.median(x)) if x.size else None,
        "max": int(x.max()) if x.size else None,
        "mean": float(x.mean()) if x.size else None,
    }


def window_sanity_report(rows: list[WindowRow], kind: str) -> dict:
    pos = [r for r in rows if r.is_positive]
    neg = [r for r in rows if not r.is_positive]
    n_samples = len({r.sample_id for r in rows})
    pos_by = {}
    neg_by = {}
    for r in pos:
        pos_by.setdefault(r.sample_id, 0)
        pos_by[r.sample_id] += 1
    for r in neg:
        neg_by.setdefault(r.sample_id, 0)
        neg_by[r.sample_id] += 1
    n_pos_overlap = sum(r.query_or_pad_overlap for r in pos)
    n_pos_fail = sum(r.placement_failed for r in pos)
    n_neg_fail = sum(r.placement_failed for r in neg)
    n_kv = sum(r.kv_fully_inside for r in pos)
    n_wrong_len = sum(r.n_tokens != L_PRIMARY for r in rows)
    missing_bg = [sid for sid in pos_by if sid not in neg_by] if kind == "single2" else []
    problems = []
    if n_pos_overlap:
        problems.append(f"{n_pos_overlap} positive windows overlap query/padding")
    if n_pos_fail or n_neg_fail:
        problems.append(f"placement_failed pos={n_pos_fail} neg={n_neg_fail}")
    if pos and n_kv != len(pos):
        problems.append(f"kv not fully inside positive window: {n_kv}/{len(pos)}")
    if n_wrong_len:
        problems.append(f"{n_wrong_len} windows with length != {L_PRIMARY}")
    if missing_bg:
        problems.append(f"{len(missing_bg)} samples with zero background windows")
    if any(v != 1 for v in pos_by.values()):
        problems.append("a sample does not have exactly 1 positive window")
    return {
        "kind": kind,
        "n_samples": n_samples,
        "n_positive_windows": len(pos),
        "n_negative_windows": len(neg),
        "kv_fully_inside_positive": n_kv,
        "kv_fully_inside_rate": (n_kv / len(pos)) if pos else None,
        "positive_query_or_pad_overlap": n_pos_overlap,
        "placement_failed_positive": n_pos_fail,
        "placement_failed_negative": n_neg_fail,
        "negative_windows_per_sample": summarize_int(neg_by.values()) if neg_by else summarize_int([]),
        "problems": problems,
        "stop": bool(problems),
    }


def phase_windows(*, force: bool, limit: int) -> dict:
    import pandas as pd

    examples, spans_df = load_task_tables()
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    all_rows: list[dict] = []
    sanity = {}
    splits = {}
    n_take = limit if limit else N_SAMPLES
    for spec in TASKS:
        task = spec["name"]
        pool = examples[(examples["task"] == task) & (examples["seq_length_label"] == SEQ_LABEL)]
        ids = sorted(int(x) for x in pool["sample_id"].tolist())
        if ids != list(range(N_SAMPLES)):
            raise SystemExit(f"{task} sample_id set is not 0..199")
        use = ids[:n_take]
        splits[task] = split_sample_ids(n_take, seed=SPLIT_SEED, salt=spec["salt"]) if n_take == N_SAMPLES else {
            "note": f"limit={n_take}; full 60/20/20 only when n=200",
            "all": use,
        }
        built: list[WindowRow] = []
        for sid in use:
            row, span_list = sample_rows(examples, spans_df, task, sid)
            kwargs = dict(
                task=task,
                sample_id=sid,
                T=int(row["T"]),
                T_pad=int(row["T_pad"]),
                query_t0=int(row["query_loss_t_start"]),
                query_t1=int(row["query_loss_t_end"]),
                spans=span_list,
                L=L_PRIMARY,
            )
            if int(row["T_pad"]) != 8192:
                raise SystemExit(f"{task} sid={sid} T_pad={row['T_pad']} != 8192")
            if spec["kind"] == "single2":
                built.extend(build_single2_windows(**kwargs))
            else:
                built.extend(build_mk2_windows(**kwargs))
        sanity[task] = window_sanity_report(built, spec["kind"])
        print(f"windows {task}: {json.dumps(sanity[task])}", flush=True)
        all_rows.extend(asdict(r) for r in built)
    stop = any(v["stop"] for v in sanity.values())
    out = {
        "L": L_PRIMARY,
        "seq_label": SEQ_LABEL,
        "n_requested": n_take,
        "split_seed": SPLIT_SEED,
        "sanity": sanity,
        "stop": stop,
    }
    dump_json(RESULT_DIR / "window_sanity.json", out, force=force)
    dump_json(RESULT_DIR / "split_ids.json", splits, force=force)
    win_path = RESULT_DIR / "window_index.parquet"
    refuse_overwrite(win_path, force=force)
    pd.DataFrame(all_rows).to_parquet(win_path, index=False)
    if stop:
        print("WINDOW SANITY STOP: see window_sanity.json. Not starting GPU.", flush=True)
    return out


def build_batch(ids: list[int], T: int, T_pad: int):
    import jax.numpy as jnp
    import numpy as np

    if len(ids) != T:
        raise SystemExit(f"input_ids len {len(ids)} != T={T}")
    if ids[0] != BOS_ID:
        raise SystemExit("sequence does not start with BOS")
    padded, T2, T_pad2 = right_pad_ids(ids, pad_id=PAD_ID, mini_batch=MINI_BATCH)
    if T2 != T or T_pad2 != T_pad:
        raise SystemExit("padding length mismatch")
    tokens = padded + [PAD_ID]
    seq = tokens_to_batch(jnp.asarray(tokens, dtype=jnp.int32), bos_token_id=BOS_ID)
    mask = np.asarray(seq.loss_masks).astype(bool)
    label_mask = np.asarray(ntp_loss_mask(T, T_pad), dtype=bool)
    if not np.array_equal(mask[: T_pad - 1], label_mask) or mask[-1] or int(mask.sum()) != T - 1:
        raise SystemExit("NTP mask does not match t < T-1")
    return seq, mask


def inspect_inexact_paths(tree) -> list[dict]:
    import equinox as eqx
    import jax

    rows = []
    for path, value in jax.tree.flatten_with_path(tree)[0]:
        if eqx.is_inexact_array(value):
            rows.append({"path": jax.tree_util.keystr(path)})
    return rows


def block_until_ready(tree):
    import jax

    jax.tree.map(lambda x: x.block_until_ready() if hasattr(x, "block_until_ready") else x, tree)
    return tree


def phase_features(*, attention_cudnn_override: bool, force: bool, limit: int) -> None:
    """Official TTT trajectory; token NLL + K=32 Ghat only. No Ahat/exact/matching."""
    import numpy as np

    from measurement_utils import pytree_max_abs_diff, rademacher_like, token_ttt_ce_from_logits

    ensure_nvrtc()
    examples, spans_df = load_task_tables()
    n_take = limit if limit else N_SAMPLES
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
    print(f"GPU={gpu.name} override={attention_cudnn_override} seq_len={cfg.model.seq_len}", flush=True)

    n_chunks = 8
    compiled = {"f": False, "jvp": False, "update": False, "prefix": False}

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

        def jvp_token_ce(inner, tangent, st, prefix_chunk, seq_chunk):
            def f(w):
                return token_ttt_ce_only(w, st, prefix_chunk, seq_chunk)

            return eqx.filter_jvp(f, (inner,), (tangent,))

        f_jit = eqx.filter_jit(token_ttt_ce_only)
        jvp_jit = eqx.filter_jit(jvp_token_ce)
        prefix_fn = eqx.filter_checkpoint(model_outer.language_model.prefix_call)

        for spec in TASKS:
            task = spec["name"]
            sids = list(range(n_take))
            nll = np.full((n_take, 8192), np.nan, dtype=np.float32)
            ghat = np.full_like(nll, np.nan)
            valid_arr = np.zeros((n_take, 8192), dtype=np.uint8)
            sample_T = np.zeros((n_take,), dtype=np.int32)
            t0 = time.perf_counter()
            for si, sid in enumerate(sids):
                row, _spans = sample_rows(examples, spans_df, task, sid)
                T = int(row["T"])
                T_pad = int(row["T_pad"])
                ids = [int(x) for x in row["input_ids"]]
                seq, valid = build_batch(ids, T, T_pad)
                sample_T[si] = T
                print(f"\n===== features {task} {si + 1}/{n_take} sid={sid} T={T} =====", flush=True)

                model_inner = clone_pytree(model_outer)
                inner_opt_state = model_outer.inner_optimizer(state_all_0).init(model_outer.inner_parameters())
                suffix_state = clone_pytree(state_suffix_0)
                state_all = clone_pytree(state_all_0)
                xt_embed = model_outer.language_model.wte_call(seq.input_ids)
                prefix_output = prefix_fn(
                    model_outer.language_model.model.h.prefix_blocks, xt_embed, state_prefix, seq
                ).last_hidden_state
                block_until_ready(prefix_output)
                compiled["prefix"] = True
                seq_chunks = tree_rearrange(seq, "(chunk token) ... -> chunk token ...", token=mini_batch)
                prefix_chunks = tree_rearrange(prefix_output, "(chunk token) ... -> chunk token ...", token=mini_batch)

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
                    g0 = c * MINI_BATCH
                    ce = np.asarray(f_jit(inner_w, fresh_state(), prefix_chunk, seq_chunk), dtype=np.float64)
                    compiled["f"] = True
                    jvp_keys = jrandom.split(jrandom.PRNGKey(JVP_SEED), K_G)
                    sumsq = None
                    for k_i in range(K_G):
                        v_k = rademacher_like(inner_w, jvp_keys[k_i])
                        _, jv_k = jvp_jit(inner_w, v_k, fresh_state(), prefix_chunk, seq_chunk)
                        proj = np.asarray(jv_k, dtype=np.float64) ** 2
                        sumsq = proj if sumsq is None else sumsq + proj
                    compiled["jvp"] = True
                    ghat_c = np.sqrt(sumsq / float(K_G))
                    nll[si, g0 : g0 + MINI_BATCH] = np.where(mask_c, ce, np.nan).astype(np.float32)
                    ghat[si, g0 : g0 + MINI_BATCH] = np.where(mask_c, ghat_c, np.nan).astype(np.float32)
                    valid_arr[si, g0 : g0 + MINI_BATCH] = mask_c.astype(np.uint8)

                    w_unmoved = pytree_max_abs_diff(inner_before, inner_w)
                    out_step = official_chunk_step(
                        model_inner,
                        inner_opt_state,
                        (state_all, suffix_state),
                        seq_chunk,
                        prefix_chunk,
                        model_outer,
                    )
                    block_until_ready(out_step.new_model)
                    compiled["update"] = True
                    if w_unmoved != 0.0:
                        raise SystemExit("measurement mutated W")
                    model_inner = out_step.new_model
                    inner_opt_state = out_step.new_optimizer_state
                    state_all, suffix_state = out_step.new_state

                inner_now, _ = eqx.partition(model_outer, spec_inner)
                w0_drift = pytree_max_abs_diff(inner_w0_ref, inner_now)
                if w0_drift != 0.0:
                    raise SystemExit(f"checkpoint W0 mutated after {task} sid={sid}: {w0_drift}")
                print(f"  done sid={sid} elapsed_all={time.perf_counter() - t0:.1f}s", flush=True)

            path = RESULT_DIR / (
                f"token_features_{task}_limit{n_take}.npz" if n_take != N_SAMPLES else f"token_features_{task}.npz"
            )
            refuse_overwrite(path, force=force)
            np.savez_compressed(
                path,
                sample_ids=np.asarray(sids, dtype=np.int32),
                T=sample_T,
                nll=nll,
                ghat=ghat,
                valid=valid_arr,
                K=np.int32(K_G),
                jvp_seed=np.int32(JVP_SEED),
            )
            print(f"wrote {path}", flush=True)


def load_token_features(task: str):
    import numpy as np

    blob = np.load(RESULT_DIR / f"token_features_{task}.npz")
    return blob


def attach_window_features(index_df, task: str):
    import numpy as np
    import pandas as pd

    blob = load_token_features(task)
    sids = np.asarray(blob["sample_ids"])
    sid_to_i = {int(s): i for i, s in enumerate(sids)}
    nll = blob["nll"]
    ghat = blob["ghat"]
    sub = index_df[index_df["task"] == task].copy()
    feats = []
    for rec in sub.to_dict("records"):
        i = sid_to_i[int(rec["sample_id"])]
        feats.append(
            window_features_from_tokens(
                nll=nll[i],
                ghat=ghat[i],
                w0=int(rec["start_t"]),
                w1=int(rec["end_t"]),
            )
        )
    feat_df = pd.DataFrame(feats, index=sub.index)
    return pd.concat([sub, feat_df], axis=1)


def model_matrix(df, keys: list[str], *, shuffled_logg=None):
    import numpy as np

    cols = []
    for k in keys:
        if k == "mean_log_ghat" and shuffled_logg is not None:
            cols.append(shuffled_logg)
        else:
            cols.append(np.asarray(df[k], dtype=np.float64))
    return np.stack(cols, axis=1)


def eval_split_tables(df, w, keys, mu, sd, *, shuffled_logg=None):
    import numpy as np

    X = standardize_apply(model_matrix(df, keys, shuffled_logg=shuffled_logg), mu, sd)
    scores = X @ w
    y = np.asarray(df["is_positive"], dtype=np.int32)
    sid = np.asarray(df["sample_id"], dtype=np.int32)
    starts = np.asarray(df["start_t"], dtype=np.int32)
    mean_m = mean_metrics_from_scores(scores, y, sid, starts)
    rows = per_sample_rank_table(scores, y, sid, starts)
    return mean_m, rows, scores


def rank_one_dataset(df, splits: dict, task: str, salt: int) -> dict:
    import numpy as np

    def subset(name: str):
        ids = set(splits[name])
        return df[df["sample_id"].isin(ids)].copy()

    train, val, test = subset("train"), subset("val"), subset("test")
    out = {"n_train": int(train["sample_id"].nunique()), "n_val": int(val["sample_id"].nunique()), "n_test": int(test["sample_id"].nunique())}
    models = {}
    val_metrics = {}
    test_metrics = {}
    test_rows = {}
    for name, keys in FEATURE_KEYS.items():
        tr = train
        va = val
        te = test
        sh_tr = sh_va = sh_te = None
        if name == "M2_shuffled":
            sh_tr = shuffle_logg_within_sample(
                np.asarray(tr["mean_log_ghat"], dtype=np.float64),
                np.asarray(tr["sample_id"]),
                seed=SHUFFLE_SEED,
                salt=salt,
            )
            sh_va = shuffle_logg_within_sample(
                np.asarray(va["mean_log_ghat"], dtype=np.float64),
                np.asarray(va["sample_id"]),
                seed=SHUFFLE_SEED,
                salt=salt + 50,
            )
            sh_te = shuffle_logg_within_sample(
                np.asarray(te["mean_log_ghat"], dtype=np.float64),
                np.asarray(te["sample_id"]),
                seed=SHUFFLE_SEED,
                salt=salt + 100,
            )
        Xtr_raw = model_matrix(tr, keys, shuffled_logg=sh_tr)
        mu, sd = standardize_fit(Xtr_raw)
        Xtr = standardize_apply(Xtr_raw, mu, sd)
        diffs = pairwise_diffs(Xtr, np.asarray(tr["is_positive"]), np.asarray(tr["sample_id"]))
        Xva = standardize_apply(model_matrix(va, keys, shuffled_logg=sh_va), mu, sd)
        lam, w, lam_info = select_lambda(
            diffs,
            Xva,
            np.asarray(va["is_positive"]),
            np.asarray(va["sample_id"]),
            np.asarray(va["start_t"]),
        )
        w = fit_linear_ranker(diffs, lam)
        val_m, _, _ = eval_split_tables(va, w, keys, mu, sd, shuffled_logg=sh_va)
        tes_m, tes_rows, _ = eval_split_tables(te, w, keys, mu, sd, shuffled_logg=sh_te)
        models[name] = {
            "keys": keys,
            "lambda": lam,
            "w": w.tolist(),
            "w_on_standardized_features": True,
            "train_mean": mu.tolist(),
            "train_std": sd.tolist(),
            "lambda_grid": lam_info,
            "note": "score = w^T ((x-mean)/std); intercept omitted (cancels in within-sample diffs)",
        }
        val_metrics[name] = val_m
        test_metrics[name] = tes_m
        test_rows[name] = tes_rows
        print(f"  {task} {name} val_MRR={val_m['mrr']:.4f} test_MRR={tes_m['mrr']:.4f} w={w.tolist()} lam={lam}", flush=True)

    deltas = bootstrap_delta(test_rows["M0"], test_rows["M2"], DELTA_KEYS, n_boot=N_BOOTSTRAP, seed=BOOTSTRAP_SEED + salt)
    deltas_sh = bootstrap_delta(
        test_rows["M0"], test_rows["M2_shuffled"], DELTA_KEYS, n_boot=N_BOOTSTRAP, seed=BOOTSTRAP_SEED + salt + 7
    )
    out.update(
        {
            "models": models,
            "validation": val_metrics,
            "test": test_metrics,
            "test_per_sample": test_rows,
            "bootstrap_M2_minus_M0": deltas,
            "bootstrap_M2shuffled_minus_M0": deltas_sh,
        }
    )
    return out


def write_report(payload: dict, path: Path, *, force: bool) -> None:
    refuse_overwrite(path, force=force)
    lines = [
        "# STEP 8A — RULER within-sample ranking (8K, 125M)",
        "",
        "Not long-range memory. 8K examples are in-SWA. Ghat is the K=32 JVP estimator.",
        "G is not importance. A positive M2−M0 delta is not a router design.",
        "",
        "## 1. Samples",
        "",
        f"- checkpoint: `{CHECKPOINT_NAME}`",
        f"- tasks: niah_single_2 / 8k and niah_multikey_2 / 8k, n={N_SAMPLES} each",
        f"- split seed {SPLIT_SEED}, 60/20/20 by sample_id",
        "",
        "## 2–3. Candidate windows",
        "",
        "```json",
        json.dumps(payload["window_sanity"], indent=2),
        "```",
        "",
        "## 4. Feature sanity",
        "",
        "```json",
        json.dumps(payload.get("feature_sanity"), indent=2),
        "```",
        "",
        "## 5–7. Metrics, deltas, coefficients",
        "",
        "```json",
        json.dumps(
            {
                t: {
                    "validation": blob["validation"],
                    "test": blob["test"],
                    "bootstrap_M2_minus_M0": blob["bootstrap_M2_minus_M0"],
                    "bootstrap_M2shuffled_minus_M0": blob["bootstrap_M2shuffled_minus_M0"],
                    "coefficients": {k: v["w"] for k, v in blob["models"].items()},
                    "lambda": {k: v["lambda"] for k, v in blob["models"].items()},
                }
                for t, blob in payload["ranking"].items()
            },
            indent=2,
        ),
        "```",
        "",
        "## 8–10. Leakage / interpretation",
        "",
        payload["interpretation"],
        "",
    ]
    path.write_text("\n".join(lines) + "\n")


def feature_sanity(df) -> dict:
    import numpy as np

    out = {}
    for task, sub in df.groupby("task"):
        g = np.asarray(sub["mean_ghat"], dtype=np.float64)
        nll = np.asarray(sub["mean_nll"], dtype=np.float64)
        logg = np.asarray(sub["mean_log_ghat"], dtype=np.float64)
        out[str(task)] = {
            "n_windows": int(len(sub)),
            "nll_finite": bool(np.all(np.isfinite(nll))),
            "ghat_finite": bool(np.all(np.isfinite(g))),
            "ghat_nonneg": bool(np.all(g >= 0)),
            "mean_nll": float(np.mean(nll)),
            "mean_ghat": float(np.mean(g)),
            "mean_log_ghat": float(np.mean(logg)),
            "spearman_window_nll_logg": None,
        }
        if np.std(nll) > 0 and np.std(logg) > 0:
            from scipy.stats import spearmanr

            out[str(task)]["spearman_window_nll_logg"] = float(spearmanr(nll, logg).statistic)
    return out


def interpret(ranking: dict, sanity: dict) -> str:
    bits = [
        "Allowed claim scope: within-sample ranking of L=32 token windows on in-SWA 8K RULER.",
        "M2 vs M0 asks whether log(Ghat) adds ranking signal beyond mean NLL on the same window.",
        "Shuffle control: M2-shuffled should be near M0. Improvement under shuffle is treated as fitting artifact, not a G signal.",
        "Surprisal remains the first confounder (STEP 7A Spearman(Ghat,NLL)≈0.80 at token grain).",
        "Do not treat a ranking gain as evidence for explicit memory, a router, or causal importance.",
        "8K is inside SWA=8192; this is not a long-range TTT-memory result.",
    ]
    for task, blob in ranking.items():
        d = blob["bootstrap_M2_minus_M0"]["mrr"]
        sh = blob["bootstrap_M2shuffled_minus_M0"]["mrr"]
        bits.append(
            f"{task} test ΔMRR(M2-M0)={d['point']:.4f} 95% CI [{d['ci95_lo']:.4f},{d['ci95_hi']:.4f}]; "
            f"ΔMRR(M2shuffled-M0)={sh['point']:.4f}."
        )
        lo, hi = d["ci95_lo"], d["ci95_hi"]
        if lo > 0:
            bits.append(f"{task}: CI excludes 0 on ΔMRR; still not a causal / long-range claim.")
        elif hi < 0:
            bits.append(f"{task}: M2 worse than M0 on ΔMRR CI.")
        else:
            bits.append(f"{task}: ΔMRR CI includes 0; no stable ranking gain of M2 over M0 on this test split.")
    return "\n".join(bits)


def phase_rank(*, force: bool) -> None:
    import pandas as pd

    index = pd.read_parquet(RESULT_DIR / "window_index.parquet")
    splits = json.loads((RESULT_DIR / "split_ids.json").read_text())
    parts = []
    for spec in TASKS:
        parts.append(attach_window_features(index, spec["name"]))
    feat = pd.concat(parts, ignore_index=True)
    feat_path = RESULT_DIR / "window_features.parquet"
    refuse_overwrite(feat_path, force=force)
    feat.to_parquet(feat_path, index=False)
    f_san = feature_sanity(feat)
    ranking = {}
    for spec in TASKS:
        task = spec["name"]
        print(f"\n--- rank {task} ---", flush=True)
        ranking[task] = rank_one_dataset(feat[feat["task"] == task].copy(), splits[task], task, spec["salt"])
    win_san = json.loads((RESULT_DIR / "window_sanity.json").read_text())
    payload = {
        "checkpoint": CHECKPOINT_NAME,
        "L": L_PRIMARY,
        "K": K_G,
        "jvp_seed": JVP_SEED,
        "split_seed": SPLIT_SEED,
        "lambda_grid": list(LAMBDA_GRID),
        "log_g_eps": LOG_G_EPS,
        "feature_sanity": f_san,
        "window_sanity": win_san["sanity"],
        "ranking": ranking,
        "interpretation": interpret(ranking, win_san),
    }
    dump_json(RESULT_DIR / "config.json", {
        "hypothesis": "Within a sample, does mean log(Ghat) add ranking signal for the needle window beyond mean NLL?",
        "independent_variable": "scorer features {NLL, log Ghat, both}",
        "dependent_metrics": list(DELTA_KEYS) + ["recall_at_1pct", "percentile_rank"],
        "baseline": "M0 mean NLL linear scorer",
        "controlled": [
            "125M 8K checkpoint",
            "official inner_loop_step trajectory",
            "K=32 JVP Ghat",
            "L=32",
            "sample_id splits",
            "in-SWA 8K",
        ],
        "confounders": ["surprisal", "needle format", "window overlap in MK2", "JVP noise in Ghat"],
        "dataset": "RULER niah_single_2 and niah_multikey_2 validation 8k",
        "n_samples": N_SAMPLES,
        "context_length": "8k padded to 8192",
        "model": CHECKPOINT_NAME,
        "seed": {"split": SPLIT_SEED, "jvp": JVP_SEED, "shuffle": SHUFFLE_SEED, "bootstrap": BOOTSTRAP_SEED},
        "precision": "official bf16 compute, Ghat from fp32 JVP accum",
        "hardware": "A100 cuDNN (script default --no-attention-cudnn-override)",
        "mini_batch": MINI_BATCH,
        "window": L_PRIMARY,
        "K": K_G,
    }, force=force)
    dump_json(RESULT_DIR / "validation_metrics.json", {t: v["validation"] for t, v in ranking.items()}, force=force)
    dump_json(RESULT_DIR / "test_metrics.json", {t: v["test"] for t, v in ranking.items()}, force=force)
    dump_json(RESULT_DIR / "coefficients.json", {t: v["models"] for t, v in ranking.items()}, force=force)
    dump_json(RESULT_DIR / "bootstrap_summary.json", {
        t: {
            "M2_minus_M0": v["bootstrap_M2_minus_M0"],
            "M2shuffled_minus_M0": v["bootstrap_M2shuffled_minus_M0"],
        }
        for t, v in ranking.items()
    }, force=force)
    dump_json(RESULT_DIR / "shuffle_control.json", {t: v["test"]["M2_shuffled"] for t, v in ranking.items()}, force=force)
    dump_json(RESULT_DIR / "ranking_full.json", ranking, force=force)
    write_report(payload, RESULT_DIR / "report.md", force=force)
    print("\n" + payload["interpretation"], flush=True)
    print(f"\nwrote {RESULT_DIR}", flush=True)


def main() -> None:
    args = parse_args()
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    if args.phase in ("windows", "all"):
        info = phase_windows(force=args.force, limit=args.limit)
        if info["stop"]:
            return
    if args.phase in ("features", "all"):
        san_path = RESULT_DIR / "window_sanity.json"
        if san_path.exists():
            san = json.loads(san_path.read_text())
            if san.get("stop"):
                print("window sanity previously STOP; not running GPU features", flush=True)
                return
        phase_features(
            attention_cudnn_override=args.attention_cudnn_override,
            force=args.force,
            limit=args.limit,
        )
    if args.phase in ("rank", "all"):
        if args.limit:
            print("rank phase expects full 200-sample features; skip under --limit", flush=True)
            return
        phase_rank(force=args.force)


if __name__ == "__main__":
    main()
