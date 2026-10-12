"""Restore the official 3B 128K checkpoint and run a short T=1024 forward.

Does not modify e2e/, does not touch 125M artifacts, does not run 128K.

Checks:
- Orbax pytree match (checkpoint vs model)
- parameter count ~3B
- tokenizer vocab/BOS/EOS
- loss_for_sequence finite
- prefix_call + suffix_call logits shape [1024, vocab]
- GPU memory after the short forward
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from dataclasses import asdict
from pathlib import Path

_BASELINE_3B = Path(__file__).resolve().parents[1]
_BASELINE_125M_COMMON = Path(__file__).resolve().parents[2] / "baseline_125m" / "common"
# 125M helpers first, then 3B common last so `baseline_config` resolves to 3B.
sys.path.insert(0, str(_BASELINE_125M_COMMON))
sys.path.insert(0, str(_BASELINE_3B / "common"))

from baseline_config import (  # noqa: E402
    CHECKPOINT_DIR,
    CHECKPOINT_GCS,
    EXPERIMENT,
    PROBE_PATH,
    SANITY_RESULT_DIR,
    TOKENIZER_PATH,
    add_e2e_to_path,
    compose_3b_128k_config,
)
from runtime_helpers import AttentionOverride, describe_gpu, tokens_to_batch  # noqa: E402

DEFAULT_SEQ_LEN = 1024


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seq-len", type=int, default=DEFAULT_SEQ_LEN)
    p.add_argument(
        "--attention-cudnn-override",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="XLA fallback. Default is official cuDNN prefix SWA (A100).",
    )
    return p.parse_args()


def leaves_by_path(tree) -> dict:
    import jax

    return {jax.tree_util.keystr(path): value for path, value in jax.tree.flatten_with_path(tree)[0]}


def try_memory_stats():
    try:
        import jax

        stats = jax.devices()[0].memory_stats()
        if not stats:
            return None
        out = {}
        for k, v in stats.items():
            out[str(k)] = int(v) if isinstance(v, int) else v
        return out
    except Exception:
        return None


def nvidia_smi_used_mib() -> int | None:
    import subprocess

    try:
        line = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            text=True,
            timeout=10,
        ).strip().splitlines()[0]
        return int(line.strip())
    except Exception:
        return None


def main() -> None:
    args = parse_args()
    cfg = compose_3b_128k_config()
    add_e2e_to_path()

    import equinox as eqx
    import jax
    import jax.numpy as jnp
    import jax.random as jrandom
    import numpy as np
    from jax.sharding import PartitionSpec as P

    from ttt.infra.checkpoint import Checkpointer, unify_dict_with_eqx_module
    from ttt.model.data import Batch
    from ttt.model.sharding import ModelSharding
    from ttt.model.transformer import BlockCollectionSplit, MetaModel
    from ttt.utils.jax_utils import eval_shape_and_sharding, initialize_distibuted, set_random_seed

    initialize_distibuted(cfg.backend)
    gpu = describe_gpu()
    override = AttentionOverride(enabled=args.attention_cudnn_override)
    override.install()

    # train.py:123 — RoPE table is 2 * seq_len. For the 128K checkpoint this is 131072.
    cfg.model.seq_len = cfg.training.seq_length
    mini_batch = int(cfg.model.mini_batch_size)
    seq_len = int(args.seq_len)
    if seq_len % mini_batch != 0:
        raise SystemExit(f"--seq-len {seq_len} is not a multiple of mini_batch_size={mini_batch}")

    from tokenizers import Tokenizer

    if not TOKENIZER_PATH.exists():
        raise FileNotFoundError(f"tokenizer missing: {TOKENIZER_PATH}")
    tok = Tokenizer.from_file(str(TOKENIZER_PATH))
    tok_ok = {
        "path": str(TOKENIZER_PATH),
        "vocab": tok.get_vocab_size(with_added_tokens=True),
        "bos": tok.token_to_id("<|begin_of_text|>"),
        "eos": tok.token_to_id("<|end_of_text|>"),
        "vocab_matches_config": tok.get_vocab_size(with_added_tokens=True) == int(cfg.model.vocab_size),
        "bos_matches_config": tok.token_to_id("<|begin_of_text|>") == int(cfg.model.bos_token_id),
        "eos_matches_config": tok.token_to_id("<|end_of_text|>") == int(cfg.model.eos_token_id),
    }
    print("--- tokenizer ---")
    for k, v in tok_ok.items():
        print(f"  {k}: {v}")
    if not (tok_ok["vocab_matches_config"] and tok_ok["bos_matches_config"] and tok_ok["eos_matches_config"]):
        raise SystemExit("tokenizer does not match 3B config")

    probe = np.fromfile(PROBE_PATH, dtype="<u4")
    if probe.size < seq_len + 1:
        raise SystemExit(f"{PROBE_PATH} has {probe.size} tokens, need {seq_len + 1}")
    seq = tokens_to_batch(jnp.asarray(probe[: seq_len + 1], dtype=jnp.int32), bos_token_id=int(cfg.model.bos_token_id))

    key = set_random_seed(cfg.training.model_seed)
    model_sharding = ModelSharding(cfg)
    mesh = model_sharding.mesh

    @eqx.filter_jit
    def create_sharded_model_and_state():
        model, state = eqx.nn.make_with_state(MetaModel)(cfg, key=key)
        state = jax.device_put(state, jax.NamedSharding(mesh, P()))
        model = model_sharding.shard_params(model)
        return model, state

    print("--- restore ---")
    print(f"checkpoint dir: {CHECKPOINT_DIR}")
    print(f"gcs:            {CHECKPOINT_GCS}")
    abstract_model_weights = eval_shape_and_sharding(lambda: create_sharded_model_and_state()[0].weights())
    abstract_n = len(leaves_by_path(abstract_model_weights))
    print(f"abstract model_weights leaves: {abstract_n}")

    loader = Checkpointer(config=cfg, for_saving=False)
    if not loader.checkpoint_exists():
        raise FileNotFoundError(
            f"No Orbax checkpoint under {CHECKPOINT_DIR}. "
            f"Download with: python scripts/baseline_3b/download_checkpoint.py"
        )
    print(f"available steps: {loader.manager.all_steps()}  latest: {loader.manager.latest_step()}")

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        out_state = loader.load_checkpoint(
            step=cfg.training.resume_step,
            targets={"model_weights": abstract_model_weights},
            restore=cfg.training.load_part,
        )
    ckpt_only = [str(w.message) for w in caught]
    restored_leaves = leaves_by_path(out_state["model_weights"])
    print(f"paths in checkpoint but missing from model: {len(ckpt_only)}")
    for message in ckpt_only:
        print(f"  WARN {message}")

    model_fresh, state = create_sharded_model_and_state()
    model, missing = unify_dict_with_eqx_module(out_state["model_weights"], model_fresh)
    print(f"paths in model but missing from checkpoint: {len(missing)}")
    for path in missing[:20]:
        print(f"  WARN {path}")
    if missing:
        raise SystemExit(f"checkpoint did not cover {len(missing)} model paths")

    fresh = leaves_by_path(model_fresh.weights())
    final = leaves_by_path(model.weights())
    unchanged = [k for k in fresh if jnp.array_equal(fresh[k], final[k])]
    n_all = int(sum(x.size for x in jax.tree_util.tree_leaves(model.weights())))
    n_trainable = int(sum(x.size for x in jax.tree_util.tree_leaves(model.trainable_parameters())))
    print(f"leaves overwritten: {len(fresh) - len(unchanged)} / {len(fresh)}")
    print(f"#all params:       {n_all:,}")
    print(f"#trainable params: {n_trainable:,}")
    del out_state, model_fresh

    state = state.set(model.step_index, jnp.array(jnp.iinfo(jnp.int32).max - 100, dtype=jnp.int32))
    mem_after_restore = try_memory_stats()
    smi_after_restore = nvidia_smi_used_mib()
    print(f"nvidia-smi used after restore: {smi_after_restore} MiB")

    @eqx.filter_jit
    def ntp(model, seq, state):
        return model.loss_for_sequence(seq, state)

    attention_cfg = override.report_for_config(
        force_flash=bool(cfg.model.force_flash),
        seq_modeling_block=str(cfg.model.seq_modeling_block),
    )
    print("--- short forward T=1024 ---")
    print(f"GPU: {gpu.name}  SWA={cfg.model.sliding_window_size}  chunk={mini_batch}")
    print(f"RoPE seq_len={cfg.model.seq_len}  suffix_len={cfg.model.suffix_len}")
    print(f"override enabled: {attention_cfg['override_enabled']}")

    with mesh:
        loss, metrics = ntp(model, seq, state)
        loss.block_until_ready()
        nll = metrics[MetaModel.MetricType.token_nll_loss]
        nll_f = nll.astype(jnp.float32)
        loss_f = float(loss)
        any_nan = bool(jnp.any(jnp.isnan(nll_f)) or jnp.isnan(loss))
        any_inf = bool(jnp.any(jnp.isinf(nll_f)) or jnp.isinf(loss))
        print(f"loss_for_sequence: {loss_f:.6f}")
        print(f"token_nll shape: {tuple(nll.shape)}  min/mean/max: {float(jnp.min(nll_f)):.4f}/{float(jnp.mean(nll_f)):.4f}/{float(jnp.max(nll_f)):.4f}")
        print(f"NaN={any_nan} Inf={any_inf}")

        # Same generation interface as Exp 01 125M: prefix_call + suffix_call.
        block_collection = model.language_model.model.h.blocks
        prime_storage = model.language_model.model.h.prime_storage
        state_prefix_suffix = state.substate(model.language_model.model.h.blocks)
        state_prefix, state_suffix = BlockCollectionSplit.split_state(state_prefix_suffix, cfg.model.suffix_len)
        split = BlockCollectionSplit(cfg.model, block_collection=block_collection, prime_storage=prime_storage, key=jrandom.PRNGKey(0))
        model_split = eqx.tree_at(lambda m: m.language_model.model.h, model, split)
        model_split = jax.tree.map(lambda p: p.astype(model_split.state_dtype), model_split)

        def logits_via_prefix_suffix(m, seq, state_prefix, state_suffix):
            xt = m.language_model.wte_call(seq.input_ids)
            prefix_out = m.language_model.prefix_call(
                m.language_model.model.h.prefix_blocks, xt, state_prefix, seq
            )
            lm_out = m.language_model.suffix_call(
                prefix_outputs=prefix_out.last_hidden_state,
                state=state_suffix,
                seq=seq,
            )
            return lm_out.logits

        logits_fn = eqx.filter_jit(logits_via_prefix_suffix)
        logits = logits_fn(model_split, seq, state_prefix, state_suffix)
        logits.block_until_ready()
        logits_shape = tuple(int(x) for x in logits.shape)
        logits_f = logits.astype(jnp.float32)
        logits_finite = bool(jnp.all(jnp.isfinite(logits_f)))
        print(f"prefix+suffix logits shape: {logits_shape} dtype={logits.dtype} all_finite={logits_finite}")
        expected = (seq_len, int(cfg.model.vocab_size))
        if logits_shape != expected:
            raise SystemExit(f"logits shape {logits_shape} != {expected}")

    mem_after_fwd = try_memory_stats()
    smi_after_fwd = nvidia_smi_used_mib()
    print(f"nvidia-smi used after short forward: {smi_after_fwd} MiB")

    peak_bytes = None
    if mem_after_fwd and "peak_bytes_in_use" in mem_after_fwd:
        peak_bytes = mem_after_fwd["peak_bytes_in_use"]
        print(f"JAX peak_bytes_in_use: {peak_bytes} ({peak_bytes / (1024**3):.2f} GiB)")

    SANITY_RESULT_DIR.mkdir(parents=True, exist_ok=True)
    result = {
        "experiment": EXPERIMENT,
        "checkpoint_gcs": CHECKPOINT_GCS,
        "checkpoint_dir": str(CHECKPOINT_DIR),
        "checkpoint_step": int(loader.manager.latest_step()) if loader.manager.latest_step() is not None else None,
        "tokenizer": tok_ok,
        "model": {
            "name": str(cfg.model.name),
            "hidden_size": int(cfg.model.hidden_size),
            "num_hidden_layers": int(cfg.model.num_hidden_layers),
            "num_attention_heads": int(cfg.model.num_attention_heads),
            "intermediate_size": int(cfg.model.intermediate_size),
            "suffix_len": int(cfg.model.suffix_len),
            "prefix_len": int(cfg.model.num_hidden_layers) - int(cfg.model.suffix_len),
            "sliding_window_size": int(cfg.model.sliding_window_size),
            "mini_batch_size": mini_batch,
            "seq_modeling_block": str(cfg.model.seq_modeling_block),
            "rope_theta": float(cfg.model.rope_theta),
            "model_seq_len_rope": int(cfg.model.seq_len),
            "training_seq_length": int(cfg.training.seq_length),
            "spec_inner": list(cfg.training.spec_inner),
            "train_mode": str(cfg.training.train_mode),
            "param_dtype": str(cfg.model.param_dtype),
            "compute_dtype": str(cfg.model.compute_dtype),
            "state_dtype": str(cfg.model.state_dtype),
        },
        "restore": {
            "abstract_leaves": abstract_n,
            "ckpt_only_warnings": len(ckpt_only),
            "model_only_missing": len(missing),
            "leaves_overwritten": len(fresh) - len(unchanged),
            "leaves_total": len(fresh),
            "n_all_params": n_all,
            "n_trainable_params": n_trainable,
        },
        "forward": {
            "T": seq_len,
            "loss": loss_f,
            "token_nll_shape": list(nll.shape),
            "any_nan": any_nan,
            "any_inf": any_inf,
            "logits_shape": list(logits_shape),
            "logits_all_finite": logits_finite,
            "prefix_suffix_interface_unchanged": True,
        },
        "gpu": asdict(gpu),
        "attention": attention_cfg,
        "memory": {
            "nvidia_smi_used_mib_after_restore": smi_after_restore,
            "nvidia_smi_used_mib_after_forward": smi_after_fwd,
            "jax_after_restore": mem_after_restore,
            "jax_after_forward": mem_after_fwd,
            "jax_peak_bytes_in_use": peak_bytes,
        },
    }
    out = SANITY_RESULT_DIR / "restore_and_forward_t1024.json"
    out.write_text(json.dumps(result, indent=2, default=str) + "\n", encoding="utf-8")
    print(f"wrote {out}")
    if any_nan or any_inf or not logits_finite:
        raise SystemExit("short forward produced NaN/Inf")


if __name__ == "__main__":
    main()
