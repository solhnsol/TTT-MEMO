"""Stage 3b: next-token loss on a real DCLM val prefix, restored vs random init.

No official-repo edits, no gradient logging, no TTT-MEMO modules. The only
question is:

    On the same DCLM validation tokens, is the restored checkpoint's next-token
    loss clearly lower than an identically configured random initialization?

Official path used: `MetaModel.loss_for_sequence` (what `eval_step_fn` calls).
We do not go through `eval_step_fn` itself, because that vmaps over a data-parallel
axis and `pmean`s -- unnecessary for a single sequence, and it still would not
return logits.

Default T is 1024 (`--seq-len`). It must be a multiple of `mini_batch_size=1024`.
`--seq-len 2048` is two inner-loop chunks; the official `metrics["loss"]` array is
then shape `[2]` (1-D arrays are not flattened). Chunk-1 loss is not assumed to
be lower than chunk-0: the texts differ, and KV cache also carries across chunks.

`cfg.model.seq_len` stays 8192. That field sizes the RoPE table (`2 * seq_len`),
which the suffix path indexes up to `sliding_window_size + mini_batch_size - 1`.
Setting it equal to T would produce NaN RoPE values. `training.seq_length`
also stays 8192: `loss_for_sequence` only uses it in `seqlen % mini_batch_size == 0`,
and the actual chunk count comes from the Batch length.

Attention kernel: this host currently cannot run the official prefix
`implementation="cudnn"` call. `--attention-cudnn-override` (default on) is a
script-local RTX 3060 compatibility workaround that rewrites only that keyword
to `"xla"`. It is not a model or attention-semantics change. Turn it off with
`--no-attention-cudnn-override`.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from baseline_config import CHECKPOINT_DIR, REPO_ROOT, add_e2e_to_path, compose_baseline_config  # noqa: E402

PROBE_PATH = REPO_ROOT / "data" / "_probe_val_head.bin"
RESULT_DIR = REPO_ROOT / "results" / "raw" / "baseline_125m"

# Smallest T the official SWA suffix path accepts under this checkpoint's config.
DEFAULT_SEQ_LEN = 1024

# Official call sites in e2e/ttt/model/attention.py for this checkpoint
# (seq_modeling_block=SWA, force_flash=False):
#   prefix: SWA.full_sw_attention  -> implementation="cudnn"  (hardcoded)
#   suffix: SWA.core_attention_op  -> implementation=None     (force_flash is False)
# Attention / SWAFull are not used by this experiment.


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--seq-len",
        type=int,
        default=DEFAULT_SEQ_LEN,
        help=(
            "Batch length T in tokens. Must be a multiple of mini_batch_size "
            f"(1024). Default {DEFAULT_SEQ_LEN}. Does not change cfg.model.seq_len."
        ),
    )
    parser.add_argument(
        "--attention-cudnn-override",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "RTX 3060 compatibility workaround: redirect official "
            "implementation='cudnn' requests to implementation='xla'. "
            "Does not change model architecture or attention semantics. "
            "Disable with --no-attention-cudnn-override."
        ),
    )
    return parser.parse_args()


@dataclass
class GpuInfo:
    name: str
    compute_capability: str | None
    jax_device: str
    jax_backend: str


def describe_gpu() -> GpuInfo:
    import jax

    device = jax.devices()[0]
    name = getattr(device, "device_kind", str(device))
    compute_capability = None
    try:
        line = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,compute_cap", "--format=csv,noheader"],
            text=True,
            timeout=10,
        ).strip().splitlines()[0]
        smi_name, smi_cc = [part.strip() for part in line.split(",", 1)]
        name = smi_name
        compute_capability = smi_cc
    except (FileNotFoundError, subprocess.SubprocessError, ValueError):
        pass
    return GpuInfo(
        name=name,
        compute_capability=compute_capability,
        jax_device=str(device),
        jax_backend=jax.default_backend(),
    )


@dataclass
class AttentionOverride:
    """Process-local RTX 3060 compatibility workaround for this sanity-check script.

    Not a change to model architecture or attention semantics. The official
    prefix SWA path hardcodes jax.nn.dot_product_attention(..., implementation="cudnn")
    in e2e/ttt/model/attention.py:312. On this host that kernel currently has no
    valid cuDNN engine; the same shapes run with implementation="xla". The root
    cause of the cuDNN failure is not established here (missing NVRTC is one
    hypothesis, not a finding).

    The wrapper only rewrites the `implementation` keyword when its value is
    exactly "cudnn". Every other argument is forwarded unchanged. For the 125M
    E2E config (force_flash=False) the only such call is prefix SWA; suffix
    already passes implementation=None, so it is not rewritten.
    """

    enabled: bool
    n_seen: int = 0
    n_redirected: int = 0
    n_passthrough: int = 0
    installed: bool = False

    def install(self) -> None:
        if not self.enabled:
            return
        import jax
        import jax.nn as jnn

        original = jnn.dot_product_attention

        def wrapped(*args, **kwargs):
            self.n_seen += 1
            if kwargs.get("implementation") == "cudnn":
                kwargs = dict(kwargs)
                kwargs["implementation"] = "xla"
                self.n_redirected += 1
            else:
                self.n_passthrough += 1
            return original(*args, **kwargs)

        jnn.dot_product_attention = wrapped
        jax.nn.dot_product_attention = wrapped
        self.installed = True

    def report_for_config(self, *, force_flash: bool, seq_modeling_block: str) -> dict:
        """Requested vs effective backends implied by the official source + this flag.

        Suffix is unaffected unless force_flash becomes True, in which case the
        official suffix path would also request cudnn and this wrapper would
        redirect it too. That is not this checkpoint's config.
        """
        prefix_requested = "cudnn"
        suffix_requested = "cudnn" if force_flash else None
        prefix_effective = "xla" if self.enabled else prefix_requested
        suffix_effective = "xla" if (self.enabled and suffix_requested == "cudnn") else suffix_requested
        return {
            "override_enabled": self.enabled,
            "override_installed": self.installed,
            "seq_modeling_block": seq_modeling_block,
            "force_flash": force_flash,
            "prefix_requested_backend": prefix_requested,
            "prefix_effective_backend": prefix_effective,
            "suffix_requested_backend": suffix_requested,
            "suffix_effective_backend": suffix_effective,
            "redirects_only_when_implementation_is_cudnn": True,
            "n_dot_product_attention_seen": self.n_seen,
            "n_cudnn_redirected_to_xla": self.n_redirected,
            "n_passthrough": self.n_passthrough,
        }


def tokens_to_batch(tokens, *, bos_token_id: int):
    """Same slicing as `ttt.dataloader.lm_dataset._to_batch`, minus the unused eos arg."""
    from ttt.model.data import Batch

    return Batch(
        input_ids=tokens[:-1],
        target_tokens=tokens[1:],
        loss_masks=(tokens[1:] != bos_token_id),
    )


def summarize_nll(name: str, nll, mask) -> dict[str, float | bool]:
    import jax.numpy as jnp

    nll_f = nll.astype(jnp.float32)
    masked = jnp.where(mask, nll_f, jnp.nan)
    finite = jnp.isfinite(nll_f)
    stats = {
        "mean_masked": float(jnp.nanmean(masked)),
        "min": float(jnp.min(nll_f)),
        "mean": float(jnp.mean(nll_f)),
        "max": float(jnp.max(nll_f)),
        "any_nan": bool(jnp.any(jnp.isnan(nll_f))),
        "any_inf": bool(jnp.any(jnp.isinf(nll_f))),
        "all_finite": bool(jnp.all(finite)),
    }
    print(f"\n{name}")
    print(f"  mean NTP loss (loss_mask): {stats['mean_masked']:.6f}")
    print(f"  per-token NLL min/mean/max: {stats['min']:.6f} / {stats['mean']:.6f} / {stats['max']:.6f}")
    print(f"  NaN: {stats['any_nan']}  Inf: {stats['any_inf']}  all_finite: {stats['all_finite']}")
    return stats


def chunk_loss_report(name: str, chunk_losses, nll, mask, mini_batch: int) -> list[dict]:
    """Read official per-chunk scalar CE (`metrics[loss]`, shape [n_chunks]) plus
    the matching slice of flattened token NLL. Does not recompute the model.
    """
    import jax.numpy as jnp
    import numpy as np

    chunk_losses = jnp.asarray(chunk_losses)
    nll = jnp.asarray(nll)
    mask = jnp.asarray(mask)
    n_chunks = int(chunk_losses.shape[0])
    rows = []
    print(f"\n{name} per-chunk NTP (official metrics['loss'] shape {tuple(chunk_losses.shape)})")
    for i in range(n_chunks):
        sl = slice(i * mini_batch, (i + 1) * mini_batch)
        nll_i = nll[sl].astype(jnp.float32)
        mask_i = mask[sl]
        row = {
            "chunk": i,
            "mean_ntp": float(chunk_losses[i]),
            "token_nll_masked_mean": float(jnp.nanmean(jnp.where(mask_i, nll_i, jnp.nan))),
            "any_nan": bool(jnp.any(jnp.isnan(chunk_losses[i])) or jnp.any(jnp.isnan(nll_i))),
            "any_inf": bool(jnp.any(jnp.isinf(chunk_losses[i])) or jnp.any(jnp.isinf(nll_i))),
        }
        rows.append(row)
        print(
            f"  chunk {i}: mean NTP={row['mean_ntp']:.6f}  "
            f"NaN={row['any_nan']}  Inf={row['any_inf']}"
        )
    overall = float(np.mean([r["mean_ntp"] for r in rows]))
    print(f"  overall mean NTP (mean of chunk scalars): {overall:.6f}")
    return rows


def main(attention_cudnn_override: bool = True, seq_len: int = DEFAULT_SEQ_LEN) -> None:
    cfg = compose_baseline_config()
    add_e2e_to_path()

    import equinox as eqx
    import jax
    import jax.numpy as jnp
    import numpy as np
    from jax.sharding import PartitionSpec as P

    from ttt.infra.checkpoint import Checkpointer, unify_dict_with_eqx_module
    from ttt.model.sharding import ModelSharding
    from ttt.model.transformer import MetaModel
    from ttt.utils.jax_utils import eval_shape_and_sharding, initialize_distibuted, set_random_seed

    initialize_distibuted(cfg.backend)
    gpu = describe_gpu()
    override = AttentionOverride(enabled=attention_cudnn_override)
    override.install()

    # train.py:123. RoPE table size. Must remain the pretrain value, not SEQ_LEN.
    cfg.model.seq_len = cfg.training.seq_length

    mini_batch = int(cfg.model.mini_batch_size)
    if seq_len % mini_batch != 0:
        raise SystemExit(f"--seq-len {seq_len} is not a multiple of mini_batch_size={mini_batch}")

    probe = np.fromfile(PROBE_PATH, dtype="<u4")
    n_needed = seq_len + 1
    if probe.size < n_needed:
        raise SystemExit(f"{PROBE_PATH} has {probe.size} tokens, need {n_needed}")

    tokens = jnp.asarray(probe[:n_needed], dtype=jnp.int32)
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

    model_random, state = create_sharded_model_and_state()
    model_restored, missing = unify_dict_with_eqx_module(out_state["model_weights"], model_random)
    if missing:
        raise SystemExit(f"checkpoint did not cover {len(missing)} model paths: {missing[:8]}")
    del out_state

    # train.py:224 -- eval uses a fully warmed-up inner LR. Without this, inner SGD
    # would run at ~0.1x the trained inner learning rate (ilr_warmup_steps=480).
    # At T=1024 the reported loss does not depend on that update, but the official
    # path still executes it, so match eval.
    state = state.set(model_restored.step_index, jnp.array(jnp.iinfo(jnp.int32).max - 100, dtype=jnp.int32))

    M = MetaModel.MetricType

    @eqx.filter_jit
    def ntp(model, seq, state):
        return model.loss_for_sequence(seq, state)

    print("--- setup ---")
    print(f"GPU name:            {gpu.name}")
    print(f"compute capability:  {gpu.compute_capability}")
    print(f"jax device:          {gpu.jax_device}")
    print(f"jax backend:         {gpu.jax_backend}")
    print(f"mesh:                {mesh}")
    attention_cfg = override.report_for_config(
        force_flash=bool(cfg.model.force_flash),
        seq_modeling_block=str(cfg.model.seq_modeling_block),
    )
    print(f"seq_modeling_block:  {attention_cfg['seq_modeling_block']}")
    print(f"force_flash:         {attention_cfg['force_flash']}")
    print(f"override enabled:    {attention_cfg['override_enabled']}")
    print(f"prefix requested:    {attention_cfg['prefix_requested_backend']}")
    print(f"prefix effective:    {attention_cfg['prefix_effective_backend']}")
    print(f"suffix requested:    {attention_cfg['suffix_requested_backend']}")
    print(f"suffix effective:    {attention_cfg['suffix_effective_backend']}")
    n_chunks = seq_len // mini_batch
    print(f"sequence length T:   {seq_len}")
    print(f"mini_batch_size:     {mini_batch}")
    print(f"n inner-loop chunks: {n_chunks}")
    print(f"train_mode:          {cfg.training.train_mode}")
    print(f"param_dtype:         {cfg.model.param_dtype}")
    print(f"compute_dtype:       {cfg.model.compute_dtype}")
    print(f"state_dtype:         {cfg.model.state_dtype}")
    print(f"model.seq_len (RoPE table = 2*this): {cfg.model.seq_len}")
    print(f"training.seq_length (assert only):   {cfg.training.seq_length}")
    print(f"input_ids:     {tuple(seq.input_ids.shape)} {seq.input_ids.dtype}")
    print(f"target_tokens: {tuple(seq.target_tokens.shape)} {seq.target_tokens.dtype}")
    print(f"loss_masks:    {tuple(seq.loss_masks.shape)}  valid={int(seq.loss_masks.sum())}/{seq.loss_masks.size}")
    # Official path never returns a logits tensor. Each inner-loop chunk materializes
    # `[mini_batch_size, vocab_size]` inside `CausalLM.suffix_call`, then discards it
    # after `lm_loss`. Concatenated equivalent would be `[T, vocab_size]`.
    print(f"logits (per chunk, official path): ({mini_batch}, {cfg.model.vocab_size})")
    print(f"logits (concatenated equivalent):  ({seq_len}, {cfg.model.vocab_size})")

    with mesh:
        print("\n--- forward: restored checkpoint ---")
        loss_ckpt, metrics_ckpt = ntp(model_restored, seq, state)
        print("\n--- forward: random init ---")
        loss_rand, metrics_rand = ntp(model_random, seq, state)

    attention_cfg = override.report_for_config(
        force_flash=bool(cfg.model.force_flash),
        seq_modeling_block=str(cfg.model.seq_modeling_block),
    )
    print("\n--- attention override counts (Python tracing of jax.nn.dot_product_attention) ---")
    print(f"seen: {attention_cfg['n_dot_product_attention_seen']}  "
          f"cudnn->xla: {attention_cfg['n_cudnn_redirected_to_xla']}  "
          f"passthrough: {attention_cfg['n_passthrough']}")
    print("passthrough includes suffix (implementation=None under force_flash=False).")

    nll_ckpt = metrics_ckpt[M.token_nll_loss]
    nll_rand = metrics_rand[M.token_nll_loss]
    chunk_loss_ckpt = metrics_ckpt[M.loss]
    chunk_loss_rand = metrics_rand[M.loss]
    # Scalar `loss` from loss_for_sequence is the mean of per-chunk masked CE.
    # metrics[loss] stays 1-D of length n_chunks (flatten leaves ndim==1 arrays).
    print(f"\nloss_for_sequence scalar (restored): {float(loss_ckpt):.6f}  dtype={loss_ckpt.dtype}")
    print(f"loss_for_sequence scalar (random):   {float(loss_rand):.6f}  dtype={loss_rand.dtype}")
    print(f"metrics[loss] shape (restored): {tuple(chunk_loss_ckpt.shape)} {chunk_loss_ckpt.dtype}")
    print(f"token_nll_loss shape/dtype (restored): {tuple(nll_ckpt.shape)} {nll_ckpt.dtype}")
    if tuple(chunk_loss_ckpt.shape) != (n_chunks,):
        raise SystemExit(
            f"expected metrics['loss'] shape {(n_chunks,)}, got {tuple(chunk_loss_ckpt.shape)}"
        )

    ckpt_chunks = chunk_loss_report("checkpoint restored", chunk_loss_ckpt, nll_ckpt, seq.loss_masks, mini_batch)
    rand_chunks = chunk_loss_report("random init", chunk_loss_rand, nll_rand, seq.loss_masks, mini_batch)

    ckpt_stats = summarize_nll("checkpoint restored", nll_ckpt, seq.loss_masks)
    rand_stats = summarize_nll("random init", nll_rand, seq.loss_masks)

    gap = rand_stats["mean_masked"] - ckpt_stats["mean_masked"]
    print("\n--- comparison (same tokens, same official meta-mode path) ---")
    print(f"random - restored (masked mean): {gap:.6f}")
    print(f"restored clearly lower: {ckpt_stats['mean_masked'] < rand_stats['mean_masked']}")
    print("no hardcoded PASS/FAIL threshold; this gap is the sanity check.")

    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    stem = RESULT_DIR / f"forward_sanity_t{seq_len}"
    np.savez(
        stem.with_suffix(".npz"),
        input_ids=np.asarray(seq.input_ids),
        target_tokens=np.asarray(seq.target_tokens),
        loss_masks=np.asarray(seq.loss_masks),
        token_nll_checkpoint=np.asarray(nll_ckpt),
        token_nll_random=np.asarray(nll_rand),
        chunk_loss_checkpoint=np.asarray(chunk_loss_ckpt),
        chunk_loss_random=np.asarray(chunk_loss_rand),
    )
    meta = {
        "seq_len": seq_len,
        "mini_batch_size": mini_batch,
        "n_inner_loop_chunks": n_chunks,
        "train_mode": str(cfg.training.train_mode),
        "param_dtype": str(cfg.model.param_dtype),
        "compute_dtype": str(cfg.model.compute_dtype),
        "gpu": asdict(gpu),
        "attention": attention_cfg,
        "checkpoint_dir": str(CHECKPOINT_DIR),
        "probe_path": str(PROBE_PATH),
        "logits_per_chunk": [mini_batch, int(cfg.model.vocab_size)],
        "logits_concat_equivalent": [seq_len, int(cfg.model.vocab_size)],
        "restored": ckpt_stats,
        "random": rand_stats,
        "restored_chunks": ckpt_chunks,
        "random_chunks": rand_chunks,
        "random_minus_restored_masked_mean": gap,
    }
    stem.with_suffix(".json").write_text(json.dumps(meta, indent=2) + "\n")
    print(f"\nraw per-token arrays: {stem.with_suffix('.npz')}")
    print(f"run metadata:         {stem.with_suffix('.json')}")


if __name__ == "__main__":
    args = parse_args()
    main(attention_cudnn_override=args.attention_cudnn_override, seq_len=args.seq_len)
