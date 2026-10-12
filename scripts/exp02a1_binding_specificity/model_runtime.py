"""Qwen3-4B-Instruct-2507 BF16 load + LoRA wrap for Exp 02A-1.

Copied from scripts/exp02a_source_free_lora/model_runtime.py so this experiment
has its own namespace. Do not change LoRA rank/alpha/targets, optimizer, LR,
precision, or checkpoint relative to the validated Exp 02A K=32 recipe.

Does not touch `e2e/` or any Exp 02A frozen datasets/results.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any

import torch
from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

from protocol import SPEC

MODEL_ID = SPEC["backbone"]["model_id"]
REVISION = SPEC["backbone"]["revision_sha"]

WATCH_FROZEN_SUBSTRINGS = (
    "layers.0.self_attn.q_proj",
    "layers.0.self_attn.o_proj",
    "layers.35.self_attn.q_proj",
    "layers.35.self_attn.o_proj",
)


@dataclass
class LoadedRuntime:
    tokenizer: Any
    model: Any
    device: torch.device
    revision: str
    model_id: str
    lora_init_cpu: dict[str, torch.Tensor]
    frozen_watch_cpu: dict[str, torch.Tensor]
    frozen_watch_hash: str
    trainable_names: list[str]
    n_trainable: int
    n_frozen: int
    lora_config: dict[str, Any]
    torch_dtype: str
    gpu_name: str


def _is_lora_param(name: str) -> bool:
    return "lora_A" in name or "lora_B" in name or "lora_embedding" in name


def lora_state_cpu(model) -> dict[str, torch.Tensor]:
    out = {}
    for name, param in model.named_parameters():
        if _is_lora_param(name):
            out[name] = param.detach().cpu().contiguous().clone()
    if not out:
        raise RuntimeError("no LoRA parameters")
    return out


def load_lora_state_(model, state: dict[str, torch.Tensor]) -> None:
    with torch.no_grad():
        found = 0
        for name, param in model.named_parameters():
            if not _is_lora_param(name):
                continue
            if name not in state:
                raise RuntimeError(f"missing LoRA tensor in snapshot: {name}")
            param.copy_(state[name].to(device=param.device, dtype=param.dtype))
            found += 1
        if found != len(state):
            raise RuntimeError(f"LoRA restore count {found} != snapshot {len(state)}")


def lora_max_abs_diff(model, state: dict[str, torch.Tensor]) -> float:
    m = 0.0
    with torch.no_grad():
        for name, param in model.named_parameters():
            if not _is_lora_param(name):
                continue
            ref = state[name].to(device=param.device, dtype=param.dtype)
            diff = (param.detach() - ref).abs().max()
            val = float(diff.item())
            if val > m:
                m = val
    return m


def lora_l2_delta(model, state: dict[str, torch.Tensor]) -> float:
    tot = 0.0
    with torch.no_grad():
        for name, param in model.named_parameters():
            if not _is_lora_param(name):
                continue
            ref = state[name].to(device=param.device, dtype=param.dtype)
            d = (param.detach().float() - ref.float()).norm()
            tot += float(d.item()) ** 2
    return float(tot ** 0.5)


def snapshot_frozen_watch(model) -> dict[str, torch.Tensor]:
    """Keep a small on-device copy of frozen q/o projections. Do not clone embeddings.

    Full embed/lm_head snapshots were first attempted and then dropped: copying
    those matrices to CPU and casting BF16->FP32 on every example pinned the
    host for minutes (the first smoke run hung there). Layer-0/35 q_proj and
    o_proj base weights are enough to detect accidental unfreezing.
    """
    out = {}
    for name, param in model.named_parameters():
        if _is_lora_param(name):
            continue
        if any(s in name for s in WATCH_FROZEN_SUBSTRINGS) and name.endswith("weight"):
            out[name] = param.detach().clone()
    if len(out) < 4:
        raise RuntimeError(f"frozen watch set too small: {list(out)}")
    return out


def frozen_watch_hash(watch: dict[str, torch.Tensor]) -> str:
    h = hashlib.sha256()
    for name in sorted(watch):
        t = watch[name].detach().contiguous().cpu()
        h.update(name.encode())
        h.update(str(tuple(t.shape)).encode())
        h.update(str(t.dtype).encode())
        h.update(t.view(torch.uint8).numpy().tobytes())
    return h.hexdigest()


def frozen_watch_max_abs_diff(model, watch: dict[str, torch.Tensor]) -> float:
    m = 0.0
    with torch.no_grad():
        by_name = dict(model.named_parameters())
        if set(watch) - set(by_name):
            raise RuntimeError("frozen watch keys missing from model")
        for name, tensor in watch.items():
            param = by_name[name]
            if _is_lora_param(name) or param.requires_grad:
                raise RuntimeError(f"watch tensor became trainable: {name}")
            diff = (param.detach() - tensor).abs().max()
            val = float(diff.item())
            if val > m:
                m = val
    return m


def make_adamw(model, *, lr: float, weight_decay: float):
    params = [p for n, p in model.named_parameters() if _is_lora_param(n)]
    if not params:
        raise RuntimeError("AdamW got no LoRA params")
    for p in params:
        p.requires_grad_(True)
    return torch.optim.AdamW(
        params,
        lr=lr,
        weight_decay=weight_decay,
        betas=(0.9, 0.999),
        eps=1e-8,
    )


def set_adapter_enabled(model, enabled: bool) -> None:
    """PEFT 0.21: toggle on the inner tuner, not PeftModel.enable_adapter_layers."""
    if enabled:
        model.base_model.enable_adapter_layers()
        model._adapters_disabled = False
        for name, param in model.named_parameters():
            if _is_lora_param(name):
                param.requires_grad_(True)
    else:
        model.base_model.disable_adapter_layers()
        model._adapters_disabled = True


def assert_base_frozen(model) -> None:
    for name, param in model.named_parameters():
        if _is_lora_param(name):
            if not param.requires_grad:
                raise RuntimeError(f"LoRA param is frozen: {name}")
            continue
        if param.requires_grad:
            raise RuntimeError(f"base param is trainable: {name}")


def load_runtime(*, device: str | None = None) -> LoadedRuntime:
    if device is None:
        if not torch.cuda.is_available():
            raise RuntimeError("Exp 02A smoke requires CUDA")
        device_obj = torch.device("cuda")
    else:
        device_obj = torch.device(device)

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_ID,
        revision=REVISION,
        use_fast=True,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        revision=REVISION,
        dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    )
    if model.dtype != torch.bfloat16:
        # Some modules may be fp32 (e.g. norms). Check parameter dtypes below.
        pass
    dtypes = {str(p.dtype) for p in model.parameters()}
    if "torch.float8_e4m3fn" in dtypes or "torch.float8_e5m2" in dtypes:
        raise RuntimeError(f"FP8 weights loaded; forbidden. dtypes={dtypes}")
    if torch.bfloat16 not in {p.dtype for p in model.parameters()}:
        raise RuntimeError(f"expected BF16 parameters, got {dtypes}")

    model.to(device_obj)
    lora_cfg = LoraConfig(
        r=int(SPEC["lora"]["r"]),
        lora_alpha=int(SPEC["lora"]["alpha"]),
        lora_dropout=float(SPEC["lora"]["dropout"]),
        bias=str(SPEC["lora"]["bias"]),
        target_modules=list(SPEC["lora"]["target_modules"]),
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
    )
    model = get_peft_model(model, lora_cfg)
    for name, param in model.named_parameters():
        if param.requires_grad:
            param.data = param.data.to(dtype=torch.float32)
    assert_base_frozen(model)
    model.config.use_cache = False

    trainable = [n for n, p in model.named_parameters() if _is_lora_param(n)]
    frozen_n = sum(1 for n, p in model.named_parameters() if not _is_lora_param(n))
    if not any(".q_proj." in n or n.endswith("q_proj.lora_A.default.weight") or "q_proj" in n for n in trainable):
        raise RuntimeError(f"no q_proj LoRA tensors: {trainable[:8]}")
    if not any("o_proj" in n for n in trainable):
        raise RuntimeError("no o_proj LoRA tensors")
    n_layers_q = len([n for n in trainable if "q_proj" in n and "lora_A" in n])
    n_layers_o = len([n for n in trainable if "o_proj" in n and "lora_A" in n])
    if n_layers_q != 36 or n_layers_o != 36:
        raise RuntimeError(f"expected LoRA on 36 layers, got q={n_layers_q} o={n_layers_o}")

    lora_init = lora_state_cpu(model)
    frozen_watch = snapshot_frozen_watch(model)
    gpu_name = torch.cuda.get_device_name(device_obj) if device_obj.type == "cuda" else "cpu"
    return LoadedRuntime(
        tokenizer=tokenizer,
        model=model,
        device=device_obj,
        revision=REVISION,
        model_id=MODEL_ID,
        lora_init_cpu=lora_init,
        frozen_watch_cpu=frozen_watch,
        frozen_watch_hash=frozen_watch_hash(frozen_watch),
        trainable_names=trainable,
        n_trainable=len(trainable),
        n_frozen=frozen_n,
        lora_config={
            "target_modules": list(SPEC["lora"]["target_modules"]),
            "r": SPEC["lora"]["r"],
            "alpha": SPEC["lora"]["alpha"],
            "dropout": SPEC["lora"]["dropout"],
            "bias": SPEC["lora"]["bias"],
            "n_q_proj_layers": n_layers_q,
            "n_o_proj_layers": n_layers_o,
            "trainable_dtype": "float32",
            "base_dtype": "bfloat16",
        },
        torch_dtype="bfloat16",
        gpu_name=gpu_name,
    )


def reset_example_state(runtime: LoadedRuntime):
    """Restore init LoRA and return a fresh AdamW. Call at the start of every example/condition."""
    set_adapter_enabled(runtime.model, True)
    load_lora_state_(runtime.model, runtime.lora_init_cpu)
    diff = lora_max_abs_diff(runtime.model, runtime.lora_init_cpu)
    if diff != 0.0:
        raise RuntimeError(f"LoRA reset failed, max_abs_diff={diff}")
    opt = make_adamw(
        runtime.model,
        lr=float(SPEC["lora"]["lr"]),
        weight_decay=float(SPEC["lora"]["weight_decay"]),
    )
    return opt
