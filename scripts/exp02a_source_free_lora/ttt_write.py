"""NTP LoRA write (inner-loop / test-time) for Exp 02A.

Gradient flow
-------------
Loss: mean causal next-token CE on oracle payload tokens only
      (HuggingFace CausalLM: labels already -100 on wrappers; the model shifts
      logits[..., :-1, :] vs labels[..., 1:] internally).
W.r.t.: LoRA A/B of q_proj and o_proj in all 36 layers (requires_grad=True,
        float32). This is the test-time inner-loop update, not an outer-loop
        meta-gradient.
Not differentiated: every base-model parameter.
Aggregation: one sequence, mean over labeled tokens, repeated for K optimizer
             steps. There is a single textual fact exposure; K is the optimizer
             budget, not K copies of the fact in the context.

After the K steps this function returns only Python scalars / lists. It does
not return input_ids, labels, logits, or past_key_values.
"""

from __future__ import annotations

from typing import Any

import torch

from model_runtime import LoadedRuntime, lora_l2_delta, lora_max_abs_diff, set_adapter_enabled
from prompts import EncodedSequence, to_tensors


@torch.no_grad()
def _loss_value(model, batch: dict[str, torch.Tensor]) -> float:
    out = model(
        input_ids=batch["input_ids"],
        attention_mask=batch["attention_mask"],
        labels=batch["labels"],
        use_cache=False,
        past_key_values=None,
    )
    # Some transformers versions still populate past_key_values internally.
    # The invariant is that we never keep or return that object.
    loss = float(out.loss.detach().float().cpu())
    del out
    return loss


def write_fact(
    runtime: LoadedRuntime,
    encoded: EncodedSequence,
    optimizer,
    *,
    k_steps: int,
) -> dict[str, Any]:
    """Run K AdamW steps on one write sequence. Drops all write tensors before return."""
    model = runtime.model
    set_adapter_enabled(model, True)
    model.train()
    batch = to_tensors(encoded, runtime.device, include_labels=True)

    loss_before_steps: list[float] = []
    loss_after_steps: list[float] = []
    trajectory = [_loss_value(model, batch)]

    for step in range(k_steps):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        out = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            labels=batch["labels"],
            use_cache=False,
            past_key_values=None,
        )
        loss = out.loss
        if loss is None:
            raise RuntimeError("write loss is None")
        loss_before_steps.append(float(loss.detach().float().cpu()))
        loss.backward()
        optimizer.step()
        del out, loss
        after = _loss_value(model, batch)
        loss_after_steps.append(after)
        trajectory.append(after)

    lora_delta = lora_max_abs_diff(model, runtime.lora_init_cpu)
    lora_l2 = lora_l2_delta(model, runtime.lora_init_cpu)
    n_labeled = encoded.n_labeled
    n_tokens = len(encoded.input_ids)

    # Source-free invariant: drop write tensors. Caller must not reuse them.
    del batch
    if runtime.device.type == "cuda":
        torch.cuda.synchronize()

    model.eval()
    return {
        "k_steps": k_steps,
        "loss_trajectory": trajectory,
        "loss_before_steps": loss_before_steps,
        "loss_after_steps": loss_after_steps,
        "loss_before": trajectory[0],
        "loss_after": trajectory[-1],
        "loss_decreased": bool(trajectory[-1] < trajectory[0]),
        "lora_max_abs_diff_from_init": lora_delta,
        "lora_l2_delta_from_init": lora_l2,
        "write_n_tokens": n_tokens,
        "write_n_labeled_tokens": n_labeled,
        "write_used_cache": False,
        "returned_past_key_values": False,
    }
