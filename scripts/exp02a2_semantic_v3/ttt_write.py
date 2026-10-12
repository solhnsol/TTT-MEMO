"""NTP LoRA write for Exp 02A-2: two independent sequences as one batch.

Gradient flow
-------------
Loss: unweighted mean of the two sequence-level mean causal CEs.
      For each sequence i, CE_i = mean_j [-log p(y_{i,j} | y_{i,<j})] over
      labeled description(+period) tokens only (labels already -100 on
      entity, scaffold, Fact wrapper, chat specials, and padding).
      L = (CE_0 + CE_1) / 2.
W.r.t.: LoRA A/B of q_proj and o_proj (float32). Inner-loop / test-time update.
Not differentiated: every base-model parameter.
Aggregation: batch size 2, one optimizer step uses both facts of the mapping.
             K=32 is the optimizer budget, not 32 textual replays.

Token-weighted CE over the concatenated labeled tokens of the batch is stored
as a diagnostic; it is not the training objective.

Write tensors and past_key_values are dropped before return.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F

from model_runtime import LoadedRuntime, lora_l2_delta, lora_max_abs_diff, set_adapter_enabled


def collate_write_batch(
    sequences: list[dict[str, Any]],
    *,
    pad_token_id: int,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    if len(sequences) != 2:
        raise ValueError(f"expected 2 independent write sequences, got {len(sequences)}")
    lengths = [len(s["input_ids"]) for s in sequences]
    max_len = max(lengths)
    input_ids = []
    attention_mask = []
    labels = []
    for seq in sequences:
        ids = list(seq["input_ids"])
        lab = list(seq["ce_labels"])
        if len(ids) != len(lab):
            raise RuntimeError("input_ids and ce_labels length mismatch")
        pad = max_len - len(ids)
        input_ids.append(ids + [pad_token_id] * pad)
        attention_mask.append([1] * len(ids) + [0] * pad)
        labels.append(lab + [-100] * pad)
    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long, device=device),
        "attention_mask": torch.tensor(attention_mask, dtype=torch.long, device=device),
        "labels": torch.tensor(labels, dtype=torch.long, device=device),
    }


def sequence_mean_ce(logits: torch.Tensor, labels: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-sequence mean CE and token-weighted mean CE.

    logits: [B, T, V] unshifted model logits.
    labels: [B, T] with -100 ignored. HF convention: logits[t] predicts labels[t+1].
    """
    shift_logits = logits[:, :-1, :].float()
    shift_labels = labels[:, 1:]
    logp = F.log_softmax(shift_logits, dim=-1)
    bsz = shift_labels.size(0)
    seq_losses = []
    token_nlls = []
    for i in range(bsz):
        tgt = shift_labels[i]
        mask = tgt != -100
        n = int(mask.sum().item())
        if n == 0:
            raise RuntimeError(f"sequence {i} has no labeled target tokens")
        nll = -logp[i][mask, tgt[mask]]
        seq_losses.append(nll.mean())
        token_nlls.append(nll)
    seq_ce = torch.stack(seq_losses)
    mean_seq_ce = seq_ce.mean()
    token_weighted = torch.cat(token_nlls).mean()
    return mean_seq_ce, token_weighted


@torch.no_grad()
def _loss_values(model, batch: dict[str, torch.Tensor]) -> tuple[float, float]:
    out = model(
        input_ids=batch["input_ids"],
        attention_mask=batch["attention_mask"],
        use_cache=False,
        past_key_values=None,
    )
    mean_seq, token_w = sequence_mean_ce(out.logits, batch["labels"])
    del out
    return float(mean_seq.detach().cpu()), float(token_w.detach().cpu())


def write_mapping_batch(
    runtime: LoadedRuntime,
    sequences: list[dict[str, Any]],
    optimizer,
    *,
    k_steps: int,
) -> dict[str, Any]:
    """K AdamW steps on two independent write sequences. Drops write tensors."""
    model = runtime.model
    set_adapter_enabled(model, True)
    model.train()
    batch = collate_write_batch(
        sequences,
        pad_token_id=int(runtime.tokenizer.pad_token_id),
        device=runtime.device,
    )
    n_labeled = [int(sum(1 for x in s["ce_labels"] if x != -100)) for s in sequences]
    n_tokens = [len(s["input_ids"]) for s in sequences]

    loss_before_steps: list[float] = []
    loss_after_steps: list[float] = []
    token_weighted_traj: list[float] = []
    first_seq, first_tok = _loss_values(model, batch)
    trajectory = [first_seq]
    token_weighted_traj.append(first_tok)

    for _step in range(k_steps):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        out = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            use_cache=False,
            past_key_values=None,
        )
        loss, token_w = sequence_mean_ce(out.logits, batch["labels"])
        loss_before_steps.append(float(loss.detach().float().cpu()))
        loss.backward()
        optimizer.step()
        del out, loss, token_w
        after_seq, after_tok = _loss_values(model, batch)
        loss_after_steps.append(after_seq)
        trajectory.append(after_seq)
        token_weighted_traj.append(after_tok)

    lora_delta = lora_max_abs_diff(model, runtime.lora_init_cpu)
    lora_l2 = lora_l2_delta(model, runtime.lora_init_cpu)

    del batch
    if runtime.device.type == "cuda":
        torch.cuda.synchronize()

    model.eval()
    return {
        "k_steps": k_steps,
        "batch_size": 2,
        "loss_aggregation": "mean_of_per_sequence_mean_ce",
        "loss_trajectory": trajectory,
        "token_weighted_ce_trajectory": token_weighted_traj,
        "loss_before_steps": loss_before_steps,
        "loss_after_steps": loss_after_steps,
        "loss_before": trajectory[0],
        "loss_after": trajectory[-1],
        "loss_decreased": bool(trajectory[-1] < trajectory[0]),
        "lora_max_abs_diff_from_init": lora_delta,
        "lora_l2_delta_from_init": lora_l2,
        "write_n_tokens_per_sequence": n_tokens,
        "write_n_labeled_tokens_per_sequence": n_labeled,
        "write_used_cache": False,
        "returned_past_key_values": False,
    }
