"""Exp 02A evaluation loop: four conditions, candidate ranking, sanity checks."""

from __future__ import annotations

import json
import traceback
from collections import defaultdict
from typing import Any

import torch

from metrics import generation_scores, paired_bootstrap_ci, rank_candidates
from model_runtime import (
    LoadedRuntime,
    frozen_watch_max_abs_diff,
    lora_max_abs_diff,
    reset_example_state,
    set_adapter_enabled,
)
from prompts import (
    encode_candidate,
    encode_recall_prompt,
    encode_write,
    to_tensors,
)
from protocol import CONDITIONS, SOURCE_FREE_CONDITIONS, SPEC, TTT_CONDITIONS
from ttt_write import write_fact


def mean(xs: list[float]) -> float:
    return float(sum(xs) / len(xs)) if xs else float("nan")


@torch.no_grad()
def score_candidate_logprobs(runtime: LoadedRuntime, encoded: dict[str, Any]) -> dict[str, Any]:
    model = runtime.model
    model.eval()
    batch = to_tensors(encoded, runtime.device, include_labels=True)
    out = model(
        input_ids=batch["input_ids"],
        attention_mask=batch["attention_mask"],
        use_cache=False,
        past_key_values=None,
    )
    logits = out.logits[0].float()
    labels = batch["labels"][0]
    logp = torch.log_softmax(logits[:-1], dim=-1)
    tgt = labels[1:]
    mask = tgt != -100
    if int(mask.sum().item()) == 0:
        raise RuntimeError("no labeled candidate tokens")
    token_logp = logp[mask, tgt[mask]]
    logp_sum = float(token_logp.sum().cpu())
    logp_mean = float(token_logp.mean().cpu())
    n_tokens = int(mask.sum().item())
    if n_tokens != encoded["n_tokens"]:
        raise RuntimeError(f"labeled {n_tokens} != encoded {encoded['n_tokens']}")
    del out, batch
    return {
        "candidate": encoded["candidate"],
        "logp_sum": logp_sum,
        "logp_mean": logp_mean,
        "nll_mean": -logp_mean,
        "n_tokens": n_tokens,
        "token_ids": encoded["input_ids"],
        "candidate_token_indices": encoded["candidate_token_indices"],
    }


@torch.no_grad()
def greedy_generate(runtime: LoadedRuntime, prompt_ids: list[int]) -> dict[str, Any]:
    model = runtime.model
    tokenizer = runtime.tokenizer
    model.eval()
    input_ids = torch.tensor([prompt_ids], dtype=torch.long, device=runtime.device)
    attention_mask = torch.ones_like(input_ids)
    gen = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_new_tokens=int(SPEC["generation"]["max_new_tokens"]),
        do_sample=False,
        use_cache=True,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    new_ids = gen[0, input_ids.shape[1] :].detach().cpu().tolist()
    text = tokenizer.decode(new_ids, skip_special_tokens=True)
    del gen, input_ids, attention_mask
    return {"generated_token_ids": new_ids, "raw_prediction": text}


def write_source_for_condition(
    example: dict[str, Any],
    condition: str,
    by_id: dict[str, dict[str, Any]],
) -> dict[str, Any] | None:
    if condition == "ttt_correct_source_free":
        return example
    if condition == "ttt_shuffled_source_free":
        partner = by_id[example["shuffle_partner_id"]]
        if partner["info_type"] != example["info_type"]:
            raise RuntimeError("shuffle partner type mismatch at eval")
        if partner["entity"] == example["entity"] or partner["answer"] == example["answer"]:
            raise RuntimeError("shuffle overlap at eval")
        return partner
    return None


def evaluate_one(
    runtime: LoadedRuntime,
    example: dict[str, Any],
    *,
    condition: str,
    by_id: dict[str, dict[str, Any]],
    k_steps: int,
) -> dict[str, Any]:
    record: dict[str, Any] = {
        "example_id": example["example_id"],
        "info_type": example["info_type"],
        "entity": example["entity"],
        "answer": example["answer"],
        "write_payload": example["write_payload"],
        "fact_sentence": example["fact_sentence"],
        "question": example["question"],
        "condition": condition,
        "shuffle_partner_id": example["shuffle_partner_id"],
        "status": "ok",
        "error": None,
        "source_present": condition in ("static_source_present",),
        "adapter_enabled": condition in TTT_CONDITIONS,
        "recall_used_past_key_values": False,
    }
    opt = None
    try:
        opt = reset_example_state(runtime)
        write_example = write_source_for_condition(example, condition, by_id)
        if write_example is not None:
            encoded_write = encode_write(runtime.tokenizer, write_example)
            if example["question"] in encoded_write.text:
                raise RuntimeError("future question leaked into write")
            write_log = write_fact(runtime, encoded_write, opt, k_steps=k_steps)
            record["write"] = write_log
            record["write_example_id"] = write_example["example_id"]
            record["write_text"] = encoded_write.text
            record["write_n_tokens"] = encoded_write.n_labeled
            record["decoded_oracle_tokens"] = runtime.tokenizer.decode(
                [encoded_write.input_ids[i] for i in encoded_write.oracle_token_indices],
                skip_special_tokens=False,
            )
            if write_log["lora_max_abs_diff_from_init"] <= 0.0:
                raise RuntimeError("LoRA did not change after TTT write")
            base_diff = frozen_watch_max_abs_diff(runtime.model, runtime.frozen_watch_cpu)
            if base_diff != 0.0:
                raise RuntimeError(f"base weights changed during write: {base_diff}")
            del encoded_write
        else:
            record["write"] = None
            record["write_example_id"] = None
            set_adapter_enabled(runtime.model, False)

        source_present = condition == "static_source_present"
        if source_present:
            set_adapter_enabled(runtime.model, False)
        elif condition == "static_source_free":
            set_adapter_enabled(runtime.model, False)
        else:
            set_adapter_enabled(runtime.model, True)

        prompt = encode_recall_prompt(
            runtime.tokenizer,
            example,
            source_present=source_present,
        )
        if condition in SOURCE_FREE_CONDITIONS:
            if example["fact_sentence"] in prompt["text"]:
                raise RuntimeError("source fact present in source-free recall prompt")
            if write_example is not None and write_example["fact_sentence"] in prompt["text"]:
                raise RuntimeError("write fact present in source-free recall prompt")

        cand_rows = []
        for cand in example["candidates"]:
            encoded_c = encode_candidate(runtime.tokenizer, prompt["text"], cand["text"])
            scored = score_candidate_logprobs(runtime, encoded_c)
            scored["is_gold"] = cand["is_gold"]
            scored["source_example_id"] = cand["source_example_id"]
            scored["candidate_token_ids"] = [
                encoded_c["input_ids"][i] for i in encoded_c["candidate_token_indices"]
            ]
            scored.pop("token_ids", None)
            cand_rows.append(scored)
            del encoded_c

        ranked = rank_candidates(cand_rows, gold=example["answer"])
        gen = greedy_generate(runtime, prompt["input_ids"])
        gen_scores = generation_scores(gen["raw_prediction"], example["answer"])
        record.update(
            {
                "recall_prompt": prompt["text"],
                "recall_n_tokens": len(prompt["input_ids"]),
                "candidates": cand_rows,
                **ranked,
                **gen,
                **gen_scores,
            }
        )
        base_diff = frozen_watch_max_abs_diff(runtime.model, runtime.frozen_watch_cpu)
        record["base_weight_max_abs_diff"] = base_diff
        if base_diff != 0.0:
            raise RuntimeError(f"base weights changed at recall: {base_diff}")
    except Exception as exc:
        record["status"] = "error"
        record["error"] = f"{type(exc).__name__}: {exc}"
        record["traceback"] = traceback.format_exc()
    finally:
        # Always restore init LoRA so the next (example, condition) cannot leak state.
        reset_example_state(runtime)
        del opt
    return record


def run_conditions(
    runtime: LoadedRuntime,
    examples: list[dict[str, Any]],
    *,
    by_id: dict[str, dict[str, Any]],
    k_steps: int,
    log_fn=print,
    conditions: tuple[str, ...] = CONDITIONS,
) -> list[dict[str, Any]]:
    records = []
    n = len(examples) * len(conditions)
    k = 0
    for ex in examples:
        for condition in conditions:
            k += 1
            rec = evaluate_one(
                runtime,
                ex,
                condition=condition,
                by_id=by_id,
                k_steps=k_steps,
            )
            records.append(rec)
            extra = ""
            if rec["status"] == "ok":
                extra = (
                    f"acc={rec['accuracy']} rank={rec['gold_rank']} "
                    f"margin_mean={rec['margin_logp_mean']:.3f} em={rec['em']}"
                )
            log_fn(f"[{k}/{n}] {ex['example_id']} {condition} {rec['status']} {extra}")
    return records


def condition_metrics(
    records: list[dict[str, Any]],
    *,
    conditions: tuple[str, ...] = CONDITIONS,
) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for condition in conditions:
        rows = [r for r in records if r["condition"] == condition and r["status"] == "ok"]
        by_type: dict[str, list] = defaultdict(list)
        for r in rows:
            by_type[r["info_type"]].append(r)
        payload = {
            "n": len(rows),
            "primary_score": "mean_token_log_likelihood",
            "candidate_accuracy": mean([r["accuracy"] for r in rows]),
            "mean_gold_rank": mean([r["gold_rank"] for r in rows]),
            "mean_margin_logp_mean": mean([r["margin_logp_mean"] for r in rows]),
            "mean_gold_nll": mean([r["gold_nll_mean"] for r in rows]),
            "mean_gold_n_tokens": mean([r["gold_n_tokens"] for r in rows]),
            "candidate_accuracy_sum_logp": mean([r["accuracy_sum_logp"] for r in rows]),
            "mean_gold_rank_sum_logp": mean([r["gold_rank_sum_logp"] for r in rows]),
            "mean_margin_logp_sum": mean([r["margin_logp_sum"] for r in rows]),
            "generation_em": mean([r["em"] for r in rows]),
            "generation_f1": mean([r["f1"] for r in rows]),
            "n_error": sum(1 for r in records if r["condition"] == condition and r["status"] != "ok"),
        }
        for info_type, subset in by_type.items():
            payload[info_type] = {
                "n": len(subset),
                "candidate_accuracy": mean([r["accuracy"] for r in subset]),
                "mean_gold_rank": mean([r["gold_rank"] for r in subset]),
                "mean_margin_logp_mean": mean([r["margin_logp_mean"] for r in subset]),
                "mean_gold_nll": mean([r["gold_nll_mean"] for r in subset]),
                "mean_gold_n_tokens": mean([r["gold_n_tokens"] for r in subset]),
                "candidate_accuracy_sum_logp": mean([r["accuracy_sum_logp"] for r in subset]),
                "mean_margin_logp_sum": mean([r["margin_logp_sum"] for r in subset]),
                "generation_em": mean([r["em"] for r in subset]),
                "generation_f1": mean([r["f1"] for r in subset]),
            }
        out[condition] = payload
    return out


def paired_ttt_effect(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Item-specific memory = TTT-correct vs TTT-shuffled, not correct vs static."""
    by_id: dict[str, dict[str, dict]] = {}
    for r in records:
        if r.get("status") != "ok":
            continue
        by_id.setdefault(r["example_id"], {})[r["condition"]] = r

    per_type: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for eid, conds in sorted(by_id.items()):
        if "ttt_correct_source_free" not in conds or "ttt_shuffled_source_free" not in conds:
            continue
        c = conds["ttt_correct_source_free"]
        s = conds["ttt_shuffled_source_free"]
        row = {
            "example_id": eid,
            "info_type": c["info_type"],
            "answer": c["answer"],
            "gold_n_tokens": c["gold_n_tokens"],
            "correct_nll": c["gold_nll_mean"],
            "shuffled_nll": s["gold_nll_mean"],
            "nll_improvement": s["gold_nll_mean"] - c["gold_nll_mean"],
            "correct_margin": c["margin_logp_mean"],
            "shuffled_margin": s["margin_logp_mean"],
            "margin_improvement": c["margin_logp_mean"] - s["margin_logp_mean"],
            "correct_rank": c["gold_rank"],
            "shuffled_rank": s["gold_rank"],
            "rank_change": s["gold_rank"] - c["gold_rank"],
            "correct_top1": c["accuracy"],
            "shuffled_top1": s["accuracy"],
            "top1_gain": c["accuracy"] - s["accuracy"],
        }
        per_type[c["info_type"]].append(row)

    out: dict[str, Any] = {
        "definition": (
            "nll_improvement = shuffled_nll - correct_nll (positive => correct better). "
            "margin_improvement = correct_margin - shuffled_margin (positive => correct better). "
            "rank_change = shuffled_rank - correct_rank (positive => correct better). "
            "Static is a baseline only; correct > static is not item-specific memory."
        ),
        "bootstrap": {"n_boot": 10000, "seed": 42, "alpha": 0.05},
        "by_type": {},
    }
    for info_type, rows in per_type.items():
        correct_nll = [r["correct_nll"] for r in rows]
        shuffled_nll = [r["shuffled_nll"] for r in rows]
        correct_margin = [r["correct_margin"] for r in rows]
        shuffled_margin = [r["shuffled_margin"] for r in rows]
        correct_rank = [float(r["correct_rank"]) for r in rows]
        shuffled_rank = [float(r["shuffled_rank"]) for r in rows]
        correct_top1 = [float(r["correct_top1"]) for r in rows]
        shuffled_top1 = [float(r["shuffled_top1"]) for r in rows]
        out["by_type"][info_type] = {
            "n": len(rows),
            "mean_nll_improvement": mean([r["nll_improvement"] for r in rows]),
            "mean_margin_improvement": mean([r["margin_improvement"] for r in rows]),
            "mean_rank_change": mean([float(r["rank_change"]) for r in rows]),
            "mean_top1_gain": mean([float(r["top1_gain"]) for r in rows]),
            "correct_accuracy": mean(correct_top1),
            "shuffled_accuracy": mean(shuffled_top1),
            "bootstrap_nll_improvement": paired_bootstrap_ci(shuffled_nll, correct_nll),
            "bootstrap_margin_improvement": paired_bootstrap_ci(correct_margin, shuffled_margin),
            "bootstrap_rank_change": paired_bootstrap_ci(shuffled_rank, correct_rank),
            "bootstrap_top1_gain": paired_bootstrap_ci(correct_top1, shuffled_top1),
            "per_example": rows,
        }
    return out


def _index_ttt_records(records: list[dict[str, Any]]) -> dict[str, dict[str, dict[str, Any]]]:
    by_id: dict[str, dict[str, dict[str, Any]]] = {}
    for r in records:
        if r.get("status") != "ok":
            continue
        if r["condition"] not in TTT_CONDITIONS:
            continue
        by_id.setdefault(r["example_id"], {})[r["condition"]] = r
    return by_id


def write_stats_for_correct(records: list[dict[str, Any]], info_type: str) -> dict[str, float]:
    rows = [
        r
        for r in records
        if r.get("status") == "ok"
        and r["condition"] == "ttt_correct_source_free"
        and r["info_type"] == info_type
        and r.get("write")
    ]
    if not rows:
        return {
            "write_loss_pre": float("nan"),
            "write_loss_post": float("nan"),
            "lora_l2_delta": float("nan"),
            "lora_max_abs_diff": float("nan"),
            "n": 0,
        }
    trajs = [r["write"]["loss_trajectory"] for r in rows]
    mean_traj = [mean([t[i] for t in trajs]) for i in range(len(trajs[0]))]
    return {
        "write_loss_pre": mean([r["write"]["loss_before"] for r in rows]),
        "write_loss_post": mean([r["write"]["loss_after"] for r in rows]),
        "lora_l2_delta": mean([r["write"]["lora_l2_delta_from_init"] for r in rows]),
        "lora_max_abs_diff": mean([r["write"]["lora_max_abs_diff_from_init"] for r in rows]),
        "mean_loss_trajectory": mean_traj,
        "n": len(rows),
    }


def cross_k_memory_table(
    records_k8: list[dict[str, Any]],
    records_k32: list[dict[str, Any]],
    *,
    info_types: tuple[str, ...] = ("exact", "semantic_direct"),
) -> dict[str, Any]:
    """Primary cross-K comparison: memory_nll_gain = shuffled_NLL - correct_NLL."""
    idx8 = _index_ttt_records(records_k8)
    idx32 = _index_ttt_records(records_k32)
    shared = sorted(set(idx8) & set(idx32))
    missing32 = sorted(set(idx8) - set(idx32))
    extra32 = sorted(set(idx32) - set(idx8))
    if missing32 or extra32:
        raise RuntimeError(f"example_id mismatch k8_only={missing32[:8]} k32_only={extra32[:8]}")

    rows_out: list[dict[str, Any]] = []
    by_type: dict[str, Any] = {}
    for info_type in info_types:
        eids = [eid for eid in shared if idx8[eid]["ttt_correct_source_free"]["info_type"] == info_type]
        gain8 = []
        gain32 = []
        per_example = []
        for eid in eids:
            c8 = idx8[eid]["ttt_correct_source_free"]
            s8 = idx8[eid]["ttt_shuffled_source_free"]
            c32 = idx32[eid]["ttt_correct_source_free"]
            s32 = idx32[eid]["ttt_shuffled_source_free"]
            g8 = s8["gold_nll_mean"] - c8["gold_nll_mean"]
            g32 = s32["gold_nll_mean"] - c32["gold_nll_mean"]
            gain8.append(g8)
            gain32.append(g32)
            per_example.append(
                {
                    "example_id": eid,
                    "info_type": info_type,
                    "memory_nll_gain_k8": g8,
                    "memory_nll_gain_k32": g32,
                    "memory_nll_gain_delta": g32 - g8,
                    "correct_nll_k8": c8["gold_nll_mean"],
                    "shuffled_nll_k8": s8["gold_nll_mean"],
                    "correct_nll_k32": c32["gold_nll_mean"],
                    "shuffled_nll_k32": s32["gold_nll_mean"],
                    "mean_logp_shift_correct": c32["gold_logp_mean"] - c8["gold_logp_mean"],
                    "margin_shift_correct": c32["margin_logp_mean"] - c8["margin_logp_mean"],
                    "rank_shift_correct": c8["gold_rank"] - c32["gold_rank"],
                    "rank_gain_k8": s8["gold_rank"] - c8["gold_rank"],
                    "rank_gain_k32": s32["gold_rank"] - c32["gold_rank"],
                    "top1_k8_correct": c8["accuracy"],
                    "top1_k32_correct": c32["accuracy"],
                    "top1_gain_k8": c8["accuracy"] - s8["accuracy"],
                    "top1_gain_k32": c32["accuracy"] - s32["accuracy"],
                    "em_k8_correct": c8["em"],
                    "em_k32_correct": c32["em"],
                    "f1_k8_correct": c8["f1"],
                    "f1_k32_correct": c32["f1"],
                }
            )

        w8 = write_stats_for_correct(records_k8, info_type)
        w32 = write_stats_for_correct(records_k32, info_type)
        zeros = [0.0] * len(gain32)
        by_type[info_type] = {
            "n": len(eids),
            "k8": {
                "write_loss_pre": w8["write_loss_pre"],
                "write_loss_post": w8["write_loss_post"],
                "lora_l2_delta": w8["lora_l2_delta"],
                "lora_max_abs_diff": w8["lora_max_abs_diff"],
                "mean_loss_trajectory": w8.get("mean_loss_trajectory"),
                "correct_gold_NLL": mean([idx8[e]["ttt_correct_source_free"]["gold_nll_mean"] for e in eids]),
                "shuffled_gold_NLL": mean([idx8[e]["ttt_shuffled_source_free"]["gold_nll_mean"] for e in eids]),
                "memory_nll_gain": mean(gain8),
                "candidate_rank_gain": mean(
                    [
                        float(
                            idx8[e]["ttt_shuffled_source_free"]["gold_rank"]
                            - idx8[e]["ttt_correct_source_free"]["gold_rank"]
                        )
                        for e in eids
                    ]
                ),
                "top1_gain": mean(
                    [
                        float(
                            idx8[e]["ttt_correct_source_free"]["accuracy"]
                            - idx8[e]["ttt_shuffled_source_free"]["accuracy"]
                        )
                        for e in eids
                    ]
                ),
                "generation_EM": mean([idx8[e]["ttt_correct_source_free"]["em"] for e in eids]),
                "generation_F1": mean([idx8[e]["ttt_correct_source_free"]["f1"] for e in eids]),
                "correct_accuracy": mean([idx8[e]["ttt_correct_source_free"]["accuracy"] for e in eids]),
                "shuffled_accuracy": mean([idx8[e]["ttt_shuffled_source_free"]["accuracy"] for e in eids]),
                "correct_mean_logp": mean([idx8[e]["ttt_correct_source_free"]["gold_logp_mean"] for e in eids]),
                "correct_margin": mean([idx8[e]["ttt_correct_source_free"]["margin_logp_mean"] for e in eids]),
            },
            "k32": {
                "write_loss_pre": w32["write_loss_pre"],
                "write_loss_post": w32["write_loss_post"],
                "lora_l2_delta": w32["lora_l2_delta"],
                "lora_max_abs_diff": w32["lora_max_abs_diff"],
                "mean_loss_trajectory": w32.get("mean_loss_trajectory"),
                "correct_gold_NLL": mean([idx32[e]["ttt_correct_source_free"]["gold_nll_mean"] for e in eids]),
                "shuffled_gold_NLL": mean([idx32[e]["ttt_shuffled_source_free"]["gold_nll_mean"] for e in eids]),
                "memory_nll_gain": mean(gain32),
                "candidate_rank_gain": mean(
                    [
                        float(
                            idx32[e]["ttt_shuffled_source_free"]["gold_rank"]
                            - idx32[e]["ttt_correct_source_free"]["gold_rank"]
                        )
                        for e in eids
                    ]
                ),
                "top1_gain": mean(
                    [
                        float(
                            idx32[e]["ttt_correct_source_free"]["accuracy"]
                            - idx32[e]["ttt_shuffled_source_free"]["accuracy"]
                        )
                        for e in eids
                    ]
                ),
                "generation_EM": mean([idx32[e]["ttt_correct_source_free"]["em"] for e in eids]),
                "generation_F1": mean([idx32[e]["ttt_correct_source_free"]["f1"] for e in eids]),
                "correct_accuracy": mean([idx32[e]["ttt_correct_source_free"]["accuracy"] for e in eids]),
                "shuffled_accuracy": mean([idx32[e]["ttt_shuffled_source_free"]["accuracy"] for e in eids]),
                "correct_mean_logp": mean([idx32[e]["ttt_correct_source_free"]["gold_logp_mean"] for e in eids]),
                "correct_margin": mean([idx32[e]["ttt_correct_source_free"]["margin_logp_mean"] for e in eids]),
            },
            "bootstrap_memory_nll_gain_k32": paired_bootstrap_ci(gain32, zeros),
            "bootstrap_memory_nll_gain_k32_minus_k8": paired_bootstrap_ci(gain32, gain8),
            "mean_logp_shift_correct": mean(
                [idx32[e]["ttt_correct_source_free"]["gold_logp_mean"] - idx8[e]["ttt_correct_source_free"]["gold_logp_mean"] for e in eids]
            ),
            "margin_shift_correct": mean(
                [idx32[e]["ttt_correct_source_free"]["margin_logp_mean"] - idx8[e]["ttt_correct_source_free"]["margin_logp_mean"] for e in eids]
            ),
            "rank_shift_correct": mean(
                [
                    float(
                        idx8[e]["ttt_correct_source_free"]["gold_rank"]
                        - idx32[e]["ttt_correct_source_free"]["gold_rank"]
                    )
                    for e in eids
                ]
            ),
            "per_example": per_example,
        }
        for k in (8, 32):
            block = by_type[info_type]["k8" if k == 8 else "k32"]
            rows_out.append(
                {
                    "type": info_type,
                    "K": k,
                    "write_loss_pre": block["write_loss_pre"],
                    "write_loss_post": block["write_loss_post"],
                    "correct_gold_NLL": block["correct_gold_NLL"],
                    "shuffled_gold_NLL": block["shuffled_gold_NLL"],
                    "memory_nll_gain": block["memory_nll_gain"],
                    "candidate_rank_gain": block["candidate_rank_gain"],
                    "top1_gain": block["top1_gain"],
                    "generation_EM": block["generation_EM"],
                }
            )
    return {
        "primary_metric": "memory_nll_gain = NLL(ttt_shuffled) - NLL(ttt_correct) using per-token gold NLL",
        "semantic_direct_top1_note": (
            "semantic_direct mean-logp top-1 failed source-present calibration at K=8 "
            "(31.2%). Do not use semantic top-1 as the primary memory conclusion."
        ),
        "table": rows_out,
        "by_type": by_type,
    }


def smoke_invariants(
    runtime: LoadedRuntime,
    examples: list[dict[str, Any]],
    records: list[dict[str, Any]],
    *,
    by_id: dict[str, dict[str, Any]],
    log_fn=print,
    require_source_present: bool = True,
) -> list[str]:
    failures: list[str] = []

    def fail(msg: str) -> None:
        failures.append(msg)
        log_fn(f"SANITY FAIL: {msg}")

    def ok(msg: str) -> None:
        log_fn(f"OK {msg}")

    log_fn("\n=== smoke sanity checks ===")

    errors = [r for r in records if r["status"] != "ok"]
    if errors:
        fail(f"{len(errors)} evaluation errors, first={errors[0]['error']}")
    else:
        ok(f"all {len(records)} records status=ok")

    # Source-present candidate accuracy: in-context upper bound. Below 0.5 on
    # 16 items would mean the scoring protocol is broken, not a memory result.
    sp = [r for r in records if r["condition"] == "static_source_present" and r["status"] == "ok"]
    if require_source_present:
        sp_acc = mean([r["accuracy"] for r in sp])
        if sp_acc < 0.5:
            fail(f"static_source_present candidate accuracy {sp_acc:.3f} < 0.5 (eval likely broken)")
        else:
            ok(f"static_source_present candidate accuracy={sp_acc:.3f} n={len(sp)}")
    elif sp:
        ok(f"source-present records present n={len(sp)} (not required this run)")
    else:
        ok("source-present skipped this run; using K-independent k8 reference")

    # Base weights unchanged at end.
    base_diff = frozen_watch_max_abs_diff(runtime.model, runtime.frozen_watch_cpu)
    if base_diff != 0.0:
        fail(f"base weights changed at end: {base_diff}")
    else:
        ok(f"base weights unchanged hash={runtime.frozen_watch_hash[:16]}...")

    # LoRA changes after TTT; write loss decreases; adapter reset afterwards.
    ttt_rows = [r for r in records if r["condition"] in TTT_CONDITIONS and r["status"] == "ok"]
    for r in ttt_rows:
        w = r.get("write") or {}
        if not w.get("loss_decreased"):
            fail(
                f"{r['example_id']} {r['condition']}: write loss did not decrease "
                f"{w.get('loss_before')} -> {w.get('loss_after')}"
            )
        if not (w.get("lora_max_abs_diff_from_init") or 0) > 0:
            fail(f"{r['example_id']} {r['condition']}: LoRA delta is 0")
        if w.get("returned_past_key_values"):
            fail(f"{r['example_id']} {r['condition']}: write returned a cache")
        if r["question"] in (r.get("write_text") or ""):
            fail(f"{r['example_id']} {r['condition']}: question in write text")
    if ttt_rows:
        ok(f"TTT writes n={len(ttt_rows)} all loss-decreased and LoRA-changed (if no fails above)")

    # Source/KV absent at recall for source-free conditions.
    for r in records:
        if r["status"] != "ok":
            continue
        if r["condition"] not in SOURCE_FREE_CONDITIONS:
            continue
        if r.get("recall_used_past_key_values"):
            fail(f"{r['example_id']} {r['condition']}: recall used past_key_values")
        prompt = r.get("recall_prompt") or ""
        if r["fact_sentence"] in prompt:
            fail(f"{r['example_id']} {r['condition']}: fact_sentence in recall prompt")
        if r["info_type"] in ("exact", "semantic_direct") and r["answer"] in prompt:
            fail(f"{r['example_id']} {r['condition']}: gold answer in source-free recall prompt")
    ok("source-free recall prompts do not contain the write fact (if no fails above)")

    # Shuffled fact has no entity/value overlap.
    for ex in examples:
        partner = by_id[ex["shuffle_partner_id"]]
        if partner["entity"] == ex["entity"] or partner["answer"] == ex["answer"]:
            fail(f"{ex['example_id']}: shuffle overlap")
        if partner["info_type"] != ex["info_type"]:
            fail(f"{ex['example_id']}: shuffle type mismatch")
    ok("shuffle partners have no entity/value overlap")

    # Per-example adapter reset: after the loop, LoRA matches init.
    reset_diff = lora_max_abs_diff(runtime.model, runtime.lora_init_cpu)
    if reset_diff != 0.0:
        fail(f"LoRA not restored to init at end: {reset_diff}")
    else:
        ok("LoRA restored to init after final example")

    # Deterministic greedy generation on one source-present prompt.
    probe = examples[0]
    set_adapter_enabled(runtime.model, False)
    prompt = encode_recall_prompt(runtime.tokenizer, probe, source_present=True)
    a = greedy_generate(runtime, prompt["input_ids"])
    b = greedy_generate(runtime, prompt["input_ids"])
    if a["generated_token_ids"] != b["generated_token_ids"]:
        fail("greedy generation is not deterministic")
    else:
        ok(f"deterministic generation ids={a['generated_token_ids'][:8]}...")
    set_adapter_enabled(runtime.model, True)
    reset_example_state(runtime)

    # No thinking blocks.
    for r in records:
        if r.get("recall_prompt") and "<think>" in r["recall_prompt"]:
            fail(f"{r['example_id']} thinking block in recall")
        if r.get("raw_prediction") and "<think>" in (r["raw_prediction"] or ""):
            fail(f"{r['example_id']} thinking block in generation")
    ok("no thinking blocks in prompts/generations (if no fails above)")

    if failures:
        log_fn(f"\n{len(failures)} sanity failure(s).")
    else:
        log_fn("\nAll smoke sanity checks passed.")
    return failures


def write_jsonl(path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for r in records:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
