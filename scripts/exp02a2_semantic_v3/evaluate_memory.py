"""Exp 02A-2 source-free evaluation: description binding + semantic label probes."""

from __future__ import annotations

import json
import traceback
from typing import Any

import torch

from encode_write import encode_candidate, encode_recall_prompt
from memory_protocol import CONDITIONS, PROBES, TTT_CONDITIONS, SPEC
from metrics import (
    b_swap,
    binding_score_from_matrix,
    flip_diagnostics,
    generation_scores,
    rank_candidates,
)
from model_runtime import (
    LoadedRuntime,
    frozen_watch_max_abs_diff,
    lora_max_abs_diff,
    lora_state_cpu,
    reset_example_state,
    set_adapter_enabled,
)
from ttt_write import write_mapping_batch


def mean(xs: list[float]) -> float:
    return float(sum(xs) / len(xs)) if xs else float("nan")


def mapping_name(condition: str) -> str | None:
    if "mapping_A" in condition:
        return "mapping_A"
    if "mapping_B" in condition:
        return "mapping_B"
    return None


def gold_value(pair: dict[str, Any], *, mapping: str, entity_key: str, probe: str) -> str:
    concept_key = pair[mapping][entity_key]
    concept = pair["concepts"][concept_key]
    if probe == "description_binding":
        return concept["description"]
    if probe == "semantic_label":
        return concept["label"]
    raise ValueError(probe)


def probe_question(pair: dict[str, Any], entity_key: str, probe: str) -> str:
    entity = pair["entities"][entity_key]
    if probe == "description_binding":
        return f"{entity} spends most workdays"
    if probe == "semantic_label":
        return pair["query_templates"]["occupation_query"].format(entity=entity)
    raise ValueError(probe)


def probe_candidates(pair: dict[str, Any], probe: str) -> list[str]:
    if probe == "description_binding":
        return [pair["concepts"]["c1"]["description"], pair["concepts"]["c2"]["description"]]
    if probe == "semantic_label":
        return list(pair["candidates"])
    raise ValueError(probe)


def write_sequences(pair: dict[str, Any], mapping: str) -> list[dict[str, Any]]:
    seqs = pair["sequences"][mapping]
    if len(seqs) != 2:
        raise RuntimeError(f"{pair['pair_id']} {mapping}: expected 2 sequences")
    out = []
    for seq in seqs:
        chat = seq["tokenization"]["chat_rendered_write"]
        if seq["label"] in seq["write_sentence"] or seq["label"] in chat["text"]:
            raise RuntimeError(f"{seq['sequence_id']}: occupation label leaked into write")
        out.append(
            {
                "sequence_id": seq["sequence_id"],
                "input_ids": chat["input_ids"],
                "ce_labels": chat["ce_labels"],
                "text": chat["text"],
                "write_sentence": seq["write_sentence"],
                "label": seq["label"],
                "description": seq["description"],
            }
        )
    return out


@torch.no_grad()
def score_candidate_logprobs(runtime: LoadedRuntime, encoded: dict[str, Any]) -> dict[str, Any]:
    model = runtime.model
    model.eval()
    input_ids = torch.tensor([encoded["input_ids"]], dtype=torch.long, device=runtime.device)
    attention_mask = torch.ones_like(input_ids)
    labels = torch.tensor([encoded["labels"]], dtype=torch.long, device=runtime.device)
    out = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        use_cache=False,
        past_key_values=None,
    )
    logits = out.logits[0].float()
    logp = torch.log_softmax(logits[:-1], dim=-1)
    tgt = labels[0, 1:]
    mask = tgt != -100
    if int(mask.sum().item()) == 0:
        raise RuntimeError("no labeled candidate tokens")
    token_logp = logp[mask, tgt[mask]]
    n_tokens = int(mask.sum().item())
    if n_tokens != encoded["n_tokens"]:
        raise RuntimeError(f"labeled {n_tokens} != encoded {encoded['n_tokens']}")
    result = {
        "candidate": encoded["candidate"],
        "logp_sum": float(token_logp.sum().cpu()),
        "logp_mean": float(token_logp.mean().cpu()),
        "nll_mean": float(-token_logp.mean().cpu()),
        "n_tokens": n_tokens,
    }
    del out, input_ids, attention_mask, labels
    return result


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


def _assert_source_free_prompt(pair: dict[str, Any], prompt_text: str) -> None:
    for mapping in ("mapping_A", "mapping_B"):
        for seq in pair["sequences"][mapping]:
            if seq["write_sentence"] in prompt_text:
                raise RuntimeError("write sentence leaked into source-free recall")
            if seq["description"] in prompt_text:
                raise RuntimeError("description leaked into source-free recall")


def mapping_fact_block(pair: dict[str, Any], mapping: str) -> str:
    return "\n".join(s["write_sentence"] for s in pair["sequences"][mapping])


def evaluate_probe(
    runtime: LoadedRuntime,
    pair: dict[str, Any],
    *,
    probe: str,
    mapping: str | None,
    source_present: bool = False,
) -> dict[str, Any]:
    cands = probe_candidates(pair, probe)
    v1, v2 = cands[0], cands[1]
    rank_mapping = mapping or "mapping_A"
    matrix_mean = {"e1": {}, "e2": {}}
    matrix_sum = {"e1": {}, "e2": {}}
    entities: dict[str, Any] = {}
    prediction_rows = []
    fact = None
    if source_present:
        if mapping is None:
            raise RuntimeError("source-present probe requires a mapping")
        fact = mapping_fact_block(pair, mapping)
    for entity_key in ("e1", "e2"):
        question = probe_question(pair, entity_key, probe)
        prompt = encode_recall_prompt(runtime.tokenizer, question=question, fact_block=fact)
        if source_present:
            if pair["sequences"][mapping][0]["write_sentence"] not in prompt["text"]:
                raise RuntimeError("source-present prompt missing write sentence")
        else:
            _assert_source_free_prompt(pair, prompt["text"])
        if probe == "semantic_label":
            for lab in pair["candidates"]:
                if lab.lower() in prompt["text"].lower() and lab.lower() not in question.lower():
                    raise RuntimeError("occupation label leaked into semantic recall prompt")
        cand_rows = []
        for cand in cands:
            encoded = encode_candidate(runtime.tokenizer, prompt["text"], cand)
            scored = score_candidate_logprobs(runtime, encoded)
            scored["is_v1"] = cand == v1
            scored["is_v2"] = cand == v2
            cand_rows.append(scored)
            del encoded
        by_text = {r["candidate"]: r for r in cand_rows}
        matrix_mean[entity_key]["v1"] = by_text[v1]["logp_mean"]
        matrix_mean[entity_key]["v2"] = by_text[v2]["logp_mean"]
        matrix_sum[entity_key]["v1"] = by_text[v1]["logp_sum"]
        matrix_sum[entity_key]["v2"] = by_text[v2]["logp_sum"]
        gold = gold_value(pair, mapping=rank_mapping, entity_key=entity_key, probe=probe)
        ranked = rank_candidates(cand_rows, gold=gold)
        gen = greedy_generate(runtime, prompt["input_ids"])
        gen_scores = generation_scores(gen["raw_prediction"], gold)
        entities[entity_key] = {
            "entity_key": entity_key,
            "entity": pair["entities"][entity_key],
            "question": question,
            "gold": gold,
            "candidates": cand_rows,
            "recall_prompt": prompt["text"],
            "recall_n_tokens": len(prompt["input_ids"]),
            **ranked,
            **gen,
            **gen_scores,
        }
        prediction_rows.append(
            {
                "pair_id": pair["pair_id"],
                "probe": probe,
                "mapping": mapping,
                "entity_key": entity_key,
                "entity": pair["entities"][entity_key],
                "gold": gold,
                "v1": v1,
                "v2": v2,
                "s_v1_mean": by_text[v1]["logp_mean"],
                "s_v2_mean": by_text[v2]["logp_mean"],
                "s_v1_sum": by_text[v1]["logp_sum"],
                "s_v2_sum": by_text[v2]["logp_sum"],
                "gold_rank": ranked["gold_rank"],
                "accuracy": ranked["accuracy"],
                "margin_logp_mean": ranked["margin_logp_mean"],
                "raw_prediction": gen["raw_prediction"],
                "em": gen_scores["em"],
                "f1": gen_scores["f1"],
            }
        )
    return {
        "probe": probe,
        "v1": v1,
        "v2": v2,
        "entities": entities,
        "matrix_mean_logp": matrix_mean,
        "matrix_sum_logp": matrix_sum,
        "prediction_rows": prediction_rows,
    }


def probes_for_condition(condition: str) -> tuple[str, ...]:
    if "source_present" in condition:
        return ("semantic_label",)
    return PROBES


def evaluate_condition(
    runtime: LoadedRuntime,
    pair: dict[str, Any],
    *,
    condition: str,
    k_steps: int,
) -> dict[str, Any]:
    mapping = mapping_name(condition)
    do_write = condition in TTT_CONDITIONS
    source_present = "source_present" in condition and "source_free" not in condition
    record: dict[str, Any] = {
        "pair_id": pair["pair_id"],
        "source_pair_id": pair.get("source_pair_id", pair["pair_id"]),
        "replicate_index": pair.get("replicate_index"),
        "info_type": pair["info_type"],
        "condition": condition,
        "mapping": mapping,
        "adapter_enabled": do_write,
        "source_present": source_present,
        "status": "ok",
        "error": None,
        "recall_used_past_key_values": False,
        "recall_updated_lora": False,
        "e1": pair["entities"]["e1"],
        "e2": pair["entities"]["e2"],
        "c1": pair["concepts"]["c1"]["label"],
        "c2": pair["concepts"]["c2"]["label"],
    }
    opt = None
    post_write_lora = None
    try:
        opt = reset_example_state(runtime)
        if do_write:
            if mapping is None:
                raise RuntimeError("TTT condition requires mapping")
            seqs = write_sequences(pair, mapping)
            write_log = write_mapping_batch(runtime, seqs, opt, k_steps=k_steps)
            record["write"] = write_log
            record["write_texts"] = [s["text"] for s in seqs]
            record["write_sentences"] = [s["write_sentence"] for s in seqs]
            if write_log["lora_max_abs_diff_from_init"] <= 0.0:
                raise RuntimeError("LoRA did not change after TTT write")
            base_diff = frozen_watch_max_abs_diff(runtime.model, runtime.frozen_watch_cpu)
            if base_diff != 0.0:
                raise RuntimeError(f"base weights changed during write: {base_diff}")
            post_write_lora = lora_state_cpu(runtime.model)
            set_adapter_enabled(runtime.model, True)
        else:
            record["write"] = None
            set_adapter_enabled(runtime.model, False)

        probes = {}
        prediction_rows = []
        for probe in probes_for_condition(condition):
            block = evaluate_probe(
                runtime,
                pair,
                probe=probe,
                mapping=mapping,
                source_present=source_present,
            )
            probes[probe] = block
            for row in block["prediction_rows"]:
                row["condition"] = condition
                prediction_rows.append(row)

        if do_write:
            recall_lora_diff = lora_max_abs_diff(runtime.model, post_write_lora)
            record["recall_lora_max_abs_diff_from_post_write"] = recall_lora_diff
            if recall_lora_diff != 0.0:
                record["recall_updated_lora"] = True
                raise RuntimeError(f"recall updated LoRA: max_abs_diff={recall_lora_diff}")
        else:
            record["recall_lora_max_abs_diff_from_post_write"] = 0.0

        base_diff = frozen_watch_max_abs_diff(runtime.model, runtime.frozen_watch_cpu)
        record["base_weight_max_abs_diff"] = base_diff
        if base_diff != 0.0:
            raise RuntimeError(f"base weights changed at recall: {base_diff}")
        record["probes"] = probes
        record["prediction_rows"] = prediction_rows
    except Exception as exc:
        record["status"] = "error"
        record["error"] = f"{type(exc).__name__}: {exc}"
        record["traceback"] = traceback.format_exc()
    finally:
        reset_example_state(runtime)
        del opt
        del post_write_lora
    return record


def run_pairs(
    runtime: LoadedRuntime,
    pairs: list[dict[str, Any]],
    *,
    k_steps: int,
    log_fn=print,
    conditions: tuple[str, ...] | None = None,
) -> list[dict[str, Any]]:
    records = []
    conds = conditions if conditions is not None else CONDITIONS
    n = len(pairs) * len(conds)
    k = 0
    for pair in pairs:
        for condition in conds:
            k += 1
            rec = evaluate_condition(runtime, pair, condition=condition, k_steps=k_steps)
            records.append(rec)
            extra = rec["status"]
            if rec["status"] == "ok" and rec.get("write"):
                extra += (
                    f" loss={rec['write']['loss_before']:.3f}->{rec['write']['loss_after']:.3f} "
                    f"lora_l2={rec['write']['lora_l2_delta_from_init']:.3f}"
                )
            log_fn(f"[{k}/{n}] {pair['pair_id']} {condition} {extra}")
    return records


def flatten_predictions(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for rec in records:
        if rec.get("status") != "ok":
            rows.append(
                {
                    "pair_id": rec["pair_id"],
                    "condition": rec["condition"],
                    "status": rec["status"],
                    "error": rec.get("error"),
                }
            )
            continue
        for row in rec.get("prediction_rows") or []:
            out = dict(row)
            out["status"] = "ok"
            if rec.get("write"):
                out["write_loss_before"] = rec["write"]["loss_before"]
                out["write_loss_after"] = rec["write"]["loss_after"]
                out["lora_l2_delta"] = rec["write"]["lora_l2_delta_from_init"]
                out["lora_max_abs_diff"] = rec["write"]["lora_max_abs_diff_from_init"]
            rows.append(out)
    return rows


def flatten_matrices(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for rec in records:
        if rec.get("status") != "ok":
            continue
        for probe, block in rec["probes"].items():
            rows.append(
                {
                    "pair_id": rec["pair_id"],
                    "condition": rec["condition"],
                    "mapping": rec["mapping"],
                    "probe": probe,
                    "matrix_mean_logp": block["matrix_mean_logp"],
                    "matrix_sum_logp": block["matrix_sum_logp"],
                    "v1": block["v1"],
                    "v2": block["v2"],
                    "e1": rec["e1"],
                    "e2": rec["e2"],
                }
            )
    return rows


def pair_binding_table(records: list[dict[str, Any]], pairs: list[dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {"by_probe": {}, "per_pair": []}
    grouped: dict[str, list[dict[str, Any]]] = {p: [] for p in PROBES}
    for pair in pairs:
        pid = pair["pair_id"]
        rec_a = next(r for r in records if r["pair_id"] == pid and r["condition"] == "ttt_mapping_A_source_free")
        rec_b = next(r for r in records if r["pair_id"] == pid and r["condition"] == "ttt_mapping_B_source_free")
        rec_st = next(r for r in records if r["pair_id"] == pid and r["condition"] == "static_source_free")
        rec_sp_a = next(
            (r for r in records if r["pair_id"] == pid and r["condition"] == "static_mapping_A_source_present"),
            None,
        )
        rec_sp_b = next(
            (r for r in records if r["pair_id"] == pid and r["condition"] == "static_mapping_B_source_present"),
            None,
        )
        pair_row: dict[str, Any] = {
            "pair_id": pid,
            "source_pair_id": pair.get("source_pair_id", pid),
            "replicate_index": pair.get("replicate_index"),
            "e1": pair["entities"]["e1"],
            "e2": pair["entities"]["e2"],
            "c1": pair["concepts"]["c1"]["label"],
            "c2": pair["concepts"]["c2"]["label"],
            "probes": {},
        }
        for probe in PROBES:
            if rec_a.get("status") != "ok" or rec_b.get("status") != "ok":
                continue
            mat_a = rec_a["probes"][probe]["matrix_mean_logp"]
            mat_b = rec_b["probes"][probe]["matrix_mean_logp"]
            mat_st = rec_st["probes"][probe]["matrix_mean_logp"] if rec_st.get("status") == "ok" else None
            bind = b_swap(mat_a, mat_b)
            flips = flip_diagnostics(mat_a, mat_b)
            static_b_a = binding_score_from_matrix(mat_st, "A") if mat_st is not None else None
            two_way_a = mean(
                [
                    float(rec_a["probes"][probe]["entities"]["e1"]["accuracy"]),
                    float(rec_a["probes"][probe]["entities"]["e2"]["accuracy"]),
                ]
            )
            two_way_b = mean(
                [
                    float(rec_b["probes"][probe]["entities"]["e1"]["accuracy"]),
                    float(rec_b["probes"][probe]["entities"]["e2"]["accuracy"]),
                ]
            )
            two_way_st = None
            if rec_st.get("status") == "ok":
                two_way_st = mean(
                    [
                        float(rec_st["probes"][probe]["entities"]["e1"]["accuracy"]),
                        float(rec_st["probes"][probe]["entities"]["e2"]["accuracy"]),
                    ]
                )
            em_ttt = mean(
                [
                    float(rec_a["probes"][probe]["entities"]["e1"]["em"]),
                    float(rec_a["probes"][probe]["entities"]["e2"]["em"]),
                    float(rec_b["probes"][probe]["entities"]["e1"]["em"]),
                    float(rec_b["probes"][probe]["entities"]["e2"]["em"]),
                ]
            )
            sp_bind = None
            sp_flips = None
            two_way_sp = None
            if (
                rec_sp_a is not None
                and rec_sp_b is not None
                and rec_sp_a.get("status") == "ok"
                and rec_sp_b.get("status") == "ok"
                and probe in rec_sp_a.get("probes", {})
                and probe in rec_sp_b.get("probes", {})
            ):
                mat_sp_a = rec_sp_a["probes"][probe]["matrix_mean_logp"]
                mat_sp_b = rec_sp_b["probes"][probe]["matrix_mean_logp"]
                sp_bind = b_swap(mat_sp_a, mat_sp_b)
                sp_flips = flip_diagnostics(mat_sp_a, mat_sp_b)
                two_way_sp = mean(
                    [
                        float(rec_sp_a["probes"][probe]["entities"]["e1"]["accuracy"]),
                        float(rec_sp_a["probes"][probe]["entities"]["e2"]["accuracy"]),
                        float(rec_sp_b["probes"][probe]["entities"]["e1"]["accuracy"]),
                        float(rec_sp_b["probes"][probe]["entities"]["e2"]["accuracy"]),
                    ]
                )
            style_row = {
                "B_A": bind["B_A"],
                "B_B": bind["B_B"],
                "B_swap": bind["B_swap"],
                "B_swap_gt_0": bind["B_swap"] > 0,
                "B_A_and_B_B_gt_0": bind["B_A"] > 0 and bind["B_B"] > 0,
                "e1_flipped": flips["e1_flipped"],
                "e2_flipped": flips["e2_flipped"],
                "both_flipped": flips["both_flipped"],
                "two_way_acc_A": two_way_a,
                "two_way_acc_B": two_way_b,
                "two_way_acc_ttt": mean([two_way_a, two_way_b]),
                "two_way_acc_static": two_way_st,
                "static_B_A": static_b_a,
                "generation_em_ttt": em_ttt,
                "source_present_B_A": None if sp_bind is None else sp_bind["B_A"],
                "source_present_B_B": None if sp_bind is None else sp_bind["B_B"],
                "source_present_B_swap": None if sp_bind is None else sp_bind["B_swap"],
                "source_present_two_way_accuracy": two_way_sp,
                "source_present_both_flipped": None if sp_flips is None else sp_flips["both_flipped"],
                "write_loss_pre_A": rec_a.get("write", {}).get("loss_before") if rec_a.get("write") else None,
                "write_loss_post_A": rec_a.get("write", {}).get("loss_after") if rec_a.get("write") else None,
                "write_loss_pre_B": rec_b.get("write", {}).get("loss_before") if rec_b.get("write") else None,
                "write_loss_post_B": rec_b.get("write", {}).get("loss_after") if rec_b.get("write") else None,
                "lora_l2_A": rec_a.get("write", {}).get("lora_l2_delta_from_init") if rec_a.get("write") else None,
                "lora_l2_B": rec_b.get("write", {}).get("lora_l2_delta_from_init") if rec_b.get("write") else None,
                "lora_max_abs_A": rec_a.get("write", {}).get("lora_max_abs_diff_from_init") if rec_a.get("write") else None,
                "lora_max_abs_B": rec_b.get("write", {}).get("lora_max_abs_diff_from_init") if rec_b.get("write") else None,
                "matrix_A": mat_a,
                "matrix_B": mat_b,
                "matrix_static": mat_st,
            }
            pair_row["probes"][probe] = style_row
            grouped[probe].append(style_row)
        out["per_pair"].append(pair_row)

    for probe, rows in grouped.items():
        if not rows:
            continue
        out["by_probe"][probe] = {
            "n_pairs": len(rows),
            "static_B_A": mean([r["static_B_A"] for r in rows if r["static_B_A"] is not None]),
            "static_two_way_accuracy_vs_mapping_A": mean(
                [r["two_way_acc_static"] for r in rows if r["two_way_acc_static"] is not None]
            ),
            "B_A": mean([r["B_A"] for r in rows]),
            "B_B": mean([r["B_B"] for r in rows]),
            "B_swap": mean([r["B_swap"] for r in rows]),
            "n_B_swap_positive": int(sum(r["B_swap"] > 0 for r in rows)),
            "fraction_B_swap_gt_0": mean([float(r["B_swap"] > 0) for r in rows]),
            "fraction_B_A_and_B_B_gt_0": mean([float(r["B_A"] > 0 and r["B_B"] > 0) for r in rows]),
            "fraction_e1_flipped": mean([float(r["e1_flipped"]) for r in rows]),
            "fraction_e2_flipped": mean([float(r["e2_flipped"]) for r in rows]),
            "fraction_both_flipped": mean([float(r["both_flipped"]) for r in rows]),
            "ttt_two_way_accuracy": mean([r["two_way_acc_ttt"] for r in rows]),
            "generation_em_ttt": mean([r["generation_em_ttt"] for r in rows]),
            "write_loss_pre": mean(
                [x for r in rows for x in (r["write_loss_pre_A"], r["write_loss_pre_B"]) if x is not None]
            ),
            "write_loss_post": mean(
                [x for r in rows for x in (r["write_loss_post_A"], r["write_loss_post_B"]) if x is not None]
            ),
            "lora_l2_delta": mean(
                [x for r in rows for x in (r["lora_l2_A"], r["lora_l2_B"]) if x is not None]
            ),
        }
    return out


def run_invariants(
    runtime: LoadedRuntime,
    pairs: list[dict[str, Any]],
    records: list[dict[str, Any]],
    *,
    log_fn=print,
) -> list[str]:
    failures: list[str] = []

    def fail(msg: str) -> None:
        failures.append(msg)
        log_fn(f"SANITY FAIL: {msg}")

    def ok(msg: str) -> None:
        log_fn(f"OK {msg}")

    log_fn("\n=== Exp 02A-2 source-free sanity checks ===")
    errors = [r for r in records if r["status"] != "ok"]
    if errors:
        fail(f"{len(errors)} evaluation errors, first={errors[0]['error']}")
    else:
        ok(f"all {len(records)} records status=ok")

    for p in pairs:
        for mapping in ("mapping_A", "mapping_B"):
            for seq in p["sequences"][mapping]:
                if seq["label"] in seq["write_sentence"]:
                    fail(f"{seq['sequence_id']}: label in write sentence")
        if not p["token_inventory_A_equals_B"]:
            fail(f"{p['pair_id']}: A/B token inventories differ")
    ok("occupation labels absent from write; A/B token inventories equal")

    for rec in records:
        if rec["status"] != "ok":
            continue
        if rec["condition"] in TTT_CONDITIONS:
            w = rec.get("write") or {}
            if not w.get("loss_decreased"):
                fail(
                    f"{rec['pair_id']} {rec['condition']}: write loss did not decrease "
                    f"{w.get('loss_before')} -> {w.get('loss_after')}"
                )
            if not (w.get("lora_max_abs_diff_from_init") or 0) > 0:
                fail(f"{rec['pair_id']} {rec['condition']}: LoRA delta is 0")
            if w.get("returned_past_key_values") or w.get("write_used_cache"):
                fail(f"{rec['pair_id']} {rec['condition']}: write used or returned cache")
            if rec.get("recall_updated_lora"):
                fail(f"{rec['pair_id']} {rec['condition']}: recall updated LoRA")
            for probe in rec["probes"]:
                for ent_key in ("e1", "e2"):
                    prompt = rec["probes"][probe]["entities"][ent_key]["recall_prompt"]
                    pair = next(p for p in pairs if p["pair_id"] == rec["pair_id"])
                    _assert_source_free_prompt(pair, prompt)
        if rec["condition"] == "static_source_free":
            for probe in rec["probes"]:
                for ent_key in ("e1", "e2"):
                    prompt = rec["probes"][probe]["entities"][ent_key]["recall_prompt"]
                    pair = next(p for p in pairs if p["pair_id"] == rec["pair_id"])
                    _assert_source_free_prompt(pair, prompt)
        if rec.get("source_present"):
            pair = next(p for p in pairs if p["pair_id"] == rec["pair_id"])
            mapping = rec["mapping"]
            for probe in rec["probes"]:
                for ent_key in ("e1", "e2"):
                    prompt = rec["probes"][probe]["entities"][ent_key]["recall_prompt"]
                    if pair["sequences"][mapping][0]["write_sentence"] not in prompt:
                        fail(f"{rec['pair_id']} {rec['condition']}: source-present missing write")
    ok("write/recall invariants (source-free, KV discarded, LoRA frozen at recall)")

    base_diff = frozen_watch_max_abs_diff(runtime.model, runtime.frozen_watch_cpu)
    if base_diff != 0.0:
        fail(f"base weights changed at end: {base_diff}")
    else:
        ok("base weights unchanged")
    reset_diff = lora_max_abs_diff(runtime.model, runtime.lora_init_cpu)
    if reset_diff != 0.0:
        fail(f"LoRA not restored to init at end: {reset_diff}")
    else:
        ok("LoRA restored to identical init after each mapping")

    if failures:
        log_fn(f"\n{len(failures)} sanity failure(s).")
    else:
        log_fn("\nAll source-free sanity checks passed.")
    return failures


def write_jsonl(path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for r in records:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")


def render_summary_md(table: dict[str, Any], *, dataset_sha: str, n_pairs: int) -> str:
    lines = [
        "# Exp 02A-2 source-free semantic binding (K=32)",
        "",
        f"Frozen dataset_v3 sha256: `{dataset_sha}`",
        f"Pairs: {n_pairs} (P1–P4). Independent-sequence batch write. No source at recall.",
        "",
        "Do **not** call this semantic memory from top-1 alone.",
        "Positive description `B_swap` = source-free relational description binding.",
        "Positive semantic-label `B_swap` with labels absent from write = semantic accessibility beyond lexical imprint.",
        "",
        "| probe | static B_A | B_A | B_B | B_swap | n B_swap>0 | e1 flip | e2 flip | both flip | TTT 2-way | greedy EM |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for probe, b in table["by_probe"].items():
        lines.append(
            "| {probe} | {st:.3f} | {ba:.3f} | {bb:.3f} | {bs:.3f} | {npos}/{n} | {e1:.3f} | {e2:.3f} | {both:.3f} | {ttt:.3f} | {em:.3f} |".format(
                probe=probe,
                st=b["static_B_A"],
                ba=b["B_A"],
                bb=b["B_B"],
                bs=b["B_swap"],
                npos=b["n_B_swap_positive"],
                n=b["n_pairs"],
                e1=b["fraction_e1_flipped"],
                e2=b["fraction_e2_flipped"],
                both=b["fraction_both_flipped"],
                ttt=b["ttt_two_way_accuracy"],
                em=b["generation_em_ttt"],
            )
        )
    lines.extend(["", "## Per pair", ""])
    lines.append("| pair | probe | B_A | B_B | B_swap | both flip | TTT 2-way |")
    lines.append("|---|---|---:|---:|---:|---:|---:|")
    for prow in table["per_pair"]:
        for probe, b in prow["probes"].items():
            lines.append(
                f"| {prow['pair_id']} | {probe} | {b['B_A']:.3f} | {b['B_B']:.3f} | "
                f"{b['B_swap']:.3f} | {b['both_flipped']} | {b['two_way_acc_ttt']:.3f} |"
            )
    lines.append("")
    return "\n".join(lines)
