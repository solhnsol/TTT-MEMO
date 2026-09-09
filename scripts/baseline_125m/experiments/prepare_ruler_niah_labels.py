"""Build compact RULER NIAH token-level labels for all validation JSONL rows.

CPU only. Does not load the checkpoint or run TTT.

    PYTHONPATH=scripts/baseline_125m/common .venv/bin/python \
        scripts/baseline_125m/experiments/prepare_ruler_niah_labels.py
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

_BASELINE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_BASELINE_ROOT / "common"))

from ruler_niah_labels import (  # noqa: E402
    EXPECTED_N_RELEVANT,
    LABEL_RESULT_DIR,
    MINI_BATCH,
    RULER_ROOT,
    TASK_FAMILY,
    TASK_VTYPE,
    TOKENIZER_PATH,
    TOKENIZER_SHA256,
    LabelError,
    example_row,
    label_example,
    processed_jsonl,
    sha256_file,
    span_rows,
)
from tokenizers import Tokenizer  # noqa: E402

SEQ_LABELS = ("8k", "16k", "32k", "64k", "128k")
TASKS = list(TASK_FAMILY)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--max-per-split", type=int, default=0, help="0 = all 200")
    p.add_argument("--tasks", nargs="*", default=None)
    p.add_argument("--seq-labels", nargs="*", default=None)
    p.add_argument(
        "--out-dir",
        type=Path,
        default=LABEL_RESULT_DIR,
        help="Parquet + summary.json destination",
    )
    return p.parse_args()


def load_jsonl(path: Path, limit: int) -> list[dict]:
    rows = []
    with path.open() as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
            if limit and len(rows) >= limit:
                break
    return rows


def main() -> None:
    args = parse_args()
    if not TOKENIZER_PATH.exists():
        raise SystemExit(f"tokenizer missing: {TOKENIZER_PATH}")
    digest = sha256_file(TOKENIZER_PATH)
    if digest != TOKENIZER_SHA256:
        raise SystemExit(f"tokenizer sha256 {digest} != pin {TOKENIZER_SHA256}")
    tok = Tokenizer.from_file(str(TOKENIZER_PATH))
    tasks = args.tasks or TASKS
    seqs = args.seq_labels or list(SEQ_LABELS)
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    import pandas as pd

    failures: list[dict] = []
    split_stats: list[dict] = []
    n_ok = 0
    n_fail = 0
    t0 = time.time()
    example_parts: list[Path] = []
    span_parts: list[Path] = []

    for task in tasks:
        for label in seqs:
            path = processed_jsonl(task, label)
            rel = str(path.relative_to(RULER_ROOT))
            if not path.exists():
                failures.append({"task": task, "seq_length_label": label, "error": f"missing {path}"})
                n_fail += 1
                print(f"MISSING {rel}", flush=True)
                continue
            recs = load_jsonl(path, args.max_per_split)
            example_rows = []
            span_acc = []
            n_rel = []
            n_dis = []
            n_out = []
            any_out = 0
            all_out = 0
            bos_ok = 0
            t_pad_ok = 0
            valid_ok = 0
            for rec in recs:
                sid = rec.get("sample_id")
                try:
                    ex = label_example(rec, tok, source_jsonl=rel, tokenizer_sha256=digest)
                    if ex.input_ids[0] != 128000:
                        raise LabelError("BOS")
                    if ex.T_pad % MINI_BATCH != 0:
                        raise LabelError("T_pad")
                    if ex.valid_loss_count != ex.T - 1:
                        raise LabelError("valid_loss_count")
                    example_rows.append(example_row(ex))
                    span_acc.extend(span_rows(ex))
                    n_rel.append(ex.n_relevant_spans)
                    n_dis.append(ex.n_distractor_spans)
                    n_out.append(ex.n_out_of_swa_spans)
                    any_out += int(ex.any_relevant_out_of_swa)
                    all_out += int(ex.all_relevant_out_of_swa)
                    bos_ok += 1
                    t_pad_ok += 1
                    valid_ok += 1
                    n_ok += 1
                except LabelError as exc:
                    n_fail += 1
                    failures.append(
                        {
                            "task": task,
                            "seq_length_label": label,
                            "sample_id": sid,
                            "source_jsonl": rel,
                            "error": str(exc),
                        }
                    )
            stem = f"{task}_{label}"
            ex_path = out_dir / f"examples_{stem}.parquet"
            sp_path = out_dir / f"spans_{stem}.parquet"
            pd.DataFrame(example_rows).to_parquet(ex_path, index=False)
            pd.DataFrame(span_acc).to_parquet(sp_path, index=False)
            example_parts.append(ex_path)
            span_parts.append(sp_path)
            expected_rel = EXPECTED_N_RELEVANT[task]
            rel_counter = dict(Counter(n_rel))
            dis_counter = dict(Counter(n_dis))
            mismatch = []
            if any(n != expected_rel for n in n_rel):
                mismatch.append(f"relevant counts {rel_counter} != expected {expected_rel}")
            if task in {"niah_single_1", "niah_single_2", "niah_single_3", "niah_multiquery", "niah_multivalue"}:
                if any(n != 0 for n in n_dis):
                    mismatch.append(f"unexpected distractors {dis_counter}")
            elif task == "niah_multikey_1":
                if any(n == 0 for n in n_dis):
                    mismatch.append(f"MK1 expected distractors, got {dis_counter}")
            elif task in {"niah_multikey_2", "niah_multikey_3"}:
                if any(n < 10 for n in n_dis):
                    mismatch.append(f"MK2/3 expected many distractors, got {dis_counter}")
            split_stats.append(
                {
                    "task": task,
                    "task_family": TASK_FAMILY[task],
                    "vtype": TASK_VTYPE[task],
                    "seq_length_label": label,
                    "n_jsonl": len(recs),
                    "n_ok": len(example_rows),
                    "n_fail": len(recs) - len(example_rows),
                    "n_relevant": rel_counter,
                    "n_distractor": {
                        "min": min(n_dis) if n_dis else None,
                        "median": sorted(n_dis)[len(n_dis) // 2] if n_dis else None,
                        "max": max(n_dis) if n_dis else None,
                        "counts_head": dict(Counter(n_dis).most_common(6)),
                    },
                    "n_out_of_swa_spans": dict(Counter(n_out)),
                    "n_any_relevant_out_of_swa": any_out,
                    "n_all_relevant_out_of_swa": all_out,
                    "bos_ok": bos_ok,
                    "t_pad_ok": t_pad_ok,
                    "valid_loss_ok": valid_ok,
                    "expectation_mismatch": mismatch,
                }
            )
            print(
                f"done {task}/{label} ok={len(example_rows)}/{len(recs)} "
                f"relevant={rel_counter} distractor_med={split_stats[-1]['n_distractor']['median']} "
                f"any_out_swa={any_out} mismatch={mismatch}",
                flush=True,
            )

    fail_path = out_dir / "failures.jsonl"
    with fail_path.open("w") as fh:
        for row in failures:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")

    summary = {
        "n_ok": n_ok,
        "n_fail": n_fail,
        "elapsed_s": round(time.time() - t0, 1),
        "tokenizer_path": str(TOKENIZER_PATH),
        "tokenizer_sha256": digest,
        "pad_id": 128000,
        "mini_batch": MINI_BATCH,
        "swa": 8192,
        "out_of_swa_rule": "distance_to_query = query_start_j - needle_last_j; out iff > 8191",
        "indexing": "loss_t = raw_j - 1; G_t attached to predicted target input_ids[t+1]",
        "chat_template": "kept as stored in JSONL input; add_special_tokens=False",
        "failures_path": str(fail_path),
        "examples_parquet": str(out_dir / "examples.parquet"),
        "spans_parquet": str(out_dir / "spans.parquet"),
        "splits": split_stats,
    }
    summary_path = out_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")

    # Combined tables for later analysis (still compact vs raw JSONL text).
    if example_parts:
        examples = pd.concat([pd.read_parquet(p) for p in example_parts], ignore_index=True)
        spans = pd.concat([pd.read_parquet(p) for p in span_parts], ignore_index=True)
        examples.to_parquet(out_dir / "examples.parquet", index=False)
        spans.to_parquet(out_dir / "spans.parquet", index=False)
        for p in example_parts + span_parts:
            p.unlink(missing_ok=True)

    print(f"\nOK={n_ok} FAIL={n_fail} wrote {out_dir}", flush=True)
    if n_fail:
        print(f"failures: {fail_path}", flush=True)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
