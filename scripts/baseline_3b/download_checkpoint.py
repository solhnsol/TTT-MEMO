"""Download the official 3B 128K TTT-E2E Orbax checkpoint from GCS.

Requester Pays bucket (documented in e2e/README.md):

    gs://ttt-e2e-checkpoints/3b_ttt_e2e_finetune_books_128k_3x_cc

Does not convert weights. Writes next to the existing 125M directory, not into it.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

_BASELINE_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(_BASELINE_ROOT / "common"))

from baseline_config import CHECKPOINT_DIR, CHECKPOINT_GCS, REPO_ROOT  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--billing-project",
        default=os.environ.get("GOOGLE_CLOUD_PROJECT") or os.environ.get("CLOUDSDK_CORE_PROJECT") or "",
        help="GCP project billed for Requester Pays. Env GOOGLE_CLOUD_PROJECT also works.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    dest = CHECKPOINT_DIR
    if dest.is_dir() and any(dest.iterdir()):
        print(f"checkpoint already present: {dest}")
        return
    gcloud = shutil.which("gcloud")
    if gcloud is None:
        raise SystemExit(
            "gcloud is not on PATH. Official download (Requester Pays):\n"
            f"  gcloud storage cp -r {CHECKPOINT_GCS} {dest}\n"
            "    --billing-project=<YOUR_GCP_PROJECT>\n"
            "The 125M checkpoint is already local; this 3B object is not."
        )
    dest.parent.mkdir(parents=True, exist_ok=True)
    cmd = [gcloud, "storage", "cp", "-r", CHECKPOINT_GCS, str(dest)]
    if args.billing_project:
        cmd.append(f"--billing-project={args.billing_project}")
    print("running:", " ".join(cmd))
    subprocess.check_call(cmd, cwd=REPO_ROOT)


if __name__ == "__main__":
    main()
