#!/usr/bin/env bash
# Runs on the A100 after `pull_from_a100.sh --restore-test` on the PC.
# Checksums the copy that went A100 -> PC -> A100 against the originals.
set -euo pipefail
REPO="$(cd "$(dirname "$0")/../.." && pwd)"
T=/tmp/ttt_restore_test
fail=0
for d in experiments/exp05_write_dynamics results scripts; do
  if out="$(cd "$REPO/$d" && find . -type f ! -path '*/__pycache__/*' -print0 | sort -z \
            | xargs -0 nice sha256sum | (cd "$T/$d" && sha256sum --quiet -c - 2>&1))"; then
    echo "OK   $d ($(find "$T/$d" -type f | wc -l) files, $(du -sh "$T/$d" | cut -f1))"
  else
    echo "DIFF $d"; echo "$out" | head; fail=1
  fi
done
exit $fail
