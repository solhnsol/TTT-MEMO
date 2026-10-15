#!/usr/bin/env bash
# Runs on the lab PC (Linux, macOS, or WSL on Windows). Pulls everything needed to
# resume TTT-MEMO from the A100 into $DEST, every 15 min via cron / Task Scheduler.
#
#   pull_from_a100.sh                 # incremental mirror (never deletes on the PC)
#   pull_from_a100.sh --restore-test  # round-trip check: push a sample back to A100 /tmp
#
# Files deleted or overwritten on the A100 are kept on the PC under
# $DEST/../_previous/<date>/ instead of being lost.
set -euo pipefail
A100="${A100:-a100}"   # ssh alias from ~/.ssh/config on the PC (see RESTORE.md)
SRC="${SRC:-/home/solhnsol/workspace/TTT-MEMO/}"
DEST="${DEST:-$HOME/a100-backup/TTT-MEMO/}"
REMOTE_RSYNC="${REMOTE_RSYNC:-/home/solhnsol/.local/bin/rsync}"  # A100 has no system rsync
BWLIMIT="${BWLIMIT:-50000}"                         # KB/s, keeps A100 disk/network light
SSH="${SSH:-ssh -o BatchMode=yes -o ConnectTimeout=20 -o ServerAliveInterval=30}"

mkdir -p "$DEST" "$DEST/../_logs"
LOG="$DEST/../_logs/pull_$(date +%F).log"
exec 9>"$DEST/../.pull.lock"
flock -n 9 || { echo "another pull is running"; exit 0; }

EXCLUDES=(
  --exclude=/.venv/ --exclude=/.venv-exp02a/ --exclude=__pycache__/
  --exclude=.jax_cache/ --exclude=jax_cache/ --exclude=.cache/
  --exclude=/.git/autosnap.lock --exclude=/.git/index.lock
)

if [ "${1:-}" = "--restore-test" ]; then
  # Round trip: A100 -> PC (done by a normal pull) -> A100 scratch dir.
  for d in experiments/exp05_write_dynamics results scripts; do
    rsync -a --rsync-path="$REMOTE_RSYNC" -e "$SSH" "$DEST$d/" "$A100:/tmp/ttt_restore_test/$d/"
  done
  echo "pushed sample to $A100:/tmp/ttt_restore_test; verify there with scripts/backup/verify_restore_test.sh"
  exit 0
fi

{
  echo "=== $(date '+%F %T') pull start"
  rsync -a --partial --human-readable --stats \
    --backup --backup-dir="../_previous/$(date +%F)" \
    --bwlimit="$BWLIMIT" --rsync-path="ionice -c3 nice -n19 $REMOTE_RSYNC" \
    -e "$SSH" "${EXCLUDES[@]}" "$A100:$SRC" "$DEST"
  # Small extras outside the repo: Claude memory for this project.
  rsync -a -e "$SSH" --rsync-path="nice $REMOTE_RSYNC" \
    "$A100:.claude/projects/-home-solhnsol-workspace-TTT-MEMO/memory/" "$DEST/../_claude_memory/" || true
  find "$DEST/../_previous" -mindepth 1 -maxdepth 1 -mtime +14 -exec rm -rf {} + 2>/dev/null || true
  date '+%F %T' > "$DEST/../LAST_OK"
  echo "=== $(date '+%F %T') pull ok"
} >>"$LOG" 2>&1
