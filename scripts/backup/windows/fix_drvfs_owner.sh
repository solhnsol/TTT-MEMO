#!/usr/bin/env bash
# Run as root inside the WSL distro:  bash fix_drvfs_owner.sh <linux-user>
# Windows drives were mounted uid=0, so the backup user cannot set file times
# (rsync -t fails and every pull re-reads everything). Mount them as that user.
set -euo pipefail
u="${1:?linux user}"
uid="$(id -u "$u")"; gid="$(id -g "$u")"
conf=/etc/wsl.conf
touch "$conf"
if grep -q '^\[automount\]' "$conf"; then
  sed -i '/^\[automount\]/,/^\[/{/^options *=/d}' "$conf"
  sed -i "/^\[automount\]/a options = \"metadata,uid=$uid,gid=$gid,umask=022\"" "$conf"
else
  printf '\n[automount]\noptions = "metadata,uid=%s,gid=%s,umask=022"\n' "$uid" "$gid" >> "$conf"
fi
cat "$conf"
echo "Now run 'wsl --shutdown' in PowerShell, then reopen WSL."
