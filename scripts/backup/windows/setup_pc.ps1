# One-time setup on the Windows lab PC (PowerShell, normal user; WSL must exist).
#   powershell -ExecutionPolicy Bypass -File setup_pc.ps1 -A100Host <ip-or-name> [-Port 22] [-User solhnsol] [-Root D:\Project\KFAI\2026\TTT_ACL] [-WinSshAlias A100Server]
# Installs rsync in WSL, creates an ssh key + `a100` alias, copies the pull script,
# and registers a hidden Task Scheduler job that pulls every 15 minutes.
param(
  [Parameter(Mandatory = $true)][string]$A100Host,
  [int]$Port = 22,
  [string]$User = "solhnsol",
  [string]$WinSshAlias = "",   # existing Windows ssh alias; if set, the new key is installed on the A100 through it
  [string]$Root = "D:\Project\KFAI\2026\TTT_ACL",
  [int]$Minutes = 15,
  [string]$Distro = "Ubuntu"    # a real Linux distro (Docker Desktop's default distro has no bash)
)
$ErrorActionPreference = "Stop"
if (-not (Get-Command wsl.exe -ErrorAction SilentlyContinue)) { throw "WSL not found. Run 'wsl --install -d Ubuntu' as admin, reboot, then rerun." }
$distros = (wsl.exe -l -q) -replace "`0", "" | ForEach-Object { $_.Trim() } | Where-Object { $_ }
if ($distros -notcontains $Distro) {
  throw "WSL distro '$Distro' not found (have: $($distros -join ', ')). Run 'wsl --install -d Ubuntu', create the Linux user it asks for, then rerun."
}

$drive = $Root.Substring(0, 1).ToLower()
$wslRoot = "/mnt/$drive" + ($Root.Substring(2) -replace '\\', '/')
New-Item -ItemType Directory -Force -Path "$Root\TTT-MEMO", "$Root\_bin" | Out-Null
$raw = "https://raw.githubusercontent.com/solhnsol/TTT-MEMO/autosnap/a100/scripts/backup/pull_from_a100.sh"
Invoke-WebRequest -UseBasicParsing $raw -OutFile "$Root\_bin\pull_from_a100.sh"

Write-Host "Installing rsync in WSL (may ask for your WSL sudo password)..."
wsl.exe -d $Distro -e bash -lc "command -v rsync >/dev/null || (sudo apt-get update -qq && sudo apt-get install -y -qq rsync)"

$sshSetup = @"
set -e
mkdir -p ~/.ssh && chmod 700 ~/.ssh
[ -f ~/.ssh/a100_backup ] || ssh-keygen -q -t ed25519 -N '' -C 'lab-pc-backup' -f ~/.ssh/a100_backup
grep -q '^Host a100$' ~/.ssh/config 2>/dev/null || printf 'Host a100\n  HostName $A100Host\n  Port $Port\n  User $User\n  IdentityFile ~/.ssh/a100_backup\n  StrictHostKeyChecking accept-new\n' >> ~/.ssh/config
chmod 600 ~/.ssh/config
sed -i 's/\r$//' '$wslRoot/_bin/pull_from_a100.sh'
"@
wsl.exe -d $Distro -e bash -lc ($sshSetup -replace "`r", "")

$cmd = "DEST='$wslRoot/TTT-MEMO/' bash '$wslRoot/_bin/pull_from_a100.sh'"
$action = New-ScheduledTaskAction -Execute "conhost.exe" -Argument "--headless wsl.exe -d $Distro -e bash -lc `"$cmd`""
$trigger = New-ScheduledTaskTrigger -Once -At (Get-Date).AddMinutes(1) -RepetitionInterval (New-TimeSpan -Minutes $Minutes)
$settings = New-ScheduledTaskSettingsSet -StartWhenAvailable -DontStopIfGoingOnBatteries -AllowStartIfOnBatteries -MultipleInstances IgnoreNew -ExecutionTimeLimit (New-TimeSpan -Hours 12)
Register-ScheduledTask -TaskName "TTT-MEMO A100 backup" -Action $action -Trigger $trigger -Settings $settings -Force | Out-Null

Write-Host "`nTask registered: every $Minutes min -> $Root\TTT-MEMO (logs: $Root\_logs, last success: $Root\LAST_OK)"
$pub = (wsl.exe -d $Distro -e bash -lc "cat ~/.ssh/a100_backup.pub").Trim()
if ($WinSshAlias) {
  $pub | ssh $WinSshAlias "mkdir -p ~/.ssh; tr -d '\r' >> ~/.ssh/authorized_keys"
  wsl.exe -d $Distro -e bash -lc "ssh -o BatchMode=yes a100 true && echo 'WSL -> A100 ssh OK'"
} else {
  Write-Host "Send this public key to Claude so it can be added on the A100:`n$pub"
}
