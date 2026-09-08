# Install/verify cannbot-skills (external operator skill library used by operator-dev).
# Usage: powershell -File install_cannbot.ps1
#        $env:CANNBOT_SKILLS_DIR = "D:\somewhere\cannbot-skills"; powershell -File install_cannbot.ps1
#        $env:CANNBOT_UPDATE = "1"; powershell -File install_cannbot.ps1   # git pull when already cloned
$ErrorActionPreference = "Stop"

$remote = if ($env:CANNBOT_REMOTE) { $env:CANNBOT_REMOTE } else { "https://gitcode.com/cann/cannbot-skills" }
$target = if ($env:CANNBOT_SKILLS_DIR) { $env:CANNBOT_SKILLS_DIR } else { Join-Path $HOME ".cannbot-skills" }
$required = @(
  "AGENTS.md",
  "ops/triton-latency-optimizer/SKILL.md",
  "ops/triton-op-coding/SKILL.md",
  "ops/triton-op-designer/SKILL.md",
  "ops/ascendc-tiling-design/SKILL.md"
)

Write-Output "== cannbot-skills install =="
Write-Output "target : $target"
Write-Output "remote : $remote"

if (Test-Path (Join-Path $target ".git")) {
  Write-Output "already cloned."
  if ($env:CANNBOT_UPDATE -eq "1") {
    Write-Output "updating..."
    git -C $target pull --ff-only
  }
} else {
  New-Item -ItemType Directory -Force -Path (Split-Path $target -Parent) | Out-Null
  git clone --depth 1 $remote $target
}

$missing = @()
foreach ($f in $required) {
  if (-not (Test-Path (Join-Path $target $f))) { $missing += $f }
}
if ($missing.Count -gt 0) {
  Write-Error ("cannbot-skills incomplete at {0}: missing {1}" -f $target, ($missing -join ", "))
}

Write-Output "OK: cannbot-skills ready at $target"
Write-Output "hint: set CANNBOT_SKILLS_DIR=$target and reuse in operator-dev tasks"
Write-Output "hint: repo bundles its own plugin installers (install.sh / install.ps1) if needed."
