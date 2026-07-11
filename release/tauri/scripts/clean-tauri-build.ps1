[CmdletBinding()]
param([switch]$KeepBundleSource)

$ErrorActionPreference = "Stop"

$repoRoot = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot "..\..\.."))
$pathsToRemove = @(
  (Join-Path $repoRoot "app\src-tauri\target"),
  (Join-Path $repoRoot "release\windows")
)
if (-not $KeepBundleSource) { $pathsToRemove += (Join-Path $repoRoot "release\tauri\.bundle-src") }

foreach ($path in $pathsToRemove) {
  if (Test-Path $path) {
    Remove-Item -Recurse -Force $path
    Write-Host "[OK] Removed: $path"
  } else {
    Write-Host "[INFO] Not found: $path"
  }
}

Write-Host "[DONE] Build cleanup complete."
