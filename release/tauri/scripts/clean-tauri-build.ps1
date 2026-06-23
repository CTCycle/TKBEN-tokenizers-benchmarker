[CmdletBinding()]
param()

$ErrorActionPreference = "Stop"

$repoRoot = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot "..\..\.."))
$pathsToRemove = @(
  (Join-Path $repoRoot "app\src-tauri\target"),
  (Join-Path $repoRoot "app\src-tauri\bundle"),
  (Join-Path $repoRoot "app\src-tauri\gen"),
  (Join-Path $repoRoot "release\windows")
)

foreach ($path in $pathsToRemove) {
  if (Test-Path $path) {
    Remove-Item -LiteralPath $path -Recurse -Force -ErrorAction Stop
    Write-Host "[OK] Removed: $path"
  } else {
    Write-Host "[INFO] Not found: $path"
  }
}

Write-Host "[DONE] Build cleanup complete."
