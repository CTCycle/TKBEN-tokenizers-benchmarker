[CmdletBinding()]
param()

$ErrorActionPreference = 'Stop'
$repo = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..\..\..'))
$bundle = Join-Path $repo 'release\tauri\.bundle-src'
$files = @{
  'app\server\app.py' = '# cargo-check resource placeholder';
  'app\client\dist\index.html' = '<!doctype html><title>TKBEN check resource</title>';
  'settings\.env.example' = '';
  'settings\configurations.json' = '{}';
  'runtimes\python\python.exe' = ''
}
foreach ($relative in $files.Keys) {
  $path = Join-Path $bundle $relative
  New-Item (Split-Path $path) -ItemType Directory -Force | Out-Null
  if (-not (Test-Path $path)) { Set-Content $path $files[$relative] -Encoding ascii }
}
Write-Host "[OK] Prepared generated cargo-check resource skeleton."
