[CmdletBinding()]
param([string]$UvPath = '')

$ErrorActionPreference = 'Stop'
$ProgressPreference = 'SilentlyContinue'
$repo = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..\..\..'))
$bundle = Join-Path $repo 'release\tauri\.bundle-src'
$pythonDir = Join-Path $bundle 'runtimes\python'
$serverDir = Join-Path $bundle 'app\server'
$clientDist = Join-Path $bundle 'app\client\dist'
$settingsDir = Join-Path $bundle 'settings'
$pythonVersion = '3.14.2'

if (Test-Path $bundle) { Remove-Item $bundle -Recurse -Force }
New-Item $pythonDir,$serverDir,$clientDist,$settingsDir -ItemType Directory -Force | Out-Null

if (-not $UvPath) {
  $localUv = Join-Path $repo 'runtimes\uv\uv.exe'
  if (Test-Path $localUv) { $UvPath = $localUv } else { $UvPath = (Get-Command uv -ErrorAction Stop).Source }
}

foreach ($required in @('app\server\uv.lock','app\client\package-lock.json','settings\.env.example','settings\configurations.json','app\client\dist\index.html')) {
  if (-not (Test-Path (Join-Path $repo $required))) { throw "Required release input is missing: $required" }
}

$zip = Join-Path $env:TEMP "tkben-package-python-$PID.zip"
Invoke-WebRequest "https://www.python.org/ftp/python/$pythonVersion/python-$pythonVersion-embed-amd64.zip" -OutFile $zip
Expand-Archive $zip $pythonDir -Force
Remove-Item $zip -Force
$pth = Join-Path $pythonDir 'python314._pth'
$pthLines = @(Get-Content $pth | Where-Object { $_ -ne '#import site' }) + 'Lib\site-packages' + 'import site'
$pthLines | Set-Content $pth -Encoding ascii

$requirements = Join-Path $bundle 'requirements.txt'
Push-Location (Join-Path $repo 'app\server')
try {
  & $UvPath export --quiet --frozen --no-dev --no-emit-project --format requirements-txt --output-file $requirements
  if ($LASTEXITCODE) { throw 'uv export failed.' }
  & $UvPath pip install --quiet --link-mode copy --python (Join-Path $pythonDir 'python.exe') --target (Join-Path $pythonDir 'Lib\site-packages') --requirements $requirements --no-deps
  if ($LASTEXITCODE) { throw 'uv pip install failed.' }
} finally { Pop-Location }
Remove-Item $requirements -Force

robocopy (Join-Path $repo 'app\server') $serverDir /E /NFL /NDL /NJH /NJS /NC /NS /XD .venv __pycache__ .pytest_cache .ruff_cache /XF uv.lock *.pyc | Out-Null
if ($LASTEXITCODE -ge 8) { throw 'Failed to copy filtered backend sources.' }
robocopy (Join-Path $repo 'app\client\dist') $clientDist /E /NFL /NDL /NJH /NJS /NC /NS | Out-Null
if ($LASTEXITCODE -ge 8) { throw 'Failed to copy frontend distribution.' }
Copy-Item (Join-Path $repo 'settings\.env.example') $settingsDir
Copy-Item (Join-Path $repo 'settings\configurations.json') $settingsDir

$sitePackages = Join-Path $pythonDir 'Lib\site-packages'
Get-ChildItem $sitePackages -Recurse -Directory -ErrorAction SilentlyContinue |
  Where-Object Name -in @('test','tests','__pycache__') |
  Sort-Object FullName -Descending |
  Remove-Item -Recurse -Force
$forbidden = Get-ChildItem $bundle -Recurse -File | Where-Object {
  $_.Name -match '(?i)(\.db|\.log|\.pyc|\.pfx|\.key)$' -or
  ($_.Name -match '(?i)\.pem$' -and $_.FullName -notmatch '(?i)\\certifi\\cacert\.pem$') -or
  $_.FullName -match '(?i)(\\tests?\\|\\__pycache__\\|\\node_modules\\)'
}
if ($forbidden) { throw "Forbidden files entered the runtime payload: $($forbidden.FullName -join ', ')" }
Write-Host "[OK] Prepared self-contained runtime: $bundle"
