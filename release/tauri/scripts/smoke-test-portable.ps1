[CmdletBinding()]
param([Parameter(Mandatory=$true)][string]$ArchivePath)

$ErrorActionPreference = 'Stop'
$archive = [IO.Path]::GetFullPath($ArchivePath)
if (-not (Test-Path $archive)) { throw "Portable archive not found: $archive" }
$extract = Join-Path $env:TEMP "tkben-portable-smoke-$PID"
$logDir = Join-Path $env:LOCALAPPDATA 'com.tkben.desktop\logs'
if (Test-Path $extract) { Remove-Item $extract -Recurse -Force }
New-Item $extract -ItemType Directory -Force | Out-Null
try {
  & tar.exe -xf $archive -C $extract
  if ($LASTEXITCODE) { throw "Portable extraction failed with exit code $LASTEXITCODE" }
  $exe = Get-ChildItem $extract -File -Filter '*.exe' | Select-Object -First 1
  if (-not $exe) { throw 'Portable archive contains no root executable.' }
  foreach ($required in @('app\server\app.py','app\client\dist\index.html','runtimes\python\python.exe','settings\.env.example')) {
    if (-not (Test-Path (Join-Path $extract $required))) { throw "Portable payload is missing $required" }
  }
  $started = Get-Date
  $process = Start-Process $exe.FullName -WorkingDirectory $extract -PassThru
  $deadline = (Get-Date).AddSeconds(120)
  do {
    $log = Get-Item (Join-Path $logDir 'desktop-backend.log') -ErrorAction SilentlyContinue
    if ($log -and $log.LastWriteTime -lt $started) { $log = $null }
    if ($log -and (Get-Content $log.FullName -Raw -ErrorAction SilentlyContinue) -match 'Uvicorn running on http://127\.0\.0\.1:') { break }
    if ($process.HasExited) { throw "Portable application exited before backend readiness with code $($process.ExitCode)." }
    Start-Sleep -Milliseconds 500
  } while ((Get-Date) -lt $deadline)
  if (-not $log) { throw 'Portable backend log was not created before timeout.' }
  Write-Host "[OK] Portable application and bundled backend started. Log: $($log.FullName)"
} finally {
  if ($process -and -not $process.HasExited) { & taskkill.exe /PID $process.Id /T /F | Out-Null }
  if (Test-Path $extract) { Remove-Item $extract -Recurse -Force }
}
