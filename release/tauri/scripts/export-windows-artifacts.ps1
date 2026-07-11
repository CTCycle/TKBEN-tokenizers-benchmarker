[CmdletBinding()]
param([string]$OutputPath = '')

$ErrorActionPreference = 'Stop'
$repo = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..\..\..'))
$config = Get-Content (Join-Path $repo 'app\src-tauri\tauri.conf.json') -Raw | ConvertFrom-Json
$version = $config.version
$targetRelease = Join-Path $repo 'app\src-tauri\target\x86_64-pc-windows-msvc\release'
if (-not (Test-Path $targetRelease)) { $targetRelease = Join-Path $repo 'app\src-tauri\target\release' }
$output = if ($OutputPath) { [IO.Path]::GetFullPath((Join-Path $repo $OutputPath)) } else { Join-Path $repo 'release\windows' }
$stage = Join-Path $output 'portable-stage'
$baseName = "TKBEN-Desktop-$version-windows-x64"

if (Test-Path $output) { Remove-Item $output -Recurse -Force }
New-Item $output,$stage -ItemType Directory -Force | Out-Null
$exe = Get-ChildItem $targetRelease -File -Filter '*.exe' | Where-Object Name -NotMatch '(?i)(setup|uninstall)' | Select-Object -First 1
if (-not $exe) { throw "Portable executable missing in $targetRelease" }
$msi = Get-ChildItem (Join-Path $targetRelease 'bundle\msi') -File -Filter '*.msi' | Select-Object -First 1
if (-not $msi) { throw 'MSI bundle was not produced.' }
Copy-Item $exe.FullName (Join-Path $stage 'TKBEN Desktop.exe')

$bundleSource = Join-Path $repo 'release\tauri\.bundle-src'
foreach($entry in @('app','settings','runtimes')) {
  $source = Join-Path $bundleSource $entry
  if(-not(Test-Path $source)){throw "Portable resource missing: $source"}
  $destination = Join-Path $stage $entry
  New-Item $destination -ItemType Directory -Force | Out-Null
  & robocopy.exe $source $destination /E /NFL /NDL /NJH /NJS /NC /NS | Out-Null
  if ($LASTEXITCODE -ge 8) { throw "Failed to copy portable resource tree: $entry (robocopy exit $LASTEXITCODE)" }
}

$portable = Join-Path $output "$baseName-portable.zip"
$installer = Join-Path $output "$baseName.msi"
Write-Host '[STEP] Creating portable ZIP...'
Push-Location $stage
try {
  & tar.exe -a -c -f "..\$baseName-portable.zip" .
  if ($LASTEXITCODE) { throw "Portable archive creation failed with exit code $LASTEXITCODE" }
} finally { Pop-Location }
Write-Host '[STEP] Copying MSI...'
Copy-Item $msi.FullName $installer
Write-Host '[STEP] Removing portable staging directory...'
Remove-Item $stage -Recurse -Force

$forbidden = @('*.db','*.log','*.pfx','*.pem','*.key') | ForEach-Object { Get-ChildItem $output -Recurse -File -Filter $_ }
if ($forbidden) { throw "Forbidden release files found: $($forbidden.FullName -join ', ')" }
$sha256 = [Security.Cryptography.SHA256]::Create()
Write-Host '[STEP] Generating SHA-256 checksums...'
try {
  @($portable,$installer) | ForEach-Object {
    $stream = [IO.File]::OpenRead($_)
    try { $hash = ([BitConverter]::ToString($sha256.ComputeHash($stream))).Replace('-','') } finally { $stream.Dispose() }
    "$hash  $([IO.Path]::GetFileName($_))"
  } | Set-Content (Join-Path $output "$baseName-SHA256SUMS.txt") -Encoding ascii
} finally { $sha256.Dispose() }
Write-Host "[OK] $portable"
Write-Host "[OK] $installer"
