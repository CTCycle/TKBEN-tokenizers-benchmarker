[CmdletBinding()]
param(
  [string]$OutputPath = ""
)

$ErrorActionPreference = "Stop"

$repoRoot = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot "..\..\.."))
$appDir = Join-Path $repoRoot "app"
$tauriConfigPath = Join-Path $appDir "src-tauri\tauri.conf.json"
$releaseDir = Join-Path $appDir "src-tauri\target\release"
$bundleDir = Join-Path $releaseDir "bundle"
$appVersion = (Get-Content -Path $tauriConfigPath -Raw | ConvertFrom-Json).version

if ([string]::IsNullOrWhiteSpace($OutputPath)) {
  $outputDir = Join-Path $repoRoot "release\windows"
} else {
  $outputDir = [System.IO.Path]::GetFullPath((Join-Path $repoRoot $OutputPath))
}

$installersDir = Join-Path $outputDir "installers"
$portableDir = Join-Path $outputDir "portable"
$portableRuntimeDir = Join-Path $portableDir "runtime"

if (-not (Test-Path $bundleDir)) {
  throw "Bundle directory not found. Run 'npm run tauri:build' first. Missing: $bundleDir"
}

if (Test-Path $outputDir) {
  Remove-Item -Recurse -Force $outputDir
}

New-Item -ItemType Directory -Path $installersDir -Force | Out-Null
New-Item -ItemType Directory -Path $portableDir -Force | Out-Null
New-Item -ItemType Directory -Path $portableRuntimeDir -Force | Out-Null

$installerArtifacts = @()

$nsisDir = Join-Path $bundleDir "nsis"
if (Test-Path $nsisDir) {
  $nsisFiles = Get-ChildItem -Path $nsisDir -Filter "*.exe" -File |
    Where-Object { $_.Name -like "*_$appVersion-*" -or $_.Name -like "*_$appVersion_*" }
  foreach ($file in $nsisFiles) {
    Copy-Item -Path $file.FullName -Destination $installersDir -Force
    $installerArtifacts += Join-Path $installersDir $file.Name
  }
}

$msiDir = Join-Path $bundleDir "msi"
if (Test-Path $msiDir) {
  $msiFiles = Get-ChildItem -Path $msiDir -Filter "*.msi" -File |
    Where-Object { $_.Name -like "*_$appVersion_*" }
  foreach ($file in $msiFiles) {
    Copy-Item -Path $file.FullName -Destination $installersDir -Force
    $installerArtifacts += Join-Path $installersDir $file.Name
  }
}

$portableExeCandidates = Get-ChildItem -Path $releaseDir -Filter "*.exe" -File |
  Where-Object { $_.Name -notmatch "(?i)(setup|installer|uninstall|updater)" }

foreach ($file in $portableExeCandidates) {
  Copy-Item -Path $file.FullName -Destination $portableDir -Force
}

$portableRequiredEntries = @(
  "app",
  "settings",
  "runtimes"
)

$portableOptionalEntries = @(
  "resources"
)

foreach ($entry in $portableRequiredEntries) {
  $sourcePath = Join-Path $releaseDir $entry
  if (-not (Test-Path $sourcePath)) {
    throw "Required portable payload is missing from release output: $sourcePath"
  }
  $destinationPath = Join-Path $portableRuntimeDir $entry
  Copy-Item -Path $sourcePath -Destination $destinationPath -Recurse -Force
}

foreach ($entry in $portableOptionalEntries) {
  $sourcePath = Join-Path $releaseDir $entry
  if (Test-Path $sourcePath) {
    $destinationPath = Join-Path $portableRuntimeDir $entry
    Copy-Item -Path $sourcePath -Destination $destinationPath -Recurse -Force
  }
}

$portableUvExe = Join-Path $portableRuntimeDir "runtimes\uv\uv.exe"
$portablePythonExe = Join-Path $portableRuntimeDir "runtimes\python\python.exe"
if (-not (Test-Path $portableUvExe)) {
  throw "Portable export is missing bundled uv runtime: $portableUvExe"
}
if (-not (Test-Path $portablePythonExe)) {
  throw "Portable export is missing bundled Python runtime: $portablePythonExe"
}

$instructions = @"
TKBEN desktop build output

1) Preferred for users:
   Open installers\ and run the setup executable (.exe) or .msi.

2) Portable executable:
   portable\ contains the app .exe and a sibling runtime\ folder.
   Keep the .exe and runtime\ folder together.

Generated from:
$bundleDir
"@
Set-Content -Path (Join-Path $outputDir "README.txt") -Value $instructions -Encoding ascii

Write-Host "[OK] Exported Windows artifacts to: $outputDir"
Write-Host "[INFO] Installers:"
if ($installerArtifacts.Count -eq 0) {
  Write-Host " - none found"
} else {
  $installerArtifacts | ForEach-Object { Write-Host " - $_" }
}
Write-Host "[INFO] Portable executables:"
$portableFiles = Get-ChildItem -Path $portableDir -Filter "*.exe" -File
if ($portableFiles.Count -eq 0) {
  Write-Host " - none found"
} else {
  $portableFiles | ForEach-Object { Write-Host " - $($_.FullName)" }
}
