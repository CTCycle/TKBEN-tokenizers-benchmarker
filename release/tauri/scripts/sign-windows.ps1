[CmdletBinding()]
param([Parameter(Mandatory=$true)][string]$FilePath)

$ErrorActionPreference = 'Stop'
if (-not $env:TKBEN_SIGNING_CERTIFICATE_PATH) {
  Write-Host "[SIGN] No certificate configured; leaving unsigned: $FilePath"
  exit 0
}
if (-not (Test-Path $env:TKBEN_SIGNING_CERTIFICATE_PATH)) { throw 'Configured signing certificate does not exist.' }
$signtool = Get-ChildItem "${env:ProgramFiles(x86)}\Windows Kits\10\bin" -Recurse -Filter signtool.exe -ErrorAction SilentlyContinue | Where-Object FullName -Match '\\x64\\' | Sort-Object FullName -Descending | Select-Object -First 1
if (-not $signtool) { throw 'signtool.exe was not found in the Windows SDK.' }
& $signtool.FullName sign /fd SHA256 /td SHA256 /tr http://timestamp.digicert.com /f $env:TKBEN_SIGNING_CERTIFICATE_PATH /p $env:TKBEN_SIGNING_CERTIFICATE_PASSWORD $FilePath
if ($LASTEXITCODE) { throw "signtool failed with exit code $LASTEXITCODE" }
