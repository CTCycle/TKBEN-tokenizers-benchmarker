[CmdletBinding()]
param([Parameter(Mandatory=$true)][string]$MsiPath)

$ErrorActionPreference = 'Stop'
$msi = [IO.Path]::GetFullPath($MsiPath)
$installLog = Join-Path $env:TEMP "tkben-msi-install-$PID.log"
$uninstallLog = Join-Path $env:TEMP "tkben-msi-uninstall-$PID.log"
$process = $null
if (-not (Test-Path $msi)) { throw "MSI not found: $msi" }
try {
  $install = Start-Process msiexec.exe -ArgumentList @('/i',"`"$msi`"",'/qn','/norestart','/l*v',"`"$installLog`"") -Wait -PassThru
  if ($install.ExitCode -ne 0) { throw "MSI install failed with code $($install.ExitCode). Log: $installLog" }
  $exe = Get-ChildItem $env:ProgramFiles,$env:LOCALAPPDATA -Recurse -File -Filter 'TKBEN Desktop.exe' -ErrorAction SilentlyContinue | Select-Object -First 1
  if (-not $exe) { throw 'Installed TKBEN executable was not found.' }
  $process = Start-Process $exe.FullName -PassThru
  Start-Sleep -Seconds 5
  if ($process.HasExited) { throw "Installed application exited early with code $($process.ExitCode)." }
  Write-Host "[OK] MSI installed and launched: $($exe.FullName)"
} finally {
  if ($process -and -not $process.HasExited) { & taskkill.exe /PID $process.Id /T /F | Out-Null }
  $uninstall = Start-Process msiexec.exe -ArgumentList @('/x',"`"$msi`"",'/qn','/norestart','/l*v',"`"$uninstallLog`"") -Wait -PassThru
  if ($uninstall.ExitCode -ne 0 -and $uninstall.ExitCode -ne 1605) { throw "MSI uninstall failed with code $($uninstall.ExitCode). Log: $uninstallLog" }
  Write-Host '[OK] MSI uninstall completed.'
}
