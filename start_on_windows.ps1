[CmdletBinding()]
param(
    [switch]$Launch
)

$ErrorActionPreference = 'Stop'
# Progress bars are reserved for quiet, measurable work; console logs clear them first.
$ProgressPreference = 'Continue'

$RepoRoot = if ($PSScriptRoot) {
    [IO.Path]::GetFullPath($PSScriptRoot)
} elseif ($PSCommandPath) {
    [IO.Path]::GetFullPath((Split-Path -Parent $PSCommandPath))
} else {
    [IO.Path]::GetFullPath((Get-Location).Path)
}
$RuntimeDir = Join-Path $RepoRoot 'runtimes'
$PythonDir = Join-Path $RuntimeDir 'python'
$UvDir = Join-Path $RuntimeDir 'uv'
$NodeDir = Join-Path $RuntimeDir 'nodejs'
$PythonExe = Join-Path $PythonDir 'python.exe'
$PythonPth = Join-Path $PythonDir 'python314._pth'
$UvExe = Join-Path $UvDir 'uv.exe'
$NodeExe = Join-Path $NodeDir 'node.exe'
$NpmCmd = Join-Path $NodeDir 'npm.cmd'
$ServerDir = Join-Path $RepoRoot 'app\server'
$ClientDir = Join-Path $RepoRoot 'app\client'
$AppDir = Join-Path $RepoRoot 'app'
$AssetsDir = Join-Path $RepoRoot 'assets'
$SettingsDir = Join-Path $RepoRoot 'settings'
$TestsDir = Join-Path $AppDir 'tests'
$VenvDir = Join-Path $ServerDir '.venv'
$VenvPython = Join-Path $VenvDir 'Scripts\python.exe'
$EnvFile = Join-Path $RepoRoot 'settings\.env'
$EnvTemplate = Join-Path $RepoRoot 'settings\.env.example'
$RuntimeCacheDir = Join-Path $RuntimeDir 'cache'
$ToolCacheDir = Join-Path $TestsDir 'cache'
$UvCacheDir = Join-Path $RuntimeCacheDir 'uv'
$PythonVersion = '3.14.2'
$NodeVersion = '22.23.1'
$script:NextProgressId = 1
$script:ActiveProgressActivities = [Collections.Generic.Dictionary[int, string]]::new()
$script:LauncherProgressEnabled = -not [Console]::IsOutputRedirected
$script:LauncherInteractive = -not [Console]::IsInputRedirected -and -not [Console]::IsOutputRedirected

# =============================================================================
# Shared output and filesystem helpers
# =============================================================================
function Write-Step([string]$Message) { Clear-LauncherProgress; Write-Host "[STEP] $Message" -ForegroundColor Cyan }
function Write-Ok([string]$Message) { Clear-LauncherProgress; Write-Host "[OK] $Message" -ForegroundColor Green }
function Write-Fatal([string]$Message) { Clear-LauncherProgress; Write-Host "[FATAL] $Message" -ForegroundColor Red }

function Start-LauncherProgress {
    param([Parameter(Mandatory)][string]$Activity, [Parameter(Mandatory)][string]$Status)
    $id = $script:NextProgressId++
    $script:ActiveProgressActivities[$id] = $Activity
    if ($script:LauncherProgressEnabled) { Write-Progress -Id $id -Activity $Activity -Status $Status }
    return $id
}

function Update-LauncherProgress {
    param(
        [Parameter(Mandatory)][int]$Id,
        [Parameter(Mandatory)][string]$Activity,
        [Parameter(Mandatory)][string]$Status,
        [Nullable[int]]$PercentComplete
    )
    if (-not $script:ActiveProgressActivities.ContainsKey($Id)) { return }
    $activity = $script:ActiveProgressActivities[$Id]
    $progress = @{ Id = $Id; Activity = $activity; Status = $Status }
    if ($null -ne $PercentComplete) { $progress.PercentComplete = $PercentComplete }
    if ($script:LauncherProgressEnabled) { Write-Progress @progress }
}

function Complete-LauncherProgress([int]$Id) {
    if ($script:ActiveProgressActivities.ContainsKey($Id)) {
        $activity = $script:ActiveProgressActivities[$Id]
        try {
            if ($script:LauncherProgressEnabled) {
                try { Write-Progress -Id $Id -Activity $activity -Completed } catch { }
            }
        }
        finally {
            [void]$script:ActiveProgressActivities.Remove($Id)
        }
    }
}

function Clear-LauncherProgress {
    foreach ($id in @($script:ActiveProgressActivities.Keys)) {
        Complete-LauncherProgress -Id $id
    }
}

function Invoke-TrackedLauncherAction {
    param(
        [Parameter(Mandatory)][string]$Name,
        [Parameter(Mandatory)][scriptblock]$Action
    )
    Write-Step "Starting $Name"
    try {
        & $Action
        Write-Ok "$Name completed"
    }
    catch {
        Write-Fatal "$Name failed: $($_.Exception.Message)"
        throw
    }
    finally {
        Clear-LauncherProgress
    }
}

function Ensure-Directory([string]$Path) {
    [IO.Directory]::CreateDirectory($Path) | Out-Null
}

function Initialize-EnvFile {
    if (-not (Test-Path -LiteralPath $EnvFile)) {
        if (-not (Test-Path -LiteralPath $EnvTemplate)) {
            throw "Missing environment template: $EnvTemplate"
        }
        Copy-Item -LiteralPath $EnvTemplate -Destination $EnvFile
        Write-Ok "Created settings\.env from settings\.env.example."
    }
}

function Read-EnvironmentFile {
    $values = @{}
    if (-not (Test-Path -LiteralPath $EnvFile)) { return $values }

    foreach ($rawLine in Get-Content -LiteralPath $EnvFile) {
        $line = $rawLine.Trim()
        if (-not $line -or $line.StartsWith('#') -or $line.StartsWith(';') -or -not $line.Contains('=')) {
            continue
        }
        $key, $value = $line.Split('=', 2)
        $key = $key.Trim()
        $value = $value.Trim()
        if (($value.StartsWith('"') -and $value.EndsWith('"')) -or ($value.StartsWith("'") -and $value.EndsWith("'"))) {
            $value = $value.Substring(1, $value.Length - 2)
        }
        if ($key) { $values[$key] = $value }
    }
    return $values
}

function Get-EnvironmentSetting {
    param(
        [Parameter(Mandatory)][string]$Key,
        [string]$DefaultValue = ''
    )

    $configuredValues = Read-EnvironmentFile
    if ($configuredValues.ContainsKey($Key)) { return [string]$configuredValues[$Key] }

    $processValue = [Environment]::GetEnvironmentVariable($Key)
    if (-not [string]::IsNullOrWhiteSpace($processValue)) { return $processValue }
    return $DefaultValue
}

function Resolve-ConfiguredPath {
    param(
        [string]$ConfiguredPath,
        [Parameter(Mandatory)][string]$DefaultPath
    )

    if ([string]::IsNullOrWhiteSpace($ConfiguredPath)) {
        return [IO.Path]::GetFullPath($DefaultPath)
    }

    $expandedPath = [Environment]::ExpandEnvironmentVariables($ConfiguredPath.Trim())
    if ([IO.Path]::IsPathRooted($expandedPath)) {
        return [IO.Path]::GetFullPath($expandedPath)
    }
    return [IO.Path]::GetFullPath((Join-Path $RepoRoot $expandedPath))
}

function Get-ApplicationDataRoot {
    return Resolve-ConfiguredPath `
        -ConfiguredPath (Get-EnvironmentSetting -Key 'TKBEN_DATA_DIR') `
        -DefaultPath (Join-Path $AppDir 'resources')
}

function Get-ApplicationLogRoot {
    param([Parameter(Mandatory)][string]$DataRoot)

    return Resolve-ConfiguredPath `
        -ConfiguredPath (Get-EnvironmentSetting -Key 'TKBEN_LOG_DIR') `
        -DefaultPath (Join-Path $DataRoot 'logs')
}

function Get-HuggingFaceMaterialPath {
    param([Parameter(Mandatory)][string]$DataRoot)

    return Resolve-ConfiguredPath `
        -ConfiguredPath (Get-EnvironmentSetting -Key 'HF_KEYS_ENCRYPTION_MATERIAL_FILE') `
        -DefaultPath (Join-Path $DataRoot 'hf-key-material.json')
}

function Invoke-DownloadAndExtract {
    param(
        [Parameter(Mandatory)][uri]$Uri,
        [Parameter(Mandatory)][string]$ArchivePath,
        [Parameter(Mandatory)][string]$Destination
    )
    $activity = "TKBEN: download and extract $([IO.Path]::GetFileName($ArchivePath))"
    $progressId = Start-LauncherProgress -Activity $activity -Status "Downloading $Uri"
    try {
        [IO.Directory]::CreateDirectory((Split-Path -Parent $ArchivePath)) | Out-Null
        [IO.Directory]::CreateDirectory($Destination) | Out-Null
        Invoke-WebRequest -Uri $Uri -OutFile $ArchivePath
        Update-LauncherProgress -Id $progressId -Activity $activity -Status 'Extracting archive'
        Expand-Archive -LiteralPath $ArchivePath -DestinationPath $Destination -Force
    } finally {
        Remove-Item -LiteralPath $ArchivePath -Force -ErrorAction SilentlyContinue
        Complete-LauncherProgress $progressId
    }
}

# =============================================================================
# Environment, runtimes, and dependency management
# =============================================================================
function Install-NodeRuntime {
    $stagingDir = Join-Path $RuntimeDir ('.nodejs-staging-' + [guid]::NewGuid().ToString('N'))
    $backupDir = Join-Path $RuntimeDir ('.nodejs-backup-' + [guid]::NewGuid().ToString('N'))
    $oldRuntimeMoved = $false
    $newRuntimeInstalled = $false

    try {
        Write-Step "Downloading Node.js $NodeVersion (portable x64)."
        $nodeArchive = Join-Path $stagingDir "node-v$NodeVersion-win-x64.zip"
        Invoke-DownloadAndExtract `
            -Uri "https://nodejs.org/dist/v$NodeVersion/node-v$NodeVersion-win-x64.zip" `
            -ArchivePath $nodeArchive `
            -Destination $stagingDir

        $nestedNodeDir = Join-Path $stagingDir "node-v$NodeVersion-win-x64"
        if (-not (Test-Path -LiteralPath (Join-Path $nestedNodeDir 'node.exe'))) {
            throw "Node.js was not found in the extracted archive at $nestedNodeDir"
        }

        if (Test-Path -LiteralPath $NodeDir) {
            Move-Item -LiteralPath $NodeDir -Destination $backupDir -ErrorAction Stop
            $oldRuntimeMoved = $true
        }
        Move-Item -LiteralPath $nestedNodeDir -Destination $NodeDir -ErrorAction Stop
        $newRuntimeInstalled = $true

        if (Test-Path -LiteralPath $backupDir) {
            [void](Remove-LauncherPath -Path $backupDir -Activity 'TKBEN: remove Node.js backup runtime' -Strict)
        }
    } catch {
        if ($newRuntimeInstalled -and (Test-Path -LiteralPath $NodeDir)) {
            [void](Remove-LauncherPath -Path $NodeDir -Activity 'TKBEN: roll back Node.js runtime')
        }
        if ($oldRuntimeMoved -and (Test-Path -LiteralPath $backupDir) -and -not (Test-Path -LiteralPath $NodeDir)) {
            Move-Item -LiteralPath $backupDir -Destination $NodeDir -ErrorAction SilentlyContinue
        }
        throw
    } finally {
        if (Test-Path -LiteralPath $stagingDir) {
            [void](Remove-LauncherPath -Path $stagingDir -Activity 'TKBEN: remove runtime staging directory')
        }
    }
}

function Invoke-PatchPth {
    param([Parameter(Mandatory)][string]$Path)
    if (-not (Test-Path -LiteralPath $Path)) { throw "Missing Python path file: $Path" }
    (Get-Content -LiteralPath $Path) -replace '^#import site$', 'import site' |
        Set-Content -LiteralPath $Path -Encoding ascii
}

function Invoke-CheckPyVer {
    param([Parameter(Mandatory)][string]$PythonExe)
    & $PythonExe -c 'import platform; print(platform.python_version())'
    if ($LASTEXITCODE -ne 0) { throw "Python version check failed with exit code $LASTEXITCODE." }
}

function Invoke-Npm {
    param([Parameter(ValueFromRemainingArguments)][string[]]$Arguments)
    if (-not (Test-Path -LiteralPath $NpmCmd)) { throw "npm was not installed at $NpmCmd" }
    $commandLine = '"' + $NpmCmd + '"'
    if ($Arguments) { $commandLine += ' ' + ($Arguments -join ' ') }
    & cmd.exe /d /c $commandLine | Out-Host
    return [int]$LASTEXITCODE
}

function Get-LogTail {
    param(
        [Parameter(Mandatory)][string]$Path,
        [ValidateRange(1, 100)][int]$Lines = 12
    )
    if (-not (Test-Path -LiteralPath $Path)) { return $null }
    $content = @(Get-Content -LiteralPath $Path -Tail $Lines -ErrorAction SilentlyContinue)
    if (-not $content) { return $null }
    return ($content -join [Environment]::NewLine)
}

function Invoke-FindUv {
    param([Parameter(Mandatory)][string]$SearchRoot)
    $match = Get-ChildItem -LiteralPath $SearchRoot -Recurse -Filter 'uv.exe' -File | Select-Object -First 1
    if ($match) { $match.FullName }
}

function Invoke-HealthCheck {
    param(
        [Parameter(Mandatory)][uri]$Uri,
        [ValidateRange(1, 3600)][int]$Attempts = 60,
        [ValidateRange(1, 60)][int]$IntervalSeconds = 1,
        [string]$Description = 'service',
        [System.Diagnostics.Process]$ProcessToMonitor,
        [string]$FailureLogPath
    )
    for ($attempt = 1; $attempt -le $Attempts; $attempt++) {
        try {
            $response = Invoke-WebRequest -UseBasicParsing -Uri $Uri -TimeoutSec 2
            if ($response.StatusCode -ge 200 -and $response.StatusCode -lt 400) { return }
        } catch {
            if ($attempt -eq $Attempts) { break }
        }

        if ($ProcessToMonitor) {
            try {
                $ProcessToMonitor.Refresh()
                if ($ProcessToMonitor.HasExited) {
                    $details = if ($FailureLogPath) { Get-LogTail -Path $FailureLogPath } else { $null }
                    $message = "$Description exited with code $($ProcessToMonitor.ExitCode) while waiting for $Uri."
                    if ($details) { $message += " Output: $details" }
                    throw $message
                }
            } catch [InvalidOperationException] {
                # The process may exit between Refresh and HasExited; the next
                # request or timeout still provides the final readiness result.
            }
        }

        if ($attempt -eq 1 -or ($attempt % 10 -eq 0)) {
            Write-Host "[WAIT] Waiting for $Description at $Uri ($attempt/$Attempts)." -ForegroundColor DarkGray
        }
        Start-Sleep -Seconds $IntervalSeconds
    }
    throw "Timed out waiting for $Uri after $Attempts attempts."
}

function Import-Environment {
    Initialize-EnvFile

    $defaults = @{
        FASTAPI_HOST = '127.0.0.1'
        FASTAPI_PORT = '5000'
        UI_HOST = '127.0.0.1'
        UI_PORT = '8000'
        RELOAD = 'false'
        # Backend logs are visible by default when the setting is absent.
        BACKEND_LOGS_VISIBLE = 'true'
    }
    foreach ($entry in $defaults.GetEnumerator()) {
        Set-Item -Path "Env:$($entry.Key)" -Value $entry.Value
    }

    foreach ($entry in (Read-EnvironmentFile).GetEnumerator()) {
        Set-Item -Path "Env:$($entry.Key)" -Value $entry.Value
    }

    if ($env:BACKEND_LOGS_VISIBLE -ieq 'true') {
        $env:BACKEND_LOGS_VISIBLE = 'true'
    } elseif ($env:BACKEND_LOGS_VISIBLE -ieq 'false') {
        $env:BACKEND_LOGS_VISIBLE = 'false'
    } else {
        throw "BACKEND_LOGS_VISIBLE must be either 'true' or 'false'."
    }

    Ensure-Directory $RuntimeCacheDir
    foreach ($cacheName in @('uv', 'pip', 'npm')) {
        Ensure-Directory (Join-Path $RuntimeCacheDir $cacheName)
    }
    Ensure-Directory $ToolCacheDir
    foreach ($cacheName in @('ruff', 'mypy', 'pycache', 'coverage', 'playwright', 'pytest', 'pytest-basetemp', 'angular')) {
        Ensure-Directory (Join-Path $ToolCacheDir $cacheName)
    }
    $env:UV_CACHE_DIR = $UvCacheDir
    $env:PIP_CACHE_DIR = Join-Path $RuntimeCacheDir 'pip'
    $env:NPM_CONFIG_CACHE = Join-Path $RuntimeCacheDir 'npm'
    $env:RUFF_CACHE_DIR = Join-Path $ToolCacheDir 'ruff'
    $env:MYPY_CACHE_DIR = Join-Path $ToolCacheDir 'mypy'
    $env:PYTHONPYCACHEPREFIX = Join-Path $ToolCacheDir 'pycache'
    $env:COVERAGE_FILE = Join-Path (Join-Path $ToolCacheDir 'coverage') '.coverage'
    $env:PLAYWRIGHT_BROWSERS_PATH = Join-Path $ToolCacheDir 'playwright'
    $env:UV_PROJECT_ENVIRONMENT = $VenvDir
    $env:UV_LINK_MODE = 'copy'
    Remove-Item Env:PYTHONHOME -ErrorAction SilentlyContinue
    Remove-Item Env:PYTHONPATH -ErrorAction SilentlyContinue
    Remove-Item Env:PYTHONNOUSERSITE -ErrorAction SilentlyContinue
    $env:PATH = "$NodeDir;$env:PATH"
}

function Install-Runtimes {
    Write-Step 'Checking portable runtimes.'
    Ensure-Directory $RuntimeDir
    Ensure-Directory $PythonDir
    Ensure-Directory $UvDir
    Ensure-Directory $NodeDir

    if (-not (Test-Path -LiteralPath $PythonExe)) {
        Write-Step "Downloading Python $PythonVersion (embeddable x64)."
        Invoke-DownloadAndExtract `
            -Uri "https://www.python.org/ftp/python/$PythonVersion/python-$PythonVersion-embed-amd64.zip" `
            -ArchivePath (Join-Path $PythonDir "python-$PythonVersion-embed-amd64.zip") `
            -Destination $PythonDir
    }
    if (-not (Test-Path -LiteralPath $PythonExe)) { throw "Python was not installed at $PythonExe" }
    Invoke-PatchPth -Path $PythonPth
    $detectedPython = Invoke-CheckPyVer -PythonExe $PythonExe
    Write-Ok "Python ready: $detectedPython"

    if (-not (Test-Path -LiteralPath $UvExe)) {
        $uvArchive = if ($env:PROCESSOR_ARCHITECTURE -eq 'ARM64') {
            'uv-aarch64-pc-windows-msvc.zip'
        } else {
            'uv-x86_64-pc-windows-msvc.zip'
        }
        Write-Step 'Downloading uv (portable).'
        Invoke-DownloadAndExtract `
            -Uri "https://github.com/astral-sh/uv/releases/latest/download/$uvArchive" `
            -ArchivePath (Join-Path $UvDir 'uv.zip') `
            -Destination $UvDir
        $foundUv = Invoke-FindUv -SearchRoot $UvDir
        if (-not $foundUv) { throw 'uv.exe was not found after extraction.' }
        if ([IO.Path]::GetFullPath($foundUv) -ne [IO.Path]::GetFullPath($UvExe)) {
            Copy-Item -LiteralPath $foundUv -Destination $UvExe -Force
        }
    }
    Write-Ok (& $UvExe --version)

    $nodeNeedsInstall = -not (Test-Path -LiteralPath $NodeExe)
    if (-not $nodeNeedsInstall) {
        $installedNodeVersion = (& $NodeExe --version).Trim().TrimStart('v')
        $nodeNeedsInstall = $installedNodeVersion -ne $NodeVersion
        if ($nodeNeedsInstall) {
            Write-Step "Replacing incompatible Node.js $installedNodeVersion with $NodeVersion."
            Stop-PortListeners -Port ([int]$env:UI_PORT)
        }
    }
    if ($nodeNeedsInstall) {
        Install-NodeRuntime
    }
    if (-not (Test-Path -LiteralPath $NodeExe)) { throw "Node.js was not installed at $NodeExe" }
    if (-not (Test-Path -LiteralPath $NpmCmd)) { throw "npm was not installed at $NpmCmd" }
    Write-Ok "Node.js ready: $(& $NodeExe --version)"
    Write-Ok 'Portable runtimes ready.'
}

function Get-FrontendDependencyFingerprint {
    $manifestPaths = @(
        (Join-Path $ClientDir 'package.json'),
        (Join-Path $ClientDir 'package-lock.json')
    )

    if ($manifestPaths | Where-Object { -not (Test-Path -LiteralPath $_) }) {
        throw 'Frontend package.json and package-lock.json are required.'
    }
    return (($manifestPaths | ForEach-Object { (Get-FileHash -LiteralPath $_ -Algorithm SHA256).Hash }) -join ':')
}

function Test-FrontendDependenciesReady {
    $nodeModulesDir = Join-Path $ClientDir 'node_modules'
    $stampPath = Join-Path $nodeModulesDir '.tkben-dependencies.json'
    $npmLockPath = Join-Path $nodeModulesDir '.package-lock.json'

    if (-not (Test-Path -LiteralPath $stampPath) -or -not (Test-Path -LiteralPath $npmLockPath)) {
        return $false
    }

    try {
        $stamp = Get-Content -LiteralPath $stampPath -Raw | ConvertFrom-Json
        return (
            $stamp.packageFingerprint -eq (Get-FrontendDependencyFingerprint) -and
            $stamp.nodeVersion -eq (& $NodeExe --version).Trim() -and
            (Test-Path -LiteralPath (Join-Path $nodeModulesDir '.bin\ng.cmd'))
        )
    } catch {
        return $false
    }
}

function Write-FrontendDependencyStamp {
    $stampPath = Join-Path $ClientDir 'node_modules\.tkben-dependencies.json'
    [ordered]@{
        packageFingerprint = Get-FrontendDependencyFingerprint
        nodeVersion = (& $NodeExe --version).Trim()
    } | ConvertTo-Json | Set-Content -LiteralPath $stampPath -Encoding utf8
}

function Sync-Dependencies {
    param(
        [switch]$BuildFrontend,
        [switch]$UseCachedFrontendDependencies,
        [switch]$RuntimesReady,
        [ValidateSet('Standard', 'Development')]
        [string]$InstallationType = 'Standard'
    )

    Import-Environment
    if (-not $RuntimesReady) { Install-Runtimes }
    Write-Step 'Installing Python dependencies.'
    $uvArguments = @('sync', '--python', $PythonExe)
    if ($InstallationType -eq 'Development') { $uvArguments += '--all-extras' }
    Push-Location $ServerDir
    try {
        $uvExitCode = 1
        for ($attempt = 1; $attempt -le 2; $attempt++) {
            & $UvExe @uvArguments
            $uvExitCode = $LASTEXITCODE
            if ($uvExitCode -eq 0) { break }
            if ($attempt -eq 1) {
                Write-Step 'uv sync failed; clearing the managed uv cache and retrying once.'
                if (Test-Path -LiteralPath $UvCacheDir) {
                    Remove-PathBestEffort -Path $UvCacheDir | Out-Null
                }
            }
        }
        if ($uvExitCode -ne 0) { throw "uv sync failed with exit code $uvExitCode." }
    } finally {
        Pop-Location
    }

    Sync-Frontend -BuildFrontend:$BuildFrontend -UseCachedFrontendDependencies:$UseCachedFrontendDependencies
}

function Sync-Frontend {
    param(
        [switch]$BuildFrontend,
        [switch]$UseCachedFrontendDependencies
    )

    Stop-PortListeners -Port ([int]$env:UI_PORT)
    Push-Location $ClientDir
    try {
        $frontendInstallRequired = -not $UseCachedFrontendDependencies -or -not (Test-FrontendDependenciesReady)
        if ($frontendInstallRequired) {
            Write-Step 'Installing frontend dependencies.'
            if (-not (Test-Path -LiteralPath (Join-Path $ClientDir 'package-lock.json'))) {
                throw 'Frontend package-lock.json is required; refusing an unlocked install.'
            }
            $npmExitCode = Invoke-Npm ci
            if ($npmExitCode -ne 0) { throw "npm dependency installation failed with exit code $npmExitCode." }
            Write-FrontendDependencyStamp
        } else {
            Write-Ok 'Frontend dependencies are unchanged; skipped clean install.'
        }

        if ($BuildFrontend) {
            Write-Step 'Building frontend.'
            $npmExitCode = Invoke-Npm run build
            if ($npmExitCode -ne 0) { throw "Frontend build failed with exit code $npmExitCode." }
        }
    } finally {
        Pop-Location
    }
}

function Test-DependenciesReady {
    $frontendPackage = Join-Path $ClientDir 'package.json'
    $frontendLock = Join-Path $ClientDir 'package-lock.json'
    $frontendModules = Join-Path $ClientDir 'node_modules'
    $frontendRunner = Join-Path $frontendModules '.bin\ng.cmd'
    $backendEntrypoint = Join-Path $AppDir 'server/app.py'

    if (-not (Test-Path -LiteralPath $PythonExe) -or
        -not (Test-Path -LiteralPath $UvExe) -or
        -not (Test-Path -LiteralPath $NodeExe) -or
        -not (Test-Path -LiteralPath $NpmCmd) -or
        -not (Test-Path -LiteralPath $VenvPython) -or
        -not (Test-Path -LiteralPath $backendEntrypoint) -or
        -not (Test-Path -LiteralPath $frontendPackage) -or
        -not (Test-Path -LiteralPath $frontendLock) -or
        -not (Test-Path -LiteralPath (Join-Path $frontendModules '.package-lock.json')) -or
        -not (Test-Path -LiteralPath $frontendRunner)) {
        return $false
    }

    & $PythonExe --version *> $null
    if ($LASTEXITCODE -ne 0) { return $false }
    & $UvExe --version *> $null
    if ($LASTEXITCODE -ne 0) { return $false }
    & $NodeExe --version *> $null
    if ($LASTEXITCODE -ne 0) { return $false }
    & $VenvPython -c 'import fastapi, uvicorn' *> $null
    if ($LASTEXITCODE -ne 0) { return $false }

    return $true
}

function Stop-PortListeners([int]$Port) {
    $listeners = netstat -ano | Select-String -Pattern ":$Port\s+.*LISTENING\s+(\d+)\s*$"
    $processIds = @($listeners | ForEach-Object {
        if ($_.Matches.Count) { [int]$_.Matches[0].Groups[1].Value }
    } | Sort-Object -Unique)
    foreach ($processId in $processIds) {
        Write-Step "Stopping PID $processId on port $Port."
        & taskkill.exe /PID $processId /T /F | Out-Null
    }
}

function Get-PortProcessId([int]$Port) {
    $listener = netstat -ano | Select-String -Pattern ":$Port\s+.*LISTENING\s+(\d+)\s*$" | Select-Object -First 1
    if ($listener -and $listener.Matches.Count) { return [int]$listener.Matches[0].Groups[1].Value }
    return $null
}

# =============================================================================
# Application lifecycle and validation
# =============================================================================
function Launch-Application {
    Import-Environment
    if (-not (Test-DependenciesReady)) {
        Write-Step 'Required application environments are missing or unusable; installing dependencies.'
        Sync-Dependencies -InstallationType 'Standard'
    }
    else {
        Write-Ok 'Application environments are ready; skipped dependency installation.'
    }
    Import-Environment

    $backendPort = [int]$env:FASTAPI_PORT
    $uiPort = [int]$env:UI_PORT
    Stop-PortListeners -Port $backendPort
    Stop-PortListeners -Port $uiPort

    $backendAppPath = Join-Path $RepoRoot 'app'
    $backendArgs = "-m uvicorn server.app:app --app-dir `"$backendAppPath`" --host `"$($env:FASTAPI_HOST)`" --port $backendPort"
    if ($env:RELOAD -ieq 'true') { $backendArgs += ' --reload' }

    Write-Step 'Starting backend.'
    if ($env:BACKEND_LOGS_VISIBLE -ieq 'true') {
        $escapedPython = $VenvPython.Replace("'", "''")
        $escapedApp = $backendAppPath.Replace("'", "''")
        $backendCommand = "& '$escapedPython' -m uvicorn server.app:app --app-dir '$escapedApp' --host $($env:FASTAPI_HOST) --port $backendPort"
        if ($env:RELOAD -ieq 'true') { $backendCommand += ' --reload' }
        $backendProcess = Start-Process -FilePath 'powershell.exe' `
            -ArgumentList @('-NoProfile', '-NoExit', '-Command', $backendCommand) `
            -WorkingDirectory $RepoRoot -WindowStyle Normal -PassThru
    } else {
        $backendProcess = Start-Process -FilePath $VenvPython -ArgumentList $backendArgs -WorkingDirectory $RepoRoot -WindowStyle Hidden -PassThru
    }

    Invoke-HealthCheck `
        -Uri "http://$($env:FASTAPI_HOST):$backendPort/api/health" `
        -Description 'backend' `
        -Attempts 60 `
        -IntervalSeconds 1
    $backendPid = if ($backendProcess) { $backendProcess.Id } else { Get-PortProcessId -Port $backendPort }

    Write-Step 'Starting frontend preview.'
    $previewCommandLine = '"' + $NpmCmd + '" run preview -- --host ' + $env:UI_HOST + ' --port ' + $uiPort + ' --strictPort'
    $frontendLogDir = Join-Path $AppDir 'resources\logs'
    Ensure-Directory -Path $frontendLogDir
    $frontendLogStem = Join-Path $frontendLogDir ('TKBEN_frontend_' + (Get-Date -Format 'yyyyMMdd_HHmmss_fff'))
    $frontendStdoutLog = "$frontendLogStem.out.log"
    $frontendStderrLog = "$frontendLogStem.err.log"
    $frontendProcess = Start-Process -FilePath 'cmd.exe' `
        -ArgumentList @('/d', '/c', $previewCommandLine) `
        -WorkingDirectory $ClientDir -WindowStyle Hidden `
        -RedirectStandardOutput $frontendStdoutLog `
        -RedirectStandardError $frontendStderrLog `
        -PassThru
    Invoke-HealthCheck `
        -Uri "http://$($env:UI_HOST):$uiPort/" `
        -Description 'frontend preview' `
        -ProcessToMonitor $frontendProcess `
        -FailureLogPath $frontendStderrLog `
        -Attempts 60 `
        -IntervalSeconds 1

    $url = "http://$($env:UI_HOST):$uiPort"
    Write-Ok 'Application started successfully.'
    Write-Host "Backend: http://$($env:FASTAPI_HOST):$backendPort (PID $backendPid)"
    Write-Host "Frontend: $url (PID $($frontendProcess.Id))"
    try {
        Start-Process -FilePath $url -ErrorAction Stop | Out-Null
    } catch {
        Write-Host "[WARN] Could not open the browser automatically. Open $url manually." -ForegroundColor Yellow
    }
}

function Install-Dependencies {
    Import-Environment
    Install-Runtimes
    $installationType = Read-InstallationType
    Sync-Dependencies -BuildFrontend -InstallationType $installationType -RuntimesReady
    Invoke-DatabaseInitialization
    if (Test-Path -LiteralPath $UvCacheDir) { Remove-PathBestEffort -Path $UvCacheDir | Out-Null }
    Write-Ok 'Dependencies installed, frontend built, and database synchronized.'
}

function Rebuild-Frontend {
    Import-Environment
    Install-Runtimes
    Sync-Frontend -BuildFrontend -UseCachedFrontendDependencies
    Write-Ok 'Frontend rebuilt.'
}

function Read-InstallationType {
    Clear-LauncherProgress
    Write-Host '  [1] Development - include Ruff, Pyright, and pytest'
    Write-Host '  [2] Standard    - install runtime dependencies only'
    $selection = (Read-Host '  Select installation profile [1-2]').Trim()
    switch ($selection) {
        '1' { return 'Development' }
        '2' { return 'Standard' }
        default { throw 'Invalid installation profile. Enter 1 for Development or 2 for Standard.' }
    }
}

function Invoke-DatabaseInitialization {
    Write-Step 'Initializing database.'
    $env:PYTHONPATH = $AppDir
    & $UvExe run --project $ServerDir --python $PythonExe python (Join-Path $RepoRoot 'app\scripts\initialize_database.py')
    if ($LASTEXITCODE -ne 0) { throw "Database initialization failed with exit code $LASTEXITCODE." }
    Write-Ok 'Database synchronized with the latest Alembic revision.'
}

function Initialize-Database {
    Import-Environment
    Install-Runtimes
    Invoke-DatabaseInitialization
}

function Run-TestSuite {
    $testScript = Join-Path $RepoRoot 'app\tests\run_tests.bat'
    Write-Step "Running test suite: $testScript"
    & cmd.exe /c $testScript
    $testExitCode = $LASTEXITCODE
    if ($testExitCode -ne 0) { throw "Test suite failed with exit code $testExitCode." }
    Write-Ok "Test suite completed with exit code $testExitCode."
}

# =============================================================================
# Data, logs, cache, and installation maintenance
# =============================================================================
function Remove-Logs {
    $logDir = Join-Path $RepoRoot 'app\resources\logs'
    $logs = @(Get-ChildItem -LiteralPath $logDir -Filter '*.log' -File -ErrorAction SilentlyContinue |
        Sort-Object @{ Expression = { $_.FullName.ToUpperInvariant() }; Descending = $false })
    $summary = @($logs | ForEach-Object { Remove-PathBestEffort -Path $_.FullName })
    $removed = [int](($summary | Measure-Object -Property RemovedCount -Sum).Sum)
    $skipped = [int](($summary | Measure-Object -Property SkippedCount -Sum).Sum)
    Write-Ok "Removed $removed log file(s); skipped $skipped locked or inaccessible file(s)."
}

function Remove-LauncherPath {
    [CmdletBinding()]
    param(
        [Parameter(Mandatory)][string]$Path,
        [switch]$KeepRoot,
        [string[]]$PreserveNames = @('.gitkeep'),
        [switch]$Strict,
        [switch]$WhatIf,
        [string]$Activity = 'TKBEN: remove files'
    )

    if ([string]::IsNullOrWhiteSpace($Path)) {
        throw 'Refusing to remove an empty path.'
    }
    $fullPath = [IO.Path]::GetFullPath($Path)
    $normalizedPath = $fullPath.TrimEnd('\')
    $filesystemRoot = [IO.Path]::GetPathRoot($fullPath).TrimEnd('\')
    $repositoryRoot = if ($script:RepoRoot) { [IO.Path]::GetFullPath([string]$script:RepoRoot).TrimEnd('\') } else { $null }
    if ($normalizedPath -eq $filesystemRoot -or
        ($repositoryRoot -and
        ($normalizedPath -eq $repositoryRoot -or
        $repositoryRoot.StartsWith("$normalizedPath\", [StringComparison]::OrdinalIgnoreCase)))) {
        throw "Refusing to remove a filesystem or repository root: $fullPath"
    }

    $plannedPaths = [Collections.Generic.List[string]]::new()
    $preservedPaths = [Collections.Generic.List[string]]::new()
    $removedPaths = [Collections.Generic.List[string]]::new()
    $skippedPaths = [Collections.Generic.List[string]]::new()
    $enumerationErrorPaths = [Collections.Generic.List[string]]::new()
    $warningMessages = [Collections.Generic.List[string]]::new()
    $result = [ordered]@{
        Target = $fullPath
        Path = $fullPath
        Planned = 0
        PlannedCount = 0
        PlannedPaths = @()
        Preserved = 0
        PreservedCount = 0
        PreservedEntries = @()
        PreservedPaths = @()
        Removed = 0
        RemovedCount = 0
        RemovedPaths = @()
        Skipped = 0
        SkippedCount = 0
        SkippedPaths = @()
        EnumerationErrors = @()
        EnumerationErrorCount = 0
        EnumerationErrorPaths = @()
        WhatIf = [bool]$WhatIf
    }

    try {
        $root = Get-Item -LiteralPath $fullPath -Force -ErrorAction Stop
    }
    catch {
        if ($_.CategoryInfo.Category -eq [System.Management.Automation.ErrorCategory]::ObjectNotFound) {
            Clear-LauncherProgress
            return [pscustomobject]$result
        }
        $message = [string]$_.Exception.Message
        [void]$skippedPaths.Add($fullPath)
        $result.Skipped = $skippedPaths.Count
        $result.SkippedCount = $skippedPaths.Count
        $result.SkippedPaths = @($skippedPaths.ToArray())
        Clear-LauncherProgress
        Write-Host "[WARN] Skipped inaccessible path: $fullPath ($message)" -ForegroundColor Yellow
        if ($Strict) { throw }
        return [pscustomobject]$result
    }

    $enumerationErrors = @()
    $entries = if ($root.PSIsContainer) {
        @(Get-ChildItem -LiteralPath $root.FullName -Force -Recurse -ErrorAction SilentlyContinue -ErrorVariable enumerationErrors)
    } else {
        @($root)
    }
    foreach ($enumerationError in @($enumerationErrors)) {
        $errorPath = [string]$enumerationError.TargetObject
        if ([string]::IsNullOrWhiteSpace($errorPath)) { $errorPath = $fullPath }
        [void]$enumerationErrorPaths.Add($errorPath)
        [void]$warningMessages.Add(("Skipped inaccessible path below {0}: {1}" -f $fullPath, $enumerationError.Exception.Message))
    }

    $protectedDirectories = [Collections.Generic.HashSet[string]]::new([StringComparer]::OrdinalIgnoreCase)
    foreach ($entry in @($entries)) {
        if ($entry.Name -in $PreserveNames) {
            [void]$preservedPaths.Add($entry.FullName)
            [void]$protectedDirectories.Add($root.FullName)
            $ancestor = [IO.Path]::GetDirectoryName($entry.FullName)
            while ($ancestor -and $ancestor.StartsWith($root.FullName.TrimEnd('\') + '\', [StringComparison]::OrdinalIgnoreCase)) {
                [void]$protectedDirectories.Add($ancestor)
                $ancestor = [IO.Path]::GetDirectoryName($ancestor)
            }
        }
    }

    $candidates = @($entries |
        Where-Object { -not $preservedPaths.Contains($_.FullName) -and -not $protectedDirectories.Contains($_.FullName) } |
        Sort-Object @{ Expression = { $_.FullName.Length }; Descending = $true }, @{ Expression = { $_.FullName.ToUpperInvariant() }; Descending = $false })
    if ($root.PSIsContainer -and -not $KeepRoot -and $preservedPaths.Count -eq 0) {
        $candidates += $root
    }
    foreach ($candidate in @($candidates)) { [void]$plannedPaths.Add($candidate.FullName) }
    $result.Planned = $plannedPaths.Count
    $result.PlannedCount = $plannedPaths.Count
    $result.PlannedPaths = @($plannedPaths.ToArray())
    $result.Preserved = $preservedPaths.Count
    $result.PreservedCount = $preservedPaths.Count
    $result.PreservedEntries = @($preservedPaths.ToArray())
    $result.PreservedPaths = @($preservedPaths.ToArray() | Sort-Object { $_.ToUpperInvariant() })
    $result.EnumerationErrors = @($enumerationErrors | ForEach-Object { [string]$_ })
    $result.EnumerationErrorCount = $enumerationErrorPaths.Count
    $result.EnumerationErrorPaths = @($enumerationErrorPaths.ToArray() | Sort-Object { $_.ToUpperInvariant() })

    $progressId = $null
    try {
        if ($plannedPaths.Count -gt 0) {
            $progressId = Start-LauncherProgress -Activity $Activity -Status "0 of $($plannedPaths.Count) items"
        }
        for ($index = 0; $index -lt $plannedPaths.Count; $index++) {
            $candidatePath = $plannedPaths[$index]
            if ($null -ne $progressId) {
                Update-LauncherProgress -Id $progressId -Activity $Activity -Status "$($index + 1) of $($plannedPaths.Count): $([IO.Path]::GetFileName($candidatePath))" -PercentComplete ([int](($index + 1) * 100 / [Math]::Max(1, $plannedPaths.Count)))
            }
            if ($WhatIf) { continue }
            try {
                Remove-Item -LiteralPath $candidatePath -Force -Confirm:$false -ErrorAction Stop
                [void]$removedPaths.Add($candidatePath)
            }
            catch {
                [void]$skippedPaths.Add($candidatePath)
                [void]$warningMessages.Add("Skipped locked or protected path: $candidatePath ($($_.Exception.Message))")
            }
        }
    }
    finally {
        if ($null -ne $progressId) { Complete-LauncherProgress -Id $progressId }
    }

    $result.Removed = $removedPaths.Count
    $result.RemovedCount = $removedPaths.Count
    $result.RemovedPaths = @($removedPaths.ToArray())
    $result.Skipped = $skippedPaths.Count
    $result.SkippedCount = $skippedPaths.Count
    $result.SkippedPaths = @($skippedPaths.ToArray())
    foreach ($message in $warningMessages.ToArray()) { Write-Host "[WARN] $message" -ForegroundColor Yellow }
    Clear-LauncherProgress
    if ($Strict -and ($result.SkippedCount -gt 0 -or $result.EnumerationErrorCount -gt 0)) {
        throw "Removal of '$fullPath' was incomplete. Skipped $($result.SkippedCount) item(s) and encountered $($result.EnumerationErrorCount) enumeration error(s)."
    }
    return [pscustomobject]$result
}

function Remove-PathBestEffort {
    param([Parameter(Mandatory)][string]$Path)
    return Remove-LauncherPath -Path $Path -Activity "TKBEN: remove $([IO.Path]::GetFileName($Path))"
}

function Assert-SafeCleanupDirectory {
    param(
        [Parameter(Mandatory)][string]$Path,
        [Parameter(Mandatory)][string]$Description
    )

    $candidate = [IO.Path]::GetFullPath($Path).TrimEnd('\')
    $repository = [IO.Path]::GetFullPath($RepoRoot).TrimEnd('\')
    $protectedDirectories = @(
        $repository,
        [IO.Path]::GetFullPath($AppDir).TrimEnd('\'),
        [IO.Path]::GetFullPath($AssetsDir).TrimEnd('\'),
        [IO.Path]::GetFullPath($SettingsDir).TrimEnd('\'),
        [IO.Path]::GetFullPath($ServerDir).TrimEnd('\'),
        [IO.Path]::GetFullPath($ClientDir).TrimEnd('\'),
        [IO.Path]::GetFullPath($RuntimeDir).TrimEnd('\'),
        [IO.Path]::GetFullPath($TestsDir).TrimEnd('\')
    )

    $filesystemRoot = [IO.Path]::GetPathRoot($candidate).TrimEnd('\')
    if ($candidate -eq $filesystemRoot -or
        $protectedDirectories -contains $candidate -or
        $repository.StartsWith("$candidate\", [StringComparison]::OrdinalIgnoreCase)) {
        throw "Refusing to remove application files through $Description at $Path."
    }
}

function Remove-DirectoryContents {
    param([Parameter(Mandatory)][string]$Path)

    if (-not (Test-Path -LiteralPath $Path -PathType Container)) {
        Clear-LauncherProgress
        return [pscustomobject]@{
            Path = [IO.Path]::GetFullPath($Path)
            PlannedCount = 0
            PlannedPaths = @()
            PreservedCount = 0
            PreservedPaths = @()
            RemovedCount = 0
            RemovedPaths = @()
            SkippedCount = 0
            SkippedPaths = @()
            EnumerationErrorCount = 0
            EnumerationErrorPaths = @()
            WhatIf = $false
        }
    }
    $result = Remove-LauncherPath -Path $Path -KeepRoot -PreserveNames @('.gitkeep') -Activity "TKBEN: clear $Path"
    return [pscustomobject]@{
        Path = $result.Path
        PlannedCount = $result.PlannedCount
        PlannedPaths = $result.PlannedPaths
        PreservedCount = $result.PreservedCount
        PreservedPaths = $result.PreservedPaths
        RemovedCount = $result.RemovedCount
        RemovedPaths = $result.RemovedPaths
        SkippedCount = $result.SkippedCount
        SkippedPaths = $result.SkippedPaths
        EnumerationErrorCount = $result.EnumerationErrorCount
        EnumerationErrorPaths = $result.EnumerationErrorPaths
        WhatIf = $result.WhatIf
    }
}

function Remove-AllData {
    Clear-LauncherProgress
    $confirmation = ([string](Read-Host 'This permanently deletes user data. Continue? [y/N]')).Trim()
    if ($confirmation -notmatch '^(?i:y|yes)$') {
        Write-Host '[INFO] Remove All Data cancelled.' -ForegroundColor DarkGray
        return
    }

    $dataRoot = Get-ApplicationDataRoot
    $logRoot = Get-ApplicationLogRoot -DataRoot $dataRoot
    if ([IO.Path]::GetFullPath($logRoot).TrimEnd('\') -eq [IO.Path]::GetFullPath($dataRoot).TrimEnd('\')) {
        throw 'The configured log directory cannot be the application data directory.'
    }
    Assert-SafeCleanupDirectory -Path $logRoot -Description 'the configured log directory'

    Write-Step "Removing user data from $dataRoot."
    $summaries = @()
    foreach ($dataFile in @(
        (Join-Path $dataRoot 'database.db'),
        (Join-Path $dataRoot 'database.db-wal'),
        (Join-Path $dataRoot 'database.db-shm'),
        (Join-Path $dataRoot 'database.db-journal'),
        (Get-HuggingFaceMaterialPath -DataRoot $dataRoot)
    ) | Select-Object -Unique) {
        if (Test-Path -LiteralPath $dataFile) {
            $summaries += @(Remove-PathBestEffort -Path $dataFile)
        }
    }

    foreach ($dataDirectory in @(
        (Join-Path $dataRoot 'sources\datasets'),
        (Join-Path $dataRoot 'sources\tokenizers'),
        $logRoot
    ) | Select-Object -Unique) {
        $summaries += @(Remove-DirectoryContents -Path $dataDirectory)
    }

    $removed = [int](($summaries | Measure-Object -Property RemovedCount -Sum).Sum)
    $skipped = [int](($summaries | Measure-Object -Property SkippedCount -Sum).Sum) +
        [int](($summaries | Measure-Object -Property EnumerationErrorCount -Sum).Sum)
    if ((Get-EnvironmentSetting -Key 'DATABASE_EMBEDDED' -DefaultValue 'true') -ine 'true') {
        Write-Host '[WARN] DATABASE_EMBEDDED is false; the external database was not modified.' -ForegroundColor Yellow
    }
    if ($skipped -gt 0) {
        Write-Ok "Removed $removed user-data file(s); skipped $skipped locked or inaccessible path(s)."
    } else {
        Write-Ok "Removed $removed user-data file(s). Application files, templates, and .gitkeep files were preserved."
    }
}

function Remove-PythonCaches {
    $cacheDirectories = @(Get-ChildItem -LiteralPath $RepoRoot -Directory -Filter '__pycache__' -Recurse -Force -ErrorAction SilentlyContinue |
        Sort-Object @{ Expression = { $_.FullName.Length }; Descending = $true }, @{ Expression = { $_.FullName.ToUpperInvariant() }; Descending = $false })
    @($cacheDirectories | ForEach-Object { Remove-PathBestEffort -Path $_.FullName })
}

function Clear-ManagedCache {
    $summaries = @()
    foreach ($cacheRoot in @($RuntimeCacheDir, $ToolCacheDir)) {
        Ensure-Directory $cacheRoot
        $entries = @(Get-ChildItem -LiteralPath $cacheRoot -Force -ErrorAction SilentlyContinue |
            Where-Object { $_.Name -ne '.gitkeep' } |
            Sort-Object @{ Expression = { $_.FullName.ToUpperInvariant() }; Descending = $false })
        $summaries += @($entries | ForEach-Object { Remove-PathBestEffort -Path $_.FullName })
    }

    $summaries
}

function Clear-Cache {
    Write-Step 'Removing development caches and temporary artifacts.'
    $summaries = @(
        Remove-PythonCaches
        Clear-ManagedCache
    )
    $skipped = [int](($summaries | Measure-Object -Property SkippedCount -Sum).Sum) +
        [int](($summaries | Measure-Object -Property EnumerationErrorCount -Sum).Sum)
    if ($skipped -gt 0) {
        Write-Ok "Caches cleared where permitted; skipped $skipped locked or inaccessible path(s)."
    } else {
        Write-Ok 'Caches cleared.'
    }
}

function Uninstall-Application {
    Write-Step 'Removing downloaded runtimes, dependencies, build output, and Python caches.'
    $summaries = @()
    $directories = @(
        $RuntimeDir,
        $VenvDir,
        (Join-Path $RepoRoot '.venv'),
        (Join-Path $ClientDir 'node_modules'),
        (Join-Path $ClientDir '.angular'),
        (Join-Path $ClientDir 'dist')
    )
    foreach ($directory in $directories) {
        if (Test-Path -LiteralPath $directory) { $summaries += @(Remove-PathBestEffort -Path $directory) }
    }
    # Project manifests, dependency lockfiles, and tool configuration remain intact.
    $summaries += @(Remove-PythonCaches)
    $summaries += @(Clear-ManagedCache)
    $removed = [int](($summaries | Measure-Object -Property RemovedCount -Sum).Sum)
    $skipped = [int](($summaries | Measure-Object -Property SkippedCount -Sum).Sum) +
        [int](($summaries | Measure-Object -Property EnumerationErrorCount -Sum).Sum)
    if ($skipped -gt 0) {
        Write-Ok "Removed $removed application path item(s) where permitted; skipped $skipped locked or inaccessible path(s). Dependency lockfiles and user data were preserved."
    } else {
        Write-Ok "Removed $removed application path item(s). Dependency lockfiles and user data were preserved."
    }
}

# =============================================================================
# Source update management
# =============================================================================
function Update-Application {
    Write-Step 'Updating the application from origin/main (fast-forward only).'
    Push-Location $RepoRoot
    try {
        $branchOutput = @(& git branch --show-current 2>$null)
        $branchExitCode = if ($null -eq $LASTEXITCODE) { 0 } else { [int]$LASTEXITCODE }
        $branch = (@($branchOutput | ForEach-Object { [string]$_ }) -join [Environment]::NewLine).Trim()
        if ($branchExitCode -ne 0 -or [string]::IsNullOrWhiteSpace($branch)) { throw 'Update requires a non-detached Git checkout.' }
        if ($branch -ne 'main') { throw "Update requires the main branch to be checked out; current branch is '$branch'. No files were changed." }
        $statusOutput = @(& git status --porcelain 2>$null)
        $statusExitCode = if ($null -eq $LASTEXITCODE) { 0 } else { [int]$LASTEXITCODE }
        if ($statusExitCode -ne 0) { throw 'Unable to inspect the Git working tree before updating.' }
        $changes = @($statusOutput | Where-Object { -not [string]::IsNullOrWhiteSpace([string]$_) })
        if ($changes.Count -gt 0) { throw 'Update requires a clean Git working tree. Commit or safely preserve local changes before retrying.' }
        & git pull --ff-only origin main
        $exitCode = if ($null -eq $LASTEXITCODE) { 0 } else { [int]$LASTEXITCODE }
        if ($exitCode -ne 0) { throw "Application update failed with exit code $exitCode." }
    } finally {
        Pop-Location
    }
    Write-Ok 'Application updated from origin/main.'
}

function Check-ForUpdates {
    Push-Location $RepoRoot
    try {
        $currentRevision = (& git rev-parse HEAD 2>&1).Trim()
        if ($LASTEXITCODE -ne 0) { throw "Could not determine the current application revision: $currentRevision" }

        # ls-remote reads the remote branch tip without fetching or applying objects.
        $remoteOutput = @(& git ls-remote origin refs/heads/main 2>&1)
        if ($LASTEXITCODE -ne 0) {
            $details = ($remoteOutput -join ' ').Trim()
            throw "Could not check origin/main for updates$(if ($details) { ": $details" })."
        }
        $remoteLine = [string]($remoteOutput | Select-Object -First 1)
        $remoteRevision = ($remoteLine -split '\s+')[0]
        if ($remoteRevision -notmatch '^[0-9a-fA-F]{40}$') {
            throw 'The origin/main revision could not be read.'
        }
    } finally {
        Pop-Location
    }

    if ($currentRevision -eq $remoteRevision) {
        Write-Ok 'The application is up to date with origin/main.'
    } else {
        Write-Host "[UPDATE] A newer origin/main revision is available ($($remoteRevision.Substring(0, 7)); current $($currentRevision.Substring(0, 7)))." -ForegroundColor Yellow
        Write-Host '         Run Update to pull the main branch.' -ForegroundColor DarkGray
    }
}

# =============================================================================
# Interactive menu
# =============================================================================
function Wait-ForMenu {
    Clear-LauncherProgress
    Write-Host
    Write-Host 'Press any key to return to the menu...' -ForegroundColor DarkGray
    if (-not $script:LauncherInteractive) { return }
    [Console]::ReadKey($true) | Out-Null
}

function Clear-MenuScreen {
    if ([Console]::IsOutputRedirected) { return }
    try {
        Clear-Host
    } catch {
        # Hosts without a usable cursor handle can still render the menu.
    }
}

function Get-LauncherMenuEntries {
    return @(
        [pscustomobject]@{ Section = 'APPLICATION'; Label = 'Launch application'; Description = 'Start the local benchmark workspace'; Key = 'Launch'; Destructive = $false }
        [pscustomobject]@{ Section = 'SETUP & VALIDATION'; Label = 'Install / update dependencies'; Description = 'Sync tooling and the database'; Key = 'Install'; Destructive = $false }
        [pscustomobject]@{ Section = 'SETUP & VALIDATION'; Label = 'Rebuild frontend'; Description = 'Build the Angular production output only'; Key = 'Rebuild'; Destructive = $false }
        [pscustomobject]@{ Section = 'SETUP & VALIDATION'; Label = 'Initialize database'; Description = 'Create or update the local database'; Key = 'Database'; Destructive = $false }
        [pscustomobject]@{ Section = 'SETUP & VALIDATION'; Label = 'Run test suite'; Description = 'Validate backend and frontend checks'; Key = 'Tests'; Destructive = $false }
        [pscustomobject]@{ Section = 'SOURCE CONTROL'; Label = 'Check for updates'; Description = 'Report whether origin/main has a newer revision'; Key = 'Check'; Destructive = $false }
        [pscustomobject]@{ Section = 'SOURCE CONTROL'; Label = 'Update application'; Description = 'Pull the application from the main branch'; Key = 'Update'; Destructive = $false }
        [pscustomobject]@{ Section = 'DATA & MAINTENANCE'; Label = 'Remove logs'; Description = 'Clear generated application logs'; Key = 'Logs'; Destructive = $true }
        [pscustomobject]@{ Section = 'DATA & MAINTENANCE'; Label = 'Clear cache'; Description = 'Remove downloaded and generated caches'; Key = 'Cache'; Destructive = $true }
        [pscustomobject]@{ Section = 'DATA & MAINTENANCE'; Label = 'Remove all data'; Description = 'Delete the database and user-created files'; Key = 'AllData'; Destructive = $true }
        [pscustomobject]@{ Section = 'DATA & MAINTENANCE'; Label = 'Uninstall application'; Description = 'Remove local runtimes and dependencies'; Key = 'Uninstall'; Destructive = $true }
        [pscustomobject]@{ Section = 'EXIT'; Label = 'Exit'; Description = 'Close this launcher'; Key = 'Exit'; Destructive = $false }
    )
}

function Write-MenuItem {
    param(
        [Parameter(Mandatory)][pscustomobject]$Entry,
        [Parameter(Mandatory)][int]$NumberWidth,
        [Parameter(Mandatory)][int]$LabelWidth
    )
    $color = if ($Entry.Destructive) { 'Yellow' } elseif ($Entry.Key -eq 'Exit') { 'DarkGray' } else { 'White' }
    Write-Host ("  {0,$NumberWidth}. {1,-$LabelWidth}  {2}" -f $Entry.Number, $Entry.Label, $Entry.Description) -ForegroundColor $color
}

function Show-Menu {
    while ($true) {
        Clear-LauncherProgress
        Clear-MenuScreen
        if (-not [Console]::IsOutputRedirected) {
            try { $host.UI.RawUI.WindowTitle = 'TKBEN | Tokenizers Benchmarker' } catch { }
        }
        Write-Host
        Write-Host '  +----------------------------------------------------------+' -ForegroundColor DarkCyan
        Write-Host '  |' -NoNewline -ForegroundColor DarkCyan
        Write-Host '  TKBEN' -NoNewline -ForegroundColor Cyan
        Write-Host '  TOKENIZERS BENCHMARKER' -NoNewline -ForegroundColor White
        Write-Host '                         |' -ForegroundColor DarkCyan
        Write-Host ('  |  {0,-56}|' -f 'Launch, maintain, and validate your local workspace.') -ForegroundColor DarkGray
        Write-Host '  +----------------------------------------------------------+' -ForegroundColor DarkCyan
        Write-Host
        $entries = @(Get-LauncherMenuEntries)
        for ($index = 0; $index -lt $entries.Count; $index++) {
            $entries[$index] = [pscustomobject]@{
                Number = $index + 1
                Section = $entries[$index].Section
                Label = $entries[$index].Label
                Description = $entries[$index].Description
                Key = $entries[$index].Key
                Destructive = $entries[$index].Destructive
            }
        }
        $numberWidth = ([string]$entries.Count).Length
        $labelWidth = ($entries | ForEach-Object { $_.Label.Length } | Measure-Object -Maximum).Maximum
        $lastSection = $null
        foreach ($entry in $entries) {
            if ($entry.Section -ne $lastSection) {
                if ($null -ne $lastSection) { Write-Host }
                Write-Host ("  {0}" -f $entry.Section) -ForegroundColor DarkCyan
                $lastSection = $entry.Section
            }
            Write-MenuItem -Entry $entry -NumberWidth $numberWidth -LabelWidth $labelWidth
        }
        Write-Host
        Write-Host '  +----------------------------------------------------------+' -ForegroundColor DarkCyan
        Write-Host
        $maxOption = $entries.Count
        if (-not $script:LauncherInteractive) { return }
        Clear-LauncherProgress
        $selection = (Read-Host "  Select an option (1-$maxOption)").Trim()

        if ($selection -notmatch '^[1-9][0-9]*$' -or [int]$selection -lt 1 -or [int]$selection -gt $maxOption) {
            Write-Fatal "Invalid option. Enter a number from 1 through $maxOption."
            Wait-ForMenu
            continue
        }
        $entry = $entries[[int]$selection - 1]
        if ($entry.Key -eq 'Exit') { break }

        try {
            Invoke-TrackedLauncherAction -Name $entry.Label -Action {
                switch ($entry.Key) {
                    'Launch' { Launch-Application; exit 0 }
                    'Install' { Install-Dependencies }
                    'Rebuild' { Rebuild-Frontend }
                    'Database' { Initialize-Database }
                    'Tests' { Run-TestSuite }
                    'Check' { Check-ForUpdates }
                    'Update' { Update-Application }
                    'Logs' { Remove-Logs }
                    'Cache' { Clear-Cache }
                    'AllData' { Remove-AllData }
                    'Uninstall' { Uninstall-Application }
                }
            }
        } catch {
            Write-Fatal $_.Exception.Message
        }
        Wait-ForMenu
    }
}

if ($Launch) {
    Invoke-TrackedLauncherAction -Name 'launch application' -Action { Launch-Application }
    exit 0
}

Show-Menu
Clear-LauncherProgress
