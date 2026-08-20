[CmdletBinding()]
param(
    [switch]$Launch
)

$ErrorActionPreference = 'Stop'
$ProgressPreference = 'SilentlyContinue'

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
$VenvDir = Join-Path $ServerDir '.venv'
$VenvPython = Join-Path $VenvDir 'Scripts\python.exe'
$EnvFile = Join-Path $RepoRoot 'settings\.env'
$EnvTemplate = Join-Path $RepoRoot 'settings\.env.example'
$CacheDir = Join-Path $RepoRoot 'assets\cache'
$UvCacheDir = Join-Path $CacheDir 'uv'
$PythonVersion = '3.14.2'
$NodeVersion = '22.23.1'

function Write-Step([string]$Message) { Write-Host "[STEP] $Message" -ForegroundColor Cyan }
function Write-Ok([string]$Message) { Write-Host "[OK] $Message" -ForegroundColor Green }
function Write-Fatal([string]$Message) { Write-Host "[FATAL] $Message" -ForegroundColor Red }

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

function Invoke-DownloadAndExtract {
    param(
        [Parameter(Mandatory)][uri]$Uri,
        [Parameter(Mandatory)][string]$ArchivePath,
        [Parameter(Mandatory)][string]$Destination
    )
    [IO.Directory]::CreateDirectory((Split-Path -Parent $ArchivePath)) | Out-Null
    [IO.Directory]::CreateDirectory($Destination) | Out-Null
    try {
        Invoke-WebRequest -Uri $Uri -OutFile $ArchivePath
        Expand-Archive -LiteralPath $ArchivePath -DestinationPath $Destination -Force
    } finally {
        Remove-Item -LiteralPath $ArchivePath -Force -ErrorAction SilentlyContinue
    }
}

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
            Remove-Item -LiteralPath $backupDir -Recurse -Force -ErrorAction Stop
        }
    } catch {
        if ($newRuntimeInstalled -and (Test-Path -LiteralPath $NodeDir)) {
            Remove-Item -LiteralPath $NodeDir -Recurse -Force -ErrorAction SilentlyContinue
        }
        if ($oldRuntimeMoved -and (Test-Path -LiteralPath $backupDir) -and -not (Test-Path -LiteralPath $NodeDir)) {
            Move-Item -LiteralPath $backupDir -Destination $NodeDir -ErrorAction SilentlyContinue
        }
        throw
    } finally {
        if (Test-Path -LiteralPath $stagingDir) {
            Remove-Item -LiteralPath $stagingDir -Recurse -Force -ErrorAction SilentlyContinue
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
        FASTAPI_PORT = '8000'
        UI_HOST = '127.0.0.1'
        UI_PORT = '8001'
        RELOAD = 'false'
        # Backend logs are visible by default when the setting is absent.
        BACKEND_LOGS_VISIBLE = 'true'
    }
    foreach ($entry in $defaults.GetEnumerator()) {
        Set-Item -Path "Env:$($entry.Key)" -Value $entry.Value
    }

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
        if ($key) {
            Set-Item -Path "Env:$key" -Value $value
        }
    }

    if ($env:BACKEND_LOGS_VISIBLE -ieq 'true') {
        $env:BACKEND_LOGS_VISIBLE = 'true'
    } elseif ($env:BACKEND_LOGS_VISIBLE -ieq 'false') {
        $env:BACKEND_LOGS_VISIBLE = 'false'
    } else {
        throw "BACKEND_LOGS_VISIBLE must be either 'true' or 'false'."
    }

    Ensure-Directory $CacheDir
    foreach ($cacheName in @('uv', 'pip', 'npm', 'ruff', 'mypy', 'pycache', 'coverage', 'playwright', 'pytest', 'pytest-basetemp', 'angular')) {
        Ensure-Directory (Join-Path $CacheDir $cacheName)
    }
    $env:UV_CACHE_DIR = $UvCacheDir
    $env:PIP_CACHE_DIR = Join-Path $CacheDir 'pip'
    $env:NPM_CONFIG_CACHE = Join-Path $CacheDir 'npm'
    $env:RUFF_CACHE_DIR = Join-Path $CacheDir 'ruff'
    $env:MYPY_CACHE_DIR = Join-Path $CacheDir 'mypy'
    $env:PYTHONPYCACHEPREFIX = Join-Path $CacheDir 'pycache'
    $env:COVERAGE_FILE = Join-Path (Join-Path $CacheDir 'coverage') '.coverage'
    $env:PLAYWRIGHT_BROWSERS_PATH = Join-Path $CacheDir 'playwright'
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
    ) | Where-Object { Test-Path -LiteralPath $_ }

    if (-not $manifestPaths) { throw 'Frontend package manifests are missing.' }
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
                    Remove-Item -LiteralPath $UvCacheDir -Recurse -Force
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
            if (Test-Path -LiteralPath (Join-Path $ClientDir 'package-lock.json')) {
                $npmExitCode = Invoke-Npm ci
            } else {
                $npmExitCode = Invoke-Npm install
            }
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
    if (Test-Path -LiteralPath $UvCacheDir) { Remove-Item -LiteralPath $UvCacheDir -Recurse -Force }
    Write-Ok 'Dependencies installed, frontend built, and database synchronized.'
}

function Rebuild-Frontend {
    Import-Environment
    Install-Runtimes
    Sync-Frontend -BuildFrontend -UseCachedFrontendDependencies
    Write-Ok 'Frontend rebuilt.'
}

function Read-InstallationType {
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

function Remove-Logs {
    $logDir = Join-Path $RepoRoot 'app\resources\logs'
    $logs = @(Get-ChildItem -LiteralPath $logDir -Filter '*.log' -File -ErrorAction SilentlyContinue)
    $logs | Remove-Item -Force
    Write-Ok "Removed $($logs.Count) log file(s)."
}

function Remove-PythonCaches {
    Get-ChildItem -LiteralPath $RepoRoot -Directory -Filter '__pycache__' -Recurse -Force -ErrorAction SilentlyContinue |
        Sort-Object FullName -Descending |
        Remove-Item -Recurse -Force
}

function Remove-LegacyDevelopmentCaches {
    Get-ChildItem -LiteralPath $RepoRoot -Directory -Recurse -Force -ErrorAction SilentlyContinue |
        Where-Object {
            $_.Name -in @('.pytest_cache', '.ruff_cache', '.mypy_cache', '.angular') -and
            $_.FullName -notlike "$CacheDir*"
        } |
        Sort-Object FullName -Descending |
        Remove-Item -Recurse -Force
}

function Clear-ManagedCache {
    Ensure-Directory $CacheDir
    Get-ChildItem -LiteralPath $CacheDir -Force -ErrorAction SilentlyContinue |
        Where-Object { $_.Name -ne '.gitkeep' } |
        Remove-Item -Recurse -Force
}

function Clear-Cache {
    Write-Step 'Removing development caches and temporary artifacts.'
    Remove-PythonCaches
    Remove-LegacyDevelopmentCaches
    Clear-ManagedCache
    Write-Ok 'Caches cleared.'
}

function Uninstall-Application {
    Write-Step 'Removing downloaded runtimes, dependencies, build output, lockfiles, and Python caches.'
    $directories = @(
        $RuntimeDir,
        $VenvDir,
        (Join-Path $RepoRoot '.venv'),
        (Join-Path $ClientDir 'node_modules'),
        (Join-Path $ClientDir '.angular'),
        (Join-Path $ClientDir 'dist')
    )
    foreach ($directory in $directories) {
        if (Test-Path -LiteralPath $directory) { Remove-Item -LiteralPath $directory -Recurse -Force }
    }
    foreach ($lockfile in @((Join-Path $ClientDir 'package-lock.json'), (Join-Path $ServerDir 'uv.lock'), (Join-Path $RepoRoot 'uv.lock'))) {
        if (Test-Path -LiteralPath $lockfile) { Remove-Item -LiteralPath $lockfile -Force }
    }
    Remove-PythonCaches
    Remove-LegacyDevelopmentCaches
    Clear-ManagedCache
    Write-Ok 'Application dependencies and generated files removed. Settings and user data were preserved.'
}

function Wait-ForMenu {
    if ([Console]::IsInputRedirected) { return }
    Write-Host
    Write-Host 'Press any key to return to the menu...' -ForegroundColor DarkGray
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

function Write-MenuItem([string]$Number, [string]$Label, [string]$Description, [ConsoleColor]$Color = [ConsoleColor]::White) {
    Write-Host "  [$Number] " -NoNewline -ForegroundColor $Color
    Write-Host $Label -NoNewline -ForegroundColor White
    Write-Host "  $Description" -ForegroundColor DarkGray
}

function Show-Menu {
    while ($true) {
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
        Write-Host '  APPLICATION' -ForegroundColor DarkCyan
        Write-MenuItem '1' 'Launch application' 'Start the local benchmark workspace' Cyan
        Write-Host
        Write-Host '  SETUP & VALIDATION' -ForegroundColor DarkCyan
        Write-MenuItem '2' 'Install / update dependencies' 'Sync tooling and the database'
        Write-MenuItem '3' 'Rebuild frontend' 'Build the Angular production output only'
        Write-MenuItem '4' 'Initialize database' 'Create or update the local database'
        Write-MenuItem '5' 'Run test suite' 'Validate backend and frontend checks'
        Write-Host
        Write-Host '  MAINTENANCE' -ForegroundColor DarkCyan
        Write-MenuItem '6' 'Remove logs' 'Clear generated application logs'
        Write-MenuItem '7' 'Clear cache' 'Remove downloaded and generated caches'
        Write-MenuItem '8' 'Uninstall application' 'Remove local runtimes and dependencies' Yellow
        Write-Host
        Write-Host '  +----------------------------------------------------------+' -ForegroundColor DarkCyan
        Write-MenuItem '9' 'Exit' 'Close this launcher' DarkGray
        Write-Host '  +----------------------------------------------------------+' -ForegroundColor DarkCyan
        Write-Host
        $selection = (Read-Host '  Select an option [1-9]').Trim()

        if ($selection -notmatch '^[1-9]$') {
            Write-Fatal 'Invalid option. Enter a number from 1 through 9.'
            Wait-ForMenu
            continue
        }
        if ($selection -eq '9') { break }

        try {
            switch ($selection) {
                '1' { Launch-Application; exit 0 }
                '2' { Install-Dependencies }
                '3' { Rebuild-Frontend }
                '4' { Initialize-Database }
                '5' { Run-TestSuite }
                '6' { Remove-Logs }
                '7' { Clear-Cache }
                '8' { Uninstall-Application }
            }
        } catch {
            Write-Fatal $_.Exception.Message
        }
        Wait-ForMenu
    }
}

if ($Launch) {
    Launch-Application
    exit 0
}

Show-Menu
