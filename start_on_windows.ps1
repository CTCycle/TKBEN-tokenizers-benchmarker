[CmdletBinding()]
param()

$ErrorActionPreference = 'Stop'
$ProgressPreference = 'SilentlyContinue'

$RepoRoot = [IO.Path]::GetFullPath($PSScriptRoot)
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
$VenvDir = Join-Path $ServerDir '.venv'
$VenvPython = Join-Path $VenvDir 'Scripts\python.exe'
$EnvFile = Join-Path $RepoRoot 'settings\.env'
$EnvTemplate = Join-Path $RepoRoot 'settings\.env.example'
$UvCacheDir = Join-Path $RuntimeDir '.uv-cache'
$PythonVersion = '3.14.2'
$NodeVersion = '22.12.0'

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

function Invoke-FindUv {
    param([Parameter(Mandatory)][string]$SearchRoot)
    $match = Get-ChildItem -LiteralPath $SearchRoot -Recurse -Filter 'uv.exe' -File | Select-Object -First 1
    if ($match) { $match.FullName }
}

function Invoke-HealthCheck {
    param(
        [Parameter(Mandatory)][uri]$Uri,
        [ValidateRange(1, 3600)][int]$Attempts = 60,
        [ValidateRange(1, 60)][int]$IntervalSeconds = 1
    )
    for ($attempt = 1; $attempt -le $Attempts; $attempt++) {
        try {
            $response = Invoke-WebRequest -UseBasicParsing -Uri $Uri -TimeoutSec 2
            if ($response.StatusCode -ge 200 -and $response.StatusCode -lt 400) { return }
        } catch {
            if ($attempt -eq $Attempts) { break }
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
        OPTIONAL_DEPENDENCIES = 'false'
        ALWAYS_REBUILD = 'true'
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

    if ($env:ALWAYS_REBUILD -ieq 'true') {
        $env:ALWAYS_REBUILD = 'true'
    } elseif ($env:ALWAYS_REBUILD -ieq 'false') {
        $env:ALWAYS_REBUILD = 'false'
    } else {
        throw "ALWAYS_REBUILD must be either 'true' or 'false'."
    }

    $env:UV_CACHE_DIR = $UvCacheDir
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

    if (-not (Test-Path -LiteralPath $NodeExe)) {
        Write-Step "Downloading Node.js $NodeVersion (portable x64)."
        $nodeArchive = Join-Path $NodeDir "node-v$NodeVersion-win-x64.zip"
        Invoke-DownloadAndExtract `
            -Uri "https://nodejs.org/dist/v$NodeVersion/node-v$NodeVersion-win-x64.zip" `
            -ArchivePath $nodeArchive `
            -Destination $NodeDir
        $nestedNodeDir = Join-Path $NodeDir "node-v$NodeVersion-win-x64"
        if (Test-Path -LiteralPath (Join-Path $nestedNodeDir 'node.exe')) {
            Get-ChildItem -LiteralPath $nestedNodeDir -Force | Move-Item -Destination $NodeDir -Force
            Remove-Item -LiteralPath $nestedNodeDir -Recurse -Force
        }
    }
    if (-not (Test-Path -LiteralPath $NodeExe)) { throw "Node.js was not installed at $NodeExe" }
    if (-not (Test-Path -LiteralPath $NpmCmd)) { throw "npm was not installed at $NpmCmd" }
    Write-Ok "Node.js ready: $(& $NodeExe --version)"
}

function Sync-Dependencies {
    param([bool]$BuildFrontend = $true)

    Import-Environment
    Install-Runtimes

    Write-Step 'Installing Python dependencies.'
    $uvArguments = @('sync', '--python', $PythonExe)
    if ($env:OPTIONAL_DEPENDENCIES -ieq 'true') { $uvArguments += '--all-extras' }
    Push-Location $ServerDir
    try {
        & $UvExe @uvArguments
        if ($LASTEXITCODE -ne 0) { throw "uv sync failed with exit code $LASTEXITCODE." }
    } finally {
        Pop-Location
    }

    Write-Step 'Installing frontend dependencies.'
    Push-Location $ClientDir
    try {
        if (Test-Path -LiteralPath (Join-Path $ClientDir 'package-lock.json')) {
            & $NpmCmd ci
        } else {
            & $NpmCmd install
        }
        if ($LASTEXITCODE -ne 0) { throw "npm dependency installation failed with exit code $LASTEXITCODE." }

        if ($BuildFrontend) {
            Write-Step 'Building frontend.'
            & $NpmCmd run build
            if ($LASTEXITCODE -ne 0) { throw "Frontend build failed with exit code $LASTEXITCODE." }
        } else {
            Write-Step 'Skipping frontend build because ALWAYS_REBUILD=false.'
        }
    } finally {
        Pop-Location
    }
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
    Sync-Dependencies -BuildFrontend ($env:ALWAYS_REBUILD -ieq 'true')
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

    Invoke-HealthCheck -Uri "http://$($env:FASTAPI_HOST):$backendPort/api/health" -Attempts 60 -IntervalSeconds 1
    $backendPid = if ($backendProcess) { $backendProcess.Id } else { Get-PortProcessId -Port $backendPort }

    Write-Step 'Starting frontend preview.'
    $frontendProcess = Start-Process -FilePath $NpmCmd `
        -ArgumentList @('run', 'preview', '--', '--host', $env:UI_HOST, '--port', "$uiPort", '--strictPort') `
        -WorkingDirectory $ClientDir -WindowStyle Hidden -PassThru
    Invoke-HealthCheck -Uri "http://$($env:UI_HOST):$uiPort/" -Attempts 60 -IntervalSeconds 1

    $url = "http://$($env:UI_HOST):$uiPort"
    Start-Process $url
    Write-Ok 'Application started successfully.'
    Write-Host "Backend: http://$($env:FASTAPI_HOST):$backendPort (PID $backendPid)"
    Write-Host "Frontend: $url (PID $($frontendProcess.Id))"
}

function Install-Dependencies {
    Sync-Dependencies
    if (Test-Path -LiteralPath $UvCacheDir) { Remove-Item -LiteralPath $UvCacheDir -Recurse -Force }
    Write-Ok 'Dependencies installed and frontend built.'
}

function Initialize-Database {
    Import-Environment
    Install-Runtimes
    Write-Step 'Initializing database.'
    & $UvExe run --project $ServerDir --python $PythonExe python (Join-Path $RepoRoot 'app\scripts\initialize_database.py') --drop-existing --seed-catalogs --force-reseed-catalogs
    if ($LASTEXITCODE -ne 0) { throw "Database initialization failed with exit code $LASTEXITCODE." }
    Write-Ok 'Database initialized.'
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

function Clear-Cache {
    Write-Step 'Removing Python and uv caches.'
    Remove-PythonCaches
    if (Test-Path -LiteralPath $UvCacheDir) { Remove-Item -LiteralPath $UvCacheDir -Recurse -Force }
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
    Write-Ok 'Application dependencies and generated files removed. Settings and user data were preserved.'
}

function Wait-ForMenu {
    if ([Console]::IsInputRedirected) { return }
    Write-Host
    Write-Host 'Press any key to return to the menu...' -ForegroundColor DarkGray
    [Console]::ReadKey($true) | Out-Null
}

function Write-MenuItem([string]$Number, [string]$Label, [string]$Description, [ConsoleColor]$Color = [ConsoleColor]::White) {
    Write-Host "  [$Number] " -NoNewline -ForegroundColor $Color
    Write-Host $Label -NoNewline -ForegroundColor White
    Write-Host "  $Description" -ForegroundColor DarkGray
}

function Show-Menu {
    while ($true) {
        Clear-Host
        $host.UI.RawUI.WindowTitle = 'TKBEN | Tokenizers Benchmarker'
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
        Write-MenuItem '2' 'Install / update dependencies' 'Sync the required local tooling'
        Write-MenuItem '3' 'Initialize database' 'Create or update the local database'
        Write-MenuItem '4' 'Run test suite' 'Validate backend and frontend checks'
        Write-Host
        Write-Host '  MAINTENANCE' -ForegroundColor DarkCyan
        Write-MenuItem '5' 'Remove logs' 'Clear generated application logs'
        Write-MenuItem '6' 'Clear cache' 'Remove downloaded and generated caches'
        Write-MenuItem '7' 'Uninstall application' 'Remove local runtimes and dependencies' Yellow
        Write-Host
        Write-Host '  +----------------------------------------------------------+' -ForegroundColor DarkCyan
        Write-MenuItem '8' 'Exit' 'Close this launcher' DarkGray
        Write-Host '  +----------------------------------------------------------+' -ForegroundColor DarkCyan
        Write-Host
        $selection = (Read-Host '  Select an option [1-8]').Trim()

        if ($selection -notmatch '^[1-8]$') {
            Write-Fatal 'Invalid option. Enter a number from 1 through 8.'
            Wait-ForMenu
            continue
        }
        if ($selection -eq '8') { break }

        try {
            switch ($selection) {
                '1' { Launch-Application; exit 0 }
                '2' { Install-Dependencies }
                '3' { Initialize-Database }
                '4' { Run-TestSuite }
                '5' { Remove-Logs }
                '6' { Clear-Cache }
                '7' { Uninstall-Application }
            }
        } catch {
            Write-Fatal $_.Exception.Message
        }
        Wait-ForMenu
    }
}

Show-Menu
