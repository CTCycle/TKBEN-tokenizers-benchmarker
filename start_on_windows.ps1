[CmdletBinding(PositionalBinding = $false)]
param(
    [ValidateSet('Menu','Launch','Install','Sync','Update','InitDatabase','Repair','Clean','Diagnostics','Logs','Stop','Test','BuildDesktop','RemoveDesktop','Uninstall')]
    [string]$Action = 'Menu',
    [switch]$NoBrowser,
    [switch]$Yes
)

$ErrorActionPreference = 'Stop'
$ProgressPreference = 'SilentlyContinue'
$RepoRoot = [IO.Path]::GetFullPath($PSScriptRoot)
$AppDir = Join-Path $RepoRoot 'app'
$ServerDir = Join-Path $AppDir 'server'
$ClientDir = Join-Path $AppDir 'client'
$RuntimeDir = Join-Path $RepoRoot 'runtimes'
$PythonDir = Join-Path $RuntimeDir 'python'
$UvDir = Join-Path $RuntimeDir 'uv'
$NodeDir = Join-Path $RuntimeDir 'nodejs'
$PythonExe = Join-Path $PythonDir 'python.exe'
$UvExe = Join-Path $UvDir 'uv.exe'
$NodeExe = Join-Path $NodeDir 'node.exe'
$NpmCmd = Join-Path $NodeDir 'npm.cmd'
$VenvDir = Join-Path $ServerDir '.venv'
$VenvPython = Join-Path $VenvDir 'Scripts\python.exe'
$UserRoot = Join-Path $env:LOCALAPPDATA 'TKBEN'
$DevState = Join-Path $UserRoot 'dev'
$DataDir = Join-Path $UserRoot 'data'
$LogDir = Join-Path $UserRoot 'logs'
$ConfigDir = Join-Path $UserRoot 'config'
$CacheDir = Join-Path $UserRoot 'cache'
$PidFile = Join-Path $DevState 'processes.json'
$PythonVersion = '3.14.2'
$NodeVersion = '22.14.0'

function Write-Step([string]$Message) { Write-Host "[STEP] $Message" -ForegroundColor Cyan }
function Write-Ok([string]$Message) { Write-Host "[OK] $Message" -ForegroundColor Green }
function Ensure-Directory([string]$Path) { [IO.Directory]::CreateDirectory($Path) | Out-Null }

function Initialize-UserLayout {
    @($DevState,$DataDir,$LogDir,$ConfigDir,$CacheDir,(Join-Path $DataDir 'sources\datasets'),(Join-Path $DataDir 'sources\tokenizers')) | ForEach-Object { Ensure-Directory $_ }
    $targetEnv = Join-Path $ConfigDir '.env'
    if (-not (Test-Path $targetEnv)) { Copy-Item (Join-Path $RepoRoot 'settings\.env.example') $targetEnv }
    $targetConfig = Join-Path $ConfigDir 'configurations.json'
    if (-not (Test-Path $targetConfig)) { Copy-Item (Join-Path $RepoRoot 'settings\configurations.json') $targetConfig }
}

function Set-TkbenEnvironment {
    Initialize-UserLayout
    $env:TKBEN_DATA_DIR = $DataDir
    $env:TKBEN_LOG_DIR = $LogDir
    $env:TKBEN_CONFIG_DIR = $ConfigDir
    $env:HF_HOME = Join-Path $CacheDir 'huggingface'
    $env:MPLCONFIGDIR = Join-Path $CacheDir 'matplotlib'
    $env:UV_CACHE_DIR = Join-Path $CacheDir 'uv'
    $env:UV_PROJECT_ENVIRONMENT = $VenvDir
    $env:PYTHONPATH = $AppDir
    $env:PATH = "$NodeDir;$env:PATH"
}

function Expand-Zip([string]$Uri,[string]$Destination,[string]$ZipName) {
    Ensure-Directory $Destination
    $zip = Join-Path $env:TEMP $ZipName
    Invoke-WebRequest -Uri $Uri -OutFile $zip
    Expand-Archive -LiteralPath $zip -DestinationPath $Destination -Force
    Remove-Item -LiteralPath $zip -Force
}

function Install-Runtimes {
    if (-not [Environment]::Is64BitOperatingSystem) { throw 'TKBEN supports Windows x64 only.' }
    if ((Test-Path $NodeExe) -and ((& $NodeExe --version).TrimStart('v') -ne $NodeVersion)) {
        Write-Step "Replacing stale Node.js runtime with $NodeVersion"
        Remove-Item $NodeDir -Recurse -Force
    }
    if (-not (Test-Path $PythonExe)) {
        Write-Step "Downloading Python $PythonVersion x64"
        Expand-Zip "https://www.python.org/ftp/python/$PythonVersion/python-$PythonVersion-embed-amd64.zip" $PythonDir "tkben-python-$PID.zip"
        $pth = Join-Path $PythonDir 'python314._pth'
        (Get-Content $pth) -replace '^#import site$','import site' | Set-Content $pth -Encoding ascii
    }
    if (-not (Test-Path $UvExe)) {
        Write-Step 'Downloading uv x64'
        $tempUv = Join-Path $env:TEMP "tkben-uv-$PID"
        Expand-Zip 'https://github.com/astral-sh/uv/releases/latest/download/uv-x86_64-pc-windows-msvc.zip' $tempUv "tkben-uv-$PID.zip"
        Ensure-Directory $UvDir
        Copy-Item (Get-ChildItem $tempUv -Recurse -Filter uv.exe | Select-Object -First 1).FullName $UvExe
        Remove-Item -LiteralPath $tempUv -Recurse -Force
    }
    if (-not (Test-Path $NodeExe)) {
        Write-Step "Downloading Node.js $NodeVersion x64"
        $tempNode = Join-Path $env:TEMP "tkben-node-$PID"
        Expand-Zip "https://nodejs.org/dist/v$NodeVersion/node-v$NodeVersion-win-x64.zip" $tempNode "tkben-node-$PID.zip"
        Ensure-Directory $NodeDir
        Copy-Item (Join-Path $tempNode "node-v$NodeVersion-win-x64\*") $NodeDir -Recurse -Force
        Remove-Item -LiteralPath $tempNode -Recurse -Force
    }
}

function Sync-Dependencies {
    Set-TkbenEnvironment
    Install-Runtimes
    Write-Step 'Synchronizing locked Python dependencies'
    & $UvExe sync --frozen --extra test --python $PythonExe --project $ServerDir
    if ($LASTEXITCODE) { throw "uv sync failed with exit code $LASTEXITCODE" }
    Write-Step 'Installing locked frontend dependencies'
    Push-Location $ClientDir
    try { & $NpmCmd ci; if ($LASTEXITCODE) { throw "npm ci failed with exit code $LASTEXITCODE" } } finally { Pop-Location }
    Write-Ok 'Dependencies are synchronized.'
}

function Update-Dependencies {
    Set-TkbenEnvironment
    Install-Runtimes
    Push-Location $ServerDir
    try { & $UvExe lock --upgrade; if ($LASTEXITCODE) { throw 'uv lock --upgrade failed.' } } finally { Pop-Location }
    Push-Location $ClientDir
    try { & $NpmCmd update --package-lock-only; if ($LASTEXITCODE) { throw 'npm lockfile update failed.' } } finally { Pop-Location }
    Sync-Dependencies
}

function Get-OwnedProcesses {
    if (-not (Test-Path $PidFile)) { return @() }
    $records = @(Get-Content $PidFile -Raw | ConvertFrom-Json)
    foreach ($record in $records) {
        $process = Get-CimInstance Win32_Process -Filter "ProcessId = $($record.pid)" -ErrorAction SilentlyContinue
        if (-not $process) { continue }
        $legacyOwned = $process.CommandLine -and $process.CommandLine.Contains($RepoRoot, [StringComparison]::OrdinalIgnoreCase)
        $manifestOwned = $record.expectedExe -and $record.commandToken -and
            $process.ExecutablePath -and $process.ExecutablePath.Equals([string]$record.expectedExe, [StringComparison]::OrdinalIgnoreCase) -and
            $process.CommandLine -and $process.CommandLine.Contains([string]$record.commandToken, [StringComparison]::OrdinalIgnoreCase)
        if ($legacyOwned -or $manifestOwned) { $process }
    }
}

function Stop-Tkben {
    $owned = @(Get-OwnedProcesses)
    foreach ($process in ($owned | Sort-Object ProcessId -Descending)) {
        & taskkill.exe /PID $process.ProcessId /T /F | Out-Null
    }
    Remove-Item $PidFile -Force -ErrorAction SilentlyContinue
    Write-Ok "Stopped $($owned.Count) owned service process(es)."
}

function Read-EnvPort([string]$Name,[int]$Default) {
    $envFile = Join-Path $ConfigDir '.env'
    $line = Get-Content $envFile | Where-Object { $_ -match "^\s*$Name\s*=" } | Select-Object -Last 1
    if ($line -and ($line.Split('=',2)[1].Trim() -as [int])) { return [int]$line.Split('=',2)[1].Trim() }
    return $Default
}

function Wait-Http([string]$Uri,[int]$Seconds=60) {
    $deadline = (Get-Date).AddSeconds($Seconds)
    do {
        try { if ((Invoke-WebRequest -UseBasicParsing -Uri $Uri -TimeoutSec 2).StatusCode -lt 400) { return } } catch {}
        Start-Sleep -Milliseconds 500
    } while ((Get-Date) -lt $deadline)
    throw "Timed out waiting for $Uri. See $LogDir."
}

function Launch-Application {
    Set-TkbenEnvironment
    if ((Get-OwnedProcesses).Count) { throw 'TKBEN developer services are already running. Use Stop first.' }
    if (-not (Test-Path $VenvPython) -or -not (Test-Path (Join-Path $ClientDir 'node_modules'))) { Sync-Dependencies }
    & $VenvPython (Join-Path $AppDir 'scripts\initialize_database.py')
    if ($LASTEXITCODE) { throw 'Database initialization failed.' }
    $apiPort = Read-EnvPort 'FASTAPI_PORT' 5000
    $uiPort = Read-EnvPort 'UI_PORT' 8000
    foreach ($port in @($apiPort,$uiPort)) {
        if (Get-NetTCPConnection -State Listen -LocalPort $port -ErrorAction SilentlyContinue) { throw "Port $port is already in use; no process was terminated." }
    }
    $backendLog = Join-Path $LogDir 'developer-backend.log'
    $frontendLog = Join-Path $LogDir 'developer-frontend.log'
    Write-Step 'Starting backend'
    $backend = Start-Process -FilePath $VenvPython -ArgumentList @('-m','uvicorn','server.app:app','--app-dir','.','--host','127.0.0.1','--port',"$apiPort") -WorkingDirectory $AppDir -RedirectStandardOutput $backendLog -RedirectStandardError "$backendLog.err" -WindowStyle Hidden -PassThru
    try {
        Wait-Http "http://127.0.0.1:$apiPort/api/health"
        Write-Step 'Starting Vite developer server'
        $viteScript = Join-Path $ClientDir 'node_modules\vite\bin\vite.js'
        $frontend = Start-Process -FilePath $NodeExe -ArgumentList @("`"$viteScript`"",'--host','127.0.0.1','--port',"$uiPort",'--strictPort') -WorkingDirectory $ClientDir -RedirectStandardOutput $frontendLog -RedirectStandardError "$frontendLog.err" -WindowStyle Hidden -PassThru
        @(
            @{name='backend';pid=$backend.Id;expectedExe=$VenvPython;commandToken='server.app:app'},
            @{name='frontend';pid=$frontend.Id;expectedExe=$NodeExe;commandToken='vite.js'}
        ) | ConvertTo-Json | Set-Content $PidFile
        Wait-Http "http://127.0.0.1:$uiPort/"
    } catch {
        $startupError = $_
        if (Get-Process -Id $backend.Id -ErrorAction SilentlyContinue) { & cmd.exe /c "taskkill /PID $($backend.Id) /T /F >nul 2>&1" }
        throw $startupError
    }
    if (-not $NoBrowser) { Start-Process "http://127.0.0.1:$uiPort/" }
    Write-Ok "TKBEN is running at http://127.0.0.1:$uiPort/"
}

function Initialize-Database { Set-TkbenEnvironment; if (-not (Test-Path $VenvPython)) { Sync-Dependencies }; & $VenvPython (Join-Path $AppDir 'scripts\initialize_database.py'); if ($LASTEXITCODE) { throw 'Database initialization failed.' } }
function Run-Tests { Sync-Dependencies; & cmd.exe /c (Join-Path $AppDir 'tests\run_tests.bat'); if ($LASTEXITCODE) { throw "Tests failed with exit code $LASTEXITCODE" } }
function Build-Desktop { Sync-Dependencies; & cmd.exe /c (Join-Path $RepoRoot 'release\tauri\build_with_tauri.bat'); if ($LASTEXITCODE) { throw "Desktop build failed with exit code $LASTEXITCODE" } }
function Remove-Desktop { & powershell.exe -NoProfile -ExecutionPolicy Bypass -File (Join-Path $RepoRoot 'release\tauri\scripts\clean-tauri-build.ps1'); if ($LASTEXITCODE) { throw 'Desktop cleanup failed.' } }
function Clean-Generated { Stop-Tkben; @((Join-Path $ClientDir 'dist'),(Join-Path $ClientDir '.vite'),(Join-Path $ServerDir '.pytest_cache'),(Join-Path $ServerDir '.ruff_cache'),(Join-Path $RuntimeDir '.uv-cache')) | ForEach-Object { if(Test-Path $_){Remove-Item $_ -Recurse -Force} }; Get-ChildItem $AppDir -Recurse -Directory -Filter __pycache__ -ErrorAction SilentlyContinue | Remove-Item -Recurse -Force; Write-Ok 'Generated caches were removed.' }
function Repair-Environment { Stop-Tkben; @($VenvDir,(Join-Path $ClientDir 'node_modules')) | ForEach-Object { if(Test-Path $_){Remove-Item $_ -Recurse -Force} }; Sync-Dependencies; Initialize-Database }
function Uninstall-Developer { Stop-Tkben; if(-not $Yes -and (Read-Host 'Remove downloaded runtimes and dependency directories? [y/N]') -notmatch '^(y|yes)$'){return}; @($RuntimeDir,$VenvDir,(Join-Path $ClientDir 'node_modules'),(Join-Path $ClientDir 'dist')) | ForEach-Object { if(Test-Path $_){Remove-Item $_ -Recurse -Force} }; Ensure-Directory $RuntimeDir; New-Item (Join-Path $RuntimeDir '.gitkeep') -ItemType File -Force | Out-Null; Write-Ok 'Developer runtimes were removed; user data was preserved.' }

function Show-Diagnostics {
    Set-TkbenEnvironment
    Write-Host "Repository: $RepoRoot"
    Write-Host "Windows x64: $([Environment]::Is64BitOperatingSystem)"
    foreach($item in @($PythonExe,$UvExe,$NodeExe,$VenvPython,(Join-Path $ClientDir 'package-lock.json'),(Join-Path $ServerDir 'uv.lock'))){ Write-Host ("{0,-5} {1}" -f $(if(Test-Path $item){'OK'}else{'MISS'}),$item) }
    Write-Host "Owned processes: $(@(Get-OwnedProcesses).Count)"
    Write-Host "Data: $DataDir"
    Write-Host "Logs: $LogDir"
}

function Show-Logs { Initialize-UserLayout; $files=@(Get-ChildItem $LogDir -File -ErrorAction SilentlyContinue | Sort-Object LastWriteTime -Descending); if(-not $files){Write-Host 'No logs found.';return}; $files | Select-Object Name,Length,LastWriteTime | Format-Table; Write-Host "`nLatest log tail:"; Get-Content $files[0].FullName -Tail 80 }

function Show-Menu {
    $entries = [ordered]@{'1'='Launch';'2'='Install';'3'='Sync';'4'='Update';'5'='InitDatabase';'6'='Repair';'7'='Clean';'8'='Diagnostics';'9'='Logs';'10'='Stop';'11'='Test';'12'='BuildDesktop';'13'='RemoveDesktop';'14'='Uninstall';'0'='Exit'}
    $labels = @{'Launch'='Launch application';'Install'='Install or initialize application';'Sync'='Synchronize locked dependencies';'Update'='Update dependencies and lockfiles';'InitDatabase'='Initialize database';'Repair'='Reset or repair local environment';'Clean'='Clean generated files and caches';'Diagnostics'='Run diagnostics';'Logs'='View logs';'Stop'='Stop running application services';'Test'='Run test suite';'BuildDesktop'='Build Windows desktop release';'RemoveDesktop'='Remove desktop build artifacts';'Uninstall'='Uninstall developer runtimes';'Exit'='Exit'}
    while($true){
        Clear-Host
        Write-Host 'TKBEN Developer and Maintenance'
        foreach($key in $entries.Keys){Write-Host "$key. $($labels[$entries[$key]])"}
        $rawChoice = Read-Host 'Select an option'
        if ($null -eq $rawChoice) { return }
        $choice = $rawChoice.Trim()
        if($choice -eq '0'){return}
        if($entries.Contains($choice)){
            try { Invoke-Action $entries[$choice] } catch { Write-Host "[ERROR] $_" -ForegroundColor Red }
            Read-Host 'Press Enter to continue' | Out-Null
        }
    }
}

function Invoke-Action([string]$Name) {
    switch($Name){
        'Launch' { Launch-Application }; 'Install' { Sync-Dependencies; Initialize-Database }; 'Sync' { Sync-Dependencies }; 'Update' { Update-Dependencies }; 'InitDatabase' { Initialize-Database }; 'Repair' { Repair-Environment }; 'Clean' { Clean-Generated }; 'Diagnostics' { Show-Diagnostics }; 'Logs' { Show-Logs }; 'Stop' { Stop-Tkben }; 'Test' { Run-Tests }; 'BuildDesktop' { Build-Desktop }; 'RemoveDesktop' { Remove-Desktop }; 'Uninstall' { Uninstall-Developer }; default { throw "Unknown action: $Name" }
    }
}

if ($Action -eq 'Menu') { Show-Menu } else { Invoke-Action $Action }
