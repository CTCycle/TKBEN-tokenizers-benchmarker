# Copyright © 2023 Thomas Virdis
# Licensed under the MIT License.

#requires -Version 5.1
[CmdletBinding()]
param()

$ErrorActionPreference = 'Stop'
$runId = [guid]::NewGuid().ToString('N')
$fixtureRoot = Join-Path ([IO.Path]::GetTempPath()) "tkben-cleanup-$runId"
$repoRoot = Join-Path $fixtureRoot 'checkout'
$dataRoot = Join-Path $fixtureRoot 'external-data'
$outsideRoot = Join-Path $fixtureRoot 'outside'
$aclOriginal = $null
$aclPath = $null
$junctionPath = $null
$blockingProcess = $null
$passed = 0

function Assert-Test {
    param([bool]$Condition, [string]$Message)
    if (-not $Condition) { throw $Message }
}

function New-TestFile {
    param([Parameter(Mandatory)][string]$Path, [string]$Content = 'fixture')
    $parent = Split-Path -Parent $Path
    if ($parent) { [void][IO.Directory]::CreateDirectory($parent) }
    [IO.File]::WriteAllText($Path, $Content)
}

function New-TestDirectory {
    param([Parameter(Mandatory)][string]$Path)
    [void][IO.Directory]::CreateDirectory($Path)
}

function Get-TestPowerShellPath {
    $windowsPowerShell = Get-Command powershell.exe -ErrorAction SilentlyContinue
    if ($windowsPowerShell) { return $windowsPowerShell.Source }
    $currentPowerShell = Get-Command pwsh.exe -ErrorAction Stop
    return $currentPowerShell.Source
}

function Get-FixturePlan {
    param(
        [Parameter(Mandatory)][ValidateSet('Cache', 'Logs', 'Uninstall', 'AllData')][string]$Action,
        [string]$DataRootPath = $dataRoot,
        [string]$SqlitePath = (Join-Path $dataRoot 'database.db'),
        [bool]$ExternalDatabase = $false,
        [string]$PathsConfig = (Join-Path $repoRoot 'settings\paths-config.json')
    )
    $arguments = @{
        Action = $Action
        RepositoryRoot = $repoRoot
        RuntimeRoot = Join-Path $repoRoot 'runtimes'
        RuntimeCacheRoot = Join-Path $repoRoot 'runtimes\cache'
        ApplicationRoot = Join-Path $repoRoot 'app'
        ClientRoot = Join-Path $repoRoot 'app\client'
        ServerRoot = Join-Path $repoRoot 'app\server'
        DataRoot = $DataRootPath
        DatasetCacheRoot = Join-Path $DataRootPath 'sources\datasets'
        TokenizerDataRoot = Join-Path $DataRootPath 'sources\tokenizers'
        LogRoot = Join-Path $DataRootPath 'logs'
        TempRoot = Join-Path $outsideRoot 'temp'
        SettingsRoot = Join-Path $repoRoot 'settings'
        PathsConfigPath = $PathsConfig
        EnvironmentFilePath = Join-Path $repoRoot 'settings\.env'
        DatabaseConfigPath = Join-Path $dataRoot 'database-config.json'
        SqliteDatabasePath = $SqlitePath
        EncryptionKeyPath = Join-Path $dataRoot 'hf-key-material.json'
        ExternalDatabase = $ExternalDatabase
    }
    return Get-TkbenCleanupPlan @arguments
}

function Assert-ContainsPath {
    param([pscustomobject]$Plan, [string]$Path, [string]$Message)
    $fullPath = [IO.Path]::GetFullPath($Path)
    Assert-Test ([bool](@($Plan.Targets | Where-Object { $_.Path -eq $fullPath }).Count)) $Message
}

function Assert-NotContainsPath {
    param([pscustomobject]$Plan, [string]$Path, [string]$Message)
    $fullPath = [IO.Path]::GetFullPath($Path)
    Assert-Test (-not [bool](@($Plan.Targets | Where-Object { $_.Path -eq $fullPath }).Count)) $Message
}

function Invoke-LauncherDryRunFromOtherDirectory {
    param([int]$Selection, [string]$ExpectedTarget)
    $launcher = Join-Path $repoRoot 'start_on_windows.ps1'
    $powershellExe = Get-TestPowerShellPath
    $startInfo = New-Object Diagnostics.ProcessStartInfo
    $startInfo.FileName = $powershellExe
    $startInfo.Arguments = "-NoProfile -ExecutionPolicy Bypass -File `"$launcher`" -DryRun"
    $startInfo.WorkingDirectory = $fixtureRoot
    $startInfo.RedirectStandardInput = $true
    $startInfo.RedirectStandardOutput = $true
    $startInfo.RedirectStandardError = $true
    $startInfo.UseShellExecute = $false
    $startInfo.CreateNoWindow = $true
    $startInfo.EnvironmentVariables['TKBEN_PATHS_CONFIG'] = Join-Path $repoRoot 'settings\paths-config.json'
    $process = [Diagnostics.Process]::Start($startInfo)
    try {
        $outputRead = $process.StandardOutput.ReadToEndAsync()
        $errorRead = $process.StandardError.ReadToEndAsync()
        $process.StandardInput.WriteLine([string]$Selection)
        $process.StandardInput.Close()
        if (-not $process.WaitForExit(120000)) {
            $process.Kill()
            $process.WaitForExit()
            throw "The launcher dry-run did not exit for menu selection $Selection. Output: $($outputRead.Result) $($errorRead.Result)"
        }
        $output = $outputRead.Result + $errorRead.Result
        Assert-Test ($process.ExitCode -eq 0) "Launcher dry-run exited with $($process.ExitCode): $output"
        Assert-Test ($output.Contains([IO.Path]::GetFullPath($ExpectedTarget))) "Dry-run selection $Selection did not print the full target path '$ExpectedTarget'. Output: $output"
        Assert-Test ($output.Contains([IO.Path]::GetFullPath($dataRoot))) 'Dry-run omitted the resolved configured data root.'
        Assert-Test ($output -match 'Dry run complete') 'Dry-run did not report its no-change completion.'
        Assert-Test (Test-Path -LiteralPath $ExpectedTarget) 'Launcher dry-run changed a targeted file.'
    }
    finally {
        if (-not $process.HasExited) { $process.Kill() }
        $process.Dispose()
    }
}

function Assert-LauncherRejectsDryRunFlags {
    param([Parameter(Mandatory)][string[]]$Arguments, [Parameter(Mandatory)][string]$ExpectedMessage)
    $launcher = Join-Path $repoRoot 'start_on_windows.ps1'
    $startInfo = New-Object Diagnostics.ProcessStartInfo
    $startInfo.FileName = Get-TestPowerShellPath
    $startInfo.Arguments = "-NoProfile -ExecutionPolicy Bypass -File `"$launcher`" $($Arguments -join ' ')"
    $startInfo.WorkingDirectory = $fixtureRoot
    $startInfo.RedirectStandardOutput = $true
    $startInfo.RedirectStandardError = $true
    $startInfo.UseShellExecute = $false
    $startInfo.CreateNoWindow = $true
    $startInfo.EnvironmentVariables['TKBEN_PATHS_CONFIG'] = Join-Path $repoRoot 'settings\paths-config.json'
    $process = [Diagnostics.Process]::Start($startInfo)
    try {
        $outputRead = $process.StandardOutput.ReadToEndAsync()
        $errorRead = $process.StandardError.ReadToEndAsync()
        if (-not $process.WaitForExit(30000)) {
            $process.Kill()
            $process.WaitForExit()
            throw "The launcher did not reject $($Arguments -join ' ')."
        }
        $output = $outputRead.Result + $errorRead.Result
        Assert-Test ($process.ExitCode -ne 0) "Launcher accepted incompatible arguments $($Arguments -join ' ')."
        Assert-Test ($output.Contains($ExpectedMessage)) "Launcher rejection did not explain the incompatible arguments: $output"
    }
    finally {
        if (-not $process.HasExited) { $process.Kill() }
        $process.Dispose()
    }
}

try {
    $moduleSource = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..\..\..\scripts\maintenance_cleanup.psm1')).Path
    $launcherSource = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..\..\..\..\start_on_windows.ps1')).Path
    $modulePath = Join-Path $repoRoot 'app\scripts\maintenance_cleanup.psm1'
    [void][IO.Directory]::CreateDirectory((Join-Path $repoRoot 'app\scripts'))
    [void][IO.Directory]::CreateDirectory((Join-Path $repoRoot 'app\client'))
    [void][IO.Directory]::CreateDirectory((Join-Path $repoRoot 'app\server'))
    [void][IO.Directory]::CreateDirectory((Join-Path $repoRoot 'app\tests'))
    [void][IO.Directory]::CreateDirectory((Join-Path $repoRoot 'settings'))
    [void][IO.Directory]::CreateDirectory((Join-Path $repoRoot 'runtimes\cache'))
    New-TestFile (Join-Path $repoRoot 'runtimes\cache\.gitkeep')
    [void][IO.Directory]::CreateDirectory((Join-Path $outsideRoot 'temp'))
    [void][IO.Directory]::CreateDirectory((Join-Path $dataRoot 'sources\datasets'))
    [void][IO.Directory]::CreateDirectory((Join-Path $dataRoot 'sources\tokenizers'))
    [void][IO.Directory]::CreateDirectory((Join-Path $dataRoot 'logs'))
    [void][IO.Directory]::CreateDirectory($outsideRoot)
    Copy-Item -LiteralPath $moduleSource -Destination $modulePath
    Copy-Item -LiteralPath $launcherSource -Destination (Join-Path $repoRoot 'start_on_windows.ps1')
    Import-Module -Name $modulePath -Force
    New-TestFile (Join-Path $repoRoot 'settings\paths-config.json') '{"data_dir":"../external-data","log_dir":"../external-data/logs"}'

    # Confirm every filesystem action supports -DryRun from a different working directory.
    $dryRunTargets = @(
        [pscustomobject]@{ Selection = 8; Path = Join-Path $dataRoot 'logs\TKBEN_dry-run.log' },
        [pscustomobject]@{ Selection = 9; Path = Join-Path $repoRoot 'runtimes\cache\dry-run-sentinel.tmp' },
        [pscustomobject]@{ Selection = 10; Path = Join-Path $dataRoot 'database.db' },
        [pscustomobject]@{ Selection = 11; Path = Join-Path $repoRoot 'runtimes\python\python.exe' }
    )
    foreach ($target in $dryRunTargets) {
        New-TestFile $target.Path
        Invoke-LauncherDryRunFromOtherDirectory -Selection $target.Selection -ExpectedTarget $target.Path
    }
    Assert-LauncherRejectsDryRunFlags -Arguments @('-DryRun', '-Launch') -ExpectedMessage 'Use -DryRun only with the maintenance menu'
    Assert-LauncherRejectsDryRunFlags -Arguments @('-DryRun', '-KillAll') -ExpectedMessage 'Use -DryRun only with the maintenance menu'
    $passed++

    # Nested, hidden backend/frontend caches and generated outputs are discovered.
    $cacheFiles = @(
        (Join-Path $repoRoot 'runtimes\cache\pytest-state\state.bin'),
        (Join-Path $repoRoot 'pytest-cache-files-stale\state.bin'),
        (Join-Path $repoRoot 'app\server\.ruff_cache\nested\state.bin'),
        (Join-Path $repoRoot 'app\server\.ruff_cache\.gitkeep'),
        (Join-Path $repoRoot 'app\server\build\TKBEN.whl'),
        (Join-Path $repoRoot 'app\server\dist\TKBEN.whl'),
        (Join-Path $repoRoot 'app\server\TKBEN.egg-info\PKG-INFO'),
        (Join-Path $repoRoot 'app\server\pkg\__pycache__\module.pyc'),
        (Join-Path $repoRoot 'app\client\node_modules\.cache\bundle\state.bin'),
        (Join-Path $repoRoot 'app\client\node_modules\.vite\deps\state.bin'),
        (Join-Path $repoRoot 'app\client\coverage\lcov.info'),
        (Join-Path $repoRoot 'app\client\.angular\cache\metadata.bin'),
        (Join-Path $repoRoot 'app\client\dist\tkben-angular\index.html'),
        (Join-Path $dataRoot 'sources\datasets\hf-source\cache.arrow'),
        (Join-Path $dataRoot '.database-config.json.partial.tmp'),
        (Join-Path $dataRoot 'hf-key-material.json.tmp'),
        (Join-Path $dataRoot 'sources\tokenizers\custom\.tokenizer.json.partial.tmp'),
        (Join-Path $repoRoot 'settings\paths-config.json.tmp')
    )
    $temporaryRuntimeFiles = @(
        (Join-Path $outsideRoot 'temp\tkben_freq_fixture.sqlite3'),
        (Join-Path $outsideRoot 'temp\tkben_freq_fixture.sqlite3-wal'),
        (Join-Path $outsideRoot 'temp\tkben-benchmark-spool-fixture.ndjson'),
        (Join-Path $outsideRoot 'temp\unrelated.sqlite3')
    )
    foreach ($path in $cacheFiles) { New-TestFile $path }
    foreach ($path in $temporaryRuntimeFiles) { New-TestFile $path }
    New-TestFile (Join-Path $dataRoot 'database.db')
    New-TestFile (Join-Path $dataRoot 'database-config.json') '{"backend":"sqlite"}'
    New-TestFile (Join-Path $dataRoot 'hf-key-material.json') '{"key":"persistent"}'
    New-TestFile (Join-Path $dataRoot 'sources\tokenizers\custom\tokenizer.json')
    New-TestFile (Join-Path $repoRoot 'settings\.env')
    New-TestFile (Join-Path $repoRoot 'settings\paths-config.json') '{"data_dir":"../external-data","log_dir":"../external-data/logs"}'
    New-TestFile (Join-Path $dataRoot 'keep-user-file.txt')
    New-TestFile (Join-Path $dataRoot 'logs\other-app.log')
    $cachePlan = Get-FixturePlan -Action Cache
    foreach ($path in $cacheFiles) { Assert-ContainsPath $cachePlan $path "Cache target was not discovered: $path" }
    foreach ($path in @($temporaryRuntimeFiles | Where-Object { $_ -notmatch 'unrelated' })) {
        Assert-ContainsPath $cachePlan $path "Recognized application temporary file was not discovered: $path"
    }
    Assert-NotContainsPath $cachePlan (Join-Path $outsideRoot 'temp\unrelated.sqlite3') 'Cache plan included an unrelated temporary database.'
    Assert-NotContainsPath $cachePlan (Join-Path $dataRoot 'database.db') 'Cache plan included the database.'
    Assert-NotContainsPath $cachePlan (Join-Path $dataRoot 'database-config.json') 'Cache plan included persistent database configuration.'
    Assert-NotContainsPath $cachePlan (Join-Path $dataRoot 'hf-key-material.json') 'Cache plan included persistent key material.'
    Assert-NotContainsPath $cachePlan (Join-Path $repoRoot 'settings\paths-config.json') 'Cache plan included persistent path configuration.'
    Assert-NotContainsPath $cachePlan (Join-Path $dataRoot 'sources\tokenizers\custom\tokenizer.json') 'Cache plan included persistent tokenizer data.'
    Assert-NotContainsPath $cachePlan (Join-Path $repoRoot 'settings\.env') 'Cache plan included bootstrap settings.'
    $dryResult = Invoke-TkbenCleanupPlan -Plan $cachePlan -DryRun
    Assert-Test ($dryResult.Removed.Count -eq 0) 'Module dry-run reported a deletion.'
    foreach ($path in $cacheFiles) { Assert-Test (Test-Path -LiteralPath $path) "Module dry-run changed $path" }
    foreach ($path in $temporaryRuntimeFiles) { Assert-Test (Test-Path -LiteralPath $path) "Module dry-run changed $path" }
    $cacheResult = Invoke-TkbenCleanupPlan -Plan $cachePlan
    Assert-Test ($cacheResult.Failed.Count -eq 0) 'Cache cleanup failed in the fixture.'
    foreach ($path in $cacheFiles) { Assert-Test (-not (Test-Path -LiteralPath $path)) "Cache remained after cleanup: $path" }
    foreach ($path in @($temporaryRuntimeFiles | Where-Object { $_ -notmatch 'unrelated' })) {
        Assert-Test (-not (Test-Path -LiteralPath $path)) "Recognized application temporary file remained: $path"
    }
    Assert-Test (Test-Path -LiteralPath (Join-Path $outsideRoot 'temp\unrelated.sqlite3')) 'Cache cleanup deleted an unrelated temporary file.'
    Assert-Test (Test-Path -LiteralPath (Join-Path $dataRoot 'database.db')) 'Cache cleanup removed the database.'
    Assert-Test (Test-Path -LiteralPath (Join-Path $repoRoot 'runtimes\cache\.gitkeep')) 'Cache cleanup removed the managed cache-root marker.'
    Assert-Test (Test-Path -LiteralPath (Join-Path $dataRoot 'database-config.json')) 'Cache cleanup removed persistent database configuration.'
    Assert-Test (Test-Path -LiteralPath (Join-Path $dataRoot 'hf-key-material.json')) 'Cache cleanup removed persistent key material.'
    Assert-Test (Test-Path -LiteralPath (Join-Path $repoRoot 'settings\paths-config.json')) 'Cache cleanup removed persistent path configuration.'
    Assert-Test (Test-Path -LiteralPath (Join-Path $dataRoot 'sources\tokenizers\custom\tokenizer.json')) 'Cache cleanup removed persistent tokenizer data.'
    Assert-Test (Test-Path -LiteralPath (Join-Path $repoRoot 'settings\.env')) 'Cache cleanup removed settings.'
    $repeatResult = Invoke-TkbenCleanupPlan -Plan (Get-FixturePlan -Action Cache)
    Assert-Test ($repeatResult.Failed.Count -eq 0) 'Repeated cache cleanup failed.'
    $passed++

    $vanishingCacheFile = Join-Path $repoRoot 'app\client\.vite\gone-after-preview.bin'
    New-TestFile $vanishingCacheFile
    $vanishingPlan = Get-FixturePlan -Action Cache
    Remove-Item -LiteralPath $vanishingCacheFile -Force
    $vanishingResult = Invoke-TkbenCleanupPlan -Plan $vanishingPlan
    Assert-Test ($vanishingResult.Failed.Count -eq 0) 'A path removed after preview caused cleanup to fail.'
    Assert-Test (@($vanishingResult.Missing | Where-Object { $_ -eq [IO.Path]::GetFullPath($vanishingCacheFile) }).Count -eq 1) 'A path removed after preview was not reported as a missing no-op.'
    $passed++

    # Locked files are reported, preserve partial results, then clean on retry.
    $lockedFile = Join-Path $repoRoot 'app\server\.ruff_cache\locked.bin'
    New-TestFile $lockedFile
    $lockedHandle = [IO.File]::Open($lockedFile, [IO.FileMode]::Open, [IO.FileAccess]::ReadWrite, [IO.FileShare]::None)
    try {
        $lockedResult = Invoke-TkbenCleanupPlan -Plan (Get-FixturePlan -Action Cache)
        Assert-Test ($lockedResult.Failed.Count -gt 0) 'Locked files were not reported as a cleanup failure.'
        Assert-Test (Test-Path -LiteralPath $lockedFile) 'Cleanup deleted a locked file.'
    }
    finally { $lockedHandle.Dispose() }
    $retryResult = Invoke-TkbenCleanupPlan -Plan (Get-FixturePlan -Action Cache)
    Assert-Test ($retryResult.Failed.Count -eq 0) 'Cleanup retry failed after releasing a locked file.'
    Assert-Test (-not (Test-Path -LiteralPath $lockedFile)) 'Cleanup retry left the released file behind.'
    $passed++

    # ACL failures are surfaced when the host can enforce a deny rule.
    $aclPath = Join-Path $repoRoot 'app\server\.mypy_cache'
    New-TestFile (Join-Path $aclPath 'blocked.bin')
    try {
        $aclOriginal = Get-Acl -LiteralPath $aclPath
        $identity = [Security.Principal.WindowsIdentity]::GetCurrent().User
        $denyRule = [Security.AccessControl.FileSystemAccessRule]::new($identity, [Security.AccessControl.FileSystemRights]::ReadAndExecute, [Security.AccessControl.AccessControlType]::Deny)
        $deniedAcl = Get-Acl -LiteralPath $aclPath
        $deniedAcl.AddAccessRule($denyRule)
        Set-Acl -LiteralPath $aclPath -AclObject $deniedAcl
        $accessDenied = $false
        try { Get-ChildItem -LiteralPath $aclPath -ErrorAction Stop | Out-Null }
        catch [System.UnauthorizedAccessException] { $accessDenied = $true }
        catch [System.Management.Automation.ItemNotFoundException] { $accessDenied = $true }
        if ($accessDenied) {
            $aclResult = Invoke-TkbenCleanupPlan -Plan (Get-FixturePlan -Action Cache)
            Assert-Test (($aclResult.Skipped.Count + $aclResult.Failed.Count) -gt 0) 'Access-denied cleanup was not reported.'
        }
        else {
            Write-Host '[SKIP] Host ACL policy did not enforce the synthetic deny rule.' -ForegroundColor Yellow
        }
    }
    catch [System.Security.SecurityException] {
        Write-Host '[SKIP] Host policy did not permit the synthetic ACL failure fixture.' -ForegroundColor Yellow
    }
    finally {
        if ($null -ne $aclOriginal -and (Test-Path -LiteralPath $aclPath)) {
            Set-Acl -LiteralPath $aclPath -AclObject $aclOriginal
        }
    }
    $passed++

    # A junction is removed as a link; the outside target remains untouched.
    $outsideSentinel = Join-Path $outsideRoot 'keep.txt'
    New-TestFile $outsideSentinel
    $junctionPath = Join-Path $repoRoot 'app\server\.ruff_cache\outside-link'
    try {
        [void][IO.Directory]::CreateDirectory((Split-Path -Parent $junctionPath))
        New-Item -ItemType Junction -Path $junctionPath -Target $outsideRoot | Out-Null
        $linkPlan = Get-FixturePlan -Action Cache
        Assert-ContainsPath $linkPlan $junctionPath 'Cleanup plan omitted the junction entry.'
        $linkResult = Invoke-TkbenCleanupPlan -Plan $linkPlan
        Assert-Test ($linkResult.Failed.Count -eq 0) 'Cleanup failed to remove a junction entry.'
        Assert-Test (Test-Path -LiteralPath $outsideSentinel) 'Cleanup followed a junction and deleted its outside target.'
    }
    catch {
        if (Test-Path -LiteralPath $outsideSentinel) {
            Write-Host '[SKIP] Host policy did not permit creating a junction fixture.' -ForegroundColor Yellow
        }
        else { throw }
    }
    finally {
        if (Test-Path -LiteralPath $junctionPath) { [IO.Directory]::Delete($junctionPath, $false) }
    }
    $passed++

    # Standard uninstall removes software/logs but retains persistent local data.
    $installFiles = @(
        (Join-Path $repoRoot 'runtimes\python\python.exe'),
        (Join-Path $repoRoot 'app\server\.venv\Scripts\python.exe'),
        (Join-Path $repoRoot '.venv\Scripts\python.exe'),
        (Join-Path $repoRoot 'app\client\node_modules\angular\package.json'),
        (Join-Path $repoRoot 'app\client\dist\tkben-angular\index.html'),
        (Join-Path $dataRoot 'logs\TKBEN_sample.log')
    )
    foreach ($path in $installFiles) { New-TestFile $path }
    $uninstallTemp = Join-Path $outsideRoot 'temp\tkben-benchmark-spool-uninstall.ndjson'
    $uninstallWrite = Join-Path $dataRoot 'sources\tokenizers\custom\.tokenizer.json.uninstall.tmp'
    New-TestFile $uninstallTemp
    New-TestFile $uninstallWrite
    $sqlitePath = Join-Path $outsideRoot 'configured.sqlite'
    New-TestFile $sqlitePath
    $uninstallPlan = Get-FixturePlan -Action Uninstall -SqlitePath $sqlitePath
    Assert-ContainsPath $uninstallPlan $uninstallTemp 'Standard uninstall omitted an application temporary file.'
    Assert-ContainsPath $uninstallPlan $uninstallWrite 'Standard uninstall omitted an interrupted application write.'
    $uninstallResult = Invoke-TkbenCleanupPlan -Plan $uninstallPlan
    Assert-Test ($uninstallResult.Failed.Count -eq 0) 'Standard uninstall failed in the fixture.'
    foreach ($path in $installFiles) { Assert-Test (-not (Test-Path -LiteralPath $path)) "Uninstall left an installation artifact: $path" }
    Assert-Test (-not (Test-Path -LiteralPath $uninstallTemp)) 'Uninstall left a recognized temporary file behind.'
    Assert-Test (-not (Test-Path -LiteralPath $uninstallWrite)) 'Uninstall left an interrupted application write behind.'
    Assert-Test (Test-Path -LiteralPath $sqlitePath) 'Standard uninstall removed the configured database.'
    Assert-Test (Test-Path -LiteralPath (Join-Path $dataRoot 'sources\tokenizers\custom\tokenizer.json')) 'Standard uninstall removed persistent tokenizer data.'
    Assert-Test (Test-Path -LiteralPath (Join-Path $dataRoot 'keep-user-file.txt')) 'Standard uninstall removed an unrecognized data file.'
    Assert-Test (Test-Path -LiteralPath (Join-Path $dataRoot 'logs\other-app.log')) 'Standard uninstall removed an unrelated log.'
    $passed++

    # Full local-data removal includes exact SQLite sidecars/configs, not unknown files.
    New-TestFile (Join-Path $dataRoot 'database-config.json') ('{"backend":"sqlite","sqlite_path":"' + $sqlitePath.Replace('\', '\\') + '"}')
    New-TestFile (Join-Path $dataRoot '.database-config.json.test.tmp') '{"backend":"sqlite"}'
    New-TestFile "$sqlitePath-wal"
    New-TestFile "$sqlitePath-shm"
    New-TestFile "$sqlitePath-journal"
    New-TestFile (Join-Path $dataRoot 'hf-key-material.json') '{"key":"local"}'
    New-TestFile (Join-Path $dataRoot 'hf-key-material.json.tmp') '{"key":"local-temp"}'
    New-TestFile (Join-Path $dataRoot 'sources\datasets\stale-download\cache.bin')
    New-TestFile (Join-Path $dataRoot 'sources\tokenizers\custom\tokenizer.json') '{"version":"1"}'
    New-TestFile (Join-Path $dataRoot 'logs\TKBEN_full-removal.log')
    New-TestFile (Join-Path $repoRoot 'settings\paths-config.json') '{"data_dir":"external-data"}'
    New-TestFile (Join-Path $repoRoot 'settings\paths-config.json.tmp') '{"data_dir":"staged"}'
    New-TestFile (Join-Path $repoRoot 'settings\.env.example') 'template'
    $completeRemovalCache = Join-Path $repoRoot 'runtimes\cache\all-data-cache.tmp'
    New-TestFile $completeRemovalCache
    $allDataPlan = Get-FixturePlan -Action AllData -SqlitePath $sqlitePath
    Assert-ContainsPath $allDataPlan $completeRemovalCache 'Complete local-data removal omitted disposable application caches.'
    foreach ($path in @(
        $sqlitePath,
        "$sqlitePath-wal",
        "$sqlitePath-shm",
        "$sqlitePath-journal",
        (Join-Path $dataRoot 'database-config.json'),
        (Join-Path $dataRoot 'hf-key-material.json'),
        (Join-Path $dataRoot '.database-config.json.test.tmp'),
        (Join-Path $dataRoot 'hf-key-material.json.tmp'),
        (Join-Path $repoRoot 'settings\paths-config.json'),
        (Join-Path $repoRoot 'settings\paths-config.json.tmp'),
        (Join-Path $repoRoot 'settings\.env'),
        (Join-Path $dataRoot 'sources\tokenizers\custom\tokenizer.json'),
        (Join-Path $dataRoot 'logs\TKBEN_full-removal.log')
    )) { Assert-ContainsPath $allDataPlan $path "Complete removal omitted an app-owned path: $path" }
    Assert-NotContainsPath $allDataPlan (Join-Path $dataRoot 'keep-user-file.txt') 'Complete removal included an unrecognized data file.'
    $allDataResult = Invoke-TkbenCleanupPlan -Plan $allDataPlan
    Assert-Test ($allDataResult.Failed.Count -eq 0) 'Complete local-data removal failed in the fixture.'
    foreach ($path in @($sqlitePath, "$sqlitePath-wal", "$sqlitePath-shm", "$sqlitePath-journal", (Join-Path $dataRoot 'database-config.json'), (Join-Path $dataRoot '.database-config.json.test.tmp'), (Join-Path $dataRoot 'hf-key-material.json'), (Join-Path $dataRoot 'hf-key-material.json.tmp'), (Join-Path $repoRoot 'settings\paths-config.json'), (Join-Path $repoRoot 'settings\paths-config.json.tmp'), (Join-Path $repoRoot 'settings\.env'), (Join-Path $dataRoot 'sources\tokenizers\custom\tokenizer.json'), (Join-Path $dataRoot 'logs\TKBEN_full-removal.log'))) {
        Assert-Test (-not (Test-Path -LiteralPath $path)) "Complete removal left a targeted file: $path"
    }
    Assert-Test (Test-Path -LiteralPath (Join-Path $repoRoot 'settings\.env.example')) 'Complete removal deleted the environment template.'
    Assert-Test (-not (Test-Path -LiteralPath $completeRemovalCache)) 'Complete removal left disposable application caches behind.'
    Assert-Test (Test-Path -LiteralPath (Join-Path $dataRoot 'keep-user-file.txt')) 'Complete removal deleted an unrecognized data file.'
    $passed++

    $unsafeDataRootRejected = $false
    try { $null = Get-FixturePlan -Action Cache -DataRootPath $repoRoot }
    catch { $unsafeDataRootRejected = $_.Exception.Message -match 'Refusing a data root' }
    Assert-Test $unsafeDataRootRejected 'Cleanup accepted a data root overlapping the repository.'
    $passed++

    # External PostgreSQL is never contacted and its local key remains available.
    New-TestFile (Join-Path $dataRoot 'database-config.json') '{"backend":"postgresql"}'
    New-TestFile (Join-Path $dataRoot 'hf-key-material.json') '{"key":"external-db-key"}'
    New-TestFile (Join-Path $dataRoot 'hf-key-material.json.tmp') '{"key":"staged-external-db-key"}'
    New-TestFile $sqlitePath
    $pgPlan = Get-FixturePlan -Action AllData -ExternalDatabase $true -SqlitePath $sqlitePath
    Assert-NotContainsPath $pgPlan (Join-Path $dataRoot 'hf-key-material.json') 'PostgreSQL cleanup plan included key material required by the external database.'
    Assert-NotContainsPath $pgPlan (Join-Path $dataRoot 'hf-key-material.json.tmp') 'PostgreSQL cleanup plan included staged key material required by the external database.'
    Assert-NotContainsPath $pgPlan $sqlitePath 'PostgreSQL cleanup plan included an unrelated SQLite path.'
    $pgResult = Invoke-TkbenCleanupPlan -Plan $pgPlan
    Assert-Test ($pgResult.Failed.Count -eq 0) 'Local PostgreSQL-configuration cleanup failed.'
    Assert-Test (Test-Path -LiteralPath (Join-Path $dataRoot 'hf-key-material.json')) 'External PostgreSQL key material was removed.'
    Assert-Test (Test-Path -LiteralPath (Join-Path $dataRoot 'hf-key-material.json.tmp')) 'External PostgreSQL staged key material was removed.'
    Assert-Test (Test-Path -LiteralPath $sqlitePath) 'External PostgreSQL cleanup removed an unrelated SQLite database.'
    $passed++

    # Maintenance process detection uses both repository identity and command type.
    $powershellExe = Get-TestPowerShellPath
    $command = "Start-Sleep -Seconds 30; Write-Output '$repoRoot ng serve'"
    $blockingProcess = Start-Process -FilePath $powershellExe -ArgumentList @('-NoProfile', '-Command', $command) -PassThru -WindowStyle Hidden
    Start-Sleep -Milliseconds 500
    $syntheticProcess = [pscustomobject]@{ ProcessId = $blockingProcess.Id; CommandLine = $command }
    $syntheticLauncher = [pscustomobject]@{
        ProcessId = $blockingProcess.Id + 100000
        CommandLine = "powershell.exe $repoRoot\start_on_windows.ps1 -Launch"
    }
    $activeIds = @(Get-TkbenProcessIdsFromSnapshot -RepositoryRoot $repoRoot -Processes @($syntheticProcess, $syntheticLauncher))
    Assert-Test ($activeIds -contains $blockingProcess.Id) 'Active application-process detection missed the checkout-owned server-like process.'
    Assert-Test ($activeIds -contains $syntheticLauncher.ProcessId) 'Active application-process detection missed a concurrent checkout launcher.'
    $refusedActiveCleanup = $false
    try { Assert-TkbenNoActiveProcesses -ProcessIds $activeIds }
    catch { $refusedActiveCleanup = $_.Exception.Message -match [string]$blockingProcess.Id }
    Assert-Test $refusedActiveCleanup 'The cleanup process guard did not refuse an active application process.'
    Assert-Test (-not $blockingProcess.HasExited) 'The cleanup process guard stopped the active application process.'
    try {
        $observedIds = @(Get-TkbenMaintenanceProcessIds -RepositoryRoot $repoRoot)
        Assert-Test ($observedIds -contains $blockingProcess.Id) 'The live process scan missed the checkout-owned server-like process.'
    }
    catch {
        if ($_.Exception.Message -match 'Could not verify that cleanup targets are not in use') {
            Write-Host '[SKIP] Host CIM policy did not permit a live process-table read; snapshot filtering and refusal were tested.' -ForegroundColor Yellow
        }
        else { throw }
    }
    $passed++

    Write-Host "PASS: $passed launcher cleanup integration checks completed." -ForegroundColor Green
}
finally {
    if ($null -ne $blockingProcess -and -not $blockingProcess.HasExited) {
        Stop-Process -Id $blockingProcess.Id -Force -ErrorAction SilentlyContinue
    }
    if ($null -ne $aclOriginal -and $aclPath -and (Test-Path -LiteralPath $aclPath)) {
        try { Set-Acl -LiteralPath $aclPath -AclObject $aclOriginal } catch { }
    }
    if ($junctionPath -and (Test-Path -LiteralPath $junctionPath)) {
        try { [IO.Directory]::Delete($junctionPath, $false) } catch { }
    }
    if (Test-Path -LiteralPath $fixtureRoot) {
        Remove-Item -LiteralPath $fixtureRoot -Recurse -Force -ErrorAction SilentlyContinue
    }
    Remove-Module maintenance_cleanup -ErrorAction SilentlyContinue
}
