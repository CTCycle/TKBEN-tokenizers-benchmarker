# Copyright © 2023 Thomas Virdis
# Licensed under the MIT License.

Set-StrictMode -Version Latest

function Get-TkbenFullPath {
    param([Parameter(Mandatory)][string]$Path)
    if ([string]::IsNullOrWhiteSpace($Path)) { throw 'Refusing an empty cleanup path.' }
    return [IO.Path]::GetFullPath($Path)
}

function Test-TkbenPathWithin {
    param(
        [Parameter(Mandatory)][string]$Path,
        [Parameter(Mandatory)][string]$Root
    )
    $candidate = (Get-TkbenFullPath $Path).TrimEnd('\')
    $boundary = (Get-TkbenFullPath $Root).TrimEnd('\')
    return $candidate.Equals($boundary, [StringComparison]::OrdinalIgnoreCase) -or
        $candidate.StartsWith($boundary + '\', [StringComparison]::OrdinalIgnoreCase)
}

function Test-TkbenNoReparseAncestor {
    param([Parameter(Mandatory)][string]$Path)
    $candidate = Get-TkbenFullPath $Path
    $current = $candidate
    while ($current) {
        try {
            $item = Get-Item -LiteralPath $current -Force -ErrorAction Stop
            if (($item.Attributes -band [IO.FileAttributes]::ReparsePoint) -ne 0) {
                return $false
            }
        }
        catch [System.Management.Automation.ItemNotFoundException] { }
        catch [System.IO.FileNotFoundException] { }
        catch { return $false }
        $parent = [IO.Path]::GetDirectoryName($current)
        if (-not $parent -or $parent -eq $current) { break }
        $current = $parent
    }
    return $true
}

function Add-TkbenPlanTarget {
    param(
        [Parameter(Mandatory)][pscustomobject]$Plan,
        [Parameter(Mandatory)][string]$Path,
        [Parameter(Mandatory)][string]$Kind,
        [Parameter(Mandatory)][string]$Reason,
        [string]$Boundary,
        [switch]$ExplicitExternalFile
    )

    try { $fullPath = Get-TkbenFullPath $Path }
    catch {
        [void]$Plan.Skipped.Add([pscustomobject]@{ Path = $Path; Reason = $_.Exception.Message })
        return
    }
    $normalized = $fullPath.TrimEnd('\')
    $volumeRoot = [IO.Path]::GetPathRoot($fullPath).TrimEnd('\')
    if (-not $normalized -or $normalized -eq $volumeRoot) {
        [void]$Plan.Skipped.Add([pscustomobject]@{ Path = $fullPath; Reason = 'Filesystem roots are never valid targets.' })
        return
    }
    if ($Boundary -and -not (Test-TkbenPathWithin -Path $fullPath -Root $Boundary)) {
        [void]$Plan.Skipped.Add([pscustomobject]@{ Path = $fullPath; Reason = "Outside approved application root '$Boundary'." })
        return
    }
    if (-not $Boundary -and -not $ExplicitExternalFile) {
        [void]$Plan.Skipped.Add([pscustomobject]@{ Path = $fullPath; Reason = 'No approved application root was supplied.' })
        return
    }
    if ($Plan.TargetIndex.ContainsKey($fullPath)) { return }

    try {
        $item = Get-Item -LiteralPath $fullPath -Force -ErrorAction Stop
    }
    catch {
        if ($_.CategoryInfo.Category -eq [System.Management.Automation.ErrorCategory]::ObjectNotFound -or
            $_.Exception -is [System.IO.FileNotFoundException] -or
            $_.Exception -is [System.IO.DirectoryNotFoundException]) {
            [void]$Plan.Missing.Add($fullPath)
        }
        else {
            [void]$Plan.Skipped.Add([pscustomobject]@{ Path = $fullPath; Reason = $_.Exception.Message })
        }
        return
    }
    if ($Kind -eq 'File' -and $item.PSIsContainer) {
        [void]$Plan.Skipped.Add([pscustomobject]@{ Path = $fullPath; Reason = 'Expected a file but found a directory.' })
        return
    }
    if (-not (Test-TkbenNoReparseAncestor -Path $fullPath)) {
        $isLink = ($item.Attributes -band [IO.FileAttributes]::ReparsePoint) -ne 0
        if (-not $isLink) {
            [void]$Plan.Skipped.Add([pscustomobject]@{ Path = $fullPath; Reason = 'A parent path is a symbolic link or junction.' })
            return
        }
    }

    $target = [pscustomobject]@{
        Path = $fullPath
        Kind = if ($item.PSIsContainer) { 'Directory' } else { 'File' }
        IsReparsePoint = (($item.Attributes -band [IO.FileAttributes]::ReparsePoint) -ne 0)
        Boundary = if ($Boundary) { Get-TkbenFullPath $Boundary } else { $null }
        AllowsExternalPath = [bool]$ExplicitExternalFile
        Reason = $Reason
    }
    $Plan.TargetIndex[$fullPath] = $true
    [void]$Plan.Targets.Add($target)
}

function Add-TkbenTreeToPlan {
    param(
        [Parameter(Mandatory)][pscustomobject]$Plan,
        [Parameter(Mandatory)][string]$Path,
        [Parameter(Mandatory)][string]$Boundary,
        [Parameter(Mandatory)][string]$Reason,
        [switch]$RemoveRoot,
        [switch]$PreserveGitKeep
    )
    try { $fullPath = Get-TkbenFullPath $Path }
    catch {
        [void]$Plan.Skipped.Add([pscustomobject]@{ Path = $Path; Reason = $_.Exception.Message })
        return
    }
    if (-not (Test-TkbenPathWithin -Path $fullPath -Root $Boundary)) {
        [void]$Plan.Skipped.Add([pscustomobject]@{ Path = $fullPath; Reason = "Outside approved application root '$Boundary'." })
        return
    }
    try { $item = Get-Item -LiteralPath $fullPath -Force -ErrorAction Stop }
    catch {
        if ($_.CategoryInfo.Category -eq [System.Management.Automation.ErrorCategory]::ObjectNotFound -or
            $_.Exception -is [System.IO.FileNotFoundException] -or
            $_.Exception -is [System.IO.DirectoryNotFoundException]) {
            [void]$Plan.Missing.Add($fullPath)
        }
        else {
            [void]$Plan.Skipped.Add([pscustomobject]@{ Path = $fullPath; Reason = $_.Exception.Message })
        }
        return
    }
    if (-not (Test-TkbenNoReparseAncestor -Path $fullPath) -and
        ($item.Attributes -band [IO.FileAttributes]::ReparsePoint) -eq 0) {
        [void]$Plan.Skipped.Add([pscustomobject]@{ Path = $fullPath; Reason = 'A parent path is a symbolic link or junction.' })
        return
    }
    if (($item.Attributes -band [IO.FileAttributes]::ReparsePoint) -ne 0) {
        Add-TkbenPlanTarget -Plan $Plan -Path $fullPath -Kind 'Any' -Reason $Reason -Boundary $Boundary
        return
    }
    if (-not $item.PSIsContainer) {
        Add-TkbenPlanTarget -Plan $Plan -Path $fullPath -Kind 'File' -Reason $Reason -Boundary $Boundary
        return
    }

    try {
        $children = @(Get-ChildItem -LiteralPath $fullPath -Force -ErrorAction Stop)
        foreach ($child in $children) {
            if ($PreserveGitKeep -and $child.Name -eq '.gitkeep') { continue }
            Add-TkbenTreeToPlan -Plan $Plan -Path $child.FullName -Boundary $Boundary -Reason $Reason -RemoveRoot -PreserveGitKeep:$PreserveGitKeep
        }
    }
    catch {
        [void]$Plan.Skipped.Add([pscustomobject]@{ Path = $fullPath; Reason = "Could not enumerate directory: $($_.Exception.Message)" })
    }
    if ($RemoveRoot) {
        Add-TkbenPlanTarget -Plan $Plan -Path $fullPath -Kind 'Directory' -Reason $Reason -Boundary $Boundary
    }
}

function Add-TkbenContentsToPlan {
    param(
        [Parameter(Mandatory)][pscustomobject]$Plan,
        [Parameter(Mandatory)][string]$Path,
        [Parameter(Mandatory)][string]$Boundary,
        [Parameter(Mandatory)][string]$Reason
    )
    try { $rootItem = Get-Item -LiteralPath $Path -Force -ErrorAction Stop }
    catch {
        if ($_.CategoryInfo.Category -eq [System.Management.Automation.ErrorCategory]::ObjectNotFound -or
            $_.Exception -is [System.IO.FileNotFoundException] -or
            $_.Exception -is [System.IO.DirectoryNotFoundException]) {
            [void]$Plan.Missing.Add((Get-TkbenFullPath $Path))
        }
        else {
            [void]$Plan.Skipped.Add([pscustomobject]@{ Path = (Get-TkbenFullPath $Path); Reason = $_.Exception.Message })
        }
        return
    }
    if (-not $rootItem.PSIsContainer) {
        [void]$Plan.Skipped.Add([pscustomobject]@{ Path = (Get-TkbenFullPath $Path); Reason = 'Expected an application-owned directory.' })
        return
    }
    if (($rootItem.Attributes -band [IO.FileAttributes]::ReparsePoint) -ne 0 -or -not (Test-TkbenNoReparseAncestor $Path)) {
        [void]$Plan.Skipped.Add([pscustomobject]@{ Path = (Get-TkbenFullPath $Path); Reason = 'Directory or parent is a symbolic link or junction; contents were not traversed.' })
        return
    }
    try { $children = @(Get-ChildItem -LiteralPath $Path -Force -ErrorAction Stop) }
    catch {
        [void]$Plan.Skipped.Add([pscustomobject]@{ Path = (Get-TkbenFullPath $Path); Reason = "Could not enumerate directory: $($_.Exception.Message)" })
        return
    }
    foreach ($child in $children) {
        if ($child.Name -eq '.gitkeep') { continue }
        Add-TkbenTreeToPlan -Plan $Plan -Path $child.FullName -Boundary $Boundary -Reason $Reason -RemoveRoot
    }
}

function Add-TkbenCacheScan {
    param(
        [Parameter(Mandatory)][pscustomobject]$Plan,
        [Parameter(Mandatory)][string]$Path,
        [Parameter(Mandatory)][string]$Boundary
    )
    try { $item = Get-Item -LiteralPath $Path -Force -ErrorAction Stop }
    catch {
        if ($_.CategoryInfo.Category -eq [System.Management.Automation.ErrorCategory]::ObjectNotFound -or
            $_.Exception -is [System.IO.FileNotFoundException] -or
            $_.Exception -is [System.IO.DirectoryNotFoundException]) { return }
        [void]$Plan.Skipped.Add([pscustomobject]@{ Path = $Path; Reason = $_.Exception.Message })
        return
    }
    if (($item.Attributes -band [IO.FileAttributes]::ReparsePoint) -ne 0 -or
        -not (Test-TkbenNoReparseAncestor -Path $Path)) {
        [void]$Plan.Skipped.Add([pscustomobject]@{ Path = $item.FullName; Reason = 'Cache scan did not follow a symbolic link or junction.' })
        return
    }

    try { $children = @(Get-ChildItem -LiteralPath $Path -Force -ErrorAction Stop) }
    catch {
        [void]$Plan.Skipped.Add([pscustomobject]@{ Path = $Path; Reason = "Could not inspect cache root: $($_.Exception.Message)" })
        return
    }
    $isNodeModulesRoot = $item.Name -eq 'node_modules'
    foreach ($child in $children) {
        if ($child.PSIsContainer -and
            ($child.Name -in @('__pycache__', '.pytest_cache', '.ruff_cache', '.mypy_cache', '.angular', '.vite', '.cache', 'coverage', 'htmlcov') -or
                $child.Name -like 'pytest-cache-files-*')) {
            Add-TkbenTreeToPlan -Plan $Plan -Path $child.FullName -Boundary $Boundary -Reason 'Recognized nested application cache' -RemoveRoot
            continue
        }
        if (-not $child.PSIsContainer -and
            ($child.Name -match '(?i)^(\.coverage.*|.*\.(pyc|pyo))$')) {
            Add-TkbenPlanTarget -Plan $Plan -Path $child.FullName -Kind 'File' -Reason 'Python bytecode or coverage artifact' -Boundary $Boundary
            continue
        }
        # The supported Node tool caches live directly in node_modules. Do not
        # crawl every installed package's shipped files or mistake package dist
        # folders for application build output.
        if ($isNodeModulesRoot) { continue }
        if ($child.PSIsContainer -and ($child.Attributes -band [IO.FileAttributes]::ReparsePoint) -eq 0) {
            Add-TkbenCacheScan -Plan $Plan -Path $child.FullName -Boundary $Boundary
        }
    }
}

function Add-TkbenNamedTempTargets {
    param(
        [Parameter(Mandatory)][pscustomobject]$Plan,
        [Parameter(Mandatory)][string]$Path,
        [Parameter(Mandatory)][string]$Boundary
    )
    try { $root = Get-Item -LiteralPath $Path -Force -ErrorAction Stop }
    catch {
        if ($_.CategoryInfo.Category -eq [System.Management.Automation.ErrorCategory]::ObjectNotFound -or
            $_.Exception -is [System.IO.FileNotFoundException] -or
            $_.Exception -is [System.IO.DirectoryNotFoundException]) { return }
        [void]$Plan.Skipped.Add([pscustomobject]@{ Path = (Get-TkbenFullPath $Path); Reason = "Could not inspect application temporary files: $($_.Exception.Message)" })
        return
    }
    if (-not $root.PSIsContainer -or ($root.Attributes -band [IO.FileAttributes]::ReparsePoint) -ne 0 -or
        -not (Test-TkbenNoReparseAncestor -Path $Path)) {
        [void]$Plan.Skipped.Add([pscustomobject]@{ Path = $root.FullName; Reason = 'Temporary directory or parent is a symbolic link or junction.' })
        return
    }
    try {
        $temporaryFiles = @(Get-ChildItem -LiteralPath $Path -Force -File -ErrorAction Stop |
            Where-Object {
                $_.Name -match '(?i)^tkben_freq_.+\.sqlite3(?:-(?:wal|shm|journal))?$' -or
                $_.Name -match '(?i)^tkben-(?:benchmark-spool-.+\.ndjson|tokenizer-upload-.+\.json)$'
            })
        foreach ($temporaryFile in $temporaryFiles) {
            Add-TkbenPlanTarget -Plan $Plan -Path $temporaryFile.FullName -Kind 'File' -Reason 'Recognized TKBEN temporary runtime file' -Boundary $Boundary
        }
    }
    catch {
        [void]$Plan.Skipped.Add([pscustomobject]@{ Path = $root.FullName; Reason = "Could not enumerate application temporary files: $($_.Exception.Message)" })
    }
}

function Add-TkbenPatternFileTargets {
    param(
        [Parameter(Mandatory)][pscustomobject]$Plan,
        [Parameter(Mandatory)][string]$Path,
        [Parameter(Mandatory)][string]$Boundary,
        [Parameter(Mandatory)][string]$NamePattern,
        [Parameter(Mandatory)][string]$Reason
    )
    try { $item = Get-Item -LiteralPath $Path -Force -ErrorAction Stop }
    catch {
        if ($_.CategoryInfo.Category -eq [System.Management.Automation.ErrorCategory]::ObjectNotFound -or
            $_.Exception -is [System.IO.FileNotFoundException] -or
            $_.Exception -is [System.IO.DirectoryNotFoundException]) { return }
        [void]$Plan.Skipped.Add([pscustomobject]@{ Path = (Get-TkbenFullPath $Path); Reason = "Could not inspect generated temporary files: $($_.Exception.Message)" })
        return
    }
    if (-not $item.PSIsContainer -or ($item.Attributes -band [IO.FileAttributes]::ReparsePoint) -ne 0 -or
        -not (Test-TkbenNoReparseAncestor -Path $Path)) {
        [void]$Plan.Skipped.Add([pscustomobject]@{ Path = $item.FullName; Reason = 'Temporary-file root or parent is a symbolic link or junction.' })
        return
    }
    try { $children = @(Get-ChildItem -LiteralPath $Path -Force -ErrorAction Stop) }
    catch {
        [void]$Plan.Skipped.Add([pscustomobject]@{ Path = $item.FullName; Reason = "Could not enumerate generated temporary files: $($_.Exception.Message)" })
        return
    }
    foreach ($child in $children) {
        if ($child.PSIsContainer) {
            if (($child.Attributes -band [IO.FileAttributes]::ReparsePoint) -eq 0) {
                Add-TkbenPatternFileTargets -Plan $Plan -Path $child.FullName -Boundary $Boundary -NamePattern $NamePattern -Reason $Reason
            }
        }
        elseif ($child.Name -match $NamePattern) {
            Add-TkbenPlanTarget -Plan $Plan -Path $child.FullName -Kind 'File' -Reason $Reason -Boundary $Boundary
        }
    }
}

function Test-TkbenSafeDirectory {
    param(
        [Parameter(Mandatory)][pscustomobject]$Plan,
        [Parameter(Mandatory)][string]$Path,
        [Parameter(Mandatory)][string]$Reason
    )
    try { $item = Get-Item -LiteralPath $Path -Force -ErrorAction Stop }
    catch {
        if ($_.CategoryInfo.Category -eq [System.Management.Automation.ErrorCategory]::ObjectNotFound -or
            $_.Exception -is [System.IO.FileNotFoundException] -or
            $_.Exception -is [System.IO.DirectoryNotFoundException]) { return $false }
        [void]$Plan.Skipped.Add([pscustomobject]@{ Path = (Get-TkbenFullPath $Path); Reason = "$Reason`: $($_.Exception.Message)" })
        return $false
    }
    if (-not $item.PSIsContainer) {
        [void]$Plan.Skipped.Add([pscustomobject]@{ Path = $item.FullName; Reason = "$Reason`: expected a directory." })
        return $false
    }
    if (($item.Attributes -band [IO.FileAttributes]::ReparsePoint) -ne 0 -or
        -not (Test-TkbenNoReparseAncestor -Path $Path)) {
        [void]$Plan.Skipped.Add([pscustomobject]@{ Path = $item.FullName; Reason = "$Reason`: directory or parent is a symbolic link or junction." })
        return $false
    }
    return $true
}

function Add-TkbenInterruptedWriteTargets {
    param([Parameter(Mandatory)][pscustomobject]$Plan)
    Add-TkbenPatternFileTargets -Plan $Plan -Path $Plan.TokenizerDataRoot -Boundary $Plan.DataRoot -NamePattern '^\.tokenizer\.json\..+\.tmp$' -Reason 'Interrupted tokenizer artifact write'
    if ($Plan.DataRoot -ne $Plan.RepositoryRoot -and
        (Test-TkbenSafeDirectory -Plan $Plan -Path $Plan.DataRoot -Reason 'Interrupted application data writes')) {
        try {
            $databaseTemps = @(Get-ChildItem -LiteralPath $Plan.DataRoot -Force -File -Filter '.database-config.json.*.tmp' -ErrorAction Stop)
            foreach ($temporaryFile in $databaseTemps) {
                Add-TkbenPlanTarget -Plan $Plan -Path $temporaryFile.FullName -Kind 'File' -Reason 'Interrupted database configuration write' -Boundary $Plan.DataRoot
            }
            if (-not $Plan.ExternalDatabase -and $Plan.EncryptionKeyPath) {
                $keyTemp = $Plan.EncryptionKeyPath + '.tmp'
                Add-TkbenPlanTarget -Plan $Plan -Path $keyTemp -Kind 'File' -Reason 'Interrupted encryption-key write' -Boundary $Plan.DataRoot
            }
        }
        catch {
            if ($_.CategoryInfo.Category -ne [System.Management.Automation.ErrorCategory]::ObjectNotFound -and
                $_.Exception -isnot [System.IO.FileNotFoundException] -and
                $_.Exception -isnot [System.IO.DirectoryNotFoundException]) {
                [void]$Plan.Skipped.Add([pscustomobject]@{ Path = $Plan.DataRoot; Reason = "Could not inspect interrupted data writes: $($_.Exception.Message)" })
            }
        }
    }
    $defaultPathsConfig = Join-Path $Plan.SettingsRoot 'paths-config.json'
    if ((Get-TkbenFullPath $Plan.PathsConfigPath).Equals((Get-TkbenFullPath $defaultPathsConfig), [StringComparison]::OrdinalIgnoreCase)) {
        Add-TkbenPlanTarget -Plan $Plan -Path ($Plan.PathsConfigPath + '.tmp') -Kind 'File' -Reason 'Interrupted application storage configuration write' -Boundary $Plan.SettingsRoot
    }
}

function Add-TkbenLogTargets {
    param(
        [Parameter(Mandatory)][pscustomobject]$Plan,
        [Parameter(Mandatory)][string]$Path,
        [Parameter(Mandatory)][string]$Boundary
    )
    try { $root = Get-Item -LiteralPath $Path -Force -ErrorAction Stop }
    catch {
        if ($_.CategoryInfo.Category -eq [System.Management.Automation.ErrorCategory]::ObjectNotFound -or
            $_.Exception -is [System.IO.FileNotFoundException] -or
            $_.Exception -is [System.IO.DirectoryNotFoundException]) {
            [void]$Plan.Missing.Add((Get-TkbenFullPath $Path))
        }
        else {
            [void]$Plan.Skipped.Add([pscustomobject]@{ Path = (Get-TkbenFullPath $Path); Reason = "Could not inspect application log directory: $($_.Exception.Message)" })
        }
        return
    }
    if (-not $root.PSIsContainer -or ($root.Attributes -band [IO.FileAttributes]::ReparsePoint) -ne 0 -or
        -not (Test-TkbenNoReparseAncestor -Path $Path)) {
        [void]$Plan.Skipped.Add([pscustomobject]@{ Path = $root.FullName; Reason = 'Log directory or parent is a symbolic link or junction.' })
        return
    }
    try { $logs = @(Get-ChildItem -LiteralPath $Path -Force -File -Filter 'TKBEN_*.log' -ErrorAction Stop) }
    catch {
        [void]$Plan.Skipped.Add([pscustomobject]@{ Path = (Get-TkbenFullPath $Path); Reason = "Could not enumerate application logs: $($_.Exception.Message)" })
        return
    }
    foreach ($log in $logs) {
        Add-TkbenPlanTarget -Plan $Plan -Path $log.FullName -Kind 'File' -Reason 'Application-generated log' -Boundary $Boundary
    }
}

function Add-TkbenPathsConfigTarget {
    param([Parameter(Mandatory)][pscustomobject]$Plan)
    $defaultConfig = Join-Path $Plan.SettingsRoot 'paths-config.json'
    if ((Get-TkbenFullPath $Plan.PathsConfigPath).Equals((Get-TkbenFullPath $defaultConfig), [StringComparison]::OrdinalIgnoreCase)) {
        Add-TkbenPlanTarget -Plan $Plan -Path $Plan.PathsConfigPath -Kind 'File' -Reason 'Application storage configuration' -Boundary $Plan.SettingsRoot
        Add-TkbenPlanTarget -Plan $Plan -Path ($Plan.PathsConfigPath + '.tmp') -Kind 'File' -Reason 'Interrupted application storage configuration write' -Boundary $Plan.SettingsRoot
        return
    }
    $fileName = [IO.Path]::GetFileName($Plan.PathsConfigPath)
    if ((Test-TkbenPathWithin -Path $Plan.PathsConfigPath -Root $Plan.RuntimeCacheRoot) -and $fileName -match '^test-paths-[^\\/]+\.json$') {
        Add-TkbenPlanTarget -Plan $Plan -Path $Plan.PathsConfigPath -Kind 'File' -Reason 'Isolated launcher test configuration' -Boundary $Plan.RuntimeCacheRoot
        Add-TkbenPlanTarget -Plan $Plan -Path ($Plan.PathsConfigPath + '.tmp') -Kind 'File' -Reason 'Interrupted isolated test configuration write' -Boundary $Plan.RuntimeCacheRoot
        return
    }
    [void]$Plan.Skipped.Add([pscustomobject]@{ Path = $Plan.PathsConfigPath; Reason = 'Custom TKBEN_PATHS_CONFIG path is not a recognized local application config file.' })
}

function Add-TkbenPackageBuildTargets {
    param([Parameter(Mandatory)][pscustomobject]$Plan)
    foreach ($rootPath in @($Plan.RepositoryRoot, $Plan.ServerRoot)) {
        try { $root = Get-Item -LiteralPath $rootPath -Force -ErrorAction Stop }
        catch {
            if ($_.CategoryInfo.Category -eq [System.Management.Automation.ErrorCategory]::ObjectNotFound -or
                $_.Exception -is [System.IO.FileNotFoundException] -or
                $_.Exception -is [System.IO.DirectoryNotFoundException]) { continue }
            [void]$Plan.Skipped.Add([pscustomobject]@{ Path = $rootPath; Reason = "Could not inspect package build artifacts: $($_.Exception.Message)" })
            continue
        }
        if (-not $root.PSIsContainer -or ($root.Attributes -band [IO.FileAttributes]::ReparsePoint) -ne 0 -or
            -not (Test-TkbenNoReparseAncestor -Path $rootPath)) {
            [void]$Plan.Skipped.Add([pscustomobject]@{ Path = $root.FullName; Reason = 'Package build root or parent is a symbolic link or junction.' })
            continue
        }
        try {
            $artifacts = @(Get-ChildItem -LiteralPath $rootPath -Force -Directory -ErrorAction Stop |
                Where-Object { $_.Name -in @('build', 'dist') -or $_.Name -like '*.egg-info' })
            foreach ($artifact in $artifacts) {
                Add-TkbenTreeToPlan -Plan $Plan -Path $artifact.FullName -Boundary $Plan.RepositoryRoot -Reason 'Python package build output or generated metadata' -RemoveRoot
            }
        }
        catch {
            [void]$Plan.Skipped.Add([pscustomobject]@{ Path = $rootPath; Reason = "Could not enumerate package build artifacts: $($_.Exception.Message)" })
        }
    }
}

function Add-TkbenCacheTargets {
    param([Parameter(Mandatory)][pscustomobject]$Plan)
    Add-TkbenContentsToPlan -Plan $Plan -Path $Plan.RuntimeCacheRoot -Boundary $Plan.RepositoryRoot -Reason 'Managed tool and test cache'
    foreach ($cacheName in @('__pycache__', '.pytest_cache', '.ruff_cache', '.mypy_cache', '.angular', '.vite', '.cache', 'coverage', 'htmlcov')) {
        Add-TkbenTreeToPlan -Plan $Plan -Path (Join-Path $Plan.RepositoryRoot $cacheName) -Boundary $Plan.RepositoryRoot -Reason 'Recognized root-level application cache' -RemoveRoot
    }
    if (Test-TkbenSafeDirectory -Plan $Plan -Path $Plan.RepositoryRoot -Reason 'Root-level application cache scan') {
        try {
            $rootPytestCaches = @(Get-ChildItem -LiteralPath $Plan.RepositoryRoot -Force -Directory -ErrorAction Stop |
                Where-Object { $_.Name -like 'pytest-cache-files-*' })
            foreach ($directory in $rootPytestCaches) {
                Add-TkbenTreeToPlan -Plan $Plan -Path $directory.FullName -Boundary $Plan.RepositoryRoot -Reason 'Root-level pytest cache artifact' -RemoveRoot
            }
        }
        catch {
            [void]$Plan.Skipped.Add([pscustomobject]@{ Path = $Plan.RepositoryRoot; Reason = "Could not inspect root-level pytest artifacts: $($_.Exception.Message)" })
        }
        try {
            $rootBytecode = @(Get-ChildItem -LiteralPath $Plan.RepositoryRoot -Force -File -ErrorAction Stop |
                Where-Object { $_.Name -match '(?i)^(\.coverage.*|.*\.(pyc|pyo))$' })
            foreach ($file in $rootBytecode) {
                Add-TkbenPlanTarget -Plan $Plan -Path $file.FullName -Kind 'File' -Reason 'Root-level Python bytecode or coverage artifact' -Boundary $Plan.RepositoryRoot
            }
        }
        catch {
            [void]$Plan.Skipped.Add([pscustomobject]@{ Path = $Plan.RepositoryRoot; Reason = "Could not inspect root-level caches: $($_.Exception.Message)" })
        }
    }
    Add-TkbenCacheScan -Plan $Plan -Path $Plan.ApplicationRoot -Boundary $Plan.RepositoryRoot
    Add-TkbenPackageBuildTargets -Plan $Plan
    Add-TkbenContentsToPlan -Plan $Plan -Path $Plan.DatasetCacheRoot -Boundary $Plan.DataRoot -Reason 'Temporary Hugging Face dataset download cache'
    Add-TkbenContentsToPlan -Plan $Plan -Path $Plan.ClientDistRoot -Boundary $Plan.RepositoryRoot -Reason 'Generated frontend build output'
    Add-TkbenContentsToPlan -Plan $Plan -Path $Plan.ClientDistSsrRoot -Boundary $Plan.RepositoryRoot -Reason 'Generated frontend SSR build output'
    try { $runtimeRootItem = Get-Item -LiteralPath $Plan.RuntimeRoot -Force -ErrorAction Stop }
    catch {
        if ($_.CategoryInfo.Category -ne [System.Management.Automation.ErrorCategory]::ObjectNotFound -and
            $_.Exception -isnot [System.IO.FileNotFoundException] -and
            $_.Exception -isnot [System.IO.DirectoryNotFoundException]) {
            [void]$Plan.Skipped.Add([pscustomobject]@{ Path = $Plan.RuntimeRoot; Reason = "Could not inspect launcher staging directories: $($_.Exception.Message)" })
        }
        return
    }
    if (-not $runtimeRootItem.PSIsContainer -or ($runtimeRootItem.Attributes -band [IO.FileAttributes]::ReparsePoint) -ne 0 -or
        -not (Test-TkbenNoReparseAncestor -Path $Plan.RuntimeRoot)) {
        [void]$Plan.Skipped.Add([pscustomobject]@{ Path = $runtimeRootItem.FullName; Reason = 'Runtime directory or parent is a symbolic link or junction.' })
        return
    }
    try {
        $staging = @(Get-ChildItem -LiteralPath $Plan.RuntimeRoot -Force -Directory -ErrorAction Stop |
            Where-Object { $_.Name -match '^\.(python|nodejs)-staging-' })
        foreach ($directory in $staging) {
            Add-TkbenTreeToPlan -Plan $Plan -Path $directory.FullName -Boundary $Plan.RepositoryRoot -Reason 'Interrupted launcher staging directory' -RemoveRoot
        }
    }
    catch {
        [void]$Plan.Skipped.Add([pscustomobject]@{ Path = $Plan.RuntimeRoot; Reason = "Could not inspect launcher staging directories: $($_.Exception.Message)" })
    }
}

function Get-TkbenCleanupPlan {
    [CmdletBinding()]
    param(
        [Parameter(Mandatory)][ValidateSet('Cache', 'Logs', 'Uninstall', 'AllData')][string]$Action,
        [Parameter(Mandatory)][string]$RepositoryRoot,
        [Parameter(Mandatory)][string]$RuntimeRoot,
        [Parameter(Mandatory)][string]$RuntimeCacheRoot,
        [Parameter(Mandatory)][string]$ApplicationRoot,
        [Parameter(Mandatory)][string]$ClientRoot,
        [Parameter(Mandatory)][string]$ServerRoot,
        [Parameter(Mandatory)][string]$DataRoot,
        [Parameter(Mandatory)][string]$DatasetCacheRoot,
        [Parameter(Mandatory)][string]$TokenizerDataRoot,
        [Parameter(Mandatory)][string]$LogRoot,
        [Parameter(Mandatory)][string]$TempRoot,
        [Parameter(Mandatory)][string]$SettingsRoot,
        [Parameter(Mandatory)][string]$PathsConfigPath,
        [Parameter(Mandatory)][string]$EnvironmentFilePath,
        [Parameter(Mandatory)][string]$DatabaseConfigPath,
        [string]$SqliteDatabasePath,
        [string]$EncryptionKeyPath,
        [switch]$ExternalDatabase
    )

    $repository = Get-TkbenFullPath $RepositoryRoot
    $dataRootFull = Get-TkbenFullPath $DataRoot
    if ($dataRootFull.Equals($repository, [StringComparison]::OrdinalIgnoreCase) -or
        $repository.StartsWith($dataRootFull.TrimEnd('\') + '\', [StringComparison]::OrdinalIgnoreCase)) {
        throw "Refusing a data root that is the repository root or one of its ancestors: $dataRootFull"
    }
    $protectedRoots = @(
        (Join-Path $repository 'app'),
        (Join-Path $repository 'assets'),
        (Join-Path $repository 'settings'),
        (Join-Path $repository 'runtimes'),
        (Join-Path $repository '.git')
    )
    foreach ($protectedRoot in $protectedRoots) {
        if ((Test-TkbenPathWithin -Path $dataRootFull -Root $protectedRoot) -or
            (Test-TkbenPathWithin -Path $protectedRoot -Root $dataRootFull)) {
            throw "Refusing a data root that overlaps a protected application or configuration directory: $dataRootFull"
        }
    }

    $plan = [pscustomobject]@{
        Action = $Action
        RepositoryRoot = Get-TkbenFullPath $RepositoryRoot
        RuntimeRoot = Get-TkbenFullPath $RuntimeRoot
        RuntimeCacheRoot = Get-TkbenFullPath $RuntimeCacheRoot
        ApplicationRoot = Get-TkbenFullPath $ApplicationRoot
        ClientRoot = Get-TkbenFullPath $ClientRoot
        ServerRoot = Get-TkbenFullPath $ServerRoot
        DataRoot = Get-TkbenFullPath $DataRoot
        DatasetCacheRoot = Get-TkbenFullPath $DatasetCacheRoot
        TokenizerDataRoot = Get-TkbenFullPath $TokenizerDataRoot
        LogRoot = Get-TkbenFullPath $LogRoot
        TempRoot = Get-TkbenFullPath $TempRoot
        SettingsRoot = Get-TkbenFullPath $SettingsRoot
        ClientDistRoot = Join-Path $ClientRoot 'dist'
        ClientDistSsrRoot = Join-Path $ClientRoot 'dist-ssr'
        PathsConfigPath = Get-TkbenFullPath $PathsConfigPath
        EnvironmentFilePath = Get-TkbenFullPath $EnvironmentFilePath
        DatabaseConfigPath = Get-TkbenFullPath $DatabaseConfigPath
        SqliteDatabasePath = $SqliteDatabasePath
        EncryptionKeyPath = $EncryptionKeyPath
        ExternalDatabase = [bool]$ExternalDatabase
        Targets = [Collections.Generic.List[object]]::new()
        Missing = [Collections.Generic.List[string]]::new()
        Skipped = [Collections.Generic.List[object]]::new()
        TargetIndex = @{}
    }

    switch ($Action) {
        'Cache' {
            Add-TkbenCacheTargets -Plan $plan
            Add-TkbenInterruptedWriteTargets -Plan $plan
            Add-TkbenNamedTempTargets -Plan $plan -Path $plan.TempRoot -Boundary $plan.TempRoot
        }
        'Logs' {
            Add-TkbenLogTargets -Plan $plan -Path $plan.LogRoot -Boundary $plan.LogRoot
        }
        'Uninstall' {
            Add-TkbenCacheTargets -Plan $plan
            foreach ($directory in @(
                (Join-Path $plan.RuntimeRoot 'python'),
                (Join-Path $plan.RuntimeRoot 'uv'),
                (Join-Path $plan.RuntimeRoot 'nodejs'),
                (Join-Path $plan.ServerRoot '.venv'),
                (Join-Path $plan.RepositoryRoot '.venv'),
                (Join-Path $plan.ClientRoot 'node_modules')
            )) {
                Add-TkbenTreeToPlan -Plan $plan -Path $directory -Boundary $plan.RepositoryRoot -Reason 'Managed runtime or dependency installation' -RemoveRoot
            }
            if (Test-TkbenSafeDirectory -Plan $plan -Path $plan.RuntimeRoot -Reason 'Launcher backup scan') {
                try {
                    $backups = @(Get-ChildItem -LiteralPath $plan.RuntimeRoot -Force -Directory -ErrorAction Stop |
                        Where-Object { $_.Name -match '^\.(python|nodejs)-backup-' })
                    foreach ($directory in $backups) {
                        Add-TkbenTreeToPlan -Plan $plan -Path $directory.FullName -Boundary $plan.RepositoryRoot -Reason 'Interrupted launcher backup runtime' -RemoveRoot
                    }
                }
                catch {
                    [void]$plan.Skipped.Add([pscustomobject]@{ Path = $plan.RuntimeRoot; Reason = "Could not inspect launcher backups: $($_.Exception.Message)" })
                }
            }
            Add-TkbenInterruptedWriteTargets -Plan $plan
            Add-TkbenNamedTempTargets -Plan $plan -Path $plan.TempRoot -Boundary $plan.TempRoot
            Add-TkbenLogTargets -Plan $plan -Path $plan.LogRoot -Boundary $plan.LogRoot
        }
        'AllData' {
            Add-TkbenContentsToPlan -Plan $plan -Path $plan.DatasetCacheRoot -Boundary $plan.DataRoot -Reason 'Application dataset source data'
            Add-TkbenContentsToPlan -Plan $plan -Path $plan.TokenizerDataRoot -Boundary $plan.DataRoot -Reason 'Persistent tokenizer artifacts'
            Add-TkbenLogTargets -Plan $plan -Path $plan.LogRoot -Boundary $plan.LogRoot
            Add-TkbenNamedTempTargets -Plan $plan -Path $plan.TempRoot -Boundary $plan.TempRoot
            Add-TkbenPlanTarget -Plan $plan -Path $plan.DatabaseConfigPath -Kind 'File' -Reason 'Application database configuration' -Boundary $plan.DataRoot
            Add-TkbenPathsConfigTarget -Plan $plan
            Add-TkbenPlanTarget -Plan $plan -Path $plan.EnvironmentFilePath -Kind 'File' -Reason 'Application bootstrap settings' -Boundary $plan.RepositoryRoot
            if (Test-TkbenSafeDirectory -Plan $plan -Path $plan.DataRoot -Reason 'Temporary database configuration scan') {
                try {
                    $databaseConfigTemps = @(Get-ChildItem -LiteralPath $plan.DataRoot -Force -File -Filter '.database-config.json.*.tmp' -ErrorAction Stop)
                    foreach ($temporaryFile in $databaseConfigTemps) {
                        Add-TkbenPlanTarget -Plan $plan -Path $temporaryFile.FullName -Kind 'File' -Reason 'Interrupted application database configuration write' -Boundary $plan.DataRoot
                    }
                }
                catch {
                    [void]$plan.Skipped.Add([pscustomobject]@{ Path = $plan.DataRoot; Reason = "Could not inspect temporary database configuration files: $($_.Exception.Message)" })
                }
            }
            if (-not $ExternalDatabase -and $EncryptionKeyPath) {
                Add-TkbenPlanTarget -Plan $plan -Path $EncryptionKeyPath -Kind 'File' -Reason 'Application encryption key material' -Boundary $plan.DataRoot
                Add-TkbenPlanTarget -Plan $plan -Path ($plan.EncryptionKeyPath + '.tmp') -Kind 'File' -Reason 'Interrupted application encryption-key write' -Boundary $plan.DataRoot
            }
            if (-not $ExternalDatabase -and $SqliteDatabasePath) {
                foreach ($databaseFile in @(
                    $SqliteDatabasePath,
                    "$SqliteDatabasePath-wal",
                    "$SqliteDatabasePath-shm",
                    "$SqliteDatabasePath-journal"
                )) {
                    Add-TkbenPlanTarget -Plan $plan -Path $databaseFile -Kind 'File' -Reason 'Configured SQLite database or sidecar' -ExplicitExternalFile
                }
            }
            Add-TkbenCacheTargets -Plan $plan
        }
    }

    return $plan
}

function Invoke-TkbenCleanupPlan {
    [CmdletBinding()]
    param(
        [Parameter(Mandatory)][pscustomobject]$Plan,
        [switch]$DryRun
    )
    $removed = [Collections.Generic.List[string]]::new()
    $failed = [Collections.Generic.List[object]]::new()
    $skipped = [Collections.Generic.List[object]]::new()
    foreach ($plannedSkip in $Plan.Skipped) { [void]$skipped.Add($plannedSkip) }
    if ($DryRun) {
        return [pscustomobject]@{
            Action = $Plan.Action
            DryRun = $true
            Removed = @()
            Failed = @()
            Missing = @($Plan.Missing.ToArray())
            Skipped = @($skipped.ToArray())
            Planned = @($Plan.Targets.ToArray())
        }
    }

    foreach ($target in $Plan.Targets) {
        try {
            $fullPath = Get-TkbenFullPath $target.Path
            if (-not $fullPath -or $fullPath.TrimEnd('\') -eq [IO.Path]::GetPathRoot($fullPath).TrimEnd('\')) {
                [void]$skipped.Add([pscustomobject]@{ Path = $target.Path; Reason = 'Filesystem roots are never valid cleanup targets.' })
                continue
            }
            if ($target.Boundary) {
                if (-not (Test-TkbenPathWithin -Path $fullPath -Root $target.Boundary)) {
                    [void]$skipped.Add([pscustomobject]@{ Path = $fullPath; Reason = 'Target moved outside its approved application boundary after preview.' })
                    continue
                }
            }
            elseif (-not $target.AllowsExternalPath) {
                [void]$skipped.Add([pscustomobject]@{ Path = $fullPath; Reason = 'Target has no approved boundary.' })
                continue
            }
            $parent = [IO.Path]::GetDirectoryName($fullPath)
            if (-not $parent -or -not (Test-TkbenNoReparseAncestor -Path $parent)) {
                [void]$skipped.Add([pscustomobject]@{ Path = $fullPath; Reason = 'A parent path became a symbolic link or junction after preview.' })
                continue
            }
            $item = Get-Item -LiteralPath $target.Path -Force -ErrorAction Stop
            $currentIsReparsePoint = (($item.Attributes -band [IO.FileAttributes]::ReparsePoint) -ne 0)
            if ([bool]$target.IsReparsePoint -ne $currentIsReparsePoint) {
                [void]$skipped.Add([pscustomobject]@{ Path = $fullPath; Reason = 'Target type changed after preview; it was not removed.' })
                continue
            }
            $currentKind = if ($item.PSIsContainer) { 'Directory' } else { 'File' }
            if ($target.Kind -ne $currentKind) {
                [void]$skipped.Add([pscustomobject]@{ Path = $fullPath; Reason = 'Target file type changed after preview; it was not removed.' })
                continue
            }
            if (($item.Attributes -band [IO.FileAttributes]::ReparsePoint) -ne 0) {
                if ($item.PSIsContainer) { [IO.Directory]::Delete($target.Path, $false) }
                else { [IO.File]::Delete($target.Path) }
            }
            elseif ($item.PSIsContainer) {
                [IO.Directory]::Delete($target.Path, $false)
            }
            else {
                [IO.File]::Delete($target.Path)
            }
            [void]$removed.Add($target.Path)
        }
        catch {
            if ($_.CategoryInfo.Category -eq [System.Management.Automation.ErrorCategory]::ObjectNotFound -or
                $_.Exception -is [System.IO.FileNotFoundException] -or
                $_.Exception -is [System.IO.DirectoryNotFoundException]) {
                [void]$Plan.Missing.Add($target.Path)
            }
            else {
                [void]$failed.Add([pscustomobject]@{ Path = $target.Path; Reason = $_.Exception.Message })
            }
        }
    }
    return [pscustomobject]@{
        Action = $Plan.Action
        DryRun = $false
        Removed = @($removed.ToArray())
        Failed = @($failed.ToArray())
        Missing = @($Plan.Missing.ToArray())
        Skipped = @($skipped.ToArray())
        Planned = @($Plan.Targets.ToArray())
    }
}

function Get-TkbenProcessIdsFromSnapshot {
    [CmdletBinding()]
    param(
        [Parameter(Mandatory)][string]$RepositoryRoot,
        [int[]]$BackendProcessIds = @(),
        [Parameter(Mandatory)][psobject[]]$Processes
    )
    $repoMarker = (Get-TkbenFullPath $RepositoryRoot).TrimEnd('\')
    $processIds = [Collections.Generic.List[int]]::new()
    foreach ($processId in $BackendProcessIds) { [void]$processIds.Add([int]$processId) }
    $blockingPattern = '(?i)(?:start_on_windows\.ps1|@angular[\\/]+cli|ng(?:\.js)?\s+(?:serve|build|test)|vite(?:\.js)?|vitest|uvicorn|server\.app:app|scripts[\\/]+preview\.mjs|npm(?:\.cmd)?\s+(?:ci|install|run\s+(?:dev|build|test:unit|preview))|pytest|ruff|basedpyright|compileall|uv(?:\.exe)?\s+(?:run|sync|venv|python|tool))'
    foreach ($process in $processes) {
        $processId = [int]$process.ProcessId
        if ($processId -eq $PID) { continue }
        $commandLine = [string]$process.CommandLine
        if ($commandLine.IndexOf($repoMarker, [StringComparison]::OrdinalIgnoreCase) -lt 0) { continue }
        if ($commandLine -match $blockingPattern) { [void]$processIds.Add($processId) }
    }
    return @($processIds | Sort-Object -Unique)
}

function Get-TkbenMaintenanceProcessIds {
    [CmdletBinding()]
    param(
        [Parameter(Mandatory)][string]$RepositoryRoot,
        [int[]]$BackendProcessIds = @()
    )
    try { $processes = @(Get-CimInstance -ClassName Win32_Process -ErrorAction Stop) }
    catch { throw "Could not verify that cleanup targets are not in use: $($_.Exception.Message)" }
    return @(Get-TkbenProcessIdsFromSnapshot -RepositoryRoot $RepositoryRoot -BackendProcessIds $BackendProcessIds -Processes $processes)
}

function Assert-TkbenNoActiveProcesses {
    [CmdletBinding()]
    param([int[]]$ProcessIds = @())
    if ($ProcessIds.Count -gt 0) {
        throw "Cleanup was not started because TKBEN processes are still using application files (PID(s): $($ProcessIds -join ', ')). Stop the application and retry."
    }
}

function Remove-TkbenOwnedPath {
    [CmdletBinding()]
    param(
        [Parameter(Mandatory)][string]$Path,
        [Parameter(Mandatory)][string]$Boundary,
        [switch]$KeepRoot,
        [switch]$Strict,
        [switch]$DryRun
    )
    $fullPath = Get-TkbenFullPath $Path
    $fullBoundary = Get-TkbenFullPath $Boundary
    if ($fullPath.Equals($fullBoundary, [StringComparison]::OrdinalIgnoreCase) -or
        $fullBoundary.StartsWith($fullPath.TrimEnd('\') + '\', [StringComparison]::OrdinalIgnoreCase)) {
        throw "Refusing to remove an approved root or its ancestor: $fullPath"
    }
    $plan = [pscustomobject]@{
        Action = 'Internal runtime maintenance'
        Targets = [Collections.Generic.List[object]]::new()
        Missing = [Collections.Generic.List[string]]::new()
        Skipped = [Collections.Generic.List[object]]::new()
        TargetIndex = @{}
    }
    if ($KeepRoot) {
        Add-TkbenContentsToPlan -Plan $plan -Path $fullPath -Boundary $fullBoundary -Reason 'Managed runtime cache'
    }
    else {
        Add-TkbenTreeToPlan -Plan $plan -Path $fullPath -Boundary $fullBoundary -Reason 'Managed launcher runtime' -RemoveRoot -PreserveGitKeep
    }
    $result = Invoke-TkbenCleanupPlan -Plan $plan -DryRun:$DryRun
    if ($Strict -and ($result.Failed.Count -gt 0 -or $result.Skipped.Count -gt 0)) {
        throw "Removal of '$Path' was incomplete. Failed $($result.Failed.Count) path(s); skipped $($result.Skipped.Count) path(s)."
    }
    return [pscustomobject]@{
        Path = [IO.Path]::GetFullPath($Path)
        PlannedCount = $result.Planned.Count
        PlannedPaths = @($result.Planned | ForEach-Object { $_.Path })
        RemovedCount = $result.Removed.Count
        RemovedPaths = $result.Removed
        SkippedCount = $result.Skipped.Count + $result.Missing.Count
        SkippedPaths = @($result.Skipped | ForEach-Object { $_.Path }) + @($result.Missing)
        EnumerationErrorCount = $result.Skipped.Count
        EnumerationErrorPaths = @($result.Skipped | ForEach-Object { $_.Path })
        WhatIf = [bool]$DryRun
    }
}

Export-ModuleMember -Function Get-TkbenCleanupPlan, Invoke-TkbenCleanupPlan, Get-TkbenMaintenanceProcessIds, Get-TkbenProcessIdsFromSnapshot, Assert-TkbenNoActiveProcesses, Remove-TkbenOwnedPath
