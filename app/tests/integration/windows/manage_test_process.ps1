# Copyright © 2023 Thomas Virdis
# Licensed under the MIT License.

#requires -Version 5.1

<#
Start and stop only processes launched by run_tests.bat. The state file
records the exact PID, executable, and creation time so a recycled PID or
unrelated listener cannot be terminated during cleanup.
#>
[CmdletBinding()]
param(
    [Parameter(Mandatory)]
    [ValidateSet('StartBackend', 'StartPreview', 'Stop', 'RemoveTestData')]
    [string]$Action,
    [string]$StateFile,
    [string]$RepositoryRoot,
    [string]$DataRoot,
    [string]$BindAddress = '127.0.0.1',
    [ValidateRange(1, 65535)][int]$Port = 5000
)

$ErrorActionPreference = 'Stop'

if ($Action -eq 'RemoveTestData') {
    if ([string]::IsNullOrWhiteSpace($RepositoryRoot) -or
        [string]::IsNullOrWhiteSpace($DataRoot)) {
        throw 'RepositoryRoot and DataRoot are required for test data cleanup.'
    }
    $root = [IO.Path]::GetFullPath($RepositoryRoot)
    $cacheRoot = [IO.Path]::GetFullPath((Join-Path $root 'runtimes\cache'))
    $cachePrefix = $cacheRoot.TrimEnd('\', '/') + [IO.Path]::DirectorySeparatorChar
    $target = [IO.Path]::GetFullPath($DataRoot)
    $leaf = Split-Path -Leaf $target
    if (-not $target.StartsWith($cachePrefix, [StringComparison]::OrdinalIgnoreCase) -or
        -not $leaf.StartsWith('test-data-', [StringComparison]::OrdinalIgnoreCase) -or
        $leaf.Length -le 'test-data-'.Length) {
        throw "Refusing to remove a path outside the isolated test data area: $target"
    }
    if (Test-Path -LiteralPath $target) {
        $entry = Get-Item -LiteralPath $target -Force
        if (-not $entry.PSIsContainer -or
            ($entry.Attributes -band [IO.FileAttributes]::ReparsePoint)) {
            throw "Refusing to remove a non-directory or reparse test data path: $target"
        }
        Remove-Item -LiteralPath $target -Recurse -Force -ErrorAction Stop
    }
    exit 0
}

if ($Action -eq 'Stop') {
    if ([string]::IsNullOrWhiteSpace($StateFile)) {
        throw 'StateFile is required to stop a test-owned process.'
    }
    if (-not (Test-Path -LiteralPath $StateFile -PathType Leaf)) {
        throw "Cannot verify the test-owned process: state file missing at $StateFile."
    }

    $state = Get-Content -LiteralPath $StateFile -Raw | ConvertFrom-Json
    $ownedProcessId = [int]$state.processId
    if ($ownedProcessId -le 0 -or $ownedProcessId -eq $PID) {
        throw 'The test process state contains an invalid PID.'
    }

    $ownedProcess = Get-Process -Id $ownedProcessId -ErrorAction SilentlyContinue
    if ($ownedProcess) {
        $actualStartTicks = [string]$ownedProcess.StartTime.ToUniversalTime().Ticks
        $processRecord = Get-CimInstance -ClassName Win32_Process -Filter "ProcessId = $ownedProcessId" -ErrorAction Stop
        $actualExecutable = [string]$processRecord.ExecutablePath
        if ($actualStartTicks -ne [string]$state.startUtcTicks -or
            [string]::IsNullOrWhiteSpace($actualExecutable) -or
            [IO.Path]::GetFullPath($actualExecutable) -ne [IO.Path]::GetFullPath([string]$state.executable)) {
            throw "PID $ownedProcessId no longer matches the process launched by this test. Refusing termination."
        }

        try {
            Stop-Process -Id $ownedProcessId -Force -ErrorAction Stop
        }
        catch {
            if (Get-Process -Id $ownedProcessId -ErrorAction SilentlyContinue) { throw }
        }
        if (-not $ownedProcess.WaitForExit(5000)) {
            throw "Test-owned PID $ownedProcessId did not exit after cleanup."
        }
    }
    Remove-Item -LiteralPath $StateFile -Force -ErrorAction Stop
    exit 0
}

if ([string]::IsNullOrWhiteSpace($RepositoryRoot)) {
    throw 'RepositoryRoot is required for process startup.'
}
if ([string]::IsNullOrWhiteSpace($StateFile)) {
    throw 'StateFile is required for process startup.'
}
if (Test-Path -LiteralPath $StateFile) {
    throw "Refusing to overwrite an existing test process state file: $StateFile."
}

$root = [IO.Path]::GetFullPath($RepositoryRoot)
if ($Action -eq 'StartBackend') {
    $executable = Join-Path $root 'app\server\.venv\Scripts\python.exe'
    $arguments = "-m uvicorn server.app:app --host $BindAddress --port $Port --log-level warning"
    $workDir = Join-Path $root 'app\server'
}
else {
    $executable = Join-Path $root 'runtimes\nodejs\node.exe'
    $previewScript = Join-Path $root 'app\client\scripts\preview.mjs'
    $arguments = "`"$previewScript`" --host $BindAddress --port $Port --strictPort"
    $workDir = Join-Path $root 'app\client'
}
if (-not (Test-Path -LiteralPath $executable -PathType Leaf)) {
    throw "The test runtime executable does not exist: $executable"
}

try {
    $address = [Net.IPAddress]::Parse($BindAddress)
    $probe = [Net.Sockets.TcpListener]::new($address, $Port)
    $probe.Start()
    $probe.Stop()
}
catch {
    if ($probe) { $probe.Stop() }
    throw "The test port $BindAddress`:$Port is occupied or unavailable. Refusing to reuse it. $($_.Exception.Message)"
}

$startedProcess = Start-Process -FilePath $executable -ArgumentList $arguments `
    -WorkingDirectory $workDir -NoNewWindow -PassThru -ErrorAction Stop
try {
    $startedProcess.Refresh()
    $startUtcTicks = [string]$startedProcess.StartTime.ToUniversalTime().Ticks
    $parentDir = Split-Path -Parent ([IO.Path]::GetFullPath($StateFile))
    New-Item -ItemType Directory -Path $parentDir -Force | Out-Null
    [ordered]@{
        processId = [int]$startedProcess.Id
        startUtcTicks = $startUtcTicks
        executable = [IO.Path]::GetFullPath($executable)
    } | ConvertTo-Json | Set-Content -LiteralPath $StateFile -Encoding UTF8 -ErrorAction Stop
}
catch {
    if (-not $startedProcess.HasExited) {
        Stop-Process -Id $startedProcess.Id -Force -ErrorAction SilentlyContinue
    }
    throw
}
