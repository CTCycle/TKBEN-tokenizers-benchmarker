#requires -Version 5.1

<#
.SYNOPSIS
    Reproduces a real denied Stop-Process attempt against a synthetic listener.

.DESCRIPTION
    This harness is deliberately safety-gated. It must be run from a disposable
    Windows VM, Windows Sandbox, or equivalent isolated Windows account and from
    a non-elevated interactive console. It clones the requested checkout into a
    disposable directory, starts one elevated synthetic listener and one
    unprivileged sentinel listener, then runs the normal interactive -Launch
    path. The elevated helper exits through its own stop marker during cleanup;
    this harness never targets a system process and never uses taskkill.exe.

    The -AllowDisposableEnvironment switch is an explicit operator assertion.
    It is not a privilege escalation mechanism and does not elevate the
    launcher under test. The elevated helper may display the normal UAC prompt.

.EXAMPLE
    .\test_launcher_port_permissions.ps1 -AllowDisposableEnvironment

    Run only inside a disposable Windows test boundary. The harness supplies
    "yes" to the launch confirmation through the shared interactive console.
#>
[CmdletBinding()]
param(
    [string]$SourceRoot,
    [string]$EvidencePath,
    [switch]$KeepCheckout,
    [switch]$AllowDisposableEnvironment
)

$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest

$SourceRoot = if ([string]::IsNullOrWhiteSpace($SourceRoot)) {
    Join-Path $PSScriptRoot '..\..\..\..'
}
else {
    $SourceRoot
}
$sourceRootFull = [IO.Path]::GetFullPath($SourceRoot)
$runId = [guid]::NewGuid().ToString('N')
$harnessRoot = Join-Path ([IO.Path]::GetTempPath()) "tkben-t0-02-permission-$runId"
$checkoutRoot = Join-Path $harnessRoot 'checkout'
$helperRoot = Join-Path $harnessRoot 'helpers'
$launcherLogPath = Join-Path $harnessRoot 'launcher.log'
$listenerScriptPath = Join-Path $helperRoot 'synthetic_listener.ps1'
$launcherWrapperPath = Join-Path $helperRoot 'invoke_launcher.ps1'
$protectedReadyPath = Join-Path $helperRoot 'protected.ready.json'
$protectedStopPath = Join-Path $helperRoot 'protected.stop'
$sentinelReadyPath = Join-Path $helperRoot 'sentinel.ready.json'
$sentinelStopPath = Join-Path $helperRoot 'sentinel.stop'
$protectedRecord = $null
$sentinelRecord = $null
$launcherProcess = $null
$protectedPort = $null
$uiPort = $null
$launcherExitCode = $null
$launcherOutput = ''
$terminationDenied = $false
$servicesStarted = $false
$cleanupPassed = $true
$resultStatus = 'UNRUN'

function Write-HarnessStep([string]$Message) {
    Write-Host "[STEP] $Message" -ForegroundColor Cyan
}

function Write-HarnessOk([string]$Message) {
    Write-Host "[OK] $Message" -ForegroundColor Green
}

function Assert-Harness([bool]$Condition, [string]$Message) {
    if (-not $Condition) { throw $Message }
}

function Get-IsElevated {
    $identity = [Security.Principal.WindowsIdentity]::GetCurrent()
    $principal = [Security.Principal.WindowsPrincipal]::new($identity)
    return $principal.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)
}

function Get-FreeTcpPort {
    $listener = [Net.Sockets.TcpListener]::new([Net.IPAddress]::Loopback, 0)
    try {
        $listener.Start()
        return ([Net.IPEndPoint]$listener.LocalEndpoint).Port
    }
    finally {
        $listener.Stop()
    }
}

function Get-NetstatListeners([int]$Port) {
    $records = @()
    foreach ($line in @(netstat.exe -ano 2>$null)) {
        $match = [regex]::Match(
            [string]$line,
            '^\s*TCP\S*\s+\S+:(\d+)\s+\S+\s+LISTENING\s+(\d+)\s*$'
        )
        if ($match.Success -and [int]$match.Groups[1].Value -eq $Port) {
            $records += [pscustomobject]@{
                Port = [int]$match.Groups[1].Value
                ProcessId = [int]$match.Groups[2].Value
            }
        }
    }
    return @($records | Sort-Object Port, ProcessId -Unique)
}

function Test-NetstatOwnership([int]$Port, [int]$ProcessId) {
    return @(
        Get-NetstatListeners -Port $Port |
            Where-Object { [int]$_.ProcessId -eq $ProcessId }
    ).Count -gt 0
}

function Get-ProcessCommandLines {
    try {
        return @(Get-CimInstance -ClassName Win32_Process -ErrorAction Stop |
            Select-Object ProcessId, Name, CommandLine)
    }
    catch {
        Write-Host "[WARN] Process command-line snapshot unavailable: $($_.Exception.Message)" -ForegroundColor Yellow
        return @()
    }
}

function Wait-ForReadyFile([string]$Path, [int]$TimeoutSeconds = 20) {
    $deadline = [datetime]::UtcNow.AddSeconds($TimeoutSeconds)
    while ([datetime]::UtcNow -lt $deadline) {
        if (Test-Path -LiteralPath $Path) {
            try {
                return (Get-Content -LiteralPath $Path -Raw | ConvertFrom-Json)
            }
            catch {
                Start-Sleep -Milliseconds 100
            }
        }
        Start-Sleep -Milliseconds 200
    }
    throw "Timed out waiting for synthetic listener readiness: $Path"
}

function Stop-SyntheticListener {
    param(
        [Parameter(Mandatory)][AllowNull()]$Record,
        [Parameter(Mandatory)][string]$StopPath,
        [int]$TimeoutSeconds = 15
    )

    if (-not $Record) { return }
    New-Item -ItemType File -Path $StopPath -Force | Out-Null
    $deadline = [datetime]::UtcNow.AddSeconds($TimeoutSeconds)
    while ([datetime]::UtcNow -lt $deadline) {
        if (-not (Test-NetstatOwnership -Port ([int]$Record.Port) -ProcessId ([int]$Record.ProcessId))) {
            return
        }
        Start-Sleep -Milliseconds 250
    }
    throw "Synthetic listener PID $($Record.ProcessId) did not release port $($Record.Port) through its privileged cleanup context."
}

function Quote-ProcessArgument([string]$Value) {
    return '"' + $Value.Replace('"', '\"') + '"'
}

function Start-SyntheticListener {
    param(
        [Parameter(Mandatory)][int]$Port,
        [Parameter(Mandatory)][string]$ReadyPath,
        [Parameter(Mandatory)][string]$StopPath,
        [switch]$Elevated
    )

    Remove-Item -LiteralPath $ReadyPath, $StopPath -Force -ErrorAction SilentlyContinue
    $argumentLine = @(
        '-NoProfile'
        '-ExecutionPolicy Bypass'
        "-File $(Quote-ProcessArgument $listenerScriptPath)"
        "-Port $Port"
        "-ReadyPath $(Quote-ProcessArgument $ReadyPath)"
        "-StopPath $(Quote-ProcessArgument $StopPath)"
    ) -join ' '

    $powershellPath = Join-Path $env:WINDIR 'System32\WindowsPowerShell\v1.0\powershell.exe'
    if ($Elevated) {
        Start-Process -FilePath $powershellPath -Verb RunAs -WindowStyle Hidden -ArgumentList $argumentLine | Out-Null
    }
    else {
        Start-Process -FilePath $powershellPath -WindowStyle Hidden -ArgumentList $argumentLine | Out-Null
    }

    return Wait-ForReadyFile -Path $ReadyPath
}

function Add-ConsoleInputWriter {
    if ('TkbenConsoleInput' -as [type]) { return }
    Add-Type -TypeDefinition @'
using System;
using System.Runtime.InteropServices;

public static class TkbenConsoleInput
{
    [StructLayout(LayoutKind.Sequential, CharSet = CharSet.Unicode)]
    public struct KeyEventRecord
    {
        [MarshalAs(UnmanagedType.Bool)] public bool KeyDown;
        public ushort RepeatCount;
        public ushort VirtualKeyCode;
        public ushort VirtualScanCode;
        public char UnicodeChar;
        public uint ControlKeyState;
    }

    [StructLayout(LayoutKind.Explicit)]
    public struct InputRecord
    {
        [FieldOffset(0)] public ushort EventType;
        [FieldOffset(4)] public KeyEventRecord KeyEvent;
    }

    [DllImport("kernel32.dll", SetLastError = true)]
    private static extern IntPtr GetStdHandle(int handle);

    [DllImport("kernel32.dll", CharSet = CharSet.Unicode, SetLastError = true)]
    private static extern bool WriteConsoleInput(
        IntPtr inputHandle,
        [In] InputRecord[] records,
        uint recordCount,
        out uint written);

    public static void WriteText(string value)
    {
        var records = new InputRecord[value.Length * 2 + 2];
        var index = 0;
        foreach (var character in value)
        {
            records[index++] = KeyRecord(true, character, 0);
            records[index++] = KeyRecord(false, character, 0);
        }
        records[index++] = KeyRecord(true, '\r', 0x0d);
        records[index] = KeyRecord(false, '\r', 0x0d);
        var handle = GetStdHandle(-10);
        uint written;
        if (!WriteConsoleInput(handle, records, (uint)records.Length, out written))
        {
            throw new InvalidOperationException("WriteConsoleInput failed with Win32 error " + Marshal.GetLastWin32Error() + ".");
        }
    }

    private static InputRecord KeyRecord(bool down, char character, ushort virtualKey)
    {
        return new InputRecord
        {
            EventType = 1,
            KeyEvent = new KeyEventRecord
            {
                KeyDown = down,
                RepeatCount = 1,
                VirtualKeyCode = virtualKey,
                VirtualScanCode = 0,
                UnicodeChar = character,
                ControlKeyState = 0
            }
        };
    }
}
'@
}

function Start-InteractiveLauncher {
    param(
        [Parameter(Mandatory)][string]$LauncherPath,
        [Parameter(Mandatory)][string]$WorkingDirectory,
        [Parameter(Mandatory)][string]$LogPath
    )

    $launcherCommand = @'
$ErrorActionPreference = 'Continue'
& __LAUNCHER_PATH__ -Launch *>&1 | Tee-Object -FilePath __LOG_PATH__
exit $LASTEXITCODE
'@
    $launcherCommand = $launcherCommand.Replace(
        '__LAUNCHER_PATH__',
        (Quote-ProcessArgument $LauncherPath)
    ).Replace(
        '__LOG_PATH__',
        (Quote-ProcessArgument $LogPath)
    )
    Set-Content -LiteralPath $launcherWrapperPath -Value $launcherCommand -Encoding UTF8

    $powershellPath = Join-Path $env:WINDIR 'System32\WindowsPowerShell\v1.0\powershell.exe'
    $processParameters = @{
        FilePath = $powershellPath
        NoNewWindow = $true
        PassThru = $true
        ArgumentList = "-NoProfile -ExecutionPolicy Bypass -File $(Quote-ProcessArgument $launcherWrapperPath)"
        WorkingDirectory = $WorkingDirectory
    }
    $process = Start-Process @processParameters

    # Import-Environment and the port prompt are intentionally before any
    # dependency/setup work. Buffering the answer in the shared console keeps
    # the child genuinely interactive while remaining unattended.
    Start-Sleep -Seconds 3
    [TkbenConsoleInput]::WriteText('yes')
    return $process
}

function Write-EvidenceRecord {
    param([string]$Status)
    if ([string]::IsNullOrWhiteSpace($EvidencePath)) { return }

    $os = Get-CimInstance -ClassName Win32_OperatingSystem -ErrorAction SilentlyContinue
    $revision = (& git -C $sourceRootFull rev-parse HEAD 2>$null).Trim()
    $serviceProof = if ($servicesStarted) {
        'unexpected startup marker present'
    }
    else {
        'no startup marker; no service process observed'
    }
    $cleanupProof = if ($cleanupPassed) { 'passed' } else { 'failed' }
    $relevantOutput = @(
        $launcherOutput -split '\r?\n' |
            Where-Object { $_ -match '(?i)Configured launch ports remain occupied|Termination errors:|No service was started|access is denied' } |
            Select-Object -First 6 |
            ForEach-Object { ($_ -replace '[A-Za-z]:\\[^\s]+', '<path>').Trim() }
    ) -join ' | '
    if ([string]::IsNullOrWhiteSpace($relevantOutput)) { $relevantOutput = 'not captured' }
    $lines = @(
        '# TKBEN T0-02 permission-denied validation'
        ''
        "Date: $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss zzz')"
        'Repository: CTCycle/TKBEN-tokenizers-benchmarker'
        "Commit SHA: $revision"
        "Status: $Status"
        "Windows: $($os.Caption) $($os.Version) (build $($os.BuildNumber))"
        'Privilege arrangement: launcher ran with a non-elevated token; the protected synthetic listener ran in a separate elevated RunAs helper token.'
        'Identity record: launcher = current non-elevated test account; listener = elevated helper identity. Account names intentionally omitted.'
        "Selected FASTAPI_PORT: $protectedPort"
        "Protected synthetic PID: $($protectedRecord.ProcessId)"
        "Sentinel port/PID: $($sentinelRecord.Port) / $($sentinelRecord.ProcessId)"
        'Launcher invocation: powershell.exe -NoProfile -ExecutionPolicy Bypass -File start_on_windows.ps1 -Launch'
        'Confirmation: yes supplied through the shared interactive console input buffer.'
        "Launcher exit code: $launcherExitCode"
        "Access-denied proof: $terminationDenied"
        "Backend/frontend start proof: $serviceProof"
        "Relevant sanitized launcher output: $relevantOutput"
        "Cleanup: $cleanupProof"
        ''
        'Sanitized launcher output is stored in the run-local launcher log; no credentials or account names are recorded here.'
    )
    $parent = Split-Path -Parent ([IO.Path]::GetFullPath($EvidencePath))
    New-Item -ItemType Directory -Path $parent -Force | Out-Null
    Set-Content -LiteralPath $EvidencePath -Value $lines -Encoding UTF8
}

try {
    Assert-Harness ([Environment]::OSVersion.Platform -eq [PlatformID]::Win32NT) 'This integration harness is Windows-only.'
    Assert-Harness $AllowDisposableEnvironment.IsPresent 'Refusing to run without -AllowDisposableEnvironment. Use a disposable Windows VM, Windows Sandbox, or equivalent privilege-isolated boundary.'
    Assert-Harness (-not [Console]::IsInputRedirected -and -not [Console]::IsOutputRedirected) 'Run this harness from a real interactive console; redirected input/output cannot prove the normal launcher prompt path.'
    Assert-Harness (-not (Get-IsElevated)) 'The launcher harness must run with a non-elevated token so the protected listener can produce a genuine access-denied result.'
    Assert-Harness (Test-Path -LiteralPath (Join-Path $sourceRootFull 'start_on_windows.ps1')) "Source checkout does not contain start_on_windows.ps1: $sourceRootFull"

    Write-HarnessStep 'Creating the disposable checkout and synthetic helper scripts.'
    New-Item -ItemType Directory -Path $helperRoot -Force | Out-Null
    & git clone --local --no-hardlinks --branch develop --single-branch --no-tags $sourceRootFull $checkoutRoot | Out-Host
    Assert-Harness ($LASTEXITCODE -eq 0) 'Could not create the disposable checkout.'
    Copy-Item -LiteralPath (Join-Path $sourceRootFull 'start_on_windows.ps1') -Destination (Join-Path $checkoutRoot 'start_on_windows.ps1') -Force

    $envExamplePath = Join-Path $checkoutRoot 'settings\.env.example'
    $envPath = Join-Path $checkoutRoot 'settings\.env'
    $listenerScript = @'
[CmdletBinding()]
param(
    [Parameter(Mandatory)][int]$Port,
    [Parameter(Mandatory)][string]$ReadyPath,
    [Parameter(Mandatory)][string]$StopPath
)
$ErrorActionPreference = 'Stop'
$listener = [Net.Sockets.TcpListener]::new([Net.IPAddress]::Loopback, $Port)
try {
    $listener.Start()
    $identity = [Security.Principal.WindowsIdentity]::GetCurrent()
    $principal = [Security.Principal.WindowsPrincipal]::new($identity)
    [pscustomobject]@{
        ProcessId = $PID
        Port = $Port
        IsElevated = $principal.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)
    } | ConvertTo-Json -Compress | Set-Content -LiteralPath $ReadyPath -Encoding UTF8
    while (-not (Test-Path -LiteralPath $StopPath)) {
        Start-Sleep -Milliseconds 200
    }
}
finally {
    $listener.Stop()
    Remove-Item -LiteralPath $ReadyPath -Force -ErrorAction SilentlyContinue
}
'@
    Set-Content -LiteralPath $listenerScriptPath -Value $listenerScript -Encoding UTF8

    $protectedPort = Get-FreeTcpPort
    do { $uiPort = Get-FreeTcpPort } while ($uiPort -eq $protectedPort)
    $envLines = Get-Content -LiteralPath $envExamplePath |
        ForEach-Object {
            if ($_ -match '^FASTAPI_PORT=') { "FASTAPI_PORT=$protectedPort" }
            elseif ($_ -match '^UI_PORT=') { "UI_PORT=$uiPort" }
            else { $_ }
        }
    Set-Content -LiteralPath $envPath -Value $envLines -Encoding UTF8

    Write-HarnessStep "Starting the elevated synthetic listener on port $protectedPort. Approve only the helper UAC prompt in the disposable boundary."
    $protectedRecord = Start-SyntheticListener -Port $protectedPort -ReadyPath $protectedReadyPath -StopPath $protectedStopPath -Elevated
    Assert-Harness ([bool]$protectedRecord.IsElevated) 'The protected synthetic listener did not receive an elevated token; refusing to treat the run as permission-denied evidence.'
    Assert-Harness (Test-NetstatOwnership -Port $protectedPort -ProcessId ([int]$protectedRecord.ProcessId)) "netstat -ano did not confirm protected PID $($protectedRecord.ProcessId) owns port $protectedPort."
    Write-HarnessOk "netstat -ano confirmed protected synthetic PID $($protectedRecord.ProcessId) owns port $protectedPort."

    $sentinelPort = Get-FreeTcpPort
    $sentinelRecord = Start-SyntheticListener -Port $sentinelPort -ReadyPath $sentinelReadyPath -StopPath $sentinelStopPath
    Assert-Harness (-not [bool]$sentinelRecord.IsElevated) 'The unrelated sentinel unexpectedly ran elevated.'
    Assert-Harness (Test-NetstatOwnership -Port $sentinelPort -ProcessId ([int]$sentinelRecord.ProcessId)) "netstat -ano did not confirm sentinel PID $($sentinelRecord.ProcessId) owns port $sentinelPort."

    Add-ConsoleInputWriter
    $beforeProcesses = @(Get-ProcessCommandLines)
    $launcherPath = Join-Path $checkoutRoot 'start_on_windows.ps1'
    Write-HarnessStep 'Running the normal interactive launcher path and supplying yes to its termination prompt.'
    $launcherProcess = Start-InteractiveLauncher -LauncherPath $launcherPath -WorkingDirectory $checkoutRoot -LogPath $launcherLogPath

    $deadline = [datetime]::UtcNow.AddSeconds(45)
    while (-not $launcherProcess.HasExited -and [datetime]::UtcNow -lt $deadline) {
        if (-not (Test-NetstatOwnership -Port $protectedPort -ProcessId ([int]$protectedRecord.ProcessId))) {
            Stop-Process -Id $launcherProcess.Id -Force -ErrorAction SilentlyContinue
            throw 'The protected synthetic listener disappeared before the launcher reported denial; the run cannot prove failure-closed behavior.'
        }
        Start-Sleep -Milliseconds 250
    }
    if (-not $launcherProcess.HasExited) {
        Stop-Process -Id $launcherProcess.Id -Force -ErrorAction SilentlyContinue
        throw 'The launcher did not exit after the denied termination attempt.'
    }

    $launcherExitCode = $launcherProcess.ExitCode
    $launcherOutput = if (Test-Path -LiteralPath $launcherLogPath) { Get-Content -LiteralPath $launcherLogPath -Raw } else { '' }
    $terminationDenied = $launcherOutput -match '(?i)access is denied|permission denied|not permitted'
    Assert-Harness ($launcherExitCode -ne 0) "The protected-listener launch unexpectedly exited with code $launcherExitCode."
    Assert-Harness $terminationDenied 'Launcher output did not contain a genuine access-denied/permission-denied termination error.'
    Assert-Harness ($launcherOutput -match '(?i)Configured launch ports remain occupied') 'Launcher output did not report the remaining occupied configured port.'
    Assert-Harness ($launcherOutput -match "(?i)PID $($protectedRecord.ProcessId)") 'Launcher output did not report the blocked protected PID.'
    Assert-Harness ($launcherOutput -match "(?i)$protectedPort") 'Launcher output did not report the blocked configured port.'
    Assert-Harness ($launcherOutput -match '(?i)Termination errors:') 'Launcher output did not include recorded termination errors.'
    Assert-Harness ($launcherOutput -match '(?i)No service was started') 'Launcher output did not state that service startup was blocked.'
    Assert-Harness (Test-NetstatOwnership -Port $protectedPort -ProcessId ([int]$protectedRecord.ProcessId)) 'The protected listener no longer owns the configured port after the denied termination attempt.'
    Assert-Harness (Test-NetstatOwnership -Port $sentinelPort -ProcessId ([int]$sentinelRecord.ProcessId)) 'The unrelated sentinel listener was terminated or lost its port.'

    $afterProcesses = @(Get-ProcessCommandLines)
    $beforeIds = @($beforeProcesses | ForEach-Object { [int]$_.ProcessId })
    $newServiceProcesses = @($afterProcesses | Where-Object {
        $isNew = $beforeIds -notcontains [int]$_.ProcessId
        $commandLine = [string]$_.CommandLine
        $isService = $commandLine -match '(?i)uvicorn\s+server\.app:app|npm(?:\.cmd)?\s+run\s+preview|npm-cli\.js.*\brun\s+preview'
        $isNew -and $isService
    })
    $servicesStarted = $newServiceProcesses.Count -gt 0 -or
        $launcherOutput -match '(?i)Starting backend|Starting frontend preview'
    Assert-Harness (-not $servicesStarted) 'FastAPI or Angular preview startup was observed despite the denied port release.'

    $resultStatus = 'PASS'
    Write-HarnessOk 'Real permission-denied launch validation passed.'
}
catch {
    $resultStatus = 'FAIL'
    Write-Host "[FAIL] $($_.Exception.Message)" -ForegroundColor Red
}
finally {
    Write-HarnessStep 'Cleaning up synthetic listeners through their helper contexts.'
    if ($sentinelRecord) {
        try { Stop-SyntheticListener -Record $sentinelRecord -StopPath $sentinelStopPath } catch { $cleanupPassed = $false; Write-Host "[WARN] Sentinel cleanup: $($_.Exception.Message)" -ForegroundColor Yellow }
    }
    if ($protectedRecord) {
        try { Stop-SyntheticListener -Record $protectedRecord -StopPath $protectedStopPath } catch { $cleanupPassed = $false; Write-Host "[WARN] Protected-listener cleanup: $($_.Exception.Message)" -ForegroundColor Yellow }
    }
    if (-not $KeepCheckout -and (Test-Path -LiteralPath $harnessRoot)) {
        try { Remove-Item -LiteralPath $harnessRoot -Recurse -Force -ErrorAction Stop } catch { $cleanupPassed = $false; Write-Host "[WARN] Disposable checkout cleanup: $($_.Exception.Message)" -ForegroundColor Yellow }
    }
    if (-not $cleanupPassed -and $resultStatus -eq 'PASS') { $resultStatus = 'FAIL' }
    try { Write-EvidenceRecord -Status $resultStatus } catch { Write-Host "[WARN] Could not write evidence record: $($_.Exception.Message)" -ForegroundColor Yellow }
}

if ($resultStatus -ne 'PASS') { exit 1 }
exit 0
