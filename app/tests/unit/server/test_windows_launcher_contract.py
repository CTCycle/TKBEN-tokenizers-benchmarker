from __future__ import annotations

import re
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[4]
LAUNCHER = (REPOSITORY_ROOT / "start_on_windows.ps1").read_text(encoding="utf-8")


def _section(function_name: str, next_function_name: str | None = None) -> str:
    start = LAUNCHER.index(f"function {function_name}")
    if next_function_name is None:
        return LAUNCHER[start:]
    end = LAUNCHER.index(f"function {next_function_name}", start)
    return LAUNCHER[start:end]


def test_launcher_has_no_always_rebuild_switch() -> None:
    assert "ALWAYS_REBUILD" not in LAUNCHER


def test_launcher_pins_and_checks_the_uv_runtime_version() -> None:
    install_runtimes = _section("Install-Runtimes", "Sync-BackendDependencies")
    readiness = _section("Test-BackendDependenciesReady", "Test-FrontendBuildReady")

    assert "$UvVersion = '0.12.17'" in LAUNCHER
    assert "/download/$UvVersion/$uvArchive" in install_runtimes
    assert "not the required $UvVersion" in install_runtimes
    assert "releases/latest" not in install_runtimes
    assert "$UvExe --version" in readiness
    assert "Escape($UvVersion)" in readiness


def test_launch_guards_ports_before_starting_services() -> None:
    launch = _section("Launch-Application", "Install-Dependencies")

    assert launch.index("Confirm-ReleaseLaunchPorts") < launch.index("Start-Process")
    assert launch.count("Confirm-ReleaseLaunchPorts") == 2


def test_normal_launch_and_frontend_sync_do_not_kill_port_listeners() -> None:
    launch = _section("Launch-Application", "Install-Dependencies")
    frontend_sync = _section("Sync-Frontend", "Test-BackendDependenciesReady")

    assert "Stop-PortListeners" not in LAUNCHER
    assert "Stop-ProcessTree" not in launch
    assert "Stop-PortListeners" not in frontend_sync


def test_port_release_uses_one_explicit_process_termination_without_tree_kill() -> None:
    release = _section("Confirm-ReleaseLaunchPorts", "Stop-ProcessTree")

    assert "Read-Host" in release
    assert "Stop-Process -Id $processId -Force" in release
    assert "Stop-ProcessTree" not in release
    assert "taskkill.exe" not in release


def test_backend_repair_is_not_coupled_to_frontend_build() -> None:
    launch = _section("Launch-Application", "Install-Dependencies")
    backend_sync = _section("Sync-BackendDependencies", "Sync-Dependencies")

    assert "Sync-BackendDependencies" in launch
    assert "Sync-Dependencies -BuildFrontend" not in launch
    assert "Stop-ApplicationProcesses" not in backend_sync


def test_explicit_install_and_rebuild_still_build_frontend() -> None:
    install = _section("Install-Dependencies", "Rebuild-Frontend")
    rebuild = _section("Rebuild-Frontend", "Read-InstallationType")

    assert "Sync-Dependencies -BuildFrontend" in install
    assert "Sync-Frontend -BuildFrontend" in rebuild


def test_normal_launch_build_is_gated_by_frontend_build_readiness() -> None:
    launch = _section("Launch-Application", "Install-Dependencies")

    assert "if (-not (Test-FrontendBuildReady))" in launch
    assert "Sync-Frontend -BuildFrontend -UseCachedFrontendDependencies" in launch


def test_production_fingerprint_uses_only_confirmed_build_inputs() -> None:
    inputs = _section("Get-FrontendBuildInputFiles", "Get-FrontendSourceFingerprint")

    assert "spec\\.ts" in inputs
    assert "proxy.conf.cjs" not in inputs
    for required_path in (
        "angular.json",
        "package.json",
        "package-lock.json",
        "tsconfig.json",
        "tsconfig.app.json",
    ):
        assert required_path in inputs
    assert "Get-ChildItem -LiteralPath (Join-Path $ClientDir 'public')" in inputs


def test_dependency_stamp_and_locked_backend_sync_are_present() -> None:
    backend_sync = _section("Sync-BackendDependencies", "Sync-Dependencies")

    assert "Get-BackendDependencyStampPath" in LAUNCHER
    assert "dependencyFingerprint" in LAUNCHER
    assert "--locked" in backend_sync
    assert "installationProfile" in LAUNCHER


def test_public_operations_initialize_environment_once() -> None:
    launch = _section("Launch-Application", "Install-Dependencies")
    sync = _section("Sync-Dependencies", "Sync-Frontend")
    frontend = _section("Sync-Frontend", "Test-BackendDependenciesReady")

    assert launch.count("Import-Environment") == 1
    assert "Import-Environment" not in sync
    assert "Import-Environment" not in frontend


def test_tracked_actions_invoke_scriptblocks_without_method_stream_conversion() -> None:
    tracked = _section("Invoke-TrackedLauncherAction", "Ensure-Directory")

    assert "& $Action" in tracked
    assert "$Action.Invoke()" not in tracked


def test_launcher_maintenance_menu_routes_are_all_dispatched() -> None:
    menu = _section("Get-LauncherMenuEntries", "Write-MenuItem")
    dispatch = _section("Show-Menu")

    menu_keys = set(re.findall(r"Key\s*=\s*'([A-Za-z]+)'", menu))
    dispatch_keys = set(re.findall(r"^\s*'([A-Za-z]+)'\s*\{", dispatch, re.MULTILINE))

    assert menu_keys == {
        "Launch",
        "Install",
        "Rebuild",
        "Database",
        "Tests",
        "Check",
        "Update",
        "Logs",
        "Cache",
        "AllData",
        "Uninstall",
        "KillAll",
        "Exit",
    }
    assert dispatch_keys == menu_keys - {"Exit"}


def test_launcher_install_and_destructive_routes_keep_explicit_guards() -> None:
    profiles = _section("Read-InstallationType", "Invoke-DatabaseInitialization")
    confirmation = _section("Confirm-DestructiveAction", "Remove-Logs")
    menu = _section("Get-LauncherMenuEntries", "Write-MenuItem")
    dispatch = _section("Show-Menu")

    assert "'Standard'" in profiles
    assert "'Development'" in profiles
    for key in ("Logs", "Cache", "AllData", "Uninstall", "KillAll"):
        assert re.search(rf"Key\s*=\s*'{key}'.*Destructive\s*=\s*\$true", menu)
    assert "requires an interactive console" in confirmation
    assert "return $false" in confirmation
    assert "-notmatch '^(?i:y|yes)$'" in confirmation
    assert "Confirm-DestructiveAction" in dispatch


def test_update_route_fails_closed_on_develop_without_switching_branches() -> None:
    update = _section("Update-Application", "Check-ForUpdates")

    assert "$branch -ne 'main'" in update
    assert "clean Git working tree" in update
    assert "No files were changed." in update
    assert not re.search(
        r"^\s*&\s*git\s+(checkout|switch)\b",
        update,
        re.MULTILINE | re.IGNORECASE,
    )


def test_killall_recognizes_quoted_npm_preview_processes() -> None:
    process_scan = _section("Get-ApplicationProcessIds", "Stop-ApplicationProcesses")
    match = re.search(
        r"\$isFrontend\s*=\s*\$commandLine\s+-match\s+'([^']+)'",
        process_scan,
    )

    assert match is not None
    frontend_pattern = match.group(1)
    assert re.search(
        frontend_pattern,
        '"C:\\TKBEN\\runtimes\\nodejs\\npm.cmd" run preview -- --port 8000',
    )
    assert re.search(
        frontend_pattern,
        '"C:\\TKBEN\\runtimes\\nodejs\\node_modules\\npm\\bin\\npm-cli.js" run preview',
    )


def test_killall_stops_only_outermost_matching_process_trees() -> None:
    process_scan = _section("Get-ApplicationProcessIds", "Stop-ApplicationProcesses")

    assert "$processesById" in process_scan
    assert "$matchedProcessIds -contains $parentProcessId" in process_scan
    assert "$hasMatchedAncestor" in process_scan
    assert "return @($rootProcessIds | Sort-Object -Unique)" in process_scan
