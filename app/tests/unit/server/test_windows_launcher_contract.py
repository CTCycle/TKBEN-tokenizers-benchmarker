from __future__ import annotations

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
