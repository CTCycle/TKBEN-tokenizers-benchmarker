from pathlib import Path


LAUNCHER = Path(__file__).parents[2] / ".." / "start_on_windows.ps1"


###############################################################################
def test_launcher_guards_console_only_menu_operations() -> None:
    source = LAUNCHER.resolve().read_text(encoding="utf-8")

    assert "function Clear-MenuScreen" in source
    assert "if ([Console]::IsOutputRedirected) { return }" in source
    assert "Clear-MenuScreen" in source
    assert "if (-not [Console]::IsOutputRedirected)" in source
    assert "Clear-Host" in source

###############################################################################
def test_launcher_keeps_safe_database_initialization_command() -> None:
    source = LAUNCHER.resolve().read_text(encoding="utf-8")

    assert "Write-MenuItem '3' 'Initialize database'" in source
    assert "initialize_database.py')" in source
    assert "--drop-existing" not in source
    assert "--seed-catalogs" not in source
    assert "--force-reseed-catalogs" not in source
