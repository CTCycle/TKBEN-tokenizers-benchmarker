from __future__ import annotations

import ast
from pathlib import Path


SERVER_ROOT = Path(__file__).resolve().parents[3] / "server"
REPOSITORY_ROOT = SERVER_ROOT.parents[1]

FORBIDDEN_IMPORTS: dict[str, tuple[str, ...]] = {
    "api": ("server.repositories",),
    "contracts": (
        "server.api",
        "server.services",
        "server.repositories",
        "fastapi",
        "sqlalchemy",
    ),
    "services": ("server.api",),
    "repositories": ("server.api", "server.services"),
    "configurations": ("server.api",),
}

LEGACY_IMPORTS = (
    "server.domain",
    "server.repositories.serialization",
    "server.configurations.management",
    "server.common.utils.types",
)

REMOVED_COMPATIBILITY_PATHS = (
    SERVER_ROOT / "domain",
    SERVER_ROOT / "repositories" / "serialization.py",
    SERVER_ROOT / "configurations" / "management.py",
    SERVER_ROOT / "common" / "utils" / "types.py",
)

LEGACY_CACHE_TOKENS = (
    "$LegacyCachePaths",
    ".uv-cache",
    ".pytest_cache",
    ".ruff_cache",
    ".mypy_cache",
)


###############################################################################
def _imported_modules(path: Path) -> list[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    modules: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            modules.append(node.module)
    return modules


###############################################################################
def _production_python_files() -> list[Path]:
    return sorted(
        path
        for path in SERVER_ROOT.rglob("*.py")
        if not any(part in {".venv", "__pycache__"} for part in path.parts)
    )


###############################################################################
def _starts_with_module(module: str, prefix: str) -> bool:
    return module == prefix or module.startswith(f"{prefix}.")


###############################################################################
def test_configuration_bootstraps_before_sensitive_imports() -> None:
    configuration_init = SERVER_ROOT / "configurations" / "__init__.py"
    tree = ast.parse(
        configuration_init.read_text(encoding="utf-8"),
        filename=str(configuration_init),
    )

    bootstrap_index = next(
        index
        for index, node in enumerate(tree.body)
        if (
            isinstance(node, ast.Expr)
            and isinstance(node.value, ast.Call)
            and isinstance(node.value.func, ast.Name)
            and node.value.func.id == "ensure_environment_loaded"
        )
    )
    sensitive_import_indices = [
        index
        for index, node in enumerate(tree.body)
        if (
            isinstance(node, ast.ImportFrom)
            and node.module
            in {
                "server.configurations.settings",
                "server.configurations.startup",
            }
        )
    ]

    assert sensitive_import_indices
    assert bootstrap_index < min(sensitive_import_indices)


###############################################################################
def test_production_layers_respect_dependency_boundaries() -> None:
    violations: list[str] = []
    for path in _production_python_files():
        relative = path.relative_to(SERVER_ROOT)
        layer = relative.parts[0] if len(relative.parts) > 1 else ""
        forbidden = FORBIDDEN_IMPORTS.get(layer, ())
        for module in _imported_modules(path):
            for prefix in forbidden:
                if _starts_with_module(module, prefix):
                    violations.append(f"{relative}: imports {module}")

    assert not violations, "\n".join(violations)


###############################################################################
def test_production_code_has_no_legacy_architecture_imports() -> None:
    violations: list[str] = []
    for path in _production_python_files():
        relative = path.relative_to(SERVER_ROOT)
        for module in _imported_modules(path):
            if any(_starts_with_module(module, prefix) for prefix in LEGACY_IMPORTS):
                violations.append(f"{relative}: imports {module}")

    assert not violations, "\n".join(violations)


###############################################################################
def test_removed_compatibility_paths_do_not_return() -> None:
    existing = [str(path.relative_to(REPOSITORY_ROOT)) for path in REMOVED_COMPATIBILITY_PATHS if path.exists()]
    assert not existing, "Removed compatibility paths were recreated:\n" + "\n".join(existing)


###############################################################################
def test_launcher_has_no_legacy_cache_compatibility_paths() -> None:
    launcher = (REPOSITORY_ROOT / "start_on_windows.ps1").read_text(encoding="utf-8")
    violations = [token for token in LEGACY_CACHE_TOKENS if token in launcher]
    assert not violations, "Launcher still contains legacy cache compatibility tokens: " + ", ".join(violations)
