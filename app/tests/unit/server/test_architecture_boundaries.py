from __future__ import annotations

import ast
from pathlib import Path


SERVER_ROOT = Path(__file__).resolve().parents[3] / "server"

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
