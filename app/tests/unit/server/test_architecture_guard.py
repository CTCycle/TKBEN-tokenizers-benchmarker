from __future__ import annotations

import ast
from pathlib import Path


SERVER_ROOT = Path(__file__).parents[4] / "server"

###############################################################################
def _python_files() -> list[Path]:
    return [path for path in SERVER_ROOT.rglob("*.py") if "__pycache__" not in path.parts]

###############################################################################
def test_api_modules_do_not_import_repositories() -> None:
    violations: list[str] = []
    for path in (SERVER_ROOT / "api").glob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module and node.module.startswith("server.repositories"):
                violations.append(f"{path}:{node.lineno}")
            if isinstance(node, ast.Import):
                violations.extend(
                    f"{path}:{node.lineno}"
                    for alias in node.names
                    if alias.name.startswith("server.repositories")
                )
    assert violations == []

###############################################################################
def test_production_imports_are_top_level_and_files_are_bounded() -> None:
    violations: list[str] = []
    for path in _python_files():
        source = path.read_text(encoding="utf-8")
        if len(source.splitlines()) > 1000:
            violations.append(f"{path}: exceeds 1000 lines")
        tree = ast.parse(source)
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                for child in ast.walk(node):
                    if child is node:
                        continue
                    if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        violations.append(f"{path}:{child.lineno}: nested function")
                    if isinstance(child, (ast.Import, ast.ImportFrom)):
                        violations.append(f"{path}:{child.lineno}: local import")
            if isinstance(node, ast.ClassDef):
                for child in ast.walk(node):
                    if isinstance(child, (ast.Import, ast.ImportFrom)):
                        violations.append(f"{path}:{child.lineno}: class-local import")

        for node in ast.walk(tree):
            if not isinstance(node, (ast.If, ast.Try)):
                continue
            for child in ast.walk(node):
                if isinstance(child, (ast.Import, ast.ImportFrom)):
                    violations.append(f"{path}:{child.lineno}: conditional import")

        for node in tree.body:
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                continue
            if isinstance(node, (ast.Expr, ast.Assign, ast.AnnAssign)):
                continue
            if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            if isinstance(node, ast.If):
                for child in ast.walk(node):
                    if isinstance(child, (ast.Import, ast.ImportFrom)):
                        violations.append(f"{path}:{child.lineno}: conditional import")

        if (
            "serialization.data import" in source
            or "repositories.serialization.data." in source
            or ("api.helpers" in source and path.parent.name != "api")
        ):
            violations.append(f"{path}: obsolete module import")
    assert violations == []
