from __future__ import annotations

import re
from pathlib import Path, PurePosixPath

IDENTIFIER_PATTERN = re.compile(
    r"^[A-Za-z0-9][A-Za-z0-9._-]*(?:/[A-Za-z0-9][A-Za-z0-9._-]*)*$"
)
SAFE_FILENAME_CHARS_PATTERN = re.compile(r"[^A-Za-z0-9._-]+")
CONTROL_CHAR_PATTERN = re.compile(r"[\x00-\x1f\x7f]")

###############################################################################
def normalize_identifier(
    value: str,
    field_name: str,
    *,
    max_length: int = 200,
) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{field_name} must be a string.")
    normalized = value.strip()
    if not normalized:
        raise ValueError(f"{field_name} must not be empty.")
    if len(normalized) > max_length:
        raise ValueError(f"{field_name} is too long (max {max_length} characters).")
    if "\\" in normalized:
        raise ValueError(f"{field_name} contains invalid path separators.")
    if not IDENTIFIER_PATTERN.fullmatch(normalized):
        raise ValueError(
            f"{field_name} contains unsupported characters. "
            "Use letters, numbers, '.', '_', '-', and '/'."
        )
    return normalized

###############################################################################
def normalize_optional_identifier(
    value: str | None,
    field_name: str,
    *,
    max_length: int = 200,
) -> str | None:
    if value is None:
        return None
    return normalize_identifier(value, field_name, max_length=max_length)

###############################################################################
def contains_control_chars(value: str) -> bool:
    return bool(CONTROL_CHAR_PATTERN.search(value))

###############################################################################
def normalize_upload_stem(filename: str, *, max_length: int = 120) -> str:
    if not isinstance(filename, str):
        raise ValueError("Uploaded filename must be a string.")
    normalized_name = filename.strip().replace("\\", "/")
    if not normalized_name:
        raise ValueError("Uploaded filename must not be empty.")
    base_name = PurePosixPath(normalized_name).name
    stem = Path(base_name).stem.strip()
    if not stem:
        raise ValueError("Uploaded filename is missing a valid stem.")
    cleaned = SAFE_FILENAME_CHARS_PATTERN.sub("_", stem).strip("._-")
    if not cleaned:
        raise ValueError("Uploaded filename stem is not valid.")
    return cleaned[:max_length]

###############################################################################
def ensure_path_is_within(base_path: str | Path, candidate_path: str | Path) -> str:
    base_abs = Path(base_path).resolve()
    candidate_abs = Path(candidate_path).resolve()
    try:
        candidate_abs.relative_to(base_abs)
    except ValueError as exc:
        message = "Resolved path escapes the allowed base directory."
        if getattr(exc, "args", ()) and "different anchors" in str(exc):
            message = "Path validation failed across different roots."
        raise ValueError(message) from exc
    return str(candidate_abs)
