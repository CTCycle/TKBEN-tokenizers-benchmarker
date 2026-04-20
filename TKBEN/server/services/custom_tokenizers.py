from __future__ import annotations

from functools import lru_cache
from threading import RLock
from typing import Any


###############################################################################
class CustomTokenizerRegistry:
    def __init__(self) -> None:
        self._lock = RLock()
        self._tokenizers: dict[str, Any] = {}

    # -------------------------------------------------------------------------
    def get(self, tokenizer_name: str) -> Any | None:
        with self._lock:
            return self._tokenizers.get(tokenizer_name)

    # -------------------------------------------------------------------------
    def set(self, tokenizer_name: str, tokenizer: Any) -> None:
        with self._lock:
            self._tokenizers[tokenizer_name] = tokenizer

    # -------------------------------------------------------------------------
    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return dict(self._tokenizers)

    # -------------------------------------------------------------------------
    def clear(self) -> None:
        with self._lock:
            self._tokenizers.clear()


###############################################################################
@lru_cache(maxsize=1)
def get_custom_tokenizer_registry() -> CustomTokenizerRegistry:
    return CustomTokenizerRegistry()
