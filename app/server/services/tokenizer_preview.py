# Copyright © 2023 Thomas Virdis
# Licensed under the MIT License.

from __future__ import annotations

from server.common.utils.logger import logger
from server.contracts.tokenizers import (
    TokenizerPreviewRequest,
    TokenizerPreviewResult,
    TokenizerPreviewToken,
)
from server.services.benchmarks import BenchmarkService
from server.services.tokenizer_adapters import UniversalTokenizerAdapter

###############################################################################
class TokenizerPreviewService:
    """Encode a short text with persisted tokenizers without storing results."""

    # -------------------------------------------------------------------------
    def __init__(self) -> None:
        self.benchmark_service = BenchmarkService()

    # -------------------------------------------------------------------------
    def preview(self, request: TokenizerPreviewRequest) -> list[TokenizerPreviewResult]:
        missing = set(
            self.benchmark_service.get_missing_persisted_tokenizers(request.tokenizers)
        )
        loaded = self.benchmark_service.load_tokenizers(
            [name for name in request.tokenizers if name not in missing]
        )
        results: list[TokenizerPreviewResult] = []
        for name in request.tokenizers:
            if name in missing:
                results.append(self._error(name, "Tokenizer is not persisted locally."))
                continue
            tokenizer = loaded.get(name)
            if tokenizer is None:
                results.append(
                    self._error(name, "Tokenizer failed to load or is incompatible.")
                )
                continue
            results.append(self._encode(name, tokenizer, request))
        return results

    # -------------------------------------------------------------------------
    def _encode(
        self, name: str, tokenizer: object, request: TokenizerPreviewRequest
    ) -> TokenizerPreviewResult:
        try:
            encoded = UniversalTokenizerAdapter(name, tokenizer).encode_batch(
                [request.text],
                add_special_tokens=request.add_special_tokens,
                padding=False,
                truncation=False,
                max_length=None,
            )
            token_ids = encoded.input_ids_by_doc[0]
            tokens = self.benchmark_service.tools.convert_ids_to_tokens(
                tokenizer, token_ids, ""
            )
            if len(tokens) != len(token_ids):
                raise ValueError("Tokenizer could not map token IDs to tokens.")
        except Exception:
            logger.warning("Tokenizer preview failed for %s", name, exc_info=True)
            return self._error(name, "Tokenizer failed to encode the preview text.")
        return TokenizerPreviewResult(
            tokenizer_name=name,
            status="success",
            tokens=[
                TokenizerPreviewToken(token_id=token_id, token=token)
                for token_id, token in zip(token_ids, tokens)
            ],
            token_count=encoded.token_counts[0],
            unknown_count=encoded.unknown_counts[0],
        )

    # -------------------------------------------------------------------------
    @staticmethod
    def _error(name: str, message: str) -> TokenizerPreviewResult:
        return TokenizerPreviewResult(
            tokenizer_name=name, status="error", error=message
        )
