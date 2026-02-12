from __future__ import annotations

from typing import Any

from huggingface_hub import HfApi

from TKBEN.server.common.utils.logger import logger


###############################################################################
class TokenizersService:
    """
    Service for fetching tokenizer information from HuggingFace.

    This is a webapp-specific service that provides tokenizer scanning
    functionality without the desktop app dependencies.
    """

    PIPELINE_TAGS = [
        "text-generation",
        "fill-mask",
        "text-classification",
        "token-classification",
        "text2text-generation",
        "question-answering",
        "sentence-similarity",
        "translation",
        "summarization",
        "conversational",
        "zero-shot-classification",
    ]

    def __init__(self, hf_access_token: str | None = None) -> None:
        self.hf_access_token = hf_access_token

    def get_tokenizer_identifiers(self, limit: int = 100) -> list[Any]:
        """
        Retrieve the most downloaded tokenizer identifiers from Hugging Face.

        Args:
            limit: Maximum number of identifiers to request (default 100).

        Returns:
            List with the identifiers of the retrieved tokenizers ordered by
            popularity (downloads).
        """
        api = HfApi(token=self.hf_access_token) if self.hf_access_token else HfApi()

        try:
            models = api.list_models(
                search="tokenizer", sort="downloads", direction=-1, limit=limit
            )
        except Exception:
            logger.warning("Failed to retrieve tokenizer identifiers from HuggingFace")
            logger.debug("Tokenizer identifier fetch failed", exc_info=True)
            return []

        identifiers = [m.modelId for m in models]

        return identifiers

