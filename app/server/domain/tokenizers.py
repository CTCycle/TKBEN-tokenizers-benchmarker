from __future__ import annotations

from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator, model_validator

from server.common.utils.security import normalize_identifier

###############################################################################
class TokenizerSignature(BaseModel):
    identifier: str
    records: list[dict[str, Any]] = Field(default_factory=list)

    # -------------------------------------------------------------------------
    @field_validator("identifier")
    @classmethod
    def validate_identifier(cls, value: str) -> str:
        return normalize_identifier(value, "Tokenizer identifier", max_length=160)

###############################################################################
class SupportedTokenizerPipeline(StrEnum):
    TEXT_GENERATION = "text-generation"
    FILL_MASK = "fill-mask"
    TEXT_CLASSIFICATION = "text-classification"
    TOKEN_CLASSIFICATION = "token-classification"
    TEXT2TEXT_GENERATION = "text2text-generation"
    QUESTION_ANSWERING = "question-answering"
    SENTENCE_SIMILARITY = "sentence-similarity"
    TRANSLATION = "translation"
    SUMMARIZATION = "summarization"
    ZERO_SHOT_CLASSIFICATION = "zero-shot-classification"


###############################################################################
class TokenizerDiscoverySort(StrEnum):
    DOWNLOADS = "downloads"
    LIKES = "likes"
    LAST_MODIFIED = "last_modified"
    CREATED_AT = "created_at"

###############################################################################
class TokenizerDiscoveryQuery(BaseModel):
    search: str | None = Field(default=None, max_length=160)
    limit: int = Field(default=50, ge=1, le=250)
    pipeline_tag: SupportedTokenizerPipeline | None = None
    author: str | None = Field(default=None, max_length=160)
    include_tags: list[str] = Field(default_factory=list)
    exclude_tags: list[str] = Field(default_factory=list)
    access: Literal["all", "public", "gated"] = "all"
    sort: TokenizerDiscoverySort = TokenizerDiscoverySort.DOWNLOADS
    vocabulary_operator: Literal["at_least", "at_most"] | None = None
    vocabulary_size: int | None = Field(default=None, ge=0)
    vocabulary_sort: Literal["none", "ascending", "descending"] = "none"

    # -------------------------------------------------------------------------
    @field_validator("search", "author", mode="before")
    @classmethod
    def normalize_optional_text(cls, value: object) -> str | None:
        if value is None:
            return None
        normalized = str(value).strip()
        return normalized or None

    # -------------------------------------------------------------------------
    @field_validator("include_tags", "exclude_tags", mode="before")
    @classmethod
    def normalize_tags(cls, value: object) -> list[str]:
        raw_values = value if isinstance(value, list) else [value]
        normalized: list[str] = []
        seen: set[str] = set()
        for raw_value in raw_values:
            if not isinstance(raw_value, str):
                raise ValueError("Tokenizer discovery tags must be strings.")
            for tag in raw_value.replace("\n", ",").split(","):
                cleaned = tag.strip()
                key = cleaned.casefold()
                if cleaned and key not in seen:
                    seen.add(key)
                    normalized.append(cleaned)
        if len(normalized) > 8:
            raise ValueError("Tokenizer discovery supports at most 8 tags per list.")
        return normalized

    # -------------------------------------------------------------------------
    @model_validator(mode="after")
    def validate_discovery_query(self) -> "TokenizerDiscoveryQuery":
        include = {tag.casefold() for tag in self.include_tags}
        exclude = {tag.casefold() for tag in self.exclude_tags}
        if include & exclude:
            raise ValueError("A tokenizer discovery tag cannot be both required and excluded.")
        if self.vocabulary_operator is not None and self.vocabulary_size is None:
            raise ValueError("Vocabulary operator requires a vocabulary size.")
        return self

###############################################################################
class TokenizerDiscoveryItem(BaseModel):
    identifier: str
    pipeline_tag: str | None = None
    library_name: str | None = None
    downloads: int | None = Field(default=None, ge=0)
    likes: int | None = Field(default=None, ge=0)
    last_modified: str | None = None
    gated: bool | str | None = None
    tags: list[str] = Field(default_factory=list)
    vocabulary_size: int | None = Field(default=None, ge=0)

###############################################################################
class TokenizerDiscoveryResponse(BaseModel):
    items: list[TokenizerDiscoveryItem] = Field(default_factory=list)
    count: int = Field(default=0, ge=0)
    fetched_count: int = Field(default=0, ge=0)

###############################################################################
class TokenizerListItem(BaseModel):
    tokenizer_name: str
    source: Literal["huggingface", "custom"]
    has_report: bool = False
    vocabulary_size: int | None = Field(default=None, ge=0)

###############################################################################
class TokenizerListResponse(BaseModel):
    tokenizers: list[TokenizerListItem] = Field(default_factory=list)
    count: int = Field(default=0)

###############################################################################
class TokenizerDownloadRequest(BaseModel):
    tokenizers: list[str] = Field(
        default_factory=list,
        description="Tokenizer IDs to download and persist",
    )

    # -------------------------------------------------------------------------
    @field_validator("tokenizers")
    @classmethod
    def validate_tokenizers(cls, value: list[str]) -> list[str]:
        normalized: list[str] = []
        seen: set[str] = set()
        for tokenizer in value:
            cleaned = normalize_identifier(
                tokenizer,
                "Tokenizer identifier",
                max_length=160,
            )
            if cleaned in seen:
                continue
            seen.add(cleaned)
            normalized.append(cleaned)
        if len(normalized) > 200:
            raise ValueError("Too many tokenizers requested (max 200).")
        return normalized

###############################################################################
class TokenizerDownloadResponse(BaseModel):
    status: str = Field(default="success")
    downloaded: list[str] = Field(default_factory=list)
    already_downloaded: list[str] = Field(default_factory=list)
    failed: list[str] = Field(default_factory=list)
    failed_details: list[str] = Field(default_factory=list)
    requested_count: int = Field(default=0)
    downloaded_count: int = Field(default=0)
    already_downloaded_count: int = Field(default=0)
    failed_count: int = Field(default=0)

###############################################################################
class TokenizerSettingsResponse(BaseModel):
    default_discovery_limit: int
    max_discovery_limit: int
    max_discovery_candidates: int
    metadata_candidate_multiplier: int

###############################################################################
class TokenizerUploadResponse(BaseModel):
    """Response schema for custom tokenizer upload."""

    status: str = Field(default="success")
    tokenizer_name: str = Field(..., description="Name assigned to uploaded tokenizer")
    is_compatible: bool = Field(..., description="Whether tokenizer is compatible")

###############################################################################
class TokenizerReportGenerateRequest(BaseModel):
    tokenizer_name: str = Field(..., description="Persisted tokenizer name")

    # -------------------------------------------------------------------------
    @field_validator("tokenizer_name")
    @classmethod
    def validate_tokenizer_name(cls, value: str) -> str:
        return normalize_identifier(value, "Tokenizer name", max_length=160)

###############################################################################
class TokenizerLengthHistogram(BaseModel):
    bins: list[str] = Field(default_factory=list)
    counts: list[int] = Field(default_factory=list)
    bin_edges: list[float] = Field(default_factory=list)
    min_length: int = Field(default=0)
    max_length: int = Field(default=0)
    mean_length: float = Field(default=0.0)
    median_length: float = Field(default=0.0)

###############################################################################
class TokenizerReportResponse(BaseModel):
    status: str = Field(default="success")
    report_id: int
    report_version: int = Field(default=1)
    created_at: str
    tokenizer_name: str
    description: str | None = None
    huggingface_url: str | None = None
    global_stats: dict[str, Any] = Field(default_factory=dict)
    token_length_histogram: TokenizerLengthHistogram = Field(
        default_factory=TokenizerLengthHistogram
    )
    vocabulary_size: int = Field(default=0)

###############################################################################
class TokenizerVocabularyItem(BaseModel):
    token_id: int
    token: str
    length: int

###############################################################################
class TokenizerVocabularyPageResponse(BaseModel):
    status: str = Field(default="success")
    report_id: int
    tokenizer_name: str
    offset: int
    limit: int
    total: int
    items: list[TokenizerVocabularyItem] = Field(default_factory=list)

###############################################################################
class CustomTokenizersDeleteResponse(BaseModel):
    status: str = Field(default="success")
    message: str

###############################################################################
class TokenizerDeleteResponse(BaseModel):
    status: str = Field(default="success")
    tokenizer_name: str
    message: str
