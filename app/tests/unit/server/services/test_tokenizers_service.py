from __future__ import annotations

from pathlib import Path

from server.services.tokenizers import TokenizersService

###############################################################################
class FakeTokenizerRepository:

    # -------------------------------------------------------------------------
    def __init__(self) -> None:
        self.inserted: list[str] = []

    # -------------------------------------------------------------------------
    def tokenizer_exists(self, tokenizer_id: str) -> bool:
        return tokenizer_id == "exists"

    # -------------------------------------------------------------------------
    def insert_if_missing(self, tokenizer_id: str) -> None:
        self.inserted.append(tokenizer_id)

    # -------------------------------------------------------------------------
    def get_latest_tokenizer_report(self, tokenizer_name: str):
        return object() if tokenizer_name == "exists" else None

    # -------------------------------------------------------------------------
    def get_tokenizer_report_by_id(self, report_id: int):
        return object() if report_id == 1 else None

###############################################################################
def test_tokenizers_service_uses_repository_layer(monkeypatch) -> None:
    service = TokenizersService()
    fake_repo = FakeTokenizerRepository()
    service.repository = fake_repo  # type: ignore[assignment]

    assert service.is_tokenizer_persisted("exists") is True
    service.insert_tokenizer_if_missing("new")
    assert fake_repo.inserted == ["new"]

###############################################################################
def test_tokenizers_service_report_prechecks(monkeypatch) -> None:
    service = TokenizersService()
    service.repository = FakeTokenizerRepository()  # type: ignore[assignment]

    monkeypatch.setattr(
        service.report_serializer,
        "load_latest_tokenizer_report",
        lambda tokenizer_name: {"tokenizer_name": tokenizer_name},
    )
    monkeypatch.setattr(
        service.report_serializer,
        "load_tokenizer_report_by_id",
        lambda report_id: {"report_id": report_id},
    )
    monkeypatch.setattr(
        service.report_serializer,
        "load_tokenizer_vocabulary_page",
        lambda report_id, offset, limit: {
            "report_id": report_id,
            "offset": offset,
            "limit": limit,
            "items": [],
        },
    )

    assert service.get_latest_tokenizer_report("exists") is not None
    assert service.get_latest_tokenizer_report("missing") is None

    assert service.get_tokenizer_report_by_id(1) is not None
    assert service.get_tokenizer_report_by_id(2) is None

    assert service.get_tokenizer_report_vocabulary(1, 0, 10) is not None
    assert service.get_tokenizer_report_vocabulary(2, 0, 10) is None


###############################################################################
def test_tokenizer_scan_keeps_only_supported_text_pipeline_models(monkeypatch) -> None:
    service = TokenizersService()

    ###############################################################################
    class FakeModel:

        # -------------------------------------------------------------------------
        def __init__(self, model_id: str, pipeline_tag: str | None) -> None:
            self.modelId = model_id
            self.pipeline_tag = pipeline_tag

    ###############################################################################
    class FakeApi:

        # -------------------------------------------------------------------------
        def list_models(self, **kwargs):
            return [
                FakeModel("bert-base-uncased", "fill-mask"),
                FakeModel("XiaomiMiMo/MiMo-Audio-Tokenizer", None),
                FakeModel("turkeyju/tokenizer_tatitok_bl128_vq", "image-tokenization"),
            ]

    monkeypatch.setattr("server.services.tokenizers.HfApi", lambda **kwargs: FakeApi())
    monkeypatch.setattr(service.key_service, "get_active_key", lambda: None)

    assert service.get_tokenizer_identifiers(limit=100) == ["bert-base-uncased"]


###############################################################################
def test_failed_tokenizer_download_cleans_partial_cache_and_returns_reason(
    monkeypatch,
) -> None:
    service = TokenizersService()
    cache_dir = "G:/TKBEN-test-cache/broken"
    removed: list[str] = []

    monkeypatch.setattr(service.key_service, "get_active_key", lambda: None)
    monkeypatch.setattr(service, "is_tokenizer_persisted", lambda _: False)
    monkeypatch.setattr(service, "get_tokenizer_cache_dir", lambda _: cache_dir)
    monkeypatch.setattr(service, "has_cached_tokenizer", lambda _: False)
    monkeypatch.setattr(
        "server.services.tokenizers.shutil.rmtree",
        lambda path, ignore_errors=False: removed.append(str(path)),
    )
    monkeypatch.setattr(
        "server.services.tokenizers.AutoTokenizer.from_pretrained",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            ValueError("Unrecognized model configuration")
        ),
    )

    result = service.download_and_persist(["broken/model"])

    assert result["failed"] == ["broken/model"]
    assert result["failed_count"] == 1
    assert "ValueError: Unrecognized model configuration" in result["failed_details"][0]
    assert removed == [str(Path(cache_dir))]
