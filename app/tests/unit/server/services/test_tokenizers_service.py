from __future__ import annotations

from pathlib import Path

import pytest

from server.services.tokenizer_reporting import TokenizerReportingService
from server.services.tokenizers import TokenizersService

###############################################################################
class FakeTokenizerRepository:

    # -------------------------------------------------------------------------
    def get_latest_tokenizer_report(self, tokenizer_name: str):
        return object() if tokenizer_name == "exists" else None

    # -------------------------------------------------------------------------
    def get_tokenizer_report_by_id(self, report_id: int):
        return object() if report_id == 1 else None

###############################################################################
def test_tokenizers_service_report_prechecks(monkeypatch) -> None:
    service = TokenizerReportingService()
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
def test_tokenizer_scan_propagates_upstream_failure(monkeypatch) -> None:
    service = TokenizersService()

    ###############################################################################
    class FailingApi:

        # -------------------------------------------------------------------------
        def list_models(self, **kwargs):
            del kwargs
            raise RuntimeError("upstream outage details")

    monkeypatch.setattr("server.services.tokenizers.HfApi", lambda **kwargs: FailingApi())
    monkeypatch.setattr(service.key_service, "get_active_key", lambda: None)

    with pytest.raises(RuntimeError, match="upstream outage details"):
        service.get_tokenizer_identifiers(limit=100)

###############################################################################
def test_failed_tokenizer_download_cleans_partial_cache_and_returns_reason(
    monkeypatch,
    tmp_path,
) -> None:
    service = TokenizersService()
    cache_dir = tmp_path / "broken"
    removed: list[str] = []

    monkeypatch.setattr(service.key_service, "get_active_key", lambda: None)
    monkeypatch.setattr(service.repository, "tokenizer_exists", lambda _: False)
    monkeypatch.setattr(
        service,
        "get_tokenizer_cache_dir",
        lambda _: str(cache_dir),
    )
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
