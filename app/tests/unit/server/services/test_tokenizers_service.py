from __future__ import annotations

import time
from pathlib import Path

import pytest

from server.contracts.tokenizers import TokenizerDiscoveryQuery
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
def tokenizer_siblings(*filenames: str) -> list[dict[str, str]]:
    return [{"rfilename": filename} for filename in filenames]

###############################################################################
def test_tokenizers_service_report_prechecks(monkeypatch) -> None:
    service = TokenizerReportingService()
    service.repository = FakeTokenizerRepository()  # type: ignore[assignment]

    monkeypatch.setattr(
        service.report_repository,
        "load_latest_tokenizer_report",
        lambda tokenizer_name: {"tokenizer_name": tokenizer_name},
    )
    monkeypatch.setattr(
        service.report_repository,
        "load_tokenizer_report_by_id",
        lambda report_id: {"report_id": report_id},
    )
    monkeypatch.setattr(
        service.report_repository,
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
def test_tokenizer_discovery_passes_native_filters_and_returns_structured_items(monkeypatch) -> None:
    service = TokenizersService()

    ###############################################################################
    class FakeModel:

        # -------------------------------------------------------------------------
        def __init__(self, model_id: str, pipeline_tag: str | None) -> None:
            self.modelId = model_id
            self.pipeline_tag = pipeline_tag
            self.library_name = "transformers"
            self.downloads = 42
            self.likes = 7
            self.lastModified = None
            self.gated = False
            self.tags = ["core-model"]
            self.config = None
            self.siblings = tokenizer_siblings("tokenizer.json", "config.json")

    ###############################################################################
    class FakeApi:

        captured: dict[str, object] = {}

        # -------------------------------------------------------------------------
        def list_models(self, **kwargs):
            self.captured = kwargs
            return [
                FakeModel("bert-base-uncased", "fill-mask"),
                FakeModel("XiaomiMiMo/MiMo-Audio-Tokenizer", None),
                FakeModel("turkeyju/tokenizer_tatitok_bl128_vq", "image-tokenization"),
            ]

    fake_api = FakeApi()
    monkeypatch.setattr("server.services.tokenizers.HfApi", lambda **kwargs: fake_api)
    monkeypatch.setattr(service.key_service, "get_active_key", lambda: None)

    response = service.discover_tokenizers(TokenizerDiscoveryQuery(
        search=" bert ",
        limit=5,
        pipeline_tag="fill-mask",
        author="google",
        include_tags=["core-model"],
        access="public",
        sort="downloads",
    ))

    assert fake_api.captured == {
        "sort": "downloads",
        "direction": -1,
        "limit": 15,
        "search": "bert",
        "author": "google",
        "pipeline_tag": "fill-mask",
        "filter": ["core-model"],
        "gated": False,
        "expand": [
            "siblings",
            "pipeline_tag",
            "library_name",
            "downloads",
            "likes",
            "lastModified",
            "gated",
            "tags",
        ],
    }
    assert response.count == 1
    assert response.fetched_count == 3
    assert response.items[0].model_dump() == {
        "identifier": "bert-base-uncased",
        "pipeline_tag": "fill-mask",
        "library_name": "transformers",
        "downloads": 42,
        "likes": 7,
        "last_modified": None,
        "gated": False,
        "tags": ["core-model"],
        "vocabulary_size": None,
    }

###############################################################################
def test_remove_custom_tokenizer_removes_registry_entry_and_persisted_row(monkeypatch) -> None:
    service = TokenizersService()
    registry = service.custom_tokenizer_registry
    registry.clear()
    registry.set("CUSTOM_sample", object())

    deleted: list[str] = []

    ###############################################################################
    class FakeRepository:

        # -------------------------------------------------------------------------
        def delete_tokenizer(self, name: str) -> bool:
            deleted.append(name)
            return True

    service.repository = FakeRepository()  # type: ignore[assignment]

    assert service.remove_downloaded_tokenizer("CUSTOM_sample") is True
    assert registry.get("CUSTOM_sample") is None
    assert deleted == ["CUSTOM_sample"]

###############################################################################
def test_remove_downloaded_tokenizer_cleans_cache_before_database_delete(tmp_path, monkeypatch) -> None:
    service = TokenizersService()
    cache_dir = tmp_path / "google-bert__bert-base-uncased"
    cache_dir.mkdir()
    (cache_dir / "tokenizer.json").write_text("{}", encoding="utf-8")
    monkeypatch.setattr(service, "get_tokenizer_cache_dir", lambda name: str(cache_dir))

    calls: list[str] = []

    ###############################################################################
    class FakeRepository:

        # -------------------------------------------------------------------------
        def delete_tokenizer(self, name: str) -> bool:
            assert not cache_dir.exists()
            calls.append(name)
            return True

    service.repository = FakeRepository()  # type: ignore[assignment]

    assert service.remove_downloaded_tokenizer("google-bert/bert-base-uncased") is True
    assert calls == ["google-bert/bert-base-uncased"]

###############################################################################
def test_remove_downloaded_tokenizer_keeps_database_row_when_cache_cleanup_fails(monkeypatch) -> None:
    service = TokenizersService()
    monkeypatch.setattr(service, "get_tokenizer_cache_dir", lambda name: "cache")
    monkeypatch.setattr("server.services.tokenizers.Path.exists", lambda self: True)
    monkeypatch.setattr("server.services.tokenizers.shutil.rmtree", lambda path: (_ for _ in ()).throw(OSError("locked")))

    deleted = False

    ###############################################################################
    class FakeRepository:

        # -------------------------------------------------------------------------
        def delete_tokenizer(self, name: str) -> bool:
            nonlocal deleted
            deleted = True
            return True

    service.repository = FakeRepository()  # type: ignore[assignment]

    with pytest.raises(RuntimeError, match="Failed to remove downloaded tokenizer files"):
        service.remove_downloaded_tokenizer("google-bert/bert-base-uncased")
    assert deleted is False

###############################################################################
def test_tokenizer_discovery_uses_bounded_overfetch_and_candidate_cap(monkeypatch) -> None:
    service = TokenizersService()

    ###############################################################################
    class FakeApi:

        captured: dict[str, object] = {}

        # -------------------------------------------------------------------------
        def list_models(self, **kwargs):
            self.captured = kwargs
            return []

    fake_api = FakeApi()
    monkeypatch.setattr("server.services.tokenizers.HfApi", lambda **kwargs: fake_api)
    monkeypatch.setattr(service.key_service, "get_active_key", lambda: None)

    service.discover_tokenizers(TokenizerDiscoveryQuery(
        limit=5,
        exclude_tags=["audio"],
        vocabulary_sort="ascending",
    ))
    assert fake_api.captured["limit"] == 15
    assert fake_api.captured["expand"][-1] == "config"
    assert "fetch_config" not in fake_api.captured

    service.discover_tokenizers(TokenizerDiscoveryQuery(
        limit=250,
        vocabulary_sort="ascending",
    ))
    assert fake_api.captured["limit"] == 750
    assert fake_api.captured["expand"][0] == "siblings"

###############################################################################
def test_tokenizer_discovery_accepts_supported_root_artifacts_only(monkeypatch) -> None:
    service = TokenizersService()

    ###############################################################################
    class FakeModel:

        # -------------------------------------------------------------------------
        def __init__(self, model_id: str, filenames: tuple[str, ...]) -> None:
            self.modelId = model_id
            self.pipeline_tag = "text-generation"
            self.tags = []
            self.siblings = tokenizer_siblings(*filenames)

    ###############################################################################
    class FakeApi:

        # -------------------------------------------------------------------------
        def list_models(self, **kwargs):
            assert kwargs["expand"][0] == "siblings"
            return [
                FakeModel("valid/fast", ("tokenizer.json", "tokenizer_config.json")),
                FakeModel("valid/sentencepiece", ("tokenizer.model", "config.json")),
                FakeModel("valid/spiece", ("spiece.model", "config.json")),
                FakeModel("valid/bpe", ("vocab.json", "merges.txt")),
                FakeModel("valid/wordpiece", ("vocab.txt", "config.json")),
                FakeModel("invalid/weights-only", ("config.json", "model.safetensors")),
                FakeModel("invalid/metadata-only", ("config.json", "tokenizer_config.json")),
                FakeModel("invalid/nested-only", ("config.json", "nested/tokenizer.json")),
                FakeModel("invalid/partial-vocab", ("vocab.json",)),
                FakeModel("invalid/no-siblings", ()),
            ]

    monkeypatch.setattr("server.services.tokenizers.HfApi", lambda **kwargs: FakeApi())
    monkeypatch.setattr(service.key_service, "get_active_key", lambda: None)

    response = service.discover_tokenizers(TokenizerDiscoveryQuery(limit=20))

    assert [item.identifier for item in response.items] == [
        "valid/fast",
        "valid/sentencepiece",
        "valid/spiece",
        "valid/bpe",
        "valid/wordpiece",
    ]
    assert response.count == 5
    assert response.fetched_count == 10

###############################################################################
def test_tokenizer_discovery_filters_excluded_and_unsupported_models(monkeypatch) -> None:
    service = TokenizersService()

    ###############################################################################
    class FakeModel:

        # -------------------------------------------------------------------------
        def __init__(self, model_id: str, pipeline_tag: str, tags: list[str]) -> None:
            self.modelId = model_id
            self.pipeline_tag = pipeline_tag
            self.tags = tags
            self.siblings = tokenizer_siblings("tokenizer.json", "config.json")

    ###############################################################################
    class FakeApi:

        # -------------------------------------------------------------------------
        def list_models(self, **kwargs):
            del kwargs
            return [
                FakeModel("text/model", "fill-mask", ["safe"]),
                FakeModel("audio/model", "audio-tokenization", ["safe"]),
                FakeModel("unsafe/model", "text-generation", ["unsafe"]),
            ]

    monkeypatch.setattr("server.services.tokenizers.HfApi", lambda **kwargs: FakeApi())
    monkeypatch.setattr(service.key_service, "get_active_key", lambda: None)

    response = service.discover_tokenizers(TokenizerDiscoveryQuery(
        limit=10,
        exclude_tags=["unsafe"],
    ))
    assert [item.identifier for item in response.items] == ["text/model"]

###############################################################################
def test_tokenizer_discovery_extracts_and_filters_vocabulary_metadata(monkeypatch) -> None:
    service = TokenizersService()

    ###############################################################################
    class FakeModel:

        # -------------------------------------------------------------------------
        def __init__(self, model_id: str, vocabulary_size: object) -> None:
            self.modelId = model_id
            self.pipeline_tag = "fill-mask"
            self.config = {"vocab_size": vocabulary_size} if vocabulary_size is not None else None
            self.siblings = tokenizer_siblings("vocab.txt", "config.json")

    ###############################################################################
    class FakeApi:

        # -------------------------------------------------------------------------
        def list_models(self, **kwargs):
            assert "config" in kwargs["expand"]
            return [FakeModel("small/model", 4), FakeModel("large/model", 12), FakeModel("unknown/model", None)]

    monkeypatch.setattr("server.services.tokenizers.HfApi", lambda **kwargs: FakeApi())
    monkeypatch.setattr(service.key_service, "get_active_key", lambda: None)

    minimum = service.discover_tokenizers(TokenizerDiscoveryQuery(
        limit=10,
        vocabulary_operator="at_least",
        vocabulary_size=10,
    ))
    assert [(item.identifier, item.vocabulary_size) for item in minimum.items] == [("large/model", 12)]

    maximum = service.discover_tokenizers(TokenizerDiscoveryQuery(
        limit=10,
        vocabulary_operator="at_most",
        vocabulary_size=5,
    ))
    assert [(item.identifier, item.vocabulary_size) for item in maximum.items] == [("small/model", 4)]

###############################################################################
def test_tokenizer_discovery_orders_known_vocabulary_before_unknown(monkeypatch) -> None:
    service = TokenizersService()

    ###############################################################################
    class FakeModel:

        # -------------------------------------------------------------------------
        def __init__(self, model_id: str, vocabulary_size: int | None) -> None:
            self.modelId = model_id
            self.pipeline_tag = "fill-mask"
            self.config = {"vocab_size": vocabulary_size} if vocabulary_size is not None else None
            self.siblings = tokenizer_siblings("vocab.json", "merges.txt")

    ###############################################################################
    class FakeApi:

        # -------------------------------------------------------------------------
        def list_models(self, **kwargs):
            del kwargs
            return [FakeModel("unknown/model", None), FakeModel("large/model", 12), FakeModel("small/model", 4)]

    monkeypatch.setattr("server.services.tokenizers.HfApi", lambda **kwargs: FakeApi())
    monkeypatch.setattr(service.key_service, "get_active_key", lambda: None)

    ascending = service.discover_tokenizers(TokenizerDiscoveryQuery(limit=10, vocabulary_sort="ascending"))
    descending = service.discover_tokenizers(TokenizerDiscoveryQuery(limit=10, vocabulary_sort="descending"))
    assert [item.identifier for item in ascending.items] == ["small/model", "large/model", "unknown/model"]
    assert [item.identifier for item in descending.items] == ["large/model", "small/model", "unknown/model"]

###############################################################################
def test_tokenizer_discovery_propagates_upstream_failure(monkeypatch) -> None:
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
        service.discover_tokenizers(TokenizerDiscoveryQuery(limit=5))

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

###############################################################################
def test_tokenizer_download_timeout_returns_failure_without_blocking_job(
    monkeypatch,
) -> None:
    service = TokenizersService()
    service.TOKENIZER_DOWNLOAD_TIMEOUT_SECONDS = 0.01
    cache_dir = Path("timeout-cache")

    monkeypatch.setattr(service.key_service, "get_active_key", lambda: None)
    monkeypatch.setattr(service.repository, "tokenizer_exists", lambda _: False)
    monkeypatch.setattr(service, "get_tokenizer_cache_dir", lambda _: str(cache_dir))
    monkeypatch.setattr(service, "has_cached_tokenizer", lambda _: False)

    def slow_load(*args, **kwargs):
        del args, kwargs
        time.sleep(0.05)
        return object()

    monkeypatch.setattr(
        "server.services.tokenizers.AutoTokenizer.from_pretrained",
        slow_load,
    )

    started = time.monotonic()
    result = service.download_and_persist(["slow/model"])

    assert time.monotonic() - started < 0.04
    assert result["failed"] == ["slow/model"]
    assert "TokenizerDownloadTimeoutError" in result["failed_details"][0]

###############################################################################
def test_tokenizer_catalog_filters_cached_sources_search_and_vocabulary(
    monkeypatch,
) -> None:
    service = TokenizersService()

    ###############################################################################
    class CustomTokenizer:

        # -------------------------------------------------------------------------
        def get_vocab_size(self) -> int:
            return 4

    monkeypatch.setattr(
        service.repository,
        "list_downloaded_tokenizer_catalog",
        lambda: [
            ("zeta/model", True, {"vocabulary_size": 100}),
            ("alpha/model", False, {"vocabulary_size": 3}),
            ("uncached/model", False, {"vocabulary_size": 1}),
        ],
    )
    monkeypatch.setattr(
        service,
        "has_cached_tokenizer",
        lambda name: name != "uncached/model",
    )
    monkeypatch.setattr(
        service.custom_tokenizer_registry,
        "snapshot",
        lambda: {"CUSTOM_demo": CustomTokenizer()},
    )
    monkeypatch.setattr(
        service.repository,
        "get_latest_tokenizer_report",
        lambda _: None,
    )

    assert [item["tokenizer_name"] for item in service.list_tokenizer_catalog()] == [
        "alpha/model",
        "CUSTOM_demo",
        "zeta/model",
    ]
    assert service.list_tokenizer_catalog(
        source="custom",
        search="DEMO",
        vocabulary_size_operator="at_most",
        vocabulary_size=5,
    ) == [{
        "tokenizer_name": "CUSTOM_demo",
        "source": "custom",
        "has_report": False,
        "vocabulary_size": 4,
    }]
    assert [item["tokenizer_name"] for item in service.list_tokenizer_catalog(
        vocabulary_size_operator="at_least",
        vocabulary_size=50,
    )] == ["zeta/model"]
