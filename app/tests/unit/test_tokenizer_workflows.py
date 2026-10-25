# Copyright © 2023 Thomas Virdis
# Licensed under the MIT License.

from __future__ import annotations

from server.repositories.frequencies import DiskBackedFrequencyStore
from datetime import datetime
from datetime import timezone
import pytest
from sqlalchemy import create_engine
from sqlalchemy import select
from sqlalchemy.orm import Session
from server.repositories.database.backend import get_database
from server.repositories.schemas.models import Base
from server.repositories.schemas.models import HFAccessKey
from server.services.keys import HFAccessKeyConflictError
from server.services.keys import HFAccessKeyNotFoundError
from server.services.keys import HFAccessKeyService
from server.services.keys import HFAccessKeyValidationError
from types import SimpleNamespace
from typing import Any
from server.repositories.queries.data import DataRepositoryQueries
from server.repositories.tokenizer_reports import TokenizerReportRepository
from server.repositories.schemas.models import Tokenizer
from server.services.tokenizer_reporting import TokenizerReportingService
from server.common.tokenizer_metrics import compute_vocabulary_shape_metrics
import threading
from pathlib import Path
from server.contracts.tokenizers import TokenizerDiscoveryQuery
from server.repositories.benchmarks import BenchmarkRepository
from server.repositories.tokenizers import TokenizerRepository
from server.services.benchmarks import BenchmarkService
from server.services.tokenizers import TokenizersService
from tokenizers import Tokenizer as Tokenizer__tokenizer_adapter_contract
from tokenizers import models
from tokenizers import pre_tokenizers
from tokenizers import processors
from server.services.tokenizer_adapters import UniversalTokenizerAdapter


###############################################################################
def test_frequency_store_flush_batches_large_token_lookups() -> None:
    store = DiskBackedFrequencyStore(memory_limit=50_000)
    try:
        total_tokens = 2_500
        for index in range(total_tokens):
            store.add(f"token-{index}")

        store.flush()

        assert store.unique_count() == total_tokens
        assert store.total_count() == total_tokens

        for index in range(total_tokens):
            store.add(f"token-{index}")

        store.flush()

        token_counts = dict(store.top_k(5_000))
        assert token_counts["token-0"] == 2
        assert token_counts["token-2499"] == 2
    finally:
        store.close()


###############################################################################
@pytest.fixture
def isolated_engine(monkeypatch: pytest.MonkeyPatch):
    engine = create_engine("sqlite+pysqlite:///:memory:", future=True)
    Base.metadata.create_all(engine, checkfirst=True)
    database = get_database()
    monkeypatch.setattr(database.backend, "engine", engine)
    try:
        yield engine
    finally:
        engine.dispose()


###############################################################################
def test_get_active_key_raises_validation_error_on_invalid_decryption(
    isolated_engine,
) -> None:
    del isolated_engine
    service = HFAccessKeyService()


    ###############################################################################
    class FailingCipher:

        # -------------------------------------------------------------------------
        def decrypt(self, encrypted_value: str) -> str:
            del encrypted_value
            raise ValueError("invalid token")

    with Session(bind=get_database().backend.engine) as session:
        session.add(
            HFAccessKey(
                key_value="encrypted-value",
                created_at=datetime.now(timezone.utc),
                is_active=True,
            )
        )
        session.commit()

    service._cipher = FailingCipher()  # type: ignore[assignment]
    with pytest.raises(HFAccessKeyValidationError, match="cannot be decrypted"):
        service.get_active_key()


###############################################################################
def test_add_key_skips_undecryptable_rows_during_duplicate_check(
    isolated_engine,
) -> None:
    del isolated_engine
    service = HFAccessKeyService()


    ###############################################################################
    class RecoveringCipher:

        # -------------------------------------------------------------------------
        def encrypt(self, plaintext: str) -> str:
            return f"enc:{plaintext}"

        # -------------------------------------------------------------------------
        def decrypt(self, encrypted_value: str) -> str:
            if encrypted_value == "stale-encrypted":
                raise ValueError("invalid token")
            if encrypted_value.startswith("enc:"):
                return encrypted_value[4:]
            return encrypted_value

    with Session(bind=get_database().backend.engine) as session:
        session.add(
            HFAccessKey(
                key_value="stale-encrypted",
                created_at=datetime.now(timezone.utc),
                is_active=False,
            )
        )
        session.commit()

    service._cipher = RecoveringCipher()  # type: ignore[assignment]
    result = service.add_key("hf_test_key")
    assert result.is_active is False

    with Session(bind=get_database().backend.engine) as session:
        rows = (
            session.execute(select(HFAccessKey).order_by(HFAccessKey.id.asc()))
            .scalars()
            .all()
        )
    assert len(rows) == 2
    assert rows[1].key_value == "enc:hf_test_key"


###############################################################################
def test_add_key_detects_duplicate_raw_key_across_encryption(
    isolated_engine,
) -> None:
    del isolated_engine
    service = HFAccessKeyService()


    ###############################################################################
    class NonDeterministicCipher:

        # -------------------------------------------------------------------------
        def __init__(self) -> None:
            self.encryptions = 0

        # -------------------------------------------------------------------------
        def encrypt(self, plaintext: str) -> str:
            self.encryptions += 1
            return f"enc:{self.encryptions}:{plaintext}"

        # -------------------------------------------------------------------------
        def decrypt(self, encrypted_value: str) -> str:
            return encrypted_value.rsplit(":", 1)[-1]

    service._cipher = NonDeterministicCipher()  # type: ignore[assignment]

    service.add_key("hf_duplicate_key")
    with pytest.raises(HFAccessKeyConflictError, match="already stored"):
        service.add_key("hf_duplicate_key")

    with Session(bind=get_database().backend.engine) as session:
        rows = session.execute(select(HFAccessKey)).scalars().all()
    assert len(rows) == 1


###############################################################################
def test_set_active_key_is_idempotent_for_already_active_key(
    isolated_engine,
) -> None:
    del isolated_engine
    service = HFAccessKeyService()

    with Session(bind=get_database().backend.engine) as session:
        session.add_all(
            [
                HFAccessKey(
                    key_value="enc:key-1",
                    created_at=datetime.now(timezone.utc),
                    is_active=True,
                ),
                HFAccessKey(
                    key_value="enc:key-2",
                    created_at=datetime.now(timezone.utc),
                    is_active=False,
                ),
            ]
        )
        session.commit()
        key_id = int(
            session.execute(
                select(HFAccessKey.id).where(HFAccessKey.key_value == "enc:key-1")
            ).scalar_one()
        )

    service.set_active_key(key_id)

    with Session(bind=get_database().backend.engine) as session:
        rows = (
            session.execute(select(HFAccessKey).order_by(HFAccessKey.id.asc()))
            .scalars()
            .all()
        )
    assert any(row.id == key_id and row.is_active for row in rows)
    assert all(row.is_active is (row.id == key_id) for row in rows)


###############################################################################
def test_activating_second_key_deactivates_first(
    isolated_engine,
) -> None:
    del isolated_engine
    service = HFAccessKeyService()

    with Session(bind=get_database().backend.engine) as session:
        session.add_all(
            [
                HFAccessKey(
                    key_value="enc:key-1",
                    created_at=datetime.now(timezone.utc),
                    is_active=False,
                ),
                HFAccessKey(
                    key_value="enc:key-2",
                    created_at=datetime.now(timezone.utc),
                    is_active=False,
                ),
            ]
        )
        session.commit()
        key_ids = list(
            session.execute(select(HFAccessKey.id).order_by(HFAccessKey.id.asc()))
            .scalars()
            .all()
        )

    service.set_active_key(int(key_ids[0]))
    service.set_active_key(int(key_ids[1]))

    with Session(bind=get_database().backend.engine) as session:
        rows = (
            session.execute(select(HFAccessKey).order_by(HFAccessKey.id.asc()))
            .scalars()
            .all()
        )
    assert rows[0].is_active is False
    assert rows[1].is_active is True


###############################################################################
def test_unknown_activation_does_not_clear_existing_active_key(
    isolated_engine,
) -> None:
    del isolated_engine
    service = HFAccessKeyService()

    with Session(bind=get_database().backend.engine) as session:
        session.add(
            HFAccessKey(
                key_value="enc:key-1",
                created_at=datetime.now(timezone.utc),
                is_active=True,
            )
        )
        session.commit()
        key_id = int(session.execute(select(HFAccessKey.id)).scalar_one())

    with pytest.raises(HFAccessKeyNotFoundError, match="not found"):
        service.set_active_key(404)

    with Session(bind=get_database().backend.engine) as session:
        row = session.get(HFAccessKey, key_id)
    assert row is not None
    assert row.is_active is True


###############################################################################
def test_compute_subword_word_stats_excludes_special_tokens_and_classifies_markers() -> (
    None
):
    service = TokenizerReportingService()
    stats = service.compute_subword_word_stats(
        vocab_tokens=[
            "##ing",
            "token",
            "▁hello",
            "he▁llo",
            "Ġword",
            "wordĠpiece",
            "sub@@",
            "Ċnewline",
            "lineĊbreak",
            "[CLS]",
            "<pad>",
        ],
        special_tokens={"[CLS]", "<pad>"},
    )

    assert stats["subword_count"] == 5
    assert stats["word_count"] == 4
    assert stats["considered_count"] == 9
    assert stats["subword_to_word_ratio"] == pytest.approx(1.25)
    assert stats["subword_percentage"] == pytest.approx(55.5555, rel=1e-3)
    assert stats["word_percentage"] == pytest.approx(44.4444, rel=1e-3)


###############################################################################
@pytest.mark.parametrize("field", ["metadata", "token_length_histogram"])
def test_tokenizer_report_response_rejects_json_encoded_storage(field: str) -> None:
    repository = TokenizerReportRepository.__new__(TokenizerReportRepository)
    storage = {
        "id": 1,
        "report_version": 1,
        "created_at": None,
        "tokenizer_name": "custom/test",
        "metadata": {},
        "token_length_histogram": {},
    }
    storage[field] = "{}"

    with pytest.raises(ValueError, match="native JSON object"):
        repository._build_tokenizer_report_response(storage)


###############################################################################
def test_resolve_hf_repo_metadata_returns_link_when_description_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service = TokenizerReportingService()

    def raise_model_info(*args: Any, **kwargs: Any) -> Any:
        raise RuntimeError("model info unavailable")

    def raise_model_card(*args: Any, **kwargs: Any) -> Any:
        raise RuntimeError("model card unavailable")

    monkeypatch.setattr(
        "server.services.tokenizer_reporting.HfApi.model_info",
        raise_model_info,
    )
    monkeypatch.setattr(
        "server.services.tokenizer_reporting.ModelCard.load",
        raise_model_card,
    )

    description, huggingface_url = service.resolve_hf_repo_metadata("bert-base-uncased")

    assert description is None
    assert huggingface_url == "https://huggingface.co/bert-base-uncased"


###############################################################################
class DummyBackendTokenizerModel:
    pass


###############################################################################
class DummyBackendTokenizer:

    # -------------------------------------------------------------------------
    def __init__(self) -> None:
        self.model = DummyBackendTokenizerModel()


###############################################################################
class DummyTokenizer:
    special_tokens_map = {"cls_token": "[CLS]"}
    all_special_tokens = ["[CLS]", "<pad>"]
    all_special_ids = [2, 99]
    added_tokens_encoder = {"<pad>": 99}
    vocab_size = 30_522
    model_max_length = 512
    padding_side = "right"
    init_kwargs = {
        "model_max_length": 512,
        "padding_side": "right",
    }

    backend_tokenizer = DummyBackendTokenizer()

    # -------------------------------------------------------------------------
    def get_vocab(self) -> dict[str, int]:
        return {
            "hello": 0,
            "##ing": 1,
            "[CLS]": 2,
            "▁world": 3,
            "he▁llo": 4,
            "Ġtoken": 5,
            "wordĠpiece": 6,
        }


###############################################################################
def test_generate_report_payload_includes_hf_url_and_subword_stats(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service = TokenizerReportingService()
    captured_report: dict[str, Any] = {}

    monkeypatch.setattr(
        service.repository,
        "get_tokenizer_source",
        lambda tokenizer_name: "huggingface",
    )
    monkeypatch.setattr(service, "has_cached_tokenizer", lambda tokenizer_name: True)
    monkeypatch.setattr(
        "server.services.tokenizer_reporting.AutoTokenizer.from_pretrained",
        lambda *args, **kwargs: DummyTokenizer(),
    )
    monkeypatch.setattr(service, "find_cached_file", lambda *args, **kwargs: None)
    monkeypatch.setattr(service, "load_json_if_present", lambda *args, **kwargs: {})
    monkeypatch.setattr(
        service,
        "resolve_hf_repo_metadata",
        lambda tokenizer_name: (None, f"https://huggingface.co/{tokenizer_name}"),
    )

    monkeypatch.setattr(
        service.report_repository,
        "replace_report_and_vocabulary",
        lambda tokenizer_name, report, vocabulary_rows: (
            captured_report.update(report) or 123
        ),
    )

    report = service.generate_and_store_report("bert-base-uncased")

    assert report["report_id"] == 123
    assert report["huggingface_url"] == "https://huggingface.co/bert-base-uncased"
    assert report["global_stats"]["persistence_mode"] == "filesystem_required"
    assert "subword_word_stats" in report["global_stats"]
    assert report["global_stats"]["base_vocabulary_size"] == 30522
    assert report["global_stats"]["model_max_length"] == 512
    assert report["global_stats"]["padding_side"] == "right"
    assert report["global_stats"]["added_tokens_count"] == 1
    assert report["global_stats"]["special_tokens_ids_count"] == 2
    assert report["global_stats"]["special_tokens_count"] == 2
    assert report["global_stats"]["vocabulary_stats"]["subword_like_count"] == 3
    shape_metric_keys = {
        "token_length_std",
        "token_length_p90",
        "token_length_cv",
        "single_character_token_percentage",
    }
    vocabulary_stats = report["global_stats"]["vocabulary_stats"]
    assert shape_metric_keys <= vocabulary_stats.keys()
    assert shape_metric_keys <= report["token_length_histogram"].keys()
    assert all(
        vocabulary_stats[key]
        == captured_report["global_stats"]["vocabulary_stats"][key]
        for key in shape_metric_keys
    )
    assert (
        report["global_stats"]["vocabulary_stats"]["special_tokens_in_vocab_count"] == 1
    )
    assert (
        captured_report["huggingface_url"] == "https://huggingface.co/bert-base-uncased"
    )
    assert (
        captured_report["global_stats"]["vocabulary_stats"]["unique_token_lengths"] == 3
    )


###############################################################################
def test_tokenizer_report_repository_roundtrip_preserves_huggingface_url() -> None:
    engine = create_engine("sqlite+pysqlite:///:memory:", future=True)
    Base.metadata.create_all(engine)
    now = datetime.now(timezone.utc)
    tokenizer_name = "test/tokenizer-report-roundtrip"
    with Session(engine) as session:
        session.add(
            Tokenizer(name=tokenizer_name, source="huggingface", created_at=now)
        )
        session.commit()
    repository = TokenizerReportRepository(
        DataRepositoryQueries(SimpleNamespace(backend=SimpleNamespace(engine=engine)))
    )
    report = {
        "report_version": 1,
        "created_at": "2026-02-17T00:00:00Z",
        "tokenizer_name": tokenizer_name,
        "description": None,
        "huggingface_url": "https://huggingface.co/test/tokenizer-report-roundtrip",
        "global_stats": {
            "vocabulary_size": 2,
            "base_vocabulary_size": 2,
            "model_max_length": 512,
            "padding_side": "right",
            "added_tokens_count": 0,
            "special_tokens_ids_count": 1,
            "subword_word_stats": {
                "heuristic": "hash_prefix_atat_suffix_sentencepiece_markers",
                "subword_count": 1,
                "word_count": 1,
                "considered_count": 2,
                "subword_percentage": 50.0,
                "word_percentage": 50.0,
                "subword_to_word_ratio": 1.0,
            },
            "vocabulary_stats": {
                "heuristic": "hash_prefix_atat_suffix_sentencepiece_markers",
                "min_token_length": 1,
                "mean_token_length": 1.0,
                "median_token_length": 1.0,
                "max_token_length": 1,
                "subword_like_count": 1,
                "subword_like_percentage": 50.0,
                "special_tokens_in_vocab_count": 0,
                "special_tokens_in_vocab_percentage": 0.0,
                "unique_token_lengths": 1,
                "empty_token_count": 0,
                "considered_non_special_count": 2,
            },
            "persistence_mode": "filesystem_required",
            "persistence_reason": "test",
        },
        "token_length_histogram": {
            "bins": ["1-1"],
            "counts": [2],
            "bin_edges": [1.0, 2.0],
            "min_length": 1,
            "max_length": 1,
            "mean_length": 1.0,
            "median_length": 1.0,
        },
        "vocabulary_size": 2,
    }
    report_id = repository.replace_report_and_vocabulary(
        report["tokenizer_name"], report, []
    )

    loaded = repository.load_tokenizer_report_by_id(report_id)

    assert loaded is not None
    assert (
        loaded["huggingface_url"]
        == "https://huggingface.co/test/tokenizer-report-roundtrip"
    )
    assert loaded["global_stats"]["subword_word_stats"]["subword_count"] == 1
    assert loaded["global_stats"]["base_vocabulary_size"] == 2
    assert loaded["global_stats"]["model_max_length"] == 512
    assert loaded["global_stats"]["vocabulary_stats"]["subword_like_count"] == 1


###############################################################################
def test_tokenizer_report_repository_does_not_create_missing_tokenizer() -> None:
    engine = create_engine("sqlite+pysqlite:///:memory:", future=True)
    Base.metadata.create_all(engine)
    repository = TokenizerReportRepository(
        DataRepositoryQueries(SimpleNamespace(backend=SimpleNamespace(engine=engine)))
    )

    with pytest.raises(ValueError, match="must exist before storing a report"):
        repository.replace_report_and_vocabulary(
            "missing/tokenizer",
            {"report_version": 1, "global_stats": {}, "token_length_histogram": {}},
            [],
        )


###############################################################################
def test_compute_vocabulary_shape_metrics_vectorized_summary() -> None:
    metrics = compute_vocabulary_shape_metrics(
        [
            {"token": "a"},
            {"token": "bb"},
            {"token": "ccc"},
            {"token": "dddd"},
        ]
    )

    assert metrics["token_length_std"] == pytest.approx(1.1180339887)
    assert metrics["token_length_p90"] == pytest.approx(3.7)
    assert metrics["token_length_cv"] == pytest.approx(0.4472135955)
    assert metrics["single_character_token_percentage"] == pytest.approx(25.0)


###############################################################################
def test_repository_persists_vocabulary_shape_metrics_in_report_json() -> None:
    engine = create_engine("sqlite+pysqlite:///:memory:", future=True)
    Base.metadata.create_all(engine)
    tokenizer_name = "test/vectorized-metrics"
    now = datetime.now(timezone.utc)
    vocabulary_rows = [
        {"token_id": 0, "token": "a", "decoded_token": "a"},
        {"token_id": 1, "token": "bb", "decoded_token": "bb"},
        {"token_id": 2, "token": "ccc", "decoded_token": "ccc"},
        {"token_id": 3, "token": "dddd", "decoded_token": "dddd"},
    ]
    vocabulary_shape_metrics = compute_vocabulary_shape_metrics(vocabulary_rows)
    with Session(engine) as session:
        session.add(
            Tokenizer(
                name=tokenizer_name,
                source="custom",
                created_at=now,
            )
        )
        session.commit()

    repository = TokenizerReportRepository(
        DataRepositoryQueries(SimpleNamespace(backend=SimpleNamespace(engine=engine)))
    )
    report_id = repository.replace_report_and_vocabulary(
        tokenizer_name,
        {
            "report_version": 1,
            "global_stats": {
                "vocabulary_size": 4,
                "token_length_measure": "character_count",
                "vocabulary_stats": {
                    "mean_token_length": 2.5,
                    **vocabulary_shape_metrics,
                },
            },
            "token_length_histogram": {
                "bins": ["1", "2", "3", "4"],
                "counts": [1, 1, 1, 1],
                "bin_edges": [1.0, 2.0, 3.0, 4.0, 5.0],
                "min_length": 1,
                "max_length": 4,
                "mean_length": 2.5,
                "median_length": 2.5,
                **vocabulary_shape_metrics,
            },
        },
        vocabulary_rows,
    )

    loaded = repository.load_tokenizer_report_by_id(report_id)

    assert loaded is not None
    vocabulary_stats = loaded["global_stats"]["vocabulary_stats"]
    histogram = loaded["token_length_histogram"]
    assert vocabulary_stats["token_length_std"] == pytest.approx(1.1180339887)
    assert vocabulary_stats["token_length_p90"] == pytest.approx(3.7)
    assert vocabulary_stats["token_length_cv"] == pytest.approx(0.4472135955)
    assert vocabulary_stats["single_character_token_percentage"] == pytest.approx(25.0)
    assert histogram["token_length_p90"] == pytest.approx(3.7)
    assert histogram["single_character_token_percentage"] == pytest.approx(25.0)


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
def test_tokenizer_discovery_passes_native_filters_and_returns_structured_items(
    monkeypatch,
) -> None:
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

    response = service.discover_tokenizers(
        TokenizerDiscoveryQuery(
            search=" bert ",
            limit=5,
            pipeline_tag="fill-mask",
            author="google",
            include_tags=["core-model"],
            access="public",
            sort="downloads",
        )
    )

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
def test_custom_tokenizer_survives_service_recreation_and_is_benchmarkable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from server.services import benchmarks as benchmark_module
    from server.services import tokenizer_storage

    monkeypatch.setattr(tokenizer_storage, "TOKENIZERS_PATH", tmp_path)
    monkeypatch.setattr(benchmark_module, "TOKENIZERS_PATH", tmp_path)

    engine = create_engine("sqlite+pysqlite:///:memory:", future=True)
    Base.metadata.create_all(engine)
    database = SimpleNamespace(backend=SimpleNamespace(engine=engine))


    ###############################################################################
    class FakeFastTokenizer:

        # -------------------------------------------------------------------------
        def get_vocab_size(self) -> int:
            return 3

    monkeypatch.setattr(
        "server.services.tokenizers.FastTokenizer.from_file",
        lambda _path: FakeFastTokenizer(),
    )
    try:
        service = TokenizersService()
        service.repository = TokenizerRepository(database)  # type: ignore[assignment]
        monkeypatch.setattr(
            service.benchmark_tools,
            "is_tokenizer_compatible",
            lambda _tokenizer: True,
        )

        result = service.register_custom_tokenizer_from_upload(
            b"canonical tokenizer artifact",
            "custom.json",
            "sample",
        )
        tokenizer_name = result["tokenizer_name"]
        assert result["is_compatible"] is True
        artifact = tmp_path / "CUSTOM_sample" / "tokenizer.json"
        assert artifact.read_bytes() == b"canonical tokenizer artifact"

        recreated = TokenizersService()
        recreated.repository = TokenizerRepository(database)  # type: ignore[assignment]
        assert recreated.repository.get_tokenizer_source(tokenizer_name) == "custom"
        assert recreated.has_available_tokenizer(tokenizer_name) is True
        assert recreated.list_tokenizer_catalog() == [
            {
                "tokenizer_name": tokenizer_name,
                "source": "custom",
                "has_report": False,
                "vocabulary_size": 3,
            }
        ]

        benchmark = BenchmarkService()
        benchmark.repository = BenchmarkRepository(database)  # type: ignore[assignment]
        monkeypatch.setattr(
            benchmark.tools,
            "is_tokenizer_compatible",
            lambda _tokenizer: True,
        )
        assert tokenizer_name in benchmark.load_tokenizers([tokenizer_name])
    finally:
        engine.dispose()


###############################################################################
def test_remove_tokenizer_cleans_cache_before_database_delete(
    tmp_path, monkeypatch
) -> None:
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

    assert service.remove_tokenizer("google-bert/bert-base-uncased") is True
    assert calls == ["google-bert/bert-base-uncased"]


###############################################################################
def test_remove_tokenizer_keeps_database_row_when_cache_cleanup_fails(
    monkeypatch,
) -> None:
    service = TokenizersService()
    monkeypatch.setattr(service, "get_tokenizer_cache_dir", lambda name: "cache")
    monkeypatch.setattr("server.services.tokenizers.Path.exists", lambda self: True)
    monkeypatch.setattr(
        "server.services.tokenizers.shutil.rmtree",
        lambda path: (_ for _ in ()).throw(OSError("locked")),
    )

    deleted = False


    ###############################################################################
    class FakeRepository:

        # -------------------------------------------------------------------------
        def delete_tokenizer(self, name: str) -> bool:
            nonlocal deleted
            deleted = True
            return True

    service.repository = FakeRepository()  # type: ignore[assignment]

    with pytest.raises(RuntimeError, match="Failed to remove tokenizer files"):
        service.remove_tokenizer("google-bert/bert-base-uncased")
    assert deleted is False


###############################################################################
def test_tokenizer_discovery_uses_bounded_overfetch_and_candidate_cap(
    monkeypatch,
) -> None:
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

    service.discover_tokenizers(
        TokenizerDiscoveryQuery(
            limit=5,
            exclude_tags=["audio"],
            vocabulary_sort="ascending",
        )
    )
    assert fake_api.captured["limit"] == 15
    assert fake_api.captured["expand"][-1] == "config"
    assert "fetch_config" not in fake_api.captured

    service.discover_tokenizers(
        TokenizerDiscoveryQuery(
            limit=250,
            vocabulary_sort="ascending",
        )
    )
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
                FakeModel(
                    "invalid/metadata-only", ("config.json", "tokenizer_config.json")
                ),
                FakeModel(
                    "invalid/nested-only", ("config.json", "nested/tokenizer.json")
                ),
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
def test_tokenizer_discovery_filters_excluded_and_unsupported_models(
    monkeypatch,
) -> None:
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

    response = service.discover_tokenizers(
        TokenizerDiscoveryQuery(
            limit=10,
            exclude_tags=["unsafe"],
        )
    )
    assert [item.identifier for item in response.items] == ["text/model"]


###############################################################################
def test_tokenizer_discovery_extracts_and_filters_vocabulary_metadata(
    monkeypatch,
) -> None:
    service = TokenizersService()


    ###############################################################################
    class FakeModel:

        # -------------------------------------------------------------------------
        def __init__(self, model_id: str, vocabulary_size: object) -> None:
            self.modelId = model_id
            self.pipeline_tag = "fill-mask"
            self.config = (
                {"vocab_size": vocabulary_size} if vocabulary_size is not None else None
            )
            self.siblings = tokenizer_siblings("vocab.txt", "config.json")


    ###############################################################################
    class FakeApi:

        # -------------------------------------------------------------------------
        def list_models(self, **kwargs):
            assert "config" in kwargs["expand"]
            return [
                FakeModel("small/model", 4),
                FakeModel("large/model", 12),
                FakeModel("unknown/model", None),
            ]

    monkeypatch.setattr("server.services.tokenizers.HfApi", lambda **kwargs: FakeApi())
    monkeypatch.setattr(service.key_service, "get_active_key", lambda: None)

    minimum = service.discover_tokenizers(
        TokenizerDiscoveryQuery(
            limit=10,
            vocabulary_operator="at_least",
            vocabulary_size=10,
        )
    )
    assert [(item.identifier, item.vocabulary_size) for item in minimum.items] == [
        ("large/model", 12)
    ]

    maximum = service.discover_tokenizers(
        TokenizerDiscoveryQuery(
            limit=10,
            vocabulary_operator="at_most",
            vocabulary_size=5,
        )
    )
    assert [(item.identifier, item.vocabulary_size) for item in maximum.items] == [
        ("small/model", 4)
    ]


###############################################################################
def test_tokenizer_discovery_orders_known_vocabulary_before_unknown(
    monkeypatch,
) -> None:
    service = TokenizersService()


    ###############################################################################
    class FakeModel:

        # -------------------------------------------------------------------------
        def __init__(self, model_id: str, vocabulary_size: int | None) -> None:
            self.modelId = model_id
            self.pipeline_tag = "fill-mask"
            self.config = (
                {"vocab_size": vocabulary_size} if vocabulary_size is not None else None
            )
            self.siblings = tokenizer_siblings("vocab.json", "merges.txt")


    ###############################################################################
    class FakeApi:

        # -------------------------------------------------------------------------
        def list_models(self, **kwargs):
            del kwargs
            return [
                FakeModel("unknown/model", None),
                FakeModel("large/model", 12),
                FakeModel("small/model", 4),
            ]

    monkeypatch.setattr("server.services.tokenizers.HfApi", lambda **kwargs: FakeApi())
    monkeypatch.setattr(service.key_service, "get_active_key", lambda: None)

    ascending = service.discover_tokenizers(
        TokenizerDiscoveryQuery(limit=10, vocabulary_sort="ascending")
    )
    descending = service.discover_tokenizers(
        TokenizerDiscoveryQuery(limit=10, vocabulary_sort="descending")
    )
    assert [item.identifier for item in ascending.items] == [
        "small/model",
        "large/model",
        "unknown/model",
    ]
    assert [item.identifier for item in descending.items] == [
        "large/model",
        "small/model",
        "unknown/model",
    ]


###############################################################################
def test_tokenizer_discovery_propagates_upstream_failure(monkeypatch) -> None:
    service = TokenizersService()


    ###############################################################################
    class FailingApi:

        # -------------------------------------------------------------------------
        def list_models(self, **kwargs):
            del kwargs
            raise RuntimeError("upstream outage details")

    monkeypatch.setattr(
        "server.services.tokenizers.HfApi", lambda **kwargs: FailingApi()
    )
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
    monkeypatch, tmp_path: Path
) -> None:
    service = TokenizersService()
    service.TOKENIZER_DOWNLOAD_TIMEOUT_SECONDS = 0.01
    cache_dir = tmp_path / "timeout-cache"
    loader_started = threading.Event()
    release_loader = threading.Event()
    loader_finished = threading.Event()

    monkeypatch.setattr(service.key_service, "get_active_key", lambda: None)
    monkeypatch.setattr(service.repository, "tokenizer_exists", lambda _: False)
    monkeypatch.setattr(service, "get_tokenizer_cache_dir", lambda _: str(cache_dir))
    monkeypatch.setattr(service, "has_cached_tokenizer", lambda _: False)

    def slow_load(*args, **kwargs):
        del args, kwargs
        loader_started.set()
        release_loader.wait(timeout=1.0)
        loader_finished.set()
        return object()

    monkeypatch.setattr(
        "server.services.tokenizers.AutoTokenizer.from_pretrained",
        slow_load,
    )

    try:
        result = service.download_and_persist(["slow/model"])
        assert loader_started.wait(timeout=1.0)
        assert not loader_finished.is_set()
    finally:
        release_loader.set()
        assert loader_finished.wait(timeout=1.0)

    assert result["failed"] == ["slow/model"]
    assert "TokenizerDownloadTimeoutError" in result["failed_details"][0]


###############################################################################
def test_tokenizer_catalog_filters_cached_sources_search_and_vocabulary(
    monkeypatch,
) -> None:
    service = TokenizersService()

    ###############################################################################
    monkeypatch.setattr(
        service.repository,
        "list_downloaded_tokenizer_catalog",
        lambda: [
            ("zeta/model", "huggingface", True, {"vocabulary_size": 100}),
            ("alpha/model", "huggingface", False, {"vocabulary_size": 3}),
            ("uncached/model", "huggingface", False, {"vocabulary_size": 1}),
            ("CUSTOM_demo", "custom", False, {"vocabulary_size": 4}),
        ],
    )
    monkeypatch.setattr(
        service,
        "has_cached_tokenizer",
        lambda name: name != "uncached/model",
    )
    monkeypatch.setattr(
        service, "has_custom_tokenizer_artifact", lambda name: name == "CUSTOM_demo"
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
    ) == [
        {
            "tokenizer_name": "CUSTOM_demo",
            "source": "custom",
            "has_report": False,
            "vocabulary_size": 4,
        }
    ]
    assert [
        item["tokenizer_name"]
        for item in service.list_tokenizer_catalog(
            vocabulary_size_operator="at_least",
            vocabulary_size=50,
        )
    ] == ["zeta/model"]


###############################################################################
class DummyTokenizer__tokenizer_adapter_contract:
    unk_token_id = 0

    # -------------------------------------------------------------------------
    def __call__(self, texts, **kwargs):  # type: ignore[no-untyped-def]
        del kwargs
        return {"input_ids": [[1, 2], [0]]}


###############################################################################
class EncodeOnlyTokenizer:
    unk_token_id = 0

    # -------------------------------------------------------------------------
    def encode(self, text: str, *, add_special_tokens: bool = True) -> list[int]:
        del add_special_tokens
        return [1, 0] if text == "unknown" else [1, 2, 3]


###############################################################################
class PaddingTokenizer:
    eos_token = "<eos>"
    pad_token_id = None
    unk_token_id = None

    # -------------------------------------------------------------------------
    def __call__(self, texts, **kwargs):  # type: ignore[no-untyped-def]
        assert kwargs["padding"] is True
        assert self.pad_token == self.eos_token
        return {"input_ids": [[1] for _ in texts]}


###############################################################################
def test_raw_tokenizer_honors_special_padding_and_truncation_options() -> None:
    tokenizer = Tokenizer__tokenizer_adapter_contract(
        models.WordLevel(
            {
                "[PAD]": 0,
                "[UNK]": 1,
                "[CLS]": 2,
                "[SEP]": 3,
                "alpha": 4,
                "beta": 5,
                "gamma": 6,
                "delta": 7,
            },
            unk_token="[UNK]",
        )
    )
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer.post_processor = processors.TemplateProcessing(
        single="[CLS] $A [SEP]",
        special_tokens=[("[CLS]", 2), ("[SEP]", 3)],
    )
    adapter = UniversalTokenizerAdapter("raw-configurable", tokenizer)
    texts = ["alpha beta", "alpha beta gamma delta"]

    without_options = adapter.encode_batch(
        texts,
        add_special_tokens=False,
        padding=False,
        truncation=False,
        max_length=None,
    )
    with_special_tokens = adapter.encode_batch(
        texts,
        add_special_tokens=True,
        padding=False,
        truncation=False,
        max_length=None,
    )
    with_padding = adapter.encode_batch(
        texts,
        add_special_tokens=False,
        padding=True,
        truncation=False,
        max_length=None,
    )
    with_truncation = adapter.encode_batch(
        texts,
        add_special_tokens=False,
        padding=False,
        truncation=True,
        max_length=3,
    )

    assert without_options.token_counts == [2, 4]
    assert with_special_tokens.token_counts == [4, 6]
    assert with_padding.token_counts == [4, 4]
    assert with_truncation.token_counts == [2, 3]
    assert tokenizer.padding is None
    assert tokenizer.truncation is None

    tokenizer.enable_padding(pad_id=0, pad_token="[PAD]")
    tokenizer.enable_truncation(max_length=3)
    original_padding = tokenizer.padding
    original_truncation = tokenizer.truncation
    without_options_again = adapter.encode_batch(
        texts,
        add_special_tokens=False,
        padding=False,
        truncation=False,
        max_length=None,
    )
    assert without_options_again.token_counts == [2, 4]
    assert tokenizer.padding == original_padding
    assert tokenizer.truncation == original_truncation


###############################################################################
def test_adapter_returns_normalized_structure() -> None:
    adapter = UniversalTokenizerAdapter(
        "x", DummyTokenizer__tokenizer_adapter_contract()
    )
    encoded = adapter.encode_batch(
        ["a", "b"],
        add_special_tokens=False,
        padding=False,
        truncation=False,
        max_length=None,
    )
    assert encoded.token_counts == [2, 1]
    assert encoded.unknown_counts == [0, 1]


###############################################################################
def test_adapter_rejects_encode_only_tokenizers() -> None:
    adapter = UniversalTokenizerAdapter("encode-only", EncodeOnlyTokenizer())

    with pytest.raises(ValueError, match="callable or support encode_batch"):
        adapter.encode_batch(
            ["known", "unknown"],
            add_special_tokens=False,
            padding=False,
            truncation=False,
            max_length=None,
        )


###############################################################################
def test_adapter_uses_eos_token_for_padding_when_pad_token_is_missing() -> None:
    tokenizer = PaddingTokenizer()
    adapter = UniversalTokenizerAdapter("padding", tokenizer)

    encoded = adapter.encode_batch(
        ["a"],
        add_special_tokens=False,
        padding=True,
        truncation=False,
        max_length=None,
    )

    assert tokenizer.pad_token == tokenizer.eos_token
    assert encoded.token_counts == [1]


###############################################################################
class CapturingTokenizer:

    # -------------------------------------------------------------------------
    def __init__(self) -> None:
        self.last_kwargs = {}

    # -------------------------------------------------------------------------
    def __call__(self, texts, **kwargs):  # type: ignore[no-untyped-def]
        del texts
        self.last_kwargs = dict(kwargs)
        return {"input_ids": [[1], [2]]}


###############################################################################
def test_adapter_uses_explicit_tokenizer_settings() -> None:
    tokenizer = CapturingTokenizer()
    adapter = UniversalTokenizerAdapter("tok", tokenizer)
    adapter.encode_batch(
        ["hello", "world"],
        add_special_tokens=False,
        padding=False,
        truncation=False,
        max_length=128,
    )
    assert tokenizer.last_kwargs["add_special_tokens"] is False
    assert tokenizer.last_kwargs["padding"] is False
    assert tokenizer.last_kwargs["truncation"] is False
    assert tokenizer.last_kwargs["max_length"] == 128


###############################################################################
def test_tokenizer_preview_returns_tokens_per_tokenizer_and_isolates_failures(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from server.contracts.tokenizers import TokenizerPreviewRequest
    from server.services.tokenizer_preview import TokenizerPreviewService

    raw = Tokenizer__tokenizer_adapter_contract(
        models.WordLevel({"[UNK]": 0, "alpha": 1, "beta": 2}, unk_token="[UNK]")
    )
    raw.pre_tokenizer = pre_tokenizers.Whitespace()


    ###############################################################################
    class FakeHFTokenizer:
        unk_token_id = 0
        vocab = {"[UNK]": 0, "alpha": 1, "beta": 2}

        # -------------------------------------------------------------------------
        def __call__(self, texts: list[str], **_kwargs: Any) -> dict[str, Any]:
            return {
                "input_ids": [
                    [self.vocab.get(word, 0) for word in text.split()] for text in texts
                ]
            }

        # -------------------------------------------------------------------------
        def convert_ids_to_tokens(self, ids: list[int]) -> list[str]:
            reverse = {value: key for key, value in self.vocab.items()}
            return [reverse[token_id] for token_id in ids]


    ###############################################################################
    class BrokenTokenizer:

        # -------------------------------------------------------------------------
        def __call__(self, *_args: Any, **_kwargs: Any) -> dict[str, Any]:
            raise RuntimeError("encode failed")

    service = TokenizerPreviewService()
    monkeypatch.setattr(
        service.benchmark_service,
        "get_missing_persisted_tokenizers",
        lambda names: [name for name in names if name == "missing"],
    )
    monkeypatch.setattr(
        service.benchmark_service,
        "load_tokenizers",
        lambda names: {
            name: tokenizer
            for name, tokenizer in {
                "raw": raw,
                "hf": FakeHFTokenizer(),
                "broken": BrokenTokenizer(),
            }.items()
            if name in names
        },
    )

    first = service.preview(
        TokenizerPreviewRequest(
            text="alpha gamma beta", tokenizers=["raw", "hf", "raw"]
        )
    )
    second = service.preview(
        TokenizerPreviewRequest(text="alpha", tokenizers=["missing", "broken"])
    )

    assert [result.tokenizer_name for result in first] == ["raw", "hf"]
    assert [(token.token_id, token.token) for token in first[0].tokens] == [
        (1, "alpha"),
        (0, "[UNK]"),
        (2, "beta"),
    ]
    assert (first[0].token_count, first[0].unknown_count) == (3, None)
    assert [token.token for token in first[1].tokens] == ["alpha", "[UNK]", "beta"]
    assert (first[1].token_count, first[1].unknown_count) == (3, 1)
    assert [(result.status, result.tokens) for result in second] == [
        ("error", []),
        ("error", []),
    ]
    assert second[0].error == "Tokenizer is not persisted locally."
