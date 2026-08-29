from __future__ import annotations

from pathlib import Path

import pytest

from server.contracts.benchmarks import BenchmarkRunRequest
from server.services.benchmarks import BenchmarkService

###############################################################################
class FakeBenchmarkRepository:

    # -------------------------------------------------------------------------
    def get_dataset_document_count(self, dataset_name: str) -> int:
        return 7 if dataset_name == "custom/sample" else 0

    # -------------------------------------------------------------------------
    def get_missing_persisted_tokenizers(self, tokenizer_ids: list[str]) -> list[str]:
        return [name for name in tokenizer_ids if name != "bert-base-uncased"]

    # -------------------------------------------------------------------------
    def get_tokenizer_sources(self, tokenizer_ids: list[str]) -> dict[str, str]:
        return {
            name: "huggingface"
            for name in tokenizer_ids
            if name == "bert-base-uncased"
        }

###############################################################################
def test_benchmark_service_uses_repository_for_dataset_and_tokenizer_checks() -> None:
    service = BenchmarkService()
    service.repository = FakeBenchmarkRepository()  # type: ignore[assignment]

    assert service.get_dataset_document_count("custom/sample") == 7
    assert service.get_dataset_document_count("missing") == 0

    missing = service.get_missing_persisted_tokenizers(["bert-base-uncased", "missing"])
    assert "missing" in missing

###############################################################################
def test_benchmark_service_preserves_repository_missing_with_cached_files(
    monkeypatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr("server.services.benchmarks.TOKENIZERS_PATH", tmp_path)
    cached_dir = tmp_path / "missing"
    cached_dir.mkdir()
    (cached_dir / "tokenizer.json").write_text("{}", encoding="utf-8")

    service = BenchmarkService()
    service.repository = FakeBenchmarkRepository()  # type: ignore[assignment]

    missing = service.get_missing_persisted_tokenizers(["missing"])

    assert missing == ["missing"]

###############################################################################
def test_prepare_run_owns_admission_checks_and_normalizes_job_payload(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service = BenchmarkService()
    monkeypatch.setattr(service, "get_dataset_document_count", lambda dataset_name: 3)
    monkeypatch.setattr(service, "get_missing_persisted_tokenizers", lambda tokenizers: [])

    payload = BenchmarkRunRequest(
        tokenizers=["CUSTOM_demo"],
        dataset_name="custom/sample",
        config={"max_documents": 2},
    )
    prepared = service.prepare_run(payload)

    assert prepared["dataset_name"] == "custom/sample"
    assert prepared["config"]["max_documents"] == 2
