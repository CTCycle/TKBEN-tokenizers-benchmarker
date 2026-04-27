from __future__ import annotations

from pathlib import Path

from TKBEN.server.services.benchmarks import BenchmarkService


class FakeBenchmarkRepository:
    def get_dataset_document_count(self, dataset_name: str) -> int:
        return 7 if dataset_name == "custom/sample" else 0

    def get_missing_persisted_tokenizers(self, tokenizer_ids: list[str]) -> list[str]:
        return [name for name in tokenizer_ids if name != "bert-base-uncased"]


def test_benchmark_service_uses_repository_for_dataset_and_tokenizer_checks() -> None:
    service = BenchmarkService()
    service.repository = FakeBenchmarkRepository()  # type: ignore[assignment]

    assert service.get_dataset_document_count("custom/sample") == 7
    assert service.get_dataset_document_count("missing") == 0

    missing = service.get_missing_persisted_tokenizers(
        ["bert-base-uncased", "missing"]
    )
    assert "missing" in missing


def test_benchmark_service_preserves_repository_missing_with_cached_files(
    monkeypatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr("TKBEN.server.services.benchmarks.TOKENIZERS_PATH", str(tmp_path))
    cached_dir = tmp_path / "missing"
    cached_dir.mkdir()
    (cached_dir / "tokenizer.json").write_text("{}", encoding="utf-8")

    service = BenchmarkService()
    service.repository = FakeBenchmarkRepository()  # type: ignore[assignment]

    missing = service.get_missing_persisted_tokenizers(["missing"])

    assert missing == ["missing"]


def test_resolve_custom_tokenizer_selection(monkeypatch) -> None:
    service = BenchmarkService()

    class DummyRegistry:
        def get(self, name: str):
            return object() if name == "CUSTOM_demo" else None

    monkeypatch.setattr(
        "TKBEN.server.services.benchmarks.get_custom_tokenizer_registry",
        lambda: DummyRegistry(),
    )
    monkeypatch.setattr(service.tools, "is_tokenizer_compatible", lambda tokenizer: True)

    resolved = service.resolve_custom_tokenizer_selection("CUSTOM_demo")
    assert "CUSTOM_demo" in resolved
