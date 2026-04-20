from __future__ import annotations

from TKBEN.server.services.tokenizers import TokenizersService


class FakeTokenizerRepository:
    def __init__(self) -> None:
        self.inserted: list[str] = []

    def tokenizer_exists(self, tokenizer_id: str) -> bool:
        return tokenizer_id == "exists"

    def insert_if_missing(self, tokenizer_id: str) -> None:
        self.inserted.append(tokenizer_id)

    def get_missing_tokenizers(self, tokenizer_ids: list[str]) -> list[str]:
        return [name for name in tokenizer_ids if name != "exists"]

    def get_latest_tokenizer_report(self, tokenizer_name: str):
        return object() if tokenizer_name == "exists" else None

    def get_tokenizer_report_by_id(self, report_id: int):
        return object() if report_id == 1 else None


def test_tokenizers_service_uses_repository_layer(monkeypatch) -> None:
    service = TokenizersService()
    fake_repo = FakeTokenizerRepository()
    service.repository = fake_repo  # type: ignore[assignment]

    assert service.is_tokenizer_persisted("exists") is True
    service.insert_tokenizer_if_missing("new")
    assert fake_repo.inserted == ["new"]

    missing = service.resolve_missing_tokenizer_names(["exists", "missing"])
    assert missing == ["missing"]


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
        lambda report_id, offset, limit: {"report_id": report_id, "offset": offset, "limit": limit, "items": []},
    )

    assert service.get_latest_tokenizer_report("exists") is not None
    assert service.get_latest_tokenizer_report("missing") is None

    assert service.get_tokenizer_report_by_id(1) is not None
    assert service.get_tokenizer_report_by_id(2) is None

    assert service.get_tokenizer_report_vocabulary(1, 0, 10) is not None
    assert service.get_tokenizer_report_vocabulary(2, 0, 10) is None
