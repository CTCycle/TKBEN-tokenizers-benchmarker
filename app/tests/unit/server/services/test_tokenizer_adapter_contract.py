from __future__ import annotations

from server.services.tokenizer_adapters import UniversalTokenizerAdapter


class DummyTokenizer:
    unk_token_id = 0

    def __call__(self, texts, **kwargs):  # type: ignore[no-untyped-def]
        del kwargs
        return {"input_ids": [[1, 2], [0]]}


def test_adapter_returns_normalized_structure() -> None:
    adapter = UniversalTokenizerAdapter("x", DummyTokenizer())
    encoded = adapter.encode_batch(["a", "b"], add_special_tokens=False, padding=False, truncation=False, max_length=None)
    assert encoded.token_counts == [2, 1]
    assert encoded.unknown_counts == [0, 1]
