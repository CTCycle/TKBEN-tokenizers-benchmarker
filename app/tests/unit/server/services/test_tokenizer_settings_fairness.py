from __future__ import annotations

from server.services.tokenizer_adapters import UniversalTokenizerAdapter


class CapturingTokenizer:
    def __init__(self) -> None:
        self.last_kwargs = {}

    def __call__(self, texts, **kwargs):  # type: ignore[no-untyped-def]
        del texts
        self.last_kwargs = dict(kwargs)
        return {"input_ids": [[1], [2]]}


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
