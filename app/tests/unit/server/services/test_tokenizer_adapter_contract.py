from __future__ import annotations

from server.services.tokenizer_adapters import UniversalTokenizerAdapter


###############################################################################
class DummyTokenizer:
    unk_token_id = 0

    # -------------------------------------------------------------------------
    def __call__(self, texts, **kwargs):  # type: ignore[no-untyped-def]
        del kwargs
        return {"input_ids": [[1, 2], [0]]}


###############################################################################
class RawEncoding:

    # -------------------------------------------------------------------------
    def __init__(self, ids: list[int]) -> None:
        self.ids = ids


###############################################################################
class RawTokenizer:
    unk_token_id = 0

    # -------------------------------------------------------------------------
    def encode(self, text: str) -> RawEncoding:
        return RawEncoding([1, 0] if text == "unknown" else [1, 2, 3])


###############################################################################
def test_adapter_returns_normalized_structure() -> None:
    adapter = UniversalTokenizerAdapter("x", DummyTokenizer())
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
def test_adapter_accepts_raw_tokenizers_encoding_objects() -> None:
    adapter = UniversalTokenizerAdapter("raw", RawTokenizer())
    encoded = adapter.encode_batch(
        ["known", "unknown"],
        add_special_tokens=False,
        padding=False,
        truncation=False,
        max_length=None,
    )
    assert encoded.input_ids_by_doc == [[1, 2, 3], [1, 0]]
    assert encoded.token_counts == [3, 2]
    assert encoded.unknown_counts == [0, 1]
