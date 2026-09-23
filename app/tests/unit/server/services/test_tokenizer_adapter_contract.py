from __future__ import annotations

from tokenizers import Tokenizer, models, pre_tokenizers, processors

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
    def encode(self, text: str, *, add_special_tokens: bool = True) -> RawEncoding:
        del add_special_tokens
        return RawEncoding([1, 0] if text == "unknown" else [1, 2, 3])

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
    tokenizer = Tokenizer(
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
