from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Mapping, Sequence
from typing import Any, Protocol, cast

###############################################################################
@dataclass(frozen=True)
class EncodedBatch:
    token_counts: list[int]
    unknown_counts: list[int | None]
    input_ids_by_doc: list[list[int]]

###############################################################################
class TokenizerAdapter(Protocol):
    tokenizer_id: str

    # -------------------------------------------------------------------------
    def encode_batch(
        self,
        texts: Sequence[str],
        *,
        add_special_tokens: bool,
        padding: bool,
        truncation: bool,
        max_length: int | None,
    ) -> EncodedBatch: ...

###############################################################################
class UniversalTokenizerAdapter:

    # -------------------------------------------------------------------------
    def __init__(self, tokenizer_id: str, tokenizer: Any) -> None:
        self.tokenizer_id = tokenizer_id
        self._tokenizer = tokenizer

    # -------------------------------------------------------------------------
    def _normalize_ids(self, encoded: Any) -> list[int]:
        ids = encoded.ids if hasattr(encoded, "ids") else encoded
        return [int(value) for value in ids]

    # -------------------------------------------------------------------------
    def _ensure_padding_token(self, padding: bool) -> None:
        if not padding or getattr(self._tokenizer, "pad_token_id", None) is not None:
            return
        eos_token = getattr(self._tokenizer, "eos_token", None)
        if eos_token is not None:
            self._tokenizer.pad_token = eos_token

    # -------------------------------------------------------------------------
    def encode_batch(
        self,
        texts: Sequence[str],
        *,
        add_special_tokens: bool,
        padding: bool,
        truncation: bool,
        max_length: int | None,
    ) -> EncodedBatch:
        as_list = list(texts)
        unk_id = getattr(self._tokenizer, "unk_token_id", None)
        self._ensure_padding_token(padding)

        if callable(getattr(self._tokenizer, "__call__", None)):
            encoded = self._tokenizer(
                as_list,
                add_special_tokens=add_special_tokens,
                padding=padding,
                truncation=truncation,
                max_length=max_length,
            )
            input_ids = encoded["input_ids"] if isinstance(encoded, Mapping) else []
            if not isinstance(input_ids, Sequence):
                input_ids = []
            normalized_ids = [
                [int(cast(int | str | float, value)) for value in ids]
                for ids in input_ids
                if isinstance(ids, Sequence)
            ]
            token_counts = [len(ids) for ids in normalized_ids]
            unknown_counts = [
                (None if unk_id is None else sum(1 for value in ids if value == unk_id))
                for ids in normalized_ids
            ]
            return EncodedBatch(
                token_counts=token_counts,
                unknown_counts=unknown_counts,
                input_ids_by_doc=normalized_ids,
            )

        encode_batch = getattr(self._tokenizer, "encode_batch", None)
        if callable(encode_batch):
            original_padding = getattr(self._tokenizer, "padding", None)
            original_truncation = getattr(self._tokenizer, "truncation", None)
            try:
                if padding:
                    enable_padding = getattr(self._tokenizer, "enable_padding", None)
                    if not callable(enable_padding):
                        raise ValueError("Tokenizer does not support padding.")
                    padding_options = self._raw_padding_options(original_padding)
                    enable_padding(**padding_options)
                elif original_padding is not None:
                    no_padding = getattr(self._tokenizer, "no_padding", None)
                    if not callable(no_padding):
                        raise ValueError("Tokenizer cannot disable configured padding.")
                    no_padding()

                if truncation:
                    enable_truncation = getattr(
                        self._tokenizer, "enable_truncation", None
                    )
                    if not callable(enable_truncation):
                        raise ValueError("Tokenizer does not support truncation.")
                    truncation_options = (
                        dict(original_truncation)
                        if isinstance(original_truncation, Mapping)
                        else {}
                    )
                    effective_max_length = max_length or truncation_options.get(
                        "max_length"
                    )
                    if not isinstance(effective_max_length, int) or isinstance(
                        effective_max_length, bool
                    ):
                        raise ValueError(
                            "A maximum length is required for tokenizer truncation."
                        )
                    truncation_options["max_length"] = effective_max_length
                    enable_truncation(**truncation_options)
                elif original_truncation is not None:
                    no_truncation = getattr(self._tokenizer, "no_truncation", None)
                    if not callable(no_truncation):
                        raise ValueError(
                            "Tokenizer cannot disable configured truncation."
                        )
                    no_truncation()

                encodings = encode_batch(
                    as_list, add_special_tokens=add_special_tokens
                )
                if not isinstance(encodings, Sequence):
                    raise ValueError("Tokenizer returned an invalid encoding batch.")
                normalized_ids = [self._normalize_ids(item) for item in encodings]
            finally:
                self._restore_raw_configuration(
                    "padding", original_padding, "enable_padding", "no_padding"
                )
                self._restore_raw_configuration(
                    "truncation",
                    original_truncation,
                    "enable_truncation",
                    "no_truncation",
                )
            token_counts = [len(ids) for ids in normalized_ids]
            unknown_counts = [
                (None if unk_id is None else sum(1 for value in ids if value == unk_id))
                for ids in normalized_ids
            ]
            return EncodedBatch(
                token_counts=token_counts,
                unknown_counts=unknown_counts,
                input_ids_by_doc=normalized_ids,
            )

        if padding or truncation:
            raise ValueError(
                "Tokenizer does not support requested padding or truncation."
            )

        token_counts: list[int] = []
        unknown_counts: list[int | None] = []
        input_ids_by_doc: list[list[int]] = []
        for text in as_list:
            try:
                ids = self._tokenizer.encode(
                    text, add_special_tokens=add_special_tokens
                )
            except TypeError:
                ids = self._tokenizer.encode(text)
            ids_list = self._normalize_ids(ids)
            input_ids_by_doc.append(ids_list)
            token_counts.append(len(ids_list))
            if unk_id is None:
                unknown_counts.append(None)
            else:
                unknown_counts.append(sum(1 for value in ids_list if value == unk_id))

        return EncodedBatch(
            token_counts=token_counts,
            unknown_counts=unknown_counts,
            input_ids_by_doc=input_ids_by_doc,
        )

    # -------------------------------------------------------------------------
    def _raw_padding_options(self, original_padding: Any) -> dict[str, Any]:
        if isinstance(original_padding, Mapping):
            return dict(original_padding)

        pad_token = getattr(self._tokenizer, "pad_token", None)
        pad_token_id = getattr(self._tokenizer, "pad_token_id", None)
        vocab = getattr(self._tokenizer, "get_vocab", None)
        vocab_items = vocab() if callable(vocab) else {}
        if not isinstance(pad_token, str) or not pad_token:
            if isinstance(vocab_items, Mapping) and "[PAD]" in vocab_items:
                pad_token = "[PAD]"
            else:
                eos_token = getattr(self._tokenizer, "eos_token", None)
                pad_token = eos_token if isinstance(eos_token, str) else None
        token_to_id = getattr(self._tokenizer, "token_to_id", None)
        if (
            not isinstance(pad_token_id, int)
            and isinstance(pad_token, str)
            and callable(token_to_id)
        ):
            pad_token_id = token_to_id(pad_token)
        if not isinstance(pad_token_id, int) or not isinstance(pad_token, str):
            raise ValueError("Tokenizer does not define a usable padding token.")
        return {
            "pad_id": pad_token_id,
            "pad_token": pad_token,
            "direction": "right",
        }

    # -------------------------------------------------------------------------
    def _restore_raw_configuration(
        self,
        attribute: str,
        original_value: Any,
        enable_method: str,
        disable_method: str,
    ) -> None:
        if original_value is None:
            disable = getattr(self._tokenizer, disable_method, None)
            if callable(disable):
                disable()
            return
        enable = getattr(self._tokenizer, enable_method, None)
        if not callable(enable) or not isinstance(original_value, Mapping):
            raise ValueError(f"Tokenizer cannot restore its configured {attribute}.")
        enable(**dict(original_value))
