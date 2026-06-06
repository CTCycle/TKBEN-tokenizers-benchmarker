from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, Sequence


@dataclass(frozen=True)
class EncodedBatch:
    token_counts: list[int]
    unknown_counts: list[int | None]
    input_ids_by_doc: list[list[int]]


class TokenizerAdapter(Protocol):
    tokenizer_id: str

    def encode_batch(
        self,
        texts: Sequence[str],
        *,
        add_special_tokens: bool,
        padding: bool,
        truncation: bool,
        max_length: int | None,
    ) -> EncodedBatch: ...


class UniversalTokenizerAdapter:
    def __init__(self, tokenizer_id: str, tokenizer: Any) -> None:
        self.tokenizer_id = tokenizer_id
        self._tokenizer = tokenizer

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

        if callable(getattr(self._tokenizer, "__call__", None)):
            encoded = self._tokenizer(
                as_list,
                add_special_tokens=add_special_tokens,
                padding=padding,
                truncation=truncation,
                max_length=max_length,
            )
            input_ids = encoded["input_ids"] if isinstance(encoded, dict) else []
            normalized_ids = [[int(value) for value in ids] for ids in input_ids]
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

        token_counts: list[int] = []
        unknown_counts: list[int | None] = []
        input_ids_by_doc: list[list[int]] = []
        for text in as_list:
            ids = self._tokenizer.encode(text)
            ids_list = [int(value) for value in ids]
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
