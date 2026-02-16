from __future__ import annotations

import hashlib
from collections import defaultdict


###############################################################################
def compute_simhash(features: list[str], bits: int = 64) -> int:
    if not features:
        return 0
    vector = [0] * bits
    for feature in features:
        digest = hashlib.blake2b(feature.encode("utf-8", errors="ignore"), digest_size=8).digest()
        value = int.from_bytes(digest, "big", signed=False)
        for bit_index in range(bits):
            mask = 1 << bit_index
            vector[bit_index] += 1 if value & mask else -1
    fingerprint = 0
    for bit_index, score in enumerate(vector):
        if score > 0:
            fingerprint |= 1 << bit_index
    return fingerprint


###############################################################################
def hamming_distance(a: int, b: int) -> int:
    return int((a ^ b).bit_count())


###############################################################################
def jaccard_like_similarity_from_hamming(a: int, b: int, bits: int = 64) -> float:
    distance = hamming_distance(a, b)
    return max(0.0, 1.0 - (distance / float(bits)))


###############################################################################
class SimHashNearDuplicateAnalyzer:
    def __init__(self, similarity_threshold: float = 0.9, bands: int = 4, bits: int = 64) -> None:
        self.similarity_threshold = float(similarity_threshold)
        self.bands = max(1, int(bands))
        self.bits = int(bits)
        self.band_size = max(1, self.bits // self.bands)
        self.buckets: dict[str, list[int]] = defaultdict(list)
        self.hashes: list[int] = []

    # -------------------------------------------------------------------------
    def _bucket_key(self, fingerprint: int, band_index: int) -> str:
        shift = band_index * self.band_size
        mask = (1 << self.band_size) - 1
        segment = (fingerprint >> shift) & mask
        return f"{band_index}:{segment}"

    # -------------------------------------------------------------------------
    def _features_from_tokens(self, tokens: list[str]) -> list[str]:
        lowered = [token.lower() for token in tokens if token]
        if len(lowered) < 3:
            return lowered
        return [" ".join(lowered[index : index + 3]) for index in range(len(lowered) - 2)]

    # -------------------------------------------------------------------------
    def check_and_add(self, tokens: list[str]) -> bool:
        features = self._features_from_tokens(tokens)
        fingerprint = compute_simhash(features, bits=self.bits)
        candidates: set[int] = set()
        for band_index in range(self.bands):
            bucket = self._bucket_key(fingerprint, band_index)
            for candidate_index in self.buckets.get(bucket, []):
                candidates.add(candidate_index)

        is_near_duplicate = False
        for candidate_index in candidates:
            candidate_hash = self.hashes[candidate_index]
            similarity = jaccard_like_similarity_from_hamming(
                fingerprint, candidate_hash, bits=self.bits
            )
            if similarity >= self.similarity_threshold:
                is_near_duplicate = True
                break

        current_index = len(self.hashes)
        self.hashes.append(fingerprint)
        for band_index in range(self.bands):
            bucket = self._bucket_key(fingerprint, band_index)
            self.buckets[bucket].append(current_index)

        return is_near_duplicate

