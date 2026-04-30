from TKBEN.server.services.metrics.frequencies import DiskBackedFrequencyStore


def test_frequency_store_flush_batches_large_token_lookups() -> None:
    store = DiskBackedFrequencyStore(memory_limit=50_000)
    try:
        total_tokens = 2_500
        for index in range(total_tokens):
            store.add(f"token-{index}")

        store.flush()

        assert store.unique_count() == total_tokens
        assert store.total_count() == total_tokens

        for index in range(total_tokens):
            store.add(f"token-{index}")

        store.flush()

        token_counts = dict(store.top_k(5_000))
        assert token_counts["token-0"] == 2
        assert token_counts["token-2499"] == 2
    finally:
        store.close()
