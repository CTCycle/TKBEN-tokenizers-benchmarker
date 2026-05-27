from __future__ import annotations

from server.services.benchmark_spool import BenchmarkTextSpool


def test_benchmark_text_spool_replays_rows_and_batches() -> None:
    spool = BenchmarkTextSpool()
    spool.append(1, "alpha")
    spool.append(2, "beta")
    spool.append(3, "gamma")
    spool.finalize()

    assert list(spool.iter_rows()) == [(1, "alpha"), (2, "beta"), (3, "gamma")]
    assert list(spool.iter_text_batches(2)) == [["alpha", "beta"], ["gamma"]]
    spool.cleanup()


def test_benchmark_text_spool_cleanup_removes_file() -> None:
    spool = BenchmarkTextSpool()
    path = spool.path
    spool.append(1, "x")
    spool.finalize()
    assert path.exists()
    spool.cleanup()
    assert not path.exists()
