from __future__ import annotations

from TKBEN.server.services.dataset_jobs import DatasetJobService


class DummyJobManager:
    def should_stop(self, job_id: str) -> bool:
        del job_id
        return False

    def update_progress(self, job_id: str, value: float) -> None:
        del job_id, value


def test_build_analysis_payload_preserves_contract() -> None:
    service = DatasetJobService()
    payload = service.build_analysis_payload(
        {
            "dataset_name": "custom/demo",
            "report_id": 1,
            "document_length_histogram": {"bins": ["1-2"], "counts": [1]},
            "word_length_histogram": {"bins": ["1-2"], "counts": [1]},
        }
    )
    assert payload["status"] == "success"
    assert payload["dataset_name"] == "custom/demo"
    assert payload["report_id"] == 1


def test_extract_configuration_handles_missing() -> None:
    service = DatasetJobService()
    assert service.extract_configuration({}) is None
    assert service.extract_configuration({"configs": {"configuration": " abc "}}) == "abc"


def test_run_download_job_returns_service_payload(monkeypatch) -> None:
    service = DatasetJobService()

    class FakeDatasetService:
        def download_and_persist(self, **kwargs):
            assert kwargs["corpus"] == "wikitext"
            return {
                "dataset_name": "wikitext",
                "text_column": "text",
                "document_count": 2,
                "saved_count": 2,
                "histogram": {"bins": ["1-2"], "counts": [2]},
            }

    monkeypatch.setattr("TKBEN.server.services.dataset_jobs.DatasetService", FakeDatasetService)

    result = service.run_download_job(
        request_payload={"corpus": "wikitext", "configs": {}},
        job_manager=DummyJobManager(),
        job_id="job1",
    )

    assert result["status"] == "success"
    assert result["dataset_name"] == "wikitext"
