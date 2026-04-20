from __future__ import annotations

from fastapi.testclient import TestClient

from TKBEN.server.app import app


class DummyJobManager:
    def __init__(self) -> None:
        self.last_job_type = ""

    def is_job_running(self, job_type: str | None = None) -> bool:
        return False

    def start_job(self, job_type, runner, args=(), kwargs=None):
        del runner, args, kwargs
        self.last_job_type = str(job_type)
        return "job-123"

    def get_job_status(self, job_id: str):
        del job_id
        return {"job_type": self.last_job_type, "status": "pending"}


def test_dataset_job_start_routes_return_202(monkeypatch) -> None:
    manager = DummyJobManager()
    monkeypatch.setattr(app.state, "job_manager", manager)

    from TKBEN.server.services.datasets import DatasetService

    monkeypatch.setattr(
        DatasetService,
        "is_dataset_in_database",
        lambda self, dataset_name: bool(dataset_name),
    )

    client = TestClient(app)

    download_resp = client.post(
        "/api/datasets/download",
        json={"corpus": "wikitext", "configs": {"configuration": "wikitext-2-v1"}},
    )
    assert download_resp.status_code == 202
    assert download_resp.json()["job_id"] == "job-123"

    upload_resp = client.post(
        "/api/datasets/upload",
        files={"file": ("sample.csv", b"text\nhello\n", "text/csv")},
    )
    assert upload_resp.status_code == 202
    assert upload_resp.json()["job_id"] == "job-123"

    analyze_resp = client.post(
        "/api/datasets/analyze",
        json={"dataset_name": "custom/sample"},
    )
    assert analyze_resp.status_code == 202
    assert analyze_resp.json()["job_id"] == "job-123"
