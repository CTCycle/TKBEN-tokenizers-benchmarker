from __future__ import annotations

from fastapi.testclient import TestClient

from server.app import app

###############################################################################
class DummyJobManager:

    # -------------------------------------------------------------------------
    def __init__(self) -> None:
        self.last_job_type = ""
        self.started_jobs = 0

    # -------------------------------------------------------------------------
    def is_job_running(self, job_type: str | None = None) -> bool:
        return False

    # -------------------------------------------------------------------------
    def start_job(self, job_type, runner, args=(), kwargs=None):
        del runner, args, kwargs
        self.started_jobs += 1
        self.last_job_type = str(job_type)
        return "job-123"

    # -------------------------------------------------------------------------
    def get_job_status(self, job_id: str):
        del job_id
        return {"job_type": self.last_job_type, "status": "pending"}

###############################################################################
def test_dataset_job_start_routes_return_202(monkeypatch) -> None:
    manager = DummyJobManager()
    monkeypatch.setattr(app.state, "job_manager", manager)

    from server.services.datasets import DatasetService

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

###############################################################################
def test_dataset_upload_rejects_oversized_file_before_job_dispatch(monkeypatch) -> None:
    manager = DummyJobManager()
    monkeypatch.setattr(app.state, "job_manager", manager)

    from server.api import datasets as datasets_api

    ###############################################################################
    class _DatasetCfg:
        allowed_extensions = (".csv", ".xls", ".xlsx")
        max_upload_bytes = 4

    ###############################################################################
    class _Settings:
        datasets = _DatasetCfg()
        jobs = type("JobsCfg", (), {"polling_interval": 1.0})()

    monkeypatch.setattr(datasets_api, "get_server_settings", lambda: _Settings())

    client = TestClient(app)
    response = client.post(
        "/api/datasets/upload",
        files={"file": ("sample.csv", b"text\nhello\n", "text/csv")},
    )

    assert response.status_code == 413
    assert manager.started_jobs == 0

###############################################################################
def test_dataset_list_passes_catalog_filters_to_service(monkeypatch) -> None:
    from server.services.datasets import DatasetService

    captured: dict[str, object] = {}

    def fake_previews(self, **filters):
        del self
        captured.update(filters)
        return [{"dataset_name": "custom/demo", "document_count": 4}]

    monkeypatch.setattr(DatasetService, "get_dataset_previews", fake_previews)

    response = TestClient(app).get(
        "/api/datasets/list?search= demo &source=custom"
        "&document_count_operator=at_most&document_count=5"
    )

    assert response.status_code == 200
    assert response.json() == {
        "datasets": [{"dataset_name": "custom/demo", "document_count": 4}],
        "count": 1,
    }
    assert captured == {
        "search": " demo ",
        "source": "custom",
        "document_count_operator": "at_most",
        "document_count": 5,
    }

###############################################################################
def test_dataset_delete_normalizes_identifier_and_returns_not_found_for_repeat(monkeypatch) -> None:
    from server.services.datasets import DatasetService

    calls: list[tuple[str, str]] = []
    available = {"custom/foo"}

    monkeypatch.setattr(
        DatasetService,
        "is_dataset_in_database",
        lambda self, name: calls.append(("exists", name)) or name in available,
    )
    monkeypatch.setattr(
        DatasetService,
        "remove_dataset",
        lambda self, name: (calls.append(("remove", name)), available.remove(name)),
    )

    client = TestClient(app)
    deleted = client.delete(
        "/api/datasets/delete",
        params={"dataset_name": "custom/foo"},
    )

    assert deleted.status_code == 200
    assert deleted.json() == {
        "status": "success",
        "dataset_name": "custom/foo",
        "message": "Dataset removed.",
    }
    assert calls == [("exists", "custom/foo"), ("remove", "custom/foo")]

    repeated = client.delete(
        "/api/datasets/delete",
        params={"dataset_name": "custom/foo"},
    )
    assert repeated.status_code == 404
    assert repeated.json()["detail"] == "Dataset 'custom/foo' not found."
