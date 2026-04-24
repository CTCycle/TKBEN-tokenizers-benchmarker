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
        return "job-xyz"

    def get_job_status(self, job_id: str):
        del job_id
        return {"job_type": self.last_job_type, "status": "pending"}


def test_tokenizer_upload_validation_and_custom_clear(monkeypatch) -> None:
    client = TestClient(app)

    invalid_ext = client.post(
        "/api/tokenizers/upload",
        files={"file": ("tokenizer.txt", b"x", "text/plain")},
    )
    assert invalid_ext.status_code == 400

    empty = client.post(
        "/api/tokenizers/upload",
        files={"file": ("tokenizer.json", b"", "application/json")},
    )
    assert empty.status_code == 400

    from TKBEN.server.api import tokenizers as tokenizers_api

    class _TokenizerCfg:
        max_upload_bytes = 1

    class _Settings:
        tokenizers = _TokenizerCfg()
        jobs = type("JobsCfg", (), {"polling_interval": 1.0})()

    monkeypatch.setattr(tokenizers_api, "get_server_settings", lambda: _Settings())

    oversized = client.post(
        "/api/tokenizers/upload",
        files={"file": ("tokenizer.json", b"{}", "application/json")},
    )
    assert oversized.status_code == 413
    monkeypatch.setattr(tokenizers_api, "get_server_settings", lambda: type(
        "Settings",
        (),
        {
            "tokenizers": type("TokenizerCfg", (), {"max_upload_bytes": 10_000_000})(),
            "jobs": type("JobsCfg", (), {"polling_interval": 1.0})(),
        },
    )())

    def fake_upload(content: bytes, normalized_filename: str, safe_stem: str):
        del content, normalized_filename, safe_stem
        return {
            "status": "success",
            "tokenizer_name": "CUSTOM_demo",
            "is_compatible": True,
        }

    monkeypatch.setattr(tokenizers_api.tokenizer_job_service, "upload_custom_tokenizer", fake_upload)

    ok_upload = client.post(
        "/api/tokenizers/upload",
        files={"file": ("tokenizer.json", b"{}", "application/json")},
    )
    assert ok_upload.status_code == 200
    assert ok_upload.json()["is_compatible"] is True

    called = {"value": False}

    def fake_clear() -> None:
        called["value"] = True

    monkeypatch.setattr(tokenizers_api.tokenizer_job_service, "clear_custom_tokenizers", fake_clear)

    cleared = client.delete("/api/tokenizers/custom")
    assert cleared.status_code == 200
    assert called["value"] is True


def test_tokenizer_job_routes_return_202(monkeypatch) -> None:
    manager = DummyJobManager()
    monkeypatch.setattr(app.state, "job_manager", manager)

    from TKBEN.server.services.keys import HFAccessKeyService
    from TKBEN.server.services.tokenizers import TokenizersService

    monkeypatch.setattr(HFAccessKeyService, "get_active_key", lambda self: "token")
    monkeypatch.setattr(
        TokenizersService,
        "has_cached_tokenizer",
        lambda self, tokenizer_name: tokenizer_name == "bert-base-uncased",
    )

    client = TestClient(app)

    download_resp = client.post(
        "/api/tokenizers/download",
        json={"tokenizers": ["bert-base-uncased"]},
    )
    assert download_resp.status_code == 202
    assert download_resp.json()["job_id"] == "job-xyz"

    report_resp = client.post(
        "/api/tokenizers/reports/generate",
        json={"tokenizer_name": "bert-base-uncased"},
    )
    assert report_resp.status_code == 202
    assert report_resp.json()["job_id"] == "job-xyz"
