from __future__ import annotations

from fastapi.testclient import TestClient

from server.app import app

###############################################################################
class DummyJobManager:

    # -------------------------------------------------------------------------
    def __init__(self) -> None:
        self.last_job_type = ""

    # -------------------------------------------------------------------------
    def is_job_running(self, job_type: str | None = None) -> bool:
        return False

    # -------------------------------------------------------------------------
    def start_job(self, job_type, runner, args=(), kwargs=None):
        del runner, args, kwargs
        self.last_job_type = str(job_type)
        return "job-xyz"

    # -------------------------------------------------------------------------
    def get_job_status(self, job_id: str):
        del job_id
        return {"job_type": self.last_job_type, "status": "pending"}

###############################################################################
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

    from server.api import tokenizers as tokenizers_api

    ###############################################################################
    class _TokenizerCfg:
        max_upload_bytes = 1

    ###############################################################################
    class _Settings:
        tokenizers = _TokenizerCfg()
        jobs = type("JobsCfg", (), {"polling_interval": 1.0})()

    monkeypatch.setattr(tokenizers_api, "get_server_settings", lambda: _Settings())

    oversized = client.post(
        "/api/tokenizers/upload",
        files={"file": ("tokenizer.json", b"{}", "application/json")},
    )
    assert oversized.status_code == 413
    monkeypatch.setattr(
        tokenizers_api,
        "get_server_settings",
        lambda: type(
            "Settings",
            (),
            {
                "tokenizers": type(
                    "TokenizerCfg", (), {"max_upload_bytes": 10_000_000}
                )(),
                "jobs": type("JobsCfg", (), {"polling_interval": 1.0})(),
            },
        )(),
    )

    def fake_upload(self, content: bytes, normalized_filename: str, safe_stem: str):
        del self, content, normalized_filename, safe_stem
        return {
            "status": "success",
            "tokenizer_name": "CUSTOM_demo",
            "is_compatible": True,
        }

    monkeypatch.setattr(
        tokenizers_api.TokenizersService,
        "register_custom_tokenizer_from_upload",
        fake_upload,
    )

    ok_upload = client.post(
        "/api/tokenizers/upload",
        files={"file": ("tokenizer.json", b"{}", "application/json")},
    )
    assert ok_upload.status_code == 200
    assert ok_upload.json()["is_compatible"] is True

    called = {"value": False}

    def fake_clear(self) -> None:
        del self
        called["value"] = True

    monkeypatch.setattr(
        tokenizers_api.TokenizersService,
        "clear_custom_tokenizers",
        fake_clear,
    )

    cleared = client.delete("/api/tokenizers/custom")
    assert cleared.status_code == 200
    assert called["value"] is True

###############################################################################
def test_tokenizer_upload_rejects_oversized_file_before_service_call(monkeypatch) -> None:
    from server.api import tokenizers as tokenizers_api

    ###############################################################################
    class _TokenizerCfg:
        max_upload_bytes = 1

    ###############################################################################
    class _Settings:
        tokenizers = _TokenizerCfg()
        jobs = type("JobsCfg", (), {"polling_interval": 1.0})()

    called = {"upload": False}

    def fake_upload(self, content: bytes, normalized_filename: str, safe_stem: str):
        del self, content, normalized_filename, safe_stem
        called["upload"] = True
        return {"status": "success", "tokenizer_name": "CUSTOM_demo"}

    monkeypatch.setattr(tokenizers_api, "get_server_settings", lambda: _Settings())
    monkeypatch.setattr(
        tokenizers_api.TokenizersService,
        "register_custom_tokenizer_from_upload",
        fake_upload,
    )

    client = TestClient(app)
    response = client.post(
        "/api/tokenizers/upload",
        files={"file": ("tokenizer.json", b"{}", "application/json")},
    )

    assert response.status_code == 413
    assert called["upload"] is False

###############################################################################
def test_tokenizer_job_routes_return_202(monkeypatch) -> None:
    manager = DummyJobManager()
    monkeypatch.setattr(app.state, "job_manager", manager)

    from server.services.keys import HFAccessKeyService
    from server.services.tokenizers import TokenizersService

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

###############################################################################
def test_tokenizer_scan_returns_sanitized_500_on_upstream_failure(monkeypatch) -> None:
    from server.services.tokenizers import TokenizersService

    def fail_scan(self, limit: int):
        del self, limit
        raise RuntimeError("private upstream credentials and response details")

    monkeypatch.setattr(TokenizersService, "get_tokenizer_identifiers", fail_scan)

    response = TestClient(app).get("/api/tokenizers/scan")

    assert response.status_code == 500
    assert response.json()["detail"] == "Failed to retrieve tokenizers from HuggingFace."
    assert "private upstream" not in response.text

###############################################################################
def test_tokenizer_list_passes_catalog_filters_to_service(monkeypatch) -> None:
    from server.services.tokenizers import TokenizersService

    captured: dict[str, object] = {}

    def fake_catalog(self, **filters):
        del self
        captured.update(filters)
        return [{
            "tokenizer_name": "CUSTOM_demo",
            "source": "custom",
            "has_report": False,
            "vocabulary_size": 3,
        }]

    monkeypatch.setattr(TokenizersService, "list_tokenizer_catalog", fake_catalog)

    response = TestClient(app).get(
        "/api/tokenizers/list?search= demo &source=custom"
        "&vocabulary_size_operator=at_most&vocabulary_size=5"
    )

    assert response.status_code == 200
    assert response.json() == {
        "tokenizers": [{
            "tokenizer_name": "CUSTOM_demo",
            "source": "custom",
            "has_report": False,
            "vocabulary_size": 3,
        }],
        "count": 1,
    }
    assert captured == {
        "search": " demo ",
        "source": "custom",
        "vocabulary_size_operator": "at_most",
        "vocabulary_size": 5,
    }
