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
def test_tokenizer_delete_supports_encoded_and_custom_names_and_returns_404(monkeypatch) -> None:
    from server.api import tokenizers as tokenizers_api

    deleted: list[str] = []

    available = {"google-bert/bert-base-uncased", "CUSTOM_sample"}

    def fake_remove(self, name: str) -> bool:
        del self
        deleted.append(name)
        if name in available:
            available.remove(name)
            return True
        return False

    monkeypatch.setattr(
        tokenizers_api.TokenizersService,
        "remove_downloaded_tokenizer",
        fake_remove,
    )

    client = TestClient(app)
    downloaded = client.delete(
        "/api/tokenizers/delete",
        params={"tokenizer_name": "google-bert/bert-base-uncased"},
    )
    custom = client.delete(
        "/api/tokenizers/delete",
        params={"tokenizer_name": "CUSTOM_sample"},
    )
    missing = client.delete(
        "/api/tokenizers/delete",
        params={"tokenizer_name": "CUSTOM_sample"},
    )

    assert downloaded.status_code == 200
    assert custom.status_code == 200
    assert missing.status_code == 404
    assert downloaded.json()["tokenizer_name"] == "google-bert/bert-base-uncased"
    assert missing.json()["detail"] == "Tokenizer 'CUSTOM_sample' is not downloaded."
    assert deleted == [
        "google-bert/bert-base-uncased",
        "CUSTOM_sample",
        "CUSTOM_sample",
    ]

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
def test_tokenizer_discovery_returns_sanitized_500_on_upstream_failure(monkeypatch) -> None:
    from server.services.tokenizers import TokenizersService

    def fail_discovery(self, query):
        del self, query
        raise RuntimeError("private upstream credentials and response details")

    monkeypatch.setattr(TokenizersService, "discover_tokenizers", fail_discovery)

    response = TestClient(app).get("/api/tokenizers/discover")

    assert response.status_code == 500
    assert response.json()["detail"] == "Failed to discover tokenizers from HuggingFace."
    assert "private upstream" not in response.text

###############################################################################
def test_tokenizer_discovery_validates_combined_query_and_structured_response(monkeypatch) -> None:
    from server.contracts.tokenizers import TokenizerDiscoveryResponse
    from server.services.tokenizers import TokenizersService

    captured = {}

    def fake_discovery(self, query):
        del self
        captured.update(query.model_dump())
        return TokenizerDiscoveryResponse.model_validate({
            "items": [{
                "identifier": "bert-base-uncased",
                "pipeline_tag": "fill-mask",
                "library_name": "transformers",
                "downloads": 10,
                "likes": 2,
                "last_modified": None,
                "gated": False,
                "tags": ["core"],
                "vocabulary_size": None,
            }],
            "count": 1,
            "fetched_count": 3,
        })

    monkeypatch.setattr(TokenizersService, "discover_tokenizers", fake_discovery)
    response = TestClient(app).get(
        "/api/tokenizers/discover?search=%20bert%20&limit=5&pipeline_tag=fill-mask"
        "&author=google&include_tags=core&exclude_tags=audio&access=public&sort=downloads"
    )

    assert response.status_code == 200
    assert response.json()["items"][0]["identifier"] == "bert-base-uncased"
    assert response.json()["count"] == 1
    assert captured["search"] == "bert"
    assert captured["limit"] == 5
    assert captured["pipeline_tag"] == "fill-mask"
    assert captured["author"] == "google"
    assert captured["include_tags"] == ["core"]
    assert captured["exclude_tags"] == ["audio"]
    assert captured["access"] == "public"

###############################################################################
def test_tokenizer_discovery_rejects_invalid_query(monkeypatch) -> None:
    from server.services.tokenizers import TokenizersService

    called = {"value": False}

    def fail_discovery(self, query):
        del self, query
        called["value"] = True
        return None

    monkeypatch.setattr(TokenizersService, "discover_tokenizers", fail_discovery)
    client = TestClient(app)
    assert client.get("/api/tokenizers/discover?pipeline_tag=image-tokenization").status_code == 422
    assert client.get("/api/tokenizers/discover?vocabulary_operator=at_least").status_code == 422
    assert client.get("/api/tokenizers/discover?vocabulary_size=-1").status_code == 422
    assert client.get("/api/tokenizers/discover?include_tags=shared&exclude_tags=shared").status_code == 422
    assert called["value"] is False

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
