from __future__ import annotations

from server.app import app

###############################################################################
JSON_ROUTE_EXPECTATIONS = [
    ("/api/health", "get", 200, "HealthResponse"),
    ("/api/datasets/list", "get", 200, "DatasetListResponse"),
    ("/api/datasets/metrics/catalog", "get", 200, "DatasetMetricCatalogResponse"),
    ("/api/datasets/download", "post", 202, "JobStartResponse"),
    ("/api/datasets/upload", "post", 202, "JobStartResponse"),
    ("/api/datasets/analyze", "post", 202, "JobStartResponse"),
    ("/api/datasets/reports/latest", "get", 200, "DatasetAnalysisResponse"),
    ("/api/datasets/reports/{report_id}", "get", 200, "DatasetAnalysisResponse"),
    ("/api/datasets/delete", "delete", 200, "DatasetDeleteResponse"),
    ("/api/tokenizers/settings", "get", 200, "TokenizerSettingsResponse"),
    ("/api/tokenizers/scan", "get", 200, "TokenizerScanResponse"),
    ("/api/tokenizers/list", "get", 200, "TokenizerListResponse"),
    ("/api/tokenizers/download", "post", 202, "JobStartResponse"),
    ("/api/tokenizers/reports/generate", "post", 202, "JobStartResponse"),
    ("/api/tokenizers/reports/latest", "get", 200, "TokenizerReportResponse"),
    ("/api/tokenizers/reports/{report_id}", "get", 200, "TokenizerReportResponse"),
    (
        "/api/tokenizers/reports/{report_id}/vocabulary",
        "get",
        200,
        "TokenizerVocabularyPageResponse",
    ),
    ("/api/tokenizers/upload", "post", 200, "TokenizerUploadResponse"),
    ("/api/tokenizers/custom", "delete", 200, "CustomTokenizersDeleteResponse"),
    ("/api/tokenizers/delete", "delete", 200, "TokenizerDeleteResponse"),
    ("/api/benchmarks/run", "post", 202, "JobStartResponse"),
    ("/api/benchmarks/reports", "get", 200, "BenchmarkReportListResponse"),
    ("/api/benchmarks/reports/{report_id}", "get", 200, "BenchmarkRunResponse"),
    (
        "/api/benchmarks/metrics/catalog",
        "get",
        200,
        "BenchmarkMetricCatalogResponse",
    ),
    ("/api/jobs/{job_id}", "get", 200, "JobStatusResponse"),
    ("/api/jobs/{job_id}/cancel", "post", 200, "JobStatusResponse"),
    ("/api/keys", "post", 201, "HFAccessKeyListItem"),
    ("/api/keys", "get", 200, "HFAccessKeyListResponse"),
    ("/api/keys/{key_id}", "delete", 200, "HFAccessKeyDeleteResponse"),
    ("/api/keys/{key_id}/activate", "post", 200, "HFAccessKeyActivateResponse"),
    ("/api/keys/{key_id}/deactivate", "post", 200, "HFAccessKeyActivateResponse"),
    ("/api/keys/{key_id}/reveal", "post", 200, "HFAccessKeyRevealResponse"),
]

###############################################################################
def test_openapi_generation_and_response_models() -> None:
    schema = app.openapi()
    assert schema

    paths = schema.get("paths", {})
    for path, method, status_code, model_name in JSON_ROUTE_EXPECTATIONS:
        assert path in paths
        operation = paths[path][method]
        content = operation["responses"][str(status_code)]["content"][
            "application/json"
        ]["schema"]
        assert content.get("$ref", "").endswith(f"/{model_name}")
