from __future__ import annotations

from TKBEN.server.app import app


def test_openapi_generation_and_response_models() -> None:
    schema = app.openapi()
    assert schema

    paths = schema.get("paths", {})

    tokenizer_expectations = {
        "/api/tokenizers/settings": "TokenizerSettingsResponse",
        "/api/tokenizers/scan": "TokenizerScanResponse",
        "/api/tokenizers/list": "TokenizerListResponse",
        "/api/tokenizers/download": "JobStartResponse",
        "/api/tokenizers/reports/generate": "JobStartResponse",
        "/api/tokenizers/reports/latest": "TokenizerReportResponse",
        "/api/tokenizers/reports/{report_id}": "TokenizerReportResponse",
        "/api/tokenizers/reports/{report_id}/vocabulary": "TokenizerVocabularyPageResponse",
        "/api/tokenizers/upload": "TokenizerUploadResponse",
        "/api/tokenizers/custom": "CustomTokenizersDeleteResponse",
    }

    benchmark_expectations = {
        "/api/benchmarks/run": "JobStartResponse",
        "/api/benchmarks/reports": "BenchmarkReportListResponse",
        "/api/benchmarks/reports/{report_id}": "BenchmarkRunResponse",
        "/api/benchmarks/metrics/catalog": "BenchmarkMetricCatalogResponse",
    }

    for path, model_name in {**tokenizer_expectations, **benchmark_expectations}.items():
        assert path in paths

        methods = paths[path]
        status_code = "202" if model_name == "JobStartResponse" and path in {
            "/api/tokenizers/download",
            "/api/tokenizers/reports/generate",
            "/api/benchmarks/run",
        } else "200"

        method_key = "post" if path in {
            "/api/tokenizers/download",
            "/api/tokenizers/reports/generate",
            "/api/tokenizers/upload",
            "/api/benchmarks/run",
        } else "delete" if path == "/api/tokenizers/custom" else "get"

        content = (
            methods[method_key]["responses"][status_code]["content"]["application/json"][
                "schema"
            ]
        )
        assert content.get("$ref", "").endswith(f"/{model_name}")
