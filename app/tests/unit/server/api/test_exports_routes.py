from __future__ import annotations

from types import SimpleNamespace

from fastapi.testclient import TestClient

from server.app import app


###############################################################################
def _export_payload() -> dict[str, object]:
    return {
        "dashboard_type": "benchmark",
        "report_name": "demo",
        "file_name": "demo",
        "dashboard_payload": {"report_id": 1},
    }


###############################################################################
def test_export_route_returns_pdf_headers_and_bytes(monkeypatch) -> None:
    from server.api import exports as exports_api

    ###############################################################################
    class FakeExportService:
        # -------------------------------------------------------------------------
        def export_dashboard_pdf(self, **kwargs):
            assert kwargs["dashboard_type"] == "benchmark"
            return SimpleNamespace(
                file_name="demo.pdf",
                page_count=2,
                pdf_bytes=b"%PDF-test",
            )

    monkeypatch.setattr(exports_api, "DashboardExportService", FakeExportService)

    response = TestClient(app).post(
        "/api/exports/dashboard/pdf", json=_export_payload()
    )

    assert response.status_code == 200
    assert response.headers["content-type"] == "application/pdf"
    assert response.headers["content-disposition"] == 'attachment; filename="demo.pdf"'
    assert response.headers["x-export-page-count"] == "2"
    assert response.content == b"%PDF-test"


###############################################################################
def test_export_route_maps_expected_and_unexpected_service_failures(
    monkeypatch,
) -> None:
    from server.api import exports as exports_api

    ###############################################################################
    class ValueErrorService:
        # -------------------------------------------------------------------------
        def export_dashboard_pdf(self, **kwargs):
            del kwargs
            raise ValueError("unsupported visualization")

    monkeypatch.setattr(exports_api, "DashboardExportService", ValueErrorService)
    invalid = TestClient(app).post("/api/exports/dashboard/pdf", json=_export_payload())
    assert invalid.status_code == 400
    assert invalid.json()["detail"] == "unsupported visualization"

    ###############################################################################
    class FailingService:
        # -------------------------------------------------------------------------
        def export_dashboard_pdf(self, **kwargs):
            del kwargs
            raise RuntimeError("private rendering details")

    monkeypatch.setattr(exports_api, "DashboardExportService", FailingService)
    failed = TestClient(app).post("/api/exports/dashboard/pdf", json=_export_payload())
    assert failed.status_code == 500
    assert failed.json()["detail"] == "Failed to export dashboard as PDF."
    assert "private rendering" not in failed.text
