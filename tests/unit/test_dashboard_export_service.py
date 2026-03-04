from __future__ import annotations

from io import BytesIO
from pathlib import Path

import pytest
from PIL import Image, ImageDraw

from TKBEN.server.services.dashboard_export import DashboardExportService


def build_dashboard_png(width: int, height: int) -> bytes:
    image = Image.new("RGB", (width, height), color=(22, 26, 29))
    draw = ImageDraw.Draw(image)
    band_count = 12
    band_height = max(40, height // band_count)
    for index in range(band_count):
        top = index * band_height
        bottom = min(height, top + band_height)
        fill = (35 + index * 8, 42 + index * 6, 56 + index * 5)
        draw.rectangle((0, top, width, bottom), fill=fill)

    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def test_export_dashboard_pdf_writes_pdf(tmp_path: Path) -> None:
    service = DashboardExportService()
    image_bytes = build_dashboard_png(width=1600, height=2200)

    result = service.export_dashboard_pdf(
        dashboard_type="dataset",
        report_name="dataset-main-report",
        output_dir=str(tmp_path),
        file_name="dataset-main-report",
        image_bytes=image_bytes,
    )

    output_path = Path(result.output_path)
    assert result.dashboard_type == "dataset"
    assert result.file_name == "dataset-main-report.pdf"
    assert result.page_count >= 1
    assert output_path.exists()
    assert output_path.suffix.lower() == ".pdf"
    assert output_path.stat().st_size > 0


def test_export_dashboard_pdf_splits_long_capture_across_pages(tmp_path: Path) -> None:
    service = DashboardExportService()
    image_bytes = build_dashboard_png(width=1400, height=9800)

    result = service.export_dashboard_pdf(
        dashboard_type="benchmark",
        report_name="benchmark-report",
        output_dir=str(tmp_path),
        file_name="benchmark-report",
        image_bytes=image_bytes,
    )

    assert result.dashboard_type == "benchmark"
    assert result.page_count >= 2
    assert Path(result.output_path).exists()


def test_export_dashboard_pdf_rejects_unsupported_dashboard_type(tmp_path: Path) -> None:
    service = DashboardExportService()
    image_bytes = build_dashboard_png(width=1200, height=1200)

    with pytest.raises(ValueError, match="Unsupported dashboard type"):
        service.export_dashboard_pdf(
            dashboard_type="invalid",
            report_name="report",
            output_dir=str(tmp_path),
            file_name="report",
            image_bytes=image_bytes,
        )

