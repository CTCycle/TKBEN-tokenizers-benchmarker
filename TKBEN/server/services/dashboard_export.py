from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
import re
from typing import Literal

from PIL import Image, UnidentifiedImageError

from TKBEN.server.common.utils.security import contains_control_chars


SAFE_FILE_CHARS_PATTERN = re.compile(r"[^A-Za-z0-9._ ()-]+")
ALLOWED_DASHBOARD_TYPES = {"dataset", "tokenizer", "benchmark"}
MAX_IMAGE_BYTES = 40 * 1024 * 1024
MAX_FILE_STEM_LENGTH = 120


###############################################################################
@dataclass(frozen=True)
class DashboardExportResult:
    dashboard_type: Literal["dataset", "tokenizer", "benchmark"]
    output_path: str
    file_name: str
    page_count: int
    image_width: int
    image_height: int


###############################################################################
class DashboardExportService:
    def export_dashboard_pdf(
        self,
        *,
        dashboard_type: str,
        report_name: str,
        output_dir: str,
        file_name: str,
        image_bytes: bytes,
    ) -> DashboardExportResult:
        normalized_dashboard_type = self._normalize_dashboard_type(dashboard_type)
        normalized_output_dir = self._normalize_output_dir(output_dir)
        normalized_file_name = self._normalize_file_name(file_name, report_name)

        if not image_bytes:
            raise ValueError("Dashboard image is empty.")
        if len(image_bytes) > MAX_IMAGE_BYTES:
            raise ValueError(
                f"Dashboard image is too large ({MAX_IMAGE_BYTES} bytes max)."
            )

        image = self._decode_image(image_bytes)
        page_images = self._build_pdf_pages(image)

        normalized_output_dir.mkdir(parents=True, exist_ok=True)
        output_path = normalized_output_dir / normalized_file_name
        page_images[0].save(
            output_path,
            format="PDF",
            save_all=True,
            append_images=page_images[1:],
            resolution=300.0,
        )

        return DashboardExportResult(
            dashboard_type=normalized_dashboard_type,
            output_path=str(output_path),
            file_name=normalized_file_name,
            page_count=len(page_images),
            image_width=image.width,
            image_height=image.height,
        )

    # -------------------------------------------------------------------------
    def _normalize_dashboard_type(
        self,
        dashboard_type: str,
    ) -> Literal["dataset", "tokenizer", "benchmark"]:
        value = dashboard_type.strip().lower()
        if value not in ALLOWED_DASHBOARD_TYPES:
            raise ValueError(
                "Unsupported dashboard type. Use one of: dataset, tokenizer, benchmark."
            )
        return value

    # -------------------------------------------------------------------------
    def _normalize_output_dir(self, output_dir: str) -> Path:
        if not isinstance(output_dir, str):
            raise ValueError("Output path must be a string.")
        value = output_dir.strip()
        if not value:
            raise ValueError("Output path is required.")
        if contains_control_chars(value):
            raise ValueError("Output path contains unsupported control characters.")

        resolved = Path(value).expanduser()
        if not resolved.is_absolute():
            resolved = Path.cwd() / resolved
        return resolved.resolve()

    # -------------------------------------------------------------------------
    def _normalize_file_name(self, file_name: str, report_name: str) -> str:
        candidate = file_name.strip() if isinstance(file_name, str) else ""
        if not candidate:
            candidate = report_name.strip() if isinstance(report_name, str) else ""
        if not candidate:
            candidate = "dashboard-report"
        if "\\" in candidate or "/" in candidate:
            raise ValueError("File name must not contain path separators.")
        if contains_control_chars(candidate):
            raise ValueError("File name contains unsupported control characters.")

        if candidate.lower().endswith(".pdf"):
            candidate = candidate[:-4]
        candidate = SAFE_FILE_CHARS_PATTERN.sub("_", candidate).strip("._- ")
        if not candidate:
            candidate = "dashboard-report"
        candidate = candidate[:MAX_FILE_STEM_LENGTH]
        return f"{candidate}.pdf"

    # -------------------------------------------------------------------------
    def _decode_image(self, image_bytes: bytes) -> Image.Image:
        try:
            with Image.open(BytesIO(image_bytes)) as image:
                decoded = image.copy()
        except UnidentifiedImageError as exc:
            raise ValueError("Unable to decode dashboard image.") from exc
        return self._ensure_rgb(decoded)

    # -------------------------------------------------------------------------
    def _ensure_rgb(self, image: Image.Image) -> Image.Image:
        if image.mode == "RGB":
            return image
        if image.mode in {"RGBA", "LA"}:
            background = Image.new("RGB", image.size, "white")
            alpha = image.getchannel("A")
            background.paste(image.convert("RGB"), mask=alpha)
            return background
        return image.convert("RGB")

    # -------------------------------------------------------------------------
    def _build_pdf_pages(self, image: Image.Image) -> list[Image.Image]:
        if image.width <= 0 or image.height <= 0:
            raise ValueError("Dashboard image has invalid dimensions.")

        is_landscape = image.width > image.height
        page_width, page_height = (3508, 2480) if is_landscape else (2480, 3508)
        scaled_height = max(1, int(round((image.height * page_width) / image.width)))
        resized = image.resize((page_width, scaled_height), Image.Resampling.LANCZOS)

        pages: list[Image.Image] = []
        for top in range(0, scaled_height, page_height):
            bottom = min(top + page_height, scaled_height)
            chunk = resized.crop((0, top, page_width, bottom))
            if chunk.height == page_height:
                pages.append(chunk)
                continue

            padded = Image.new("RGB", (page_width, page_height), "white")
            padded.paste(chunk, (0, 0))
            pages.append(padded)
        return pages
