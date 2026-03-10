from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
import json
import re
from typing import Any, Literal

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from TKBEN.server.common.utils.security import contains_control_chars


DashboardType = Literal["dataset", "tokenizer", "benchmark"]

SAFE_FILE_CHARS_PATTERN = re.compile(r"[^A-Za-z0-9._ ()-]+")
MAX_FILE_STEM_LENGTH = 120
MAX_PDF_BYTES = 80 * 1024 * 1024
ALLOWED_DASHBOARD_TYPES = {"dataset", "tokenizer", "benchmark"}

PRIMARY_COLOR = "#facc15"
SECONDARY_COLOR = "#38bdf8"
TERTIARY_COLOR = "#22c55e"
MUTED_TEXT = "#5b6472"


###############################################################################
@dataclass(frozen=True)
class DashboardPdfDocument:
    file_name: str
    page_count: int
    pdf_bytes: bytes


###############################################################################
class DashboardExportService:
    def export_dashboard_pdf(
        self,
        *,
        dashboard_type: str,
        report_name: str,
        file_name: str,
        dashboard_payload: dict[str, Any] | None,
    ) -> DashboardPdfDocument:
        normalized_dashboard_type = self._normalize_dashboard_type(dashboard_type)
        normalized_file_name = self._normalize_file_name(file_name, report_name)
        payload = dashboard_payload if isinstance(dashboard_payload, dict) else {}

        buffer = BytesIO()
        page_count = 0
        with PdfPages(buffer) as pdf:
            if normalized_dashboard_type == "dataset":
                page_count += self._render_dataset_dashboard(pdf, report_name, payload)
            elif normalized_dashboard_type == "tokenizer":
                page_count += self._render_tokenizer_dashboard(
                    pdf, report_name, payload
                )
            else:
                page_count += self._render_benchmark_dashboard(
                    pdf, report_name, payload
                )

        pdf_bytes = buffer.getvalue()
        if not pdf_bytes:
            raise ValueError("Failed to generate dashboard PDF.")
        if len(pdf_bytes) > MAX_PDF_BYTES:
            raise ValueError(f"Generated PDF is too large ({MAX_PDF_BYTES} bytes max).")

        return DashboardPdfDocument(
            file_name=normalized_file_name,
            page_count=max(1, page_count),
            pdf_bytes=pdf_bytes,
        )

    # -------------------------------------------------------------------------
    def _normalize_dashboard_type(self, dashboard_type: str) -> DashboardType:
        value = dashboard_type.strip().lower()
        if value not in ALLOWED_DASHBOARD_TYPES:
            raise ValueError(
                "Unsupported dashboard type. Use one of: dataset, tokenizer, benchmark."
            )
        return value  # type: ignore[return-value]

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
    def _render_dataset_dashboard(
        self,
        pdf: PdfPages,
        report_name: str,
        payload: dict[str, Any],
    ) -> int:
        report = self._extract_nested(payload, "report")
        source = report if report else payload
        aggregate = source.get("aggregate_statistics")
        if not isinstance(aggregate, dict):
            aggregate = {}

        fig = plt.figure(figsize=(11.69, 8.27), constrained_layout=True)
        grid = fig.add_gridspec(3, 2, height_ratios=[0.28, 0.36, 0.36])

        title_ax = fig.add_subplot(grid[0, :])
        title_ax.axis("off")
        dataset_name = str(source.get("dataset_name") or "N/A")
        created_at = str(source.get("created_at") or "")
        title_ax.text(
            0.0,
            0.9,
            "Dataset Dashboard Report",
            fontsize=18,
            fontweight="bold",
            color="#111827",
        )
        title_ax.text(
            0.0,
            0.6,
            f"Report: {report_name or dataset_name}",
            fontsize=11,
            color=MUTED_TEXT,
        )
        title_ax.text(
            0.0,
            0.38,
            f"Dataset: {dataset_name}",
            fontsize=11,
            color=MUTED_TEXT,
        )
        if created_at:
            title_ax.text(
                0.0,
                0.16,
                f"Generated from snapshot: {created_at}",
                fontsize=10,
                color=MUTED_TEXT,
            )

        aggregate_ax = fig.add_subplot(grid[1, 0])
        aggregate_rows = [
            ("Documents", self._format_count(source.get("document_count"))),
            ("Mean length", self._format_number(aggregate.get("doc.length_mean"), 2)),
            ("Min length", self._format_count(aggregate.get("doc.length_min"))),
            ("Max length", self._format_count(aggregate.get("doc.length_max"))),
            ("Length CV", self._format_number(aggregate.get("doc.length_cv"), 4)),
            ("p50", self._format_count(aggregate.get("doc.length_p50"))),
            ("p90", self._format_count(aggregate.get("doc.length_p90"))),
            ("p99", self._format_count(aggregate.get("doc.length_p99"))),
        ]
        self._render_table_card(aggregate_ax, "Aggregate Stats", aggregate_rows)

        word_metrics_ax = fig.add_subplot(grid[1, 1])
        word_rows = [
            (
                "Vocabulary size",
                self._format_count(aggregate.get("corpus.unique_words")),
            ),
            ("MATTR", self._format_number(aggregate.get("corpus.mattr"), 4)),
            ("Entropy", self._format_number(aggregate.get("words.shannon_entropy"), 4)),
            ("Hapax ratio", self._format_number(aggregate.get("words.hapax_ratio"), 4)),
            ("Zipf slope", self._format_number(aggregate.get("words.zipf_slope"), 4)),
            ("Gini", self._format_number(aggregate.get("words.frequency_gini"), 4)),
            ("HHI", self._format_number(aggregate.get("words.hhi"), 6)),
        ]
        self._render_table_card(word_metrics_ax, "Word Metrics", word_rows)

        histogram_left_ax = fig.add_subplot(grid[2, 0])
        self._render_histogram(
            histogram_left_ax,
            source.get("document_length_histogram"),
            "Document Length Histogram",
            PRIMARY_COLOR,
        )

        histogram_right_ax = fig.add_subplot(grid[2, 1])
        self._render_histogram(
            histogram_right_ax,
            source.get("word_length_histogram"),
            "Word Length Histogram",
            SECONDARY_COLOR,
        )

        pdf.savefig(fig)
        plt.close(fig)

        # Page 2: pie + zipf + top words
        fig2 = plt.figure(figsize=(11.69, 8.27), constrained_layout=True)
        grid2 = fig2.add_gridspec(2, 2, height_ratios=[0.5, 0.5])

        composition_ax = fig2.add_subplot(grid2[0, 0])
        composition_rows = [
            ("Whitespace", self._to_number(aggregate.get("chars.whitespace_ratio"))),
            ("Punctuation", self._to_number(aggregate.get("chars.punctuation_ratio"))),
            ("Digits", self._to_number(aggregate.get("chars.digit_ratio"))),
            ("Uppercase", self._to_number(aggregate.get("chars.uppercase_ratio"))),
            ("Non-ASCII", self._to_number(aggregate.get("chars.non_ascii_ratio"))),
            ("Control", self._to_number(aggregate.get("chars.control_ratio"))),
            ("Other", self._to_number(aggregate.get("chars.other_ratio"))),
        ]
        labels = [name for name, value in composition_rows if value > 0]
        values = [value for _, value in composition_rows if value > 0]
        composition_ax.set_title(
            "Character Composition", fontsize=12, fontweight="bold"
        )
        if values:
            composition_ax.pie(values, labels=labels, autopct="%1.1f%%", startangle=90)
        else:
            composition_ax.axis("off")
            composition_ax.text(
                0.5,
                0.5,
                "No composition data",
                ha="center",
                va="center",
                color=MUTED_TEXT,
            )

        zipf_ax = fig2.add_subplot(grid2[0, 1])
        zipf_points = self._parse_zipf_curve(
            aggregate.get("words.zipf_curve") or aggregate.get("words.zipf")
        )
        zipf_ax.set_title("Zipf Curve", fontsize=12, fontweight="bold")
        if zipf_points:
            zipf_ax.plot(
                [point["rank"] for point in zipf_points],
                [point["frequency"] for point in zipf_points],
                color=SECONDARY_COLOR,
                linewidth=2.0,
            )
            zipf_ax.set_xlabel("Rank")
            zipf_ax.set_ylabel("Frequency")
            zipf_ax.grid(alpha=0.25)
        else:
            zipf_ax.axis("off")
            zipf_ax.text(
                0.5,
                0.5,
                "No Zipf curve data",
                ha="center",
                va="center",
                color=MUTED_TEXT,
            )

        indicators_ax = fig2.add_subplot(grid2[1, 0])
        indicators_ax.axis("off")
        indicators_ax.set_title(
            "Quality Indicators", fontsize=12, fontweight="bold", loc="left"
        )
        indicator_lines = [
            f"Duplicate rate: {self._format_percent(aggregate.get('quality.duplicate_rate'))}",
            f"Near-duplicate rate: {self._format_percent(aggregate.get('quality.near_duplicate_rate'))}",
            f"Top-k concentration: {self._format_percent(aggregate.get('words.topk_concentration'))}",
            f"Rare tail mass: {self._format_percent(aggregate.get('words.rare_tail_mass'))}",
            f"Normalized entropy: {self._format_percent(aggregate.get('words.normalized_entropy'))}",
        ]
        indicators_ax.text(
            0.0,
            0.9,
            "\n".join(indicator_lines),
            fontsize=10.5,
            color="#111827",
            va="top",
            linespacing=1.5,
        )

        top_words_ax = fig2.add_subplot(grid2[1, 1])
        top_words_ax.set_title("Most Common Words", fontsize=12, fontweight="bold")
        top_words = self._parse_word_frequency(source.get("most_common_words"))[:10]
        if top_words:
            labels = [word["word"] for word in top_words]
            counts = [word["count"] for word in top_words]
            top_words_ax.barh(labels[::-1], counts[::-1], color=TERTIARY_COLOR)
            top_words_ax.grid(axis="x", alpha=0.25)
            top_words_ax.set_xlabel("Count")
        else:
            top_words_ax.axis("off")
            top_words_ax.text(
                0.5,
                0.5,
                "No word frequency data",
                ha="center",
                va="center",
                color=MUTED_TEXT,
            )

        pdf.savefig(fig2)
        plt.close(fig2)
        return 2

    # -------------------------------------------------------------------------
    def _render_tokenizer_dashboard(
        self,
        pdf: PdfPages,
        report_name: str,
        payload: dict[str, Any],
    ) -> int:
        report = self._extract_nested(payload, "report")
        source = report if report else payload
        global_stats = source.get("global_stats")
        if not isinstance(global_stats, dict):
            global_stats = {}

        fig = plt.figure(figsize=(11.69, 8.27), constrained_layout=True)
        grid = fig.add_gridspec(2, 2, height_ratios=[0.42, 0.58])

        title_ax = fig.add_subplot(grid[0, :])
        title_ax.axis("off")
        tokenizer_name = str(source.get("tokenizer_name") or "N/A")
        title_ax.text(
            0.0,
            0.9,
            "Tokenizer Dashboard Report",
            fontsize=18,
            fontweight="bold",
            color="#111827",
        )
        title_ax.text(
            0.0,
            0.62,
            f"Report: {report_name or tokenizer_name}",
            fontsize=11,
            color=MUTED_TEXT,
        )
        title_ax.text(
            0.0, 0.42, f"Tokenizer: {tokenizer_name}", fontsize=11, color=MUTED_TEXT
        )
        title_ax.text(
            0.0,
            0.22,
            f"Report ID: {source.get('report_id') or 'N/A'}",
            fontsize=10,
            color=MUTED_TEXT,
        )

        basics_ax = fig.add_subplot(grid[1, 0])
        basics_rows = [
            ("Tokenizer class", str(global_stats.get("tokenizer_class") or "N/A")),
            ("Vocabulary size", self._format_count(source.get("vocabulary_size"))),
            (
                "Base vocabulary",
                self._format_count(global_stats.get("base_vocabulary_size")),
            ),
            (
                "Model max length",
                self._format_count(global_stats.get("model_max_length")),
            ),
            ("Padding side", str(global_stats.get("padding_side") or "N/A")),
            (
                "Special tokens",
                self._format_count(global_stats.get("special_tokens_count")),
            ),
            (
                "Added tokens",
                self._format_count(global_stats.get("added_tokens_count")),
            ),
            ("Hugging Face URL", str(source.get("huggingface_url") or "N/A")),
        ]
        self._render_table_card(basics_ax, "Basics", basics_rows, font_size=9)

        histogram_ax = fig.add_subplot(grid[1, 1])
        self._render_histogram(
            histogram_ax,
            source.get("token_length_histogram"),
            "Token Length Histogram",
            SECONDARY_COLOR,
        )
        pdf.savefig(fig)
        plt.close(fig)

        vocabulary_items = payload.get("vocabulary_items")
        rows = self._parse_vocabulary_items(vocabulary_items)
        if not rows:
            return 1

        pages = 0
        chunk_size = 45
        for start in range(0, min(len(rows), 180), chunk_size):
            chunk = rows[start : start + chunk_size]
            fig_page = plt.figure(figsize=(11.69, 8.27), constrained_layout=True)
            ax = fig_page.add_subplot(111)
            ax.axis("off")
            ax.set_title(
                f"Vocabulary Preview ({start + 1}-{start + len(chunk)} of {len(rows)})",
                fontsize=13,
                fontweight="bold",
                loc="left",
            )
            table = ax.table(
                cellText=[
                    [
                        str(item.get("token_id", "")),
                        str(item.get("token", "")),
                        str(item.get("length", "")),
                    ]
                    for item in chunk
                ],
                colLabels=["token_id", "token", "length"],
                cellLoc="left",
                colLoc="left",
                bbox=[0, 0.02, 1, 0.9],
            )
            table.auto_set_font_size(False)
            table.set_fontsize(8)
            pdf.savefig(fig_page)
            plt.close(fig_page)
            pages += 1
        return 1 + pages

    # -------------------------------------------------------------------------
    def _render_benchmark_dashboard(
        self,
        pdf: PdfPages,
        report_name: str,
        payload: dict[str, Any],
    ) -> int:
        report = self._extract_nested(payload, "report")
        source = report if report else payload
        global_metrics = source.get("global_metrics")
        if not isinstance(global_metrics, list):
            global_metrics = []
        chart_data = source.get("chart_data")
        if not isinstance(chart_data, dict):
            chart_data = {}

        fig = plt.figure(figsize=(11.69, 8.27), constrained_layout=True)
        grid = fig.add_gridspec(3, 2, height_ratios=[0.28, 0.36, 0.36])

        title_ax = fig.add_subplot(grid[0, :])
        title_ax.axis("off")
        dataset_name = str(source.get("dataset_name") or "N/A")
        title_ax.text(
            0.0,
            0.9,
            "Benchmark Dashboard Report",
            fontsize=18,
            fontweight="bold",
            color="#111827",
        )
        title_ax.text(
            0.0,
            0.62,
            f"Report: {report_name or dataset_name}",
            fontsize=11,
            color=MUTED_TEXT,
        )
        title_ax.text(
            0.0, 0.42, f"Dataset: {dataset_name}", fontsize=11, color=MUTED_TEXT
        )
        title_ax.text(
            0.0,
            0.22,
            f"Documents: {self._format_count(source.get('documents_processed'))} | "
            f"Tokenizers: {self._format_count(source.get('tokenizers_count'))}",
            fontsize=10,
            color=MUTED_TEXT,
        )

        speed_ax = fig.add_subplot(grid[1, 0])
        speed_ax.set_title("Speed Comparison", fontsize=12, fontweight="bold")
        speed_rows = chart_data.get("speed_metrics")
        if isinstance(speed_rows, list) and speed_rows:
            labels = [
                self._short_name(str(item.get("tokenizer") or ""))
                for item in speed_rows
            ]
            values = [
                self._to_number(item.get("tokens_per_second")) for item in speed_rows
            ]
            speed_ax.bar(labels, values, color=SECONDARY_COLOR)
            speed_ax.set_ylabel("tokens/sec")
            speed_ax.tick_params(axis="x", rotation=35)
            speed_ax.grid(axis="y", alpha=0.25)
        else:
            speed_ax.axis("off")
            speed_ax.text(
                0.5,
                0.5,
                "No speed metrics available",
                ha="center",
                va="center",
                color=MUTED_TEXT,
            )

        rates_ax = fig.add_subplot(grid[1, 1])
        rates_ax.set_title("Global Rates", fontsize=12, fontweight="bold")
        if global_metrics:
            labels = [
                self._short_name(str(item.get("tokenizer") or ""))
                for item in global_metrics
            ]
            oov = [self._as_percent(item.get("oov_rate")) for item in global_metrics]
            round_trip = [
                self._as_percent(item.get("round_trip_fidelity_rate"))
                for item in global_metrics
            ]
            x = list(range(len(labels)))
            width = 0.38
            rates_ax.bar(
                [value - width / 2 for value in x],
                oov,
                width=width,
                label="OOV %",
                color="#f87171",
            )
            rates_ax.bar(
                [value + width / 2 for value in x],
                round_trip,
                width=width,
                label="Round-trip %",
                color=TERTIARY_COLOR,
            )
            rates_ax.set_xticks(x, labels)
            rates_ax.tick_params(axis="x", rotation=35)
            rates_ax.set_ylabel("Percent")
            rates_ax.legend()
            rates_ax.grid(axis="y", alpha=0.25)
        else:
            rates_ax.axis("off")
            rates_ax.text(
                0.5,
                0.5,
                "No global rate metrics available",
                ha="center",
                va="center",
                color=MUTED_TEXT,
            )

        vocab_ax = fig.add_subplot(grid[2, 0])
        vocab_ax.set_title("Vocabulary Size", fontsize=12, fontweight="bold")
        vocabulary_rows = chart_data.get("vocabulary_stats")
        if isinstance(vocabulary_rows, list) and vocabulary_rows:
            labels = [
                self._short_name(str(item.get("tokenizer") or ""))
                for item in vocabulary_rows
            ]
            values = [
                self._to_number(item.get("vocabulary_size")) for item in vocabulary_rows
            ]
            vocab_ax.bar(labels, values, color=PRIMARY_COLOR)
            vocab_ax.tick_params(axis="x", rotation=35)
            vocab_ax.grid(axis="y", alpha=0.25)
        else:
            vocab_ax.axis("off")
            vocab_ax.text(
                0.5,
                0.5,
                "No vocabulary stats available",
                ha="center",
                va="center",
                color=MUTED_TEXT,
            )

        dist_ax = fig.add_subplot(grid[2, 1])
        dist_ax.set_title("Token Length Distribution", fontsize=12, fontweight="bold")
        distribution_rows = chart_data.get("token_length_distributions")
        selected_distribution = str(
            payload.get("selected_distribution_tokenizer") or ""
        )
        selected = self._select_distribution(distribution_rows, selected_distribution)
        if selected:
            bins = selected.get("bins")
            if isinstance(bins, list) and bins:
                labels = [
                    f"{int(self._to_number(item.get('bin_start')))}-{int(self._to_number(item.get('bin_end')))}"
                    for item in bins
                ]
                counts = [self._to_number(item.get("count")) for item in bins]
                dist_ax.bar(labels, counts, color="#a78bfa")
                dist_ax.tick_params(axis="x", rotation=45)
                dist_ax.grid(axis="y", alpha=0.25)
            else:
                dist_ax.axis("off")
                dist_ax.text(
                    0.5,
                    0.5,
                    "No distribution bins",
                    ha="center",
                    va="center",
                    color=MUTED_TEXT,
                )
        else:
            dist_ax.axis("off")
            dist_ax.text(
                0.5,
                0.5,
                "No token length distribution",
                ha="center",
                va="center",
                color=MUTED_TEXT,
            )

        pdf.savefig(fig)
        plt.close(fig)

        metrics_page = plt.figure(figsize=(11.69, 8.27), constrained_layout=True)
        ax = metrics_page.add_subplot(111)
        ax.axis("off")
        ax.set_title(
            "Per Tokenizer Additional Metrics",
            fontsize=13,
            fontweight="bold",
            loc="left",
        )

        table_rows = self._build_benchmark_metrics_table(global_metrics)
        table = ax.table(
            cellText=table_rows,
            colLabels=[
                "Tokenizer",
                "Word Recovery %",
                "Coverage %",
                "Subword Fertility",
                "Entropy",
            ],
            cellLoc="left",
            colLoc="left",
            bbox=[0, 0.02, 1, 0.92],
        )
        table.auto_set_font_size(False)
        table.set_fontsize(8.5)
        pdf.savefig(metrics_page)
        plt.close(metrics_page)
        return 2

    # -------------------------------------------------------------------------
    def _render_table_card(
        self,
        ax: Any,
        title: str,
        rows: list[tuple[str, str]],
        *,
        font_size: float = 9.5,
    ) -> None:
        ax.axis("off")
        ax.set_title(title, fontsize=12, fontweight="bold", loc="left")
        table = ax.table(
            cellText=[[label, value] for label, value in rows],
            colLabels=["Metric", "Value"],
            cellLoc="left",
            colLoc="left",
            bbox=[0, 0.02, 1, 0.9],
        )
        table.auto_set_font_size(False)
        table.set_fontsize(font_size)

    # -------------------------------------------------------------------------
    def _render_histogram(
        self,
        ax: Any,
        histogram: Any,
        title: str,
        color: str,
    ) -> None:
        ax.set_title(title, fontsize=12, fontweight="bold")
        if not isinstance(histogram, dict):
            ax.axis("off")
            ax.text(
                0.5,
                0.5,
                "No histogram data",
                ha="center",
                va="center",
                color=MUTED_TEXT,
            )
            return

        bins = histogram.get("bins")
        counts = histogram.get("counts")
        if not isinstance(bins, list) or not isinstance(counts, list) or not counts:
            ax.axis("off")
            ax.text(
                0.5,
                0.5,
                "No histogram data",
                ha="center",
                va="center",
                color=MUTED_TEXT,
            )
            return

        x = list(range(len(counts)))
        numeric_counts = [self._to_number(value) for value in counts]
        labels = [str(value) for value in bins]
        if len(labels) > 14:
            step = max(1, len(labels) // 12)
            tick_positions = x[::step]
            tick_labels = labels[::step]
        else:
            tick_positions = x
            tick_labels = labels

        ax.bar(x, numeric_counts, color=color)
        ax.set_xticks(tick_positions, tick_labels)
        ax.tick_params(axis="x", rotation=45, labelsize=8)
        ax.grid(axis="y", alpha=0.25)

    # -------------------------------------------------------------------------
    def _parse_zipf_curve(self, value: Any) -> list[dict[str, float]]:
        parsed = self._parse_json_like(value)
        if not isinstance(parsed, list):
            return []
        points: list[dict[str, float]] = []
        for index, item in enumerate(parsed):
            if isinstance(item, list) and len(item) >= 2:
                rank = self._to_number(item[0], index + 1)
                freq = self._to_number(item[1], 0.0)
            elif isinstance(item, dict):
                rank = self._to_number(item.get("rank"), index + 1)
                freq = self._to_number(item.get("frequency") or item.get("count"), 0.0)
            else:
                continue
            if rank > 0 and freq > 0:
                points.append({"rank": rank, "frequency": freq})
        points.sort(key=lambda item: item["rank"])
        return points[:300]

    # -------------------------------------------------------------------------
    def _parse_word_frequency(self, value: Any) -> list[dict[str, int]]:
        if not isinstance(value, list):
            return []
        rows: list[dict[str, int]] = []
        for item in value:
            if not isinstance(item, dict):
                continue
            word = str(item.get("word") or item.get("token") or "").strip()
            if not word:
                continue
            count = int(round(max(0.0, self._to_number(item.get("count"), 0.0))))
            if count <= 0:
                continue
            rows.append({"word": word, "count": count})
        rows.sort(key=lambda item: (-item["count"], item["word"]))
        return rows

    # -------------------------------------------------------------------------
    def _parse_vocabulary_items(self, value: Any) -> list[dict[str, Any]]:
        if not isinstance(value, list):
            return []
        rows: list[dict[str, Any]] = []
        for item in value:
            if not isinstance(item, dict):
                continue
            rows.append(
                {
                    "token_id": int(round(self._to_number(item.get("token_id"), 0))),
                    "token": str(item.get("token") or ""),
                    "length": int(round(self._to_number(item.get("length"), 0))),
                }
            )
        return rows

    # -------------------------------------------------------------------------
    def _select_distribution(
        self, distributions: Any, selected_tokenizer: str
    ) -> dict[str, Any] | None:
        if not isinstance(distributions, list) or not distributions:
            return None
        if selected_tokenizer:
            for item in distributions:
                if (
                    isinstance(item, dict)
                    and str(item.get("tokenizer") or "") == selected_tokenizer
                ):
                    return item
        for item in distributions:
            if isinstance(item, dict):
                return item
        return None

    # -------------------------------------------------------------------------
    def _build_benchmark_metrics_table(
        self, global_metrics: list[Any]
    ) -> list[list[str]]:
        rows: list[list[str]] = []
        for item in global_metrics[:18]:
            if not isinstance(item, dict):
                continue
            rows.append(
                [
                    self._short_name(str(item.get("tokenizer") or "N/A")),
                    self._format_percent(item.get("word_recovery_rate")),
                    self._format_percent(item.get("character_coverage")),
                    self._format_number(item.get("subword_fertility"), 4),
                    self._format_number(item.get("token_distribution_entropy"), 4),
                ]
            )
        if not rows:
            return [["N/A", "N/A", "N/A", "N/A", "N/A"]]
        return rows

    # -------------------------------------------------------------------------
    def _extract_nested(self, payload: dict[str, Any], key: str) -> dict[str, Any]:
        candidate = payload.get(key)
        return candidate if isinstance(candidate, dict) else {}

    # -------------------------------------------------------------------------
    def _short_name(self, tokenizer_name: str, max_length: int = 24) -> str:
        trimmed = tokenizer_name.strip()
        if not trimmed:
            return "N/A"
        short = trimmed.split("/")[-1] or trimmed
        if len(short) <= max_length:
            return short
        return f"{short[: max(1, max_length - 3)]}..."

    # -------------------------------------------------------------------------
    def _to_number(self, value: Any, fallback: float = 0.0) -> float:
        if isinstance(value, (int, float)):
            return float(value)
        if isinstance(value, str):
            try:
                return float(value)
            except ValueError:
                return fallback
        return fallback

    # -------------------------------------------------------------------------
    def _format_count(self, value: Any) -> str:
        return f"{int(round(max(0.0, self._to_number(value, 0.0)))):,}"

    # -------------------------------------------------------------------------
    def _format_number(self, value: Any, decimals: int) -> str:
        return f"{self._to_number(value, 0.0):.{decimals}f}"

    # -------------------------------------------------------------------------
    def _format_percent(self, value: Any) -> str:
        numeric = self._to_number(value, 0.0)
        if numeric <= 1.0:
            numeric *= 100.0
        return f"{numeric:.2f}%"

    # -------------------------------------------------------------------------
    def _as_percent(self, value: Any) -> float:
        numeric = self._to_number(value, 0.0)
        if numeric <= 1.0:
            numeric *= 100.0
        return numeric

    # -------------------------------------------------------------------------
    def _parse_json_like(self, value: Any) -> Any:
        if not isinstance(value, str):
            return value
        candidate = value.strip()
        if not candidate:
            return None
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            return value
