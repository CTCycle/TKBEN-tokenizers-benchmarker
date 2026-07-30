from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
import re
from typing import Any, cast, get_args

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from server.common.utils.security import contains_control_chars
from server.domain.exports import DashboardType
from server.domain.benchmarks import BenchmarkVisualizationKind
from server.services.dashboard_export_helpers import DashboardExportFormatting


SAFE_FILE_CHARS_PATTERN = re.compile(r"[^A-Za-z0-9._ ()-]+")
MAX_FILE_STEM_LENGTH = 120
MAX_PDF_BYTES = 80 * 1024 * 1024
ALLOWED_DASHBOARD_TYPES = set(get_args(DashboardType))

PRIMARY_COLOR = "#facc15"
SECONDARY_COLOR = "#38bdf8"
TERTIARY_COLOR = "#22c55e"
MUTED_TEXT = "#5b6472"
CHART_SERIES_COLORS = (SECONDARY_COLOR, TERTIARY_COLOR, PRIMARY_COLOR, "#f472b6", "#a78bfa", "#2dd4bf")

###############################################################################
@dataclass(frozen=True)
class DashboardPdfDocument:
    file_name: str
    page_count: int
    pdf_bytes: bytes

###############################################################################
class DashboardExportService(DashboardExportFormatting):

    # -------------------------------------------------------------------------
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
        return cast(DashboardType, value)

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
            aggregate.get("words.zipf_curve")
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
                bbox=cast(Any, [0, 0.02, 1, 0.9]),
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
        widgets = self._normalize_benchmark_dashboard_widgets(source, payload)
        return self._render_normalized_benchmark_widgets(pdf, report_name, source, widgets)

    # -------------------------------------------------------------------------
    def _normalize_benchmark_dashboard_widgets(
        self, source: dict[str, Any], payload: dict[str, Any]
    ) -> list[dict[str, Any]]:
        dashboard = source.get("dashboard")
        if not isinstance(dashboard, dict):
            return []
        raw_widgets = dashboard.get("widgets")
        if not isinstance(raw_widgets, list):
            return []
        widgets_by_id = {
            widget.get("widget_id"): widget
            for widget in raw_widgets
            if isinstance(widget, dict) and isinstance(widget.get("widget_id"), str)
        }
        visible_ids = payload.get("visible_widget_ids")
        ordered_ids = payload.get("ordered_widget_ids")
        visible = visible_ids if isinstance(visible_ids, list) else list(widgets_by_id)
        ordered = ordered_ids if isinstance(ordered_ids, list) else list(widgets_by_id)
        overrides = payload.get("visualization_by_widget_id", {})
        if not isinstance(overrides, dict):
            raise ValueError("visualization_by_widget_id must be an object.")
        unknown_override_ids = set(overrides) - set(widgets_by_id)
        if unknown_override_ids:
            raise ValueError(
                "visualization_by_widget_id contains unknown widget IDs: "
                + ", ".join(sorted(str(widget_id) for widget_id in unknown_override_ids))
                + "."
            )
        visible_set = {widget_id for widget_id in visible if isinstance(widget_id, str)}
        normalized: list[dict[str, Any]] = []
        seen: set[str] = set()
        for widget_id in ordered:
            if not isinstance(widget_id, str) or widget_id in seen or widget_id not in visible_set:
                continue
            widget = widgets_by_id.get(widget_id)
            if widget is not None:
                default = widget.get("default_visualization")
                compatible = widget.get("compatible_visualizations")
                if not isinstance(default, str) or not isinstance(compatible, list) or not all(isinstance(item, str) for item in compatible):
                    raise ValueError(f"Widget '{widget_id}' has an invalid visualization contract.")
                try:
                    selected = overrides.get(widget_id, default)
                    selected_kind = BenchmarkVisualizationKind(selected)
                except ValueError as exc:
                    raise ValueError(f"Widget '{widget_id}' has an unknown visualization override.") from exc
                if selected_kind.value not in compatible:
                    raise ValueError(f"Visualization '{selected_kind.value}' is incompatible with widget '{widget_id}'.")
                normalized.append({**widget, "visualization": selected_kind.value})
                seen.add(widget_id)
        return normalized

    # -------------------------------------------------------------------------
    def _render_normalized_benchmark_widgets(
        self, pdf: PdfPages, report_name: str, source: dict[str, Any], widgets: list[dict[str, Any]]
    ) -> int:
        if not widgets:
            fig = plt.figure(figsize=(11.69, 8.27), constrained_layout=True)
            ax = fig.add_subplot(111)
            ax.axis("off")
            ax.text(0.0, 0.92, "Benchmark Dashboard Report", fontsize=18, fontweight="bold", color="#111827")
            ax.text(0.0, 0.74, f"Report: {report_name or source.get('dataset_name') or 'N/A'}", fontsize=11, color=MUTED_TEXT)
            ax.text(0.5, 0.45, "No selected metric widgets are available for this export.", ha="center", va="center", color=MUTED_TEXT)
            pdf.savefig(fig)
            plt.close(fig)
            return 1

        pages: list[list[dict[str, Any]]] = []
        current: list[dict[str, Any]] = []
        for widget in widgets:
            if widget.get("width") == "wide":
                if current:
                    pages.append(current)
                pages.append([widget])
                current = []
            else:
                current.append(widget)
                if len(current) == 2:
                    pages.append(current)
                    current = []
        if current:
            pages.append(current)

        for page_index, page_widgets in enumerate(pages, start=1):
            fig = plt.figure(figsize=(11.69, 8.27))
            fig.text(
                0.06,
                0.94,
                "Benchmark Dashboard Report",
                fontsize=16,
                fontweight="bold",
                color="#111827",
            )
            fig.text(
                0.06,
                0.89,
                f"Report: {report_name or source.get('dataset_name') or 'N/A'} | Page {page_index} of {len(pages)}",
                fontsize=10,
                color=MUTED_TEXT,
            )
            if len(page_widgets) == 1:
                axes = [fig.add_axes([0.07, 0.1, 0.88, 0.66])]
            else:
                axes = [
                    fig.add_axes([0.07, 0.1, 0.4, 0.66]),
                    fig.add_axes([0.55, 0.1, 0.4, 0.66]),
                ]
            for axis, widget in zip(axes, page_widgets, strict=True):
                self._render_normalized_benchmark_widget(axis, widget)
            pdf.savefig(fig)
            plt.close(fig)
        return len(pages)

    # -------------------------------------------------------------------------
    def _render_normalized_benchmark_widget(self, ax: Any, widget: dict[str, Any]) -> None:
        label = str(widget.get("label") or "Metric")
        unit = str(widget.get("unit") or "")
        ax.set_title(
            f"{label} ({unit})" if unit else label,
            fontsize=12,
            fontweight="bold",
            loc="left",
            pad=10,
        )
        visualization = widget.get("visualization")
        if visualization == "box_plot":
            rows = [row for row in widget.get("distributions", []) if isinstance(row, dict)]
            if rows:
                labels = [self._short_name(str(row.get("tokenizer") or "")) for row in rows]
                stats = [{"label": label, "whislo": self._to_number(row.get("min")), "q1": self._to_number(row.get("q1")), "med": self._to_number(row.get("median")), "q3": self._to_number(row.get("q3")), "whishi": self._to_number(row.get("max")), "fliers": []} for label, row in zip(labels, rows, strict=True)]
                ax.bxp(stats, orientation="horizontal", showfliers=False, patch_artist=True, boxprops={"facecolor": SECONDARY_COLOR, "alpha": 0.75}, medianprops={"color": PRIMARY_COLOR, "linewidth": 2})
                minimum = min(stat["whislo"] for stat in stats)
                maximum = max(stat["whishi"] for stat in stats)
                if minimum > 0 and maximum / minimum >= 50:
                    ax.set_xscale("log")
                    ax.set_xlabel(f"{unit} (log scale)")
            else:
                ax.text(0.5, 0.5, "No distribution data", ha="center", va="center", color=MUTED_TEXT)
        elif visualization == "histogram":
            rows = [row for row in widget.get("histogram_bins", []) if isinstance(row, dict)]
            tokenizers = list(dict.fromkeys(str(row.get("tokenizer") or "") for row in rows))
            if rows:
                for index, tokenizer in enumerate(tokenizers):
                    tokenizer_rows = [row for row in rows if str(row.get("tokenizer") or "") == tokenizer]
                    lows = [self._to_number(row.get("bin_low")) for row in tokenizer_rows]
                    widths = [self._to_number(row.get("bin_high")) - low for row, low in zip(tokenizer_rows, lows, strict=True)]
                    counts = [self._to_number(row.get("count")) for row in tokenizer_rows]
                    ax.bar(lows, counts, width=widths, align="edge", alpha=0.35, label=self._short_name(tokenizer), color=CHART_SERIES_COLORS[index % len(CHART_SERIES_COLORS)])
                ax.legend(fontsize=8)
                ax.set_xlabel(unit)
                ax.set_ylabel("Count")
            else:
                ax.text(0.5, 0.5, "No histogram data", ha="center", va="center", color=MUTED_TEXT)
        elif visualization in {"grouped_bar", "heatmap"}:
            rows = [row for row in widget.get("buckets", []) if isinstance(row, dict)]
            tokenizers = list(dict.fromkeys(str(row.get("tokenizer") or "") for row in rows))
            buckets = list(dict.fromkeys(str(row.get("bucket") or "") for row in rows))
            lookup = {(str(row.get("tokenizer") or ""), str(row.get("bucket") or "")): self._to_number(row.get("value")) for row in rows}
            if rows and visualization == "grouped_bar":
                x = list(range(len(buckets)))
                width = 0.8 / max(len(tokenizers), 1)
                for index, tokenizer in enumerate(tokenizers):
                    ax.bar([item + (index - (len(tokenizers) - 1) / 2) * width for item in x], [lookup.get((tokenizer, bucket), 0.0) for bucket in buckets], width=width, label=self._short_name(tokenizer), color=CHART_SERIES_COLORS[index % len(CHART_SERIES_COLORS)])
                ax.set_xticks(x, [self._short_name(bucket) for bucket in buckets])
                ax.legend(fontsize=8)
            elif rows:
                matrix = [[lookup.get((tokenizer, bucket), float("nan")) for bucket in buckets] for tokenizer in tokenizers]
                image = ax.imshow(matrix, aspect="auto", cmap="viridis")
                ax.set_xticks(range(len(buckets)), [self._short_name(bucket) for bucket in buckets])
                ax.set_yticks(range(len(tokenizers)), [self._short_name(tokenizer) for tokenizer in tokenizers])
                ax.figure.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
            else:
                ax.text(0.5, 0.5, "No bucket data", ha="center", va="center", color=MUTED_TEXT)
        elif visualization == "horizontal_bar":
            rows = [row for row in widget.get("points", []) if isinstance(row, dict)]
            labels = [self._short_name(str(row.get("tokenizer") or "")) for row in rows]
            values = [self._to_number(row.get("value")) for row in rows]
            if rows:
                ax.barh(labels, values, color=SECONDARY_COLOR)
                ax.set_xlabel(unit)
            else:
                ax.text(0.5, 0.5, "No metric data", ha="center", va="center", color=MUTED_TEXT)
        elif visualization == "dot_whisker":
            rows = [row for row in widget.get("points", []) if isinstance(row, dict)]
            labels = [self._short_name(str(row.get("tokenizer") or "")) for row in rows]
            values = [self._to_number(row.get("value")) for row in rows]
            lows = [self._to_number(row.get("interval_low")) for row in rows]
            highs = [self._to_number(row.get("interval_high")) for row in rows]
            if rows:
                errors = [[max(0.0, value - low) for value, low in zip(values, lows, strict=True)], [max(0.0, high - value) for value, high in zip(values, highs, strict=True)]]
                ax.errorbar(values, labels, xerr=errors, fmt="o", color=PRIMARY_COLOR, ecolor=SECONDARY_COLOR, capsize=4)
            else:
                ax.text(0.5, 0.5, "No metric data", ha="center", va="center", color=MUTED_TEXT)
        else:
            rows = [row for row in widget.get("points", []) if isinstance(row, dict)]
            if rows:
                labels = [self._short_name(str(row.get("tokenizer") or "")) for row in rows]
                values = [self._to_number(row.get("value")) for row in rows]
                ax.bar(labels, values, color=SECONDARY_COLOR)
                if visualization == "interval_bar":
                    lows = [self._to_number(row.get("interval_low")) for row in rows]
                    highs = [self._to_number(row.get("interval_high")) for row in rows]
                    errors = [[max(0.0, value - low) for value, low in zip(values, lows, strict=True)], [max(0.0, high - value) for value, high in zip(values, highs, strict=True)]]
                    ax.errorbar(labels, values, yerr=errors, fmt="none", ecolor=PRIMARY_COLOR, capsize=3)
                ax.tick_params(axis="x", rotation=35, labelsize=8)
                ax.grid(axis="y", alpha=0.25)
            else:
                ax.text(0.5, 0.5, "No metric data", ha="center", va="center", color=MUTED_TEXT)
        ax.set_ylabel(unit)
        ax.set_axisbelow(True)

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
