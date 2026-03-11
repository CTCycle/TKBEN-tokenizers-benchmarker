from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from playwright.sync_api import Page, TimeoutError as PlaywrightTimeoutError, sync_playwright


ROOT = Path(__file__).resolve().parents[2]
ENV_PATH = ROOT / "TKBEN" / "settings" / ".env"
FIGURES_DIR = ROOT / "assets" / "figures"
MANIFEST_PATH = FIGURES_DIR / "manifest.json"


@dataclass
class CaptureConfig:
    name: str
    route: str
    file_base: str
    action: Callable[[Page, str], list[str]]
    force_segmented: bool = False
    capture_internal_scroll: bool = False


def parse_env_file(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    if not path.exists():
        return values

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key.strip()] = value.strip().strip('"').strip("'")
    return values


def wait_for_http(url: str, timeout_s: int = 120, interval_s: float = 1.0) -> bool:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        try:
            req = Request(url=url, method="GET")
            with urlopen(req, timeout=3) as response:
                if 200 <= response.status < 500:
                    return True
        except (HTTPError, URLError, TimeoutError):
            pass
        time.sleep(interval_s)
    return False


def goto_with_retry(page: Page, url: str, attempts: int = 3) -> None:
    last_error: Exception | None = None
    for attempt in range(1, attempts + 1):
        try:
            page.goto(url, wait_until="domcontentloaded", timeout=30000)
            wait_for_stable_render(page)
            return
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            if attempt < attempts:
                page.wait_for_timeout(900)
    if last_error:
        raise last_error


def wait_for_stable_render(page: Page) -> None:
    try:
        page.wait_for_load_state("networkidle", timeout=12000)
    except PlaywrightTimeoutError:
        pass
    try:
        page.wait_for_function(
            "() => !document.fonts || document.fonts.status === 'loaded'",
            timeout=5000,
        )
    except PlaywrightTimeoutError:
        pass
    page.add_style_tag(
        content=(
            "*,:before,:after{"
            "animation:none !important;"
            "transition:none !important;"
            "scroll-behavior:auto !important;"
            "}"
        )
    )
    page.wait_for_timeout(400)


def layout_metrics(page: Page) -> dict[str, float]:
    return page.evaluate(
        """
        () => {
          const doc = document.documentElement;
          const body = document.body;
          const docWidth = Math.max(
            doc.scrollWidth,
            doc.offsetWidth,
            doc.clientWidth,
            body ? body.scrollWidth : 0,
            body ? body.offsetWidth : 0
          );
          const docHeight = Math.max(
            doc.scrollHeight,
            doc.offsetHeight,
            doc.clientHeight,
            body ? body.scrollHeight : 0,
            body ? body.offsetHeight : 0
          );
          let maxScrollableWidth = 0;
          for (const el of Array.from(document.querySelectorAll('*'))) {
            const style = window.getComputedStyle(el);
            if (!/(auto|scroll)/.test(style.overflowX) && !/(auto|scroll)/.test(style.overflowY)) {
              continue;
            }
            maxScrollableWidth = Math.max(maxScrollableWidth, el.scrollWidth);
          }
          const tableWidth = Math.max(
            0,
            ...Array.from(document.querySelectorAll('table')).map((el) => el.scrollWidth)
          );
          return {
            docWidth,
            docHeight,
            clientWidth: doc.clientWidth,
            clientHeight: doc.clientHeight,
            overflowX: Math.max(0, docWidth - doc.clientWidth),
            maxScrollableWidth,
            tableWidth,
          };
        }
        """
    )


def adapt_viewport(page: Page) -> tuple[dict[str, int], list[str]]:
    notes: list[str] = []
    page.set_viewport_size({"width": 1440, "height": 960})
    page.wait_for_timeout(150)

    metrics = layout_metrics(page)
    doc_width = int(metrics["docWidth"])
    doc_height = int(metrics["docHeight"])
    target_width = max(1240, min(2200, doc_width + 80))
    target_width = max(
        target_width,
        min(2200, int(metrics["maxScrollableWidth"]) + 120),
        min(2200, int(metrics["tableWidth"]) + 120),
    )
    if doc_width < 1200:
        target_width = max(1240, min(1600, target_width))

    if doc_height <= 1100:
        target_height = max(860, min(1100, doc_height + 80))
    elif doc_height <= 1900:
        target_height = 1160
    else:
        target_height = 1300

    page.set_viewport_size({"width": target_width, "height": target_height})
    page.wait_for_timeout(250)

    for _ in range(2):
        adjusted = layout_metrics(page)
        overflow_x = int(adjusted["overflowX"])
        if overflow_x <= 20:
            break
        target_width = min(2200, target_width + max(120, overflow_x + 48))
        page.set_viewport_size({"width": target_width, "height": target_height})
        page.wait_for_timeout(200)
        notes.append(f"increased viewport width to reduce horizontal clipping (+{overflow_x}px overflow)")

    viewport = {"width": target_width, "height": target_height}
    notes.append(f"adaptive viewport selected ({viewport['width']}x{viewport['height']})")
    return viewport, notes


def capture_with_strategy(
    page: Page,
    file_base: str,
    force_segmented: bool = False,
    capture_internal_scroll: bool = False,
) -> tuple[list[str], dict[str, int], list[str]]:
    viewport, notes = adapt_viewport(page)
    metrics = layout_metrics(page)
    doc_height = int(metrics["docHeight"])

    files: list[str] = []
    if force_segmented or doc_height > 2500:
        step = max(500, viewport["height"] - 170)
        segment_count = max(2, math.ceil((doc_height - 60) / step))
        segment_count = min(segment_count, 4)
        notes.append(f"segmented capture used ({segment_count} parts) for long page ({doc_height}px)")
        for index in range(segment_count):
            y = min(index * step, max(0, doc_height - viewport["height"]))
            page.evaluate("(top) => window.scrollTo(0, top)", y)
            page.wait_for_timeout(280)
            filename = f"{file_base}-part{index + 1}.png"
            page.screenshot(path=str(FIGURES_DIR / filename), full_page=False)
            files.append(filename)
        page.evaluate("() => window.scrollTo(0, 0)")
    else:
        filename = f"{file_base}.png"
        page.screenshot(path=str(FIGURES_DIR / filename), full_page=True)
        files.append(filename)

    if capture_internal_scroll:
        scrolled = page.evaluate(
            """
            () => {
              let target = null;
              let score = 0;
              for (const el of Array.from(document.querySelectorAll('*'))) {
                const style = window.getComputedStyle(el);
                if (!/(auto|scroll)/.test(style.overflowY)) {
                  continue;
                }
                const delta = el.scrollHeight - el.clientHeight;
                if (delta < 180 || el.clientHeight < 180 || el.clientWidth < 300) {
                  continue;
                }
                const current = delta * Math.min(el.clientWidth, 2200);
                if (current > score) {
                  score = current;
                  target = el;
                }
              }
              if (!target) {
                return null;
              }
              target.scrollTop = Math.min(target.scrollHeight - target.clientHeight, Math.floor(target.clientHeight * 0.75));
              return {scrollHeight: target.scrollHeight, clientHeight: target.clientHeight};
            }
            """
        )
        if scrolled:
            page.wait_for_timeout(250)
            filename = f"{file_base}-internal-scroll.png"
            page.screenshot(path=str(FIGURES_DIR / filename), full_page=False)
            files.append(filename)
            notes.append("captured additional internal-scroll container state")

    return files, viewport, notes


def discover_routes(page: Page, base_url: str) -> list[str]:
    discovered: list[str] = []
    goto_with_retry(page, base_url)
    discovered.append(page.url)
    for label in ("Datasets", "Tokenizers", "Cross Benchmark"):
        try:
            page.get_by_role("button", name=label).click()
            wait_for_stable_render(page)
            discovered.append(page.url)
        except Exception:  # noqa: BLE001
            continue
    unique: list[str] = []
    for url in discovered:
        if url not in unique:
            unique.append(url)
    return unique


def open_dataset_page(page: Page, base_url: str) -> list[str]:
    notes: list[str] = []
    goto_with_retry(page, f"{base_url}/dataset")
    page.get_by_text("Dataset Usage").first.wait_for(state="visible", timeout=15000)
    return notes


def open_dataset_dashboard(page: Page, base_url: str) -> list[str]:
    notes = open_dataset_page(page, base_url)
    load_button = page.get_by_role("button", name="Load latest saved report").first
    if load_button.count() > 0:
        try:
            load_button.click(timeout=8000)
            wait_for_stable_render(page)
            notes.append("loaded latest persisted dataset report")
        except Exception:  # noqa: BLE001
            notes.append("dataset report load click failed; captured current dashboard state")
    else:
        notes.append("no report-load button found; captured default dashboard state")
    return notes


def open_dataset_list_modal(page: Page, base_url: str) -> list[str]:
    notes = open_dataset_page(page, base_url)
    page.get_by_role("button", name="Add dataset").click(timeout=8000)
    page.get_by_text("Predefined Datasets").first.wait_for(state="visible", timeout=15000)
    notes.append("captured dataset selector modal (list-style view)")
    return notes


def open_tokenizers_page(page: Page, base_url: str) -> list[str]:
    notes: list[str] = []
    goto_with_retry(page, f"{base_url}/tokenizers")
    page.get_by_text("Tokenizer Selection").first.wait_for(state="visible", timeout=15000)
    return notes


def open_tokenizer_detail(page: Page, base_url: str) -> list[str]:
    notes = open_tokenizers_page(page, base_url)
    open_report_button = page.locator("button[aria-label^='Generate or open tokenizer report for']").first
    if open_report_button.count() == 0:
        notes.append("no tokenizer detail action was available in the preview list")
        return notes

    try:
        open_report_button.click(timeout=8000)
        page.get_by_text("Latest loaded report:").first.wait_for(state="visible", timeout=60000)
        wait_for_stable_render(page)
        notes.append("opened tokenizer report detail from preview list")
    except Exception:  # noqa: BLE001
        notes.append("tokenizer report open timed out; captured fallback detail layout")
    return notes


def open_settings_modal(page: Page, base_url: str) -> list[str]:
    notes = open_dataset_page(page, base_url)
    page.get_by_role("button", name="Manage Hugging Face keys").click(timeout=8000)
    page.get_by_text("Hugging Face Keys").first.wait_for(state="visible", timeout=15000)
    notes.append("captured key manager settings popover")
    return notes


def open_cross_benchmark(page: Page, base_url: str) -> list[str]:
    notes: list[str] = []
    goto_with_retry(page, f"{base_url}/cross-benchmark")
    page.get_by_text("Tokenizer Benchmark Dashboard").first.wait_for(state="visible", timeout=20000)
    selector = page.locator("#benchmark-report-selector")
    if selector.count() > 0:
        try:
            option_count = selector.locator("option").count()
            if option_count > 1:
                selector.select_option(index=1)
                wait_for_stable_render(page)
                notes.append("selected first available benchmark report")
        except Exception:  # noqa: BLE001
            notes.append("kept current benchmark report selection")
    return notes


def main() -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    env = parse_env_file(ENV_PATH)
    ui_host = env.get("UI_HOST", "127.0.0.1")
    ui_port = env.get("UI_PORT", "8000")
    api_host = env.get("FASTAPI_HOST", "127.0.0.1")
    api_port = env.get("FASTAPI_PORT", "5000")
    base_url = f"http://{ui_host}:{ui_port}"
    api_url = f"http://{api_host}:{api_port}"

    if not wait_for_http(base_url, timeout_s=120):
        raise RuntimeError(f"Frontend is not reachable at {base_url}")
    if not wait_for_http(f"{api_url}/docs", timeout_s=120):
        raise RuntimeError(f"Backend is not reachable at {api_url}")

    capture_plan = [
        CaptureConfig("Home", "/", "home", open_dataset_page),
        CaptureConfig("Dataset Dashboard", "/dataset", "dashboard", open_dataset_dashboard),
        CaptureConfig("Datasets List", "/dataset", "datasets-list", open_dataset_list_modal),
        CaptureConfig("Tokenizers List", "/tokenizers", "tokenizers-list", open_tokenizers_page),
        CaptureConfig(
            "Tokenizer Detail",
            "/tokenizers",
            "tokenizer-detail",
            open_tokenizer_detail,
            capture_internal_scroll=True,
        ),
        CaptureConfig("Settings", "/dataset", "settings", open_settings_modal),
        CaptureConfig(
            "Reports Overview",
            "/cross-benchmark",
            "reports-overview",
            open_cross_benchmark,
            force_segmented=True,
        ),
    ]

    manifest: dict[str, object] = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "base_url": base_url,
        "api_url": api_url,
        "discovered_routes": [],
        "captures": [],
        "failures": [],
    }

    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=True)
        context = browser.new_context(
            viewport={"width": 1440, "height": 960},
            reduced_motion="reduce",
        )
        page = context.new_page()
        page.set_default_timeout(20000)

        discovered = discover_routes(page, base_url)
        manifest["discovered_routes"] = discovered

        for capture in capture_plan:
            errors: list[str] = []
            completed = False
            for attempt in range(1, 4):
                try:
                    action_notes = capture.action(page, base_url)
                    files, viewport, capture_notes = capture_with_strategy(
                        page=page,
                        file_base=capture.file_base,
                        force_segmented=capture.force_segmented,
                        capture_internal_scroll=capture.capture_internal_scroll,
                    )
                    entry = {
                        "name": capture.name,
                        "page_title": page.title(),
                        "route": capture.route,
                        "url": page.url,
                        "files": [
                            {
                                "filename": filename,
                                "viewport": viewport,
                            }
                            for filename in files
                        ],
                        "notes": action_notes + capture_notes + ["no authentication required"],
                    }
                    captures = manifest["captures"]
                    assert isinstance(captures, list)
                    captures.append(entry)
                    completed = True
                    break
                except Exception as exc:  # noqa: BLE001
                    errors.append(f"attempt {attempt}: {exc}")
                    page.wait_for_timeout(900)
            if not completed:
                failures = manifest["failures"]
                assert isinstance(failures, list)
                failures.append(
                    {
                        "name": capture.name,
                        "route": capture.route,
                        "reason": "; ".join(errors),
                    }
                )

        context.close()
        browser.close()

    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Wrote manifest: {MANIFEST_PATH}")


if __name__ == "__main__":
    main()
