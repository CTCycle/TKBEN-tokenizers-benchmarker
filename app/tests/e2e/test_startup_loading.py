"""Browser coverage for frontend-first startup and readiness polling."""

import re

from playwright.sync_api import Page, expect


def _route_health(page: Page, failures: int | None) -> dict[str, int]:
    calls = {"count": 0}

    def handle_health(route) -> None:
        calls["count"] += 1
        if failures is None or calls["count"] <= failures:
            route.fulfill(status=503, json={"detail": "backend warming up"})
            return
        route.fulfill(json={"status": "ok"})

    page.route("**/api/health", handle_health)
    return calls


def test_startup_screen_recovers_after_transient_backend_unavailability(
    page: Page, base_url: str
) -> None:
    calls = _route_health(page, failures=2)

    page.goto(f"{base_url}/dataset")

    expect(
        page.get_by_role("heading", name="Preparing tokenizer benchmarks")
    ).to_be_visible()
    expect(page.locator(".startup-token")).to_have_count(6)
    expect(page.get_by_text("Dataset Usage")).to_be_visible(timeout=10_000)
    expect(page.locator(".startup-screen")).to_have_count(0)
    assert calls["count"] >= 3


def test_startup_screen_supports_reduced_motion_and_narrow_viewport(
    page: Page, base_url: str
) -> None:
    page.set_viewport_size({"width": 640, "height": 800})
    page.emulate_media(reduced_motion="reduce")
    _route_health(page, failures=None)

    page.goto(f"{base_url}/dataset")

    expect(
        page.get_by_role("heading", name="Preparing tokenizer benchmarks")
    ).to_be_visible()
    token_duration = page.locator(".startup-token").first.evaluate(
        "element => getComputedStyle(element).animationDuration"
    )
    assert float(token_duration.removesuffix("s")) <= 0.01
    assert page.evaluate("document.documentElement.scrollWidth <= window.innerWidth")
    expect(
        page.get_by_text(re.compile(r"Backend startup is taking a little longer"))
    ).to_be_visible(timeout=17_000)
