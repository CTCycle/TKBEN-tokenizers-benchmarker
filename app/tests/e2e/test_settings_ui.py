"""Browser coverage for the Settings page and runtime settings lifecycle."""

from __future__ import annotations

from urllib.parse import parse_qs, urlparse

from playwright.sync_api import APIRequestContext, Page, expect

###############################################################################
def _restore_runtime_settings(
    api_context: APIRequestContext,
    original: dict,
) -> None:
    current_response = api_context.get("/api/settings")
    assert current_response.ok, current_response.text()
    current = current_response.json()

    reset_response = api_context.post(
        "/api/settings/reset",
        data={"expected_revision": current["revision"], "all": True},
    )
    assert reset_response.ok, reset_response.text()

    overridden_keys = original.get("overridden_keys", [])
    if not overridden_keys:
        return

    patch: dict = {"expected_revision": reset_response.json()["revision"]}
    original_settings = original["settings"]
    for key in overridden_keys:
        group, field = key.split(".", maxsplit=1)
        patch.setdefault(group, {})[field] = original_settings[group][field]

    restore_response = api_context.patch("/api/settings", data=patch)
    assert restore_response.ok, restore_response.text()

###############################################################################
def test_settings_page_round_trip_and_runtime_effect(
    page: Page,
    base_url: str,
    api_context: APIRequestContext,
) -> None:
    """Settings are typed, persistent, revisioned, and applied to new workflows."""
    original_response = api_context.get("/api/settings")
    assert original_response.ok, original_response.text()
    original = original_response.json()
    defaults = original["defaults"]
    histogram_default = defaults["datasets"]["histogram_bins"]
    histogram_value = histogram_default + 1 if histogram_default < 100 else 99
    candidate_cap = original["settings"]["tokenizers"]["max_discovery_candidates"]
    target_max = min(candidate_cap, 9)
    target_default = min(target_max, 7)
    benchmark_document_default = defaults["benchmarks"]["default_max_documents"]
    target_benchmark_documents = 1234 if benchmark_document_default != 1234 else 1235

    try:
        page.goto(f"{base_url}/dataset")
        settings_link = page.get_by_role("link", name="Open settings")
        expect(settings_link).to_be_visible()
        settings_link.click()
        expect(page).to_have_url(f"{base_url}/settings")

        for tab in ("Data", "Tokenizers", "Benchmarks", "Runtime"):
            expect(page.get_by_role("tab", name=tab)).to_be_visible()

        expected_labels_by_tab = {
            "Data": [
                "Histogram bins",
                "Dataset upload limit (MiB)",
                "Dataset download timeout (seconds)",
                "Download retry attempts",
                "Download retry backoff (seconds)",
            ],
            "Tokenizers": [
                "Default discovery limit",
                "Maximum discovery limit",
                "Discovery candidate cap",
                "Metadata candidate multiplier",
                "Tokenizer upload limit (MiB)",
            ],
            "Benchmarks": [
                "Default document cap",
                "Default tokenizer batch size",
                "Default parallelism",
                "Benchmark streaming batch size",
            ],
            "Runtime": [
                "Dataset streaming batch size",
                "Job polling interval (seconds)",
            ],
        }
        for tab_name, labels in expected_labels_by_tab.items():
            page.get_by_role("tab", name=tab_name).click()
            for label in labels:
                expect(page.get_by_label(label)).to_be_visible()

        page.get_by_role("tab", name="Data").click()
        save_button = page.get_by_role("button", name="Save changes")
        histogram = page.get_by_label("Histogram bins")
        histogram.fill("4")
        expect(save_button).to_be_disabled()
        histogram.fill(str(histogram_value))

        page.get_by_role("tab", name="Tokenizers").click()
        page.get_by_label("Default discovery limit").fill(str(target_default))
        page.get_by_label("Maximum discovery limit").fill(str(target_max))

        page.get_by_role("tab", name="Benchmarks").click()
        page.get_by_label("Default document cap").fill(str(target_benchmark_documents))

        with page.expect_response(
            lambda response: response.request.method == "PATCH"
            and response.url.endswith("/api/settings")
            and response.ok
        ) as save_response:
            save_button.click()
        saved_settings = save_response.value.json()["settings"]
        assert saved_settings["datasets"]["histogram_bins"] == histogram_value
        assert saved_settings["benchmarks"]["default_max_documents"] == target_benchmark_documents

        page.reload()
        expect(page.get_by_label("Histogram bins")).to_have_value(str(histogram_value))
        expect(page.get_by_label("Default discovery limit")).to_have_value(str(target_default))
        expect(page.get_by_label("Maximum discovery limit")).to_have_value(str(target_max))
        expect(page.get_by_label("Default document cap")).to_have_value(str(target_benchmark_documents))

        page.get_by_role("button", name="Cross Benchmark").click()
        expect(page).to_have_url(f"{base_url}/cross-benchmark")
        run_benchmark = page.get_by_role("button", name="Run benchmark")
        expect(run_benchmark).to_be_enabled()
        run_benchmark.click()
        page.get_by_role("button", name="Next").click()
        expect(page.locator("#benchmark-documents")).to_have_value(str(target_benchmark_documents))
        page.get_by_role("button", name="Close benchmark wizard").click()

        over_limit = api_context.get(
            f"/api/tokenizers/discover?limit={target_max + 1}"
        )
        assert over_limit.status == 422

        page.get_by_role("button", name="Tokenizers").click()
        expect(page).to_have_url(f"{base_url}/tokenizers")
        add_tokenizer = page.get_by_role("button", name="Add tokenizer")
        expect(add_tokenizer).to_be_visible()
        with page.expect_request(
            lambda request: "/api/tokenizers/discover" in request.url
        ) as discovery_request:
            add_tokenizer.click()
        query = parse_qs(urlparse(discovery_request.value.url).query)
        assert query["limit"] == [str(target_default)]

        close_manager = page.get_by_role(
            "button", name="Close tokenizer manager"
        ).last
        expect(close_manager).to_be_visible()
        close_manager.click()
        page.get_by_role("link", name="Open settings").click()
        expect(page).to_have_url(f"{base_url}/settings")
        histogram_row = page.locator(".settings-row").filter(has_text="Histogram bins")
        with page.expect_response(
            lambda response: response.request.method == "POST"
            and response.url.endswith("/api/settings/reset")
            and response.ok
        ):
            histogram_row.get_by_role("button", name="Reset").click()
        expect(page.get_by_label("Histogram bins")).to_have_value(str(histogram_default))

        conflict_value = histogram_default + 1 if histogram_default < 100 else 99
        page.get_by_label("Histogram bins").fill(str(conflict_value))
        current_response = api_context.get("/api/settings")
        assert current_response.ok, current_response.text()
        current = current_response.json()
        polling = current["settings"]["jobs"]["polling_interval"]
        external_polling = 2.0 if polling != 2.0 else 3.0
        external_response = api_context.patch(
            "/api/settings",
            data={
                "expected_revision": current["revision"],
                "jobs": {"polling_interval": external_polling},
            },
        )
        assert external_response.ok, external_response.text()

        with page.expect_response(
            lambda response: response.request.method == "PATCH"
            and response.url.endswith("/api/settings")
            and response.status == 409
        ):
            save_button.click()
        expect(page.get_by_role("alert")).to_contain_text("Settings changed elsewhere")
        reload_button = page.get_by_role("button", name="Reload settings")
        reload_button.click()
        expect(page.get_by_label("Histogram bins")).to_have_value(str(histogram_default))
    finally:
        _restore_runtime_settings(api_context, original)
