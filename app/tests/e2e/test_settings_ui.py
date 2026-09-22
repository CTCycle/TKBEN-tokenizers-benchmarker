"""Browser coverage for the Settings page and runtime settings lifecycle."""

from __future__ import annotations

from dataclasses import dataclass
from math import ceil
from urllib.parse import parse_qs, urlparse
from uuid import uuid4

from playwright.sync_api import APIRequestContext, Locator, Page, expect
from server.configurations.settings import (
    BenchmarkSettings,
    DatasetSettings,
    JobsSettings,
    TokenizerSettings,
)

###############################################################################
Number = int | float
BYTES_PER_MIB = 1024 * 1024


@dataclass(frozen=True)
class InvalidSettingCase:
    value: Number
    expected_error: str


@dataclass(frozen=True)
class SettingFieldSpec:
    tab: str
    label: str
    control_name: str
    api_key: str
    integer: bool
    api_multiplier: int
    backend_minimum: Number
    backend_maximum: Number | None
    minimum: Number
    maximum: Number | None
    valid_boundary_values: tuple[Number, ...]
    invalid_cases: tuple[InvalidSettingCase, ...]
    restore_value: Number
    valid_fractional_value: Number | None
    persisted_value: Number


SETTING_MODELS = {
    "datasets": DatasetSettings,
    "tokenizers": TokenizerSettings,
    "benchmarks": BenchmarkSettings,
    "jobs": JobsSettings,
}


# The blueprint owns only rendered control metadata and the values used by the
# browser campaign. Numeric limits come from the backend Pydantic fields below.
_FIELD_BLUEPRINTS = (
    (
        "Data",
        "Histogram bins",
        "histogramBins",
        "datasets",
        "histogram_bins",
        True,
        1,
        25,
        None,
    ),
    (
        "Data",
        "Dataset upload limit (MiB)",
        "datasetMaxUploadMiB",
        "datasets",
        "max_upload_bytes",
        True,
        BYTES_PER_MIB,
        12,
        None,
    ),
    (
        "Data",
        "Dataset download timeout (seconds)",
        "downloadTimeoutSeconds",
        "datasets",
        "download_timeout_seconds",
        False,
        1,
        12.5,
        1.5,
    ),
    (
        "Data",
        "Download retry attempts",
        "downloadRetryAttempts",
        "datasets",
        "download_retry_attempts",
        True,
        1,
        4,
        None,
    ),
    (
        "Data",
        "Download retry backoff (seconds)",
        "downloadRetryBackoffSeconds",
        "datasets",
        "download_retry_backoff_seconds",
        False,
        1,
        2.5,
        0.5,
    ),
    (
        "Tokenizers",
        "Default discovery limit",
        "defaultDiscoveryLimit",
        "tokenizers",
        "default_discovery_limit",
        True,
        1,
        7,
        None,
    ),
    (
        "Tokenizers",
        "Maximum discovery limit",
        "maxDiscoveryLimit",
        "tokenizers",
        "max_discovery_limit",
        True,
        1,
        11,
        None,
    ),
    (
        "Tokenizers",
        "Discovery candidate cap",
        "maxDiscoveryCandidates",
        "tokenizers",
        "max_discovery_candidates",
        True,
        1,
        17,
        None,
    ),
    (
        "Tokenizers",
        "Metadata candidate multiplier",
        "metadataCandidateMultiplier",
        "tokenizers",
        "metadata_candidate_multiplier",
        True,
        1,
        4,
        None,
    ),
    (
        "Tokenizers",
        "Tokenizer upload limit (MiB)",
        "tokenizerMaxUploadMiB",
        "tokenizers",
        "max_upload_bytes",
        True,
        BYTES_PER_MIB,
        12,
        None,
    ),
    (
        "Benchmarks",
        "Default document cap",
        "benchmarkDefaultMaxDocuments",
        "benchmarks",
        "default_max_documents",
        True,
        1,
        1234,
        None,
    ),
    (
        "Benchmarks",
        "Default tokenizer batch size",
        "benchmarkDefaultBatchSize",
        "benchmarks",
        "default_batch_size",
        True,
        1,
        24,
        None,
    ),
    (
        "Benchmarks",
        "Default parallelism",
        "benchmarkDefaultParallelism",
        "benchmarks",
        "default_parallelism",
        True,
        1,
        2,
        None,
    ),
    (
        "Benchmarks",
        "Benchmark streaming batch size",
        "benchmarkStreamingBatchSize",
        "benchmarks",
        "streaming_batch_size",
        True,
        1,
        1100,
        None,
    ),
    (
        "Runtime",
        "Dataset streaming batch size",
        "datasetStreamingBatchSize",
        "datasets",
        "streaming_batch_size",
        True,
        1,
        11000,
        None,
    ),
    (
        "Runtime",
        "Job polling interval (seconds)",
        "jobPollingInterval",
        "jobs",
        "polling_interval",
        False,
        1,
        1.25,
        0.75,
    ),
)


def _backend_bound(group: str, field: str, attribute: str) -> Number | None:
    field_info = SETTING_MODELS[group].model_fields[field]
    for constraint in field_info.metadata:
        value = getattr(constraint, attribute, None)
        if value is not None:
            return value
    return None


def _number_text(value: Number) -> str:
    return f"{value:g}" if isinstance(value, float) else str(value)


def _below_bound(value: Number) -> Number:
    if isinstance(value, int):
        return value - 1
    return round(value - 0.01, 2)


def _above_bound(value: Number) -> Number:
    if isinstance(value, int):
        return value + 1
    return round(value + 0.01, 2)


def _alternate_setting_value(group: str, field: str, current: Number) -> Number:
    minimum = _backend_bound(group, field, "ge")
    maximum = _backend_bound(group, field, "le")
    step = 1 if isinstance(current, int) else 0.25
    candidate = current + step
    if maximum is not None and candidate > maximum:
        candidate = current - step
    assert minimum is not None and candidate >= minimum
    assert maximum is None or candidate <= maximum
    return candidate


def _build_setting_specs() -> tuple[SettingFieldSpec, ...]:
    specs: list[SettingFieldSpec] = []
    for (
        tab,
        label,
        control_name,
        group,
        field,
        integer,
        api_multiplier,
        persisted_value,
        valid_fractional_value,
    ) in _FIELD_BLUEPRINTS:
        backend_minimum = _backend_bound(group, field, "ge")
        backend_maximum = _backend_bound(group, field, "le")
        assert backend_minimum is not None, (
            f"Missing backend minimum for {group}.{field}"
        )
        minimum = (
            ceil(backend_minimum / api_multiplier)
            if api_multiplier != 1
            else backend_minimum
        )
        maximum = (
            backend_maximum / api_multiplier
            if backend_maximum is not None and api_multiplier != 1
            else backend_maximum
        )
        invalid_cases = [
            InvalidSettingCase(
                _below_bound(minimum),
                f"Must be at least {_number_text(minimum)}.",
            ),
        ]
        if maximum is not None:
            expected_maximum_error = (
                "Must not exceed Maximum discovery limit."
                if group == "tokenizers" and field == "default_discovery_limit"
                else f"Must be no more than {_number_text(maximum)}."
            )
            invalid_cases.append(
                InvalidSettingCase(
                    _above_bound(maximum),
                    expected_maximum_error,
                )
            )
        if integer:
            invalid_cases.append(
                InvalidSettingCase(minimum + 0.5, "Use a whole number.")
            )
        boundary_values = (minimum,)
        if maximum is not None:
            boundary_values += (maximum,)
        specs.append(
            SettingFieldSpec(
                tab=tab,
                label=label,
                control_name=control_name,
                api_key=f"{group}.{field}",
                integer=integer,
                api_multiplier=api_multiplier,
                backend_minimum=backend_minimum,
                backend_maximum=backend_maximum,
                minimum=minimum,
                maximum=maximum,
                valid_boundary_values=boundary_values,
                invalid_cases=tuple(invalid_cases),
                restore_value=boundary_values[0],
                valid_fractional_value=valid_fractional_value,
                persisted_value=persisted_value,
            )
        )
    return tuple(specs)


SETTINGS_FIELDS = _build_setting_specs()
ALL_RUNTIME_SETTING_KEYS = tuple(spec.api_key for spec in SETTINGS_FIELDS)
TOKENIZER_RELATION_KEYS = frozenset(
    {
        "tokenizers.default_discovery_limit",
        "tokenizers.max_discovery_limit",
        "tokenizers.max_discovery_candidates",
    }
)
assert len(SETTINGS_FIELDS) == 16
assert set(ALL_RUNTIME_SETTING_KEYS) == {
    "datasets.histogram_bins",
    "datasets.max_upload_bytes",
    "datasets.download_timeout_seconds",
    "datasets.download_retry_attempts",
    "datasets.download_retry_backoff_seconds",
    "tokenizers.default_discovery_limit",
    "tokenizers.max_discovery_limit",
    "tokenizers.max_discovery_candidates",
    "tokenizers.metadata_candidate_multiplier",
    "tokenizers.max_upload_bytes",
    "benchmarks.default_max_documents",
    "benchmarks.default_batch_size",
    "benchmarks.default_parallelism",
    "benchmarks.streaming_batch_size",
    "datasets.streaming_batch_size",
    "jobs.polling_interval",
}


def _input_text(value: Number) -> str:
    return _number_text(value)


def _fill_control(control: Locator, value: Number) -> None:
    control.fill(_input_text(value))
    control.press("Tab")


def _api_value(spec: SettingFieldSpec, value: Number) -> Number:
    if spec.api_multiplier == 1:
        return value
    return int(value * spec.api_multiplier)


def _set_tokenizer_values(
    page: Page,
    *,
    default_limit: Number,
    maximum_limit: Number,
    candidate_cap: Number,
) -> None:
    page.get_by_role("tab", name="Tokenizers").click()
    _fill_control(page.get_by_label("Default discovery limit"), default_limit)
    _fill_control(page.get_by_label("Maximum discovery limit"), maximum_limit)
    _fill_control(page.get_by_label("Discovery candidate cap"), candidate_cap)


def _set_tokenizer_boundary_context(
    page: Page,
    spec: SettingFieldSpec,
    value: Number,
) -> None:
    if spec.api_key == "tokenizers.default_discovery_limit":
        maximum_limit = max(value, 11)
        candidate_cap = max(maximum_limit, 17)
        _set_tokenizer_values(
            page,
            default_limit=value,
            maximum_limit=maximum_limit,
            candidate_cap=candidate_cap,
        )
    elif spec.api_key == "tokenizers.max_discovery_limit":
        _set_tokenizer_values(
            page,
            default_limit=min(value, 7),
            maximum_limit=value,
            candidate_cap=max(value, 17),
        )
    elif spec.api_key == "tokenizers.max_discovery_candidates":
        _set_tokenizer_values(
            page,
            default_limit=1,
            maximum_limit=1,
            candidate_cap=value,
        )
    else:
        _set_tokenizer_values(
            page,
            default_limit=7,
            maximum_limit=11,
            candidate_cap=17,
        )


def _set_tokenizer_invalid_context(
    page: Page,
    spec: SettingFieldSpec,
    value: Number,
) -> None:
    if (
        spec.api_key == "tokenizers.default_discovery_limit"
        and spec.maximum is not None
        and value > spec.maximum
    ):
        _set_tokenizer_values(
            page,
            default_limit=7,
            maximum_limit=250,
            candidate_cap=250,
        )
        return
    _set_tokenizer_values(
        page,
        default_limit=7,
        maximum_limit=11,
        candidate_cap=17,
    )


def _assert_restored_runtime_settings(
    api_context: APIRequestContext,
    original: dict,
) -> None:
    response = api_context.get("/api/settings")
    assert response.ok, response.text()
    restored = response.json()
    assert restored["settings"] == original["settings"]
    assert restored["overridden_keys"] == original.get("overridden_keys", [])


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
def test_settings_all_field_boundaries_and_persistence(
    page: Page,
    base_url: str,
    api_context: APIRequestContext,
) -> None:
    """Every rendered runtime setting honors its backend bounds and persists."""
    original_response = api_context.get("/api/settings")
    assert original_response.ok, original_response.text()
    original = original_response.json()
    save_button = None

    try:
        page.goto(f"{base_url}/settings")
        expect(page.locator(".settings-form")).to_be_visible()
        save_button = page.get_by_role("button", name="Save changes")

        for spec in SETTINGS_FIELDS:
            page.get_by_role("tab", name=spec.tab).click()
            control = page.get_by_label(spec.label)

            for invalid_case in spec.invalid_cases:
                if spec.tab == "Tokenizers":
                    _set_tokenizer_invalid_context(page, spec, invalid_case.value)
                    control = page.get_by_label(spec.label)
                _fill_control(control, invalid_case.value)
                expect(control).to_have_attribute("aria-invalid", "true")
                error = page.locator(f"#settings-error-{spec.control_name}")
                expect(error).to_be_visible()
                expect(error).to_have_text(invalid_case.expected_error)
                expect(save_button).to_be_disabled()

                if spec.tab == "Tokenizers":
                    _set_tokenizer_values(
                        page,
                        default_limit=7,
                        maximum_limit=11,
                        candidate_cap=17,
                    )
                    control = page.get_by_label(spec.label)
                    if spec.api_key not in TOKENIZER_RELATION_KEYS:
                        _fill_control(control, spec.restore_value)
                else:
                    _fill_control(control, spec.restore_value)
                expect(control).not_to_have_attribute("aria-invalid", "true")
                expect(error).not_to_be_visible()
                expect(save_button).to_be_enabled()

            for boundary_value in spec.valid_boundary_values:
                if spec.tab == "Tokenizers":
                    _set_tokenizer_boundary_context(page, spec, boundary_value)
                    control = page.get_by_label(spec.label)
                    if spec.api_key not in TOKENIZER_RELATION_KEYS:
                        _fill_control(control, boundary_value)
                else:
                    _fill_control(control, boundary_value)
                expect(control).not_to_have_attribute("aria-invalid", "true")
                expect(save_button).to_be_enabled()

            if spec.valid_fractional_value is not None:
                _fill_control(control, spec.valid_fractional_value)
                expect(control).not_to_have_attribute("aria-invalid", "true")
                expect(save_button).to_be_enabled()

            if spec.tab == "Tokenizers":
                _set_tokenizer_values(
                    page,
                    default_limit=7,
                    maximum_limit=11,
                    candidate_cap=17,
                )
                if spec.api_key not in TOKENIZER_RELATION_KEYS:
                    _fill_control(page.get_by_label(spec.label), spec.restore_value)
            else:
                _fill_control(control, spec.restore_value)

        _set_tokenizer_values(
            page,
            default_limit=12,
            maximum_limit=11,
            candidate_cap=17,
        )
        expect(page.get_by_label("Default discovery limit")).to_have_attribute(
            "aria-invalid", "true"
        )
        expect(page.get_by_label("Maximum discovery limit")).to_have_attribute(
            "aria-invalid", "true"
        )
        expect(page.locator("#settings-error-defaultDiscoveryLimit")).to_have_text(
            "Must not exceed Maximum discovery limit."
        )
        expect(page.locator("#settings-error-maxDiscoveryLimit")).to_have_text(
            "Must be at least Default discovery limit."
        )
        expect(page.locator(".settings-inline-error")).to_have_text(
            "Default discovery limit must not exceed the maximum, and the maximum must not exceed the candidate cap."
        )
        expect(save_button).to_be_disabled()

        _set_tokenizer_values(
            page,
            default_limit=7,
            maximum_limit=18,
            candidate_cap=17,
        )
        expect(page.get_by_label("Maximum discovery limit")).to_have_attribute(
            "aria-invalid", "true"
        )
        expect(page.get_by_label("Discovery candidate cap")).to_have_attribute(
            "aria-invalid", "true"
        )
        expect(page.locator("#settings-error-maxDiscoveryLimit")).to_have_text(
            "Must not exceed Discovery candidate cap."
        )
        expect(page.locator("#settings-error-maxDiscoveryCandidates")).to_have_text(
            "Must be at least Maximum discovery limit."
        )
        expect(page.locator(".settings-inline-error")).to_be_visible()
        expect(save_button).to_be_disabled()

        _set_tokenizer_values(
            page,
            default_limit=18,
            maximum_limit=12,
            candidate_cap=11,
        )
        expect(page.get_by_label("Default discovery limit")).to_have_attribute(
            "aria-invalid", "true"
        )
        expect(page.get_by_label("Maximum discovery limit")).to_have_attribute(
            "aria-invalid", "true"
        )
        expect(page.get_by_label("Discovery candidate cap")).to_have_attribute(
            "aria-invalid", "true"
        )
        expect(page.locator("#settings-error-defaultDiscoveryLimit")).to_have_text(
            "Must not exceed Maximum discovery limit."
        )
        expect(page.locator("#settings-error-maxDiscoveryLimit")).to_have_text(
            "Must be at least Default discovery limit and no more than Discovery candidate cap."
        )
        expect(page.locator("#settings-error-maxDiscoveryCandidates")).to_have_text(
            "Must be at least Maximum discovery limit."
        )
        expect(page.locator(".settings-inline-error")).to_be_visible()
        expect(save_button).to_be_disabled()

        _set_tokenizer_values(
            page,
            default_limit=7,
            maximum_limit=11,
            candidate_cap=17,
        )
        expect(save_button).to_be_enabled()

        for tab in ("Data", "Tokenizers", "Benchmarks", "Runtime"):
            page.get_by_role("tab", name=tab).click()
            for spec in SETTINGS_FIELDS:
                if spec.tab == tab:
                    _fill_control(page.get_by_label(spec.label), spec.persisted_value)

        expected_settings: dict[str, dict[str, Number]] = {}
        for spec in SETTINGS_FIELDS:
            group, field = spec.api_key.split(".", maxsplit=1)
            expected_value = _api_value(spec, spec.persisted_value)
            expected_settings.setdefault(group, {})[field] = expected_value
            assert expected_value != original["defaults"][group][field]

        with page.expect_response(
            lambda response: (
                response.request.method == "PATCH"
                and response.url.endswith("/api/settings")
                and response.ok
            )
        ) as save_response:
            save_button.click()
        saved = save_response.value.json()
        assert saved["revision"] == original["revision"] + 1
        assert set(saved["overridden_keys"]) == set(ALL_RUNTIME_SETTING_KEYS)
        for group, fields in expected_settings.items():
            for field, expected_value in fields.items():
                assert saved["settings"][group][field] == expected_value

        persisted_response = api_context.get("/api/settings")
        assert persisted_response.ok, persisted_response.text()
        persisted = persisted_response.json()
        assert persisted["revision"] == saved["revision"]
        assert persisted["settings"] == saved["settings"]
        assert set(persisted["overridden_keys"]) == set(ALL_RUNTIME_SETTING_KEYS)

        page.reload()
        expect(page.locator(".settings-form")).to_be_visible()
        for tab in ("Data", "Tokenizers", "Benchmarks", "Runtime"):
            page.get_by_role("tab", name=tab).click()
            for spec in SETTINGS_FIELDS:
                if spec.tab == tab:
                    expect(page.get_by_label(spec.label)).to_have_value(
                        _input_text(spec.persisted_value)
                    )
    finally:
        _restore_runtime_settings(api_context, original)
        _assert_restored_runtime_settings(api_context, original)


###############################################################################
def test_settings_page_round_trip_and_runtime_effect(
    page: Page,
    base_url: str,
    api_context: APIRequestContext,
    job_waiter,
    tiny_tokenizer_json: bytes,
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
    target_benchmark_batch = _alternate_setting_value(
        "benchmarks",
        "default_batch_size",
        defaults["benchmarks"]["default_batch_size"],
    )
    target_benchmark_parallelism = _alternate_setting_value(
        "benchmarks",
        "default_parallelism",
        defaults["benchmarks"]["default_parallelism"],
    )
    target_polling_interval = _alternate_setting_value(
        "jobs",
        "polling_interval",
        defaults["jobs"]["polling_interval"],
    )
    dataset_filename = f"settings_runtime_{uuid4().hex}.csv"
    dataset_name = f"custom/{dataset_filename.removesuffix('.csv')}"
    tokenizer_filename = f"settings_runtime_{uuid4().hex}.json"
    dataset_created = False
    tokenizer_name: str | None = None

    try:
        page.goto(f"{base_url}/dataset")
        settings_link = page.get_by_role("link", name="Open settings")
        expect(settings_link).to_be_visible()
        settings_link.click()
        expect(page).to_have_url(f"{base_url}/settings")
        expect(page.locator(".settings-panel")).to_have_count(0)
        expect(page.locator(".settings-sidebar")).to_be_visible()
        expect(page.get_by_role("tablist", name="Settings sections")).to_have_attribute(
            "aria-orientation", "vertical"
        )

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
        expect(histogram).to_have_attribute("aria-invalid", "true")
        expect(page.get_by_text("Must be at least 5.")).to_be_visible()
        histogram.fill(str(histogram_value))

        page.get_by_role("tab", name="Tokenizers").click()
        page.get_by_label("Default discovery limit").fill(str(target_default))
        page.get_by_label("Maximum discovery limit").fill(str(target_max))

        page.get_by_role("tab", name="Benchmarks").click()
        page.get_by_label("Default document cap").fill(str(target_benchmark_documents))
        page.get_by_label("Default tokenizer batch size").fill(
            str(target_benchmark_batch)
        )
        page.get_by_label("Default parallelism").fill(str(target_benchmark_parallelism))

        page.get_by_role("tab", name="Runtime").click()
        page.get_by_label("Job polling interval (seconds)").fill(
            str(target_polling_interval)
        )

        with page.expect_response(
            lambda response: (
                response.request.method == "PATCH"
                and response.url.endswith("/api/settings")
                and response.ok
            )
        ) as save_response:
            save_button.click()
        saved_settings = save_response.value.json()["settings"]
        assert saved_settings["datasets"]["histogram_bins"] == histogram_value
        assert (
            saved_settings["benchmarks"]["default_max_documents"]
            == target_benchmark_documents
        )
        assert (
            saved_settings["benchmarks"]["default_batch_size"] == target_benchmark_batch
        )
        assert (
            saved_settings["benchmarks"]["default_parallelism"]
            == target_benchmark_parallelism
        )
        assert saved_settings["jobs"]["polling_interval"] == target_polling_interval

        page.reload()
        expect(page.get_by_label("Histogram bins")).to_have_value(str(histogram_value))
        expect(page.get_by_label("Default discovery limit")).to_have_value(
            str(target_default)
        )
        expect(page.get_by_label("Maximum discovery limit")).to_have_value(
            str(target_max)
        )
        expect(page.get_by_label("Default document cap")).to_have_value(
            str(target_benchmark_documents)
        )
        expect(page.get_by_label("Default tokenizer batch size")).to_have_value(
            str(target_benchmark_batch)
        )
        expect(page.get_by_label("Default parallelism")).to_have_value(
            str(target_benchmark_parallelism)
        )
        expect(page.get_by_label("Job polling interval (seconds)")).to_have_value(
            str(target_polling_interval)
        )

        upload_response = api_context.post(
            "/api/datasets/upload",
            multipart={
                "file": {
                    "name": dataset_filename,
                    "mimeType": "text/csv",
                    "buffer": b"text\nhello there\nanother small document\n",
                }
            },
        )
        assert upload_response.status == 202, upload_response.text()
        upload_job = upload_response.json()
        dataset_created = True
        assert upload_job["poll_interval"] == target_polling_interval
        upload_status = job_waiter(
            upload_job["job_id"],
            poll_interval=upload_job["poll_interval"],
            timeout_seconds=300.0,
        )
        assert upload_status.get("status") == "completed", upload_status.get("error")
        upload_result = upload_status.get("result", {})
        assert upload_result.get("dataset_name") == dataset_name
        uploaded_histogram = upload_result.get("histogram", {})
        assert len(uploaded_histogram.get("bins", [])) == histogram_value

        tokenizer_upload = api_context.post(
            "/api/tokenizers/upload",
            multipart={
                "file": {
                    "name": tokenizer_filename,
                    "mimeType": "application/json",
                    "buffer": tiny_tokenizer_json,
                }
            },
        )
        assert tokenizer_upload.ok, tokenizer_upload.text()
        tokenizer_payload = tokenizer_upload.json()
        assert tokenizer_payload.get("is_compatible") is True
        tokenizer_name = tokenizer_payload["tokenizer_name"]

        page.get_by_role("button", name="Cross Benchmark").click()
        expect(page).to_have_url(f"{base_url}/cross-benchmark")
        run_benchmark = page.get_by_role("button", name="Run benchmark")
        expect(run_benchmark).to_be_enabled()
        run_benchmark.click()
        page.get_by_role("button", name="Next").click()
        expect(page.locator("#benchmark-documents")).to_have_value(
            str(target_benchmark_documents)
        )
        page.get_by_placeholder("Search tokenizers").fill(tokenizer_name)
        tokenizer_option = page.locator(".benchmark-wizard-tokenizer-option").filter(
            has_text=tokenizer_name
        )
        expect(tokenizer_option).to_be_visible()
        tokenizer_option.get_by_role("checkbox").check()
        page.get_by_label("Dataset").select_option(dataset_name)
        page.get_by_role("button", name="Next").click()
        expect(page.get_by_label("Batch size")).to_have_value(
            str(target_benchmark_batch)
        )
        expect(page.get_by_label("Parallelism")).to_have_value(
            str(target_benchmark_parallelism)
        )
        page.get_by_role("button", name="Close benchmark wizard").click()

        over_limit = api_context.get(f"/api/tokenizers/discover?limit={target_max + 1}")
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

        close_manager = page.get_by_role("button", name="Close tokenizer manager").last
        expect(close_manager).to_be_visible()
        close_manager.click()
        page.get_by_role("link", name="Open settings").click()
        expect(page).to_have_url(f"{base_url}/settings")
        histogram_row = page.locator(".settings-row").filter(has_text="Histogram bins")
        with page.expect_response(
            lambda response: (
                response.request.method == "POST"
                and response.url.endswith("/api/settings/reset")
                and response.ok
            )
        ):
            histogram_row.get_by_role("button", name="Reset").click()
        expect(page.get_by_label("Histogram bins")).to_have_value(
            str(histogram_default)
        )

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
            lambda response: (
                response.request.method == "PATCH"
                and response.url.endswith("/api/settings")
                and response.status == 409
            )
        ):
            save_button.click()
        expect(page.get_by_role("alert")).to_contain_text("Settings changed elsewhere")
        reload_button = page.get_by_role("button", name="Reload settings")
        reload_button.click()
        expect(page.get_by_label("Histogram bins")).to_have_value(
            str(histogram_default)
        )
    finally:
        try:
            if dataset_created:
                deleted = api_context.delete(
                    "/api/datasets/delete",
                    params={"dataset_name": dataset_name},
                )
                assert deleted.status == 200, deleted.text()
        finally:
            try:
                if tokenizer_name is not None:
                    deleted_tokenizer = api_context.delete(
                        "/api/tokenizers/delete",
                        params={"tokenizer_name": tokenizer_name},
                    )
                    assert deleted_tokenizer.status == 200, deleted_tokenizer.text()
            finally:
                _restore_runtime_settings(api_context, original)
