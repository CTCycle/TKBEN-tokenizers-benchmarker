"""
E2E tests for UI navigation and page rendering.
Targets datasets, tokenizers, and cross benchmark workflows.
"""
import re
from playwright.sync_api import Page, expect


class TestAppShell:
    """Tests for core layout and routing."""

    def test_root_redirects_to_dataset(self, page: Page, base_url: str) -> None:
        """The root route should redirect to the dataset page."""
        page.goto(base_url)
        expect(page).to_have_url(re.compile(r".*/dataset/?$"))
        expect(page.get_by_text("Dataset Usage")).to_be_visible()

    def test_sidebar_links_are_visible(self, page: Page, base_url: str) -> None:
        """Sidebar navigation should expose primary sections."""
        page.goto(f"{base_url}/dataset")
        expect(page.get_by_role("button", name="Datasets")).to_be_visible()
        expect(page.get_by_role("button", name="Tokenizers")).to_be_visible()
        expect(page.get_by_role("button", name="Cross Benchmark")).to_be_visible()

    def test_unknown_route_redirects_to_dataset(self, page: Page, base_url: str) -> None:
        """Unknown routes should redirect back to the dataset page."""
        page.goto(f"{base_url}/does-not-exist")
        expect(page).to_have_url(re.compile(r".*/dataset/?$"))


class TestDatasetPage:
    """Tests for dataset page UI elements."""

    def test_dataset_page_panels_render(self, page: Page, base_url: str) -> None:
        """Dataset page should show the main layout panels."""
        page.goto(f"{base_url}/dataset")
        expect(page.get_by_text("Dataset Usage")).to_be_visible()
        expect(page.get_by_text("Dataset Preview")).to_be_visible()
        expect(page.get_by_role("button", name="Add dataset")).to_be_visible()


class TestTokenizersPage:
    """Tests for tokenizers page UI elements."""

    def test_tokenizers_page_loads(self, page: Page, base_url: str) -> None:
        """Tokenizers page should render selection and report panels."""
        page.goto(f"{base_url}/tokenizers")
        expect(page.get_by_text("Tokenizer Selection")).to_be_visible()
        expect(page.get_by_text("Tokenizers Dashboard")).to_be_visible()


class TestCrossBenchmarkPage:
    """Tests for cross benchmark page UI elements."""

    def test_cross_benchmark_page_loads_controls(self, page: Page, base_url: str) -> None:
        """Cross benchmark page should render control panel and report selector."""
        page.goto(f"{base_url}/cross-benchmark")
        expect(page.get_by_text("Tokenizer Benchmark")).to_be_visible()
        expect(page.get_by_role("button", name="Start benchmark")).to_be_visible()
        expect(page.locator("#benchmark-report-selector")).to_be_visible()

    def test_cross_benchmark_wizard_navigation_and_validation(
        self,
        page: Page,
        base_url: str,
    ) -> None:
        """Wizard should open, navigate to step 2, and enforce required step-2 selections."""
        page.goto(f"{base_url}/cross-benchmark")
        page.get_by_role("button", name="Start benchmark").click()
        expect(page.get_by_text("1. Metrics")).to_be_visible()
        expect(page.get_by_text("2. Inputs")).to_be_visible()
        expect(page.get_by_text("3. Summary")).to_be_visible()
        page.get_by_role("button", name="Next").click()
        expect(page.get_by_role("button", name="Back")).to_be_enabled()
        expect(page.get_by_role("button", name="Next")).to_be_disabled()
