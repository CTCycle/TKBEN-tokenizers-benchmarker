"""
E2E tests for UI navigation and page rendering.
Targets the dataset, tokenizers, and database browser pages.
"""
import re
from playwright.sync_api import Page, expect


class TestAppShell:
    """Tests for core layout and routing."""

    def test_root_redirects_to_dataset(self, page: Page, base_url: str):
        """The root route should redirect to the dataset page."""
        page.goto(base_url)
        expect(page).to_have_url(re.compile(r".*/dataset/?$"))
        expect(page.get_by_role("heading", name="TKBEN Dashboard")).to_be_visible()

    def test_sidebar_links_are_visible(self, page: Page, base_url: str):
        """Sidebar navigation should expose primary sections."""
        page.goto(f"{base_url}/dataset")
        expect(page.get_by_role("link", name="Datasets")).to_be_visible()
        expect(page.get_by_role("link", name="Tokenizers")).to_be_visible()
        expect(page.get_by_role("link", name="Database Browser")).to_be_visible()

    def test_unknown_route_redirects_to_dataset(self, page: Page, base_url: str):
        """Unknown routes should redirect back to the dataset page."""
        page.goto(f"{base_url}/does-not-exist")
        expect(page).to_have_url(re.compile(r".*/dataset/?$"))


class TestDatasetPage:
    """Tests for dataset page UI elements."""

    def test_dataset_page_panels_render(self, page: Page, base_url: str):
        """Dataset page should show the main panels and inputs."""
        page.goto(f"{base_url}/dataset")
        expect(page.get_by_text("Download dataset")).to_be_visible()
        expect(page.get_by_text("Analysis Tools")).to_be_visible()
        expect(page.get_by_text("Dataset overview")).to_be_visible()
        expect(page.locator("#corpus-input")).to_be_visible()
        expect(page.locator("#config-input")).to_be_visible()
        expect(page.locator("#analysis-dataset-select")).to_be_visible()


class TestTokenizersPage:
    """Tests for tokenizers page UI elements."""

    def test_tokenizers_page_loads(self, page: Page, base_url: str):
        """Tokenizers page should render core controls."""
        page.goto(f"{base_url}/tokenizers")
        expect(page.get_by_text("Select tokenizers")).to_be_visible()
        expect(page.get_by_role("button", name="Run Benchmarks")).to_be_visible()
        expect(page.locator("#datasetSelect")).to_be_visible()
        expect(page.locator("#maxDocs")).to_be_visible()


class TestDatabaseBrowserPage:
    """Tests for database browser UI elements."""

    def test_database_browser_page_loads(self, page: Page, base_url: str):
        """Database browser page should render controls."""
        page.goto(f"{base_url}/database")
        expect(page.get_by_text("Database Browser")).to_be_visible()
        expect(page.locator("#table-select")).to_be_visible()
        expect(page.get_by_role("button", name="Refresh")).to_be_visible()
