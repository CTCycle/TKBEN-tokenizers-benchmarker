from __future__ import annotations

from pathlib import Path

# [PATHS]
###############################################################################
ROOT_DIR = Path(__file__).resolve().parents[3]
PROJECT_DIR = ROOT_DIR / "app"
SETTING_PATH = ROOT_DIR / "settings"
RESOURCES_PATH = PROJECT_DIR / "resources"
SOURCES_PATH = RESOURCES_PATH / "sources"
DATASETS_PATH = SOURCES_PATH / "datasets"
TOKENIZERS_PATH = SOURCES_PATH / "tokenizers"
LOGS_PATH = RESOURCES_PATH / "logs"
TEMPLATES_PATH = RESOURCES_PATH / "templates"
ENV_FILE_PATH = SETTING_PATH / ".env"
DATABASE_FILENAME = "database.db"


###############################################################################
CONFIGURATIONS_FILE = SETTING_PATH / "configurations.json"


###############################################################################
FASTAPI_TITLE = "TKBEN_webapp Tokenizers Benchmark Backend"
FASTAPI_DESCRIPTION = "FastAPI backend"
FASTAPI_VERSION = "1.2.0"


###############################################################################
API_ROUTE_ROOT = "/"
API_ROUTE_DOCS = "/docs"
API_ROUTER_PREFIX_DATASETS = "/datasets"
API_ROUTER_PREFIX_TOKENIZERS = "/tokenizers"
API_ROUTER_PREFIX_BENCHMARKS = "/benchmarks"
API_ROUTER_PREFIX_JOBS = "/jobs"
API_ROUTER_PREFIX_KEYS = "/keys"
API_ROUTER_PREFIX_EXPORTS = "/exports"
API_ROUTE_DATASETS_LIST = "/list"
API_ROUTE_DATASETS_DOWNLOAD = "/download"
API_ROUTE_DATASETS_UPLOAD = "/upload"
API_ROUTE_DATASETS_ANALYZE = "/analyze"
API_ROUTE_DATASETS_DELETE = "/delete"
API_ROUTE_DATASETS_REPORT_LATEST = "/reports/latest"
API_ROUTE_DATASETS_REPORT_BY_ID = "/reports/{report_id}"
API_ROUTE_DATASETS_METRICS_CATALOG = "/metrics/catalog"
API_ROUTE_TOKENIZERS_SETTINGS = "/settings"
API_ROUTE_TOKENIZERS_SCAN = "/scan"
API_ROUTE_TOKENIZERS_LIST = "/list"
API_ROUTE_TOKENIZERS_DOWNLOAD = "/download"
API_ROUTE_TOKENIZERS_UPLOAD = "/upload"
API_ROUTE_TOKENIZERS_CUSTOM = "/custom"
API_ROUTE_TOKENIZERS_REPORT_GENERATE = "/reports/generate"
API_ROUTE_TOKENIZERS_REPORT_LATEST = "/reports/latest"
API_ROUTE_TOKENIZERS_REPORT_BY_ID = "/reports/{report_id}"
API_ROUTE_TOKENIZERS_REPORT_VOCABULARY = "/reports/{report_id}/vocabulary"
API_ROUTE_BENCHMARKS_RUN = "/run"
API_ROUTE_BENCHMARKS_REPORTS = "/reports"
API_ROUTE_BENCHMARKS_REPORT_BY_ID = "/reports/{report_id}"
API_ROUTE_BENCHMARKS_METRICS_CATALOG = "/metrics/catalog"
API_ROUTE_EXPORTS_DASHBOARD_PDF = "/dashboard/pdf"
API_ROUTE_JOBS_STATUS = "/{job_id}"
API_ROUTE_KEYS_CREATE = ""
API_ROUTE_KEYS_LIST = ""
API_ROUTE_KEYS_DELETE = "/{key_id}"
API_ROUTE_KEYS_ACTIVATE = "/{key_id}/activate"
API_ROUTE_KEYS_DEACTIVATE = "/{key_id}/deactivate"
API_ROUTE_KEYS_REVEAL = "/{key_id}/reveal"
