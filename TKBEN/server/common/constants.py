from __future__ import annotations

from os.path import abspath, join

# [PATHS]
###############################################################################
ROOT_DIR = abspath(join(__file__, "../../../.."))
PROJECT_DIR = join(ROOT_DIR, "TKBEN")
SETTING_PATH = join(PROJECT_DIR, "settings")
RESOURCES_PATH = join(PROJECT_DIR, "resources")
SOURCES_PATH = join(RESOURCES_PATH, "sources")
DATASETS_PATH = join(SOURCES_PATH, "datasets")
TOKENIZERS_PATH = join(SOURCES_PATH, "tokenizers")
LOGS_PATH = join(RESOURCES_PATH, "logs")
TEMPLATES_PATH = join(RESOURCES_PATH, "templates")
ENV_FILE_PATH = join(SETTING_PATH, ".env")
DATABASE_FILENAME = "database.db"


###############################################################################
CONFIGURATIONS_FILE = join(SETTING_PATH, "configurations.json")


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


