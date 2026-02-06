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
TOKENIZERS_PATH = join(SOURCES_PATH, "tokenizers"
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
API_ROUTER_PREFIX_BROWSER = "/browser"
API_ROUTER_PREFIX_JOBS = "/jobs"
API_ROUTE_DATASETS_LIST = "/list"
API_ROUTE_DATASETS_DOWNLOAD = "/download"
API_ROUTE_DATASETS_UPLOAD = "/upload"
API_ROUTE_DATASETS_ANALYZE = "/analyze"
API_ROUTE_DATASETS_DELETE = "/delete"
API_ROUTE_TOKENIZERS_SETTINGS = "/settings"
API_ROUTE_TOKENIZERS_SCAN = "/scan"
API_ROUTE_TOKENIZERS_UPLOAD = "/upload"
API_ROUTE_TOKENIZERS_CUSTOM = "/custom"
API_ROUTE_BENCHMARKS_RUN = "/run"
API_ROUTE_BROWSER_TABLES = "/tables"
API_ROUTE_BROWSER_DATA = "/data"
API_ROUTE_JOBS_STATUS = "/{job_id}"


###############################################################################
MODELS_LIST = [
    "Langmuir",
    "Sips",
    "Freundlich",
    "Temkin",
    "Toth",
    "Dubinin-Radushkevich",
    "Dual-Site Langmuir",
    "Redlich-Peterson",
    "Jovanovic",
]

MODEL_PARAMETER_DEFAULTS: dict[str, dict[str, tuple[float, float]]] = {
    "Langmuir": {
        "k": (1e-06, 10.0),
        "qsat": (0.0, 100.0),
    },
    "Sips": {
        "k": (1e-06, 10.0),
        "qsat": (0.0, 100.0),
        "exponent": (0.1, 10.0),
    },
    "Freundlich": {
        "k": (1e-06, 10.0),
        "exponent": (0.1, 10.0),
    },
    "Temkin": {
        "k": (1e-06, 10.0),
        "beta": (0.1, 10.0),
    },
    "Toth": {
        "k": (1e-06, 10.0),
        "qsat": (0.0, 100.0),
        "exponent": (0.1, 10.0),
    },
    "Dubinin-Radushkevich": {
        "qsat": (0.0, 100.0),
        "beta": (1e-06, 10.0),
    },
    "Dual-Site Langmuir": {
        "k1": (1e-06, 10.0),
        "qsat1": (0.0, 100.0),
        "k2": (1e-06, 10.0),
        "qsat2": (0.0, 100.0),
    },
    "Redlich-Peterson": {
        "k": (1e-06, 10.0),
        "a": (1e-06, 10.0),
        "beta": (0.1, 1.0),
    },
    "Jovanovic": {
        "k": (1e-06, 10.0),
        "qsat": (0.0, 100.0),
    },
}

DEFAULT_DATASET_COLUMN_MAPPING = {
    "experiment": "experiment",
    "temperature": "temperature [K]",
    "pressure": "pressure [Pa]",
    "uptake": "uptake [mol/g]",
}

DATASET_FALLBACK_DELIMITERS = (";", "\t", "|")

FITTING_MODEL_NAMES = (
    "LANGMUIR",
    "SIPS",
    "FREUNDLICH",
    "TEMKIN",
    "TOTH",
    "DUBININ_RADUSHKEVICH",
    "DUAL_SITE_LANGMUIR",
    "REDLICH_PETERSON",
    "JOVANOVIC",
)
