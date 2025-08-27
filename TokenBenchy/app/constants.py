from os.path import abspath, join

# [PATHS]
###############################################################################
ROOT_DIR = abspath(join(__file__, "../../.."))
PROJECT_DIR = join(ROOT_DIR, "TokenBenchy")
SETUP_PATH = join(ROOT_DIR, "setup")
RSC_PATH = join(PROJECT_DIR, "resources")
DATA_PATH = join(RSC_PATH, "database")
EVALUATION_PATH = join(DATA_PATH, "evaluation")
TOKENIZER_PATH = join(DATA_PATH, "tokenizers")
DATASETS_PATH = join(DATA_PATH, "datasets")
CONFIG_PATH = join(RSC_PATH, "configurations")
LOGS_PATH = join(RSC_PATH, "logs")

# [UI LAYOUT PATH]
###############################################################################
UI_PATH = join(PROJECT_DIR, "app", "assets", "window_layout.ui")
