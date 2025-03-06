import json
from os.path import join, abspath 

# [PATHS]
###############################################################################
ROOT_DIR = abspath(join(__file__, "../../.."))
PROJECT_DIR = abspath(join(__file__, "../.."))
DATA_PATH = join(PROJECT_DIR, 'resources')
BENCHMARK_PATH = join(DATA_PATH, 'benchmarks')
BENCHMARK_FIGURES_PATH = join(BENCHMARK_PATH, 'figures')
TOKENIZER_PATH = join(DATA_PATH, 'tokenizers')
DATASETS_PATH = join(DATA_PATH, 'datasets')
LOGS_PATH = join(DATA_PATH, 'logs')



# [CONFIGURATIONS]
###############################################################################
CONFIG_PATH = join(PROJECT_DIR, 'settings', 'configurations.json')
with open(CONFIG_PATH, 'r') as file:
    CONFIG = json.load(file)