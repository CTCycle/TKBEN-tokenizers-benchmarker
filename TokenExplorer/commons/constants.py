import json
from os.path import join, dirname, abspath 

PROJECT_DIR = dirname(dirname(abspath(__file__)))
DATA_PATH = join(PROJECT_DIR, 'resources')
BENCHMARK_PATH = join(DATA_PATH, 'benchmarks')
BENCHMARK_RESULTS_PATH = join(BENCHMARK_PATH, 'results')
BENCHMARK_FIGURES_PATH = join(BENCHMARK_PATH, 'figures')
TOKENIZER_PATH = join(DATA_PATH, 'tokenizers', 'tokenizers')
DATASETS_PATH = join(DATA_PATH, 'datasets', 'datasets')
CUSTOM_DATASET_PATH = join(DATA_PATH, 'tokenizers', 'custom tokenizer')
CUSTOM_TOKENIZER_PATH = join(DATA_PATH, 'datasets', 'custom dataset')

LOGS_PATH = join(DATA_PATH, 'logs')

CONFIG_PATH = join(PROJECT_DIR, 'settings', 'configurations.json')
with open(CONFIG_PATH, 'r') as file:
    CONFIG = json.load(file)