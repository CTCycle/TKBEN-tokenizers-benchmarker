import os

# [SETTING ENVIRONMENT VARIABLES]
from TokenExplorer.commons.variables import EnvironmentVariables
EV = EnvironmentVariables()

# [SETTING WARNINGS]
import warnings
warnings.simplefilter(action='ignore', category=Warning)

# [IMPORT CUSTOM MODULES]
from TokenExplorer.commons.utils.downloads import DownloadManager
from TokenExplorer.commons.utils.processing import ProcessDataSet
from TokenExplorer.commons.utils.analyzer.benchmarks import BenchmarkTokenizers, normalized_sequence_length
from TokenExplorer.commons.constants import CONFIG, DATASETS_PATH, BENCHMARK_PATH, BENCHMARK_VALIDATION_PATH
from TokenExplorer.commons.logger import logger


# [RUN MAIN]
###############################################################################
if __name__ == '__main__':

    # 1. [DOWNLOAD TOKENIZERS AND DATASETS]
    #--------------------------------------------------------------------------      
    logger.info('Download tokenizers and text dataset')       
    manager = DownloadManager()    
    tokenizers = manager.tokenizer_download()
    datasets = manager.dataset_download()
    
    # extract wikitext-103-v1 data and split train and text corpora    
    processor = ProcessDataSet(datasets)
    documents = processor.split_dataset()    

    # aggregate text dataset statistics and save as .json    
    benchmark = BenchmarkTokenizers(tokenizers)
    if not os.path.exists(os.path.join(DATASETS_PATH, f'{processor.target_dataset}_stats.csv')):
        benchmark.aggregate_dataset_stats(documents, DATASETS_PATH, 
                                          max_number=CONFIG["benchmarks"]["MAX_NUM_DOCS"])        
        
    # run benchmark on selected dataset and generate a series of dataframes with
    # results with various metrics       
    benchmark_results = benchmark.run_tokenizer_benchmarks(documents, BENCHMARK_VALIDATION_PATH,                                                     
                                                           max_number=CONFIG["benchmarks"]["MAX_NUM_DOCS"], 
                                                           reduce_size=CONFIG["benchmarks"]["REDUCE_CSV_SIZE"])

    # run Normalized Sequence Length (NSL) benchmark using the custom tokenizer over
    # the series of tokenizers as baseline     
    logger.info('Calculate Normalized Sequence Length (NSL)')
    normalized_sequence_length(BENCHMARK_VALIDATION_PATH, BENCHMARK_PATH)