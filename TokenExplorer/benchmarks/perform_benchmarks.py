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
from TokenExplorer.commons.utils.analyzer.benchmarks import BenchmarkTokenizers
from TokenExplorer.commons.constants import CONFIG
from TokenExplorer.commons.logger import logger


# [RUN MAIN]
###############################################################################
if __name__ == '__main__':

    # 1. [DOWNLOAD TOKENIZERS AND DATASETS]
    #--------------------------------------------------------------------------    
    manager = DownloadManager(CONFIG)   
    logger.info('Download tokenizers and text dataset based on user configurations')       
    tokenizers = manager.tokenizer_download()
    datasets = manager.dataset_download()
    
    # 2. [PROCESS TEXT DATASET]
    #--------------------------------------------------------------------------
    # extract documents from text dataset and split train and text corpora    
    processor = ProcessDataSet(CONFIG, datasets)
    documents, clean_documents = processor.split_text_dataset()  
    logger.info(f'Total number of documents: {len(documents)}')
    logger.info(f'Number of valid documents: {len(clean_documents)}')  

    # 3. [PERFORM BENCHMARKS]
    #--------------------------------------------------------------------------
    # aggregate text dataset statistics and save as .json    
    benchmark = BenchmarkTokenizers(CONFIG, tokenizers)    
    benchmark.aggregate_dataset_stats(documents)        
        
    # run benchmark on selected dataset and generate a series of dataframes with
    # results with various metrics       
    benchmark_results = benchmark.run_tokenizer_benchmarks(documents)

    # run Normalized Sequence Length (NSL) benchmark using the custom tokenizer over
    # the series of tokenizers as baseline     
    logger.info('Calculate Normalized Sequence Length (NSL) vs custom tokenizer')
    NSL_benchmarks = benchmark.normalized_sequence_length()