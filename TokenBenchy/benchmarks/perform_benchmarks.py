# [SETTING ENVIRONMENT VARIABLES]
from TokenBenchy.commons.variables import EnvironmentVariables
EV = EnvironmentVariables()

# [SETTING WARNINGS]
import warnings
warnings.simplefilter(action='ignore', category=Warning)

# [IMPORT CUSTOM MODULES]
from TokenBenchy.commons.utils.data.downloads import DownloadManager
from TokenBenchy.commons.utils.data.processing import ProcessDataset
from TokenBenchy.commons.utils.evaluation.benchmarks import BenchmarkTokenizers
from TokenBenchy.commons.constants import CONFIG
from TokenBenchy.commons.logger import logger


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
    processor = ProcessDataset(CONFIG, datasets)
    documents, clean_documents = processor.split_text_dataset()  
    logger.info(f'Total number of documents: {len(documents)}')
    logger.info(f'Number of valid documents: {len(clean_documents)}')  

    # 3. [PERFORM BENCHMARKS]
    #--------------------------------------------------------------------------
    # aggregate text dataset statistics and save as .json    
    benchmark = BenchmarkTokenizers(CONFIG, tokenizers)  
    logger.info('Calculate text dataset statistics')  
    benchmark.calculate_dataset_stats(clean_documents)        
        
    # run benchmark on selected dataset and generate a series of dataframes with
    # results with various metrics       
    benchmark_results = benchmark.run_tokenizer_benchmarks(documents)

    # run Normalized Sequence Length (NSL) benchmark using the custom tokenizer over
    # the series of tokenizers as baseline     
    logger.info('Calculate Normalized Sequence Length (NSL) vs custom tokenizer')
    NSL_benchmarks = benchmark.normalized_sequence_length(benchmark_results)