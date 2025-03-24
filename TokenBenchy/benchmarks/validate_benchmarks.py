# [SETTING ENVIRONMENT VARIABLES]
from TokenBenchy.commons.variables import EnvironmentVariables
EV = EnvironmentVariables()

# [SETTING WARNINGS]
import warnings
warnings.simplefilter(action='ignore', category=Warning)

# [IMPORT CUSTOM MODULES]
from TokenBenchy.commons.utils.data.downloads import DownloadManager
from TokenBenchy.commons.utils.evaluation.explorer import ExploreTokenizers
from TokenBenchy.commons.constants import CONFIG 
from TokenBenchy.commons.logger import logger


# [RUN MAIN]
###############################################################################
if __name__ == '__main__':

    # 1. [DOWNLOAD TOKENIZERS AND DATASETS]
    #--------------------------------------------------------------------------      
    logger.info('Download tokenizers and text dataset')       
    manager = DownloadManager(CONFIG)
    tokenizers = manager.tokenizer_download()
    datasets = manager.dataset_download()
    
    # 2. [EXPLORE TOKENIZERS]
    #--------------------------------------------------------------------------
    # extract wikitext-103-v1 data and split train and text corpora    
    explorer = ExploreTokenizers(tokenizers)
    logger.info('Check length of tokenizers vocabulary by using two methods:')
    logger.info('1) Extraction of the embedded vocabulary')
    logger.info('2) Decoding through vocabulary mapping')
    logger.info('Comparison should highlight any possible discrepancy')
    explorer.get_vocabulary_report()     

    logger.info('Analyze distribution of token by characters length using histograms and boxplots')
    explorer.plot_vocabulary_size()
    explorer.plot_histogram_tokens_length()
    explorer.plot_boxplot_tokens_length()
    explorer.plot_subwords_vs_words()

    logger.info('Plot a series of metrics to evaluate the performance of the tokenizers on the given dataset')
    logger.info('1) Evaluate number of generate tokens versus number of words in text (by document)')
    logger.info('2) Average character length of tokens versus average length of words (average by document)')
    logger.info('3) Bytes per token calculated by dividing UTF-8 bytes by tokens number')
    #explorer.boxplot_from_benchmarks_dataset() # temporarely removed
    




    

    