from tqdm import tqdm
tqdm.pandas()

from TokenExplorer.commons.constants import CONFIG
from TokenExplorer.commons.logger import logger

             
# [TOKENIZERS EXPLORER]
###############################################################################
class ProcessDataSet:

    def __init__(self, datasets):
        self.datasets = datasets
        self.target_dataset = CONFIG["DATASET"][0]["corpus"]
        self.target_config = CONFIG["DATASET"][0]["config"]

    #--------------------------------------------------------------------------
    def split_dataset(self):

        benchmark_data = self.datasets[self.target_config]
        documents = benchmark_data['train']['text']  
        clean_docs = [x for x in documents if len(x) > 0] 
        logger.info(f'Total number of documents: {len(documents)}')
        logger.info(f'Number of valid documents: {len(clean_docs)}\n')

        return clean_docs    
    
    