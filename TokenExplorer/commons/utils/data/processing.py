
from TokenExplorer.commons.constants import CONFIG
from TokenExplorer.commons.logger import logger

             
# [TOKENIZERS EXPLORER]
###############################################################################
class ProcessDataset:

    def __init__(self, configurations, datasets):
        self.datasets = datasets     
        self.target_dataset = configurations["DATASET"]["corpus"]
        self.target_config = configurations["DATASET"]["config"]

    #--------------------------------------------------------------------------
    def split_text_dataset(self):
        benchmark_data = self.datasets[self.target_config]
        documents = benchmark_data['train']['text']  
        clean_docs = [x for x in documents if len(x) > 0]         

        return documents, clean_docs    
    
    