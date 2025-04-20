from TokenBenchy.commons.constants import CONFIG
from TokenBenchy.commons.logger import logger

             
# [TOKENIZERS EXPLORER]
###############################################################################
class ProcessDataset:

    def __init__(self, configurations, datasets):
        self.datasets = datasets     
        self.target_dataset = configurations["DATASET"].get("corpus", "wikitext")
        self.target_config = configurations["DATASET"].get("config", "wikitext-103-v1")

    #--------------------------------------------------------------------------
    def split_text_dataset(self):
        benchmark_data = self.datasets[self.target_config]
        documents = benchmark_data['train']['text']  
        clean_docs = [x for x in documents if len(x) > 0]         

        return documents, clean_docs    
    
    