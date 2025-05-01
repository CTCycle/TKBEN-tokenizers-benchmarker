from TokenBenchy.commons.constants import DATA_PATH
from TokenBenchy.commons.logger import logger

             
# [TOKENIZERS EXPLORER]
###############################################################################
class ProcessDataset:

    def __init__(self, configurations, datasets):
        self.datasets = datasets
        self.clean_docs = configurations.get('remove_invalid_documents', True)
        self.dataset_config = configurations.get("DATASET", {})             
        self.target_dataset = self.dataset_config.get("corpus", "wikitext")
        self.target_config = self.dataset_config.get("config", "wikitext-103-v1")

    #--------------------------------------------------------------------------
    def split_text_dataset(self):
        benchmark_data = self.datasets[self.target_config]
        documents = benchmark_data['train']['text']  
        clean_docs = [x for x in documents if len(x) > 0] if self.clean_docs else documents         

        return documents, clean_docs    
    
    