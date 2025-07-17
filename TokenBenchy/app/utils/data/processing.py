             
# [TOKENIZERS EXPLORER]
###############################################################################
class ProcessDataset:

    def __init__(self, configuration, datasets):
        self.datasets = datasets
        self.clean_docs = configuration.get('remove_invalid_documents', True)
        self.dataset_config = configuration.get("DATASET", {})             
        self.target_dataset = self.dataset_config.get("corpus", "wikitext")
        self.target_config = self.dataset_config.get("config", "wikitext-103-v1")
        self.text_data = self.datasets[self.target_config]
        self.documents = self.text_data['train']['text']
        self.num_documents = len(self.documents)

    #--------------------------------------------------------------------------
    def clean_dataset(self):        
        clean_docs = [x for x in self.documents if len(x) > 0] if self.clean_docs else self.documents         

        return clean_docs    
    
    