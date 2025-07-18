             
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
    def process_text_dataset(self):
        processed_docs = self.documents
        if self.clean_docs:
            processed_docs = [x for x in processed_docs if len(x) > 0]           
        processed_docs = list(dict.fromkeys(processed_docs))         

        return processed_docs   
    
    