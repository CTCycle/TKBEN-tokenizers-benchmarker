

###############################################################################
class Configuration:
    
    def __init__(self):
        self.configuration = {'use_custom_dataset' : False,
                              'remove_invalid_documents' : True, 
                              'include_custom_tokenizer' : False,
                              'include_NSL' : False,
                              'num_documents' : 0,
                              'reduce_output_size' : False,
                              'save_images' : True,
                              "DATASET": {"corpus" : "wikitext", 
                                           "config": "wikitext-103-v1"},
                              "TOKENIZERS": []} 

    #--------------------------------------------------------------------------  
    def get_configuration(self):
        return self.configuration
    
    #--------------------------------------------------------------------------
    def update_value(self, key: str, value: bool):       
        self.configuration[key] = value