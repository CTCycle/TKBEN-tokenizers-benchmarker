

###############################################################################
class Configurations:
    
    def __init__(self):
        self.configurations = {'include_custom_dataset' : False,
                               'include_custom_tokenizer' : False,
                               'include_NSL' : False,
                               'num_documents' : 0,
                               "DATASET": {"corpus" : "wikitext", 
                                           "config": "wikitext-103-v1"}} 

    #--------------------------------------------------------------------------  
    def get_configurations(self):
        return self.configurations
    
    #--------------------------------------------------------------------------
    def update_value(self, key: str, value: bool):       
        self.configurations[key] = value