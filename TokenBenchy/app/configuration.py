import os
import json

from TokenBenchy.app.constants import CONFIG_PATH

###############################################################################
class Configuration:
    
    def __init__(self):
        self.configuration = {
            'use_custom_dataset' : False,
            'remove_invalid_documents' : True, 
            'include_custom_tokenizer' : False,
            'perform_NSL' : False,
            'num_documents' : 50000,
            'image_resolution' : 400,                          
            "DATASET": {"corpus" : "wikitext", 
                        "config": "wikitext-103-v1"},
            "TOKENIZERS": []} 

    #--------------------------------------------------------------------------  
    def get_configuration(self):
        return self.configuration
    
    #--------------------------------------------------------------------------
    def update_value(self, key: str, value: bool):       
        self.configuration[key] = value

    #--------------------------------------------------------------------------
    def save_configuration_to_json(self, name : str):  
        full_path = os.path.join(CONFIG_PATH, f'{name}.json')      
        with open(full_path, 'w') as f:
            json.dump(self.configuration, f, indent=4)