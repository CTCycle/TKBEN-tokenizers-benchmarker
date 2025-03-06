import os
import glob
from datasets import load_dataset
from tokenizers import Tokenizer
from transformers import AutoTokenizer

from TokenExplorer.commons.variables import EnvironmentVariables
from TokenExplorer.commons.constants import DATASETS_PATH, TOKENIZER_PATH
from TokenExplorer.commons.logger import logger


# [DOWNLOADS]
###############################################################################
class DownloadManager:

    def __init__(self, configuration):
        # get Hugging Face access token from environmental variables
        EV = EnvironmentVariables()
        self.hf_access_token = EV.get_HF_access_token()
        # extract configurations from given JSON
        self.tokenizers = configuration["TOKENIZERS"]                           
        self.dataset = configuration["DATASET"]        
        self.get_custom_tokenizer = configuration["benchmarks"]["INCLUDE_CUSTOM_TOKENIZER"]   
        self.get_custom_dataset = configuration["benchmarks"]["INCLUDE_CUSTOM_DATASET"]    

    #--------------------------------------------------------------------------
    def tokenizer_download(self):
        tokenizers = {}
        for tokenizer_id in self.tokenizers: 
            try:            
                tokenizer_name = tokenizer_id.replace('/', '_')                 
                tokenizer_save_path = os.path.join(TOKENIZER_PATH, 'open', tokenizer_name)           
                os.mkdir(tokenizer_save_path) if not os.path.exists(tokenizer_save_path) else None            
                logger.info(f'Downloading and saving tokenizer: {tokenizer_id}')
                tokenizer = AutoTokenizer.from_pretrained(
                    tokenizer_id, cache_dir=tokenizer_save_path, token=self.hf_access_token) 
                tokenizers[tokenizer_id] = tokenizer
            except Exception as e:
                logger.error(f"Failed to download tokenizer {tokenizer_id}: {e}", exc_info=True)
    

        # load custom tokenizer in target subfolder if .json files are found 
        custom_tokenizer_path = os.path.join(TOKENIZER_PATH, 'custom')      
        if os.path.exists(custom_tokenizer_path):            
            search_pattern = os.path.join(custom_tokenizer_path, '*.json')   
            json_files = glob.glob(search_pattern)
            if len(json_files) > 0 and self.get_custom_tokenizer:
                logger.info(f'Loading custom tokenizers from {custom_tokenizer_path}')                
                for js in json_files:
                    tokenizer = Tokenizer.from_file(js) 
                    tokenizer_name = os.path.basename(js).split('.')[0]              
                    tokenizers[f'CUSTOM {tokenizer_name}'] = tokenizer                 

        return tokenizers 

    #--------------------------------------------------------------------------
    def dataset_download(self):        
        datasets = {}                
        corpus, config = self.dataset['corpus'], self.dataset['config']
        dataset_path = os.path.join(DATASETS_PATH, 'open', f'{corpus}_{config}') 
        os.mkdir(dataset_path) if not os.path.exists(dataset_path) else None              
        logger.info(f'Downloading and saving dataset: {corpus} - {config}')
        dataset = load_dataset(corpus, config, cache_dir=dataset_path)            
        datasets[config] = dataset
        
        return datasets
        
       
