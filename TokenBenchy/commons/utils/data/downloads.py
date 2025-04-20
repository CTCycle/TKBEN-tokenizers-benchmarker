import os
import glob
from datasets import load_dataset
from tokenizers import Tokenizer
from transformers import AutoTokenizer

from TokenBenchy.commons.constants import DATASETS_PATH, TOKENIZER_PATH
from TokenBenchy.commons.logger import logger


# [DOWNLOADS]
###############################################################################
class DatasetDownloadManager:

    def __init__(self, configuration, hf_access_token):
        self.hf_access_token = hf_access_token        
        # extract configurations from given JSON                                 
        self.dataset = configuration.get("DATASET", {})  
        self.dataset_corpus = self.dataset.get('corpus', 'wikitext')
        self.dataset_config = self.dataset.get('config', 'wikitext-103-v1')   
        self.has_custom_dataset = configuration.get('include_custom_dataset', False)      

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
        
       
# [DOWNLOADS]
###############################################################################
class TokenizersDownloadManager:

    def __init__(self, configuration, hf_access_token):
        self.hf_access_token = hf_access_token    
        self.tokenizers = configuration.get("TOKENIZERS", [])
        self.has_custom_tokenizer = configuration.get('include_custom_tokenizer', False)      
        
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
            if len(json_files) > 0 and self.has_custom_tokenizer:
                logger.info(f'Loading custom tokenizers from {custom_tokenizer_path}')                
                for js in json_files:
                    tokenizer = Tokenizer.from_file(js) 
                    tokenizer_name = os.path.basename(js).split('.')[0]              
                    tokenizers[f'CUSTOM {tokenizer_name}'] = tokenizer                 

        return tokenizers 

        