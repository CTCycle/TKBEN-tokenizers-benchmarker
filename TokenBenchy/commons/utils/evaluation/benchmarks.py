import os
import numpy as np
import pandas as pd
import transformers
from tqdm import tqdm

from TokenBenchy.commons.utils.data.database import TOKENDatabase
from TokenBenchy.commons.constants import CONFIG, DATASETS_PATH, EVALUATION_PATH
from TokenBenchy.commons.logger import logger

             
# [TOKENIZERS EXPLORER]
###############################################################################
class BenchmarkTokenizers:

    def __init__(self, configuration : dict):
        transformers.utils.logging.set_verbosity_error()        
        self.max_docs_number = configuration.get('num_documents', 0)
        self.reduce_size = configuration.get("reduce_output_size", False)
        self.database = TOKENDatabase(configuration) 
        self.configuration = configuration           

    #--------------------------------------------------------------------------
    def calculate_dataset_stats(self, documents):
        if self.max_docs_number !=0 and self.max_docs_number <= len(documents):
            documents = documents[:self.max_docs_number]
        dataset_stats = pd.DataFrame()        
        dataset_stats['Text'] = documents      
        dataset_stats['Words count'] = dataset_stats['Text'].apply(
            lambda doc : len(doc.split()))
        dataset_stats['AVG word length'] = dataset_stats['Text'].apply(
            lambda doc : np.mean([len(w) for w in doc.split()]))    
        dataset_stats['STD word length'] = dataset_stats['Text'].apply(
            lambda doc : np.std([len(w) for w in doc.split()]))        
       
        self.database.save_dataset_statistics(dataset_stats)          
    
    #--------------------------------------------------------------------------
    def run_tokenizer_benchmarks(self, documents, tokenizers : dict):        
        if self.max_docs_number is not None and self.max_docs_number <= len(documents):
            documents = documents[:self.max_docs_number]
        
        all_tokenizers = []
        for tokenizer_name, tokenizer in tokenizers.items():
            k_rep = tokenizer_name.replace('/', '_')
            logger.info(f'Decoding documents with {tokenizer_name}')
            data = pd.DataFrame({'Tokenizer': tokenizer_name,'Text': documents})
            data['Text characters'] = data['Text'].str.len()            
            data['Words count'] = data['Text'].apply(lambda x : len(x.split()))
            data['AVG words length'] = data['Text'].apply(
                lambda text: np.mean([len(word) for word in text.split()]) if text else 0)

            if 'CUSTOM' in tokenizer_name:
                data['Tokens'] = data['Text'].apply(
                    lambda text: tokenizer.decode(tokenizer.encode(text).ids))
                data['Tokens split'] = data['Tokens'].str.split()
            else:
                data['Tokens split'] = data['Text'].apply(tokenizer.tokenize)
                data['Tokens'] = data['Tokens split'].str.join(' ')

            data['Tokens count'] = data['Tokens split'].str.len()
            data['Tokens characters'] = data['Tokens'].str.len()
            data['AVG tokens length'] = data['Tokens split'].apply(
                lambda tokens: np.mean([len(tok) for tok in tokens]) if tokens else 0)

            data['Tokens to words ratio'] = np.where(
                data['Words count'] > 0, data['Tokens count'] / data['Words count'], 0)
            data['Bytes per token'] = np.where(
                data['Tokens count'] > 0, data['Text characters'] / data['Tokens count'], 0)

            if self.reduce_size:
                data = data.drop(columns=['Text', 'Tokens', 'Tokens split'])           

            self.database.save_benchmark_results(data, table_name=k_rep)
            all_tokenizers.append(data)

        merged_data = pd.concat(all_tokenizers, ignore_index=True)
        self.database.save_benchmark_results(merged_data)

        return merged_data

    #--------------------------------------------------------------------------
    def normalized_sequence_length(self, benchmark_results : pd.DataFrame):                    
        data_custom = benchmark_results[
            benchmark_results['Tokenizer'].str.contains('custom tokenizer', case=False, na=False)]   

        data = []
        tokenizer_names = list(benchmark_results['Tokenizer'].unique())
        if data_custom.empty:
            logger.warning('NSL value cannot be calculated without a custom tokenizer as reference')
            return None
        else:
            for tok in tqdm(tokenizer_names):
                logger.info(f'NSL value is calculated for {tok} versus custom tokenizers')
                data_chunk = benchmark_results[benchmark_results['Tokenizer'] == tok]                                                 
                data_chunk['NSL'] = [
                    x/y if y != 0 else 0 for x, y in zip(
                    data_custom['Tokens count'].to_list(),
                    data_chunk['Tokens count'].to_list())]            
                data.append(data_chunk)
            
            data_NSL = pd.concat(data, ignore_index=True)
            self.database.save_benchmark_results(data_NSL, table_name='NSL')          

        return data_NSL 


    


    
            
                    
        
      


