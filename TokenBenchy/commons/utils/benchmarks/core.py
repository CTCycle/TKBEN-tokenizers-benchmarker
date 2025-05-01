import os
import numpy as np
import pandas as pd
import transformers
from tqdm import tqdm

from TokenBenchy.commons.utils.data.database import TOKENDatabase
from TokenBenchy.commons.constants import DATASETS_PATH, EVALUATION_PATH
from TokenBenchy.commons.logger import logger

             
# [TOKENIZERS EXPLORER]
###############################################################################
class BenchmarkTokenizers:

    def __init__(self, configuration : dict):
        transformers.utils.logging.set_verbosity_error()        
        self.max_docs_number = configuration.get('num_documents', 0)
        self.reduce_size = configuration.get("reduce_output_size", False)
        self.include_custom_tokenizer = configuration.get("include_custom_tokenizer", False)
        self.include_NSL = configuration.get("include_NSL", False)
        self.database = TOKENDatabase(configuration) 
        self.configuration = configuration       

    #--------------------------------------------------------------------------
    def calculate_dataset_stats(self, documents):
        if self.max_docs_number !=0 and self.max_docs_number <= len(documents):
            documents = documents[:self.max_docs_number]
        dataset_stats = pd.DataFrame()        
        dataset_stats['text'] = documents      
        dataset_stats['words_count'] = dataset_stats['text'].apply(
            lambda doc : len(doc.split()))
        dataset_stats['AVG word length'] = dataset_stats['text'].apply(
            lambda doc : np.mean([len(w) for w in doc.split()]))    
        dataset_stats['STD word length'] = dataset_stats['text'].apply(
            lambda doc : np.std([len(w) for w in doc.split()]))        
       
        self.database.save_dataset_statistics(dataset_stats)

    #--------------------------------------------------------------------------
    def calculate_vocabulary_statistics(self, tokenizers):
        vocabularies = {k : v.get_vocab() for k, v in tokenizers.items()}        
        self.vocab_len = {k: len(v) for k, v in vocabularies.items()}  
        self.vocab_decoded = {}
        self.vocab_len_decoded = {} 
                
        vocab_data = pd.DataFrame({'Tokenizer': name})
        for i, (name, tokenizer) in enumerate(tokenizers.items()):
            vocabulary = vocabularies[name] 
            vocab_indexes = list(vocabulary.values())
            vocab_words = list(vocabulary.keys())
            decoded_words = tokenizer.decode(vocab_indexes).split()                  
                        
            intersection = set(vocab_words).intersection(set(decoded_words))    
            not_intersecting = set(vocab_words).symmetric_difference(set(decoded_words))
            vocab_data['number_of_tokens_from_vocabulary'] = len(vocab_words)
            vocab_data['number_of_tokens_from_decode'] = len(decoded_words)
            vocab_data['number_shared_tokens'] = len(intersection)
            vocab_data['number_unshared_tokens'] = len(not_intersecting)

        return vocab_data
                    
    
    #--------------------------------------------------------------------------
    def run_tokenizer_benchmarks(self, documents, tokenizers : dict, progress_callback=None):        
        if self.max_docs_number is not None and self.max_docs_number <= len(documents):
            documents = documents[:self.max_docs_number]
        
        all_tokenizers = []
        for i, (tokenizer_name, tokenizer) in enumerate(tokenizers.items()):
            k_rep = tokenizer_name.replace('/', '_')
            logger.info(f'Decoding documents with {tokenizer_name}')
            data = pd.DataFrame({'Tokenizer': tokenizer_name,'text': documents})            
            
            data['text_characters'] = data['text'].str.len()            
            data['words_count'] = data['text'].apply(lambda x : len(x.split()))
            data['AVG_words_length'] = data['text'].apply(
                lambda text: np.mean([len(word) for word in text.split()]) if text else 0)

            if 'CUSTOM' in tokenizer_name and self.include_custom_tokenizer:
                data['tokens'] = data['text'].apply(
                    lambda text: tokenizer.decode(tokenizer.encode(text).ids))
                data['tokens split'] = data['tokens'].str.split()
            else:
                data['tokens split'] = data['text'].apply(tokenizer.tokenize)
                data['tokens'] = data['tokens split'].str.join(' ')

            data['tokens_count'] = data['tokens split'].str.len()
            data['tokens_characters'] = data['tokens'].str.len()
            data['AVG_tokens_length'] = data['tokens split'].apply(
                lambda tokens: np.mean([len(tok) for tok in tokens]) if tokens else 0)

            data['tokens_to_words_ratio'] = np.where(
                data['words_count'] > 0, data['tokens_count'] / data['words_count'], 0)
            data['bytes_per_token'] = np.where(
                data['tokens_count'] > 0, data['text_characters'] / data['tokens_count'], 0)
            
            drop_cols = ['tokens split']
            if self.reduce_size:
                drop_cols.extend(['text', 'tokens'])
            data = data.drop(columns=drop_cols)           

            self.database.save_benchmark_results(data, table_name=k_rep)
            self.database.save_benchmark_results(data, table_name=k_rep)
            all_tokenizers.append(data)

            if progress_callback is not None:
                total = len(tokenizers.items())
                percent = int((i + 1) * 100 / total)
                progress_callback(percent)

        benchmark_results = pd.concat(all_tokenizers, ignore_index=True)
        self.database.save_benchmark_results(benchmark_results)

        if self.include_NSL and self.include_custom_tokenizer:
            NSL_results = self.normalized_sequence_length(benchmark_results)

        return benchmark_results

    #--------------------------------------------------------------------------
    def normalized_sequence_length(self, benchmark_results : pd.DataFrame):                    
        data_custom = benchmark_results[
            benchmark_results['Tokenizer'].str.contains(
                'custom tokenizer', case=False, na=False)]   

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
                    data_custom['tokens_count'].to_list(),
                    data_chunk['tokens_count'].to_list())]            
                data.append(data_chunk)
            
            data_NSL = pd.concat(data, ignore_index=True)
            self.database.save_benchmark_results(data_NSL, table_name='NSL')          

        return data_NSL 


    


    
            
                    
        
      


