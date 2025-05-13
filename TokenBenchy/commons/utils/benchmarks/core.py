import numpy as np
import pandas as pd
import transformers
from tqdm import tqdm

from TokenBenchy.commons.utils.data.database import TokenBenchyDatabase
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
        self.database = TokenBenchyDatabase(configuration)
        self.vocab_columns = ['id', 'vocabulary_tokens', 'decoded_tokens'] 
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
        rows = []
        for name, tokenizer in tokenizers.items():
            vocab = tokenizer.get_vocab()              
            vocab_words = list(vocab.keys())
            vocab_indices = list(vocab.values())
            subwords = [x for x in vocab_words if '##' in x]
            true_words = [x for x in vocab_words if x not in subwords]
            
            decoded_words = tokenizer.decode(vocab_indices).split()            
            shared = set(vocab_words).intersection(decoded_words)
            unshared = set(vocab_words).symmetric_difference(decoded_words)            
           
            subwords_perc = len(subwords)/(len(true_words) + len(subwords)) * 100
            words_perc = len(true_words)/(len(true_words) + len(subwords)) * 100           
            
            rows.append({
                'tokenizer': name,
                'number_tokens_from_vocabulary': len(vocab_words),
                'number_tokens_from_decode': len(decoded_words),
                'number_shared_tokens': len(shared),
                'number_unshared_tokens': len(unshared),
                'percentage_subwords': subwords_perc,
                'percentage_true_words' : words_perc})            

            # save entire vocabulary into database
            vocabulary = pd.DataFrame({
                self.vocab_columns[0]: pd.Series(vocab_indices),
                self.vocab_columns[1]: pd.Series(vocab_words),
                self.vocab_columns[2]: pd.Series(decoded_words)})
            self.database.save_vocabulary_tokens(vocabulary, name) 

        # save vocabulary statistics into database 
        vocabulary_stats = pd.DataFrame(rows)
        self.database.save_vocabulary_results(vocabulary_stats)           
        
        return vocabulary, vocabulary_stats
    
    #--------------------------------------------------------------------------
    def run_tokenizer_benchmarks(self, documents, tokenizers : dict, progress_callback=None):  
        vocab, vocab_stats = self.calculate_vocabulary_statistics(tokenizers)
        if self.max_docs_number is not None and self.max_docs_number <= len(documents):
            documents = documents[:self.max_docs_number]      
        
        all_tokenizers = []
        for i, (tokenizer_name, tokenizer) in enumerate(tokenizers.items()):
            k_rep = tokenizer_name.replace('/', '_')
            logger.info(f'Decoding documents with {tokenizer_name}')
            data = pd.DataFrame({'tokenizer': tokenizer_name, 'text': documents})            
            
            data['num_characters'] = data['text'].apply(lambda x : len(str(x)))       
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
                data['tokens_count'] > 0, data['num_characters'] / data['tokens_count'], 0)
            
            drop_cols = ['tokens split']
            if self.reduce_size:
                drop_cols.extend(['text', 'tokens'])
            data = data.drop(columns=drop_cols)           

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
            benchmark_results['tokenizer'].str.contains(
                'custom tokenizer', case=False, na=False)]   

        data = []
        tokenizer_names = list(benchmark_results['tokenizer'].unique())
        if data_custom.empty:
            logger.warning('NSL value cannot be calculated without a custom tokenizer as reference')
            return None
        else:
            for tok in tqdm(tokenizer_names):
                logger.info(f'NSL value is calculated for {tok} versus custom tokenizers')
                data_chunk = benchmark_results[benchmark_results['tokenizer'] == tok]                                                 
                data_chunk['NSL'] = [
                    x/y if y != 0 else 0 for x, y in zip(
                    data_custom['tokens_count'].to_list(),
                    data_chunk['tokens_count'].to_list())]            
                data.append(data_chunk)
            
            data_NSL = pd.concat(data, ignore_index=True)
            self.database.save_benchmark_results(data_NSL, table_name='NSL')          

        return data_NSL 


    


    
            
                    
        
      


