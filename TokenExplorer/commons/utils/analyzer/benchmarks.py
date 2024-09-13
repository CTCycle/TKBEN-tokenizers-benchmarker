import os
import numpy as np
import pandas as pd
import transformers
from tqdm import tqdm
tqdm.pandas()

from TokenExplorer.commons.constants import CONFIG
from TokenExplorer.commons.logger import logger

             
# [TOKENIZERS EXPLORER]
###############################################################################
class BenchmarkTokenizers:

    def __init__(self, tokenizers):
        self.tokenizers = tokenizers 
        transformers.utils.logging.set_verbosity_error() 

    #--------------------------------------------------------------------------
    def aggregate_dataset_stats(self, documents, path, max_number=None):

        if max_number is not None and max_number <= len(documents):
            documents = documents[:max_number]
        dataset_stats = pd.DataFrame()
        logger.info('Aggregating dataset statistics')
        dataset_stats['Text'] = documents      
        dataset_stats['Words count'] = dataset_stats['Text'].progress_apply(lambda x : len(x.split()))
        dataset_stats['Words length'] = dataset_stats['Text'].progress_apply(lambda doc : [len(x) for x in doc.split()])

        filename = os.path.join(path, 'dataset_stats.csv')
        dataset_stats.to_csv(filename, encoding='utf-8', sep=';', index=False)           
    
    #--------------------------------------------------------------------------
    def run_tokenizer_benchmarks(self, documents, path, max_number=None, reduce_size=False):        
        
        try:
            os.remove(os.path.join(path, 'tokenizers_benchmark.csv'))
        except:
            logger.debug('No tokenizers_benchmark.csv file found')
        if max_number is not None and max_number <= len(documents):
            documents = documents[:max_number]
        for k, v in self.tokenizers.items():
            k_rep = k.replace('/', '_')            
            df = pd.DataFrame() 
            df['Tokenizer'] = [k] * len(documents)       
            df['Text'] = documents
            df['Text characters'] = df['Text'].apply(lambda x : len(x))
            df['Words count'] = df['Text'].apply(lambda x : len(x.split()))
            df['Words length'] = df['Text'].apply(lambda doc : [len(x) for x in doc.split()])
            df['AVG words length'] = df['Words length'].apply(lambda x : np.mean(x))

            # tokenize each document looping over tokenizers        
            logger.info(f'Decoding documents with {k}') 
            if 'custom tokenizer' in k:                
                df['Tokens'] = df['Text'].progress_apply(lambda text: v.encode(text).ids)
                df['Tokens'] = df['Tokens'].progress_apply(lambda ids: v.decode(ids))
                df['Tokens characters'] = df['Tokens'].progress_apply(lambda x : len(x))
                df['Tokens count'] = df['Tokens'].progress_apply(lambda x : len(x.split()))
                df['Tokens length'] = df['Tokens'].progress_apply(lambda doc : [len(x) for x in doc.split()])
                df['AVG tokens length'] = df['Tokens length'].progress_apply(lambda x : np.mean(x))                      
                df['Tokens/words ratio'] = df.progress_apply(lambda row: row['Tokens count']/row['Words count'] 
                                                                         if row['Words count'] > 0 else 0, axis=1) 
                df['Bytes per token'] = df.progress_apply(lambda row: row['Text characters']/row['Tokens count'] 
                                                                         if row['Tokens count'] > 0 else 0, axis=1)              
            else:                
                df['Tokens'] = df['Text'].progress_apply(lambda text: v.tokenize(text))
                df['Tokens characters'] = df['Tokens'].progress_apply(lambda x : len(' '.join(x)))               
                df['Tokens count'] = df['Tokens'].progress_apply(lambda x : len(x))
                df['Tokens length'] = df['Tokens'].progress_apply(lambda doc : [len(x) for x in doc])  
                df['AVG tokens length'] = df['Tokens length'].progress_apply(lambda x : np.mean(x))                    
                df['Tokens/words ratio'] = df.progress_apply(lambda row: row['Tokens count']/row['Words count'] 
                                                                         if row['Words count'] > 0 else 0, axis=1)  
                df['Bytes per token'] = df.progress_apply(lambda row: row['Text characters']/row['Tokens count'] 
                                                                         if row['Tokens count'] > 0 else 0, axis=1)  

            # save single .csv file for each tokenizer
            filename = os.path.join(path, f'{k_rep}_benchmark.csv')
            df['Tokens'] = df['Tokens'].apply(lambda x: ' '.join(str(token) for token in x))
            df['Words length'] = df['Words length'].apply(lambda x: ' '.join(str(token) for token in x))
            df['Tokens length'] = df['Tokens length'].apply(lambda x: ' '.join(str(token) for token in x))            
            if reduce_size:
                df = df.drop(columns=['Text', 'Tokens'], axis=1)               
            df.to_csv(filename, encoding='utf-8', sep=';', index=False)        
    
        # merge all .csv files in a single benchmark dataset
        csv_files = [file for file in os.listdir(path) if file.endswith('.csv')]   
        csv_paths = [os.path.join(path, x) for x in csv_files]                 
        data_frames = [pd.read_csv(x, encoding='utf-8', sep=';') for x in csv_paths]        
        merged_data = pd.concat(data_frames, ignore_index=True)       
        filename = os.path.join(path, 'tokenizers_benchmark.csv')
        merged_data.to_csv(filename, index=False, encoding='utf-8', sep=';')     


###############################################################################
def normalized_sequence_length(source_path, save_path):            
    
    # load benchmarks and isolate df with only target columns
    filepath = os.path.join(source_path, 'tokenizers_benchmark.csv')
    df_tokens = pd.read_csv(filepath, sep=';', encoding='utf-8', usecols=['Tokenizer', 'Tokens count'])
    df_custom = df_tokens[df_tokens['Tokenizer'].str.contains('custom tokenizer', case=False, na=False)]   

    data = []
    tokenizer_names = list(df_tokens['Tokenizer'].unique())
    if df_custom.empty:
        logger.info('NSL value cannot be calculated without a custom tokenizer as reference')
    else:
        for tok in tqdm(tokenizer_names):
            logger.info(f'NSL value is calculated for {tok} versus custom tokenizers')
            df_chunk = df_tokens[df_tokens['Tokenizer'] == tok]                                                 
            df_chunk['NSL'] = [x/y if y != 0 else 0 for x, y in zip(df_custom['Tokens count'].to_list(),
                                                                    df_chunk['Tokens count'].to_list())]            
            data.append(df_chunk)

        # merge all datasets and save them into .csv file
        df_NSL = pd.concat(data, ignore_index=True)
        filename = os.path.join(save_path, 'NSL_benchmark.csv')
        df_NSL.to_csv(filename, index=False, encoding='utf-8', sep=';') 


    


    
