import os
import re
import sqlite3
import pandas as pd

from TokenBenchy.app.src.constants import DATA_PATH
from TokenBenchy.app.src.logger import logger


###############################################################################
class BenchmarkResultsTable:

    def __init__(self):
        self.name = 'BENCHMARK_RESULTS'
        self.dtypes = {
            'tokenizer': 'VARCHAR',
            'num_characters': 'INTEGER',
            'words_count': 'INTEGER',
            'AVG_words_length': 'FLOAT',
            'tokens_count': 'INTEGER',
            'tokens_characters': 'INTEGER',
            'AVG_tokens_length': 'FLOAT',
            'tokens_to_words_ratio': 'FLOAT',
            'bytes_per_token': 'FLOAT'}
        
    #--------------------------------------------------------------------------
    def get_dtypes(self):
        return self.dtypes
    
    #--------------------------------------------------------------------------
    def create_table(self, cursor):
        query = f'''
        CREATE TABLE IF NOT EXISTS {self.name} (
            tokenizer VARCHAR,
            text_characters INTEGER,
            words_count INTEGER,
            AVG_words_length FLOAT,
            tokens_count INTEGER,
            tokens_characters INTEGER,
            AVG_tokens_length FLOAT,
            tokens_to_words_ratio FLOAT,
            bytes_per_token FLOAT
        );
        '''

        cursor.execute(query)


###############################################################################
class VocabularyStatsTable:

    def __init__(self):
        self.name = 'VOCABULARY_STATISTICS'
        self.dtypes = {
            'tokenizer': 'VARCHAR',
            'number_tokens_from_vocabulary': 'INTEGER',
            'number_tokens_from_decode': 'INTEGER',
            'number_shared_tokens': 'INTEGER',
            'number_unshared_tokens': 'INTEGER',
            'percentage_subwords': 'FLOAT',
            'percentage_true_words': 'FLOAT'}
        
    #--------------------------------------------------------------------------
    def get_dtypes(self):
        return self.dtypes
    
    #--------------------------------------------------------------------------
    def create_table(self, cursor):
        query = f'''
        CREATE TABLE IF NOT EXISTS {self.name} (
            tokenizer VARCHAR,
            number_tokens_from_vocabulary INTEGER,
            number_tokens_from_decode INTEGER,
            number_shared_tokens INTEGER,
            number_unshared_tokens INTEGER,
            percentage_subwords FLOAT,
            percentage_true_words FLOAT
        );
        '''
       
        cursor.execute(query)


###############################################################################
class VocabularyTable:

    def __init__(self):
        self.name = 'VOCABULARY'
        self.dtypes = {
            'id': 'INTEGER',
            'vocabulary_tokens': 'VARCHAR',
            'decoded_tokens': 'VARCHAR'}
        
    #--------------------------------------------------------------------------
    def get_dtypes(self):
        return self.dtypes
    
    #--------------------------------------------------------------------------
    def create_table(self, cursor):
        query = f'''
        CREATE TABLE IF NOT EXISTS {self.name} (
            id INTEGER,
            vocabulary_tokens VARCHAR
            decoded_tokens VARCHAR
        );
        '''
       
        cursor.execute(query)    


###############################################################################
class DatasetStatsTable:

    def __init__(self):
        self.name = 'DATASET_STATISTICS'
        self.dtypes = {
            'text': 'VARCHAR',
            'words_count': 'INTEGER',
            'AVG_word_length': 'FLOAT',
            'STD_word_length': 'FLOAT'}
        
    #--------------------------------------------------------------------------
    def get_dtypes(self):
        return self.dtypes
    
    #--------------------------------------------------------------------------
    def create_table(self, cursor):
        query = f'''
        CREATE TABLE IF NOT EXISTS {self.name} (
            text VARCHAR,
            words_count INTEGER,
            AVG_word_length FLOAT,
            STD_word_length FLOAT
        );
        '''       
        cursor.execute(query)     

# [DATABASE]
###############################################################################
class TokenBenchyDatabase:

    def __init__(self, configuration):                   
        self.db_path = os.path.join(DATA_PATH, 'TokenBenchy_database.db') 
        self.configuration = configuration 
        self.benchmark_results = BenchmarkResultsTable()
        self.vocabulary_results = VocabularyStatsTable()
        self.vocabulary_tokens = VocabularyTable()
        self.dataset_summary = DatasetStatsTable()       
       
    #--------------------------------------------------------------------------       
    def initialize_database(self):        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor() 
        self.benchmark_results.create_table(cursor)  
        self.vocabulary_results.create_table(cursor)  
        self.dataset_summary.create_table(cursor)   
        
        conn.commit()
        conn.close() 

    #--------------------------------------------------------------------------
    def load_benchmark_results(self):            
        conn = sqlite3.connect(self.db_path)        
        benchmarks = pd.read_sql_query(
            f"SELECT * FROM {self.benchmark_results.name}", conn)
        stats = pd.read_sql_query(
            f"SELECT * FROM {self.vocabulary_results.name}", conn)
        conn.close()  

        return benchmarks, stats  

    #--------------------------------------------------------------------------
    def load_vocabulary_tokens(self, table_name=None):
        table_name = re.sub(r'[^0-9A-Za-z_]', '_', table_name) if table_name is not None else None
        table_name = self.vocabulary_tokens.name if table_name is None else f'{table_name}_VOCABULARY'       
        conn = sqlite3.connect(self.db_path)        
        vocabulary = pd.read_sql_query(
            f"SELECT * FROM {table_name}", conn)       
        conn.close()  

        return vocabulary

    #--------------------------------------------------------------------------
    def save_dataset_statistics(self, data):         
        conn = sqlite3.connect(self.db_path)         
        data.to_sql(self.dataset_summary.name, conn, if_exists='replace', index=False,
            dtype=self.dataset_summary.get_dtypes())
        conn.commit()
        conn.close() 
        
    #--------------------------------------------------------------------------
    def save_benchmark_results(self, data, table_name=None):
        table_name = re.sub(r'[^0-9A-Za-z_]', '_', table_name) if table_name is not None else None
        table_name = self.vocabulary_tokens.name if table_name is None else f'{table_name}_BENCHMARK_RESULTS'       
        conn = sqlite3.connect(self.db_path)         
        data.to_sql(table_name, conn, if_exists='replace', index=False,
            dtype=self.benchmark_results.get_dtypes())
        conn.commit()
        conn.close() 

    #--------------------------------------------------------------------------
    def save_vocabulary_results(self, data):        
        conn = sqlite3.connect(self.db_path)         
        data.to_sql(
            self.vocabulary_results.name, conn, if_exists='replace', index=False,
            dtype=self.vocabulary_results.get_dtypes())
        conn.commit()
        conn.close() 

    #--------------------------------------------------------------------------
    def save_vocabulary_tokens(self, data, table_name=None):
        table_name = re.sub(r'[^0-9A-Za-z_]', '_', table_name) if table_name is not None else None
        table_name = self.vocabulary_tokens.name if table_name is None else f'{table_name}_VOCABULARY'       
        conn = sqlite3.connect(self.db_path)         
        data.to_sql(
            table_name, conn, if_exists='replace', index=False,
            dtype=self.vocabulary_tokens.get_dtypes())
        conn.commit()
        conn.close() 

    

   

 
    
    