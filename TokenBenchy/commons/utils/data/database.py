import os
import sqlite3
import pandas as pd

from TokenBenchy.commons.constants import DATA_PATH
from TokenBenchy.commons.logger import logger


###############################################################################
class BenchmarkResultsTable:

    def __init__(self):
        self.name = 'BENCHMARK_RESULTS'
        self.dtypes = {
            'tokenizer': 'VARCHAR',
            'text_characters': 'INTEGER',
            'words_count': 'INTEGER',
            'AVG_words_length': 'FLOAT',
            'Tokens_count': 'INTEGER',
            'Tokens_characters': 'INTEGER',
            'AVG_tokens_length': 'FLOAT',
            'Tokens_to_words_ratio': 'FLOAT',
            'Bytes_per_token': 'FLOAT'}
        
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
            Tokens_count INTEGER,
            Tokens_characters INTEGER,
            AVG_tokens_length FLOAT,
            Tokens_to_words_ratio FLOAT,
            Bytes_per_token FLOAT
        );
        '''

        cursor.execute(query)


###############################################################################
class DatasetStatsTable:

    def __init__(self):
        self.name = 'DATASET_STATISTICS'
        self.dtypes = {
            'Text': 'VARCHAR',
            'Words_count': 'INTEGER',
            'AVG_word_length': 'FLOAT',
            'STD_word_length': 'FLOAT'}
        
    #--------------------------------------------------------------------------
    def get_dtypes(self):
        return self.dtypes
    
    #--------------------------------------------------------------------------
    def create_table(self, cursor):
        query = f'''
        CREATE TABLE IF NOT EXISTS {self.name} (
            Text VARCHAR,
            Words_count INTEGER,
            AVG_word_length FLOAT,
            STD_word_length FLOAT
        );
        '''
       
        cursor.execute(query)
    
      


# [DATABASE]
###############################################################################
class TOKENDatabase:

    def __init__(self, configuration):                   
        self.db_path = os.path.join(DATA_PATH, 'TokenBenchy_database.db') 
        self.configuration = configuration 
        self.benchmark_results = BenchmarkResultsTable()
        self.dataset_summary = DatasetStatsTable()         
        self.initialize_database()  
       
    #--------------------------------------------------------------------------       
    def initialize_database(self):        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor() 
        self.benchmark_results.create_table(cursor)  
        self.dataset_summary.create_table(cursor)   
        conn.commit()
        conn.close() 

    #--------------------------------------------------------------------------
    def load_benchmark_results(self):            
        conn = sqlite3.connect(self.db_path)        
        data = pd.read_sql_query(
            f"SELECT * FROM {self.benchmark_results.name}", conn)
        conn.close()  

        return data       

    #--------------------------------------------------------------------------
    def save_dataset_statistics(self, data : pd.DataFrame):         
        conn = sqlite3.connect(self.db_path)         
        data.to_sql(self.dataset_summary.name, conn, if_exists='replace', index=False,
            dtype=self.dataset_summary.get_dtypes())
        conn.commit()
        conn.close() 
        
    #--------------------------------------------------------------------------
    def save_benchmark_results(self, data : pd.DataFrame, table_name=None):
        table_name = self.benchmark_results.name if table_name is None else f'{table_name}_BENCHMARK_RESULTS'
        conn = sqlite3.connect(self.db_path)         
        data.to_sql(table_name, conn, if_exists='replace', index=False,
            dtype=self.benchmark_results.get_dtypes())
        conn.commit()
        conn.close() 

    

   

 
    
    