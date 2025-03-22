import os
import sqlite3
import pandas as pd

from TokenExplorer.commons.constants import DATA_PATH
from TokenExplorer.commons.logger import logger

# [DATABASE]
###############################################################################
class TOKENDatabase:

    def __init__(self, configuration):                   
        self.db_path = os.path.join(DATA_PATH, 'TOKENEXP_database.db') 
        self.configuration = configuration 
        self.initialize_database()

    #--------------------------------------------------------------------------       
    def initialize_database(self):        
        # Connect to the SQLite database and create the database if does not exist
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        create_overall_benchmark_table = '''
        CREATE TABLE IF NOT EXISTS OVERALL_BENCHMARK_RESULTS (
           id INTEGER PRIMARY KEY AUTOINCREMENT,
           tokenizer TEXT,
           text_characters INTEGER,
           words_count INTEGER,
           AVG_words_length REAL,
           Tokens_count INTEGER,
           Tokens_characters INTEGER,
           AVG_tokens_length REAL,
           Tokens_to_words_ratio REAL,
           Bytes_per_token REAL
        );
        '''

        create_dataset_stats_table = '''
        CREATE TABLE IF NOT EXISTS OVERALL_BENCHMARK_RESULTS (
           id INTEGER PRIMARY KEY AUTOINCREMENT,
           Text TEXT,
           Words_count INTEGER,
           AVG_word_length REAL,
           STD_word_length REAL       
        );
        '''
      
        cursor.execute(create_overall_benchmark_table)  
        cursor.execute(create_dataset_stats_table)  
        
        conn.commit()
        conn.close()

    #--------------------------------------------------------------------------
    def load_benchmark_results(self, table_name=None):
        # Connect to the database and inject a select all query
        # convert the extracted data directly into a pandas dataframe          
        conn = sqlite3.connect(self.db_path)        
        data = pd.read_sql_query(f"SELECT * FROM {table_name}", conn)
        conn.close()  

        return data       

    #--------------------------------------------------------------------------
    def save_dataset_statistics(self, data : pd.DataFrame):        
        # connect to sqlite database and save the preprocessed data as table
        conn = sqlite3.connect(self.db_path)         
        data.to_sql('DATASET_STATISTICS', conn, if_exists='replace')
        conn.commit()
        conn.close() 
        
    #--------------------------------------------------------------------------
    def save_benchmark_results(self, processed_data : pd.DataFrame, table_name=None):
        table_name = 'OVERALL_BENCHMARK_RESULTS' if table_name is None else f'{table_name}_BENCHMARK_RESULTS'
        # Connect to the database and inject a select all query
        # convert the extracted data directly into a pandas dataframe          
        conn = sqlite3.connect(self.db_path)        
        processed_data.to_sql(table_name, conn, if_exists='replace')
        conn.close()

   

 
    
    