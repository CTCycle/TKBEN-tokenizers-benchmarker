import os
import sqlite3
import pandas as pd

from TokenExplorer.commons.constants import DATA_PATH
from TokenExplorer.commons.logger import logger

# [DATABASE]
###############################################################################
class TOKENDatabase:

    def __init__(self, configuration):             
        self.db_path = os.path.join(DATA_PATH, 'XREPORT_database.db') 
        self.configuration = configuration 

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
        processed_data.to_sql('PROCESSED_DATA', conn, if_exists='replace')
        conn.close()

   

 
    
    